import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from tqdm import tqdm
import os
import pickle


from THINGSImagedataset import THINGSImageDataset    # assume this is in your project
from globals import get_rating_cols, get_SD_cols, get_all_cols
from visulizing import plot_loss_curves

# =====================================================================
# 1. Model Definition (借鉴 CEBRA 中的模型设计思想)
# =====================================================================
class CEBRAContrastiveModel(nn.Module):
    """
    这个模型参考了 CEBRA 在 `cebra/models.py` 中的编码器 + 投影头结构。
    - CEBRA 中先用主干网络提取特征，再接一个 MLP 作为空间嵌入投影头。
    - 投影头常用两层全连接 + 激活 + 归一化。
    """
    def __init__(self, base_encoder: nn.Module, 
                 output_dim:int = 128, property_dim:int = None):
        super().__init__()
        
        # 如果没有显式传入 property_dim，就动态取 len(get_all_cols())
        if property_dim is None:
            property_dim = len(get_rating_cols())

        # CLIP 图像编码器作为主干，等同于在 CEBRA 中使用的 backbone
        self.encoder = base_encoder
        self._setup_trainable_layers()
        
        # ===== CEBRA 投影头结构 =====
        # 在 CEBRA 源码 `cebra/models/spatial_projector.py` 类似：
        # Linear -> Activation -> Linear -> Normalize
        feat_dim = self.encoder.visual.output_dim  # CLIP ViT 输出维度
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )
        #----属性预测 head 输出维度为 property_dim_mean (10)----
        self.property_head = nn.Linear(output_dim, property_dim)
        # CLIP 在 cuda 下默认用 float16，所以这里确保 projection 也是 float16
        # encoder_param = next(self.encoder.parameters())
        # self.projection = self.projection.to(device=encoder_param.device)
        # self.projection = self.projection.to(
        #     device=encoder_param.device,
        #     dtype=encoder_param.dtype
        #     )
    
    def _setup_trainable_layers(self):
        # 冻结整个模型
        for param in self.encoder.parameters():
            param.requires_grad = False
        # 只解冻视觉投影层
        if hasattr(self.encoder.visual, 'proj'):
            # ViT: proj 是 Parameter
            self.encoder.visual.proj.requires_grad = True
        elif hasattr(self.encoder.visual, 'head'):
            # ResNet/GPT 版本: head 是 Module
            for param in self.encoder.visual.head.parameters():
                param.requires_grad = True
                
    def forward(self, x):
        # 提取主干特征，参考 CEBRA 中 encoder.forward
        features = self.encoder.encode_image(x)  # [B, D]
        features = F.normalize(features, dim=1)  # CEBRA 建议先归一化

        # 投影头：获得可解释嵌入
        z = self.projection(features)
        z = F.normalize(z, dim=1)  # 最后再归一化，匹配 CEBRA 的 L2 约束
        
        p_hat = self.property_head(z)  # 预测属性值[B, 10]，只预测 mean
        
        return z, p_hat

# =====================================================================
# 2. 对比损失函数 InfoNCE + SmoothL1Loss 结合评分信息
# =====================================================================
huber_loss = nn.SmoothL1Loss(reduction='none')
def contrastive_loss(z_i, z_j, temperature=0.1):
    """
    在 CEBRA 中，这一函数实现了他们的 InfoNCE 对比损失：
    - 计算一批正样本对的相似度矩阵
    - 以对角线位置为正例标签，用交叉熵损失
    - 对 i->j 和 j->i 对称计算再平均
    """
    batch_size = z_i.shape[0]
    # 余弦相似度矩阵，并除以温度
    sim = torch.matmul(z_i, z_j.t()) / temperature
    labels = torch.arange(batch_size, device=z_i.device)
    loss_i = F.cross_entropy(sim, labels)
    loss_j = F.cross_entropy(sim.t(), labels)
    return (loss_i + loss_j) * 0.5

def weighted_huber_loss(p_hat: torch.Tensor,
                        mean: torch.Tensor,
                        sd: torch.Tensor,
                        huber_loss_fn: torch.nn.Module,
                        eps: float = 1e-6,
                        pdf_min: float = 1e-3) -> torch.Tensor:
    """
    计算单个 view 的“加权 Huber 损失”，
    权重由真实分布 N(mean, sd) 的 pdf 决定(weight = 1 / pdf)。
    但对 pdf 最小值做 clamp 以避免权重过大。
    """
    # 1) 计算 elementwise Huber：huber_elem 形状 [B, C]
    huber_elem = huber_loss_fn(p_hat, mean)  # [B, C]

    # 2) 基于真实 sd 计算“置信度 pdf”：
    #    z = (p_hat - mean) / sd
    z_val = (p_hat - mean) / (sd + eps)   # [B, C]
    pdf = (1.0 / ((2 * torch.pi) ** 0.5 * (sd + eps))) \
          * torch.exp(-0.5 * z_val * z_val)  # [B, C]
    pdf = torch.clamp(pdf, min=pdf_min)  # 避免权重过大
    # 3) 计算权重：weight = 1 / (pdf + eps)
    weight = 1.0 / (pdf + eps)  # [B, C]

    # 4) 加权 Huber，最后对所有元素取平均
    weighted_huber = weight * huber_elem  # [B, C]
    return weighted_huber.mean()         # 标量

# =====================================================================
# 3. 数据采样 —— Positive Pairs (参照 CEBRA 中的 PairLoader)
# =====================================================================
class PairDataset(torch.utils.data.Dataset):
    """
    参考 CEBRA 数据加载器中的同类对构造方式：
    - 根据 uniqueID 将同类图像分组
    - 对每个组内构造正样本对
    - CEBRA 还支持分层采样、负样本队列等，可按需扩展
    """
    def __init__(self, base_dataset):
        self.base = base_dataset
        self._build_index()
        self.pairs = self._generate_pairs()

    def _build_index(self) -> None:
        """构建ID到索引的映射"""
        self.class_to_indices = {}
        for idx in range(len(self.base)):
            uid = self.base.fields['uniqueID'][idx]
            if isinstance(uid, bytes): 
                uid = uid.decode()
            self.class_to_indices.setdefault(uid, []).append(idx)

    def _generate_pairs(self) -> list:
        """生成正样本对列表"""
        pairs = []
        for indices in self.class_to_indices.values():
            n = len(indices)
            # 每个类生成n*(n-1)/2个正样本对
            for i in range(n):
                for j in range(i+1, n):
                    pairs.append((indices[i], indices[j]))
        return pairs
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        sample_i = self.base[i]
        sample_j = self.base[j]

        img_i = sample_i['image']
        img_j = sample_j['image']

        # 分别读取 10 维 mean 和 10 维 SD
        mean_cols = get_rating_cols()
        sd_cols = get_SD_cols()
        # 构建长度为 len(all_cols) 的属性向量
        prop_mean_i = torch.tensor([sample_i[col] for col in mean_cols], dtype=torch.float32)  # [10,]
        prop_sd_i   = torch.tensor([sample_i[col] for col in sd_cols  ], dtype=torch.float32)  # [10,]
        prop_mean_j = torch.tensor([sample_j[col] for col in mean_cols], dtype=torch.float32)  # [10,]
        prop_sd_j   = torch.tensor([sample_j[col] for col in sd_cols  ], dtype=torch.float32)  # [10,]
        
        return {'image_i': img_i, 'image_j': img_j,
                'mean_i'    : prop_mean_i,   # [10,]
                'sd_i'      : prop_sd_i,     # [10,]
                'mean_j'    : prop_mean_j,   # [10,]
                'sd_j'      : prop_sd_j      # [10,]
            }

# =====================================================================
# 4. 训练流程
# =====================================================================
def train(model, full_train_ds, optimizer, device,
          sample_size = None,
          epochs=10, temperature: float = 0.1,
          max_lambda: float = 10.0, random_seed: int = 42,
          batch_size: int = 64, num_workers: int = 4,
          scheduler=None, start_epoch: int = 1):
    """
    基于 CEBRA 的训练脚本框架：
    - 先 model.train(), 冻结层不更新
    - 每个 batch 计算 InfoNCE, 反向+优化
    - 打印 epoch 平均损失
    - 模型只预测 10 维 mean,真实 SD 用于加权 Huber
    """
    model.to(device)
    model.train()
    eps = 1e-6

    epoch_losses = []

    total_train = len(full_train_ds)
    for ep in range(start_epoch, epochs+1):

        if sample_size is not None:
            curr_sample_n = min(sample_size, total_train)
            gen = torch.Generator().manual_seed(random_seed + ep)  
            # 用 ep 变化的种子，保证每个 epoch 都不同，但可复现
            rand_indices = torch.randperm(total_train, generator=gen)[:curr_sample_n].tolist()
            epoch_subset = Subset(full_train_ds, rand_indices)
            print(f"[Epoch {ep}] Sampled {curr_sample_n}/{total_train} pairs.")
        else:
            # 若 sample_size 设为 None，则整个数据集都用上
            epoch_subset = full_train_ds

        # 基于本 epoch 的子集构造 DataLoader
        loader = DataLoader(
            epoch_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=True,
            prefetch_factor=2
        )

        pbar = tqdm(loader, desc=f"Epoch {ep}", dynamic_ncols=True)
        total_loss = 0.0

        for batch_idx, batch in enumerate(pbar):
            img_i = batch['image_i'].to(device,non_blocking=True)
            img_j = batch['image_j'].to(device,non_blocking=True)
            mean_i   = batch['mean_i'].to(device, non_blocking=True)   # [B, 10]
            sd_i     = batch['sd_i'].to(device, non_blocking=True)     # [B, 10]
            mean_j   = batch['mean_j'].to(device, non_blocking=True)   # [B, 10]
            sd_j     = batch['sd_j'].to(device, non_blocking=True)     # [B, 10]

            optimizer.zero_grad()
            
            z_i,p_hat_i = model(img_i) # [B, D], [B, 2]
            z_j,p_hat_j = model(img_j) # [B, D], [B, 2]

            loss_contrastive = contrastive_loss(z_i, z_j, temperature)
            # 计算属性损失，附加评分信息
            loss_prop_i = weighted_huber_loss(p_hat_i, mean_i, sd_i, huber_loss, eps)
            loss_prop_j = weighted_huber_loss(p_hat_j, mean_j, sd_j, huber_loss, eps)
            loss_prop = (loss_prop_i + loss_prop_j) * 0.5

            # loss = loss_contrastive + lambda_prop * loss_prop
            dynamic_lambda = (loss_contrastive.detach() / (loss_prop.detach() + eps)).clamp(max=max_lambda)
            loss = loss_contrastive + dynamic_lambda * loss_prop
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # avg_loss = total_loss / (batch_idx+1)
            pbar.set_postfix({
                'contra_loss': f"{loss_contrastive.item():.4f}",
                'prop_loss': f"{loss_prop.item():.4f}"
            }) # 这里loss_prop是没有经过lambda_prop相乘的,经过测试一般初始在140左右，后期会变小
            # loss_contrastive一般初始在4左右，然后会逐渐减小
        pbar.close()

        avg_loss = total_loss / len(loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {ep}: Avg Loss={avg_loss:.4f}")

        if scheduler is not None:
            scheduler.step(avg_loss) # scheduler 根据 avg_loss 决定是否降低 LR
            current_lr = scheduler.get_last_lr()
            print(f"[Epoch {ep}] Avg Loss={avg_loss:.4f}, Learning Rate={current_lr[0]:.2e}")
        # 保存模型
        torch.save(model.state_dict(), f"./results/models/model_epoch_{ep}.pth")
    plot_loss_curves(epoch_losses)


def create_datasets(
    meta_path: str, 
    img_root: str, 
    transform: transforms.Compose
) -> tuple:
    """创建基础数据集和正样本对数据集"""
    base_ds = THINGSImageDataset(meta_path, img_root, transform)
    pair_ds = PairDataset(base_ds)

    os.makedirs(CONFIG['split_save_dir'], exist_ok=True)
    train_pkl = os.path.join(CONFIG['split_save_dir'], 'train_indices.pkl')
    val_pkl   = os.path.join(CONFIG['split_save_dir'], 'val_indices.pkl')
    test_pkl  = os.path.join(CONFIG['split_save_dir'], 'test_indices.pkl')

    # 如果三个划分文件都存在，则直接加载
    if os.path.isfile(train_pkl) and os.path.isfile(val_pkl) and os.path.isfile(test_pkl):
        print("Found existing split files. Loading train/val/test indices from disk.")
        with open(train_pkl, 'rb') as f:
            train_indices = pickle.load(f)
        with open(val_pkl, 'rb') as f:
            val_indices = pickle.load(f)
        with open(test_pkl, 'rb') as f:
            test_indices = pickle.load(f)

        # 用 Subset 构造对应的数据集
        train_ds = Subset(pair_ds, train_indices)
        val_ds   = Subset(pair_ds, val_indices)
        test_ds  = Subset(pair_ds, test_indices)

        total_pairs = len(pair_ds)
        print(f"Loaded existing splits:"
              f" train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
              f" (total={total_pairs})")
        return train_ds, val_ds, test_ds
    else:
        print("Creating new split files.")
        # 如果三个划分文件都不存在，则随机划分数据集
        total_pairs = len(pair_ds)
        train_size = int(total_pairs * CONFIG['split_ratio'][0])
        val_size = int(total_pairs * CONFIG['split_ratio'][1])
        test_size = total_pairs - train_size - val_size

        generator = torch.Generator().manual_seed(CONFIG['random_seed'])
        train_ds, val_ds, test_ds = random_split(
            pair_ds, [train_size, val_size, test_size], generator=generator
            )
        
        # 保存划分结果
        with open(train_pkl, 'wb') as f:
            pickle.dump(train_ds.indices, f)
        with open(val_pkl, 'wb') as f:
            pickle.dump(val_ds.indices, f)
        with open(test_pkl, 'wb') as f:
            pickle.dump(test_ds.indices, f)

        print(f"Performed new split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
            f" (total={total_pairs})")
        return train_ds, val_ds, test_ds

def setup_model(
    model_name: str = 'ViT-B/32', 
    output_dim: int = 128,
    device: torch.device = torch.device('cpu')
) -> nn.Module:
    """初始化CLIP模型和对比学习模型"""
    clip_model, _ = clip.load(model_name, device=device, jit=False)
    clip_model = clip_model.float() #把 CLIP 主干和投影层都改为 float32
    return CEBRAContrastiveModel(clip_model, output_dim=output_dim)

# =====================================================================
# 5. 主运行入口
# =====================================================================
if __name__ == '__main__':
    # 配置参数
    CONFIG = {
        'meta_path': './preprocessed/things_image.npz',
        'img_root': './image/_image_database_things',
        'batch_size': 128,
        'num_workers': 4,
        'lr': 1e-4,
        'epochs': 20,
        'output_dim': 128,
        'sample_size': 12800,
        'clip_model': 'ViT-B/32',
        'split_ratio': [0.8,0.1,0.1],
        'random_seed': 42,
        'split_save_dir': './splits',
        'resume': True,
        'resume_epoch': 17
    }
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(224), # 这里调用了CLIP预训练的ViT视觉编码器，因此输入大小为224x224
        transforms.ToTensor(),
    ])
    
    # ----创建数据集并进行划分----
    full_train_ds, val_ds, test_ds = create_datasets(
        CONFIG['meta_path'], 
        CONFIG['img_root'], 
        transform
    )
    
    # 初始化模型
    model = setup_model(
        CONFIG['clip_model'],
        CONFIG['output_dim'],
        device
    )
    
    # 设置优化器 (只训练可训练参数)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=CONFIG['lr']
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',            # 监控指标越小越好
        factor=0.5,            # 每次降低为原来的 0.5 倍
        patience=3,            # 若 3 个 epoch 内 loss 无下降，则触发降 LR
        min_lr=1e-7            # LR 最低不低于 1e-7
    )

    start_epoch = 1
    if CONFIG.get('resume', False):
        # 加载之前保存的模型
        ckpt_path = f"./results/models/model_epoch_{CONFIG['resume_epoch']}.pth"
        if os.path.isfile(ckpt_path):
            print(f"Resuming training: loading checkpoint {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
            start_epoch = CONFIG['resume_epoch'] + 1
        else:
            raise ValueError(f"Checkpoint {ckpt_path} not found.")
        
    # 训练
    train(
        model, full_train_ds, optimizer, device,
        sample_size=CONFIG['sample_size'],
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        scheduler=scheduler,
        start_epoch=start_epoch
    )