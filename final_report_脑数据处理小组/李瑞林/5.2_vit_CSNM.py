# 导入必要的库（新增部分已标注）
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from THINGSfastdataset import THINGSfastDataset  # 自定义fMRI数据集
from visualizing_CSNM import plot_loss_curves

# ---------------------- CSNM模块定义 ----------------------
class CSNMModule(nn.Module):
    """
    跨尺度邻域匹配模块（Cross-Scale Neighborhood Matching）
    输入多尺度patch嵌入，输出融合后的跨尺度特征
    
    参数：
        scales: 尺度数量（如[大, 中, 小]三种尺度）
        embed_dim: 单尺度嵌入维度（多尺度共享）
        kernel_size: 邻域匹配的局部窗口大小（如3x3x3）
    """
    def __init__(self, scales=3, embed_dim=512, kernel_size=(3,3,3)):
        super().__init__()
        self.scales = scales
        self.kernel_size = kernel_size
        
        # 注意力权重计算（用于跨尺度匹配）
        self.attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim*2, embed_dim),  # 拼接两尺度特征后降维
                nn.ReLU(),
                nn.Linear(embed_dim, 1)             # 输出匹配权重
            ) for _ in range(scales*(scales-1)//2)  # 每对尺度计算一次注意力
        ])
        
        # 融合层（将多尺度特征加权求和）
        self.fusion = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim*scales, embed_dim)
        )

    def forward(self, multi_scale_embeds):
        """
        参数：
            multi_scale_embeds: 多尺度patch嵌入列表，每个元素形状为 [B, N_p, D]
        """
        B, _, D = multi_scale_embeds[0].shape
        all_pairs = []
        
        # 计算所有尺度对的邻域匹配权重
        for i in range(self.scales):
            for j in range(i+1, self.scales):
                # 提取i尺度和j尺度的特征
                feat_i = multi_scale_embeds[i]  # [B, N_p_i, D]
                feat_j = multi_scale_embeds[j]  # [B, N_p_j, D]
                
                # 邻域窗口采样（简化版：取局部区域的平均）
                # 注：实际可根据需求设计更复杂的邻域采样（如3D卷积）
                window_i = nn.functional.avg_pool3d(
                    feat_i.reshape(B, -1, *self.grid_size[i]),  # 恢复空间形状
                    kernel_size=self.kernel_size, 
                    stride=1, 
                    padding=self.kernel_size[0]//2
                ).flatten(2).transpose(1,2)  # [B, N_p_i, D]
                
                window_j = nn.functional.avg_pool3d(
                    feat_j.reshape(B, -1, *self.grid_size[j]),
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size[0]//2
                ).flatten(2).transpose(1,2)  # [B, N_p_j, D]
                
                # 计算跨尺度注意力权重（i→j和j→i）
                concat_ij = torch.cat([window_i, window_j], dim=-1)  # [B, N_p_i, 2D]
                concat_ji = torch.cat([window_j, window_i], dim=-1)  # [B, N_p_j, 2D]
                attn_ij = self.attention[i*(self.scales-1)+j](concat_ij).softmax(dim=1)  # [B, N_p_i, 1]
                attn_ji = self.attention[i*(self.scales-1)+j](concat_ji).softmax(dim=1)  # [B, N_p_j, 1]
                
                # 加权融合邻域信息
                matched_ij = (attn_ij * feat_j).sum(dim=1, keepdim=True)  # [B, 1, D]（j尺度信息融合到i）
                matched_ji = (attn_ji * feat_i).sum(dim=1, keepdim=True)  # [B, 1, D]（i尺度信息融合到j）
                all_pairs.extend([matched_ij, matched_ji])
        
        # 融合所有尺度和匹配对的特征（简化版：拼接后线性融合）
        fused = torch.cat([*multi_scale_embeds, *all_pairs], dim=1)  # [B, N_p*scales + 2*pair_num, D]
        return self.fusion(fused.mean(dim=1))  # [B, D]（全局平均后融合）


# ---------------------- CSNM优化的3D ViT模型 ----------------------
class CSNM_ViT3D(nn.Module):
    """
    基于CSNM的3D视觉Transformer模型，增强跨尺度局部上下文建模
    """
    def __init__(self, 
                 input_shape=(72, 91, 75),
                 patch_sizes=[(12,13,15), (6,13,15), (12,13,15)],  # 多尺度patch尺寸
                 embed_dim=512,
                 num_heads=8,
                 num_layers=6,
                 num_classes=12,
                 csnm_kernel=(3,3,3)):
        super().__init__()
        
        # 检查多尺度patch尺寸是否可整除输入形状
        for p in patch_sizes:
            assert all([i % p_dim == 0 for i, p_dim in zip(input_shape, p)]), \
                f"Input shape {input_shape} not divisible by patch size {p}"
        
        self.input_shape = input_shape
        self.patch_sizes = patch_sizes
        self.scales = len(patch_sizes)
        
        # 多尺度patch嵌入层（每个尺度独立的3D卷积）
        self.multi_patch_embed = nn.ModuleList([
            nn.Conv3d(
                in_channels=1, 
                out_channels=embed_dim, 
                kernel_size=p, 
                stride=p
            ) for p in patch_sizes
        ])
        
        # 计算各尺度的网格大小和总patch数
        self.grid_sizes = [tuple(i//p for i, p in zip(input_shape, ps)) for ps in patch_sizes]
        self.num_patches_per_scale = [np.prod(gs) for gs in self.grid_sizes]
        
        # CSNM跨尺度匹配模块
        self.csnm = CSNMModule(
            scales=self.scales, 
            embed_dim=embed_dim, 
            kernel_size=csnm_kernel
        )
        # 为CSNM模块绑定各尺度的网格大小（用于邻域采样）
        self.csnm.grid_size = self.grid_sizes  # 传递给CSNMModule
        
        # 可学习的cls token（全局特征）
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # 位置编码（多尺度共享，或为每个尺度单独设计）
        self.pos_embed = nn.Parameter(torch.randn(1, sum(self.num_patches_per_scale)+1, embed_dim))
        
        # Transformer编码器（使用增强后的跨尺度特征）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4*embed_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 回归头
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        
        # 1. 多尺度patch嵌入
        multi_embeds = []
        for embed_layer, ps in zip(self.multi_patch_embed, self.patch_sizes):
            feat = embed_layer(x)  # [B, D, D', H', W']
            feat = feat.flatten(2).transpose(1,2)  # [B, N_p, D]（N_p=D'×H'×W'）
            multi_embeds.append(feat)
        
        # 2. 跨尺度邻域匹配（CSNM模块）
        csnm_feat = self.csnm(multi_embeds)  # [B, D]（融合后的跨尺度特征）
        
        # 3. 拼接多尺度patch和cls token
        all_patches = torch.cat(multi_embeds, dim=1)  # [B, sum(N_p), D]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, D]
        x = torch.cat([cls_tokens, all_patches, csnm_feat.unsqueeze(1)], dim=1)  # [B, sum(N_p)+2, D]
        x = x + self.pos_embed  # 位置编码
        
        # 4. Transformer编码
        x = self.transformer(x)  # [B, sum(N_p)+2, D]
        
        # 5. 取cls token输出
        x = x[:, 0, :]  # [B, D]
        return self.head(x)  # [B, 12]


# ---------------------- 训练函数 ----------------------
def train_model(train_dataset, batch_size, model, criterion, optimizer, device, rating_cols, epochs=10, batches_per_epoch=32):
    """
    训练模型（子采样加速训练）
    
    参数：
        train_dataset: 训练数据集（THINGSfastDataset实例）
        batch_size: 每批次样本数
        model: 待训练模型（ViT3D实例）
        criterion: 损失函数（如MSE）
        optimizer: 优化器（如AdamW）
        device: 计算设备（GPU/CPU）
        rating_cols: 目标评分的列名（12维，对应数据集中的评分字段）
        epochs: 训练轮数
        batches_per_epoch: 每轮训练的批次数（子采样，减少计算量）
    """
    train_losses = []  # 记录每轮的平均损失
    model.train()      # 开启训练模式（激活Dropout等正则化层）
    dataset_size = len(train_dataset)  # 数据集总样本数

    for epoch in range(epochs):
        # 1. 随机子采样：每轮只训练 batches_per_epoch × batch_size 个样本（加速训练）
        num_samples = batches_per_epoch * batch_size  # 每轮训练的总样本数
        indices = torch.randperm(dataset_size)[:num_samples].tolist()  # 随机选择样本索引（前num_samples个）
        sampler = SubsetRandomSampler(indices)  # 子采样器（随机采样指定索引的样本）
        
        # 2. 构建数据加载器（使用子采样器）
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,    # 使用子采样器（替代默认的顺序采样）
            num_workers=4,      # 多进程加载数据（提高IO效率）
            pin_memory=True     # 锁页内存（加速GPU数据传输，仅在GPU可用时有效）
        )
        
        total_loss = 0.0  # 累计损失（每轮清零）
        
        # 3. 训练进度条（tqdm显示训练进度）
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            # 加载数据和目标（fMRI数据和12维评分）
            data = batch['fmri'].to(device)  # fMRI数据（形状：[batch_size, 1, D, H, W]）
            # 拼接12维目标评分（从字典中按列名提取，转为浮点张量）
            targets = torch.stack([
                batch[col].to(device).float() for col in rating_cols
            ], dim=1)  # 形状：[batch_size, 12]（12维评分）
            
            # 前向传播
            outputs = model(data)  # 预测评分（形状：[batch_size, 12]）
            loss = criterion(outputs, targets)  # 计算MSE损失（预测值与真实值的均方误差）
            
            # 反向传播
            optimizer.zero_grad()  # 清空优化器梯度（避免累积）
            loss.backward()        # 计算梯度（反向传播）
            optimizer.step()       # 更新模型参数（根据梯度）
            
            total_loss += loss.item()  # 累计当前批次的损失（标量值）
            
            # 打印中间日志（每12个batch）
            if batch_idx % 12 == 0:
                pbar.write(f"Epoch: {epoch+1:03d}/{epochs} | Batch: {batch_idx:03d}/{batches_per_epoch} | Loss: {loss.item():.4f}")
            pbar.set_postfix_str(f"loss={loss.item():.4f}")  # 进度条显示当前批次损失
        
        # 计算并记录每轮平均损失（总损失 / 批次数）
        avg_loss = total_loss / batches_per_epoch
        print(f"\nEpoch: {epoch+1:03d}/{epochs} | Avg Loss: {avg_loss:.4f}\n")
        train_losses.append(avg_loss)  # 保存每轮平均损失

    return train_losses  # 返回各轮训练损失列表


# ---------------------- 测试函数 ----------------------
def test_model(test_dataset, batch_size, model, device, rating_cols, criterion, batches_to_test=16):
    """
    测试模型（子采样验证）
    
    参数：
        test_dataset: 测试数据集（THINGSfastDataset实例）
        batch_size: 每批次样本数
        model: 待测试模型（ViT3D实例）
        device: 计算设备（GPU/CPU）
        rating_cols: 目标评分列名（12维）
        criterion: 损失函数（如MSE）
        batches_to_test: 测试的批次数（子采样，减少计算量）
    """
    model.eval()  # 开启评估模式（关闭Dropout等随机层）
    total_test_loss = 0.0  # 累计测试损失
    total_error = 0.0      # 累计绝对误差（评估预测准确性）
    dataset_size = len(test_dataset)  # 测试集总样本数

    with torch.no_grad():  # 关闭梯度计算（加速推理）
        # 子采样测试数据（仅测试batches_to_test × batch_size个样本）
        num_samples = batches_to_test * batch_size
        indices = torch.randperm(dataset_size)[:num_samples].tolist()
        sampler = SubsetRandomSampler(indices)
        loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
        
        # 测试进度条
        pbar = tqdm(loader, desc="Testing", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            data = batch['fmri'].to(device)  # fMRI数据（形状：[batch_size, 1, D, H, W]）
            targets = torch.stack([
                batch[col].to(device).float() for col in rating_cols
            ], dim=1)  # 目标评分（形状：[batch_size, 12]）
            
            outputs = model(data)  # 预测评分（形状：[batch_size, 12]）
            loss = criterion(outputs, targets)  # 计算MSE损失
            total_test_loss += loss.item()     # 累计测试损失
            
            # 计算绝对误差（预测值与真实值的平均绝对差）
            total_error += torch.abs(outputs - targets).mean().item()
            
            # 打印中间日志（每batches_to_test个batch）
            if batch_idx % batches_to_test == 0:
                tqdm.write(f"Test Batch {batch_idx}/{batches_to_test} | Loss: {loss.item():.4f}")
    
    # 计算平均测试损失和平均绝对误差
    avg_test_loss = total_test_loss / batches_to_test
    avg_test_error = total_error / batches_to_test
    
    print(f"\nTest Loss: {avg_test_loss:.4f} | Test Avg Error: {avg_test_error:.4f}")
    # 可视化预测结果（最后一个batch的样本）
    visualize_fmri_slice(batch, model, device, rating_cols)

    return avg_test_loss  # 返回平均测试损失


# ---------------------- 可视化函数 ----------------------
def visualize_fmri_slice(batch, model, device, rating_cols, z_slice=30):
    """
    可视化fMRI切片和预测-真实评分对比
    
    参数：
        batch: 当前批次数据（包含fMRI、评分、图像路径等）
        model: 模型（用于生成预测）
        device: 计算设备
        rating_cols: 评分列名（12维）
        z_slice: 可视化的z轴切片位置（深度方向）
    """
    was_training = model.training  # 记录模型原模式（训练/评估）
    model.eval()  # 临时切换到评估模式（关闭随机层）
    
    # 加载数据和目标
    data = batch['fmri'].to(device)  # fMRI数据（形状：[batch_size, 1, D, H, W]）
    targets = torch.stack([batch[col].to(device) for col in rating_cols], dim=1)  # 真实评分（形状：[batch_size, 12]）
    image_paths = batch['image_path']  # 对应的图像路径（用于标注）
    
    with torch.no_grad():  # 关闭梯度计算
        outputs = model(data)  # 预测评分（形状：[batch_size, 12]）
    
    # 提取第一个样本的fMRI切片和评分（可视化单个样本）
    fmri_slice = data[0, 0, :, :, z_slice].cpu().numpy()  # 形状：[D, H]（z_slice层的二维切片）
    pred_vec = outputs[0].cpu().numpy()  # 预测评分（形状：[12]）
    true_vec = targets[0].cpu().numpy()  # 真实评分（形状：[12]）
    image_path = image_paths[0]          # 对应的图像路径（字符串）
    
    # 绘制fMRI切片和评分对比图（1行2列）
    plt.figure(figsize=(12, 6))  # 画布大小
    
    # 子图1：fMRI切片（灰度图）
    plt.subplot(1, 2, 1)
    plt.imshow(fmri_slice, cmap='gray')  # 显示灰度切片
    plt.title(f"z={z_slice} Slice (Image: {image_path})")  # 标题标注切片位置和图像路径
    plt.axis('off')  # 关闭坐标轴
    
    # 子图2：评分对比条形图
    plt.subplot(1, 2, 2)
    x = range(len(rating_cols))  # x轴位置（12个评分维度）
    plt.bar([i-0.2 for i in x], true_vec, width=0.4, label='True')  # 真实评分（左偏移0.2）
    plt.bar([i+0.2 for i in x], pred_vec, width=0.4, label='Pred')  # 预测评分（右偏移0.2）
    plt.xticks(x, rating_cols, rotation=90)  # x轴标签（评分列名，旋转90度避免重叠）
    plt.ylabel('Rating Value')  # y轴标签（评分值）
    plt.title('True vs Predicted Ratings')  # 子图标题
    plt.legend()  # 显示图例（True/Pred）
    plt.tight_layout()  # 调整子图布局（避免重叠）
    plt.savefig('trans.png')  # 保存可视化结果
    plt.close()
    
    # 恢复模型原模式（如果之前是训练模式）
    if was_training:
        model.train()

# ---------------------- 主程序（超参数调整） ----------------------
if __name__ == "__main__":
    # 调整超参数以适配CSNM（新增多尺度相关参数）
    config = {
        "input_shape": (72, 91, 75),
        "patch_sizes": [(12,13,15), (6,13,15), (12,13,15)],  # 多尺度patch（需整除输入形状）
        "embed_dim": 512,
        "num_heads": 8,
        "num_layers": 6,
        "num_classes": 12,
        "batch_size": 8,  # 多尺度特征增加计算量，可能需要减小batch_size
        "lr": 5e-5,       # 微调学习率（多尺度模型更复杂）
        "epochs": 12,
        "train_ratio": 0.8,
        "batches_to_train": 32,
        "batches_to_test": 12,
        "csnm_kernel": (3,3,3)  # CSNM邻域窗口大小
    }
    
    # 初始化数据集（同原始代码）
    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = THINGSfastDataset(
        meta_npz_path="./preprocessed/things_meta.npz",
        image_root_dir="./image/_image_database_things",
        transform=transform
    )
    
    # 划分训练集和测试集（同原始代码）
    train_size = int(config["train_ratio"] * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # 初始化设备和模型（替换为CSNM_ViT3D）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSNM_ViT3D(
        input_shape=config["input_shape"],
        patch_sizes=config["patch_sizes"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"],
        csnm_kernel=config["csnm_kernel"]
    ).to(device)

    # 定义目标评分的列名（12维，对应数据集中的评分字段）
    rating_cols = [
        'image-label_nameability_mean','image-label_consistency_mean',
        'property_manmade_mean','property_precious_mean','property_lives_mean',
        'property_heavy_mean','property_natural_mean','property_moves_mean',
        'property_grasp_mean','property_hold_mean','property_be-moved_mean','property_pleasant_mean'
    ]

    # 定义损失函数、优化器（同原始代码）
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    
    # 训练、测试、保存模型（同原始代码）
    print("Starting training...")
    train_losses = train_model(train_dataset, config["batch_size"], model, criterion, optimizer, device, rating_cols, config["epochs"], config["batches_to_train"])
    print("Training completed!")
    
    model_dir = Path("./model")
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "csnm_vit3d.pth")
    
    plot_loss_curves(train_losses)
    
    print("Testing model...")
    test_model(test_dataset, config["batch_size"], model, device, rating_cols, criterion, config["batches_to_test"])
    print("Testing completed!")