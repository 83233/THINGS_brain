import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

from THINGSfastdataset import THINGSfastDataset
from visulizing import plot_loss_curves

# 3D Vision Transformer模型
class ViT3D(nn.Module):
    def __init__(self, 
                 input_shape=(72, 91, 75),
                 patch_size=(12, 13, 15),
                 embed_dim=512,
                 num_heads=8,
                 num_layers=6,
                 num_classes=12):
        super().__init__()
        
        # 验证输入尺寸能被patch尺寸整除
        assert all([i % p == 0 for i, p in zip(input_shape, patch_size)]), \
            "Input dimensions must be divisible by patch size"
        
        self.input_shape = input_shape
        self.patch_size = patch_size
        
        # 计算patch数量和各维度分割数
        self.grid_size = tuple([i // p for i, p in zip(input_shape, patch_size)])
        self.num_patches = np.prod(self.grid_size)
        # patch_volume = np.prod(patch_size)
        
        # 网络结构
        self.patch_embed = nn.Conv3d(
                                    in_channels=1,
                                    out_channels=embed_dim,
                                    kernel_size=self.patch_size,
                                    stride=self.patch_size
                                )  # :contentReference[oaicite:0]{index=0}
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4*embed_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,enable_nested_tensor=True)
        
        # 回归头
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x, return_attn = False):
        batch_size = x.shape[0]
        
        # x: [B, 1, D, H, W]
        x = self.patch_embed(x)             # [B, E, D', H', W']
        # 展平并转置到 [B, N_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)    # :contentReference[oaicite:1]{index=1}
        
        # 添加 cls token & 位置编码
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        attn_weights = []
        for layer in self.transformer.layers:
            # 利用原生 TransformerEncoderLayer，需手动提取 attn
            # 这里我们 monkey-patch multihead_attention to return weights
            src2, attn = layer.self_attn(x, x, x, need_weights=True)
            attn_weights.append(attn)  # attn: [B, heads, N, N]
            x = layer.norm1(x + layer.dropout1(src2))
            x = layer.norm2(x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))))
        
        # Transformer处理
        x = self.transformer(x)
        
        # 取cls token进行回归
        x = x[:, 0, :]
        if return_attn:
            return self.head(x), attn_weights
        return self.head(x)

# 训练函数
def train_model(train_dataset, batch_size, model, criterion, optimizer, device, rating_cols, epochs=10, batches_per_epoch=32):
    """
    train_dataloader: 原始 Dataset 对象
    batch_size:      DataLoader 的 batch_size
    batches_per_epoch: 每个 epoch 抽取的小批次数
    """
    train_losses = []
    model.train()
    dataset_size = len(train_dataset)

    for epoch in range(epochs):
        # 1. 随机抽取 batches_per_epoch * batch_size 个样本索引
        num_samples = batches_per_epoch * batch_size
        indices = torch.randperm(dataset_size)[:num_samples].tolist()
        sampler = SubsetRandomSampler(indices)
        
        # 2. 基于子采样 sampler 构造 DataLoader
        loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=4,  
                            pin_memory=True)
        
        total_loss = 0.0
        
        # 使用tqdm进度条显示训练进度
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            data = batch['fmri'].to(device)
            # targets = batch['image-label_nameability_mean'].to(device).float().view(-1, 1)

            # stack 出 [batch_size, 12]
            targets = torch.stack([
                batch[col].to(device).float()
                for col in rating_cols
            ], dim=1)  # shape: (B,12)


            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 每隔若干 batch 更新 bar 的 postfix
            if batch_idx % 12 == 0:
                pbar.write(f"Epoch: {epoch+1:03d}/{epochs} | "
                           f"Batch: {batch_idx:03d}/{batches_per_epoch} | "
                           f"Loss: {loss.item():.4f}")
            pbar.set_postfix_str(f"loss={loss.item():.4f}")
            
        # 进行可视化预测
        # visualize_fmri_slice(batch, model, device, rating_cols)
        # 学习器调度
        scheduler.step(avg_loss)
        # 每个 epoch 结束后打印平均损失
        avg_loss = total_loss / batches_per_epoch
        print(f"\nEpoch: {epoch+1:03d}/{epochs} | Avg Loss: {avg_loss:.4f}\n")
        train_losses.append(avg_loss)

        # test_model(train_dataloader, model, device)

    return train_losses

def test_model(test_dataset,batch_size, model, device, rating_cols, criterion, batches_to_test=16):
    model.eval()
    total_test_loss = 0.0
    total_error = 0.0
    dataset_size = len(test_dataset)
    with torch.no_grad():
        # 使用 tqdm 可视化测试进度
        # pbar = tqdm(test_loader, desc="Testing", unit="batch")
        # for batch_idx, batch in enumerate(pbar):
        # 只测试固定数量的 batch
        num_samples = batches_to_test * batch_size
        indices = torch.randperm(dataset_size)[:num_samples].tolist()
        sampler = SubsetRandomSampler(indices)

        loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=4,
                            pin_memory=True)
        
        pbar = tqdm(loader, desc="Testing", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            
            data = batch['fmri'].to(device)
            targets = torch.stack(
                [batch[col].to(device).float() for col in rating_cols],
                dim=1
                )
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_test_loss += loss.item()
            total_error += torch.abs(outputs - targets).mean().item()
            if batch_idx % batches_to_test == 0:
                tqdm.write(f"Test Batch {batch_idx}/{batches_to_test} | "
                           f"Loss: {loss.item():.4f}")
    avg_test_loss = total_test_loss / batches_to_test
    avg_test_error = total_error / batches_to_test
    
    # 打印测试结果
    print(f"\nTest Loss: {avg_test_loss:.4f} | "
          f"Test Avg Error: {avg_test_error:.4f}")
    visualize_fmri_slice(batch, model, device, rating_cols)

def visualize_fmri_slice(batch, model, device,rating_cols, z_slice=30):
    was_training = model.training

    model.eval()
    data = batch['fmri'].to(device)
    targets = torch.stack([batch[col].to(device) for col in rating_cols], dim=1)  # shape (B,12)
    image_paths = batch['image_path']
    
    with torch.no_grad():
        outputs = model(data)
    
    # 取第一个样本
    fmri_slice = data[0, 0, :, :, z_slice].cpu().numpy()
    pred_vec = outputs[0].cpu().numpy()   # (12,)
    true_vec = targets[0].cpu().numpy()   # (12,)
    image_path = image_paths[0]
    
    # 绘制 fMRI 切片
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(fmri_slice, cmap='gray')
    plt.title(f"z={z_slice} Slice")

    # 绘制 12 维评分对比条形图
    plt.subplot(1, 2, 2)
    x = range(len(rating_cols))
    plt.bar([i-0.2 for i in x], true_vec, width=0.4, label='True')
    plt.bar([i+0.2 for i in x], pred_vec, width=0.4, label='Pred')
    plt.xticks(x, rating_cols, rotation=90)
    plt.ylabel('Rating')
    plt.title('True vs Predicted Ratings')
    plt.legend()
    plt.tight_layout()
    plt.show()
    if was_training:
        model.train()

# 主程序
if __name__ == "__main__":
    # 超参数
    split_path = "./preprocessed/dataset_splits.npz"
    config = {
        "input_shape": (72, 91, 75),
        "patch_size": (12, 13, 15),
        "embed_dim": 512,
        "num_heads": 8,
        "num_layers": 6,
        "num_classes": 12,
        "batch_size": 16,
        "lr": 1e-4,
        "epochs": 10,
        "train_ratio": 0.8,
        "batches_to_train": 64,
        "batches_to_test": 16
    }
    
    # 初始化数据集和数据加载器
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                     std=[0.229, 0.224, 0.225])
    ])
    # image\01_image-level\image-paths.csv
    # sub-01_ses-things01_run-01_conditions.tsv
    # image\02_object-level\_concepts-metadata_things.tsv
    dataset = THINGSfastDataset(
        meta_npz_path="./preprocessed/things_meta.npz",
        image_root_dir="./image/_image_database_things",
        transform=transform
    )
    
    if os.path.exists(split_path):
    # 已存在划分索引，直接加载
        print("Loading existing dataset splits...")
        split = np.load(split_path)
        # 转为 Python list，确保每个索引都是内置 int
        train_indices = split['train_indices'].tolist()
        val_indices   = split['val_indices'].tolist()
        test_indices  = split['test_indices'].tolist()
    else:
        # 创建新的划分并保存
        print("Creating new dataset splits...")
        num_samples = len(dataset)
        indices = np.random.permutation(num_samples)

        train_end = int(config["train_ratio"] * num_samples)
        val_end   = int((config["train_ratio"] + 0.1) * num_samples)

        # 先用 numpy array 截取，然后马上转成 Python list 防止出现报错
        train_indices = indices[:train_end].tolist()
        val_indices   = indices[train_end:val_end].tolist()
        test_indices  = indices[val_end:].tolist()

        # 保存索引
        np.savez(split_path, 
                 train_indices=np.array(train_indices, dtype=int),
                 val_indices=np.array(val_indices, dtype=int),
                 test_indices=np.array(test_indices, dtype=int))
        print("Dataset splits saved!")

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ViT3D(
        input_shape=config["input_shape"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"]
    ).to(device)

    # 12个评分定义
    rating_cols = [
                'image-label_nameability_mean','image-label_consistency_mean',
                'property_manmade_mean','property_precious_mean','property_lives_mean',
                'property_heavy_mean','property_natural_mean','property_moves_mean',
                'property_grasp_mean','property_hold_mean','property_be-moved_mean','property_pleasant_mean'
            ]
    
    # 定义损失函数和优化器
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                              factor=0.5, patience=3,
                              threshold=1e-4, cooldown=1,
                              min_lr=1e-7) 
    
    # 开始训练
    print("Starting training...")
    train_losses = train_model(train_dataset,config["batch_size"], model, criterion, optimizer,
                               device, rating_cols, config["epochs"], config["batches_to_train"])
    print("Training completed!")

    # 保存模型
    torch.save(model.state_dict(), "./results/model/vit3d.pth")

    plot_loss_curves(train_losses)

    print("Testing model...")
    test_model(test_dataset,config["batch_size"], model, device, 
               rating_cols, criterion, config["batches_to_test"])
    print("Testing completed!")

