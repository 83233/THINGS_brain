# 导入必要的库和模块
import torch                          # PyTorch核心库
import torch.nn as nn                 # 神经网络模块
import numpy as np                    # 数值计算库
from torch.utils.data import DataLoader, SubsetRandomSampler  # 数据加载和子采样工具
import torchvision.transforms as T    # 图像变换工具（虽用于图像，但数据集可能关联图像元数据）
from torch.utils.data import random_split  # 数据集随机划分工具
import matplotlib.pyplot as plt       # 可视化库
from tqdm import tqdm                 # 进度条工具
from pathlib import Path              # 路径管理工具

# 自定义数据集和可视化工具（需用户自行实现或确保路径正确）
from THINGSfastdataset import THINGSfastDataset  # 自定义fMRI数据集类
from visualizing_1 import plot_loss_curves          # 自定义损失曲线绘制函数


# ---------------------- 3D Vision Transformer 模型定义 ----------------------
class ViT3D(nn.Module):
    """
    3D视觉Transformer模型，用于fMRI数据的回归任务（预测12维评分）
    
    参数：
        input_shape: 输入fMRI数据的三维形状 (D, H, W)（深度×高度×宽度）
        patch_size: 每个patch的三维尺寸 (D_p, H_p, W_p)（需能整除input_shape）
        embed_dim: patch嵌入后的特征维度
        num_heads: Transformer多头注意力的头数
        num_layers: Transformer编码器层数
        num_classes: 输出的评分维度（本任务为12维）
    """
    def __init__(self, 
                 input_shape=(72, 91, 75),
                 patch_size=(12, 13, 15),
                 embed_dim=512,
                 num_heads=8,
                 num_layers=6,
                 num_classes=12):
        super().__init__()
        
        # 检查输入尺寸是否能被patch尺寸整除（否则无法均匀分割patch）
        assert all([i % p == 0 for i, p in zip(input_shape, patch_size)]), \
            "Input dimensions must be divisible by patch size"  # 断言确保数据可分
        
        self.input_shape = input_shape  # 保存输入形状
        self.patch_size = patch_size    # 保存patch尺寸
        
        # 计算patch的网格数量（每个维度分割的块数）和总patch数
        self.grid_size = tuple([i // p for i, p in zip(input_shape, patch_size)])  # (D块数, H块数, W块数)
        self.num_patches = np.prod(self.grid_size)  # 总patch数 = D块数 × H块数 × W块数
        
        # 3D Patch嵌入层：通过3D卷积将输入分割为patch并嵌入到高维空间
        # 输入通道为1（fMRI数据单通道），输出通道为embed_dim（嵌入维度）
        # 卷积核和步长均为patch_size，确保每个patch独立处理（无重叠）
        self.patch_embed = nn.Conv3d(
            in_channels=1,              # fMRI数据为单通道
            out_channels=embed_dim,     # 嵌入后的维度
            kernel_size=self.patch_size,  # 卷积核大小等于patch尺寸
            stride=self.patch_size       # 步长等于patch尺寸（无重叠）
        )
        
        # 可学习的cls token（用于全局特征表示，类似BERT的[CLS]）
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # 形状：[1, 1, embed_dim]
        
        # 位置编码（用于区分不同patch的位置信息，避免丢失空间位置）
        # +1是因为需要包含cls token的位置
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))  # 形状：[1, num_patches+1, embed_dim]
        
        # Transformer编码器层（批量优先模式，输入形状为[batch, seq_len, embed_dim]）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,            # 输入特征维度（与嵌入维度一致）
            nhead=num_heads,              # 多头注意力头数
            dim_feedforward=4*embed_dim,  # 前馈网络隐藏层维度（通常为4倍d_model）
            dropout=0.1,                  # Dropout正则化（防止过拟合）
            batch_first=True              # 输入形状为[batch, seq_len, embed_dim]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # 堆叠多层编码器
        
        # 回归头：将cls token的特征映射到12维评分
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),      # 层归一化（稳定训练）
            nn.Linear(embed_dim, num_classes)  # 线性层输出12维评分
        )

    def forward(self, x):
        """
        前向传播流程
        
        参数：
            x: 输入fMRI数据，形状 [batch_size, 1, D, H, W]（1为通道数）
        
        返回：
            预测的12维评分，形状 [batch_size, 12]
        """
        batch_size = x.shape[0]  # 获取批次大小
        
        # 1. 提取patch嵌入
        x = self.patch_embed(x)  # 输出形状：[batch_size, embed_dim, D', H', W']（D'=D/patch_size[0]等）
        x = x.flatten(2)         # 展平空间维度（将D', H', W'合并为序列长度），形状：[batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)    # 调整维度顺序为[batch_size, num_patches, embed_dim]（序列长度为num_patches）
        
        # 2. 添加cls token和位置编码
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # 复制cls token到当前批次（形状：[batch_size, 1, embed_dim]）
        x = torch.cat((cls_tokens, x), dim=1)  # 拼接cls token（形状：[batch_size, num_patches+1, embed_dim]）
        x = x + self.pos_embed  # 加上位置编码（区分patch位置信息）
        
        # 3. 输入Transformer编码器
        x = self.transformer(x)  # 输出形状：[batch_size, num_patches+1, embed_dim]（保留所有patch和cls token的特征）
        
        # 4. 取cls token的特征，通过回归头输出
        x = x[:, 0, :]  # 提取cls token的特征（第0个位置，形状：[batch_size, embed_dim]）
        return self.head(x)  # 输出12维评分（形状：[batch_size, 12]）


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


# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    # 超参数配置（可根据任务需求调整）
    config = {
        "input_shape": (72, 91, 75),    # fMRI数据形状（深度×高度×宽度）
        "patch_size": (12, 13, 15),     # patch尺寸（需整除input_shape：72/12=6, 91/13=7, 75/15=5）
        "embed_dim": 512,               # patch嵌入维度
        "num_heads": 8,                 # Transformer头数（多头注意力）
        "num_layers": 6,                # Transformer编码器层数
        "num_classes": 12,              # 输出评分维度（12维）
        "batch_size": 16,               # 批次大小（根据GPU显存调整）
        "lr": 1e-4,                     # 学习率（AdamW优化器参数）
        "epochs": 12,                    # 训练轮数（示例用1轮，实际需调参）
        "train_ratio": 0.8,             # 训练集比例（80%训练，20%测试）
        "batches_to_train": 32,          # 每轮训练的批次数（子采样，减少计算量）
        "batches_to_test": 12            # 测试的批次数（子采样）
    }
    
    # 初始化数据集（需确保路径正确）
    transform = T.Compose([
        T.Resize(256),                  # 图像缩放（若数据包含图像，统一尺寸）
        T.CenterCrop(224),              # 中心裁剪（提取图像主体）
        T.ToTensor(),                   # 转为张量（通道优先，范围[0,1]）
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化（使用ImageNet统计量）
    ])
    dataset = THINGSfastDataset(
        meta_npz_path="./preprocessed/things_meta.npz",  # 元数据路径（包含fMRI与评分的对应关系）
        image_root_dir="./image/_image_database_things",  # 图像根目录（若数据关联图像）
        transform=transform  # 图像变换（用于处理关联的图像数据）
    )
    
    # 划分训练集和测试集（随机划分）
    train_size = int(config["train_ratio"] * len(dataset))  # 训练集样本数
    test_size = len(dataset) - train_size  # 测试集样本数
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])  # 随机划分
    
    # 初始化设备（优先使用GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型并移动到设备
    model = ViT3D(
        input_shape=config["input_shape"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"]
    ).to(device)  # 模型移动到GPU/CPU
    
    # 定义目标评分的列名（12维，对应数据集中的评分字段）
    rating_cols = [
        'image-label_nameability_mean','image-label_consistency_mean',
        'property_manmade_mean','property_precious_mean','property_lives_mean',
        'property_heavy_mean','property_natural_mean','property_moves_mean',
        'property_grasp_mean','property_hold_mean','property_be-moved_mean','property_pleasant_mean'
    ]
    
    # 定义损失函数（均方误差，用于回归任务）和优化器（AdamW，带权重衰减的Adam）
    criterion = nn.MSELoss()  # MSE损失：(预测值-真实值)^2的平均
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)  # 权重衰减正则化
    
    # 开始训练
    print("Starting training...")
    train_losses = train_model(
        train_dataset,
        config["batch_size"],
        model,
        criterion,
        optimizer,
        device,
        rating_cols,
        config["epochs"],
        config["batches_to_train"]
    )
    print("Training completed!")
    
    # 保存模型（仅保存参数，需同时记录配置）
    model_dir = Path("./model")  # 模型保存目录
    model_dir.mkdir(parents=True, exist_ok=True)  # 创建目录（若不存在）
    torch.save(model.state_dict(), model_dir / "vit3d.pth")  # 保存模型参数
    
    # 绘制损失曲线（调用自定义函数，需确保visualizing模块存在）
    plot_loss_curves(train_losses)
    
    # 测试模型
    print("Testing model...")
    test_model(
        test_dataset,
        config["batch_size"],
        model,
        device,
        rating_cols,
        criterion,
        config["batches_to_test"]
    )
    print("Testing completed!")