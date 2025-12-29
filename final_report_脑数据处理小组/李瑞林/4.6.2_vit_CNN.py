# 导入必要的库和模块
import torch                      # PyTorch深度学习框架
import torch.nn as nn             # 神经网络模块（如层、损失函数）
import numpy as np                # 数值计算库
from torch.utils.data import DataLoader, SubsetRandomSampler  # 数据加载和子采样工具
import torchvision.transforms as T  # 图像变换工具（如归一化、裁剪）
from torch.utils.data import random_split  # 数据集随机划分工具
import matplotlib.pyplot as plt   # 可视化库（绘制损失曲线、fMRI切片等）
from tqdm import tqdm             # 进度条工具（显示训练/测试进度）
from pathlib import Path          # 路径管理工具（创建目录、保存模型）

# 导入自定义模块（需用户根据实际路径调整）
from THINGSfastdataset import THINGSfastDataset  # 自定义fMRI数据集类
from visualizing_2 import plot_loss_curves          # 自定义损失曲线绘制函数


# ---------------------- 3D CNN特征提取器 ----------------------
class CNN3D(nn.Module):
    """
    3D卷积神经网络，用于提取fMRI数据的局部特征。
    输入：单通道fMRI体积数据（形状：[batch_size, 1, D, H, W]）
    输出：降维后的局部特征（形状：[batch_size, 64, D/4, H/4, W/4]）
    """
    def __init__(self, input_channels=1):
        """
        参数：
            input_channels: 输入数据的通道数（fMRI为单通道，默认1）
        """
        super().__init__()  # 调用父类（nn.Module）初始化
        
        # 第一层卷积块：3D卷积 → 批归一化 → ReLU激活 → 最大池化
        self.conv1 = nn.Conv3d(
            in_channels=input_channels,  # 输入通道数（fMRI为1）
            out_channels=32,             # 输出通道数（提取32种局部特征）
            kernel_size=3,               # 卷积核尺寸（3×3×3）
            stride=1,                    # 步长（不跳过像素）
            padding=1                    # 填充（保持空间尺寸不变）
        )
        self.bn1 = nn.BatchNorm3d(32)    # 批归一化（稳定训练，加速收敛）
        self.relu1 = nn.ReLU()           # 非线性激活（引入非线性特征）
        
        # 第二层卷积块：3D卷积 → 批归一化 → ReLU激活 → 最大池化
        self.conv2 = nn.Conv3d(
            in_channels=32,              # 输入通道数（上一层输出32）
            out_channels=64,             # 输出通道数（提取64种更复杂特征）
            kernel_size=3,               # 卷积核尺寸（3×3×3）
            stride=1,                    # 步长（不跳过像素）
            padding=1                    # 填充（保持空间尺寸不变）
        )
        self.bn2 = nn.BatchNorm3d(64)    # 批归一化
        self.relu2 = nn.ReLU()           # 非线性激活
        
        # 最大池化层：空间下采样（尺寸减半）
        self.pool = nn.MaxPool3d(
            kernel_size=2,               # 池化核尺寸（2×2×2）
            stride=2                     # 步长（等于核尺寸，无重叠）
        )

    def forward(self, x):
        """
        前向传播（特征提取）
        参数：
            x: 输入fMRI数据（形状：[batch_size, 1, D, H, W]）
        返回：
            提取的局部特征（形状：[batch_size, 64, D/4, H/4, W/4]）
        """
        # 第一层卷积块处理
        x = self.conv1(x)         # 卷积：[B, 1, D, H, W] → [B, 32, D, H, W]
        x = self.bn1(x)           # 批归一化（标准化特征分布）
        x = self.relu1(x)         # 激活：引入非线性
        x = self.pool(x)          # 池化：空间尺寸减半 → [B, 32, D/2, H/2, W/2]
        
        # 第二层卷积块处理
        x = self.conv2(x)         # 卷积：[B, 32, D/2, H/2, W/2] → [B, 64, D/2, H/2, W/2]
        x = self.bn2(x)           # 批归一化
        x = self.relu2(x)         # 激活
        x = self.pool(x)          # 池化：空间尺寸再次减半 → [B, 64, D/4, H/4, W/4]
        
        return x  # 输出CNN提取的局部特征


# ---------------------- CNN+3D Vision Transformer 模型 ----------------------
class CNN_ViT3D(nn.Module):
    """
    结合3D CNN和Vision Transformer的模型，用于fMRI数据的12维评分回归任务。
    流程：CNN提取局部特征 → 特征分块嵌入 → Transformer建模全局依赖 → 回归头输出评分
    """
    def __init__(self, 
                 input_shape=(72, 91, 75),  # 输入fMRI的三维尺寸（D, H, W）
                 patch_size=(6, 11, 6),     # Transformer的patch尺寸（D_p, H_p, W_p）
                 embed_dim=512,             # patch嵌入维度（特征向量长度）
                 num_heads=8,               # Transformer多头注意力的头数
                 num_layers=6,              # Transformer编码器层数
                 num_classes=12):           # 输出评分维度（12维）
        """
        参数：
            input_shape: 输入fMRI数据的三维尺寸（如72×91×75）
            patch_size: 分块的三维尺寸（需与CNN输出尺寸匹配）
            embed_dim: patch嵌入后的特征维度
            num_heads: Transformer注意力头数（多头注意力并行计算）
            num_layers: Transformer编码器层数（层数越多，模型复杂度越高）
            num_classes: 输出评分的维度（12个评分）
        """
        super().__init__()
        
        # 初始化3D CNN特征提取器（预处理模块）
        self.cnn = CNN3D(input_channels=1)  # 输入fMRI为单通道
        
        # 计算CNN输出的空间尺寸（两次池化后，原尺寸除以4）
        cnn_output_shape = (
            input_shape[0] // 4,  # 深度方向尺寸（原72 → 72/4=18）
            input_shape[1] // 4,  # 高度方向尺寸（原91 → 91/4≈22）
            input_shape[2] // 4   # 宽度方向尺寸（原75 → 75/4≈18）
        )
        self.cnn_output_shape = cnn_output_shape  # 保存CNN输出尺寸
        
        # 验证：CNN输出尺寸必须能被patch_size整除（确保均匀分块）
        assert all([i % p == 0 for i, p in zip(cnn_output_shape, patch_size)]), \
            "CNN输出尺寸必须能被patch尺寸整除（否则无法均匀分块）"
        
        # 计算分块的网格数量（每个维度的块数）和总patch数
        self.grid_size = tuple([i // p for i, p in zip(cnn_output_shape, patch_size)])  # (18/6, 22/11, 18/6) → (3, 2, 3)
        self.num_patches = np.prod(self.grid_size)  # 总patch数 = 3×2×3 = 18
        
        # 3D Patch嵌入层（将CNN输出的局部特征分块并嵌入到高维空间）
        self.patch_embed = nn.Conv3d(
            in_channels=64,          # 输入通道数（CNN输出的64通道特征）
            out_channels=embed_dim,  # 输出通道数（嵌入后的特征维度）
            kernel_size=patch_size,  # 卷积核尺寸（等于patch尺寸，确保每个patch独立处理）
            stride=patch_size        # 步长（等于patch尺寸，无重叠分块）
        )
        
        # 可学习的分类token（cls_token）：用于全局特征表示
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # 形状：[1, 1, embed_dim]
        
        # 位置编码（学习每个patch的位置信息，避免顺序丢失）
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))  # +1为cls_token
        
        # Transformer编码器层（批量优先模式）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,          # 输入特征维度（与嵌入维度一致）
            nhead=num_heads,            # 注意力头数（多头注意力并行计算）
            dim_feedforward=4*embed_dim,  # 前馈网络隐藏层维度（通常为嵌入维度的4倍）
            dropout=0.0,                # 关闭Dropout（避免正则化干扰对比实验）
            batch_first=True            # 输入形状为[batch_size, seq_len, embed_dim]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # 堆叠多层编码器
        
        # 回归头（将cls_token的特征映射到12维评分）
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),  # 层归一化（稳定特征分布）
            nn.Linear(embed_dim, num_classes)  # 线性层（输出12维评分）
        )

    def forward(self, x):
        """
        前向传播流程
        参数：
            x: 输入fMRI数据（形状：[batch_size, 1, D, H, W]）
        返回：
            预测的12维评分（形状：[batch_size, 12]）
        """
        batch_size = x.shape[0]  # 获取批次大小
        
        # 1. CNN提取局部特征（预处理）
        x = self.cnn(x)  # 输出形状：[batch_size, 64, D/4, H/4, W/4]
        
        # 2. 分块嵌入（将CNN特征分块并转换为序列）
        x = self.patch_embed(x)  # 输出形状：[batch_size, embed_dim, grid_D, grid_H, grid_W]（如[B, 512, 3, 2, 3]）
        x = x.flatten(2)         # 展平空间维度 → [batch_size, embed_dim, num_patches]（如[B, 512, 18]）
        x = x.transpose(1, 2)    # 调整为序列格式 → [batch_size, num_patches, embed_dim]（如[B, 18, 512]）
        
        # 3. 添加cls_token和位置编码
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # 复制cls_token到当前批次（形状：[B, 1, 512]）
        x = torch.cat((cls_tokens, x), dim=1)  # 拼接cls_token（形状：[B, 19, 512]）
        x = x + self.pos_embed  # 添加位置编码（区分不同patch的位置）
        
        # 4. 输入Transformer编码器（建模全局依赖）
        x = self.transformer(x)  # 输出形状：[B, 19, 512]（每个位置的特征）
        
        # 5. 提取cls_token的特征，通过回归头输出评分
        x = x[:, 0, :]  # 取cls_token的特征（形状：[B, 512]）
        return self.head(x)  # 输出12维评分（形状：[B, 12]）


# ---------------------- 模型训练函数 ----------------------
def train_model(train_dataset, batch_size, model, criterion, optimizer, device, rating_cols, epochs=10, batches_per_epoch=32):
    """
    训练模型（使用子采样加速训练，每轮仅训练部分批次）
    参数：
        train_dataset: 训练数据集（THINGSfastDataset实例）
        batch_size: 每批次样本数（显存限制时需调小）
        model: 待训练的模型（如CNN_ViT3D）
        criterion: 损失函数（如MSE）
        optimizer: 优化器（如AdamW）
        device: 计算设备（GPU/CPU）
        rating_cols: 目标评分的列名（12维）
        epochs: 训练轮数（遍历数据集的次数）
        batches_per_epoch: 每轮训练的批次数（子采样数量）
    返回：
        train_losses: 每轮的平均训练损失（列表）
    """
    train_losses = []  # 记录每轮的平均损失
    model.train()      # 开启训练模式（激活BatchNorm等）
    dataset_size = len(train_dataset)  # 训练集总样本数

    for epoch in range(epochs):  # 遍历数据集epochs次
        # 1. 子采样策略：每轮随机选择batches_per_epoch×batch_size个样本
        num_samples = batches_per_epoch * batch_size  # 每轮训练的总样本数
        indices = torch.randperm(dataset_size)[:num_samples].tolist()  # 随机选择索引
        sampler = SubsetRandomSampler(indices)  # 子采样器（随机采样指定索引）
        
        # 2. 构建数据加载器（按批次加载数据）
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,       # 每批次样本数
            sampler=sampler,             # 使用子采样器（仅训练部分样本）
            num_workers=4,               # 多进程加载数据（加速）
            pin_memory=True              # 锁页内存（加速GPU数据传输）
        )
        
        total_loss = 0.0  # 累计每轮的总损失
        
        # 3. 训练进度条（显示当前轮次和批次进度）
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            # 加载fMRI数据和对应的12维评分（从批次字典中提取）
            data = batch['fmri'].to(device)  # fMRI数据（形状：[B, 1, D, H, W]）
            # 拼接12维目标评分（从字典中按列名提取，转换为张量）
            targets = torch.stack([
                batch[col].to(device).float()  # 转换为浮点型并移动到设备（GPU/CPU）
                for col in rating_cols  # 遍历12个评分列名
            ], dim=1)  # 形状：[B, 12]（批次大小×评分维度）
            
            # 前向传播：输入数据，得到预测评分
            outputs = model(data)  # 预测评分（形状：[B, 12]）
            
            # 计算损失：预测评分与真实评分的MSE（均方误差）
            loss = criterion(outputs, targets)  # 标量损失值
            
            # 反向传播：计算梯度并更新参数
            optimizer.zero_grad()  # 清空之前的梯度
            loss.backward()        # 计算当前损失的梯度
            optimizer.step()       # 根据梯度更新模型参数
            
            # 累计损失（用于计算每轮平均损失）
            total_loss += loss.item()  # 提取损失的标量值（避免梯度累积）
            
            # 打印中间日志（每12个批次）
            if batch_idx % 12 == 0:
                pbar.write(f"Epoch: {epoch+1:03d}/{epochs} | Batch: {batch_idx:03d}/{batches_per_epoch} | Loss: {loss.item():.4f}")
            
            # 进度条显示当前批次的损失
            pbar.set_postfix_str(f"loss={loss.item():.4f}")
        
        # 计算并记录每轮的平均损失
        avg_loss = total_loss / batches_per_epoch  # 总损失 / 批次数
        print(f"\nEpoch: {epoch+1:03d}/{epochs} | Avg Loss: {avg_loss:.4f}\n")
        train_losses.append(avg_loss)  # 保存当前轮次的平均损失

    return train_losses  # 返回各轮次的平均损失


# ---------------------- 模型测试函数 ----------------------
def test_model(test_dataset, batch_size, model, device, rating_cols, criterion, batches_to_test=16):
    """
    测试模型（评估训练后的模型在测试集上的性能）
    参数：
        test_dataset: 测试数据集（THINGSfastDataset实例）
        batch_size: 每批次样本数（与训练一致）
        model: 已训练的模型（如CNN_ViT3D）
        device: 计算设备（GPU/CPU）
        rating_cols: 目标评分的列名（12维）
        criterion: 损失函数（如MSE）
        batches_to_test: 测试的批次数（子采样数量）
    返回：
        avg_test_loss: 平均测试损失
    """
    model.eval()  # 开启评估模式（关闭Dropout、BatchNorm的更新）
    total_test_loss = 0.0  # 累计测试总损失
    total_error = 0.0      # 累计绝对误差（用于评估预测准确性）
    dataset_size = len(test_dataset)  # 测试集总样本数

    with torch.no_grad():  # 关闭梯度计算（加速测试，节省显存）
        # 1. 子采样策略：随机选择batches_to_test×batch_size个测试样本
        num_samples = batches_to_test * batch_size  # 测试的总样本数
        indices = torch.randperm(dataset_size)[:num_samples].tolist()  # 随机选择索引
        sampler = SubsetRandomSampler(indices)  # 子采样器
        
        # 2. 构建测试数据加载器
        loader = DataLoader(
            test_dataset,
            batch_size=batch_size,       # 每批次样本数
            sampler=sampler,             # 使用子采样器（仅测试部分样本）
            num_workers=4,               # 多进程加载数据
            pin_memory=True              # 锁页内存（加速GPU数据传输）
        )
        
        # 3. 测试进度条（显示测试批次进度）
        pbar = tqdm(loader, desc="Testing", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            # 加载fMRI数据和对应的12维评分
            data = batch['fmri'].to(device)  # fMRI数据（形状：[B, 1, D, H, W]）
            targets = torch.stack([
                batch[col].to(device).float()  # 转换为浮点型并移动到设备
                for col in rating_cols  # 遍历12个评分列名
            ], dim=1)  # 形状：[B, 12]（真实评分）
            
            # 前向传播：输入数据，得到预测评分
            outputs = model(data)  # 预测评分（形状：[B, 12]）
            
            # 计算损失和绝对误差
            loss = criterion(outputs, targets)  # MSE损失（标量）
            total_test_loss += loss.item()      # 累计损失
            total_error += torch.abs(outputs - targets).mean().item()  # 累计绝对误差（平均每个评分的绝对偏差）
            
            # 打印中间日志（每batches_to_test个批次）
            if batch_idx % batches_to_test == 0:
                tqdm.write(f"Test Batch {batch_idx}/{batches_to_test} | Loss: {loss.item():.4f}")
    
    # 计算平均测试损失和平均绝对误差
    avg_test_loss = total_test_loss / batches_to_test  # 平均测试损失
    avg_test_error = total_error / batches_to_test     # 平均绝对误差（评分的平均绝对偏差）
    
    print(f"\nTest Loss: {avg_test_loss:.4f} | Test Avg Error: {avg_test_error:.4f}")
    
    # 可视化最后一个批次的预测结果（调用自定义函数）
    visualize_fmri_slice(batch, model, device, rating_cols)

    return avg_test_loss  # 返回平均测试损失


# ---------------------- 预测结果可视化函数 ----------------------
def visualize_fmri_slice(batch, model, device, rating_cols, z_slice=30):
    """
    可视化fMRI切片和预测-真实评分对比图（辅助分析模型性能）
    参数：
        batch: 当前测试批次数据（包含fMRI、图像路径、评分等）
        model: 已训练的模型（用于生成预测）
        device: 计算设备（GPU/CPU）
        rating_cols: 评分列名（12维）
        z_slice: 可视化的z轴切片位置（深度方向）
    """
    was_training = model.training  # 记录模型原模式（训练/评估）
    model.eval()  # 临时切换到评估模式（避免Dropout等影响预测）
    
    # 加载数据和目标（从批次中提取）
    data = batch['fmri'].to(device)  # fMRI数据（形状：[B, 1, D, H, W]）
    targets = torch.stack([
        batch[col].to(device)  # 真实评分（转换为张量并移动到设备）
        for col in rating_cols
    ], dim=1)  # 形状：[B, 12]（真实评分）
    image_paths = batch['image_path']  # 对应的图像路径（用于标注）
    
    with torch.no_grad():  # 关闭梯度计算（加速预测）
        outputs = model(data)  # 预测评分（形状：[B, 12]）
    
    # 提取第一个样本的fMRI切片和评分（用于可视化）
    fmri_slice = data[0, 0, :, :, z_slice].cpu().numpy()  # 提取z_slice层的切片（形状：[D, H]）
    pred_vec = outputs[0].cpu().numpy()  # 预测评分（12维向量）
    true_vec = targets[0].cpu().numpy()  # 真实评分（12维向量）
    image_path = image_paths[0]          # 对应的图像路径（用于标题）
    
    # 绘制fMRI切片和评分对比图（两个子图）
    plt.figure(figsize=(12, 6))  # 设置画布大小（12英寸宽，6英寸高）
    
    # 子图1：fMRI切片（z轴第z_slice层）
    plt.subplot(1, 2, 1)  # 1行2列，第1个子图
    plt.imshow(fmri_slice, cmap='gray')  # 灰度图显示切片
    plt.title(f"fMRI Slice (z={z_slice}) | Image: {image_path}")  # 标题（包含图像路径）
    plt.xlabel("Width")  # x轴标签（宽度方向）
    plt.ylabel("Height")  # y轴标签（高度方向）
    
    # 子图2：预测-真实评分对比（条形图）
    plt.subplot(1, 2, 2)  # 1行2列，第2个子图
    x = range(len(rating_cols))  # 评分维度索引（0~11）
    plt.bar([i-0.2 for i in x], true_vec, width=0.4, label='True', color='blue')  # 真实评分（蓝色）
    plt.bar([i+0.2 for i in x], pred_vec, width=0.4, label='Pred', color='orange')  # 预测评分（橙色）
    plt.xticks(x, rating_cols, rotation=90)  # x轴刻度（评分列名，旋转90度避免重叠）
    plt.ylabel('Rating Value')  # y轴标签（评分值）
    plt.title('True vs Predicted Ratings')  # 标题
    plt.legend()  # 显示图例（True/Pred）
    plt.tight_layout()  # 调整布局（避免子图重叠）
    plt.savefig('trans_CNN.png')  # 保存可视化结果到本地文件
    plt.close()  # 关闭画布（释放内存）
    
    # 恢复模型原模式（如果之前是训练模式）
    if was_training:
        model.train()


# ---------------------- 主程序（模型训练与测试流程） ----------------------
if __name__ == "__main__":
    # ---------------------- 超参数配置 ----------------------
    # 固定超参数（与原始ViT3D模型对齐，确保对比公平）
    config = {
        "input_shape": (72, 91, 75),    # 输入fMRI数据的三维尺寸（深度×高度×宽度）
        "patch_size": (6, 11, 6),       # Transformer的patch尺寸（与CNN输出匹配，确保均匀分块）
        "embed_dim": 512,               # patch嵌入维度（特征向量长度，与原始ViT一致）
        "num_heads": 8,                 # Transformer注意力头数（与原始ViT一致）
        "num_layers": 6,                # Transformer编码器层数（与原始ViT一致）
        "num_classes": 12,              # 输出评分维度（12个评分）
        "batch_size": 16,                # 批次大小（受显存限制，与原始ViT子采样策略一致）
        "lr": 1e-4,                     # 学习率（与原始ViT优化器一致）
        "epochs": 1,                   # 训练轮数（示例用小值，实际需根据收敛情况调整）
        "train_ratio": 0.8,             # 训练集比例（80%训练，20%测试）
        "batches_to_train": 32,          # 每轮训练的批次数（子采样加速）
        "batches_to_test": 12            # 测试的批次数（子采样加速）
    }
    
    # ---------------------- 数据集初始化 ----------------------
    # 定义图像变换（若数据包含对应的图像，用于预处理）
    transform = T.Compose([
        T.Resize(256),                  # 图像缩放（长边缩放到256）
        T.CenterCrop(224),              # 中心裁剪（224×224，与ImageNet标准一致）
        T.ToTensor(),                   # 转换为张量（0~1范围）
        T.Normalize(                     # 归一化（使用ImageNet统计量，增强模型泛化性）
            mean=[0.485, 0.456, 0.406],  # 均值（红、绿、蓝通道）
            std=[0.229, 0.224, 0.225]     # 标准差（红、绿、蓝通道）
        )
    ])
    
    # 初始化自定义fMRI数据集（需用户根据实际路径调整）
    dataset = THINGSfastDataset(
        meta_npz_path="./preprocessed/things_meta.npz",  # 元数据路径（包含fMRI与图像的对应关系）
        image_root_dir="./image/_image_database_things",  # 图像根目录（存储对应的真实图像）
        transform=transform  # 图像变换（预处理图像数据）
    )
    
    # 划分训练集和测试集（按train_ratio比例随机划分）
    train_size = int(config["train_ratio"] * len(dataset))  # 训练集样本数
    test_size = len(dataset) - train_size  # 测试集样本数
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])  # 随机划分
    
    # ---------------------- 设备配置 ----------------------
    # 优先使用GPU（若可用），否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  # 打印当前使用的设备
    
    # ---------------------- 模型初始化 ----------------------
    # 定义12维评分的列名（与元数据中的列名一致）
    rating_cols = [
        'image-label_nameability_mean', 'image-label_consistency_mean',
        'property_manmade_mean', 'property_precious_mean', 'property_lives_mean',
        'property_heavy_mean', 'property_natural_mean', 'property_moves_mean',
        'property_grasp_mean', 'property_hold_mean', 'property_be-moved_mean', 'property_pleasant_mean'
    ]
    
    # 初始化CNN+ViT3D模型，并移动到目标设备（GPU/CPU）
    model = CNN_ViT3D(
        input_shape=config["input_shape"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"]
    ).to(device)
    
    # ---------------------- 训练配置 ----------------------
    # 定义损失函数（均方误差，用于回归任务）
    criterion = nn.MSELoss()  # MSE = 1/N * Σ(y_pred - y_true)²
    
    # 定义优化器（AdamW，带权重衰减的Adam，通常优于普通Adam）
    optimizer = torch.optim.AdamW(
        model.parameters(),  # 优化模型的所有可训练参数
        lr=config["lr"],     # 学习率（控制参数更新步长）
        weight_decay=0.0     # 权重衰减（正则化，防止过拟合，此处关闭以对齐原始ViT）
    )
    
    # ---------------------- 模型训练 ----------------------
    print("\n===== Starting Training =====")
    train_losses = train_model(
        train_dataset=train_dataset,    # 训练数据集
        batch_size=config["batch_size"],  # 批次大小
        model=model,                    # 待训练的模型
        criterion=criterion,            # 损失函数
        optimizer=optimizer,            # 优化器
        device=device,                  # 计算设备
        rating_cols=rating_cols,        # 目标评分列名
        epochs=config["epochs"],        # 训练轮数
        batches_per_epoch=config["batches_to_train"]  # 每轮训练的批次数
    )
    print("===== Training Completed =====")
    
    # ---------------------- 模型保存 ----------------------
    # 创建模型保存目录（若不存在）
    model_dir = Path("./model_cnn_vit")
    model_dir.mkdir(parents=True, exist_ok=True)  # parents=True：自动创建父目录；exist_ok=True：目录存在时不报错
    
    # 保存模型参数（仅保存可训练参数，不保存模型结构）
    torch.save(model.state_dict(), model_dir / "cnn_vit3d.pth")
    print(f"Model saved to: {model_dir / 'cnn_vit3d.pth'}")
    
    # ---------------------- 损失曲线绘制 ----------------------
    # 调用自定义函数绘制训练损失曲线（需visualizing模块支持）
    plot_loss_curves(train_losses)  # 输入各轮次的平均损失，输出损失曲线图像
    
    # ---------------------- 模型测试 ----------------------
    print("\n===== Starting Testing =====")
    test_model(
        test_dataset=test_dataset,    # 测试数据集
        batch_size=config["batch_size"],  # 批次大小（与训练一致）
        model=model,                  # 已训练的模型
        device=device,                # 计算设备
        rating_cols=rating_cols,      # 目标评分列名
        criterion=criterion,          # 损失函数
        batches_to_test=config["batches_to_test"]  # 测试的批次数
    )
    print("===== Testing Completed =====")