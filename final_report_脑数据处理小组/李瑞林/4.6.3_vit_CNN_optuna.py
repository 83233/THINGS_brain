# 导入必要的库和模块
import torch                          # PyTorch核心库
import torch.nn as nn                 # 神经网络模块
import numpy as np                    # 数值计算库
from torch.utils.data import DataLoader, SubsetRandomSampler  # 数据加载和子采样工具
import torchvision.transforms as T    # 图像变换工具
from torch.utils.data import random_split  # 数据集随机划分工具
import matplotlib.pyplot as plt       # 可视化库
from tqdm import tqdm                 # 进度条工具
from pathlib import Path              # 路径管理工具
import optuna                         # 超参数优化框架（自动搜索最优超参数）

# 自定义数据集和可视化工具（需用户自行实现或确保路径正确）
from THINGSfastdataset import THINGSfastDataset  # 自定义fMRI数据集类
from visualizing_3 import plot_loss_curves          # 自定义损失曲线绘制函数


# ---------------------- Optuna目标函数（超参数搜索） ----------------------
def objective(trial: optuna.Trial, fixed_config, train_dataset, test_dataset, rating_cols, device):
    """
    Optuna的目标函数，定义超参数搜索空间并返回验证损失（用于优化）
    
    参数：
        trial: Optuna的试验对象（用于建议超参数）
        fixed_config: 固定超参数（不参与搜索的参数，如输入形状、批次大小）
        train_dataset: 训练数据集
        test_dataset: 测试数据集
        rating_cols: 目标评分列名（12维）
        device: 计算设备（GPU/CPU）
    """
    # 1. 定义动态超参数搜索空间（需优化的参数）
    dynamic_params = {
        # 嵌入维度（从候选值中选择，减少搜索复杂度）
        "embed_dim": trial.suggest_categorical("embed_dim", [256, 512, 768]),
        # Transformer层数（整数范围3-8）
        "num_layers": trial.suggest_int("num_layers", 3, 8),
        # 学习率（对数空间搜索，覆盖1e-5到1e-3）
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    }
    
    # 2. 初始化模型（无Dropout和权重衰减，避免正则化干扰调参）
    model = CNN_ViT3D(
        input_shape=fixed_config["input_shape"],  # 输入形状（固定）
        patch_size=fixed_config["patch_size"],    # patch尺寸（基于CNN输出，固定）
        embed_dim=dynamic_params["embed_dim"],    # 嵌入维度（搜索得到）
        num_heads=fixed_config["num_heads"],      # 头数（固定，减少搜索维度）
        num_layers=dynamic_params["num_layers"],  # Transformer层数（搜索得到）
        num_classes=fixed_config["num_classes"],  # 输出维度（固定12）
        dropout=0.0  # 显式关闭Dropout（调参阶段先不引入正则化）
    ).to(device)  # 模型移动到设备
    
    # 3. 定义优化器（关闭权重衰减，避免干扰）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=dynamic_params["lr"],  # 学习率（搜索得到）
        weight_decay=0.0  # 无权重衰减（调参阶段先不引入正则化）
    )
    
    # 4. 训练模型（使用固定的训练轮数和批次数，快速评估）
    _ = train_model(
        train_dataset=train_dataset,
        batch_size=fixed_config["batch_size"],
        model=model,
        criterion=nn.MSELoss(),  # MSE损失（回归任务）
        optimizer=optimizer,
        device=device,
        rating_cols=rating_cols,
        epochs=fixed_config["epochs"],          # 训练轮数（固定，示例用1轮）
        batches_per_epoch=fixed_config["batches_to_train"]  # 每轮批次数（固定）
    )
    
    # 5. 验证模型并返回测试损失（Optuna根据此值优化，目标：最小化损失）
    test_loss = test_model(
        test_dataset=test_dataset,
        batch_size=fixed_config["batch_size"],
        model=model,
        device=device,
        rating_cols=rating_cols,
        criterion=nn.MSELoss(),
        batches_to_test=fixed_config["batches_to_test"]  # 测试批次数（固定）
    )
    return test_loss  # 返回测试损失（Optuna优化目标）


# ---------------------- 3D CNN特征提取器（无正则化） ----------------------
class CNN3D(nn.Module):
    """
    3D卷积神经网络，用于提取fMRI的局部特征（无Dropout，仅保留基础层）
    作用：通过卷积操作提取局部空间特征，缩小数据尺寸，为后续Transformer降维
    """
    def __init__(self, input_channels=1):
        super().__init__()
        # 卷积块1：3D卷积 → 批归一化 → ReLU激活 → 最大池化
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, stride=1, padding=1)  # 输入通道1→输出32
        self.bn1 = nn.BatchNorm3d(32)  # 批归一化（稳定训练，加速收敛）
        self.relu1 = nn.ReLU()         # 非线性激活（引入非线性特征）
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # 最大池化（缩小空间尺寸，减少计算量）
        
        # 卷积块2：3D卷积 → 批归一化 → ReLU激活 → 最大池化
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)  # 输入32→输出64（增加通道数）
        self.bn2 = nn.BatchNorm3d(64)  # 批归一化
        self.relu2 = nn.ReLU()         # 非线性激活
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # 最大池化（尺寸再次减半）

    def forward(self, x):
        """
        前向传播（特征提取）
        
        参数：
            x: 输入fMRI数据 [batch_size, 1, D, H, W]（1为通道数）
        
        返回：
            提取的特征 [batch_size, 64, D/4, H/4, W/4]（两次池化，尺寸减半两次）
        """
        x = self.relu1(self.bn1(self.conv1(x)))  # 卷积→归一化→激活（形状：[B, 32, D, H, W]）
        x = self.pool1(x)                        # 池化（尺寸减半，形状：[B, 32, D/2, H/2, W/2]）
        x = self.relu2(self.bn2(self.conv2(x)))  # 卷积→归一化→激活（形状：[B, 64, D/2, H/2, W/2]）
        x = self.pool2(x)                        # 池化（尺寸再次减半，形状：[B, 64, D/4, H/4, W/4]）
        return x


# ---------------------- CNN+3D Vision Transformer 模型 ----------------------
class CNN_ViT3D(nn.Module):
    """
    结合3D CNN和Transformer的模型（无正则化，用于Optuna调参）
    结构：CNN提取局部特征 → Transformer学习全局依赖
    
    参数：
        input_shape: 输入fMRI数据形状 (D, H, W)
        patch_size: Transformer的patch尺寸（基于CNN输出特征）
        embed_dim: patch嵌入维度
        num_heads: Transformer头数
        num_layers: Transformer层数
        num_classes: 输出评分维度（12维）
        dropout: Transformer的dropout率（调参阶段设为0）
    """
    def __init__(self, 
                 input_shape=(72, 91, 75),
                 patch_size=(6, 11, 6),
                 embed_dim=512,
                 num_heads=8,
                 num_layers=6,
                 num_classes=12,
                 dropout=0.0):
        super().__init__()
        
        # 初始化3D CNN特征提取器（输入fMRI为单通道）
        self.cnn = CNN3D(input_channels=1)  # 输出通道64，空间尺寸为input_shape/4
        
        # 计算CNN输出的空间尺寸（两次池化后尺寸减半两次，即除以4）
        cnn_output_shape = (
            input_shape[0] // 4,  # D方向尺寸（72/4=18）
            input_shape[1] // 4,  # H方向尺寸（91/4≈22.75→取整？需与patch_size匹配）
            input_shape[2] // 4   # W方向尺寸（75/4=18.75→取整？）
        )
        self.cnn_output_shape = cnn_output_shape  # 保存CNN输出形状
        
        # 检查CNN输出尺寸能否被patch_size整除（确保均匀分割patch）
        assert all([i % p == 0 for i, p in zip(cnn_output_shape, patch_size)]), \
            "CNN output dimensions must be divisible by patch size"  # 断言确保数据可分
        
        # 计算patch的网格数量和总patch数（基于CNN输出特征）
        self.grid_size = tuple([i // p for i, p in zip(cnn_output_shape, patch_size)])  # (D块数, H块数, W块数)
        self.num_patches = np.prod(self.grid_size)  # 总patch数 = D块数 × H块数 × W块数
        
        # 3D Patch嵌入层（输入通道为CNN的输出通道64，输出为embed_dim）
        self.patch_embed = nn.Conv3d(
            in_channels=64,          # CNN输出通道数（固定为64）
            out_channels=embed_dim,  # 嵌入后的维度（搜索参数）
            kernel_size=patch_size,  # 卷积核大小等于patch尺寸（基于CNN输出）
            stride=patch_size        # 步长等于patch尺寸（无重叠）
        )
        
        # 可学习的cls token（用于全局特征表示）
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # 形状：[1, 1, embed_dim]
        
        # 位置编码（包含cls token，区分patch位置）
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))  # 形状：[1, num_patches+1, embed_dim]
        
        # Transformer编码器层（无Dropout，调参阶段不引入正则化）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,            # 输入特征维度（与嵌入维度一致）
            nhead=num_heads,              # 多头注意力头数（固定）
            dim_feedforward=4*embed_dim,  # 前馈网络隐藏层维度（4倍d_model）
            dropout=dropout,              # Dropout率（调参阶段设为0）
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
        前向传播流程（CNN特征提取 → Transformer处理）
        
        参数：
            x: 输入fMRI数据 [batch_size, 1, D, H, W]（1为通道数）
        
        返回：
            预测的12维评分 [batch_size, 12]
        """
        batch_size = x.shape[0]  # 获取批次大小
        
        # 1. 通过CNN提取局部特征
        x = self.cnn(x)  # 输出形状：[batch_size, 64, D/4, H/4, W/4]（CNN处理后）
        
        # 2. 提取patch嵌入（基于CNN特征）
        x = self.patch_embed(x)  # 输出形状：[batch_size, embed_dim, D'', H'', W'']（D''=D/4/patch_size[0]等）
        x = x.flatten(2)         # 展平空间维度（合并D'', H'', W''为序列长度），形状：[batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)    # 调整维度顺序为[batch_size, num_patches, embed_dim]（序列长度为num_patches）
        
        # 3. 添加cls token和位置编码
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # 复制cls token到当前批次（形状：[batch_size, 1, embed_dim]）
        x = torch.cat((cls_tokens, x), dim=1)  # 拼接cls token（形状：[batch_size, num_patches+1, embed_dim]）
        x = x + self.pos_embed  # 加上位置编码（区分patch位置信息）
        
        # 4. 输入Transformer编码器
        x = self.transformer(x)  # 输出形状：[batch_size, num_patches+1, embed_dim]（保留所有patch和cls token的特征）
        
        # 5. 取cls token的特征，通过回归头输出
        x = x[:, 0, :]  # 提取cls token的特征（第0个位置，形状：[batch_size, embed_dim]）
        return self.head(x)  # 输出12维评分（形状：[batch_size, 12]）


# ---------------------- 训练函数（与原始模型一致） ----------------------
def train_model(train_dataset, batch_size, model, criterion, optimizer, device, rating_cols, epochs=10, batches_per_epoch=32):
    """（注释同vit_regression.py，此处省略重复注释，仅保留关键逻辑）"""
    train_losses = []
    model.train()
    dataset_size = len(train_dataset)

    for epoch in range(epochs):
        num_samples = batches_per_epoch * batch_size
        indices = torch.randperm(dataset_size)[:num_samples].tolist()
        sampler = SubsetRandomSampler(indices)
        
        loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=4,
                            pin_memory=True)
        
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            data = batch['fmri'].to(device)
            targets = torch.stack([batch[col].to(device).float() for col in rating_cols], dim=1)
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 12 == 0:
                pbar.write(f"Epoch: {epoch+1:03d}/{epochs} | Batch: {batch_idx:03d}/{batches_per_epoch} | Loss: {loss.item():.4f}")
            pbar.set_postfix_str(f"loss={loss.item():.4f}")
        
        avg_loss = total_loss / batches_per_epoch
        print(f"\nEpoch: {epoch+1:03d}/{epochs} | Avg Loss: {avg_loss:.4f}\n")
        train_losses.append(avg_loss)

    return train_losses


# ---------------------- 测试函数（与原始模型一致） ----------------------
def test_model(test_dataset, batch_size, model, device, rating_cols, criterion, batches_to_test=16):
    """（注释同vit_regression.py，此处省略重复注释，仅保留关键逻辑）"""
    model.eval()
    total_test_loss = 0.0
    total_error = 0.0
    dataset_size = len(test_dataset)
    with torch.no_grad():
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
            targets = torch.stack([batch[col].to(device).float() for col in rating_cols], dim=1)
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_test_loss += loss.item()
            total_error += torch.abs(outputs - targets).mean().item()
            if batch_idx % batches_to_test == 0:
                tqdm.write(f"Test Batch {batch_idx}/{batches_to_test} | Loss: {loss.item():.4f}")
    
    avg_test_loss = total_test_loss / batches_to_test
    avg_test_error = total_error / batches_to_test
    
    print(f"\nTest Loss: {avg_test_loss:.4f} | Test Avg Error: {avg_test_error:.4f}")
    visualize_fmri_slice(batch, model, device, rating_cols)
    return avg_test_loss


# ---------------------- 可视化函数（与原始模型一致） ----------------------
def visualize_fmri_slice(batch, model, device, rating_cols, z_slice=30):
    """（注释同vit_regression.py，此处省略重复注释，仅保留关键逻辑）"""
    was_training = model.training
    model.eval()
    
    data = batch['fmri'].to(device)
    targets = torch.stack([batch[col].to(device) for col in rating_cols], dim=1)
    image_paths = batch['image_path']
    
    with torch.no_grad():
        outputs = model(data)
    
    fmri_slice = data[0, 0, :, :, z_slice].cpu().numpy()
    pred_vec = outputs[0].cpu().numpy()
    true_vec = targets[0].cpu().numpy()
    image_path = image_paths[0]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(fmri_slice, cmap='gray')
    plt.title(f"z={z_slice} Slice (Image: {image_path})")
    
    plt.subplot(1, 2, 2)
    x = range(len(rating_cols))
    plt.bar([i-0.2 for i in x], true_vec, width=0.4, label='True')
    plt.bar([i+0.2 for i in x], pred_vec, width=0.4, label='Pred')
    plt.xticks(x, rating_cols, rotation=90)
    plt.ylabel('Rating Value')
    plt.title('True vs Predicted Ratings')
    plt.legend()
    plt.tight_layout()
    plt.savefig('trans_CNN_optuna.png')  # 保存可视化结果
    plt.close() 
    
    if was_training:
        model.train()


# ---------------------- 主程序（Optuna调参+最终模型训练） ----------------------
if __name__ == "__main__":
    # 固定超参数配置（不参与搜索的参数）
    fixed_config = {
        "input_shape": (72, 91, 75),    # fMRI输入形状（深度×高度×宽度）
        "patch_size": (6, 11, 6),       # Transformer的patch尺寸（基于CNN输出，需整除CNN输出形状）
        "num_heads": 8,                 # Transformer头数（固定以减少搜索维度）
        "num_classes": 12,              # 输出评分维度（12维）
        "batch_size": 16,                # 批次大小（小批次减少显存占用，适应调参）
        "epochs": 12,                    # 调参阶段训练轮数（示例用1轮，快速评估）
        "train_ratio": 0.8,             # 训练集比例（80%训练，20%测试）
        "batches_to_train": 32,          # 每轮训练批次数（子采样）
        "batches_to_test": 12            # 测试批次数（子采样）
    }
    
    # 初始化数据集（与原始模型一致）
    transform = T.Compose([
        T.Resize(256),                  # 图像缩放
        T.CenterCrop(224),              # 中心裁剪
        T.ToTensor(),                   # 转为张量
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])
    dataset = THINGSfastDataset(
        meta_npz_path="./preprocessed/things_meta.npz",  # 元数据路径
        image_root_dir="./image/_image_database_things",  # 图像根目录
        transform=transform  # 图像变换
    )
    
    # 划分训练集和测试集（随机划分）
    train_size = int(fixed_config["train_ratio"] * len(dataset))  # 训练集样本数
    test_size = len(dataset) - train_size  # 测试集样本数
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])  # 随机划分
    
    # 初始化设备（优先GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 定义目标评分列名（与原始模型一致）
    rating_cols = [
        'image-label_nameability_mean','image-label_consistency_mean',
        'property_manmade_mean','property_precious_mean','property_lives_mean',
        'property_heavy_mean','property_natural_mean','property_moves_mean',
        'property_grasp_mean','property_hold_mean','property_be-moved_mean','property_pleasant_mean'
    ]
    
    # 运行Optuna超参数优化（搜索最优超参数）
    print("Starting Optuna optimization...")
    study = optuna.create_study(direction="minimize")  # 优化方向：最小化测试损失
    # 执行1次试验（示例用小值，实际需增加n_trials以提高搜索准确性）
    study.optimize(lambda trial: objective(trial, fixed_config, train_dataset, test_dataset, rating_cols, device), 
                   n_trials=1, show_progress_bar=True)  # 启动优化
    
    # 输出最优结果（找到的超参数组合）
    print("\n==================== 最优超参数 ====================")
    print(f"最优测试损失: {study.best_value:.4f}")
    print(f"最优参数: {study.best_params}")  # 显示embed_dim、num_layers、lr的最优值
    
    # 重新训练最终模型（使用最优超参数，增加训练轮数）
    print("\n重新训练最终模型...")
    final_model = CNN_ViT3D(
        input_shape=fixed_config["input_shape"],
        patch_size=fixed_config["patch_size"],
        embed_dim=study.best_params["embed_dim"],  # 使用最优嵌入维度
        num_heads=fixed_config["num_heads"],        # 固定头数
        num_layers=study.best_params["num_layers"],  # 使用最优层数
        num_classes=fixed_config["num_classes"],
        dropout=0.0  # 仍无Dropout（或根据需求后续添加正则化）
    ).to(device)  # 模型移动到设备
    
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=study.best_params["lr"],  # 使用最优学习率
        weight_decay=0.0  # 仍无权重衰减（或根据需求调整）
    )
    
    # 训练轮数加倍（实际需根据任务调整）
    train_losses = train_model(
        train_dataset,
        fixed_config["batch_size"],
        final_model,
        criterion=nn.MSELoss(),
        optimizer=final_optimizer,
        device=device,
        rating_cols=rating_cols,
        epochs=fixed_config["epochs"],  # 训练1轮
        batches_per_epoch=fixed_config["batches_to_train"]
    )
    
    # 保存最终模型（参数）
    model_dir = Path("./model_optuna")  # 模型保存目录（区分原始模型）
    model_dir.mkdir(parents=True, exist_ok=True)  # 创建目录
    torch.save(final_model.state_dict(), model_dir / "best_vit3d.pth")  # 保存最优模型参数
    
    # 绘制损失曲线（调用自定义函数）
    plot_loss_curves(train_losses)
 
    # 最终模型测试（评估泛化能力）
    print("\n最终模型测试...")
    test_model(
        test_dataset,
        fixed_config["batch_size"],
        final_model,
        device,
        rating_cols,
        criterion=nn.MSELoss(),
        batches_to_test=fixed_config["batches_to_test"]
    )