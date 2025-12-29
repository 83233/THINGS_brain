import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# 导入用户定义的模型和数据集代码
from contrast_sample import setup_model  
from THINGSfast_fmri_dataset import THINGSfastDataset
from globals import CONFIG

try:
    from brain_score import sco
except ImportError:
    print("警告: 'brain_score.py' 未找到或其中没有 'sco' 函数。将无法进行最终计算。")
    sco = None

def main():
    # 配置路径
    model_path = './results/models/model_epoch_60.pth'
    meta_npz_path = './preprocessed/things_meta.npz'
    image_root_dir = './image/_image_database_things'
    
    # 文件路径
    img_feat_path = './brain_score_code/act_ly_dr.npy'
    fmri_data_path = './brain_score_code/act_clu_fmri_raw491400.npy'  # 原始大文件
    
    # 检查图像特征文件是否存在
    if os.path.exists(img_feat_path):
        print(f"找到已存在的图像特征文件，直接加载: {img_feat_path}")
        img_mat = np.load(img_feat_path)
    else:
        print(f"未找到图像特征文件，开始执行模型推理以生成: {img_feat_path}")
        # 图像预处理，保持与模型训练时一致
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
        ])

        # 构建 THINGS 数据集和 DataLoader
        dataset = THINGSfastDataset(
            meta_npz_path=meta_npz_path,
            image_root_dir=image_root_dir,
            transform=transform
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

        # 初始化模型
        model = setup_model(CONFIG['clip_model'], CONFIG['output_dim'], device=CONFIG['device'])
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint)
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # 存储列表
        img_feat_list = []

        # 遍历数据集：同时获取 fmri 和 image
        for batch in tqdm(loader, desc='Processing batches'):
            # 处理图像
            imgs = batch['image'].to(device)
            with torch.no_grad():
                z, _ = model(imgs)
            img_feat_list.append(z.cpu().numpy())

        # 拼接所有样本
        img_mat = np.vstack(img_feat_list)  # (N_samples, output_dim)
        # 确保保存路径存在
        output_dir = os.path.dirname(img_feat_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 保存结果以便后续加载
        np.save(img_feat_path, img_mat)
        print(f"图像特征提取完成，并已保存 with shape {img_mat.shape}")

    # ===== 处理大型 fMRI 文件 =====
    # 检查 fMRI 数据文件是否存在
    if not os.path.exists(fmri_data_path):
        print(f"错误: 找不到 fMRI 数据文件 {fmri_data_path}")
        return
        
    # 使用直接加载方式处理大型文件
    print(f"正在加载大型 fMRI 数据: {fmri_data_path}")
    
    try:
        # 尝试标准加载方式
        fmri_mat = np.load(fmri_data_path)
        print(f"成功加载 fMRI 数据: shape {fmri_mat.shape}")
    except Exception as e:
        print(f"标准加载失败，错误信息: {e}")
        print(f"标准加载失败，尝试备选加载方式...")
        # 备选方法：使用文件大小推断数组形状
        file_size = os.path.getsize(fmri_data_path)
        expected_size = 9804 * 491400 * 4  # 9804样本 * 491400特征 * 4字节(float32)
        
        if file_size != expected_size:
            print(f"警告: 文件大小不匹配! 期望: {expected_size}字节, 实际: {file_size}字节")
            print("尝试调整样本数...")
            # 计算实际样本数
            actual_samples = file_size // (491400 * 4)
            print(f"推断样本数: {actual_samples}")
            
            # 使用内存映射创建数组视图
            fmri_mat = np.memmap(fmri_data_path, dtype=np.float32, mode='r', 
                                 shape=(actual_samples, 491400))
        else:
            # 文件大小匹配预期，直接创建内存映射
            fmri_mat = np.memmap(fmri_data_path, dtype=np.float32, mode='r', 
                                shape=(9804, 491400))
    
    # 检查样本数是否一致
    if fmri_mat.shape[0] != img_mat.shape[0]:
        print(f"警告: fMRI数据 ({fmri_mat.shape[0]}个样本) 与图像特征 ({img_mat.shape[0]}个样本) 的数量不匹配！")
        print("将使用最小样本数进行计算")
        min_samples = min(fmri_mat.shape[0], img_mat.shape[0])
        img_mat = img_mat[:min_samples]
        # 如果是内存映射对象，需要切片处理
        if isinstance(fmri_mat, np.memmap):
            # 创建新的内存映射视图
            fmri_mat = fmri_mat[:min_samples]
        else:
            fmri_mat = fmri_mat[:min_samples]
        print(f"使用前 {min_samples} 个样本进行计算")
    
    # 分批处理参数
    batch_size = 200  # 每次处理的样本数
    total_samples = fmri_mat.shape[0]
    
    # --- 分批计算 Brain-Score ---
    if sco is not None:
        print("\n开始分批计算 Brain-Score...")
        
        # 存储每批的结果
        batch_scores = []
        
        # 分批处理数据
        for start_idx in tqdm(range(0, total_samples, batch_size), desc="处理fMRI批次"):
            end_idx = min(start_idx + batch_size, total_samples)
            
            # 读取当前批次的 fMRI 数据
            fmri_batch = fmri_mat[start_idx:end_idx]
            
            # 获取对应的图像特征
            img_batch = img_mat[start_idx:end_idx]
            
            # 计算当前批次的分数
            batch_score = sco(data1=img_batch, data2=fmri_batch)
            batch_scores.append(batch_score.values.item())  # 存储分数值
        
        # 计算平均分数
        avg_score = np.mean(batch_scores)
        
        print("\n--- Brain-Score 计算结果 ---")
        print(f"处理批次: {len(batch_scores)}")
        print(f"各批次分数: {batch_scores}")
        print(f"平均对齐分数: {avg_score:.4f}")
        
        # 保存结果
        score_path = os.path.join(os.path.dirname(img_feat_path), 'brain_score_results.txt')
        with open(score_path, 'w') as f:
            f.write(f"Batch Scores: {batch_scores}\n")
            f.write(f"Average Score: {avg_score:.4f}\n")
        print(f"结果已保存至 {score_path}")
    
    else:
        print("无法计算 Brain-Score,缺少 sco 函数")


if __name__ == '__main__':
    main()