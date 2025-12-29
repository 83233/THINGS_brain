import os
import numpy as np
from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# 导入用户定义的 THINGSfastDataset
from THINGSfast_fmri_dataset import THINGSfastDataset
from globals import CONFIG


def main():
    # 数据路径配置
    meta_npz_path = './preprocessed/things_meta.npz'
    image_root_dir = './image/_image_database_things'
    model_output_dir = './brain_score_code'

    os.makedirs(model_output_dir, exist_ok=True)

    # 图像预处理（此处可保留，但我们只提取 fMRI）
    transform = transforms.Compose([  # 保持与加载时一致
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])

    # 构建数据集和 DataLoader，仅用于读取 fMRI
    dataset = THINGSfastDataset(
        meta_npz_path=meta_npz_path,
        image_root_dir=image_root_dir,
        transform=transform
    )
    
    # 创建内存映射文件
    output_path = os.path.join(model_output_dir, 'act_clu_fmri_raw491400.npy')
    total_samples = len(dataset)
    voxel_dims = dataset[0]['fmri'].numel()  # 获取单个样本的体素数量
    
    # 创建预分配的内存映射文件
    fp = np.memmap(output_path, dtype='float32', mode='w+', 
                  shape=(total_samples, voxel_dims))
    
    # 分批处理数据
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    start_idx = 0
    
    for batch in tqdm(loader, desc="Processing fMRI data"):
        fmri_batch = batch['fmri']  # shape: (B, 1, X, Y, Z)
        B = fmri_batch.size(0)
        
        # 展平并转换为numpy
        fmri_flat = fmri_batch.view(B, -1).numpy().astype('float32')
        
        # 写入内存映射文件
        end_idx = start_idx + B
        fp[start_idx:end_idx] = fmri_flat
        fp.flush()  # 确保写入磁盘
        start_idx = end_idx

    print(f"Saved Raw fMRI data to {output_path} with shape ({total_samples}, {voxel_dims})")
    
    # 清理内存映射对象
    del fp

if __name__ == '__main__':
    main()