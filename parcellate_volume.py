import os
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm

# 引入 nilearn 用于下载和处理图像
from nilearn import image as nilearn_image
from nilearn import datasets

# 假设您已将 THINGSfastDataset 类保存在名为 things_dataset.py 的文件中
# 同时，假设 globals.py 也存在且包含了 get_rating_cols 和 get_SD_cols 函数
from THINGSfast_fmri_dataset import THINGSfastDataset 

# --- 配置参数 ---

# 1. 输入数据路径 (与您原始代码中的路径一致)
META_NPZ_PATH = "./preprocessed/things_meta.npz"
IMAGE_ROOT_DIR = "./image/_image_database_things"

# 2. 输出文件路径 (处理后的数据将保存在这里)
OUTPUT_NPZ_PATH = "./brain_score_code/things_fmri_compressed_700d.npy"

# 3. 目标维度 (由分区模板决定)
NUM_PARCELS = 700

def compress_fmri_with_atlas(dataset, num_parcels):
    """
    使用大脑分区模板对数据集中的 fMRI 数据进行降维。
    此版本会自动下载模板并进行重采样以匹配 fMRI 数据空间。

    Args:
        dataset (Dataset): 实例化的 THINGSfastDataset。
        num_parcels (int): 模板中的分区数量。

    Returns:
        tuple: (包含所有降维后 fMRI 数据的 numpy 数组, 对应的 uniqueID 列表)
    """
    # 1. 使用 nilearn 自动下载 Schaefer 2018 模板
    # 我们选择与您数据体素大小匹配的 2mm 分辨率
    print(f"正在下载 Schaefer 2018 Atlas ({num_parcels} 分区, 2mm 分辨率)...")
    # nilearn 会将数据缓存，后续运行会直接从本地加载
    schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=num_parcels, resolution_mm=2)
    atlas_nii = nilearn_image.load_img(schaefer_atlas.maps)
    print("模板下载/加载成功。")
    print(f"原始模板维度: {atlas_nii.shape}")

    # 2. 【关键步骤】将模板重采样到 fMRI 数据的空间
    # 我们需要一个 fMRI 样本作为重采样的目标参考
    # 从数据集中获取第一个样本的 fMRI 文件路径
    first_fmri_path = dataset.fields['fmri_path'][0].decode() if isinstance(dataset.fields['fmri_path'][0], bytes) else dataset.fields['fmri_path'][0]
    target_fmri_nii = nib.load(first_fmri_path) # 使用 nibabel 加载以获取其仿射和形状
    
    print(f"目标 fMRI 维度: {target_fmri_nii.shape}")
    print("正在将模板重采样至 fMRI 空间...")
    
    # 使用 'nearest' 插值方法，这对于保留分区的离散整数标签至关重要
    resampled_atlas_nii = nilearn_image.resample_to_img(
        source_img=atlas_nii,
        target_img=target_fmri_nii,
        interpolation='nearest'
    )
    
    # 从重采样后的模板获取数据数组
    resampled_atlas_data = resampled_atlas_nii.get_fdata().astype(np.int32)
    print(f"模板重采样成功。新的模板维度: {resampled_atlas_data.shape}")
    
    # 确认维度匹配
    if target_fmri_nii.shape[:3] != resampled_atlas_data.shape:
        raise RuntimeError("重采样后维度依然不匹配，请检查输入数据。")

    all_compressed_fmri = []
    all_unique_ids = []

    # 3. 遍历数据集中的每一个样本
    print(f"开始处理 {len(dataset)} 个 fMRI 样本...")
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        
        # 获取 fMRI 数据 (Tensor -> numpy, 并移除单通道维度)
        fmri_data = sample['fmri'].squeeze().numpy()
        unique_id = sample['uniqueID']
        
        # 初始化当前样本的降维向量
        compressed_vector = np.zeros(num_parcels)
        
        # 4. 对每个分区计算信号均值 (现在使用重采样后的模板)
        for parcel_id in range(1, num_parcels + 1):
            mask = (resampled_atlas_data == parcel_id)
            if np.any(mask):
                voxels_in_parcel = fmri_data[mask]
                compressed_vector[parcel_id - 1] = np.mean(voxels_in_parcel)
        
        all_compressed_fmri.append(compressed_vector)
        all_unique_ids.append(unique_id)
        
    return np.array(all_compressed_fmri), np.array(all_unique_ids)


if __name__ == '__main__':
    # 实例化数据集 (transform=None)
    print("正在实例化数据集...")
    dataset = THINGSfastDataset(
        meta_npz_path=META_NPZ_PATH,
        image_root_dir=IMAGE_ROOT_DIR,
        transform=None
    )
    print(f"数据集加载完成，共 {len(dataset)} 个样本。")

    # 执行降维，不再需要传入 atlas_path
    compressed_data, unique_ids = compress_fmri_with_atlas(
        dataset=dataset,
        num_parcels=NUM_PARCELS
    )

    # 5. 保存结果
    if compressed_data is not None and unique_ids is not None:
        print(f"数据处理完成。降维后数据维度: {compressed_data.shape}")
        
        output_dir = os.path.dirname(OUTPUT_NPZ_PATH)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        np.savez(
            OUTPUT_NPZ_PATH,
            fmri=compressed_data, 
            uniqueID=unique_ids
        )
        print(f"已将降维后的数据保存到: {OUTPUT_NPZ_PATH}")