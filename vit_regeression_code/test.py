import os
import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting
from PIL import Image

# ======================================================================
# 1. 定义文件路径（需根据实际路径修改）
# ======================================================================
fmri_root_path = "betas_vol/scalematched/sub-01/ses-things01"
nii_path = os.path.join(fmri_root_path, "sub-01_ses-things01_run-01_betas.nii.gz")
tsv_path = os.path.join(fmri_root_path, "sub-01_ses-things01_run-01_conditions.tsv")
# 图像根目录
image_root_path = "image/_image_database_things/object_images"
# image_path = os.path.join(image_root_path, "")
# 试次索引
trial_idx = 0

# ======================================================================
# 2. 读取 .nii.gz 文件（fMRI beta权重或激活图）
# ======================================================================
# 加载NIfTI文件
nii_img = nib.load(nii_path)
nii_data = nii_img.get_fdata()  # 获取数据矩阵（维度通常为 [x, y, z, 时间点/条件]）
nii_header = nii_img.header    # 获取头文件元数据（如体素尺寸、坐标系等）

# 打印关键信息
print("\n===== NIfTI 文件信息 =====")
print(f"数据维度: {nii_data.shape}")        #  (72, 91, 75, 92)  
print(f"体素尺寸 (mm): {nii_header.get_zooms()[:3]}")  # 体素尺寸 (mm): (2.0, 2.0, 2.0)
print(f"坐标系: {nii_header.get_best_affine()}")       # 空间变换矩阵

# ======================================================================
# 3. 读取 .tsv 文件（试次条件信息）
# ======================================================================
# 加载TSV文件
tsv_df = pd.read_csv(tsv_path, sep="\t")

# 输出 TSV 文件的总行数
print(f"TSV 文件共有 {len(tsv_df)} 行数据")

# 打印关键信息
print("\n===== TSV 文件信息 =====")
print(f"列名: {tsv_df.columns.tolist()}")  # 查看所有列名（如 onset, duration, condition）
print("\n前5行数据:")
print(tsv_df.head())                       # 显示前5行数据

# ======================================================================
# 4. 关联NIfTI数据与试次条件（示例）
# ======================================================================
# 假设每个试次对应NIfTI数据中的第4维度（条件/时间点）
for idx, row in tsv_df.iterrows():
    condition_id = idx  # 或根据实际列名（如 row['condition_id']）
    beta_map = nii_data[..., condition_id]  # 提取对应条件的beta图
    
    # 打印试次条件与beta图的关联信息
    print(f"\n试次 {idx}:")
    print(f"- 条件名称: {row.get('condition', 'N/A')}")  # 假设列名为'condition'
    print(f"- Beta图维度: {beta_map.shape}")
    print(f"- Beta图均值: {np.nanmean(beta_map):.2f}")

    if idx == trial_idx:  # 选择特定试次
        break