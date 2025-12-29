# preprocess_images.py

import os
import numpy as np
import pandas as pd
from globals import *

def preprocess_and_save(image_root_dir,
                        property_ratings_tsv,
                        save_npz_path):
    """
    image_root_dir: 应包含子目录 images/{uniqueID}/ 下的所有图片
    property_ratings_tsv: 物体级别评分表，含 uniqueID, 各 property_mean, property_sd 列
    save_npz_path: 输出 .npz 文件路径
    """
    # 1. 读取 ratings（含 mean 和 SD）
    prop = pd.read_csv(property_ratings_tsv, sep='\t')
    rating_cols = get_rating_cols()
    SD_cols = get_SD_cols()
    sel_cols = ['uniqueID'] + rating_cols + SD_cols
    prop = prop[sel_cols]

    # 2. 遍历每个 uniqueID 目录，收集所有图像路径
    records = []
    images_root = os.path.join(image_root_dir, 'images')
    for uid in prop['uniqueID'].astype(str):
        uid_dir = os.path.join(images_root, uid)
        if not os.path.isdir(uid_dir):
            continue
        for fname in os.listdir(uid_dir):
            if fname.lower().endswith(('.jpg','.jpeg','.png')):
                rel_path = os.path.join('images', uid, fname)
                records.append({'uniqueID': uid, 'filepath': rel_path})

    meta = pd.DataFrame(records)
    # 3. 合并评分
    meta = meta.merge(prop, on='uniqueID', how='inner')
    
    # 打印前几行信息以进行检查
    print("Before normalization:")
    print(meta.head())
    for col in rating_cols:
        # mean 归一化： (x - 1) / 6
        meta[col] = (meta[col] - 1.0) / 6.0

    for col in SD_cols:
        # SD 归一化： x / 6
        meta[col] = meta[col] / 6.0

    print("After normalization (mean -> (mean-1)/6, SD -> SD/6):")
    print(meta.head())
    # 4. 保存为压缩 npz
    np_dict = {col: meta[col].values for col in meta.columns}
    np.savez_compressed(save_npz_path, **np_dict)
    print(f"Saved NPZ to {save_npz_path}")

if __name__ == "__main__":
    preprocess_and_save(
        image_root_dir="./image/_image_database_things",
        property_ratings_tsv="./image/02_object-level/_property-ratings.tsv",
        save_npz_path="./preprocessed/things_image.npz"
    )
