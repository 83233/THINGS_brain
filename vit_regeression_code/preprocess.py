import os
import numpy as np
import pandas as pd

from globals import *
def preprocess_and_save(fmri_root_dir,
                        conditions_tsv_pattern,
                        image_paths_csv,
                        property_ratings_tsv,
                        save_npz_path=None,
                        save_h5_path=None):
    # 1. 读取 ratings & image paths
    prop = pd.read_csv(property_ratings_tsv, sep='\t')
    imgs = pd.read_csv(image_paths_csv, header=None, names=['filepath'])
    imgs['filename'] = imgs['filepath'].apply(lambda x: os.path.basename(x).split('.')[0])
    
    # 2. 读取所有 conditions
    conds = []
    for ses in sorted(os.listdir(fmri_root_dir)):
        if not ses.startswith('ses-things'):
            continue
        ses_dir = os.path.join(fmri_root_dir, ses)
        run = 1
        while True:
            name = conditions_tsv_pattern.format(ses=ses, run=run)
            path = os.path.join(ses_dir, name)
            if not os.path.exists(path):
                break
            df = pd.read_csv(path, sep='\t')
            df['session'], df['run'] = ses, run
            conds.append(df)
            run += 1
    conds = pd.concat(conds, ignore_index=True)
    
    # 3. 建立 metadata
    conds['base_filename'] = conds['image_filename'].apply(
        lambda x: os.path.basename(x).split('.')[0]
    )
    meta = pd.merge(conds, imgs, left_on='base_filename', right_on='filename', how='inner') \
             .drop(columns=['filename'])
    
    meta['uniqueID'] = meta['image_filename'].apply(
        lambda x: os.path.normpath(x).split(os.sep)[0]
    )
    all_cols = get_all_cols()
    # 合并 ratings，并立即归一化
    sel = ['uniqueID'] + all_cols
    meta = pd.merge(meta, prop[sel], on='uniqueID', how='inner')
    
    # 不做归一化,利用标准差构建正态分布
    # prop_cols = [
    #     'property_manmade_mean','property_precious_mean','property_lives_mean',
    #     'property_heavy_mean','property_natural_mean','property_moves_mean',
    #     'property_grasp_mean','property_hold_mean','property_be-moved_mean','property_pleasant_mean'
    # ]
    # norm = lambda col: (col - 1.0) / 6.0
    # meta[prop_cols] = meta[prop_cols].apply(norm)

    meta = meta.rename(columns={'Unnamed: 0': 'trial_number'})

    # 4. 计算 fmri 绝对路径列表
    fmri_paths = []
    for _, row in meta.iterrows():
        ses, run = row['session'], row['run']
        fname = f"sub-01_{ses}_run-{run:02d}_betas.nii.gz"
        fmri_paths.append(os.path.join(fmri_root_dir, ses, fname))
    meta['fmri_path'] = fmri_paths

    # 5. 保存
    if save_npz_path:
        # 转成 numpy-friendly dict
        np_dict = {col: meta[col].values for col in meta.columns}
        np.savez_compressed(save_npz_path, **np_dict)
        print(f"Saved NPZ to {save_npz_path}")
    if save_h5_path:
        meta.to_hdf(save_h5_path, key='meta', mode='w')
        print(f"Saved HDF5 to {save_h5_path}")

if __name__ == "__main__":
    preprocess_and_save(
        fmri_root_dir="./betas_vol/scalematched/sub-01",
        conditions_tsv_pattern="sub-01_{ses}_run-{run:02d}_conditions.tsv",
        image_paths_csv="./image/01_image-level/image-paths.csv",
        property_ratings_tsv="./image/02_object-level/_property-ratings.tsv",
        save_npz_path="./preprocessed/things_meta.npz"
    )