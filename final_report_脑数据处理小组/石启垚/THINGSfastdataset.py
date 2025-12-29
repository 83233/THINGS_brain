import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import nibabel as nib

from globals import *

class THINGSfastDataset(Dataset):
    def __init__(self, 
                 meta_npz_path,
                 image_root_dir,
                 transform=None):
        """
        meta_npz_path: 预处理阶段保存的 .npz 文件路径
        image_root_dir: 图像根目录（用于拼接 filepath)
        """
        data = np.load(meta_npz_path,allow_pickle=True)
        # 取出所有列为 numpy array
        self.fields = {k: data[k] for k in data.files}
        self.image_root = image_root_dir
        self.transform = transform
        self.len = len(self.fields['filepath'])

        # 2. 构建分层索引 run_trial_map
        #    按 ses_idx -> run -> trial_number 映射到全局 idx
        self.run_trial_map = {}
        # session 字段可能是 bytes，要先解码
        sess_arr = self.fields['session']
        run_arr  = self.fields['run']
        trial_arr = self.fields['trial_number']
        
        for idx in range(self.len):
            # 取 session 字符串
            raw_ses = sess_arr[idx]
            if isinstance(raw_ses, bytes):
                raw_ses = raw_ses.decode()
            # 抽出数字部分并补齐两位
            ses_idx = raw_ses.split('ses-things')[-1].zfill(2)
            
            run   = int(run_arr[idx])
            trial = int(trial_arr[idx])
            
            # 初始化多级字典
            self.run_trial_map.setdefault(ses_idx, {})
            self.run_trial_map[ses_idx].setdefault(run, {})
            # 存储 idx
            self.run_trial_map[ses_idx][run][trial] = idx

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        """
        支持三种索引模式：
        1. 传统整数索引: dataset[12]
        2. 两层索引: dataset[run, trial] 此处默认 ses=01
        3. 三层索引: dataset[ses_things_idx, run, trial]
        """

        # 支持三元索引 (ses, run, trial)
        if isinstance(index, tuple) and len(index) == 3:
            ses, run, trial = index
            # 补齐 ses 为两位字符
            ses_idx = str(ses).zfill(2)
            try:
                idx = self.run_trial_map[ses_idx][run][trial]
            except KeyError:
                raise KeyError(f"Index ({ses_idx}, {run}, {trial}) not found.")
        elif isinstance(index, tuple) and len(index) == 2:
            # 保留原有 (run, trial) 兼容模式（可选）
            run, trial = index
            idx = self.run_trial_map['01'][run][trial]  # 默认 ses=01
        elif isinstance(index, int):
            # 传统整数索引
            idx = index
        else:
            raise ValueError("Invalid index type")
        
        # 1. 加载 fmri
        fmri_path = self.fields['fmri_path'][idx].decode() if isinstance(self.fields['fmri_path'][idx], bytes) else self.fields['fmri_path'][idx]
        img_nii = nib.load(fmri_path)  # 需 import nibabel
        data = img_nii.get_fdata()[..., int(self.fields['run'][idx])]
        fmri = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        fmri = (fmri - fmri.mean()) / fmri.std()

        # 2. 加载图像
        rel = self.fields['filepath'][idx].decode() if isinstance(self.fields['filepath'][idx], bytes) else self.fields['filepath'][idx]
        img = Image.open(os.path.join(self.image_root, rel)).convert('RGB')
        if self.transform:
            img = self.transform(img)

        # 3. ratings
        rating_cols = get_rating_cols()

        ratings = {col: torch.tensor(float(self.fields[col][idx]), dtype=torch.float32)
                   for col in rating_cols}
        
        SD_cols = get_SD_cols()
        
        SDs = {col: torch.tensor(float(self.fields[col][idx]), dtype=torch.float32)
                   for col in SD_cols}
        
        

        # 4. 【新增】加载 uniqueID
        raw_uid = self.fields['uniqueID'][idx]
        if isinstance(raw_uid, bytes):
            raw_uid = raw_uid.decode()
        unique_id = str(raw_uid)   # 明确转换为 Python 字符串

        return {
            'fmri': fmri,
            'image': img,
            **ratings,
            **SDs,
            'image_path': os.path.join(self.image_root, rel),
            'uniqueID': unique_id,
        }

# 使用示例
if __name__ == '__main__':
    import torchvision.transforms as T
    
    # 定义转换
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
        transform = transform
    )
    
    # 获取单个样本 索引形如[ses_things_idx, run, trial_number]
    sample = dataset[1,1,0]
    print("FMRI shape:", sample['fmri'].shape)
    print("Image shape:", sample['image'].shape)
    print("Rating value:", sample['image-label_nameability_mean'].item())