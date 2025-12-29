# things_image_dataset.py

import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
import matplotlib.pyplot as plt
import csv

from globals import *


class THINGSImageDataset(Dataset):
    def __init__(self, 
                 meta_npz_path,
                 image_root_dir,
                 category27_tsv_path=None,
                 transform=None):
        """
        meta_npz_path: preprocess_images.py 生成的 .npz 文件
        image_root_dir: 与 .npz 中 filepath 相拼接的根目录
        transform: torchvision.transforms
        """
        data = np.load(meta_npz_path, allow_pickle=True)
        # 转成 dict of arrays
        self.fields = {k: data[k] for k in data.files}
        self.image_root = image_root_dir
        self.transform = transform
        self.len = len(self.fields['filepath'])
        self.uid_to_category = None

        # 构建 uniqueID -> [idx list] 索引（可选）
        self.uid_map = {}
        for idx, uid in enumerate(self.fields['uniqueID']):
            uid_str = uid.decode() if isinstance(uid, bytes) else str(uid)
            self.uid_map.setdefault(uid_str, []).append(idx)
        # 加载 category27.tsv 映射关系（可选）
        if category27_tsv_path:
            sorted_uids = sorted(self.uid_map.keys())
            # 解析TSV文件
            self.uid_to_category = {}
            with open(category27_tsv_path, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                headers = next(reader)  # 读取头部（27个类别名）
                
                # 按行读取，每行对应一个sorted_uids中的uniqueID
                for idx, row in enumerate(reader):
                    if idx >= len(sorted_uids):
                        break
                    uid = sorted_uids[idx]
                    # 将01向量转换为类别索引
                    one_hot = [int(x) for x in row]
                    s = sum(one_hot)
                    if s == 1:
                        # 正常情况，只有一个类别为1
                        category_idx = one_hot.index(1)
                    elif s == 0:
                        # 没有归属任何大类，标记为 -1
                        category_idx = -1
                    else:
                        # 存在多个类别为1，取第一个
                        category_idx = one_hot.index(1)
                        print(f"Warning: {uid} has multiple categories with 1, using {category_idx}")
                    self.uid_to_category[uid] = category_idx
                
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        """
        支持：
         1. 整数索引 dataset[i]
         2. 按 uniqueID 索引 dataset['aardvark'] 返回该 ID 下第一张图（可扩展）
        """
        if isinstance(index, str):
            # 按 uniqueID，默认取该组第一张
            idx = self.uid_map[index][0]
        elif isinstance(index, int):
            idx = index
        else:
            raise ValueError("Index must be int or uniqueID string")

        # 1. 加载图像
        rel = self.fields['filepath'][idx]
        if isinstance(rel, bytes):
            rel = rel.decode()
        img = Image.open(os.path.join(self.image_root, rel)).convert('RGB')
        if self.transform:
            img = self.transform(img)

        # 2. 加载评分
        rating_cols = get_rating_cols()
        SD_cols = get_SD_cols()
        ratings = {col: torch.tensor(float(self.fields[col][idx]), dtype=torch.float32)
                   for col in rating_cols}
        SDs = {col: torch.tensor(float(self.fields[col][idx]), dtype=torch.float32)
                   for col in SD_cols}

        # 3. uniqueID
        raw_uid = self.fields['uniqueID'][idx]
        if isinstance(raw_uid, bytes):
            raw_uid = raw_uid.decode()
        unique_id = str(raw_uid)

        # 4. 类别（可选）
        if self.uid_to_category:
            category_idx = self.uid_to_category.get(unique_id, -1)
            return {
                'image': img,
                'uniqueID': unique_id,
                'category27': category_idx,
                **ratings,
                **SDs,
                'filepath': rel,
            }
        else:
            return {
                'image': img,
                'uniqueID': unique_id,
                **ratings,
                **SDs,
                'filepath': rel,
            }

if __name__ == "__main__":
    # 使用示例
    import torchvision.transforms as T
    transform = T.Compose([
        T.transforms.Resize(224), # 这里调用了CLIP预训练的ViT视觉编码器，因此输入大小为224x224
        T.ToTensor()
    ])
    dataset = THINGSImageDataset(
        meta_npz_path=CONFIG['meta_path'],
        image_root_dir=CONFIG['img_root'],
        # category27_tsv_path=CONFIG['category27_tsv_path'],
        transform=transform
    )
    # 取一张 aardvark 图
    sample = dataset['aardvark']
    print("Image tensor shape:", sample['image'].shape)
    print("uniqueID:", sample['uniqueID'])
    print("Manmade mean:", sample['property_manmade_mean'].item())
    # print("category27:", sample['category27'])
    # 显示图像
    plt.imshow(sample['image'].permute(1, 2, 0))
    plt.show()

