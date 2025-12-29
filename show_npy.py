import numpy as np

# 读取 .npy 文件
# brain_score_code\things_fmri_compressed_700d.npy
file_path = './brain_score_code/things_fmri_compressed_700d.npy'  # 替换为你的文件路径
data = np.load(file_path)

# 打印数据
print(data)
print(data['fmri'].shape)