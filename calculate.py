#  C:\Users\26065\nilearn_data\schaefer_2018
import os
import numpy as np
import torch
from torch.utils.data  import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# 导入用户定义的模型和数据集代码
from contrast_sample import setup_model  
from THINGSfast_fmri_dataset import THINGSfastDataset  # 使用更新后的数据集定义
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
    # 如果需要限制 ses/run/trial，可在 DataLoader 或索引时设定

    img_feat_path = './brain_score_code/act_ly_dr.npy'
    fmri_data_path = './brain_score_code/things_fmri_compressed_700d.npy'

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
            ])

        # 构建 THINGS 数据集和 DataLoader
        dataset = THINGSfastDataset(
            meta_npz_path=meta_npz_path,
            image_root_dir=image_root_dir,
            transform=transform
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

        # 初始化模型
        # model = CEBRAContrastiveModel(base_encoder=base_encoder, output_dim=128)
        model = setup_model(CONFIG['clip_model'], CONFIG['output_dim'], device=CONFIG['device'])
        checkpoint = torch.load(model_path, map_location='cpu',weights_only=True)
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
        np.save('./brain_score_code/act_ly_dr.npy', img_mat)
        print(f"图像特征提取完成，并已保存 with shape {img_mat.shape}")

    # 加载降维后的 fmri 数据
    if not os.path.exists(fmri_data_path):
        print(f"error:找不到降维后的fMRI数据文件 {fmri_data_path}。请先运行数据降维脚本。")
        return
    print(f"正在加载降维后的fMRI数据: {fmri_data_path}")
    fmri_npz = np.load(fmri_data_path)
    # 根据您之前保存的键 'fmri' 来提取数据
    fmri_mat = fmri_npz['fmri']
    print(f"加载的fMRI特征矩阵维度: {fmri_mat.shape}")

    # --- 4. 调用 sco 函数进行计算 ---
    if sco is not None:
        # 检查样本数是否一致
        if fmri_mat.shape[0] != img_mat.shape[0]:
            print(f"ERROR: fMRI数据 ({fmri_mat.shape[0]}个样本) 与图像特征 ({img_mat.shape[0]}个样本) 的数量不匹配！")
            print("请检查数据集和特征提取过程是否对应。")
            return
            
        print("\n开始计算 Brain-Score...")
        # 调用 sco 函数，传入两个二维矩阵 (样本数, 特征数)
        score = sco(data1=img_mat, data2=fmri_mat)
        
        # score 对象是 brain-score 的 DataArray，可以直接打印查看详细信息
        print("\n--- Brain-Score 计算结果 ---")
        print(score)
        # 提取核心的分数值
        print(f"\n核心对齐分数 (PLS-Pearsonr): {score.values.item():.4f}")


if __name__ == '__main__':
    main()