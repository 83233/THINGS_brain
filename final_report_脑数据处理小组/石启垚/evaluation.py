import os
import torch
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
import nibabel as nib
from nilearn import plotting, datasets
import plotly.graph_objects as go

from THINGSfastdataset import THINGSfastDataset
from vit_regression_subSamper_scheduler import ViT3D
from globals import *

def compute_similarity(pred_vector, test_labels, metric='cosine'):
    # pred_vector: [1, D], test_labels: [N, D]
    pred = pred_vector.squeeze(0)
    if metric == 'cosine':
        p = F.normalize(pred, dim=0)
        t = F.normalize(test_labels, dim=1)
        return torch.matmul(t, p)            # [N]
    else:
        d = torch.norm(test_labels - pred, dim=1)
        return -d                           # [N]

def visualize_attention_on_fmri(sample, cls2patch, patch_size, input_shape, head_idx=0, z_slices=[30, 40, 50]):
    """
    将attention映射到fMRI原始空间并可视化。
    sample: 一个样本的原始fmri张量 shape [1, D, H, W]
    cls2patch: attention权重,shape [heads, num_patches]
    patch_size: patch尺寸(如 (12, 13, 15))
    input_shape: 输入尺寸，如(72, 91, 75)
    head_idx: 要可视化的head索引
    z_slices: 选择可视化的z轴切片索引
    """
    # 1. 计算 patch grid 尺寸
    grid_D, grid_H, grid_W = [input_shape[i] // patch_size[i] for i in range(3)]
    num_patches = grid_D * grid_H * grid_W
    assert cls2patch.shape[1] == num_patches, "patch数量不匹配"

    # 2. 取出某个 head 的 attention
    attn_weights = cls2patch[head_idx]  # shape: [num_patches]
    attn_map = attn_weights.view(grid_D, grid_H, grid_W)  # attention分布到每个 patch

    # 3. 构建 attention 叠加图（原始分辨率）
    full_attention = torch.zeros(input_shape)
    for d in range(grid_D):
        for h in range(grid_H):
            for w in range(grid_W):
                val = attn_map[d, h, w].item()
                d_start, d_end = d * patch_size[0], (d + 1) * patch_size[0]
                h_start, h_end = h * patch_size[1], (h + 1) * patch_size[1]
                w_start, w_end = w * patch_size[2], (w + 1) * patch_size[2]
                full_attention[d_start:d_end, h_start:h_end, w_start:w_end] = val

    # 4. 可视化z轴切片
    fmri_np = sample.squeeze(0).cpu().numpy()
    attn_np = full_attention.numpy()

    for z in z_slices:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # —— 原始 fMRI 切片 —— #
        im0 = axes[0].imshow(fmri_np[:, :, z], cmap='gray')
        axes[0].set_title(f'fMRI Slice z={z}')
        axes[0].axis('off')
        # 添加 fMRI 强度 colorbar
        cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        cbar0.set_label('fMRI Intensity', rotation=270, labelpad=15)

        # —— Attention 热力图 —— #
        im1 = axes[1].imshow(attn_np[:, :, z], cmap='hot')
        axes[1].set_title(f'Attention Heatmap z={z} (Head {head_idx+1})')
        axes[1].axis('off')
        # 添加 Attention 权重 colorbar
        cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        cbar1.set_label('Attention Weight', rotation=270, labelpad=15)

        plt.tight_layout()
        plt.show()

def attention_to_nifti(attn_map: np.ndarray,
                       input_shape: tuple,
                       patch_size: tuple,
                       affine: np.ndarray = None) -> nib.Nifti1Image:
    """
    将 attention_map([num_patches]）映射到与 fMRI 相同的三维体素空间，
    并返回 NIfTI1Image。
    """
    # 1. 计算 grid 尺寸
    grid = tuple(input_shape[i] // patch_size[i] for i in range(3))
    attn_3d = attn_map.reshape(grid)
    
    # 2. 扩展到原始体素
    full = np.zeros(input_shape, dtype=np.float32)
    for d in range(grid[0]):
        for h in range(grid[1]):
            for w in range(grid[2]):
                val = attn_3d[d, h, w]
                ds, de = d*patch_size[0], (d+1)*patch_size[0]
                hs, he = h*patch_size[1], (h+1)*patch_size[1]
                ws, we = w*patch_size[2], (w+1)*patch_size[2]
                full[ds:de, hs:he, ws:we] = val

    # 3. 构造仿射（若无提供，默认单位矩阵）
    if affine is None:
        affine = np.eye(4)
    return nib.Nifti1Image(full, affine)

# -----------------------------
# 4. 主程序：划分 / 训练 / 检索 / 可视化
# -----------------------------
if __name__ == "__main__":
    # 配置
    split_path = "./preprocessed/dataset_splits.npz"
    model_path = "./results/model/vit3d.pth"
    cache_path = "./preprocessed./cached_test_labels.pt"
    Retrieval = False    # 是否检索
    Atten_Vis = True
    config = {
        "input_shape": (72, 91, 75),
        "patch_size": (6, 7, 5),
        "embed_dim": 256,
        "num_heads": 8,
        "num_layers": 3,
        "num_classes": 10,
        "batch_size": 8,
        "lr": 1e-4,
        "epochs": 50,
        "train_ratio": 0.8,
        "batches_to_train": 32,
        "batches_to_test": 16
    }
    rating_cols = get_rating_cols()

    # 数据集
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                     std=[0.229, 0.224, 0.225])
    ])

    dataset = THINGSfastDataset(
        meta_npz_path="./preprocessed/things_meta.npz",
        image_root_dir="./image/_image_database_things",
        transform=transform
    )

    # 划分索引
    if os.path.exists(split_path):
        arr = np.load(split_path)
        train_idx = arr['train_indices'].tolist()
        val_idx   = arr['val_indices'].tolist()
        test_idx  = arr['test_indices'].tolist()
    else:
        N = len(dataset)
        inds = np.random.permutation(N)
        t_end = int(config['train_ratio'] * N)
        v_end = t_end + int(config['val_ratio'] * N)
        train_idx = inds[:t_end].tolist()
        val_idx   = inds[t_end:v_end].tolist()
        test_idx  = inds[v_end:].tolist()
        np.savez(split_path,
                 train_indices=np.array(train_idx, dtype=int),
                 val_indices  =np.array(val_idx,   dtype=int),
                 test_indices =np.array(test_idx,  dtype=int))

    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)
    test_ds  = Subset(dataset, test_idx)

    # 设备 & 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViT3D(
        input_shape=config["input_shape"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"]
    )
    model.load_state_dict(torch.load(model_path,weights_only=True))
    model.to(device)

    test_loader = DataLoader(val_ds, batch_size=1, pin_memory=True, num_workers=4)

    if Retrieval:
        # 准备所有测试集真实 labels
        if os.path.exists(cache_path):
            # 直接加载
            data = torch.load(cache_path,weights_only=True)
            all_test_ids = data['all_test_ids']
            test_labels  = data['test_labels']
            print(f"Loaded cached test labels from '{cache_path}'.")
        
        else:
            # 被检索库：从 test_ds 中按 rating_cols 拼成 [N_test,12]
            all_test_labels = []
            all_test_ids    = []
            for batch in tqdm(test_loader, desc="Loading Test Labels"):
                # 假设 dataset 返回 batch['uniqueID'] (scalar tensor)
                uid = batch['uniqueID']
                all_test_ids.append(uid)
                # 按顺序拼接 12 维评分->10维
                vec = torch.stack(
                    [ batch[col].float().squeeze(0) for col in rating_cols ],
                    dim=0
                )  # shape (12,)
                all_test_labels.append(vec)
            # 最终矩阵 [N_test,12]
            test_labels = torch.stack(all_test_labels, dim=0)

            # 缓存到文件
            torch.save({
                'all_test_ids': all_test_ids,
                'test_labels' : test_labels
                }, cache_path)
            print(f"Saved test labels cache to '{cache_path}'.")
        
        # val_loader = DataLoader(val_ds, batch_size=1, pin_memory=True)
        topk_counts = {1:0, 5:0, 10:0}
        total_q = 0

        for batch in tqdm(test_loader, desc="Retrieval"):
            # 取出查询样本的 uniqueID，并构建输入
            uid_q = batch['uniqueID']
            data = batch['fmri'].to(device)
            with torch.no_grad():
                pred_vec = model(data).cpu()  # (1,12)
            # 计算相似度 scores: [N_test]
            scores = compute_similarity(pred_vec, test_labels, metric='cosine')
            idx_sorted = torch.argsort(scores, descending=True)
            
            total_q += 1
            # 按 uniqueID 判断是否命中
            for k in topk_counts:
                topk_idx = idx_sorted[:k].tolist()
                topk_ids = [ all_test_ids[i] for i in topk_idx ]
                if uid_q in topk_ids:
                    topk_counts[k] += 1

        for k, cnt in topk_counts.items():
            print(f"Top-{k} recall: {cnt/total_q * 100:.4f}%")

    if Atten_Vis:
        # -------------------------
        # Attention 可视化
        # -------------------------
        
        sample_idx = val_idx[0]
        sample_x= dataset[sample_idx]
        # sample_x = sample_x.unsqueeze(0).to(device)
        data = sample_x['fmri'].unsqueeze(0).to(device)
        with torch.no_grad():
            _, attn = model(data, return_attn=True)
        # attn: [1, heads, seq, seq]
        attn = attn[0]             # [heads, seq, seq]
        cls2patch = attn[:, 0, 1:] # [heads, num_patches]
        # H, P = cls2patch.shape

        # 切片注意力可视化
        # for head_idx  in range(cls2patch.shape[0]):
        #     visualize_attention_on_fmri(
        #     sample_x['fmri'], cls2patch, config["patch_size"], config["input_shape"],
        #     head_idx=head_idx, z_slices=[30, 40, 50]
        # )
        # 使用 Nilearn 可视化3d
        # 1. 构建 NIfTI
        provided_affine = np.array([
            [ 2.        ,  0.        ,  0.        , -71.1632843 ],
            [ 0.        ,  2.        ,  0.        , -80.94287109],
            [ 0.        ,  0.        ,  2.        , -68.46676636],
            [ 0.        ,  0.        ,  0.        ,   1.        ]
        ])
        
        attention_nifti = attention_to_nifti(
            attn_map=cls2patch[0].cpu().numpy(),             # 某 head 的注意力向量
            input_shape=config["input_shape"],
            patch_size=config["patch_size"],
            affine=provided_affine
        )
        from nilearn.datasets import load_mni152_template
        from nilearn.image import resample_to_img

        mni_template = load_mni152_template(resolution=2)  # fMRIPrep 默认空间
        attention_mni = resample_to_img(
            source_img=attention_nifti,
            target_img=mni_template,
            interpolation='nearest',     # 最近邻插值
            copy_header=True,
            force_resample=True
        )
        affine = attention_mni.affine

        # 2. Glass brain 半透明渲染
        plotting.plot_glass_brain(
            attention_mni,
            title="Head 1 Attention in MNI152NLin2009cAsym (Glass Brain)",
            display_mode='lyrz',
            colorbar=True
        )
        # 4. 交互式 3D 切片视图（已对齐 MNI）
        view = plotting.view_img(
            attention_mni,
            threshold="90%",
            cmap='hot'
        )

        view.open_in_browser()  # 在浏览器中查看交互式渲染

        # ============= 使用 Plotly 体积渲染 =============

        data = attention_mni.get_fdata()
        x, y, z = np.indices(data.shape)

        fig = go.Figure(data=go.Volume(
            x=x.flatten(), y=y.flatten(), z=z.flatten(),
            value=data.flatten(),
            isomin=np.percentile(data, 75),
            isomax=data.max(),
            opacityscale=[(0, 0.1), (0.5, 0.5), (1, 1.0)],
            surface_count=20,  # 等值面数量
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))
        fig.update_layout(title="Head 1 Attention Volume Rendering")
        fig.show()
        # ============= 3D 散点图显示 Top‐N Attention Voxels =============
        # 取 top N voxels 的世界坐标和对应权重
        flat = data.flatten()
        N = 200
        top_idxs = np.argsort(flat)[-N:]
        coords_vox = np.array(np.unravel_index(top_idxs, data.shape)).T
        coords_world = nib.affines.apply_affine(attention_mni.affine, coords_vox)

        # ============= 脑区标注 & 占比统计 =============

        # Load Harvard–Oxford atlas
        atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        atlas_img = atlas.maps
        labels = atlas.labels

        # 取 attention 前 N 大的 voxel 坐标（与上面散点图一致）
        top_vox = top_idxs
        coords = coords_vox  # voxel indices

        # 转换到世界坐标，再查询标签
        xyz_world = coords_world

        for coord in xyz_world[:10]:
            # 查询对应标签
            i, j, k = coord.astype(int)
            label_idx = atlas_img.get_fdata()[i, j, k]
            print(f"Coord {coord} → Region: {labels[int(label_idx)]}")

        # 统计各脑区占比
        atlas_data = atlas_img.get_fdata().astype(int)
        # 查询 top voxels 所在标签
        top_labels = atlas_data[coords_vox[:, 0],
                                 coords_vox[:, 1],
                                 coords_vox[:, 2]]
        unique_labels, counts = np.unique(top_labels, return_counts=True)
        print("\n=== Top-N Attention Voxels Region Proportions ===")
        total = len(top_labels)
        for lbl, cnt in zip(unique_labels, counts):
            region = labels[int(lbl)]
            print(f"{region:30s}: {cnt/total*100:5.2f}%")
