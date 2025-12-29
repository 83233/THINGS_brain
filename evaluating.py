# interpret_embeddings.py
import os
import torch
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt

from THINGSImagedataset import THINGSImageDataset
from contrast_sample import setup_model
from globals import get_rating_cols, get_SD_cols, CONFIG
from visulizing import tsne_visualization, pca_score_correlation, umap_visualization
from preprocess import load_category_names

# os.environ['OMP_NUM_THREADS'] = '1'

# =============================================================================
# 1. 加载验证集--这里我选定的是从每个 uniqueID 中随机抽取 2 张图片作为评估集。
# 当然也可以修改代码,从已分割的验证集中抽取图片。但是由于最初分割时没有考虑到可视化评估的完整部分,所以这里我选择了重新抽取。
# =============================================================================

def load_validation_set(sample_size=2):
    """
    重新从所有 uniqueID 中各抽取 n 张图片作为评估集。
    base_ds.fields 中有字段 'uniqueID',表示每张图片的 ID。
    n 可以自行修改,默认为 2。
    新增大类归属,对没有大类归属(category27 == -1)的图片，将其归于 "Unknown" 类。
    """
    sample_path = CONFIG['valuation_sample_path']
    parent_dir = os.path.dirname(sample_path)     # './preprocessed'
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    
    base_ds = THINGSImageDataset(
        meta_npz_path=CONFIG['meta_path'],
        image_root_dir=CONFIG['img_root'],
        category27_tsv_path=CONFIG['category27_tsv_path'],
        transform=transform
        )
    
    
    if os.path.isfile(sample_path):
        data = np.load(sample_path)
        sampled_indices = data['indices'].tolist()
        print(f"Loaded sampled_indices from {sample_path}")
    else:
        # 获取所有 uniqueID 列表
        uid_list = [u.decode() if isinstance(u, bytes) else u for u in base_ds.fields['uniqueID']]
        # 按 uniqueID 收集所有索引
        uid_to_indices = {}
        for idx, uid in enumerate(uid_list):
            uid_to_indices.setdefault(uid, []).append(idx)

        # # 过滤，只保留在 base_ds.uid_to_category 中有归属（≠ -1）的 uniqueID
        # valid_uid_to_indices = {}
        # for uid, indices in uid_to_indices.items():
        #     if base_ds.uid_to_category.get(uid, -1) != -1:
        #         valid_uid_to_indices[uid] = indices
        #     else:
        #         print(f"Filtering out '{uid}' without major category.")
        # uid_to_indices = valid_uid_to_indices

        # 不过滤 unknown 类 (category27 == -1)，统一保留，用于后续标注为 "Unknown"
        # 从每个 uniqueID 中随机抽取 n 个索引
        sampled_indices = []
        rng = np.random.RandomState(42)
        for uid, indices in uid_to_indices.items():
            if len(indices) >= sample_size:
                chosen = rng.choice(indices, size=sample_size, replace=False).tolist()
            else:
                chosen = indices[:]  # 如果该 uniqueID 下不足 n 张,就全部保留
                print(f"Warning: {uid} has only {len(indices)} images, sample {sample_size} images instead.")
            sampled_indices.extend(chosen)

        # 保存到 NPZ
        np.savez(sample_path, indices=np.array(sampled_indices, dtype=np.int64))
        print(f"Saved sampled_indices to {sample_path}")

    # 构建 Subset 来评估单张图片
    val_ds = Subset(base_ds, sampled_indices)
    return val_ds

# =============================================================================
# 2. 评估模型并收集特征和评分
# =============================================================================
def evaluate_model(model, val_ds):
    """
    对单张图片数据集 val_ds 进行评估：
    只计算每张图像的嵌入、预测评分、置信度（基于正态分布公式）。
    """
    model.eval()
    loader = DataLoader(
        val_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=(CONFIG['device'].type == 'cuda')
    )

    all_z = []           # 存储128维嵌入
    all_mean_true = []   # 存储真实的mean评分
    all_mean_pred = []   # 存储模型预测的mean评分
    confidence_pred = []     # 存储置信度 

    rating_cols = get_rating_cols()
    sd_cols = get_SD_cols()

    with torch.no_grad():
        pbar = tqdm(loader, desc='Evaluating', dynamic_ncols=True)
        for batch in pbar:
            img = batch['image'].to(CONFIG['device'], non_blocking=True)
            true_ratings = torch.stack(
                [batch[col] for col in rating_cols],
                dim=1
            ).to(CONFIG['device'], non_blocking=True)  # (B, M)

            # —— 拼接所有 SD 维度（shape: [B, M]） ——
            true_sds = torch.stack(
                [batch[col] for col in sd_cols],
                dim=1
            ).to(CONFIG['device'], non_blocking=True)  # (B, M)
            
            z, p_hat = model(img)
            all_z.append(z.cpu().numpy())
            all_mean_true.append(true_ratings.cpu().numpy())

            p_hat_np = p_hat.cpu().numpy()
            all_mean_pred.append(p_hat_np)
            # 计算基于正态分布的置信度
            sd_np = true_sds.cpu().numpy()
            sd_np = np.clip(sd_np, 1e-6, None)
            conf = np.exp(-0.5 * ((p_hat_np - true_ratings.cpu().numpy()) / sd_np) ** 2)
            confidence_pred.append(conf)

    all_z = np.concatenate(all_z, axis=0)
    all_mean_true = np.concatenate(all_mean_true, axis=0)
    all_mean_pred = np.concatenate(all_mean_pred, axis=0)
    confidence_pred = np.concatenate(confidence_pred, axis=0)

    mae = np.mean(np.abs(all_mean_pred - all_mean_true))
    print(f"Validation MAE: {mae:.4f}")

    return all_z, all_mean_true, all_mean_pred, confidence_pred

# =============================================================================
# 3. 嵌入维度 与 评分维度 相关性排名
# =============================================================================
def dimension_score_correlation_ranking(embeddings, raw_scores, rating_cols, top_k=5, save_dir=None):
    """
    计算每个嵌入维度 (128) 与每个评分维度 (M) 的 Pearson 相关系数,
    并针对每个评分维度输出相关性绝对值最高的前 top_k 个嵌入维度索引及系数。
    可选地将结果保存为 CSV。
    """
    # 计算相关系数：先拼接 embeddings (N,128) 和 raw_scores (N,M),按列计算协方差矩阵
    combined = np.concatenate([embeddings, raw_scores], axis=1)  # (N, 128+M)
    corr = np.corrcoef(combined, rowvar=False)  # (128+M, 128+M)
    corr_dim_score = corr[:embeddings.shape[1], embeddings.shape[1]:]  # (128, M)

    results = {}
    for j, col in enumerate(rating_cols):
        # 对第 j 个评分维度,按相关性绝对值排序
        abs_corrs = np.abs(corr_dim_score[:, j])
        idx_sorted = np.argsort(-abs_corrs)
        top_dims = idx_sorted[:top_k]
        top_values = corr_dim_score[top_dims, j]
        results[col] = list(zip(top_dims.tolist(), top_values.tolist()))

    # 如果需要保存到 CSV
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        rows = []
        for col, pairs in results.items():
            for dim_idx, corr_val in pairs:
                rows.append({'评分维度': col, '嵌入维度': dim_idx, '相关系数': corr_val})
        df_out = pd.DataFrame(rows)
        path = os.path.join(save_dir, 'dim_score_correlation_ranking.csv')
        df_out.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"Saved dimension-score correlation ranking at: {path}")

    return results

# =============================================================================
# 4. 读取或保存 evaluate_model 结果
# =============================================================================
def load_or_compute_evaluation(model, val_ds):
    """
    检查 CONFIG['val_results_dir'] 下是否已有保存的 all_z, all_mean_true, all_mean_pred, confidence_pred 文件。
    如果有,则读取并返回；否则调用 evaluate_model 计算并保存后再返回。
    """
    os.makedirs(CONFIG['val_results_dir'], exist_ok=True)
    z_path = os.path.join(CONFIG['val_results_dir'], 'all_z.npy')
    mean_true_path = os.path.join(CONFIG['val_results_dir'], 'all_mean_true.npy')
    mean_pred_path = os.path.join(CONFIG['val_results_dir'], 'all_mean_pred.npy')
    confidence_pred_path = os.path.join(CONFIG['val_results_dir'], 'confidence_pred.npy')

    # 若所有文件都存在,则直接加载
    if all(os.path.isfile(p) for p in [z_path, mean_true_path, mean_pred_path, confidence_pred_path]):
        print("Found cached evaluation results. Loading from disk...")
        all_z = np.load(z_path)
        all_mean_true = np.load(mean_true_path)
        all_mean_pred = np.load(mean_pred_path)
        confidence_pred = np.load(confidence_pred_path)
    else:
        # 否则重新计算并保存
        print("No cached results found. Running evaluation...")
        all_z, all_mean_true, all_mean_pred, confidence_pred = evaluate_model(model, val_ds)

        np.save(z_path, all_z)
        np.save(mean_true_path, all_mean_true)
        np.save(mean_pred_path, all_mean_pred)
        np.save(confidence_pred_path, confidence_pred)
        print(f"Saved evaluation results to {CONFIG['val_results_dir']}")

    return all_z, all_mean_true, all_mean_pred, confidence_pred

def visualize_example(model, val_ds):
    idx = CONFIG['example_index']
    example_vis_dir = os.path.join(CONFIG['results_dir'], 'images')
    example_imge_path = os.path.join(example_vis_dir, f"example_{idx}.png")
    example = val_ds[idx]

    rel_path = example['filepath']
    if isinstance(rel_path, bytes):
        rel_path = rel_path.decode()
    img_full_path = os.path.join(CONFIG['img_root'], rel_path)
    img = Image.open(img_full_path).convert("RGB")

    # 将图像保存下来
    img.save(example_imge_path)
    print(f"Saved example image at: {example_imge_path}")

    rating_cols = get_rating_cols()
    
    # 提取该样本的真实评分向量
    true_props = np.array(
        [example[col].item() for col in rating_cols],
        dtype=np.float32
    )

    # 模型预测
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    x = transform(img).unsqueeze(0).to(CONFIG['device'])
    model.eval()
    with torch.no_grad():
        z, p_hat = model(x)
    p_hat = p_hat.cpu().numpy().flatten()

    # 使用原始评分均值和方差构造正态分布,计算预测评分的置信度（0-1）
    SD_cols = get_SD_cols()
    true_sds = np.array(
        [example[col].item() for col in SD_cols],
        dtype=np.float32
    )

    # 为避免除以0,将方差中极小值裁剪
    true_sds = np.clip(true_sds, a_min=1e-6, a_max=None)
    # 置信度 = exp(-0.5 * ((pred - true_mean)/true_sd)^2)
    conf = np.exp(-0.5 * ((p_hat - true_props) / true_sds) ** 2)
    # 百分比表示
    conf_percent = conf * 100

    cols = get_rating_cols()
    x_pos = np.arange(len(rating_cols))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_pos - width/2, true_props, width, label='True')
    ax.bar(x_pos + width/2, p_hat, width, label='Predicted', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cols, rotation=45, ha='right')
    ax.set_ylabel('Rating')
    ax.set_title('True vs Predicted Ratings')
    ax.legend()
    # 在每个 x 轴标签上方添加置信度百分比注释
    for i, pct in enumerate(conf_percent):
        bar_height = max(true_props[i], p_hat[i])
        ax.text(i + width/2, bar_height + 0.02 * bar_height, f"{pct:.1f}%",
                ha='center', va='bottom')
    plt.tight_layout()
    os.makedirs(example_vis_dir, exist_ok=True)
    example_path = os.path.join(example_vis_dir, 'example_rating_comparison.png')
    plt.savefig(example_path)
    plt.show()
    plt.close()
    print(f"Saved example rating comparison at: {example_path}")

# =============================================================================
# 5. 主函数
# =============================================================================
def main():
    # 1. 加载模型
    model = setup_model(CONFIG['clip_model'], CONFIG['output_dim'], CONFIG['device'])
    ckpt_path = f"./results/models/model_epoch_{CONFIG['resume_epoch']}.pth"
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found.")
    state_dict = torch.load(ckpt_path, map_location=CONFIG['device'],weights_only=True)
    model.load_state_dict(state_dict)
    model.to(CONFIG['device'])

    # 2. 加载验证集
    val_ds = load_validation_set(CONFIG['uniqueID_sample_size'])

    # 3. 评估模型,获得 128 维嵌入与预测/真实评分、置信度
    all_z, all_mean_true, all_mean_pred, confidence_pred =  load_or_compute_evaluation(model, val_ds)

    raw_scores = all_mean_true
    rating_cols = get_rating_cols()
    save_vis_dir = os.path.join(CONFIG['results_dir'], 'images')
    
    # 5. 2D 可视化：t-SNE & UMAP
    if CONFIG['2d_visualizing']:
        # 先提取每个样本对应的大类索引
        categories_orig = [ val_ds[i]['category27'] for i in range(len(val_ds)) ]
        

        # 加载大类名称列表,并追加unknown类
        category_names = load_category_names(CONFIG['category27_tsv_path'])
        category_names.append('Unknown')

        # 将-1映射到unknown类
        unk_idx = len(category_names) - 1
        categories = [c if c != -1 else unk_idx for c in categories_orig]
        
        # ----检查类别下样本数量----
        # from collections import Counter
        # # categories 列表里包含了所有样本的类别编号（-1 已被映射为 unk_idx）
        # cnt = Counter(categories)
        # print("各类别样本数量：")
        # for cat_idx, n in cnt.items():
        #     # 如果是 unknown 类，就特别标注
        #     label = category_names[cat_idx]
        #     print(f"  类别 {cat_idx} ({label}): {n} 张")

        # —— 只保留需要可视化的类别（例如选 0,1,2,3,4,5,6,7 共8 个） ——
        selected_cats = [0,1,2,3,4,5,6,7]
        mask = [c in selected_cats for c in categories]
        z_sel = all_z[mask]
        cats_sel = [c for c in categories if c in selected_cats]
        names_sel = category_names  # 也可只把需要的名字摘出来，但下面直接用索引也行
        
        tsne_visualization(z_sel, cats_sel, names_sel, save_vis_dir)
        umap_visualization(z_sel, cats_sel, names_sel, save_vis_dir)

        # tsne_visualization(all_z, categories, category_names, save_vis_dir)
        # umap_visualization(all_z, categories, category_names, save_vis_dir)

    if CONFIG['pca_analysis']:
        # 6. PCA 与 评分相关性分析
        pca_score_correlation(all_z, raw_scores, rating_cols, save_vis_dir)

        # 7. 嵌入维度 与 评分维度 相关性排名
        ranking = dimension_score_correlation_ranking(
            all_z, raw_scores, rating_cols,
            top_k=5,
            save_dir=save_vis_dir
        )
        # 打印前5个维度示例
        for col, pairs in ranking.items():
            print(f"评分维度 '{col}' 最相关的前5个嵌入维度 (维度索引, 相关系数):")
            for dim_idx, corr_val in pairs:
                print(f"  维度 {dim_idx}: 相关系数 {corr_val:.4f}")
            print()

    # 如需可视化单个示例（可选）
    if CONFIG['example_visualizing']:
        visualize_example(model, val_ds)

    # 8. （3D 可视化示例,若需要）
    if CONFIG['example_3d_visualizing']:
        # 仅对示例样本做3D可视化：真实与预测评分、置信度,略去,此处留空或按需实现
        pass


if __name__ == '__main__':
    main()
