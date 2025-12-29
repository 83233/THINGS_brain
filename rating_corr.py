import torch
import os
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import numpy as np


from globals import CONFIG, get_rating_cols, get_SD_cols
from evaluating import setup_model,load_validation_dataset,load_or_compute_evaluation
from visulizing import plot_2d_heatmap





if __name__ == '__main__':
    # 1. 加载模型
    model = setup_model(CONFIG['clip_model'], CONFIG['output_dim'], CONFIG['device'])
    ckpt_path = f"./results/models/model_epoch_{CONFIG['resume_epoch']}.pth"
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found.")
    state_dict = torch.load(ckpt_path, map_location=CONFIG['device'],weights_only=True)
    model.load_state_dict(state_dict)
    model.to(CONFIG['device'])

    save_vis_dir = os.path.join(CONFIG['results_dir'], 'images')
    # 2. 加载验证集
    val_ds = load_validation_dataset(CONFIG['uniqueID_sample_size'])

    # 3. 评估模型,获得 128 维嵌入与预测/真实评分、置信度
    all_z, all_mean_true, all_mean_pred, confidence_pred =  load_or_compute_evaluation(model, val_ds)

    # ===== 1. 所有属性的预测 vs 真实 相关性检验 =====
    rating_cols = get_rating_cols()
    corrs = {}
    for i, col in enumerate(rating_cols):
        r, _ = pearsonr(all_mean_pred[:, i], all_mean_true[:, i])
        corrs[col] = r
        print(f"属性 '{col}' 预测 vs 真实 Pearson r: {r:.4f}")

    # ===== 2. 可视化 property_head 权重 & 失连接实验 =====
    # 2.1 取出模型中的全连接层权重
    W = model.property_head.weight.data.cpu().numpy()   # shape: (属性数, 隐空间维度)

    # 2.2 绘制权重热力图
    plot_2d_heatmap(W, save_dir=save_vis_dir)

    # 2.3 Top-k 维度失连接后影响
    top_k = 5
    for j, col in enumerate(rating_cols):
        dims = np.argsort(-np.abs(W[j]))[:top_k]
        Z_ablated = all_z.copy()
        Z_ablated[:, dims] = 0
        # 直接用权重矩阵进行预测
        p_hat_abl = Z_ablated.dot(W.T)                  # (N, 属性数)
        r_abl, _ = pearsonr(p_hat_abl[:, j], all_mean_true[:, j])
        print(f"属性 '{col}' 失连接维度 {dims.tolist()} 后 Pearson r: {r_abl:.4f}")
