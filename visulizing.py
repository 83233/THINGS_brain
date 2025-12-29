import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import umap
import re
import os
import numpy as np

from globals import rating_cols

def plot_loss_curves(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Curves')
    plt.savefig('./results/loss_curves.png')
    plt.show()

def read_loss_history_from_file(file_path):
    loss_history = []
    # 定义一个正则，用来从每行中提取 Avg Loss=后面的浮点数
    pattern = re.compile(r"Avg Loss=([\d\.eE+-]+)")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = pattern.search(line)
            if match:
                loss_value = float(match.group(1))
                loss_history.append(loss_value)
            else:
                # 如果某行不符合预期格式，可以打印提醒或直接跳过
                print(f"[Warning] 无法解析这一行的 Loss 值：{line}")
    return loss_history

def tsne_visualization(embeddings,categories,category_names, save_dir):
    """
    对 embeddings (N, 128) 使用 t-SNE 降到 2D,
    并根据 categories 做分组绘制散点，图例显示 category_names。
    """
    os.makedirs(save_dir, exist_ok=True)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_tsne = tsne.fit_transform(embeddings)  # (N, 2)

    plt.figure(figsize=(6, 5))
    unique_cats = sorted(set(categories))
    for cat in unique_cats:
        mask = [c == cat for c in categories]
        plt.scatter(
            emb_tsne[mask, 0], emb_tsne[mask, 1],
            label=category_names[cat], s=15, alpha=0.8
        )
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title("t-SNE 2D vislualization")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.tight_layout()
    path = os.path.join(save_dir, 'tsne_2d.png')
    plt.savefig(path)
    plt.show()
    plt.close()
    print(f"Saved t-SNE 2D plot at: {path}")

    # =====计算聚类系数(Silhouette Score)=====
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    n_clusters = len(set(categories))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(emb_tsne)
    sil = silhouette_score(emb_tsne, kmeans.labels_)
    print(f"t-SNE 聚类系数 (Silhouette Score), {n_clusters} 簇: {sil:.4f}")

def umap_visualization(embeddings, categories, category_names, save_dir):
    """
    对 embeddings (N, 128) 使用 UMAP 降到 2D,
    并根据 categories 做分组绘制散点，图例显示 category_names。
    """
    os.makedirs(save_dir, exist_ok=True)
    reducer = umap.UMAP(n_components=2, random_state=42)
    emb_umap = reducer.fit_transform(embeddings)  # (N, 2)

    plt.figure(figsize=(6, 5))
    unique_cats = sorted(set(categories))
    for cat in unique_cats:
        mask = [c == cat for c in categories]
        plt.scatter(
            emb_umap[mask, 0], emb_umap[mask, 1],
            label=category_names[cat], s=15, alpha=0.8
        )
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("UMAP 2D visualization ")
    plt.xlabel("UMAP dimension 1")
    plt.ylabel("UMAP dimension 2")
    plt.tight_layout()
    path = os.path.join(save_dir, 'umap_2d.png')
    plt.savefig(path)
    plt.show()
    plt.close()
    print(f"Saved UMAP 2D plot at: {path}")

    # =====计算聚类系数(Silhouette Score)=====
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    n_clusters = len(set(categories))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(emb_umap)
    sil = silhouette_score(emb_umap, kmeans.labels_)
    print(f"UMAP 聚类系数 (Silhouette Score), {n_clusters} 簇: {sil:.4f}")

def pca_score_correlation(embeddings, raw_scores, rating_cols, save_dir):
    """
    对 embeddings 进行 PCA,提取前10个主成分,并计算与 raw_scores 各评分维度的 Pearson 相关系数。
    保存热力图到文件。
    """
    os.makedirs(save_dir, exist_ok=True)
    n_comps = 10
    pca = PCA(n_components=n_comps)
    pcs = pca.fit_transform(embeddings)  # (N, n_comps)

    # 构建 DataFrame,方便计算相关性矩阵
    df = pd.DataFrame(pcs, columns=[f'PC{i+1}' for i in range(n_comps)])
    for j, col in enumerate(rating_cols):
        df[col] = raw_scores[:, j]

    corr_matrix = df.corr(method='pearson')
    # 只关心评分与主成分之间的相关性
    corr_pc_score = corr_matrix.loc[rating_cols, [f'PC{i+1}' for i in range(n_comps)]]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_pc_score,
        annot=True, fmt=".2f",
        center=0, cmap='RdBu_r',
        cbar_kws={'label': 'Pearson correlation coefficient'}
    )
    plt.title("Heatmap of the correlation between evaluation dimensions and PCA main components (top 10)")
    plt.xlabel("principal component")
    plt.ylabel("Rating dimension")
    plt.tight_layout()
    path = os.path.join(save_dir, 'pca_score_correlation_heatmap.png')
    plt.savefig(path)
    plt.show()
    plt.close()
    print(f"Saved PCA-score correlation heatmap at: {path}")

def plot_2d_heatmap(weights, save_dir):
    """
    绘制 2D 热力图，显示权重矩阵。
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    sns.heatmap(weights, cmap='viridis',
                xticklabels=np.arange(weights.shape[1]),
                yticklabels=rating_cols)
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Rating Dimension')
    plt.title('Embedding Weights Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'property_head_weight_heatmap.png'))
    plt.show()
    print(f"Saved weight heatmap at: {os.path.join(save_dir, 'property_head_weight_heatmap.png')}")

if __name__ == '__main__':
    loss_file = "./results/loss.txt"  # 根据实际情况填写路径
    loss_history = read_loss_history_from_file(loss_file)
    plot_loss_curves(loss_history)