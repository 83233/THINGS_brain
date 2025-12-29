import numpy as np
import matplotlib.pyplot as plt
from nilearn import image, datasets, plotting

# --- 配置参数 ---

# 1. 指向您降维后保存的 .npz 文件
#    注意：在前一个脚本中您将它保存为了 .npz 文件，键为 'fmri'
FMRI_COMPRESSED_PATH = "./brain_score_code/things_fmri_compressed_700d.npy"

# 2. 我们将再次“获取”Schaefer模板，以便使用它的坐标和网络信息
#    nilearn会使用缓存，所以不会重新下载
NUM_PARCELS = 700

def visualize_fmri_connectivity(fmri_path, num_parcels):
    """
    加载降维后的fMRI数据,计算并可视化其功能连接矩阵和脑网络图。
    """
    # --- 1. 加载降维后的fMRI数据 ---
    print(f"正在加载降维后的fMRI数据: {fmri_path}")
    try:
        data_npz = np.load(fmri_path)
        # 假设您在保存时使用的键是 'fmri'
        fmri_matrix = data_npz['fmri']
    except KeyError:
        # 如果您之前是直接用 np.save 保存的 .npy 文件，请使用下面这行
        # fmri_matrix = np.load(fmri_path)
        print(f"错误：在 {fmri_path} 中找不到键 'fmri'。请确认文件格式或键名。")
        return
        
    print(f"数据加载成功，维度: {fmri_matrix.shape}")

    # --- 2. 识别并剔除恒定列 (这是解决问题的关键) ---
    print("正在诊断并过滤恒定信号分区...")
    variance = fmri_matrix.var(axis=0)
    # 创建一个布尔掩码，只保留方差大于0（即非恒定）的分区
    valid_parcels_mask = variance > 0
    
    num_valid_parcels = np.sum(valid_parcels_mask)
    num_constant_parcels = num_parcels - num_valid_parcels
    
    if num_constant_parcels > 0:
        print(f"[诊断结果] 发现 {num_constant_parcels} 个恒定分区 (无信号或全零)。")
        print(f"保留 {num_valid_parcels} 个有信号变化的分区进行分析。")
    else:
        print("[诊断结果] 所有分区均有信号变化。")

    # 应用掩码，只保留有效分区的数据
    fmri_matrix_filtered = fmri_matrix[:, valid_parcels_mask]

    # --- 3. 计算功能连接矩阵 ---
    # 我们计算不同分区在所有样本上的活动相关性
    # np.corrcoef 需要特征在行上，所以我们先转置
    print("正在计算功能连接矩阵...")
    correlation_matrix = np.corrcoef(fmri_matrix_filtered.T)
    # 将对角线（自身相关性）设为0，以便更好地可视化
    np.fill_diagonal(correlation_matrix, 0)
    print(f"功能连接矩阵计算完成，维度: {correlation_matrix.shape}")

    # --- 4. 获取分区的坐标和网络信息 ---
    print("正在获取分区坐标和网络信息...")
    schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=num_parcels)
    
    # 获取所有700个分区的中心坐标
    all_coords = plotting.find_parcellation_cut_coords(schaefer_atlas.maps)
    # 获取所有700个分区所属的网络名称 (如 'VisCent', 'Default')
    all_network_labels = schaefer_atlas.labels
    print(all_network_labels)
    # 使用相同的掩码过滤坐标和网络标签，确保与连接矩阵对齐
    valid_coords = all_coords[valid_parcels_mask]
    # valid_network_labels = np.array(all_network_labels, dtype=object)[valid_parcels_mask]
    
    # --- 5. 可视化 ---
    print("开始绘图...")
    
    # a) 绘制功能连接矩阵热图
    # plt.figure(figsize=(10, 8))
    plt.tight_layout() 
    plotting.plot_matrix(
        correlation_matrix,
        # labels=valid_network_labels,
        colorbar=True,
        vmax=0.8, vmin=-0.8,
        title=f"Functional connectivity matrix ({num_valid_parcels} valid parcels)"
    )

    # b) 绘制脑网络连接图 (Connectome)
    # 我们只显示最强的连接，否则会一团乱麻。这里我们显示前1%的强连接。
    # plt.figure(figsize=(10, 8))
    plotting.plot_connectome(
        adjacency_matrix=correlation_matrix,
        node_coords=valid_coords,
        edge_threshold="99.5%",  # 只显示数值排在前1%的连接
        node_size=10,
        title=f"Brain network connectivity map (top 1% connections)"
    )
    
    # 显示所有图像
    plotting.show()


if __name__ == '__main__':
    visualize_fmri_connectivity(FMRI_COMPRESSED_PATH, NUM_PARCELS)