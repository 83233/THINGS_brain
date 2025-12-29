import torch

CONFIG = {
    'meta_path': './preprocessed/things_image.npz',
    'img_root': './image/_image_database_things',
    'split_save_dir': './splits',
    'clip_model': 'ViT-B/32',
    'valuation_sample_path': './preprocessed/valuation_sample.npz',
    'category27_tsv_path':'./image/03_category-level/category27_manual.tsv',
    'output_dim': 128,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'batch_size': 64,
    'num_workers': 4,
    'epochs': 60,
    'resume_epoch': 50,
    'dim_reduction_method': 'tsne',  # 可选 'pca', 'tsne', 'umap'
    'pca_dims': 32,
    'vis_dims': [0, 1, 2],
    'example_index': 2,
    'example_visualizing': False,
    '2d_visualizing': True,
    'example_3d_visualizing': False,
    'pca_analysis': False,
    'results_dir': './results',
    'val_results_dir': 'val_results',
    'uniqueID_sample_size': 2,
}

rating_cols = [
            'property_manmade_mean','property_precious_mean','property_lives_mean',
            'property_heavy_mean','property_natural_mean','property_moves_mean',
            'property_grasp_mean','property_hold_mean','property_be-moved_mean','property_pleasant_mean'
        ]

SD_cols = [
            'property_manmade_SD','property_precious_SD','property_lives_SD',
            'property_heavy_SD','property_natural_SD','property_moves_SD',
            'property_grasp_SD','property_hold_SD','property_be-moved_SD','property_pleasant_SD'
        ]

def get_rating_cols():
    return rating_cols

def get_SD_cols():
    return SD_cols

def get_all_cols():
    return rating_cols + SD_cols
