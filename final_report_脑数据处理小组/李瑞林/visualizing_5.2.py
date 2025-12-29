import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curves(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')  # 明确标签为训练损失
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Curves')  # 标题与实际绘制内容一致（若后续添加测试损失再调整）
    plt.savefig('loss_trans_CSNM.png', dpi=300, bbox_inches='tight')  # 保存为高清图片
    plt.close()