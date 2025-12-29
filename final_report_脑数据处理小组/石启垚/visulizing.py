import matplotlib.pyplot as plt

def plot_loss_curves(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Testing Loss Curves')
    plt.savefig('./results/loss_curves.png')
    plt.show()