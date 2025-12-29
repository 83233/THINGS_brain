import torch
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from THINGSdataset1 import THINGSDataset
import torchvision.transforms as T
from torch.utils.data import random_split
import matplotlib
matplotlib.use('TkAgg')  
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  

class FMRI3DClassificationDataset(Dataset):
    def __init__(self, base_dataset, train=False):
        self.base_dataset = base_dataset
        self.train = train  

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        fmri = sample['fmri'].squeeze(0)[::2, ::2, ::2]  # 下采样
        fmri = (fmri - fmri.mean()) / (fmri.std() + 1e-6)  # 标准化
        fmri = fmri.unsqueeze(0) 

        # 训练阶段加噪声
        if self.train:
            noise = torch.randn_like(fmri) * 0.05
            fmri = fmri + noise

        label_idx = sample['category_id']
        return fmri.float(), label_idx


class FMRI3DCNNClassifier(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            conv_out = self.conv(x)
            flattened_size = conv_out.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def evaluate(model, dataloader, topk=5):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for fmri, label in dataloader:
            output = model(fmri)
            topk_preds = torch.topk(output, k=topk, dim=1).indices
            correct += sum(label[i].item() in topk_preds[i] for i in range(len(label)))
            total += label.size(0)
    acc = correct / total * 100
    print(f"Top-{topk} Accuracy: {acc:.2f}%")
    model.train()


def plot_train_val_loss(train_loss, val_loss):
    import matplotlib.pyplot as plt  

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_loss) + 1), train_loss, marker='o', label='Train Loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, marker='s', label='Val Loss')
    plt.title("Training & Validation Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    with open('./class_mapping.json', 'r') as f:
        word_to_label = json.load(f)

    base_dataset = THINGSDataset(
        fmri_root_dir="./betas_vol/scalematched/sub-01",
        conditions_tsv_pattern="sub-01_ses-things01_run-{run:02d}_conditions.tsv",
        image_root_dir="./image/_image_database_things",
        image_paths_csv="./image/01_image-level/image-paths.csv",
        property_ratings_tsv="./image/02_object-level/_property-ratings.tsv"
    )

    dataset = FMRI3DClassificationDataset(base_dataset, train=True)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    val_dataset.dataset.train = False
    test_dataset.dataset.train = False
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    sample_input, sample_idx = train_dataset[0]
    input_shape = sample_input.shape
    num_classes = len(word_to_label)

    model = FMRI3DCNNClassifier(input_shape, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 记录训练和验证 loss
    train_losses = []
    val_losses = []

    # 训练
    for epoch in range(5):
        total_loss = 0
        model.train()
        for fmri, label in train_loader:
            output = model(fmri)
            label = label.long()
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for fmri, label in val_loader:
                output = model(fmri)
                label = label.long()
                loss = criterion(output, label)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 测试集评估
    evaluate(model, test_loader, topk=10)
    plot_train_val_loss(train_losses, val_losses)