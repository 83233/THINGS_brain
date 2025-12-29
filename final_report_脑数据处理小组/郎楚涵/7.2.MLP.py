import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from THINGSdataset1 import THINGSDataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('TkAgg')  # 避免 PyCharm 兼容性问题

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#  数据集封装 
class FMRIClassificationDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        fmri = sample['fmri'].squeeze(0)[::2, :2:, ::2].flatten()
        label = sample['category_id']
        return fmri, label


# MLP 
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


# eval函数
def evaluate(model, dataloader, topk=10):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for fmri, label in dataloader:
            pred = model(fmri)
            topk_preds = torch.topk(pred, k=topk, dim=1).indices
            correct += sum(label[i].item() in topk_preds[i] for i in range(len(label)))
            total += label.size(0)
    print(f"Top-{topk} Accuracy: {correct / total * 100:.2f}%")
    model.train()


# 可视化
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
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    dataset = THINGSDataset(
        fmri_root_dir="./betas_vol/scalematched/sub-01",
        conditions_tsv_pattern="sub-01_ses-things01_run-{run:02d}_conditions.tsv",
        image_root_dir="./image/_image_database_things",
        image_paths_csv="./image/01_image-level/image-paths.csv",
        property_ratings_tsv="./image/02_object-level/_property-ratings.tsv",
        transform=transform
    )

    clf_dataset = FMRIClassificationDataset(dataset)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(clf_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    sample_input, _ = train_dataset[0]
    input_dim = sample_input.numel()
    num_classes = len(dataset.class_to_idx)

    model = MLPClassifier(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loss_history = []
    val_loss_history = []

    for epoch in range(5):
        model.train()
        total_train_loss = 0
        for fmri, label in train_loader:
            output = model(fmri)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for fmri, label in val_loader:
                output = model(fmri)
                loss = criterion(output, label)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    evaluate(model, test_loader)

    plot_train_val_loss(train_loss_history, val_loss_history)
