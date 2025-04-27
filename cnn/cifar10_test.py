# cifar10_final_test.py

import torch
import torchvision
import torchvision.transforms as transforms
from cifarcnn import CifarCNN
from pytorch_lightning import Trainer
from lightning_fabric.plugins.environments import LightningEnvironment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# =========================================
# Data Loading
# =========================================
def load_data(batch_size=64):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                 download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)
    return test_loader

# =========================================
# Model Loading
# =========================================
def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CifarCNN.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    return model, device

# =========================================
# Model Evaluation
# =========================================
def evaluate_model(model, device, dataloader, classes):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)

def print_test_accuracy(all_labels, all_preds):
    accuracy = (all_preds == all_labels).sum() / len(all_labels)
    print(f"\nOverall Test Accuracy: {accuracy * 100:.2f}%")

# =========================================
# Confusion Matrix
# =========================================
def plot_confusion_matrix(all_labels, all_preds, classes, save_path="confusion_matrix_cnn.png"):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.title("Confusion Matrix (CNN)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    # plt.show()

# =========================================
# Precision, Recall, F1 Table
# =========================================
def plot_classification_report(all_labels, all_preds, classes, save_path="metrics_table.png"):
    precision_per_class = precision_score(all_labels, all_preds, average=None)
    recall_per_class = recall_score(all_labels, all_preds, average=None)
    f1_per_class = f1_score(all_labels, all_preds, average=None)

    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    metrics_table = pd.DataFrame({
        'Precision': list(precision_per_class) + [precision_macro],
        'Recall': list(recall_per_class) + [recall_macro],
        'F1-Score': list(f1_per_class) + [f1_macro]
    }, index=classes + ['Average'])

    print(metrics_table.round(4))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.axis('tight')

    table = ax.table(
        cellText=metrics_table.round(4).values,
        colLabels=metrics_table.columns,
        rowLabels=metrics_table.index,
        loc='center',
        cellLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    # plt.show()

# =========================================
# Train vs Test Accuracy Curve
# =========================================
def plot_accuracy_curves(metrics_csv_path, save_path="accuracy_curve_cnn.png"):
    df = pd.read_csv(metrics_csv_path)

    val_df = df.dropna(subset=["val_acc"])
    train_acc_per_epoch = df.groupby("epoch")["train_acc"].mean()
    val_acc_per_epoch = val_df.set_index("epoch")["val_acc"]

    plt.figure(figsize=(8, 6))
    plt.plot(train_acc_per_epoch.index, train_acc_per_epoch, label="Train Accuracy")
    plt.plot(val_acc_per_epoch.index, val_acc_per_epoch, label="Test Accuracy")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("Accuracy Curve (CNN)")
    plt.ylim(0, 1)
    plt.xlim(0, 50)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    # plt.show()

# =========================================
# Train vs Test Loss Curve
# =========================================
def plot_loss_curves(metrics_csv_path, save_path="loss_curve_cnn.png"):
    df = pd.read_csv(metrics_csv_path)

    val_df = df.dropna(subset=["val_loss"])
    train_loss_per_epoch = df.groupby("epoch")["train_loss"].mean()
    val_loss_per_epoch = val_df.set_index("epoch")["val_loss"]

    plt.figure(figsize=(8, 6))
    plt.plot(train_loss_per_epoch.index, train_loss_per_epoch, label="Train Loss")
    plt.plot(val_loss_per_epoch.index, val_loss_per_epoch, label="Test Loss")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("Loss Curve (CNN)")
    plt.ylim(0, 1.7)
    plt.xlim(0, 50)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    checkpoint_path = "final_checkpoints/lightning_logs/version_0/checkpoints/epoch=49-step=39100.ckpt"
    metrics_csv_path = "final_checkpoints/lightning_logs/version_0/metrics.csv"
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

    test_loader = load_data(batch_size=64)
    model, device = load_model(checkpoint_path)
    all_labels, all_preds = evaluate_model(model, device, test_loader, classes)
    print_test_accuracy(all_labels, all_preds)

    plot_confusion_matrix(all_labels, all_preds, classes)
    plot_classification_report(all_labels, all_preds, classes)
    plot_loss_curves(metrics_csv_path)
    plot_accuracy_curves(metrics_csv_path)