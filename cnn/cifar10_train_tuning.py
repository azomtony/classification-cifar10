# cifar10_train_tuning.py

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from lightning_fabric.plugins.environments import LightningEnvironment
from pytorch_lightning.callbacks import TQDMProgressBar
from cifarcnn import CifarCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =====================================
# Data Preparation
# =====================================

# Data augmentation for training set
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# No augmentation for test/validation set
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 64

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=2)

val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_test)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# =====================================
# Custom Progress Bar
# =====================================

class TrainOnlyProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.disable = True
        return bar

# =====================================
# Hyperparameter Tuning
# =====================================

learning_rates = [1e-2, 1e-3, 1e-4]
dropout_convs = [0.25, 0.4]
dropout_fcs = [0.5, 0.3]

results = []
for lr in learning_rates:
    for d_conv in dropout_convs:
        for d_fc in dropout_fcs:
            print(f"\nTraining model with lr={lr}, dropout_conv={d_conv}, dropout_fc={d_fc}\n")

            model = CifarCNN(learning_rate=lr, dropout_conv=d_conv, dropout_fc=d_fc)
            model = model.to(device)

            trainer = Trainer(
                max_epochs=15,  
                callbacks=[TrainOnlyProgressBar()],
                default_root_dir=f"./checkpoints/lr_{lr}_dconv_{d_conv}_dfc_{d_fc}",
                plugins=[LightningEnvironment()],
                enable_progress_bar=True,
                logger=False  
            )

            trainer.fit(model, train_loader, val_loader)

            val_acc = trainer.callback_metrics["val_acc"].item()

            results.append((lr, d_conv, d_fc, val_acc))

# =====================================
# Tuning Results
# =====================================

# Sorting by best validation accuracy
results.sort(key=lambda x: x[3], reverse=True)

print("\nBest Hyperparameters Found:")
print(f"Learning Rate: {results[0][0]}")
print(f"Dropout Conv: {results[0][1]}")
print(f"Dropout FC: {results[0][2]}")
print(f"Validation Accuracy: {results[0][3]:.4f}")


results_df = pd.DataFrame(results, columns=["Learning Rate", "Dropout Conv", "Dropout FC", "Validation Accuracy"])
results_df.to_csv("hyperparameter_tuning_results.csv", index=False)
print("\nAll results saved to hyperparameter_tuning_results.csv")