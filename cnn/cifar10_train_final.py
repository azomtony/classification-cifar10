import torch
import torchvision
import torchvision.transforms as transforms
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
# Training
# =====================================

# Instantiate model with best hyperparameters
model = CifarCNN(learning_rate=1e-3, dropout_conv=0.25, dropout_fc=0.3)
model = model.to(device)

trainer = Trainer(
    max_epochs=50,
    callbacks=[TrainOnlyProgressBar()],
    default_root_dir="./final_checkpoints/",
    plugins=[LightningEnvironment()],
)

# Train the model
trainer.fit(model, train_loader, val_loader)