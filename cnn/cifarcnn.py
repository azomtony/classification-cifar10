import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class CifarCNN(pl.LightningModule):
    """
    A 4-layer Convolutional Neural Network for classification,
    """

    def __init__(self, learning_rate=1e-3, dropout_conv=0.25, dropout_fc=0.5):
        """
        Args:
            learning_rate (float): learning rate for optimizer
            dropout_conv (float): dropout rate after conv layers
            dropout_fc (float): dropout rate after fully connected layer
        """
        super().__init__()
        self.save_hyperparameters()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv_layer = nn.Dropout2d(dropout_conv)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 2 * 2, 512)  
        self.dropout_fc_layer = nn.Dropout(dropout_fc)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  
        x = self.dropout_conv_layer(x)
        x = torch.flatten(x, 1)  
        x = F.relu(self.fc1(x))
        x = self.dropout_fc_layer(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, targets)
        acc = (outputs.argmax(dim=1) == targets).float().mean()
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, targets)
        acc = (outputs.argmax(dim=1) == targets).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, targets)
        acc = (outputs.argmax(dim=1) == targets).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
