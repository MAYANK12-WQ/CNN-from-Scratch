"""
Custom CNN Architecture for CIFAR-10 Classification

This module implements a convolutional neural network from scratch with:
- 3 Convolutional blocks with batch normalization
- MaxPooling for spatial downsampling
- Fully connected layers with dropout
- Total parameters: ~2.8M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    """
    Custom CNN Architecture for image classification.

    Architecture:
        Conv Block 1: Conv(3→64) + BatchNorm + ReLU + MaxPool
        Conv Block 2: Conv(64→128) + BatchNorm + ReLU + MaxPool
        Conv Block 3: Conv(128→256) + BatchNorm + ReLU + MaxPool
        FC Layers: Flatten → FC(256*4*4→512) → Dropout → FC(512→10)

    Args:
        num_classes (int): Number of output classes (default: 10 for CIFAR-10)
        dropout_rate (float): Dropout probability (default: 0.5)
    """

    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(CustomCNN, self).__init__()

        # Convolutional Block 1: 3 → 64 channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 → 16x16

        # Convolutional Block 2: 64 → 128 channels
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 → 8x8

        # Convolutional Block 3: 128 → 256 channels
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8 → 4x4

        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Flatten
        x = x.view(x.size(0), -1)  # (batch_size, 256*4*4)

        # Fully Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def _initialize_weights(self):
        """
        Initialize network weights using Kaiming initialization for ReLU activations.
        This helps with training stability and convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def get_num_parameters(self):
        """
        Calculate total number of trainable parameters.

        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_model():
    """Test function to verify model architecture and output shape."""
    model = CustomCNN(num_classes=10)

    # Create dummy input (batch_size=4, channels=3, height=32, width=32)
    x = torch.randn(4, 3, 32, 32)

    # Forward pass
    output = model(x)

    print("Model Architecture Test:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {model.get_num_parameters():,}")
    print("\nModel Summary:")
    print(model)


if __name__ == "__main__":
    test_model()
