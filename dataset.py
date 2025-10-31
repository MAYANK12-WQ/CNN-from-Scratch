"""
Dataset loading and preprocessing for CIFAR-10

Handles:
- Downloading CIFAR-10 dataset
- Data augmentation (training)
- Normalization
- DataLoader creation
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# CIFAR-10 statistics for normalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_data_loaders(batch_size=128, num_workers=2, data_dir='./data'):
    """
    Create training and testing data loaders for CIFAR-10.

    Training augmentation:
        - Random horizontal flip
        - Random crop with padding
        - Normalization

    Test preprocessing:
        - Normalization only

    Args:
        batch_size (int): Batch size for training and testing
        num_workers (int): Number of worker processes for data loading
        data_dir (str): Directory to store dataset

    Returns:
        tuple: (train_loader, test_loader, classes)
    """

    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    # Download and load training dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    # Download and load test dataset
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # CIFAR-10 class names
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"Dataset loaded successfully!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {classes}")
    print(f"Batch size: {batch_size}")

    return train_loader, test_loader, classes


def get_sample_batch(data_loader):
    """
    Get a single batch of data for visualization or testing.

    Args:
        data_loader: PyTorch DataLoader

    Returns:
        tuple: (images, labels) batch
    """
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    return images, labels


def denormalize(tensor, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    """
    Denormalize a tensor for visualization.

    Args:
        tensor (torch.Tensor): Normalized image tensor
        mean (tuple): Mean values used for normalization
        std (tuple): Std values used for normalization

    Returns:
        torch.Tensor: Denormalized tensor
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


if __name__ == "__main__":
    # Test data loading
    train_loader, test_loader, classes = get_data_loaders(batch_size=128)

    # Get a sample batch
    images, labels = get_sample_batch(train_loader)
    print(f"\nSample batch shape: {images.shape}")
    print(f"Sample labels shape: {labels.shape}")
    print(f"First 10 labels: {labels[:10]}")
