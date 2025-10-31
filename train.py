"""
Training script for Custom CNN on CIFAR-10

Features:
- Training loop with validation
- Learning rate scheduling
- Model checkpointing
- Training history logging
- Progress tracking with tqdm
"""

import os
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import CustomCNN
from dataset import get_data_loaders


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train model for one epoch.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc='Training', leave=False)

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, test_loader, criterion, device):
    """
    Validate model on test set.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Validation', leave=False):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def train_model(args):
    """
    Main training function.

    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader, classes = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Initialize model
    print("\nInitializing model...")
    model = CustomCNN(num_classes=10, dropout_rate=args.dropout)
    model = model.to(device)
    print(f"Total parameters: {model.get_num_parameters():,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                          weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Create checkpoint directory
    os.makedirs(args.save_path, exist_ok=True)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }

    best_acc = 0.0
    print("\nStarting training...")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)

        # Print epoch results
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }
            save_file = os.path.join(args.save_path, 'best_model.pth')
            torch.save(checkpoint, save_file)
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")

    print("\n" + "=" * 60)
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")

    # Save training history
    history_file = os.path.join(args.save_path, 'training_history.json')
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_file}")


def main():
    parser = argparse.ArgumentParser(description='Train Custom CNN on CIFAR-10')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Initial learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (L2 regularization) (default: 1e-4)')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate (default: 0.5)')

    # System settings
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use: cuda or cpu (default: cuda)')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of data loading workers (default: 2)')
    parser.add_argument('--save-path', type=str, default='./checkpoints',
                       help='Path to save model checkpoints (default: ./checkpoints)')

    args = parser.parse_args()

    # Print configuration
    print("Training Configuration:")
    print("-" * 60)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("-" * 60)

    # Train model
    train_model(args)


if __name__ == "__main__":
    main()
