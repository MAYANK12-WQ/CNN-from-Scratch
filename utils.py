"""
Utility functions for visualization and analysis

Features:
- Filter/kernel visualization
- Feature map visualization
- Training history plots
- Grad-CAM implementation
- Confusion matrix
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report


def visualize_filters(model, layer_name='conv1', num_filters=64, save_path=None):
    """
    Visualize learned convolutional filters.

    Args:
        model: Trained CNN model
        layer_name (str): Name of convolutional layer
        num_filters (int): Number of filters to visualize
        save_path (str): Path to save figure (optional)
    """
    # Get the layer
    layer = getattr(model, layer_name)
    filters = layer.weight.data.cpu()

    # Normalize filters for visualization
    filters = (filters - filters.min()) / (filters.max() - filters.min())

    num_filters = min(num_filters, filters.shape[0])
    grid_size = int(np.ceil(np.sqrt(num_filters)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle(f'Learned Filters - {layer_name}', fontsize=16, fontweight='bold')

    for idx in range(grid_size * grid_size):
        row = idx // grid_size
        col = idx % grid_size
        ax = axes[row, col] if grid_size > 1 else axes

        if idx < num_filters:
            # Handle single-channel and multi-channel filters
            filter_img = filters[idx].numpy()

            if filter_img.shape[0] == 3:  # RGB filters
                filter_img = np.transpose(filter_img, (1, 2, 0))
            else:  # Take first channel
                filter_img = filter_img[0]

            ax.imshow(filter_img, cmap='viridis')
            ax.set_title(f'Filter {idx}', fontsize=8)
        else:
            ax.axis('off')

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Filter visualization saved to {save_path}")

    plt.show()


def visualize_feature_maps(model, image, layer_name='conv1', num_maps=16, save_path=None):
    """
    Visualize feature maps (activations) for a given image.

    Args:
        model: Trained CNN model
        image (torch.Tensor): Input image tensor (C, H, W) or (1, C, H, W)
        layer_name (str): Name of layer to visualize
        num_maps (int): Number of feature maps to show
        save_path (str): Path to save figure (optional)
    """
    model.eval()

    if image.dim() == 3:
        image = image.unsqueeze(0)

    # Register hook to capture activations
    activations = {}

    def hook_fn(module, input, output):
        activations['output'] = output.detach()

    layer = getattr(model, layer_name)
    hook = layer.register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        _ = model(image)

    hook.remove()

    # Get activations
    feature_maps = activations['output'][0].cpu()
    num_maps = min(num_maps, feature_maps.shape[0])
    grid_size = int(np.ceil(np.sqrt(num_maps)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle(f'Feature Maps - {layer_name}', fontsize=16, fontweight='bold')

    for idx in range(grid_size * grid_size):
        row = idx // grid_size
        col = idx % grid_size
        ax = axes[row, col] if grid_size > 1 else axes

        if idx < num_maps:
            feature_map = feature_maps[idx].numpy()
            ax.imshow(feature_map, cmap='viridis')
            ax.set_title(f'Map {idx}', fontsize=8)
        else:
            ax.axis('off')

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature map visualization saved to {save_path}")

    plt.show()


def plot_training_history(history_path, save_path=None):
    """
    Plot training history curves.

    Args:
        history_path (str): Path to training history JSON file
        save_path (str): Path to save figure (optional)
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot learning rate
    axes[2].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")

    plt.show()


def compute_gradcam(model, image, target_class, target_layer='conv3'):
    """
    Compute Grad-CAM visualization for interpretability.

    Args:
        model: Trained model
        image (torch.Tensor): Input image (1, C, H, W)
        target_class (int): Target class index
        target_layer (str): Layer name to compute Grad-CAM on

    Returns:
        numpy.ndarray: Grad-CAM heatmap
    """
    model.eval()

    # Hooks for gradients and activations
    gradients = {}
    activations = {}

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]

    def forward_hook(module, input, output):
        activations['value'] = output

    # Register hooks
    layer = getattr(model, target_layer)
    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(image)
    model.zero_grad()

    # Backward pass for target class
    class_loss = output[0, target_class]
    class_loss.backward()

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    # Compute Grad-CAM
    grads = gradients['value'].cpu().data.numpy()[0]
    acts = activations['value'].cpu().data.numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)  # ReLU
    cam = cam / (cam.max() + 1e-8)  # Normalize

    return cam


def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        save_path (str): Path to save figure (optional)
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.show()

    # Print classification report
    print("\nClassification Report:")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=classes))


if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")
    print("Available functions:")
    print("  - visualize_filters()")
    print("  - visualize_feature_maps()")
    print("  - plot_training_history()")
    print("  - compute_gradcam()")
    print("  - plot_confusion_matrix()")
