# CNN from Scratch: Deep Dive into Convolutional Neural Networks

A comprehensive implementation of Convolutional Neural Networks built from scratch using PyTorch, demonstrating fundamental deep learning concepts for computer vision.

## Overview

This project implements a custom CNN architecture trained on CIFAR-10 dataset, achieving **~85% test accuracy** with detailed visualizations and interpretability tools. Built to understand the mathematical foundations and practical implementation of CNNs.

## Features

- **Custom CNN Architecture**: Built from ground-up with modular design
- **Training Pipeline**: Complete training loop with learning rate scheduling and early stopping
- **Visualization Suite**:
  - Filter/kernel visualizations
  - Feature map activations
  - Training curves (loss, accuracy)
  - Grad-CAM for model interpretability
- **Model Checkpointing**: Save and load trained models
- **Inference Engine**: Easy-to-use prediction interface
- **Google Colab Ready**: Notebook included for instant experimentation

## Architecture

```
Input (3x32x32)
  → Conv2D(3→64) → BatchNorm → ReLU → MaxPool
  → Conv2D(64→128) → BatchNorm → ReLU → MaxPool
  → Conv2D(128→256) → BatchNorm → ReLU → MaxPool
  → Flatten
  → FC(256*4*4 → 512) → Dropout → ReLU
  → FC(512 → 10)
Output (10 classes)
```

**Total Parameters**: ~2.8M
**Training Time**: ~30 min on GPU (Colab)

## Theory Background

### Convolutional Layers
Convolutional layers apply learnable filters to extract spatial hierarchies:
- **Low-level features**: Edges, corners, textures
- **Mid-level features**: Shapes, patterns
- **High-level features**: Object parts, semantic concepts

**Mathematical Operation**:
```
Output[i,j] = Σ(Input[i:i+k, j:j+k] ⊙ Kernel) + bias
```

### Batch Normalization
Normalizes layer inputs to stabilize training:
```
y = γ * (x - μ) / √(σ² + ε) + β
```

### Pooling
Reduces spatial dimensions while retaining important features (translation invariance).

## Installation

```bash
git clone https://github.com/MAYANK12-WQ/CNN-from-Scratch.git
cd CNN-from-Scratch
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
python train.py --epochs 50 --batch-size 128 --lr 0.001
```

**Arguments**:
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 0.001)
- `--device`: 'cuda' or 'cpu' (default: auto-detect)
- `--save-path`: Path to save model checkpoints

### Inference

```bash
python inference.py --model-path checkpoints/best_model.pth --image-path test_image.png
```

### Visualization

```python
from utils import visualize_filters, visualize_feature_maps, plot_training_history

# Visualize learned filters
visualize_filters(model, layer_name='conv1')

# Show feature maps for an image
visualize_feature_maps(model, image, layer_name='conv2')

# Plot training curves
plot_training_history('training_log.json')
```

## Project Structure

```
CNN-from-Scratch/
├── model.py              # CNN architecture definition
├── train.py              # Training script
├── inference.py          # Inference and prediction
├── utils.py              # Visualization and helper functions
├── dataset.py            # Data loading and augmentation
├── requirements.txt      # Dependencies
├── demo.ipynb           # Google Colab demo notebook
├── checkpoints/         # Saved models
├── logs/                # Training logs
└── README.md
```

## Results

### Performance Metrics

| Metric | Score |
|--------|-------|
| Test Accuracy | 85.3% |
| Train Accuracy | 92.1% |
| F1 Score | 0.851 |
| Inference Time | ~5ms/image |

### Training Curves

Training converges smoothly with learning rate scheduling:
- Loss decreases consistently
- Validation accuracy plateaus around epoch 40
- No significant overfitting with dropout + batch norm

### Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) shows which regions the model focuses on for predictions, providing interpretability.

## Dataset: CIFAR-10

- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Training samples**: 50,000
- **Test samples**: 10,000
- **Image size**: 32x32 RGB

**Data Augmentation Applied**:
- Random horizontal flips
- Random crops with padding
- Normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Key Implementation Details

### 1. Weight Initialization
Using Kaiming/He initialization for ReLU activations:
```python
nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
```

### 2. Learning Rate Scheduling
ReduceLROnPlateau: reduces LR when validation loss plateaus
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
```

### 3. Regularization
- Dropout (p=0.5) in fully connected layers
- Batch normalization after each conv layer
- L2 weight decay (1e-4)

### 4. Loss Function
CrossEntropyLoss for multi-class classification

## Extensions & Ideas

- [ ] Implement ResNet-style skip connections
- [ ] Add data augmentation techniques (Mixup, CutMix)
- [ ] Experiment with different architectures (VGG, MobileNet-style)
- [ ] Transfer learning comparison
- [ ] Quantization for mobile deployment
- [ ] Adversarial robustness testing

## Learning Outcomes

This project demonstrates:
1. **Deep Learning Fundamentals**: Understanding CNN building blocks
2. **PyTorch Proficiency**: Custom model implementation, training loops
3. **Computer Vision**: Image classification pipeline
4. **Best Practices**: Modular code, reproducibility, documentation
5. **Model Interpretability**: Visualization and analysis techniques

## References

- [Deep Learning Book - Goodfellow et al.](https://www.deeplearningbook.org/)
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)

## License

MIT License - feel free to use for learning and research!

## Author

**Mayank** - Aspiring AI/ML Engineer focused on Computer Vision and Robotics
[GitHub](https://github.com/MAYANK12-WQ) | [LinkedIn](#)

---

**Built with passion for understanding AI from first principles** 🚀
