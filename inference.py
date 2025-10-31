"""
Inference script for trained CNN model

Features:
- Load trained model
- Single image prediction
- Batch prediction
- Top-k predictions with confidence scores
"""

import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from model import CustomCNN


# CIFAR-10 class names
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# CIFAR-10 normalization parameters
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def load_model(model_path, device='cuda'):
    """
    Load trained model from checkpoint.

    Args:
        model_path (str): Path to model checkpoint
        device (str): Device to load model on

    Returns:
        model: Loaded PyTorch model in eval mode
    """
    # Initialize model
    model = CustomCNN(num_classes=10)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Validation accuracy: {checkpoint.get('val_acc', 'unknown'):.2f}%")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def preprocess_image(image_path):
    """
    Preprocess image for model input.

    Args:
        image_path (str): Path to input image

    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    return image_tensor


def predict(model, image_tensor, device='cuda', top_k=5):
    """
    Make prediction on image.

    Args:
        model: Trained model
        image_tensor (torch.Tensor): Preprocessed image tensor
        device (str): Device to run inference on
        top_k (int): Number of top predictions to return

    Returns:
        list: List of (class_name, probability) tuples
    """
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)

        # Get top-k predictions
        top_probs, top_indices = probabilities.topk(top_k, dim=1)

        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            results.append((CLASSES[idx], prob.item() * 100))

    return results


def main():
    parser = argparse.ArgumentParser(description='CNN Inference on CIFAR-10')

    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image-path', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use: cuda or cpu (default: cuda)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top predictions to show (default: 5)')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load model
    print("Loading model...")
    model = load_model(args.model_path, device)
    print(f"Model loaded successfully!\n")

    # Preprocess image
    print(f"Loading image: {args.image_path}")
    image_tensor = preprocess_image(args.image_path)
    print(f"Image shape: {image_tensor.shape}\n")

    # Make prediction
    print("Making prediction...")
    predictions = predict(model, image_tensor, device, args.top_k)

    # Display results
    print("\n" + "=" * 50)
    print("Predictions:")
    print("=" * 50)
    for i, (class_name, prob) in enumerate(predictions, 1):
        bar = "█" * int(prob / 2)
        print(f"{i}. {class_name:12s} {prob:5.2f}% {bar}")
    print("=" * 50)

    # Print top prediction
    top_class, top_prob = predictions[0]
    print(f"\nTop Prediction: {top_class} ({top_prob:.2f}% confidence)")


if __name__ == "__main__":
    main()
