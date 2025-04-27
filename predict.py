import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import argparse
from torchvision import transforms
from model import ResNet
from config import INFER_CONFIG


def load_model(checkpoint_path):
    """
    Load a trained iris detection model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        
    Returns:
        Loaded model
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet()
    # Load checkpoint
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    model = model.to(device)
    
    return model


def preprocess_image(image_path):
    """
    Preprocess an image for inference.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        Preprocessed image tensor
    """
    # Define the same transformations used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def detect_iris(image_path, model):
    """
    Detect iris coordinates in a facial image.
    
    Args:
        image_path: Path to the input image
        model: Trained iris detection model
        
    Returns:
        A 4-dimensional vector: (xleft, yleft, xright, yright)
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Preprocess the image
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Perform inference
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Convert predictions to numpy array
    iris_coordinates = predictions[0].cpu().numpy()
    
    return iris_coordinates


def batch_detect_iris(image_paths, model):
    """
    Detect iris coordinates in multiple facial images.
    
    Args:
        image_paths: List of paths to input images
        model: Trained iris detection model
        
    Returns:
        List of 4-dimensional vectors: (xleft, yleft, xright, yright) for each image
    """
    results = []
    
    for image_path in image_paths:
        try:
            coordinates = detect_iris(image_path, model)
            results.append({
                'image_path': image_path,
                'coordinates': coordinates,
                'success': True
            })
        except Exception as e:
            results.append({
                'image_path': image_path,
                'coordinates': None,
                'success': False,
                'error': str(e)
            })
    
    return results

def visualize_iris_detection(image_path, coordinates):
    """
    Visualize the detected iris keypoints on the input image.
    
    Args:
        image_path: Path to the input image
        coordinates: 4D vector of iris coordinates (xleft, yleft, xright, yright)
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        ])
    
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.permute(1, 2, 0)
    # image = Image.open(image_path).convert('RGB')

    plt.figure(figsize=(10, 8))
    plt.imshow(np.array(image))

    # Plot the left iris
    plt.plot(coordinates[0], coordinates[1], 'ro', markersize=8, label='Left Iris')
    
    # Plot the right iris
    plt.plot(coordinates[2], coordinates[3], 'bo', markersize=8, label='Right Iris')
    
    plt.title('Iris Detection Results')
    plt.legend()
    
    # Save the visualization
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path).replace('.jpg', '_result.jpg'))
    plt.savefig(output_path)
    plt.close()
    
    print(f'Visualization saved to {output_path}')
    return output_path


def process_single_image(image_path, model, visualize=True):
    """
    Process a single image and detect iris keypoints.
    
    Args:
        image_path: Path to the input image
        model: Trained iris detection model
        visualize: Whether to visualize the results
        
    Returns:
        Detected iris coordinates and visualization path (if visualize=True)
    """
    # Detect iris coordinates
    coordinates = detect_iris(image_path, model)
    
    print(f'\nDetected iris coordinates for {os.path.basename(image_path)}:')
    print(f'Left eye (x, y): ({coordinates[0]:.2f}, {coordinates[1]:.2f})')
    print(f'Right eye (x, y): ({coordinates[2]:.2f}, {coordinates[3]:.2f})')
    
    # Visualize results if requested
    vis_path = None
    if visualize:
        vis_path = visualize_iris_detection(image_path, coordinates)
    
    return coordinates, vis_path


def main():
    parser = argparse.ArgumentParser(description='Iris Detection Interface')
    parser.add_argument('--image', type=str, help='Path to input image or directory')
    parser.add_argument('--checkpoint', type=str, default=INFER_CONFIG['checkpoint_path'],
                        help='Path to model checkpoint')
    parser.add_argument('--no-visualize', action='store_true', help='Disable visualization')
    args = parser.parse_args()

    # Check if image path is provided
    if not args.image:
        print('Please provide an input image or directory path using --image')
        return

    # Load the model
    try:
        model = load_model(args.checkpoint)
        print(f'Model loaded successfully from {args.checkpoint}')
    except Exception as e:
        print(f'Error loading model: {str(e)}')
        return

    # Check if input is a directory or a file
    if os.path.isdir(args.image):
        # Process all images in the directory
        exts = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [os.path.join(args.image, f) for f in os.listdir(args.image)
                       if os.path.splitext(f)[1].lower() in exts]
        if not image_files:
            print(f'No image files found in directory {args.image}')
            return
        print(f'Processing {len(image_files)} images in directory {args.image}')
        for img_path in image_files:
            try:
                coordinates, vis_path = process_single_image(
                    img_path, model, visualize=not args.no_visualize
                )
                print(f'Image: {os.path.basename(img_path)}')
                print(f'Output vector (xleft, yleft, xright, yright): ({coordinates[0]:.2f}, {coordinates[1]:.2f}, {coordinates[2]:.2f}, {coordinates[3]:.2f})')
                if vis_path:
                    print(f'Visualization saved to {vis_path}')
            except Exception as e:
                print(f'Error processing {img_path}: {str(e)}')
    elif os.path.isfile(args.image):
        # Process a single image
        try:
            coordinates, vis_path = process_single_image(
                args.image, model, visualize=not args.no_visualize
            )
            print('\nOutput vector (xleft, yleft, xright, yright):')
            print(f'({coordinates[0]:.2f}, {coordinates[1]:.2f}, {coordinates[2]:.2f}, {coordinates[3]:.2f})')
        except Exception as e:
            print(f'Error processing image: {str(e)}')
    else:
        print(f'Error: {args.image} is not a valid file or directory')


if __name__ == '__main__':
    main()