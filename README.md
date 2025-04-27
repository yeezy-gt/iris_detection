# Iris Detection System

## Project Overview
This project implements an iris detection system using deep learning, specifically focusing on keypoint detection. The system accurately identifies and localizes the irises in a given facial image using a ResNet-based architecture.

## Problem Description
The goal is to develop a system that can detect the coordinates of both irises in a facial image. This is a keypoint detection problem where we need to predict the exact (x, y) coordinates of the left and right irises.

## Input and Output
- **Input**: A .jpg image of a person's face
- **Output**: A 4-dimensional vector: (xleft, yleft, xright, yright) where:
  - xleft, yleft: Coordinates of the left iris
  - xright, yright: Coordinates of the right iris

## Dataset
The dataset consists of facial images with corresponding JSON annotation files. The annotations contain keypoints for both the left and right irises, labeled as 'leye' and 'reye' respectively. My implementation includes robust validation to handle missing or corrupted files, ensuring smooth training even with imperfect datasets.

Data augmentation was performed using the Albumentations library to improve model robustness and generalization.

## Model Architecture
The model utilizes a custom implementation of ResNet34 (Residual Network) architecture, as introduced by He et al. (2016). Instead of using a pre-trained model, we've built ResNet34 from scratch to tailor it specifically for iris detection. ResNet is chosen for this problem because:

1. Keypoint detection requires capturing fine-grained spatial details, which ResNet's deep architecture allows by learning rich hierarchical features.
2. ResNet's skip connections help propagate gradients effectively, addressing the vanishing gradient problem in deep networks.
3. ResNet's ability to generalize across different datasets ensures robustness to variations in lighting, pose, and occlusions—factors commonly encountered in facial images.
4. Our custom implementation allows for precise control over the network architecture, enabling optimizations specific to iris detection.

## Project Structure
- `model.py`: Defines the ResNet-based iris detection model
- `dataset.py`: Handles data loading and preprocessing
- `train.py`: Contains the training loop and related functions
- `predict.py`: Implements inference functionality
- `interface.py`: Provides a user-friendly interface for using the model
- `config.py`: Centralizes configuration parameters

## Usage Instructions

### Installation
Ensure you have the required dependencies:
```
pip install torch torchvision numpy matplotlib pillow
```

Note: This implementation supports MPS (Metal Performance Shaders) for accelerated training on Mac GPUs. The system automatically detects and utilizes available GPU resources.

### Training
To train the model:
```
python train.py
```

### Inference
To detect irises in a single image:
```
python interface.py --image path/to/image.jpg
```

Additional options:
- `--checkpoint path/to/checkpoint.pth`: Specify a custom model checkpoint
- `--no-visualize`: Disable visualization of results

## Results
The model outputs a 4-dimensional vector representing the coordinates of both irises. Additionally, the interface provides a visualization of the detected keypoints overlaid on the input image.

## Future Improvements
- Explore even deeper backbone architectures (ResNet50, ResNet101)
- Add support for real-time iris detection using webcam input
- Implement ensemble methods to improve accuracy
- Optimize inference speed for mobile deployment
