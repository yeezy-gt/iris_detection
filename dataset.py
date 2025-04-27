# Unlabled data downloaded from: https://www.kaggle.com/datasets/atulanandjha/lfwpeople/data 
# I labeled a subset of this dataset manually using labelme, and then performed data augmentation
# using albumentations.

import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class IrisDataset(Dataset):
    """
    Dataset class for iris keypoint detection.
    Loads images and their corresponding iris keypoint annotations.
    """
    def __init__(self, data_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the images and JSON annotation files
            transform: Optional transforms to apply to the images
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Get valid JSON files with corresponding images
        self.json_files = []
        for f in os.listdir(data_dir):
            if f.endswith('.json'):
                img_name = json.load(open(os.path.join(data_dir, f)))['imagePath']
                if os.path.exists(os.path.join(data_dir, img_name)):
                    self.json_files.append(f)
        
    def __len__(self):
        return len(self.json_files)
    
    def __getitem__(self, idx):
        # Get the JSON file name
        json_file = self.json_files[idx]
        
        # Load the annotation data
        with open(os.path.join(self.data_dir, json_file), 'r') as f:
            annotation = json.load(f)
        
        # Get the image path
        img_path = os.path.join(self.data_dir, annotation['imagePath'])
        
        # Load the image with error handling
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image file: {img_path}")
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading {img_path}") from e
        
        # Extract iris keypoints
        keypoints = []
        for shape in annotation['shapes']:
            if shape['label'] == 'leye':
                keypoints.extend(shape['points'][0])  # [xleft, yleft]
            elif shape['label'] == 'reye':
                keypoints.extend(shape['points'][0])  # [xright, yright]
        
        # Convert keypoints to tensor
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        
        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)
        
        return image, keypoints

def get_data_loaders(data_dir, batch_size=32, train_split=0.8, val_split=0.1):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for the data loaders
        train_split: Proportion of data to use for training
        val_split: Proportion of data to use for validation
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Create the dataset
    dataset = IrisDataset(data_dir, transform=transform)
    
    # Split the dataset
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader