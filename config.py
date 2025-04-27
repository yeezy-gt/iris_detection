# Configuration file for the iris detection system

# Dataset configuration
DATASET_CONFIG = {
    'data_dir': 'data',
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
    'batch_size': 32,
    'num_workers': 4
}

# Model configuration
MODEL_CONFIG = {
    'dropout_rate': 0.2
}

# Training configuration
TRAIN_CONFIG = {
    'num_epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'checkpoint_dir': 'checkpoints',
    'save_freq': 5  # Save checkpoint every N epochs
}

# Inference configuration
INFER_CONFIG = {
    'checkpoint_path': 'checkpoints/final_weights.pth',
    'image_size': (224, 224)  # Input image size for the model
}