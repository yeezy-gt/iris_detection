import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import get_data_loaders

def train_model(model, num_epochs, train_loader, val_loader, loss_fn, optimizer, device, checkpoint_dir='checkpoints'):
    """Train the iris detection model.
    
    Args:
        model: The model to train
        num_epochs: Number of epochs to train for
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        Trained model and training history
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            # Move data to device
            images = images.to(device)
            targets = targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = loss_fn(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # Calculate average training loss for this epoch
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                loss = loss_fn(outputs, targets)
                
                val_loss += loss.item()
        
        # Calculate average validation loss
        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
        
        # Save checkpoint if this is the best model so far
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
            }, os.path.join(checkpoint_dir, 'final_weights.pth'))
            print(f'Checkpoint saved at epoch {epoch+1}')
    
    # Save the final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
    }, os.path.join(checkpoint_dir, 'final_weights.pth'))
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, 'loss_curve.png'))
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses}


def main():
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Hyperparameters
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    
    # Get data loaders
    data_dir = 'data'
    train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size=batch_size)
    
    # Import the model from model.py
    from model import ResNet
    
    # Initialize model
    model = ResNet()
    model = model.to(device)
    
    # Define loss function and optimizer
    loss_fn = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    trained_model, history = train_model(
        model=model,
        num_epochs=num_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device
    )
    
    print('Training completed!')


if __name__ == '__main__':
    main()