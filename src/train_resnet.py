# src/train_resnet.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from datasets import PickledCIFAR10Dataset
from utils import setup_logging, plot_training_loss
from data_utils import load_split_datasets  # Removed set_seed from import
import os
import logging
from tqdm import tqdm  # For progress bars
import random
import numpy as np

def set_seed(seed):
    """
    Sets the seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_resnet():
    # Setup Logging
    log_file = '/ssd1/divake/doubly_conformal/logs/resnet_training.log'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)  # Create log directory if it doesn't exist
    setup_logging(log_file)
    logger = logging.getLogger()
    logger.info("Starting ResNet-18 training...")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Hyperparameters
    num_epochs = 50
    batch_size = 256
    learning_rate = 0.001
    model_save_path = '/ssd1/divake/doubly_conformal/models/resnet18_cifar10.pth'
    plot_save_path = '/ssd1/divake/doubly_conformal/plots/resnet_training_loss.png'
    
    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Define Transformations for Training and Testing
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),              # Data Augmentation
        transforms.RandomCrop(32, padding=4),           # Data Augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),    # CIFAR-10 Mean
                             (0.2023, 0.1994, 0.2010))    # CIFAR-10 Std
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),    # CIFAR-10 Mean
                             (0.2023, 0.1994, 0.2010))    # CIFAR-10 Std
    ])
    
    # Load Datasets
    # Assuming load_split_datasets returns (train_loader, calibration_loader, test_loader)
    # Since calibration is not used here, we can ignore it
    split_dir = '/ssd1/divake/doubly_conformal/data/splits/'
    dataset_path = '/ssd1/divake/doubly_conformal/data/cifar-10-batches-py/cifar-10-batches-py/'
    
    # Load all three loaders, but we'll use only train_loader and test_loader
    train_loader, calibration_loader, test_loader = load_split_datasets(
        split_dir=split_dir,
        transform_train=transform_train,
        transform_calibration=None,   # Not used in this step
        transform_test=transform_test,
        dataset_path=dataset_path
    )
    logger.info("Loaded Training and Test DataLoaders.")
    
    # Verify DataLoaders
    logger.info(f"Number of training batches: {len(train_loader)}")
    logger.info(f"Number of test batches: {len(test_loader)}")
    
    # Initialize ResNet-18 Model
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 has 10 classes
    model = model.to(device)
    logger.info("Initialized ResNet-18 model for CIFAR-10.")
    
    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    train_losses = []
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward Pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward Pass and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)  # Accumulate loss
            
            # Update Progress Bar
            progress_bar.set_postfix({'Loss': loss.item()})
        
        # Calculate Average Loss for the Epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    # Save the Trained Model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"ResNet-18 model saved to {model_save_path}")
    
    # Plot Training Loss
    plot_training_loss(train_losses, [], plot_save_path)  # Assuming plot_training_loss handles empty validation losses
    logger.info(f"Training loss plot saved to {plot_save_path}")
    
    # Evaluate the Model on Test Dataset
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating on Test Set', leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * correct / total
    logger.info(f"Test Accuracy: {test_accuracy:.2f}%")
    
    logger.info("ResNet-18 training and evaluation completed.")

if __name__ == "__main__":
    train_resnet()
