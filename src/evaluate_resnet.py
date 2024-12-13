# src/evaluate_resnet.py

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from data_utils import PickledCIFAR10Dataset, load_split_datasets  # Corrected import
from utils import setup_logging  # Corrected import
import os
import logging
from tqdm import tqdm  # For progress bars

def evaluate_resnet():
    # Setup Logging
    log_file = '/ssd1/divake/doubly_conformal/logs/resnet_evaluation.log'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)  # Create log directory if it doesn't exist
    setup_logging(log_file)
    logger = logging.getLogger()
    logger.info("Starting ResNet-18 evaluation...")
    
    # Hyperparameters
    model_save_path = '/ssd1/divake/doubly_conformal/models/resnet18_cifar10.pth'
    plot_save_path = '/ssd1/divake/doubly_conformal/plots/resnet_test_accuracy.txt'
    
    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Define Transformations for Testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),    # CIFAR-10 Mean
                             (0.2023, 0.1994, 0.2010))    # CIFAR-10 Std
    ])
    
    # Load Datasets
    split_dir = '/ssd1/divake/doubly_conformal/data/splits/'
    dataset_path = '/ssd1/divake/doubly_conformal/data/cifar-10-batches-py/cifar-10-batches-py/'
    
    # Load all three loaders, but we'll use only test_loader
    train_loader, calibration_loader, test_loader = load_split_datasets(
        split_dir=split_dir,
        transform_train=None,           # Not used here
        transform_calibration=None,     # Not used here
        transform_test=transform_test,
        dataset_path=dataset_path
    )
    logger.info("Loaded Test DataLoader.")
    
    # Verify DataLoader
    logger.info(f"Number of test batches: {len(test_loader)}")
    
    # Initialize ResNet-18 Model
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 has 10 classes
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"The model file {model_save_path} does not exist. Please train the model first.")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    model.to(device)
    logger.info("Loaded ResNet-18 model.")
    
    # Evaluation
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
    
    accuracy = 100 * correct / total
    logger.info(f"Test Accuracy: {accuracy:.2f}%")
    
    # Save accuracy to a text file
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    with open(plot_save_path, 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n")
    logger.info(f"Test accuracy written to {plot_save_path}")
    
    logger.info("ResNet-18 evaluation completed.")

if __name__ == "__main__":
    evaluate_resnet()
