# resnet_cifar10.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import pickle
from PIL import Image
import os
from collections import defaultdict
import random

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 50
batch_size = 256
learning_rate = 0.001
model_save_path = "/ssd1/divake/doubly_conformal/models/resnet18_cifar10.pth"  # Updated path to save the model

# Transformations for data augmentation and normalization
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_calibration = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Path to your dataset
dataset_path = "/ssd1/divake/doubly_conformal/data/cifar-10-batches-py/cifar-10-batches-py/"

class CIFAR10Dataset(Dataset):
    def __init__(self, batch_path, transform=None):
        """
        Initializes the CIFAR-10 dataset from a batch file.
        
        Args:
            batch_path (str): Path to the CIFAR-10 batch file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        if not os.path.exists(batch_path):
            raise FileNotFoundError(f"The file {batch_path} does not exist.")
        
        with open(batch_path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        self.data = batch[b"data"]  # Image data
        self.labels = batch[b"labels"]  # Corresponding labels
        self.transform = transform

        # Reshape data into (N, 3, 32, 32) format
        self.data = self.data.reshape(-1, 3, 32, 32).astype("float32") / 255.0  # Normalize to [0, 1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        # Convert numpy array to PIL image
        image = Image.fromarray((image.transpose(1, 2, 0) * 255).astype("uint8"))  # Convert back to HWC and scale to [0, 255]
        
        # Apply transformation if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label

class PickledCIFAR10Dataset(Dataset):
    def __init__(self, pickle_path, transform=None):
        """
        Initializes the dataset by loading data and labels from a pickle file.
        
        Args:
            pickle_path (str): Path to the pickle file containing data and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"The file {pickle_path} does not exist.")
        
        with open(pickle_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        self.data = data_dict['data']  # Shape: (N, 3072)
        self.labels = data_dict['labels']  # List of length N
        self.transform = transform
        
        # Reshape data into (N, 3, 32, 32) format
        self.data = self.data.reshape(-1, 3, 32, 32).astype('float32') / 255.0  # Normalize to [0, 1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        # Convert numpy array to PIL image
        image = Image.fromarray((image.transpose(1, 2, 0) * 255).astype('uint8'))  # Convert back to HWC and scale to [0, 255]
        
        # Apply transformation if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Load the datasets
def load_datasets(split_dir='/ssd1/divake/doubly_conformal/data/splits/', batch_path_test=os.path.join('/ssd1/divake/doubly_conformal/data/cifar-10-batches-py/cifar-10-batches-py/', 'test_batch')):
    """
    Load the training, calibration, and test datasets.
    
    Args:
        split_dir (str): Directory where the split pickle files are stored.
        batch_path_test (str): Path to the test batch file.
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: train_loader, calibration_loader, test_loader
    """
    # Paths to split datasets
    calibration_pickle_path = os.path.join(split_dir, 'calibration_data.pkl')
    remaining_train_pickle_path = os.path.join(split_dir, 'remaining_train_data.pkl')
    
    # Create Dataset instances
    calibration_dataset = PickledCIFAR10Dataset(pickle_path=calibration_pickle_path, transform=transform_calibration)
    remaining_train_dataset = PickledCIFAR10Dataset(pickle_path=remaining_train_pickle_path, transform=transform_train)
    
    # Test Dataset using existing CIFAR10Dataset class
    test_dataset = CIFAR10Dataset(batch_path=batch_path_test, transform=transform_test)
    
    # Create DataLoaders
    train_loader = DataLoader(dataset=remaining_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    calibration_loader = DataLoader(dataset=calibration_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, calibration_loader, test_loader

def load_frozen_resnet18(model_save_path="/ssd1/divake/doubly_conformal/models/resnet18_cifar10.pth"):
    """
    Loads the pretrained ResNet-18 model with frozen parameters.
    
    Args:
        model_save_path (str): Path to the saved ResNet-18 model weights.
    
    Returns:
        nn.Module: Frozen ResNet-18 model.
    """
    model = models.resnet18(pretrained=False)  # Set to True if using pretrained weights like ImageNet
    model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 has 10 classes
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"The model file {model_save_path} does not exist. Please train the model first.")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()  # Set to evaluation mode

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    model.to(device)
    return model

# Training function
def train_model():
    train_loader, _, _ = load_datasets()
    model = models.resnet18(pretrained=False)  # Set to True if using pretrained weights like ImageNet
    model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 has 10 classes
    model = model.to(device)

    # Do NOT freeze the model parameters during training
    for param in model.parameters():
        param.requires_grad = True

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    # Save the trained model weights
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# Testing function
def test_model():
    _, _, test_loader = load_datasets()
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"The model file {model_save_path} does not exist. Please train the model first.")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# Main execution
if __name__ == "__main__":
    # Step 1: Train the ResNet-18 model
    print("Starting training...")
    train_model()
    
    # Step 2: Evaluate the trained model on the test set
    print("Evaluating the model on the test set...")
    test_model()
    
    # Step 3: Load and freeze the trained model for further steps
    print("Loading and freezing the trained ResNet-18 model...")
    frozen_model = load_frozen_resnet18(model_save_path=model_save_path)
    print("Model loaded and frozen successfully.")
    
    # The frozen_model can now be used for the next steps, such as feeding softmax scores to the Scoring Function.
