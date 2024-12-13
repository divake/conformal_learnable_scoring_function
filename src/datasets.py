# src/datasets.py

import pickle
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os

class CIFAR10Dataset(Dataset):
    def __init__(self, batch_path: str, transform=None):
        """
        Initialize CIFAR-10 dataset from a batch file.
        
        Args:
            batch_path (str): Path to the CIFAR-10 batch file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        if not os.path.exists(batch_path):
            raise FileNotFoundError(f"The file {batch_path} does not exist.")
        
        with open(batch_path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        self.data = batch[b'data']  # Shape: (10000, 3072)
        self.labels = batch[b'labels']  # List of length 10000
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

class PickledCIFAR10Dataset(Dataset):
    def __init__(self, pickle_path: str, transform=None):
        """
        Initialize CIFAR-10 dataset from a pickle file.
        
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
