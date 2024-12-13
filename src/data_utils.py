# src/data_utils.py

import os
import pickle
import numpy as np
from collections import defaultdict
import random
from torch.utils.data import DataLoader
from datasets import CIFAR10Dataset, PickledCIFAR10Dataset
from utils import set_seed

def load_cifar_batch(batch_path: str):
    """
    Load a single batch of CIFAR-10 data.
    
    Args:
        batch_path (str): Path to the batch file.
    
    Returns:
        Tuple[np.ndarray, list]: Data and labels.
    """
    with open(batch_path, 'rb') as file:
        dict = pickle.load(file, encoding='bytes')
        data = dict[b'data']  # Shape: (10000, 3072)
        labels = dict[b'labels']  # List of length 10000
    return data, labels

def load_cifar10(data_dir: str):
    """
    Load all CIFAR-10 data from batch files.
    
    Args:
        data_dir (str): Directory containing CIFAR-10 batch files.
    
    Returns:
        Tuple[np.ndarray, list, np.ndarray, list]: Train data, train labels, test data, test labels.
    """
    train_data = []
    train_labels = []
    # Load all training batches
    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f'data_batch_{i}')
        data, labels = load_cifar_batch(batch_path)
        train_data.append(data)
        train_labels.extend(labels)
    train_data = np.concatenate(train_data)  # Shape: (50000, 3072)
    
    # Load test batch
    test_batch_path = os.path.join(data_dir, 'test_batch')
    test_data, test_labels = load_cifar_batch(test_batch_path)  # Shape: (10000, 3072), labels: list
    
    return train_data, train_labels, test_data, test_labels

def create_calibration_set(data: np.ndarray, labels: list, calibration_ratio: float =0.1, num_classes: int =10, seed: int =42):
    """
    Create a balanced calibration set from the training data.
    
    Args:
        data (np.ndarray): Training data.
        labels (list): Training labels.
        calibration_ratio (float): Fraction of data to use for calibration.
        num_classes (int): Number of classes.
        seed (int): Random seed.
    
    Returns:
        Tuple[np.ndarray, list, np.ndarray, list]: Calibration data, calibration labels, remaining data, remaining labels.
    """
    set_seed(seed)
    class_data = defaultdict(list)
    for idx, label in enumerate(labels):
        class_data[label].append(idx)
    
    calibration_indices = []
    for cls in range(num_classes):
        cls_indices = class_data[cls]
        num_cal = max(1, int(len(cls_indices) * calibration_ratio))  # Ensure at least one sample per class
        selected = random.sample(cls_indices, num_cal)
        calibration_indices.extend(selected)
    
    # Remaining indices
    all_indices = set(range(len(labels)))
    remaining_indices = list(all_indices - set(calibration_indices))
    
    # Extract data
    calibration_data = data[calibration_indices]
    calibration_labels = [labels[idx] for idx in calibration_indices]
    
    remaining_data = data[remaining_indices]
    remaining_labels = [labels[idx] for idx in remaining_indices]
    
    return calibration_data, calibration_labels, remaining_data, remaining_labels

def save_split_datasets(calibration_data: np.ndarray, calibration_labels: list, remaining_train_data: np.ndarray, remaining_train_labels: list, save_dir: str):
    """
    Save calibration and remaining training datasets as pickle files.
    
    Args:
        calibration_data (np.ndarray): Calibration data.
        calibration_labels (list): Calibration labels.
        remaining_train_data (np.ndarray): Remaining training data.
        remaining_train_labels (list): Remaining training labels.
        save_dir (str): Directory to save the splits.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'calibration_data.pkl'), 'wb') as f:
        pickle.dump({'data': calibration_data, 'labels': calibration_labels}, f)
    
    with open(os.path.join(save_dir, 'remaining_train_data.pkl'), 'wb') as f:
        pickle.dump({'data': remaining_train_data, 'labels': remaining_train_labels}, f)
    
    print(f"Datasets saved to {save_dir}")

def load_split_datasets(split_dir: str, transform_train, transform_calibration, transform_test, dataset_path: str):
    """
    Load split datasets and create DataLoaders.
    
    Args:
        split_dir (str): Directory where split pickle files are stored.
        transform_train: Transformations for training data.
        transform_calibration: Transformations for calibration data.
        transform_test: Transformations for test data.
        dataset_path (str): Path to CIFAR-10 batch files.
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: train_loader, calibration_loader, test_loader
    """
    calibration_pickle_path = os.path.join(split_dir, 'calibration_data.pkl')
    remaining_train_pickle_path = os.path.join(split_dir, 'remaining_train_data.pkl')
    
    calibration_dataset = PickledCIFAR10Dataset(pickle_path=calibration_pickle_path, transform=transform_calibration)
    remaining_train_dataset = PickledCIFAR10Dataset(pickle_path=remaining_train_pickle_path, transform=transform_train)
    
    # Test dataset
    test_batch_path = os.path.join(dataset_path, 'test_batch')
    test_dataset = CIFAR10Dataset(batch_path=test_batch_path, transform=transform_test)
    
    # Create DataLoaders
    train_loader = DataLoader(dataset=remaining_train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    calibration_loader = DataLoader(dataset=calibration_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, calibration_loader, test_loader
