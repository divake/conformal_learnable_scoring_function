# src/scoring_function.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from src.model import ScoringFunction
from src.data_utils import load_split_datasets, extract_softmax_scores
from src.utils import setup_logging
import os
import logging
from torchvision import models

def plot_softmax_vs_nonconformity(scoring_fn, resnet, calibration_loader, device, save_path: str):
    """
    Plot the relationship between softmax scores and non-conformity scores for each class.
    
    Args:
        scoring_fn (nn.Module): Trained Scoring Function.
        resnet (nn.Module): Trained ResNet-18 model.
        calibration_loader (DataLoader): DataLoader for calibration set.
        device (torch.device): Device to perform computations on.
        save_path (str): Path to save the plot.
    """
    scoring_fn.eval()
    resnet.eval()
    softmax = nn.Softmax(dim=1)
    
    # Extract softmax scores
    all_softmax = []
    all_labels = []
    with torch.no_grad():
        for images, labels in calibration_loader:
            images = images.to(device)
            outputs = resnet(images)
            probs = softmax(outputs)
            all_softmax.append(probs.cpu())
            all_labels.extend(labels)
    
    all_softmax = torch.cat(all_softmax, dim=0)  # Shape: (N, num_classes)
    
    # Get non-conformity scores
    with torch.no_grad():
        non_conformity = scoring_fn(all_softmax.to(device)).cpu().numpy()
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    plt.figure(figsize=(12, 8))
    for cls in range(10):
        cls_softmax = all_softmax[:, cls].numpy()
        cls_non_conf = non_conformity[:, cls]
        plt.scatter(cls_softmax, cls_non_conf, label=class_names[cls], alpha=0.5, s=10)
    
    plt.title('Softmax Score vs. Non-Conformity Score for Each Class')
    plt.xlabel('Softmax Score')
    plt.ylabel('Non-Conformity Score')
    plt.legend(title='Classes', fontsize='small', markerscale=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Setup
    log_file = '/ssd1/divake/doubly_conformal/logs/plot_scoring_function.log'
    setup_logging(log_file)
    logging = logging.getLogger()
    logging.info("Starting Scoring Function plotting...")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load ResNet-18
    resnet_path = '/ssd1/divake/doubly_conformal/models/resnet18_cifar10.pth'
    resnet = models.resnet18(pretrained=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, 10)
    if not os.path.exists(resnet_path):
        raise FileNotFoundError(f"The ResNet-18 model file {resnet_path} does not exist. Please train the ResNet-18 model first.")
    resnet.load_state_dict(torch.load(resnet_path, map_location=device))
    resnet.eval()
    resnet.to(device)
    logging.info("Loaded ResNet-18 model.")
    
    # Load Scoring Function
    scoring_fn_path = '/ssd1/divake/doubly_conformal/models/scoring_function.pth'
    scoring_fn = ScoringFunction(input_dim=10, hidden_dims=[64, 32], output_dim=10).to(device)
    if not os.path.exists(scoring_fn_path):
        raise FileNotFoundError(f"The Scoring Function model file {scoring_fn_path} does not exist. Please train the Scoring Function first.")
    scoring_fn.load_state_dict(torch.load(scoring_fn_path, map_location=device))
    scoring_fn.eval()
    logging.info("Loaded Scoring Function model.")
    
    # Load calibration DataLoader
    split_dir = '/ssd1/divake/doubly_conformal/data/splits/'
    dataset_path = '/ssd1/divake/doubly_conformal/data/cifar-10-batches-py/cifar-10-batches-py/'
    _, calibration_loader, _ = load_split_datasets(
        split_dir=split_dir,
        transform_train=None,
        transform_calibration=None,
        transform_test=None,
        dataset_path=dataset_path
    )
    logging.info("Loaded calibration DataLoader.")
    
    # Plot
    plot_path = '/ssd1/divake/doubly_conformal/plots/softmax_vs_nonconformity.png'
    plot_softmax_vs_nonconformity(scoring_fn, resnet, calibration_loader, device, plot_path)
    logging.info(f"Softmax vs. Non-Conformity plot saved to {plot_path}")
    
    logging.info("Scoring Function plotting completed.")

if __name__ == "__main__":
    main()
