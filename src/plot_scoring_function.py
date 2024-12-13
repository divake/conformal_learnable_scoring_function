# src/visualize_scoring.py

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from data_utils import PickledCIFAR10Dataset, load_split_datasets, set_seed
from utils import setup_logging
from model import ScoringFunction
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def extract_scores(model, scoring_fn, dataloader, device):
    """
    Extract softmax scores of the true class and non-conformity scores for all samples in the dataloader.

    Args:
        model (nn.Module): Trained ResNet-18 model.
        scoring_fn (nn.Module): Trained Scoring Function model.
        dataloader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to perform computations on.

    Returns:
        list of float: Softmax scores of the true class.
        list of float: Non-conformity scores.
    """
    model.eval()
    scoring_fn.eval()
    softmax = nn.Softmax(dim=1)
    softmax_scores = []
    non_conformity_scores = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Extracting Scores', leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = softmax(outputs)
            true_softmax = probs[torch.arange(labels.size(0)), labels].unsqueeze(1)  # Shape: (batch_size,1)
            soft_scores = true_softmax.cpu().numpy().flatten().tolist()
            non_conf_scores = scoring_fn(true_softmax).squeeze(1).cpu().numpy().flatten().tolist()

            softmax_scores.extend(soft_scores)
            non_conformity_scores.extend(non_conf_scores)

    return softmax_scores, non_conformity_scores

def plot_softmax_vs_nonconformity(softmax_scores, non_conformity_scores, save_path):
    """
    Plot the relationship between softmax scores and non-conformity scores.

    Args:
        softmax_scores (list of float): Softmax scores of the true class.
        non_conformity_scores (list of float): Non-conformity scores.
        save_path (str): Path to save the plot image.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(softmax_scores, non_conformity_scores, alpha=0.5, edgecolors='b', s=20)
    plt.xlabel('Softmax Score of True Class')
    plt.ylabel('Non-Conformity Score')
    plt.title('Variation Between Softmax Score and Non-Conformity Score')
    plt.grid(True)

    # Optionally, add trend line
    z = np.polyfit(softmax_scores, non_conformity_scores, 1)
    p = np.poly1d(z)
    plt.plot(softmax_scores, p(softmax_scores), "r--", label=f'Trend Line: y={z[0]:.2f}x + {z[1]:.2f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Setup Logging
    log_file = '/ssd1/divake/doubly_conformal/logs/visualization.log'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    setup_logging(log_file)
    logger = logging.getLogger()
    logger.info("Starting Visualization Script...")

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set Seed for Reproducibility
    set_seed(42)

    # Load the Trained ResNet-18 Model
    resnet_path = '/ssd1/divake/doubly_conformal/models/resnet18_cifar10.pth'
    if not os.path.exists(resnet_path):
        logger.error(f"The ResNet-18 model file {resnet_path} does not exist. Please ensure it is trained and saved correctly.")
        return

    resnet = models.resnet18(pretrained=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, 10)  # Adjust for CIFAR-10
    resnet.load_state_dict(torch.load(resnet_path, map_location=device))
    resnet.eval()
    resnet.to(device)
    logger.info("Loaded ResNet-18 model.")

    # Load the Trained Scoring Function
    scoring_fn_path = '/ssd1/divake/doubly_conformal/models/scoring_function.pth'
    if not os.path.exists(scoring_fn_path):
        logger.error(f"The Scoring Function model file {scoring_fn_path} does not exist. Please ensure it is trained and saved correctly.")
        return

    scoring_fn = ScoringFunction(input_dim=1, hidden_dims=[64, 32], output_dim=1).to(device)
    scoring_fn.load_state_dict(torch.load(scoring_fn_path, map_location=device))
    scoring_fn.eval()
    logger.info("Loaded Scoring Function model.")

    # Define Transforms (use the same as used during training)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))
    ])

    # Load Test DataLoader
    split_dir = '/ssd1/divake/doubly_conformal/data/splits/'
    dataset_path = '/ssd1/divake/doubly_conformal/data/cifar-10-batches-py/cifar-10-batches-py/'

    _, _, test_loader = load_split_datasets(
        split_dir=split_dir,
        transform_train=None,
        transform_calibration=None,
        transform_test=transform_test,  # Apply test transform
        dataset_path=dataset_path
    )
    logger.info("Loaded Test DataLoader.")

    # Extract Scores
    logger.info("Extracting softmax and non-conformity scores...")
    softmax_scores, non_conformity_scores = extract_scores(resnet, scoring_fn, test_loader, device)
    logger.info(f"Extracted {len(softmax_scores)} softmax scores and {len(non_conformity_scores)} non-conformity scores.")

    # Plot and Save the Visualization
    plot_save_path = '/ssd1/divake/doubly_conformal/plots/softmax_vs_nonconformity.png'
    logger.info("Plotting the relationship between softmax scores and non-conformity scores...")
    plot_softmax_vs_nonconformity(softmax_scores, non_conformity_scores, plot_save_path)
    logger.info(f"Plot saved to {plot_save_path}")

    logger.info("Visualization completed successfully.")

if __name__ == "__main__":
    main()
