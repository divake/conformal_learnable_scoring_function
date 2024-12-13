# src/evaluate_conformal.py

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from datasets import CIFAR10Dataset, PickledCIFAR10Dataset
from utils import setup_logging, plot_conformal_metrics
from data_utils import load_split_datasets, set_seed
from model import ScoringFunction  # Ensure this import points to your model.py
import os
import numpy as np
from collections import defaultdict
import logging  # Imported logging module

def extract_softmax_scores(model, dataloader, device):
    """
    Extract softmax scores from the model for all samples in the dataloader.

    Args:
        model (nn.Module): Trained ResNet-18 model.
        dataloader (DataLoader): DataLoader for calibration or test set.
        device (torch.device): Device to perform computations on.

    Returns:
        Tuple[torch.Tensor, list]: Softmax scores tensor and corresponding labels.
    """
    model.eval()
    softmax = nn.Softmax(dim=1)
    all_softmax = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = softmax(outputs)
            all_softmax.append(probs.cpu())
            # Convert labels to list of integers
            all_labels.extend(labels.tolist())

    all_softmax = torch.cat(all_softmax, dim=0)  # Shape: (N, num_classes)
    return all_softmax, all_labels

def compute_tau(non_conformity_scores: np.ndarray, confidence: float =0.9):
    """
    Compute the tau threshold based on non-conformity scores and desired confidence level.

    Args:
        non_conformity_scores (np.ndarray): Array of non-conformity scores.
        confidence (float): Desired confidence level (e.g., 0.9 for 90%).

    Returns:
        float: Tau threshold.
    """
    tau = np.quantile(non_conformity_scores, confidence)
    return tau

def generate_prediction_sets(softmax_scores: torch.Tensor, scoring_fn: nn.Module, tau: float, device):
    """
    Generate prediction sets based on softmax scores, scoring function, and tau.

    Args:
        softmax_scores (torch.Tensor): Softmax scores of shape (N, num_classes).
        scoring_fn (nn.Module): Trained Scoring Function.
        tau (float): Threshold for non-conformity scores.
        device (torch.device): Device to perform computations on.

    Returns:
        list: List of prediction sets for each sample.
    """
    scoring_fn.eval()
    with torch.no_grad():
        non_conformity = scoring_fn(softmax_scores.to(device)).cpu()

    prediction_sets = []
    for nc_scores in non_conformity:
        prediction = []
        for cls_idx, score in enumerate(nc_scores):
            if score <= tau:
                prediction.append(cls_idx)
        prediction_sets.append(prediction)
    return prediction_sets

def define_targets(softmax_scores, labels, num_classes=10):
    """
    Define target non-conformity scores based on softmax scores.
    Assign lower scores to the true class and higher scores to other classes.

    Args:
        softmax_scores (torch.Tensor): Softmax scores tensor of shape (N, num_classes).
        labels (list): True labels.
        num_classes (int): Number of classes.

    Returns:
        torch.Tensor: Target non-conformity scores of shape (N, num_classes)
    """
    # Initialize targets with high non-conformity scores
    targets = torch.full_like(softmax_scores, 1.0)  # Shape: (N, num_classes)
    for i, label in enumerate(labels):
        # Assign lower non-conformity score to the true class
        targets[i, label] = -torch.log(softmax_scores[i, label] + 1e-10)  # Negative log prob for true class
    return targets

def evaluate_conformal():
    # Setup Logging
    log_file = '/ssd1/divake/doubly_conformal/logs/conformal_evaluation.log'
    setup_logging(log_file)
    logger = logging.getLogger()  # Renamed to 'logger'
    logger.info("Starting Conformal Prediction evaluation...")

    # Hyperparameters and Paths
    model_resnet_path = '/ssd1/divake/doubly_conformal/models/resnet18_cifar10.pth'
    scoring_fn_path = '/ssd1/divake/doubly_conformal/models/scoring_function.pth'
    plot_save_path = '/ssd1/divake/doubly_conformal/plots/conformal_metrics.png'

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set Seed for Reproducibility (Optional)
    set_seed(42)

    # Load frozen ResNet-18 model
    resnet = models.resnet18(pretrained=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, 10)
    if not os.path.exists(model_resnet_path):
        logger.error(f"The ResNet-18 model file {model_resnet_path} does not exist. Please train the ResNet-18 model first.")
        return
    resnet.load_state_dict(torch.load(model_resnet_path, map_location=device))
    resnet.eval()
    resnet.to(device)
    logger.info("Loaded ResNet-18 model.")

    # Load trained Scoring Function
    scoring_fn = ScoringFunction(input_dim=10, hidden_dims=[128, 64, 32], output_dim=10).to(device)
    if not os.path.exists(scoring_fn_path):
        logger.error(f"The Scoring Function model file {scoring_fn_path} does not exist. Please train the Scoring Function first.")
        return
    scoring_fn.load_state_dict(torch.load(scoring_fn_path, map_location=device))
    scoring_fn.eval()
    logger.info("Loaded Scoring Function model.")

    # Define Calibration and Test Transforms
    transform_calibration = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))
    ])

    # Load datasets with the defined transforms
    split_dir = '/ssd1/divake/doubly_conformal/data/splits/'
    dataset_path = '/ssd1/divake/doubly_conformal/data/cifar-10-batches-py/cifar-10-batches-py/'

    _, calibration_loader, test_loader = load_split_datasets(
        split_dir=split_dir,
        transform_train=None,
        transform_calibration=transform_calibration,  # Apply calibration transform
        transform_test=transform_test,                # Apply test transform
        dataset_path=dataset_path
    )
    logger.info("Loaded calibration and test DataLoaders with transformations.")

    # Debugging: Check sample labels
    try:
        sample_calibration_images, sample_calibration_labels = next(iter(calibration_loader))
        sample_test_images, sample_test_labels = next(iter(test_loader))
        logger.info(f"Sample calibration labels: {sample_calibration_labels[:5]}")
        logger.info(f"Sample test labels: {sample_test_labels[:5]}")
    except Exception as e:
        logger.error(f"Error fetching a sample from DataLoader: {e}")

    # Extract softmax scores for calibration set
    logger.info("Extracting softmax scores for calibration set...")
    calibration_softmax, calibration_labels = extract_softmax_scores(resnet, calibration_loader, device)
    logger.info(f"Extracted softmax scores for {calibration_softmax.shape[0]} calibration samples.")
    logger.info(f"Sample calibration labels: {calibration_labels[:5]}")

    # Define targets for calibration set
    logger.info("Defining target non-conformity scores for calibration set...")
    calibration_targets = define_targets(calibration_softmax, calibration_labels, num_classes=10)

    # Compute non-conformity scores for calibration set
    logger.info("Computing non-conformity scores for calibration set...")
    with torch.no_grad():
        calibration_non_conformity = scoring_fn(calibration_softmax.to(device)).cpu().numpy()  # Shape: (N, num_classes)

    # Extract non-conformity scores for true classes
    calibration_scores = calibration_non_conformity[np.arange(calibration_non_conformity.shape[0]), calibration_labels]

    # Compute tau
    tau = compute_tau(calibration_scores, confidence=0.9)
    logger.info(f"Computed tau threshold: {tau:.4f}")

    # Extract softmax scores for test set
    logger.info("Extracting softmax scores for test set...")
    test_softmax, test_labels = extract_softmax_scores(resnet, test_loader, device)
    logger.info(f"Extracted softmax scores for {test_softmax.shape[0]} test samples.")
    logger.info(f"Sample test labels: {test_labels[:5]}")

    # Generate prediction sets for test set
    logger.info("Generating prediction sets for test set...")
    prediction_sets = generate_prediction_sets(test_softmax, scoring_fn, tau, device)

    # Evaluate coverage and set sizes
    logger.info("Evaluating coverage and set sizes...")
    coverage = 0
    set_sizes = []
    per_class_metrics = defaultdict(lambda: {'coverage':0, 'set_size':0, 'count':0})

    for idx, prediction in enumerate(prediction_sets):
        true_label = test_labels[idx]
        set_sizes.append(len(prediction))
        per_class_metrics[true_label]['count'] +=1
        if true_label in prediction:
            coverage +=1
            per_class_metrics[true_label]['coverage'] +=1
        per_class_metrics[true_label]['set_size'] += len(prediction)

    overall_coverage = coverage / len(test_labels)
    average_set_size = np.mean(set_sizes)

    # Per-class metrics
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    metrics = {}
    for cls in range(10):
        if per_class_metrics[cls]['count'] >0:
            metrics[class_names[cls]] = {
                'coverage': per_class_metrics[cls]['coverage'] / per_class_metrics[cls]['count'] *100,
                'set_size': per_class_metrics[cls]['set_size'] / per_class_metrics[cls]['count']
            }
        else:
            metrics[class_names[cls]] = {
                'coverage': 0.0,
                'set_size': 0.0
            }

    logger.info(f"Overall Coverage: {overall_coverage*100:.2f}%")
    logger.info(f"Average Set Size: {average_set_size:.2f}")

    # Log per-class metrics
    for cls in class_names:
        logger.info(f"Class {cls}: Coverage: {metrics[cls]['coverage']:.2f}%, Average Set Size: {metrics[cls]['set_size']:.2f}")

    # Ensure the plots directory exists
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)

    # Plot metrics
    plot_conformal_metrics(metrics, plot_save_path)
    logger.info(f"Conformal prediction metrics plot saved to {plot_save_path}")

    logger.info("Conformal Prediction evaluation completed.")

if __name__ == "__main__":
    evaluate_conformal()
