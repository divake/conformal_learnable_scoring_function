# src/train_scoring_function.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from data_utils import PickledCIFAR10Dataset, load_split_datasets, set_seed
from utils import setup_logging, plot_training_loss
from model import ScoringFunction
import os
import logging
import numpy as np
from tqdm import tqdm  # For progress bars

def extract_softmax_scores_true_class(model, dataloader, device):
    """
    Extract softmax scores of the true class from the model for all samples in the dataloader.

    Args:
        model (nn.Module): Trained ResNet-18 model.
        dataloader (DataLoader): DataLoader for dataset.
        device (torch.device): Device to perform computations on.

    Returns:
        torch.Tensor: Softmax scores of the true class of shape (N,1).
        list: Corresponding true labels.
    """
    model.eval()
    softmax = nn.Softmax(dim=1)
    all_softmax_true = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Extracting Softmax Scores', leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = softmax(outputs)
            true_softmax = probs[torch.arange(labels.size(0)), labels].unsqueeze(1)  # Shape: (batch_size,1)
            all_softmax_true.append(true_softmax.cpu())
            all_labels.extend(labels.tolist())

    all_softmax_true = torch.cat(all_softmax_true, dim=0)  # Shape: (N,1)
    return all_softmax_true, all_labels

def compute_tau(calibration_scores, percentile=90):
    """
    Compute tau as the given percentile of the calibration non-conformity scores.

    Args:
        calibration_scores (torch.Tensor): Non-conformity scores from calibration set.
        percentile (float): Desired percentile for tau.

    Returns:
        float: Tau value.
    """
    tau = torch.quantile(calibration_scores, percentile / 100.0).item()
    return tau

def evaluate_scoring_function(scoring_fn, model, test_loader, device, tau):
    """
    Evaluate coverage and average set size on the test dataset.

    Args:
        scoring_fn (nn.Module): Trained Scoring Function model.
        model (nn.Module): Trained ResNet-18 model.
        test_loader (DataLoader): DataLoader for test set.
        device (torch.device): Device to perform computations on.
        tau (float): Threshold for non-conformity scores.

    Returns:
        float: Coverage percentage.
        float: Average set size.
    """
    scoring_fn.eval()
    model.eval()
    softmax = nn.Softmax(dim=1)
    correct = 0
    total = 0
    total_set_size = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating Scoring Function', leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = softmax(outputs)
            true_softmax = probs[torch.arange(labels.size(0)), labels].unsqueeze(1)  # Shape: (batch_size,1)
            non_conformity = scoring_fn(true_softmax).squeeze(1)  # Shape: (batch_size,)

            # Prediction set includes the true class if non_conformity <= tau
            coverage = (non_conformity <= tau).float().mean().item() * 100  # Percentage
            correct += coverage * labels.size(0) / 100  # Accumulate correct predictions
            total += labels.size(0)
            # Set size is always 1 since prediction set includes/excludes the true class
            total_set_size += 1 * labels.size(0)

    overall_coverage = (correct / total) * 100
    average_set_size = total_set_size / total  # Should be close to 1
    return overall_coverage, average_set_size

def train_scoring_function():
    # Setup Logging
    log_file = '/ssd1/divake/doubly_conformal/logs/scoring_function_training.log'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    setup_logging(log_file)
    logger = logging.getLogger()
    logger.info("Starting Scoring Function training...")

    # Hyperparameters
    num_epochs = 50
    batch_size = 256
    learning_rate = 0.001
    margin = 0.1  # Margin for ranking loss
    lambda_rank = 1.0  # Weight for ranking loss
    model_save_path = '/ssd1/divake/doubly_conformal/models/scoring_function.pth'
    plot_save_path = '/ssd1/divake/doubly_conformal/plots/scoring_function_loss.png'
    results_save_path = '/ssd1/divake/doubly_conformal/plots/scoring_function_evaluation.txt'

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
    logger.info("Loaded and froze ResNet-18 model.")

    # Define Calibration and Training Transforms
    transform_calibration = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))
    ])

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))
    ])

    # Load Datasets with the Calibration and Training Transforms
    split_dir = '/ssd1/divake/doubly_conformal/data/splits/'
    dataset_path = '/ssd1/divake/doubly_conformal/data/cifar-10-batches-py/cifar-10-batches-py/'

    # Load all loaders, but use only calibration_loader and training_loader
    train_loader, calibration_loader, _ = load_split_datasets(
        split_dir=split_dir,
        transform_train=transform_train,               # Apply training transform
        transform_calibration=transform_calibration,   # Apply calibration transform
        transform_test=None,
        dataset_path=dataset_path
    )
    logger.info("Loaded Training and Calibration DataLoaders.")

    # Extract Softmax Scores and Labels for Training Set
    logger.info("Extracting softmax scores for training set...")
    train_softmax, train_labels = extract_softmax_scores_true_class(resnet, train_loader, device)
    logger.info(f"Extracted softmax scores for {train_softmax.size(0)} training samples.")

    # Initialize Scoring Function
    scoring_fn = ScoringFunction(input_dim=1, hidden_dims=[64, 32], output_dim=1).to(device)
    logger.info("Initialized Scoring Function model.")

    # Define Optimizer
    optimizer = optim.Adam(scoring_fn.parameters(), lr=learning_rate)

    # Phase 1: Compute Tau from Calibration Set
    logger.info("Phase 1: Computing tau from calibration set...")
    calibration_softmax, calibration_labels = extract_softmax_scores_true_class(resnet, calibration_loader, device)
    calibration_non_conformity = scoring_fn(calibration_softmax.to(device)).squeeze(1)  # Shape: (N,)
    tau = compute_tau(calibration_non_conformity.cpu(), percentile=90)
    logger.info(f"Tau set to: {tau:.4f}")

    # Phase 2: Train Scoring Function with Pairwise Ranking Loss
    logger.info("Phase 2: Training Scoring Function with Pairwise Ranking Loss...")
    train_losses = []
    ranking_loss_fn = nn.MarginRankingLoss(margin=margin)

    for epoch in range(num_epochs):
        scoring_fn.train()
        running_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(range(0, train_softmax.size(0), batch_size), desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)

        for i in progress_bar:
            batch_softmax = train_softmax[i:i+batch_size].to(device)  # Shape: (batch_size,1)
            batch_size_actual = batch_softmax.size(0)

            # Forward Pass
            predictions = scoring_fn(batch_softmax).squeeze(1)  # Shape: (batch_size,)

            # Create all possible pairs within the batch
            if batch_size_actual < 2:
                continue  # Need at least two samples to form a pair

            # Generate indices for all possible pairs
            idx_i, idx_j = torch.triu_indices(batch_size_actual, batch_size_actual, offset=1)

            # Get softmax scores for pairs
            soft_i = batch_softmax[idx_i].squeeze(1)  # Shape: (num_pairs,)
            soft_j = batch_softmax[idx_j].squeeze(1)  # Shape: (num_pairs,)

            # Get predictions for pairs
            pred_i = predictions[idx_i]  # Shape: (num_pairs,)
            pred_j = predictions[idx_j]  # Shape: (num_pairs,)

            # Determine target: +1 if soft_i < soft_j, else -1
            target = torch.where(soft_i < soft_j, torch.ones_like(pred_i), -torch.ones_like(pred_i)).to(device)  # Shape: (num_pairs,)

            # Compute loss
            loss = ranking_loss_fn(pred_i, pred_j, target)

            # Backward Pass and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches +=1

            # Update Progress Bar
            progress_bar.set_postfix({'Loss': loss.item()})

        avg_loss = running_loss / num_batches if num_batches > 0 else 0
        train_losses.append(avg_loss)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save the Trained Scoring Function
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(scoring_fn.state_dict(), model_save_path)
    logger.info(f"Scoring Function model saved to {model_save_path}")

    # Plot Training Loss
    plot_training_loss(train_losses, [], plot_save_path)  # Assuming plot_training_loss handles empty validation losses
    logger.info(f"Training loss plot saved to {plot_save_path}")

    # Phase 3: Evaluate on Test Set
    logger.info("Phase 3: Evaluating Scoring Function on test set...")
    _, _, test_loader = load_split_datasets(
        split_dir=split_dir,
        transform_train=None,
        transform_calibration=None,
        transform_test=transform_calibration,  # Reuse calibration transform for test data
        dataset_path=dataset_path
    )
    logger.info("Loaded Test DataLoader.")

    # Extract Non-Conformity Scores from Test Set
    def extract_non_conformity_scores(scoring_fn, dataloader, device):
        scoring_fn.eval()
        all_non_conformity = []
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='Extracting Non-Conformity Scores', leave=False):
                images = images.to(device)
                labels = labels.to(device)
                # Assuming you want non-conformity scores for the true class
                # If scoring_fn takes in softmax scores, compute them
                resnet = models.resnet18(pretrained=False)
                resnet.fc = nn.Linear(resnet.fc.in_features, 10)
                # Load resnet state if needed or assume it's the same as before
                # Here, for simplicity, we assume that the resnet model is already loaded and frozen
                # So we would need to pass the softmax scores to the scoring_fn
                # To avoid confusion, we'll use the existing resnet model
                # Extract softmax scores
                softmax = nn.Softmax(dim=1)
                outputs = resnet(images)
                probs = softmax(outputs)
                true_softmax = probs[torch.arange(labels.size(0)), labels].unsqueeze(1)  # Shape: (batch_size,1)
                non_conf = scoring_fn(true_softmax).squeeze(1)  # Shape: (batch_size,)
                all_non_conformity.extend(non_conf.cpu().numpy().tolist())
        return all_non_conformity

    # Alternatively, reuse the evaluate_scoring_function function
    # Compute tau again after training
    # Extract non-conformity scores from calibration set with the trained scoring function
    logger.info("Recomputing tau from calibration set with trained Scoring Function...")
    calibration_non_conformity = scoring_fn(calibration_softmax.to(device)).squeeze(1)  # Shape: (N,)
    tau = compute_tau(calibration_non_conformity.cpu(), percentile=90)
    logger.info(f"Tau recomputed and set to: {tau:.4f}")

    # Evaluate Coverage and Set Size on Test Set
    coverage, avg_set_size = evaluate_scoring_function(scoring_fn, resnet, test_loader, device, tau)
    logger.info(f"Coverage on Test Set: {coverage:.2f}%")
    logger.info(f"Average Set Size on Test Set: {avg_set_size:.2f}")

    # Save results to a text file
    os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
    with open(results_save_path, 'w') as f:
        f.write(f"Tau (90th percentile): {tau:.4f}\n")
        f.write(f"Coverage on Test Set: {coverage:.2f}%\n")
        f.write(f"Average Set Size on Test Set: {avg_set_size:.2f}\n")
    logger.info(f"Evaluation results saved to {results_save_path}")

    logger.info("Scoring Function training and evaluation completed.")

if __name__ == "__main__":
    train_scoring_function()
