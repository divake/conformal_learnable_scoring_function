# src/utils.py

import logging
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seed(seed: int = 42):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging(log_file: str):
    """
    Setup logging configuration.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def plot_training_loss(train_losses: list, val_losses: list, save_path: str):
    """
    Plot training and validation loss curves.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_per_class_scores(class_scores: dict, save_path: str):
    """
    Plot per-class non-conformity scores.
    """
    classes = list(class_scores.keys())
    scores = [class_scores[cls] for cls in classes]
    
    plt.figure(figsize=(12, 8))
    plt.bar(classes, scores, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Average Non-Conformity Score')
    plt.title('Per-Class Non-Conformity Scores')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_conformal_metrics(metrics: dict, save_path: str):
    """
    Plot coverage and set size metrics.
    """
    labels = list(metrics.keys())
    coverage = [metrics[label]['coverage'] for label in labels]
    set_size = [metrics[label]['set_size'] for label in labels]
    
    x = np.arange(len(labels))
    width = 0.35  # width of the bars
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    color = 'tab:blue'
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Coverage (%)', color=color)
    ax1.bar(x - width/2, coverage, width, label='Coverage', color=color, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:red'
    ax2.set_ylabel('Average Set Size', color=color)  # we already handled the x-label with ax1
    ax2.bar(x + width/2, set_size, width, label='Set Size', color=color, alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.title('Conformal Prediction Metrics per Class')
    plt.savefig(save_path)
    plt.close()
