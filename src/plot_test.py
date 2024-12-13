# visualize_all_classes_non_conformity.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model import ScoringFunction  # Ensure this import matches your project structure
import os
import logging

def setup_logging():
    """
    Sets up logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def load_scoring_function(model_path: str, device: torch.device):
    """
    Loads the trained Scoring Function model.
    
    Args:
        model_path (str): Path to the trained model's state_dict.
        device (torch.device): Device to load the model on.
    
    Returns:
        nn.Module: Loaded Scoring Function model.
    """
    if not os.path.exists(model_path):
        logging.error(f"Scoring Function model file not found at {model_path}.")
        raise FileNotFoundError(f"Scoring Function model file not found at {model_path}.")
    
    # Initialize the model with the same architecture used during training
    scoring_fn = ScoringFunction(input_dim=10, hidden_dims=[128, 64, 32], output_dim=10).to(device)
    
    # Load the state_dict
    scoring_fn.load_state_dict(torch.load(model_path, map_location=device))
    scoring_fn.eval()
    logging.info("Loaded Scoring Function model successfully.")
    
    return scoring_fn

def generate_softmax_vectors(target_class: int, num_classes: int =10, steps: int =100):
    """
    Generates softmax score vectors by varying the score of the target class from 0 to 1.
    
    Args:
        target_class (int): Index of the target class to vary.
        num_classes (int): Total number of classes.
        steps (int): Number of steps between 0 and 1.
    
    Returns:
        List[torch.Tensor]: List of softmax score tensors.
        List[float]: Corresponding softmax scores for the target class.
    """
    softmax_vectors = []
    target_scores = np.linspace(0, 1, steps)
    
    for score in target_scores:
        remaining_score = 1.0 - score
        if remaining_score < 0:
            remaining_score = 0.0
        # Distribute the remaining score equally among other classes
        if num_classes >1:
            other_score = remaining_score / (num_classes -1)
        else:
            other_score = 0.0
        # Create softmax vector
        softmax = np.full(num_classes, other_score)
        softmax[target_class] = score
        # Normalize to ensure sum to 1 (optional, since we already distribute remaining_score)
        softmax = softmax / softmax.sum()
        softmax_vectors.append(torch.tensor(softmax, dtype=torch.float32))
    return softmax_vectors, target_scores

def visualize_non_conformity_all_classes(model_path: str, class_names: list, save_path: str, num_classes: int =10, steps: int =100):
    """
    Visualizes the variation in non-conformity scores for each class by varying their softmax scores,
    plotting all classes in a single graph.
    
    Args:
        model_path (str): Path to the trained Scoring Function model.
        class_names (list): List of class names corresponding to class indices.
        save_path (str): Path to save the consolidated plot.
        num_classes (int): Total number of classes.
        steps (int): Number of steps between 0 and 1 for softmax score variation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained Scoring Function
    scoring_fn = load_scoring_function(model_path, device)
    
    # Prepare the plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.get_cmap('tab10', num_classes)  # Get a colormap with distinct colors
    
    for cls_idx, cls_name in enumerate(class_names):
        logging.info(f"Processing class '{cls_name}' (Index: {cls_idx})...")
        softmax_vectors, target_scores = generate_softmax_vectors(cls_idx, num_classes=num_classes, steps=steps)
        non_conformity_scores = []
        
        # Process in batches for efficiency
        batch_size = 50
        for i in range(0, len(softmax_vectors), batch_size):
            batch = torch.stack(softmax_vectors[i:i+batch_size]).to(device)  # Shape: (batch_size, num_classes)
            with torch.no_grad():
                outputs = scoring_fn(batch)  # Shape: (batch_size, num_classes)
                # Extract non-conformity scores for the target class
                target_nc = outputs[:, cls_idx].cpu().numpy()
                non_conformity_scores.extend(target_nc.tolist())
        
        # Plot the results
        plt.plot(target_scores, non_conformity_scores, label=cls_name, color=colors(cls_idx))
    
    plt.xlabel('Softmax Score')
    plt.ylabel('Non-Conformity Score')
    plt.title('Non-Conformity Score vs Softmax Score for All Classes')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Consolidated non-conformity scores plot saved to {save_path}")

def main():
    setup_logging()
    
    # Configuration
    model_path = '/ssd1/divake/doubly_conformal/models/scoring_function.pth'  # Update if different
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    save_path = '/ssd1/divake/doubly_conformal/plots/all_classes_non_conformity_variation.png'
    
    visualize_non_conformity_all_classes(model_path, class_names, save_path, num_classes=10, steps=100)

if __name__ == "__main__":
    main()
