# src/model.py

import torch
import torch.nn as nn

class ScoringFunction(nn.Module):
    def __init__(self, input_dim: int =1, hidden_dims: list = [64, 32], output_dim: int =1):
        """
        Initialize the Scoring Function model.
        
        Args:
            input_dim (int): Dimension of input features (softmax score of true class).
            hidden_dims (list): List of hidden layer dimensions.
            output_dim (int): Dimension of output (non-conformity score).
        """
        super(ScoringFunction, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.ReLU())  # Ensure non-negative output
        # layers.append(nn.Sigmoid())  # Ensure output is in the range (0, 1)
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass to compute non-conformity score.
        
        Args:
            x (Tensor): Softmax score of the true class (batch_size, 1)
        
        Returns:
            Tensor: Non-conformity score (batch_size, 1)
        """
        return self.network(x)
