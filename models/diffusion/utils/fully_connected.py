import torch
import torch.nn as nn
import numpy as np

class FullyConnected(nn.Module):
    """
        This class implements a FC layer.
        It is used to pass from `input_dim` --> `output_dim` with two extra dimensions. 
    """
    
    def __init__(self, input_dim, output_dim):
        """
        Args:
            input_dim: input dimensions
            output_dim: output dimensions
        """
        super().__init__()
        self.fully_connected = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fully_connected(x).unsqueeze(-1).unsqueeze(-1)
