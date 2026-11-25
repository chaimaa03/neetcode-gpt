import torch
import torch.nn as nn
import math
from typing import List
import numpy as np


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        weights=torch.empty(fan_out,fan_in)
        torch.manual_seed(0)
        
        
        nn.init.xavier_normal_(weights)
        # Use torch.manual_seed(0) for reproducibility
        
        
        return np.round(weights,4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        weights=torch.empty(fan_out,fan_in)
        torch.manual_seed(0)
        nn.init.kaiming_normal_(weights,mode="fan_in",nonlinearity="relu")
        # Use torch.manual_seed(0) for reproducibility
        return np.round(weights,4).tolist()
        
    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Use torch.manual_seed(0) once at the start.
        torch.manual_seed(0)
        # Sample all weights in layer order with torch.randn(out_dim, in_dim) * std:
        weights=[]
        # first (hidden_dim, input_dim), then (hidden_dim, hidden_dim).
        
        for i in range(num_layers):
            if i==0:
                in_dim=input_dim
            else:
                in_dim=hidden_dim
            out_dim=hidden_dim
            if init_type=="xavier":
                std=math.sqrt(2/(in_dim+out_dim))
            elif init_type=="kaiming":
                std=math.sqrt(2/in_dim)
            else:
                std=1

            w=torch.randn(out_dim,in_dim)*std
            weights.append(w)

        x = torch.randn(1, input_dim)
        std_activation=[]
        for w in weights:
            x = torch.relu(x @ w.T)
            std_activation.append(round(x.std().item(),2))
        return std_activation
        # Keep weights unrounded; do not call the init helpers, which reset the seed.
        # Then sample x = torch.randn(1, input_dim) without resetting the seed.
        # Apply x = torch.relu(x @ W.T) at every layer, with no biases.
        # Return the std of activations after each layer, rounded to 2 decimals.
        # Use round(x.std().item(), 2) with PyTorch's default sample std.
        
