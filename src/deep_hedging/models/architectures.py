import torch
import torch.nn as nn
import torch.optim as optim

class DeepHedgingMLP(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, num_hidden_layers: int = 3):
        super().__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            
        # Output layer maps to a single hedge ratio
        # Sigmoid bounds the output to (0, 1) which is mathematically correct for a Call Option delta
        layers.extend([nn.Linear(hidden_dim, 1), nn.Sigmoid()])
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, input_dim) representing the enriched state I_tilde
        Returns:
            delta: Tensor of shape (batch_size,) representing the target hedge ratio
        """
        return self.network(x).squeeze(-1)