import torch
import torch.nn as nn
from typing import Tuple, Optional

class DeepHedgingMLP(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, num_hidden_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            
        layers.extend([nn.Linear(hidden_dim, 1), nn.Sigmoid()])
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output shape: (batch_size,)
        return self.network(x).squeeze(-1)


class DeepHedgingLSTM(nn.Module):
    """
    LSTM variant that maintains a persistent hidden state across the unrolled trajectory.
    """
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Ensure x has sequence dimension: (batch, 1, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        out, hx = self.lstm(x, hx)
        delta = self.fc(out.squeeze(1))
        return delta.squeeze(-1), hx