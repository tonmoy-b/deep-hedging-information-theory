import torch
import torch.nn as nn

class GainWeightedDeepHedgingLoss(nn.Module):
    def __init__(self, alpha: float = 0.95, lambda_gain: float = 0.1):
        """
        Args:
            alpha: Confidence level for CVaR (e.g., 0.95 for 95% CVaR).
            lambda_gain: Weight assigned to the positive PnL reward.
        """
        super().__init__()
        self.alpha = alpha
        self.lambda_gain = lambda_gain

    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        # Sort PnL ascending to isolate the left tail (worst outcomes)
        sorted_pnl, _ = torch.sort(pnl)
        
        # Calculate the index threshold for the worst (1 - alpha) cases
        k = int(pnl.shape[0] * (1.0 - self.alpha))
        k = max(1, k)
        
        # CVaR is the negative expectation of the worst cases
        worst_cases = sorted_pnl[:k]
        cvar_loss = -worst_cases.mean()
        
        # Gain penalty: actively rewards the model for positive PnL
        gain_penalty = self.lambda_gain * torch.relu(pnl).mean()
        
        return cvar_loss - gain_penalty