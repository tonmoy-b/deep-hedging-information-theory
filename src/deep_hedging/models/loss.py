import math
import torch
import torch.nn as nn

class GainWeightedDeepHedgingLoss(nn.Module):
    """
    Implements L_composite(Theta; lambda_gain) = rho(X) - lambda_gain * E[ln(1 + ReLU(X))]
    Supports both 'entropic' and 'cvar' baseline convex risk measures.
    """
    def __init__(
        self,
        risk_type: str = "entropic",
        gamma: float = 1.0,
        alpha: float = 0.95,
        lambda_gain: float = 0.0,
    ):
        super().__init__()
        self.risk_type = risk_type.lower()
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_gain = lambda_gain

    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pnl: Terminal portfolio PnL tensor X of shape (batch_size,)
        """
        if self.risk_type == "entropic":
            # Log-sum-exp trick for numerical stability: (1/gamma) * ln E[exp(-gamma * X)]
            n = pnl.shape[0]
            risk_loss = (torch.logsumexp(-self.gamma * pnl, dim=0) - math.log(n)) / self.gamma
        elif self.risk_type == "cvar":
            # Empirical CVaR at tail level (1 - alpha)
            losses = -pnl
            var_threshold = torch.quantile(losses.detach(), self.alpha)
            tail_hinge = torch.relu(losses - var_threshold)
            risk_loss = var_threshold + tail_hinge.mean() / (1.0 - self.alpha)
        else:
            raise ValueError(f"Unsupported risk_type: {self.risk_type}")

        if self.lambda_gain > 0.0:
            # Concave upside utility prevents unbounded speculative leverage
            upside_utility = torch.log1p(torch.relu(pnl)).mean()
            return risk_loss - self.lambda_gain * upside_utility

        return risk_loss

