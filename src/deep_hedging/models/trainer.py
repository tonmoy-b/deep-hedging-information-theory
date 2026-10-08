import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

class DeepHedgingTrainer:
    def __init__(
        self, 
        model: nn.Module, 
        loss_fn: nn.Module, 
        strike_price: float = 100.0,
        transaction_cost: float = 0.001,  # 10 bps
        lr: float = 1e-3
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.strike = strike_price
        self.c = transaction_cost
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def european_call_payoff(self, S_T: torch.Tensor) -> torch.Tensor:
        return torch.relu(S_T - self.strike)

    def train_step(
        self, 
        prices: torch.Tensor, 
        rolling_features: torch.Tensor, 
        dt: float
    ) -> Tuple[float, torch.Tensor]:
        self.model.train()
        self.optimizer.zero_grad()
        
        batch_size, n_steps = rolling_features.shape[0], rolling_features.shape[1]
        device = prices.device
        
        cash = torch.zeros(batch_size, device=device)
        delta_prev = torch.zeros(batch_size, device=device)
        total_time = n_steps * dt
        
        # 1. Unroll the hedging trajectory
        for t in range(n_steps):
            S_t = prices[:, t]
            tau = torch.full((batch_size,), total_time - (t * dt), device=device)
            
            # Augmented state I_tilde: [S_t, tau, delta_prev, vol, skew, kurt]
            state_t = torch.cat([
                S_t.unsqueeze(-1),
                tau.unsqueeze(-1),
                delta_prev.unsqueeze(-1),
                rolling_features[:, t, :]
            ], dim=-1)
            
            delta_t = self.model(state_t)
            
            turnover = torch.abs(delta_t - delta_prev)
            cost_t = self.c * turnover * S_t
            
            cash = cash - (delta_t - delta_prev) * S_t - cost_t
            delta_prev = delta_t
            
        # 2. Terminal boundary liquidation
        S_T = prices[:, -1]
        cost_T = self.c * torch.abs(delta_prev) * S_T
        cash_T = cash + (delta_prev * S_T) - cost_T
        
        # 3. Liability settlement
        liability = self.european_call_payoff(S_T)
        pnl = cash_T - liability
        
        # 4. Compute risk and optimize
        loss = self.loss_fn(pnl)
        loss.backward()
        
        # Gradient clipping prevents explosions during deep unrolling
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item(), pnl.detach()