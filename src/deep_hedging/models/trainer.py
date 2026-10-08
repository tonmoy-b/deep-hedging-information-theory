import torch
import torch.nn as nn
import torch.optim as optim

class DeepHedgingTrainer:
    def __init__(
        self, 
        model: nn.Module, 
        loss_fn: nn.Module, 
        strike_price: float = 100.0,
        transaction_cost: float = 0.001, # e.g., 10 bps
        lr: float = 1e-3
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.strike = strike_price
        self.c = transaction_cost
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def european_call_payoff(self, S_T: torch.Tensor) -> torch.Tensor:
        """Terminal liability Z(S_T) = max(S_T - K, 0)"""
        return torch.relu(S_T - self.strike)

    def train_step(
        self, 
        prices: torch.Tensor, 
        rolling_features: torch.Tensor, 
        dt: float
    ) -> float:
        """
        Executes one full trajectory unroll, computes PnL, and backpropagates.
        
        Args:
            prices: (batch_size, n_steps + 1)
            rolling_features: (batch_size, n_steps, 3) containing [vol, skew, kurtosis]
            dt: Time increment per step (e.g., 1/252)
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        batch_size, n_steps = rolling_features.shape[0], rolling_features.shape[1]
        device = prices.device
        
        # Initialize tracking tensors
        cash = torch.zeros(batch_size, device=device)
        delta_prev = torch.zeros(batch_size, device=device)
        total_time = n_steps * dt
        
        # 1. Unroll the trajectory through time
        for t in range(n_steps):
            S_t = prices[:, t]
            tau = torch.full((batch_size,), total_time - (t * dt), device=device)
            
            # Construct augmented state I_tilde
            # Shape: (batch_size, 6) -> [S_t, tau, delta_prev, vol, skew, kurt]
            state_t = torch.cat([
                S_t.unsqueeze(-1),
                tau.unsqueeze(-1),
                delta_prev.unsqueeze(-1),
                rolling_features[:, t, :]
            ], dim=-1)
            
            # Forward pass to get new target hedge
            delta_t = self.model(state_t)
            
            # Calculate transaction costs: c * |delta_t - delta_prev| * S_t
            turnover = torch.abs(delta_t - delta_prev)
            cost_t = self.c * turnover * S_t
            
            # Self-financing portfolio update
            cash = cash - (delta_t - delta_prev) * S_t - cost_t
            
            # Advance state
            delta_prev = delta_t
            
        # 2. Terminal Liquidation at t = n
        S_T = prices[:, -1]
        
        # Liquidate the final held position (delta_prev goes to 0)
        cost_T = self.c * torch.abs(delta_prev) * S_T
        cash_T = cash + (delta_prev * S_T) - cost_T
        
        # 3. Settle liability and compute final PnL
        liability = self.european_call_payoff(S_T)
        pnl = cash_T - liability
        
        # 4. Compute composite risk loss and backpropagate
        loss = self.loss_fn(pnl)
        loss.backward()
        
        # Optional: Gradient clipping to prevent explosion from deep unrolling
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()