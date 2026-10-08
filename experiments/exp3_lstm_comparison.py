import torch
import torch.optim as optim
from deep_hedging.data.generator import CompositePathBuilder, ModelType, GBMParams, BatesParams
from deep_hedging.metrics.analyzer import SampleAnalyzer
from deep_hedging.models.architectures import DeepHedgingMLP, DeepHedgingLSTM
from deep_hedging.models.loss import GainWeightedDeepHedgingLoss
from deep_hedging.models.trainer import DeepHedgingTrainer

def train_lstm_naive(model, prices, dt, epochs=50, c=0.001, strike=100.0):
    device = prices.device
    batch_size, n_steps = prices.shape[0], prices.shape[1] - 1
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = GainWeightedDeepHedgingLoss(alpha=0.95, lambda_gain=0.1)
    
    total_time = n_steps * dt
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        cash = torch.zeros(batch_size, device=device)
        delta_prev = torch.zeros(batch_size, device=device)
        hx = None # LSTM hidden state
        
        for t in range(n_steps):
            S_t = prices[:, t]
            tau = torch.full((batch_size,), total_time - (t * dt), device=device)
            
            # Naive state: ONLY Price, Time, and Previous Delta (No rolling moments)
            state_t = torch.cat([S_t.unsqueeze(-1), tau.unsqueeze(-1), delta_prev.unsqueeze(-1)], dim=-1)
            
            delta_t, hx = model(state_t, hx)
            
            cost_t = c * torch.abs(delta_t - delta_prev) * S_t
            cash = cash - (delta_t - delta_prev) * S_t - cost_t
            delta_prev = delta_t
            
        S_T = prices[:, -1]
        cost_T = c * torch.abs(delta_prev) * S_T
        cash_T = cash + (delta_prev * S_T) - cost_T
        
        pnl = cash_T - torch.relu(S_T - strike)
        loss = loss_fn(pnl)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"  LSTM Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
            
    # Calculate final CVaR
    model.eval()
    sorted_pnl, _ = torch.sort(pnl.detach())
    k = max(1, int(pnl.shape[0] * 0.05))
    return -sorted_pnl[:k].mean().item()

def run_experiment_3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Experiment 3: Enriched MLP vs Naive LSTM on {device} ---")

    dt = 1.0 / 252.0
    builder = CompositePathBuilder(num_paths=5000, dt=dt, device=device)
    builder.add_regime(ModelType.GBM, steps=60, params=GBMParams(mu=0.05, sigma=0.15))
    builder.add_regime(ModelType.BATES, steps=40, params=BatesParams(mu=-0.2, kappa=3.0, theta=0.08, xi=0.4, rho=-0.7, lam=5.0, mu_j=-0.05, sigma_j=0.02))
    
    sample = builder.build()
    features = SampleAnalyzer.extract_rolling_features(sample)

    print("\n1. Training Naive LSTM (Implicit Memory)...")
    lstm_model = DeepHedgingLSTM(input_dim=3, hidden_dim=64, num_layers=1).to(device)
    lstm_cvar = train_lstm_naive(lstm_model, sample.prices, dt)

    print("\n2. Training Enriched MLP (Explicit Telemetry)...")
    mlp_model = DeepHedgingMLP(input_dim=6, hidden_dim=64, num_hidden_layers=3).to(device)
    trainer = DeepHedgingTrainer(mlp_model, GainWeightedDeepHedgingLoss(), lr=2e-3)
    
    for epoch in range(50):
        mlp_loss, mlp_pnl = trainer.train_step(sample.prices, features, dt)
        if (epoch + 1) % 10 == 0:
            print(f"  MLP Epoch {epoch+1}/50 | Loss: {mlp_loss:.4f}")
            
    sorted_pnl, _ = torch.sort(mlp_pnl)
    k = max(1, int(mlp_pnl.shape[0] * 0.05))
    mlp_cvar = -sorted_pnl[:k].mean().item()

    print("\nRESULTS FOR ARCHITECTURE COMPARISON:")
    print("-" * 65)
    print(f"{'Architecture':<25} | {'State Input':<20} | {'Realized CVaR'}")
    print("-" * 65)
    print(f"{'LSTM (Recurrent)':<25} | {'Naive (3 dims)':<20} | ${lstm_cvar:.2f}")
    print(f"{'MLP (Feedforward)':<25} | {'Enriched (6 dims)':<20} | ${mlp_cvar:.2f}")
    print("-" * 65)

if __name__ == "__main__":
    run_experiment_3()