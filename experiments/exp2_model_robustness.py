import torch
from deep_hedging.data.generator import CompositePathBuilder, ModelType, GBMParams, BatesParams
from deep_hedging.data.mutator import SampleMutator
from deep_hedging.metrics.analyzer import SampleAnalyzer
from deep_hedging.metrics.divergence import DivergenceMetrics
from deep_hedging.models.architectures import DeepHedgingMLP
from deep_hedging.models.loss import GainWeightedDeepHedgingLoss
from deep_hedging.models.trainer import DeepHedgingTrainer

@torch.no_grad()
def evaluate_cvar(model, prices, features, dt, strike=100.0, c=0.001, alpha=0.95):
    """Evaluates the 95% CVaR of a trained policy on a given test set."""
    model.eval()
    batch_size, n_steps = features.shape[0], features.shape[1]
    device = prices.device
    
    cash = torch.zeros(batch_size, device=device)
    delta_prev = torch.zeros(batch_size, device=device)
    total_time = n_steps * dt
    
    for t in range(n_steps):
        S_t = prices[:, t]
        tau = torch.full((batch_size,), total_time - (t * dt), device=device)
        
        state_t = torch.cat([
            S_t.unsqueeze(-1), tau.unsqueeze(-1), delta_prev.unsqueeze(-1), features[:, t, :]
        ], dim=-1)
        
        delta_t = model(state_t)
        cost_t = c * torch.abs(delta_t - delta_prev) * S_t
        cash = cash - (delta_t - delta_prev) * S_t - cost_t
        delta_prev = delta_t
        
    S_T = prices[:, -1]
    cost_T = c * torch.abs(delta_prev) * S_T
    cash_T = cash + (delta_prev * S_T) - cost_T
    
    pnl = cash_T - torch.relu(S_T - strike)
    
    # Calculate 95% CVaR
    sorted_pnl, _ = torch.sort(pnl)
    k = max(1, int(pnl.shape[0] * (1.0 - alpha)))
    cvar = -sorted_pnl[:k].mean().item()
    return cvar

def run_experiment_2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Experiment 2: Policy Robustness on {device} ---")

    dt = 1.0 / 252.0
    gbm_calm = GBMParams(mu=0.05, sigma=0.15)
    bates_crash = BatesParams(mu=-0.2, kappa=3.0, theta=0.08, xi=0.4, rho=-0.7, lam=5.0, mu_j=-0.05, sigma_j=0.02)

    # 1. Generate Training Data
    print("Generating training databank...")
    builder = CompositePathBuilder(num_paths=5000, dt=dt, device=device)
    builder.add_regime(ModelType.GBM, steps=60, params=gbm_calm)
    builder.add_regime(ModelType.BATES, steps=40, params=bates_crash)
    train_sample = builder.build()
    train_features = SampleAnalyzer.extract_rolling_features(train_sample)

    # 2. Train the Neural Network
    print("Training Enriched Deep Hedging MLP (50 epochs)...")
    model = DeepHedgingMLP(input_dim=6, hidden_dim=64, num_hidden_layers=3).to(device)
    loss_fn = GainWeightedDeepHedgingLoss(alpha=0.95, lambda_gain=0.1)
    trainer = DeepHedgingTrainer(model, loss_fn, lr=2e-3)

    for epoch in range(50):
        loss_val, _ = trainer.train_step(train_sample.prices, train_features, dt)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/50 | Composite Loss: {loss_val:.4f}")

    # 3. Establish Baseline on Clean Test Set
    test_sample_clean = builder.build()
    test_features_clean = SampleAnalyzer.extract_rolling_features(test_sample_clean)
    baseline_cvar = evaluate_cvar(model, test_sample_clean.prices, test_features_clean, dt)

    # 4. Inject Market Shocks (OOD Test Sets)
    print("\nInjecting Out-of-Distribution Market Shocks...")
    
    # Shock A: Sudden 3x Volatility Multiplier in the final 20 days
    mutated_vol = SampleMutator.inject_volatility_multiplier(
        test_sample_clean, start_step=80, end_step=100, vol_multiplier=3.0
    )
    features_vol = SampleAnalyzer.extract_rolling_features(mutated_vol)
    
    # Shock B: Extreme Poisson Flash Crashes across the entire timeline
    mutated_crash = SampleMutator.inject_poisson_shocks(
        test_sample_clean, intensity=25.0, shock_mu=-0.15, shock_sigma=0.05
    )
    features_crash = SampleAnalyzer.extract_rolling_features(mutated_crash)

    # 5. Measure Observability Divergence & Model Degradation
    print("Computing Jensen-Shannon Divergence telemetry...")
    jsd_vol = DivergenceMetrics.compute_jensen_shannon(train_features, features_vol)["D_JS_normalized"]
    jsd_crash = DivergenceMetrics.compute_jensen_shannon(train_features, features_crash)["D_JS_normalized"]

    cvar_vol = evaluate_cvar(model, mutated_vol.prices, features_vol, dt)
    cvar_crash = evaluate_cvar(model, mutated_crash.prices, features_crash, dt)

    # 6. Output Table 2
    print("\nRESULTS FOR TABLE 2 (Policy Degradation vs. Observability Divergence):")
    print("-" * 75)
    print(f"{'Mutation Intensity':<20} | {'Injected Shock Type':<25} | {'JSD (Drift)':<10} | {'Realized CVaR'}")
    print("-" * 75)
    print(f"{'0.0x (Baseline)':<20} | {'None (In-Distribution)':<25} | {0.000:<10.3f} | ${baseline_cvar:.2f}")
    print(f"{'3.0x Multiplier':<20} | {'Late-stage Vol Scaler':<25} | {jsd_vol:<10.3f} | ${cvar_vol:.2f}")
    print(f"{'25.0 Intensity':<20} | {'Poisson Flash Crashes':<25} | {jsd_crash:<10.3f} | ${cvar_crash:.2f}")
    print("-" * 75)

if __name__ == "__main__":
    run_experiment_2()