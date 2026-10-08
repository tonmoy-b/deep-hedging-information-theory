import torch
from deep_hedging.data.generator import CompositePathBuilder, ModelType, GBMParams, BatesParams
from deep_hedging.metrics.analyzer import SampleAnalyzer
from deep_hedging.models.architectures import DeepHedgingMLP
from deep_hedging.models.loss import GainWeightedDeepHedgingLoss
from deep_hedging.models.trainer import DeepHedgingTrainer

def run_experiment_4():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Experiment 4: Transaction Cost Ablation on {device} ---")

    dt = 1.0 / 252.0
    builder = CompositePathBuilder(num_paths=3000, dt=dt, device=device)
    builder.add_regime(ModelType.GBM, steps=60, params=GBMParams(mu=0.05, sigma=0.15))
    builder.add_regime(ModelType.BATES, steps=40, params=BatesParams(mu=-0.2, kappa=3.0, theta=0.08, xi=0.4, rho=-0.7, lam=5.0, mu_j=-0.05, sigma_j=0.02))
    
    sample = builder.build()
    features = SampleAnalyzer.extract_rolling_features(sample)
    
    cost_levels = {
        "Frictionless (0 bps)": 0.000,
        "Moderate (10 bps)": 0.001,
        "High Friction (50 bps)": 0.005
    }
    
    results = {}
    
    for label, cost in cost_levels.items():
        print(f"\nTraining with {label}...")
        model = DeepHedgingMLP(input_dim=6, hidden_dim=64, num_hidden_layers=3).to(device)
        trainer = DeepHedgingTrainer(model, GainWeightedDeepHedgingLoss(), transaction_cost=cost, lr=2e-3)
        
        final_pnl = None
        for epoch in range(40):
            loss, pnl = trainer.train_step(sample.prices, features, dt)
            final_pnl = pnl
            
        sorted_pnl, _ = torch.sort(final_pnl)
        k = max(1, int(final_pnl.shape[0] * 0.05))
        cvar = -sorted_pnl[:k].mean().item()
        results[label] = cvar
        print(f"  Converged CVaR: ${cvar:.2f}")

    print("\nRESULTS FOR TRANSACTION COST SENSITIVITY:")
    print("-" * 50)
    print(f"{'Friction Regime':<25} | {'Realized CVaR'}")
    print("-" * 50)
    for label, cvar in results.items():
        print(f"{label:<25} | ${cvar:.2f}")
    print("-" * 50)

if __name__ == "__main__":
    run_experiment_4()