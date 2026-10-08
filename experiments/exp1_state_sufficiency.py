import torch
from deep_hedging.data.generator import CompositePathBuilder, ModelType, GBMParams, BatesParams
from deep_hedging.metrics.analyzer import SampleAnalyzer

def run_experiment_1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Experiment 1: Information Gain on {device} ---")

    # 1. Setup multi-regime training distribution
    dt = 1.0 / 252.0
    gbm_calm = GBMParams(mu=0.05, sigma=0.15)
    bates_crash = BatesParams(
        mu=-0.2, kappa=3.0, theta=0.08, xi=0.4, rho=-0.7, 
        lam=5.0, mu_j=-0.05, sigma_j=0.02
    )

    builder = CompositePathBuilder(num_paths=2000, dt=dt, device=device)
    
    # Simulate a market that is calm for 3 months, then crashes for 2 months
    builder.add_regime(ModelType.GBM, steps=60, params=gbm_calm)
    builder.add_regime(ModelType.BATES, steps=40, params=bates_crash)
    
    sample = builder.build()
    
    # 2. Extract Information Metrics
    print("Computing Conditional Entropy (this may take a few seconds on KDE)...")
    metrics = SampleAnalyzer.analyze_information_gain(sample, window_size=20)
    
    # 3. Output formatted for LaTeX Table 1
    print("\nRESULTS FOR TABLE 1:")
    print("-" * 50)
    print(f"H(Z | I_raw) [Naive State]:      {metrics['H(Z|I_raw)']:.4f} nats")
    print(f"H(Z | I_tilde) [Enriched State]: {metrics['H(Z|I_tilde)']:.4f} nats")
    print(f"Information Gain I(Z ; M):       {metrics['Information_Gain_nats']:.4f} nats")
    print(f"State Ambiguity Reduction:       {metrics['Ambiguity_Reduction_pct']:.2f}%")
    print("-" * 50)
    
if __name__ == "__main__":
    run_experiment_1()