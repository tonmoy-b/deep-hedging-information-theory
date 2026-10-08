# Deep Hedging: An Information-Theoretic and Metrics-Driven Framework

This repository contains the official PyTorch implementation for the paper: **An Information-Theoretic and Metrics-Driven Framework for Deep Hedging: From Path Generation to Production Observability** by Tonmoy T. Bhattacharya.

Building upon the foundational Deep Hedging framework (Buehler et al., 2018), this codebase introduces a complete production-grade telemetry system. It quantifies state-space sufficiency during policy optimization and utilizes Jensen-Shannon Divergence (JSD) to flag out-of-distribution market shocks before they trigger financial model degradation.

The codebase provides three primary subsystems:
1. **Composite Path Generation (`src/deep_hedging/data/`):** Vectorized synthesis of non-stationary market trajectories (GBM, Heston, Bates) with strict boundary continuity and rejection sampling.
2. **Deep Learning Engine (`src/deep_hedging/models/`):** PyTorch-based policy optimization utilizing rolling-window state enrichment and a gain-weighted composite risk loss to prevent gradient starvation.
3. **Production Observability (`src/deep_hedging/metrics/`):** Runtime telemetry that computes the Jensen-Shannon Divergence (JSD) between training and live data distributions to flag out-of-distribution market shocks.

---

## Key Empirical Results

The `experiments/` directory contains the automated scripts to reproduce the paper's core findings.

### 1. Production Observability & Model Degradation (`exp2_model_robustness.py`)
Demonstrates that statistical divergence (JSD) acts as a highly accurate leading indicator for the financial breakdown (CVaR) of the hedging agent under out-of-distribution market shocks.

| Mutation Intensity | Injected Shock Type | JSD (Drift) | Realized CVaR (95%) |
| :--- | :--- | :--- | :--- |
| **0.0x (Baseline)** | None (In-Distribution) | 0.000 | $12.68 |
| **3.0x Multiplier** | Late-stage Vol Scaler | 0.059 | $24.07 |
| **25.0 Intensity** | Poisson Flash Crashes | 0.509 | $38.76 |

### 2. Implicit vs. Explicit State Representation (`exp3_lstm_comparison.py`)
Benchmarks an Enriched MLP (using explicitly engineered rolling moments) against a Recurrent LSTM (forced to learn a latent memory state organically). Proves that Deep Hedging operates optimally as a POMDP where recurrent memory outperforms explicit, lagged statistical telemetry.

| Architecture | State Input | Latent Memory | Realized CVaR (95%) |
| :--- | :--- | :--- | :--- |
| **LSTM (Recurrent)** | Naive (3 dims) | Implicit (Dynamic) | $8.97 |
| **MLP (Feedforward)** | Enriched (6 dims) | Explicit (Lagged) | $12.39 |

### 3. Transaction Cost Sensitivity (`exp4_transaction_costs.py`)
Validates the mechanical integrity of the custom Gain-Weighted Composite Loss. As friction scales, the network rationally learns to tolerate higher local delta mismatches rather than incurring prohibitive continuous rebalancing costs.

| Friction Regime | Transaction Cost ($c$) | Realized CVaR (95%) |
| :--- | :--- | :--- |
| **Frictionless** | 0 bps (0.000) | $12.80 |
| **Moderate Friction** | 10 bps (0.001) | $12.89 |
| **High Friction** | 50 bps (0.005) | $13.10 |

---

## Quickstart & Reproducibility

This project strictly utilizes the `src/` layout for clean module imports and isolated environments.

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install --upgrade pip
```
### 2. Install Dependencies and Package 
```bash
pip install -r requirements.txt
pip install -e .
```
### 3. Run the Experimental Suite
```bash
python experiments/exp1_state_sufficiency.py
python experiments/exp2_model_robustness.py
python experiments/exp3_lstm_comparison.py
python experiments/exp4_transaction_costs.py
```

## Repository Architecture
* src/deep_hedging/data/: SDE path generators (GBM, Heston, Bates), Markov-switching trajectory stitcher, and Poisson/Vol mutation injectors.

* src/deep_hedging/models/: Deep Hedging MLP/LSTM architectures, Gain-Weighted Composite Loss, and the unrolled computational graph trainer.

* src/deep_hedging/metrics/: Information-theoretic telemetry engines (Conditional Entropy, Rolling Moments, Jensen-Shannon Divergence).

* tests/: Automated pytest suite ensuring Feller condition compliance and SDE boundary continuity.

* experiments/: Executable scripts generating the tables presented in the manuscript.