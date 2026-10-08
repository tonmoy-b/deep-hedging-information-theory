This repository contains the experimental suite for evaluating deep hedging models through an information-theoretic lens. 
It addresses the one-to-many state mapping problem in Markov-switching markets by framing deep hedging as a closed-loop information system.

The codebase provides three primary subsystems:

1. Composite Path Generation: Vectorized synthesis of non-stationary market trajectories (GBM, Heston, Bates) with strict boundary continuity and rejection sampling.

2. Deep Learning Engine: PyTorch-based policy optimization utilizing rolling-window state enrichment and a gain-weighted composite risk loss to prevent gradient starvation.

3. Production Observability: Runtime telemetry that computes the Jensen-Shannon Divergence (JSD) between training and live data distributions to flag out-of-distribution market shocks.
