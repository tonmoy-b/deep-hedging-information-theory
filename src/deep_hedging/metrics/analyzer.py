import math
import torch
from typing import Dict, Optional
from deep_hedging.data.generator import SyntheticSample


class SampleAnalyzer:
    """
    Computes information-theoretic metrics and rolling statistical moments 
    to quantify the state-space sufficiency of Deep Hedging training data.
    """

    @staticmethod
    def extract_rolling_features(sample: SyntheticSample, window_size: int = 20) -> torch.Tensor:
        log_rets = torch.diff(torch.log(sample.prices), dim=1)
        windows = log_rets.unfold(dimension=1, size=window_size, step=1)
        
        mean = windows.mean(dim=-1, keepdim=True)
        std = windows.std(dim=-1, unbiased=True, keepdim=True).clamp(min=1e-8)
        
        diffs = windows - mean
        skew = (diffs.pow(3).mean(dim=-1, keepdim=True)) / std.pow(3)
        kurt = (diffs.pow(4).mean(dim=-1, keepdim=True)) / std.pow(4) - 3.0
        
        m_features = torch.cat([std, skew, kurt], dim=-1)
        
        padding = torch.zeros(
            sample.prices.shape[0], 
            window_size - 1, 
            3, 
            device=sample.prices.device
        )
        return torch.cat([padding, m_features], dim=1)

    @staticmethod
    @torch.no_grad()
    def compute_conditional_entropy_kde(
        features: torch.Tensor, 
        labels: torch.Tensor, 
        num_regimes: int, 
        bandwidth: Optional[float] = None,
        max_samples: int = 5000  # <--- SAFEGUARD ADDED HERE
    ) -> torch.Tensor:
        """
        Estimates Shannon Conditional Entropy H(Z | X) in nats.
        Uses random subsampling to prevent OOM errors on large trajectory batches.
        """
        X = features.reshape(-1, features.shape[-1])
        Z = labels.reshape(-1)
        
        # --- OOM Prevention: Randomly subsample if the dataset is too massive ---
        N_total = X.shape[0]
        if N_total > max_samples:
            indices = torch.randperm(N_total, device=X.device)[:max_samples]
            X = X[indices]
            Z = Z[indices]
            
        N, d = X.shape
        device = X.device
        
        std = X.std(dim=0, keepdim=True).clamp(min=1e-6)
        x_norm = (X - X.mean(dim=0, keepdim=True)) / std

        if bandwidth is None:
            bandwidth = float(N) ** (-1.0 / (d + 4.0))

        dist_sq = torch.cdist(x_norm, x_norm, p=2.0).pow(2)
        log_weights = -dist_sq / (2.0 * (bandwidth ** 2))
        
        eye_mask = torch.eye(N, dtype=torch.bool, device=device)
        log_weights.masked_fill_(eye_mask, float("-inf"))
        
        weights = torch.softmax(log_weights, dim=-1)
        z_onehot = torch.nn.functional.one_hot(Z.long(), num_classes=num_regimes).float()
        
        posterior_probs = weights @ z_onehot
        true_regime_probs = posterior_probs.gather(dim=-1, index=Z.long().unsqueeze(-1)).squeeze(-1)
        
        return -torch.log(true_regime_probs.clamp(min=1e-12)).mean()

    @classmethod
    def analyze_information_gain(
        cls, 
        sample: SyntheticSample, 
        window_size: int = 20,
        max_samples: int = 5000
    ) -> Dict[str, float]:
        """
        Evaluates how much the rolling moments M_t reduce the ambiguity of Z_t.
        """
        i_raw = sample.prices[:, 1:].unsqueeze(-1)
        m_features = cls.extract_rolling_features(sample, window_size)
        i_tilde = torch.cat([i_raw, m_features], dim=-1)
        labels = sample.regime_labels[:, 1:]
        
        h_raw = cls.compute_conditional_entropy_kde(
            i_raw, labels, sample.num_regimes, max_samples=max_samples
        )
        h_aug = cls.compute_conditional_entropy_kde(
            i_tilde, labels, sample.num_regimes, max_samples=max_samples
        )
        
        mi_gain = torch.clamp(h_raw - h_aug, min=0.0)
        
        return {
            "H(Z|I_raw)": h_raw.item(),
            "H(Z|I_tilde)": h_aug.item(),
            "Information_Gain_nats": mi_gain.item(),
            "Ambiguity_Reduction_pct": (100.0 * mi_gain / h_raw.clamp(min=1e-12)).item()
        }