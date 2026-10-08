import math
import torch
from typing import Dict, Optional

class DivergenceMetrics:
    @staticmethod
    @torch.no_grad()
    def compute_jensen_shannon(
        m_train: torch.Tensor,
        m_live: torch.Tensor,
        bandwidth: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Computes multivariate Jensen-Shannon Divergence D_JS(P_train || Q_live)
        over rolling-window moment vectors M_t in R^d using Gaussian Parzen KDE.
        
        Args:
            m_train: Reference moments from training bank, shape (N_p, d)
            m_live: Observed moments from live market stream, shape (N_q, d)
        """
        # Flatten temporal and batch dimensions to treat as a set of observed states
        m_train = m_train.reshape(-1, m_train.shape[-1])
        m_live = m_live.reshape(-1, m_live.shape[-1])
        
        n_p, d = m_train.shape
        n_q = m_live.shape[0]

        # Standardize using reference training bank mean and std
        mu_ref = m_train.mean(dim=0, keepdim=True)
        std_ref = m_train.std(dim=0, keepdim=True).clamp(min=1e-6)

        p_norm = (m_train - mu_ref) / std_ref
        q_norm = (m_live - mu_ref) / std_ref

        if bandwidth is None:
            bandwidth = float(0.5 * (n_p + n_q)) ** (-1.0 / (d + 4.0))

        def log_kde_density(eval_pts: torch.Tensor, ref_pts: torch.Tensor, is_self: bool) -> torch.Tensor:
            dist_sq = torch.cdist(eval_pts, ref_pts, p=2.0).pow(2)
            log_k = -dist_sq / (2.0 * (bandwidth ** 2))
            
            if is_self:
                # Leave-one-out to prevent zero-distance infinity spikes
                eye = torch.eye(eval_pts.shape[0], dtype=torch.bool, device=eval_pts.device)
                log_k.masked_fill_(eye, float("-inf"))
                n_eff = ref_pts.shape[0] - 1
            else:
                n_eff = ref_pts.shape[0]
                
            norm_const = -0.5 * d * math.log(2.0 * math.pi * (bandwidth ** 2)) - math.log(n_eff)
            return torch.logsumexp(log_k, dim=-1) + norm_const

        # Evaluate log P(x) and log Q(x) at x ~ P_train
        log_p_at_p = log_kde_density(p_norm, p_norm, is_self=True)
        log_q_at_p = log_kde_density(p_norm, q_norm, is_self=False)
        log_m_at_p = torch.logaddexp(log_p_at_p, log_q_at_p) - math.log(2.0)

        # Evaluate log P(y) and log Q(y) at y ~ Q_live
        log_q_at_q = log_kde_density(q_norm, q_norm, is_self=True)
        log_p_at_q = log_kde_density(q_norm, p_norm, is_self=False)
        log_m_at_q = torch.logaddexp(log_p_at_q, log_q_at_q) - math.log(2.0)

        kl_p_m = (log_p_at_p - log_m_at_p).mean().clamp(min=0.0)
        kl_q_m = (log_q_at_q - log_m_at_q).mean().clamp(min=0.0)

        jsd_nats = torch.clamp(0.5 * (kl_p_m + kl_q_m), min=0.0, max=math.log(2.0)).item()
        jsd_normalized = jsd_nats / math.log(2.0)

        return {
            "D_JS_nats": jsd_nats,
            "D_JS_normalized": jsd_normalized,
        }