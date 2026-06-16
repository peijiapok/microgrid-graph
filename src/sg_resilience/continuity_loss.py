"""Differentiable windowed-continuity surrogate (ADR-0001 §4 / spec §8).

The PRIMARY metric (metrics_v1.capacity_normalized_continuity) uses hard
thresholds and is non-differentiable. This module is the soft training
surrogate that actually drives continuity through gradients — distinct from the
magnitude-smoothness term ||a_t - a_{t-1}|| (anti-chatter only).

    s[t,i]   = sigmoid(beta * (a[t,i] - m[i] d[t,i]))         # soft served
    q[t,i,L] = prod_{tau=t-L+1..t} s[tau,i]   = exp(sum log s) # soft continuity
    loss     = - sum_{L,t,i} w[i] q[t,i,L] / normalizer

As beta -> inf, q -> the hard windowed-continuity indicator. Outaged steps are
made transparent (soft-served set to 1) so the policy is not penalized for
exogenous outages, matching the metric's outage-conditioning.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

LOG_EPS = 1e-8


def soft_served(a: torch.Tensor, d: torch.Tensor, m: torch.Tensor, beta: float) -> torch.Tensor:
    """sigmoid(beta (a - m d)), shape (T,k)."""
    return torch.sigmoid(beta * (a - m.unsqueeze(0) * d))


def windowed_continuity_loss(
    a: torch.Tensor,
    d: torch.Tensor,
    m: torch.Tensor,
    w: torch.Tensor,
    windows: Sequence[int] = (2, 4, 8),
    beta: float = 10.0,
    outage: torch.Tensor | None = None,
) -> torch.Tensor:
    """Negative weighted soft windowed-continuity, averaged over windows.

    a, d : (T, k) ; m, w : (k,) ; outage : (T, k) bool or None.
    Returns a scalar tensor (lower = more continuous critical service).
    """
    T, k = a.shape
    s = soft_served(a, d, m, beta)  # (T,k)
    if outage is not None:
        # outaged steps transparent: soft-served = 1 there (detached path)
        s = torch.where(outage.bool(), torch.ones_like(s), s)
    log_s = torch.log(s.clamp_min(LOG_EPS))  # (T,k)
    # prefix sums for O(1) windowed products: cs[t] = sum_{tau<t} log_s[tau]
    cs = torch.cat([torch.zeros(1, k, dtype=log_s.dtype, device=log_s.device),
                    torch.cumsum(log_s, dim=0)], dim=0)  # (T+1, k)

    terms = []
    for L in windows:
        if L > T:
            continue
        # q[t] for t = L-1 .. T-1 : exp(cs[t+1] - cs[t+1-L])
        end = cs[L:]            # cs[L .. T]   -> (T-L+1, k)
        start = cs[: T - L + 1]  # cs[0 .. T-L]
        q = torch.exp(end - start)  # (T-L+1, k) soft windowed continuity
        wq = q * w.unsqueeze(0)
        terms.append(wq.mean())  # mean over windows and nodes (weighted)
    if not terms:
        return torch.zeros((), dtype=a.dtype, device=a.device)
    return -torch.stack(terms).mean()
