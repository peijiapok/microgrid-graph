"""Tests for the Markov outage process and the continuity surrogate loss."""
from __future__ import annotations

import numpy as np
import torch

from sg_resilience.continuity_loss import windowed_continuity_loss
from sg_resilience.outages import (
    generate_markov_outage_matrix,
    stationary_outage_rate,
)


# ---- Markov outage process ----
def test_stationary_rate_formula():
    assert abs(stationary_outage_rate(0.05, 0.85) - 0.05 / (0.05 + 0.15)) < 1e-12
    assert stationary_outage_rate(0.0, 0.9) == 0.0


def test_markov_empirical_rate_and_persistence():
    rng = np.random.default_rng(0)
    p_out, p_stay = 0.05, 0.9
    mat = generate_markov_outage_matrix(200, 2000, p_out, p_stay, rng)
    # empirical outage rate ~ stationary
    emp = mat.mean()
    assert abs(emp - stationary_outage_rate(p_out, p_stay)) < 0.02
    # persistence: mean outage run length ~ 1/(1-p_stay) = 10
    runs = []
    for col in mat.T:
        run = 0
        for v in col:
            if v:
                run += 1
            elif run:
                runs.append(run)
                run = 0
        if run:
            runs.append(run)
    mean_run = np.mean(runs)
    assert 7.0 < mean_run < 13.0  # around 1/(1-0.9)=10


def test_persistence_beats_iid():
    rng = np.random.default_rng(1)
    persistent = generate_markov_outage_matrix(100, 1000, 0.05, 0.9, rng)
    iid = generate_markov_outage_matrix(100, 1000, 0.2, 0.2, rng)  # ~same rate, no memory

    def mean_run(mat):
        runs = []
        for col in mat.T:
            r = 0
            for v in col:
                if v:
                    r += 1
                elif r:
                    runs.append(r); r = 0
            if r:
                runs.append(r)
        return np.mean(runs) if runs else 0.0

    assert mean_run(persistent) > mean_run(iid)


# ---- continuity surrogate loss ----
def test_loss_differentiable():
    T, k = 8, 3
    a = torch.full((T, k), 0.5, requires_grad=True)
    d = torch.ones(T, k)
    m = torch.full((k,), 0.5)
    w = torch.tensor([3.0, 2.0, 1.0])
    loss = windowed_continuity_loss(a, d, m, w, windows=(2, 4), beta=10.0)
    loss.backward()
    assert a.grad is not None
    assert torch.isfinite(a.grad).all()
    assert a.grad.abs().sum() > 0


def test_more_service_lowers_loss():
    T, k = 8, 3
    d = torch.ones(T, k)
    m = torch.full((k,), 0.5)
    w = torch.ones(k)
    served = windowed_continuity_loss(torch.ones(T, k), d, m, w, windows=(2, 4), beta=15.0)
    unserved = windowed_continuity_loss(torch.zeros(T, k), d, m, w, windows=(2, 4), beta=15.0)
    assert served.item() < unserved.item()


def test_outage_transparency():
    T, k = 6, 1
    # served everywhere except an outaged middle step
    a = torch.tensor([[1.0], [1.0], [0.0], [1.0], [1.0], [1.0]])
    d = torch.ones(T, k)
    m = torch.full((k,), 0.5)
    w = torch.ones(k)
    outage = torch.tensor([[0], [0], [1], [0], [0], [0]], dtype=torch.bool)
    with_out = windowed_continuity_loss(a, d, m, w, windows=(4,), beta=15.0, outage=outage)
    without = windowed_continuity_loss(a, d, m, w, windows=(4,), beta=15.0, outage=None)
    # making the outaged step transparent yields MORE continuity (lower loss)
    assert with_out.item() < without.item()
