"""Tests for the mission-aware EV fleet + readiness (run: PYTHONPATH=. pytest)."""
from __future__ import annotations

import numpy as np

from ev_fleet import EVFleet, FleetConfig, readiness_continuity


def _fleet(n_emerg=3, n_reg=3, soc=0.3, node=0):
    m = n_emerg + n_reg
    return EVFleet(
        node_of_vehicle=np.full(m, node),
        is_emergency=np.array([True] * n_emerg + [False] * n_reg),
        capacity_mwh=np.full(m, 0.064),  # 64 kWh
        soc=np.full(m, soc),
        cfg=FleetConfig(mission_rate_per_h=0.0),  # no missions unless overridden
    )


def test_charging_raises_soc_and_readiness():
    f = _fleet(soc=0.3)
    rng = np.random.default_rng(0)
    assert f.readiness() == 0                      # all below theta=0.6
    for _ in range(40):                            # charge for 10 h
        f.step({0: 0.3}, rng)                       # ample charger power
    assert f.readiness() == 3                       # all emergency now ready
    assert f.soc.mean() > 0.3


def test_charging_prioritizes_emergency():
    f = _fleet(soc=0.3)
    rng = np.random.default_rng(0)
    f.step({0: 0.05}, rng)                          # scarce: ~1 vehicle worth of energy
    # highest SOC among emergency should have risen most; a regular should not beat emergency
    assert f.soc[:3].sum() >= f.soc[3:].sum()


def test_v2g_discharge_preserves_emergency_readiness():
    f = _fleet(soc=0.8)                             # all ready/high
    rng = np.random.default_rng(0)
    e0, r0 = f.soc[:3].copy(), f.soc[3:].copy()
    f.step({0: -0.2}, rng)                          # V2G discharge
    # regular vehicles drained more than emergency (readiness preserved by design)
    assert (r0 - f.soc[3:]).sum() > (e0 - f.soc[:3]).sum()


def test_missions_drain_and_fail_when_none_ready():
    f = _fleet(n_emerg=1, n_reg=0, soc=0.3)
    f.cfg.mission_rate_per_h = 100.0                # force missions every step
    rng = np.random.default_rng(1)
    failed = 0
    for _ in range(8):
        failed += f.step({0: 0.0}, rng)["missions_failed"]  # no charging
    assert failed > 0                               # SOC 0.3 < theta 0.6 -> not ready -> fail


def test_readiness_gated_dispatch_and_recharge_cycle():
    f = _fleet(n_emerg=2, n_reg=0, soc=0.9)
    f.cfg.mission_rate_per_h = 4.0
    rng = np.random.default_rng(2)
    series = []
    for _ in range(96):                             # 24 h
        series.append(f.step({0: 0.1}, rng)["readiness"])
    series = np.array(series)
    assert series.max() >= 1                         # sometimes ready
    c = readiness_continuity(series, n_min=1)
    assert 0.0 <= c <= 1.0


def test_readiness_continuity_metric():
    assert readiness_continuity(np.array([2, 2, 1, 0, 2]), n_min=2) == 0.6
