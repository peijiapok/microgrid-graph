"""Tests for metrics_v1 — capacity-normalized windowed continuity (spec §9)."""
from __future__ import annotations

import numpy as np

from sg_resilience import metrics_v1 as M


def _served_to_alloc(served_seq, m=0.5, d=1.0):
    """Build (a,d,m) so served_matrix reproduces a desired served sequence."""
    served_seq = np.asarray(served_seq, dtype=float)
    a = np.where(served_seq > 0, d, 0.0)  # served -> a=d>=m*d ; unserved -> a=0
    return a.reshape(-1, 1), np.full_like(a, d).reshape(-1, 1), np.array([m])


# 1. Discrimination: 1,0,1,0 and 1,1,0,0 -> equal adequacy, different continuity.
def test_discrimination_adequacy_vs_continuity():
    a1, d1, m1 = _served_to_alloc([1, 0, 1, 0])
    a2, d2, m2 = _served_to_alloc([1, 1, 0, 0])
    out = np.zeros((4, 1), dtype=bool)
    s1, s2 = M.served_matrix(a1, d1, m1), M.served_matrix(a2, d2, m2)
    assert abs(M.critical_load_adequacy(s1, out) - 0.5) < 1e-9
    assert abs(M.critical_load_adequacy(s2, out) - 0.5) < 1e-9
    # length-2 windowed continuity: 1,0,1,0 has no 2-run; 1,1,0,0 has one.
    c1 = M.windowed_continuity_per_node(s1, out, 2)[0]
    c2 = M.windowed_continuity_per_node(s2, out, 2)[0]
    assert c1 == 0.0
    assert c2 > c1  # 1/3 of the three length-2 windows


# 2. Anti-gaming: one node served forever vs spread -> coverage distinguishes.
def test_anti_gaming_coverage():
    T = 8
    # policy A: node 0 served all T, nodes 1,2 never.
    served_A = np.zeros((T, 3), dtype=bool)
    served_A[:, 0] = True
    # policy B: all three served all T.
    served_B = np.ones((T, 3), dtype=bool)
    out = np.zeros((T, 3), dtype=bool)
    w = np.array([1.0, 1.0, 1.0])
    covA = M.coverage_diagnostics(served_A, out, w, 4)
    covB = M.coverage_diagnostics(served_B, out, w, 4)
    assert covA["critical_coverage"] < covB["critical_coverage"]
    assert covA["weighted_starvation"] > covB["weighted_starvation"]
    assert abs(covB["weighted_starvation"]) < 1e-9


# 3. Oracle sanity: policy = oracle -> C = 1; degraded policy -> C < 1.
def test_oracle_normalization():
    T, k = 8, 3
    d = np.ones((T, k))
    m = np.full(k, 0.5)
    w = np.array([3.0, 2.0, 1.0])
    outage = np.zeros((T, k), dtype=bool)
    power = np.full(T, k * 0.5)  # exactly enough to keep all at min-service
    # oracle commits all three (budget fits all min-service every step)
    oracle_cont = M.greedy_global_budget_oracle(d, m, w, power, outage)
    assert oracle_cont.sum() == 3
    # policy serves everyone fully -> C == 1
    a_full = np.ones((T, k))
    s_full = M.served_matrix(a_full, d, m)
    res_full = M.capacity_normalized_continuity(s_full, outage, w, (2, 4), oracle_cont)
    assert abs(res_full["C"] - 1.0) < 1e-9
    # policy drops node 2 entirely -> C < 1
    a_drop = np.ones((T, k))
    a_drop[:, 2] = 0.0
    s_drop = M.served_matrix(a_drop, d, m)
    res_drop = M.capacity_normalized_continuity(s_drop, outage, w, (2, 4), oracle_cont)
    assert res_drop["C"] < 1.0


# 4. Outage conditioning: outaged steps don't count as interruptions.
def test_outage_conditioning():
    # served = 1,1,(out),1,1 ; with conditioning the run is unbroken length 4.
    served = np.array([[1], [1], [0], [1], [1]], dtype=bool)
    outage = np.array([[0], [0], [1], [0], [0]], dtype=bool)
    cond = M.windowed_continuity_per_node(served, outage, 4)[0]
    assert cond == 1.0  # 4 non-outaged served steps -> one full length-4 window
    # outage-inclusive (no conditioning): the zero breaks the run.
    no_out = np.zeros_like(outage)
    incl = M.windowed_continuity_per_node(served, no_out, 4)[0]
    assert incl < 1.0


# 5. Determinism / vectorization vs reference loop.
def test_window_fraction_reference():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(1, 20))
        L = int(rng.integers(1, 6))
        seq = rng.integers(0, 2, size=n).astype(bool)
        # reference: brute-force all windows
        if n < L:
            ref = 0.0
        else:
            wins = [bool(np.all(seq[t:t + L])) for t in range(n - L + 1)]
            ref = sum(wins) / len(wins)
        got = M._node_window_fraction(seq, L)
        assert abs(got - ref) < 1e-9


# 6. Bundle smoke.
def test_rollout_bundle_smoke():
    T, k = 16, 4
    rng = np.random.default_rng(1)
    d = rng.uniform(0.5, 1.5, size=(T, k))
    m = np.full(k, 0.5)
    w = rng.uniform(1.0, 5.0, size=k)
    power = np.full(T, float(np.sum(m * d.mean(axis=0))))
    outage = rng.uniform(size=(T, k)) < 0.1
    a = d.copy()  # serve everyone fully
    res = M.rollout_metrics(a, d, m, w, power, outage)
    assert 0.0 <= res["C"] <= 1.0
    assert 0.0 <= res["critical_load_adequacy"] <= 1.0
    assert set(["critical_coverage", "weighted_starvation"]).issubset(res)
