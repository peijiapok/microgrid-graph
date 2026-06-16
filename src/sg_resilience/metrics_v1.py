"""v1 metrics — capacity-normalized windowed continuity (ADR-0001).

Implements the Jia-side, policy-facing parts of `notes/continuity_metric_spec.md`
(§2,3,5,6,7) plus a global-budget greedy oracle (§4 fast fallback). The
flow-constrained oracle and the differentiable training surrogate (§4 canonical,
§8) are deferred to the C3 feasible-set finalization with Fanchen and are NOT in
this module; the oracle is injected via callable so this code is unchanged when
the flow-constrained feasible set lands.

All functions operate on numpy arrays over a single rollout:
    a, d, outage : float/bool arrays of shape (T, k) for the k critical nodes
    m, w         : float arrays of shape (k,)
Non-critical nodes are excluded by the caller before constructing these arrays.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------- #
# §2 served indicator
# --------------------------------------------------------------------------- #
def served_matrix(a: np.ndarray, d: np.ndarray, m: np.ndarray) -> np.ndarray:
    """served[t,i] = 1[a[t,i] >= m[i]*d[t,i]]. Shape (T,k) bool."""
    a = np.asarray(a, dtype=float)
    d = np.asarray(d, dtype=float)
    m = np.asarray(m, dtype=float)
    return a >= (m[None, :] * d) - EPS


def _outage_conditioned_columns(
    served: np.ndarray, outage: np.ndarray
) -> list[np.ndarray]:
    """Per node, the served sequence with outaged timesteps removed (§2).

    Measures policy behavior, not exogenous outages. Returns a list of k
    1-D arrays, each possibly shorter than T.
    """
    served = np.asarray(served, dtype=bool)
    outage = np.asarray(outage, dtype=bool)
    cols = []
    for i in range(served.shape[1]):
        keep = ~outage[:, i]
        cols.append(served[keep, i])
    return cols


# --------------------------------------------------------------------------- #
# §3 windowed continuity (numerator)
# --------------------------------------------------------------------------- #
def _node_window_fraction(seq: np.ndarray, L: int) -> float:
    """Fraction of length-L windows over `seq` that are all-served.

    Returns 0.0 if the (outage-conditioned) sequence is shorter than L.
    """
    seq = np.asarray(seq, dtype=bool)
    n = seq.shape[0]
    if n < L:
        return 0.0
    # window is all-served iff it contains no zeros: use run-based counting.
    # number of length-L all-ones windows = sum over maximal runs r of max(r-L+1,0)
    num_windows = n - L + 1
    if num_windows <= 0:
        return 0.0
    good = 0
    run = 0
    for v in seq:
        if v:
            run += 1
        else:
            if run >= L:
                good += run - L + 1
            run = 0
    if run >= L:
        good += run - L + 1
    return good / num_windows


def windowed_continuity_per_node(
    served: np.ndarray, outage: np.ndarray, L: int
) -> np.ndarray:
    """cont[i,L] per node (§3), on outage-conditioned sequences. Shape (k,)."""
    cols = _outage_conditioned_columns(served, outage)
    return np.array([_node_window_fraction(seq, L) for seq in cols], dtype=float)


def weighted_continuity_numerator(
    served: np.ndarray, outage: np.ndarray, w: np.ndarray, L: int
) -> float:
    """num_L = sum_i w_i cont[i,L]."""
    cont = windowed_continuity_per_node(served, outage, L)
    return float(np.sum(np.asarray(w, dtype=float) * cont))


# --------------------------------------------------------------------------- #
# §4 oracle frontier (global-budget greedy fallback; injected for flow case)
# --------------------------------------------------------------------------- #
def greedy_global_budget_oracle(
    d: np.ndarray,
    m: np.ndarray,
    w: np.ndarray,
    power: np.ndarray,
    outage: np.ndarray,
) -> np.ndarray:
    """Greedy priority-ordered oracle continuity per node (§4 fast fallback).

    Commits critical nodes in descending w; a node is committed iff its
    min-service demand m_i*d_{t,i} fits within remaining budget at *every*
    non-outaged step. Committed -> cont_oracle[i]=1 for all L; else 0.

    Returns cont_oracle of shape (k,) in {0,1}. Only models the GLOBAL budget;
    the flow-constrained oracle is injected separately (ADR-0001 / spec §4).
    """
    d = np.asarray(d, dtype=float)
    m = np.asarray(m, dtype=float)
    w = np.asarray(w, dtype=float)
    power = np.asarray(power, dtype=float)
    outage = np.asarray(outage, dtype=bool)
    T, k = d.shape
    need = m[None, :] * d  # (T,k) min-service demand
    remaining = power.astype(float).copy()  # (T,)
    committed = np.zeros(k, dtype=float)
    order = np.argsort(-w)  # high priority first
    for i in order:
        active = ~outage[:, i]
        req = np.where(active, need[:, i], 0.0)
        if np.all(req <= remaining + EPS):
            remaining = remaining - req
            committed[i] = 1.0
    return committed


# --------------------------------------------------------------------------- #
# capacity-normalized continuity (primary metric)
# --------------------------------------------------------------------------- #
def capacity_normalized_continuity(
    served: np.ndarray,
    outage: np.ndarray,
    w: np.ndarray,
    windows: tuple[int, ...],
    oracle_cont: np.ndarray,
) -> dict:
    """C (primary), per-L curve, and a cap flag (spec §4).

    oracle_cont[i] in [0,1] is the achievable continuity per node from an
    (offline) oracle; for the greedy global-budget oracle it is in {0,1} and
    constant across L, so den_L = sum_i w_i oracle_cont[i].
    """
    w = np.asarray(w, dtype=float)
    den = float(np.sum(w * np.asarray(oracle_cont, dtype=float)))
    per_L = {}
    capped = False
    for L in windows:
        num = weighted_continuity_numerator(served, outage, w, L)
        if den <= EPS:
            c = 0.0
        else:
            c = num / den
            if c > 1.0:
                capped = True
                c = 1.0
        per_L[L] = c
    C = float(np.mean(list(per_L.values()))) if per_L else 0.0
    return {"C": C, "per_L": per_L, "den": den, "capped": capped}


# --------------------------------------------------------------------------- #
# §5 anti-gaming diagnostics
# --------------------------------------------------------------------------- #
def coverage_diagnostics(
    served: np.ndarray, outage: np.ndarray, w: np.ndarray, L_star: int
) -> dict:
    cont = windowed_continuity_per_node(served, outage, L_star)
    w = np.asarray(w, dtype=float)
    total_w = float(np.sum(w)) + EPS
    served_any = cont > 0.0
    starved = ~served_any
    coverage = float(np.mean(served_any)) if cont.size else 0.0
    weighted_starvation = float(np.sum(w[starved]) / total_w)
    worst = float(np.min(cont)) if cont.size else 0.0
    # gini over cont
    if cont.size and np.sum(cont) > EPS:
        x = np.sort(cont)
        n = x.size
        gini = float((2 * np.sum((np.arange(1, n + 1)) * x) / (n * np.sum(x))) - (n + 1) / n)
    else:
        gini = 0.0
    return {
        "critical_coverage": coverage,
        "weighted_starvation": weighted_starvation,
        "worst_served_critical": worst,
        "gini_continuity": gini,
    }


# --------------------------------------------------------------------------- #
# §6 retained guard metric
# --------------------------------------------------------------------------- #
def critical_load_adequacy(
    served: np.ndarray, outage: np.ndarray
) -> float:
    """Mean over non-outaged critical (t,i) of served (docs/05 §2). Guard, not primary."""
    served = np.asarray(served, dtype=bool)
    outage = np.asarray(outage, dtype=bool)
    keep = ~outage
    n = int(np.sum(keep))
    if n == 0:
        return 0.0
    return float(np.sum(served[keep]) / n)


# --------------------------------------------------------------------------- #
# §7 transfer gap
# --------------------------------------------------------------------------- #
def transfer_gap(
    mu_train: float, mu_ood: float, higher_is_better: bool = True
) -> float:
    """Positive = OOD worse (docs/05 §2)."""
    return (mu_train - mu_ood) if higher_is_better else (mu_ood - mu_train)


# --------------------------------------------------------------------------- #
# convenience: full per-rollout metric bundle
# --------------------------------------------------------------------------- #
def rollout_metrics(
    a: np.ndarray,
    d: np.ndarray,
    m: np.ndarray,
    w: np.ndarray,
    power: np.ndarray,
    outage: np.ndarray,
    windows: tuple[int, ...] = (2, 4, 8, 16),
    L_star: int = 4,
    oracle: Callable[..., np.ndarray] | None = None,
) -> dict:
    """Compute the v1 metric bundle for one rollout.

    `oracle(d,m,w,power,outage) -> cont(k,)` is injectable; defaults to the
    global-budget greedy oracle. The flow-constrained oracle (Path A) plugs in
    here unchanged.
    """
    served = served_matrix(a, d, m)
    if oracle is None:
        oracle = greedy_global_budget_oracle
    oracle_cont = oracle(d, m, w, power, outage)
    T = d.shape[0]
    valid_windows = tuple(L for L in windows if L <= T)
    cont = capacity_normalized_continuity(served, outage, w, valid_windows, oracle_cont)
    cov = coverage_diagnostics(served, outage, w, min(L_star, T))
    return {
        "C": cont["C"],
        "C_per_L": cont["per_L"],
        "C_den": cont["den"],
        "C_capped": cont["capped"],
        "critical_load_adequacy": critical_load_adequacy(served, outage),
        **cov,
    }
