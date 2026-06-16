"""Per-node two-state Markov outage process (ADR-0001 / spec §5).

Continuity is only policy-controllable if outages are *persistent* rather than
i.i.d. per step (else continuity is dictated by exogenous flicker). This module
generates per-node outage trajectories from an independent two-state Markov
chain per node, using real node ids (the existing scenario_generator inserts
synthetic global outage names, which cannot bind to nodes).

State per node: up (0) / out (1).
    P(up  -> out) = p_out     (failure hazard)
    P(out -> out) = p_stay    (persistence; higher => longer outages)
Stationary outage rate = p_out / (p_out + (1 - p_stay)).
Edge/spatially-correlated outages are a separate, Fanchen-coupled design and
are NOT implemented here.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from .scenario_schema import Scenario


def stationary_outage_rate(p_out: float, p_stay: float) -> float:
    denom = p_out + (1.0 - p_stay)
    return 0.0 if denom <= 0 else p_out / denom


def generate_markov_outage_matrix(
    n_nodes: int,
    horizon: int,
    p_out: float,
    p_stay: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """(horizon, n_nodes) bool matrix; True = outaged. Starts from stationary."""
    out = np.zeros((horizon, n_nodes), dtype=bool)
    pi_out = stationary_outage_rate(p_out, p_stay)
    state = rng.uniform(size=n_nodes) < pi_out  # stationary init
    for t in range(horizon):
        out[t] = state
        # next state
        u = rng.uniform(size=n_nodes)
        nxt = np.where(state, u < p_stay, u < p_out)
        state = nxt
    return out


def apply_markov_outages(
    base: Scenario,
    p_out: float = 0.05,
    p_stay: float = 0.85,
    seed: int = 0,
    node_subset: Sequence[str] | None = None,
) -> Scenario:
    """Return a copy of `base` with per-node Markov outages on its states.

    Outages are drawn over `node_subset` (default: all load nodes) and written
    into each TimeStepState.outages as real node ids.
    """
    rng = np.random.default_rng(seed)
    node_ids = list(node_subset) if node_subset is not None else [n.node_id for n in base.nodes]
    horizon = len(base.states)
    mat = generate_markov_outage_matrix(len(node_ids), horizon, p_out, p_stay, rng)

    states = []
    for t, st in enumerate(base.states):
        outaged = tuple(sorted(node_ids[i] for i in range(len(node_ids)) if mat[t, i]))
        # demands for outaged nodes are unserveable; keep demand but mark outage
        states.append(replace(st, outages=tuple(sorted(set(st.outages) | set(outaged)))))

    return Scenario(
        name=f"{base.name}_markov_out",
        nodes=base.nodes,
        states=tuple(states),
        adjacency_matrix=base.adjacency_matrix,
        metadata={
            **base.metadata,
            "outage_process": "markov_two_state",
            "p_out": str(p_out),
            "p_stay": str(p_stay),
            "stationary_outage_rate": f"{stationary_outage_rate(p_out, p_stay):.4f}",
        },
    )
