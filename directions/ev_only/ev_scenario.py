"""Wire the mission-aware fleet onto a (damaged) IEEE feeder and run episodes.

Roles are assigned to surviving load buses (hospital / emergency-charger /
regular-charger). Per 15-min step: compute per-node demand, allocate scarce power
under Δ_grid via the core flow-aware greedy allocator (lexicographic: hospital
floor first, then emergency > regular), feed charger power to the fleet, advance
fleet dynamics, and record readiness + hospital service. A `flow_aware` flag lets
us compare topology-aware vs topology-blind allocation (the de-risking check:
does topology matter for readiness under damage?).
"""
from __future__ import annotations

import numpy as np

from sg_resilience.flow_projection import (
    greedy_flow_allocation,
    node_ancestor_edges,
    project_onto_delta_grid,
)

from ev_fleet import EVFleet, FleetConfig, readiness_continuity

D_HOSP_MW = 0.4          # hospital critical load
HOSP_PRIORITY = 1000.0
EMERG_PRIORITY = 100.0
REG_PRIORITY = 1.0


def assign_roles(load_ids, rng, n_hosp=2, frac_emerg=0.35):
    ids = list(load_ids)
    rng.shuffle(ids)
    roles = {}
    for i, nid in enumerate(ids):
        if i < n_hosp:
            roles[nid] = "hospital"
        elif i < n_hosp + int(frac_emerg * (len(ids) - n_hosp)):
            roles[nid] = "emergency"
        else:
            roles[nid] = "regular"
    return roles


def build_fleet(roles, rng, veh_per_node=3, cap_mwh=0.064, soc0=0.5, cfg=None):
    node_of, is_emerg = [], []
    for nid, role in roles.items():
        if role in ("emergency", "regular"):
            for _ in range(veh_per_node):
                node_of.append(nid)
                is_emerg.append(role == "emergency")
    m = len(node_of)
    return EVFleet(
        node_of_vehicle=np.array(node_of), is_emergency=np.array(is_emerg),
        capacity_mwh=np.full(m, cap_mwh), soc=np.full(m, soc0),
        cfg=cfg or FleetConfig(),
    )


def _step_demand(load_ids, roles, fleet):
    """Per-node demand this step: hospital = fixed; charger = idle-vehicle charging need."""
    rate = fleet.cfg.max_rate_mw
    idle = fleet.idle()
    d = np.zeros(len(load_ids))
    for j, nid in enumerate(load_ids):
        if roles[nid] == "hospital":
            d[j] = D_HOSP_MW
        else:
            n_idle = int(np.sum((fleet.node_of_vehicle == nid) & idle))
            d[j] = n_idle * rate
    return d


def run_episode(tree, load_ids, roles, fleet, T, scarcity, rng, flow_aware=True):
    """One episode; returns readiness continuity, hospital continuity, missions failed."""
    priorities = np.array([{"hospital": HOSP_PRIORITY, "emergency": EMERG_PRIORITY,
                            "regular": REG_PRIORITY}[roles[n]] for n in load_ids])
    minfrac = np.array([1.0 if roles[n] == "hospital" else 0.0 for n in load_ids])
    critical = np.array([roles[n] == "hospital" for n in load_ids])
    # DELIVERY always respects real branch-flow (physics); PLANNING may ignore it
    # (topology-blind). A blind plan that violates flow is projected to what can
    # actually be delivered -> fair comparison of decision quality.
    real_anc, real_caps = node_ancestor_edges(tree, load_ids)
    plan_anc, plan_caps = (real_anc, real_caps) if flow_aware else ([[] for _ in load_ids], np.zeros(0))
    charger_idx = {n: j for j, n in enumerate(load_ids) if roles[n] != "hospital"}
    R, hosp_ok, missions_failed = [], [], 0
    for _ in range(T):
        d = _step_demand(load_ids, roles, fleet)
        budget = scarcity * float(d.sum())
        outage = np.zeros(len(load_ids), dtype=bool)
        a = greedy_flow_allocation(d, budget, outage, plan_anc, plan_caps, priorities,
                                   minfrac, critical, score=priorities)
        if not flow_aware:      # deliver only what real branch-flow allows
            a = project_onto_delta_grid(a, d, budget, outage, tree, load_ids)
        # hospital service continuity
        served = all(a[j] >= D_HOSP_MW - 1e-9 for j, n in enumerate(load_ids)
                     if roles[n] == "hospital")
        hosp_ok.append(served)
        # charger power -> fleet
        node_power = {n: float(a[j]) for n, j in charger_idx.items()}
        s = fleet.step(node_power, rng)
        R.append(s["readiness"]); missions_failed += s["missions_failed"]
    R = np.array(R)
    n_emerg = int(np.sum(fleet.is_emergency))
    n_min = max(1, n_emerg // 2)
    return {
        "readiness_continuity": readiness_continuity(R, n_min),
        "hospital_continuity": float(np.mean(hosp_ok)),
        "missions_failed": missions_failed,
        "mean_readiness": float(R.mean()),
        "n_emerg": n_emerg, "n_min": n_min, "n_surviving": len(load_ids),
    }
