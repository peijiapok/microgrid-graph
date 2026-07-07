"""EV-only resilience: adaptive V2G control, regular EVs, and standard metrics.

Metrics (grounded in power-system resilience literature + GPT-5.5):
  - RC  readiness continuity (availability analogue): fraction of time
        >= N_min emergency vehicles are dispatch-ready  [HEADLINE].
  - ESC emergency service continuity = 1 - unserved/arrived missions
        (EENS/EUE analogue: fraction of emergency demand served)  [OUTCOME].
  - hosp_avail hospital availability (1 - Energy-Not-Served fraction).
  - reg_evac regular-EV evacuation service (fraction of regular EVs charged to
        an evacuation target over time)  [regular-EV story, required].
V2G carries a round-trip efficiency (~85%, empirical): delivering E to the
hospital drains E/eff from EVs. Controllers: fixed-level vs ADAPTIVE (cover only
the hospital deficit, draining regular EVs first — protects emergency readiness).
"""
from __future__ import annotations

import numpy as np

from ev_fleet import EVFleet, FleetConfig
from ev_scenario import D_HOSP_MW, assign_roles, build_fleet

V2G_EFF = 0.85
EVAC_TARGET = 0.5


def fixed_v2g(level):
    def policy(hosp_demand, grid_feed, fleet, state):
        return level * hosp_demand
    return policy


def adaptive_v2g():
    """Use V2G to FREE grid budget for emergency charging (not merely cover a
    hospital shortfall). Deliver just enough V2G that the grid left after the
    hospital covers this step's emergency-charging need — protecting readiness
    while minimizing EV drain (fleet drains regular EVs first)."""
    def policy(hosp_demand, grid_feed, fleet, state):
        rate = fleet.cfg.max_rate_mw
        idle = fleet.idle()
        below = fleet.soc < fleet.theta
        emerg_need = int(np.sum(fleet.is_emergency & idle & below)) * rate   # MW to charge
        # want grid_left = grid_feed - (hosp_demand - v2g) >= emerg_need
        return float(np.clip(emerg_need + hosp_demand - grid_feed, 0.0, hosp_demand))
    return policy


def run_episode(roles, fleet: EVFleet, T, v2g_policy, feed_factor, rng,
                v2g_eff=V2G_EFF, evac_target=EVAC_TARGET, coverage_frac=0.5):
    load_ids = list(roles.keys())
    emerg_nodes = [n for n in load_ids if roles[n] == "emergency"]
    reg_nodes = [n for n in load_ids if roles[n] == "regular"]
    charger_nodes = emerg_nodes + reg_nodes
    rate = fleet.cfg.max_rate_mw
    n_hosp = sum(r == "hospital" for r in roles.values())
    hosp_demand = n_hosp * D_HOSP_MW
    grid_feed = feed_factor * hosp_demand
    dt = fleet.cfg.dt_h
    n_emerg = int(np.sum(fleet.is_emergency))
    n_reg = int(np.sum(~fleet.is_emergency))
    n_min = max(1, int(np.ceil(coverage_frac * n_emerg)))

    R, evac, arrived, failed, hosp_ens = [], [], 0, 0, 0.0

    def node_need(nodes):
        idle = fleet.idle()
        return {n: int(np.sum((fleet.node_of_vehicle == n) & idle)) * rate for n in nodes}

    for _ in range(T):
        e_need, r_need = node_need(emerg_nodes), node_need(reg_nodes)
        state = {"readiness": fleet.readiness(),
                 "min_emerg_soc": float(np.min(fleet.soc[fleet.is_emergency])) if n_emerg else 1.0}
        # V2G delivered to hospital (bounded by demand); drained from EVs w/ efficiency
        v2g_deliver = float(np.clip(v2g_policy(hosp_demand, grid_feed, fleet, state), 0, hosp_demand))
        hosp_from_grid = min(hosp_demand - v2g_deliver, grid_feed)
        hosp_served = hosp_from_grid + v2g_deliver
        hosp_ens += max(hosp_demand - hosp_served, 0.0) * dt
        B = grid_feed - hosp_from_grid                       # grid left for charging
        # emergency charging (spread), then regular
        e_tot = sum(e_need.values()); frac_e = min(1.0, B / e_tot) if e_tot else 0.0
        charge = {n: e_need[n] * frac_e for n in emerg_nodes}; B -= sum(charge.values())
        r_tot = sum(r_need.values()); frac_r = min(1.0, B / r_tot) if r_tot else 0.0
        charge.update({n: r_need[n] * frac_r for n in reg_nodes})
        # V2G draw from EVs (efficiency loss), spread as negative power
        v2g_draw = v2g_deliver / v2g_eff
        v2g_share = v2g_draw / max(len(charger_nodes), 1)
        net = {n: charge.get(n, 0.0) - v2g_share for n in charger_nodes}

        s = fleet.step(net, rng)
        R.append(s["readiness"]); arrived += s["missions_arrived"]; failed += s["missions_failed"]
        evac.append(fleet.evac_ready(evac_target))

    R = np.array(R)
    return {
        "RC": float(np.mean(R >= n_min)),
        "ESC": float(1 - failed / max(arrived, 1)),
        "hosp_avail": float(1 - hosp_ens / max(hosp_demand * T * dt, 1e-9)),
        "reg_evac": float(np.mean(evac) / max(n_reg, 1)),
        "missions_failed": failed, "missions_arrived": arrived,
    }


def _scenario(seed, feed_factor, n_remove=4, mission_rate=3.0, drain=0.35,
              theta=0.6, veh_per_node=3):
    import pandapower.networks as ppn
    from damage import damage_feeder
    rng = np.random.default_rng(seed)
    _, surv = damage_feeder(ppn.case33bw(), n_remove=n_remove, rng=rng)
    if len(surv) < 6:
        return None
    roles = assign_roles(surv, rng)
    fleet = build_fleet(roles, rng, veh_per_node=veh_per_node,
                        cfg=FleetConfig(mission_rate_per_h=mission_rate,
                                        mission_drain_frac=drain, emergency_theta=theta))
    return roles, fleet, rng


def eval_policy(policy, feed_factor, seeds=range(12), T=96, **kw):
    keys = ["RC", "ESC", "hosp_avail", "reg_evac", "missions_failed"]
    acc = {k: [] for k in keys}
    for s in seeds:
        sc = _scenario(s, feed_factor, **kw)
        if sc is None:
            continue
        roles, fleet, rng = sc
        r = run_episode(roles, fleet, T, policy, feed_factor, rng)
        for k in keys:
            acc[k].append(r[k])
    return {k: float(np.mean(v)) for k, v in acc.items()}


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    print("Adaptive V2G vs fixed levels — across degraded-feed strength (IEEE-33, damaged, 24 h)")
    print(f"{'feed':>5} {'policy':>10} {'RC':>6} {'ESC':>6} {'hosp':>6} {'reg_evac':>8} {'miss_fail':>9}")
    for ff in (0.6, 0.8, 1.0, 1.5):
        for name, pol in [("v2g=0", fixed_v2g(0.0)), ("v2g=0.5", fixed_v2g(0.5)),
                          ("v2g=1", fixed_v2g(1.0)), ("ADAPTIVE", adaptive_v2g())]:
            m = eval_policy(pol, ff)
            print(f"{ff:5.2f} {name:>10} {m['RC']:6.3f} {m['ESC']:6.3f} {m['hosp_avail']:6.3f} "
                  f"{m['reg_evac']:8.3f} {m['missions_failed']:9.1f}")
        print()
    print("Expect: ADAPTIVE keeps hospital up AND readiness high across feeds; fixed levels fail at one end.")
