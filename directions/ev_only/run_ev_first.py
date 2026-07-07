"""First EV-only result on IEEE-33 (case33bw): readiness under damaged feeders,
and the de-risking check — does topology (branch-flow) matter for readiness?

Run: PYTHONPATH=src:directions/ev_only python directions/ev_only/run_ev_first.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandapower.networks as ppn

from damage import damage_feeder
from ev_scenario import assign_roles, build_fleet, run_episode
from ev_fleet import FleetConfig

warnings.simplefilter("ignore")
T, SCARCITY = 96, 0.5      # 24 h, 50%-of-demand budget (degraded grid)


def one(seed, n_remove, flow_aware):
    rng = np.random.default_rng(seed)
    net = ppn.case33bw()
    tree, surviving = damage_feeder(net, n_remove=n_remove, rng=rng)
    if len(surviving) < 6:
        return None
    roles = assign_roles(surviving, rng)
    fleet = build_fleet(roles, rng, cfg=FleetConfig(mission_rate_per_h=3.0))
    return run_episode(tree, surviving, roles, fleet, T, SCARCITY, rng, flow_aware=flow_aware)


def main():
    print("EV-only first result — IEEE-33, 24 h, scarce (50% budget), emergency missions")
    print(f"{'damage(lines removed)':>22} {'#surv':>6} {'readiness_cont':>14} {'hosp_cont':>10} {'missions_failed':>15}")
    for n_remove in (0, 3, 6, 9):
        rows = [one(s, n_remove, flow_aware=True) for s in range(8)]
        rows = [r for r in rows if r]
        if not rows:
            continue
        rc = np.mean([r["readiness_continuity"] for r in rows])
        hc = np.mean([r["hospital_continuity"] for r in rows])
        mf = np.mean([r["missions_failed"] for r in rows])
        ns = np.mean([r["n_surviving"] for r in rows])
        print(f"{n_remove:22d} {ns:6.1f} {rc:14.3f} {hc:10.3f} {mf:15.1f}")

    print("\nDE-RISKING: does topology (branch-flow) matter for readiness under damage?")
    print(f"{'damage':>8} {'flow_aware_RC':>14} {'topo_blind_RC':>14} {'Δ(flow-blind)':>14}")
    for n_remove in (3, 6, 9):
        fa = [one(s, n_remove, True) for s in range(8)]
        tb = [one(s, n_remove, False) for s in range(8)]
        fa = [r["readiness_continuity"] for r in fa if r]
        tb = [r["readiness_continuity"] for r in tb if r]
        n = min(len(fa), len(tb))
        if n == 0:
            continue
        d = np.mean(fa[:n]) - np.mean(tb[:n])
        print(f"{n_remove:8d} {np.mean(fa[:n]):14.3f} {np.mean(tb[:n]):14.3f} {d:14.3f}")
    print("\n(Δ≈0 => branch-flow/topology doesn't bite for readiness at these caps; "
          "Δ>0 => flow-aware allocation keeps more emergency readiness under damage.)")


if __name__ == "__main__":
    main()
