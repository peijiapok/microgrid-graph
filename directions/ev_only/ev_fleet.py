"""Mission-aware EV fleet with V2G, for the resilient emergency-EV direction.

Models emergency-response readiness as an ENERGY-STATE process (not a static
load): vehicles charge from allocated charger power, get dispatched on stochastic
missions that drain SOC, and return needing recharge. Fleet-level readiness =
number of emergency vehicles idle AND above their dispatch-readiness SOC.

V2G is native: per-node power is SIGNED — positive charges vehicles, negative
discharges them (V2G support). Discharge preferentially drains regular/high-SOC
vehicles to preserve emergency readiness — but the controller can still drain an
ambulance, which is exactly the readiness-vs-V2G tradeoff we study.

Units: power MW, energy MWh, SOC in [0,1] of per-vehicle capacity, dt in hours.
This module is fleet dynamics only; the feeder / Delta_grid allocation that
produces `node_power` lives in the core and is wired in separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FleetConfig:
    dt_h: float = 0.25                 # 15-min step
    max_rate_mw: float = 0.05          # per-vehicle charge/discharge cap (50 kW)
    emergency_theta: float = 0.6       # emergency vehicle "ready" if SOC >= theta
    mission_rate_per_h: float = 2.0    # Poisson intensity of emergency missions (fleet-wide)
    mission_duration_steps: int = 4    # ~1 h out
    mission_drain_frac: float = 0.35   # SOC drained by one mission
    v2g_soc_floor: float = 0.05        # never discharge below this


@dataclass
class EVFleet:
    node_of_vehicle: np.ndarray        # (M,) feeder node id each vehicle charges at
    is_emergency: np.ndarray           # (M,) bool
    capacity_mwh: np.ndarray           # (M,)
    soc: np.ndarray                    # (M,) in [0,1]
    cfg: FleetConfig = field(default_factory=FleetConfig)
    on_mission: np.ndarray = field(default=None)  # (M,) remaining mission steps; 0 = idle

    def __post_init__(self):
        self.soc = np.clip(np.asarray(self.soc, float), 0, 1)
        self.is_emergency = np.asarray(self.is_emergency, bool)
        self.capacity_mwh = np.asarray(self.capacity_mwh, float)
        self.node_of_vehicle = np.asarray(self.node_of_vehicle)
        if self.on_mission is None:
            self.on_mission = np.zeros(len(self.soc), dtype=int)
        self.theta = np.where(self.is_emergency, self.cfg.emergency_theta, 0.0)

    # --- state queries ---
    def readiness(self) -> int:
        """# emergency vehicles idle and dispatch-ready."""
        return int(np.sum(self.is_emergency & (self.on_mission == 0) & (self.soc >= self.theta)))

    def idle(self) -> np.ndarray:
        return self.on_mission == 0

    # --- one 15-min step ---
    def step(self, node_power: dict, rng: np.random.Generator) -> dict:
        """Advance one step. node_power[node] = signed MW (+charge / -V2G discharge).
        Returns per-step readiness, missions_failed, and SOC summaries."""
        dt = self.cfg.dt_h
        rate_e = self.cfg.max_rate_mw * dt                       # per-vehicle energy cap (MWh)
        for node, p in node_power.items():
            vids = np.where((self.node_of_vehicle == node) & self.idle())[0]
            if len(vids) == 0:
                continue
            budget = abs(float(p)) * dt                          # MWh available this step
            if p >= 0:   # CHARGE: emergency first, then lowest SOC
                order = sorted(vids, key=lambda i: (not self.is_emergency[i], self.soc[i]))
                for i in order:
                    room = (1.0 - self.soc[i]) * self.capacity_mwh[i]
                    e = min(rate_e, budget, room)
                    if e <= 0:
                        continue
                    self.soc[i] += e / self.capacity_mwh[i]
                    budget -= e
                    if budget <= 1e-12:
                        break
            else:        # V2G DISCHARGE: regular first, then highest SOC (preserve readiness)
                order = sorted(vids, key=lambda i: (self.is_emergency[i], -self.soc[i]))
                for i in order:
                    avail = (self.soc[i] - self.cfg.v2g_soc_floor) * self.capacity_mwh[i]
                    e = min(rate_e, budget, max(avail, 0.0))
                    if e <= 0:
                        continue
                    self.soc[i] -= e / self.capacity_mwh[i]
                    budget -= e
                    if budget <= 1e-12:
                        break

        # emergency missions (readiness-gated dispatch)
        missions_failed = 0
        n_new = rng.poisson(self.cfg.mission_rate_per_h * dt)
        for _ in range(int(n_new)):
            cand = np.where(self.is_emergency & (self.on_mission == 0) & (self.soc >= self.theta))[0]
            if len(cand) == 0:
                missions_failed += 1
                continue
            i = cand[np.argmax(self.soc[cand])]                  # send the fullest ready vehicle
            self.on_mission[i] = self.cfg.mission_duration_steps

        # mission progress (drain + tick down)
        active = self.on_mission > 0
        drain_per_step = self.cfg.mission_drain_frac / max(self.cfg.mission_duration_steps, 1)
        self.soc[active] = np.clip(self.soc[active] - drain_per_step, 0, 1)
        self.on_mission[active] -= 1

        return {
            "readiness": self.readiness(),
            "missions_failed": missions_failed,
            "mean_soc": float(np.mean(self.soc)),
            "min_emergency_soc": float(np.min(self.soc[self.is_emergency])) if self.is_emergency.any() else 1.0,
        }


def readiness_continuity(readiness_series: np.ndarray, n_min: int) -> float:
    """Fraction of the horizon the emergency fleet stayed at/above the readiness floor."""
    r = np.asarray(readiness_series)
    return float(np.mean(r >= n_min)) if r.size else 0.0
