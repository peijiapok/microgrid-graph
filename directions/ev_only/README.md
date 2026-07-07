# Direction: EV-only

> Fork of the microgrid-graph line where the served loads are **EV charging
> points only**. Shares the core framework (`src/sg_resilience/`); this folder
> holds EV-specific load models, configs, and experiments.

## 1. Scope & framing
Scarce power on a radial distribution feeder must be allocated across **EV
chargers**, step by step, under budget + branch-flow (`Δ_grid`) constraints, and
the policy must generalize zero-shot to unseen feeder topologies. The resilience
regime (crisis / grid-constrained charging) is when charging demand exceeds
available power and the controller must decide **who charges, how much, and
without interrupting** priority vehicles.

## 2. Load model (what specializes the core)
- **Nodes** = EV chargers on feeder buses. `n` varies across feeders.
- **Demand `d_{t,i}`** = per-charger power request, driven by connected-EV state:
  SOC deficit, dwell time, target departure SOC. Time-varying via arrivals/
  departures (activity-based schedules) — reuse from `ev-charging-rl` / NHTS.
- **Priority / criticality `w_i, c_i, m_i`** = which EVs must keep charging
  (e.g., low-SOC, essential/fleet vehicles) and/or **equity weights** (income/
  housing-context, from EV-equity PCFI). `m_i` = minimum charging rate to count
  as "served."
- **Continuity `C`** = uninterrupted charging service to priority EVs (a vehicle
  that charges → stops → charges is worse than a steady session).
- **Outages** = charger/feeder-segment outages (Markov, core `outages.py`).

## 3. Research question
> Can one graph policy allocate scarce feeder power across EV chargers to keep
> priority/equity-critical charging **continuous**, generalize to unseen feeders,
> and respect branch-flow limits — better than topology-blind allocation?

## 4. Reuse from existing EV work (Jia's projects)
- `ev-charging-rl` — Green Handshake signal, activity-based EV mobility, IEEE 33/118-bus.
- `thriftydeath` — behavioral EV charging (deferability, loss aversion) → realistic demand deferral.
- `EV equity` — PCFI equity metric → equity-weighted priorities `w_i`.

## 5. Distinctive vs full-demand
Homogeneous, deferrable, equity-laden loads; the interesting tradeoff is
**equity + continuity under scarcity**, not heterogeneous-priority conflict. EVs
are flexible (deferable/interruptible in principle) — so "continuity" is a
*service-quality* choice, not a hard must-serve like a datacenter.

## 6. Target / connection
Fits the EV-charging/equity venue line (e.g., resilient-charging / e-mobility).
The graph angle (topology generalization) + equity + continuity is the novelty
hook; honest control-side caveat (topology-light decision) applies.

## 7. Next steps
- [ ] `ev_loads.py` — EV charger demand model (SOC/arrival-driven) producing `d_{t,i}`, `m_i`, `w_i` per feeder.
- [ ] EV scenario configs (which feeder buses host chargers; charger counts).
- [ ] Adapter into `_FeederCtx` so the core eval/metric pipeline runs unchanged.
- [ ] Equity-weighted priority + continuity baseline result.
