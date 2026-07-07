# Direction: EV-only — Resilient emergency-EV charging under disaster scarcity

> Fork of the microgrid-graph line. Shares the core framework
> (`src/sg_resilience/`: `Δ_grid` allocation, continuity metric, oracle-calibrated
> eval, topology). Problem sharpened with GPT-5.5 (2026-07-07).

## 0. Decisions locked (2026-07-08)
- **Network:** IEEE test feeders (13 / 33 / 34 / 123; case33bw already in pipeline). Damaged variants (remove lines → islands) = the topology family; Fanchen's RGM generation expands it. **Simulation study first; real EMS/EV/outage data deferred.**
- **Action (1a):** per-charger power allocation; vehicles pre-assigned to chargers (no routing).
- **Dispatch (2):** only *ready* vehicles dispatch; if none ready, emergency mission **fails** (counted). Trained proxy = **readiness continuity** (continuous power to keep chargers/vehicles served); report **missions-failed** as the outcome.
- **Objective (3):** lexicographic — hospital service ≫ emergency-EV readiness ≫ regular-EV fairness.
- **Time (4):** 15-min steps; primary 24 h (96 steps), robustness 72 h (288 steps).
- **Supply (5):** degraded grid import + renewables (solar/wind) + storage + **V2G** as the core. **SMR = named add-on, deferred.**
- **Readiness (6):** count-based (`R(t)=#ready emergency vehicles`); small microgrid → spatial coverage deferred.
- **Fleet (8):** all EVs modeled; ambulance/emergency readiness is the priority focus in the emergency.
- **Scope (10):** standalone paper.
- **Venue (9):** aim **Nature Energy** on the *insight* (electrified emergency-response blackout vulnerability + the **V2G-vs-readiness tradeoff**); fallback ladder Joule / Nature Comms / Applied Energy / IEEE TSG. IEEE feeders are the testbed, not the NE grounding — data layer added later.

**Core novel element:** the **V2G-vs-readiness tradeoff** — draining an ambulance's battery (V2G) to keep a hospital powered *now* costs its dispatch *readiness* later.

## 1. Story & setting
A near-future, fully-electrified world. A disaster (earthquake, storm) damages
the distribution grid: the surviving feeder is **partially destroyed / islanded
with scarce power**. We must simultaneously (a) keep **hospitals** and other
fixed critical loads powered, and (b) **charge EVs** — above all **emergency EVs
(ambulances, fire, police, now electric)** that must stay *operationally ready to
respond* — plus regular EVs for evacuation / essential mobility. You cannot
retrain on the specific feeder that survived the quake, so the controller must
**generalize zero-shot to unseen, damaged feeder topologies**.

Framing discipline: present electrified emergency fleets as a **resilience
stress-test scenario**, not speculative 2050 storytelling. The technical claim
stands on the model, not the date.

## 2. The intellectual contribution (NOT "prioritize ambulances")
"Give ambulances high weight" is trivial. The contribution is that **emergency
response capability is an energy-STATE process, not a static load**: post-disaster
feeder operation becomes a *coupled infrastructure-readiness* problem where
**mobile emergency energy demand, fixed critical-load service, and damaged-grid
feasibility interact under zero-shot topology shift**. The control problem is
created by readiness being *dynamic, fleet-level, and mission-coupled*.

## 3. Sharpest research question
> How can a controller allocate scarce power on a **damaged, islanded radial
> feeder** to jointly preserve **hospital service** and **fleet-level emergency-EV
> operational readiness**, while generalizing **zero-shot to unseen post-disaster
> topologies and damage patterns**?

## 4. Formulation (specializes the core)
- **Graph / feasibility:** radial feeder; per-step scarce-power allocation to
  nodes under budget + radial branch-flow caps (`Δ_grid`, core). **Damage =
  outages of feeder segments/chargers → the surviving topology varies per scenario.**
- **Nodes / loads:** hospitals (fixed critical), **emergency-EV chargers**,
  regular-EV chargers.
- **Mission-aware readiness (the key modeling choice):** each emergency vehicle
  has SOC, a **minimum dispatch energy** (readiness threshold θ), stochastic/
  episodic **missions** (dispatch → drain SOC → return → needs recharge), and
  return times. Charging replenishes SOC; missions drain it. Do **not** model
  this as ordinary high-weight demand.
- **Fleet-level readiness continuity (the metric):** keep enough emergency
  vehicles dispatch-ready *over time* — e.g. `R(t) = #{v : SOC_v(t) ≥ θ}` and the
  objective is continuity of `R(t) ≥ N_min` (fraction of time the fleet stays
  above the readiness floor), plus hospital-service continuity, plus fair
  regular-EV access. Fleet-level, because emergency capability is **substitutable
  across vehicles/stations** — "the ambulance system degraded" is the right
  abstraction, not "charger 7 was underserved."
- **Limited mobility (defer full routing):** a vehicle may charge at any charger
  in a **precomputed reachable, energized set** (given its location + surviving
  topology), with a travel-energy penalty. This keeps the key insight without
  turning it into a transportation/routing paper.

## 5. Where topology GENUINELY matters (and why this escapes the prior trap)
Earlier finding (`notes/h1_negative_finding.md`): with a shared feasibility
allocator, topology was non-causal for *static per-step* control. This setup is
different — **topology affects readiness *trajectories*, not just constraint
projection**:
- which chargers remain **electrically reachable / energized** under damage,
- branch **bottlenecks near hospitals** vs charger banks,
- **spatial mismatch** between where emergency vehicles are and which chargers are
  energized,
- **mobility access** to surviving nodes.
So the test is: do topology-aware policies beat topology-agnostic feasible
allocators **specifically under damaged-feeder islands, charger loss, and fleet
relocation** — i.e., where topology shapes the readiness dynamics over time. This
is the honest bar and the reason the graph angle can be load-bearing here.

## 6. Baselines & why learning (pre-empt the "just solve OPF online" attack)
Compare against **OPF/MPC and priority heuristics** that solve each damaged
topology online. Learning must justify itself: **speed** (real-time repeated
decisions during a crisis), **incomplete/uncertain models** (unknown exact
damage), and **amortization across many uncertain damage states** (a policy
trained over a damage distribution vs re-solving each). State this explicitly.

## 7. Reviewer attacks → defenses
| Attack | Defense |
|---|---|
| "Just weighted load shedding with ambulances prioritized" | Readiness is dynamic, fleet-level, mission-coupled — an energy-state process, not a static weight |
| "Unrealistic 2050 assumptions" | Framed as a resilience stress-test; claim rests on the model, not the year |
| "Online optimization solves each damaged topology" | OPF/MPC baselines + learning justified by speed / model uncertainty / amortization over damage states |
| "Topology is decorative" (the prior finding) | Topology drives readiness *trajectories* (reachability, spatial mismatch, mobility) — tested specifically under damage |

## 8. Reuse & what's new
- **Reuse (Jia):** core `Δ_grid`/`metrics`/`eval` framework; EV mobility &
  activity models from `ev-charging-rl`; behavioral/deferral from `thriftydeath`;
  equity weighting from `EV equity`.
- **Reuse (Fanchen):** feeder structure characterization / RGM-based **damaged-
  feeder topology generation** — perfect fit: generate many post-disaster feeder
  islands to train/eval zero-shot transfer (F2 in his work order).
- **New here:** mission-aware readiness dynamics, fleet-level readiness-continuity
  metric, limited-mobility charger assignment, damage-scenario generation.

## 9. Next steps
- [ ] `ev_fleet.py` — mission-aware emergency + regular EV fleet: SOC, θ, missions (Poisson dispatch, drain, return), charger assignment (reachable/energized set).
- [ ] `readiness_metric.py` — fleet readiness `R(t)` + readiness-continuity (specializes `metrics_v1`).
- [ ] Damage-scenario generator — feeder-segment/charger outages producing varied surviving topologies (with Fanchen's generation for scale).
- [ ] `_FeederCtx` adapter placing hospitals + emergency/regular chargers on buses.
- [ ] First result: readiness-continuity of a priority heuristic vs OPF vs topology-aware policy, under damaged topologies.
