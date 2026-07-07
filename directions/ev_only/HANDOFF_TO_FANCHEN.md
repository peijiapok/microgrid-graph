# Handoff — EV-only direction, where Fanchen's structural work plugs in

> 2026-07-08. Jia has built the full control/empirics side on a single real feeder
> (IEEE-33) with damage variants. The next steps genuinely need Fanchen's
> **structural graph mining** (feeder generation + characterization). This is the
> boundary.

## What's DONE (Jia side, committed, tested)
- **Core framework** (`src/sg_resilience/`): flow-constrained `Δ_grid` allocation, continuity metric, oracle-calibrated eval, topology (tree + `F_e`).
- **Mission-aware EV fleet + readiness + native V2G** (`ev_fleet.py`, 6 tests): emergency readiness as an energy-state process; fleet readiness = # emergency vehicles idle & above dispatch threshold; V2G via signed power.
- **Damaged-feeder generator + feeder integration + episode loop** (`damage.py`, `ev_scenario.py`): remove IEEE-33 lines → surviving topology; place hospitals + emergency/regular chargers; allocate scarce power; run fleet; measure readiness + hospital + missions-failed.
- **HEADLINE result** (`ev_v2g.py`): the **V2G-vs-readiness interior optimum** — coordinated V2G at ~0.25–0.5 keeps both hospital and emergency fleet alive; either extreme is catastrophic. This is the paper's spine and stands on ONE feeder.

## The boundary — why we need Fanchen now
Everything above is on **IEEE-33 + its damage variants only**. Two claims the paper needs cannot be made from one feeder, and both are Fanchen's expertise:

### F-EV-1 — Generate a diverse family of damaged/synthetic radial feeders (his RGM signature)
We need **many structurally-varied radial distribution feeders** (and their post-disaster damaged islands) to show:
- the **V2G interior-optimum + readiness findings generalize** across the topology space (not an IEEE-33 artifact);
- a **zero-shot topology-generalization** study is possible (train a controller on some damaged topologies, evaluate on unseen ones — the "after a quake you don't know which feeder survived" claim).
Adapt his generation machinery to **radial trees with realistic structure** (degree/branching, depth, subtree sizes) + attach loads and line capacities `F_e`. Target: ≥20 feeders spanning a structural range, each with a damage-scenario distribution.
- **Interface:** produce pandapower radial nets (or an equivalent the loader/`_FeederCtx` accepts); Jia adds a thin net→`_FeederCtx` adapter so the existing fleet/eval pipeline runs unchanged.

### F-EV-2 — Structural characterization: does structure predict the readiness/V2G behavior? (his `d_struct`)
Characterize each (damaged) feeder with graph-mining fingerprints (degree/spectral/depth/subtree/community — tree-native, since feeders have clustering≈0), define `d_struct`, and test whether **structural distance predicts** how the readiness outcome or the V2G-optimum shifts across topologies. This is his signature "predict behavior from structure alone," and it's the natural way to explain **when** topology matters here.

## Honest open question he should know
Jia's control-side probes (main line + this EV first result) found the **control decision is often topology-light** once a shared allocator handles feasibility — the flow-aware vs blind comparison was confounded by allocation *strategy* (concentrate vs spread). So: **whether real feeder topology is causal for the control is still open.** Fanchen's F-EV-1 (scale) + F-EV-2 (structure) are exactly what's needed to settle it — either structure predicts the readiness/V2G behavior (a positive result), or it doesn't (an honest negative, still publishable). Don't assume topology is load-bearing; test it with his tools.

## What NOT to do yet
- Don't build a GNN control operator (control decision may be topology-light; and it's not his specialty).
- Don't pull real EMS/EV/outage data yet (deferred; simulation-first).

## Pointers
- Direction plan + headline: `directions/ev_only/README.md`
- Run the headline: `PYTHONPATH=src:directions/ev_only python directions/ev_only/ev_v2g.py`
- Load a feeder's graph in 3 lines: `_FeederCtx` (see `docs/12`); `topology.build_radial_tree`.
- Repo: github.com/peijiapok/microgrid-graph, branch `reframe-continuity-flow`.
