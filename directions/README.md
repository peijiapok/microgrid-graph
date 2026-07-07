# Directions — two load-scope forks of the microgrid-graph line

> Created 2026-07-07. Both directions **share the same core framework** in
> `src/sg_resilience/` (radial-feeder topology, flow-constrained `Δ_grid`
> allocation, capacity-normalized continuity metric `C`, oracle-calibrated
> evaluation ADR-0003, Markov outages). They differ only in the **load model /
> application scope**. Direction-specific load models, scenario configs, and
> experiments live in each subfolder; the shared methodology is imported, not
> duplicated.

| Direction | Folder | Loads served | Anchor question |
|---|---|---|---|
| **EV-only** | `directions/ev_only/` | EV charging points only | Fair, continuous EV charging allocation under feeder flow constraints + scarcity, generalizing across topologies |
| **Full demand** | `directions/full_demand/` | Buildings + EV + **AI datacenter** | Continuity-preserving allocation across heterogeneous priorities (datacenter ≫ buildings > EV) under crisis/flow limits |

## What is shared vs forked
- **Shared (core, `src/sg_resilience/`):** `Δ_grid` feasible set, `metrics_v1` continuity `C`, `eval_harness` oracle calibration, `topology` (tree + `F_e`), `outages`, reference projection + greedy allocator, control baselines.
- **Forked (per direction):** node/load semantics, demand generation, priority/criticality assignment, scenario configs, and the application framing + target venue.

## Cross-cutting (applies to both, unchanged)
- **Control-side H1 caveat** (`notes/h1_negative_finding.md`): the *control decision* is topology-light under a shared allocator — holds for both directions; it's about the policy, not the load model.
- **Fanchen's structural side** (`notes/work_order_fanchen_2026-07-06.md`): `d_struct` + feeder-structure characterization + RGM-based feeder generation apply to both directions (structure of the feeder, independent of what loads sit on it).

## Why fork instead of one model
Keeps each paper's scope crisp: the EV-only direction connects to the EV-charging/equity line and is a self-contained resilience-charging story; the full-demand direction is the datacenter-centric unified-microgrid story where the datacenter's criticality/burstiness drives the priority tradeoffs. Same methodology, two defensible papers.
