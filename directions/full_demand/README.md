# Direction: Full demand (buildings + EV + AI datacenter)

> Fork of the microgrid-graph line serving the **full heterogeneous load mix**,
> with the **AI datacenter** as the dominant critical load. Shares the core
> framework (`src/sg_resilience/`); this folder holds the heterogeneous load
> models, configs, and experiments.

## 1. Scope & framing
Scarce power on a radial feeder must be allocated across **heterogeneous loads —
AI datacenter(s), campus buildings, and EV chargers** — under budget + branch-flow
(`Δ_grid`) constraints, generalizing zero-shot across topologies. The resilience
regime (supply collapse / crisis) forces a **priority tradeoff**: keep the
datacenter (and other critical infrastructure) continuously served while EVs and
non-critical buildings are shed.

## 2. Load model (what specializes the core)
- **Nodes** = mixed load types on feeder buses:
  - **AI datacenter** — large, high-priority (`c_i=1`, top `w_i`), **bursty**:
    base load + GPU-training spikes (10–50 MW, ~30 s ramp) — reuse the spike
    dynamics from the carbon-paper work (Azure/Alibaba traces). Must stay
    continuously served; `m_i` high.
  - **Campus buildings** — medium priority, daily profiles (labs/dorms/cafeterias),
    partially sheddable.
  - **EV chargers** — low priority, deferrable (from the EV-only direction's model).
- **Priority ordering** typically datacenter ≫ critical buildings > buildings > EV.
- **Continuity `C`** = uninterrupted service to the datacenter + critical loads;
  the metric weights by `w_i` so the datacenter dominates.
- **Outages / crisis** = feeder-segment loss + supply shortfall (Markov core).

## 3. Research question
> Under crisis-level scarcity and branch-flow limits, can a topology-generalizing
> graph policy keep the **AI datacenter + critical loads continuously served**
> while gracefully shedding EVs/non-critical buildings — and does the feeder
> topology matter for *which* loads can be kept continuous?

## 4. Reuse from existing work (Jia's projects)
- `unified-microgrid-research` / carbon paper — datacenter spike dynamics, SMR/
  renewable generation, building profiles, the "shape drives stress" insight.
- Core `metrics_v1` continuity — directly measures datacenter continuity.
- EV model from the `ev_only` direction (shared).

## 5. Distinctive vs EV-only
**Heterogeneous priorities and the datacenter-vs-everything tradeoff.** This is
where branch-flow topology is *most likely* to matter: a datacenter deep behind a
capacity-limited branch competing with clustered building/EV load is exactly the
"who can be kept continuous" decision where structure could bite (cf. the
decision-room diagnostic — congestion + heterogeneity is where room appears).

## 6. Target / connection
The datacenter-centric unified-microgrid story (Nature-Energy/Joule-adjacent for
the systems framing; NeurIPS-adjacent for the graph-generalization framing).
Strongest tie to the project's original catastrophe/critical-infrastructure motivation.

## 7. Next steps
- [ ] `hetero_loads.py` — datacenter (spike) + building + EV demand generators → `d_{t,i}`, `w_i`, `c_i`, `m_i`.
- [ ] Configs placing a datacenter + buildings + chargers on feeder buses (per split feeder).
- [ ] Adapter into `_FeederCtx`; run core eval/metric pipeline.
- [ ] Test whether datacenter-continuity transfer depends on where it sits in the tree (topology × heterogeneity — the case most likely to show a real topology effect).
