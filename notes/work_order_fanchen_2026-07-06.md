# Work order — Fanchen (CORRECTED to match his expertise). 2026-07-06

> **Supersedes `notes/work_order_fanchen_2026-06-16.md`**, which wrongly framed Fanchen around a GNN control operator (C1). After reading his ICDM 2025 paper *"Edge Probability Graph Models Beyond Edge Independency"* (random graph models / graph generation / structural graph mining, Bu & Shin, KAIST), his role is repositioned onto the **structure** side, which is his actual specialty. The GNN-control experiments Jia ran are a **separate, preliminary control-side probe** and are NOT a verdict on this work.

## 1. What Fanchen actually brings (from his research)
His toolkit: random graph models (ER / Chung-Lu / SBM / Kronecker), **edge-dependency "binding"** generation, **closed-form subgraph/motif density tractability**, fitting graph statistics (degree, clustering, motifs, community), scalable controlled graph generation. This is **graph structure / graph-mining**, not GNN control. His natural contributions here are C6 (`d_struct`) + C2 (structural atlas) + a new generation angle — NOT C1.

## 2. His three tasks (structure side)

### F1 — Feeder structural atlas + `d_struct` (C6/C2, his signature)
Compute graph-mining-native **structural fingerprints** for every feeder in the atlas skeleton `configs/feeder_atlas.yaml` (currently empty): degree-distribution moments, spectral (algebraic connectivity, spectral gap, leading Laplacian eigenvalues), shape (diameter, avg path length), motif/graphlet counts up to k=4, community (modularity, community-size entropy). Then define a computable **`d_struct(G, G')`** over these fingerprints.
- **Caveat (radial feeders are TREES):** distribution feeders are radial → no cycles → clustering/triangles ≈ 0. So the *clustering-specific* part of your binding paper doesn't transfer directly; the relevant structural axes for trees are **degree/branching distribution, depth/height, subtree-size distribution, path-length distribution, and load/attribute placement**. Your fingerprint + tractability machinery adapts to these; picking the right tree-native features is your call.
- Signature claim to test: does `d_struct(G_train, G_ood)` **predict the transfer gap** (F3)?
- Done-when: `d_struct` computable in <1s/feeder; atlas populated for all 6 real feeders; per-feeder fingerprints reported.

### F2 — RGM-based synthetic radial-feeder generation (NEW — highest-leverage)
The project's biggest limitation: **only 6 real feeders**; v2 needs ≥20 train-test topology pairs for any structural-transfer claim. Use your random-graph-model / generation expertise to **generate synthetic radial distribution feeders with controlled structural statistics** (matching real feeders' degree/depth/subtree distributions), expanding the topology family to dozens/hundreds and letting us **sweep structural properties** systematically.
- Adaptation needed: trees, not high-clustering graphs → tree-native RGMs (random recursive / preferential-attachment / configuration-model trees) with controlled fingerprints; attach loads + line capacities `F_e` realistically. Your tractability approach gives controllable statistics.
- Done-when: a generator producing valid radial feeders (connected tree, realistic degree/depth, per-edge `F_e`, load placement) with tunable structural fingerprints; ≥20 generated feeders spanning the atlas.

### F3 — Does structure predict transfer? (the core scientific question, joint)
With F1+F2, test whether **structural distance predicts controller transfer difficulty** across many (real + synthetic) topology pairs. This is the graph contribution — independent of whether the *control policy* uses a GNN. Note Jia's preliminary control-side finding (§4) suggests the control decision is topology-light under a shared allocator; so an honest hypothesis is *"transfer difficulty is/ isn't predictable from structure"* — either result is publishable and is yours to establish.

## 3. What Jia provides (the control/eval side — ready)
Everything to run controllers and measure transfer on any feeder (real or your synthetic ones):
- 3-line graph load: `_FeederCtx({...}, horizon)` → `.tree` (edges, `edge_capacity_mw` = F_e, `subtree_loads`), `.adjacency()` (sparse real topology), `.node_order`, criticals.
- Oracle-calibrated eval + continuity metric `C` + transfer gap (`eval_harness`, `metrics_v1`, ADR-0003).
- Feed your synthetic feeders in via the same `_FeederCtx` path (loader accepts pandapower/simbench nets; a synthetic-net adapter is a small Jia-side add).

## 4. On C1 (GNN operator) and C3 (projection) — deprioritized / not yours
- **C3 (differentiable projection):** Jia's side (reference impl exists; differentiable version optional, not blocking).
- **C1 (flow-aware GNN control operator):** optional and **not your specialty** — leave it to Jia/joint if the control-policy direction is pursued at all. Jia's preliminary probe (a quick GraphSAGE) found the control decision topology-light; that is a control-side result, orthogonal to F1–F3.

## 5. Open questions for the sync
- Which tree-native structural features best characterize radial feeders for `d_struct`?
- Can binding/EPGM ideas be adapted to controlled radial-tree generation, or do we use tree-specific RGMs?
- Given the control decision may be topology-light, is the right claim "structure predicts *transfer difficulty*" (for any controller), rather than "graph control needs topology"?
- Repo: github.com/peijiapok/microgrid-graph, branch `reframe-continuity-flow`. Atlas skeleton: `configs/feeder_atlas.yaml`.
