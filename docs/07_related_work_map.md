# 07 — Related Work Map

> Last revised 2026-04-22. Buckets + anchor paper names + TODO-markers for full citations. The collaborator fills BibTeX as papers are read; do not invent references. Every `[TODO verify]` is a citation whose exact details must be confirmed before the draft is circulated.

## 1. Six buckets, with the claim each supports

| # | Bucket | Supports |
|---|---|---|
| 1 | Resilient distribution grid operation | Problem framing, classical baselines |
| 2 | Graph / multi-agent RL for grid control | Prior-art for learned graph control |
| 3 | Flow-aware / physics-aware GNNs | C1 (flow-aware message passing) |
| 4 | OOD / topology generalization in GNNs | C2 (topology transfer) |
| 5 | Differentiable optimization / projection layers | C3 (differentiable feasibility) |
| 6 | Safe / distributionally-robust learning | C4 (CVaR-minimax + regret bound) |

## 2. Bucket-by-bucket anchors

### Bucket 1 — Resilient distribution grid operation

Anchors carried forward from `notes/legacy/paper_comparison_execution_plan.md` (verify all):
- **Gao et al.**, IEEE Trans. Smart Grid, 2016 — critical-load restoration via microgrid formation. `[TODO verify]`
- **Aghajan et al.**, 2024 — priority-aware restoration under stress. `[TODO verify]`
- **Lorca et al.**, Operations Research, 2016 — robust/stochastic control under uncertainty. `[TODO verify]`
- **Jacob et al.**, Nature Communications, 2024 — outage-aware graph control. `[TODO verify]`
- **Hassouna et al.**, 2024 — graph-RL survey for distribution operation. `[TODO verify]`

Use: (i) establish that continuity / critical service preservation is the standard operational objective; (ii) cite as prior baselines to contrast against our topology-transfer claim.

### Bucket 2 — Graph / multi-agent RL for grid control

Areas to locate representative papers in:
- Grid2Op / L2RPN benchmark series and leading submissions.
- Graph-structured RL for Volt-Var control, topology control, voltage management.
- Multi-agent RL on distribution grids.

`[TODO locate]` — collaborator adds canonical references during weeks 1–2.

### Bucket 3 — Flow-aware / physics-aware GNNs for power systems

Areas to locate:
- Power-flow-embedded GNNs (recent ICML/NeurIPS workshop papers).
- Physics-informed GNNs for AC-OPF surrogate modeling.
- Edge-feature-aware message passing (MPNN, directional message passing).
- DimeNet-style directional / geometric GNNs — adapted to directed flow.

`[TODO locate]` — critical for the C1 methods section.

### Bucket 4 — OOD / topology generalization in GNNs

Key conceptual anchors:
- **Xu et al.**, "How Powerful are Graph Neural Networks?" (ICLR 2019) — WL framework, expressivity. `[TODO verify]`
- **Yehudai et al.**, "On Size Generalization in Graph Neural Networks" — size-generalization foundations. `[TODO verify]`
- Structural extrapolation and size-transfer results in GNN literature.
- Domain adaptation on graphs.

`[TODO locate]` — these anchor the theory.

### Bucket 5 — Differentiable optimization / projection layers

- **Amos & Kolter**, "OptNet: Differentiable Optimization as a Layer in Neural Networks" (ICML 2017). `[TODO verify]`
- **Agrawal et al.**, CVXPY Layers / cone-program differentiation. `[TODO verify]`
- **Martins & Astudillo**, sparsemax (already used in our code). `[TODO verify]`
- Water-filling projections in wireless and power systems (classical references). `[TODO locate]`

Use: position our projection as a simpler, closed-form alternative to OptNet for the specific constrained-allocation polytope.

### Bucket 6 — Safe / distributionally-robust learning

- **Rockafellar & Uryasev**, "Optimization of conditional value-at-risk" (J. Risk, 2000). `[TODO verify]`
- **Shapiro, Dentcheva, Ruszczyński**, "Lectures on Stochastic Programming" — CVaR minimax reference. `[TODO verify]`
- **Namkoong & Duchi**, distributionally robust learning. `[TODO verify]`
- **Achiam et al.**, Constrained Policy Optimization. `[TODO verify]`
- **Paternain et al.**, primal-dual safe RL. `[TODO verify]`

Use: cite for Theorem 3 machinery and for the safety-adjacent framing.

## 3. What we do NOT cite

- Unrelated large-language-model work.
- Single-feeder restoration studies with no ML content — they are context, not related work in an ML paper.
- Papers with no public artifact and no reproducible experiments, unless central to theory.

## 4. Citation discipline

- Every citation in the draft must have a BibTeX entry in `paper/refs.bib` with DOI or ArXiv ID. No verbal references.
- Numbers quoted from a cited paper must link to the table/figure and be reproducible from that paper's artifact if available.
- Prior-paper comparison numbers used as contextual references only, never as direct head-to-head, unless we re-run under identical conditions. (Rule inherited from `notes/legacy/public_benchmark_comparison_note.md`.)

## 5. Work to do on this map

Before week 3:
- Fill in `[TODO verify]` with exact citation details.
- Fill in `[TODO locate]` with anchors (≥ 3 per bucket).
- For each bucket, write one sentence ("what this bucket says; how we differ") to use verbatim in the paper's related work section.

This section of the paper is typically the single biggest signal of reviewer confidence. Do not under-invest in it.
