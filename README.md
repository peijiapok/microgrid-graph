# microgrid-graph

**Topology-generalizing graph control for catastrophic grid events**, instantiated on electrical distribution feeders. Targeted at NeurIPS.

## One-paragraph pitch

When a distribution grid is hit by a catastrophic event — a hurricane, a wildfire, a cascading outage, an extreme-heat overload — supply collapses below demand and a controller must allocate the scarce power that remains so critical infrastructure (hospitals, data centers, communications, cold-chain pharmacy) stays on. The easy version of this problem — "allocate well on a feeder you have trained on" — is already handled by priority heuristics and classical optimization. The hard version, and the one this project addresses, is acting on **feeders the controller has never seen**, with formal guarantees. During a real crisis you cannot retrain a model on the specific feeder that just failed; either the policy generalizes zero-shot across topologies, or it is useless when it matters. We build (i) a graph-structured policy that generalizes zero-shot across feeder topologies, (ii) a graph-mining-native structural-similarity metric that predicts how well it will transfer before you train, and (iii) a bound on the worst-case continuity gap in terms of GNN expressivity, structural coverage, and adversary budget. The combined story is the paper.

## Status

- Empirical pipeline (GraphSAGE + sparsemax + projected allocation, multi-term continuity-aware loss) working on pandapower + SimBench feeders. Code in `src/sg_resilience/`.
- A reframe is in progress: the current paper (see `paper/neurips_activityloss_paper.tex`) reads as an application paper; the new direction targets a genuine graph-learning contribution.
- Active collaboration with a graph-learning expert. See `docs/03_collaborator_brief.md`.

## Layout

```
microgrid-graph/
├── README.md                   # this file
├── docs/                       # authoritative project docs (read in order 01 → 09)
├── paper/                      # LaTeX draft and compiled PDF
├── src/sg_resilience/          # Python package (controller, training, baselines, evaluation)
├── configs/                    # experiment configs (topology splits, hyperparameters)
├── tests/                      # unit + integration tests (empty today; populate as C1–C5 land)
├── results/                    # run outputs (ignored by git; legacy zips in results/legacy_zips/)
└── notes/                      # working notes, ADRs, legacy MD notes under notes/legacy/
```

## How to read this repo

1. `docs/01_problem_statement.md` — the formal problem.
2. `docs/02_research_goal.md` — the five contributions and what we will and will not claim.
3. `docs/03_collaborator_brief.md` — onboarding for the graph-learning collaborator.
4. `docs/05_experimental_protocol.md` — how we test the claims.
5. The rest of `docs/` — theory sketch, related work, timeline, reproducibility.

If you are the collaborator: start at `docs/03`.

## Collaboration

This is a two-person research project. See `docs/04_division_of_labor.md` for ownership, review policy, and decision escalation.

## License

TBD. Pick before any external code or data release.

## Citation

Not published yet. Do not cite.
