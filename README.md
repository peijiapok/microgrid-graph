# microgrid-graph

A research codebase for **topology-generalizing graph control of constrained sequential resource allocation**, instantiated on electrical distribution-feeder resilience. Targeted at NeurIPS.

## One-paragraph pitch

A learned graph policy allocates a scarce power budget across feeder loads at each time step, preserving critical service under stress. The ML question is not "how do we allocate well?" — priority heuristics and classical optimization already do that. The ML question is: *can one graph-structured policy, trained on a family of feeder topologies with a physics-aware operator and a distributionally-robust objective, generalize zero-shot to unseen topologies — and can we bound the worst-case continuity gap?* The answer and the theory around it are the paper.

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
