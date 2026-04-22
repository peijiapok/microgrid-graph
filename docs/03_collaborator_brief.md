# 03 — Collaborator Brief

> You are reading this because you are the graph-learning expert on this project. This doc is the single artifact you need to go from zero-context to productive. Reading time: 20 minutes. Budget 2 hours for the onboarding task at the end.

## 1. What this project is, in 30 seconds

A learned graph policy for constrained sequential resource allocation on electrical distribution feeders. The empirical half of the pipeline works (code in `src/sg_resilience/`). The graph-learning half is underdeveloped: adjacency is symmetric and unweighted, there are no learned GNN baselines, and there is no topology generalization evaluation. That is why you are here.

The target is NeurIPS. We will not rush a weak submission — we are aiming for workshop 2026, main 2027, and iterating until the paper is the best we can make it.

## 2. What exists today

- `src/sg_resilience/` — Python package, 16 modules, 1590 LOC.
- `controller_model.py` — 3-layer GraphSAGE (32 hidden), sparsemax, iterative water-filling projection.
- `training.py` — 7-term loss: weighted surplus + floor + critical floor + critical activity + switching + continuity-drop + SmoothL1 imitation of an optimization teacher. Lexicographic worst-case checkpoint selection.
- `baseline_opt.py`, `baseline_rule.py` — non-learned references.
- `benchmark_loader.py`, `scenario_schema.py`, `scenario_generator.py` — SimBench + pandapower harness.
- `physics_eval.py` — post hoc pandapower convergence audit.
- `results/legacy_zips/` — 40+ prior experiment artifacts (checkpoints + metrics + figures). Treat as regression fixtures, not as ground truth.
- `paper/neurips_activityloss_paper.tex` — the **old** draft. It is an application paper and will be rewritten under the new framing. Read it for context, not for direction.

## 3. The problem with the current state

Three things a NeurIPS reviewer will say within 30 seconds of reading the current draft:

1. "The graph is decorative. You use a symmetric unweighted adjacency; edge features are ignored; an MLP baseline is not reported. Why is this a graph paper?"
2. "You train and test on the same feeder. There is no graph-generalization claim, which is the one thing that would justify the graph framing."
3. "Your 'worst-case-aware selection' is just picking the best checkpoint. There is no theory."

Your job is to make those three sentences impossible to say about the revised paper.

## 4. What you own

The five contributions, in `02_research_goal.md`. Summary of ownership:

| # | Contribution | You | Jia |
|---|---|---|---|
| C1 | Flow-aware message passing | **Design + implement** | Integrate + evaluate |
| C2 | Topology-generalizing policy | **Design protocol + splits** | Run experiments |
| C3 | Differentiable feasibility projection | **Theory + reference impl** | Production impl |
| C4 | CVaR-minimax + regret bound | **Lead** | Co-author |
| C5 | GNN baseline zoo | Specify | **Implement + run** |

You also own the related-work chapter on GNNs, graph generalization, and safe graph learning. I own the resilience side.

## 5. First-month milestones

**Week 1 (this week).**
- Read `docs/01_problem_statement.md`, `docs/02_research_goal.md`, `docs/05_experimental_protocol.md` in that order.
- Clone the repo. Run the smoke test (see `docs/09_repro_and_env.md`).
- Open an issue titled "C1 design proposal" with a 1-page sketch of the flow-aware operator you propose.

**Week 2.**
- Implement C1 as a drop-in replacement for `GraphSAGEBlock`. Gate: pass the existing smoke metrics within a small tolerance.
- Open "C2 topology-split proposal": which feeders go in `G_train`, which in `G_ood`, why.

**Week 3.**
- C1 end-to-end result on at least case33bw and rural1.
- First MLP and scrambled-edge ablations — we need to know by end of week 3 whether the graph is actually load-bearing.

**Week 4.**
- First cross-topology result. This is the make-or-break moment. If transfer is uniformly catastrophic we go back and reframe before investing further.
- Draft Theorem 1 (transfer bound) statement; proof can follow in weeks 5–6.

Each milestone ends with a short written note in `notes/` and a decision in the weekly sync.

## 6. How we collaborate

- **Repo.** https://github.com/peijiapok/microgrid-graph. Branch per contribution (`c1-flow-aware-mp`, `c2-topology-splits`, etc.). PR review required before merge to `main`.
- **Sync.** One 60-minute call per week. Start with "what moved, what blocked, what next." Written update in `notes/weekly/YYYY-MM-DD.md` before the call.
- **Decisions.** Any decision that changes `docs/01` or `docs/02` goes into `notes/adr/NNNN-title.md` with rationale and alternatives considered. Short is fine — one page — but written.
- **Authorship.** To be settled explicitly before first submission. Default expectation: you are lead on the ML/theory, Jia is lead on resilience/empirics, authorship order reflects who drove the final writing.
- **Pace.** "Work until it's the best paper" — not "ship something in 3 weeks." If a milestone slips because an experiment forced a rethink, that is good news and is what the schedule exists to surface.

## 7. Onboarding task (2 hours, do before first sync)

1. Read docs 01, 02, 05 (45 min total).
2. Clone + run the smoke in `docs/09_repro_and_env.md` (30 min).
3. In `notes/` create `onboarding_<your-handle>.md` with:
   - Three questions about the problem statement where the text is ambiguous or you disagree.
   - Your 1-paragraph sketch for C1 — what operator, why, which published operator it adapts.
   - One risk you see that is not already flagged in `02_research_goal.md` Section 3.

This note is the first artifact we discuss in the first sync. Do not skip it.

## 8. Reading list before week 2

The related-work map (`docs/07_related_work_map.md`) has the buckets. At minimum, before week 2, skim the anchor papers in:
- Flow-aware / physics-aware GNNs for power systems.
- OOD graph generalization (size-generalization, structural extrapolation).
- Differentiable optimization / convex projection layers.
- CVaR / distributionally robust learning.

You will fill in the missing BibTeX as you go — the doc has TODO markers for this.
