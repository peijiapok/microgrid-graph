# Research Progress

> Living snapshot of the project's current state. Updated every sync. Historic per-week snapshots go under `notes/weekly/`. If this file disagrees with `docs/`, fix the docs — they are authoritative; this file is the changelog.

**Last updated.** 2026-04-23
**Days since kickoff.** 1 (kickoff 2026-04-22)
**Current gate.** G-A — verify graph is load-bearing (target 2026-05-end)
**Status.** Infrastructure complete; collaborator onboarding pending; first real experiment = C1 drop-in + MLP ablation, due end of May 2026.

---

## Completed since kickoff

### Repo & infrastructure

- [x] Repository created at https://github.com/peijiapok/microgrid-graph (SSH authentication working).
- [x] Directory restructured: `paper/`, `docs/`, `src/`, `configs/`, `tests/`, `notes/`, `results/`.
- [x] Legacy experiment zips (~80 MB, 40+ archives) parked under `results/legacy_zips/` and gitignored.
- [x] `src/src/` duplicate and `__MACOSX/` artifacts removed from a prior bad zip extraction.
- [x] `pyproject.toml` at root with src-layout; `pip install -e .` works from a fresh clone.
- [x] Smoke test at `src/sg_resilience/smoke.py`; verified passing locally in a few seconds on CPU.
  - case33bw, 32 nodes, 4 time steps, 5 scenarios, 10 epochs
  - `hidden_dim=16`, `num_layers=2`, no teacher imitation (CPU-cheap)
  - Exact feasibility asserts: `0 ≤ a_i ≤ d_i`, `Σa ≤ P`, per step
  - Representative output captured in `docs/09_repro_and_env.md`

### Docs (authoritative)

All nine under `docs/`:

- [x] `01_problem_statement.md` — catastrophic grid events as motivation; topology-generalizing constrained allocation as ML kernel.
- [x] `02_research_goal.md` — six contributions C1–C6, explicit non-goals, hierarchical success criteria.
- [x] `03_collaborator_brief.md` — onboarding brief assuming zero context; week-1 to week-4 milestones; onboarding task.
- [x] `04_division_of_labor.md` — ownership table, review policy, decision-escalation protocol.
- [x] `05_experimental_protocol.md` — topology splits, atlas, ablations, baselines, compute budget.
- [x] `06_theory_sketch.md` — Theorems 1–3 statements, proof directions, known obstacles, non-vacuity gate.
- [x] `07_related_work_map.md` — six buckets with `[TODO verify]` markers for BibTeX to be filled by collaborator.
- [x] `08_timeline.md` — six decision gates G-A through G-F; multi-milestone plan through 2027.
- [x] `09_repro_and_env.md` — setup, smoke contract, seeds, artifact provenance.

### Configs

- [x] `configs/smoke_case33bw.yaml` — smoke benchmark, consumed by `python -m sg_resilience.smoke`.
- [x] `configs/topology_split_v1.yaml` — first frozen split. G_train = {case33bw, rural1, rural2}; G_ood = {lv_urban, ieee_34, ieee_123}. Validation pending (GitHub Issue #3).
- [x] `configs/feeder_atlas.yaml` — v0 skeleton; owner = collaborator (C6); feature schema documented inline; fields marked `null` to be populated.

### External

- [x] `PROPOSAL.md` — formal research proposal for external review.
- [x] `PROGRESS.md` (this file) — living status snapshot.

### Commits on `main`

| Commit | Purpose |
|---|---|
| `cf4b90f` | Initial restructure; docs 01–09; src; legacy archive |
| `2a6439e` | Expand C2 with structural atlas; add C6 topology-similarity metric |
| `3456072` | Rebalance framing: catastrophic grid events explicit in openings |
| `02ae70a` | Smoke test, configs, pyproject |
| *(this commit)* | Add `PROPOSAL.md` and `PROGRESS.md` |

---

## In flight (this week)

- [ ] Send collaborator the repo link and onboarding message (owner: Jia).
- [ ] Open three GitHub issues (owner: Jia):
  - [ ] `[C1] Design proposal: flow-aware message-passing operator` (assign collaborator).
  - [ ] `[Infra] Smoke test + CI gate` (assign Jia).
  - [ ] `[C2] Topology split v1 — rationale and invariant checks` (assign Jia).
- [ ] Write GitHub Actions workflow `.github/workflows/smoke.yml` so PRs cannot merge on a red smoke.
- [ ] Draft `notes/adr/0001-authorship.md` (authorship agreement) before the first substantive sync.

---

## Next 1–2 weeks (toward G-A, 2026-05-end)

**By 2026-05-06:**
- Collaborator onboarded; onboarding note filed at `notes/onboarding_<handle>.md`.
- First weekly sync completed; note in `notes/weekly/`.
- Collaborator has opened "C1 design proposal" issue with their one-page sketch.
- Smoke CI green on every PR.

**By 2026-05-13:**
- C1 drop-in reference implementation in a branch; smoke still green under the new operator.
- First MLP ablation run — does the graph matter? This is G-A's primary question.
- `configs/feeder_atlas.yaml` v0 populated for the three G_train feeders (degree, spectral, shape features).

**By 2026-05-30 (G-A):**
- MLP and scrambled-edge ablations both complete.
- Decision at G-A: go / iterate / reframe.
- First cross-topology result (even if weak) on at least one G_ood feeder. Catastrophic transfer failure triggers reframe; modest success unlocks G-B.

---

## Open questions / blockers

- **IEEE 123 loader.** `configs/topology_split_v1.yaml` references `ieee_123`; pandapower may not provide it directly and OpenDSS conversion may be required. Fallback: drop from v1 and add a synthetic radial feeder of similar size. Owner: Jia, Issue #3, due 2026-05-end.
- **Authorship.** To be settled explicitly in `notes/adr/0001-authorship.md` before first submission.
- **GitHub Actions budget.** Free-tier CPU runners are sufficient for smoke (seconds). If C2/C5 training jobs grow, a self-hosted runner or a budgeted cloud runner is needed — decide at G-B.
- **Compute provisioning.** Workshop target: ~100 GPU-hours. Provider not yet selected; decide by end of May.

---

## Upcoming decision gates

| Gate | Target | Question | If no |
|---|---|---|---|
| G-A | 2026-05-end | Is the graph load-bearing (MLP ablation fails on ≥1 metric)? | Reframe the ML contribution |
| G-B | 2026-06-end | Does cross-topology transfer work at all on v1 split? | Re-scope C2; consider narrower claims |
| G-C | 2026-07-end | Does `d_struct` predict transfer with R² > 0.5 on ≥1 metric? Workshop draft coherent? | C6 becomes a negative result; headline shifts |
| G-D | 2026-12-end | Is Theorem 1 non-vacuous on empirical data with the measured `L_Π`? | Drop theory to weaker empirical form |
| G-E | 2027-03-end | Is the main-conf draft at the quality bar? | Delay to ICML 2027 or NeurIPS 2028 |
| G-F | 2027-05 | Submit. | — |

---

## Risk register (current)

| Risk | Status | Note |
|---|---|---|
| Transfer is catastrophic across feeders | Unknown — tested at G-B | Week-4 milestone exists specifically to surface this early |
| `d_struct` fails to predict transfer (R² < 0.5) | Unknown — tested at G-C | Paper's signature claim depends on this; negative result is still reportable |
| Theorem 1 not non-vacuous | Unknown — tested at G-D | Mitigation: Theorem 2+3 sufficient for viability |
| Collaborator bandwidth unpredictable | Green at kickoff | Zero-blocker to start; monthly milestone checks |
| Compute bottleneck on large OOD feeders | Yellow | Not yet exercised; provisioning decision pending |
| Code quality issues surface under new ablations | Yellow | Budget 1 week of refactor in 2026-05 if needed |

---

## Pointers

- Proposal: `PROPOSAL.md`
- Problem: `docs/01_problem_statement.md`
- Goal: `docs/02_research_goal.md`
- Collaborator brief: `docs/03_collaborator_brief.md`
- Timeline: `docs/08_timeline.md`
- Repository: https://github.com/peijiapok/microgrid-graph
