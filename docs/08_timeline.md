# 08 — Timeline

> Last revised 2026-04-22. Multi-milestone plan. The posture is "work until this is the best paper we can make," not "ship by the next deadline." Deadlines are decision gates, not panic points.

## 1. Horizon and targets

- **Primary target.** NeurIPS 2027 main conference (submission ≈ May 2027).
- **Interim target.** NeurIPS 2026 workshop (submission window July–August 2026).
- **Fallback target.** ICML 2027 (if NeurIPS 2027 misses) or a journal (IEEE TPS, JMLR) with extended content.

Each target is a **gate**, not a commitment. We decide at each gate whether we go or whether we iterate.

## 2. Monthly plan

### 2026-04 (now)
- Repo live on GitHub, docs 01–09 circulated.
- Collaborator onboarded; onboarding note filed.
- First sync completed with shared reading list.

**Exit condition.** Collaborator's 1-page C1 proposal in an issue.

### 2026-05
- **C1 reference implementation** lands in a branch; passes smoke tests.
- **MLP and scrambled-edge ablations** complete — we know whether the graph is load-bearing by end of month.
- **Topology-split v1** frozen in `configs/topology_split_v1.yaml`.

**Exit condition.** At least one cross-topology result exists, even if weak. If transfer is uniformly catastrophic, we reframe in June instead of continuing.

### 2026-06
- **C2 topology-transfer** full sweep on v1 split.
- **C5 baseline zoo** implemented (GCN, GAT, GIN, EGNN, Graph Transformer).
- **Main result table** populated for the workshop draft.

**Exit condition.** Draft tables for main result + transfer gap. Workshop go/no-go decision.

### 2026-07
- **C3 differentiable projection** implemented; feasibility invariant at 0 violations.
- **C4 CVaR-minimax training** implemented; comparison against lexicographic selection.
- **Theory sketches** for Theorems 1 and 2 complete in prose (proofs still rough).
- **Workshop paper drafting** begins from the reframed problem statement.

**Exit condition.** Workshop draft circulated internally for review by end of month.

### 2026-08
- **Workshop submission.**
- **First proof of Theorem 2 (feasibility)** completed.
- **First non-vacuity check of Theorem 1 bound** on empirical data.

### 2026-09 to 2026-12 — consolidation
- Reviewer feedback from workshop incorporated.
- **Theorem 1 proof** formalized.
- **Theorem 3 (CVaR regret)** proof drafted.
- Ablations hardened; new OOD families added (IEEE 123, additional synthetic families).
- Scale-up to multi-seed, multi-size protocol.
- Writing of the main-conference draft begins.

**Decision gate (end of 2026-12).** Is Theorem 1 non-vacuous on our empirical setup? If not, re-scope the theory in January.

### 2027-01 to 2027-03 — polish
- Final ablation suite.
- Physics audit across all reported runs.
- Related-work section fully cited (no `[TODO]` markers remaining).
- Internal review round with a senior advisor.
- Rebuttal-ready supplementary materials prepared in advance.

### 2027-04 — submission runway
- Final camera-quality figures.
- All tables regenerated from frozen run artifacts.
- Public artifact release candidate prepared (Zenodo record).

### 2027-05 — NeurIPS 2027 submission.

### 2027-06 to 2027-09 — review cycle
- Rebuttal during author-response window.
- Camera-ready if accepted; otherwise iterate to next venue.

## 3. Decision gates (explicit)

Every gate has a go / iterate / re-scope decision. "Iterate" is not "more of the same" — it must specify what changes.

| Gate | Date (approx) | Question |
|---|---|---|
| G-A | 2026-05-end | Is the graph load-bearing (MLP ablation fails)? |
| G-B | 2026-06-end | Does any cross-topology transfer work at all? |
| G-C | 2026-07-end | Is the workshop draft telling a coherent story? |
| G-D | 2026-12-end | Is Theorem 1 non-vacuous on data? |
| G-E | 2027-03-end | Is the main-conf draft at the quality bar? |
| G-F | 2027-05 | Submit. |

A `no` at G-A or G-B triggers reframing. A `no` at G-D triggers theory re-scope (possibly dropping Theorem 1 or moving to a weaker form). A `no` at G-E delays submission to ICML 2027 or NeurIPS 2028 — we do not submit papers we don't believe in.

## 4. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Transfer is uniformly catastrophic | Medium | Re-scope to C1+C3+C4 narrative |
| Theory proves intractable | Low-medium | Drop Theorem 3, keep 1+2 |
| Compute bottleneck on large OOD feeders | Medium | Start OOD runs in 2026-06, not later |
| Existing code quality issues surface under new ablations | Medium | Budget 1 week of refactor in 2026-05 |
| Collaborator bandwidth unpredictable | Always | Monthly check on milestones; re-scope rather than slip |

## 5. Non-negotiables

- No submission without all five invariants from `01_problem_statement.md` Section 5 satisfied.
- No claim in the paper without a corresponding row in `05_experimental_protocol.md` and a regenerated table artifact.
- No citation without a verified BibTeX entry.
- No silent scope change. Write the ADR first.
