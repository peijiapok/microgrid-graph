# DEV-META - DO NOT INGEST

# Microgrid Graph Research Execution Plan

Created: 2026-05-08  
Scope: execution plan for `graph_research_plan_full.md` and the current
`microgrid-graph` repository.  
Claude review verdict: `revise`.

This is a planning and audit artifact. It is not research evidence, not a
manuscript section, and not input to training, simulation, reward design, or
runtime model context.

## 1. Done Definition

The near-term research plan is done when the project has:

- a single lead framing for the paper;
- frozen early topology-transfer gates;
- falsifiable hypotheses with pre-registered failure criteria;
- a minimal experiment path that can kill or justify the graph-learning claim;
- generated-result and physics-audit requirements for every empirical claim;
- explicit cut points for workshop, full sweep, and main-conference scope.

This plan does not claim that the method works, that `d_struct` predicts
transfer, that the theory is non-vacuous, or that any empirical result exists.

Minimum useful cut point: a May-July workshop evidence gate that can decide
whether to proceed with the topology-transfer graph-learning paper or reframe.

## 2. Framing Decision

Primary framing:

> Topology-generalizing graph control for catastrophic grid events.

The critical-infrastructure and continuity-aware microgrid story remains the
motivation and testbed. The paper should lead with the graph-learning claim:
zero-shot feeder topology transfer, structural distance as a predictor of
transfer, and continuity preservation under stochastic topology degradation.

The AAAI-style critical-infrastructure narrative in
`graph_research_plan_full.md` should be treated as legacy source material unless
it is explicitly merged into the current NeurIPS/topology-transfer framing.

Acceptance:

- `PASS`: the authoritative proposal, progress notes, collaborator brief, and
  experiment protocol all describe the same lead claim.
- `FAIL`: the project presents itself sometimes as an application/reward paper
  and sometimes as a graph-learning/topology-transfer paper.
- `INVALID`: no paper-facing framing decision is needed because the project is
  paused before any collaborator or advisor review.

## 3. Independent Hypotheses And Falsifiers

### H1. Graph structure is load-bearing under topology shift.

Claim to test: graph-structured policies transfer better than non-graph or
wrong-graph ablations on held-out feeder topologies.

Falsifier:

- MLP, random-edge, or identity-adjacency ablations match or beat GraphSAGE/C1
  on `transfer_gap[critical_continuity_ratio]` under OOD feeder evaluation.

Acceptance:

- `PASS`: at least one graph model beats the MLP baseline by a pre-registered
  effect size on at least one OOD feeder, across three seeds, with paired
  uncertainty excluding zero.
- `FAIL`: graph and non-graph baselines are statistically indistinguishable
  under topology shift.
- `INVALID`: feeder loading or result generation fails before comparison.

### H2. The first useful experiment is a standalone refutation test, not the full sweep.

Claim to test: one train feeder and one OOD feeder already reveal whether graph
transfer is plausible.

Falsifier:

- A minimal `case33bw -> ieee_34` or nearest available OOD test shows MLP
  transfer gap no worse than GraphSAGE.

Acceptance:

- `PASS`: one generated JSON result compares GraphSAGE and MLP on one OOD
  feeder and one continuity metric.
- `FAIL`: the minimal test refutes the graph-transfer claim.
- `INVALID`: the selected OOD feeder cannot be loaded or audited.

### H3. `d_struct` must be fixed before it becomes a theory or prediction claim.

Claim to test: structural distance can explain or predict transfer degradation.

Falsifier:

- `d_struct` has no stable functional form, changes after seeing results, or
  fails a permutation-null check against transfer gap.

Acceptance:

- `PASS`: `d_struct` functional form is frozen before the full transfer sweep;
  R2 exceeds the 95th percentile of a label-permutation null on at least one
  metric.
- `FAIL`: distance does not track transfer better than the null.
- `INVALID`: fewer than the minimum planned OOD feeders have valid result JSON.

### H4. Microreactor detail is scenario context, not core novelty.

Claim to test: continuity-aware graph transfer remains meaningful without
making microreactor constraints the headline.

Falsifier:

- Removing microreactor-specific constraints makes continuity trivial or makes
  the method indistinguishable from ordinary restoration.

Acceptance:

- `PASS`: the core experiments can be described in feeder/topology/control
  terms without relying on microreactor-specific novelty.
- `FAIL`: the only nontrivial behavior comes from microreactor constraints.
- `INVALID`: the scenario generator does not support this comparison.

### H5. Claims must be artifact-traceable.

Claim to test: every manuscript result can be regenerated from protocol rows,
result JSON, table generators, and physics audits.

Falsifier:

- A planned paper claim lacks a corresponding protocol row, result artifact,
  table-generation path, or audit status.

Acceptance:

- `PASS`: every table row maps to `docs/05_experimental_protocol.md`, a JSON
  artifact under `results/`, and a script under `scripts/`.
- `FAIL`: any number is hand-entered or untraceable.
- `INVALID`: the project stops before paper-table generation.

## 4. Gates

### G-0: Onboarding Gate

Target: immediately / first collaborator milestone.

Oracle:

- collaborator has one merged PR or signed onboarding note;
- C1 design sketch is filed as an issue or note;
- ownership of C1, C6 atlas, and `d_struct` form is explicit.

Outcome:

- `PASS`: proceed to implementation gates.
- `FAIL`: re-scope May work to Jia-owned MLP, loader, CI, and reporting tasks.
- `INVALID`: collaborator is no longer part of the project.

### G-A.5: Feeder Loader Gate

Target: before training sweeps.

Oracle:

- every feeder in `configs/topology_split_v1.yaml` can be loaded and converted
  into the training/evaluation graph representation;
- load-only smoke test runs without training.

Outcome:

- `PASS`: topology split v1 is runnable.
- `FAIL`: substitute or remove broken feeders through an ADR before training.
- `INVALID`: topology split v1 is intentionally replaced.

### G-A: Graph Load-Bearing Gate

Target: May 2026.

Oracle:

- GraphSAGE or C1 is compared against MLP and wrong-graph ablations under at
  least one OOD feeder;
- effect size threshold `Delta_pre` is written before results are inspected.

Outcome:

- `PASS`: graph structure improves OOD continuity by `Delta_pre`.
- `FAIL`: graph structure is not load-bearing; reframe away from graph novelty.
- `INVALID`: OOD comparison cannot run.

### G-B: Cross-Topology Transfer Gate

Target: June 2026.

Oracle:

- main transfer table exists for at least GraphSAGE, MLP, and rule/optimization
  baselines;
- at least one OOD feeder preserves nontrivial critical continuity.

Outcome:

- `PASS`: continue full topology-transfer story.
- `FAIL`: if transfer is uniformly catastrophic, reframe to constrained graph
  control or negative characterization.
- `INVALID`: loader or compute failure prevents full comparison.

Partial-pass rule:

- If transfer works on small/near OOD feeders but fails on large/far feeders,
  demote prediction claims and frame the result as characterizing transfer
  limits by feeder class.

### G-C: `d_struct` Prediction Gate

Target: July 2026.

Oracle:

- `d_struct` functional form is frozen before final transfer results;
- R2 against transfer gap is reported with a permutation-null comparison.

Outcome:

- `PASS`: keep `d_struct` as a headline contribution.
- `FAIL`: demote `d_struct` to exploratory analysis or appendix.
- `INVALID`: too few valid OOD feeders for even descriptive correlation.

Open statistical issue:

- With only three OOD feeders, R2 is weak evidence. Either expand to at least
  six OOD feeders before making a prediction claim, or label G-C descriptive
  until more splits exist.

### G-D: Theory Non-Vacuity Gate

Target: December 2026.

Oracle:

- Theorem 1 bound is computed on the actual atlas using the frozen `d_struct`
  form and an explicit Lipschitz strategy.

Outcome:

- `PASS`: bound is below the trivial ceiling and supports the paper narrative.
- `FAIL`: move theory to a weaker proposition, appendix, or future work.
- `INVALID`: no stable `d_struct` form exists.

## 5. Cut Points

### CUT A: Minimum Viable Workshop Evidence

Required:

- framing resolved;
- topology split v1 runnable;
- MLP vs GraphSAGE standalone OOD refutation test;
- at least one generated transfer table;
- result JSON and basic physics audit path;
- clear go/reframe decision.

Defer:

- C3 differentiable projection;
- C4 CVaR-minimax;
- full baseline zoo;
- formal non-vacuity proof.

### CUT B: Full Sweep

Required:

- C1 reference implementation;
- C5 baseline zoo;
- all planned OOD feeders loaded;
- three seeds minimum for claim-bearing comparisons;
- ablation tables generated from JSON;
- `d_struct` correlation and permutation-null analysis.

### CUT C: Main-Conference Scope

Required:

- expanded OOD families or repeated topology splits;
- non-vacuous theory or explicitly weakened theory;
- full physics audit;
- reproducibility package;
- no hand-entered numbers;
- advisor/internal review before submission.

## 6. Parallel And Serialized Work

Serialized critical path:

1. Resolve framing and legacy document status.
2. Freeze topology split v1.
3. Verify feeder loaders.
4. Run minimal OOD refutation test.
5. Run transfer sweep.
6. Freeze `d_struct` form before interpreting correlation.
7. Check theory non-vacuity only after `d_struct` is stable.

Parallelizable now:

- MLP baseline;
- wrong-graph ablations;
- load-only smoke test for each feeder;
- report scripts against stub JSON;
- physics-audit harness;
- collaborator C1 design sketch;
- feeder atlas v0 feature computation;
- `d_struct` functional-form proposal.

Do not parallelize:

- edits to the same experiment protocol or split file;
- training runs that share mutable result destinations;
- claim wording before the relevant gate has passed;
- theory edits that depend on an unfrozen `d_struct`.

## 7. Claude Review Issues Accepted

Accepted:

- G-A must test graph load-bearing under topology shift, not only in-family
  performance.
- A loader gate must precede training sweeps.
- The small-sample R2 issue is real; a permutation null or expanded OOD set is
  needed before making a prediction claim.
- Partial transfer needs a prewritten branch instead of a binary pass/fail.
- The untracked legacy AAAI-style plan creates framing ambiguity.
- C3 and C4 should not block the minimum workshop evidence path.

Rejected or narrowed:

- Collaborator latency is a risk, but not a reason to block Jia-owned work.
  The plan routes around it by moving MLP, loader, CI, reporting, and minimal
  OOD tests forward.
- Deleting `graph_research_plan_full.md` is not required. Archiving or adding a
  header may be enough, because it contains useful motivation and slide content.

Final decision: revise the plan now, then proceed with CUT A.

## 8. Open Tiebreaks

- `d_struct`: weighted interpretable distance vs learned composite.
  Recommendation: use a weighted/interpretable form for the first theory-backed
  version; learned composite can be a later empirical variant.
- OOD count: keep three OOD feeders for workshop speed vs expand to six before
  claiming predictivity.
  Recommendation: three is acceptable for descriptive workshop evidence, not
  for a strong prediction claim.
- C3/C4: include in workshop vs defer.
  Recommendation: defer unless CUT A is already passing early.
- Legacy plan disposition: archive, header, or merge.
  Recommendation: add a clear header or move under `notes/legacy/` before
  sharing the repo externally.

## 9. Runtime Contamination Boundary

Forbidden runtime inputs:

- this file;
- Claude review text;
- Codex discussion notes;
- planning gates, thresholds, or reviewer-style critiques;
- `dev_planning/**`;
- `.codex/skills/**` or `~/.codex/skills/**`.

Allowed research evidence:

- raw feeder and scenario data;
- generated result JSON/CSV/log files;
- analysis scripts and reproducible outputs;
- committed protocol and split files;
- manuscript source, tables, figures, references, and physics-audit artifacts.

Planning artifacts may guide human decisions and offline reporting checks only.
They must not enter the controller observation, reward, scenario generator,
training context, evaluation rollouts, or any model-facing prompt.

