# H1 experiment plan — is graph structure load-bearing? (GPT-5.5-backed)

> 2026-06-16. The first learned-policy milestone. Tooling: torch only (no torch_geometric, no cvxpylayers).

## Claim under test
Among **learned** policies, true feeder topology is load-bearing for transferable critical-load prioritization under binding radial flow constraints. Decisive contrasts (same features/allocator/loss/checkpoint-selection):
- **real-graph GNN > DeepSets** (no-graph) on OOD `C`, by ≥ `Delta_graph`=0.05;
- **real-graph GNN > degree-preserving-rewire GNN** (wrong-graph) by ≥ `Delta_rewire`=0.05.
NOT "GNN beats the greedy rule" — the rule is a near-oracle ceiling, not the comparison.

## Training (tractable, torch-only)
Train by **imitation/ranking against the flow-aware greedy rule's allocation**: policy outputs per-node scores; loss = regression+ranking of scores vs the rule's per-node allocation share (the rule encodes flow info via cap-limited amounts). No differentiable `Delta_grid` needed.
- Checkpoint selection by **validation hard `C`** (not soft loss).
- Anneal/keep flow-awareness pressure via the imitation target (the target already respects flow caps).

## Evaluation (hard, identical for all policies)
Score-driven **flow-aware greedy** allocator onto `Delta_grid` (order by policy score, fill respecting budget+box+outage+flow). Feasibility enforced identically for every method → isolates *decision* quality. Metric: capacity-normalized continuity `C` (metrics_v1), oracle-calibrated matched scarcity (ADR-0003), train + OOD feeders, paired seeds.

## Baselines (all share features/allocator/loss)
| Method | Isolates |
|---|---|
| Flow-aware greedy RULE | near-oracle ceiling (not learned) |
| Feature-only greedy (no flow lookahead) | fair non-graph heuristic |
| DeepSets (no-graph) | are node features alone enough? |
| Degree-preserving rewire GNN | real topology vs generic message passing |
| GraphSAGE (primary), GCN, GAT, GIN | learned graph operators |

## Fairness (separate policy-info from feasibility-info)
All methods use the hard flow-aware allocator at eval (feasibility, not decision). Only the GNN perceives true topology for *decisions* via message passing. The rule is labeled a handcrafted near-oracle ceiling. Claim = graph-necessity **among learned policies**.

## Expected shape (the core tension)
On easy scarcity all methods converge (rule already ~0.97–0.99). The graph gap should appear as flow constraints bind harder and criticals cluster on shared bottlenecks. If methods don't separate in the calibrated band, push to harder flow-binding regimes / clustered critical placement — separation-vs-difficulty is itself a reportable curve.

## Size-generalization
Fully shared, permutation-equivariant; NO node/feeder IDs or positional encodings. Per-feeder/per-step normalization (demands by budget, caps by total demand, depth by max depth). Optional pooled global-context vector broadcast to nodes. Train on ≤118-load feeders, eval zero-shot on 242-load.

## Build order
1. score-driven greedy allocator (order by policy score) — eval any policy.
2. `policies.py`: feature builder + DeepSets + GraphSAGE/GCN/GAT/GIN score heads (shared).
3. `h1_train.py`: imitation training vs flow-aware greedy + hard-C checkpointing + eval.
4. Pilot: DeepSets vs GraphSAGE on the calibrated split → first H1 signal; expand if it separates.
