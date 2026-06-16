# Spec — capacity-normalized windowed continuity (primary metric, ADR-0001)

> Status: spec for `src/sg_resilience/metrics_v1.py`. Date 2026-06-16. Implements the primary metric from ADR-0001. Notation follows `docs/05 §2`. Jia-side (post-hoc eval metric); the only Fanchen-coupled part is the oracle feasible set (§4), abstracted behind a callable.

## 0. Why a new metric
`critical_load_adequacy` (`mean_{t,i} 1[a≥m·d]`) is pointwise and gameable by flicker. ADR-0001 makes **temporal continuity** primary. Naive `mean_i max_run(served)/T` is also gameable (serve one easy node forever). This spec defines a **capacity-normalized, priority-weighted, windowed** continuity score that rewards keeping *as much critical load as the budget allows* continuously served, plus anti-gaming diagnostics.

## 1. Inputs (from a rollout)
Per feeder-scenario rollout, arrays over horizon `T` and critical node set `C` (`|C|=k`):
- `a[t,i]` allocation, `d[t,i]` demand, `m[i]` min-service fraction, `w[i]` service weight (priority-weighted critical demand), `O[t,i]∈{0,1}` outage mask. Source: `Scenario` (`scenario_schema.py`) + policy rollout. `w_i, m_i, c_i` come from `LoadNode` / `configs/feeder_attributes_v1.yaml`.

## 2. Served indicator (outage-conditioned)
```
served[t,i] = 1[ a[t,i] >= m[i] * d[t,i] ]          for i in C
```
**Outage conditioning (primary):** drop timesteps where `O[t,i]=1` from node i's sequence before any run/window computation — we measure *policy* behavior, not exogenous outages. Report an **outage-inclusive** variant separately (outaged steps count as not-served), per `docs/05 §2`.

## 3. Windowed continuity (numerator)
Window-length set `𝓛` (config; default `{2,4,8,16}` capped at `T`, plus report the full curve). A critical node `i` is *continuously served over a length-L window ending at t* iff `served[τ,i]=1` for all `τ∈(t−L+1..t)` (on its outage-conditioned sequence). Per-node sustained fraction:
```
cont[i,L] = (# length-L windows i is continuously served) / (# length-L windows)
```
Policy weighted continuity at scale L:  `num_L = Σ_{i∈C} w[i] · cont[i,L]`.

## 4. Capacity normalization (denominator = oracle frontier B*)
Normalize by what an **offline, full-hindsight** allocator could have sustained under the *same* per-step feasible set `Δ(G,s_t)` (global-budget v0) or `Δ_grid(G,s_t)` (flow-constrained, Path A):
```
den_L = B*_L = Σ_{i∈C} w[i] · cont_oracle[i,L]
C_L   = num_L / max(den_L, eps)            # in [0,1] when oracle is true optimum
C     = mean_{L∈𝓛} C_L                     # PRIMARY metric; also report per-L curve
```
**Oracle options (pick canonical + fast fallback):**
- **Canonical (exact-ish):** offline solve maximizing `Σ_{i,L} w_i cont[i,L]` s.t. `a_t∈Δ(·)` ∀t — i.e. the offline version of the training objective with full future knowledge. Reuse the existing continuity-aware LP/QP (`baseline_opt.py`) given the whole horizon. Guarantees `C_L≤1`.
- **Fast fallback:** greedy priority-ordered feasible oracle — sort `C` by `w_i` desc; commit each node to continuous min-service across the horizon if per-step feasible given already-committed higher-priority nodes (honoring budget + flow caps). Feasible ⇒ lower-bound oracle ⇒ `C_L` may exceed 1 (flag/cap; report fraction capped).
- **Abstraction:** `metrics_v1` takes a `feasible_set` / `oracle` callable so it is identical for v0 (global budget) and Path-A (flow-constrained). This is the only place that touches Fanchen's C3 feasible-set definition — keep it injected, not hardcoded.

## 5. Anti-gaming diagnostics (required alongside C)
- **critical coverage:** `mean_{i∈C} 1[cont[i,L*]>0]` at a canonical `L*` — did we keep *many* critical nodes up, not one?
- **weighted starvation:** `(Σ_{i: never continuously served} w_i) / Σ_{i∈C} w_i`.
- **worst-served critical node:** `min_{i∈C} cont[i,L*]` and Gini over `cont[·,L*]`.
A high `C` with low coverage / high starvation = "favorite-node" gaming → must be visible.

## 6. Retained / secondary metrics
- `critical_load_adequacy` — kept as a **guard** (so the policy can't trade all adequacy for continuity); reported, non-primary.
- `temporal_continuity` (`mean_i max_run/T`) — keep for backward-compat/reporting, NOT primary (gameable).
- `switching_count`, `weighted_served_energy`, `unserved_critical_demand`, `feasibility_violations` — unchanged.
- `critical_continuity_ratio` (legacy in `priority_metrics.py`) → maps to `critical_load_adequacy` in reports (already noted in `docs/05 §2`).

## 7. Transfer gap (higher-is-better; docs/05 §2 convention)
```
transfer_gap[C]      = mu_train(C) − mu_ood(C)
per_feeder_gap[C,Gj] = mu_train(C) − mean_seed mu(seed, Gj, C)
```
`Delta_transfer` and `Delta_graph` thresholds must be **re-pinned on C** (currently on adequacy in `docs/05 §2.5`) — a pre-run ADR, since C is now the H1/H2 primary. That ADR is part of the docs/05 rewrite after Fanchen sync.

## 8. Training surrogate (cross-ref ADR-0001 §4 — separate from this metric)
The metric above uses **hard** thresholds. Training uses the **soft** surrogate (do not conflate, per `docs/10 §9`):
```
s[t,i]   = σ(β (a[t,i] − m[i] d[t,i]))
q[t,i,L] = Π_{τ=t−L+1..t} s[τ,i]      # = exp(Σ log s) for stability
loss_cont = − Σ_{L,t} Σ_{i∈C} w[i] q[t,i,L] / normalizer
```
Calibrated relaxation of §3; → exact as β→∞. Replaces `λ_s‖a_t−a_{t-1}‖₁` as the temporal driver (`λ_s` kept small as anti-chatter only).

## 9. Tests (`tests/test_metrics_v1.py`)
1. **Discrimination:** sequences `1,0,1,0` and `1,1,0,0` give equal adequacy (0.5) but different continuity (the doc-10 example). 
2. **Anti-gaming:** "one node served forever" vs "spread evenly" — same/aggregate `C` distinguished by coverage + starvation diagnostics.
3. **Oracle sanity:** policy = oracle ⇒ `C_L = 1`; random feasible policy ⇒ `C_L < 1`.
4. **Outage-conditioning:** outaged steps removed do not count as interruptions; outage-inclusive variant does count them.
5. **Determinism / vectorization:** numpy path matches a reference loop within `1e-9`.

## 10. Build order (what's buildable now without Fanchen)
- §2,3,5,6,7 (served, windowed continuity numerator, coverage/fairness, transfer gap) — **fully Jia-side, build now** with the global-budget oracle as default `feasible_set`.
- §4 flow-constrained oracle and §8 wiring through the projection — **wait for the C3 feasible-set finalization** (Fanchen), since `Δ_grid` and the differentiable projection are his. Inject via callable so the rest doesn't block.
