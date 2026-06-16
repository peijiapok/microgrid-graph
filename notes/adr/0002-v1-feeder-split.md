# ADR-0002 — v1 feeder split (frozen, validation-grounded)

> Status: **PROPOSED** (Jia-side; ready to freeze pending one Fanchen-coupled check noted in §5). Date 2026-06-16. Resolves blocker #1 in `docs/11`. Supersedes placeholder IDs in `configs/topology_split_v1.yaml`. Validation script: `scripts/validate_feeders_v1.py` (re-runnable).

## 1. Context
The old split used placeholder feeder IDs (`simbench_rural1`, `simbench_lv_urban`) and an unvalidated assumption that a pure-LV split is credible. `docs/11` flagged this as the #1 blocker. ADR-0001 (continuity primary → flow constraints) adds a new requirement: **every feeder must be radial**, because the branch-flow constraint `|f_e(a)|≤F_e` uses subtree sums `f_e(a)`, which are only well-defined on a tree.

## 2. Validation findings (2026-06-16, `scripts/validate_feeders_v1.py`)
All candidates loaded via `simbench.get_simbench_net` / `pandapower.networks`. Counts are buses / loads.

| Feeder | Source | Bus | Load | Total load (MW) | Radial (honoring switches)? | Line cap `max_i_ka`? | Load profiles? |
|---|---|---:|---:|---:|---|---|---|
| case33bw | pandapower | 33 | 32 | 3.72 | **Yes** (5 tie-lines are `in_service=False`) | 100% | No (synthetic) |
| 1-LV-rural1--0-sw | SimBench | 15 | 13 | 0.08 | Yes | 100% | Yes |
| 1-LV-rural2--0-sw | SimBench | 97 | 99 | 0.20 | Yes | 100% | Yes |
| 1-LV-rural3--0-sw | SimBench | 129 | 118 | 0.33 | Yes | 100% | Yes |
| 1-LV-urban6--0-sw | SimBench | 59 | 111 | 0.44 | Yes | 100% | Yes |
| 1-MVLV-urban-5.303-0-sw | SimBench | 254 | 242 | 49.7 | **Yes** (15 open line-switches) | 100% | Yes |
| 1-MVLV-urban-6.305-0-sw | SimBench | 202 | 249 | 49.7 | Yes (open switches) | 100% | Yes |
| 1-MVLV-urban-6.309-0-sw | SimBench | 202 | 249 | 49.7 | Yes (open switches) | 100% | Yes |

**Three findings that drive the decision:**
1. **All feeders are operationally radial** once `in_service` (case33bw) and open switches (MVLV-urban) are honored → subtree-sum flow constraints are well-defined on all of them.
2. **Line thermal capacity (`max_i_ka`) is present on 100% of lines everywhere** → `F_e` for the flow constraints is directly available (transformers also carry `sn_mva`; MVLV feeders have 3 trafos each).
3. **The only feeders larger than the train set are the MVLV-urban ones** (242–249 loads vs rural3's 118). Since they radialize cleanly, they are valid size-gen OOD feeders.

## 3. Decision — frozen v1 split

**G_train (in-family):**
| Feeder | Loads | Class (radiality, feeder_type) |
|---|---:|---|
| case33bw | 32 | (radial, semiurban) |
| 1-LV-rural2--0-sw | 99 | (radial, rural) |
| 1-LV-rural3--0-sw | 118 | (radial, rural) |

**G_ood (held-out topology):**
| Feeder | Loads | Class | Satisfies |
|---|---:|---|---|
| 1-LV-urban6--0-sw | 111 | (radial, urban) | feeder_type-gen (urban absent from train) |
| 1-MVLV-urban-5.303-0-sw | 242 | (radial, urban/MV) | **size-gen** (242 > 118) + feeder_type-gen |
| 1-MVLV-urban-6.305-0-sw | 249 | (radial, urban/MV) | size-gen + feeder_type-gen (3rd OOD for robustness) |

**Split-rule check (docs/05 §1 "Split Rules"):**
1. No feeder in both sets ✓
2. ≥1 OOD with |V| > max train |V| (118): MVLV-urban 242, 249 ✓
3. ≥1 OOD with feeder_type absent from train: urban (train has only rural+semiurban) ✓
4. Class labels recorded for every feeder ✓ (this ADR + the yaml)

**Rejected:** `1-LV-rural1--0-sw` (13 loads — too small for claim-bearing training, per `docs/11`).

## 4. Required follow-up (Jia-side, non-Fanchen)
1. **Fix `benchmark_loader._build_network_graph`** to honor `line.in_service` and open switches; otherwise the adjacency, static features, AND flow constraints are built on the wrong (meshed) graph. This is a correctness bug exposed by the validation. Add a radiality assertion (`|E| = |V|-1` on the active graph) at load time.
2. Extend the loader to expose **edge capacity `F_e`** (`line.max_i_ka` → MW proxy via nominal voltage; `trafo.sn_mva` for transformer edges) and the **rooted-tree structure** (orient edges from the `ext_grid` slack) so subtree sums `f_e(a)` are computable.
3. Add the **workload audit** (docs/05 §9): the MVLV-urban feeders are ~150× the rural load scale; confirm normalized demand-scale and budget-ratio distributions are comparable, else C2 is reframed as mixed topology/scale shift for those feeders.

## 5. One Fanchen-coupled check before final freeze
The size-gen OOD feeders are **MV+LV** (transformers + two voltage levels), while the train feeders are pure-LV. For the flow constraints this means edge capacities mix line ampacity and transformer `sn_mva`. Confirm with Fanchen that the C3 flow-projection treats transformer edges as ordinary capacitated tree edges (expected yes) — if it needs per-voltage-level handling, the OOD slot may need a pure-LV alternative (but none larger than rural3 exists, so size-gen would be lost). Flagging, not blocking.

## 5b. case33bw line ratings — resolved (2026-06-16)
`topology.build_radial_tree` now accepts `line_capacity_mw` / `trafo_capacity_mw` overrides. case33bw is assigned **6.0 MVA per line** (≈274 A at 12.66 kV; `configs/topology_split_v1.yaml::line_capacity_mw_override`) — an explicit **assumption**, since the canonical Baran–Wu 33-bus case has no thermal data and pandapower's `max_i_ka` yields a ~2e6 MW placeholder. At the 3.715 MW feeder peak this cap is **slack**, so case33bw is the small radial **budget-binding** train feeder; the SimBench feeders (real, binding caps) carry the **flow-binding** topology evidence. Revisit the value if a stronger case33bw flow story is wanted.

## 6. Deferred / unchanged
- **IEEE 123 / IEEE 34** OpenDSS conversion (old blocker #3) is **no longer on the v1 critical path** — the MVLV-urban feeders give a validated, larger, radial OOD without conversion risk. Keep IEEE feeders as a v2 expansion for the structural-association pair count (C6).
