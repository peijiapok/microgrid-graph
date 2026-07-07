# Research Plan & Progress — Resilient Emergency-EV Charging under Disaster Scarcity

> EV-only direction of the microgrid-graph line. Last updated 2026-07-08.
> Standalone plan + living progress log. Code: `directions/ev_only/` on branch
> `reframe-continuity-flow`. Companion: `README.md` (headline results),
> `HANDOFF_TO_FANCHEN.md` (collaborator boundary).

---

## 1. One-paragraph summary
In a near-future fully-electrified world, a disaster (earthquake/storm) leaves a
distribution feeder damaged and islanded with scarce power. The operator must
simultaneously keep **hospitals** powered and **charge EVs** — above all
**emergency EVs (ambulances, fire, police)** that must stay *dispatch-ready* — plus
regular EVs for evacuation. We model emergency-response capability as an
**energy-state process** (readiness = fleet SOC above a dispatch threshold, drained
by stochastic missions), formulate scarce-power allocation on the damaged feeder,
and show that keeping hospitals *and* emergency readiness alive requires
**state-adaptive Vehicle-to-Grid (V2G) coordination**: no fixed V2G policy is robust
across grid-degradation levels, and the equity burden of V2G falls on regular EVs.

## 2. Motivation
- Emergency services are electrifying (electric ambulances/fire/police piloted in JP/KR/CN/AU).
- Disasters cause the highest-stakes, hardest grid conditions: supply collapses, the
  surviving feeder is unpredictable, and — in an electrified world — **emergency
  response itself depends on charging**. A dead ambulance battery = an unanswered emergency.
- V2G lets EVs *support* the grid, creating a genuine dilemma: draining an EV to
  power a hospital *now* costs its readiness later. This tension is unstudied at the
  fleet-readiness level.

## 3. Research questions & contributions
**RQ.** How can a controller allocate scarce power on a damaged islanded radial
feeder to jointly preserve hospital service and **fleet-level emergency-EV
operational readiness** (generalizing to unseen post-disaster topologies)?

**Contributions.**
1. **Readiness-as-energy-state formulation** — emergency capability modeled as a
   dynamic, fleet-level, mission-coupled energy process (not a static high-weight load).
2. **The V2G-vs-readiness dilemma, quantified** — an interior optimum whose location
   shifts with grid degradation, so **fixed V2G is never robust**.
3. **Adaptive V2G control** that matches/beats the hindsight-best fixed policy across
   degradation levels and parameters — a *necessity* argument for state-adaptive V2G.
4. **Equity finding** — regular-EV evacuation-readiness is the systematic casualty of
   V2G under scarcity.
5. **(Pending, Fanchen)** structural characterization + damaged-feeder generation to
   test zero-shot topology generalization.

## 4. Formulation
**System.** Radial IEEE feeder (case33bw = IEEE-33; 13/34/123 available). Damage =
remove lines → surviving source-connected topology varies per scenario. Nodes host
**hospitals** (fixed critical), **emergency-EV chargers**, **regular-EV chargers**.
Degraded grid feed = scarce budget; V2G lets EVs discharge to support loads.

**Feasibility (core `Δ_grid`).** Per-step allocation `a` with `0≤a≤d`, `Σa≤P`,
outage mask, and radial branch-flow caps `|f_e(a)|≤F_e`. Time: 15-min steps, 24 h
(96 steps); 72 h robustness.

**Fleet dynamics (`ev_fleet.py`).** Per vehicle: SOC, capacity, emergency flag,
dispatch threshold θ. Missions ~ Poisson; only a *ready* (SOC≥θ) emergency vehicle
can be dispatched, else the emergency is **unserved**; a mission drains SOC over its
duration; vehicles return needing recharge. Charging fills SOC from allocated charger
power; **V2G** = signed power (negative = discharge), round-trip efficiency **~85%**
(empirical), draining regular/high-SOC vehicles first.

**Metrics (literature-grounded — ENS/EENS, SAIDI/availability, VoLL).**
- **RC — readiness continuity** (HEADLINE; availability analogue): fraction of time
  ≥N_min emergency vehicles are dispatch-ready.
- **ESC — emergency service continuity** (OUTCOME; EENS/EUE analogue) = 1 −
  unserved/arrived missions = fraction of emergency demand served.
- **hosp_avail** = 1 − hospital Energy-Not-Served fraction.
- **reg_evac** = regular-EV evacuation service (fraction charged to evac SOC over time).
GPT-5.5 + resilience literature: keep RC as headline, validate against ESC (show they correlate).

## 5. Methods
**Controllers (all share the allocator + fleet):**
- **Fixed-V2G(level)** — serve `level × hospital` from EV batteries regardless of state.
- **Adaptive-V2G** — deliver just enough V2G to free grid budget for *this step's*
  emergency-charging need (`v2g = clip(emerg_charge_need + hosp_demand − grid_feed, 0, hosp_demand)`),
  draining regular EVs first. Covers the hospital, protects readiness, minimizes drain.
- Planned baselines: OPF/MPC (online), priority heuristic; learning justified by
  speed / model uncertainty / amortization over damage states.

## 6. Results so far (IEEE-33, damaged, 24 h)
**Disaster model behaves correctly.** As damage grows (more lines removed →
fewer surviving nodes): readiness 1.0→0.49, hospital 1.0→0.50, missions-failed
0→36. **Connectivity (which nodes survive) is the first-order driver.**

**V2G interior optimum (motivates adaptive).** Under a degraded feed, the V2G level
maximizing readiness is interior: v2g=0 → grid all to hospital, EVs never charge,
readiness collapses; v2g=1 → over-drain; v2g≈0.25–0.5 → both alive.

**Adaptive V2G dominates (ESC across feed strength):**
| feed | v2g=0 | v2g=0.5 | v2g=1 | ADAPTIVE |
|---|---|---|---|---|
| 0.6 | 0.00 | 0.00 | 0.79 | **0.86** |
| 0.8 | 0.00 | 0.93 | 0.80 | **0.94** |
| 1.0 | 0.00 | 0.97 | 0.80 | **0.97** |
| 1.5 | 0.99 | 0.97 | 0.80 | **0.99** |

**Robustness sweep (feed 0.8, ESC; margin = adaptive − hindsight-best-fixed):**
adaptive wins **14/15** settings (mission rate, drain, θ, damage, fleet size),
margin +0.01…+0.12, **growing in harder regimes**; one small exception (θ=0.5, −0.02).

**Equity finding.** V2G drains regular EVs first → regular-EV evacuation collapses
under scarcity (reg_evac ≈0.01); adaptive recovers it fastest as the feed improves
(0.80 vs 0.38 for fixed-0.5 at feed 1.5).

## 7. Progress status
| Item | Status | Where |
|---|---|---|
| Core `Δ_grid` framework, metrics, eval | ✅ done | `src/sg_resilience/` |
| Mission-aware fleet + readiness + V2G (tested) | ✅ done | `ev_fleet.py` (6 tests) |
| Damaged-feeder generator | ✅ done | `damage.py` |
| Feeder integration / episode loop | ✅ done | `ev_scenario.py` |
| V2G interior-optimum result | ✅ done | `ev_v2g.py` |
| Adaptive controller + std metrics + robustness | ✅ done | `ev_resilience.py` |
| Regular-EV equity metric + finding | ✅ done | `ev_resilience.py` |
| Literature grounding + references | ✅ done | `README.md` |
| OPF/MPC baselines | ⬜ pending | (Jia) |
| RC↔ESC correlation validation | ⬜ pending | (Jia) |
| Sensitivity: V2G efficiency, hospital load, SOC₀ | 🟡 partial | extend sweep |
| Multi-feeder / zero-shot topology generalization | ⬜ pending | **Fanchen (F-EV-1)** |
| `d_struct`: structure predicts behavior | ⬜ pending | **Fanchen (F-EV-2)** |
| Real EMS/EV/outage data | ⬜ deferred | later (NE lift) |
| SMR add-on | ⬜ deferred | optional |

## 8. Decisions locked
Per-charger allocation (no routing); readiness-gated dispatch (unserved = failed);
lexicographic hospital ≫ emergency ≫ regular; 15-min / 24–72 h; supply = degraded
grid + renewables + storage + V2G (SMR deferred); count-based readiness; all EVs
modeled, ambulance-focused; standalone; IEEE feeders simulation-first; V2G RTE 85%.

## 9. Open questions / risks
- **Is this a graph paper?** The headline (adaptive V2G) is topology-independent;
  the control decision has repeatedly looked topology-light. The graph angle survives
  only if Fanchen's F-EV-1/F-EV-2 show structure predicts the readiness/V2G behavior.
  Otherwise this is a (strong) energy-systems paper, not a graph-ML one.
- **Parameter realism.** Results de-risked by the robustness sweep, but EV/mission
  params are modeled, not real. Real EMS data is the main NE lift.
- **V2G physics** still idealized (efficiency added; flow-feasible V2G delivery + losses
  could be refined).
- **Adaptive vs optimal.** Adaptive beats fixed and the hindsight-best-fixed, but we
  haven't compared to a true optimal (LP/MPC) controller — needed to bound the gap.

## 10. Division of labor
- **Jia (control/empirics) — DONE to the pre-coauthor boundary:** environment,
  feasibility, fleet/readiness/V2G, adaptive control, metrics, robustness.
- **Fanchen (structural graph mining) — NEXT:** F-EV-1 generate diverse damaged/
  synthetic radial feeders (RGM) → generalization + zero-shot study; F-EV-2 `d_struct`
  → does structure predict readiness/V2G behavior. See `HANDOFF_TO_FANCHEN.md`.

## 11. Next steps (priority order)
1. OPF/MPC + priority-heuristic baselines; bound adaptive-vs-optimal gap.
2. Validate RC↔ESC correlation; finalize the metric story.
3. Extend robustness (V2G efficiency, hospital load, SOC₀, 72 h horizon).
4. Fanchen: multi-feeder generation + `d_struct` → settle the graph question.
5. (NE lift) source real EMS/EV/outage data; one real-event case study.

## 12. Venue strategy
Aim **Nature Energy** on the systems insight (electrified emergency-response
blackout vulnerability + the necessity of adaptive V2G coordination + the equity
cost). Honest: NE needs real-data grounding; **fallback ladder** Joule / Nature
Communications / Applied Energy / IEEE Trans. Smart Grid — where the current
IEEE-feeder + adaptive-control + robustness result is already a clear win.

## 13. Reproducibility / code map
- `ev_fleet.py` — fleet + readiness + V2G (`pytest directions/ev_only/test_ev_fleet.py`).
- `damage.py` — damaged-feeder generation.
- `ev_scenario.py` — roles + episode loop (+ topology-aware/blind comparison).
- `ev_v2g.py` — interior-optimum frontier.
- `ev_resilience.py` — adaptive/fixed controllers + std metrics + robustness sweep.
  Run: `PYTHONPATH=src:directions/ev_only python directions/ev_only/ev_resilience.py`
- Results JSON under `directions/ev_only/results/`.

## 14. References
See `README.md` "References" — hospital-microgrid resilience (Scientific Reports),
V2G disaster coordination (Symmetry), EVs as mobile storage for critical-load
restoration, V2G round-trip efficiency (IEEE/JRC, ~80–87%), resilience metrics
(ENS/EENS, SAIDI, availability — Energies/MDPI + standard defs).
