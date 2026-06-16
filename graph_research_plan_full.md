# Continuity-Aware Graph Reinforcement Learning for Resilient Critical Infrastructure under Stochastic Grid Failures

**Master research plan and slide content — full reference document.**

Target venue: AAAI (primary). Fallback: Nature Communications, KDD, ICLR workshops, IEEE Transactions on Smart Grid, Applied Energy.

---

## Table of Contents

1. [Title, Explained Word by Word](#1-title-explained-word-by-word)  
2. [Setting](#2-setting)  
3. [Problem — Stochastic Graph Failure Control](#3-problem--stochastic-graph-failure-control)  
4. [Why a Graph](#4-why-a-graph)  
5. [Motivation](#5-motivation)  
6. [Literature Review](#6-literature-review)  
7. [Project Workflow](#7-project-workflow)  
8. [System Architecture](#8-system-architecture)  
9. [Methodology](#9-methodology)  
10. [Formal Problem Formulation](#10-formal-problem-formulation)  
11. [Data Sources](#11-data-sources)  
12. [Microreactor Component](#12-microreactor-component)  
13. [Roles and Division of Work](#13-roles-and-division-of-work)  
14. [Detailed Lead-Author Tasks](#14-detailed-lead-author-tasks)  
15. [Three-Month Timeline](#15-three-month-timeline)  
16. [AAAI Positioning](#16-aaai-positioning)

---

## 1\. Title, Explained Word by Word

The full title is: *"Continuity-Aware Graph Reinforcement Learning for Resilient Critical Infrastructure under Stochastic Grid Failures."*

| Word / phrase | Meaning |
| :---- | :---- |
| **Continuity-Aware** | Cares whether service stays uninterrupted over time — "did the hospital stay on?" not "how much did we deliver?" |
| **Graph** | Infrastructure as a network. Nodes \= hospitals, shelters, batteries, microreactors. Edges \= power lines, switches, routes. Topology matters. |
| **Reinforcement Learning** | The controller learns by interacting: observe state → take action → receive reward → improve. Actions \= allocate, shed, switch, island, reroute. |
| **for** | Signals the application this method supports. |
| **Resilient** | Keeps functioning when damaged. Critical function preserved despite failures. |
| **Critical Infrastructure** | Infrastructure whose failure has severe social consequences: hospitals, shelters, communications, emergency response. |
| **under** | Tested in adverse conditions, not normal operation. |
| **Stochastic** | Random and uncertain. The controller doesn't know when, where, or how the grid will fail. |
| **Grid Failures** | Parts of the network stop working: line outages, node disconnections, overloads, damaged substations, cascades. |

**One-sentence summary:** an AI controller that learns to keep hospitals continuously powered when the grid is randomly damaged.

---

## 2\. Setting

Post-disaster microgrids supporting critical infrastructure. A controller must allocate limited electrical resources over a dynamically failing network — earthquakes, cascading overloads, wartime disruption — while keeping critical loads continuously powered.

**Critical loads:** hospitals, emergency shelters, communication infrastructure, military / remote installations.

**Energy assets:** PV generation, battery storage, nuclear microreactors (stable baseload, low ramp), switchable distribution edges.

**Benchmark systems:** IEEE 33-bus and IEEE 123-bus, augmented with disaster scenarios and microreactor integration. Synthetic graphs (random, scale-free, grid) are also used to demonstrate generality across topologies.

---

## 3\. Problem — Stochastic Graph Failure Control

A new AI problem setting. At each timestep:

- Graph topology changes  
- Edges and nodes may disappear  
- Demand shifts unpredictably  
- Failures cascade  
- The controller must reallocate limited resources

**What existing methods optimize:** maximum restored load (additive), minimum immediate violations, instantaneous reward, step-by-step efficiency.

**What we optimize:** uninterrupted service trajectories, trajectory-level continuity for critical nodes, long-horizon survival probability, stability under partial observability.

A hospital losing power briefly may constitute catastrophic failure — even if total delivered energy is high.

---

## 4\. Why a Graph

The problem is fundamentally topological. Continuity-aware control over a degrading network is about *which paths still exist* after each failure. Flat vectors and time series cannot represent that.

**Failures are topological events.** When a line drops, "is the hospital still reachable?" depends on connectivity. That's a graph question, not a numeric one. A flat feature vector loses the structure that makes the question answerable.

**Power flow follows paths.** "Can we route power from the microreactor to the hospital?" is a path question. It only makes sense over a graph with edges, switches, and routes. The controller has to reason about reachability, not just totals.

**The state space keeps changing shape.** When nodes or edges fail, the input dimension of a flat vector would change. Graphs naturally handle variable structure — same model, evolving topology — letting one policy survive across many disaster scenarios.

**Local reasoning \+ global reach.** A GNN lets the controller look at each node and its neighborhood while aggregating global structure across the whole network. The same architecture generalizes from IEEE 33-bus to 123-bus without re-design.

---

## 5\. Motivation

### 5.1 Why this matters now (scale and frequency)

- U.S. weather-related major grid outages have roughly doubled over the last decade (DOE / EIA event data)  
- Hurricane Maria's grid collapse in Puerto Rico contributed to \~3,000 excess deaths — hospital and dialysis fallout was the dominant pathway  
- The 2021 Texas winter storm left hospitals on diesel for days; multiple deaths tied to medical-equipment power loss  
- DoD has formally classified grid disruption as a mission-assurance risk — the reason Project Pele exists

**Framing line:** *Catastrophic grid degradation is no longer a tail event — it's a recurring operational reality, and the loads that fail first are the ones society can least afford to lose.*

### 5.2 Why a brief outage is catastrophic (clinical anchor)

- A 30-second power loss can interrupt surgery mid-procedure (electrocautery, anesthesia monitors, perfusion pumps)  
- ICU ventilator backup batteries last \~30 minutes; advanced respiratory patients tolerate seconds, not minutes  
- Dialysis mid-cycle has to be aborted with blood already extracorporeal  
- Cold-chain failure spoils biologics — a single freezer outage can destroy \>$100k of vaccines or cell therapies  
- Elevators lose service, trapping patients mid-transfer between OR / ICU / wards

**Framing line:** *In hospitals, the cost of an outage is measured in surgeries aborted and ventilators failed — not kilowatt-hours lost.*

### 5.3 The operational gap (what's done today)

**Today's default:** diesel/gas backup generators at each critical facility, manual operator restoration, static rule-based feeder reconfiguration, pre-planned switching procedures.

**Why it breaks under catastrophe:**

- Diesel runs out — the refueling chain itself fails in a disaster  
- Manual restoration is too slow when failures cascade across hundreds of nodes  
- Rule-based reconfiguration doesn't adapt to topologies it wasn't designed for  
- None of these *plan* for continuity — they *react* to outages after they happen

### 5.4 The academic gap (what literature misses)

What existing literature already covers: safe DRL for restoration, digital twins for topology estimation, RL-based microgrid dispatch, multi-agent restoration, graph RL in power systems.

What existing literature does **not** cover:

- Trajectory-level continuity optimization  
- Uninterrupted critical-service preservation  
- Stochastic graph destruction with continuity constraints  
- Resilient control under unobservable degradation  
- Microreactor constraints in dynamic graph degradation

### 5.5 Continuity, defined

**Continuity** \= a critical load is served *without interruption* throughout the simulation horizon. Measured in uptime and interruption events — not in total energy delivered.

### 5.6 Why classical optimization isn't enough

Classical optimization (MPC, MILP) hits a wall here because the problem combines:

- **Combinatorial action space** — which switches, which loads to shed, which paths to keep alive  
- **Stochastic non-stationary topology** — the model the optimizer is built on is wrong by the next timestep  
- **Partial observability** — sensor coverage drops as the network fails  
- **Long-horizon credit assignment** — continuity is a trajectory property, not a per-step optimum

RL is built for sequential decisions under uncertainty. Graph encoding lets the policy reason about reachability, not just totals. One trained policy generalizes across topologies it hasn't seen — same model handles many disaster shapes.

**Framing line:** *MPC and MILP can solve the problem for the network you assumed. They can't solve it for the network you'll actually have ten seconds from now.*

### 5.7 Hypothesis

A continuity-aware graph RL framework can achieve significantly more stable critical service under stochastic topology degradation than conventional restoration-oriented methods.

---

## 6\. Literature Review

Four representative recent works frame the landscape, and none target continuity-aware control under stochastic catastrophic degradation.

| \# | Title | Source | Does | Gap |
| :---- | :---- | :---- | :---- | :---- |
| 1 | Real-time outage management in active distribution networks using RL over graphs | Nature Communications, 2024 | Graph-based RL for outage response in distribution networks | Does **not** model continuity. Does **not** model unknown catastrophic degradation. |
| 2 | Explainable DRL for resilient and battery-aware microgrid control | Energy Conversion and Management, 2026 | Resilient & battery-aware microgrid control with explainable DRL | **No graph learning** — ignores topology of the network |
| 3 | Graph reinforcement learning for power grids: A comprehensive survey | Energy and AI, 2025 | Survey of graph RL methods applied to power systems | Surveys existing work — none target trajectory-level continuity under catastrophic degradation |
| 4 | Multitask active transfer learning load shedding via GCN–Transformer for transient stability | Applied Energy, 2025 | GCN+Transformer load shedding for transient stability with missing data | Focuses on transient stability, not catastrophic infrastructure survival or continuity |

**Where we sit:** trajectory-level continuity preservation \+ stochastic catastrophic graph degradation \+ microreactor-aware planning — an unaddressed intersection.

---

## 7\. Project Workflow

Six-step research workflow as captured in the project diagram:

1. **Disaster & motivation** — storms, earthquakes, cascading outages put hospitals/shelters at risk  
2. **Problem formulation** — dynamic failing graph G(V, E\_t); objective is continuity, not max restored load  
3. **Simulator** — failures evolve over time; agent observes, decides, environment updates  
4. **Proposed AI controller** — Continuity-Aware Graph RL: graph encoder → GNN → memory → RL policy outputting allocate / shed / island / reroute actions  
5. **Baselines & experiments** — rule-based, greedy, MPC, RL w/o graph, graph RL w/o continuity. Tested on synthetic graphs and IEEE 33- & 123-bus microgrids  
6. **Evaluation & main message** — metrics measure uptime, interruptions, outage duration, survival probability — showing continuity-aware control sustains critical service longer

**AI contribution:** decision-making under stochastic graph degradation with continuity constraints.

---

## 8\. System Architecture

**Offline phase — scenario design.** Disaster scenarios \+ infrastructure data (hospitals, shelters, batteries, microreactors) feed a Dynamic Graph Failure Generator that produces time-varying damaged graphs and demand trajectories.

**Online phase — control loop.** Resilient Microgrid Environment ↔ Continuity-Aware Graph RL Agent. Each step: state s\_t → action a\_t (allocate / shed / switch / island) → reward r\_t (continuity, ramping cost) → next state s\_{t+1}.

**Outputs.** Evaluation metrics (hospital uptime, interruption count, outage duration, served critical load) plus a learned resilient policy that prioritizes critical nodes under stochastic degradation.

**Main message:** the controller learns to preserve continuous service — not maximize instant restored load.

---

## 9\. Methodology

### 9.1 The Whole Idea

The conceptual move at the heart of the paper: stop measuring how much load you've restored, start measuring whether critical service has stayed on.

**Conventional restoration view:**

- Score the controller by total kWh delivered each step  
- Treats every kilowatt-hour as equal  
- Treats brief outages as recoverable  
- Rewards aggressive reconfiguration that maximizes throughput  
- Result: a hospital can flicker dark and the score still looks great

**Continuity-aware view:**

- Score the controller by whether critical loads have *never gone down*  
- An interruption is the cost — not under-delivery  
- Stable, slightly suboptimal routing beats aggressive but flicker-prone control  
- The agent learns to *defend* a small set of critical loads relentlessly  
- Brief outage at a hospital \= catastrophic, even if total energy is high

It looks like a reward tweak. It is actually a different planning problem.

### 9.2 Energy Planning

What the planner sees:

- Criticality tiers per load (hospital → shelter → residential)  
- Current availability of each source — microreactor, PV, battery  
- Which network paths are still up — and which look fragile  
- Recent service history of every critical load

Source-level tradeoffs:

- **Microreactor:** stable but inflexible — can't ramp to cover a spike, can't ramp down to avoid waste  
- **PV:** free energy but disappears with weather and time of day  
- **Battery:** scarce — has to be reserved for moments that matter most

Decisions a continuity-aware planner makes:

- Pre-emptively shed lower-priority loads *before* the network forces it  
- Reserve battery headroom for predicted-fragile critical loads  
- Choose routing paths that survive the *next* failure, not just the current one  
- Form islands when the main grid is too damaged to be useful  
- Hold microreactor output steady — and absorb mismatch on the demand side  
- Accept lower total throughput to keep critical service uninterrupted

This whole layer — simulator, disaster scenarios, criticality tiers, dispatch logic, continuity-based metrics — is the lead author's contribution.

### 9.3 Graph Component (co-author territory)

The graph-AI piece is the co-author's territory. The lead author describes:

**What I hand in:** buildings and their criticality, lines that connect them, which lines are working right now, what's loaded where.

**What I expect back:** a representation the policy can consume; captures "where am I in the network, what's reachable, what's about to break"; stays useful as the network changes shape.

**What this should enable:** adapt smoothly when edges drop or nodes go offline; generalize across different network topologies — no retraining per scenario; reason about path-level structure.

**Boundary:** the lead author provides the simulator, the scenarios, and the evaluation criteria. The co-author brings the right graph machinery.

---

## 10\. Formal Problem Formulation

A complete problem formulation has five blocks: action space, state space, constraints, dynamics, and an objective function.

### 10.1 Action Space

At each timestep the controller picks decision variables across multiple categories.

**Generation setpoints:**

- Microreactor output (within ramp limits)  
- PV curtailment  
- Battery charge/discharge rate  
- Emergency diesel generator on/off \+ setpoint

**Load actions:**

- Per-critical-load: serve fully / shed / partial supply (continuous fraction)

**Topology actions:**

- Switch open/close  
- Breaker positions  
- Intentional islanding

**Routing actions:**

- Re-route power through alternative paths when primary lines are down

**Mobile-asset actions (EVs, ambulances):**

- Dispatch EV/ambulance to a node  
- Set V2G mode (charging vs. discharging)

**Restoration actions:**

- Initiate repair on a failed component (consumes time \+ crew resource)

**Emergency generator activation:**

- Start, fuel-allocation, dispatch level

### 10.2 State Space

What the controller observes (and what is partially observed):

- Graph topology G\_t (which edges are alive)  
- Per-node features: criticality tier, current demand, current supply, last-period service status  
- Per-asset state: battery SOC, microreactor current output, generator fuel inventory, EV location \+ SOC  
- Failure indicators (which lines/nodes are damaged)  
- Time-of-day, weather (for PV), demand forecast horizon

Some elements are partially observed: sensor coverage drops as the network fails.

### 10.3 Constraints

**Physical / electrical:**

- Power balance at every node: supply \= demand \+ losses  
- Voltage limits at every bus (e.g., 0.95–1.05 p.u.)  
- Line thermal capacity — line flow ≤ rating  
- Battery SOC ∈ \[SOC\_min, SOC\_max\] — typically \[0.1, 0.9\]  
- Battery charge/discharge rate ≤ C-rate limit  
- Microreactor: P\_min ≤ P ≤ P\_max, |ΔP/Δt| ≤ ramp rate  
- Diesel generator: fuel inventory ≥ 0, P\_min when on, startup time  
- Critical-load minimum survival power: P\_load ≥ P\_min\_critical (e.g., hospital can't operate below \~80% nominal)

**Topological / operational:**

- Network must remain radial (no parallel paths) — restricts which switches can be open simultaneously  
- Frequency stability: total inertia from spinning sources must exceed minimum  
- Minimum up-time / down-time for generators  
- Maximum simultaneous load-shed events (operational realism)

**Logical / incompatibility:**

- Can't dispatch an offline generator  
- Can't form an island that contains zero generation  
- Can't charge an EV that's not at a charging node  
- Can't shed and serve the same load in the same step  
- Can't restore a line whose repair hasn't completed

### 10.4 Objective Function — Everything in Dollars

Convert all consequences to a single $ scalar. This is the most important and hardest-to-defend section.

**Cost components:**

- **Energy cost:** fuel for diesel ($/gallon × consumption), microreactor O\&M ($/MWh), battery degradation cost per kWh-cycle ($)  
- **Switching cost:** small $ penalty per switch operation (equipment wear, \~$1–10 per operation)  
- **Value of Lost Load (VOLL):** $/kWh of unserved load, varies by load type:  
  - Residential: \~$5/kWh  
  - Commercial: \~$20/kWh  
  - Industrial: \~$50/kWh  
  - Hospitals: much higher (specific values from EPRI and DOE studies)  
- **Value of Statistical Life (VSL)** for outage-induced harm:  
  - U.S. DOT current value: \~$11–13M (2022 dollars); $4M is older / lower-bound — pick a citable number and stick with it  
  - EU: \~€1–3M depending on member state  
  - Multiply VSL × P(harm | outage of duration t at facility type X)  
- **Hospital damage cost:** lost revenue per hour of outage \+ patient-harm probability × VSL \+ equipment damage (refrigeration spoilage at $100k/freezer event) \+ medication loss  
- **EV/ambulance opportunity cost:** if an ambulance is not charged, the cost is missed emergency response × VSL × probability of fatal delay  
- **Repair cost:** crew dispatch \+ parts \+ time

**Continuity penalty layer:** each interruption event at a critical load incurs a *step penalty* (a fixed $ cost per interruption event) on top of the per-kWh VOLL — this is what makes interruptions asymmetrically expensive vs. underdelivery.

**Objective:** minimize expected total $ cost over the simulation horizon.

### 10.5 Dynamics

How the state evolves given an action:

- **Failure process:** how lines/nodes fail over time (random, cascading, hurricane-pattern, adversarial)  
- **Restoration dynamics:** once a component is being repaired, time-to-repair distribution; once repaired, comes back online  
- **Demand evolution:** load profiles change with time-of-day \+ criticality  
- **PV evolution:** irradiance time series  
- **EV mobility:** how EVs move between nodes  
- **Microreactor dynamics:** ramp constraints couple consecutive timesteps

---

## 11\. Data Sources

The simulator draws on standard power-engineering benchmarks, public load and renewable datasets, microreactor design specs, and historical disaster records.

**Network topology:** IEEE 33-bus & 123-bus distribution feeders; synthetic graphs (random, grid, scale-free); OpenStreetMap for real distribution layouts.

**Load profiles:** NREL OpenEI hospital / commercial / residential profiles; smart-meter datasets (AMPds, REDD); criticality tiers tagged on each load.

**Renewable generation:** NREL NSRDB solar irradiance; weather records correlated with PV output; capacity-factor & seasonality profiles.

**Microreactor specs:** DOE Microreactor Program documentation; public designs (Oklo Aurora, eVinci, Project Pele, Xe-Mobile); ramp rates, fuel cycles, capacity factors.

**Disaster & failure scenarios:** EIA outage records, DOE event archives; historical hurricane / earthquake / wildfire damage data; synthetic failure injection (random, cascading, adversarial).

**Simulation environment:** OpenDSS, GridLAB-D, PandaPower for grid physics; PyPSA for high-level dispatch; custom RL wrapper feeding GNN \+ policy.

Each rollout is a distinct disaster scenario over a real or benchmark distribution network.

---

## 12\. Microreactor Component

### 12.1 Why microreactors specifically

Nuclear microreactors (1–20 MW, factory-built, transportable) are emerging as a dedicated backbone for disaster-resilience and remote-site power. Their physics make them behave fundamentally differently from PV, wind, or batteries.

**What makes a microreactor different:**

- Highly reliable: not weather-dependent, not fuel-resupply-dependent for years  
- Large continuous baseload: always on, always producing  
- Limited ramp flexibility: cannot follow load like a battery or gas turbine  
- Long autonomy: single fuel load can run multi-year

**Why that reshapes the problem:**

- Energy is plentiful and stable — but the *delivery network* is what fails  
- Mismatch: stable inflexible source \+ dynamic uncertain network  
- Optimal control: route, switch, and shed to preserve continuity  
- Existing solar/wind/battery RL methods do not capture this regime

A microreactor changes the question from "how do we generate enough?" to "how do we deliver what we already have, continuously?"

### 12.2 Microreactor vs SMR vs Large NPP

| Attribute | Large NPP | SMR (Small Modular) | Microreactor |
| :---- | :---- | :---- | :---- |
| Capacity | 1,000–1,600 MW | 50–300 MW | **1–20 MW** |
| Footprint | Large site \+ exclusion zone | Compact plant, fixed site | **\~1–3 acres · containerizable** |
| Construction | 10–15+ years on-site build | 3–7 years, modular shipping | **Factory-built, 1–3 years deploy** |
| Mobility | Permanently fixed | Modules shipped, plant fixed | **Truck / rail / ship transportable** |
| Refueling cycle | 18–24 months | 4–7 years | **5–20 years (single fuel load)** |
| Load-following | Poor — designed for steady baseload | Limited | **Better, still inflexible vs. battery** |
| Operators | Hundreds on site | Tens | **Minimal · semi-autonomous** |
| Primary use case | Bulk grid baseload | Replacing coal plants, large industrial | **Forward bases · remote sites · disaster microgrids** |
| Example designs | AP1000, EPR, APR1400 | NuScale (77 MW), BWRX-300, Rolls-Royce SMR | **Oklo Aurora, eVinci, Project Pele, Xe-Mobile** |

**Implication:** large NPPs and SMRs are *grid-scale* assets — they require the grid to deliver value. A microreactor is the only nuclear option that can sit *inside* a damaged, islanded microgrid — exactly the regime where continuity-aware control matters.

### 12.3 Safety and Siting

Anticipated reviewer question: "Is a microreactor safe? And isn't a microgrid close to humans?"

**The engineering answer.** Microreactors are deliberately built around passive and inherent safety. Standard pitch points: TRISO fuel acts as its own containment up to \~1800°C; passive decay-heat removal via natural circulation; negative temperature reactivity feedback; smaller radioactive inventory caps worst-case release. This is why the marketing language is "walk-away safe."

**The siting answer.** The Emergency Planning Zone (EPZ) for large nuclear plants (10-mile plume, 50-mile ingestion) was set in the 1970s based on big-LWR worst-case accidents. The NRC's new framework (10 CFR Part 53\) lets the EPZ scale with the credible accident, not with a fixed historical distance. Microreactor designs target *site-boundary* EPZ — NuScale was the first to win NRC sign-off on this; Oklo, BWXT, X-energy, and eVinci are pursuing the same path.

**Real precedents:** DoD Project Pele is explicitly building a transportable microreactor for forward operating bases; DOE Microreactor Demonstration targets remote communities, military installations, end-user industrial sites; Eielson AFB in Alaska is getting one on-base; university research reactors (1–10 MW) have operated on campuses for decades.

**Rhetorical move for the paper:** *"We are not designing a reactor and we are not making safety claims about one. The microreactor enters our simulator as a power source with known operational characteristics — capacity, ramp rate, fuel cycle. Reactor licensing is the vendor's and regulator's responsibility (NRC, DOE). Our problem is grid resilience under stochastic failure, given that such assets exist and are being actively deployed."*

---

## 13\. Roles and Division of Work

The structure is intentional: the *main idea* — continuity-aware resilient control under stochastic graph degradation — stays with the lead author. Graph AI is a critical enabling component, not the identity of the paper.

| You — Lead / First Author | Co-Author — Graph AI Expert |
| :---- | :---- |
| Problem framing | Method component |
| System & environment design | Graph encoder |
| Continuity concept | GNN implementation |
| Experiments & ablations | Representation learning |
| Paper narrative & positioning | Graph architecture |

This boundary matters for: authorship clarity, AAAI positioning, keeping the paper coherent rather than "generic graph ML."

### 13.1 Lead Author (You) — full ownership

- Problem formulation: stochastic graph failure control; continuity-aware resilience  
- Infrastructure system: microgrid, disaster scenarios, microreactor \+ PV \+ battery \+ critical loads  
- Continuity objective: uninterrupted-service metric, interruption penalties, continuity-based evaluation. **Strongest contribution.**  
- Experiments: benchmarks, baselines, scenarios, ablations, analysis  
- Writing: intro, motivation, related work, experiments, discussion, AAAI framing

### 13.2 Co-Author (Graph AI Expert) — full ownership

- Graph encoding: node features, edge features, dynamic-topology representation  
- GNN implementation: GraphSAGE / GAT / dynamic graph embeddings  
- RL integration: connecting graph embeddings to policy / value network  
- Dynamic graph reasoning: handling missing edges, topology adaptation, encoding graph degradation

---

## 14\. Detailed Lead-Author Tasks

### 14.1 Problem formulation (desk work, before any code)

**What you produce:** a 5–8 page formal problem-formulation document with action space, state space, constraints, objective, and dynamics. Becomes Section 3 of the paper.

**Order to write it in:** action space → constraints → objective → state → dynamics. Action space first because reviewers ask about it first; constraints right after because they bound feasibility; objective third because that's where the continuity story lives; state and dynamics last because they're mostly bookkeeping.

### 14.2 Infrastructure system (the simulator)

- Take IEEE 33-bus and 123-bus topologies and load them in PandaPower / OpenDSS / GridLAB-D / PyPSA  
- Tag a subset of buses as critical (hospital, shelter) with criticality tiers  
- Attach energy assets to selected buses: PV (irradiance profile), battery (SOC \+ efficiency), microreactor (capacity \+ ramp \+ fuel cycle)  
- Implement disaster scenarios as failure-injection modules: random, cascading overload, hurricane-pattern, earthquake-pattern, adversarial  
- Wrap in a clean Gym-style RL environment with `reset()`, `step(action)`, `observation`, `reward`  
- Generate synthetic graph variants (random, scale-free, grid)

**Artifact:** a reproducible simulator your co-author can plug their GNN into without needing to understand the power-system internals.

### 14.3 Continuity objective (your strongest contribution)

- Define exactly what counts as an "interruption" — full power loss for ≥1 timestep, or below-threshold supply for ≥k timesteps. Pick one and justify it.  
- Design the reward function: weight on critical-load uptime, penalty on interruption events, penalty on switching instability, penalty on overload risk  
- Define evaluation metrics that aren't the same as the reward: hospital uptime fraction, interruption count, outage duration, critical-node survival probability  
- Justify why total-restored-load is the wrong metric for this problem  
- Run a sensitivity sweep showing the reward weights matter (so reviewers see it's not arbitrary)  
- Show via experiment that an agent optimizing total-restored-load *fails* on continuity metrics — that comparison is your money shot

**Artifact:** Section 4 of the paper, plus the metric definitions everyone in your codebase agrees on, plus the reward function in code.

### 14.4 Experiments

- **Benchmarks:** synthetic graphs \+ IEEE 33-bus (deep ablations) \+ IEEE 123-bus (headline scalability)  
- **Baselines:** rule-based restoration, greedy-centrality, MPC, RL-without-graph, **graph-RL-without-continuity** (this last one is critical — it isolates the contribution of your continuity objective specifically)  
- **Scenarios:** at least 3 disaster patterns × 2–3 severity levels × 10+ random seeds for confidence intervals  
- **Ablations:** vary reward weights, swap graph encoder (GraphSAGE vs GAT), vary observation noise, vary failure intensity  
- **Analysis:** confidence intervals on every plot, statistical significance for headline claims, qualitative case-study trajectories showing *why* continuity-aware behaves differently  
- **Honest failure modes:** where does your method break?

**Artifact:** Section 5 with \~6–10 figures and 2–3 tables, all with reproducible scripts.

### 14.5 Writing

- **Intro (1.5 pages):** hook the AAAI reviewer in the first paragraph as an *AI problem*, not a power-systems problem. Contributions list at the end.  
- **Motivation:** weave together the 5 motivation pieces (scale, clinical stakes, operational gap, continuity definition, why-RL-not-MPC) into 1–1.5 pages.  
- **Related work:** 4–6 paragraphs positioning against the four papers from the literature review. Be precise about gaps.  
- **Experiments narrative:** write prose around your figures. Don't just describe — interpret. "We find X. This means Y."  
- **Discussion:** limitations, future work, broader impact.  
- **AAAI framing:** ruthlessly cut power-systems jargon from the intro and contributions. The first 2 pages should read as an AI paper that uses grids as a testbed, not a grid paper that uses AI.  
- **Coordination:** integrate co-author's writing on the graph method, normalize notation, fix terminology drift.  
- **Final polish:** references, formatting, supplementary material.

**Artifact:** the AAAI submission PDF, in your name first.

---

## 15\. Three-Month Timeline

**Month 1**

- Formalize problem (the 5–8 page formulation document)  
- Implement simulator (IEEE 33/123-bus \+ asset overlays \+ failure scenarios)  
- Build baseline RL (without graph, without continuity)  
- Define metrics

**Month 2**

- Implement GNN (co-author)  
- Implement continuity-aware reward  
- Run experiments across scenarios and seeds  
- Conduct ablations

**Month 3**

- Finalize experiments, fill any gaps the writing reveals  
- Write paper  
- Generate figures  
- Polish AAAI submission, supplementary material

---

## 16\. AAAI Positioning

**Frame as:** *"AI decision-making under stochastic graph degradation with continuity constraints."*

**Not:** *"AI for microgrids."*

Microgrids serve as the motivating application, the realism benchmark, and the catastrophic infrastructure testbed — not the identity of the paper.

**Final positioning statement:** We propose a continuity-aware graph reinforcement learning framework for decision-making under stochastic graph failures, enabling uninterrupted support for critical infrastructure during catastrophic network degradation.

---

## Appendix: Key Concepts in One Place

**Topology** \= the structure of how things are connected. Which dots have lines between them in the network graph. Two grids with identical equipment can have completely different topologies if the wiring is different. When a line fails, the topology changes — the hospital didn't move and the microreactor didn't break, but the *path* between them is gone.

**Continuity** \= a critical load is served without interruption throughout the simulation horizon. Measured in uptime and interruption events — not in total energy delivered.

**Stochastic Graph Failure Control (SGFC)** \= the new AI problem class this paper introduces. A controller must allocate limited resources over a graph whose topology, demand, and failures all evolve unpredictably, with the objective of preserving uninterrupted service to a designated subset of critical nodes.

**Value of Statistical Life (VSL)** \= the monetary value used to convert mortality risk into dollars. U.S. DOT: \~$11–13M (2022). EU: \~€1–3M depending on member state. Used to express clinical harm consequences in the same units as energy and equipment costs.

**Value of Lost Load (VOLL)** \= $/kWh of unserved electrical demand, varying by load type. Residential \~$5/kWh, commercial \~$20/kWh, industrial \~$50/kWh, hospitals much higher.

**Emergency Planning Zone (EPZ)** \= the regulatory zone around a nuclear facility for emergency planning. Large LWR plants: 10-mile plume, 50-mile ingestion. Microreactors target site-boundary EPZ under NRC's new consequence-based framework (10 CFR Part 53).  
