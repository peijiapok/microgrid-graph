# 04 — Division of Labor

> Last revised 2026-04-22. Who owns what, who reviews what, how decisions escalate. Update on any ownership change.

## 1. Roles

- **Jia (lead, resilience + empirics).** Problem definition, resilience metrics, SimBench/pandapower harness, scenario generation, physics evaluation, production engineering, final writing integration, paper submission.
- **Collaborator (lead, graph learning + theory).** Graph operator design, OOD topology protocol, differentiable projection, CVaR-minimax objective, theory, GNN-baseline specs, ML-side related work.

## 2. Contribution ownership

| Contribution | Designer | Implementer | Reviewer | Writer (primary) |
|---|---|---|---|---|
| C1. Flow-aware MP | Collaborator | Collaborator | Jia | Collaborator |
| C2. Topology-generalizing policy + atlas | Collaborator | Collaborator (atlas) + Jia (splits harness) | Jia | Collaborator |
| C3. Differentiable feasibility projection | Collaborator | Joint | Joint | Collaborator |
| C4. CVaR-minimax + regret bound | Collaborator | Joint | Jia | Collaborator |
| C5. GNN baseline zoo | Collaborator (specs) | Jia | Collaborator | Jia |
| C6. Topology-similarity metric d_struct | Collaborator | Collaborator | Jia | Collaborator |

## 3. Cross-cutting work

| Area | Primary | Secondary |
|---|---|---|
| Problem statement (`docs/01`) | Jia | Collaborator (graph-learning framing) |
| Research goal (`docs/02`) | Jia | Collaborator |
| Experimental protocol (`docs/05`) | Jia | Collaborator (splits + ablations) |
| Theory (`docs/06`) | Collaborator | Jia (empirical checks) |
| Related work (`docs/07`) | Collaborator (graph, theory buckets) | Jia (resilience, OPF buckets) |
| Paper intro + abstract | Jia (first draft) | Collaborator (revise) |
| Paper method | Collaborator | Jia |
| Paper experiments | Jia | Collaborator |
| Paper theory section | Collaborator | Jia |
| Paper discussion + limitations | Jia | Collaborator |
| Rebuttal (when it comes) | Joint | — |

## 4. Review policy

- Every PR touching `src/sg_resilience/` requires the **other** person's review before merge.
- Every edit to `docs/01` or `docs/02` requires written acknowledgement in the relevant PR by both.
- `docs/06` (theory) may be edited unilaterally by the collaborator; empirical-check notes appended by Jia.
- Experiment-only PRs (new runs, new figures) may be self-approved but must log in `notes/weekly/`.

## 5. Decision escalation

Four classes of decision, from lightest to heaviest:

1. **Tactical.** How to implement a function, naming, minor refactor. Unilateral.
2. **Design.** A new module, a new objective term, a choice of GNN operator. Requires an ADR in `notes/adr/`, single reviewer approval.
3. **Framing.** A change to the problem statement, the claim set, or the non-goals. Requires both to agree in writing (comment in the ADR, or a sign-off in the sync notes).
4. **Scope.** Adding a contribution, dropping a contribution, changing the target venue. Requires a written deliberation (rationale, alternatives, cost) and both to sign off.

When in doubt, escalate one level. A 20-minute conversation upstream saves a week downstream.

## 6. Authorship

To be settled in writing before the first submission. The default expectation recorded here, revisable on request:

- Jia: first author.
- Collaborator: co-first author or second author, depending on how theory-heavy the final paper is.
- Any additional contributors named in decreasing order of contribution.

This line will be replaced with a concrete, signed agreement in `notes/adr/0001-authorship.md` before we submit anywhere.

## 7. Exit conditions

If the framing proves infeasible after Gate 1 (see `02_research_goal.md` Section 4), we re-scope — not silently add more and more scope. Explicit re-scope options, to be decided jointly:

- Drop C2 and refocus on C1+C3+C4 as a physics-aware safe-control paper (narrower, still NeurIPS-viable).
- Drop C4 and refocus on C1+C2+C5 as a pure graph-generalization paper (less theory, more empirics).
- Escalate to a journal (IEEE TPS, J. Mach. Learn. Res.) if the ML story is thin but the application is strong.

Do not silently drop a contribution. Record the re-scope.
