"""B-viability test: does graph beat no-graph when BOTH conditions hold —
congestion (decision room exists) AND training against the LP-OPTIMAL
set-selection (not the suboptimal greedy rule)?

If graph > no-graph here, Option B (reformulate toward congested, optimal-target
control) is viable. If not, the negative finding is essentially final.
Run: PYTHONPATH=src python scripts/h1_lp_target_test.py
"""
from __future__ import annotations
import warnings, numpy as np, torch, yaml
warnings.simplefilter("ignore")

from sg_resilience.eval_harness import _FeederCtx, calibrate_budget_oracle
from sg_resilience.policies import build_policy, node_features
from sg_resilience.h1_train import _adj_tensor, eval_policy_C
from scripts.diagnose_decision_room import _critical_subtree_matrix

CAP_SCALE = 0.03


def lp_target(ctx, need, budget, S, caps):
    """LP-optimal per-critical served fraction x* (the topology-aware target)."""
    import cvxpy as cp
    k = need.shape[0]
    x = cp.Variable(k)
    cons = [x >= 0, x <= 1, need @ x <= budget, S @ cp.multiply(need, x) <= caps]
    cp.Problem(cp.Maximize(ctx.cw @ x), cons).solve(solver=cp.OSQP, verbose=False)
    return np.clip(x.value, 0, 1) if x.value is not None else np.zeros(k)


def build_lp_dataset(ctx, seeds, ps):
    adj = _adj_tensor(ctx); S, caps = _critical_subtree_matrix(ctx)
    samples = []
    for seed in seeds:
        D, OUT, P = ctx.arrays(seed, 0.05, 0.85, ps)
        prev = np.zeros(D.shape[1])
        for t in range(D.shape[0]):
            dc = D[t, ctx.cidx] * (~OUT[t, ctx.cidx])
            xstar = lp_target(ctx, dc, float(dc.sum()), S, caps)
            x = node_features(D[t], prev, OUT[t], ctx.minfrac, ctx.priorities,
                              ctx.critical_mask, P[t])
            target = torch.zeros(D.shape[1])
            target[ctx.cidx] = torch.tensor(xstar, dtype=torch.float32)
            samples.append((x, adj, target))
            prev = D[t] * 0  # no teacher-forced alloc needed for ranking target
    return samples


def train(kind, ctxs, scales, seeds, epochs=40, seed=0):
    torch.manual_seed(seed)
    model = build_policy(kind, hidden=64)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3); lossf = torch.nn.MSELoss()
    data = []
    for ctx, sc in zip(ctxs, scales):
        data += build_lp_dataset(ctx, seeds, sc)
    best, bestv = None, -1
    for ep in range(epochs):
        model.train(); np.random.default_rng(ep).shuffle(data)
        for x, adj, tgt in data:
            opt.zero_grad(); loss = lossf(model(x, adj), tgt); loss.backward(); opt.step()
        if ep % 5 == 4:
            v = np.mean([eval_policy_C(model, c, seeds, s)["C"] for c, s in zip(ctxs, scales)])
            if v > bestv: bestv, best = v, {k: vv.clone() for k, vv in model.state_dict().items()}
    if best: model.load_state_dict(best)
    return model


def main():
    split = yaml.safe_load(open("configs/topology_split_v1.yaml"))
    tr = [_FeederCtx(dict(e), 48, cap_scale=CAP_SCALE) for e in split["G_train"]]
    oo = [_FeederCtx(dict(e), 48, cap_scale=CAP_SCALE) for e in split["G_ood"]]
    cs = tuple(range(100, 106))
    trs = [calibrate_budget_oracle(c, cs)["power_scale"] for c in tr]
    oos = [calibrate_budget_oracle(c, cs)["power_scale"] for c in oo]
    tseeds, eseeds = tuple(range(5)), tuple(range(5, 13))
    res = {}
    for kind in ("deepsets", "graphsage"):
        m = train(kind, tr, trs, tseeds)
        ood = np.mean([eval_policy_C(m, c, eseeds, s)["C"] for c, s in zip(oo, oos)])
        res[kind] = float(ood)
        print(f"{kind:10} OOD C (LP-target, cap_scale={CAP_SCALE}) = {ood:.3f}")
    print(f"\nB-VIABILITY: graph - nograph OOD C = {res['graphsage'] - res['deepsets']:+.3f}")
    print("  positive => graph helps under congestion+optimal-target => Option B viable")


if __name__ == "__main__":
    main()
