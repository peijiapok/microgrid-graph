"""Validate candidate v1 feeders for the topology split (ADR-0002).

Checks, per feeder: bus/load/line counts, total load, single-connectedness,
radiality (tree => |E| = |V|-1 on the connected bus graph), availability of
line thermal capacity (max_i_ka) needed for Path-A flow constraints (F_e),
and presence of load profiles for scenario generation.

Run: python scripts/validate_feeders_v1.py
"""
from __future__ import annotations

import sys
import traceback

import networkx as nx


def bus_graph(net):
    g = nx.Graph()
    for b in net.bus.index:
        g.add_node(int(b))
    for _, r in net.line.iterrows():
        g.add_edge(int(r["from_bus"]), int(r["to_bus"]))
    if hasattr(net, "trafo") and len(net.trafo):
        for _, r in net.trafo.iterrows():
            g.add_edge(int(r["hv_bus"]), int(r["lv_bus"]))
    if hasattr(net, "switch") and len(net.switch):
        # closed bus-bus switches fuse buses electrically
        for _, r in net.switch.iterrows():
            if r.get("et") == "b" and bool(r.get("closed", True)):
                g.add_edge(int(r["bus"]), int(r["element"]))
    return g


def line_capacity_coverage(net):
    if "max_i_ka" not in net.line.columns or len(net.line) == 0:
        return 0.0
    col = net.line["max_i_ka"]
    return float((col.notna() & (col > 0)).mean())


def summarize(net):
    g = bus_graph(net)
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    connected = nx.is_connected(g) if n_nodes else False
    ncomp = nx.number_connected_components(g) if n_nodes else 0
    # radiality on the largest connected component
    if n_nodes:
        comp = max(nx.connected_components(g), key=len)
        sub = g.subgraph(comp)
        radial = sub.number_of_edges() == sub.number_of_nodes() - 1
        comp_frac = len(comp) / n_nodes
    else:
        radial, comp_frac = False, 0.0
    has_profiles = bool(getattr(net, "profiles", None)) and "load" in (net.profiles or {})
    return {
        "n_bus": int(len(net.bus)),
        "n_load": int(len(net.load)),
        "n_line": int(len(net.line)),
        "n_trafo": int(len(net.trafo)) if hasattr(net, "trafo") else 0,
        "n_ext_grid": int(len(net.ext_grid)),
        "total_load_mw": round(float(net.load["p_mw"].sum()), 4) if len(net.load) else 0.0,
        "connected": connected,
        "n_components": int(ncomp),
        "largest_comp_frac": round(comp_frac, 3),
        "radial_largest_comp": radial,
        "edges_minus_nodes": n_edges - (n_nodes - 1),
        "line_cap_coverage": round(line_capacity_coverage(net), 3),
        "has_load_profiles": has_profiles,
    }


def main() -> int:
    import simbench
    import pandapower.networks as ppn

    simbench_codes = [
        "1-LV-rural1--0-sw",
        "1-LV-rural2--0-sw",
        "1-LV-rural3--0-sw",
        "1-LV-urban6--0-sw",
        "1-MVLV-urban-5.303-0-sw",
        "1-MVLV-urban-6.305-0-sw",
        "1-MVLV-urban-6.309-0-sw",
    ]
    rows = {}
    # pandapower case33bw (current G_train anchor)
    try:
        rows["case33bw (pp)"] = summarize(ppn.case33bw())
    except Exception as e:  # noqa: BLE001
        rows["case33bw (pp)"] = {"error": repr(e)}

    for code in simbench_codes:
        try:
            net = simbench.get_simbench_net(code)
            rows[code] = summarize(net)
        except Exception as e:  # noqa: BLE001
            rows[code] = {"error": repr(e)}
            traceback.print_exc()

    cols = [
        "n_bus", "n_load", "n_line", "n_trafo", "n_ext_grid", "total_load_mw",
        "connected", "n_components", "largest_comp_frac", "radial_largest_comp",
        "edges_minus_nodes", "line_cap_coverage", "has_load_profiles",
    ]
    print("\n=== FEEDER VALIDATION ===")
    header = f"{'feeder':28} | " + " | ".join(f"{c}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, r in rows.items():
        if "error" in r:
            print(f"{name:28} | ERROR: {r['error']}")
            continue
        print(f"{name:28} | " + " | ".join(str(r[c]) for c in cols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
