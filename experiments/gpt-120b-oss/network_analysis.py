#!/usr/bin/env python3
"""Topology of the direct versus FCA-backed knowledge graph (GPT-OSS 120B pilot).

The FCA layer is rebuilt deterministically from baseline_graph.json using the same
selected_formal_concepts() routine as run_experiment.py. Concept labels (LLM-written)
do not affect topology, so concept ids are used as labels. Graphs are analysed as
undirected simple graphs over node ids.
"""
import json
import statistics
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import selected_formal_concepts  # noqa: E402

HERE = Path(__file__).parent
base = json.loads((HERE / "baseline_graph.json").read_text())


def fca_graph(base):
    g = json.loads(json.dumps(base))
    concepts = selected_formal_concepts(base)
    for c in concepts:
        g["nodes"].append({"node_id": f"concept:{c['concept_id']}", "type": "fca_formal_concept"})
        for e in c["extent"]:
            g["statements"].append({"subject_id": e, "object_id": f"concept:{c['concept_id']}", "predicate": "member_of_formal_concept"})
    for ch in concepts:
        for pa in concepts:
            if ch is pa:
                continue
            ce, pe, ci, pi = set(ch["extent"]), set(pa["extent"]), set(ch["intent"]), set(pa["intent"])
            if ce < pe and pi < ci and not any(ce < set(m["extent"]) < pe and pi < set(m["intent"]) < ci for m in concepts if m is not ch and m is not pa):
                g["statements"].append({"subject_id": f"concept:{ch['concept_id']}", "object_id": f"concept:{pa['concept_id']}", "predicate": "formal_subconcept_of"})
    return g, concepts


def to_nx(g):
    G = nx.Graph()
    G.add_nodes_from(n["node_id"] for n in g["nodes"])
    G.add_edges_from((s["subject_id"], s["object_id"]) for s in g["statements"] if s["subject_id"] != s["object_id"])
    return G


def metrics(G):
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    L = G.subgraph(comps[0])
    deg = dict(G.degree())
    bc = nx.betweenness_centrality(L)
    top = max(deg.values())
    return {
        "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
        "density": round(nx.density(G), 4), "mean_degree": round(statistics.mean(deg.values()), 2),
        "max_degree": top, "components": len(comps), "largest_component": len(comps[0]),
        "largest_frac": round(len(comps[0]) / G.number_of_nodes(), 3),
        "isolated_pairs_reachable": round(sum(len(c) * (len(c) - 1) for c in comps) / (G.number_of_nodes() * (G.number_of_nodes() - 1)), 3),
        "avg_shortest_path_lcc": round(nx.average_shortest_path_length(L), 2),
        "diameter_lcc": nx.diameter(L),
        "clustering": round(nx.average_clustering(G), 3),
        "max_betweenness_lcc": round(max(bc.values()), 3),
    }


direct = to_nx(base)
fg, concepts = fca_graph(base)
fca = to_nx(fg)
out = {"direct": metrics(direct), "fca": metrics(fca), "n_concepts": len(concepts), "statements_fca": len(fg["statements"]),
       "concept_extent_sizes": sorted(len(c["extent"]) for c in concepts)}
# Robustness: remove 10 highest-degree entity nodes (hubs) and measure largest-component fraction.
for name, G in (("direct", direct), ("fca", fca)):
    H = G.copy()
    hubs = [n for n, _ in sorted(H.degree(), key=lambda x: -x[1]) if not n.startswith("concept:")][:10]
    H.remove_nodes_from(hubs)
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    out[f"hubremoval10_{name}"] = {"largest_frac": round(len(comps[0]) / H.number_of_nodes(), 3), "components": len(comps)}
# Concept-node ablation: dropping all concept nodes from the FCA graph must recover the direct graph.
H = fca.copy(); H.remove_nodes_from([n for n in fca if n.startswith("concept:")])
out["ablate_concepts_recovers_direct"] = (set(H.edges()) == set(direct.edges()) and H.number_of_nodes() == direct.number_of_nodes())
# Ablation: drop the two broad concepts (extent > 50) and keep the 19 narrow ones.
big = [f"concept:{c['concept_id']}" for c in concepts if len(c["extent"]) > 50]
H = fca.copy(); H.remove_nodes_from(big)
comps = sorted(nx.connected_components(H), key=len, reverse=True)
out["drop_broad_concepts"] = {"removed": len(big), "components": len(comps), "largest_component": len(comps[0]), "largest_frac": round(len(comps[0]) / H.number_of_nodes(), 3), "edges": H.number_of_edges()}
H = fca.copy(); H.remove_nodes_from([f"concept:{c['concept_id']}" for c in concepts if len(c["extent"]) <= 50])
comps = sorted(nx.connected_components(H), key=len, reverse=True)
out["keep_only_broad_concepts"] = {"components": len(comps), "largest_component": len(comps[0]), "largest_frac": round(len(comps[0]) / H.number_of_nodes(), 3)}
print(json.dumps(out, indent=2))
(HERE / "network_analysis_results.json").write_text(json.dumps(out, indent=2))
