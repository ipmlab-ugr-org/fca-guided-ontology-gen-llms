#!/usr/bin/env python3
"""Extended network analysis: hierarchy, robustness (random and targeted), communities, degree skew."""
import json
import random
import statistics
import sys
from pathlib import Path

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities, modularity

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import network_analysis as na  # noqa: E402  (rebuilds both graphs; prints its own JSON)

direct, fca, fg, concepts = na.direct, na.fca, na.fg, na.concepts


def lcc_frac(G, n0):
    return max((len(c) for c in nx.connected_components(G)), default=0) / n0


def gini(xs):
    xs = sorted(xs); n = len(xs); s = sum(xs)
    return round(sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * s), 3) if s else 0.0


out = {}
# Hierarchy: formal-subconcept cover relations among the 21 concepts.
H = nx.DiGraph((s["subject_id"], s["object_id"]) for s in fg["statements"] if s["predicate"] == "formal_subconcept_of")
out["hierarchy"] = {"cover_edges": H.number_of_edges(), "concepts_in_hierarchy": H.number_of_nodes(),
                    "longest_chain_edges": nx.dag_longest_path_length(H) if H.number_of_edges() else 0}
# Redundancy: statements that collapse to an already-present undirected edge.
for name, g, G in (("direct", na.base, direct), ("fca", fg, fca)):
    out[f"redundant_statements_{name}"] = len(g["statements"]) - G.number_of_edges()
# Degree skew.
for name, G in (("direct", direct), ("fca", fca)):
    d = [x for _, x in G.degree()]
    out[f"degree_{name}"] = {"gini": gini(d), "median": statistics.median(d), "isolated_nodes": sum(1 for x in d if x == 0)}
# Communities (greedy modularity) on each graph.
for name, G in (("direct", direct), ("fca", fca)):
    comms = list(greedy_modularity_communities(G))
    out[f"communities_{name}"] = {"count_nonsingleton": sum(len(c) > 1 for c in comms), "modularity": round(modularity(G, comms), 3)}
# Robustness: remove k% of nodes at random (mean of 200 trials) and by degree (targeted, concepts included).
random.seed(7)
for name, G in (("direct", direct), ("fca", fca)):
    n0 = G.number_of_nodes(); res = {}
    for pct in (5, 10, 20):
        k = round(n0 * pct / 100)
        vals = []
        for _ in range(200):
            Hh = G.copy(); Hh.remove_nodes_from(random.sample(list(G.nodes()), k)); vals.append(lcc_frac(Hh, n0))
        Hh = G.copy(); Hh.remove_nodes_from([n for n, _ in sorted(G.degree(), key=lambda x: -x[1])[:k]])
        res[f"{pct}pct"] = {"random_lcc_frac": round(statistics.mean(vals), 3), "targeted_lcc_frac": round(lcc_frac(Hh, n0), 3)}
    out[f"robustness_{name}"] = res
# Targeted attack on the two broad concepts only.
Hh = fca.copy(); Hh.remove_nodes_from([f"concept:{c['concept_id']}" for c in concepts if len(c["extent"]) > 50])
out["fca_after_removing_two_broad_hubs_lcc_frac"] = round(lcc_frac(Hh, fca.number_of_nodes()), 3)
print("=== EXTENDED ===")
print(json.dumps(out, indent=1))
(HERE / "network_extended_results.json").write_text(json.dumps(out, indent=1))
