#!/usr/bin/env python3
"""Document-stratified null model (replaces the unfair pooled null in network_null_model.py).

Finding that motivated it: every formal concept's extent lies inside ONE source document, and the two
FCA components are exactly the two documents (no entity occurs in both). A pooled random null therefore
bridges documents by construction. Here each concept keeps its size AND its document; members are drawn
uniformly from that document's entities. Membership links only (cover links dropped on both sides).
Seed 20261005, 2000 draws. p = share of null draws at least as extreme as observed, in the direction noted.
"""
import collections
import contextlib
import io
import json
import random
import statistics
import sys
from pathlib import Path

import networkx as nx

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
with contextlib.redirect_stdout(io.StringIO()):
    import network_analysis as na

SEED, DRAWS = 20261005, 2000
ent_doc = {}
for s in na.base["statements"]:
    for k in ("subject_id", "object_id"):
        ent_doc[s[k]] = s["document_id"]
pool = collections.defaultdict(list)
for e, d in sorted(ent_doc.items()):
    pool[d].append(e)
cdoc = [ent_doc[next(iter(c["extent"]))] for c in na.concepts]
sizes = [len(c["extent"]) for c in na.concepts]
assert all(len({ent_doc[e] for e in c["extent"]}) == 1 for c in na.concepts)


def measures(G):
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    L = G.subgraph(comps[0])
    return {"components": len(comps), "largest_frac": len(comps[0]) / G.number_of_nodes(),
            "diameter_lcc": nx.diameter(L), "avg_path_lcc": nx.average_shortest_path_length(L),
            "mean_degree": 2 * G.number_of_edges() / G.number_of_nodes(), "max_degree": max(d for _, d in G.degree())}


def build(extents):
    G = na.direct.copy()
    for i, ext in enumerate(extents):
        G.add_edges_from((f"concept:{i}", e) for e in ext)
    return G


obs = measures(build([c["extent"] for c in na.concepts]))
rng = random.Random(SEED)
runs = [measures(build([rng.sample(pool[d], s) for d, s in zip(cdoc, sizes)])) for _ in range(DRAWS)]
out = {"seed": SEED, "draws": DRAWS, "docs": {d: len(v) for d, v in pool.items()},
       "concept_docs": dict(collections.Counter(cdoc)), "observed": {k: round(v, 3) for k, v in obs.items()}, "null": {}}
for k in obs:
    v = [r[k] for r in runs]
    out["null"][k] = {"mean": round(statistics.mean(v), 3), "sd": round(statistics.pstdev(v), 3), "min": round(min(v), 3), "max": round(max(v), 3),
                      "share_null_le_obs": round(sum(x <= obs[k] for x in v) / DRAWS, 4), "share_null_ge_obs": round(sum(x >= obs[k] for x in v) / DRAWS, 4)}
(HERE / "network_null_model_by_document_results.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
