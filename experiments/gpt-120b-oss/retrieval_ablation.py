#!/usr/bin/env python3
"""Retrieval-reach analysis and concept ablation for the GPT-OSS 120B pilot (no LLM calls).

For each of the 34 held-out questions, run the pilot's own deterministic retrieve() over the direct
graph, the FCA-backed graph, and two ablated FCA graphs, and check whether the retrieved statements
contain each required answer atom (an upper bound on what any answerer can be right about).
FCA concept labels were written by the LLM and are not stored, so concept ids are used as labels;
this can shift retrieval slightly relative to the graph the answerer originally saw.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run_experiment as rx  # noqa: E402

rx.label_concepts = lambda model, concepts, model_dir: {}
base = json.loads((HERE / "baseline_graph.json").read_text())
questions = json.loads((HERE / "heldout_questions.json").read_text())
full = rx.enrich_fca_graph("rebuilt", base, HERE)
concepts = full["fca"]["concepts"]
broad = {f"concept:{c['concept_id']}" for c in concepts if len(c["extent"]) > 50}


def keep(graph, predicate):
    g = json.loads(json.dumps(graph))
    g["statements"] = [s for s in g["statements"] if predicate(s)]
    return g


touches = lambda s, ids: s["subject_id"] in ids or s["object_id"] in ids
graphs = {
    "direct": base,
    "fca_full": full,
    "fca_without_broad": keep(full, lambda s: not touches(s, broad)),
    "fca_only_broad": keep(full, lambda s: s.get("document_id") != "fca_derived" or touches(s, broad)),
}
derived = lambda s: s.get("document_id") == "fca_derived"


def text_of(stmts):
    return " ".join(rx.norm(" ".join([s["subject"], s["predicate"], s["object"], " ".join(s.get("qualifiers", []))])) for s in stmts)


def has_group(text, group):
    compact = text.replace(" ", "")
    return any((t := rx.norm(a)) and (t in text or t.replace(" ", "") in compact) for a in group)


rows, summary = [], {}
for name, g in graphs.items():
    cov, fullcov, size, share = [], 0, [], []
    for q in questions:
        sub = rx.retrieve(g, q["question"])
        text = text_of(sub)
        groups = q["required_answer_atoms"]
        c = sum(has_group(text, grp) for grp in groups) / len(groups)
        cov.append(c); fullcov += c == 1.0; size.append(len(sub))
        share.append(sum(derived(s) for s in sub) / len(sub) if sub else 0.0)
        rows.append({"condition": name, "question_id": q["id"], "evidence_atom_coverage": round(c, 3), "retrieved": len(sub), "fca_derived_share": round(share[-1], 3)})
    summary[name] = {"statements": len(g["statements"]), "mean_atom_coverage": round(sum(cov) / len(cov), 3), "questions_fully_covered": fullcov,
                     "mean_retrieved": round(sum(size) / len(size), 1), "mean_fca_derived_share": round(sum(share) / len(share), 3)}

# Link retrieval reach to the reported QA outcomes (original run, direct = baseline condition).
qa = {}
with open(HERE / "full_qwen_fca_comparison.csv") as f:
    for r in csv.DictReader(f):
        if r["construction_model"] == "openai/gpt-oss-120b":
            qa[(r["condition"], r["question_id"])] = int(r["exact_correct"])
cond_map = {"direct": "baseline", "fca_full": "fca"}
link = {}
for ours, theirs in cond_map.items():
    got = {r["question_id"]: r["evidence_atom_coverage"] for r in rows if r["condition"] == ours}
    right = [q for q in got if qa.get((theirs, q)) == 1]
    wrong = [q for q in got if qa.get((theirs, q)) == 0]
    link[ours] = {"correct_questions": len(right), "mean_coverage_when_correct": round(sum(got[q] for q in right) / max(1, len(right)), 3),
                  "mean_coverage_when_wrong": round(sum(got[q] for q in wrong) / max(1, len(wrong)), 3),
                  "correct_but_not_fully_covered": sum(got[q] < 1.0 for q in right)}
disagree = []
for q in questions:
    a, b = qa.get(("baseline", q["id"])), qa.get(("fca", q["id"]))
    if a != b:
        disagree.append({"id": q["id"], "direct_correct": a, "fca_correct": b,
                         "direct_cov": next(r["evidence_atom_coverage"] for r in rows if r["condition"] == "direct" and r["question_id"] == q["id"]),
                         "fca_cov": next(r["evidence_atom_coverage"] for r in rows if r["condition"] == "fca_full" and r["question_id"] == q["id"])})
out = {"summary": summary, "qa_link": link, "disagreeing_questions": disagree}
print(json.dumps(out, indent=1))
(HERE / "retrieval_ablation_results.json").write_text(json.dumps(out, indent=1))
with open(HERE / "retrieval_ablation_per_question.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
