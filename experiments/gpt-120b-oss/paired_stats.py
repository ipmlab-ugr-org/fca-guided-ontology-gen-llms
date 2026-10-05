#!/usr/bin/env python3
"""Paired statistics for the 34-question comparison (direct vs FCA-backed), from the stored per-question CSV."""
import csv, json, math, random
from pathlib import Path
HERE = Path(__file__).parent
rows = [r for r in csv.DictReader(open(HERE / "full_qwen_fca_comparison.csv")) if r["construction_model"] == "openai/gpt-oss-120b"]
by = {(r["condition"], r["question_id"]): r for r in rows}
qs = sorted({r["question_id"] for r in rows})
def sign_p(d):
    nz = [x for x in d if x]; n = len(nz); k = min(sum(x > 0 for x in nz), sum(x < 0 for x in nz))
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
random.seed(11); out = {}
for m in ("exact_correct", "grounded", "valid_citation", "atom_recall", "unsupported"):
    d = [float(by[("fca", q)][m]) - float(by[("baseline", q)][m]) for q in qs]
    boots = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(5000))
    out[m] = {"fca_wins": sum(x > 0 for x in d), "direct_wins": sum(x < 0 for x in d), "ties": sum(x == 0 for x in d),
              "mean_diff": round(sum(d) / len(d), 3), "boot95": [round(boots[125], 3), round(boots[4875], 3)], "sign_test_p": round(sign_p(d), 3)}
print(json.dumps(out, indent=1)); (HERE / "paired_stats_results.json").write_text(json.dumps(out, indent=1))
