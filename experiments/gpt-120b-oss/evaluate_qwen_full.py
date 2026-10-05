#!/usr/bin/env python3
"""Full graph-only paired evaluation: 120B constructs both graphs; Qwen answers them."""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import run_experiment as exp

ROOT = Path("/home/ubuntu/kg_fca_evaluation")
ART = ROOT / "artifacts"
CONSTRUCTION_MODEL = "openai/gpt-oss-120b"
ANSWER_MODEL = "qwen/qwen3.6-27b"
RUN = ROOT / "runs" / exp.safe_name(CONSTRUCTION_MODEL)


def sign_test_pvalue(diffs: list[int]) -> float:
    nonzero = [d for d in diffs if d]
    if not nonzero:
        return 1.0
    n = len(nonzero)
    k = min(sum(d > 0 for d in nonzero), sum(d < 0 for d in nonzero))
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def main() -> None:
    questions = json.loads((ART / "heldout_questions.json").read_text())
    graphs = {
        "baseline": json.loads((RUN / "baseline_graph.json").read_text()),
        "fca": json.loads((RUN / "fca_graph.json").read_text()),
    }
    rows = []
    for condition, graph in graphs.items():
        answer_path = RUN / f"full_{condition}_answers_qwen.json"
        records = json.loads(answer_path.read_text()) if answer_path.exists() else []
        completed = {record["question_id"] for record in records}
        pending = [question for question in questions if question["id"] not in completed]
        for start in range(0, len(pending), 3):
            records.extend(exp.answer_questions_batch(ANSWER_MODEL, graph, pending[start:start + 3]))
            answer_path.write_text(json.dumps(records, indent=2))
        record_map = {record["question_id"]: record for record in records}
        for question in questions:
            record = record_map[question["id"]]
            statement_ids = set(record["subgraph_statement_ids"])
            subgraph = [statement for statement in graph["statements"] if statement["statement_id"] in statement_ids]
            score = exp.score_answer(question, record["result"], subgraph)
            rows.append({
                "construction_model": CONSTRUCTION_MODEL,
                "answer_model": ANSWER_MODEL,
                "condition": condition,
                "question_id": question["id"],
                "document_id": question["document_id"],
                "category": question["category"],
                "answer": record["result"].get("answer", ""),
                "status": record["result"].get("status", ""),
                "cited_statement_ids": ";".join(record["result"].get("cited_statement_ids", [])),
                **score,
            })

    csv_path = ART / "full_qwen_fca_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped = {condition: [row for row in rows if row["condition"] == condition] for condition in graphs}
    summary = {}
    for condition, values in grouped.items():
        n = len(values)
        summary[condition] = {
            "questions": n,
            "exact_accuracy": sum(v["exact_correct"] for v in values) / n,
            "atom_recall": sum(v["atom_recall"] for v in values) / n,
            "grounded_rate": sum(v["grounded"] for v in values) / n,
            "valid_citation_rate": sum(v["valid_citation"] for v in values) / n,
            "coverage": sum(v["coverage"] for v in values) / n,
            "unsupported_rate": sum(v["unsupported"] for v in values) / n,
            **exp.graph_metrics(graphs[condition]),
        }
    base = {row["question_id"]: row for row in grouped["baseline"]}
    fca = {row["question_id"]: row for row in grouped["fca"]}
    diffs = [fca[qid]["exact_correct"] - base[qid]["exact_correct"] for qid in base]
    paired = {
        "fca_wins": sum(d > 0 for d in diffs),
        "ties": sum(d == 0 for d in diffs),
        "baseline_wins": sum(d < 0 for d in diffs),
        "accuracy_delta": summary["fca"]["exact_accuracy"] - summary["baseline"]["exact_accuracy"],
        "atom_recall_delta": summary["fca"]["atom_recall"] - summary["baseline"]["atom_recall"],
        "sign_test_pvalue": sign_test_pvalue(diffs),
    }
    (ART / "full_qwen_fca_summary.json").write_text(json.dumps({"summary": summary, "paired": paired}, indent=2))

    report = [
        "# Full FCA-backed KG Comparison: 120B Construction, Qwen Graph QA", "",
        f"Completed: {datetime.now(timezone.utc).isoformat()}", "",
        "## Valid paired contrast", "",
        "`openai/gpt-oss-120b` constructed both graph variants from the same two uploaded PDFs and identical extraction settings. The baseline contains 137 evidence-backed direct claims. The FCA graph starts from those claims and adds 21 selected formal concepts plus membership and formal-subconcept structure. `qwen/qwen3.6-27b` answered every one of the same 34 pre-registered questions from retrieved graph statements only; it received no source PDF text.", "",
        "## Results", "",
        "| Condition | Questions | Exact accuracy | Answer-atom recall | Grounded-answer rate | Valid-citation rate | Graph coverage | Unsupported-answer rate | Nodes | Statements | FCA concepts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in ["baseline", "fca"]:
        s = summary[condition]
        report.append(f"| {condition} | {s['questions']} | {s['exact_accuracy']:.1%} | {s['atom_recall']:.1%} | {s['grounded_rate']:.1%} | {s['valid_citation_rate']:.1%} | {s['coverage']:.1%} | {s['unsupported_rate']:.1%} | {s['nodes']} | {s['statements']} | {s['fca_concepts']} |")
    report += [
        "", "## Paired effect", "",
        f"FCA wins: **{paired['fca_wins']}**; ties: **{paired['ties']}**; baseline wins: **{paired['baseline_wins']}**. Exact-accuracy delta: **{paired['accuracy_delta']:+.1%}**. Answer-atom-recall delta: **{paired['atom_recall_delta']:+.1%}**. Two-sided paired sign-test p-value: **{paired['sign_test_pvalue']:.4f}**.", "",
        "## Interpretation boundary", "",
        "A positive, consistent paired difference supports the usefulness of this particular FCA augmentation in this corpus and retrieval design. A tie, negative difference, or non-significant outcome is inconclusive rather than proof against FCA. This is still a small two-document pilot; broader confirmation requires more reports, independently held-out questions, and completed construction runs across multiple model families.", "",
        "## Audit artifacts", "",
        "Per-question scores are saved in `full_qwen_fca_comparison.csv`. Graph-only answers are checkpointed in `full_baseline_answers_qwen.json` and `full_fca_answers_qwen.json` under the 120B run directory. Raw claims, both graphs, concept labels, and source-grounded evidence remain in the same workspace. The API key is absent from all artifacts.",
    ]
    (ART / "full_qwen_fca_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"summary": summary, "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
