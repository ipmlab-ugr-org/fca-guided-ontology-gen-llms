# FCA-guided LLM ontology and knowledge graph construction

Code and results for the paper "Enhancing LLM-Based Ontology and Knowledge Graph Creation via Formal Concept Analysis for Wind-Turbine Maintenance: From Class Recognition to Source-Grounded Construction" (Khan, Bensalem, Chiachio-Ruano, Chiachio-Ruano).

Accepted at the Fifteenth International Conference on Complex Networks and their Applications (COMPLEX NETWORKS 2026), Granada; the proceedings are published by Springer.

IPM Lab, Universidad de Granada.

The study used confidential maintenance documents together with an NREL technical report and a Suzlon half-yearly maintenance checklist. The knowledge graph in this repository comes from the NREL and Suzlon documents. The confidential documents and their derived data are not shared, and no source documents are distributed.

## Contents

| Path | What it holds |
|---|---|
| `experiments/gpt-120b-oss/run_experiment.py` | Construction pilot: GPT-OSS 120B builds a knowledge graph with and without the FCA layer |
| `experiments/gpt-120b-oss/evaluate_qwen_full.py` | Graph-only question answering with Qwen 3.6 27B |
| `experiments/gpt-120b-oss/baseline_graph.json` | Extracted knowledge graph (direct condition), built from the NREL and Suzlon documents |
| `experiments/gpt-120b-oss/heldout_questions.json` | The 34 held-out questions |
| `experiments/gpt-120b-oss/full_qwen_fca_comparison.csv` | Per-question answers and scores, both conditions |
| `experiments/gpt-120b-oss/paired_stats.py` | Paired comparison and sign test |
| `experiments/gpt-120b-oss/retrieval_ablation.py` | Retrieval-level ablation |
| `experiments/gpt-120b-oss/network_analysis.py`, `network_extended.py` | Topology of the direct and FCA-backed graphs, hierarchy, robustness, communities |
| `experiments/gpt-120b-oss/network_null_model_by_document.py` | Null model: concept members drawn at random within their own source document |
| `experiments/gpt-120b-oss/*_results.json` | Saved outputs of the scripts above |
| `results/paper_tables/` | LaTeX tables of the paper (formal context, concepts, recognition results, question families) |

## Reproducing the network analysis

`network_analysis.py` rebuilds the FCA layer from `baseline_graph.json` and needs only `networkx`:

```
cd experiments/gpt-120b-oss
python3 network_analysis.py
python3 network_extended.py
python3 network_null_model_by_document.py
```

The FCA graph has 226 nodes and 404 edges; the direct graph has 205 nodes and 131 edges.

## Licence

Apache License 2.0, see `LICENSE`. Copyright 2026 IPM Lab, Universidad de Granada.

## Status

Code and results of the conference paper only. Extended experiments for the journal version will be added later.
