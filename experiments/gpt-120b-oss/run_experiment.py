#!/usr/bin/env python3
"""Paired pilot: conventional LLM KG versus FCA-backed LLM KG.

The construction routine deliberately does not read heldout_questions.json.  The QA
routine receives retrieved graph statements only, never PDF text or stored quotes.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

ROOT = Path("/home/ubuntu/kg_fca_evaluation")
TEXT_DIR = ROOT / "extracted_text"
ARTIFACT_DIR = ROOT / "artifacts"
RUNS_DIR = ROOT / "runs"
KEY_PATH = Path("/home/ubuntu/.secrets/groq_evaluation_key")
API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Use the two production models that support strict structured outputs. The account also exposes a preview Qwen model, but it lacks strict-schema support and is excluded to preserve a controlled comparison.
MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
SEED = 270818
MAX_RETRIES = 5

DOCUMENTS = {
    "nrel_82704": {
        "title": "NREL gearbox maintenance report",
        "path": TEXT_DIR / "nrel_82704_layout.txt",
    },
    "suzlon_wdoq179": {
        "title": "Suzlon half-yearly WTG maintenance checklist",
        "path": TEXT_DIR / "suzlon_checklist_layout.txt",
    },
}

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "subject_type": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "object_type": {"type": "string"},
                    "qualifiers": {"type": "array", "items": {"type": "string"}},
                    "evidence_quote": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "subject", "subject_type", "predicate", "object", "object_type",
                    "qualifiers", "evidence_quote", "confidence",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept_id": {"type": "string"},
                    "label": {"type": "string"},
                    "definition": {"type": "string"},
                },
                "required": ["concept_id", "label", "definition"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "status": {"type": "string", "enum": ["answered", "insufficient_graph_evidence"]},
        "cited_statement_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "status", "cited_statement_ids"],
    "additionalProperties": False,
}


BATCH_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_id": {"type": "string"},
                    "answer": {"type": "string"},
                    "status": {"type": "string", "enum": ["answered", "insufficient_graph_evidence"]},
                    "cited_statement_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["question_id", "answer", "status", "cited_statement_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["answers"],
    "additionalProperties": False,
}


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def norm(value: str) -> str:
    value = value.lower().replace("²", "2").replace("∙", " ")
    value = value.replace("°", " ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokens(value: str) -> set[str]:
    stop = {"the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "is", "was", "are", "be", "what", "which", "how", "when", "with", "that", "this", "it", "as", "by", "from", "during", "does", "do", "were"}
    return {t for t in norm(value).split() if t not in stop and len(t) > 1}


def page_chunks(path: Path, document_id: str) -> list[dict[str, Any]]:
    """Create page-aware extraction windows bounded for the account's token rate limit."""
    raw_pages = path.read_text(encoding="utf-8", errors="ignore").split("\f")
    page_records: list[tuple[int, str]] = []
    for page_number, text in enumerate(raw_pages, start=1):
        cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(cleaned) >= 80:
            page_records.append((page_number, cleaned))
    chunks: list[dict[str, Any]] = []
    current: list[tuple[int, str]] = []
    current_length = 0
    max_chars = 11000
    for page_number, cleaned in page_records:
        if current and current_length + len(cleaned) > max_chars:
            first, last = current[0][0], current[-1][0]
            chunks.append({
                "document_id": document_id,
                "page": str(first) if first == last else f"{first}-{last}",
                "chunk_index": len(chunks) + 1,
                "text": "\n\n".join(f"[PDF page {p}]\n{text}" for p, text in current),
            })
            current, current_length = [], 0
        current.append((page_number, cleaned))
        current_length += len(cleaned)
    if current:
        first, last = current[0][0], current[-1][0]
        chunks.append({
            "document_id": document_id,
            "page": str(first) if first == last else f"{first}-{last}",
            "chunk_index": len(chunks) + 1,
            "text": "\n\n".join(f"[PDF page {p}]\n{text}" for p, text in current),
        })
    return chunks


def load_api_key() -> str:
    key = KEY_PATH.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError("Groq API key file is empty")
    return key


def parse_json_content(content: str) -> dict[str, Any]:
    """Parse the first JSON object, tolerating stray trailing model text."""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", content):
        try:
            parsed, _ = decoder.raw_decode(content[match.start():])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("No JSON object found", content, 0)


def groq_call(model: str, system: str, user: str, schema: dict[str, Any], max_tokens: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call Groq with prompt-directed JSON and locally parse the returned object."""
    key = load_api_key()
    # Groq's reasoning documentation recommends embedding task instructions in the user prompt for GPT-OSS models.
    combined_prompt = system + "\n\n" + user + "\n\nReturn a single valid JSON object that follows the requested schema exactly; do not add Markdown fences or commentary."
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": combined_prompt}],
        "temperature": 0.6,
        "max_completion_tokens": max_tokens,
        "seed": SEED,
    }
    if model.startswith("openai/gpt-oss"):
        payload["include_reasoning"] = False
        payload["reasoning_effort"] = "low"
    elif model.startswith("qwen/"):
        payload["reasoning_effort"] = "none"
        payload["reasoning_format"] = "hidden"

    last_error = "unknown"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=180,
            )
            if response.status_code in {429, 500, 502, 503, 504}:
                last_error = f"HTTP {response.status_code}: {response.text[:2000]}"
                time.sleep(min(20, 2 ** attempt))
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:4000]}")
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"] or "{}"
            return parse_json_content(content), body
        except Exception as exc:  # API or parse failure; retry with bounded backoff.
            last_error = repr(exc)
            if attempt < MAX_RETRIES:
                time.sleep(min(20, 2 ** attempt))
    raise RuntimeError(f"Groq call failed for {model}: {last_error}")


def normalize_qualifiers(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, dict)):
        values = [values]
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for value in values:
        if isinstance(value, dict):
            normalized.extend(f"{safe_name(str(key))}:{str(item)}" for key, item in value.items() if str(item).strip())
        elif str(value).strip():
            normalized.append(str(value).strip())
    return normalized


def claim_items_from_response(parsed: dict[str, Any], raw_content: str) -> list[dict[str, Any]]:
    """Accept claims/facts envelopes and recover complete item objects from a truncated array."""
    for key in ("claims", "facts", "triples", "relations", "items"):
        values = parsed.get(key)
        if isinstance(values, list):
            return [value for value in values if isinstance(value, dict)]
    if all(key in parsed for key in ("subject", "predicate")) and ("object" in parsed or "value" in parsed):
        # The outer array can be truncated; parsing may have found only the first complete fact object.
        recovered: list[dict[str, Any]] = []
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", raw_content):
            try:
                value, _ = decoder.raw_decode(raw_content[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and all(key in value for key in ("subject", "predicate")) and ("object" in value or "value" in value):
                recovered.append(value)
        return recovered or [parsed]
    return []


def recover_evidence_quote(source_text: str, subject: str, object_: str) -> str:
    """Select a compact source sentence or line with the greatest entity-token overlap."""
    candidates = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+|\n+", source_text) if len(piece.strip()) >= 20]
    target_tokens = tokens(subject) | tokens(object_)
    scored = [(len(target_tokens & tokens(candidate)), candidate) for candidate in candidates]
    scored.sort(key=lambda item: (-item[0], len(item[1])))
    return scored[0][1][:600] if scored and scored[0][0] else source_text[:600].strip()


def extraction_prompt(chunk: dict[str, Any]) -> tuple[str, str]:
    system = """You are a meticulous knowledge-graph fact extractor for wind-turbine technical documents.
Use only the supplied text. Extract atomic, directly supported facts useful for maintenance, reliability, operational history, component hierarchy, procedures, numeric measurements, dates, thresholds, and conditions. Do not infer causal relations, complete missing information, or use external knowledge.

Use short, readable predicate names in lower_snake_case. Preserve names, values, units, and identifiers as written. A claim must have a verb-like predicate and a source quote that exactly appears in or is a short faithful fragment of the supplied text. Include the relevant action, prerequisite, expected outcome, and interval as separate facts when applicable. Do not output document boilerplate or references."""
    user = f"""Document: {chunk['document_id']}
Physical PDF page: {chunk['page']}
Page chunk: {chunk['chunk_index']}

Extract at most 20 atomic claims from this text:
---
{chunk['text']}
---

Return JSON matching the schema. For qualifiers, use compact strings such as `condition:WTG idling slowly`, `unit:ppm`, `procedure:WD00030`, or `frequency:half-yearly`."""
    return system, user


def build_raw_claims(model: str, model_dir: Path) -> list[dict[str, Any]]:
    raw_path = model_dir / "raw_claims.json"
    if raw_path.exists():
        return json.loads(raw_path.read_text())
    claims: list[dict[str, Any]] = []
    api_log: list[dict[str, Any]] = []
    for document_id, meta in DOCUMENTS.items():
        for chunk in page_chunks(meta["path"], document_id):
            system, user = extraction_prompt(chunk)
            result, response = groq_call(model, system, user, EXTRACTION_SCHEMA, max_tokens=1700)
            api_log.append({
                "stage": "extract", "document_id": document_id, "page": chunk["page"],
                "chunk_index": chunk["chunk_index"], "usage": response.get("usage", {}),
                "response_id": response.get("id", ""),
            })
            raw_content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            for seq, item in enumerate(claim_items_from_response(result, raw_content), start=1):
                object_value = item.get("object", item.get("value", ""))
                if not isinstance(item, dict) or not str(item.get("subject", "")).strip() or not str(item.get("predicate", "")).strip() or not str(object_value).strip():
                    continue
                subject = str(item["subject"]).strip()
                object_ = str(object_value).strip()
                evidence_quote = str(item.get("evidence_quote", item.get("source", ""))).strip() or recover_evidence_quote(chunk["text"], subject, object_)
                confidence_value = item.get("confidence", 0.70)
                try:
                    confidence = float(confidence_value)
                except (TypeError, ValueError):
                    confidence = 0.70
                claim = {
                    "claim_id": f"{document_id}_p{chunk['page']}_c{chunk['chunk_index']}_{seq}",
                    "document_id": document_id,
                    "page": chunk["page"],
                    "chunk_index": chunk["chunk_index"],
                    "subject": subject,
                    "subject_type": str(item.get("subject_type", "entity")).strip() or "entity",
                    "predicate": safe_name(str(item["predicate"])) or "related_to",
                    "object": object_,
                    "object_type": str(item.get("object_type", "entity")).strip() or "entity",
                    "qualifiers": normalize_qualifiers(item.get("qualifiers", [])),
                    "evidence_quote": evidence_quote,
                    "confidence": max(0.0, min(1.0, confidence)),
                }
                claims.append(claim)
    raw_path.write_text(json.dumps(claims, indent=2))
    (model_dir / "api_usage_extraction.json").write_text(json.dumps(api_log, indent=2))
    return claims


def node_id(label: str, kind: str) -> str:
    digest = hashlib.sha1(f"{kind}|{norm(label)}".encode()).hexdigest()[:12]
    return f"{kind}:{digest}"


def base_graph(claims: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    statements: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, tuple[str, ...], str]] = set()
    for claim in claims:
        sid = node_id(claim["subject"], claim["subject_type"])
        oid = node_id(claim["object"], claim["object_type"])
        nodes.setdefault(sid, {"node_id": sid, "label": claim["subject"], "type": claim["subject_type"]})
        nodes.setdefault(oid, {"node_id": oid, "label": claim["object"], "type": claim["object_type"]})
        fingerprint = (sid, claim["predicate"], oid, tuple(sorted(norm(q) for q in claim["qualifiers"])), claim["document_id"])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        statements.append({
            "statement_id": claim["claim_id"],
            "subject_id": sid,
            "subject": claim["subject"],
            "subject_type": claim["subject_type"],
            "predicate": claim["predicate"],
            "object_id": oid,
            "object": claim["object"],
            "object_type": claim["object_type"],
            "qualifiers": claim["qualifiers"],
            "document_id": claim["document_id"],
            "page": claim["page"],
            "confidence": claim["confidence"],
        })
    return {"nodes": list(nodes.values()), "statements": statements, "fca": {"concepts": []}}


def entity_attributes(graph: dict[str, Any]) -> dict[str, set[str]]:
    attrs: dict[str, set[str]] = defaultdict(set)
    for node in graph["nodes"]:
        attrs[node["node_id"]].add(f"type:{safe_name(node['type'])}")
    for statement in graph["statements"]:
        s, o = statement["subject_id"], statement["object_id"]
        pred = statement["predicate"]
        attrs[s].add(f"out:{pred}")
        attrs[s].add(f"out_type:{pred}:{safe_name(statement['object_type'])}")
        attrs[o].add(f"in:{pred}")
        attrs[o].add(f"in_type:{pred}:{safe_name(statement['subject_type'])}")
        for qualifier in statement["qualifiers"]:
            key = safe_name(qualifier.split(":", 1)[0])
            if key:
                attrs[s].add(f"qualifier:{key}")
        attrs[s].add(f"document:{statement['document_id']}")
        attrs[o].add(f"document:{statement['document_id']}")
    return attrs


def closure(seed: frozenset[str], objects: list[str], attrs: dict[str, set[str]]) -> tuple[frozenset[str], frozenset[str]]:
    extent = frozenset(obj for obj in objects if seed.issubset(attrs[obj]))
    if not extent:
        return extent, frozenset()
    intent = set(attrs[next(iter(extent))])
    for obj in extent:
        intent.intersection_update(attrs[obj])
    return extent, frozenset(intent)


def selected_formal_concepts(graph: dict[str, Any], min_support: int = 2, max_attrs: int = 50, max_concepts: int = 25) -> list[dict[str, Any]]:
    """Enumerate closures seeded by empty, one-attribute, and two-attribute sets.

    Each retained pair is a valid formal concept because it is closed by the closure operator.
    The bounded seed set is intentional: an unrestricted lattice can explode on noisy LLM facts.
    """
    attrs = entity_attributes(graph)
    objects = sorted(attrs)
    freq = Counter(a for values in attrs.values() for a in values)
    vocabulary = [a for a, count in freq.most_common(max_attrs) if count >= min_support]
    candidates: dict[tuple[frozenset[str], frozenset[str]], None] = {}
    seeds: list[frozenset[str]] = [frozenset()]
    seeds.extend(frozenset([a]) for a in vocabulary)
    for i, a in enumerate(vocabulary):
        for b in vocabulary[i + 1:]:
            seeds.append(frozenset([a, b]))
    for seed in seeds:
        extent, intent = closure(seed, objects, attrs)
        if len(extent) >= min_support and len(intent) >= 2:
            candidates[(extent, intent)] = None
    concepts = []
    node_lookup = {n["node_id"]: n for n in graph["nodes"]}
    for extent, intent in candidates:
        score = len(extent) * max(1, len(intent) - 1)
        concepts.append({
            "extent": sorted(extent),
            "intent": sorted(intent),
            "support": len(extent),
            "score": score,
            "examples": [node_lookup[e]["label"] for e in sorted(extent)[:5]],
        })
    concepts.sort(key=lambda c: (-c["score"], -len(c["intent"]), -c["support"], c["intent"]))
    # Retain nontrivial concepts, excluding the universal root where no attribute distinction exists.
    concepts = [c for c in concepts if c["support"] < len(objects) or len(c["intent"]) > 2][:max_concepts]
    for idx, concept in enumerate(concepts, start=1):
        concept["concept_id"] = f"fca_concept_{idx:03d}"
    return concepts


def label_concepts(model: str, concepts: list[dict[str, Any]], model_dir: Path) -> dict[str, dict[str, str]]:
    path = model_dir / "fca_concept_labels.json"
    if path.exists():
        return {x["concept_id"]: x for x in json.loads(path.read_text()).get("labels", [])}
    if not concepts:
        path.write_text(json.dumps({"labels": []}, indent=2))
        return {}
    cards = [{"concept_id": c["concept_id"], "intent": c["intent"], "support": c["support"], "examples": c["examples"]} for c in concepts]
    system = """You are an ontology engineer. Name only the supplied FCA concepts. A concept is defined by its shared structural attributes (intent) and example entities (extent). Produce a concise readable label and a one-sentence definition based solely on each concept card. Do not invent technical facts or merge different concepts. If no meaningful semantic name is justified, use a conservative structural label."""
    user = "Formal-concept cards:\n" + json.dumps(cards, indent=2)
    result, response = groq_call(model, system, user, LABEL_SCHEMA, max_tokens=1800)
    valid_ids = {c["concept_id"] for c in concepts}
    labels = [x for x in result.get("labels", []) if x.get("concept_id") in valid_ids]
    path.write_text(json.dumps({"labels": labels, "usage": response.get("usage", {})}, indent=2))
    return {x["concept_id"]: x for x in labels}


def enrich_fca_graph(model: str, base: dict[str, Any], model_dir: Path) -> dict[str, Any]:
    concepts = selected_formal_concepts(base)
    labels = label_concepts(model, concepts, model_dir)
    graph = json.loads(json.dumps(base))
    graph["fca"]["concepts"] = concepts
    existing_ids = {node["node_id"] for node in graph["nodes"]}
    for concept in concepts:
        label_info = labels.get(concept["concept_id"], {})
        label = label_info.get("label") or concept["concept_id"]
        cid = f"concept:{concept['concept_id']}"
        graph["nodes"].append({"node_id": cid, "label": label, "type": "fca_formal_concept", "definition": label_info.get("definition", "")})
        for entity_id in concept["extent"]:
            statement_id = f"{concept['concept_id']}__member__{entity_id.split(':')[-1]}"
            entity_label = next(n["label"] for n in graph["nodes"] if n["node_id"] == entity_id)
            graph["statements"].append({
                "statement_id": statement_id, "subject_id": entity_id, "subject": entity_label,
                "subject_type": "entity", "predicate": "member_of_formal_concept", "object_id": cid,
                "object": label, "object_type": "fca_formal_concept", "qualifiers": [f"intent:{'|'.join(concept['intent'])}", f"support:{concept['support']}"],
                "document_id": "fca_derived", "page": 0, "confidence": 1.0,
            })
    # Add formal-concept specialization edges based on extents and intents.
    for child in concepts:
        for parent in concepts:
            if child["concept_id"] == parent["concept_id"]:
                continue
            child_extent, parent_extent = set(child["extent"]), set(parent["extent"])
            child_intent, parent_intent = set(child["intent"]), set(parent["intent"])
            if child_extent < parent_extent and parent_intent < child_intent:
                # Cover relation only: do not add if an intermediate selected concept exists.
                intermediate = any(
                    child_extent < set(mid["extent"]) < parent_extent and parent_intent < set(mid["intent"]) < child_intent
                    for mid in concepts
                    if mid["concept_id"] not in {child["concept_id"], parent["concept_id"]}
                )
                if not intermediate:
                    child_id = f"concept:{child['concept_id']}"
                    parent_id = f"concept:{parent['concept_id']}"
                    child_label = labels.get(child["concept_id"], {}).get("label", child["concept_id"])
                    parent_label = labels.get(parent["concept_id"], {}).get("label", parent["concept_id"])
                    graph["statements"].append({
                        "statement_id": f"{child['concept_id']}__subconcept__{parent['concept_id']}",
                        "subject_id": child_id, "subject": child_label, "subject_type": "fca_formal_concept",
                        "predicate": "formal_subconcept_of", "object_id": parent_id, "object": parent_label,
                        "object_type": "fca_formal_concept", "qualifiers": [], "document_id": "fca_derived", "page": 0, "confidence": 1.0,
                    })
    return graph


def statement_text(statement: dict[str, Any]) -> str:
    qualifiers = "; ".join(statement.get("qualifiers", []))
    suffix = f" [{qualifiers}]" if qualifiers else ""
    return f"{statement['statement_id']} | {statement['subject']} --{statement['predicate']}--> {statement['object']}{suffix} | source={statement['document_id']}:p{statement['page']}"


def retrieve(graph: dict[str, Any], question: str, top_k: int = 10, expansion_limit: int = 6) -> list[dict[str, Any]]:
    q_tokens = tokens(question)
    scored: list[tuple[float, dict[str, Any]]] = []
    for statement in graph["statements"]:
        fields = " ".join([statement["subject"], statement["predicate"], statement["object"], " ".join(statement.get("qualifiers", [])), statement.get("subject_type", ""), statement.get("object_type", "")])
        overlap = len(q_tokens & tokens(fields))
        phrase_bonus = 0.5 if norm(statement["object"]) in norm(question) or norm(statement["subject"]) in norm(question) else 0.0
        if overlap:
            scored.append((overlap + phrase_bonus, statement))
    scored.sort(key=lambda x: (-x[0], x[1]["statement_id"]))
    selected = [s for _, s in scored[:top_k]]
    if not selected:
        return []
    node_ids = {s["subject_id"] for s in selected} | {s["object_id"] for s in selected}
    additions = []
    for statement in graph["statements"]:
        if len(additions) >= expansion_limit:
            break
        if statement in selected:
            continue
        if statement["subject_id"] in node_ids or statement["object_id"] in node_ids:
            additions.append(statement)
    result = {s["statement_id"]: s for s in selected + additions}
    return [result[k] for k in sorted(result)]


def answer_questions_batch(model: str, graph: dict[str, Any], questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = [(question, retrieve(graph, question["question"])) for question in questions]
    blocks = []
    for question, subgraph in prepared:
        serial = "\n".join(statement_text(statement) for statement in subgraph) if subgraph else "(no retrieved graph statements)"
        blocks.append(f"QUESTION_ID: {question['id']}\nQUESTION: {question['question']}\nSTATEMENTS:\n{serial}")
    system = """You answer technical questions using only the supplied graph statements. The question blocks are independent. Do not use outside knowledge, memory of documents, or unstated assumptions. Give a concise direct answer only when graph statements support all material parts; otherwise use status `insufficient_graph_evidence`. Cite only statement identifiers that directly support the answer. Return exactly one object in `answers` for every QUESTION_ID."""
    user = "\n\n---\n\n".join(blocks)
    result, response = groq_call(model, system, user, BATCH_ANSWER_SCHEMA, max_tokens=2300)
    raw_content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    (ARTIFACT_DIR / f"last_qa_raw_{safe_name(model)}.txt").write_text(raw_content or "")
    raw_answers = result.get("answers", [])
    if isinstance(raw_answers, str):
        try:
            raw_answers = json.loads(raw_answers)
        except json.JSONDecodeError:
            raw_answers = []
    if isinstance(raw_answers, dict):
        nested = raw_answers.get("answers", raw_answers.get("items"))
        if isinstance(nested, list):
            raw_answers = nested
        elif all(isinstance(value, dict) for value in raw_answers.values()):
            raw_answers = [{"question_id": key, **value} for key, value in raw_answers.items()]
        else:
            raw_answers = []
    if not isinstance(raw_answers, list):
        raw_answers = []
    normalized_answers = []
    for item in raw_answers:
        if not isinstance(item, dict):
            continue
        canonical = dict(item)
        canonical["question_id"] = str(canonical.get("question_id", canonical.get("QUESTION_ID", canonical.get("id", ""))))
        if "cited_statement_ids" not in canonical:
            canonical["cited_statement_ids"] = canonical.get("citations", canonical.get("evidence", canonical.get("cited_statements", [])))
        if "status" not in canonical:
            canonical["status"] = "insufficient_graph_evidence" if "insufficient" in str(canonical.get("answer", "")).lower() else "answered"
        normalized_answers.append(canonical)
    response_map = {item["question_id"]: item for item in normalized_answers if item["question_id"]}
    records = []
    for question, subgraph in prepared:
        answer = response_map.get(question["id"], {
            "answer": "Insufficient graph evidence.",
            "status": "insufficient_graph_evidence",
            "cited_statement_ids": [],
        })
        records.append({
            "question_id": question["id"], "question": question["question"], "result": answer,
            "subgraph_statement_ids": [statement["statement_id"] for statement in subgraph],
            "usage": response.get("usage", {}),
        })
    return records


def answer_question(model: str, graph: dict[str, Any], question: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    subgraph = retrieve(graph, question)
    if not subgraph:
        return {"answer": "Insufficient graph evidence.", "status": "insufficient_graph_evidence", "cited_statement_ids": []}, [], {}
    serial = "\n".join(statement_text(s) for s in subgraph)
    system = """You answer technical questions using only the supplied graph statements. Do not use outside knowledge, memory of documents, or unstated assumptions. Give a concise direct answer only when graph statements support all material parts; otherwise use status `insufficient_graph_evidence`. Cite the statement identifiers that directly support the answer. Do not cite source documents or page numbers as substitutes for graph statement identifiers."""
    user = f"Question: {question}\n\nRetrieved graph statements:\n{serial}\n\nReturn JSON following the schema."
    result, response = groq_call(model, system, user, ANSWER_SCHEMA, max_tokens=500)
    result.setdefault("answer", "")
    result.setdefault("status", "insufficient_graph_evidence")
    result.setdefault("cited_statement_ids", [])
    return result, subgraph, response.get("usage", {})


def answer_has_group(answer: str, group: list[str]) -> bool:
    normal = norm(answer)
    compact = normal.replace(" ", "")
    for alternative in group:
        target = norm(alternative)
        if target and (target in normal or target.replace(" ", "") in compact):
            return True
    return False


def score_answer(question: dict[str, Any], result: dict[str, Any], subgraph: list[dict[str, Any]]) -> dict[str, Any]:
    groups = question["required_answer_atoms"]
    answered_groups = sum(answer_has_group(result.get("answer", ""), group) for group in groups)
    group_recall = answered_groups / len(groups) if groups else 0.0
    valid_ids = {s["statement_id"] for s in subgraph}
    cited = [str(x) for x in result.get("cited_statement_ids", [])]
    valid_citations = bool(cited) and all(x in valid_ids for x in cited)
    status = result.get("status", "insufficient_graph_evidence")
    grounded = status == "answered" and valid_citations
    unsupported = status == "answered" and not valid_citations
    return {
        "atom_recall": group_recall,
        "exact_correct": int(answered_groups == len(groups)),
        "valid_citation": int(valid_citations),
        "grounded": int(grounded),
        "unsupported": int(unsupported),
        "coverage": int(bool(subgraph)),
        "answered_groups": answered_groups,
        "total_groups": len(groups),
    }


def graph_metrics(graph: dict[str, Any]) -> dict[str, int]:
    return {
        "nodes": len(graph["nodes"]),
        "statements": len(graph["statements"]),
        "fca_concepts": len(graph.get("fca", {}).get("concepts", [])),
    }


def sign_test_pvalue(differences: list[int]) -> float:
    nonzero = [d for d in differences if d != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    k = min(sum(d > 0 for d in nonzero), sum(d < 0 for d in nonzero))
    cumulative = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * cumulative)


def write_report(rows: list[dict[str, Any]], graph_stats: list[dict[str, Any]]) -> None:
    csv_path = ARTIFACT_DIR / "evaluation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    stats_path = ARTIFACT_DIR / "graph_statistics.json"
    stats_path.write_text(json.dumps(graph_stats, indent=2))

    model_condition: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        model_condition[(row["model"], row["condition"])].append(row)
    summary_rows = []
    for (model, condition), values in sorted(model_condition.items()):
        count = len(values)
        summary_rows.append({
            "model": model, "condition": condition, "questions": count,
            "exact_accuracy": sum(r["exact_correct"] for r in values) / count,
            "atom_recall": sum(r["atom_recall"] for r in values) / count,
            "grounded_rate": sum(r["grounded"] for r in values) / count,
            "valid_citation_rate": sum(r["valid_citation"] for r in values) / count,
            "coverage": sum(r["coverage"] for r in values) / count,
            "unsupported_rate": sum(r["unsupported"] for r in values) / count,
        })
    (ARTIFACT_DIR / "evaluation_summary.json").write_text(json.dumps(summary_rows, indent=2))

    lines = [
        "# FCA-backed KG Evaluation Results", "",
        f"Run completed: {datetime.now(timezone.utc).isoformat()}", "",
        "## Design", "",
        "This pilot compares a conventional direct-claim KG with an FCA-backed KG built from the same model-specific extracted claims. Question answering receives only retrieved graph statements—not PDF text or gold answers. Results therefore test graph construction, structure, retrieval, and graph-grounded answer generation together.", "",
        "## Model-level outcomes", "",
        "| Model | Condition | Questions | Exact accuracy | Answer-atom recall | Grounded-answer rate | Valid-citation rate | Graph coverage | Unsupported-answer rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary_rows:
        lines.append(
            f"| {item['model']} | {item['condition']} | {item['questions']} | {item['exact_accuracy']:.1%} | {item['atom_recall']:.1%} | {item['grounded_rate']:.1%} | {item['valid_citation_rate']:.1%} | {item['coverage']:.1%} | {item['unsupported_rate']:.1%} |"
        )
    lines += ["", "## Paired exact-accuracy comparison", "", "| Model | FCA wins | Ties | Baseline wins | FCA minus baseline | Two-sided paired sign-test p-value |", "|---|---:|---:|---:|---:|---:|"]
    comparisons = []
    for model in MODELS:
        base = {r["question_id"]: r for r in model_condition.get((model, "baseline"), [])}
        fca = {r["question_id"]: r for r in model_condition.get((model, "fca"), [])}
        pairs = [(fca[q]["exact_correct"] - base[q]["exact_correct"]) for q in sorted(set(base) & set(fca))]
        wins = sum(d > 0 for d in pairs)
        losses = sum(d < 0 for d in pairs)
        ties = sum(d == 0 for d in pairs)
        delta = (sum(fca[q]["exact_correct"] for q in fca) / len(fca) - sum(base[q]["exact_correct"] for q in base) / len(base)) if base and fca else 0.0
        pvalue = sign_test_pvalue(pairs)
        comparisons.append({"model": model, "wins": wins, "ties": ties, "losses": losses, "delta": delta, "pvalue": pvalue})
        lines.append(f"| {model} | {wins} | {ties} | {losses} | {delta:+.1%} | {pvalue:.4f} |")
    lines += ["", "## Graph construction statistics", "", "| Model | Condition | Nodes | Statements | FCA concepts |", "|---|---|---:|---:|---:|"]
    for item in graph_stats:
        lines.append(f"| {item['model']} | {item['condition']} | {item['nodes']} | {item['statements']} | {item['fca_concepts']} |")
    lines += [
        "", "## Interpretation rule", "",
        "A positive difference indicates that the FCA-backed graph answered more held-out questions exactly in this corpus. A pilot result with a high p-value or inconsistent direction across models is **inconclusive**, not confirmation of general benefit. This two-document evaluation is intended to test feasibility and identify failure modes; it does not establish general superiority.",
        "", "## Reproducibility artifacts", "",
        "Raw claims, graph JSON files, formal-concept labels, model answers, per-question scores, source-free retrieved subgraphs, and API usage logs are saved under `runs/`. The API key is not present in any artifact.",
    ]
    (ARTIFACT_DIR / "evaluation_report.md").write_text("\n".join(lines) + "\n")
    (ARTIFACT_DIR / "paired_comparisons.json").write_text(json.dumps(comparisons, indent=2))


def run() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    # Construction deliberately runs before the held-out questions are loaded.
    built: dict[str, dict[str, dict[str, Any]]] = {}
    graph_stats: list[dict[str, Any]] = []
    for model in MODELS:
        model_dir = RUNS_DIR / safe_name(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        claims = build_raw_claims(model, model_dir)
        baseline = base_graph(claims)
        fca = enrich_fca_graph(model, baseline, model_dir)
        (model_dir / "baseline_graph.json").write_text(json.dumps(baseline, indent=2))
        (model_dir / "fca_graph.json").write_text(json.dumps(fca, indent=2))
        built[model] = {"baseline": baseline, "fca": fca}
        for condition, graph in built[model].items():
            graph_stats.append({"model": model, "condition": condition, **graph_metrics(graph)})

    questions = json.loads((ARTIFACT_DIR / "heldout_questions.json").read_text())
    all_rows: list[dict[str, Any]] = []
    for model in MODELS:
        model_dir = RUNS_DIR / safe_name(model)
        for condition in ["baseline", "fca"]:
            answers_path = model_dir / f"{condition}_answers.json"
            if answers_path.exists():
                answer_records = json.loads(answers_path.read_text())
            else:
                answer_records = []
                batch_size = 5
                for start in range(0, len(questions), batch_size):
                    answer_records.extend(answer_questions_batch(model, built[model][condition], questions[start:start + batch_size]))
                answers_path.write_text(json.dumps(answer_records, indent=2))
            record_map = {r["question_id"]: r for r in answer_records}
            for question in questions:
                record = record_map[question["id"]]
                subgraph_ids = set(record["subgraph_statement_ids"])
                subgraph = [s for s in built[model][condition]["statements"] if s["statement_id"] in subgraph_ids]
                scores = score_answer(question, record["result"], subgraph)
                all_rows.append({
                    "model": model, "condition": condition, "question_id": question["id"],
                    "document_id": question["document_id"], "category": question["category"],
                    "answer": record["result"].get("answer", ""), "status": record["result"].get("status", ""),
                    "cited_statement_ids": ";".join(record["result"].get("cited_statement_ids", [])), **scores,
                })
    write_report(all_rows, graph_stats)
    print(f"Completed {len(all_rows)} scored model-condition-question rows.")
    print(ARTIFACT_DIR / "evaluation_report.md")


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        print(f"FAILED: {error}", file=sys.stderr)
        raise
