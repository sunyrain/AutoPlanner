"""Process-style literature evidence normalization.

These helpers keep whole-cell, fermentation, and other process evidence out of
small-molecule exact reaction rows.  The rows are advisory anchors for route
objectives and guided search; they are not reaction SMILES or solved proofs.
"""
from __future__ import annotations

import re
from typing import Any


PROCESS_EVIDENCE_ROW_SCHEMA = "literature_process_evidence_row.v1"


_PROCESS_TERMS = {
    "advanced intermediate",
    "advanced ketal ester",
    "biotransformation",
    "calcium salt",
    "counter-ion exchange",
    "deprotection",
    "ester hydrolysis",
    "fermentation",
    "hemi-calcium",
    "whole-cell",
    "whole cell",
    "kilogram",
    "kilogram-scale",
    "multi-kilogram",
    "microbial",
    "mycobacterium",
    "enzyme",
    "enzymatic",
    "paal-knorr",
    "preparation",
    "process route",
    "pyrrole ring",
    "salt formation",
    "strain",
    "producer",
    "phytosterol",
    "phytosterols",
    "substrate",
    "feedstock",
    "kstD".lower(),
    "hsd4A".lower(),
    "fadA5".lower(),
}

_SUBSTRATE_TERMS = {
    "advanced intermediate",
    "advanced ketal ester",
    "diketone",
    "phytosterol",
    "phytosterols",
    "intermediate",
    "ketal ester",
    "paal-knorr precursor",
    "side-chain amine",
    "side chain amine",
    "sterol",
    "sterols",
    "substrate",
    "feedstock",
    "precursor",
}

_BIOCATALYST_TERMS = {
    "calcium acetate",
    "calcium salt formation",
    "counter-ion exchange",
    "deprotection",
    "ester hydrolysis",
    "ethyl acetate extraction",
    "mycobacterium",
    "paal-knorr",
    "strain",
    "enzyme",
    "enzymatic",
    "whole-cell",
    "whole cell",
    "biocatalyst",
    "producer",
    "kstD".lower(),
    "hsd4A".lower(),
    "fadA5".lower(),
    "delta",
}


def process_evidence_rows_from_visual_result(
    result: dict[str, Any],
    *,
    payload: dict[str, Any] | None = None,
    artifact_ref: str = "",
) -> list[dict[str, Any]]:
    """Infer process evidence rows from a visual/text extraction result.

    The inference is intentionally conservative: it only emits rows when the
    source text/gaps contain process language and a target/product label.  No
    molecule-level reaction fields are produced.
    """
    rows = _explicit_process_rows(result, artifact_ref=artifact_ref)
    if rows:
        return _dedupe_rows(rows)

    payload = dict(payload or {})
    candidate = _candidate_payload(result)
    labels = _labels_from_payloads(result, payload, candidate)
    text = _joined_text(result, payload, candidate, labels)
    if not _has_any(text, _PROCESS_TERMS):
        return []

    source_ref = _first_text(
        result.get("source_ref"),
        candidate.get("source_ref"),
        payload.get("source_ref"),
        result.get("doi"),
        candidate.get("doi"),
        payload.get("doi"),
    )
    source_title = _first_text(
        result.get("source_title"),
        candidate.get("source_title"),
        payload.get("source_title"),
        payload.get("title"),
        candidate.get("title"),
    )
    endpoint_labels = _endpoint_labels(labels, text)
    substrate_labels = _labels_matching(labels, _SUBSTRATE_TERMS)
    biocatalyst_labels = _labels_matching(labels, _BIOCATALYST_TERMS)
    if not endpoint_labels:
        endpoint_labels = _endpoint_labels_from_text(text)
    if not (endpoint_labels and (substrate_labels or biocatalyst_labels)):
        return []

    row = {
        "schema_version": PROCESS_EVIDENCE_ROW_SCHEMA,
        "row_id": _row_id(source_ref or source_title or artifact_ref, endpoint_labels[0]),
        "evidence_class": "process_literature_endpoint",
        "process_type": _process_type(text),
        "source_ref": source_ref,
        "source_title": source_title,
        "artifact_ref": artifact_ref,
        "endpoint_labels": endpoint_labels[:6],
        "substrate_or_feedstock_labels": substrate_labels[:6],
        "biocatalyst_or_process_labels": biocatalyst_labels[:8],
        "evidence_refs": _dedupe_texts(
            [
                *[str(item) for item in result.get("evidence_refs") or []],
                *[str(item) for item in candidate.get("evidence_refs") or []],
            ]
        )[:12],
        "source_locator": _first_text(
            result.get("source_locator"),
            candidate.get("source_locator"),
            _gap_locator(candidate),
        ),
        "confidence": _process_confidence(endpoint_labels, substrate_labels, biocatalyst_labels, text),
        "allowed_use": "route_objective_anchor_and_guided_hint_only",
        "not_exact_literature_segment": True,
        "not_parent_route_proof": True,
        "not_reaction_smiles": True,
        "requires_objective_specific_verification": True,
        "verification_required": [
            "product_identity_audit",
            "process_endpoint_acceptability",
            "feedstock_or_organism_evidence_review",
            "do_not_compile_as_exact_reaction_smiles",
        ],
        "risk_flags": _risk_flags(text),
        "summary": _summary(endpoint_labels, substrate_labels, biocatalyst_labels),
        "no_solved_claim": True,
    }
    return [row]


def process_evidence_rows_from_pdf_result(
    result: dict[str, Any],
    *,
    payload: dict[str, Any] | None = None,
    artifact_ref: str = "",
    max_chars: int = 120_000,
) -> list[dict[str, Any]]:
    """Infer process evidence directly from locally extracted PDF text."""
    payload = dict(payload or {})
    fulltext = _read_text_file(result.get("fulltext_path"), max_chars=max_chars)
    labels = _labels_from_payloads(result, payload)
    title = _first_text(
        result.get("source_title"),
        payload.get("source_title"),
        payload.get("title"),
    )
    text = " ".join(
        part
        for part in [
            str(result.get("source_ref") or ""),
            title,
            str(payload.get("route_sequence_hint") or ""),
            " ".join(labels),
            fulltext,
        ]
        if str(part or "").strip()
    )
    if not text or not _has_any(text, _PROCESS_TERMS):
        return []

    endpoint_labels = _dedupe_texts([*_endpoint_labels(labels, text.lower()), *_endpoint_labels_from_text(text)])
    substrate_labels = _dedupe_texts([*_labels_matching(labels, _SUBSTRATE_TERMS), *_substrate_labels_from_text(text)])
    biocatalyst_labels = _dedupe_texts([*_labels_matching(labels, _BIOCATALYST_TERMS), *_biocatalyst_labels_from_text(text)])
    if not (endpoint_labels and (substrate_labels or biocatalyst_labels)):
        return []

    source_ref = _first_text(
        result.get("source_ref"),
        payload.get("source_ref"),
        result.get("doi"),
        payload.get("doi"),
    )
    row = {
        "schema_version": PROCESS_EVIDENCE_ROW_SCHEMA,
        "row_id": _row_id(source_ref or title or artifact_ref, endpoint_labels[0]),
        "evidence_class": "process_literature_endpoint",
        "process_type": _process_type(text.lower()),
        "source_ref": source_ref,
        "source_title": title,
        "source_pdf_path": _first_text(result.get("source_pdf_path"), result.get("pdf_path"), payload.get("pdf_path")),
        "artifact_ref": artifact_ref,
        "endpoint_labels": endpoint_labels[:6],
        "substrate_or_feedstock_labels": substrate_labels[:6],
        "biocatalyst_or_process_labels": biocatalyst_labels[:8],
        "quantitative_evidence": _quantitative_evidence_from_text(text),
        "evidence_refs": _dedupe_texts([source_ref, str(result.get("source_pdf_path") or "")])[:12],
        "source_locator": "pdf_fulltext",
        "confidence": _process_confidence(endpoint_labels, substrate_labels, biocatalyst_labels, text.lower()),
        "allowed_use": "route_objective_anchor_and_guided_hint_only",
        "not_exact_literature_segment": True,
        "not_parent_route_proof": True,
        "not_reaction_smiles": True,
        "requires_objective_specific_verification": True,
        "verification_required": [
            "product_identity_audit",
            "process_endpoint_acceptability",
            "feedstock_or_organism_evidence_review",
            "do_not_compile_as_exact_reaction_smiles",
        ],
        "risk_flags": _risk_flags(text.lower()),
        "summary": _summary(endpoint_labels, substrate_labels, biocatalyst_labels),
        "no_solved_claim": True,
    }
    return [row]


def semisynthesis_anchors_from_process_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_id = str(row.get("row_id") or "")
        endpoint = str((row.get("endpoint_labels") or ["process_endpoint"])[0] or "process_endpoint")
        anchors.append(
            {
                "schema_version": "route_objective_anchor.v1",
                "anchor_id": f"process_anchor:{_safe_token(row_id or endpoint)}",
                "anchor_type": "biotransformation_or_process_endpoint",
                "name": endpoint,
                "source_ref": str(row.get("source_ref") or ""),
                "source_title": str(row.get("source_title") or ""),
                "process_evidence_row_id": row_id,
                "endpoint_labels": [str(item) for item in row.get("endpoint_labels") or [] if str(item).strip()],
                "substrate_or_feedstock_labels": [
                    str(item) for item in row.get("substrate_or_feedstock_labels") or [] if str(item).strip()
                ],
                "biocatalyst_or_process_labels": [
                    str(item) for item in row.get("biocatalyst_or_process_labels") or [] if str(item).strip()
                ],
                "allowed_use": "route_objective_anchor_and_guided_hint_only",
                "not_exact_literature_segment": True,
                "not_parent_route_proof": True,
                "no_solved_claim": True,
            }
        )
    return anchors


def _explicit_process_rows(result: dict[str, Any], *, artifact_ref: str) -> list[dict[str, Any]]:
    explicit = result.get("literature_process_evidence_rows") or result.get("process_evidence_rows") or []
    rows: list[dict[str, Any]] = []
    if not isinstance(explicit, list):
        return rows
    for raw in explicit:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        row.setdefault("schema_version", PROCESS_EVIDENCE_ROW_SCHEMA)
        row.setdefault("allowed_use", "route_objective_anchor_and_guided_hint_only")
        row.setdefault("not_exact_literature_segment", True)
        row.setdefault("not_parent_route_proof", True)
        row.setdefault("not_reaction_smiles", True)
        row.setdefault("no_solved_claim", True)
        if artifact_ref and not str(row.get("artifact_ref") or "").strip():
            row["artifact_ref"] = artifact_ref
        if not str(row.get("row_id") or "").strip():
            row["row_id"] = _row_id(str(row.get("source_ref") or artifact_ref), str((row.get("endpoint_labels") or [""])[0]))
        rows.append(row)
    return rows


def _candidate_payload(result: dict[str, Any]) -> dict[str, Any]:
    for key in ("candidate_chain", "parsed_output"):
        value = result.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _labels_from_payloads(*payloads: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for payload in payloads:
        for key in ("expected_labels", "missing_expected_labels", "gap_labels", "condition_gap_labels"):
            labels.extend(str(item) for item in payload.get(key) or [] if str(item or "").strip())
        for gap in payload.get("extraction_gaps") or []:
            if not isinstance(gap, dict):
                continue
            raw = gap.get("labels") if isinstance(gap.get("labels"), list) else [gap.get("label")]
            labels.extend(str(item) for item in raw if str(item or "").strip())
    return _dedupe_texts(labels)


def _joined_text(result: dict[str, Any], payload: dict[str, Any], candidate: dict[str, Any], labels: list[str]) -> str:
    parts: list[str] = []
    for source in (result, payload, candidate):
        for key in (
            "source_ref",
            "source_title",
            "title",
            "source_locator",
            "route_sequence_hint",
            "raw_last_message",
            "source_excerpt",
            "relevance_rationale",
        ):
            if str(source.get(key) or "").strip():
                parts.append(str(source.get(key) or ""))
        for key in ("evidence_refs", "reasons", "extraction_task_recommendations"):
            parts.extend(str(item) for item in source.get(key) or [] if str(item or "").strip())
        for gap in source.get("extraction_gaps") or []:
            if isinstance(gap, dict):
                parts.extend(str(gap.get(key) or "") for key in ("label", "reason", "source_locator") if str(gap.get(key) or "").strip())
    parts.extend(labels)
    return " ".join(parts).lower()


def _endpoint_labels(labels: list[str], text: str) -> list[str]:
    rows: list[str] = []
    for label in labels:
        clean = str(label).strip()
        lower = clean.lower()
        if not clean:
            continue
        if _has_any(lower, _SUBSTRATE_TERMS | _BIOCATALYST_TERMS):
            continue
        if any(
            token in lower
            for token in ("product", "target", "peak", "oh", "hp", "pregn", "steroid", "atorvastatin")
        ):
            rows.append(clean)
            continue
        if clean.lower() in text and len(clean) >= 4:
            rows.append(clean)
    return _dedupe_texts(rows)


def _endpoint_labels_from_text(text: str) -> list[str]:
    rows: list[str] = []
    if re.search(r"\batorvastatin\s+hemi[- ]?calcium\b", text, flags=re.I):
        rows.append("atorvastatin hemi-calcium salt")
    if re.search(r"\batorvastatin\s+calcium\b", text, flags=re.I):
        rows.append("atorvastatin calcium")
    if re.search(r"\batorvastatin\s+sodium\b", text, flags=re.I):
        rows.append("atorvastatin sodium solution")
    if re.search(r"\batorvastatin\s+free\s+acid\b", text, flags=re.I):
        rows.append("atorvastatin free acid")
    if re.search(r"\batorvastatin\b", text, flags=re.I):
        rows.append("atorvastatin")
    for pattern in (
        r"\b9[- ]oh[- ]4[- ]hp\b",
        r"\b9,21[- ]dihydroxy[- ]20[- ]methyl[- ]pregna[- ]4[- ]en[- ]3[- ]one\b",
    ):
        for match in re.finditer(pattern, text, flags=re.I):
            rows.append(match.group(0))
    return _dedupe_texts(rows)


def _substrate_labels_from_text(text: str) -> list[str]:
    rows: list[str] = []
    if re.search(r"\bphytosterols?\b", text, flags=re.I):
        rows.append("phytosterols")
    if re.search(r"\badvanced\s+ketal\s+ester\s+intermediate\s*\(?4\)?\b", text, flags=re.I):
        rows.append("advanced ketal ester intermediate 4")
    elif re.search(r"\bketal\s+ester\s+intermediate\s*\(?4\)?\b", text, flags=re.I):
        rows.append("ketal ester intermediate 4")
    if re.search(r"\bintermediate\s*\(?4\)?\b", text, flags=re.I):
        rows.append("intermediate 4")
    if re.search(r"\bdiol\s*\(?5\)?\b", text, flags=re.I):
        rows.append("diol 5")
    if re.search(r"\b1,4[- ]diketone\s*\(?2\)?\b", text, flags=re.I):
        rows.append("1,4-diketone 2")
    if re.search(r"\b(?:protected\s+)?side[- ]chain\s+amine\s*\(?3\)?\b", text, flags=re.I):
        rows.append("protected side-chain amine 3")
    if re.search(r"\bpaal[- ]knorr\b", text, flags=re.I):
        rows.append("Paal-Knorr precursor set")
    return _dedupe_texts(rows)


def _biocatalyst_labels_from_text(text: str) -> list[str]:
    rows: list[str] = []
    if re.search(r"\bpaal[- ]knorr\b", text, flags=re.I):
        rows.append("Paal-Knorr pyrrole construction")
    if re.search(r"\bketal\s+deprotection\b", text, flags=re.I):
        rows.append("ketal deprotection")
    if re.search(r"\bester\s+hydrolysis\b|\bhydrolysis\b", text, flags=re.I):
        rows.append("ester hydrolysis")
    if re.search(r"\bcounter[- ]ion\s+exchange\b", text, flags=re.I):
        rows.append("counter-ion exchange")
    if re.search(r"\bcalcium\s+(?:salt|acetate)\b|\bhemi[- ]?calcium\b", text, flags=re.I):
        rows.append("calcium salt formation")
    if re.search(r"\beth?yl\s+acetate\s+extraction\b|\bethyl\s+acetate\b", text, flags=re.I):
        rows.append("ethyl acetate extraction")
    if re.search(r"\bmycobacterium\s+neoaurum\b", text, flags=re.I):
        rows.append("Mycobacterium neoaurum")
    if re.search(r"\bDSM\s*44074\b", text, flags=re.I):
        rows.append("DSM 44074")
    gene_tokens = []
    for token in ("kstD", "hsd4A", "fadA5", "katE", "nox"):
        if re.search(rf"\b{re.escape(token)}\b", text, flags=re.I):
            gene_tokens.append(token)
    if gene_tokens:
        rows.append(" ".join(gene_tokens))
    return _dedupe_texts(rows)


def _quantitative_evidence_from_text(text: str) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    normalized = _normalize_metric_text(text)
    metric_specs = {
        "product_titer_g_per_l": [
            r"\b9[- ]oh[- ]4[- ]hp\s+production\s+reached\s+([0-9]+(?:\.[0-9]+)?)\s*g\s*l\s*-?1\b",
            r"\bhighest\s+yield\s+of\s+9[- ]oh[- ]4[- ]hp\s+was\s+([0-9]+(?:\.[0-9]+)?)\s*g\s*l\s*-?1\b",
        ],
        "phytosterol_loading_g_per_l": [
            r"\bfrom\s+([0-9]+(?:\.[0-9]+)?)\s*g\s*l\s*-?1\s+phytosterols?\b",
            r"\bwith\s+([0-9]+(?:\.[0-9]+)?)\s*g\s*l\s*-?1\s+phytosterols?\b",
        ],
        "product_purity_percent": [
            r"\bpurity\s+of\s+9[- ]oh[- ]4[- ]hp[^.]{0,160}?(?:improved\s+to|reached|was|at)\s+([0-9]+(?:\.[0-9]+)?)\s*%",
            r"\b9[- ]oh[- ]4[- ]hp[^.]{0,160}?purity[^.]{0,120}?(?:improved\s+to|reached|was|at)?\s+([0-9]+(?:\.[0-9]+)?)\s*%",
            r"\bpurity[^.]{0,120}?(?:improved\s+to|reached|was|at)\s+([0-9]+(?:\.[0-9]+)?)\s*%",
        ],
        "molar_yield_percent": [
            r"\b(?:best|highest|final)[^.]{0,120}?molar\s+yield[^.]{0,80}?([0-9]+(?:\.[0-9]+)?)\s*%",
            r"\bmolar\s+yield\s+of\s+9[- ]oh[- ]4[- ]hp[^.]{0,120}?([0-9]+(?:\.[0-9]+)?)\s*%",
            r"\bmolar\s+yield[^.]{0,80}?([0-9]+(?:\.[0-9]+)?)\s*%",
        ],
    }
    candidates: list[dict[str, Any]] = []
    for metric, patterns in metric_specs.items():
        metric_candidates = _metric_candidates(normalized, metric=metric, patterns=patterns)
        if not metric_candidates:
            continue
        best = max(metric_candidates, key=lambda row: (float(row.get("score") or 0), float(row.get("value") or 0)))
        evidence[metric] = best["value"]
        candidates.extend(metric_candidates[:6])
    if candidates:
        evidence["candidate_metrics"] = sorted(
            candidates,
            key=lambda row: (str(row.get("metric") or ""), -float(row.get("score") or 0), -float(row.get("value") or 0)),
        )[:20]
    return evidence


def _normalize_metric_text(text: str) -> str:
    normalized = str(text or "")
    replacements = {
        "\u00ad": "",
        "\u00a0": " ",
        "\u2009": " ",
        "\u202f": " ",
        "\u2212": "-",
        "\u2011": "-",
        "\u2013": "-",
        "\u2014": "-",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _metric_candidates(text: str, *, metric: str, patterns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, float, str]] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            try:
                value = float(match.group(1))
            except (TypeError, ValueError):
                continue
            context = _metric_context(text, match.start(), match.end())
            key = (metric, value, context[:80])
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "metric": metric,
                    "value": value,
                    "score": _score_metric_context(context, matched_text=match.group(0)),
                    "context": context[:360],
                }
            )
    return sorted(rows, key=lambda row: (-float(row.get("score") or 0), -float(row.get("value") or 0)))


def _metric_context(text: str, start: int, end: int, *, radius: int = 140) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return str(text[left:right]).strip()


def _score_metric_context(context: str, *, matched_text: str = "") -> float:
    lower = context.lower()
    matched = str(matched_text or "").lower()
    score = 0.0
    if "9-oh-4-hp" in lower or "9 oh 4 hp" in lower:
        score += 2.0
    if "phytosterol" in lower:
        score += 1.0
    if "ultimately" in lower:
        score += 2.0
    if any(token in matched for token in ("final", "best", "highest", "improved to")):
        score += 5.0
    if "reached" in matched:
        score += 0.5
    if any(token in lower for token in ("-nk", "kate", "nox")):
        score += 2.0
    if re.search(r"\b5\s*g\s*l\s*-?1\s+phytosterols?\b", lower):
        score += 2.0
    if any(token in lower for token in ("by-product", "didn't show satisfactory", "didn’t show satisfactory")):
        score -= 4.0
    if "9-oh-ad" in lower and "9-oh-4-hp" not in lower:
        score -= 4.0
    if "previous reported" in lower:
        score -= 1.0
    return score


def _labels_matching(labels: list[str], terms: set[str]) -> list[str]:
    rows = []
    for label in labels:
        lower = str(label).lower()
        if _has_any(lower, terms):
            rows.append(str(label).strip())
    return _dedupe_texts(rows)


def _process_type(text: str) -> str:
    lower = str(text or "").lower()
    small_molecule_markers = (
        "total synthesis",
        "semisynthesis",
        "semi-synthesis",
        "paal-knorr",
        "ketal",
        "hydrolysis",
        "calcium salt",
        "counter-ion",
        "side-chain coupling",
        "side chain coupling",
        "esterification",
        "deprotection",
    )
    if any(token in lower for token in small_molecule_markers):
        return "small_molecule_process_route"
    explicit_whole_cell = any(
        token in lower
        for token in ("mycobacterium", "whole-cell", "whole cell", "kst", "hsd4a", "fada5")
    )
    strain_with_biology = bool(
        re.search(r"\bstrain\b", lower)
        and re.search(r"\b(culture|mutant|gene|biotransformation|fermentation|titer|microb(?:e|ial))\b", lower)
    )
    if explicit_whole_cell or strain_with_biology:
        return "whole_cell_biotransformation"
    if "fermentation" in lower:
        return "fermentation_endpoint"
    if re.search(r"\benzyme\b|\benzymatic\b", lower):
        return "enzymatic_biotransformation"
    return "process_endpoint"


def _process_confidence(endpoint: list[str], substrates: list[str], catalysts: list[str], text: str) -> str:
    score = 0
    if endpoint:
        score += 2
    if substrates:
        score += 2
    if catalysts:
        score += 2
    if "doi:" in text or "10." in text:
        score += 1
    if "figure" in text or "fig." in text or "table" in text:
        score += 1
    if any(token in text for token in ("kilogram", "kg scale", "7 kg", "multi-kilogram")):
        score += 1
    if any(token in text for token in ("paal-knorr", "ketal deprotection", "ester hydrolysis", "counter-ion exchange")):
        score += 1
    if score >= 7:
        return "medium_high"
    if score >= 5:
        return "medium"
    return "low"


def _risk_flags(text: str) -> list[str]:
    flags = ["not_small_molecule_exact_reaction_row"]
    if re.search(r"\b(?:feedstock|substrate)\s+mixture\b|\bphytosterols?\b", text, flags=re.I):
        flags.append("feedstock_mixture_or_class")
    if any(token in text for token in ("strain", "mycobacterium", "whole-cell", "whole cell")):
        flags.append("organism_or_strain_condition_required")
    if any(token in text for token in ("paal-knorr", "ketal", "hydrolysis", "calcium salt", "counter-ion")):
        flags.append("process_route_anchor_not_stepwise_exact_conditions")
    return flags


def _summary(endpoint: list[str], substrates: list[str], catalysts: list[str]) -> str:
    left = " / ".join(substrates or ["process substrate/feedstock"])
    middle = " / ".join(catalysts or ["process or biocatalyst"])
    right = " / ".join(endpoint or ["endpoint product"])
    return f"{left} via {middle} to {right}"


def _gap_locator(candidate: dict[str, Any]) -> str:
    for gap in candidate.get("extraction_gaps") or []:
        if isinstance(gap, dict) and str(gap.get("source_locator") or "").strip():
            return str(gap.get("source_locator") or "")
    return ""


def _has_any(text: str, terms: set[str]) -> bool:
    lower = str(text).lower()
    return any(term.lower() in lower for term in terms)


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _read_text_file(path_value: Any, *, max_chars: int) -> str:
    path_text = str(path_value or "").strip()
    if not path_text:
        return ""
    try:
        with open(path_text, "r", encoding="utf-8", errors="ignore") as handle:
            return handle.read(max(1, int(max_chars)))
    except OSError:
        return ""


def _row_id(source: str, endpoint: str) -> str:
    return f"process_evidence:{_safe_token(source or 'source')}:{_safe_token(endpoint or 'endpoint')}"


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return token[:80] or "item"


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("row_id") or row.get("summary") or row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _dedupe_texts(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out
