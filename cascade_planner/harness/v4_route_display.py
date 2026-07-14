"""Deterministic display semantics for one canonical V4 route DAG.

This module assigns presentation stages and provenance labels.  It never
changes canonical chemistry, proof, stock, or route acceptance.
"""
from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Mapping, Sequence


ORIGIN_LABELS = {
    "chemenzy": "ChemEnzy",
    "codex": "Codex",
    "codex_global_director": "Codex 全局规划",
    "host_product_grounded_repair": "主机结构修复",
    "literature": "文献抽取",
    "literature_visual_extraction": "文献视觉候选",
    "literature_replay": "文献重放",
    "manual": "人工方案",
    "self_evo_patent_template": "专利自进化",
    "template": "通用模板",
}

EVIDENCE_LABELS = {
    "curated_registry": "权威登记库",
    "paper": "论文",
    "paper_si": "论文/SI",
    "patent": "专利",
    "registry": "登记库",
}


def compile_route_display_rows(
    edge_ids: Sequence[str],
    *,
    edge_rows: Mapping[str, Mapping[str, Any]],
    edge_inspectors: Mapping[str, Mapping[str, Any]],
    nodes_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return synthesis-topological, source-labelled display rows."""

    selected = [str(value) for value in edge_ids if str(value) in edge_rows]
    product_owner = {
        str(edge_rows[edge_id].get("product_molecule_id") or ""): edge_id
        for edge_id in selected
        if str(edge_rows[edge_id].get("product_molecule_id") or "")
    }
    dependencies = {
        edge_id: {
            product_owner[precursor_id]
            for precursor_id in edge_rows[edge_id].get("precursor_molecule_ids") or []
            if precursor_id in product_owner and product_owner[precursor_id] != edge_id
        }
        for edge_id in selected
    }
    stages, ordered = _topological_stages(selected, dependencies)
    grouped: dict[int, list[str]] = defaultdict(list)
    for edge_id in ordered:
        grouped[stages[edge_id]].append(edge_id)
    stage_labels: dict[str, str] = {}
    for stage, values in sorted(grouped.items()):
        ranked = sorted(values)
        for index, edge_id in enumerate(ranked):
            suffix = _alpha(index) if len(ranked) > 1 else ""
            stage_labels[edge_id] = f"S{stage}{suffix}"

    rows = []
    for edge_id in ordered:
        edge = edge_rows[edge_id]
        inspector = edge_inspectors.get(edge_id, {})
        source_labels = _source_step_labels(inspector)
        producer_kinds = _producer_kinds(edge, inspector)
        evidence_kinds = _evidence_kinds(edge, inspector)
        precursors = [str(value) for value in edge.get("precursor_molecule_ids") or []]
        main_ids, auxiliary_ids = _partition_inputs(
            precursors,
            produced_ids=set(product_owner),
            nodes_by_id=nodes_by_id,
        )
        stage_label = stage_labels[edge_id]
        native_label = " / ".join(source_labels[:2])
        rows.append(
            {
                "edge_id": edge_id,
                "synthesis_stage": stages[edge_id],
                "stage_label": stage_label,
                "display_label": f"{stage_label} · {native_label}" if native_label else stage_label,
                "source_step_labels": source_labels,
                "producer_kinds": producer_kinds,
                "producer_label": _joined_labels(producer_kinds, ORIGIN_LABELS, "来源未标记"),
                "evidence_kinds": evidence_kinds,
                "evidence_label": _joined_labels(evidence_kinds, EVIDENCE_LABELS, "无精确证据"),
                "main_precursor_ids": main_ids,
                "auxiliary_precursor_ids": auxiliary_ids,
            }
        )
    return rows


def _topological_stages(
    edge_ids: Sequence[str],
    dependencies: Mapping[str, set[str]],
) -> tuple[dict[str, int], list[str]]:
    remaining = set(edge_ids)
    stages: dict[str, int] = {}
    ordered: list[str] = []
    while remaining:
        completed = set(stages)
        ready = sorted(
            edge_id
            for edge_id in remaining
            if dependencies.get(edge_id, set()) <= completed
        )
        if not ready:
            # Canonical acceptance should already reject cycles.  Keep display
            # deterministic and visibly bounded if a corrupt legacy row leaks in.
            ready = [min(remaining)]
        for edge_id in ready:
            parents = dependencies.get(edge_id, set())
            stages[edge_id] = 1 + max((stages.get(value, 0) for value in parents), default=0)
            ordered.append(edge_id)
            remaining.remove(edge_id)
    return stages, sorted(ordered, key=lambda value: (stages[value], value))


def _partition_inputs(
    precursor_ids: Sequence[str],
    *,
    produced_ids: set[str],
    nodes_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    if len(precursor_ids) <= 1:
        return list(precursor_ids), []
    counts = {
        molecule_id: int(nodes_by_id.get(molecule_id, {}).get("heavy_atom_count") or 0)
        for molecule_id in precursor_ids
    }
    largest = max(counts.values(), default=0)
    auxiliary = []
    for molecule_id in precursor_ids:
        if molecule_id in produced_ids:
            continue
        count = counts[molecule_id]
        if count <= 7 or (largest >= 12 and count < largest * 0.45):
            auxiliary.append(molecule_id)
    main = [value for value in precursor_ids if value not in auxiliary]
    return (main or list(precursor_ids)), auxiliary if main else []


def _producer_kinds(
    edge: Mapping[str, Any],
    inspector: Mapping[str, Any],
) -> list[str]:
    values = {
        str(value).lower()
        for value in edge.get("origin_kinds") or []
        if str(value)
    }
    values.update(
        str(value.get("origin_kind") or "").lower()
        for value in inspector.get("provenance") or []
        if isinstance(value, Mapping) and str(value.get("origin_kind") or "")
    )
    return sorted(values, key=lambda value: (value not in ORIGIN_LABELS, value))


def _evidence_kinds(
    edge: Mapping[str, Any],
    inspector: Mapping[str, Any],
) -> list[str]:
    values = {
        str(value).lower()
        for value in edge.get("source_kinds") or []
        if str(value)
    }
    values.update(
        str(value.get("source_kind") or "").lower()
        for value in inspector.get("sources") or []
        if isinstance(value, Mapping) and str(value.get("source_kind") or "")
    )
    return sorted(values, key=lambda value: (value not in EVIDENCE_LABELS, value))


def _source_step_labels(inspector: Mapping[str, Any]) -> list[str]:
    candidates: list[str] = []
    for value in inspector.get("exact_records") or []:
        if isinstance(value, Mapping):
            candidates.append(str(value.get("claim_scope_id") or ""))
    for value in inspector.get("provenance") or []:
        if isinstance(value, Mapping):
            candidates.append(str(value.get("proposal_id") or ""))
    labels = []
    for raw in candidates:
        label = _native_step_label(raw)
        if label and label not in labels:
            labels.append(label)
    return sorted(labels, key=lambda value: (not value.startswith("P-"), value))


def _native_step_label(raw: str) -> str:
    token = raw.rsplit(":", 1)[-1].removeprefix("hypothesis:")
    patent = re.search(r"patent_(c?)(\d+)", token, re.IGNORECASE)
    if patent:
        marker = "C" if patent.group(1) else ""
        return f"P-{marker}{patent.group(2)}"
    science = re.match(r"T(\d+)(?:_|$)", token, re.IGNORECASE)
    if science:
        return f"S-T{science.group(1)}"
    compound = re.match(r"compound_(\d+)(?:_|$)", token, re.IGNORECASE)
    if compound:
        return f"S-{compound.group(1)}"
    return ""


def _joined_labels(
    values: Sequence[str],
    labels: Mapping[str, str],
    fallback: str,
) -> str:
    rendered = [labels.get(value, value) for value in values]
    return " + ".join(rendered) if rendered else fallback


def _alpha(index: int) -> str:
    return chr(ord("a") + index) if 0 <= index < 26 else str(index + 1)


__all__ = ["EVIDENCE_LABELS", "ORIGIN_LABELS", "compile_route_display_rows"]
