"""Audit transform-label sanity in route-pool evidence review rows.

This is a lightweight diagnostic, not a chemistry oracle.  It checks whether
the transform labels attached to ChemEnzy route blocks are grossly consistent
with simple reactant/product structural changes.  The goal is to catch obvious
cases where a route block is labeled as e.g. hydrolysis while the shown reaction
looks like amination or ester formation.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger

from cascade_planner.cascade_search.v4_product_value import canonical_smiles


SCHEMA_VERSION = "route_pool_review_transform_sanity.v1"
DEFAULT_MISMATCH_WARNING_RATE = 0.20


SMARTS = {
    "ester": "[CX3](=O)[OX2H0][#6]",
    "amide": "[CX3](=O)[NX3]",
    "carboxyl": "[CX3](=O)[OX1-,OX2H1]",
    "alcohol": "[OX2H][CX4]",
    "carbonyl": "[CX3]=O",
}

PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in SMARTS.items()}


def audit_route_pool_review_transform_sanity(
    *,
    review_jsonl: Path,
    output_json: Path,
    output_md: Path | None = None,
    mismatch_warning_rate: float = DEFAULT_MISMATCH_WARNING_RATE,
) -> dict[str, Any]:
    RDLogger.DisableLog("rdApp.*")
    rows = _read_jsonl(review_jsonl)
    audited = [_audit_row(row) for row in rows]
    summary = _summary(audited)
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "review_jsonl": str(review_jsonl),
            "output_json": str(output_json),
            "output_md": str(output_md or output_json.with_suffix(".md")),
            "mismatch_warning_rate": mismatch_warning_rate,
        },
        "summary": summary,
        "interpretation": _interpretation(summary, mismatch_warning_rate=mismatch_warning_rate),
        "by_evidence_class": _group_summary(audited, "evidence_class"),
        "by_source_pool": _group_summary(audited, "source_pool"),
        "examples": _examples(audited),
        "rows": audited,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = output_md or output_json.with_suffix(".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_markdown(result), encoding="utf-8")
    return result


def _audit_row(row: dict[str, Any]) -> dict[str, Any]:
    block = row.get("review_block") or {}
    upstream = _audit_step(
        label=block.get("upstream_transform"),
        main_reactant=block.get("upstream_main_reactant"),
        product=block.get("upstream_product"),
        rxn=block.get("upstream_rxn"),
    )
    downstream = _audit_step(
        label=block.get("downstream_transform"),
        main_reactant=block.get("downstream_main_reactant"),
        product=block.get("downstream_product"),
        rxn=block.get("downstream_rxn"),
    )
    mismatch_count = int(bool(upstream.get("label_mismatch"))) + int(bool(downstream.get("label_mismatch")))
    return {
        "review_id": row.get("review_id"),
        "source_pool": row.get("source_pool"),
        "evidence_class": row.get("evidence_class"),
        "target_smiles": row.get("target_smiles"),
        "native_rank": row.get("native_rank"),
        "block_transform_pair": block.get("transform_pair"),
        "diagnostic_scores": row.get("diagnostic_scores") or {},
        "upstream": upstream,
        "downstream": downstream,
        "block_has_missing_reaction_detail": bool(upstream.get("missing_detail") or downstream.get("missing_detail")),
        "block_label_mismatch_count": mismatch_count,
        "block_has_label_mismatch": bool(mismatch_count),
        "block_mismatch_reasons": sorted(set((upstream.get("mismatch_reasons") or []) + (downstream.get("mismatch_reasons") or []))),
    }


def _audit_step(*, label: Any, main_reactant: Any, product: Any, rxn: Any) -> dict[str, Any]:
    label_norm = _norm_label(label)
    main = canonical_smiles(str(main_reactant or ""))
    prod = canonical_smiles(str(product or ""))
    missing = not main or not prod
    main_features = _mol_features(main)
    product_features = _mol_features(prod)
    delta = {
        key: product_features.get(key, 0) - main_features.get(key, 0)
        for key in sorted(set(main_features) | set(product_features))
    }
    inferred = _inferred_classes(main, prod, delta)
    mismatch, reasons = _label_mismatch(label_norm, inferred=inferred, delta=delta, missing=missing)
    return {
        "label": label_norm,
        "rxn": rxn,
        "main_reactant": main,
        "product": prod,
        "missing_detail": missing,
        "feature_delta": delta,
        "inferred_classes": inferred,
        "label_mismatch": bool(mismatch),
        "mismatch_reasons": reasons,
    }


def _mol_features(smiles: str) -> dict[str, int]:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return {}
    atoms = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    features = {
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "C": int(atoms.get("C", 0)),
        "N": int(atoms.get("N", 0)),
        "O": int(atoms.get("O", 0)),
        "S": int(atoms.get("S", 0)),
        "P": int(atoms.get("P", 0)),
        "halogen": int(sum(atoms.get(sym, 0) for sym in ("F", "Cl", "Br", "I"))),
    }
    for name, pattern in PATTERNS.items():
        features[f"{name}_count"] = len(mol.GetSubstructMatches(pattern)) if pattern is not None else 0
    return features


def _inferred_classes(main: str, product: str, delta: dict[str, int]) -> list[str]:
    if not main or not product:
        return ["missing_detail"]
    if main == product:
        return ["identity_or_noop"]
    out = []
    if delta.get("N", 0) > 0:
        out.append("amination_like")
    if delta.get("ester_count", 0) > 0:
        out.append("esterification_like")
    if delta.get("amide_count", 0) > 0:
        out.append("amidation_or_acylation_like")
    if delta.get("ester_count", 0) < 0 or delta.get("amide_count", 0) < 0:
        out.append("hydrolysis_like")
    if delta.get("C", 0) >= 2 and delta.get("heavy_atoms", 0) >= 2:
        out.append("carbon_chain_growth_like")
    if delta.get("O", 0) > 0 and delta.get("N", 0) == 0 and delta.get("ester_count", 0) <= 0:
        out.append("oxygenation_or_oxidation_like")
    if delta.get("O", 0) < 0 and delta.get("N", 0) == 0:
        out.append("deoxygenation_or_reduction_like")
    if abs(delta.get("heavy_atoms", 0)) >= 8:
        out.append("large_heavy_atom_delta")
    return out or ["unclassified_change"]


def _label_mismatch(label: str, *, inferred: list[str], delta: dict[str, int], missing: bool) -> tuple[bool, list[str]]:
    if missing:
        return True, ["missing_reaction_detail"]
    if label in {"", "unknown", "other"}:
        return False, []
    inferred_set = set(inferred)
    reasons = []
    if label == "hydrolysis" and "hydrolysis_like" not in inferred_set and "identity_or_noop" not in inferred_set:
        reasons.append("label_hydrolysis_without_hydrolysis_like_change")
    if label == "amination" and delta.get("N", 0) <= 0:
        reasons.append("label_amination_without_nitrogen_gain")
    if label == "esterification" and delta.get("ester_count", 0) <= 0:
        reasons.append("label_esterification_without_ester_gain")
    if label == "amidation" and delta.get("amide_count", 0) <= 0:
        reasons.append("label_amidation_without_amide_gain")
    if label == "acylation" and delta.get("ester_count", 0) <= 0 and delta.get("amide_count", 0) <= 0 and delta.get("C", 0) <= 0:
        reasons.append("label_acylation_without_acyl_like_growth")
    if label == "c_c_coupling" and delta.get("C", 0) <= 0:
        reasons.append("label_c_c_coupling_without_carbon_gain")
    return bool(reasons), reasons


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    step_rows = [step for row in rows for step in (row.get("upstream") or {}, row.get("downstream") or {})]
    return {
        "rows": total,
        "steps": len(step_rows),
        "rows_with_missing_reaction_detail": sum(1 for row in rows if row.get("block_has_missing_reaction_detail")),
        "rows_with_label_mismatch": sum(1 for row in rows if row.get("block_has_label_mismatch")),
        "row_label_mismatch_rate": _rate(sum(1 for row in rows if row.get("block_has_label_mismatch")), total),
        "step_label_mismatch_rate": _rate(sum(1 for step in step_rows if step.get("label_mismatch")), len(step_rows)),
        "top_mismatch_reasons": dict(Counter(reason for row in rows for reason in row.get("block_mismatch_reasons") or []).most_common(20)),
        "top_inferred_classes": dict(Counter(cls for step in step_rows for cls in step.get("inferred_classes") or []).most_common(20)),
        "top_labels": dict(Counter(step.get("label") for step in step_rows).most_common(20)),
    }


def _interpretation(summary: dict[str, Any], *, mismatch_warning_rate: float) -> dict[str, Any]:
    mismatch_rate = float(summary.get("row_label_mismatch_rate") or 0.0)
    missing_rows = int(summary.get("rows_with_missing_reaction_detail") or 0)
    high_noise = mismatch_rate >= float(mismatch_warning_rate) or missing_rows > 0
    return {
        "transform_labels_safe_for_training_without_review": not high_noise,
        "label_noise_warning": high_noise,
        "recommended_use": (
            "review_triage_only; do not use route-block transform labels as supervised training labels without "
            "chemistry/LLM review and calibration"
            if high_noise
            else "diagnostic sanity check passed the configured warning rate; still treat as heuristic review triage"
        ),
        "reason": {
            "row_label_mismatch_rate": mismatch_rate,
            "mismatch_warning_rate": float(mismatch_warning_rate),
            "rows_with_missing_reaction_detail": missing_rows,
        },
    }


def _group_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field) or "")].append(row)
    return {
        name: {
            "rows": len(group),
            "rows_with_label_mismatch": sum(1 for row in group if row.get("block_has_label_mismatch")),
            "row_label_mismatch_rate": _rate(sum(1 for row in group if row.get("block_has_label_mismatch")), len(group)),
        }
        for name, group in sorted(grouped.items())
    }


def _examples(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    mismatches = [row for row in rows if row.get("block_has_label_mismatch")]
    clean = [row for row in rows if not row.get("block_has_label_mismatch")]
    mismatches.sort(key=lambda row: (-int(row.get("block_label_mismatch_count") or 0), str(row.get("review_id") or "")))
    clean.sort(key=lambda row: str(row.get("review_id") or ""))
    return {
        "label_mismatch_examples": [_compact(row) for row in mismatches[:20]],
        "no_mismatch_examples": [_compact(row) for row in clean[:10]],
    }


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "review_id": row.get("review_id"),
        "source_pool": row.get("source_pool"),
        "evidence_class": row.get("evidence_class"),
        "block_transform_pair": row.get("block_transform_pair"),
        "block_mismatch_reasons": row.get("block_mismatch_reasons"),
        "upstream": {
            "label": (row.get("upstream") or {}).get("label"),
            "rxn": (row.get("upstream") or {}).get("rxn"),
            "inferred_classes": (row.get("upstream") or {}).get("inferred_classes"),
            "mismatch_reasons": (row.get("upstream") or {}).get("mismatch_reasons"),
        },
        "downstream": {
            "label": (row.get("downstream") or {}).get("label"),
            "rxn": (row.get("downstream") or {}).get("rxn"),
            "inferred_classes": (row.get("downstream") or {}).get("inferred_classes"),
            "mismatch_reasons": (row.get("downstream") or {}).get("mismatch_reasons"),
        },
    }


def _norm_label(value: Any) -> str:
    return str(value or "unknown").strip().lower()


def _rate(num: int | float, denom: int | float) -> float:
    return round(float(num) / float(denom), 6) if denom else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _markdown(result: dict[str, Any]) -> str:
    summary = result.get("summary") or {}
    interpretation = result.get("interpretation") or {}
    lines = [
        "# Route Pool Review Transform Sanity Audit",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- transform labels safe for training without review: `{interpretation.get('transform_labels_safe_for_training_without_review')}`",
            f"- label noise warning: `{interpretation.get('label_noise_warning')}`",
            f"- recommended use: `{interpretation.get('recommended_use')}`",
        ]
    )
    lines.extend(["", "## By Evidence Class", "", "| Class | Rows | Mismatch Rows | Mismatch Rate |", "|---|---:|---:|---:|"])
    for cls, row in (result.get("by_evidence_class") or {}).items():
        lines.append(f"| `{cls}` | `{row.get('rows')}` | `{row.get('rows_with_label_mismatch')}` | `{row.get('row_label_mismatch_rate')}` |")
    lines.extend(["", "## Mismatch Examples", ""])
    for row in (result.get("examples") or {}).get("label_mismatch_examples", [])[:5]:
        lines.append(f"- `{row.get('review_id')}` `{row.get('block_transform_pair')}` reasons=`{row.get('block_mismatch_reasons')}`")
    lines.extend(["", "## Contract", "", "This is a heuristic sanity audit, not a definitive chemistry classifier.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit route-pool review transform-label sanity")
    parser.add_argument("--review-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--mismatch-warning-rate", type=float, default=DEFAULT_MISMATCH_WARNING_RATE)
    args = parser.parse_args()
    result = audit_route_pool_review_transform_sanity(
        review_jsonl=Path(args.review_jsonl),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md) if args.output_md else None,
        mismatch_warning_rate=args.mismatch_warning_rate,
    )
    print(
        json.dumps(
            {"summary": result["summary"], "interpretation": result["interpretation"], "output_json": args.output_json},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
