"""Run candidate-level bridge gate ablation for verifier v0.

This is the P2 smoke test before wiring the bridge verifier into route search.
It asks a narrow question:

    If a chemical frontier molecule triggers enzyme-bridge candidates, how much
    does a verifier gate reduce false enzyme candidates compared with ungated
    retrieval or Tanimoto-only filtering?

The script deliberately uses the frozen verifier test split and precomputed
scores from ``bridge_verifier_v0``. It therefore measures gate behavior on the
weak-label bridge benchmark, not final route-level solved rate.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_PACK_DIR = Path("data/bridge_pack_v0")
DEFAULT_MODEL_DIR = Path("results/shared/bridge_verifier_v0_20260527")
DEFAULT_OUTPUT_DIR = Path("results/shared/bridge_gate_ablation_v0_20260527")
DEFAULT_VERIFIER_THRESHOLD = 0.8409896871324669


def read_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows) if rows else pa.table({}), path)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def percentile(values: list[int | float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(float(value) for value in values)
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    return float(values[lo] * (hi - pos) + values[hi] * (pos - lo))


def align_rows(test_rows: list[dict[str, Any]], score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(test_rows) != len(score_rows):
        raise ValueError(f"test rows and score rows have different lengths: {len(test_rows)} vs {len(score_rows)}")
    out: list[dict[str, Any]] = []
    mismatches = 0
    for idx, (row, score_row) in enumerate(zip(test_rows, score_rows)):
        row = dict(row)
        if (
            str(row.get("chemical_inchikey") or "") != str(score_row.get("chemical_inchikey") or "")
            or str(row.get("enzyme_inchikey") or "") != str(score_row.get("enzyme_inchikey") or "")
        ):
            mismatches += 1
        row["verifier_score"] = float(score_row.get("verifier_score") or 0.0)
        row["row_id"] = idx
        out.append(row)
    if mismatches:
        raise ValueError(f"test rows and score rows are not aligned; mismatches={mismatches}")
    return out


def policy_accepts(row: dict[str, Any], policy: str, verifier_threshold: float) -> bool:
    tanimoto = float(row.get("tanimoto") or 0.0)
    same_inchikey = str(row.get("chemical_inchikey") or "") == str(row.get("enzyme_inchikey") or "")
    score = float(row.get("verifier_score") or 0.0)
    if policy == "native_no_bridge":
        return False
    if policy == "ungated_bridge":
        return True
    if policy == "tanimoto_ge_0_50":
        return tanimoto >= 0.50 or same_inchikey
    if policy == "tanimoto_ge_0_80":
        return tanimoto >= 0.80 or same_inchikey
    if policy == "verifier_ge_0_50":
        return score >= 0.50
    if policy == "verifier_precision_gate":
        return score >= verifier_threshold
    if policy == "tanimoto_0_80_and_verifier_precision_gate":
        return (tanimoto >= 0.80 or same_inchikey) and score >= verifier_threshold
    raise ValueError(f"unknown policy: {policy}")


def summarize_policy(rows: list[dict[str, Any]], policy: str, verifier_threshold: float) -> dict[str, Any]:
    total_pos = sum(1 for row in rows if int(row.get("label") or 0) == 1)
    total_neg = len(rows) - total_pos
    accepted: list[dict[str, Any]] = [row for row in rows if policy_accepts(row, policy, verifier_threshold)]
    tp = sum(1 for row in accepted if int(row.get("label") or 0) == 1)
    fp = len(accepted) - tp
    fn = total_pos - tp
    tn = total_neg - fp
    accepted_by_chemical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        accepted_by_chemical[str(row.get("chemical_inchikey") or "")].append(row)
    per_chemical_counts = [len(value) for value in accepted_by_chemical.values()]
    useful_chemicals = sum(
        1 for group in accepted_by_chemical.values() if any(int(row.get("label") or 0) == 1 for row in group)
    )
    accepted_label_types = Counter(str(row.get("label_type") or "unknown") for row in accepted)
    accepted_negative_label_types = Counter(
        str(row.get("label_type") or "unknown") for row in accepted if int(row.get("label") or 0) == 0
    )
    accepted_scores = [float(row.get("verifier_score") or 0.0) for row in accepted]
    return {
        "policy": policy,
        "rows": len(rows),
        "accepted": len(accepted),
        "acceptance_rate": safe_div(len(accepted), len(rows)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, total_pos),
        "false_positive_rate": safe_div(fp, total_neg),
        "negative_rejection_rate": safe_div(tn, total_neg),
        "cost_per_true_positive": safe_div(len(accepted), tp),
        "accepted_chemicals": len(accepted_by_chemical),
        "useful_chemicals": useful_chemicals,
        "mean_candidates_per_accepted_chemical": float(statistics.mean(per_chemical_counts)) if per_chemical_counts else 0.0,
        "p95_candidates_per_accepted_chemical": percentile(per_chemical_counts, 0.95),
        "mean_verifier_score_accepted": float(statistics.mean(accepted_scores)) if accepted_scores else 0.0,
        "accepted_label_types": dict(accepted_label_types.most_common()),
        "accepted_negative_label_types": dict(accepted_negative_label_types.most_common()),
    }


def top_evidence_cards(rows: list[dict[str, Any]], threshold: float, *, limit: int) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if int(row.get("label") or 0) == 1 and float(row.get("verifier_score") or 0.0) >= threshold
    ]
    candidates.sort(
        key=lambda row: (
            str(row.get("source") or "") != "similarity_bridge_filtered",
            -float(row.get("verifier_score") or 0.0),
            -float(row.get("tanimoto") or 0.0),
            str(row.get("label_type") or ""),
        )
    )
    cards: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in candidates:
        key = (
            str(row.get("chemical_inchikey") or ""),
            str(row.get("enzyme_inchikey") or ""),
            str(row.get("bridge_direction") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        ecs = []
        try:
            ecs = [str(item) for item in json.loads(str(row.get("enzyme_ec_sample_json") or "[]")) if item]
        except Exception:
            ecs = []
        card = {
            "bridge_id": f"bridge_v0_{len(cards) + 1:03d}",
            "chemical_smiles": row.get("chemical_smiles") or "",
            "enzyme_smiles": row.get("enzyme_smiles") or "",
            "chemical_inchikey": row.get("chemical_inchikey") or "",
            "enzyme_inchikey": row.get("enzyme_inchikey") or "",
            "bridge_direction": row.get("bridge_direction") or "",
            "label_type": row.get("label_type") or "",
            "source": row.get("source") or "",
            "tanimoto": round(float(row.get("tanimoto") or 0.0), 4),
            "verifier_score": round(float(row.get("verifier_score") or 0.0), 6),
            "enzyme_ec_sample": ecs[:8],
            "evidence": [
                "heldout verifier test split positive",
                f"verifier_score >= {threshold:.3f}",
                "enzyme EC evidence present" if ecs else "enzyme EC evidence missing",
                "exact InChIKey bridge" if key[0] == key[1] else "high-similarity bridge",
            ],
        }
        cards.append(card)
        if len(cards) >= limit:
            break
    return cards


def render_markdown(report: dict[str, Any], cards: list[dict[str, Any]]) -> str:
    lines = [
        "# Bridge Gate Ablation v0",
        "",
        "This is a candidate-level P2 smoke test. It evaluates enzyme-bridge candidate filtering before route-search integration.",
        "",
        "## Inputs",
        "",
        f"- Test split rows: {report['inputs']['test_rows']:,}",
        f"- Positives: {report['inputs']['positives']:,}",
        f"- Negatives: {report['inputs']['negatives']:,}",
        f"- Verifier precision gate threshold: {report['inputs']['verifier_threshold']:.6f}",
        "",
        "## Gate Comparison",
        "",
        "| policy | accepted | precision | recall | FPR | negative rejection | cost / TP | useful chemicals |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["policies"]:
        lines.append(
            "| {policy} | {accepted:,} | {precision:.4f} | {recall:.4f} | {false_positive_rate:.4f} | "
            "{negative_rejection_rate:.4f} | {cost_per_true_positive:.2f} | {useful_chemicals:,} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Key Conclusion",
            "",
            report["conclusion"],
            "",
            "## Top Evidence Bridge Cards",
            "",
        ]
    )
    for card in cards:
        lines.extend(
            [
                f"### {card['bridge_id']}",
                "",
                f"- Direction: `{card['bridge_direction']}`",
                f"- Label type: `{card['label_type']}`",
                f"- Tanimoto: {card['tanimoto']}",
                f"- Verifier score: {card['verifier_score']}",
                f"- EC sample: {', '.join(card['enzyme_ec_sample']) if card['enzyme_ec_sample'] else 'N/A'}",
                f"- Chemical: `{card['chemical_smiles']}`",
                f"- Enzyme-side molecule: `{card['enzyme_smiles']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Scope Note",
            "",
            "This report does not claim route-level solved-rate improvement. It closes the verifier-gate smoke step and should be followed by route-level native / ungated / gated / gated+verifier ablation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge verifier gate ablation v0")
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=DEFAULT_VERIFIER_THRESHOLD)
    parser.add_argument("--evidence-card-limit", type=int, default=20)
    args = parser.parse_args()

    started = time.monotonic()
    test_rows = read_rows(args.pack_dir / "verifier_test.parquet")
    score_rows = read_rows(args.model_dir / "test_scores.parquet")
    rows = align_rows(test_rows, score_rows)
    positives = sum(1 for row in rows if int(row.get("label") or 0) == 1)
    negatives = len(rows) - positives
    policies = [
        "native_no_bridge",
        "ungated_bridge",
        "tanimoto_ge_0_50",
        "tanimoto_ge_0_80",
        "verifier_ge_0_50",
        "verifier_precision_gate",
        "tanimoto_0_80_and_verifier_precision_gate",
    ]
    policy_rows = [summarize_policy(rows, policy, args.threshold) for policy in policies]
    by_label_type = Counter(str(row.get("label_type") or "unknown") for row in rows)
    cards = top_evidence_cards(rows, args.threshold, limit=max(0, int(args.evidence_card_limit)))
    ungated = next(row for row in policy_rows if row["policy"] == "ungated_bridge")
    gated = next(row for row in policy_rows if row["policy"] == "verifier_precision_gate")
    conclusion = (
        "Verifier gating changes the enzyme bridge sidecar from a high-recall but noisy candidate pool "
        f"to a high-precision pool: precision {ungated['precision']:.4f} -> {gated['precision']:.4f}, "
        f"FPR {ungated['false_positive_rate']:.4f} -> {gated['false_positive_rate']:.4f}, "
        f"cost/TP {ungated['cost_per_true_positive']:.2f} -> {gated['cost_per_true_positive']:.2f}. "
        "The next required step is route-level wiring and ablation."
    )
    report = {
        "schema_version": "bridge_gate_ablation_v0.candidate_level.v1",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "inputs": {
            "pack_dir": str(args.pack_dir),
            "model_dir": str(args.model_dir),
            "test_rows": len(rows),
            "positives": positives,
            "negatives": negatives,
            "verifier_threshold": float(args.threshold),
            "label_type_distribution": dict(by_label_type.most_common()),
        },
        "policies": policy_rows,
        "conclusion": conclusion,
        "outputs": {
            "report_json": str(args.output_dir / "bridge_gate_ablation_report.json"),
            "report_md": str(args.output_dir / "bridge_gate_ablation_report.md"),
            "evidence_cards_json": str(args.output_dir / "bridge_evidence_cards.json"),
            "evidence_cards_parquet": str(args.output_dir / "bridge_evidence_cards.parquet"),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "bridge_gate_ablation_report.json", report)
    write_json(args.output_dir / "bridge_evidence_cards.json", cards)
    write_parquet(args.output_dir / "bridge_evidence_cards.parquet", cards)
    (args.output_dir / "bridge_gate_ablation_report.md").write_text(render_markdown(report, cards), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "conclusion": conclusion}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
