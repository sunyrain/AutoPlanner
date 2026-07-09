"""Audit condition-data coverage in route artifacts and training packs."""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CONDITION_FIELDS = ["T", "pH", "solvent", "catalyst", "ec", "enzyme_uid"]
EVIDENCE_FIELDS = ["doi", "pmid", "uniprot_accession", "cofactor", "cofactor_regeneration_mode"]


def audit_paths(paths: Iterable[Path]) -> dict[str, Any]:
    totals = Counter()
    by_source: dict[str, Counter] = defaultdict(Counter)
    files_loaded = 0
    files_skipped = 0
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            files_skipped += 1
            continue
        files_loaded += 1
        for step in iter_steps(data):
            update_counts(totals, step, prefix="step")
            by_source[step.get("source") or "unknown"].update(source_counts(step, prefix="step"))
            for cand in ((step.get("candidate_pool") or {}).get("top_candidates") or []):
                update_counts(totals, cand, prefix="candidate")
                by_source[cand.get("source") or "unknown"].update(source_counts(cand, prefix="candidate"))
    return finalize_report(totals, by_source, files_loaded=files_loaded, files_skipped=files_skipped)


def audit_training_pack(pack_dir: Path) -> dict[str, Any]:
    totals = Counter()
    by_source: dict[str, Counter] = defaultdict(Counter)
    path = pack_dir / "candidate_ranking.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing candidate_ranking.jsonl: {pack_dir}")
    rows = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows += 1
        row = json.loads(line)
        cand = row.get("candidate") or {}
        update_counts(totals, cand, prefix="candidate")
        by_source[cand.get("source") or "unknown"].update(source_counts(cand, prefix="candidate"))
    report = finalize_report(totals, by_source, files_loaded=1, files_skipped=0)
    report["training_pack_rows"] = rows
    report["training_pack"] = str(pack_dir)
    return report


def iter_steps(value: Any):
    if isinstance(value, dict):
        if isinstance(value.get("steps"), list):
            for step in value.get("steps") or []:
                yield step
        for child in value.values():
            yield from iter_steps(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_steps(child)


def update_counts(counter: Counter, row: dict[str, Any], *, prefix: str) -> None:
    counter[f"{prefix}_total"] += 1
    if has_any(row, CONDITION_FIELDS):
        counter[f"{prefix}_has_any_condition_field"] += 1
    if has_all(row, ["T", "pH"]):
        counter[f"{prefix}_has_T_and_pH"] += 1
    for field in CONDITION_FIELDS:
        if present(row.get(field)):
            counter[f"{prefix}_has_{field}"] += 1
    evidence = row.get("evidence") or {}
    if any(present(row.get(field)) or present(evidence.get(field)) for field in EVIDENCE_FIELDS):
        counter[f"{prefix}_has_evidence"] += 1
    for field in EVIDENCE_FIELDS:
        if present(row.get(field)) or present(evidence.get(field)):
            counter[f"{prefix}_has_{field}"] += 1


def source_counts(row: dict[str, Any], *, prefix: str) -> Counter:
    counter = Counter()
    update_counts(counter, row, prefix=prefix)
    return counter


def present(value: Any) -> bool:
    return value not in (None, "", [], {})


def has_any(row: dict[str, Any], fields: list[str]) -> bool:
    return any(present(row.get(field)) for field in fields)


def has_all(row: dict[str, Any], fields: list[str]) -> bool:
    return all(present(row.get(field)) for field in fields)


def finalize_report(
    totals: Counter,
    by_source: dict[str, Counter],
    *,
    files_loaded: int,
    files_skipped: int,
) -> dict[str, Any]:
    return {
        "files_loaded": files_loaded,
        "files_skipped": files_skipped,
        "totals": dict(totals),
        "coverage": coverage(totals),
        "by_source": {
            source: {"totals": dict(counts), "coverage": coverage(counts)}
            for source, counts in sorted(by_source.items())
        },
    }


def coverage(counts: Counter) -> dict[str, float | None]:
    out = {}
    for prefix in ("step", "candidate"):
        total = counts.get(f"{prefix}_total", 0)
        for key, value in counts.items():
            if key.startswith(f"{prefix}_has_"):
                out[key] = round(value / total, 6) if total else None
    return out


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Condition Data Coverage Audit",
        "",
        f"- files loaded: `{report.get('files_loaded')}`",
        f"- files skipped: `{report.get('files_skipped')}`",
    ]
    if report.get("training_pack"):
        lines.append(f"- training pack: `{report.get('training_pack')}`")
        lines.append(f"- training pack rows: `{report.get('training_pack_rows')}`")
    lines.extend(["", "## Overall", "", coverage_table(report.get("totals") or {})])
    lines.extend(["", "## By Source", ""])
    for source, row in (report.get("by_source") or {}).items():
        lines.extend([f"### `{source}`", "", coverage_table(row.get("totals") or {}), ""])
    lines.extend([
        "## Interpretation",
        "",
        "Candidate-level condition models need non-empty T, pH, solvent, catalyst, and evidence fields. Low candidate coverage means route-level condition/compatibility learning is weakly supervised.",
        "",
    ])
    return "\n".join(lines)


def coverage_table(counts: dict[str, int]) -> str:
    rows = ["| Metric | Count | Coverage |", "|---|---:|---:|"]
    for prefix in ("step", "candidate"):
        total = int(counts.get(f"{prefix}_total") or 0)
        rows.append(f"| `{prefix}_total` | {total} | - |")
        for key in sorted(k for k in counts if k.startswith(f"{prefix}_has_")):
            value = int(counts.get(key) or 0)
            cov = value / total if total else 0.0
            rows.append(f"| `{key}` | {value} | {cov:.3f} |")
    return "\n".join(rows)


def expand_patterns(patterns: list[str]) -> list[Path]:
    out = []
    seen = set()
    for pattern in patterns:
        matches = glob.glob(pattern) or [pattern]
        for match in matches:
            path = Path(match)
            if path.is_file() and path not in seen:
                out.append(path)
                seen.add(path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit condition-data coverage")
    ap.add_argument("--inputs", nargs="*", default=["results/v2/*.json"])
    ap.add_argument("--training-pack", default=None)
    ap.add_argument("--json-output", default="results/v2/condition_data_coverage_20260507.json")
    ap.add_argument("--md-output", default="results/v2/condition_data_coverage_20260507.md")
    args = ap.parse_args()
    report = audit_training_pack(Path(args.training_pack)) if args.training_pack else audit_paths(expand_patterns(args.inputs))
    Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.md_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md_output).write_text(report_markdown(report), encoding="utf-8")
    print(json.dumps({
        "json_output": args.json_output,
        "md_output": args.md_output,
        "candidate_total": report.get("totals", {}).get("candidate_total", 0),
        "candidate_has_T_and_pH": report.get("totals", {}).get("candidate_has_T_and_pH", 0),
    }, indent=2))


if __name__ == "__main__":
    main()
