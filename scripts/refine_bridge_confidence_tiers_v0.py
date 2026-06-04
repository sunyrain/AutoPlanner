"""Refine exact bridge tiers with conservative metabolite-like flags."""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


ELEMENT_RE = re.compile(r"([A-Z][a-z]?)(\\d*)")


def parse_formula(formula: str) -> dict[str, int]:
    # RDKit formulas can end with charge markers such as -4. Element parsing is
    # still reliable for the simple flags used here.
    out: dict[str, int] = {}
    for element, count in ELEMENT_RE.findall(formula or ""):
        out[element] = out.get(element, 0) + int(count or 1)
    return out


def bridge_flags(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    formula = str(row.get("formula") or "")
    counts = parse_formula(formula)
    heavy = int(row.get("heavy_atoms") or 0)
    ec_unique = int(row.get("enzyme_ec_unique") or 0)
    enzyme_occ = int(row.get("enzyme_occurrences") or 0)

    if bool(row.get("is_common_or_cofactor_like")):
        flags.append("common_or_cofactor_blacklist")
    if heavy <= 4:
        flags.append("very_small_molecule")
    if enzyme_occ >= 1000:
        flags.append("very_high_enzyme_frequency")
    if ec_unique >= 30:
        flags.append("enzyme_promiscuous_connector")
    if counts.get("P", 0) >= 1 and heavy <= 35:
        flags.append("small_phosphate_metabolite_like")
    if counts.get("P", 0) >= 2:
        flags.append("polyphosphate_or_nucleotide_like")
    if counts.get("N", 0) >= 5 and counts.get("P", 0) >= 1:
        flags.append("nucleotide_like")
    if counts.get("O", 0) >= 6 and heavy <= 18:
        flags.append("oxygen_rich_central_metabolite_like")
    if formula.endswith(("+", "-")) or re.search(r"[+-]\\d+$", formula):
        flags.append("charged_species")
    return sorted(set(flags))


def refined_tier(row: dict[str, Any], flags: list[str]) -> str:
    direction = str(row.get("bridge_direction") or "")
    if "common_or_cofactor_blacklist" in flags or "very_small_molecule" in flags:
        return "tier5_common_or_cofactor_artifact"
    if any(
        flag in flags
        for flag in [
            "small_phosphate_metabolite_like",
            "polyphosphate_or_nucleotide_like",
            "nucleotide_like",
            "oxygen_rich_central_metabolite_like",
            "enzyme_promiscuous_connector",
            "very_high_enzyme_frequency",
        ]
    ):
        return "tier4_metabolite_like_exact_bridge"
    if direction.endswith("substrate"):
        return "tier1_strict_exact_substrate_bridge"
    if direction.endswith("product"):
        return "tier2_strict_exact_product_bridge"
    return "tier3_exact_bridge_uncertain_direction"


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows) if rows else pa.table({}), path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine bridge confidence tiers")
    parser.add_argument("--pack-dir", default="data/bridge_pack_v0")
    args = parser.parse_args()

    started = time.time()
    root = Path(args.pack_dir)
    rows = pq.read_table(root / "exact_bridge_all.parquet").to_pylist()
    refined = []
    tier_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    for row in rows:
        flags = bridge_flags(row)
        tier = refined_tier(row, flags)
        row = dict(row)
        row["original_confidence_tier"] = row.get("confidence_tier") or ""
        row["confidence_tier"] = tier
        row["bridge_flags_json"] = json.dumps(flags, sort_keys=True)
        row["is_strict_training_positive"] = tier in {
            "tier1_strict_exact_substrate_bridge",
            "tier2_strict_exact_product_bridge",
        }
        refined.append(row)
        tier_counts[tier] += 1
        flag_counts.update(flags)

    strict = [row for row in refined if row["is_strict_training_positive"]]
    audit_only = [row for row in refined if not row["is_strict_training_positive"]]
    files = {
        "bridge_confidence_tiers_refined": str(root / "bridge_confidence_tiers_refined.parquet"),
        "exact_bridge_strict": str(root / "exact_bridge_strict.parquet"),
        "exact_bridge_audit_only": str(root / "exact_bridge_audit_only.parquet"),
        "refined_manifest": str(root / "refined_manifest.json"),
        "refined_report": str(root / "refined_report.md"),
    }
    write_parquet(Path(files["bridge_confidence_tiers_refined"]), refined)
    write_parquet(Path(files["exact_bridge_strict"]), strict)
    write_parquet(Path(files["exact_bridge_audit_only"]), audit_only)
    manifest = {
        "schema_version": "bridge_pack_v0.refined_tiers.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "pack_dir": str(root),
        "files": files,
        "counts": {
            "input_exact_bridge_all_rows": len(rows),
            "strict_training_positive_rows": len(strict),
            "audit_only_rows": len(audit_only),
            "tier_counts": dict(tier_counts),
            "flag_counts": dict(flag_counts),
        },
    }
    Path(files["refined_manifest"]).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(files["refined_report"]).write_text(render_report(manifest), encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2, ensure_ascii=False))


def render_report(manifest: dict[str, Any]) -> str:
    counts = manifest["counts"]
    lines = [
        "# Bridge Pack v0 Refined Tier Report",
        "",
        f"- generated_at: `{manifest['generated_at']}`",
        f"- pack_dir: `{manifest['pack_dir']}`",
        "",
        "## Summary",
        "",
        f"- input exact bridge rows: `{counts['input_exact_bridge_all_rows']}`",
        f"- strict training positives: `{counts['strict_training_positive_rows']}`",
        f"- audit-only rows: `{counts['audit_only_rows']}`",
        "",
        "## Tier Counts",
        "",
        "| Tier | Count |",
        "|---|---:|",
    ]
    for tier, count in sorted((counts.get("tier_counts") or {}).items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{tier}` | {count} |")
    lines.extend(["", "## Flag Counts", "", "| Flag | Count |", "|---|---:|"])
    for flag, count in sorted((counts.get("flag_counts") or {}).items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{flag}` | {count} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`exact_bridge_strict.parquet` is the conservative positive set for initial retriever/verifier training. "
            "`exact_bridge_audit_only.parquet` retains common, cofactor-like, and central-metabolite-like matches for review and future lower-weight training.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
