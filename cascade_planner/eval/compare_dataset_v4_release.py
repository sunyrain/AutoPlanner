"""Compare the v4 cascade release against earlier local datasets."""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


def compare_dataset_v4_release(
    *,
    v4_dir: Path,
    old_dataset: Path,
    benchmark: Path,
    training_pack: Path,
    output_json: Path,
    output_md: Path,
) -> dict[str, Any]:
    old = summarize_old_dataset(old_dataset)
    v4 = summarize_v4_release(v4_dir)
    benchmark_summary = summarize_benchmark_overlap(benchmark, old_dataset=old_dataset, v4_dir=v4_dir)
    training = summarize_training_pack(training_pack)
    comparison = {
        "metadata": {
            "v4_dir": str(v4_dir),
            "old_dataset": str(old_dataset),
            "benchmark": str(benchmark),
            "training_pack": str(training_pack),
        },
        "old_dataset": old,
        "v4_release": v4,
        "benchmark_overlap": benchmark_summary,
        "training_pack": training,
        "deltas": dataset_deltas(old, v4),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report_markdown(comparison), encoding="utf-8")
    return comparison


def summarize_old_dataset(path: Path) -> dict[str, Any]:
    records = json.loads(path.read_text(encoding="utf-8"))
    cascades = []
    for article in records if isinstance(records, list) else []:
        doi = norm(article.get("doi"))
        for cascade in article.get("cascades") or []:
            cascades.append({"doi": doi, "article": article, "cascade": cascade})
    steps = [step for row in cascades for step in row["cascade"].get("steps") or []]
    catalysts = [cat for step in steps for cat in step.get("catalyst_components") or []]
    input_species = [sp for step in steps for sp in step.get("input_species") or []]
    output_species = [sp for step in steps for sp in step.get("output_species") or []]
    scope = [entry for row in cascades for entry in row["cascade"].get("substrate_scope") or []]
    return {
        "path": str(path),
        "article_rows": len(records) if isinstance(records, list) else 0,
        "cascade_rows": len(cascades),
        "unique_doi": len({row["doi"] for row in cascades if row["doi"]}),
        "route_domain_counts": dict(Counter(norm(row["cascade"].get("route_domain")) or "missing" for row in cascades)),
        "operation_mode_counts": dict(Counter(norm(row["cascade"].get("operation_mode")) or "missing" for row in cascades)),
        "field_coverage": {
            "target_product_smiles": count_cascades(cascades, lambda c: any(norm(x.get("smiles")) for x in c.get("target_products") or [])),
            "starting_material_smiles": count_cascades(cascades, lambda c: any(norm(x.get("smiles")) for x in c.get("starting_materials") or [])),
            "overall_yield": count_cascades(cascades, lambda c: present((c.get("overall_outcome") or {}).get("overall_yield_percent"))),
            "overall_ee": count_cascades(cascades, lambda c: present((c.get("overall_outcome") or {}).get("overall_ee_percent"))),
            "total_reaction_time": count_cascades(cascades, lambda c: present((c.get("global_conditions") or {}).get("reaction_time_h"))),
            "conditions_temperature": count_cascades(cascades, lambda c: present((c.get("global_conditions") or {}).get("temperature_c"))),
            "conditions_ph": count_cascades(cascades, lambda c: present((c.get("global_conditions") or {}).get("ph"))),
            "conditions_solvent": count_cascades(cascades, lambda c: present((c.get("global_conditions") or {}).get("solvent_name"))),
            "conditions_buffer": count_cascades(cascades, lambda c: present((c.get("global_conditions") or {}).get("buffer_name"))),
        },
        "normalized_counts": {
            "steps": len(steps),
            "catalyst_components": len(catalysts),
            "input_species": len(input_species),
            "output_species": len(output_species),
            "substrate_scope_entries": len(scope),
        },
        "normalized_coverage": {
            "steps_rxn_smiles": count_rows(steps, lambda row: present(row.get("rxn_smiles"))),
            "steps_temperature": count_rows(steps, lambda row: present(((row.get("step_conditions") or {}).get("temperature_c")))),
            "steps_ph": count_rows(steps, lambda row: present(((row.get("step_conditions") or {}).get("ph")))),
            "steps_reaction_time": count_rows(steps, lambda row: present(((row.get("step_conditions") or {}).get("reaction_time_h")))),
            "catalysts_ec_number": count_rows(catalysts, lambda row: present(row.get("ec_number"))),
            "catalysts_uniprot_id": count_rows(catalysts, lambda row: present(row.get("uniprot_id"))),
            "input_species_smiles": count_rows(input_species, lambda row: present(row.get("smiles"))),
            "output_species_smiles": count_rows(output_species, lambda row: present(row.get("smiles"))),
            "substrate_scope_entries": len(scope),
        },
    }


def summarize_v4_release(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    all_rows = read_csv(root / "cascade_v4_all_quality_index.csv")
    hq_rows = read_csv(root / "cascade_v4_high_quality.csv")
    con = sqlite3.connect(root / "cascade_v4_high_quality.db")
    try:
        db_counts = {
            table: con.execute(f"select count(*) from {table}").fetchone()[0]
            for table in ("reactions", "steps", "catalysts", "species", "substrate_scope")
        }
        db_coverage = {
            "steps_rxn_smiles": db_count(con, "steps", "rxn_smiles"),
            "steps_temperature": db_count(con, "steps", "temperature_c"),
            "steps_ph": db_count(con, "steps", "ph"),
            "steps_reaction_time": db_count(con, "steps", "reaction_time_h"),
            "catalysts_ec_number": db_count(con, "catalysts", "ec_number"),
            "catalysts_uniprot_id": db_count(con, "catalysts", "uniprot_id"),
            "species_smiles": db_count(con, "species", "smiles"),
            "substrate_scope_entries": db_counts["substrate_scope"],
        }
        catalyst_classes = dict(con.execute(
            "select coalesce(nullif(catalyst_class,''),'missing'), count(*) from catalysts "
            "group by coalesce(nullif(catalyst_class,''),'missing') order by count(*) desc"
        ).fetchall())
        step_modes = dict(con.execute(
            "select coalesce(nullif(step_mode,''),'missing'), count(*) from steps "
            "group by coalesce(nullif(step_mode,''),'missing') order by count(*) desc"
        ).fetchall())
        rxn_status = dict(con.execute(
            "select coalesce(nullif(rxn_smiles_status,''),'missing'), count(*) from steps "
            "group by coalesce(nullif(rxn_smiles_status,''),'missing') order by count(*) desc"
        ).fetchall())
    finally:
        con.close()
    compat = Counter()
    evidence = Counter()
    preferred_mode = Counter()
    for line in (root / "cascade_v4_high_quality.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        comp = row.get("compatibility") or {}
        compat[norm(comp.get("compatibility_label")) or "missing"] += 1
        evidence[norm(comp.get("evidence_strength")) or "missing"] += 1
        preferred_mode[norm((comp.get("comparison_to_baseline") or {}).get("comparator_preferred_mode")) or "missing"] += 1
    return {
        "path": str(root),
        "manifest_quality_summary": manifest.get("quality_summary") or {},
        "all_rows": len(all_rows),
        "high_quality_rows": len(hq_rows),
        "unique_doi_all": len({norm(row.get("doi")) for row in all_rows if norm(row.get("doi"))}),
        "unique_doi_high_quality": len({norm(row.get("doi")) for row in hq_rows if norm(row.get("doi"))}),
        "quality_tier_counts": dict(Counter(norm(row.get("quality_tier")) or "missing" for row in all_rows)),
        "route_domain_counts": dict(Counter(norm(row.get("cascade_type")) or "missing" for row in hq_rows)),
        "field_coverage_all": {
            "target_product_smiles": count_rows(all_rows, lambda row: present(row.get("target_product_smiles"))),
            "starting_material_smiles": count_rows(all_rows, lambda row: present(row.get("starting_material_smiles"))),
            "overall_yield": count_rows(all_rows, lambda row: present(row.get("overall_yield"))),
            "overall_ee": count_rows(all_rows, lambda row: present(row.get("overall_ee"))),
            "total_reaction_time": count_rows(all_rows, lambda row: present(row.get("total_reaction_time"))),
            "has_conditions": count_rows(all_rows, lambda row: truthy(row.get("has_conditions"))),
            "has_compatibility_annotation": count_rows(all_rows, lambda row: truthy(row.get("has_compatibility_annotation"))),
            "rxn_smiles_steps": sum_int(all_rows, "rxn_smiles_steps"),
        },
        "field_coverage_high_quality": {
            "target_product_smiles": count_rows(hq_rows, lambda row: present(row.get("target_product_smiles"))),
            "starting_material_smiles": count_rows(hq_rows, lambda row: present(row.get("starting_material_smiles"))),
            "overall_yield": count_rows(hq_rows, lambda row: present(row.get("overall_yield"))),
            "overall_ee": count_rows(hq_rows, lambda row: present(row.get("overall_ee"))),
            "total_reaction_time": count_rows(hq_rows, lambda row: present(row.get("total_reaction_time"))),
            "has_conditions": count_rows(hq_rows, lambda row: truthy(row.get("has_conditions"))),
            "has_compatibility_annotation": count_rows(hq_rows, lambda row: truthy(row.get("has_compatibility_annotation"))),
            "rxn_smiles_steps": sum_int(hq_rows, "rxn_smiles_steps"),
        },
        "normalized_counts_high_quality": db_counts,
        "normalized_coverage_high_quality": db_coverage,
        "catalyst_class_counts": catalyst_classes,
        "step_mode_counts": step_modes,
        "rxn_smiles_status_counts": rxn_status,
        "compatibility_label_counts": dict(compat),
        "compatibility_evidence_strength_counts": dict(evidence),
        "preferred_mode_counts_top20": dict(preferred_mode.most_common(20)),
    }


def summarize_benchmark_overlap(benchmark: Path, *, old_dataset: Path, v4_dir: Path) -> dict[str, Any]:
    rows = json.loads(benchmark.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("targets") or rows.get("records") or []
    benchmark_keys = {(norm(row.get("doi")), norm(row.get("cascade_id"))) for row in rows}
    old_records = json.loads(old_dataset.read_text(encoding="utf-8"))
    old_keys = {
        (norm(article.get("doi")), norm(cascade.get("cascade_id")))
        for article in old_records
        for cascade in article.get("cascades") or []
    }
    v4_rows = read_csv(v4_dir / "cascade_v4_all_quality_index.csv")
    v4_keys = {(norm(row.get("doi")), norm(row.get("cascade_id"))) for row in v4_rows}
    v4_hq_rows = read_csv(v4_dir / "cascade_v4_high_quality.csv")
    v4_hq_keys = {(norm(row.get("doi")), norm(row.get("cascade_id"))) for row in v4_hq_rows}
    return {
        "benchmark_rows": len(rows),
        "benchmark_unique_doi": len({norm(row.get("doi")) for row in rows if norm(row.get("doi"))}),
        "in_old_dataset": len(benchmark_keys & old_keys),
        "in_v4_all": len(benchmark_keys & v4_keys),
        "in_v4_high_quality": len(benchmark_keys & v4_hq_keys),
        "missing_from_v4_high_quality": sorted([list(key) for key in benchmark_keys - v4_hq_keys])[:50],
    }


def summarize_training_pack(pack_dir: Path) -> dict[str, Any]:
    manifest_path = pack_dir / "manifest.json"
    if not manifest_path.exists():
        return {"path": str(pack_dir), "missing": True}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "path": str(pack_dir),
        "counts": manifest.get("counts") or {},
        "quality": manifest.get("quality") or {},
    }


def dataset_deltas(old: dict[str, Any], v4: dict[str, Any]) -> dict[str, Any]:
    old_counts = old.get("normalized_counts") or {}
    new_counts = v4.get("normalized_counts_high_quality") or {}
    return {
        "cascade_rows": (v4.get("all_rows") or 0) - (old.get("cascade_rows") or 0),
        "high_quality_rows_vs_old_cascades": (v4.get("high_quality_rows") or 0) - (old.get("cascade_rows") or 0),
        "unique_doi_all": (v4.get("unique_doi_all") or 0) - (old.get("unique_doi") or 0),
        "steps": (new_counts.get("steps") or 0) - (old_counts.get("steps") or 0),
        "catalyst_components": (new_counts.get("catalysts") or 0) - (old_counts.get("catalyst_components") or 0),
        "species": (new_counts.get("species") or 0) - ((old_counts.get("input_species") or 0) + (old_counts.get("output_species") or 0)),
        "substrate_scope_entries": (new_counts.get("substrate_scope") or 0) - (old_counts.get("substrate_scope_entries") or 0),
    }


def report_markdown(data: dict[str, Any]) -> str:
    old = data["old_dataset"]
    v4 = data["v4_release"]
    deltas = data["deltas"]
    overlap = data["benchmark_overlap"]
    training = data["training_pack"]
    lines = [
        "# Dataset v4 vs Old Dataset",
        "",
        "## Scale",
        "",
        "| Metric | Old v3 | v4 all | v4 high quality | Delta |",
        "|---|---:|---:|---:|---:|",
        f"| cascade rows | {old.get('cascade_rows')} | {v4.get('all_rows')} | {v4.get('high_quality_rows')} | {deltas.get('cascade_rows')} |",
        f"| unique DOI | {old.get('unique_doi')} | {v4.get('unique_doi_all')} | {v4.get('unique_doi_high_quality')} | {deltas.get('unique_doi_all')} |",
        f"| steps | {(old.get('normalized_counts') or {}).get('steps')} | - | {(v4.get('normalized_counts_high_quality') or {}).get('steps')} | {deltas.get('steps')} |",
        f"| catalyst components | {(old.get('normalized_counts') or {}).get('catalyst_components')} | - | {(v4.get('normalized_counts_high_quality') or {}).get('catalysts')} | {deltas.get('catalyst_components')} |",
        f"| species | {(old.get('normalized_counts') or {}).get('input_species', 0) + (old.get('normalized_counts') or {}).get('output_species', 0)} | - | {(v4.get('normalized_counts_high_quality') or {}).get('species')} | {deltas.get('species')} |",
        f"| substrate scope | {(old.get('normalized_counts') or {}).get('substrate_scope_entries')} | - | {(v4.get('normalized_counts_high_quality') or {}).get('substrate_scope')} | {deltas.get('substrate_scope_entries')} |",
        "",
        "## Quality Tiers",
        "",
        f"- v4 tiers: `{v4.get('quality_tier_counts')}`",
        f"- v4 compatibility labels: `{v4.get('compatibility_label_counts')}`",
        f"- v4 evidence strengths: `{v4.get('compatibility_evidence_strength_counts')}`",
        "",
        "## Benchmark Overlap",
        "",
        f"- benchmark rows: `{overlap.get('benchmark_rows')}`",
        f"- in old dataset: `{overlap.get('in_old_dataset')}`",
        f"- in v4 all: `{overlap.get('in_v4_all')}`",
        f"- in v4 high quality: `{overlap.get('in_v4_high_quality')}`",
        "",
        "## Old Training Pack",
        "",
        f"- pack: `{training.get('path')}`",
        f"- counts: `{training.get('counts')}`",
        f"- quality: `{training.get('quality')}`",
        "",
        "## Field Coverage",
        "",
        "| Field | Old v3 | v4 all | v4 high quality |",
        "|---|---:|---:|---:|",
    ]
    old_cov = old.get("field_coverage") or {}
    v4_all = v4.get("field_coverage_all") or {}
    v4_hq = v4.get("field_coverage_high_quality") or {}
    for key in sorted(set(old_cov) | set(v4_all) | set(v4_hq)):
        lines.append(f"| `{key}` | {old_cov.get(key, '')} | {v4_all.get(key, '')} | {v4_hq.get(key, '')} |")
    lines.extend(["", "## Normalized Coverage", "", "| Field | Old v3 | v4 high quality |", "|---|---:|---:|"])
    old_norm = old.get("normalized_coverage") or {}
    v4_norm = v4.get("normalized_coverage_high_quality") or {}
    for key in sorted(set(old_norm) | set(v4_norm)):
        lines.append(f"| `{key}` | {old_norm.get(key, '')} | {v4_norm.get(key, '')} |")
    lines.append("")
    return "\n".join(lines)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def db_count(con: sqlite3.Connection, table: str, column: str) -> int:
    return int(con.execute(f"select count(*) from {table} where {column} is not null and trim(cast({column} as text)) != ''").fetchone()[0])


def count_cascades(rows: list[dict[str, Any]], predicate) -> int:
    return sum(1 for row in rows if predicate(row["cascade"]))


def count_rows(rows: list[Any], predicate) -> int:
    return sum(1 for row in rows if predicate(row))


def sum_int(rows: list[dict[str, Any]], key: str) -> int:
    total = 0
    for row in rows:
        try:
            total += int(float(row.get(key) or 0))
        except (TypeError, ValueError):
            pass
    return total


def present(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text and text.lower() not in {"none", "null", "nan", "unknown", "not specified"})


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def norm(value: Any) -> str:
    return str(value or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare dataset_v4_release with old local datasets")
    ap.add_argument("--v4-dir", default="dataset_v4_release")
    ap.add_argument("--old-dataset", default="cascade_dataset_v3.json")
    ap.add_argument("--benchmark", default="data/benchmark_v2_100.json")
    ap.add_argument("--training-pack", default="results/shared/training_pack/broad_20260507")
    ap.add_argument("--output-json", default="results/shared/dataset_v4_analysis/v4_vs_v3_comparison.json")
    ap.add_argument("--output-md", default="results/shared/dataset_v4_analysis/v4_vs_v3_comparison.md")
    args = ap.parse_args()
    report = compare_dataset_v4_release(
        v4_dir=Path(args.v4_dir),
        old_dataset=Path(args.old_dataset),
        benchmark=Path(args.benchmark),
        training_pack=Path(args.training_pack),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md),
    )
    print(json.dumps({
        "output_json": args.output_json,
        "output_md": args.output_md,
        "deltas": report["deltas"],
        "benchmark_overlap": report["benchmark_overlap"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
