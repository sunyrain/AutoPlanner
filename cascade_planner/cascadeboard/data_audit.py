"""Audit active CascadeBoard data assets and write a compact report."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def _pct(n: int | float, d: int | float) -> float:
    return round(100.0 * float(n) / float(d), 2) if d else 0.0


def _file_info(paths: list[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for path in paths:
        p = Path(path)
        out[path] = {
            "exists": p.exists(),
            "size_mb": round(p.stat().st_size / (1024 * 1024), 3) if p.exists() else None,
        }
    return out


def _ec_level(ec: str | None) -> str:
    if not ec:
        return "missing"
    if ";" in ec or "," in ec or "|" in ec:
        return "multi"
    parts = [p for p in ec.split(".") if p]
    return str(len(parts))


def _first_ec(step: dict[str, Any]) -> str | None:
    for cat in step.get("catalyst_components") or []:
        ec = cat.get("ec_number")
        if ec:
            return str(ec)
    return None


def _has_uniprot(step: dict[str, Any]) -> bool:
    return any((cat.get("uniprot_id") for cat in step.get("catalyst_components") or []))


def _audit_normalized_dataset(path: str) -> dict[str, Any]:
    data = _load_json(path, {})
    records = data.get("records_kept", []) or []
    excluded = data.get("records_excluded", []) or []

    cascades = []
    steps = []
    species_roles = Counter()
    species_missing_smiles = Counter()
    catalyst_classes = Counter()
    enzyme_status = Counter()
    ec_partial = 0
    with_cofactor = 0
    for rec in records:
        for cas in rec.get("cascades") or []:
            cascades.append(cas)
            for step in cas.get("steps") or []:
                steps.append(step)
                for sp in (step.get("input_species") or []) + (step.get("output_species") or []):
                    role = sp.get("role") or "unknown"
                    species_roles[role] += 1
                    if not sp.get("smiles"):
                        species_missing_smiles[role] += 1
                for cat in step.get("catalyst_components") or []:
                    catalyst_classes[cat.get("catalyst_class") or "unknown"] += 1
                    if cat.get("uniprot_status"):
                        enzyme_status[cat.get("uniprot_status")] += 1
                    if cat.get("ec_number_partial"):
                        ec_partial += 1
                    if cat.get("cofactor_required") is not None:
                        with_cofactor += 1

    rxn_status = Counter(step.get("rxn_smiles_status") or "missing" for step in steps)
    roles = Counter(step.get("step_role") or "unknown" for step in steps)
    transforms = Counter(step.get("transformation_superclass") or "unknown" for step in steps)
    domains = Counter(cas.get("route_domain") or "unknown" for cas in cascades)
    operation_modes = Counter(cas.get("operation_mode") or "unknown" for cas in cascades)
    declared_lengths = Counter(int(cas.get("total_steps") or 0) for cas in cascades)
    actual_lengths = Counter(len(cas.get("steps") or []) for cas in cascades)
    total_steps_mismatch = sum(
        1
        for cas in cascades
        if int(cas.get("total_steps") or 0) != len(cas.get("steps") or [])
    )

    with_temp = sum(1 for step in steps if (step.get("step_conditions") or {}).get("temperature_c") is not None)
    with_ph = sum(1 for step in steps if (step.get("step_conditions") or {}).get("ph") is not None)
    with_solvent = sum(1 for step in steps if (step.get("step_conditions") or {}).get("solvent") is not None)
    with_yield = sum(1 for step in steps if (step.get("step_outcome") or {}).get("step_yield_percent") is not None)
    with_ee = sum(1 for step in steps if (step.get("step_outcome") or {}).get("step_ee_percent") is not None)
    with_conversion = sum(1 for step in steps if (step.get("step_outcome") or {}).get("step_conversion_percent") is not None)
    with_ec = sum(1 for step in steps if _first_ec(step))
    with_ec4 = sum(1 for step in steps if _ec_level(_first_ec(step)) == "4")
    with_uniprot = sum(1 for step in steps if _has_uniprot(step))

    return {
        "path": path,
        "n_records": len(records),
        "n_records_excluded": len(excluded),
        "n_cascades": len(cascades),
        "n_steps": len(steps),
        "route_domains": dict(domains.most_common()),
        "operation_modes": dict(operation_modes.most_common()),
        "cascade_lengths_declared": dict(sorted(declared_lengths.items())),
        "cascade_lengths_actual": dict(sorted(actual_lengths.items())),
        "total_steps_mismatch": total_steps_mismatch,
        "step_roles_top20": dict(roles.most_common(20)),
        "transformations_top20": dict(transforms.most_common(20)),
        "rxn_smiles_status": dict(rxn_status.most_common()),
        "field_coverage": {
            "rxn_ok": {"n": rxn_status.get("ok", 0), "pct": _pct(rxn_status.get("ok", 0), len(steps))},
            "ec_any": {"n": with_ec, "pct": _pct(with_ec, len(steps))},
            "ec_4level": {"n": with_ec4, "pct": _pct(with_ec4, len(steps))},
            "uniprot": {"n": with_uniprot, "pct": _pct(with_uniprot, len(steps))},
            "temperature": {"n": with_temp, "pct": _pct(with_temp, len(steps))},
            "ph": {"n": with_ph, "pct": _pct(with_ph, len(steps))},
            "solvent": {"n": with_solvent, "pct": _pct(with_solvent, len(steps))},
            "yield": {"n": with_yield, "pct": _pct(with_yield, len(steps))},
            "ee": {"n": with_ee, "pct": _pct(with_ee, len(steps))},
            "conversion": {"n": with_conversion, "pct": _pct(with_conversion, len(steps))},
        },
        "species_roles": dict(species_roles.most_common()),
        "species_missing_smiles_by_role": dict(species_missing_smiles.most_common()),
        "catalyst_classes": dict(catalyst_classes.most_common()),
        "enzyme_uniprot_status": dict(enzyme_status.most_common()),
        "ec_partial_annotations": ec_partial,
        "cofactor_required_field_present": with_cofactor,
    }


def _audit_benchmark(path: str) -> dict[str, Any]:
    rows = _load_json(path, [])
    domains = Counter(r.get("route_domain") or "unknown" for r in rows)
    depths = Counter(int(r.get("depth") or 0) for r in rows)
    step_domains = defaultdict(int)
    n_steps = 0
    n_ec_steps = 0
    for r in rows:
        for step in r.get("gt_route") or []:
            n_steps += 1
            if step.get("ec_number"):
                n_ec_steps += 1
            step_domains[r.get("route_domain") or "unknown"] += 1
    return {
        "path": path,
        "n_targets": len(rows),
        "route_domains": dict(domains.most_common()),
        "depths": dict(sorted(depths.items())),
        "n_gt_steps": n_steps,
        "gt_steps_with_ec": n_ec_steps,
        "gt_steps_with_ec_pct": _pct(n_ec_steps, n_steps),
        "gt_steps_by_domain": dict(step_domains),
    }


def _audit_candidate_assets() -> dict[str, Any]:
    rc = _load_json("results/shared/retrochimera_candidates_depth2.json", {})
    enz_summary = _load_json("results/shared/enzexpand_candidates_100.json.summary.json", {})
    dt_summary = _load_json("results/shared/enzexpand_dualtower_candidates_100.json.summary.json", {})
    cand_sup = _load_json("results/shared/cascadeboard_candidate_supervision_v1.json", {})
    cand_train = _load_json("results/v2/cascadeboard_candidate_supervision_report.json", {})
    strict_rc = _load_json("results/v2/cascadeboard_real_benchmark.json", {})
    strict_dt = _load_json("results/v2/cascadeboard_real_benchmark_enzexpand.json", {})
    pref = _load_json("results/v2/cascadeboard_preference_data_report.json", {})
    esm_index = _load_json("results/shared/esm_cache/cache_index.json", {})

    rc_candidates = sum(len(v or []) for v in rc.values()) if isinstance(rc, dict) else 0
    rc_nonempty = sum(1 for v in rc.values() if v) if isinstance(rc, dict) else 0

    return {
        "retrochimera_cache": {
            "n_products": len(rc) if isinstance(rc, dict) else 0,
            "n_products_nonempty": rc_nonempty,
            "n_candidates": rc_candidates,
        },
        "enzexpand_cache": enz_summary,
        "enzexpand_dual_tower_annotation": dt_summary,
        "esm_cache": {
            "n_embeddings": len(esm_index) if isinstance(esm_index, dict) else 0,
            "with_uniprot_id": sum(1 for e in esm_index.values() if e.get("uniprot_id")) if isinstance(esm_index, dict) else 0,
            "with_ec_number": sum(1 for e in esm_index.values() if e.get("ec_number")) if isinstance(esm_index, dict) else 0,
        },
        "candidate_supervision_dataset": {
            "metadata": cand_sup.get("metadata"),
            "overall": cand_sup.get("overall"),
            "all_enzymatic": (cand_sup.get("by_domain") or {}).get("all_enzymatic"),
            "chemoenzymatic": (cand_sup.get("by_domain") or {}).get("chemoenzymatic"),
            "by_ec1": cand_sup.get("by_ec1"),
        },
        "candidate_reranker_training": {
            "metadata": cand_train.get("metadata"),
            "best_val_loss": cand_train.get("best_val_loss"),
        },
        "strict_benchmarks": {
            "retrochimera_only": strict_rc.get("overall"),
            "retrochimera_plus_enzexpand_dualtower": strict_dt.get("overall"),
            "by_domain_dualtower": strict_dt.get("by_domain"),
        },
        "preference_pairs": pref.get("counts"),
    }


def build_data_audit(
    *,
    normalized_data: str,
    benchmark: str,
) -> dict[str, Any]:
    quality = _load_json("cascade_dataset_v2.quality.json", {})
    strict_filter = _load_json("cascade_dataset_v2.strict.report.json", {})
    assets = [
        "cascade_dataset_v2.json",
        "cascade_dataset_v2.normalized.json",
        "cascade_dataset_v2.quality.json",
        "cascade_dataset_v2.strict.report.json",
        "data/benchmark_v2_100.json",
        "results/shared/retrochimera_candidates_depth2.json",
        "results/shared/enzexpand_candidates_100.json",
        "results/shared/enzexpand_dualtower_candidates_100.json",
        "results/shared/esm_cache/cache_index.json",
        "results/shared/cascadeboard_candidate_supervision_v1.json",
        "data/cascadeboard_preference_pairs.schema.json",
    ]
    audit = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d"),
            "purpose": "CascadeBoard data content and gap audit",
        },
        "active_files": _file_info(assets),
        "normalized_dataset": _audit_normalized_dataset(normalized_data),
        "quality_report_existing": quality,
        "strict_filter_report_existing": strict_filter,
        "benchmark_v2_100": _audit_benchmark(benchmark),
        "candidate_and_result_assets": _audit_candidate_assets(),
    }
    audit["strengthening_priorities"] = _strengthening_priorities(audit)
    return audit


def _strengthening_priorities(audit: dict[str, Any]) -> list[dict[str, Any]]:
    ds = audit["normalized_dataset"]
    cov = ds["field_coverage"]
    strict = audit.get("strict_filter_report_existing") or {}
    cand = audit["candidate_and_result_assets"]["candidate_supervision_dataset"]
    all_enz = cand.get("all_enzymatic") or {}
    overall = cand.get("overall") or {}
    strict_dt = audit["candidate_and_result_assets"]["strict_benchmarks"].get("retrochimera_plus_enzexpand_dualtower") or {}
    pref = audit["candidate_and_result_assets"].get("preference_pairs") or {}
    return [
        {
            "priority": "P0",
            "area": "enzymatic_candidate_recall",
            "current_evidence": {
                "all_enzymatic_candidate_pool_coverage": all_enz.get("candidate_pool_coverage"),
                "all_enzymatic_exact_gt_in_pool": all_enz.get("exact_gt_in_pool"),
                "all_enzymatic_strict_gt_at_5": (audit["candidate_and_result_assets"]["strict_benchmarks"].get("by_domain_dualtower") or {}).get("all_enzymatic", {}).get("gt_at_5"),
            },
            "needed": [
                "step-level main substrate vs cofactor/buffer/salt/water roles",
                "productive transformation vs cofactor regeneration/workup/identity/no-op labels",
                "atom-map/stereo audit labels",
                "4-level EC and UniProt coverage for enzymatic steps",
            ],
            "target": {
                "all_enzymatic_candidate_pool_coverage": ">= 50%",
                "all_enzymatic_exact_gt_in_pool": ">= 20%",
            },
        },
        {
            "priority": "P0",
            "area": "rxn_smiles_completeness_and_cleanliness",
            "current_evidence": {
                "rxn_ok_pct": cov["rxn_ok"]["pct"],
                "strict_steps_kept_pct": strict.get("step_retention_pct"),
                "dropped_rxn_status_not_ok": (strict.get("drop_reasons") or {}).get("rxn_status_not_ok"),
                "dropped_identity_rxn": (strict.get("drop_reasons") or {}).get("identity_rxn"),
            },
            "needed": [
                "complete both sides of rxn_smiles",
                "remove identity/no-op reactions from productive training targets",
                "separate incomplete reaction records from benchmark/training positives",
            ],
            "target": {"strict_step_retention_pct": ">= 60%"},
        },
        {
            "priority": "P0",
            "area": "objective_specific_preference_pairs",
            "current_evidence": pref,
            "needed": [
                "route pair labels for industrial/green/novelty/balanced objectives",
                "tie and incomparable labels kept explicit",
                "quality/risk vectors attached to both routes",
            ],
            "target": {"bt_trainable_pairs": ">= 2000 initial, >= 10000 preferred"},
        },
        {
            "priority": "P1",
            "area": "cascade_structure_consistency",
            "current_evidence": {
                "total_steps_mismatch_cascades": ds.get("total_steps_mismatch"),
                "declared_lengths": ds.get("cascade_lengths_declared"),
                "actual_lengths": ds.get("cascade_lengths_actual"),
            },
            "needed": [
                "make total_steps equal to actual steps length",
                "exclude or separately label 0/1-step records from cascade-route training",
                "standardize cascade_id/step_index ordering",
            ],
            "target": {"total_steps_mismatch_cascades": "0"},
        },
        {
            "priority": "P1",
            "area": "condition_and_outcome_labels",
            "current_evidence": {
                "temperature_pct": cov["temperature"]["pct"],
                "ph_pct": cov["ph"]["pct"],
                "yield_pct": cov["yield"]["pct"],
                "ee_pct": cov["ee"]["pct"],
                "conversion_pct": cov["conversion"]["pct"],
            },
            "needed": [
                "typed temperature/pH/solvent/buffer units",
                "yield/ee/conversion per step and route-level normalization",
                "failure/risk labels for calibration",
            ],
            "target": {
                "ph_pct": ">= 70%",
                "yield_pct": ">= 50%",
                "ee_pct": ">= 50%",
            },
        },
        {
            "priority": "P1",
            "area": "enzyme_identity_and_sequence",
            "current_evidence": {
                "dataset_uniprot_pct": cov["uniprot"]["pct"],
                "esm_embeddings": audit["candidate_and_result_assets"]["esm_cache"],
            },
            "needed": [
                "UniProt ID for each enzymatic catalyst where possible",
                "organism and enzyme name normalization",
                "sequence availability for ESM/condition heads",
            ],
            "target": {"uniprot_pct": ">= 70%"},
        },
    ]


def _fmt_pct(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        return f"{100 * float(v):.1f}%"
    except Exception:
        return str(v)


def write_markdown(audit: dict[str, Any], output: str) -> None:
    ds = audit["normalized_dataset"]
    cov = ds["field_coverage"]
    bench = audit["benchmark_v2_100"]
    assets = audit["candidate_and_result_assets"]
    cand = assets["candidate_supervision_dataset"]
    strict = assets["strict_benchmarks"]

    lines = [
        "# CascadeBoard 数据内容审计",
        "",
        f"更新时间：{audit['metadata']['date']}",
        "",
        "## 当前数据内容",
        "",
        f"- 主数据：`{ds['path']}`，{ds['n_records']} records，{ds['n_cascades']} cascades，{ds['n_steps']} steps。",
        f"- Benchmark：`{bench['path']}`，{bench['n_targets']} targets，{bench['n_gt_steps']} GT steps。",
        f"- 候选监督集：`results/shared/cascadeboard_candidate_supervision_v1.json`，{(cand.get('metadata') or {}).get('n_examples')} examples。",
        f"- ESM cache：{assets['esm_cache']['n_embeddings']} enzyme embeddings。",
        "",
        "## 主数据分布",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        f"| route domains | `{json.dumps(ds['route_domains'], ensure_ascii=False)}` |",
        f"| operation modes | `{json.dumps(ds['operation_modes'], ensure_ascii=False)}` |",
        f"| cascade lengths declared | `{json.dumps(ds['cascade_lengths_declared'], ensure_ascii=False)}` |",
        f"| cascade lengths actual | `{json.dumps(ds['cascade_lengths_actual'], ensure_ascii=False)}` |",
        f"| total_steps mismatch | {ds['total_steps_mismatch']} cascades |",
        f"| top transformations | `{json.dumps(dict(list(ds['transformations_top20'].items())[:10]), ensure_ascii=False)}` |",
        f"| rxn_smiles status | `{json.dumps(ds['rxn_smiles_status'], ensure_ascii=False)}` |",
        "",
        "## 字段覆盖率",
        "",
        "| 字段 | 覆盖 |",
        "|---|---:|",
    ]
    for key in ["rxn_ok", "ec_any", "ec_4level", "uniprot", "temperature", "ph", "solvent", "yield", "ee", "conversion"]:
        item = cov[key]
        lines.append(f"| {key} | {item['n']} / {ds['n_steps']} ({item['pct']}%) |")

    strict_filter = audit.get("strict_filter_report_existing") or {}
    lines.extend([
        "",
        "## 严格过滤结果",
        "",
        f"- steps in: {strict_filter.get('n_steps_in')}",
        f"- steps out: {strict_filter.get('n_steps_out')}",
        f"- step retention: {strict_filter.get('step_retention_pct')}%",
        f"- drop reasons: `{json.dumps(strict_filter.get('drop_reasons'), ensure_ascii=False)}`",
        "",
        "## Benchmark 与候选池结果",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| benchmark targets | {bench['n_targets']} |",
        f"| benchmark GT steps with EC | {bench['gt_steps_with_ec']} / {bench['n_gt_steps']} ({bench['gt_steps_with_ec_pct']}%) |",
        f"| RC cache products/candidates | {assets['retrochimera_cache']['n_products']} / {assets['retrochimera_cache']['n_candidates']} |",
        f"| EnzExpand candidates | {(assets['enzexpand_cache'] or {}).get('n_candidates')} |",
        f"| dual-tower annotated candidates | {(assets['enzexpand_dual_tower_annotation'] or {}).get('n_annotated')} |",
        f"| overall candidate_pool_coverage | {_fmt_pct((cand.get('overall') or {}).get('candidate_pool_coverage'))} |",
        f"| overall exact GT-in-pool | {_fmt_pct((cand.get('overall') or {}).get('exact_gt_in_pool'))} |",
        f"| all_enzymatic candidate_pool_coverage | {_fmt_pct((cand.get('all_enzymatic') or {}).get('candidate_pool_coverage'))} |",
        f"| all_enzymatic exact GT-in-pool | {_fmt_pct((cand.get('all_enzymatic') or {}).get('exact_gt_in_pool'))} |",
        f"| strict GT@5 RC only | {_fmt_pct((strict.get('retrochimera_only') or {}).get('gt_at_5'))} |",
        f"| strict GT@5 RC+Enz+dual-tower | {_fmt_pct((strict.get('retrochimera_plus_enzexpand_dualtower') or {}).get('gt_at_5'))} |",
        "",
        "## 需要精准加强的部分",
        "",
    ])
    for item in audit["strengthening_priorities"]:
        lines.extend([
            f"### {item['priority']} {item['area']}",
            "",
            f"- 当前证据：`{json.dumps(item['current_evidence'], ensure_ascii=False)}`",
            f"- 需要加强：{'; '.join(item['needed'])}",
            f"- 建议目标：`{json.dumps(item['target'], ensure_ascii=False)}`",
            "",
        ])

    Path(output).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cascade_dataset_v2.normalized.json")
    ap.add_argument("--benchmark", default="data/benchmark_v2_100.json")
    ap.add_argument("--output-json", default="results/v2/cascadeboard_data_audit.json")
    ap.add_argument("--output-md", default="DATA_CONTENT_AUDIT_CN.md")
    args = ap.parse_args()

    audit = build_data_audit(normalized_data=args.data, benchmark=args.benchmark)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(audit, args.output_md)
    print(json.dumps({
        "output_json": args.output_json,
        "output_md": args.output_md,
        "n_steps": audit["normalized_dataset"]["n_steps"],
        "candidate_pool_coverage": audit["candidate_and_result_assets"]["candidate_supervision_dataset"]["overall"].get("candidate_pool_coverage"),
        "all_enzymatic_exact_gt_in_pool": audit["candidate_and_result_assets"]["candidate_supervision_dataset"]["all_enzymatic"].get("exact_gt_in_pool"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
