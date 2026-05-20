"""Summarize CascadeBench Phase II selector experiments.

This script turns the scattered smoke reports into a reproducible decision
artifact.  It intentionally does not claim a promoted model; it checks whether
the observed metrics support search integration.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = "cascadebench_phase2_decision_summary.v1"
MIN_MEANINGFUL_DELTA = 0.01


def summarize_cascadebench_phase2(*, root: Path, output_json: Path) -> dict[str, Any]:
    reports = {
        "transform_pair_train_vocab": _selector_report(
            root / "v4_transform_pair_selector_smoke_connector_trainvocab" / "selector_report.json",
            baseline_key="baseline_frequency",
        ),
        "transform_pair_evidence_lightgbm": _selector_report(
            root / "v4_transform_pair_selector_smoke_connector_trainvocab_evidence" / "selector_report.json",
            baseline_key="baseline_frequency",
        ),
        "template_fullmap_analog": _selector_report(
            root / "v4_template_selector_train50_to_test50_fullmap_analog" / "selector_report.json",
            baseline_key="baseline_proposal_score",
        ),
        "template_twostage_freq_top3_analog": _selector_report(
            root / "v4_template_selector_train50_to_test50_twostage_freq_top3_analog" / "selector_report.json",
            baseline_key="baseline_proposal_score",
        ),
        "template_twostage_freq_top3_pair": _selector_report(
            root / "v4_template_selector_train50_to_test50_twostage_freq_top3_pair" / "selector_report.json",
            baseline_key="baseline_proposal_score",
        ),
        "template_twostage_freq_top3_app_analog": _selector_report(
            root / "v4_template_selector_train50_to_test50_twostage_freq_top3_app_analog" / "selector_report.json",
            baseline_key="baseline_proposal_score",
        ),
        "template_twostage_freq_top3_app_pair": _selector_report(
            root / "v4_template_selector_train50_to_test50_twostage_freq_top3_app_pair" / "selector_report.json",
            baseline_key="baseline_proposal_score",
        ),
        "template_twostage_freq_top3_rc_analog": _selector_report(
            root / "v4_template_selector_train50_to_test50_twostage_freq_top3_rc_analog" / "selector_report.json",
            baseline_key="baseline_proposal_score",
        ),
        "template_twostage_freq_top3_rc_pair": _selector_report(
            root / "v4_template_selector_train50_to_test50_twostage_freq_top3_rc_pair" / "selector_report.json",
            baseline_key="baseline_proposal_score",
        ),
    }
    template_generation = {
        "global_top80": _template_generation(root / "v4_template_bridge_test50_fullmap" / "template_bridge.json"),
        "oracle_transform_upper_bound": _template_generation(root / "v4_template_bridge_test50_fullmap_oracle_transform" / "template_bridge.json"),
        "frequency_top3": _template_generation(root / "v4_template_bridge_test50_two_stage_frequency_top3" / "template_bridge.json"),
        "frequency_top5": _template_generation(root / "v4_template_bridge_test50_two_stage_frequency_top5" / "template_bridge.json"),
        "selector_top3": _template_generation(root / "v4_template_bridge_test50_two_stage_selector_top3" / "template_bridge.json"),
        "selector_top5": _template_generation(root / "v4_template_bridge_test50_two_stage_selector_top5" / "template_bridge.json"),
        "selector_top10": _template_generation(root / "v4_template_bridge_test50_two_stage_selector_top10" / "template_bridge.json"),
        "frequency_plus_evidence_top3": _template_generation(root / "v4_template_bridge_test50_two_stage_freq_joint_alpha2_top3" / "template_bridge.json"),
        "frequency_top3_rc": _template_generation(root / "v4_template_bridge_test50_two_stage_frequency_top3_rc" / "template_bridge.json"),
    }
    rc_probe = _proposal_score_probe(
        root / "v4_template_bridge_test50_two_stage_frequency_top3_rc" / "template_bridge.json",
        specs={
            "proposal_score": lambda row: float(row.get("proposal_score") or 0.0),
            "rc_inherited_atom_fraction": lambda row: float(row.get("rc_inherited_atom_fraction") or 0.0),
            "proposal_plus_rc_matched_fraction": lambda row: float(row.get("proposal_score") or 0.0)
            + float(row.get("rc_template_matched_fraction") or 0.0),
        },
    )
    gates = _decision_gates(reports=reports, template_generation=template_generation, rc_probe=rc_probe)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "root": str(root),
            "output_json": str(output_json),
        },
        "selector_reports": reports,
        "template_generation": template_generation,
        "rc_score_probe": rc_probe,
        "decision_gates": gates,
        "decision": _decision(gates),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    output_json.with_suffix(".md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def _selector_report(path: Path, *, baseline_key: str) -> dict[str, Any]:
    payload = _read_json(path)
    if payload is None:
        return {"path": str(path), "status": "missing"}
    baseline = ((payload.get(baseline_key) or {}).get("eval") or {})
    selector = ((payload.get("selector") or {}).get("eval") or {})
    return {
        "path": str(path),
        "status": "ok",
        "counts": payload.get("counts") or {},
        "baseline": _compact_eval(baseline),
        "selector": _compact_eval(selector),
        "delta": _delta(_compact_eval(selector), _compact_eval(baseline)),
    }


def _template_generation(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload is None:
        return {"path": str(path), "status": "missing"}
    summary = payload.get("summary") or {}
    meta = payload.get("metadata") or {}
    return {
        "path": str(path),
        "status": "ok",
        "policy": meta.get("transform_policy"),
        "top_m": meta.get("transform_top_m"),
        "elapsed_s": meta.get("elapsed_s"),
        "targets": summary.get("targets"),
        "proposals": summary.get("template_proposals"),
        "analog_any": summary.get("targets_analog_template_any"),
        "pair_and_analog_any": summary.get("targets_pair_and_analog_template_any"),
        "analog_at_5": summary.get("analog_template_at_5"),
        "pair_and_analog_at_5": summary.get("pair_and_analog_template_at_5"),
        "analog_at_50": summary.get("analog_template_at_50"),
        "pair_and_analog_at_50": summary.get("pair_and_analog_template_at_50"),
    }


def _proposal_score_probe(path: Path, *, specs: dict[str, Callable[[dict[str, Any]], float]]) -> dict[str, Any]:
    payload = _read_json(path)
    if payload is None:
        return {"path": str(path), "status": "missing"}
    rows = []
    for target in payload.get("targets") or []:
        for proposal in target.get("proposals") or []:
            if isinstance(proposal, dict):
                row = dict(proposal)
                row["target_smiles"] = target.get("target_smiles")
                rows.append(row)
    out: dict[str, Any] = {"path": str(path), "status": "ok", "rows": len(rows)}
    for label in ("analog_hit", "pair_and_analog"):
        out[label] = {name: _evaluate_rows(rows, score_fn, label=label) for name, score_fn in specs.items()}
    return out


def _evaluate_rows(rows: list[dict[str, Any]], score_fn: Callable[[dict[str, Any]], float], *, label: str) -> dict[str, Any]:
    by_target: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row in rows:
        by_target[str(row.get("target_smiles") or "")].append((row, score_fn(row)))
    ranks = []
    hits = {1: 0, 3: 0, 5: 0, 10: 0, 20: 0, 50: 0}
    positive_targets = 0
    for items in by_target.values():
        ranked = sorted(items, key=lambda item: item[1], reverse=True)
        pos = [idx for idx, (row, _) in enumerate(ranked, start=1) if row.get(label)]
        if pos:
            positive_targets += 1
            ranks.append(min(pos))
        for k in hits:
            hits[k] += int(any(row.get(label) for row, _ in ranked[:k]))
    denom = max(len(by_target), 1)
    return {
        "positive_targets": positive_targets,
        "mrr": round(sum(1.0 / rank for rank in ranks) / denom, 6),
        **{f"hit_at_{k}": round(value / denom, 6) for k, value in hits.items()},
    }


def _decision_gates(*, reports: dict[str, Any], template_generation: dict[str, Any], rc_probe: dict[str, Any]) -> dict[str, Any]:
    transform = reports.get("transform_pair_train_vocab") or {}
    transform_delta = transform.get("delta") or {}
    frequency = template_generation.get("frequency_top3") or {}
    selector = template_generation.get("selector_top3") or {}
    oracle = template_generation.get("oracle_transform_upper_bound") or {}
    twostage_analog = reports.get("template_twostage_freq_top3_analog") or {}
    twostage_pair = reports.get("template_twostage_freq_top3_pair") or {}
    rc_pair = reports.get("template_twostage_freq_top3_rc_pair") or {}
    rc_probe_pair = ((rc_probe.get("pair_and_analog") or {}).get("rc_inherited_atom_fraction") or {})
    proposal_pair = ((rc_probe.get("pair_and_analog") or {}).get("proposal_score") or {})
    return {
        "min_meaningful_delta": MIN_MEANINGFUL_DELTA,
        "transform_selector_top1_or_mrr_improves": _meaningful_gain(transform_delta.get("mrr_all_groups") or transform_delta.get("mrr")),
        "transform_selector_top5_improves": _meaningful_gain(transform_delta.get("hit_at_5")),
        "two_stage_selector_pair_beats_frequency": bool(
            (selector.get("pair_and_analog_any") or 0) > (frequency.get("pair_and_analog_any") or 0)
        ),
        "oracle_pair_upper_bound_exists": bool((oracle.get("pair_and_analog_any") or 0) > (frequency.get("pair_and_analog_any") or 0)),
        "template_twostage_analog_selector_improves_mrr": _meaningful_gain((twostage_analog.get("delta") or {}).get("mrr")),
        "template_twostage_pair_selector_improves_mrr": _meaningful_gain((twostage_pair.get("delta") or {}).get("mrr")),
        "rc_pair_selector_improves_mrr": _meaningful_gain((rc_pair.get("delta") or {}).get("mrr")),
        "rc_feature_pair_probe_improves_mrr": _meaningful_gain((rc_probe_pair.get("mrr") or 0.0) - (proposal_pair.get("mrr") or 0.0)),
    }


def _meaningful_gain(delta: Any) -> bool:
    try:
        return float(delta or 0.0) >= MIN_MEANINGFUL_DELTA
    except (TypeError, ValueError):
        return False


def _decision(gates: dict[str, Any]) -> dict[str, Any]:
    search_ready = all(
        bool(gates.get(key))
        for key in (
            "transform_selector_top1_or_mrr_improves",
            "two_stage_selector_pair_beats_frequency",
            "template_twostage_pair_selector_improves_mrr",
            "rc_pair_selector_improves_mrr",
        )
    )
    if search_ready:
        recommendation = "continue_to_search_integration"
    else:
        recommendation = "do_not_integrate_search; treat as benchmark/tooling or collect template-outcome supervision"
    return {
        "search_integration_ready": search_ready,
        "recommendation": recommendation,
        "rationale": [
            "pair_and_analog positives remain sparse",
            "selector models do not beat proposal/frequency baselines on pair_and_analog",
            "analog signals exist but do not support cascade-coherent block recovery claim",
        ],
    }


def _compact_eval(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "mrr",
        "mrr_all_groups",
        "mrr_positive_groups",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "hit_at_10",
        "hit_at_20",
        "hit_at_50",
        "analog_at_1",
        "analog_at_3",
        "analog_at_5",
        "analog_at_10",
        "analog_at_20",
        "analog_at_50",
        "pair_and_analog_at_1",
        "pair_and_analog_at_3",
        "pair_and_analog_at_5",
        "pair_and_analog_at_10",
        "pair_and_analog_at_20",
        "pair_and_analog_at_50",
    )
    return {key: row.get(key) for key in keys if key in row}


def _delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in left.items():
        if isinstance(value, (int, float)) and isinstance(right.get(key), (int, float)):
            out[key] = round(float(value) - float(right[key]), 6)
    return out


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# CascadeBench Phase II Decision Summary",
        "",
        "## Decision",
        "",
        f"- search_integration_ready: `{summary['decision']['search_integration_ready']}`",
        f"- recommendation: `{summary['decision']['recommendation']}`",
        "",
        "## Decision Gates",
        "",
        "| Gate | Pass |",
        "|---|---:|",
    ]
    for key, value in (summary.get("decision_gates") or {}).items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Template Generation", "", "| Method | Proposals | Analog Any | Pair+Analog Any | Analog@50 | Pair+Analog@50 |", "|---|---:|---:|---:|---:|---:|"])
    for name, row in (summary.get("template_generation") or {}).items():
        if row.get("status") != "ok":
            continue
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                name,
                row.get("proposals"),
                row.get("analog_any"),
                row.get("pair_and_analog_any"),
                row.get("analog_at_50"),
                row.get("pair_and_analog_at_50"),
            )
        )
    lines.extend(["", "## Selector Reports", "", "| Report | Baseline MRR | Selector MRR | Delta MRR |", "|---|---:|---:|---:|"])
    for name, row in (summary.get("selector_reports") or {}).items():
        if row.get("status") != "ok":
            continue
        base = row.get("baseline") or {}
        sel = row.get("selector") or {}
        delta = row.get("delta") or {}
        base_mrr = base.get("mrr", base.get("mrr_all_groups"))
        sel_mrr = sel.get("mrr", sel.get("mrr_all_groups"))
        delta_mrr = delta.get("mrr", delta.get("mrr_all_groups"))
        lines.append(f"| `{name}` | `{base_mrr}` | `{sel_mrr}` | `{delta_mrr}` |")
    lines.extend(["", "## Rationale", ""])
    for item in summary.get("decision", {}).get("rationale") or []:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CascadeBench Phase II selector experiments")
    parser.add_argument("--root", default="results/shared/cascadebench_strict_20260516")
    parser.add_argument("--output-json", default="results/shared/cascadebench_strict_20260516/phase2_decision_summary.json")
    args = parser.parse_args()
    summary = summarize_cascadebench_phase2(root=Path(args.root), output_json=Path(args.output_json))
    print(json.dumps({"decision": summary["decision"], "output_json": args.output_json}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
