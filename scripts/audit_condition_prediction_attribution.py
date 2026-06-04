"""Attribute route condition warnings for benchmark route rows."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.eval.product_route_feasibility_audit import (
    audit_route_condition_profile,
    build_product_route_feasibility_audit,
)


ENZYME_ROUTE_FLAG_KEYS = ("has_enzyme_step", "has_sp_v1_accepted_enzyme_step")


def main() -> None:
    parser = argparse.ArgumentParser(description="Attribute condition warnings in route benchmark rows.")
    parser.add_argument(
        "--rows",
        default=(
            "results/shared/statin_enhanced_formal_depth20_budget480_top20_20260602/"
            "enhanced_all9/native_vs_enhanced_route_rows.jsonl"
        ),
    )
    parser.add_argument(
        "--targets",
        default="results/shared/statin_enhanced_combo_20260601/statin_target_rows_all9.jsonl",
    )
    parser.add_argument(
        "--output-json",
        default=(
            "results/shared/statin_enhanced_formal_depth20_budget480_top20_20260602/"
            "enhanced_all9/condition_prediction_attribution.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=(
            "results/shared/statin_enhanced_formal_depth20_budget480_top20_20260602/"
            "enhanced_all9/condition_prediction_attribution.md"
        ),
    )
    parser.add_argument("--examples-per-bucket", type=int, default=5)
    args = parser.parse_args()

    rows_path = Path(args.rows)
    target_path = Path(args.targets)
    rows = _read_jsonl(rows_path)
    target_rows = _read_jsonl(target_path)
    name_by_smiles = {str(row.get("target_smiles") or ""): str(row.get("name") or "") for row in target_rows}

    audit = build_product_route_feasibility_audit(
        {
            "metadata": {"source": "condition_prediction_attribution"},
            "targets": [_audit_target_payload(idx, row, name_by_smiles) for idx, row in enumerate(rows)],
        }
    )
    report = build_attribution_report(
        rows=rows,
        audit=audit,
        name_by_smiles=name_by_smiles,
        rows_path=rows_path,
        examples_per_bucket=max(0, int(args.examples_per_bucket)),
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps({"json": str(output_json), "markdown": str(output_md)}, indent=2))


def build_attribution_report(
    *,
    rows: list[dict[str, Any]],
    audit: dict[str, Any],
    name_by_smiles: dict[str, str],
    rows_path: Path,
    examples_per_bucket: int,
) -> dict[str, Any]:
    route_records: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    route_risks: Counter[str] = Counter()
    route_issues: Counter[str] = Counter()
    route_classes: Counter[str] = Counter()
    route_issue_by_target: dict[str, Counter[str]] = defaultdict(Counter)
    route_risk_by_target: dict[str, Counter[str]] = defaultdict(Counter)
    step_risks: Counter[str] = Counter()
    step_issues: Counter[str] = Counter()
    step_domains: Counter[str] = Counter()
    step_sources: Counter[str] = Counter()
    step_source_issue_counts: dict[str, Counter[str]] = defaultdict(Counter)
    step_domain_issue_counts: dict[str, Counter[str]] = defaultdict(Counter)
    condition_sources: Counter[str] = Counter()
    condition_sources_by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    reaction_types: Counter[str] = Counter()
    route_warning_causes: Counter[str] = Counter()
    enzyme_route_risks: Counter[str] = Counter()
    enzyme_route_issues: Counter[str] = Counter()
    enzyme_step_risks: Counter[str] = Counter()
    enzyme_step_issues: Counter[str] = Counter()

    for target_index, (row, audit_target) in enumerate(zip(rows, audit.get("targets") or []), start=1):
        target = _target_name(row, name_by_smiles, target_index)
        routes = [route for route in row.get("routes") or [] if isinstance(route, dict)]
        audit_by_rank = {
            int(route_audit.get("rank") or 0): route_audit
            for route_audit in audit_target.get("routes") or []
            if isinstance(route_audit, dict)
        }
        target_route_count = 0
        target_warn_route_count = 0
        target_high_route_count = 0
        target_predicted_step_count = 0
        target_step_count = 0
        target_step_issue_counts: Counter[str] = Counter()

        for route_index, route in enumerate(routes, start=1):
            steps = [step for step in route.get("steps") or [] if isinstance(step, dict)]
            condition_audit = audit_route_condition_profile(steps)
            product_audit = audit_by_rank.get(route_index) or {}
            route_class = str(product_audit.get("route_class") or "missing")
            route_risk = str(condition_audit.get("route_risk") or "missing")
            route_issue_list = [str(issue) for issue in condition_audit.get("route_issues") or []]
            issue_counts = Counter(
                {
                    str(key): int(value or 0)
                    for key, value in (condition_audit.get("issue_counts") or {}).items()
                }
            )
            warning_causes = _route_warning_causes(condition_audit)
            strict_enzyme_route = _is_strict_enzyme_route(route)

            target_route_count += 1
            target_warn_route_count += int(route_risk == "warn")
            target_high_route_count += int(route_risk == "high")
            target_predicted_step_count += int(condition_audit.get("predicted_condition_count") or 0)
            target_step_count += int(condition_audit.get("step_count") or len(steps))
            target_step_issue_counts.update(issue_counts)

            route_risks[route_risk] += 1
            route_issues.update(route_issue_list)
            route_classes[route_class] += 1
            route_issue_by_target[target].update(route_issue_list)
            route_risk_by_target[target][route_risk] += 1
            route_warning_causes.update(warning_causes)
            if strict_enzyme_route:
                enzyme_route_risks[route_risk] += 1
                enzyme_route_issues.update(route_issue_list)

            route_record = {
                "target": target,
                "route_rank": route_index,
                "route_class": route_class,
                "route_solved": _is_solved_route(route),
                "strict_enzyme_route": strict_enzyme_route,
                "n_steps": len(steps),
                "condition_risk": route_risk,
                "condition_route_issues": route_issue_list,
                "condition_step_issue_counts": dict(sorted(issue_counts.items())),
                "warning_causes": warning_causes,
                "temperature_min_c": condition_audit.get("temperature_min_c"),
                "temperature_max_c": condition_audit.get("temperature_max_c"),
                "temperature_span_c": condition_audit.get("temperature_span_c"),
                "predicted_condition_count": condition_audit.get("predicted_condition_count"),
                "top1_score_mean": condition_audit.get("top1_score_mean"),
            }
            route_records.append(route_record)
            _add_examples(examples, "route_risk:" + route_risk, route_record, examples_per_bucket)
            for cause in warning_causes:
                _add_examples(examples, "route_cause:" + cause, route_record, examples_per_bucket)

            for step_index, step in enumerate(steps, start=1):
                step_audit = (condition_audit.get("steps") or [{}])[step_index - 1]
                step_source = str(step.get("source") or "missing")
                reaction_type = str(step.get("reaction_type") or "missing")
                domain = str(step_audit.get("domain") or "missing")
                risk = str(step_audit.get("risk") or "missing")
                issues = [str(issue) for issue in step_audit.get("issues") or []]
                condition = _top_condition(step)
                condition_source = _condition_source(condition)
                is_enzyme_step = _is_enzyme_step(step)

                step_risks[risk] += 1
                step_issues.update(issues)
                step_domains[domain] += 1
                step_sources[step_source] += 1
                reaction_types[reaction_type] += 1
                condition_sources[condition_source] += 1
                condition_sources_by_domain[domain][condition_source] += 1
                step_source_issue_counts[step_source].update(issues)
                step_domain_issue_counts[domain].update(issues)
                if is_enzyme_step or strict_enzyme_route:
                    enzyme_step_risks[risk] += 1
                    enzyme_step_issues.update(issues)

                step_record = {
                    "target": target,
                    "route_rank": route_index,
                    "step_index": step_index,
                    "route_class": route_class,
                    "strict_enzyme_route": strict_enzyme_route,
                    "is_enzyme_step": is_enzyme_step,
                    "step_source": step_source,
                    "reaction_type": reaction_type,
                    "domain": domain,
                    "risk": risk,
                    "issues": issues,
                    "notes": [str(note) for note in step_audit.get("notes") or []],
                    "condition_source": condition_source,
                    "condition_keys": sorted(str(key) for key in condition.keys()) if isinstance(condition, dict) else [],
                    "temperature_c": step_audit.get("temperature_c"),
                    "condition_score": step_audit.get("condition_score"),
                    "enzyme_confidence": step_audit.get("enzyme_confidence"),
                    "solvent": step_audit.get("solvent"),
                    "reagent": step_audit.get("reagent"),
                    "catalyst": step_audit.get("catalyst"),
                    "ec": step.get("ec") or condition.get("ec") if isinstance(condition, dict) else step.get("ec"),
                }
                step_records.append(step_record)
                _add_examples(examples, "step_risk:" + risk, step_record, examples_per_bucket)
                for issue in issues:
                    _add_examples(examples, "step_issue:" + issue, step_record, examples_per_bucket)

        target_rows.append(
            {
                "target": target,
                "route_count": target_route_count,
                "condition_warn_routes": target_warn_route_count,
                "condition_high_routes": target_high_route_count,
                "step_count": target_step_count,
                "predicted_condition_steps": target_predicted_step_count,
                "condition_coverage": _rate(target_predicted_step_count, target_step_count),
                "route_risk_counts": dict(sorted(route_risk_by_target[target].items())),
                "route_issue_counts": dict(sorted(route_issue_by_target[target].items())),
                "step_issue_counts": dict(sorted(target_step_issue_counts.items())),
            }
        )

    total_steps = len(step_records)
    predicted_steps = sum(1 for row in step_records if row.get("condition_source") != "missing")
    report = {
        "schema_version": "condition_prediction_attribution.v1",
        "inputs": {"rows": str(rows_path)},
        "summary": {
            "targets": len(rows),
            "routes": len(route_records),
            "steps": total_steps,
            "routes_with_condition_warning": route_risks.get("warn", 0),
            "routes_with_condition_high_risk": route_risks.get("high", 0),
            "predicted_condition_steps": predicted_steps,
            "condition_step_coverage": _rate(predicted_steps, total_steps),
            "strict_enzyme_routes": sum(1 for row in route_records if row.get("strict_enzyme_route")),
            "strict_enzyme_route_risks": dict(sorted(enzyme_route_risks.items())),
            "strict_enzyme_route_issues": dict(sorted(enzyme_route_issues.items())),
        },
        "route_level": {
            "risk_counts": dict(sorted(route_risks.items())),
            "route_issue_counts": dict(sorted(route_issues.items())),
            "route_class_counts": dict(sorted(route_classes.items())),
            "warning_cause_counts": dict(sorted(route_warning_causes.items())),
        },
        "step_level": {
            "risk_counts": dict(sorted(step_risks.items())),
            "issue_counts": dict(sorted(step_issues.items())),
            "domain_counts": dict(sorted(step_domains.items())),
            "source_counts": dict(sorted(step_sources.items())),
            "reaction_type_counts": dict(sorted(reaction_types.items())),
            "condition_source_counts": dict(sorted(condition_sources.items())),
            "condition_source_by_domain": _nested_counter_dict(condition_sources_by_domain),
            "issue_counts_by_source": _nested_counter_dict(step_source_issue_counts),
            "issue_counts_by_domain": _nested_counter_dict(step_domain_issue_counts),
        },
        "enzyme_step_level": {
            "risk_counts": dict(sorted(enzyme_step_risks.items())),
            "issue_counts": dict(sorted(enzyme_step_issues.items())),
        },
        "targets": target_rows,
        "examples": dict(sorted(examples.items())),
        "records": {
            "routes": route_records,
            "steps": step_records,
        },
        "interpretation": [],
    }
    report["interpretation"] = _interpret(report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    route_level = report.get("route_level") or {}
    step_level = report.get("step_level") or {}
    enzyme_level = report.get("enzyme_step_level") or {}
    lines = [
        "# Condition Prediction Attribution",
        "",
        "## Summary",
        "",
        f"- Targets: `{summary.get('targets')}`",
        f"- Routes: `{summary.get('routes')}`",
        f"- Steps: `{summary.get('steps')}`",
        f"- Routes with condition warning: `{summary.get('routes_with_condition_warning')}`",
        f"- Routes with high-risk conditions: `{summary.get('routes_with_condition_high_risk')}`",
        f"- Predicted condition steps: `{summary.get('predicted_condition_steps')}`",
        f"- Condition step coverage: `{summary.get('condition_step_coverage')}`",
        f"- Strict enzyme routes: `{summary.get('strict_enzyme_routes')}`",
        f"- Strict enzyme route risks: `{json.dumps(summary.get('strict_enzyme_route_risks'), sort_keys=True)}`",
        "",
        "## Route-Level Attribution",
        "",
        f"- Risk counts: `{json.dumps(route_level.get('risk_counts'), sort_keys=True)}`",
        f"- Route issues: `{json.dumps(route_level.get('route_issue_counts'), sort_keys=True)}`",
        f"- Warning causes: `{json.dumps(route_level.get('warning_cause_counts'), sort_keys=True)}`",
        "",
        "## Step-Level Attribution",
        "",
        f"- Risk counts: `{json.dumps(step_level.get('risk_counts'), sort_keys=True)}`",
        f"- Issue counts: `{json.dumps(step_level.get('issue_counts'), sort_keys=True)}`",
        f"- Domain counts: `{json.dumps(step_level.get('domain_counts'), sort_keys=True)}`",
        f"- Condition sources: `{json.dumps(step_level.get('condition_source_counts'), sort_keys=True)}`",
        f"- Enzyme step risks: `{json.dumps(enzyme_level.get('risk_counts'), sort_keys=True)}`",
        f"- Enzyme step issues: `{json.dumps(enzyme_level.get('issue_counts'), sort_keys=True)}`",
        "",
        "## Targets",
        "",
        "| target | routes | warn | high | steps | condition coverage | top step issues |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report.get("targets") or []:
        top_issues = _top_counter_items(row.get("step_issue_counts") or {}, limit=4)
        lines.append(
            "| `{target}` | {routes} | {warn} | {high} | {steps} | {coverage} | `{issues}` |".format(
                target=row.get("target"),
                routes=row.get("route_count"),
                warn=row.get("condition_warn_routes"),
                high=row.get("condition_high_routes"),
                steps=row.get("step_count"),
                coverage=row.get("condition_coverage"),
                issues=json.dumps(top_issues, sort_keys=True),
            )
        )
    lines.extend(["", "## Main Findings", ""])
    for item in report.get("interpretation") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Example Buckets", ""])
    for bucket, rows in (report.get("examples") or {}).items():
        lines.append(f"### `{bucket}`")
        for row in rows[:5]:
            lines.append(
                "- "
                + json.dumps(
                    {
                        key: row.get(key)
                        for key in (
                            "target",
                            "route_rank",
                            "step_index",
                            "route_class",
                            "step_source",
                            "domain",
                            "risk",
                            "issues",
                            "condition_source",
                            "temperature_c",
                            "condition_score",
                        )
                        if key in row
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        lines.append("")
    return "\n".join(lines)


def _interpret(report: dict[str, Any] | None) -> list[str]:
    if not report:
        return []
    step_issues = ((report.get("step_level") or {}).get("issue_counts") or {})
    route_causes = ((report.get("route_level") or {}).get("warning_cause_counts") or {})
    condition_sources = ((report.get("step_level") or {}).get("condition_source_counts") or {})
    findings: list[str] = []
    if step_issues.get("missing_condition_prediction"):
        findings.append(
            "Missing condition predictions are a primary cause; fill coverage before retraining accuracy-heavy models."
        )
    if step_issues.get("low_condition_score"):
        findings.append(
            "Low top-1 condition scores are present; condition confidence should be calibrated and used for reranking."
        )
    if route_causes.get("stepwise_required"):
        findings.append(
            "Many route warnings are stepwise-compatibility warnings, not necessarily invalid chemistry."
        )
    if condition_sources.get("brenda_condition_prior"):
        findings.append(
            "BRENDA priors are active for some enzyme steps; the next pass should compare enzyme-step warnings with BRENDA-covered steps."
        )
    if not findings:
        findings.append("No dominant condition-warning cause was identified by the current audit rules.")
    return findings


def _route_warning_causes(condition_audit: dict[str, Any]) -> list[str]:
    causes = set(str(issue) for issue in condition_audit.get("route_issues") or [])
    issue_counts = condition_audit.get("issue_counts") or {}
    for issue, count in issue_counts.items():
        if int(count or 0) > 0:
            causes.add(str(issue))
    if condition_audit.get("stepwise_required"):
        causes.add("stepwise_required")
    if condition_audit.get("low_score_step_count"):
        causes.add("low_condition_score")
    return sorted(causes)


def _audit_target_payload(idx: int, row: dict[str, Any], name_by_smiles: dict[str, str]) -> dict[str, Any]:
    target_smiles = str(row.get("target_smiles") or row.get("target_canonical") or "")
    return {
        "index": idx,
        "cascade_id": name_by_smiles.get(target_smiles) or f"target_{idx + 1}",
        "target_id": name_by_smiles.get(target_smiles) or f"target_{idx + 1}",
        "target_smiles": target_smiles,
        "metrics": {"strict_stock_solve_any": bool(row.get("solved_routes"))},
        "planner_output": {"routes": [_normalize_route(route) for route in row.get("routes") or []]},
    }


def _normalize_route(route: dict[str, Any]) -> dict[str, Any]:
    out = dict(route)
    metrics = dict(out.get("metrics") or {})
    metrics.setdefault("strict_stock_solve", bool(route.get("route_solved") or metrics.get("strict_stock_solve")))
    metrics.setdefault(
        "route_solved",
        bool(route.get("route_solved") or metrics.get("route_solved") or metrics.get("strict_stock_solve")),
    )
    metrics.setdefault("filled_route", bool(route.get("steps") or metrics.get("filled_route")))
    out["metrics"] = metrics
    return out


def _top_condition(step: dict[str, Any]) -> dict[str, Any]:
    for row in step.get("condition_predictions") or []:
        if isinstance(row, dict):
            return row
    conditions = step.get("step_conditions") or step.get("conditions") or {}
    return conditions if isinstance(conditions, dict) else {}


def _condition_source(condition: dict[str, Any]) -> str:
    if not condition:
        return "missing"
    for key in ("source", "model", "condition_model", "condition_label"):
        value = condition.get(key)
        if value not in (None, "", [], {}):
            return str(value)
    return "unknown"


def _is_strict_enzyme_route(route: dict[str, Any]) -> bool:
    return any(bool(route.get(key)) for key in ENZYME_ROUTE_FLAG_KEYS)


def _is_enzyme_step(step: dict[str, Any]) -> bool:
    source = str(step.get("source") or "")
    return bool(
        step.get("is_enzymatic")
        or step.get("ec")
        or step.get("enzyme_uid")
        or source in {"enzyme_precedent", "enzexpand", "enzyformer"}
    )


def _is_solved_route(route: dict[str, Any]) -> bool:
    metrics = route.get("metrics") or {}
    return bool(route.get("route_solved") or metrics.get("route_solved") or metrics.get("strict_stock_solve"))


def _target_name(row: dict[str, Any], name_by_smiles: dict[str, str], idx: int) -> str:
    target_smiles = str(row.get("target_smiles") or row.get("target_canonical") or "")
    return name_by_smiles.get(target_smiles) or f"target_{idx}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def _nested_counter_dict(value: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {key: dict(sorted(counter.items())) for key, counter in sorted(value.items())}


def _top_counter_items(counter: dict[str, int], *, limit: int) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-int(item[1]), item[0]))[:limit])


def _add_examples(
    examples: dict[str, list[dict[str, Any]]],
    bucket: str,
    row: dict[str, Any],
    limit: int,
) -> None:
    if limit <= 0:
        return
    if len(examples[bucket]) < limit:
        examples[bucket].append(row)


if __name__ == "__main__":
    main()
