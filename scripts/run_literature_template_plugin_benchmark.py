#!/usr/bin/env python3
"""Offline A/B benchmark for executable literature-template one-step plugin."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade_planner.agent.chem_enzy_policy import apply_literature_template_plugin_policy
from cascade_planner.agent.literature_templates import (
    LiteratureTemplateCard,
    audit_native_run_for_literature,
    default_literature_template_cards,
)
from cascade_planner.agent.template_applicability import assess_template_applicability
from cascade_planner.baselines.literature_one_step_plugin import LiteratureOneStepPlugin
from cascade_planner.baselines.route_contract import RouteSearchConfig


BENCHMARK_SCHEMA = "literature_template_plugin_benchmark.v1"


@dataclass(frozen=True)
class LiteratureTemplateBenchmarkCase:
    case_id: str
    target_name: str
    target_smiles: str
    expected_template_id: str
    native_status: str
    negative_control: bool = False


def default_benchmark_cases() -> list[LiteratureTemplateBenchmarkCase]:
    return [
        LiteratureTemplateBenchmarkCase(
            case_id="bufotalin_bufadienolide_c17_pyrone",
            target_name="Bufotalin-like C17 pyrone",
            target_smiles="CC(C)(C)[Si](C)(C)O[C@H]1CC[C@@]2(C)[C@H](CC[C@@H]3[C@@H]2CC[C@]2(C)[C@@H](c4ccc(=O)oc4)[C@@H](O)C[C@]32O)C1",
            expected_template_id="lit_tpl_bufadienolide_c17_pyrone_split_v1",
            native_status="failed",
        ),
        LiteratureTemplateBenchmarkCase(
            case_id="taxane_semisynthesis_c13_side_chain",
            target_name="Taxane C13 side-chain",
            target_smiles="CC(=O)OC1CC(O)C2(C)C(OC(=O)c3ccccc3)C3OC3C(O)C12",
            expected_template_id="lit_tpl_taxane_c13_side_chain_split_v1",
            native_status="unclosed",
        ),
        LiteratureTemplateBenchmarkCase(
            case_id="artemisinin_peroxide_anchor",
            target_name="Artemisinin peroxide anchor",
            target_smiles="CC(C)C1OC2OOCC1CC2=O",
            expected_template_id="lit_tpl_artemisinin_peroxide_anchor_v1",
            native_status="failed",
        ),
        LiteratureTemplateBenchmarkCase(
            case_id="macrolactonization",
            target_name="Macrocyclic lactone",
            target_smiles="O=C1CCCCCCCCCCCCO1",
            expected_template_id="lit_tpl_macrolactone_split_v1",
            native_status="unclosed",
        ),
        LiteratureTemplateBenchmarkCase(
            case_id="corey_lactone_prostaglandin",
            target_name="Corey lactone side-chain",
            target_smiles="O=C1OCC2CCC1C2CC=O",
            expected_template_id="lit_tpl_corey_lactone_side_chain_split_v1",
            native_status="failed",
        ),
        LiteratureTemplateBenchmarkCase(
            case_id="phenolic_glycoside_native_solved_negative_control",
            target_name="Phenolic O-glycoside native solved control",
            target_smiles="Oc1ccccc1OC1COC(O)C(O)C1O",
            expected_template_id="lit_tpl_o_glycoside_split_v1",
            native_status="solved",
            negative_control=True,
        ),
    ]


def run_literature_template_plugin_benchmark(
    *,
    cases: list[LiteratureTemplateBenchmarkCase] | None = None,
    template_cards: list[LiteratureTemplateCard] | None = None,
) -> dict[str, Any]:
    cases = cases or default_benchmark_cases()
    templates = template_cards or default_literature_template_cards()
    plugin = LiteratureOneStepPlugin(template_cards=templates)
    rows = []
    for case in cases:
        native_result = _native_result_for_case(case)
        route_audit = _route_audit_for_case(case)
        trigger_report = audit_native_run_for_literature(native_result, route_audit=route_audit)
        native_config = RouteSearchConfig(target_smiles=case.target_smiles)
        policy_config = apply_literature_template_plugin_policy(native_config, trigger_report={"should_trigger": False})
        plugin_config = apply_literature_template_plugin_policy(native_config, trigger_report=trigger_report)
        plugin_rows = plugin.predict(case.target_smiles, top_k=8) if trigger_report["should_trigger"] else []
        applicability = [
            assess_template_applicability(
                target_smiles=case.target_smiles,
                frontier_smiles=case.target_smiles,
                template_card=card,
            ).to_dict()
            for card in templates
        ]
        row = {
            "case_id": case.case_id,
            "target_name": case.target_name,
            "target_smiles": case.target_smiles,
            "negative_control": case.negative_control,
            "expected_template_id": case.expected_template_id,
            "native": _config_metrics(native_result, native_config),
            "policy_only": _config_metrics(native_result, policy_config),
            "plugin": _plugin_metrics(native_result, plugin_config, plugin_rows, case),
            "trigger_report": trigger_report,
            "applicability": applicability,
        }
        rows.append(row)
    summary = summarize_benchmark_rows(rows)
    return {
        "schema_version": BENCHMARK_SCHEMA,
        "case_count": len(rows),
        "configs": ["native", "policy_only", "plugin"],
        "metrics": {
            "solved_rate": "Native-style solved claim rate; plugin candidates do not directly decide solved.",
            "route_count": "Per-case route/proposal count by configuration.",
            "stock_closure_rate": "Strict stock closure proof rate from native/audit inputs.",
            "fake_closure_rejection_rate": "Rate of fake closure audit rejection.",
            "route_audit_pass_rate": "Strict route audit pass rate.",
            "literature_plugin_step_precision": "Expected-template validated plugin rows / plugin rows.",
            "reconstruction_pass_rate": "Template validation reconstruction pass rate for plugin rows.",
        },
        "summary": summary,
        "cases": rows,
    }


def summarize_benchmark_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    plugin_rows = [row for row in rows if row["plugin"]["route_count"] > 0]
    improved = [
        row["case_id"]
        for row in rows
        if row["plugin"]["improved_over_native"] and not row.get("negative_control")
    ]
    negative_controls_ok = all(
        not row["plugin"]["improved_over_native"]
        and row["plugin"]["route_count"] == 0
        and not row["trigger_report"]["should_trigger"]
        for row in rows
        if row.get("negative_control")
    )
    return {
        "n_cases": n,
        "native_solved_rate": _rate(row["native"]["solved"] for row in rows),
        "policy_only_solved_rate": _rate(row["policy_only"]["solved"] for row in rows),
        "plugin_solved_rate": _rate(row["plugin"]["solved"] for row in rows),
        "native_total_route_count": sum(int(row["native"]["route_count"]) for row in rows),
        "plugin_total_route_count": sum(int(row["plugin"]["route_count"]) for row in rows),
        "stock_closure_rate": _rate(row["plugin"]["stock_closure_passed"] for row in rows),
        "fake_closure_rejection_rate": _rate(row["plugin"]["fake_closure_rejected"] for row in rows),
        "route_audit_pass_rate": _rate(row["plugin"]["route_audit_passed"] for row in rows),
        "literature_plugin_step_precision": _mean(row["plugin"]["literature_plugin_step_precision"] for row in plugin_rows),
        "reconstruction_pass_rate": _mean(row["plugin"]["reconstruction_pass_rate"] for row in plugin_rows),
        "improved_case_ids": improved,
        "improved_native_failed_or_unclosed_count": len(improved),
        "negative_controls_ok": negative_controls_ok,
        "all_plugin_steps_have_evidence_and_validation": all(
            row["plugin"]["all_steps_have_evidence_refs"]
            and row["plugin"]["all_steps_have_validation_report"]
            for row in plugin_rows
        ),
    }


def _native_result_for_case(case: LiteratureTemplateBenchmarkCase) -> dict[str, Any]:
    if case.native_status == "solved":
        return {
            "target_smiles": case.target_smiles,
            "backend": "offline_native",
            "solved": True,
            "route_count": 1,
            "routes": [
                {
                    "target_smiles": case.target_smiles,
                    "solved": True,
                    "stock_status": {"native_stock_leaf": True},
                    "steps": [{"stock_status": {"native_stock_leaf": True}}],
                }
            ],
            "failures": [],
        }
    if case.native_status == "unclosed":
        return {
            "target_smiles": case.target_smiles,
            "backend": "offline_native",
            "solved": False,
            "route_count": 1,
            "routes": [
                {
                    "target_smiles": case.target_smiles,
                    "solved": False,
                    "stock_status": {"advanced_anchor": False},
                    "steps": [{"stock_status": {"advanced_anchor": False}}],
                }
            ],
            "failures": [],
        }
    return {
        "target_smiles": case.target_smiles,
        "backend": "offline_native",
        "solved": False,
        "route_count": 0,
        "routes": [],
        "failures": [{"category": "no_route_found", "message": "offline native miss"}],
    }


def _route_audit_for_case(case: LiteratureTemplateBenchmarkCase) -> dict[str, Any]:
    if case.native_status == "solved":
        return {
            "schema_version": "route_audit_report.v1",
            "route_status": "solved",
            "stock_audit_passed": True,
            "fake_closure_rejected": False,
            "reasons": [],
        }
    if case.native_status == "unclosed":
        return {
            "schema_version": "route_audit_report.v1",
            "route_status": "unresolved",
            "stock_audit_passed": False,
            "fake_closure_rejected": False,
            "reasons": ["unclosed_route"],
        }
    return {
        "schema_version": "route_audit_report.v1",
        "route_status": "unresolved",
        "stock_audit_passed": False,
        "fake_closure_rejected": False,
        "reasons": ["native_failed"],
    }


def _config_metrics(native_result: dict[str, Any], config: RouteSearchConfig) -> dict[str, Any]:
    return {
        "solved": bool(native_result.get("solved")),
        "route_count": int(native_result.get("route_count") or 0),
        "search_flags": dict(config.search_flags or {}),
        "stock_closure_passed": _stock_closed(native_result),
        "route_audit_passed": bool(native_result.get("solved") and _stock_closed(native_result)),
        "fake_closure_rejected": False,
    }


def _plugin_metrics(
    native_result: dict[str, Any],
    config: RouteSearchConfig,
    plugin_rows: list[dict[str, Any]],
    case: LiteratureTemplateBenchmarkCase,
) -> dict[str, Any]:
    route_count = len(plugin_rows)
    expected_hits = [
        row for row in plugin_rows
        if str(row.get("template") or "") == case.expected_template_id
        and (row.get("template_validation_report") or {}).get("allowed_for_one_step_source")
    ]
    reconstruction_passed = [
        row for row in plugin_rows
        if ((row.get("template_validation_report") or {}).get("reconstruction_report") or {}).get("passed")
    ]
    return {
        "solved": bool(native_result.get("solved")),
        "route_count": route_count,
        "search_flags": dict(config.search_flags or {}),
        "stock_closure_passed": _stock_closed(native_result),
        "route_audit_passed": bool(native_result.get("solved") and _stock_closed(native_result)),
        "fake_closure_rejected": False,
        "literature_plugin_step_precision": len(expected_hits) / route_count if route_count else None,
        "reconstruction_pass_rate": len(reconstruction_passed) / route_count if route_count else None,
        "improved_over_native": route_count > 0 and int(native_result.get("route_count") or 0) == 0
        or (route_count > 0 and not native_result.get("solved")),
        "all_steps_have_evidence_refs": all(bool(row.get("evidence_refs")) for row in plugin_rows),
        "all_steps_have_validation_report": all(bool(row.get("template_validation_report")) for row in plugin_rows),
        "plugin_rows": plugin_rows,
    }


def _stock_closed(result: dict[str, Any]) -> bool:
    routes = result.get("routes") or []
    if not routes:
        return False
    for route in routes:
        for value in (route.get("stock_status") or {}).values():
            if value is not True:
                return False
        for step in route.get("steps") or []:
            for value in (step.get("stock_status") or {}).values():
                if value is not True:
                    return False
    return True


def _rate(values: Any) -> float:
    vals = [bool(value) for value in values]
    return sum(1 for value in vals if value) / max(len(vals), 1)


def _mean(values: Any) -> float | None:
    vals = [float(value) for value in values if value is not None]
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="", help="Write JSON report to this path.")
    args = parser.parse_args()
    payload = run_literature_template_plugin_benchmark()
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
