"""Compile the P9 blind-retrosynthesis release gate from immutable run artifacts.

The evaluator never repairs a route and never upgrades scientific authority.  It
only reads completed target-only reports plus their digest-bound workbenches,
retains every failed case, and translates the published P9 thresholds into a
single reproducible pass/fail record.
"""

from __future__ import annotations

from collections import Counter
import hashlib
from html import escape
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


RELEASE_GATE_SCHEMA = "v4_blind_release_gate.v1"
REQUIRED_ABLATIONS = ("no-chemenzy", "no-self-evo", "no-replan")


def compile_v4_blind_release_gate(
    baseline_status_path: str | Path,
    *,
    ablation_status_paths: Iterable[str | Path] = (),
    repository_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    baseline_path = Path(baseline_status_path).expanduser().resolve()
    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root
        else Path(__file__).resolve().parents[2]
    )
    baseline, baseline_rows = _panel_rows(baseline_path, repository_root=root)
    target_count = len(baseline_rows)
    diversity = _diversity(baseline_rows)
    ablations: dict[str, dict[str, Any]] = {}
    for raw_path in ablation_status_paths:
        path = Path(raw_path).expanduser().resolve()
        status, rows = _panel_rows(path, repository_root=root)
        arm = str(status.get("ablation") or "")
        ablations[arm or path.parent.name] = _ablation_summary(
            status, rows, baseline_rows=baseline_rows
        )

    costs = [dict(row.get("cost") or {}) for row in baseline_rows]
    chem_times = [float(row.get("chemenzy_time_s") or 0.0) for row in baseline_rows]
    required_validated = max(6, math.ceil(target_count * 0.75))
    required_procurement = max(4, math.ceil(target_count * 0.50))
    required_conditions = max(3, math.ceil(target_count * 0.375))
    validated_cases = sum(row.get("two_distinct_validated_routes") is True for row in baseline_rows)
    procurement_cases = sum(
        int(row.get("procurement_closed_routes") or 0) > 0 for row in baseline_rows
    )
    condition_cases = sum(
        int(row.get("condition_complete_routes") or 0) > 0 for row in baseline_rows
    )
    extended_preflight_cases = sum(
        row.get("extended_preflight_complete") is True for row in baseline_rows
    )
    repo_preflight_cases = sum(
        row.get("repository_preflight_complete") is True for row in baseline_rows
    )
    auditable_cases = sum(row.get("auditable_outcome") is True for row in baseline_rows)
    within_budget_cases = sum(row.get("within_resource_budget") is True for row in baseline_rows)
    benchmark_procurement_violations = sum(
        int(row.get("benchmark_procurement_false_claim_count") or 0) for row in baseline_rows
    )
    predicted_exact_violations = sum(
        int(row.get("predicted_condition_exact_claim_count") or 0) for row in baseline_rows
    )
    false_closure_claims = sum(
        int(row.get("false_closure_claim_count") or 0) for row in baseline_rows
    )
    frozen = dict(baseline.get("frozen_snapshot") or {})
    snapshot_bound_cases = sum(row.get("snapshot_receipt_valid") is True for row in baseline_rows)
    ablation_complete = _ablation_gate(baseline, baseline_rows=baseline_rows, ablations=ablations)
    cost_summary = {
        "median_model_invocations": _median(costs, "model_invocations"),
        "median_input_tokens": _median(costs, "input_tokens"),
        "median_output_tokens": _median(costs, "output_tokens"),
        "median_model_wall_time_s": _median(costs, "wall_time_s"),
        "total_model_invocations": sum(int(row.get("model_invocations") or 0) for row in costs),
        "total_input_tokens": sum(int(row.get("input_tokens") or 0) for row in costs),
        "total_output_tokens": sum(int(row.get("output_tokens") or 0) for row in costs),
        "total_model_wall_time_s": round(
            sum(float(row.get("wall_time_s") or 0.0) for row in costs), 3
        ),
        "median_chemenzy_time_s": round(float(median(chem_times)), 3) if chem_times else 0.0,
        "total_chemenzy_time_s": round(sum(chem_times), 3),
    }
    gates = {
        "target_panel_and_structural_diversity": _gate(
            target_count >= 8 and diversity["distinct_scaffold_count"] >= 6,
            actual={"target_count": target_count, **diversity},
            required={"minimum_target_count": 8, "minimum_distinct_scaffold_count": 6},
        ),
        "repository_and_extended_leakage_preflight": _gate(
            repo_preflight_cases == target_count and extended_preflight_cases == target_count,
            actual={
                "repository_preflight_case_count": repo_preflight_cases,
                "extended_synonym_and_intermediate_case_count": extended_preflight_cases,
            },
            required={"case_count": target_count},
        ),
        "frozen_provider_template_inventory_snapshot": _gate(
            bool(frozen.get("content_sha256"))
            and bool(frozen.get("base_environment_sha256"))
            and bool(frozen.get("self_evo_library_sha256"))
            and bool(frozen.get("inventory_snapshot_sha256"))
            and baseline.get("_frozen_snapshot_valid") is True
            and snapshot_bound_cases == target_count,
            actual={
                "panel_snapshot_present_and_digest_valid": baseline.get("_frozen_snapshot_valid")
                is True,
                "template_snapshot_present": bool(frozen.get("self_evo_library_sha256")),
                "inventory_snapshot_present": bool(frozen.get("inventory_snapshot_sha256")),
                "case_snapshot_receipt_count": snapshot_bound_cases,
            },
            required={
                "provider_snapshot_present": True,
                "template_snapshot_present": True,
                "inventory_snapshot_present": True,
                "case_snapshot_receipt_count": target_count,
            },
        ),
        "auditable_success_or_explicit_unresolved_within_budget": _gate(
            auditable_cases == target_count and within_budget_cases == target_count,
            actual={
                "auditable_case_count": auditable_cases,
                "within_budget_case_count": within_budget_cases,
            },
            required={"case_count": target_count},
        ),
        "no_false_closure_or_authority_laundering": _gate(
            false_closure_claims == 0
            and benchmark_procurement_violations == 0
            and predicted_exact_violations == 0,
            actual={
                "false_closure_claim_count": false_closure_claims,
                "benchmark_leaf_claimed_procurement_count": benchmark_procurement_violations,
                "predicted_condition_claimed_source_exact_count": predicted_exact_violations,
            },
            required={"all_counts": 0},
        ),
        "two_strategically_distinct_reaction_validated_routes": _gate(
            validated_cases >= required_validated,
            actual={"qualifying_case_count": validated_cases},
            required={
                "qualifying_case_count": required_validated,
                "rate_scaled_from": "6_of_8",
            },
        ),
        "real_procurement_closed_route": _gate(
            procurement_cases >= required_procurement,
            actual={"qualifying_case_count": procurement_cases},
            required={
                "qualifying_case_count": required_procurement,
                "rate_scaled_from": "4_of_8",
            },
        ),
        "condition_complete_route": _gate(
            condition_cases >= required_conditions,
            actual={"qualifying_case_count": condition_cases},
            required={
                "qualifying_case_count": required_conditions,
                "rate_scaled_from": "3_of_8",
            },
        ),
        "codex_cost_envelope": _gate(
            cost_summary["median_model_invocations"] <= 2
            and cost_summary["median_input_tokens"] <= 60_000
            and cost_summary["median_output_tokens"] <= 15_000,
            actual={
                key: cost_summary[key]
                for key in (
                    "median_model_invocations",
                    "median_input_tokens",
                    "median_output_tokens",
                )
            },
            required={
                "maximum_median_model_invocations": 2,
                "maximum_median_input_tokens": 60_000,
                "maximum_median_output_tokens": 15_000,
            },
        ),
        "chemenzy_cost_and_independent_gain": _gate(
            cost_summary["median_chemenzy_time_s"] <= 120
            and ablation_complete["independent_chemenzy_gain_reported"],
            actual={
                "median_chemenzy_time_s": cost_summary["median_chemenzy_time_s"],
                "independent_gain_reported": ablation_complete[
                    "independent_chemenzy_gain_reported"
                ],
            },
            required={"maximum_median_chemenzy_time_s": 120, "independent_gain_reported": True},
        ),
        "three_required_ablations": _gate(
            ablation_complete["passed"],
            actual=ablation_complete,
            required={"arms": list(REQUIRED_ABLATIONS)},
        ),
        "three_profile_readout": _gate(
            all(
                dict(row.get("profiles") or {}).keys()
                >= {"fast_explore", "validated_plan", "process_dossier"}
                for row in baseline_rows
            ),
            actual=_profile_totals(baseline_rows),
            required={"profiles": ["fast_explore", "validated_plan", "process_dossier"]},
        ),
    }
    failure_counts = Counter(
        reason for row in baseline_rows for reason in row.get("failure_reasons") or []
    )
    report = {
        "schema_version": RELEASE_GATE_SCHEMA,
        "baseline_panel_status": str(baseline_path),
        "target_count": target_count,
        "accepted": all(row["passed"] for row in gates.values()),
        "gates": gates,
        "coverage": {
            "profiles": _profile_totals(baseline_rows),
            "two_distinct_validated_case_count": validated_cases,
            "procurement_closed_case_count": procurement_cases,
            "condition_complete_case_count": condition_cases,
            "low_confidence_or_open_route_count": sum(
                int(row.get("open_visible_route_count") or 0) for row in baseline_rows
            ),
        },
        "cost": cost_summary,
        "diversity": diversity,
        "ablations": ablations,
        "failure_classification_counts": dict(sorted(failure_counts.items())),
        "cases": baseline_rows,
        "semantics": {
            "failed_cases_are_retained": True,
            "benchmark_stock_never_counts_as_procurement": True,
            "predicted_conditions_never_count_as_source_exact": True,
            "route_length_is_descriptive_not_an_objective": True,
            "release_gate_grants_no_route_or_reaction_authority": True,
            "thresholds_scale_from_the_published_eight_case_gate": True,
        },
    }
    report["content_sha256"] = _json_sha256(report)
    if output_dir is not None:
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        _write_json(destination / "summary.json", report)
        (destination / "index.html").write_text(
            render_v4_blind_release_gate_html(report), encoding="utf-8"
        )
    return report


def render_v4_blind_release_gate_html(report: Mapping[str, Any]) -> str:
    gates = dict(report.get("gates") or {})
    gate_rows = "".join(
        "<tr><td>{}</td><td class='{}'>{}</td><td><code>{}</code></td><td><code>{}</code></td></tr>".format(
            escape(name.replace("_", " ")),
            "pass" if dict(value).get("passed") else "fail",
            "通过" if dict(value).get("passed") else "未通过",
            escape(json.dumps(dict(value).get("actual"), ensure_ascii=False)),
            escape(json.dumps(dict(value).get("required"), ensure_ascii=False)),
        )
        for name, value in gates.items()
    )
    case_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            escape(str(row.get("case_id") or row.get("target_name") or "")),
            "是" if row.get("auditable_outcome") else "否",
            int(row.get("reaction_validated_routes") or 0),
            int(row.get("procurement_closed_routes") or 0),
            int(row.get("condition_complete_routes") or 0),
            escape(str(row.get("claim") or "unresolved")),
        )
        for row in report.get("cases") or []
    )
    accepted = report.get("accepted") is True
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>V4 fresh blind 发布门</title><style>
:root{{--ink:#14213d;--muted:#667085;--line:#dbe3f0;--good:#087443;--bad:#b42318;--bg:#f5f7fb}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 Inter,'Microsoft YaHei',sans-serif}}main{{max-width:1320px;margin:auto;padding:30px 24px 56px}}h1{{margin:0 0 6px}}.lead{{color:var(--muted)}}.banner{{margin:20px 0;padding:16px 18px;border-radius:14px;background:{"#ecfdf3" if accepted else "#fff1f0"};border:1px solid {"#abefc6" if accepted else "#fecdca"};font-weight:750}}section{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:18px;margin-top:16px;overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:850px}}th,td{{padding:10px;border-bottom:1px solid #edf0f5;text-align:left;vertical-align:top}}th{{font-size:12px;color:var(--muted)}}code{{white-space:pre-wrap;font-size:11px}}.pass{{color:var(--good);font-weight:750}}.fail{{color:var(--bad);font-weight:750}}</style></head><body><main>
<h1>V4 fresh blind 发布门</h1><p class='lead'>目标输入、结构路线、反应验证、条件、真实采购、成本与消融分轴验收；未通过项不会被隐藏。</p>
<div class='banner'>{"发布门通过" if accepted else "发布门尚未通过"} · {int(report.get("target_count") or 0)} targets</div>
<section><h2>门禁</h2><table><thead><tr><th>门</th><th>状态</th><th>实测</th><th>要求</th></tr></thead><tbody>{gate_rows}</tbody></table></section>
<section><h2>逐目标</h2><table><thead><tr><th>Case</th><th>可审计终态</th><th>反应验证路线</th><th>真实采购闭合</th><th>条件完整</th><th>当前档位</th></tr></thead><tbody>{case_rows}</tbody></table></section>
</main></body></html>"""


def _panel_rows(
    status_path: Path, *, repository_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    status = _read_json(status_path)
    if status.get("schema_version") != "v4_blind_panel_status.v1":
        raise ValueError("v4_blind_panel_status_schema_invalid")
    status["_frozen_snapshot_valid"] = _frozen_snapshot_valid(status, status_path=status_path)
    panel_root = status_path.parent
    rows = [
        _case_row(
            target_name=str(name),
            status_row=dict(raw) if isinstance(raw, Mapping) else {},
            panel_root=panel_root,
            repository_root=repository_root,
            frozen_snapshot=dict(status.get("frozen_snapshot") or {}),
            frozen_snapshot_valid=status["_frozen_snapshot_valid"] is True,
        )
        for name, raw in dict(status.get("targets") or {}).items()
    ]
    return status, rows


def _case_row(
    *,
    target_name: str,
    status_row: Mapping[str, Any],
    panel_root: Path,
    repository_root: Path,
    frozen_snapshot: Mapping[str, Any],
    frozen_snapshot_valid: bool,
) -> dict[str, Any]:
    report_path = Path(str(status_row.get("report_path") or ""))
    if not report_path.is_file():
        report_path = panel_root / "runs" / target_name / "target-only-solve-report.json"
    if not report_path.is_file():
        return {
            "case_id": str(status_row.get("case_id") or target_name),
            "target_name": target_name,
            "status": str(status_row.get("status") or "missing"),
            "auditable_outcome": False,
            "within_resource_budget": False,
            "failure_reasons": ["target_report_missing"],
        }
    report = _read_json(report_path)
    report_digest_valid = _json_digest_valid(report)
    preflight = dict(report.get("preflight") or {})
    workbench, artifact_error = _load_workbench(report, panel_root=panel_root)
    routes = [dict(value) for value in dict(workbench.get("routes") or {}).values()]
    edges = dict(workbench.get("edges") or {})
    inspectors = dict(dict(workbench.get("inspectors") or {}).get("edges") or {})
    validated = [
        route
        for route in routes
        if dict(route.get("acceptance_profiles") or {}).get("reaction_validated") is True
    ]
    validated_families = {
        str(route.get("route_family_id") or route.get("strategy") or route.get("route_id") or "")
        for route in validated
    }
    validated_edge_sets = {
        tuple(sorted(str(edge_id) for edge_id in route.get("edge_ids") or []))
        for route in validated
        if route.get("edge_ids")
    }
    procurement = [
        route
        for route in routes
        if dict(route.get("acceptance_profiles") or {}).get("procurement_closed") is True
    ]
    conditions = [
        route
        for route in routes
        if dict(route.get("acceptance_profiles") or {}).get("condition_complete") is True
    ]
    process = [
        route
        for route in routes
        if dict(route.get("acceptance_profiles") or {}).get("process_ready") is True
    ]
    exploration = [
        route
        for route in routes
        if dict(route.get("acceptance_profiles") or {}).get("exploration_closed") is True
    ]
    benchmark_procurement = sum(
        str(dict(route.get("proof_vector") or {}).get("stock") or "") == "benchmark_hit"
        for route in procurement
    )
    predicted_exact = 0
    for edge_id, raw_edge in edges.items():
        edge = dict(raw_edge)
        vector = dict(edge.get("proof_vector") or {})
        inspector = dict(inspectors.get(edge_id) or {})
        if vector.get("conditions") != "source_exact":
            continue
        if (
            str(inspector.get("condition_status") or "") == "model_predicted"
            or int(vector.get("exact_procedure_record_count") or 0) < 1
        ):
            predicted_exact += 1
    claim = dict(report.get("claim") or {})
    disposition = report.get("current_disposition")
    disposition_reasons = (
        [str(value) for value in dict(disposition).get("reasons") or []]
        if isinstance(disposition, Mapping)
        else []
    )
    accepted = claim.get("accepted_under_configured_policy") is True
    explicit_unresolved = (
        not accepted
        and claim.get("no_unqualified_complete_claim") is True
        and (str(claim.get("achieved_profile") or "") == "unresolved" or bool(disposition_reasons))
    )
    repo_preflight = False
    try:
        repo_preflight = (
            preflight.get("accepted") is True
            and Path(str(preflight.get("repository_root") or "")).resolve() == repository_root
            and dict(preflight.get("semantics") or {}).get(
                "target_name_smiles_and_inchikey_checked"
            )
            is True
        )
    except OSError:
        repo_preflight = False
    snapshot_receipt = dict(status_row.get("snapshot_receipt") or {})
    supervisor = dict(snapshot_receipt.get("supervisor_preflight") or {})
    supervisor_semantics = dict(supervisor.get("semantics") or {})
    supervisor_preflight_valid = _json_digest_valid(supervisor)
    snapshot_receipt_valid = (
        frozen_snapshot_valid
        and _json_digest_valid(snapshot_receipt)
        and supervisor_preflight_valid
        and snapshot_receipt.get("panel_snapshot_sha256") == frozen_snapshot.get("content_sha256")
        and snapshot_receipt.get("base_environment_sha256")
        == frozen_snapshot.get("base_environment_sha256")
    )
    extended_preflight = (
        supervisor_preflight_valid
        and supervisor.get("accepted") is True
        and supervisor_semantics.get("target_synonym_needles_checked") is True
        and supervisor_semantics.get("key_intermediate_needles_checked") is True
    )
    resource = dict(report.get("resource_envelope") or {})
    within_budget = resource.get("within_budget") is True
    stage_timings = {
        str(row.get("stage") or ""): float(row.get("elapsed_s") or 0.0)
        for row in report.get("stages") or []
        if isinstance(row, Mapping)
    }
    failure_reasons = [
        *disposition_reasons,
        *([] if report_digest_valid else ["target_report_digest_invalid"]),
        *([] if not artifact_error else [artifact_error]),
        *([] if repo_preflight else ["repository_preflight_incomplete"]),
        *([] if extended_preflight else ["extended_leakage_preflight_incomplete"]),
        *([] if within_budget else ["resource_budget_exceeded"]),
    ]
    return {
        "case_id": str(
            dict(preflight.get("case") or {}).get("case_id") or report.get("run_id") or target_name
        ),
        "target_name": target_name,
        "status": str(status_row.get("status") or ""),
        "claim": str(claim.get("achieved_profile") or "unresolved"),
        "auditable_outcome": report_digest_valid
        and not artifact_error
        and (accepted or explicit_unresolved),
        "accepted_under_configured_policy": accepted,
        "explicit_unresolved": explicit_unresolved,
        "within_resource_budget": within_budget,
        "repository_preflight_complete": repo_preflight,
        "extended_preflight_complete": extended_preflight,
        "snapshot_receipt": snapshot_receipt,
        "snapshot_receipt_valid": snapshot_receipt_valid,
        "supervisor_preflight_digest_valid": supervisor_preflight_valid,
        "false_closure_claim_count": int(
            dict(report.get("gates") or {}).get("false_closure_claim_count") or 0
        ),
        "benchmark_procurement_false_claim_count": benchmark_procurement,
        "predicted_condition_exact_claim_count": predicted_exact,
        "reaction_validated_routes": len(validated),
        "distinct_validated_route_family_count": len(validated_families),
        "distinct_validated_edge_set_count": len(validated_edge_sets),
        "two_distinct_validated_routes": len(validated) >= 2
        and len(validated_families) >= 2
        and len(validated_edge_sets) >= 2,
        "procurement_closed_routes": len(procurement),
        "condition_complete_routes": len(conditions),
        "process_ready_routes": len(process),
        "open_visible_route_count": sum(
            not dict(route.get("acceptance_profiles") or {}).get("reaction_validated")
            for route in routes
        ),
        "profiles": {
            "fast_explore": len(exploration),
            "validated_plan": len(validated),
            "process_dossier": len(process),
        },
        "cost": dict(report.get("model_cost") or {}),
        "chemenzy_time_s": round(
            sum(value for key, value in stage_timings.items() if key.startswith("chemenzy_")),
            3,
        ),
        "target_smiles_for_diversity": str(
            dict(preflight.get("case") or {}).get("target_smiles") or ""
        ),
        "failure_reasons": sorted(set(failure_reasons)),
    }


def _load_workbench(report: Mapping[str, Any], *, panel_root: Path) -> tuple[dict[str, Any], str]:
    ref = dict(report.get("workbench_ref") or {})
    object_path = str(ref.get("object_path") or "")
    path = panel_root / "artifacts" / object_path
    if not object_path or not path.is_file():
        return {}, "workbench_artifact_missing"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != str(ref.get("sha256") or ""):
        return {}, "workbench_artifact_digest_invalid"
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}, "workbench_artifact_unreadable"
    if not isinstance(value, Mapping) or not _json_digest_valid(value):
        return {}, "workbench_scientific_digest_invalid"
    return dict(value), ""


def _frozen_snapshot_valid(status: Mapping[str, Any], *, status_path: Path) -> bool:
    frozen = dict(status.get("frozen_snapshot") or {})
    raw_path = str(frozen.get("path") or "").strip()
    if not raw_path:
        return False
    path = Path(raw_path)
    if not path.is_absolute():
        path = status_path.parent / path
    if not path.is_file():
        return False
    snapshot = _read_json(path)
    if (
        snapshot.get("schema_version") != "v4_blind_benchmark_snapshot.v1"
        or not _json_digest_valid(snapshot)
        or snapshot.get("content_sha256") != frozen.get("content_sha256")
        or snapshot.get("base_environment_sha256") != frozen.get("base_environment_sha256")
    ):
        return False
    knowledge = dict(snapshot.get("knowledge") or {})
    providers = dict(snapshot.get("provider_snapshot") or {})
    selected_case_ids = sorted(
        str(
            dict(row).get("case_id")
            or dict(dict(row).get("snapshot_receipt") or {}).get("case_id")
            or dict(row).get("run_id")
            or ""
        )
        for row in dict(status.get("targets") or {}).values()
        if isinstance(row, Mapping)
    )
    return (
        snapshot.get("manifest_sha256") == frozen.get("manifest_sha256")
        and sorted(str(value) for value in snapshot.get("selected_case_ids") or [])
        == selected_case_ids
        and snapshot.get("ablation") == status.get("ablation")
        and providers.get("model") == status.get("model")
        and providers.get("reasoning_effort") == status.get("reasoning_effort")
        and providers.get("execution_profile") == status.get("execution_profile")
        and int(providers.get("worker_count") or 0) == int(status.get("worker_count") or 0)
        and knowledge.get("self_evo_library_sha256") == frozen.get("self_evo_library_sha256")
        and knowledge.get("inventory_snapshot_sha256") == frozen.get("inventory_snapshot_sha256")
        and bool(dict(providers.get("codex_cli") or {}).get("sha256"))
        and bool(dict(providers.get("host_python") or {}).get("sha256"))
        and bool(dict(providers.get("chemenzy_python") or {}).get("sha256"))
    )


def _diversity(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    molecules = [
        Chem.MolFromSmiles(str(row.get("target_smiles_for_diversity") or "")) for row in rows
    ]
    valid = [molecule for molecule in molecules if molecule is not None]
    scaffolds = {
        MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=True)
        for molecule in valid
    }
    fingerprints = [
        AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048) for molecule in valid
    ]
    similarities = [
        DataStructs.TanimotoSimilarity(fingerprints[left], fingerprints[right])
        for left in range(len(fingerprints))
        for right in range(left + 1, len(fingerprints))
    ]
    return {
        "valid_structure_count": len(valid),
        "distinct_scaffold_count": len(scaffolds),
        "maximum_pairwise_morgan_similarity": round(max(similarities), 6) if similarities else 0.0,
        "median_pairwise_morgan_similarity": round(float(median(similarities)), 6)
        if similarities
        else 0.0,
    }


def _ablation_summary(
    status: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    *,
    baseline_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_by_id = {str(row.get("case_id") or ""): row for row in baseline_rows}
    by_id = {str(row.get("case_id") or ""): row for row in rows}
    return {
        "arm": str(status.get("ablation") or ""),
        "target_count": len(rows),
        "case_ids_match_baseline": set(by_id) == set(baseline_by_id),
        "frozen_snapshot_valid": status.get("_frozen_snapshot_valid") is True,
        "snapshot_bound_case_count": sum(row.get("snapshot_receipt_valid") is True for row in rows),
        "base_environment_sha256": str(
            dict(status.get("frozen_snapshot") or {}).get("base_environment_sha256") or ""
        ),
        "auditable_case_count": sum(row.get("auditable_outcome") is True for row in rows),
        "two_distinct_validated_case_count": sum(
            row.get("two_distinct_validated_routes") is True for row in rows
        ),
        "reaction_validated_route_count": sum(
            int(row.get("reaction_validated_routes") or 0) for row in rows
        ),
        "process_ready_route_count": sum(int(row.get("process_ready_routes") or 0) for row in rows),
        "total_model_invocations": sum(
            int(dict(row.get("cost") or {}).get("model_invocations") or 0) for row in rows
        ),
    }


def _ablation_gate(
    baseline: Mapping[str, Any],
    *,
    baseline_rows: list[Mapping[str, Any]],
    ablations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_env = str(
        dict(baseline.get("frozen_snapshot") or {}).get("base_environment_sha256") or ""
    )
    present = [arm for arm in REQUIRED_ABLATIONS if arm in ablations]
    matched = [
        arm
        for arm in present
        if dict(ablations[arm]).get("case_ids_match_baseline") is True
        and dict(ablations[arm]).get("frozen_snapshot_valid") is True
        and int(dict(ablations[arm]).get("snapshot_bound_case_count") or 0) == len(baseline_rows)
        and baseline_env
        and dict(ablations[arm]).get("base_environment_sha256") == baseline_env
    ]
    chemenzy = dict(ablations.get("no-chemenzy") or {})
    baseline_validated = sum(
        int(row.get("reaction_validated_routes") or 0) for row in baseline_rows
    )
    baseline_model_invocations = sum(
        int(dict(row.get("cost") or {}).get("model_invocations") or 0) for row in baseline_rows
    )
    validated_deltas = {
        arm: baseline_validated
        - int(dict(ablations.get(arm) or {}).get("reaction_validated_route_count") or 0)
        for arm in present
    }
    model_invocation_deltas = {
        arm: baseline_model_invocations
        - int(dict(ablations.get(arm) or {}).get("total_model_invocations") or 0)
        for arm in present
    }
    chemenzy_delta = validated_deltas["no-chemenzy"] if chemenzy else None
    independent_gain_reported = chemenzy_delta is not None and chemenzy_delta > 0
    return {
        "passed": len(matched) == len(REQUIRED_ABLATIONS),
        "present_arms": present,
        "same_case_and_environment_arms": matched,
        "independent_chemenzy_gain_reported": independent_gain_reported,
        "chemenzy_reaction_validated_route_delta": chemenzy_delta,
        "reaction_validated_route_delta_vs_disabled": validated_deltas,
        "model_invocation_delta_vs_disabled": model_invocation_deltas,
    }


def _profile_totals(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    values = {"fast_explore": 0, "validated_plan": 0, "process_dossier": 0}
    for row in rows:
        profiles = dict(row.get("profiles") or {})
        for key in values:
            values[key] += int(profiles.get(key) or 0)
    return values


def _gate(
    passed: bool, *, actual: Mapping[str, Any], required: Mapping[str, Any]
) -> dict[str, Any]:
    return {"passed": bool(passed), "actual": dict(actual), "required": dict(required)}


def _median(rows: list[Mapping[str, Any]], key: str) -> float:
    return round(float(median(float(row.get(key) or 0.0) for row in rows)), 3) if rows else 0.0


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"json_unreadable:{path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"json_not_object:{path}")
    return dict(value)


def _json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _json_digest_valid(value: Mapping[str, Any]) -> bool:
    material = dict(value)
    observed = str(material.pop("content_sha256", ""))
    return bool(observed) and observed == _json_sha256(material)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "RELEASE_GATE_SCHEMA",
    "compile_v4_blind_release_gate",
    "render_v4_blind_release_gate_html",
]
