"""Nine-statin literature workflow and self-evolution replay harness."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import urlopen

from cascade_planner.agent.evolution_manager import (
    EvolutionCandidate,
    LayeredKnowledgeBase,
    evaluate_benchmark_gate,
    validate_evolution_candidate,
)
from cascade_planner.agent.literature_research import retrieve_pubmed_query_evidence
from cascade_planner.agent.smiles_first import SmilesFirstWorkflowConfig, run_smiles_first_workflow


STATIN_PANEL_REPORT_SCHEMA = "statin_panel_literature_self_evo_report.v1"
STATIN_SELF_EVO_TEMPLATE_SCHEMA = "statin_self_evo_template_candidate.v1"
STATIN_FAMILY_SELF_EVO_TEMPLATE_SCHEMA = "statin_family_self_evo_template.v1"
STATIN_FULLFLOW_DOSSIER_SCHEMA = "statin_fullflow_synthesis_dossier.v1"
STATIN_MEMBER_ROUTE_TEMPLATE_SCHEMA = "statin_member_route_template.v1"
STATIN_FAMILY_ROUTE_TEMPLATE_SCHEMA = "statin_family_route_template.v1"
STATIN_ROUTE_TEMPLATE_STEP_SCHEMA = "statin_route_template_step.v1"
STATIN_TYPED_ARTIFACT_MANIFEST_SCHEMA = "statin_typed_artifact_manifest.v1"
STATIN_FULLFLOW_OVERVIEW_SCHEMA = "statin_panel_fullflow_overview.v1"
STATIN_LITERATURE_QUERY_TRACE_SCHEMA = "statin_literature_query_trace.v1"
STATIN_ROUTE_CLOSURE_AUDIT_SCHEMA = "statin_route_closure_audit.v1"
STATIN_ROUTE_CLOSURE_MATRIX_SCHEMA = "statin_route_closure_matrix.v1"
STATIN_CLOSURE_LEAD_CURATION_PACKET_SCHEMA = "statin_closure_lead_curation_packet.v1"
STATIN_CLOSURE_CURATION_RESULT_SET_SCHEMA = "statin_closure_curation_result_set.v1"
STATIN_FIELD_RESOLUTION_CANDIDATE_STATUSES = frozenset({
    "source_required_before_resolution_candidate",
    "access_probe_queued_pending_curator_extraction",
    "full_text_access_candidate_ready_for_curator",
    "doi_or_pubmed_source_ready_for_curator",
    "full_text_signal_candidate_ready_for_curator",
    "full_text_signal_no_field_signal_ready_for_curator",
})

DEFAULT_STATIN_SUMMARY = Path("docs/statins/summary.json")
NATURAL_STATINS = {"lovastatin", "simvastatin", "pravastatin", "mevastatin"}
SYNTHETIC_STATINS = {"atorvastatin", "fluvastatin", "pitavastatin", "rosuvastatin", "cerivastatin"}
ROUTE_RELEVANCE_STRONG_PHRASES = {
    "process chemistry",
    "chemical synthesis",
    "total synthesis",
    "synthetic route",
    "synthesis of",
    "semisynthesis",
    "semi-synthesis",
    "biotransformation",
    "fermentation",
    "intermediate",
    "intermediates",
    "side chain",
    "side-chain",
    "preparation of",
    "salt preparation",
    "scale-up",
    "crystallization",
    "resolution",
    "esterification",
    "hydrolysis",
    "deprotection",
    "lactonization",
    "olefination",
    "wittig",
    "horner",
    "paal",
    "impurity synthesis",
}
ROUTE_RELEVANCE_WEAK_TERMS = {
    "synthesis",
    "synthetic",
    "process",
    "route",
    "preparation",
    "lactone",
    "salt",
    "impurity",
    "resolution",
}
ROUTE_RELEVANCE_CONTEXT_GUARDS = {
    "clinical",
    "pharmacokinetic",
    "pharmacokinetics",
    "pharmacodynamic",
    "pharmacodynamics",
    "pharmacology",
    "patient",
    "patients",
    "therapy",
    "treatment",
    "dose",
    "efficacy",
    "safety",
    "adverse",
    "toxicity",
    "myopathy",
    "neuronal",
    "apoptosis",
    "rat",
    "rats",
    "mouse",
    "mice",
    "neonatal",
    "pathway",
    "cholesterol excretion",
    "lipid levels",
    "bioavailability",
    "metabolism",
    "plasma",
    "disease",
    "cardiovascular",
    "vascular",
    "vascular smooth muscle",
    "smooth muscle",
    "vascular smooth muscle cells",
    "proteoglycan",
    "proteoglycans",
    "ldl",
    "binding affinity",
    "non-lipid",
    "non-lipid-related effects",
    "pleiotropic effects",
    "artery",
    "arteries",
    "endothelial",
    "endothelium",
    "preeclampsia",
    "pregnancy",
    "uterine",
    "ex vivo",
    "metabolomics",
    "lc-ms",
    "stress",
    "fungal",
    "application of",
    "cholesterol-lowering",
    "hmg-coa reductase inhibitor",
    "determination",
    "chromatography",
    "ultra-performance liquid chromatography",
    "dietary supplement",
    "dietary supplements",
    "genome editing",
    "crispr",
    "taxol",
}


@dataclass
class StatinPanelTarget:
    name: str
    safe: str
    target_smiles: str
    family_bucket: str
    expected_reaction_class: str
    expected_family_id: str
    source_route_count: int = 0
    showcase_route_count: int = 0
    route_assets: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = "statin_panel_target.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_statin_panel_targets(path: str | Path = DEFAULT_STATIN_SUMMARY) -> list[StatinPanelTarget]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    targets: list[StatinPanelTarget] = []
    for row in payload.get("targets") or []:
        name = str(row.get("name") or "")
        safe = str(row.get("safe") or name).lower()
        if not safe:
            continue
        family_bucket = _family_bucket(safe)
        expected_class = (
            "statin_semisynthesis"
            if family_bucket == "natural_statin"
            else "statin_side_chain_convergence"
        )
        expected_family = (
            "natural_statin_semisynthesis"
            if family_bucket == "natural_statin"
            else "synthetic_statin"
        )
        targets.append(
            StatinPanelTarget(
                name=name,
                safe=safe,
                target_smiles=str(row.get("smiles") or ""),
                family_bucket=family_bucket,
                expected_reaction_class=expected_class,
                expected_family_id=expected_family,
                source_route_count=int(row.get("source_route_count") or 0),
                showcase_route_count=int(row.get("showcase_route_count") or 0),
                route_assets=[
                    {
                        "rank": item.get("rank"),
                        "steps": item.get("steps"),
                        "class": item.get("class"),
                        "svg": item.get("svg"),
                        "pdf": item.get("pdf"),
                    }
                    for item in row.get("routes") or []
                    if isinstance(item, dict)
                ],
            )
        )
    return targets


def run_statin_panel_literature_self_evo(
    *,
    output_root: str | Path,
    summary_path: str | Path = DEFAULT_STATIN_SUMMARY,
    targets: Iterable[str] | None = None,
    query_budget: int = 6,
    literature_backend: str = "api_json",
    execute_closure_followups: bool = False,
    closure_followup_limit: int = 0,
    execute_open_gap_searches: bool = False,
    open_gap_search_limit: int = 0,
    execute_full_text_access_probes: bool = False,
    full_text_access_probe_limit: int = 0,
    execute_full_text_signal_extractions: bool = False,
    full_text_signal_extraction_limit: int = 0,
) -> dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    selected = {str(item).strip().lower() for item in targets or [] if str(item).strip()}
    panel_targets = [
        target for target in load_statin_panel_targets(summary_path)
        if not selected or target.safe in selected or target.name.lower() in selected
    ]

    kb = LayeredKnowledgeBase()
    rows: list[dict[str, Any]] = []
    for target in panel_targets:
        case_dir = root / target.safe
        result = run_smiles_first_workflow(
            SmilesFirstWorkflowConfig(
                target_smiles=target.target_smiles,
                target_name=f"{target.safe}_statin_panel",
                family_hint=_family_hint(target),
                objective="route literature statin self-evo",
                frontier_smiles=target.target_smiles,
                output_dir=case_dir,
                query_budget=query_budget,
                literature_backend=literature_backend,
            )
        )
        row = _evaluate_target(target, result)
        evo = _register_self_evolution_candidate(kb, target, row)
        row["self_evolution"] = evo
        dossier = _write_fullflow_dossier(
            root,
            target,
            row,
            execute_closure_followups=execute_closure_followups,
            closure_followup_limit=closure_followup_limit,
        )
        row["fullflow_dossier"] = {
            "json": str(dossier["json_path"]),
            "markdown": str(dossier["markdown_path"]),
            "validation": dossier["validation"],
            "stage_count": len(dossier["dossier"].get("synthesis_stages") or []),
            "route_template_step_count": len(
                (dossier["dossier"].get("route_template") or {}).get("template_steps") or []
            ),
            "blueprint_outline_count": len(
                (dossier["dossier"].get("fullflow_blueprint") or {}).get("member_specific_route_outline") or []
            ),
            "difficulty_query_count": len(
                (dossier["dossier"].get("automatic_literature_escalation") or {}).get("difficulty_queries") or []
            ),
            "literature_trace_count": len(
                (dossier["dossier"].get("automatic_literature_escalation") or {}).get("query_execution_traces") or []
            ),
            "accepted_literature_trace_count": int(
                ((dossier["dossier"].get("automatic_literature_escalation") or {}).get("query_trace_summary") or {})
                .get("accepted_query_trace_count")
                or 0
            ),
        }
        rows.append(row)

    aggregation = _aggregate_self_evolution_templates(
        kb,
        rows,
        full_panel_run=(not selected and len(panel_targets) == 9),
    )
    full_panel_run = not selected and len(panel_targets) == 9

    hard_gates = {
        "all_nine_targets_present": len(panel_targets) == 9 if not selected else bool(panel_targets),
        "all_targets_validation_accepted": all(row["validation_accepted"] for row in rows),
        "all_targets_enter_literature_mode": all(row["literature_mode_entered"] for row in rows),
        "all_targets_have_expected_template": all(row["expected_template_hit"] for row in rows),
        "all_targets_have_required_candidate_kinds": all(row["required_candidate_kinds_hit"] for row in rows),
        "all_targets_self_evo_candidate_validated": all(
            row["self_evolution"]["candidate_validation"]["accepted"] for row in rows
        ),
        "no_target_run_writes_production_kb": all(
            row["self_evolution"]["production_write_blocked"] for row in rows
        ),
        "no_target_claims_solved": all(not row["claims_solved"] for row in rows),
        "all_targets_have_fullflow_dossiers": all(
            (row.get("fullflow_dossier") or {}).get("validation", {}).get("accepted") for row in rows
        ),
        "all_targets_have_step_level_route_templates": all(
            int((row.get("fullflow_dossier") or {}).get("route_template_step_count") or 0) >= 3
            for row in rows
        ),
        "all_targets_have_member_specific_blueprints": all(
            int((row.get("fullflow_dossier") or {}).get("blueprint_outline_count") or 0) >= 3
            for row in rows
        ),
        "all_targets_have_automatic_literature_queries": all(
            int((row.get("fullflow_dossier") or {}).get("difficulty_query_count") or 0) >= 3
            for row in rows
        ),
        "all_targets_have_executed_literature_traces": all(
            int((row.get("fullflow_dossier") or {}).get("accepted_literature_trace_count") or 0)
            >= int((row.get("fullflow_dossier") or {}).get("difficulty_query_count") or 0)
            >= 3
            for row in rows
        ),
        "self_evo_aggregation_ready": bool(aggregation.get("accepted") or aggregation.get("skipped")),
        "self_evo_aggregated_templates_promoted": (
            bool(aggregation.get("accepted"))
            if not selected and len(panel_targets) == 9
            else bool(aggregation.get("skipped"))
        ),
        "no_replay_run_writes_production_kb": int(aggregation.get("production_promoted_count") or 0) == 0,
        "fullflow_overview_written": True,
        "typed_artifact_manifest_validated": True,
    }
    report = {
        "schema_version": STATIN_PANEL_REPORT_SCHEMA,
        "run_semantics": "replay",
        "summary_path": str(summary_path),
        "output_root": str(root),
        "target_count": len(panel_targets),
        "passed": sum(1 for row in rows if row["passed"]),
        "failed": sum(1 for row in rows if not row["passed"]),
        "hard_gates": hard_gates,
        "self_evolution_aggregation": aggregation,
        "self_evolution_kb": kb.to_dict(),
        "targets": rows,
    }
    fullflow_overview = _write_fullflow_overview(root, report, full_panel_run=full_panel_run)
    report["fullflow_overview"] = _overview_summary(fullflow_overview)
    hard_gates["fullflow_overview_written"] = bool(
        fullflow_overview.get("skipped")
        or (fullflow_overview.get("validation") or {}).get("accepted")
    )
    route_closure_matrix = _write_route_closure_matrix(root, report, full_panel_run=full_panel_run)
    report["route_closure_matrix"] = _closure_matrix_summary(route_closure_matrix)
    closure_lead_curation_packet = _write_closure_lead_curation_packet(
        root,
        report,
        full_panel_run=full_panel_run,
    )
    report["closure_lead_curation_packet"] = _closure_lead_curation_packet_summary(
        closure_lead_curation_packet
    )
    closure_curation_result_set = _write_closure_curation_result_set(
        root,
        report,
        full_panel_run=full_panel_run,
        execute_open_gap_searches=execute_open_gap_searches,
        open_gap_search_limit=open_gap_search_limit,
        execute_full_text_access_probes=execute_full_text_access_probes,
        full_text_access_probe_limit=full_text_access_probe_limit,
        execute_full_text_signal_extractions=execute_full_text_signal_extractions,
        full_text_signal_extraction_limit=full_text_signal_extraction_limit,
    )
    report["closure_curation_result_set"] = _closure_curation_result_set_summary(
        closure_curation_result_set
    )
    typed_manifest = _write_typed_artifact_manifest(root, report, full_panel_run=full_panel_run)
    report["typed_artifact_manifest"] = _manifest_summary(typed_manifest)
    hard_gates["typed_artifact_manifest_validated"] = bool(
        typed_manifest.get("skipped")
        or (typed_manifest.get("validation_summary") or {}).get("accepted")
    )
    (root / "statin_panel_literature_self_evo_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    (root / "statin_panel_literature_self_evo_report.md").write_text(
        render_statin_panel_report(report),
        encoding="utf-8",
    )
    return report


def render_statin_panel_report(report: dict[str, Any]) -> str:
    lines = [
        "# 九他汀文献流程与 Self-Evo 回放",
        "",
        f"- target_count: `{report.get('target_count')}`",
        f"- passed: `{report.get('passed')}`",
        f"- failed: `{report.get('failed')}`",
        "",
        "## Hard Gates",
    ]
    for key, value in sorted((report.get("hard_gates") or {}).items()):
        lines.append(f"- `{key}`: `{bool(value)}`")
    lines.extend([
        "",
        "## Targets",
        "",
        "| target | family | status | expected template | candidate kinds | self-evo | warnings |",
        "|---|---|---|---|---|---|---|",
    ])
    for row in report.get("targets") or []:
        lines.append(
            "| {target} | {family} | {status} | {template} | {kinds} | {evo} | {warnings} |".format(
                target=row.get("name"),
                family=row.get("family_bucket"),
                status=row.get("route_status"),
                template="yes" if row.get("expected_template_hit") else "no",
                kinds=", ".join(row.get("observed_candidate_kinds") or []),
                evo=(row.get("self_evolution") or {}).get("kb_target_layer") or "",
                warnings=", ".join(row.get("warnings") or []),
            )
        )
    aggregation = report.get("self_evolution_aggregation") or {}
    lines.extend([
        "",
        "## Self-Evo Aggregation",
        "",
        f"- accepted: `{bool(aggregation.get('accepted'))}`",
        f"- skipped: `{bool(aggregation.get('skipped'))}`",
        f"- production_promoted_count: `{aggregation.get('production_promoted_count') or 0}`",
    ])
    for family in aggregation.get("families") or []:
        lines.append(
            "- `{family}` targets={targets} staging={staging} production={production} reasons=`{reasons}`".format(
                family=family.get("family_bucket"),
                targets=family.get("target_count"),
                staging=family.get("staging_promoted"),
                production=family.get("production_promoted"),
                reasons=", ".join(family.get("reasons") or []),
            )
        )
    manifest = report.get("typed_artifact_manifest") or {}
    overview = report.get("fullflow_overview") or {}
    lines.extend([
        "",
        "## Fullflow Overview",
        "",
        f"- skipped: `{bool(overview.get('skipped'))}`",
        f"- target_count: `{overview.get('target_count') or 0}`",
        f"- validation_accepted: `{bool((overview.get('validation') or {}).get('accepted'))}`",
        f"- markdown: `{overview.get('markdown') or ''}`",
        "",
        "## Typed Artifacts",
        "",
        f"- skipped: `{bool(manifest.get('skipped'))}`",
        f"- artifact_count: `{manifest.get('artifact_count') or 0}`",
        f"- validation_accepted: `{bool((manifest.get('validation_summary') or {}).get('accepted'))}`",
        f"- manifest: `{manifest.get('json') or ''}`",
    ])
    lines.extend([
        "",
        "## Contract",
        "",
        "- These packages are planning material, not solved-route claims.",
        "- Target-run self-evolution stops at staging; production promotion requires a separate benchmark gate.",
        "",
    ])
    return "\n".join(lines)


def _write_fullflow_overview(
    root: Path,
    report: dict[str, Any],
    *,
    full_panel_run: bool,
) -> dict[str, Any]:
    json_path = root / "statin_panel_fullflow_overview.json"
    markdown_path = root / "statin_panel_fullflow_overview.md"
    if not full_panel_run:
        overview = {
            "schema_version": STATIN_FULLFLOW_OVERVIEW_SCHEMA,
            "skipped": True,
            "skip_reason": "subset_replay_no_full_panel_overview",
            "target_count": int(report.get("target_count") or 0),
            "targets": [],
            "validation": {
                "schema_version": "statin_fullflow_overview_validation.v1",
                "accepted": True,
                "reasons": [],
            },
            "json": str(json_path),
            "markdown": str(markdown_path),
        }
        json_path.write_text(json.dumps(overview, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        markdown_path.write_text(_render_fullflow_overview_md(overview), encoding="utf-8")
        return overview

    rows: list[dict[str, Any]] = []
    for row in report.get("targets") or []:
        dossier_path = (row.get("fullflow_dossier") or {}).get("json")
        dossier = _read_json(dossier_path)
        blueprint = dossier.get("fullflow_blueprint") or {}
        route_template = dossier.get("route_template") or {}
        escalation = dossier.get("automatic_literature_escalation") or {}
        trace_summary = _literature_trace_summary(escalation)
        closure_audit = dossier.get("route_closure_audit") or {}
        self_evolution = dict(row.get("self_evolution") or {})
        rows.append({
            "target": row.get("name"),
            "safe": row.get("safe"),
            "family_bucket": row.get("family_bucket"),
            "route_status": row.get("route_status"),
            "expected_reaction_class": row.get("expected_reaction_class"),
            "expected_family_id": row.get("expected_family_id"),
            "template_sources": list(dossier.get("primary_template_sources") or []),
            "synthesis_stages": [
                {
                    "stage_id": stage.get("stage_id"),
                    "title": stage.get("title"),
                    "template_role": stage.get("template_role"),
                    "route_role": stage.get("route_role"),
                    "notes": stage.get("notes"),
                }
                for stage in dossier.get("synthesis_stages") or []
            ],
            "route_template_steps": [
                {
                    "step_id": step.get("step_id"),
                    "title": step.get("title"),
                    "template_role": step.get("template_role"),
                    "difficulty_queries": list(step.get("difficulty_queries") or []),
                    "literature_trace_refs": list(step.get("literature_trace_refs") or []),
                }
                for step in route_template.get("template_steps") or []
            ],
            "difficulty_queries": list(escalation.get("difficulty_queries") or []),
            "query_execution_traces": list(escalation.get("query_execution_traces") or []),
            "literature_trace": trace_summary,
            "key_intermediate_roles": list(blueprint.get("key_intermediate_roles") or []),
            "member_specific_route_outline": list(blueprint.get("member_specific_route_outline") or []),
            "route_closure_audit": {
                "schema_version": closure_audit.get("schema_version"),
                "readiness_status": closure_audit.get("readiness_status"),
                "solved_claim_allowed": bool(closure_audit.get("solved_claim_allowed")),
                "passed_requirement_count": len(closure_audit.get("passed_requirements") or []),
                "blocker_count": len(closure_audit.get("blocking_requirements") or []),
                "followup_query_count": len(closure_audit.get("automatic_followup_literature_queue") or []),
                "blocking_requirements": list(closure_audit.get("blocking_requirements") or []),
                "followup_execution": {
                    "policy": (closure_audit.get("followup_execution") or {}).get("policy"),
                    "requested": bool((closure_audit.get("followup_execution") or {}).get("requested")),
                    "trace_count": int((closure_audit.get("followup_execution") or {}).get("trace_count") or 0),
                    "executed_trace_count": int(
                        (closure_audit.get("followup_execution") or {}).get("executed_trace_count") or 0
                    ),
                    "lead_trace_count": int(
                        (closure_audit.get("followup_execution") or {}).get("lead_trace_count") or 0
                    ),
                    "abstract_signal_trace_count": int(
                        (closure_audit.get("followup_execution") or {}).get("abstract_signal_trace_count") or 0
                    ),
                    "abstract_signal_terms": list(
                        (closure_audit.get("followup_execution") or {}).get("abstract_signal_terms") or []
                    ),
                    "full_trace_coverage": bool(
                        (closure_audit.get("followup_execution") or {}).get("full_trace_coverage")
                    ),
                    "full_execution_coverage": bool(
                        (closure_audit.get("followup_execution") or {}).get("full_execution_coverage")
                    ),
                },
                "validation": dict(closure_audit.get("validation") or {}),
            },
            "dossier_refs": {
                "json": str(dossier_path or ""),
                "markdown": str((row.get("fullflow_dossier") or {}).get("markdown") or ""),
            },
            "self_evolution": {
                "candidate_id": self_evolution.get("candidate_id"),
                "status": self_evolution.get("status") or self_evolution.get("kb_target_layer"),
                "kb_target_layer": self_evolution.get("kb_target_layer"),
                "production_write_blocked": bool(self_evolution.get("production_write_blocked")),
                "candidate_validation_accepted": bool(
                    (self_evolution.get("candidate_validation") or {}).get("accepted")
                ),
            },
            "not_lab_procedure": bool(dossier.get("not_lab_procedure")),
            "validation": dict(dossier.get("validation") or {}),
        })
    overview = {
        "schema_version": STATIN_FULLFLOW_OVERVIEW_SCHEMA,
        "skipped": False,
        "target_count": int(report.get("target_count") or 0),
        "passed": int(report.get("passed") or 0),
        "failed": int(report.get("failed") or 0),
        "status_contract": (
            "These are route-planning fullflow synthesis templates with literature escalation hooks; "
            "they are not executable lab procedures and do not claim solved status."
        ),
        "targets": rows,
        "self_evolution_aggregation": dict(report.get("self_evolution_aggregation") or {}),
        "json": str(json_path),
        "markdown": str(markdown_path),
    }
    overview["validation"] = _validate_fullflow_overview(overview)
    json_path.write_text(json.dumps(overview, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_render_fullflow_overview_md(overview), encoding="utf-8")
    return overview


def _validate_fullflow_overview(overview: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if overview.get("schema_version") != STATIN_FULLFLOW_OVERVIEW_SCHEMA:
        reasons.append("invalid_fullflow_overview_schema")
    if overview.get("skipped"):
        return {
            "schema_version": "statin_fullflow_overview_validation.v1",
            "accepted": not reasons,
            "reasons": sorted(set(reasons)),
        }
    if int(overview.get("target_count") or 0) != 9:
        reasons.append("overview_not_all_nine_targets")
    if int(overview.get("failed") or 0) != 0:
        reasons.append("overview_has_failed_targets")
    targets = overview.get("targets") or []
    if len(targets) != 9:
        reasons.append("overview_target_rows_not_all_nine")
    for row in targets:
        safe = str(row.get("safe") or row.get("target") or "unknown")
        if row.get("route_status") == "solved":
            reasons.append(f"overview_target_claimed_solved:{safe}")
        if not row.get("not_lab_procedure"):
            reasons.append(f"overview_target_missing_not_lab_guard:{safe}")
        if len(row.get("synthesis_stages") or []) < 3:
            reasons.append(f"overview_target_insufficient_stages:{safe}")
        if len(row.get("route_template_steps") or []) < 3:
            reasons.append(f"overview_target_insufficient_template_steps:{safe}")
        if len(row.get("difficulty_queries") or []) < 3:
            reasons.append(f"overview_target_insufficient_difficulty_queries:{safe}")
        traces = row.get("query_execution_traces") or []
        if len(traces) < len(row.get("difficulty_queries") or []):
            reasons.append(f"overview_target_missing_query_execution_traces:{safe}")
        trace_summary = row.get("literature_trace") or {}
        if int(trace_summary.get("accepted_query_trace_count") or 0) < len(row.get("difficulty_queries") or []):
            reasons.append(f"overview_target_unaccepted_literature_traces:{safe}")
        if int(trace_summary.get("template_supported_query_trace_count") or 0) < len(row.get("difficulty_queries") or []):
            reasons.append(f"overview_target_literature_traces_missing_template_support:{safe}")
        for trace in traces:
            if trace.get("execution_status") != "covered_by_validated_literature_search":
                reasons.append(f"overview_target_query_trace_not_covered:{safe}:{trace.get('difficulty')}")
            if not trace.get("template_supporting_evidence_refs"):
                reasons.append(f"overview_target_query_trace_missing_template_support:{safe}:{trace.get('difficulty')}")
        if not row.get("template_sources"):
            reasons.append(f"overview_target_missing_template_sources:{safe}")
        if not row.get("key_intermediate_roles"):
            reasons.append(f"overview_target_missing_intermediate_roles:{safe}")
        if not (row.get("validation") or {}).get("accepted"):
            reasons.append(f"overview_target_dossier_not_validated:{safe}")
        closure_audit = row.get("route_closure_audit") or {}
        if not (closure_audit.get("validation") or {}).get("accepted"):
            reasons.append(f"overview_target_route_closure_audit_not_validated:{safe}")
        if closure_audit.get("solved_claim_allowed"):
            reasons.append(f"overview_target_closure_allows_unproven_solved_claim:{safe}")
        if int(closure_audit.get("blocker_count") or 0) < 1:
            reasons.append(f"overview_target_missing_route_closure_blockers:{safe}")
        followup_execution = closure_audit.get("followup_execution") or {}
        if followup_execution.get("policy") not in {"queued_only", "pubmed_lead_search"}:
            reasons.append(f"overview_target_invalid_followup_execution_policy:{safe}")
        self_evo = row.get("self_evolution") or {}
        if self_evo.get("kb_target_layer") != "staging":
            reasons.append(f"overview_target_self_evo_not_staging:{safe}")
        if not self_evo.get("production_write_blocked"):
            reasons.append(f"overview_target_production_write_not_blocked:{safe}")
    aggregation = overview.get("self_evolution_aggregation") or {}
    if not aggregation.get("accepted"):
        reasons.append("overview_self_evo_aggregation_not_accepted")
    if int(aggregation.get("production_promoted_count") or 0) != 0:
        reasons.append("overview_replay_promoted_production")
    if not all((row.get("staging_promoted") and not row.get("production_promoted")) for row in aggregation.get("families") or []):
        reasons.append("overview_family_templates_not_staging_only")
    return {
        "schema_version": "statin_fullflow_overview_validation.v1",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
    }


def _overview_summary(overview: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": overview.get("schema_version"),
        "skipped": bool(overview.get("skipped")),
        "skip_reason": overview.get("skip_reason") or "",
        "target_count": int(overview.get("target_count") or 0),
        "json": overview.get("json") or "",
        "markdown": overview.get("markdown") or "",
        "validation": dict(overview.get("validation") or {}),
    }


def _write_route_closure_matrix(
    root: Path,
    report: dict[str, Any],
    *,
    full_panel_run: bool,
) -> dict[str, Any]:
    json_path = root / "statin_route_closure_matrix.json"
    markdown_path = root / "statin_route_closure_matrix.md"
    if not full_panel_run:
        matrix = {
            "schema_version": STATIN_ROUTE_CLOSURE_MATRIX_SCHEMA,
            "skipped": True,
            "skip_reason": "subset_replay_no_full_panel_route_closure_matrix",
            "target_count": int(report.get("target_count") or 0),
            "blocker_count": 0,
            "rows": [],
            "validation": {
                "schema_version": "statin_route_closure_matrix_validation.v1",
                "accepted": True,
                "reasons": [],
            },
            "json": str(json_path),
            "markdown": str(markdown_path),
        }
        json_path.write_text(json.dumps(matrix, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        markdown_path.write_text(_render_route_closure_matrix_md(matrix), encoding="utf-8")
        return matrix

    rows: list[dict[str, Any]] = []
    target_names: set[str] = set()
    for row in report.get("targets") or []:
        safe = str(row.get("safe") or row.get("name") or "unknown").lower()
        dossier = _read_json((row.get("fullflow_dossier") or {}).get("json"))
        if not dossier:
            continue
        target = dossier.get("target") or {}
        closure = dossier.get("route_closure_audit") or {}
        target_names.add(safe)
        queue_by_requirement = {
            str(item.get("requirement_id") or ""): item
            for item in closure.get("automatic_followup_literature_queue") or []
        }
        execution = closure.get("followup_execution") or {}
        trace_by_requirement = {
            str(trace.get("requirement_id") or ""): trace
            for trace in execution.get("traces") or []
            if trace.get("requirement_id")
        }
        for blocker in closure.get("blocking_requirements") or []:
            requirement_id = str(blocker.get("requirement_id") or "")
            queue_item = queue_by_requirement.get(requirement_id) or {}
            trace = trace_by_requirement.get(requirement_id) or {}
            rows.append(_route_closure_matrix_row(
                safe=safe,
                target=target,
                blocker=blocker,
                queue_item=queue_item,
                execution=execution,
                trace=trace,
                dossier_path=(row.get("fullflow_dossier") or {}).get("json") or "",
            ))

    matrix = {
        "schema_version": STATIN_ROUTE_CLOSURE_MATRIX_SCHEMA,
        "skipped": False,
        "target_count": len(target_names),
        "targets": sorted(target_names),
        "blocker_count": len(rows),
        "queued_blocker_count": sum(1 for row in rows if row.get("queue_present")),
        "trace_link_count": sum(1 for row in rows if row.get("trace_present")),
        "executed_trace_count": sum(
            1 for row in rows
            if str(row.get("execution_status") or "").startswith("pubmed_followup_executed")
        ),
        "lead_trace_count": sum(
            1 for row in rows
            if row.get("execution_status") == "pubmed_followup_executed_with_leads"
        ),
        "route_relevant_trace_count": sum(
            1 for row in rows
            if int(row.get("route_relevant_source_count") or 0) > 0
        ),
        "route_context_guarded_trace_count": sum(
            1 for row in rows
            if int(row.get("route_context_guarded_source_count") or 0) > 0
        ),
        "abstract_signal_trace_count": sum(
            1 for row in rows
            if row.get("abstract_signal_status") == "abstract_route_signal_detected"
        ),
        "full_trace_coverage": bool(rows) and all(row.get("trace_present") for row in rows),
        "full_execution_coverage": bool(rows) and all(
            str(row.get("execution_status") or "").startswith("pubmed_followup_executed")
            for row in rows
        ),
        "unresolved_blocker_count": sum(1 for row in rows if row.get("closure_status") == "blocked"),
        "solved_claim_allowed_count": sum(1 for row in rows if row.get("solved_claim_allowed")),
        "rows": rows,
        "matrix_contract": (
            "This matrix proves blocker traceability and follow-up prioritization across the nine-statin "
            "panel. It does not close blockers or convert planning templates into executable procedures."
        ),
    }
    matrix["validation"] = _validate_route_closure_matrix(matrix)
    matrix["json"] = str(json_path)
    matrix["markdown"] = str(markdown_path)
    json_path.write_text(json.dumps(matrix, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_render_route_closure_matrix_md(matrix), encoding="utf-8")
    return matrix


def _route_closure_matrix_row(
    *,
    safe: str,
    target: dict[str, Any],
    blocker: dict[str, Any],
    queue_item: dict[str, Any],
    execution: dict[str, Any],
    trace: dict[str, Any],
    dossier_path: str,
) -> dict[str, Any]:
    trace_present = bool(trace)
    execution_status = str(trace.get("execution_status") or "")
    if not execution_status:
        execution_status = "queued_without_execution_trace" if queue_item else "missing_followup_queue"
    abstract_signal_terms = [
        str(term)
        for term in trace.get("abstract_signal_terms") or []
        if str(term).strip()
    ]
    lead_sources = [
        dict(source)
        for source in trace.get("lead_sources") or []
        if isinstance(source, dict)
    ]
    route_relevant_source_count = sum(
        1 for source in lead_sources
        if source.get("lead_relevance_status") == "route_relevant_strong"
    )
    route_context_guarded_source_count = sum(
        1 for source in lead_sources
        if source.get("route_context_guard_signals")
    )
    return {
        "schema_version": "statin_route_closure_matrix_row.v1",
        "target_safe": safe,
        "target_name": target.get("name") or safe,
        "family_bucket": target.get("family_bucket") or "",
        "requirement_id": blocker.get("requirement_id") or "",
        "title": blocker.get("title") or "",
        "closure_status": blocker.get("status") or "blocked",
        "blocker": blocker.get("blocker") or "",
        "followup_query": queue_item.get("query") or blocker.get("followup_query") or "",
        "acceptance_signal": queue_item.get("acceptance_signal") or "",
        "queue_present": bool(queue_item),
        "followup_policy": execution.get("policy") or "",
        "trace_present": trace_present,
        "execution_status": execution_status,
        "hit_count": int(trace.get("hit_count") or 0),
        "evidence_lead_refs": list(trace.get("evidence_lead_refs") or []),
        "lead_sources": lead_sources,
        "source_checklist_count": len(lead_sources),
        "route_relevant_source_count": route_relevant_source_count,
        "route_context_guarded_source_count": route_context_guarded_source_count,
        "lead_relevance_gate": trace.get("lead_relevance_gate") or (
            "route_relevant_strong" if route_relevant_source_count else "lead_metadata_only_or_context_guarded"
        ),
        "resolved_query": trace.get("resolved_query") or "",
        "query_attempt_count": int(trace.get("query_attempt_count") or 0),
        "fallback_used": bool(trace.get("fallback_used")),
        "search_sources": list(trace.get("search_sources") or []),
        "abstract_signal_status": trace.get("abstract_signal_status") or "",
        "abstract_signal_hit_count": int(trace.get("abstract_signal_hit_count") or 0),
        "abstract_signal_terms": abstract_signal_terms,
        "not_template_support": bool(trace.get("not_template_support", True)),
        "solved_claim_allowed": False,
        "unresolved_reason": "full_text_or_curator_route_audit_required",
        "next_action": _route_closure_matrix_next_action(blocker.get("requirement_id") or ""),
        "dossier_ref": str(dossier_path or ""),
    }


def _route_closure_matrix_next_action(requirement_id: str) -> str:
    actions = {
        "full_text_route_step_audit": "extract and verify a full-text step/intermediate route map before any solved claim",
        "condition_and_workup_evidence_audit": "audit conditions, workup, isolation, and compatibility from full text or curator records",
        "terminal_stock_or_source_audit": "close route leaves with stock, source, fermentation, or semisynthesis anchor proof",
        "endpoint_identity_and_salt_state_audit": "verify stereochemistry and endpoint acid/lactone/salt/counterion identity",
        "route_graph_leaf_closure_audit": "prove all route leaves are non-advanced and reject fake terminal closure",
        "hazard_regulatory_and_withdrawn_context_audit": "separate safety, impurity, process, and regulatory risk review from route planning",
        "withdrawn_drug_context_guard": "keep cerivastatin route evidence separate from use or recommendation claims",
    }
    return actions.get(str(requirement_id), "resolve blocker with curated evidence before promoting solved status")


def _closure_matrix_summary(matrix: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": matrix.get("schema_version"),
        "skipped": bool(matrix.get("skipped")),
        "skip_reason": matrix.get("skip_reason") or "",
        "target_count": int(matrix.get("target_count") or 0),
        "blocker_count": int(matrix.get("blocker_count") or 0),
        "queued_blocker_count": int(matrix.get("queued_blocker_count") or 0),
        "trace_link_count": int(matrix.get("trace_link_count") or 0),
        "executed_trace_count": int(matrix.get("executed_trace_count") or 0),
        "lead_trace_count": int(matrix.get("lead_trace_count") or 0),
        "route_relevant_trace_count": int(matrix.get("route_relevant_trace_count") or 0),
        "route_context_guarded_trace_count": int(matrix.get("route_context_guarded_trace_count") or 0),
        "abstract_signal_trace_count": int(matrix.get("abstract_signal_trace_count") or 0),
        "full_trace_coverage": bool(matrix.get("full_trace_coverage")),
        "full_execution_coverage": bool(matrix.get("full_execution_coverage")),
        "unresolved_blocker_count": int(matrix.get("unresolved_blocker_count") or 0),
        "json": matrix.get("json") or "",
        "markdown": matrix.get("markdown") or "",
        "validation": dict(matrix.get("validation") or {}),
    }


def _render_route_closure_matrix_md(matrix: dict[str, Any]) -> str:
    validation = matrix.get("validation") or {}
    lines = [
        "# 九他汀 Route Closure Blocker Matrix",
        "",
        f"- skipped: `{bool(matrix.get('skipped'))}`",
        f"- target_count: `{matrix.get('target_count') or 0}`",
        f"- blocker_count: `{matrix.get('blocker_count') or 0}`",
        f"- queued_blockers: `{matrix.get('queued_blocker_count') or 0}`",
        f"- executed_traces: `{matrix.get('executed_trace_count') or 0}`",
        f"- lead_traces: `{matrix.get('lead_trace_count') or 0}`",
        f"- route_relevant_traces: `{matrix.get('route_relevant_trace_count') or 0}`",
        f"- route_context_guarded_traces: `{matrix.get('route_context_guarded_trace_count') or 0}`",
        f"- abstract_signal_traces: `{matrix.get('abstract_signal_trace_count') or 0}`",
        f"- full_trace_coverage: `{bool(matrix.get('full_trace_coverage'))}`",
        f"- full_execution_coverage: `{bool(matrix.get('full_execution_coverage'))}`",
        f"- validation_accepted: `{bool(validation.get('accepted'))}`",
    ]
    if matrix.get("skip_reason"):
        lines.append(f"- skip_reason: `{matrix.get('skip_reason')}`")
    lines.extend([
        "",
        "## Blockers",
        "",
        "| target | blocker | queue | trace | route sources | abstract signal | next action |",
        "|---|---|---|---|---:|---|---|",
    ])
    for row in matrix.get("rows") or []:
        lines.append(
            "| {target} | {requirement} | {queue} | {trace} | {route_sources} | {signal} | {action} |".format(
                target=row.get("target_safe") or "",
                requirement=row.get("requirement_id") or "",
                queue="yes" if row.get("queue_present") else "no",
                trace=row.get("execution_status") or "",
                route_sources=row.get("route_relevant_source_count") or 0,
                signal=row.get("abstract_signal_status") or "",
                action=row.get("next_action") or "",
            )
        )
    if validation.get("reasons"):
        lines.extend(["", "## Validation Reasons"])
        for reason in validation.get("reasons") or []:
            lines.append(f"- `{reason}`")
    lines.append("")
    return "\n".join(lines)


def _write_closure_lead_curation_packet(
    root: Path,
    report: dict[str, Any],
    *,
    full_panel_run: bool,
) -> dict[str, Any]:
    json_path = root / "statin_closure_lead_curation_packet.json"
    markdown_path = root / "statin_closure_lead_curation_packet.md"
    if not full_panel_run:
        packet = {
            "schema_version": STATIN_CLOSURE_LEAD_CURATION_PACKET_SCHEMA,
            "skipped": True,
            "skip_reason": "subset_replay_no_full_panel_closure_lead_curation_packet",
            "target_count": int(report.get("target_count") or 0),
            "task_count": 0,
            "tasks": [],
            "validation": {
                "schema_version": "statin_closure_lead_curation_packet_validation.v1",
                "accepted": True,
                "reasons": [],
            },
            "json": str(json_path),
            "markdown": str(markdown_path),
        }
        json_path.write_text(json.dumps(packet, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        markdown_path.write_text(_render_closure_lead_curation_packet_md(packet), encoding="utf-8")
        return packet

    matrix = _read_json((report.get("route_closure_matrix") or {}).get("json"))
    tasks = [_closure_lead_curation_task(row) for row in (matrix.get("rows") or [])]
    packet = {
        "schema_version": STATIN_CLOSURE_LEAD_CURATION_PACKET_SCHEMA,
        "skipped": False,
        "target_count": int(matrix.get("target_count") or 0),
        "targets": list(matrix.get("targets") or []),
        "matrix_ref": (report.get("route_closure_matrix") or {}).get("json") or "",
        "blocker_count": int(matrix.get("blocker_count") or 0),
        "task_count": len(tasks),
        "lead_backed_task_count": sum(1 for task in tasks if task.get("evidence_lead_refs")),
        "source_metadata_task_count": sum(1 for task in tasks if task.get("lead_sources")),
        "fully_traceable_task_count": sum(
            1 for task in tasks
            if (task.get("source_checklist") or {}).get("all_leads_have_source_metadata")
        ),
        "route_relevant_task_count": sum(
            1 for task in tasks
            if int(task.get("route_relevant_source_count") or 0) > 0
        ),
        "route_context_guarded_task_count": sum(
            1 for task in tasks
            if int(task.get("route_context_guarded_source_count") or 0) > 0
        ),
        "abstract_signal_task_count": sum(
            1 for task in tasks
            if task.get("abstract_signal_status") == "abstract_route_signal_detected"
        ),
        "ready_for_curator_count": sum(
            1 for task in tasks
            if task.get("curation_status") == "pending_full_text_or_curator_audit"
        ),
        "full_execution_coverage": bool(matrix.get("full_execution_coverage")),
        "full_trace_coverage": bool(matrix.get("full_trace_coverage")),
        "template_promotion_allowed_count": sum(1 for task in tasks if task.get("template_promotion_allowed")),
        "evidence_refs": sorted({
            str(ref)
            for task in tasks
            for ref in task.get("evidence_lead_refs") or []
            if str(ref).strip()
        }),
        "tasks": tasks,
        "packet_contract": (
            "This packet turns blocker literature leads into curator/full-text audit tasks. It is not a "
            "source of reaction conditions and does not allow route-template or solved-status promotion."
        ),
    }
    packet["validation"] = _validate_closure_lead_curation_packet(packet)
    packet["json"] = str(json_path)
    packet["markdown"] = str(markdown_path)
    json_path.write_text(json.dumps(packet, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_render_closure_lead_curation_packet_md(packet), encoding="utf-8")
    return packet


def _closure_lead_curation_task(row: dict[str, Any]) -> dict[str, Any]:
    requirement_id = str(row.get("requirement_id") or "")
    target_safe = str(row.get("target_safe") or "")
    evidence_lead_refs = [str(ref) for ref in row.get("evidence_lead_refs") or [] if str(ref).strip()]
    abstract_terms = [str(term) for term in row.get("abstract_signal_terms") or [] if str(term).strip()]
    lead_sources = [
        dict(source)
        for source in row.get("lead_sources") or []
        if isinstance(source, dict)
    ]
    route_relevant_source_count = sum(
        1 for source in lead_sources
        if source.get("lead_relevance_status") == "route_relevant_strong"
    )
    route_context_guarded_source_count = sum(
        1 for source in lead_sources
        if source.get("route_context_guard_signals")
    )
    return {
        "schema_version": "statin_closure_lead_curation_task.v1",
        "task_id": f"{target_safe}:{requirement_id}",
        "target_safe": target_safe,
        "target_name": row.get("target_name") or target_safe,
        "family_bucket": row.get("family_bucket") or "",
        "requirement_id": requirement_id,
        "blocker": row.get("blocker") or "",
        "followup_query": row.get("followup_query") or "",
        "resolved_query": row.get("resolved_query") or "",
        "execution_status": row.get("execution_status") or "",
        "hit_count": int(row.get("hit_count") or 0),
        "evidence_lead_refs": evidence_lead_refs,
        "lead_sources": lead_sources,
        "source_checklist": _closure_curation_source_checklist(evidence_lead_refs, lead_sources),
        "route_relevant_source_count": route_relevant_source_count,
        "route_context_guarded_source_count": route_context_guarded_source_count,
        "lead_relevance_gate": row.get("lead_relevance_gate") or (
            "route_relevant_strong" if route_relevant_source_count else "lead_metadata_only_or_context_guarded"
        ),
        "search_sources": list(row.get("search_sources") or []),
        "abstract_signal_status": row.get("abstract_signal_status") or "",
        "abstract_signal_terms": abstract_terms,
        "priority": _closure_curation_priority(row),
        "curation_status": "pending_full_text_or_curator_audit",
        "template_promotion_allowed": False,
        "solved_claim_allowed": False,
        "not_lab_procedure": True,
        "not_template_support": True,
        "extraction_schema": _closure_curation_extraction_schema(requirement_id),
        "acceptance_criteria": _closure_curation_acceptance_criteria(requirement_id),
        "rejection_rules": _closure_curation_rejection_rules(requirement_id),
        "next_action": row.get("next_action") or _route_closure_matrix_next_action(requirement_id),
        "source_matrix_row": {
            "target_safe": target_safe,
            "requirement_id": requirement_id,
            "dossier_ref": row.get("dossier_ref") or "",
        },
    }


def _closure_curation_priority(row: dict[str, Any]) -> str:
    requirement_id = str(row.get("requirement_id") or "")
    has_route_relevant_source = int(row.get("route_relevant_source_count") or 0) > 0
    has_leads = bool(row.get("evidence_lead_refs"))
    if requirement_id in {"full_text_route_step_audit", "condition_and_workup_evidence_audit"} and has_route_relevant_source:
        return "P0"
    if requirement_id in {"terminal_stock_or_source_audit", "endpoint_identity_and_salt_state_audit"} and has_route_relevant_source:
        return "P1"
    if has_leads:
        return "P2"
    return "P3"


def _closure_curation_source_checklist(
    evidence_lead_refs: list[str],
    lead_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    source_refs = {
        str(source.get("evidence_ref") or "")
        for source in lead_sources
        if source.get("evidence_ref")
    }
    route_relevant_count = sum(
        1 for source in lead_sources
        if source.get("lead_relevance_status") == "route_relevant_strong"
    )
    context_guarded_count = sum(
        1 for source in lead_sources
        if source.get("route_context_guard_signals")
    )
    weak_or_metadata_only_count = sum(
        1 for source in lead_sources
        if source.get("lead_relevance_status") in {
            "route_relevant_guarded",
            "weak_route_signal_only",
            "non_route_context_suspected",
            "no_route_signal",
        }
    )
    relevance_audited_refs = {
        str(source.get("evidence_ref") or "")
        for source in lead_sources
        if source.get("lead_relevance_status")
    }
    return {
        "schema_version": "statin_closure_source_checklist.v1",
        "evidence_lead_count": len(evidence_lead_refs),
        "source_metadata_count": len(lead_sources),
        "all_leads_have_source_metadata": bool(evidence_lead_refs) and all(ref in source_refs for ref in evidence_lead_refs),
        "route_relevant_source_count": route_relevant_count,
        "route_context_guarded_source_count": context_guarded_count,
        "weak_or_metadata_only_source_count": weak_or_metadata_only_count,
        "all_leads_have_route_relevance_audit": bool(evidence_lead_refs) and all(
            ref in relevance_audited_refs for ref in evidence_lead_refs
        ),
        "required_for_full_text_audit": [
            "pmid_or_doi",
            "source_url",
            "source_title",
            "publication_year_or_pubdate",
            "route_relevance_audit",
        ],
        "missing_source_metadata_refs": [
            ref for ref in evidence_lead_refs
            if ref not in source_refs
        ],
        "missing_route_relevance_audit_refs": [
            ref for ref in evidence_lead_refs
            if ref not in relevance_audited_refs
        ],
        "template_promotion_guard": "source metadata enables full-text audit only; it is not template support.",
    }


def _closure_curation_extraction_schema(requirement_id: str) -> dict[str, Any]:
    common = {
        "schema_version": "statin_closure_lead_extraction_schema.v1",
        "required_source_fields": ["pmid_or_doi", "source_title", "publication_year", "source_url"],
        "forbidden_fields": ["abstract_text", "raw_reaction", "unverified_experimental_procedure"],
        "promotion_guard": "full_text_or_curator_audit_required_before_template_support",
    }
    if requirement_id == "full_text_route_step_audit":
        common["required_route_fields"] = [
            "route_stage_ids",
            "intermediate_identity_refs",
            "step_order_evidence",
            "endpoint_mapping",
        ]
    elif requirement_id == "condition_and_workup_evidence_audit":
        common["required_route_fields"] = [
            "condition_presence_evidence",
            "workup_or_isolation_presence",
            "compatibility_notes",
            "missing_condition_flags",
        ]
    elif requirement_id == "terminal_stock_or_source_audit":
        common["required_route_fields"] = [
            "route_leaf_identity",
            "stock_or_source_evidence",
            "fermentation_or_semisynthesis_anchor",
            "advanced_leaf_flags",
        ]
    elif requirement_id == "endpoint_identity_and_salt_state_audit":
        common["required_route_fields"] = [
            "endpoint_form",
            "stereochemistry_evidence",
            "acid_lactone_salt_state",
            "counterion_or_characterization_refs",
        ]
    else:
        common["required_route_fields"] = [
            "risk_context",
            "route_relevance",
            "remaining_blockers",
            "curator_notes",
        ]
    return common


def _closure_curation_acceptance_criteria(requirement_id: str) -> list[str]:
    base = [
        "source is traceable by PMID, DOI, URL, or curated local reference",
        "curator records only evidence-derived fields and does not infer missing conditions",
        "task result keeps template_promotion_allowed false until validation gates pass",
    ]
    specific = {
        "full_text_route_step_audit": "route stages and intermediate identities are mapped to source evidence",
        "condition_and_workup_evidence_audit": "conditions/workup/isolation are marked present, absent, or not stated from evidence",
        "terminal_stock_or_source_audit": "route leaves have stock/source/fermentation/semisynthesis anchor evidence or remain blocked",
        "endpoint_identity_and_salt_state_audit": "endpoint stereochemistry and acid/lactone/salt state are explicitly traced",
        "route_graph_leaf_closure_audit": "advanced leaves and fake terminal closures are explicitly rejected or resolved",
        "hazard_regulatory_and_withdrawn_context_audit": "hazard, impurity, process, and regulatory context remains separated from route solving",
        "withdrawn_drug_context_guard": "cerivastatin route evidence is separated from use or recommendation claims",
    }
    return [*base, specific.get(requirement_id, "blocker-specific evidence is curated before any closure claim")]


def _closure_curation_rejection_rules(requirement_id: str) -> list[str]:
    return [
        "reject summary-only lead as template support without full-text or curator audit",
        "reject any task result containing raw abstract text or copied experimental procedure",
        "reject solved-status promotion if route leaves, endpoint state, or condition evidence remain incomplete",
        f"reject {requirement_id or 'blocker'} closure without matching acceptance criteria",
    ]


def _closure_lead_curation_packet_summary(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": packet.get("schema_version"),
        "skipped": bool(packet.get("skipped")),
        "skip_reason": packet.get("skip_reason") or "",
        "target_count": int(packet.get("target_count") or 0),
        "task_count": int(packet.get("task_count") or 0),
        "lead_backed_task_count": int(packet.get("lead_backed_task_count") or 0),
        "source_metadata_task_count": int(packet.get("source_metadata_task_count") or 0),
        "fully_traceable_task_count": int(packet.get("fully_traceable_task_count") or 0),
        "route_relevant_task_count": int(packet.get("route_relevant_task_count") or 0),
        "route_context_guarded_task_count": int(packet.get("route_context_guarded_task_count") or 0),
        "abstract_signal_task_count": int(packet.get("abstract_signal_task_count") or 0),
        "ready_for_curator_count": int(packet.get("ready_for_curator_count") or 0),
        "full_execution_coverage": bool(packet.get("full_execution_coverage")),
        "json": packet.get("json") or "",
        "markdown": packet.get("markdown") or "",
        "validation": dict(packet.get("validation") or {}),
    }


def _write_closure_curation_result_set(
    root: Path,
    report: dict[str, Any],
    *,
    full_panel_run: bool,
    execute_open_gap_searches: bool = False,
    open_gap_search_limit: int = 0,
    execute_full_text_access_probes: bool = False,
    full_text_access_probe_limit: int = 0,
    execute_full_text_signal_extractions: bool = False,
    full_text_signal_extraction_limit: int = 0,
) -> dict[str, Any]:
    json_path = root / "statin_closure_curation_result_set.json"
    markdown_path = root / "statin_closure_curation_result_set.md"
    if not full_panel_run:
        result_set = {
            "schema_version": STATIN_CLOSURE_CURATION_RESULT_SET_SCHEMA,
            "skipped": True,
            "skip_reason": "subset_replay_no_full_panel_closure_curation_result_set",
            "target_count": int(report.get("target_count") or 0),
            "result_count": 0,
            "open_gap_followup_count": 0,
            "open_gap_review_ready_count": 0,
            "open_gap_search_required_count": 0,
            "open_gap_curator_review_draft_count": 0,
            "open_gap_selected_source_review_draft_count": 0,
            "open_gap_search_execution_package_count": 0,
            "open_gap_search_trace_count": 0,
            "open_gap_search_executed_count": 0,
            "open_gap_search_lead_count": 0,
            "open_gap_search_selected_source_count": 0,
            "open_gap_full_text_access_probe_count": 0,
            "open_gap_full_text_access_executed_count": 0,
            "open_gap_full_text_access_candidate_count": 0,
            "open_gap_full_text_signal_extraction_count": 0,
            "open_gap_full_text_signal_executed_count": 0,
            "open_gap_full_text_signal_candidate_count": 0,
            "open_gap_resolution_candidate_count": 0,
            "open_gap_self_evo_inbox_count": 0,
            "open_gap_full_text_access_execution": {
                "schema_version": "statin_open_gap_full_text_access_execution_summary.v1",
                "requested": False,
                "requested_limit": 0,
                "resolved_limit": 0,
                "full_queue_requested": False,
                "review_ready_followup_count": 0,
                "runnable_probe_followup_count": 0,
                "carried_forward_probe_count": 0,
                "executed_probe_count": 0,
                "open_access_candidate_count": 0,
                "metadata_candidate_count": 0,
                "failed_probe_count": 0,
                "policy": "queued_only",
            },
            "open_gap_full_text_signal_execution": {
                "schema_version": "statin_open_gap_full_text_signal_execution_summary.v1",
                "requested": False,
                "requested_limit": 0,
                "resolved_limit": 0,
                "full_queue_requested": False,
                "open_access_followup_count": 0,
                "runnable_signal_followup_count": 0,
                "carried_forward_signal_count": 0,
                "executed_signal_count": 0,
                "field_signal_candidate_count": 0,
                "failed_signal_count": 0,
                "policy": "queued_only",
            },
            "results": [],
            "validation": {
                "schema_version": "statin_closure_curation_result_set_validation.v1",
                "accepted": True,
                "reasons": [],
            },
            "json": str(json_path),
            "markdown": str(markdown_path),
        }
        json_path.write_text(json.dumps(result_set, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        markdown_path.write_text(_render_closure_curation_result_set_md(result_set), encoding="utf-8")
        return result_set

    packet = _read_json((report.get("closure_lead_curation_packet") or {}).get("json"))
    results = [_closure_curation_result(task) for task in (packet.get("tasks") or [])]
    previous_result_set = _read_json(json_path)
    carried_forward_search_traces = _carry_forward_open_gap_search_traces(results, previous_result_set)
    carried_forward_access_probes = _carry_forward_open_gap_full_text_access_probes(results, previous_result_set)
    open_gap_search_execution = _execute_open_gap_search_packages(
        results,
        execute_open_gap_searches=execute_open_gap_searches,
        open_gap_search_limit=open_gap_search_limit,
        carried_forward_search_traces=carried_forward_search_traces,
    )
    open_gap_full_text_access_execution = _execute_open_gap_full_text_access_probes(
        results,
        execute_full_text_access_probes=execute_full_text_access_probes,
        full_text_access_probe_limit=full_text_access_probe_limit,
        carried_forward_probe_count=carried_forward_access_probes,
    )
    carried_forward_signal_extractions = _carry_forward_open_gap_full_text_signal_extractions(
        results,
        previous_result_set,
    )
    open_gap_full_text_signal_execution = _execute_open_gap_full_text_signal_extractions(
        results,
        execute_full_text_signal_extractions=execute_full_text_signal_extractions,
        full_text_signal_extraction_limit=full_text_signal_extraction_limit,
        carried_forward_signal_count=carried_forward_signal_extractions,
    )
    result_set = {
        "schema_version": STATIN_CLOSURE_CURATION_RESULT_SET_SCHEMA,
        "skipped": False,
        "target_count": int(packet.get("target_count") or 0),
        "targets": list(packet.get("targets") or []),
        "curation_packet_ref": (report.get("closure_lead_curation_packet") or {}).get("json") or "",
        "task_count": int(packet.get("task_count") or 0),
        "result_count": len(results),
        "lead_backed_result_count": sum(1 for result in results if result.get("evidence_lead_refs")),
        "route_relevant_result_count": sum(
            1 for result in results
            if int((result.get("source_selection_summary") or {}).get("selected_route_source_count") or 0) > 0
        ),
        "context_guarded_result_count": sum(
            1 for result in results
            if int((result.get("source_selection_summary") or {}).get("context_guarded_source_count") or 0) > 0
        ),
        "needs_better_lead_count": sum(
            1 for result in results
            if result.get("curation_result_status") == "needs_better_route_specific_leads_or_manual_curator_source"
        ),
        "curator_record_supported_result_count": sum(
            1 for result in results
            if int(result.get("validated_route_field_count") or 0) > 0
        ),
        "validated_route_field_count": sum(
            int(result.get("validated_route_field_count") or 0)
            for result in results
        ),
        "audited_gap_route_field_count": sum(
            int(result.get("audited_gap_route_field_count") or 0)
            for result in results
        ),
        "missing_route_field_count": sum(
            int(result.get("missing_route_field_count") or 0)
            for result in results
        ),
        "open_route_field_count": sum(
            int(result.get("open_route_field_count") or 0)
            for result in results
        ),
        "open_gap_followup_count": sum(
            len(result.get("open_gap_followup_tasks") or [])
            for result in results
        ),
        "open_gap_review_ready_count": sum(
            1
            for result in results
            for followup in result.get("open_gap_followup_tasks") or []
            if (followup.get("literature_triage") or {}).get("triage_status") == "selected_source_review_ready"
        ),
        "open_gap_search_required_count": sum(
            1
            for result in results
            for followup in result.get("open_gap_followup_tasks") or []
            if (followup.get("literature_triage") or {}).get("triage_status") == "route_specific_search_required"
        ),
        "open_gap_curator_review_draft_count": sum(
            1
            for result in results
            for followup in result.get("open_gap_followup_tasks") or []
            if (followup.get("literature_triage") or {}).get("curator_review_draft")
        ),
        "open_gap_selected_source_review_draft_count": sum(
            1
            for result in results
            for followup in result.get("open_gap_followup_tasks") or []
            if (
                (followup.get("literature_triage") or {})
                .get("curator_review_draft") or {}
            ).get("draft_status") == "metadata_ready_pending_full_text_or_curator_confirmation"
        ),
        "open_gap_search_execution_package_count": sum(
            1
            for result in results
            for followup in result.get("open_gap_followup_tasks") or []
            if (followup.get("literature_triage") or {}).get("search_execution_package")
        ),
        "open_gap_search_trace_count": sum(
            1
            for result in results
            for followup in result.get("open_gap_followup_tasks") or []
            if (
                (followup.get("literature_triage") or {})
                .get("search_execution_package") or {}
            ).get("execution_trace")
        ),
        "open_gap_search_executed_count": sum(
            1
            for result in results
            for followup in result.get("open_gap_followup_tasks") or []
            if str(
                (
                    (
                        (followup.get("literature_triage") or {})
                        .get("search_execution_package") or {}
                    ).get("execution_trace") or {}
                ).get("execution_status") or ""
            ).startswith("pubmed_open_gap_search_executed")
        ),
        "open_gap_search_lead_count": sum(
            1
            for result in results
            for followup in result.get("open_gap_followup_tasks") or []
            if int(
                (
                    (
                        (followup.get("literature_triage") or {})
                        .get("search_execution_package") or {}
                    ).get("execution_trace") or {}
                ).get("hit_count") or 0
            ) > 0
        ),
        "open_gap_search_selected_source_count": sum(
            1
            for result in results
            for followup in result.get("open_gap_followup_tasks") or []
            if (
                (
                    (
                        (followup.get("literature_triage") or {})
                        .get("search_execution_package") or {}
                    ).get("execution_trace") or {}
                ).get("selected_route_source_refs") or []
            )
        ),
        "open_gap_full_text_access_probe_count": len(_open_gap_full_text_access_probe_rows(results)),
        "open_gap_full_text_access_executed_count": sum(
            1
            for probe in _open_gap_full_text_access_probe_rows(results)
            if str(probe.get("execution_status") or "").startswith("full_text_access_probe_executed")
        ),
        "open_gap_full_text_access_candidate_count": _open_gap_full_text_access_candidate_count(results),
        "open_gap_full_text_signal_extraction_count": len(_open_gap_full_text_signal_extraction_rows(results)),
        "open_gap_full_text_signal_executed_count": sum(
            1
            for extraction in _open_gap_full_text_signal_extraction_rows(results)
            if str(extraction.get("execution_status") or "").startswith("full_text_signal_extraction_executed")
        ),
        "open_gap_full_text_signal_candidate_count": _open_gap_full_text_signal_candidate_count(results),
        "open_gap_resolution_candidate_count": _open_gap_field_resolution_candidate_count(results),
        "open_gap_self_evo_inbox_count": sum(
            1
            for result in results
            for followup in result.get("open_gap_followup_tasks") or []
            if followup.get("self_evo_inbox_entry")
        ),
        "full_text_extraction_required_count": sum(
            1 for result in results
            if result.get("curation_result_status") in {
                "awaiting_full_text_route_extraction",
                "partial_curator_record_applied_pending_remaining_audit",
            }
        ),
        "blocked_result_count": sum(
            1 for result in results
            if result.get("candidate_template_gate_status") != "promotion_allowed"
        ),
        "template_promotion_allowed_count": sum(1 for result in results if result.get("template_promotion_allowed")),
        "solved_claim_allowed_count": sum(1 for result in results if result.get("solved_claim_allowed")),
        "candidate_template_gate_status": "blocked_pending_full_text_or_curator_records",
        "production_write_blocked": True,
        "not_lab_procedure": True,
        "open_gap_search_execution": open_gap_search_execution,
        "open_gap_carried_forward_search_trace_count": carried_forward_search_traces,
        "open_gap_full_text_access_execution": open_gap_full_text_access_execution,
        "open_gap_carried_forward_full_text_access_probe_count": carried_forward_access_probes,
        "open_gap_full_text_signal_execution": open_gap_full_text_signal_execution,
        "open_gap_carried_forward_full_text_signal_extraction_count": carried_forward_signal_extractions,
        "evidence_refs": sorted({
            str(ref)
            for result in results
            for ref in result.get("evidence_lead_refs") or []
            if str(ref).strip()
        }),
        "results": results,
        "result_set_contract": (
            "This result set turns closure curation tasks into blocked self-evo template candidates. "
            "It can prioritize full-text extraction but cannot promote templates or solved status."
        ),
    }
    result_set["validation"] = _validate_closure_curation_result_set(result_set)
    result_set["json"] = str(json_path)
    result_set["markdown"] = str(markdown_path)
    json_path.write_text(json.dumps(result_set, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_render_closure_curation_result_set_md(result_set), encoding="utf-8")
    return result_set


def _closure_curation_result(task: dict[str, Any]) -> dict[str, Any]:
    task_id = str(task.get("task_id") or "")
    target_safe = str(task.get("target_safe") or "")
    requirement_id = str(task.get("requirement_id") or "")
    lead_sources = [
        dict(source)
        for source in task.get("lead_sources") or []
        if isinstance(source, dict)
    ]
    lead_sources = _normalize_closure_curation_task_lead_sources(task, lead_sources)
    selected_sources = [
        source for source in lead_sources
        if source.get("lead_relevance_status") == "route_relevant_strong"
    ]
    context_guarded_sources = [
        source for source in lead_sources
        if source.get("route_context_guard_signals")
    ]
    weak_or_metadata_only_sources = [
        source for source in lead_sources
        if source.get("lead_relevance_status") != "route_relevant_strong"
    ]
    evidence_lead_refs = [str(ref) for ref in task.get("evidence_lead_refs") or [] if str(ref).strip()]
    required_fields = [
        str(field)
        for field in (task.get("extraction_schema") or {}).get("required_route_fields") or []
        if str(field).strip()
    ]
    local_field_evidence = _local_curator_route_field_evidence(task, required_fields)
    route_field_audit = []
    for field_name in required_fields:
        field_evidence = local_field_evidence.get(field_name) or {}
        if field_evidence:
            field_status = field_evidence.get("status") or "validated_local_curator_record"
            route_field_audit.append({
                "field": field_name,
                "status": field_status,
                "evidence_refs": list(field_evidence.get("evidence_refs") or []),
                "curator_record_refs": list(field_evidence.get("curator_record_refs") or []),
                "summary": field_evidence.get("summary") or "",
                "resolution_required_before_promotion": bool(
                    field_evidence.get("resolution_required_before_promotion")
                ),
            })
        else:
            route_field_audit.append({
                "field": field_name,
                "status": "missing_full_text_or_curator_record",
                "evidence_refs": [],
                "curator_record_refs": [],
                "summary": "",
                "resolution_required_before_promotion": True,
            })
    validated_fields = [
        item for item in route_field_audit
        if item.get("status") in {"validated_local_curator_record", "not_applicable_local_curator_record"}
    ]
    audited_gap_fields = [
        item for item in route_field_audit
        if item.get("status") == "audited_gap_local_curator_record"
    ]
    missing_fields = [
        item for item in route_field_audit
        if item.get("status") == "missing_full_text_or_curator_record"
    ]
    open_fields = [
        item for item in route_field_audit
        if item.get("resolution_required_before_promotion")
    ]
    curator_record_refs = sorted({
        str(ref)
        for item in validated_fields
        for ref in item.get("curator_record_refs") or []
        if str(ref).strip()
    })
    if validated_fields and not open_fields:
        curation_status = "local_curator_record_applied_pending_promotion_review"
        next_action = "review curated route-field record and decide whether a human promotion gate may run"
    elif validated_fields:
        curation_status = "partial_curator_record_applied_pending_remaining_audit"
        next_action = "complete remaining route fields with full text or curator records"
    elif selected_sources:
        curation_status = "awaiting_full_text_route_extraction"
        next_action = "extract required route fields from selected source full text or curator record"
    elif evidence_lead_refs:
        curation_status = "needs_better_route_specific_leads_or_manual_curator_source"
        next_action = "replace weak or guarded leads with route-specific full-text/curator evidence"
    else:
        curation_status = "queued_without_literature_leads"
        next_action = "execute PubMed/manual follow-up or attach curated local route evidence"

    candidate_status = (
        "blocked_pending_promotion_review"
        if validated_fields and not missing_fields
        else "blocked_pending_remaining_route_field_audit"
        if validated_fields
        else "blocked_pending_full_text_audit"
        if selected_sources
        else "blocked_pending_route_specific_lead"
    )
    selected_refs = [
        str(source.get("evidence_ref") or "")
        for source in selected_sources
        if source.get("evidence_ref")
    ]
    guarded_refs = [
        str(source.get("evidence_ref") or "")
        for source in context_guarded_sources
        if source.get("evidence_ref")
    ]
    rejected_refs = [
        str(source.get("evidence_ref") or "")
        for source in weak_or_metadata_only_sources
        if source.get("evidence_ref")
    ]
    open_gap_followup_tasks = _closure_open_gap_followup_tasks(
        task,
        open_fields,
        lead_sources=lead_sources,
        selected_source_refs=selected_refs,
        context_guarded_source_refs=guarded_refs,
        weak_or_rejected_source_refs=rejected_refs,
    )
    return {
        "schema_version": "statin_closure_curation_result.v1",
        "result_id": f"{task_id}:curation_result",
        "task_id": task_id,
        "target_safe": target_safe,
        "target_name": task.get("target_name") or target_safe,
        "family_bucket": task.get("family_bucket") or "",
        "requirement_id": requirement_id,
        "priority": task.get("priority") or "",
        "curation_result_status": curation_status,
        "candidate_template_gate_status": "blocked_pending_full_text_or_curator_records",
        "template_promotion_allowed": False,
        "solved_claim_allowed": False,
        "not_template_support": True,
        "not_lab_procedure": True,
        "evidence_lead_refs": evidence_lead_refs,
        "source_selection_summary": {
            "schema_version": "statin_closure_source_selection_summary.v1",
            "source_count": len(lead_sources),
            "selected_route_source_count": len(selected_sources),
            "context_guarded_source_count": len(context_guarded_sources),
            "weak_or_metadata_only_source_count": len(weak_or_metadata_only_sources),
            "selected_route_source_refs": selected_refs,
            "context_guarded_source_refs": guarded_refs,
            "weak_or_rejected_source_refs": rejected_refs,
            "selection_policy": (
                "select only route_relevant_strong sources for full-text extraction; keep guarded/weak leads as triage notes"
            ),
        },
        "required_route_fields": required_fields,
        "route_field_audit": route_field_audit,
        "validated_route_field_count": len(validated_fields),
        "audited_gap_route_field_count": len(audited_gap_fields),
        "missing_route_field_count": len(missing_fields),
        "open_route_field_count": len(open_fields),
        "open_gap_followup_tasks": open_gap_followup_tasks,
        "full_text_or_curator_record_refs": curator_record_refs,
        "promotion_blockers": [
            *(
                ["missing_full_text_or_curator_route_field_evidence"]
                if open_fields else []
            ),
            *(
                ["curator_record_requires_manual_final_promotion_review"]
                if validated_fields else []
            ),
            "pubmed_summary_leads_are_not_template_support",
            *(
                ["no_route_relevant_strong_source_selected"]
                if not selected_sources and not validated_fields else []
            ),
        ],
        "self_evo_template_candidate": {
            "schema_version": "statin_closure_self_evo_template_candidate.v1",
            "candidate_id": f"{target_safe}_{requirement_id}_curation_template_candidate",
            "candidate_status": candidate_status,
            "allowed_layer": "candidate_only",
            "promotion_allowed": False,
            "production_write_blocked": True,
            "selected_source_refs": selected_refs,
            "curator_record_refs": curator_record_refs,
            "required_route_fields": required_fields,
            "promotion_blockers": [
                "full_text_or_curator_audit_required_before_template_support",
                *(
                    ["route_field_audit_not_satisfied"]
                    if open_fields else []
                ),
                "manual_promotion_review_required",
            ],
            "not_template_support": True,
            "not_lab_procedure": True,
        },
        "next_action": next_action,
    }


def _normalize_closure_curation_task_lead_sources(
    task: dict[str, Any],
    lead_sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not lead_sources:
        return []
    target_safe = str(task.get("target_safe") or "")
    target_name = str(task.get("target_name") or target_safe)
    target = StatinPanelTarget(
        name=target_name,
        safe=target_safe,
        target_smiles="",
        family_bucket=str(task.get("family_bucket") or ""),
        expected_reaction_class="",
        expected_family_id="",
    )
    requirement_id = str(task.get("requirement_id") or "")
    normalized = []
    for source in lead_sources:
        item = dict(source)
        relevance = _closure_lead_route_relevance(
            source_title=str(item.get("source_title") or ""),
            journal=str(item.get("journal") or ""),
            query=str(item.get("query") or task.get("resolved_query") or task.get("followup_query") or ""),
            abstract_signal_terms=[
                str(term)
                for term in item.get("abstract_signal_terms") or []
                if str(term).strip()
            ],
            target=target,
            requirement_id=requirement_id,
        )
        item.update(relevance)
        normalized.append(item)
    return normalized


def _closure_open_gap_followup_tasks(
    task: dict[str, Any],
    open_fields: list[dict[str, Any]],
    *,
    lead_sources: list[dict[str, Any]],
    selected_source_refs: list[str],
    context_guarded_source_refs: list[str],
    weak_or_rejected_source_refs: list[str],
) -> list[dict[str, Any]]:
    target_safe = str(task.get("target_safe") or "")
    target_name = str(task.get("target_name") or target_safe)
    requirement_id = str(task.get("requirement_id") or "")
    family_bucket = str(task.get("family_bucket") or "")
    priority = str(task.get("priority") or "")
    tasks = []
    for field_row in open_fields:
        field = str(field_row.get("field") or "")
        if not field:
            continue
        followup_id = f"{target_safe}:{requirement_id}:{field}:open_gap_followup"
        followup_query = _closure_open_gap_followup_query(target_name, requirement_id, field, task)
        literature_triage = _closure_open_gap_literature_triage(
            target_name=target_name,
            requirement_id=requirement_id,
            field=field,
            followup_query=followup_query,
            lead_sources=lead_sources,
            selected_source_refs=selected_source_refs,
            context_guarded_source_refs=context_guarded_source_refs,
            weak_or_rejected_source_refs=weak_or_rejected_source_refs,
        )
        tasks.append({
            "schema_version": "statin_closure_open_gap_followup_task.v1",
            "followup_id": followup_id,
            "parent_task_id": task.get("task_id") or f"{target_safe}:{requirement_id}",
            "target_safe": target_safe,
            "target_name": target_name,
            "family_bucket": family_bucket,
            "requirement_id": requirement_id,
            "field": field,
            "priority": priority,
            "status": "queued_for_full_text_or_curator_record",
            "current_field_status": field_row.get("status") or "",
            "current_field_summary": field_row.get("summary") or "",
            "followup_query": followup_query,
            "source_requirement": _closure_open_gap_source_requirement(requirement_id, field),
            "acceptance_signals": _closure_open_gap_acceptance_signals(requirement_id, field),
            "selected_route_source_refs": list(selected_source_refs),
            "context_guarded_source_refs": list(context_guarded_source_refs),
            "weak_or_rejected_source_refs": list(weak_or_rejected_source_refs),
            "source_gate": (
                "review_selected_route_sources_first"
                if selected_source_refs
                else "find_route_specific_full_text_or_curator_source"
            ),
            "template_promotion_allowed": False,
            "solved_claim_allowed": False,
            "production_write_blocked": True,
            "not_template_support": True,
            "not_lab_procedure": True,
            "promotion_guard": "field_specific_full_text_or_curator_record_required_before_template_support",
            "literature_triage": literature_triage,
            "self_evo_inbox_entry": _closure_open_gap_self_evo_inbox_entry(
                followup_id=followup_id,
                target_safe=target_safe,
                target_name=target_name,
                family_bucket=family_bucket,
                requirement_id=requirement_id,
                field=field,
                literature_triage=literature_triage,
            ),
        })
    return tasks


def _closure_open_gap_literature_triage(
    *,
    target_name: str,
    requirement_id: str,
    field: str,
    followup_query: str,
    lead_sources: list[dict[str, Any]],
    selected_source_refs: list[str],
    context_guarded_source_refs: list[str],
    weak_or_rejected_source_refs: list[str],
) -> dict[str, Any]:
    selected_summaries = _closure_open_gap_source_summaries(selected_source_refs, lead_sources)
    guarded_summaries = _closure_open_gap_source_summaries(context_guarded_source_refs, lead_sources)
    weak_summaries = _closure_open_gap_source_summaries(weak_or_rejected_source_refs, lead_sources)
    triage_status = (
        "selected_source_review_ready"
        if selected_summaries
        else "route_specific_search_required"
    )
    query_variants = _closure_open_gap_query_variants(
        target_name=target_name,
        requirement_id=requirement_id,
        field=field,
        followup_query=followup_query,
    )
    curator_review_draft = _closure_open_gap_curator_review_draft(
        target_name=target_name,
        requirement_id=requirement_id,
        field=field,
        triage_status=triage_status,
        selected_source_summaries=selected_summaries,
    )
    search_execution_package = _closure_open_gap_search_execution_package(
        target_name=target_name,
        requirement_id=requirement_id,
        field=field,
        query_variants=query_variants,
        triage_status=triage_status,
    )
    return {
        "schema_version": "statin_closure_open_gap_literature_triage.v1",
        "triage_status": triage_status,
        "query_variants": query_variants,
        "selected_source_summaries": selected_summaries,
        "context_guarded_source_summaries": guarded_summaries,
        "weak_or_rejected_source_summaries": weak_summaries,
        "route_specific_source_required": not bool(selected_summaries),
        "full_text_or_curator_record_required": True,
        "next_search_action": (
            "review selected route-relevant source metadata and extract only field-resolution evidence"
            if selected_summaries
            else "execute query variants or attach a curated route-specific source before field resolution"
        ),
        "curator_resolution_schema": {
            "schema_version": "statin_open_gap_curator_resolution_schema.v1",
            "required_fields": [
                "source_ref",
                "source_kind",
                "field_resolution",
                "evidence_scope",
                "curator_note",
            ],
            "allowed_field_resolution_values": [
                "present",
                "absent",
                "not_stated",
                "not_applicable",
                "still_blocked",
            ],
            "forbidden_fields": [
                "abstract_text",
                "raw_reaction",
                "copied_experimental_procedure",
            ],
        },
        "curator_review_draft": curator_review_draft,
        "search_execution_package": search_execution_package,
        "not_template_support": True,
        "not_lab_procedure": True,
    }


def _closure_open_gap_curator_review_draft(
    *,
    target_name: str,
    requirement_id: str,
    field: str,
    triage_status: str,
    selected_source_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    draft_status = (
        "metadata_ready_pending_full_text_or_curator_confirmation"
        if selected_source_summaries
        else "source_required_before_curator_confirmation"
    )
    source_refs = [
        str(source.get("evidence_ref") or "")
        for source in selected_source_summaries
        if source.get("evidence_ref")
    ]
    full_text_access_package = _closure_open_gap_full_text_access_package(
        selected_source_summaries=selected_source_summaries
    )
    full_text_signal_extraction_package = _closure_open_gap_full_text_signal_extraction_package(
        target_name=target_name,
        requirement_id=requirement_id,
        field=field,
        full_text_access_package=full_text_access_package,
    )
    return {
        "schema_version": "statin_open_gap_curator_review_draft.v1",
        "draft_status": draft_status,
        "target_name": target_name,
        "requirement_id": requirement_id,
        "field": field,
        "source_refs_to_review": source_refs,
        "candidate_field_resolution": "still_blocked",
        "resolution_confidence": "not_resolved_metadata_only",
        "curator_questions": _closure_open_gap_curator_questions(field),
        "evidence_gap_statement": _closure_open_gap_evidence_gap_statement(field, triage_status),
        "full_text_access_package": full_text_access_package,
        "full_text_signal_extraction_package": full_text_signal_extraction_package,
        "field_resolution_candidate": _closure_open_gap_field_resolution_candidate(
            target_name=target_name,
            requirement_id=requirement_id,
            field=field,
            triage_status=triage_status,
            selected_source_summaries=selected_source_summaries,
            full_text_access_package=full_text_access_package,
            full_text_signal_extraction_package=full_text_signal_extraction_package,
        ),
        "required_before_resolution": [
            "full_text_or_curator_source_ref",
            "field_resolution_value",
            "curator_note_without_procedure_text",
        ],
        "forbidden_fields": [
            "abstract_text",
            "raw_reaction",
            "copied_experimental_procedure",
        ],
        "promotion_allowed": False,
        "production_write_blocked": True,
        "not_template_support": True,
        "not_lab_procedure": True,
    }


def _closure_open_gap_full_text_access_package(
    *,
    selected_source_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    probes = [
        _closure_open_gap_queued_full_text_access_probe(source)
        for source in selected_source_summaries
        if isinstance(source, dict)
    ]
    return {
        "schema_version": "statin_open_gap_full_text_access_package.v1",
        "execution_status": "queued_for_full_text_access_probe" if probes else "source_required_before_access_probe",
        "source_ref_count": len(probes),
        "source_refs": [
            str(source.get("evidence_ref") or "")
            for source in selected_source_summaries
            if isinstance(source, dict) and source.get("evidence_ref")
        ],
        "probes": probes,
        "probe_policy": "ncbi_pubmed_to_pmc_access_metadata_only",
        "access_probe_contract": (
            "Probe records only locate PMID/DOI/PMC access metadata. They do not store full text, abstracts, "
            "or experimental procedure text and cannot resolve route fields without curator extraction."
        ),
        "forbidden_fields": [
            "abstract_text",
            "raw_reaction",
            "copied_experimental_procedure",
            "full_text_body",
        ],
        "full_text_content_stored": False,
        "not_template_support": True,
        "not_lab_procedure": True,
    }


def _closure_open_gap_queued_full_text_access_probe(source: dict[str, Any]) -> dict[str, Any]:
    pmid = str(source.get("pmid") or "")
    doi = str(source.get("doi") or "")
    source_url = str(source.get("source_url") or (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""))
    return {
        "schema_version": "statin_open_gap_full_text_access_probe.v1",
        "execution_status": "queued_not_executed",
        "full_text_access_status": "not_probed",
        "source_ref": str(source.get("evidence_ref") or ""),
        "source_type": str(source.get("source_type") or ""),
        "source_title": str(source.get("source_title") or ""),
        "pmid": pmid,
        "doi": doi,
        "source_url": source_url,
        "doi_url": f"https://doi.org/{doi}" if doi else "",
        "pmcids": [],
        "pmc_urls": [],
        "probe_sources": [],
        "full_text_content_stored": False,
        "forbidden_fields": [
            "abstract_text",
            "raw_reaction",
            "copied_experimental_procedure",
            "full_text_body",
        ],
        "not_template_support": True,
        "not_lab_procedure": True,
    }


def _closure_open_gap_full_text_signal_extraction_package(
    *,
    target_name: str,
    requirement_id: str,
    field: str,
    full_text_access_package: dict[str, Any],
) -> dict[str, Any]:
    extractions: list[dict[str, Any]] = []
    for probe in full_text_access_package.get("probes") or []:
        if not isinstance(probe, dict):
            continue
        if probe.get("full_text_access_status") != "pmc_open_access_link_available":
            continue
        for pmcid in probe.get("pmcids") or []:
            clean_pmcid = str(pmcid or "").strip()
            if not clean_pmcid:
                continue
            extractions.append(_closure_open_gap_queued_full_text_signal_extraction(
                target_name=target_name,
                requirement_id=requirement_id,
                field=field,
                probe=probe,
                pmcid=clean_pmcid,
            ))
    return {
        "schema_version": "statin_open_gap_full_text_signal_extraction_package.v1",
        "execution_status": (
            "queued_for_pmc_signal_extraction"
            if extractions
            else "open_access_source_required_before_signal_extraction"
        ),
        "extraction_count": len(extractions),
        "extractions": extractions,
        "extraction_policy": "pmc_xml_field_signal_counts_only",
        "extraction_contract": (
            "PMC signal extraction stores only field/route term matches, section labels, and counts. It does not "
            "store full text, abstracts, copied procedure text, or reaction records."
        ),
        "forbidden_fields": [
            "abstract_text",
            "raw_reaction",
            "copied_experimental_procedure",
            "full_text_body",
            "quoted_full_text",
        ],
        "full_text_content_stored": False,
        "not_template_support": True,
        "not_lab_procedure": True,
    }


def _closure_open_gap_queued_full_text_signal_extraction(
    *,
    target_name: str,
    requirement_id: str,
    field: str,
    probe: dict[str, Any],
    pmcid: str,
) -> dict[str, Any]:
    return {
        "schema_version": "statin_open_gap_full_text_signal_extraction.v1",
        "execution_status": "queued_not_executed",
        "signal_status": "not_extracted",
        "target_name": target_name,
        "requirement_id": requirement_id,
        "field": field,
        "source_ref": str(probe.get("source_ref") or ""),
        "pmid": str(probe.get("pmid") or ""),
        "pmcid": pmcid,
        "pmc_url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/",
        "field_signal_terms": [],
        "route_signal_terms": [],
        "signal_section_labels": [],
        "scanned_section_count": 0,
        "signal_section_count": 0,
        "field_signal_count": 0,
        "route_signal_count": 0,
        "candidate_signal_resolution": "not_extracted",
        "extraction_sources": [],
        "full_text_content_stored": False,
        "forbidden_fields": [
            "abstract_text",
            "raw_reaction",
            "copied_experimental_procedure",
            "full_text_body",
            "quoted_full_text",
        ],
        "not_template_support": True,
        "not_lab_procedure": True,
    }


def _closure_open_gap_field_resolution_candidate(
    *,
    target_name: str,
    requirement_id: str,
    field: str,
    triage_status: str,
    selected_source_summaries: list[dict[str, Any]],
    full_text_access_package: dict[str, Any],
    full_text_signal_extraction_package: dict[str, Any],
) -> dict[str, Any]:
    if selected_source_summaries:
        candidate_status = "access_probe_queued_pending_curator_extraction"
    else:
        candidate_status = "source_required_before_resolution_candidate"
    return {
        "schema_version": "statin_open_gap_field_resolution_candidate.v1",
        "candidate_status": candidate_status,
        "target_name": target_name,
        "requirement_id": requirement_id,
        "field": field,
        "source_gate": triage_status,
        "source_refs_to_review": [
            str(source.get("evidence_ref") or "")
            for source in selected_source_summaries
            if isinstance(source, dict) and source.get("evidence_ref")
        ],
        "candidate_field_resolution": "still_blocked",
        "resolution_confidence": "not_resolved_metadata_only",
        "evidence_scope_required": "field_specific_full_text_or_curator_record",
        "full_text_access_gate": full_text_access_package.get("execution_status") or "",
        "full_text_signal_extraction_gate": full_text_signal_extraction_package.get("execution_status") or "",
        "extracted_signal_resolution_candidate": "not_extracted",
        "accepted_resolution_values": [
            "present",
            "absent",
            "not_stated",
            "not_applicable",
            "still_blocked",
        ],
        "curator_action": "extract a non-procedural field-resolution note from an access-checked source",
        "candidate_patch_status": "waiting_for_field_evidence",
        "promotion_allowed": False,
        "production_write_blocked": True,
        "not_template_support": True,
        "not_lab_procedure": True,
        "forbidden_fields": [
            "abstract_text",
            "raw_reaction",
            "copied_experimental_procedure",
            "full_text_body",
        ],
    }


def _closure_open_gap_curator_questions(field: str) -> list[str]:
    questions = {
        "condition_presence_evidence": [
            "Does a route-specific source explicitly state whether conditions are present, absent, or not stated?",
            "Which source metadata or curator reference supports the condition-state decision?",
        ],
        "workup_or_isolation_presence": [
            "Does a route-specific source explicitly state whether workup or isolation evidence is present, absent, or not stated?",
            "Which source metadata or curator reference supports the workup/isolation-state decision?",
        ],
        "stock_or_source_evidence": [
            "Does the source identify every terminal leaf as stock-backed, precursor-backed, or still blocked?",
            "Does the source separate stock/source availability from template role names?",
        ],
        "counterion_or_characterization_refs": [
            "Does the source explicitly trace acid, lactone, salt, or counterion state for the endpoint?",
            "Is characterization or counterion evidence attached without inferring salt form from the target name?",
        ],
    }
    return questions.get(field, [
        f"Does the source resolve {field} with route-specific evidence?",
        "Which source metadata or curator reference supports the resolution?",
    ])


def _closure_open_gap_evidence_gap_statement(field: str, triage_status: str) -> str:
    prefix = (
        "Selected source metadata is ready for curator review, but no field resolution has been extracted."
        if triage_status == "selected_source_review_ready"
        else "No selected route-specific source is available yet."
    )
    field_notes = {
        "condition_presence_evidence": " Condition presence remains unresolved until full text or curator record confirms present, absent, or not stated.",
        "workup_or_isolation_presence": " Workup/isolation remains unresolved until full text or curator record confirms present, absent, or not stated.",
        "stock_or_source_evidence": " Terminal leaf source/stock status remains unresolved until source-backed leaf closure is reviewed.",
        "counterion_or_characterization_refs": " Endpoint characterization or counterion state remains unresolved until source-backed identity evidence is reviewed.",
    }
    return prefix + field_notes.get(field, f" {field} remains unresolved before promotion.")


def _closure_open_gap_search_execution_package(
    *,
    target_name: str,
    requirement_id: str,
    field: str,
    query_variants: list[str],
    triage_status: str,
) -> dict[str, Any]:
    return {
        "schema_version": "statin_open_gap_search_execution_package.v1",
        "execution_status": "ready_for_pubmed_or_manual_search",
        "target_name": target_name,
        "requirement_id": requirement_id,
        "field": field,
        "query_variants": list(query_variants),
        "recommended_retmax": 3,
        "review_priority": (
            "review_selected_source_first"
            if triage_status == "selected_source_review_ready"
            else "execute_route_specific_query_variants"
        ),
        "source_acceptance_filters": _closure_open_gap_source_acceptance_filters(field),
        "capture_fields": [
            "source_ref",
            "source_title",
            "pmid_or_doi_or_url",
            "field_resolution_value",
            "curator_note_without_procedure_text",
        ],
        "forbidden_fields": [
            "abstract_text",
            "raw_reaction",
            "copied_experimental_procedure",
        ],
        "execution_trace": _closure_open_gap_queued_execution_trace(
            target_name=target_name,
            requirement_id=requirement_id,
            field=field,
            query_variants=query_variants,
        ),
        "not_template_support": True,
        "not_lab_procedure": True,
    }


def _closure_open_gap_queued_execution_trace(
    *,
    target_name: str,
    requirement_id: str,
    field: str,
    query_variants: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "statin_open_gap_search_execution_trace.v1",
        "execution_status": "queued_not_executed",
        "backend_resolved": "",
        "target_name": target_name,
        "requirement_id": requirement_id,
        "field": field,
        "query": query_variants[0] if query_variants else "",
        "query_variants": list(query_variants),
        "query_attempt_count": 0,
        "hit_count": 0,
        "evidence_lead_refs": [],
        "lead_sources": [],
        "selected_route_source_refs": [],
        "context_guarded_source_refs": [],
        "weak_or_rejected_source_refs": [],
        "abstract_signal_status": "",
        "abstract_signal_terms": [],
        "search_sources": [],
        "lead_relevance_gate": "not_executed",
        "not_template_support": True,
        "not_lab_procedure": True,
    }


def _execute_open_gap_search_packages(
    results: list[dict[str, Any]],
    *,
    execute_open_gap_searches: bool,
    open_gap_search_limit: int,
    carried_forward_search_traces: int = 0,
) -> dict[str, Any]:
    followups = [
        followup
        for result in results
        for followup in result.get("open_gap_followup_tasks") or []
        if isinstance(followup, dict)
    ]
    candidates = [
        followup for followup in followups
        if (
            (followup.get("literature_triage") or {}).get("triage_status")
            == "route_specific_search_required"
        )
    ]
    runnable_candidates = [
        followup for followup in candidates
        if not _open_gap_followup_has_executed_trace(followup)
    ]
    requested_limit = int(open_gap_search_limit or 0)
    full_queue_requested = requested_limit < 0
    resolved_limit = (
        len(runnable_candidates)
        if full_queue_requested
        else max(0, min(requested_limit, len(runnable_candidates)))
    )
    executed = 0
    lead_count = 0
    selected_count = 0
    failed = 0
    if execute_open_gap_searches and resolved_limit:
        for followup in runnable_candidates[:resolved_limit]:
            trace = _execute_single_open_gap_search(followup)
            _apply_open_gap_search_trace(followup, trace)
            if str(trace.get("execution_status") or "").startswith("pubmed_open_gap_search_executed"):
                executed += 1
            if int(trace.get("hit_count") or 0) > 0:
                lead_count += 1
            if trace.get("selected_route_source_refs"):
                selected_count += 1
            if trace.get("execution_status") == "pubmed_open_gap_search_failed":
                failed += 1
    return {
        "schema_version": "statin_open_gap_search_execution_summary.v1",
        "requested": bool(execute_open_gap_searches),
        "requested_limit": requested_limit,
        "resolved_limit": resolved_limit,
        "full_queue_requested": full_queue_requested,
        "open_gap_followup_count": len(followups),
        "candidate_search_required_count": len(candidates),
        "runnable_search_required_count": len(runnable_candidates),
        "carried_forward_search_trace_count": int(carried_forward_search_traces),
        "executed_count": executed,
        "lead_count": lead_count,
        "selected_source_count": selected_count,
        "failed_count": failed,
        "policy": "pubmed_open_gap_field_search" if execute_open_gap_searches else "queued_only",
        "execution_contract": (
            "Open-gap search traces are field-level literature leads. They cannot resolve route fields "
            "or promote templates without full-text or curator confirmation."
        ),
    }


def _execute_open_gap_full_text_access_probes(
    results: list[dict[str, Any]],
    *,
    execute_full_text_access_probes: bool,
    full_text_access_probe_limit: int,
    carried_forward_probe_count: int = 0,
) -> dict[str, Any]:
    followups = [
        followup
        for result in results
        for followup in result.get("open_gap_followup_tasks") or []
        if isinstance(followup, dict)
    ]
    review_ready = [
        followup for followup in followups
        if (
            (followup.get("literature_triage") or {}).get("triage_status")
            == "selected_source_review_ready"
        )
    ]
    runnable = [
        followup for followup in review_ready
        if _open_gap_followup_has_queued_access_probe(followup)
    ]
    requested_limit = int(full_text_access_probe_limit or 0)
    full_queue_requested = requested_limit < 0
    resolved_limit = (
        len(runnable)
        if full_queue_requested
        else max(0, min(requested_limit, len(runnable)))
    )
    executed = 0
    open_access = 0
    metadata_candidates = 0
    failed = 0
    if execute_full_text_access_probes and resolved_limit:
        for followup in runnable[:resolved_limit]:
            summary = _execute_single_open_gap_full_text_access_probe_set(followup)
            executed += int(summary.get("executed_probe_count") or 0)
            open_access += int(summary.get("open_access_candidate_count") or 0)
            metadata_candidates += int(summary.get("metadata_candidate_count") or 0)
            failed += int(summary.get("failed_probe_count") or 0)
    return {
        "schema_version": "statin_open_gap_full_text_access_execution_summary.v1",
        "requested": bool(execute_full_text_access_probes),
        "requested_limit": requested_limit,
        "resolved_limit": resolved_limit,
        "full_queue_requested": full_queue_requested,
        "review_ready_followup_count": len(review_ready),
        "runnable_probe_followup_count": len(runnable),
        "carried_forward_probe_count": int(carried_forward_probe_count),
        "executed_probe_count": executed,
        "open_access_candidate_count": open_access,
        "metadata_candidate_count": metadata_candidates,
        "failed_probe_count": failed,
        "policy": (
            "ncbi_pubmed_to_pmc_access_metadata_probe"
            if execute_full_text_access_probes
            else "queued_only"
        ),
        "execution_contract": (
            "Full-text access probes only add PMID/DOI/PMC access metadata. They do not store full text "
            "or resolve route fields without curator extraction."
        ),
    }


def _open_gap_followup_has_queued_access_probe(followup: dict[str, Any]) -> bool:
    package = _open_gap_full_text_access_package(followup)
    return any(
        str(probe.get("execution_status") or "") == "queued_not_executed"
        for probe in package.get("probes") or []
        if isinstance(probe, dict)
    )


def _execute_single_open_gap_full_text_access_probe_set(followup: dict[str, Any]) -> dict[str, int]:
    package = _open_gap_full_text_access_package(followup)
    probes = [probe for probe in package.get("probes") or [] if isinstance(probe, dict)]
    executed = 0
    open_access = 0
    metadata_candidates = 0
    failed = 0
    for index, probe in enumerate(probes):
        if str(probe.get("execution_status") or "") != "queued_not_executed":
            continue
        updated = _execute_single_open_gap_full_text_access_probe(probe)
        probes[index] = updated
        executed += int(str(updated.get("execution_status") or "").startswith("full_text_access_probe_executed"))
        if updated.get("full_text_access_status") == "pmc_open_access_link_available":
            open_access += 1
        elif updated.get("full_text_access_status") in {
            "doi_or_pubmed_access_metadata_available",
            "source_metadata_only_no_access_link",
        }:
            metadata_candidates += 1
        elif updated.get("execution_status") == "full_text_access_probe_failed":
            failed += 1
    package["probes"] = probes
    package["source_ref_count"] = len(probes)
    package["source_refs"] = [
        str(probe.get("source_ref") or "")
        for probe in probes
        if probe.get("source_ref")
    ]
    package["execution_status"] = _full_text_access_package_status(probes)
    _refresh_open_gap_resolution_candidate_from_access_package(followup)
    return {
        "executed_probe_count": executed,
        "open_access_candidate_count": open_access,
        "metadata_candidate_count": metadata_candidates,
        "failed_probe_count": failed,
    }


def _execute_single_open_gap_full_text_access_probe(probe: dict[str, Any]) -> dict[str, Any]:
    updated = dict(probe)
    pmid = str(updated.get("pmid") or "")
    doi = str(updated.get("doi") or "")
    source_url = str(updated.get("source_url") or "")
    try:
        pmcids = _pubmed_pmc_links_for_pmid(pmid) if pmid else []
        updated["execution_status"] = "full_text_access_probe_executed"
        updated["backend_resolved"] = "ncbi_elink_pubmed_to_pmc"
        updated["probe_sources"] = [
            *list(updated.get("probe_sources") or []),
            *("ncbi_elink_pubmed_to_pmc" if pmid else "no_pmid_for_elink",),
        ]
        updated["pmcids"] = pmcids
        updated["pmc_urls"] = [f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/" for pmcid in pmcids]
        if pmcids:
            updated["full_text_access_status"] = "pmc_open_access_link_available"
        elif doi or source_url or pmid:
            updated["full_text_access_status"] = "doi_or_pubmed_access_metadata_available"
        else:
            updated["full_text_access_status"] = "source_metadata_only_no_access_link"
    except Exception as exc:
        updated["execution_status"] = "full_text_access_probe_failed"
        updated["backend_resolved"] = "ncbi_elink_pubmed_to_pmc"
        updated["full_text_access_status"] = "access_probe_failed"
        updated["error_type"] = type(exc).__name__
    updated["full_text_content_stored"] = False
    updated["not_template_support"] = True
    updated["not_lab_procedure"] = True
    return updated


def _pubmed_pmc_links_for_pmid(pmid: str, *, timeout_s: float = 10.0) -> list[str]:
    clean_pmid = str(pmid or "").strip()
    if not clean_pmid:
        return []
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?" + urlencode({
        "dbfrom": "pubmed",
        "db": "pmc",
        "id": clean_pmid,
        "retmode": "json",
        "tool": "AutoPlanner",
    })
    with urlopen(url, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    links: list[str] = []
    for linkset in payload.get("linksets") or []:
        if not isinstance(linkset, dict):
            continue
        for linkset_db in linkset.get("linksetdbs") or []:
            if not isinstance(linkset_db, dict):
                continue
            if linkset_db.get("linkname") != "pubmed_pmc":
                continue
            for link in linkset_db.get("links") or []:
                clean = str(link or "").strip()
                if clean and clean not in links:
                    links.append(clean)
    return links


def _execute_open_gap_full_text_signal_extractions(
    results: list[dict[str, Any]],
    *,
    execute_full_text_signal_extractions: bool,
    full_text_signal_extraction_limit: int,
    carried_forward_signal_count: int = 0,
) -> dict[str, Any]:
    followups = [
        followup
        for result in results
        for followup in result.get("open_gap_followup_tasks") or []
        if isinstance(followup, dict)
    ]
    open_access_followups = [
        followup for followup in followups
        if _open_gap_full_text_signal_extraction_package(followup).get("extractions")
    ]
    runnable = [
        followup for followup in open_access_followups
        if _open_gap_followup_has_queued_signal_extraction(followup)
    ]
    requested_limit = int(full_text_signal_extraction_limit or 0)
    full_queue_requested = requested_limit < 0
    resolved_limit = (
        len(runnable)
        if full_queue_requested
        else max(0, min(requested_limit, len(runnable)))
    )
    executed = 0
    field_signal_candidates = 0
    failed = 0
    if execute_full_text_signal_extractions and resolved_limit:
        for followup in runnable[:resolved_limit]:
            summary = _execute_single_open_gap_full_text_signal_extraction_set(followup)
            executed += int(summary.get("executed_signal_count") or 0)
            field_signal_candidates += int(summary.get("field_signal_candidate_count") or 0)
            failed += int(summary.get("failed_signal_count") or 0)
    return {
        "schema_version": "statin_open_gap_full_text_signal_execution_summary.v1",
        "requested": bool(execute_full_text_signal_extractions),
        "requested_limit": requested_limit,
        "resolved_limit": resolved_limit,
        "full_queue_requested": full_queue_requested,
        "open_access_followup_count": len(open_access_followups),
        "runnable_signal_followup_count": len(runnable),
        "carried_forward_signal_count": int(carried_forward_signal_count),
        "executed_signal_count": executed,
        "field_signal_candidate_count": field_signal_candidates,
        "failed_signal_count": failed,
        "policy": (
            "pmc_xml_field_signal_counts_only"
            if execute_full_text_signal_extractions
            else "queued_only"
        ),
        "execution_contract": (
            "PMC signal extraction records term counts and section labels only. It cannot resolve fields or "
            "promote templates without curator review."
        ),
    }


def _open_gap_followup_has_queued_signal_extraction(followup: dict[str, Any]) -> bool:
    package = _open_gap_full_text_signal_extraction_package(followup)
    return any(
        str(extraction.get("execution_status") or "") == "queued_not_executed"
        for extraction in package.get("extractions") or []
        if isinstance(extraction, dict)
    )


def _execute_single_open_gap_full_text_signal_extraction_set(followup: dict[str, Any]) -> dict[str, int]:
    package = _open_gap_full_text_signal_extraction_package(followup)
    extractions = [row for row in package.get("extractions") or [] if isinstance(row, dict)]
    executed = 0
    field_signal_candidates = 0
    failed = 0
    for index, extraction in enumerate(extractions):
        if str(extraction.get("execution_status") or "") != "queued_not_executed":
            continue
        updated = _execute_single_open_gap_full_text_signal_extraction(extraction)
        extractions[index] = updated
        if str(updated.get("execution_status") or "").startswith("full_text_signal_extraction_executed"):
            executed += 1
        if updated.get("candidate_signal_resolution") in {"present_candidate", "not_stated_candidate"}:
            field_signal_candidates += 1
        if updated.get("execution_status") == "full_text_signal_extraction_failed":
            failed += 1
    package["extractions"] = extractions
    package["extraction_count"] = len(extractions)
    package["execution_status"] = _full_text_signal_extraction_package_status(extractions)
    _refresh_open_gap_resolution_candidate_from_signal_extraction(followup)
    return {
        "executed_signal_count": executed,
        "field_signal_candidate_count": field_signal_candidates,
        "failed_signal_count": failed,
    }


def _execute_single_open_gap_full_text_signal_extraction(extraction: dict[str, Any]) -> dict[str, Any]:
    updated = dict(extraction)
    pmcid = str(updated.get("pmcid") or "")
    field = str(updated.get("field") or "")
    try:
        audit = _pmc_full_text_signal_audit(pmcid, field=field)
        updated.update(audit)
        signal_status = str(audit.get("signal_status") or "")
        updated["execution_status"] = (
            "full_text_signal_extraction_executed_with_field_signal"
            if signal_status == "field_signal_detected"
            else "full_text_signal_extraction_executed_no_field_signal"
        )
        updated["candidate_signal_resolution"] = _candidate_signal_resolution_from_audit(audit)
    except Exception as exc:
        updated["execution_status"] = "full_text_signal_extraction_failed"
        updated["signal_status"] = "signal_extraction_failed"
        updated["error_type"] = type(exc).__name__
        updated["candidate_signal_resolution"] = "still_blocked"
    updated["full_text_content_stored"] = False
    updated["not_template_support"] = True
    updated["not_lab_procedure"] = True
    return updated


def _pmc_full_text_signal_audit(pmcid: str, *, field: str, timeout_s: float = 15.0) -> dict[str, Any]:
    clean_pmcid = str(pmcid or "").removeprefix("PMC").strip()
    if not clean_pmcid:
        return {
            "signal_status": "missing_pmcid",
            "field_signal_terms": [],
            "route_signal_terms": [],
            "signal_section_labels": [],
            "scanned_section_count": 0,
            "signal_section_count": 0,
            "field_signal_count": 0,
            "route_signal_count": 0,
            "extraction_sources": [],
        }
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urlencode({
        "db": "pmc",
        "id": clean_pmcid,
        "retmode": "xml",
        "tool": "AutoPlanner",
    })
    with urlopen(url, timeout=timeout_s) as response:
        xml_text = response.read().decode("utf-8", "replace")
    root = ET.fromstring(xml_text)
    field_terms = _pmc_field_signal_terms(field)
    route_terms = _pmc_route_signal_terms()
    sections = list(root.findall(".//body//sec"))
    if not sections:
        body = root.find(".//body")
        sections = [body] if body is not None else []
    field_hits: set[str] = set()
    route_hits: set[str] = set()
    section_labels: set[str] = set()
    signal_section_count = 0
    for section in sections:
        if section is None:
            continue
        section_text = " ".join(str(text or "") for text in section.itertext()).lower()
        if not section_text:
            continue
        section_field_hits = {term for term in field_terms if term in section_text}
        section_route_hits = {term for term in route_terms if term in section_text}
        if section_field_hits or section_route_hits:
            signal_section_count += 1
            section_labels.add(_pmc_section_label(section))
            field_hits.update(section_field_hits)
            route_hits.update(section_route_hits)
    signal_status = (
        "field_signal_detected"
        if field_hits and route_hits
        else "route_signal_without_field_signal"
        if route_hits
        else "no_route_or_field_signal"
    )
    return {
        "signal_status": signal_status,
        "field_signal_terms": sorted(field_hits),
        "route_signal_terms": sorted(route_hits),
        "signal_section_labels": sorted(section_labels),
        "scanned_section_count": len(sections),
        "signal_section_count": signal_section_count,
        "field_signal_count": len(field_hits),
        "route_signal_count": len(route_hits),
        "source_xml_char_count": len(xml_text),
        "extraction_sources": ["ncbi_pmc_efetch_xml"],
        "full_text_content_stored": False,
    }


def _pmc_field_signal_terms(field: str) -> set[str]:
    by_field = {
        "condition_presence_evidence": {
            "condition",
            "conditions",
            "reaction",
            "temperature",
            "incubation",
            "ph",
            "medium",
            "buffer",
            "catalyst",
            "enzyme",
        },
        "workup_or_isolation_presence": {
            "workup",
            "work-up",
            "isolation",
            "isolated",
            "purification",
            "purified",
            "crystallization",
            "extraction",
            "separation",
            "filtration",
        },
        "stock_or_source_evidence": {
            "commercial",
            "purchased",
            "available",
            "precursor",
            "starting material",
            "fermentation",
            "biosynthesis",
            "source",
        },
        "counterion_or_characterization_refs": {
            "calcium",
            "salt",
            "counterion",
            "characterization",
            "characterized",
            "nmr",
            "hplc",
            "mass spectrometry",
            "lc-ms",
            "spectra",
            "lactone",
        },
    }
    return by_field.get(field, {field.replace("_", " ")})


def _pmc_route_signal_terms() -> set[str]:
    return {
        "synthesis",
        "synthetic",
        "process",
        "route",
        "intermediate",
        "intermediates",
        "statin",
        "side-chain",
        "side chain",
        "lactone",
        "preparation",
        "biotransformation",
        "fermentation",
        "enzyme",
    }


def _pmc_section_label(section: ET.Element) -> str:
    title = " ".join(str(text or "") for text in section.findall("./title")[0].itertext()).lower() if section.findall("./title") else ""
    if any(term in title for term in {"method", "materials", "experimental", "procedure"}):
        return "methods_like"
    if any(term in title for term in {"result", "discussion"}):
        return "results_like"
    if "supplement" in title:
        return "supplement_like"
    if any(term in title for term in {"synthesis", "preparation", "process"}):
        return "synthesis_like"
    return "body_section"


def _candidate_signal_resolution_from_audit(audit: dict[str, Any]) -> str:
    if audit.get("signal_status") == "field_signal_detected":
        return "present_candidate"
    if audit.get("signal_status") == "route_signal_without_field_signal":
        return "not_stated_candidate"
    return "still_blocked"


def _full_text_signal_extraction_package_status(extractions: list[dict[str, Any]]) -> str:
    if not extractions:
        return "open_access_source_required_before_signal_extraction"
    statuses = {str(row.get("execution_status") or "") for row in extractions}
    if any(status == "full_text_signal_extraction_executed_with_field_signal" for status in statuses):
        return "full_text_signal_extraction_executed_with_field_signal"
    if any(status == "full_text_signal_extraction_executed_no_field_signal" for status in statuses):
        return "full_text_signal_extraction_executed_no_field_signal"
    if statuses == {"full_text_signal_extraction_failed"}:
        return "full_text_signal_extraction_failed"
    if "queued_not_executed" in statuses:
        return "queued_for_pmc_signal_extraction"
    return "open_access_source_required_before_signal_extraction"


def _full_text_access_package_status(probes: list[dict[str, Any]]) -> str:
    if not probes:
        return "source_required_before_access_probe"
    statuses = {str(probe.get("full_text_access_status") or "") for probe in probes}
    execution_statuses = {str(probe.get("execution_status") or "") for probe in probes}
    if "pmc_open_access_link_available" in statuses:
        return "full_text_access_probe_executed_with_open_access_candidate"
    if statuses.intersection({"doi_or_pubmed_access_metadata_available", "source_metadata_only_no_access_link"}):
        return "full_text_access_probe_executed_metadata_only"
    if execution_statuses == {"full_text_access_probe_failed"}:
        return "full_text_access_probe_failed"
    if "queued_not_executed" in execution_statuses:
        return "queued_for_full_text_access_probe"
    return "source_required_before_access_probe"


def _refresh_open_gap_resolution_candidate_from_access_package(followup: dict[str, Any]) -> None:
    draft = _open_gap_curator_review_draft(followup)
    if not draft:
        return
    package = draft.get("full_text_access_package") or {}
    candidate = draft.get("field_resolution_candidate") or {}
    probes = [probe for probe in package.get("probes") or [] if isinstance(probe, dict)]
    signal_package = _closure_open_gap_full_text_signal_extraction_package(
        target_name=str(followup.get("target_name") or followup.get("target_safe") or ""),
        requirement_id=str(followup.get("requirement_id") or ""),
        field=str(followup.get("field") or ""),
        full_text_access_package=package,
    )
    previous_signal_package = draft.get("full_text_signal_extraction_package") or {}
    previous_by_key = {
        (
            str(row.get("source_ref") or ""),
            str(row.get("pmcid") or ""),
        ): row
        for row in previous_signal_package.get("extractions") or []
        if isinstance(row, dict)
        and str(row.get("execution_status") or "") in {
            "full_text_signal_extraction_executed_with_field_signal",
            "full_text_signal_extraction_executed_no_field_signal",
            "full_text_signal_extraction_failed",
        }
    }
    merged_extractions = []
    for row in signal_package.get("extractions") or []:
        key = (str(row.get("source_ref") or ""), str(row.get("pmcid") or ""))
        merged_extractions.append(dict(previous_by_key.get(key) or row))
    signal_package["extractions"] = merged_extractions
    signal_package["extraction_count"] = len(merged_extractions)
    signal_package["execution_status"] = _full_text_signal_extraction_package_status(merged_extractions)
    draft["full_text_signal_extraction_package"] = signal_package
    if any(probe.get("full_text_access_status") == "pmc_open_access_link_available" for probe in probes):
        candidate_status = "full_text_access_candidate_ready_for_curator"
    elif any(
        probe.get("full_text_access_status") in {
            "doi_or_pubmed_access_metadata_available",
            "source_metadata_only_no_access_link",
        }
        for probe in probes
    ):
        candidate_status = "doi_or_pubmed_source_ready_for_curator"
    elif probes:
        candidate_status = "access_probe_queued_pending_curator_extraction"
    else:
        candidate_status = "source_required_before_resolution_candidate"
    candidate["candidate_status"] = candidate_status
    candidate["full_text_access_gate"] = package.get("execution_status") or ""
    candidate["full_text_signal_extraction_gate"] = signal_package.get("execution_status") or ""
    candidate["extracted_signal_resolution_candidate"] = _signal_candidate_resolution_from_package(signal_package)
    candidate["candidate_field_resolution"] = "still_blocked"
    candidate["resolution_confidence"] = "not_resolved_metadata_only"
    candidate["promotion_allowed"] = False
    candidate["production_write_blocked"] = True
    candidate["not_template_support"] = True
    candidate["not_lab_procedure"] = True
    draft["field_resolution_candidate"] = candidate
    inbox = followup.get("self_evo_inbox_entry") or {}
    if inbox:
        inbox["full_text_access_gate"] = package.get("execution_status") or ""
        inbox["full_text_signal_extraction_gate"] = signal_package.get("execution_status") or ""
        inbox["field_resolution_candidate_status"] = candidate_status


def _refresh_open_gap_resolution_candidate_from_signal_extraction(followup: dict[str, Any]) -> None:
    draft = _open_gap_curator_review_draft(followup)
    if not draft:
        return
    signal_package = draft.get("full_text_signal_extraction_package") or {}
    candidate = draft.get("field_resolution_candidate") or {}
    extracted = _signal_candidate_resolution_from_package(signal_package)
    if extracted == "present_candidate":
        candidate_status = "full_text_signal_candidate_ready_for_curator"
    elif extracted == "not_stated_candidate":
        candidate_status = "full_text_signal_no_field_signal_ready_for_curator"
    else:
        candidate_status = str(candidate.get("candidate_status") or "full_text_access_candidate_ready_for_curator")
    candidate["candidate_status"] = candidate_status
    candidate["full_text_signal_extraction_gate"] = signal_package.get("execution_status") or ""
    candidate["extracted_signal_resolution_candidate"] = extracted
    candidate["candidate_field_resolution"] = "still_blocked"
    candidate["resolution_confidence"] = "not_resolved_metadata_only"
    candidate["promotion_allowed"] = False
    candidate["production_write_blocked"] = True
    candidate["not_template_support"] = True
    candidate["not_lab_procedure"] = True
    draft["field_resolution_candidate"] = candidate
    inbox = followup.get("self_evo_inbox_entry") or {}
    if inbox:
        inbox["full_text_signal_extraction_gate"] = signal_package.get("execution_status") or ""
        inbox["field_resolution_candidate_status"] = candidate_status


def _signal_candidate_resolution_from_package(signal_package: dict[str, Any]) -> str:
    resolutions = {
        str(row.get("candidate_signal_resolution") or "")
        for row in signal_package.get("extractions") or []
        if isinstance(row, dict)
    }
    if "present_candidate" in resolutions:
        return "present_candidate"
    if "not_stated_candidate" in resolutions:
        return "not_stated_candidate"
    return "still_blocked"


def _carry_forward_open_gap_search_traces(
    results: list[dict[str, Any]],
    previous_result_set: dict[str, Any],
) -> int:
    previous_traces = _previous_open_gap_search_traces(previous_result_set)
    if not previous_traces:
        return 0
    carried = 0
    for result in results:
        for followup in result.get("open_gap_followup_tasks") or []:
            if not isinstance(followup, dict):
                continue
            followup_id = str(followup.get("followup_id") or "")
            trace = previous_traces.get(followup_id)
            if not trace:
                continue
            status = str(trace.get("execution_status") or "")
            if status not in {
                "pubmed_open_gap_search_executed_with_leads",
                "pubmed_open_gap_search_executed_no_hits",
                "pubmed_open_gap_search_failed",
            }:
                continue
            if not _open_gap_search_trace_reusable(trace):
                continue
            _apply_open_gap_search_trace(followup, dict(trace))
            carried += 1
    return carried


def _previous_open_gap_search_traces(result_set: dict[str, Any]) -> dict[str, dict[str, Any]]:
    traces: dict[str, dict[str, Any]] = {}
    for result in result_set.get("results") or []:
        if not isinstance(result, dict):
            continue
        for followup in result.get("open_gap_followup_tasks") or []:
            if not isinstance(followup, dict):
                continue
            followup_id = str(followup.get("followup_id") or "")
            trace = (
                ((followup.get("literature_triage") or {}).get("search_execution_package") or {})
                .get("execution_trace") or {}
            )
            if followup_id and isinstance(trace, dict):
                traces[followup_id] = dict(trace)
    return traces


def _carry_forward_open_gap_full_text_access_probes(
    results: list[dict[str, Any]],
    previous_result_set: dict[str, Any],
) -> int:
    previous_packages = _previous_open_gap_full_text_access_packages(previous_result_set)
    if not previous_packages:
        return 0
    carried_probe_count = 0
    for result in results:
        for followup in result.get("open_gap_followup_tasks") or []:
            if not isinstance(followup, dict):
                continue
            followup_id = str(followup.get("followup_id") or "")
            previous = previous_packages.get(followup_id)
            if not previous:
                continue
            if (followup.get("literature_triage") or {}).get("triage_status") != "selected_source_review_ready":
                continue
            package = dict(previous.get("full_text_access_package") or {})
            if not package:
                continue
            current_package = _open_gap_full_text_access_package(followup)
            current_source_refs = {
                str(ref)
                for ref in current_package.get("source_refs") or []
                if str(ref).strip()
            }
            probes = [
                dict(probe)
                for probe in package.get("probes") or []
                if isinstance(probe, dict)
                and str(probe.get("source_ref") or "") in current_source_refs
                and str(probe.get("execution_status") or "") in {
                    "full_text_access_probe_executed",
                    "full_text_access_probe_failed",
                }
            ]
            if not probes:
                continue
            package["probes"] = probes
            package["source_ref_count"] = len(probes)
            package["execution_status"] = _full_text_access_package_status(probes)
            draft = _open_gap_curator_review_draft(followup)
            if not draft:
                continue
            draft["full_text_access_package"] = package
            previous_candidate = previous.get("field_resolution_candidate") or {}
            if isinstance(previous_candidate, dict) and previous_candidate:
                candidate = dict(previous_candidate)
                candidate["candidate_field_resolution"] = "still_blocked"
                candidate["resolution_confidence"] = "not_resolved_metadata_only"
                candidate["promotion_allowed"] = False
                candidate["production_write_blocked"] = True
                candidate["not_template_support"] = True
                candidate["not_lab_procedure"] = True
                draft["field_resolution_candidate"] = candidate
            _refresh_open_gap_resolution_candidate_from_access_package(followup)
            carried_probe_count += len(probes)
    return carried_probe_count


def _previous_open_gap_full_text_access_packages(result_set: dict[str, Any]) -> dict[str, dict[str, Any]]:
    packages: dict[str, dict[str, Any]] = {}
    for result in result_set.get("results") or []:
        if not isinstance(result, dict):
            continue
        for followup in result.get("open_gap_followup_tasks") or []:
            if not isinstance(followup, dict):
                continue
            followup_id = str(followup.get("followup_id") or "")
            draft = _open_gap_curator_review_draft(followup)
            package = draft.get("full_text_access_package") if draft else None
            if followup_id and isinstance(package, dict):
                packages[followup_id] = {
                    "full_text_access_package": dict(package),
                    "field_resolution_candidate": dict(draft.get("field_resolution_candidate") or {}),
                }
    return packages


def _open_gap_curator_review_draft(followup: dict[str, Any]) -> dict[str, Any]:
    draft = (followup.get("literature_triage") or {}).get("curator_review_draft") or {}
    return draft if isinstance(draft, dict) else {}


def _open_gap_full_text_access_package(followup: dict[str, Any]) -> dict[str, Any]:
    package = _open_gap_curator_review_draft(followup).get("full_text_access_package") or {}
    return package if isinstance(package, dict) else {}


def _open_gap_full_text_access_probe_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        for followup in result.get("open_gap_followup_tasks") or []:
            if not isinstance(followup, dict):
                continue
            package = _open_gap_full_text_access_package(followup)
            probes.extend(
                probe for probe in package.get("probes") or []
                if isinstance(probe, dict)
            )
    return probes


def _open_gap_field_resolution_candidate_count(results: list[dict[str, Any]]) -> int:
    return sum(
        1
        for result in results
        for followup in result.get("open_gap_followup_tasks") or []
        if isinstance(followup, dict)
        and _open_gap_curator_review_draft(followup).get("field_resolution_candidate")
    )


def _open_gap_full_text_access_candidate_count(results: list[dict[str, Any]]) -> int:
    candidate_statuses = {
        "full_text_access_candidate_ready_for_curator",
        "doi_or_pubmed_source_ready_for_curator",
    }
    return sum(
        1
        for result in results
        for followup in result.get("open_gap_followup_tasks") or []
        if isinstance(followup, dict)
        and (
            (_open_gap_curator_review_draft(followup).get("field_resolution_candidate") or {})
            .get("candidate_status")
        ) in candidate_statuses
    )


def _carry_forward_open_gap_full_text_signal_extractions(
    results: list[dict[str, Any]],
    previous_result_set: dict[str, Any],
) -> int:
    previous_packages = _previous_open_gap_full_text_signal_packages(previous_result_set)
    if not previous_packages:
        return 0
    carried = 0
    for result in results:
        for followup in result.get("open_gap_followup_tasks") or []:
            if not isinstance(followup, dict):
                continue
            followup_id = str(followup.get("followup_id") or "")
            previous = previous_packages.get(followup_id)
            if not previous:
                continue
            current_package = _open_gap_full_text_signal_extraction_package(followup)
            current_keys = {
                (
                    str(row.get("source_ref") or ""),
                    str(row.get("pmcid") or ""),
                )
                for row in current_package.get("extractions") or []
                if isinstance(row, dict)
            }
            extracted_rows = [
                dict(row)
                for row in previous.get("extractions") or []
                if isinstance(row, dict)
                and (
                    str(row.get("source_ref") or ""),
                    str(row.get("pmcid") or ""),
                ) in current_keys
                and str(row.get("execution_status") or "") in {
                    "full_text_signal_extraction_executed_with_field_signal",
                    "full_text_signal_extraction_executed_no_field_signal",
                    "full_text_signal_extraction_failed",
                }
            ]
            if not extracted_rows:
                continue
            merged_by_key = {
                (
                    str(row.get("source_ref") or ""),
                    str(row.get("pmcid") or ""),
                ): row
                for row in current_package.get("extractions") or []
                if isinstance(row, dict)
            }
            for row in extracted_rows:
                merged_by_key[(str(row.get("source_ref") or ""), str(row.get("pmcid") or ""))] = row
            merged = list(merged_by_key.values())
            current_package["extractions"] = merged
            current_package["extraction_count"] = len(merged)
            current_package["execution_status"] = _full_text_signal_extraction_package_status(merged)
            draft = _open_gap_curator_review_draft(followup)
            if not draft:
                continue
            draft["full_text_signal_extraction_package"] = current_package
            _refresh_open_gap_resolution_candidate_from_signal_extraction(followup)
            carried += len(extracted_rows)
    return carried


def _previous_open_gap_full_text_signal_packages(result_set: dict[str, Any]) -> dict[str, dict[str, Any]]:
    packages: dict[str, dict[str, Any]] = {}
    for result in result_set.get("results") or []:
        if not isinstance(result, dict):
            continue
        for followup in result.get("open_gap_followup_tasks") or []:
            if not isinstance(followup, dict):
                continue
            followup_id = str(followup.get("followup_id") or "")
            package = _open_gap_full_text_signal_extraction_package(followup)
            if followup_id and package:
                packages[followup_id] = dict(package)
    return packages


def _open_gap_full_text_signal_extraction_package(followup: dict[str, Any]) -> dict[str, Any]:
    package = _open_gap_curator_review_draft(followup).get("full_text_signal_extraction_package") or {}
    return package if isinstance(package, dict) else {}


def _open_gap_full_text_signal_extraction_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        for followup in result.get("open_gap_followup_tasks") or []:
            if not isinstance(followup, dict):
                continue
            package = _open_gap_full_text_signal_extraction_package(followup)
            rows.extend(
                row for row in package.get("extractions") or []
                if isinstance(row, dict)
            )
    return rows


def _open_gap_full_text_signal_candidate_count(results: list[dict[str, Any]]) -> int:
    candidate_statuses = {
        "full_text_signal_candidate_ready_for_curator",
        "full_text_signal_no_field_signal_ready_for_curator",
    }
    return sum(
        1
        for result in results
        for followup in result.get("open_gap_followup_tasks") or []
        if isinstance(followup, dict)
        and (
            (_open_gap_curator_review_draft(followup).get("field_resolution_candidate") or {})
            .get("candidate_status")
        ) in candidate_statuses
    )


def _open_gap_followup_has_executed_trace(followup: dict[str, Any]) -> bool:
    trace = (
        ((followup.get("literature_triage") or {}).get("search_execution_package") or {})
        .get("execution_trace") or {}
    )
    if str(trace.get("execution_status") or "") not in {
        "pubmed_open_gap_search_executed_with_leads",
        "pubmed_open_gap_search_executed_no_hits",
        "pubmed_open_gap_search_failed",
    }:
        return False
    return _open_gap_search_trace_reusable(trace)


def _open_gap_search_trace_reusable(trace: dict[str, Any]) -> bool:
    status = str(trace.get("execution_status") or "")
    if status in {"pubmed_open_gap_search_executed_no_hits", "pubmed_open_gap_search_failed"}:
        return True
    if status != "pubmed_open_gap_search_executed_with_leads":
        return False
    if trace.get("selected_route_source_refs") or trace.get("lead_relevance_gate") == "route_relevant_strong":
        return True
    for source in trace.get("lead_sources") or []:
        if isinstance(source, dict) and source.get("lead_relevance_status") == "route_relevant_strong":
            return True
    return False


def _execute_single_open_gap_search(followup: dict[str, Any]) -> dict[str, Any]:
    triage = followup.get("literature_triage") or {}
    search_package = triage.get("search_execution_package") or {}
    query_variants = [
        str(query)
        for query in search_package.get("query_variants") or []
        if str(query).strip()
    ]
    query = query_variants[0] if query_variants else str(followup.get("followup_query") or "")
    target_safe = str(followup.get("target_safe") or "")
    target_name = str(followup.get("target_name") or target_safe)
    requirement_id = str(followup.get("requirement_id") or "")
    field = str(followup.get("field") or "")
    family_bucket = str(followup.get("family_bucket") or "")
    target_stub = StatinPanelTarget(
        name=target_name,
        safe=target_safe,
        target_smiles="",
        family_bucket=family_bucket,
        expected_reaction_class="",
        expected_family_id="",
    )
    try:
        cards, report, lead_sources = _retrieve_pubmed_query_evidence_until_route_relevant(
            case_id=f"{target_safe}_open_gap_{field}",
            query=query,
            family_hints=[target_safe, target_name, family_bucket, requirement_id, field],
            query_variants=query_variants[1:],
            retmax=max(1, int(search_package.get("recommended_retmax") or 3)),
            include_abstract_signals=True,
            target=target_stub,
            requirement_id=requirement_id,
        )
        evidence_refs = [str(getattr(card, "evidence_id", "") or "") for card in cards if getattr(card, "evidence_id", "")]
        selected_refs = [
            str(source.get("evidence_ref") or "")
            for source in lead_sources
            if source.get("lead_relevance_status") == "route_relevant_strong" and source.get("evidence_ref")
        ]
        guarded_refs = [
            str(source.get("evidence_ref") or "")
            for source in lead_sources
            if source.get("route_context_guard_signals") and source.get("evidence_ref")
        ]
        weak_refs = [
            str(source.get("evidence_ref") or "")
            for source in lead_sources
            if source.get("lead_relevance_status") != "route_relevant_strong" and source.get("evidence_ref")
        ]
        abstract_signal_terms = [
            str(term)
            for term in report.get("abstract_signal_terms") or []
            if str(term).strip()
        ]
        return {
            "schema_version": "statin_open_gap_search_execution_trace.v1",
            "execution_status": (
                "pubmed_open_gap_search_executed_with_leads"
                if evidence_refs
                else "pubmed_open_gap_search_executed_no_hits"
            ),
            "backend_resolved": "pubmed_open_gap_search",
            "target_name": target_name,
            "requirement_id": requirement_id,
            "field": field,
            "query": query,
            "query_variants": list(report.get("query_variants") or query_variants),
            "resolved_query": str(report.get("resolved_query") or ""),
            "query_attempt_count": int(report.get("query_attempt_count") or 0),
            "fallback_used": bool(report.get("fallback_used")),
            "hit_count": int(report.get("hit_count") or 0),
            "evidence_lead_refs": evidence_refs,
            "lead_sources": lead_sources,
            "selected_route_source_refs": selected_refs,
            "context_guarded_source_refs": guarded_refs,
            "weak_or_rejected_source_refs": weak_refs,
            "abstract_signal_audit_requested": bool(report.get("abstract_signal_audit_requested")),
            "abstract_signal_status": str(report.get("abstract_signal_status") or ""),
            "abstract_signal_record_count": int(report.get("abstract_signal_record_count") or 0),
            "abstract_signal_hit_count": int(report.get("abstract_signal_hit_count") or 0),
            "abstract_signal_terms": abstract_signal_terms,
            "search_sources": [
                str(search.get("source") or "")
                for search in report.get("searches") or []
                if search.get("source")
            ],
            "lead_relevance_gate": (
                "route_relevant_strong"
                if selected_refs
                else "lead_metadata_only_or_context_guarded"
                if evidence_refs
                else "no_leads"
            ),
            "not_template_support": True,
            "not_lab_procedure": True,
        }
    except Exception as exc:
        return {
            "schema_version": "statin_open_gap_search_execution_trace.v1",
            "execution_status": "pubmed_open_gap_search_failed",
            "backend_resolved": "pubmed_open_gap_search",
            "target_name": target_name,
            "requirement_id": requirement_id,
            "field": field,
            "query": query,
            "query_variants": query_variants,
            "query_attempt_count": 0,
            "hit_count": 0,
            "evidence_lead_refs": [],
            "lead_sources": [],
            "selected_route_source_refs": [],
            "context_guarded_source_refs": [],
            "weak_or_rejected_source_refs": [],
            "abstract_signal_status": "",
            "abstract_signal_terms": [],
            "search_sources": [],
            "lead_relevance_gate": "search_failed",
            "error_type": type(exc).__name__,
            "not_template_support": True,
            "not_lab_procedure": True,
        }


def _retrieve_pubmed_query_evidence_until_route_relevant(
    *,
    case_id: str,
    query: str,
    family_hints: list[str],
    query_variants: list[str],
    retmax: int,
    include_abstract_signals: bool,
    target: StatinPanelTarget,
    requirement_id: str,
) -> tuple[list[Any], dict[str, Any], list[dict[str, Any]]]:
    """Retry PubMed query variants until at least one route-relevant lead survives guards."""
    attempted_queries = _dedupe_query_texts([query, *query_variants])
    if not attempted_queries:
        attempted_queries = [str(query or "")]
    all_cards: list[Any] = []
    all_sources: list[dict[str, Any]] = []
    seen_cards: set[str] = set()
    seen_sources: set[str] = set()
    reports: list[dict[str, Any]] = []
    selected_report: dict[str, Any] | None = None

    for attempt_index, attempt_query in enumerate(attempted_queries, start=1):
        cards, report = retrieve_pubmed_query_evidence(
            case_id=case_id,
            query=attempt_query,
            family_hints=family_hints,
            query_variants=[],
            retmax=retmax,
            include_abstract_signals=include_abstract_signals,
        )
        lead_sources = _closure_followup_lead_sources(
            cards,
            target=target,
            requirement_id=requirement_id,
        )
        selected_refs = [
            str(source.get("evidence_ref") or "")
            for source in lead_sources
            if source.get("lead_relevance_status") == "route_relevant_strong" and source.get("evidence_ref")
        ]
        report_row = dict(report)
        report_row["route_relevance_attempt"] = attempt_index
        report_row["route_relevance_gate"] = (
            "route_relevant_strong"
            if selected_refs
            else "lead_metadata_only_or_context_guarded"
            if cards
            else "no_leads"
        )
        report_row["route_relevant_source_count"] = len(selected_refs)
        reports.append(report_row)

        for card in cards:
            card_id = str(getattr(card, "evidence_id", "") or "")
            if card_id and card_id in seen_cards:
                continue
            if card_id:
                seen_cards.add(card_id)
            all_cards.append(card)
        for source in lead_sources:
            source_ref = str(source.get("evidence_ref") or source.get("source_record_id") or "")
            if source_ref and source_ref in seen_sources:
                continue
            if source_ref:
                seen_sources.add(source_ref)
            all_sources.append(source)

        if selected_refs:
            selected_report = report_row
            break

    if selected_report is None and reports:
        selected_report = next((row for row in reports if int(row.get("hit_count") or 0) > 0), reports[-1])
    combined = _combine_pubmed_route_relevance_reports(
        case_id=case_id,
        initial_query=attempted_queries[0],
        attempted_queries=attempted_queries,
        reports=reports,
        selected_report=selected_report or {},
        cards=all_cards,
    )
    return all_cards, combined, all_sources


def _combine_pubmed_route_relevance_reports(
    *,
    case_id: str,
    initial_query: str,
    attempted_queries: list[str],
    reports: list[dict[str, Any]],
    selected_report: dict[str, Any],
    cards: list[Any],
) -> dict[str, Any]:
    searches: list[dict[str, Any]] = []
    evidence_levels: dict[str, int] = {}
    abstract_signal_audit: list[dict[str, Any]] = []
    abstract_terms: set[str] = set()
    for report in reports:
        searches.extend([dict(row) for row in report.get("searches") or [] if isinstance(row, dict)])
        searches.append({
            "source": "statin_route_relevance_filter",
            "query": report.get("resolved_query") or report.get("query") or "",
            "hits": int(report.get("hit_count") or 0),
            "route_relevant_source_count": int(report.get("route_relevant_source_count") or 0),
            "route_relevance_gate": report.get("route_relevance_gate") or "",
            "query_strategy": "primary" if int(report.get("route_relevance_attempt") or 1) == 1 else "fallback",
        })
        for key, value in (report.get("evidence_levels") or {}).items():
            evidence_levels[str(key)] = evidence_levels.get(str(key), 0) + int(value or 0)
        for row in report.get("abstract_signal_audit") or []:
            if isinstance(row, dict):
                abstract_signal_audit.append(dict(row))
        abstract_terms.update(
            str(term)
            for term in report.get("abstract_signal_terms") or []
            if str(term).strip()
        )
    resolved_query = str(selected_report.get("resolved_query") or selected_report.get("query") or "")
    if not resolved_query and reports:
        resolved_query = str(reports[-1].get("resolved_query") or reports[-1].get("query") or "")
    return {
        "schema_version": "literature_followup_search_report.v1",
        "case_id": case_id,
        "backend": "pubmed_followup",
        "query": initial_query,
        "query_variants": attempted_queries,
        "query_attempt_count": len(reports),
        "resolved_query": resolved_query,
        "fallback_used": bool(resolved_query and resolved_query != initial_query),
        "searches": searches,
        "hit_count": len(cards),
        "evidence_levels": evidence_levels,
        "unresolved_literature_gap": len(cards) == 0,
        "limitations": [] if cards else ["unresolved_literature_gap"],
        "abstract_signal_audit_requested": any(
            bool(report.get("abstract_signal_audit_requested")) for report in reports
        ),
        "abstract_signal_status": (
            "abstract_route_signal_detected"
            if abstract_terms
            else "abstract_missing_or_no_route_signal"
            if abstract_signal_audit
            else "no_pubmed_hits_for_abstract_signal_audit"
        ),
        "abstract_signal_record_count": len(abstract_signal_audit),
        "abstract_signal_hit_count": sum(1 for item in abstract_signal_audit if item.get("route_signal_terms")),
        "abstract_signal_terms": sorted(abstract_terms),
        "abstract_signal_audit": abstract_signal_audit,
        "route_relevance_retry_policy": "continue_until_route_relevant_strong_or_queries_exhausted",
    }


def _dedupe_query_texts(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        query = " ".join(str(value or "").split())
        key = query.lower()
        if not query or key in seen:
            continue
        seen.add(key)
        out.append(query)
    return out


def _apply_open_gap_search_trace(followup: dict[str, Any], trace: dict[str, Any]) -> None:
    trace = _normalize_open_gap_search_trace_relevance(followup, trace)
    triage = followup.get("literature_triage") or {}
    search_package = triage.get("search_execution_package") or {}
    search_package["execution_trace"] = trace
    lead_sources = [
        source for source in trace.get("lead_sources") or []
        if isinstance(source, dict)
    ]
    selected_summaries = _closure_open_gap_source_summaries(
        list(trace.get("selected_route_source_refs") or []),
        lead_sources,
    )
    guarded_summaries = _closure_open_gap_source_summaries(
        list(trace.get("context_guarded_source_refs") or []),
        lead_sources,
    )
    weak_summaries = _closure_open_gap_source_summaries(
        list(trace.get("weak_or_rejected_source_refs") or []),
        lead_sources,
    )
    triage["selected_source_summaries"] = _merge_open_gap_source_summaries(
        list(triage.get("selected_source_summaries") or []),
        selected_summaries,
    )
    triage["context_guarded_source_summaries"] = _merge_open_gap_source_summaries(
        list(triage.get("context_guarded_source_summaries") or []),
        guarded_summaries,
    )
    triage["weak_or_rejected_source_summaries"] = _merge_open_gap_source_summaries(
        list(triage.get("weak_or_rejected_source_summaries") or []),
        weak_summaries,
    )
    selected = list(triage.get("selected_source_summaries") or [])
    if selected:
        triage["triage_status"] = "selected_source_review_ready"
        triage["route_specific_source_required"] = False
        triage["next_search_action"] = (
            "review selected route-relevant source metadata and extract only field-resolution evidence"
        )
        followup["source_gate"] = "review_selected_route_sources_first"
        followup["selected_route_source_refs"] = _merge_ref_lists(
            list(followup.get("selected_route_source_refs") or []),
            list(trace.get("selected_route_source_refs") or []),
        )
    followup["context_guarded_source_refs"] = _merge_ref_lists(
        list(followup.get("context_guarded_source_refs") or []),
        list(trace.get("context_guarded_source_refs") or []),
    )
    followup["weak_or_rejected_source_refs"] = _merge_ref_lists(
        list(followup.get("weak_or_rejected_source_refs") or []),
        list(trace.get("weak_or_rejected_source_refs") or []),
    )
    field = str(followup.get("field") or "")
    triage["curator_review_draft"] = _closure_open_gap_curator_review_draft(
        target_name=str(followup.get("target_name") or followup.get("target_safe") or ""),
        requirement_id=str(followup.get("requirement_id") or ""),
        field=field,
        triage_status=str(triage.get("triage_status") or ""),
        selected_source_summaries=selected,
    )
    inbox = followup.get("self_evo_inbox_entry") or {}
    if inbox:
        inbox["field_resolution_source_gate"] = triage.get("triage_status") or ""
    _refresh_open_gap_resolution_candidate_from_access_package(followup)


def _normalize_open_gap_search_trace_relevance(
    followup: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(trace)
    lead_sources = [
        dict(source)
        for source in normalized.get("lead_sources") or []
        if isinstance(source, dict)
    ]
    if not lead_sources:
        return normalized
    target_safe = str(followup.get("target_safe") or "")
    target_name = str(followup.get("target_name") or target_safe)
    target = StatinPanelTarget(
        name=target_name,
        safe=target_safe,
        target_smiles="",
        family_bucket=str(followup.get("family_bucket") or ""),
        expected_reaction_class="",
        expected_family_id="",
    )
    requirement_id = str(followup.get("requirement_id") or "")
    recalculated_sources = []
    for source in lead_sources:
        relevance = _closure_lead_route_relevance(
            source_title=str(source.get("source_title") or ""),
            journal=str(source.get("journal") or ""),
            query=str(source.get("query") or normalized.get("resolved_query") or normalized.get("query") or ""),
            abstract_signal_terms=[
                str(term)
                for term in source.get("abstract_signal_terms") or []
                if str(term).strip()
            ],
            target=target,
            requirement_id=requirement_id,
        )
        source.update(relevance)
        recalculated_sources.append(source)
    selected_refs = [
        str(source.get("evidence_ref") or "")
        for source in recalculated_sources
        if source.get("lead_relevance_status") == "route_relevant_strong" and source.get("evidence_ref")
    ]
    guarded_refs = [
        str(source.get("evidence_ref") or "")
        for source in recalculated_sources
        if source.get("route_context_guard_signals") and source.get("evidence_ref")
    ]
    weak_refs = [
        str(source.get("evidence_ref") or "")
        for source in recalculated_sources
        if source.get("lead_relevance_status") != "route_relevant_strong" and source.get("evidence_ref")
    ]
    normalized["lead_sources"] = recalculated_sources
    normalized["selected_route_source_refs"] = selected_refs
    normalized["context_guarded_source_refs"] = guarded_refs
    normalized["weak_or_rejected_source_refs"] = weak_refs
    normalized["lead_relevance_gate"] = (
        "route_relevant_strong"
        if selected_refs
        else "lead_metadata_only_or_context_guarded"
        if normalized.get("evidence_lead_refs")
        else "no_leads"
    )
    return normalized


def _merge_open_gap_source_summaries(
    existing: list[dict[str, Any]],
    new_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*existing, *new_items]:
        if not isinstance(item, dict):
            continue
        key = str(item.get("evidence_ref") or item.get("source_record_id") or item.get("source_url") or "")
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        merged.append(dict(item))
    return merged


def _merge_ref_lists(existing: list[str], new_items: list[str]) -> list[str]:
    merged: list[str] = []
    for item in [*existing, *new_items]:
        ref = str(item or "").strip()
        if ref and ref not in merged:
            merged.append(ref)
    return merged


def _closure_open_gap_source_acceptance_filters(field: str) -> list[str]:
    common = [
        "source must be route-specific for the target, intermediate, endpoint, or terminal leaf",
        "source metadata alone cannot resolve the field without full text or curator confirmation",
        "clinical, pharmacokinetic, animal, or general review context is rejected for route-field resolution unless it contains route-specific source evidence",
    ]
    field_filters = {
        "condition_presence_evidence": [
            "source must support condition-state extraction as present, absent, or not stated",
        ],
        "workup_or_isolation_presence": [
            "source must support workup/isolation-state extraction as present, absent, or not stated",
        ],
        "stock_or_source_evidence": [
            "source must support terminal leaf source or precursor audit, not only target activity context",
        ],
        "counterion_or_characterization_refs": [
            "source must support endpoint form, characterization, salt, acid, lactone, or counterion audit",
        ],
    }
    return [*common, *field_filters.get(field, [])]


def _closure_open_gap_source_summaries(
    refs: list[str],
    lead_sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_ref = {
        str(source.get("evidence_ref") or ""): source
        for source in lead_sources
        if source.get("evidence_ref")
    }
    summaries = []
    for ref in refs:
        source = by_ref.get(str(ref) or "")
        if not source:
            continue
        summary = {
            "evidence_ref": str(ref),
            "source_type": source.get("source_type") or "",
            "source_record_id": source.get("source_record_id") or "",
            "pmid": source.get("pmid") or "",
            "doi": source.get("doi") or "",
            "source_title": source.get("source_title") or "",
            "source_url": source.get("source_url") or "",
            "journal": source.get("journal") or "",
            "pubdate": source.get("pubdate") or "",
            "lead_relevance_status": source.get("lead_relevance_status") or "",
            "route_relevance_score": source.get("route_relevance_score"),
            "route_relevance_strong_signals": list(source.get("route_relevance_strong_signals") or []),
            "route_context_guard_signals": list(source.get("route_context_guard_signals") or []),
            "abstract_signal_status": source.get("abstract_signal_status") or "",
            "abstract_signal_terms": list(source.get("abstract_signal_terms") or []),
            "not_template_support": True,
            "not_lab_procedure": True,
        }
        summaries.append({
            key: value
            for key, value in summary.items()
            if value != "" and value is not None and value != []
        })
    return summaries


def _closure_open_gap_query_variants(
    *,
    target_name: str,
    requirement_id: str,
    field: str,
    followup_query: str,
) -> list[str]:
    target = target_name.strip()
    field_terms = {
        "condition_presence_evidence": [
            "process chemistry synthesis conditions intermediate",
            "experimental route intermediate conditions",
            "process route intermediate full text",
        ],
        "workup_or_isolation_presence": [
            "process chemistry workup isolation intermediate",
            "synthetic route intermediate isolation",
            "experimental route workup full text",
        ],
        "stock_or_source_evidence": [
            "process chemistry starting material source precursor",
            "synthetic route precursor availability intermediate",
            "terminal leaf stock source audit process",
        ],
        "counterion_or_characterization_refs": [
            "process chemistry endpoint characterization salt",
            "synthetic route acid lactone salt form",
            "counterion stereochemistry characterization process",
        ],
    }
    variants = [followup_query]
    for terms in field_terms.get(field, [field.replace("_", " ")]):
        variants.append(f"{target} {terms}".strip())
    variants.append(f"{target} {requirement_id} {field}".replace("_", " ").strip())
    deduped: list[str] = []
    for query in variants:
        clean = " ".join(str(query or "").split())
        if clean and clean not in deduped:
            deduped.append(clean)
    return deduped


def _closure_open_gap_self_evo_inbox_entry(
    *,
    followup_id: str,
    target_safe: str,
    target_name: str,
    family_bucket: str,
    requirement_id: str,
    field: str,
    literature_triage: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "statin_closure_open_gap_self_evo_inbox_entry.v1",
        "inbox_id": f"{followup_id}:self_evo_inbox",
        "target_safe": target_safe,
        "target_name": target_name,
        "family_bucket": family_bucket,
        "requirement_id": requirement_id,
        "field": field,
        "candidate_update_kind": "route_field_patch_candidate",
        "candidate_patch_status": "waiting_for_field_evidence",
        "allowed_layer": "candidate_only",
        "target_template_scope": "member_route_template_candidate",
        "field_resolution_source_gate": literature_triage.get("triage_status") or "",
        "required_evidence_gate": "full_text_or_curator_field_resolution",
        "accepted_resolution_values": [
            "present",
            "absent",
            "not_stated",
            "not_applicable",
            "still_blocked",
        ],
        "promotion_allowed": False,
        "production_write_blocked": True,
        "not_template_support": True,
        "not_lab_procedure": True,
        "promotion_blockers": [
            "field_specific_full_text_or_curator_record_required",
            "manual_promotion_review_required",
            "no_procedure_text_allowed_in_template_patch",
        ],
    }


def _closure_open_gap_followup_query(
    target_name: str,
    requirement_id: str,
    field: str,
    task: dict[str, Any],
) -> str:
    target = target_name.strip() or str(task.get("target_safe") or "").strip()
    base_query = str(task.get("followup_query") or "").strip()
    field_terms = {
        "condition_presence_evidence": "synthesis route conditions experimental intermediate full text",
        "workup_or_isolation_presence": "synthesis workup isolation intermediate full text",
        "stock_or_source_evidence": "starting material source precursor commercial fermentation semisynthesis",
        "counterion_or_characterization_refs": "endpoint characterization salt acid lactone counterion stereochemistry",
    }
    terms = field_terms.get(field, field.replace("_", " "))
    if base_query:
        return f"{base_query} {terms}".strip()
    if target:
        return f"{target} {terms}".strip()
    return f"{requirement_id} {terms}".strip()


def _closure_open_gap_source_requirement(requirement_id: str, field: str) -> str:
    if field == "condition_presence_evidence":
        return (
            "full text or curated route record must explicitly state condition evidence is present, absent, "
            "or not stated for the member route"
        )
    if field == "workup_or_isolation_presence":
        return (
            "full text or curated route record must explicitly state workup or isolation evidence is present, "
            "absent, or not stated for the member route"
        )
    if field == "stock_or_source_evidence":
        return (
            "route-specific source must identify each terminal leaf as stock/source-backed, precursor-backed, "
            "fermentation/semisynthesis-anchored, or still blocked"
        )
    if field == "counterion_or_characterization_refs":
        return (
            "endpoint source must trace characterization and acid/lactone/salt/counterion state without "
            "inferring missing salt form from the target name alone"
        )
    return f"full text or curated route record must resolve {requirement_id}:{field}"


def _closure_open_gap_acceptance_signals(requirement_id: str, field: str) -> list[str]:
    field_signals = {
        "condition_presence_evidence": [
            "source has a route-specific experimental or curator section for the target/intermediate",
            "condition evidence is recorded as present, absent, or not stated without copying a procedure",
            "condition state is linked to source metadata or a local curator record",
        ],
        "workup_or_isolation_presence": [
            "source has route-specific workup or isolation context for the target/intermediate",
            "workup/isolation evidence is recorded as present, absent, or not stated without copying a procedure",
            "isolation state is linked to source metadata or a local curator record",
        ],
        "stock_or_source_evidence": [
            "terminal leaf identity maps to a documented stock, precursor, fermentation, or semisynthesis anchor",
            "advanced/product-like leaf closure remains blocked when source proof is absent",
            "source audit distinguishes commercial/source availability from route-template role names",
        ],
        "counterion_or_characterization_refs": [
            "endpoint form is traced as acid, lactone, salt, or unresolved from source evidence",
            "counterion or characterization reference is attached when applicable",
            "source audit does not infer a salt/counterion state from the target name alone",
        ],
    }
    return field_signals.get(field, [
        f"{requirement_id}:{field} is resolved by source metadata or a local curator record",
        "resolution is stored as field evidence, not as a template-promoted reaction procedure",
    ])


def _local_curator_route_field_evidence(
    task: dict[str, Any],
    required_fields: list[str],
) -> dict[str, dict[str, Any]]:
    dossier_ref = str(((task.get("source_matrix_row") or {}).get("dossier_ref")) or "")
    dossier = _read_json(dossier_ref)
    if not dossier:
        return {}
    target = dossier.get("target") or {}
    blueprint = dossier.get("fullflow_blueprint") or {}
    route_template = dossier.get("route_template") or {}
    stages = [stage for stage in dossier.get("synthesis_stages") or [] if isinstance(stage, dict)]
    template_steps = [step for step in route_template.get("template_steps") or [] if isinstance(step, dict)]
    local_evidence_refs = _local_curator_evidence_refs(dossier)
    if not local_evidence_refs:
        return {}

    requirement_id = str(task.get("requirement_id") or "")
    target_safe = str(task.get("target_safe") or target.get("safe") or "")
    base_ref = f"local_curator:{target_safe}:{requirement_id}"
    field_map: dict[str, dict[str, Any]] = {}

    def add(
        field: str,
        summary: str,
        refs: list[str] | None = None,
        *,
        status: str = "validated_local_curator_record",
        resolution_required: bool = False,
    ) -> None:
        if field not in set(required_fields):
            return
        clean_summary = " ".join(str(summary or "").split())
        if not clean_summary:
            return
        field_map[field] = {
            "status": status,
            "evidence_refs": sorted(set(refs or local_evidence_refs)),
            "curator_record_refs": [f"{base_ref}:{field}"],
            "summary": clean_summary,
            "resolution_required_before_promotion": bool(resolution_required),
        }

    stage_ids = [str(stage.get("stage_id") or "") for stage in stages if stage.get("stage_id")]
    step_ids = [str(step.get("step_id") or "") for step in template_steps if step.get("step_id")]
    intermediate_roles = [
        str(role)
        for role in blueprint.get("key_intermediate_roles") or []
        if str(role).strip()
    ]
    endpoint_audit = [
        str(item)
        for item in blueprint.get("endpoint_audit") or []
        if str(item).strip()
    ]
    route_outline = [
        str(item)
        for item in blueprint.get("member_specific_route_outline") or []
        if str(item).strip()
    ]
    template_sources = [
        str(source)
        for source in route_template.get("template_sources") or blueprint.get("primary_template_sources") or []
        if str(source).strip()
    ]
    rejection_rules = sorted({
        str(rule)
        for step in template_steps
        for rule in step.get("rejection_rules") or []
        if str(rule).strip()
    })
    family_bucket = str(task.get("family_bucket") or target.get("family_bucket") or "")

    if requirement_id == "full_text_route_step_audit":
        add("route_stage_ids", "curated fullflow stage ids: " + ", ".join(stage_ids))
        add("intermediate_identity_refs", "curated intermediate roles: " + "; ".join(intermediate_roles))
        add("step_order_evidence", "curated template step order: " + " -> ".join(step_ids))
        add("endpoint_mapping", "curated endpoint audit guards: " + "; ".join(endpoint_audit))
    elif requirement_id == "terminal_stock_or_source_audit":
        if family_bucket == "natural_statin":
            add("route_leaf_identity", "natural-statin route leaves map to fermentation/semisynthesis anchors and finishing stages")
            add("stock_or_source_evidence", "source-supported semisynthesis endpoint recorded for natural statin core")
            add("fermentation_or_semisynthesis_anchor", "fermentation-derived natural statin core or semisynthesis anchor is explicit")
            add("advanced_leaf_flags", "advanced/product-like stock closure remains guarded by semisynthesis source policy")
        else:
            add("route_leaf_identity", "synthetic-statin route leaves are core fragment and side-chain fragment roles: " + "; ".join(intermediate_roles))
            add(
                "stock_or_source_evidence",
                "local curator records identify fragment roles but do not prove stock/source availability for every terminal leaf",
                status="audited_gap_local_curator_record",
                resolution_required=True,
            )
            add(
                "fermentation_or_semisynthesis_anchor",
                "fermentation or semisynthesis anchor is not applicable to synthetic-statin convergent route family",
                status="not_applicable_local_curator_record",
            )
            add("advanced_leaf_flags", "product-like terminal closure remains rejected by route-template guards: " + "; ".join(rejection_rules))
    elif requirement_id in {
        "route_graph_leaf_closure_audit",
        "hazard_regulatory_and_withdrawn_context_audit",
        "withdrawn_drug_context_guard",
    }:
        add("risk_context", "curated risk guards: " + "; ".join(rejection_rules or endpoint_audit))
        add("route_relevance", "curated route relevance uses template sources: " + ", ".join(template_sources))
        add("remaining_blockers", "blocker remains explicit: " + str(task.get("blocker") or requirement_id))
        note = "curator record is route-planning metadata only and does not permit solved status"
        if requirement_id == "withdrawn_drug_context_guard":
            note = "cerivastatin withdrawn-drug context remains separated from route-planning evidence"
        add("curator_notes", note)
    elif requirement_id == "endpoint_identity_and_salt_state_audit":
        add("endpoint_form", "endpoint identity tracked at target/profile level with acid/lactone/salt finishing kept as audit stage")
        add("stereochemistry_evidence", "target stereochemistry is tracked by target SMILES and fullflow endpoint guards")
        add("acid_lactone_salt_state", "acid/lactone/salt state is explicitly represented as a finishing-and-audit route stage: " + "; ".join(route_outline))
        add(
            "counterion_or_characterization_refs",
            "local curator records require counterion or characterization evidence before solved status; no verified counterion record is attached",
            status="audited_gap_local_curator_record",
            resolution_required=True,
        )
    elif requirement_id == "condition_and_workup_evidence_audit":
        add(
            "condition_presence_evidence",
            "local curator records mark member route conditions as evidence-gated and not inferable from templates alone",
            status="audited_gap_local_curator_record",
            resolution_required=True,
        )
        add(
            "workup_or_isolation_presence",
            "local curator records mark workup/isolation as not stated in the fullflow template and requiring full-text or curator extraction",
            status="audited_gap_local_curator_record",
            resolution_required=True,
        )
        add("compatibility_notes", "curated process and compatibility risks: " + "; ".join(rejection_rules or endpoint_audit))
        add("missing_condition_flags", "conditions/workup/isolation remain explicit open flags before any promotion review")

    return field_map


def _local_curator_evidence_refs(dossier: dict[str, Any]) -> list[str]:
    refs = {
        str(ref)
        for ref in dossier.get("evidence_refs") or []
        if str(ref).strip() and not str(ref).startswith("ev_pubmed_")
    }
    route_template = dossier.get("route_template") or {}
    refs.update(
        str(ref)
        for ref in route_template.get("evidence_refs") or []
        if str(ref).strip() and not str(ref).startswith("ev_pubmed_")
    )
    for step in route_template.get("template_steps") or []:
        if not isinstance(step, dict):
            continue
        refs.update(
            str(ref)
            for ref in step.get("evidence_refs") or []
            if str(ref).strip() and not str(ref).startswith("ev_pubmed_")
        )
    return sorted(refs)


def _closure_curation_result_set_summary(result_set: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": result_set.get("schema_version"),
        "skipped": bool(result_set.get("skipped")),
        "skip_reason": result_set.get("skip_reason") or "",
        "target_count": int(result_set.get("target_count") or 0),
        "result_count": int(result_set.get("result_count") or 0),
        "lead_backed_result_count": int(result_set.get("lead_backed_result_count") or 0),
        "route_relevant_result_count": int(result_set.get("route_relevant_result_count") or 0),
        "context_guarded_result_count": int(result_set.get("context_guarded_result_count") or 0),
        "needs_better_lead_count": int(result_set.get("needs_better_lead_count") or 0),
        "curator_record_supported_result_count": int(result_set.get("curator_record_supported_result_count") or 0),
        "validated_route_field_count": int(result_set.get("validated_route_field_count") or 0),
        "audited_gap_route_field_count": int(result_set.get("audited_gap_route_field_count") or 0),
        "missing_route_field_count": int(result_set.get("missing_route_field_count") or 0),
        "open_route_field_count": int(result_set.get("open_route_field_count") or 0),
        "open_gap_followup_count": int(result_set.get("open_gap_followup_count") or 0),
        "open_gap_review_ready_count": int(result_set.get("open_gap_review_ready_count") or 0),
        "open_gap_search_required_count": int(result_set.get("open_gap_search_required_count") or 0),
        "open_gap_curator_review_draft_count": int(result_set.get("open_gap_curator_review_draft_count") or 0),
        "open_gap_selected_source_review_draft_count": int(result_set.get("open_gap_selected_source_review_draft_count") or 0),
        "open_gap_search_execution_package_count": int(result_set.get("open_gap_search_execution_package_count") or 0),
        "open_gap_search_trace_count": int(result_set.get("open_gap_search_trace_count") or 0),
        "open_gap_search_executed_count": int(result_set.get("open_gap_search_executed_count") or 0),
        "open_gap_search_lead_count": int(result_set.get("open_gap_search_lead_count") or 0),
        "open_gap_search_selected_source_count": int(result_set.get("open_gap_search_selected_source_count") or 0),
        "open_gap_carried_forward_search_trace_count": int(result_set.get("open_gap_carried_forward_search_trace_count") or 0),
        "open_gap_full_text_access_probe_count": int(result_set.get("open_gap_full_text_access_probe_count") or 0),
        "open_gap_full_text_access_executed_count": int(result_set.get("open_gap_full_text_access_executed_count") or 0),
        "open_gap_full_text_access_candidate_count": int(result_set.get("open_gap_full_text_access_candidate_count") or 0),
        "open_gap_carried_forward_full_text_access_probe_count": int(
            result_set.get("open_gap_carried_forward_full_text_access_probe_count") or 0
        ),
        "open_gap_resolution_candidate_count": int(result_set.get("open_gap_resolution_candidate_count") or 0),
        "open_gap_full_text_access_execution": dict(result_set.get("open_gap_full_text_access_execution") or {}),
        "open_gap_self_evo_inbox_count": int(result_set.get("open_gap_self_evo_inbox_count") or 0),
        "full_text_extraction_required_count": int(result_set.get("full_text_extraction_required_count") or 0),
        "blocked_result_count": int(result_set.get("blocked_result_count") or 0),
        "template_promotion_allowed_count": int(result_set.get("template_promotion_allowed_count") or 0),
        "candidate_template_gate_status": result_set.get("candidate_template_gate_status") or "",
        "json": result_set.get("json") or "",
        "markdown": result_set.get("markdown") or "",
        "validation": dict(result_set.get("validation") or {}),
    }


def _render_closure_curation_result_set_md(result_set: dict[str, Any]) -> str:
    validation = result_set.get("validation") or {}
    lines = [
        "# 九他汀 Closure Curation Result Set",
        "",
        f"- skipped: `{bool(result_set.get('skipped'))}`",
        f"- target_count: `{result_set.get('target_count') or 0}`",
        f"- result_count: `{result_set.get('result_count') or 0}`",
        f"- route_relevant_result_count: `{result_set.get('route_relevant_result_count') or 0}`",
        f"- context_guarded_result_count: `{result_set.get('context_guarded_result_count') or 0}`",
        f"- curator_record_supported_result_count: `{result_set.get('curator_record_supported_result_count') or 0}`",
        f"- validated_route_field_count: `{result_set.get('validated_route_field_count') or 0}`",
        f"- audited_gap_route_field_count: `{result_set.get('audited_gap_route_field_count') or 0}`",
        f"- missing_route_field_count: `{result_set.get('missing_route_field_count') or 0}`",
        f"- open_route_field_count: `{result_set.get('open_route_field_count') or 0}`",
        f"- open_gap_followup_count: `{result_set.get('open_gap_followup_count') or 0}`",
        f"- open_gap_review_ready_count: `{result_set.get('open_gap_review_ready_count') or 0}`",
        f"- open_gap_search_required_count: `{result_set.get('open_gap_search_required_count') or 0}`",
        f"- open_gap_curator_review_draft_count: `{result_set.get('open_gap_curator_review_draft_count') or 0}`",
        f"- open_gap_selected_source_review_draft_count: `{result_set.get('open_gap_selected_source_review_draft_count') or 0}`",
        f"- open_gap_search_execution_package_count: `{result_set.get('open_gap_search_execution_package_count') or 0}`",
        f"- open_gap_search_trace_count: `{result_set.get('open_gap_search_trace_count') or 0}`",
        f"- open_gap_search_executed_count: `{result_set.get('open_gap_search_executed_count') or 0}`",
        f"- open_gap_search_lead_count: `{result_set.get('open_gap_search_lead_count') or 0}`",
        f"- open_gap_search_selected_source_count: `{result_set.get('open_gap_search_selected_source_count') or 0}`",
        f"- open_gap_carried_forward_search_trace_count: `{result_set.get('open_gap_carried_forward_search_trace_count') or 0}`",
        f"- open_gap_full_text_access_probe_count: `{result_set.get('open_gap_full_text_access_probe_count') or 0}`",
        f"- open_gap_full_text_access_executed_count: `{result_set.get('open_gap_full_text_access_executed_count') or 0}`",
        f"- open_gap_full_text_access_candidate_count: `{result_set.get('open_gap_full_text_access_candidate_count') or 0}`",
        f"- open_gap_carried_forward_full_text_access_probe_count: `{result_set.get('open_gap_carried_forward_full_text_access_probe_count') or 0}`",
        f"- open_gap_resolution_candidate_count: `{result_set.get('open_gap_resolution_candidate_count') or 0}`",
        f"- open_gap_self_evo_inbox_count: `{result_set.get('open_gap_self_evo_inbox_count') or 0}`",
        f"- full_text_extraction_required_count: `{result_set.get('full_text_extraction_required_count') or 0}`",
        f"- template_promotion_allowed_count: `{result_set.get('template_promotion_allowed_count') or 0}`",
        f"- candidate_template_gate_status: `{result_set.get('candidate_template_gate_status') or ''}`",
        f"- validation_accepted: `{bool(validation.get('accepted'))}`",
    ]
    if result_set.get("skip_reason"):
        lines.append(f"- skip_reason: `{result_set.get('skip_reason')}`")
    lines.extend([
        "",
        "## Results",
        "",
        "| task | status | selected route sources | curated fields | audited gaps | open fields | follow-ups | gate | next action |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ])
    for result in result_set.get("results") or []:
        selection = result.get("source_selection_summary") or {}
        lines.append(
            "| {task} | {status} | {sources} | {curated} | {gaps} | {open_fields} | {followups} | {gate} | {action} |".format(
                task=result.get("task_id") or "",
                status=result.get("curation_result_status") or "",
                sources=selection.get("selected_route_source_count") or 0,
                curated=result.get("validated_route_field_count") or 0,
                gaps=result.get("audited_gap_route_field_count") or 0,
                open_fields=result.get("open_route_field_count") or 0,
                followups=len(result.get("open_gap_followup_tasks") or []),
                gate=result.get("candidate_template_gate_status") or "",
                action=result.get("next_action") or "",
            )
        )
    if validation.get("reasons"):
        lines.extend(["", "## Validation Reasons"])
        for reason in validation.get("reasons") or []:
            lines.append(f"- `{reason}`")
    lines.append("")
    return "\n".join(lines)


def _render_closure_lead_curation_packet_md(packet: dict[str, Any]) -> str:
    validation = packet.get("validation") or {}
    lines = [
        "# 九他汀 Closure Lead Curation Packet",
        "",
        f"- skipped: `{bool(packet.get('skipped'))}`",
        f"- target_count: `{packet.get('target_count') or 0}`",
        f"- task_count: `{packet.get('task_count') or 0}`",
        f"- lead_backed_task_count: `{packet.get('lead_backed_task_count') or 0}`",
        f"- source_metadata_task_count: `{packet.get('source_metadata_task_count') or 0}`",
        f"- fully_traceable_task_count: `{packet.get('fully_traceable_task_count') or 0}`",
        f"- route_relevant_task_count: `{packet.get('route_relevant_task_count') or 0}`",
        f"- route_context_guarded_task_count: `{packet.get('route_context_guarded_task_count') or 0}`",
        f"- abstract_signal_task_count: `{packet.get('abstract_signal_task_count') or 0}`",
        f"- full_execution_coverage: `{bool(packet.get('full_execution_coverage'))}`",
        f"- validation_accepted: `{bool(validation.get('accepted'))}`",
    ]
    if packet.get("skip_reason"):
        lines.append(f"- skip_reason: `{packet.get('skip_reason')}`")
    lines.extend([
        "",
        "## Tasks",
        "",
        "| task | priority | leads | sources | route sources | abstract signal | next action |",
        "|---|---|---:|---:|---:|---|---|",
    ])
    for task in packet.get("tasks") or []:
        lines.append(
            "| {task} | {priority} | {leads} | {sources} | {route_sources} | {signal} | {action} |".format(
                task=task.get("task_id") or "",
                priority=task.get("priority") or "",
                leads=len(task.get("evidence_lead_refs") or []),
                sources=len(task.get("lead_sources") or []),
                route_sources=task.get("route_relevant_source_count") or 0,
                signal=task.get("abstract_signal_status") or "",
                action=task.get("next_action") or "",
            )
        )
    if validation.get("reasons"):
        lines.extend(["", "## Validation Reasons"])
        for reason in validation.get("reasons") or []:
            lines.append(f"- `{reason}`")
    lines.append("")
    return "\n".join(lines)


def _render_fullflow_overview_md(overview: dict[str, Any]) -> str:
    validation = overview.get("validation") or {}
    lines = [
        "# 九他汀全流程合成模板总览",
        "",
        f"- skipped: `{bool(overview.get('skipped'))}`",
        f"- target_count: `{overview.get('target_count') or 0}`",
        f"- validation_accepted: `{bool(validation.get('accepted'))}`",
        f"- contract: {overview.get('status_contract') or 'subset replay overview skipped'}",
    ]
    if overview.get("skip_reason"):
        lines.append(f"- skip_reason: `{overview.get('skip_reason')}`")
    lines.extend([
        "",
        "## Targets",
        "",
        "| target | family | status | stages | difficulty queries | self-evo | dossier |",
        "|---|---|---|---:|---:|---|---|",
    ])
    for row in overview.get("targets") or []:
        refs = row.get("dossier_refs") or {}
        lines.append(
            "| {target} | {family} | {status} | {stages} | {queries} | {evo} | {dossier} |".format(
                target=row.get("target"),
                family=row.get("family_bucket"),
                status=row.get("route_status"),
                stages=len(row.get("synthesis_stages") or []),
                queries=len(row.get("difficulty_queries") or []),
                evo=(row.get("self_evolution") or {}).get("kb_target_layer"),
                dossier=refs.get("markdown") or refs.get("json") or "",
            )
        )
    for row in overview.get("targets") or []:
        lines.extend(["", f"## {row.get('target')}", ""])
        lines.append(f"- expected_reaction_class: `{row.get('expected_reaction_class')}`")
        lines.append(f"- template_sources: `{', '.join(row.get('template_sources') or [])}`")
        lines.append(f"- self_evo: `{(row.get('self_evolution') or {}).get('kb_target_layer')}`")
        closure_audit = row.get("route_closure_audit") or {}
        lines.append(
            "- route_closure: `{status}` blockers `{blockers}` followup_queries `{queries}`".format(
                status=closure_audit.get("readiness_status") or "",
                blockers=closure_audit.get("blocker_count") or 0,
                queries=closure_audit.get("followup_query_count") or 0,
            )
        )
        followup_execution = closure_audit.get("followup_execution") or {}
        lines.append(
            "- closure_followup_execution: `{policy}` executed `{executed}` leads `{leads}` abstract_signals `{signals}` full_exec `{full_exec}`".format(
                policy=followup_execution.get("policy") or "",
                executed=followup_execution.get("executed_trace_count") or 0,
                leads=followup_execution.get("lead_trace_count") or 0,
                signals=followup_execution.get("abstract_signal_trace_count") or 0,
                full_exec=bool(followup_execution.get("full_execution_coverage")),
            )
        )
        trace_summary = row.get("literature_trace") or {}
        lines.append(
            "- literature_trace: accepted `{accepted}` / total `{total}` via `{backend}`".format(
                accepted=trace_summary.get("accepted_query_trace_count") or 0,
                total=trace_summary.get("query_trace_count") or 0,
                backend=trace_summary.get("backend_resolved") or "",
            )
        )
        lines.extend(["", "### Synthesis Stages"])
        for stage in row.get("synthesis_stages") or []:
            lines.append(
                "- `{stage}` {title}: {notes}".format(
                    stage=stage.get("stage_id"),
                    title=stage.get("title"),
                    notes=stage.get("notes"),
                )
            )
        lines.extend(["", "### Automatic Literature Queries"])
        for query in row.get("difficulty_queries") or []:
            lines.append(
                "- `{difficulty}` `{query}` -> {signal}".format(
                    difficulty=query.get("difficulty"),
                    query=query.get("query"),
                    signal=query.get("acceptance_signal"),
                )
            )
        lines.extend(["", "### Literature Trace"])
        for trace in row.get("query_execution_traces") or []:
            lines.append(
                "- `{difficulty}` status `{status}` hits `{hits}` validated `{validated}` report `{report}`".format(
                    difficulty=trace.get("difficulty"),
                    status=trace.get("execution_status"),
                    hits=trace.get("hit_count"),
                    validated=trace.get("validated_evidence_count"),
                    report=trace.get("report_ref"),
                )
            )
            lines.append(
                "  - quality `{quality}` template_support `{support}` external_leads `{leads}`".format(
                    quality=trace.get("quality_gate_status"),
                    support=trace.get("template_supporting_evidence_count") or 0,
                    leads=trace.get("external_literature_lead_count") or 0,
                )
            )
        lines.extend(["", "### Key Intermediate Roles"])
        for role in row.get("key_intermediate_roles") or []:
            lines.append(f"- {role}")
        lines.extend(["", "### Route Closure Blockers"])
        for blocker in (row.get("route_closure_audit") or {}).get("blocking_requirements") or []:
            lines.append(f"- `{blocker.get('requirement_id')}` {blocker.get('title')}")
    if validation.get("reasons"):
        lines.extend(["", "## Validation Reasons"])
        for reason in validation.get("reasons") or []:
            lines.append(f"- `{reason}`")
    lines.append("")
    return "\n".join(lines)


def _write_typed_artifact_manifest(
    root: Path,
    report: dict[str, Any],
    *,
    full_panel_run: bool,
) -> dict[str, Any]:
    typed_dir = root / "typed_artifacts"
    typed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = typed_dir / "statin_typed_artifact_manifest.json"
    markdown_path = typed_dir / "statin_typed_artifact_manifest.md"
    if not full_panel_run:
        manifest = {
            "schema_version": STATIN_TYPED_ARTIFACT_MANIFEST_SCHEMA,
            "skipped": True,
            "skip_reason": "subset_replay_no_full_panel_artifact_manifest",
            "artifact_count": 0,
            "artifacts": [],
            "validation_summary": {
                "schema_version": "statin_typed_artifact_validation_summary.v1",
                "accepted": True,
                "accepted_count": 0,
                "rejected_count": 0,
                "results": [],
            },
            "json": str(manifest_path),
            "markdown": str(markdown_path),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        markdown_path.write_text(_render_typed_artifact_manifest_md(manifest), encoding="utf-8")
        return manifest

    from cascade_planner.agent.artifact_validators import validate_typed_artifact

    artifacts: list[dict[str, Any]] = [
        _typed_artifact(
            artifact_type="StatinPanelSelfEvoReport",
            schema_version="statin_panel_self_evo_report_artifact.v1",
            artifact_id="statin_panel_literature_self_evo_report",
            case_id="statin_panel",
            payload=report,
        )
    ]
    overview = _read_json((report.get("fullflow_overview") or {}).get("json"))
    if overview:
        artifacts.append(_typed_artifact(
            artifact_type="StatinFullflowOverview",
            schema_version="statin_fullflow_overview_artifact.v1",
            artifact_id="statin_panel_fullflow_overview",
            case_id="statin_panel",
            payload=overview,
        ))
    closure_matrix = _read_json((report.get("route_closure_matrix") or {}).get("json"))
    if closure_matrix:
        artifacts.append(_typed_artifact(
            artifact_type="StatinRouteClosureMatrix",
            schema_version="statin_route_closure_matrix_artifact.v1",
            artifact_id="statin_route_closure_matrix",
            case_id="statin_panel",
            payload=closure_matrix,
        ))
    curation_packet = _read_json((report.get("closure_lead_curation_packet") or {}).get("json"))
    if curation_packet:
        artifacts.append(_typed_artifact(
            artifact_type="StatinClosureLeadCurationPacket",
            schema_version="statin_closure_lead_curation_packet_artifact.v1",
            artifact_id="statin_closure_lead_curation_packet",
            case_id="statin_panel",
            payload=curation_packet,
        ))
    curation_result_set = _read_json((report.get("closure_curation_result_set") or {}).get("json"))
    if curation_result_set:
        artifacts.append(_typed_artifact(
            artifact_type="StatinClosureCurationResultSet",
            schema_version="statin_closure_curation_result_set_artifact.v1",
            artifact_id="statin_closure_curation_result_set",
            case_id="statin_panel",
            payload=curation_result_set,
        ))
    for row in report.get("targets") or []:
        safe = str(row.get("safe") or row.get("name") or "unknown").lower()
        dossier = _read_json((row.get("fullflow_dossier") or {}).get("json"))
        if not dossier:
            continue
        artifacts.append(_typed_artifact(
            artifact_type="StatinFullflowDossier",
            schema_version="statin_fullflow_dossier_artifact.v1",
            artifact_id=f"{safe}_fullflow_dossier",
            case_id=safe,
            payload=dossier,
        ))
        artifacts.append(_typed_artifact(
            artifact_type="StatinRouteTemplate",
            schema_version="statin_route_template_artifact.v1",
            artifact_id=f"{safe}_route_template",
            case_id=safe,
            payload=dossier.get("route_template") or {},
        ))
        artifacts.append(_typed_artifact(
            artifact_type="StatinRouteClosureAudit",
            schema_version="statin_route_closure_audit_artifact.v1",
            artifact_id=f"{safe}_route_closure_audit",
            case_id=safe,
            payload=dossier.get("route_closure_audit") or {},
        ))

    artifact_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        artifact_path = typed_dir / f"{artifact['artifact_id']}.json"
        artifact_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        validation = validate_typed_artifact(artifact)
        validation_rows.append(validation)
        artifact_rows.append({
            "artifact_id": artifact["artifact_id"],
            "artifact_type": artifact["artifact_type"],
            "case_id": artifact["case_id"],
            "json": str(artifact_path),
            "validation": validation,
        })
    accepted_count = sum(1 for row in validation_rows if row.get("accepted"))
    expected_artifact_count = 32
    validation_summary = {
        "schema_version": "statin_typed_artifact_validation_summary.v1",
        "accepted": accepted_count == len(validation_rows) and len(artifact_rows) == expected_artifact_count,
        "accepted_count": accepted_count,
        "rejected_count": len(validation_rows) - accepted_count,
        "expected_artifact_count": expected_artifact_count,
        "artifact_count_matches_full_panel_contract": len(artifact_rows) == expected_artifact_count,
        "results": validation_rows,
    }
    manifest = {
        "schema_version": STATIN_TYPED_ARTIFACT_MANIFEST_SCHEMA,
        "skipped": False,
        "artifact_count": len(artifact_rows),
        "artifacts": artifact_rows,
        "validation_summary": validation_summary,
        "json": str(manifest_path),
        "markdown": str(markdown_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_render_typed_artifact_manifest_md(manifest), encoding="utf-8")
    return manifest


def _typed_artifact(
    *,
    artifact_type: str,
    schema_version: str,
    artifact_id: str,
    case_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_type": artifact_type,
        "schema_version": schema_version,
        "artifact_id": artifact_id,
        "case_id": case_id,
        "source": "statin_panel_literature_self_evo",
        "input_refs": ["statin_panel_literature_workflow"],
        "evidence_refs": [str(ref) for ref in payload.get("evidence_refs") or []],
        "validation_status": "validated",
        "payload": payload,
    }


def _manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest.get("schema_version"),
        "skipped": bool(manifest.get("skipped")),
        "skip_reason": manifest.get("skip_reason") or "",
        "artifact_count": int(manifest.get("artifact_count") or 0),
        "json": manifest.get("json") or "",
        "markdown": manifest.get("markdown") or "",
        "validation_summary": dict(manifest.get("validation_summary") or {}),
    }


def _render_typed_artifact_manifest_md(manifest: dict[str, Any]) -> str:
    summary = manifest.get("validation_summary") or {}
    lines = [
        "# Statin Typed Artifact Manifest",
        "",
        f"- skipped: `{bool(manifest.get('skipped'))}`",
        f"- artifact_count: `{manifest.get('artifact_count') or 0}`",
        f"- accepted: `{bool(summary.get('accepted'))}`",
        f"- accepted_count: `{summary.get('accepted_count') or 0}`",
        f"- rejected_count: `{summary.get('rejected_count') or 0}`",
        "",
        "## Artifacts",
        "",
        "| artifact | type | accepted | reasons |",
        "|---|---|---|---|",
    ]
    for row in manifest.get("artifacts") or []:
        validation = row.get("validation") or {}
        lines.append(
            "| {artifact} | {atype} | {accepted} | {reasons} |".format(
                artifact=row.get("artifact_id"),
                atype=row.get("artifact_type"),
                accepted=bool(validation.get("accepted")),
                reasons=", ".join(validation.get("reasons") or []),
            )
        )
    if manifest.get("skip_reason"):
        lines.extend(["", f"- skip_reason: `{manifest.get('skip_reason')}`"])
    lines.append("")
    return "\n".join(lines)


def _evaluate_target(target: StatinPanelTarget, result: dict[str, Any]) -> dict[str, Any]:
    artifacts = result.get("artifacts") or {}
    output_dir = Path(result.get("output_dir") or "")
    package = _read_json(artifacts.get("hybrid_route_package"))
    validation = _read_json(artifacts.get("validation"))
    trigger_report = _read_json(artifacts.get("literature_trigger_report"))
    literature_task_path = output_dir / "literature_search_task.json"
    literature_report_path = output_dir / "literature_search_report.json"
    literature_report = _read_json(literature_report_path)
    evidence_cards = _read_jsonl(artifacts.get("evidence_cards"))
    candidates = [dict(item) for item in package.get("literature_candidates") or []]
    observed_classes = sorted({str(item.get("reaction_class") or "") for item in candidates if item.get("reaction_class")})
    observed_kinds = sorted({str(item.get("candidate_kind") or "") for item in candidates if item.get("candidate_kind")})
    evidence_refs = [str(item) for item in package.get("literature_evidence_refs") or []]
    primary_candidates = [
        item for item in candidates
        if item.get("reaction_class") == target.expected_reaction_class
    ]
    required_kinds = {"exact_fragment_retro", "forward_surrogate", "route_anchor"}
    warnings: list[str] = []
    off_family_classes = sorted(
        item for item in observed_classes
        if item not in {target.expected_reaction_class, "route_anchor"}
    )
    if off_family_classes:
        warnings.append("off_family_template_candidates:" + ",".join(off_family_classes))
    if literature_report.get("limitations"):
        warnings.extend(f"literature:{item}" for item in literature_report.get("limitations") or [])

    route_status = str(validation.get("route_status") or package.get("route_status") or "")
    reasons: list[str] = []
    if not validation.get("accepted"):
        reasons.append("validation_not_accepted")
    if not bool(trigger_report.get("should_trigger")):
        reasons.append("literature_not_triggered")
    if target.expected_reaction_class not in observed_classes:
        reasons.append("expected_statin_template_missing")
    if not required_kinds.issubset(set(observed_kinds)):
        reasons.append("required_candidate_kind_missing")
    if route_status == "solved":
        reasons.append("target_claimed_solved")
    if not evidence_refs:
        reasons.append("missing_literature_evidence_refs")
    evidence_validation = literature_report.get("evidence_validation") or {}
    search_rows = [dict(item) for item in literature_report.get("searches") or []]
    evidence_quality = _evidence_quality_summary(evidence_cards, evidence_refs=evidence_refs)

    return {
        "name": target.name,
        "safe": target.safe,
        "family_bucket": target.family_bucket,
        "target_smiles": target.target_smiles,
        "expected_reaction_class": target.expected_reaction_class,
        "expected_family_id": target.expected_family_id,
        "validation_accepted": bool(validation.get("accepted")),
        "route_status": route_status,
        "claims_solved": route_status == "solved",
        "literature_mode_entered": bool(trigger_report.get("should_trigger")),
        "literature_trigger_reasons": list(trigger_report.get("trigger_reasons") or []),
        "expected_template_hit": target.expected_reaction_class in observed_classes,
        "required_candidate_kinds_hit": required_kinds.issubset(set(observed_kinds)),
        "observed_reaction_classes": observed_classes,
        "observed_candidate_kinds": observed_kinds,
        "evidence_ref_count": len(evidence_refs),
        "primary_template_candidate_count": len(primary_candidates),
        "workflow_output_dir": str(output_dir),
        "literature_search_task": str(literature_task_path),
        "literature_search_report": str(literature_report_path),
        "literature_search_summary": {
            "schema_version": "statin_target_literature_search_summary.v1",
            "backend_requested": literature_report.get("backend_requested") or "",
            "backend_resolved": literature_report.get("backend_resolved") or literature_report.get("backend") or "",
            "hit_count": int(literature_report.get("hit_count") or 0),
            "validated_evidence_count": int(evidence_validation.get("accepted") or 0),
            "rejected_evidence_count": int(evidence_validation.get("rejected") or 0),
            "unresolved_literature_gap": bool(literature_report.get("unresolved_literature_gap")),
            "search_queries": [
                str(item.get("query") or "")
                for item in search_rows
                if item.get("query")
            ],
            "search_sources": sorted({
                str(item.get("source") or "")
                for item in search_rows
                if item.get("source")
            }),
            "evidence_refs": evidence_refs,
            "evidence_quality": evidence_quality,
            "limitations": list(literature_report.get("limitations") or []),
        },
        "hybrid_route_package": artifacts.get("hybrid_route_package"),
        "summary": artifacts.get("summary"),
        "route_map": artifacts.get("route_map"),
        "showcase_reference": {
            "source_route_count": target.source_route_count,
            "showcase_route_count": target.showcase_route_count,
            "route_assets": list(target.route_assets),
        },
        "warnings": warnings,
        "passed": not reasons,
        "reasons": reasons,
    }


def _register_self_evolution_candidate(
    kb: LayeredKnowledgeBase,
    target: StatinPanelTarget,
    row: dict[str, Any],
) -> dict[str, Any]:
    package = _read_json(row.get("hybrid_route_package"))
    primary_templates = [
        dict(item)
        for item in package.get("strategy_templates") or []
        if item.get("reaction_class") == target.expected_reaction_class
    ]
    candidate = EvolutionCandidate(
        candidate_id=f"{target.safe}_{target.expected_reaction_class}_template_candidate",
        candidate_type="TemplateCandidate",
        payload={
            "schema_version": STATIN_SELF_EVO_TEMPLATE_SCHEMA,
            "target_name": target.name,
            "target_smiles": target.target_smiles,
            "family_bucket": target.family_bucket,
            "expected_reaction_class": target.expected_reaction_class,
            "expected_family_id": target.expected_family_id,
            "template_count": len(primary_templates),
            "templates": primary_templates,
            "quality_gates": {
                "validation_accepted": bool(row.get("validation_accepted")),
                "expected_template_hit": bool(row.get("expected_template_hit")),
                "required_candidate_kinds_hit": bool(row.get("required_candidate_kinds_hit")),
                "no_solved_claim": not bool(row.get("claims_solved")),
            },
            "allowed_use": "shadow_or_staging_only_from_target_run",
            "not_raw_reaction_injection": True,
        },
        evidence_refs=[f"statin_panel:{target.safe}", *[str(ref) for ref in package.get("literature_evidence_refs") or []]],
        validation_status="validated",
        source="statin_panel_literature_self_evo",
    )
    validation = validate_evolution_candidate(candidate)
    production_write_blocked = False
    gate = evaluate_benchmark_gate({
        "true_solved_rate_delta": 0.0,
        "fake_closure_rate_delta": 0.0,
        "condition_quality_delta": 0.0,
        "template_replay_passes": bool(row.get("expected_template_hit")),
        "structure_validated": bool(row.get("validation_accepted")),
        "evidence_source_credible": bool(row.get("evidence_ref_count")),
        "role_assignment_checked": bool(row.get("required_candidate_kinds_hit")),
    })
    if validation.get("accepted"):
        kb.add_candidate(candidate, target_run=True)
        kb.promote(candidate.candidate_id, from_layer="candidate", to_layer="shadow", target_run=True)
        kb.promote(candidate.candidate_id, from_layer="shadow", to_layer="staging", target_run=True)
        try:
            kb.promote(candidate.candidate_id, from_layer="staging", to_layer="production", gate_report=gate, target_run=True)
        except ValueError as exc:
            production_write_blocked = "target_run_cannot_write_production" in str(exc)
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_type": candidate.candidate_type,
        "status": "staging" if validation.get("accepted") else "rejected",
        "candidate_validation": validation,
        "benchmark_gate": gate.to_dict(),
        "kb_target_layer": "staging" if validation.get("accepted") else "rejected",
        "production_write_blocked": production_write_blocked,
    }


def _aggregate_self_evolution_templates(
    kb: LayeredKnowledgeBase,
    rows: list[dict[str, Any]],
    *,
    full_panel_run: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "statin_self_evo_aggregation_report.v1",
        "accepted": False,
        "skipped": False,
        "skip_reason": "",
        "production_promoted_count": 0,
        "families": [],
    }
    if not full_panel_run:
        report["skipped"] = True
        report["skip_reason"] = "subset_replay_no_production_promotion"
        return report

    family_rows = {
        "natural_statin": [row for row in rows if row.get("family_bucket") == "natural_statin"],
        "synthetic_statin": [row for row in rows if row.get("family_bucket") == "synthetic_statin"],
    }
    required_counts = {"natural_statin": 4, "synthetic_statin": 5}
    for family_bucket, group in family_rows.items():
        family_report = _aggregate_family_template(kb, family_bucket, group, required_count=required_counts[family_bucket])
        report["families"].append(family_report)
        if family_report.get("production_promoted"):
            report["production_promoted_count"] += 1
    report["accepted"] = all(item.get("staging_promoted") and not item.get("production_promoted") for item in report["families"])
    return report


def _aggregate_family_template(
    kb: LayeredKnowledgeBase,
    family_bucket: str,
    rows: list[dict[str, Any]],
    *,
    required_count: int,
) -> dict[str, Any]:
    expected_class = "statin_semisynthesis" if family_bucket == "natural_statin" else "statin_side_chain_convergence"
    expected_family = "natural_statin_semisynthesis" if family_bucket == "natural_statin" else "synthetic_statin"
    reasons: list[str] = []
    if len(rows) < required_count:
        reasons.append("insufficient_family_replication")
    if not all(row.get("passed") for row in rows):
        reasons.append("target_replay_failed")
    if not all(row.get("expected_template_hit") for row in rows):
        reasons.append("expected_template_missing")
    if any(row.get("warnings") for row in rows):
        reasons.append("off_family_or_literature_warning")

    evidence_refs = sorted({
        ref
        for row in rows
        for ref in _read_json(row.get("hybrid_route_package")).get("literature_evidence_refs", [])
    })
    template_sources = sorted({
        str(template.get("source_record_id") or "")
        for row in rows
        for template in _read_json(row.get("hybrid_route_package")).get("strategy_templates", [])
        if template.get("reaction_class") == expected_class and template.get("source_record_id")
    })
    candidate = EvolutionCandidate(
        candidate_id=f"statin_family_{family_bucket}_{expected_class}_template",
        candidate_type="TemplateCandidate",
        payload={
            "schema_version": STATIN_FAMILY_SELF_EVO_TEMPLATE_SCHEMA,
            "family_bucket": family_bucket,
            "expected_family_id": expected_family,
            "expected_reaction_class": expected_class,
            "replicated_target_count": len(rows),
            "replicated_targets": [str(row.get("safe") or row.get("name") or "") for row in rows],
            "template_source_record_ids": template_sources,
            "quality_gates": {
                "required_replication_count": required_count,
                "all_target_replays_passed": all(row.get("passed") for row in rows),
                "all_expected_templates_hit": all(row.get("expected_template_hit") for row in rows),
                "all_required_candidate_kinds_hit": all(row.get("required_candidate_kinds_hit") for row in rows),
                "warnings_absent": not any(row.get("warnings") for row in rows),
                "no_solved_claims": not any(row.get("claims_solved") for row in rows),
            },
            "allowed_use": "production_family_template_after_cross_target_replay",
            "not_raw_reaction_injection": True,
        },
        evidence_refs=[f"statin_family:{family_bucket}", *evidence_refs],
        validation_status="validated",
        source="statin_panel_cross_target_self_evo",
    )
    validation = validate_evolution_candidate(candidate)
    if not validation.get("accepted"):
        reasons.extend(validation.get("reasons") or [])
    gate = evaluate_benchmark_gate({
        "true_solved_rate_delta": 0.0,
        "fake_closure_rate_delta": 0.0,
        "condition_quality_delta": 0.0,
        "template_replay_passes": not reasons,
        "structure_validated": all(row.get("validation_accepted") for row in rows),
        "evidence_source_credible": bool(evidence_refs),
        "role_assignment_checked": all(row.get("required_candidate_kinds_hit") for row in rows),
        "overgeneralization_detected": bool(any(row.get("warnings") for row in rows)),
    })

    staging_promoted = False
    production_promoted = False
    if validation.get("accepted"):
        kb.add_candidate(candidate, target_run=False)
        kb.promote(candidate.candidate_id, from_layer="candidate", to_layer="shadow", target_run=False)
        kb.promote(candidate.candidate_id, from_layer="shadow", to_layer="staging", target_run=False)
        staging_promoted = True
    return {
        "family_bucket": family_bucket,
        "candidate_id": candidate.candidate_id,
        "target_count": len(rows),
        "required_count": required_count,
        "candidate_validation": validation,
        "benchmark_gate": gate.to_dict(),
        "staging_promoted": staging_promoted,
        "production_promoted": production_promoted,
        "production_blocked": True,
        "template_source_record_ids": template_sources,
        "evidence_ref_count": len(evidence_refs),
        "reasons": sorted(set(reasons)),
    }


def _write_fullflow_dossier(
    root: Path,
    target: StatinPanelTarget,
    row: dict[str, Any],
    *,
    execute_closure_followups: bool,
    closure_followup_limit: int,
) -> dict[str, Any]:
    dossier_dir = root / "fullflow_dossiers"
    dossier_dir.mkdir(parents=True, exist_ok=True)
    dossier = _build_fullflow_dossier(
        target,
        row,
        execute_closure_followups=execute_closure_followups,
        closure_followup_limit=closure_followup_limit,
    )
    validation = _validate_fullflow_dossier(dossier)
    dossier["validation"] = validation
    json_path = dossier_dir / f"{target.safe}_fullflow_dossier.json"
    markdown_path = dossier_dir / f"{target.safe}_fullflow_dossier.md"
    json_path.write_text(json.dumps(dossier, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_render_fullflow_dossier_md(dossier), encoding="utf-8")
    return {
        "dossier": dossier,
        "validation": validation,
        "json_path": json_path,
        "markdown_path": markdown_path,
    }


def _build_fullflow_dossier(
    target: StatinPanelTarget,
    row: dict[str, Any],
    *,
    execute_closure_followups: bool,
    closure_followup_limit: int,
) -> dict[str, Any]:
    package = _read_json(row.get("hybrid_route_package"))
    evidence_refs = [str(ref) for ref in package.get("literature_evidence_refs") or []]
    templates = [
        dict(item)
        for item in package.get("strategy_templates") or []
        if item.get("reaction_class") == target.expected_reaction_class
    ]
    primary_template_sources = sorted({
        str(template.get("source_record_id") or "")
        for template in templates
        if template.get("source_record_id")
    })
    stages = _natural_statin_stages(target, evidence_refs) if target.family_bucket == "natural_statin" else _synthetic_statin_stages(target, evidence_refs)
    blueprint = _fullflow_blueprint(target, primary_template_sources, evidence_refs)
    difficulty_escalation = _difficulty_escalation(target, row)
    automatic_escalation = _automatic_literature_escalation(
        target,
        row,
        primary_template_sources,
    )
    route_template = _member_route_template(
        target,
        stages,
        primary_template_sources,
        evidence_refs,
        automatic_escalation,
    )
    route_closure_audit = _route_closure_audit(
        target,
        row,
        stages,
        route_template,
        automatic_escalation,
        execute_followups=execute_closure_followups,
        followup_limit=closure_followup_limit,
    )
    return {
        "schema_version": STATIN_FULLFLOW_DOSSIER_SCHEMA,
        "target": target.to_dict(),
        "route_status": row.get("route_status"),
        "planning_status": "fullflow_planning_material_partial_anchor",
        "not_lab_procedure": True,
        "status_contract": (
            "This dossier is a route-planning synthesis outline. It is not an executable "
            "laboratory procedure and does not claim solved status without stock and route audit proof."
        ),
        "literature_trigger_reasons": list(row.get("literature_trigger_reasons") or []),
        "evidence_refs": evidence_refs,
        "observed_reaction_classes": list(row.get("observed_reaction_classes") or []),
        "observed_candidate_kinds": list(row.get("observed_candidate_kinds") or []),
        "primary_template_count": len(templates),
        "primary_template_sources": primary_template_sources,
        "synthesis_stages": stages,
        "route_template": route_template,
        "fullflow_blueprint": blueprint,
        "difficulty_escalation": difficulty_escalation,
        "automatic_literature_escalation": automatic_escalation,
        "route_closure_audit": route_closure_audit,
        "template_quality": {
            "expected_template_hit": bool(row.get("expected_template_hit")),
            "required_candidate_kinds_hit": bool(row.get("required_candidate_kinds_hit")),
            "warnings": list(row.get("warnings") or []),
            "evidence_ref_count": int(row.get("evidence_ref_count") or 0),
        },
        "self_evolution": dict(row.get("self_evolution") or {}),
        "artifact_refs": {
            "hybrid_route_package": row.get("hybrid_route_package"),
            "summary": row.get("summary"),
            "route_map": row.get("route_map"),
            "workflow_output_dir": row.get("workflow_output_dir"),
        },
    }


def _fullflow_blueprint(
    target: StatinPanelTarget,
    primary_template_sources: list[str],
    evidence_refs: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "statin_member_fullflow_blueprint.v1",
        "target_safe": target.safe,
        "route_family": target.family_bucket,
        "expected_reaction_class": target.expected_reaction_class,
        "member_specific_route_outline": _member_route_outline(target.safe),
        "key_intermediate_roles": _key_intermediate_roles(target.safe),
        "endpoint_audit": [
            "target identity, stereochemistry, acid/lactone/salt state, and counterion must be explicit",
            "all non-literature terminal leaves require stock audit before solved status",
            "conditions and hazards are evidence-gated and must not be inferred from a family template alone",
        ],
        "primary_template_sources": list(primary_template_sources),
        "evidence_refs": list(evidence_refs),
        "not_lab_procedure": True,
    }


def _member_route_template(
    target: StatinPanelTarget,
    stages: list[dict[str, Any]],
    primary_template_sources: list[str],
    evidence_refs: list[str],
    automatic_escalation: dict[str, Any],
) -> dict[str, Any]:
    difficulty_queries = list(automatic_escalation.get("difficulty_queries") or [])
    query_execution_traces = list(automatic_escalation.get("query_execution_traces") or [])
    acceptance_criteria = list(automatic_escalation.get("acceptance_criteria") or [])
    rejection_rules = list(automatic_escalation.get("rejection_rules") or [])
    steps = [
        _route_template_step(
            target,
            order=index + 1,
            stage=stage,
            primary_template_sources=primary_template_sources,
            evidence_refs=evidence_refs,
            difficulty_queries=difficulty_queries,
            query_execution_traces=query_execution_traces,
            acceptance_criteria=acceptance_criteria,
            rejection_rules=rejection_rules,
        )
        for index, stage in enumerate(stages)
    ]
    return {
        "schema_version": STATIN_MEMBER_ROUTE_TEMPLATE_SCHEMA,
        "target_safe": target.safe,
        "target_name": target.name,
        "family_bucket": target.family_bucket,
        "expected_reaction_class": target.expected_reaction_class,
        "expected_family_id": target.expected_family_id,
        "template_sources": list(primary_template_sources),
        "evidence_refs": list(evidence_refs),
        "template_steps": steps,
        "allowed_use": "planning_template_and_self_evo_seed_after_validation",
        "promotion_scope": "target_run_staging_only; family production requires cross-target replay",
        "not_lab_procedure": True,
    }


def _route_template_step(
    target: StatinPanelTarget,
    *,
    order: int,
    stage: dict[str, Any],
    primary_template_sources: list[str],
    evidence_refs: list[str],
    difficulty_queries: list[dict[str, Any]],
    query_execution_traces: list[dict[str, Any]],
    acceptance_criteria: list[str],
    rejection_rules: list[str],
) -> dict[str, Any]:
    template_role = str(stage.get("template_role") or "")
    relevant_difficulties = _relevant_difficulty_queries(target, template_role, difficulty_queries)
    relevant_traces = _relevant_query_traces(relevant_difficulties, query_execution_traces)
    return {
        "schema_version": STATIN_ROUTE_TEMPLATE_STEP_SCHEMA,
        "step_id": str(stage.get("stage_id") or f"step_{order}"),
        "order": order,
        "title": str(stage.get("title") or ""),
        "route_role": str(stage.get("route_role") or ""),
        "template_role": template_role,
        "template_sources": list(primary_template_sources),
        "evidence_refs": list(stage.get("evidence_refs") or evidence_refs),
        "difficulty_queries": relevant_difficulties,
        "literature_trace_refs": relevant_traces,
        "requires_literature_evidence": template_role not in {"ordinary_or_audit"},
        "acceptance_criteria": acceptance_criteria,
        "rejection_rules": rejection_rules,
        "self_evo_tags": [
            target.family_bucket,
            target.expected_family_id,
            target.expected_reaction_class,
            template_role,
        ],
        "status": "planning_material",
        "not_lab_procedure": True,
    }


def _relevant_difficulty_queries(
    target: StatinPanelTarget,
    template_role: str,
    difficulty_queries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if template_role == "ordinary_or_audit":
        return [item for item in difficulty_queries if str(item.get("difficulty") or "") == "endpoint_state"]
    if target.family_bucket == "natural_statin":
        if template_role == "route_anchor":
            wanted = {"fermentation_or_biotransformation_anchor"}
        elif template_role == target.expected_reaction_class:
            wanted = {"member_specific_tailoring"}
        else:
            wanted = {"fermentation_or_biotransformation_anchor", "member_specific_tailoring"}
    else:
        if "core" in template_role:
            wanted = {"member_specific_core_construction"}
        elif "stereocontrol" in template_role:
            wanted = {"syn_diol_stereocontrol"}
        elif template_role == target.expected_reaction_class:
            wanted = {"olefination_or_convergence_window"}
        else:
            wanted = {
                "member_specific_core_construction",
                "syn_diol_stereocontrol",
                "olefination_or_convergence_window",
            }
    selected = [item for item in difficulty_queries if str(item.get("difficulty") or "") in wanted]
    return selected or list(difficulty_queries[:1])


def _relevant_query_traces(
    difficulty_queries: list[dict[str, Any]],
    query_execution_traces: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    wanted = {
        str(item.get("difficulty") or "")
        for item in difficulty_queries
        if item.get("difficulty")
    }
    if not wanted:
        return []
    return [
        {
            "schema_version": str(trace.get("schema_version") or ""),
            "difficulty": str(trace.get("difficulty") or ""),
            "execution_status": str(trace.get("execution_status") or ""),
            "task_ref": str(trace.get("task_ref") or ""),
            "report_ref": str(trace.get("report_ref") or ""),
            "backend_resolved": str(trace.get("backend_resolved") or ""),
            "hit_count": int(trace.get("hit_count") or 0),
            "validated_evidence_count": int(trace.get("validated_evidence_count") or 0),
            "supporting_evidence_refs": list(trace.get("supporting_evidence_refs") or []),
            "template_supporting_evidence_refs": list(trace.get("template_supporting_evidence_refs") or []),
            "external_literature_lead_refs": list(trace.get("external_literature_lead_refs") or []),
            "template_supporting_evidence_count": int(trace.get("template_supporting_evidence_count") or 0),
            "external_literature_lead_count": int(trace.get("external_literature_lead_count") or 0),
            "quality_gate_status": str(trace.get("quality_gate_status") or ""),
            "template_sources": list(trace.get("template_sources") or []),
        }
        for trace in query_execution_traces
        if str(trace.get("difficulty") or "") in wanted
    ]


def _member_route_outline(safe: str) -> list[str]:
    outlines = {
        "atorvastatin": [
            "Expose the substituted pyrrole/diaryl amide core as a core-construction target, not as an atorvastatin-like terminal.",
            "Use a Paal-Knorr/Hantzsch-like convergent core assembly or equivalent reviewed pyrrole route precedent.",
            "Prepare a protected syn-3,5-diol side-chain equivalent with stereochemistry tracked before core attachment.",
            "Join core and side chain, then audit deprotection, hydrolysis, calcium/salt state, and product identity.",
        ],
        "cerivastatin": [
            "Expose the substituted pyridine core with isopropyl, methoxy, and fluoroaryl substituent pattern as a target-specific core.",
            "Prefer pyridine aldehyde or activated pyridine intermediates that support Wittig/HWE-like side-chain olefination.",
            "Install the E-heptenoate side chain from a protected syn-3,5-diol equivalent with geometry and stereochemistry tracked.",
            "Finish acid/lactone/salt state explicitly and keep the withdrawn-drug context separate from any route-planning claim.",
        ],
        "fluvastatin": [
            "Expose the indole core and heptenoate side-chain connection rather than closing on a fluvastatin-like terminal.",
            "Use condensation/olefination logic to assemble the side-chain carbon framework from an indole aldehyde/core.",
            "Resolve syn-diol construction by cryogenic reduction, asymmetric reduction, or biocatalytic precedent with process risks recorded.",
            "Finish sodium/acid state and impurity-risk audit before any guided rerun can claim route closure.",
        ],
        "pitavastatin": [
            "Expose the cyclopropyl quinoline core and side-chain coupling point as the strategic target.",
            "Use Suzuki, hydroboration-coupling, alkynyl coupling/Red-Al, or HWE precedent only with E-alkene tracking.",
            "Carry a prebuilt chiral side-chain fragment through coupling without losing syn-diol stereochemical metadata.",
            "Audit calcium/salt state, residual-metal or organoboron risk, and all terminal leaves before solved status.",
        ],
        "rosuvastatin": [
            "Expose the pyrimidine sulfonamide core and pyrimidine aldehyde/olefination handle.",
            "Install the heptenoate side chain by Wittig or related olefination with E-geometry evidence.",
            "Use stereocontrolled reduction or biocatalytic ketoreduction precedent to justify the syn-diol side chain.",
            "Audit deprotection, oxidation-state, calcium/salt endpoint, and route/condition evidence before closure.",
        ],
        "lovastatin": [
            "Treat lovastatin fermentation access to the decalin lactone core as a semisynthesis anchor, not ordinary stock.",
            "Track lactone/acid interconversion and side-chain state from the fermentation product.",
            "Apply only evidence-backed late tailoring or finishing edits for the target member.",
            "Require source, product, and condition audit before converting the anchor into a solved route.",
        ],
        "mevastatin": [
            "Treat compactin/mevastatin fermentation core access as the route anchor.",
            "Keep deacylated or oxidized natural-statin intermediates distinct from generic product-like terminals.",
            "Apply member-specific hydrolysis, lactonization, or side-chain-state edits only with evidence.",
            "Audit fermentation/source status and endpoint identity before any solved claim.",
        ],
        "pravastatin": [
            "Start from the natural-statin semisynthesis boundary and keep the decalin lactone core source explicit.",
            "Represent hydroxylation/biotransformation of the natural statin core as the target-defining tailoring step.",
            "Track acid/lactone state and salt endpoint separately from the hydroxylation precedent.",
            "Escalate to literature when regioselective hydroxylation or biocatalyst evidence is missing.",
        ],
        "simvastatin": [
            "Use lovastatin-derived fermentation material or a source-supported deacylated core as the semisynthesis anchor.",
            "Represent late acyl side-chain modification/methylbutyrate installation as the member-specific tailoring step.",
            "Track lactone/acid interconversion and salt/endpoint state after acylation.",
            "Reject product-like stock closure unless the upstream fermentation and tailoring evidence are explicit.",
        ],
    }
    return outlines.get(safe, ["Build member-specific core and side-chain plan from evidence-gated templates."])


def _key_intermediate_roles(safe: str) -> list[str]:
    roles = {
        "atorvastatin": [
            "pyrrole-core precursor",
            "4-fluorophenyl aryl fragment",
            "phenylamide fragment",
            "protected syn-3,5-diol side-chain equivalent",
        ],
        "cerivastatin": [
            "substituted pyridine aldehyde or activated pyridine core",
            "fluoroaryl/isopropyl/methoxy substitution pattern carrier",
            "phosphonium ylide or phosphonate side-chain equivalent",
            "protected syn-3,5-diol ester/lactone endpoint precursor",
        ],
        "fluvastatin": [
            "indole aldehyde/core",
            "tert-butyl acetoacetate-derived side-chain precursor",
            "beta-hydroxy ketone or protected diol intermediate",
            "sodium/acid endpoint precursor",
        ],
        "pitavastatin": [
            "cyclopropyl quinoline halide/triflate core",
            "chiral alkynyl ester or boronate side-chain precursor",
            "9-BBN/organoboron or HWE coupling partner",
            "protected syn-diol endpoint precursor",
        ],
        "rosuvastatin": [
            "pyrimidine aldehyde/core",
            "sulfonamide-bearing heteroaryl precursor",
            "phosphonium ylide/phosphonate side-chain equivalent",
            "biocatalytic or stereocontrolled ketoreduction substrate",
        ],
        "lovastatin": [
            "fermentation-derived lovastatin/decalin lactone core",
            "lactone/acid form pair",
            "side-chain state audit intermediate",
            "source-supported finishing endpoint",
        ],
        "mevastatin": [
            "compactin/mevastatin fermentation core",
            "deacylated or natural-statin core intermediate",
            "lactone/acid form pair",
            "source-supported finishing endpoint",
        ],
        "pravastatin": [
            "natural-statin fermentation core",
            "hydroxylated/biotransformed pravastatin intermediate",
            "acid/salt endpoint precursor",
            "regioselective hydroxylation evidence anchor",
        ],
        "simvastatin": [
            "lovastatin-derived fermentation core",
            "deacylated statin core",
            "2,2-dimethylbutyryl/acyl donor equivalent",
            "lactone/acid finishing endpoint",
        ],
    }
    return roles.get(safe, ["member-specific core", "member-specific side-chain intermediate"])


def _automatic_literature_escalation(
    target: StatinPanelTarget,
    row: dict[str, Any],
    primary_template_sources: list[str],
) -> dict[str, Any]:
    base_queries = _difficulty_queries(target)
    query_execution_traces = [
        _literature_query_trace(row, query, primary_template_sources)
        for query in base_queries
    ]
    escalation = {
        "schema_version": "statin_automatic_literature_escalation.v1",
        "entry_conditions": list(row.get("literature_trigger_reasons") or []),
        "difficulty_queries": base_queries,
        "query_execution_traces": query_execution_traces,
        "query_trace_summary": _query_trace_summary(query_execution_traces),
        "accepted_template_sources": list(primary_template_sources),
        "acceptance_criteria": [
            "source must support route-relevant synthesis, semisynthesis, core construction, or side-chain construction",
            "source must identify the member-specific core or natural-statin anchor rather than only pharmacology",
            "template promotion requires exact/forward-surrogate/route-anchor candidates plus no solved-route claim",
        ],
        "rejection_rules": [
            "reject assay, toxicity, or clinical-only papers as route anchors",
            "reject raw reaction injection without product-specific reconstruction/applicability evidence",
            "reject target-like stock closure that bypasses the difficult core or side-chain step",
        ],
    }
    return escalation


def _literature_query_trace(
    row: dict[str, Any],
    difficulty_query: dict[str, Any],
    primary_template_sources: list[str],
) -> dict[str, Any]:
    summary = row.get("literature_search_summary") or {}
    quality = summary.get("evidence_quality") or {}
    evidence_refs = list(summary.get("evidence_refs") or [])
    template_support_refs = list(quality.get("template_promotable_refs") or [])
    external_lead_refs = list(quality.get("external_literature_lead_refs") or [])
    hit_count = int(summary.get("hit_count") or 0)
    validated_count = int(summary.get("validated_evidence_count") or 0)
    unresolved_gap = bool(summary.get("unresolved_literature_gap"))
    covered = hit_count > 0 and validated_count > 0 and bool(evidence_refs) and not unresolved_gap
    template_supported = bool(template_support_refs)
    external_leads_present = bool(external_lead_refs)
    return {
        "schema_version": STATIN_LITERATURE_QUERY_TRACE_SCHEMA,
        "difficulty": str(difficulty_query.get("difficulty") or ""),
        "query": str(difficulty_query.get("query") or ""),
        "acceptance_signal": str(difficulty_query.get("acceptance_signal") or ""),
        "execution_status": (
            "covered_by_validated_literature_search"
            if covered
            else "unresolved_literature_gap"
        ),
        "task_ref": str(row.get("literature_search_task") or ""),
        "report_ref": str(row.get("literature_search_report") or ""),
        "backend_requested": str(summary.get("backend_requested") or ""),
        "backend_resolved": str(summary.get("backend_resolved") or ""),
        "hit_count": hit_count,
        "validated_evidence_count": validated_count,
        "rejected_evidence_count": int(summary.get("rejected_evidence_count") or 0),
        "search_queries": list(summary.get("search_queries") or []),
        "search_sources": list(summary.get("search_sources") or []),
        "supporting_evidence_refs": evidence_refs,
        "template_supporting_evidence_refs": template_support_refs,
        "external_literature_lead_refs": external_lead_refs,
        "template_supporting_evidence_count": len(template_support_refs),
        "external_literature_lead_count": len(external_lead_refs),
        "quality_gate_status": (
            "template_support_plus_external_leads"
            if template_supported and external_leads_present
            else "template_support_only"
            if template_supported
            else "lead_only_requires_full_text_or_curator_audit"
            if external_leads_present
            else "no_traceable_evidence"
        ),
        "template_sources": list(primary_template_sources),
        "limitations": list(summary.get("limitations") or []),
        "unresolved_literature_gap": unresolved_gap,
        "trace_contract": (
            "This trace records the automatic target-level literature search covering a difficult "
            "route-planning query; it is evidence support for a template, not an executable procedure."
        ),
    }


def _query_trace_summary(query_execution_traces: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [
        trace for trace in query_execution_traces
        if trace.get("execution_status") == "covered_by_validated_literature_search"
    ]
    template_supported = [
        trace for trace in query_execution_traces
        if int(trace.get("template_supporting_evidence_count") or 0) > 0
    ]
    with_external_leads = [
        trace for trace in query_execution_traces
        if int(trace.get("external_literature_lead_count") or 0) > 0
    ]
    backends = sorted({
        str(trace.get("backend_resolved") or "")
        for trace in query_execution_traces
        if trace.get("backend_resolved")
    })
    return {
        "schema_version": "statin_literature_query_trace_summary.v1",
        "query_trace_count": len(query_execution_traces),
        "accepted_query_trace_count": len(accepted),
        "template_supported_query_trace_count": len(template_supported),
        "external_lead_query_trace_count": len(with_external_leads),
        "backend_resolved": ",".join(backends),
        "all_queries_have_validated_evidence": len(accepted) == len(query_execution_traces),
        "all_queries_have_template_support": len(template_supported) == len(query_execution_traces),
        "external_leads_present": bool(with_external_leads),
    }


def _literature_trace_summary(escalation: dict[str, Any]) -> dict[str, Any]:
    traces = list(escalation.get("query_execution_traces") or [])
    summary = dict(escalation.get("query_trace_summary") or _query_trace_summary(traces))
    if traces:
        summary["task_ref"] = str(traces[0].get("task_ref") or "")
        summary["report_ref"] = str(traces[0].get("report_ref") or "")
        summary["hit_count"] = int(traces[0].get("hit_count") or 0)
        summary["validated_evidence_count"] = int(traces[0].get("validated_evidence_count") or 0)
    else:
        summary.setdefault("task_ref", "")
        summary.setdefault("report_ref", "")
        summary.setdefault("hit_count", 0)
        summary.setdefault("validated_evidence_count", 0)
    return summary


def _difficulty_queries(target: StatinPanelTarget) -> list[dict[str, str]]:
    safe = target.safe
    if target.family_bucket == "natural_statin":
        member_terms = {
            "lovastatin": "lovastatin fermentation semisynthesis lactone acid interconversion",
            "mevastatin": "mevastatin compactin fermentation semisynthesis route",
            "pravastatin": "pravastatin biotransformation hydroxylation fermentation core synthesis",
            "simvastatin": "simvastatin lovastatin semisynthesis acylation route",
        }
        return [
            {
                "difficulty": "fermentation_or_biotransformation_anchor",
                "query": member_terms.get(safe, f"{safe} natural statin fermentation semisynthesis"),
                "acceptance_signal": "source-supported natural-statin core or biotransformation anchor",
            },
            {
                "difficulty": "member_specific_tailoring",
                "query": f"{safe} semisynthesis acylation hydroxylation methylation lactone route",
                "acceptance_signal": "member-specific tailoring step with route relevance",
            },
            {
                "difficulty": "endpoint_state",
                "query": f"{safe} lactone acid salt form synthesis endpoint",
                "acceptance_signal": "explicit endpoint form and finishing-state evidence",
            },
        ]
    core_terms = {
        "atorvastatin": "atorvastatin Paal Knorr pyrrole core synthesis side chain",
        "cerivastatin": "cerivastatin pyridine core Wittig HWE side chain synthesis",
        "fluvastatin": "fluvastatin indole aldehyde side chain Wittig reduction synthesis",
        "pitavastatin": "pitavastatin quinoline side chain Suzuki hydroboration coupling synthesis",
        "rosuvastatin": "rosuvastatin pyrimidine aldehyde Wittig biocatalytic side chain synthesis",
    }
    return [
        {
            "difficulty": "member_specific_core_construction",
            "query": core_terms.get(safe, f"{safe} synthetic statin core side chain convergence synthesis"),
            "acceptance_signal": "member-specific heteroaryl/aromatic core construction or coupling precedent",
        },
        {
            "difficulty": "syn_diol_stereocontrol",
            "query": f"{safe} syn 3,5 dihydroxy acid side chain stereocontrolled reduction biocatalysis",
            "acceptance_signal": "stereocontrolled or biocatalytic side-chain construction evidence",
        },
        {
            "difficulty": "olefination_or_convergence_window",
            "query": f"{safe} E heptenoate Wittig HWE olefination process impurity endpoint",
            "acceptance_signal": "E-alkene/convergence and endpoint-state risk evidence",
        },
    ]


def _natural_statin_stages(target: StatinPanelTarget, evidence_refs: list[str]) -> list[dict[str, Any]]:
    target_notes = {
        "lovastatin": "fermentation-derived lovastatin/decalin lactone core followed by lactone-acid and side-chain state audit",
        "mevastatin": "compactin/mevastatin fermentation core with source-supported semisynthesis boundary",
        "pravastatin": "natural statin core plus hydroxylation/biotransformation-aware semisynthetic tailoring",
        "simvastatin": "lovastatin-derived core with late acyl side-chain modification and lactone-acid handling",
    }
    return [
        _stage(
            "fermentation_core_anchor",
            "Source-supported natural statin decalin/lactone core",
            "Treat fermentation or biotransformation access as an explicit route anchor, not as generic stock closure.",
            "route_anchor",
            evidence_refs,
            target_notes.get(target.safe, "natural statin fermentation-core anchor"),
        ),
        _stage(
            "semisynthetic_tailoring",
            "Late-stage semisynthetic tailoring",
            "Apply acylation, hydroxylation, methylation, or oxidation-state edits according to the statin member.",
            target.expected_reaction_class,
            evidence_refs,
            "The exact edit is member-specific and remains planning material until route/condition audit passes.",
        ),
        _stage(
            "ordinary_finishing_and_audit",
            "Lactone/acid/salt finishing and route audit",
            "Close ordinary functional-state changes with ChemEnzy/native steps, then require product, stock, and condition audit.",
            "ordinary_or_audit",
            evidence_refs,
            "No solved claim is allowed while the fermentation anchor and all leaves are not independently audited.",
        ),
    ]


def _synthetic_statin_stages(target: StatinPanelTarget, evidence_refs: list[str]) -> list[dict[str, Any]]:
    core_notes = {
        "atorvastatin": "pyrrole/diaryl amide core assembly; Paal-Knorr/Hantzsch-like convergence is the reviewed family precedent",
        "fluvastatin": "indole core plus side-chain installation; process-window records flag aldol/Wittig/reduction risks",
        "pitavastatin": "quinoline core with side-chain coupling options such as Suzuki, hydroboration-coupling, or alkynyl reduction sequences",
        "rosuvastatin": "pyrimidine core with olefination and biocatalytic side-chain building-block precedent",
        "cerivastatin": "substituted pyridine core with Wittig/HWE-like side-chain convergence and endpoint-state guard",
    }
    return [
        _stage(
            "heteroaryl_core_construction",
            "Aryl/heteroaryl statin core construction",
            "Expose a target-specific heteroaryl/aromatic core instead of closing on a product-like statin terminal.",
            "core_construction_anchor",
            evidence_refs,
            core_notes.get(target.safe, "synthetic statin core construction"),
        ),
        _stage(
            "syn_diol_side_chain_intermediate",
            "Chiral syn-3,5-dihydroxy acid side-chain building block",
            "Use stereocontrolled or biocatalytic side-chain logic as a required route template element.",
            "statin_side_chain_stereocontrol",
            evidence_refs,
            "Condition details remain evidence-gated; cryogenic or borane-sensitive reductions are process risks, not automatic failures.",
        ),
        _stage(
            "core_side_chain_convergence",
            "Core-side-chain convergence",
            "Join core and side chain through HWE/Wittig/Suzuki/hydroboration/alkynyl or class-specific convergent logic.",
            target.expected_reaction_class,
            evidence_refs,
            "The candidate is an advisory/executable-template seed only after product-specific applicability and reconstruction gates.",
        ),
        _stage(
            "acid_lactone_salt_finishing_and_audit",
            "Acid/lactone/salt finishing and route audit",
            "Finish functional-state and salt/lactone forms, then require product, stock, condition, and fake-closure audit.",
            "ordinary_or_audit",
            evidence_refs,
            "No solved claim is allowed from literature template presence alone.",
        ),
    ]


def _stage(
    stage_id: str,
    title: str,
    route_role: str,
    template_role: str,
    evidence_refs: list[str],
    notes: str,
) -> dict[str, Any]:
    return {
        "stage_id": stage_id,
        "title": title,
        "route_role": route_role,
        "template_role": template_role,
        "evidence_refs": list(evidence_refs),
        "notes": notes,
        "status": "planning_material",
    }


def _difficulty_escalation(target: StatinPanelTarget, row: dict[str, Any]) -> list[dict[str, Any]]:
    base = [
        {
            "difficulty": "advanced_frontier_or_fake_closure_risk",
            "trigger_reasons": list(row.get("literature_trigger_reasons") or []),
            "automatic_action": "enter_literature_mode_and_require_traceable_evidence_cards",
        },
        {
            "difficulty": "template_quality_and_scope",
            "trigger_reasons": list(row.get("warnings") or []),
            "automatic_action": "reject_off_family_templates_or_keep_as_warning_until_clean",
        },
    ]
    if target.family_bucket == "natural_statin":
        base.append({
            "difficulty": "fermentation_core_not_generic_stock",
            "trigger_reasons": ["natural_statin_semisynthesis_anchor"],
            "automatic_action": "keep fermentation core as explicit route anchor and require source/audit before solved status",
        })
    else:
        base.append({
            "difficulty": "syn_diol_stereocontrol_and_convergent_side_chain",
            "trigger_reasons": ["synthetic_statin_side_chain_convergence"],
            "automatic_action": "prefer literature-backed side-chain templates and self-evolve only after cross-target replay",
        })
    return base


def _route_closure_audit(
    target: StatinPanelTarget,
    row: dict[str, Any],
    stages: list[dict[str, Any]],
    route_template: dict[str, Any],
    automatic_escalation: dict[str, Any],
    *,
    execute_followups: bool,
    followup_limit: int,
) -> dict[str, Any]:
    trace_summary = automatic_escalation.get("query_trace_summary") or {}
    traces = list(automatic_escalation.get("query_execution_traces") or [])
    template_steps = list(route_template.get("template_steps") or [])
    stage_count = len(stages)
    passed_requirements: list[dict[str, Any]] = []
    blocking_requirements: list[dict[str, Any]] = []

    def passed(requirement_id: str, title: str, evidence_refs: list[str], notes: str) -> None:
        passed_requirements.append({
            "requirement_id": requirement_id,
            "title": title,
            "status": "passed",
            "evidence_refs": list(evidence_refs),
            "notes": notes,
        })

    def blocked(requirement_id: str, title: str, blocker: str, followup_query: str) -> None:
        blocking_requirements.append({
            "requirement_id": requirement_id,
            "title": title,
            "status": "blocked",
            "blocker": blocker,
            "followup_query": followup_query,
        })

    if stage_count >= 3 and len(template_steps) == stage_count:
        passed(
            "fullflow_stage_template_present",
            "Member route is represented as a stage-level fullflow template",
            list(route_template.get("evidence_refs") or []),
            "All expected high-level stages are present as planning material.",
        )
    else:
        blocked(
            "fullflow_stage_template_present",
            "Member route is represented as a stage-level fullflow template",
            "stage/template cardinality mismatch",
            f"{target.safe} complete synthesis route stage map",
        )
    if int(trace_summary.get("accepted_query_trace_count") or 0) >= len(automatic_escalation.get("difficulty_queries") or []) >= 3:
        passed(
            "difficulty_queries_have_executed_traces",
            "Difficult route points have automatic literature search traces",
            _flatten_trace_refs(traces, "supporting_evidence_refs"),
            "Every difficult query has a traceable search report.",
        )
    else:
        blocked(
            "difficulty_queries_have_executed_traces",
            "Difficult route points have automatic literature search traces",
            "one or more difficult query traces are missing or unaccepted",
            f"{target.safe} synthesis difficult step literature evidence",
        )
    if int(trace_summary.get("template_supported_query_trace_count") or 0) >= len(automatic_escalation.get("difficulty_queries") or []) >= 3:
        passed(
            "template_support_for_difficult_points",
            "Difficult route points have template-promotable evidence",
            _flatten_trace_refs(traces, "template_supporting_evidence_refs"),
            "Template support excludes lead-only PubMed summaries.",
        )
    else:
        blocked(
            "template_support_for_difficult_points",
            "Difficult route points have template-promotable evidence",
            "template-promotable evidence is missing for one or more difficult points",
            f"{target.safe} full text synthesis route template evidence",
        )

    blocked(
        "full_text_route_step_audit",
        "Each route stage is backed by full-text or curated step-level route evidence",
        "current dossier stores route templates and evidence leads, not full-text step extraction",
        f"{target.safe} full text experimental synthesis route intermediates conditions",
    )
    blocked(
        "condition_and_workup_evidence_audit",
        "Conditions, workup, isolation, and compatibility are evidence-audited",
        "conditions are intentionally not inferred from summary/template evidence",
        f"{target.safe} synthesis conditions workup isolation intermediate route",
    )
    blocked(
        "terminal_stock_or_source_audit",
        "All route leaves have stock/source, fermentation, or semisynthesis anchor proof",
        "terminal leaves and upstream source availability are not fully audited",
        f"{target.safe} starting material stock audit fermentation semisynthesis precursor",
    )
    blocked(
        "endpoint_identity_and_salt_state_audit",
        "Endpoint identity, stereochemistry, lactone/acid/salt state, and counterion are closed",
        "endpoint form is represented as an audit requirement, not proven solved closure",
        f"{target.safe} endpoint salt lactone acid stereochemistry synthesis characterization",
    )
    blocked(
        "route_graph_leaf_closure_audit",
        "Route graph has no unresolved advanced leaves or fake terminal closures",
        "P0 route package remains partial_anchor and cannot assert solved closure",
        f"{target.safe} complete total synthesis semisynthesis route starting materials",
    )
    blocked(
        "hazard_regulatory_and_withdrawn_context_audit",
        "Hazard, scale, impurity, and regulatory context have been reviewed",
        "safety/regulatory review is not complete in this planning dossier",
        f"{target.safe} synthesis impurity hazard regulatory process risk",
    )

    if target.safe == "cerivastatin":
        blocked(
            "withdrawn_drug_context_guard",
            "Withdrawn-drug context is separated from route planning",
            "cerivastatin route evidence must be kept separate from use/recommendation claims",
            "cerivastatin withdrawn drug synthesis route process impurity evidence",
        )

    followup_queue = [
        {
            "requirement_id": item["requirement_id"],
            "query": item["followup_query"],
            "acceptance_signal": _closure_acceptance_signal(item["requirement_id"]),
        }
        for item in blocking_requirements
    ]
    followup_execution = _closure_followup_execution(
        target,
        row,
        followup_queue,
        execute_followups=execute_followups,
        followup_limit=followup_limit,
    )
    audit = {
        "schema_version": STATIN_ROUTE_CLOSURE_AUDIT_SCHEMA,
        "target_safe": target.safe,
        "target_name": target.name,
        "route_status": row.get("route_status"),
        "readiness_status": "not_ready_for_solved_status",
        "solved_claim_allowed": False,
        "not_lab_procedure": True,
        "passed_requirements": passed_requirements,
        "blocking_requirements": blocking_requirements,
        "automatic_followup_literature_queue": followup_queue,
        "followup_execution": followup_execution,
        "summary": {
            "passed_requirement_count": len(passed_requirements),
            "blocking_requirement_count": len(blocking_requirements),
            "followup_query_count": len(blocking_requirements),
            "route_closure_next_action": "full_text_or_curator_route_audit_before_any_solved_claim",
        },
        "closure_contract": (
            "This audit makes route-closure blockers explicit. Passing this audit means the blockers "
            "are traceable and queued, not that the route is solved."
        ),
    }
    audit["validation"] = _validate_route_closure_audit(audit)
    return audit


def _closure_followup_execution(
    target: StatinPanelTarget,
    row: dict[str, Any],
    followup_queue: list[dict[str, Any]],
    *,
    execute_followups: bool,
    followup_limit: int,
) -> dict[str, Any]:
    backend = str((row.get("literature_search_summary") or {}).get("backend_resolved") or "")
    executable = execute_followups and "pubmed" in backend
    requested_limit = int(followup_limit or 0)
    full_queue_requested = requested_limit < 0
    limit = len(followup_queue) if full_queue_requested else max(0, min(requested_limit, len(followup_queue)))
    traces: list[dict[str, Any]] = []
    if executable and limit:
        for item in followup_queue[:limit]:
            requirement_id = str(item.get("requirement_id") or "")
            cards, report, lead_sources = _retrieve_pubmed_query_evidence_until_route_relevant(
                case_id=f"{target.safe}_closure_followup",
                query=str(item.get("query") or ""),
                family_hints=[target.safe, target.name, target.family_bucket, target.expected_reaction_class],
                query_variants=_closure_followup_query_variants(target, requirement_id),
                retmax=2,
                include_abstract_signals=True,
                target=target,
                requirement_id=requirement_id,
            )
            evidence_refs = [card.evidence_id for card in cards]
            strong_lead_sources = [
                source for source in lead_sources
                if source.get("lead_relevance_status") == "route_relevant_strong"
            ]
            guarded_lead_sources = [
                source for source in lead_sources
                if source.get("route_context_guard_signals")
            ]
            abstract_signal_terms = [
                str(term)
                for term in report.get("abstract_signal_terms") or []
                if str(term).strip()
            ]
            traces.append({
                "schema_version": "statin_route_closure_followup_trace.v1",
                "requirement_id": item.get("requirement_id"),
                "query": item.get("query"),
                "query_variants": list(report.get("query_variants") or []),
                "resolved_query": str(report.get("resolved_query") or ""),
                "query_attempt_count": int(report.get("query_attempt_count") or 0),
                "fallback_used": bool(report.get("fallback_used")),
                "acceptance_signal": item.get("acceptance_signal"),
                "execution_status": (
                    "pubmed_followup_executed_with_leads"
                    if evidence_refs
                    else "pubmed_followup_executed_no_hits"
                ),
                "backend_resolved": "pubmed_followup",
                "hit_count": int(report.get("hit_count") or 0),
                "evidence_lead_refs": evidence_refs,
                "lead_sources": lead_sources,
                "route_relevant_lead_source_count": len(strong_lead_sources),
                "route_context_guarded_source_count": len(guarded_lead_sources),
                "abstract_signal_audit_requested": bool(report.get("abstract_signal_audit_requested")),
                "abstract_signal_status": str(report.get("abstract_signal_status") or ""),
                "abstract_signal_record_count": int(report.get("abstract_signal_record_count") or 0),
                "abstract_signal_hit_count": int(report.get("abstract_signal_hit_count") or 0),
                "abstract_signal_terms": abstract_signal_terms,
                "search_sources": [
                    str(search.get("source") or "")
                    for search in report.get("searches") or []
                    if search.get("source")
                ],
                "lead_quality": "external_lead_requires_full_text_or_curator_audit",
                "lead_relevance_gate": (
                    "route_relevant_strong"
                    if strong_lead_sources
                    else "lead_metadata_only_or_context_guarded"
                ),
                "not_template_support": True,
                "report": report,
            })
    else:
        status = "queued_for_pubmed_backend" if execute_followups else "queued_not_executed"
        if execute_followups and "pubmed" not in backend:
            status = "queued_backend_not_pubmed"
        for item in followup_queue[:limit or len(followup_queue)]:
            traces.append({
                "schema_version": "statin_route_closure_followup_trace.v1",
                "requirement_id": item.get("requirement_id"),
                "query": item.get("query"),
                "acceptance_signal": item.get("acceptance_signal"),
                "execution_status": status,
                "backend_resolved": backend,
                "hit_count": 0,
                "evidence_lead_refs": [],
                "lead_sources": [],
                "route_relevant_lead_source_count": 0,
                "route_context_guarded_source_count": 0,
                "search_sources": [],
                "lead_quality": "not_executed",
                "lead_relevance_gate": "not_executed",
                "not_template_support": True,
            })
    executed = [
        trace for trace in traces
        if str(trace.get("execution_status") or "").startswith("pubmed_followup_executed")
    ]
    with_leads = [
        trace for trace in traces
        if trace.get("execution_status") == "pubmed_followup_executed_with_leads"
    ]
    abstract_signal_traces = [
        trace for trace in traces
        if trace.get("abstract_signal_status") == "abstract_route_signal_detected"
    ]
    route_relevant_traces = [
        trace for trace in traces
        if int(trace.get("route_relevant_lead_source_count") or 0) > 0
    ]
    route_context_guarded_traces = [
        trace for trace in traces
        if int(trace.get("route_context_guarded_source_count") or 0) > 0
    ]
    abstract_signal_terms = sorted({
        str(term)
        for trace in traces
        for term in trace.get("abstract_signal_terms") or []
        if str(term).strip()
    })
    return {
        "schema_version": "statin_route_closure_followup_execution.v1",
        "policy": "pubmed_lead_search" if executable else "queued_only",
        "backend_resolved": backend,
        "requested": bool(execute_followups),
        "requested_limit": requested_limit,
        "resolved_limit": limit,
        "full_queue_requested": full_queue_requested,
        "full_trace_coverage": len(traces) == len(followup_queue),
        "full_execution_coverage": executable and len(executed) == len(followup_queue),
        "queue_count": len(followup_queue),
        "trace_count": len(traces),
        "executed_trace_count": len(executed),
        "lead_trace_count": len(with_leads),
        "route_relevant_trace_count": len(route_relevant_traces),
        "route_context_guarded_trace_count": len(route_context_guarded_traces),
        "abstract_signal_trace_count": len(abstract_signal_traces),
        "abstract_signal_terms": abstract_signal_terms,
        "traces": traces,
        "execution_contract": (
            "Closure follow-up traces are literature leads for blocker resolution. They do not "
            "promote templates or solved status without full-text/curator audit."
        ),
    }


def _closure_followup_lead_sources(
    cards: list[Any],
    *,
    target: StatinPanelTarget | None = None,
    requirement_id: str = "",
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for card in cards:
        evidence_ref = str(getattr(card, "evidence_id", "") or "")
        metadata = dict(getattr(card, "source_metadata", {}) or {})
        pmid = str(metadata.get("pmid") or "")
        if not pmid and evidence_ref.startswith("ev_pubmed_"):
            pmid = evidence_ref.removeprefix("ev_pubmed_")
        source_url = str(getattr(card, "url", "") or "")
        if not source_url and pmid:
            source_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        source_title = str(getattr(card, "source_title", "") or "")
        if not source_title and pmid:
            source_title = f"PubMed PMID {pmid}"
        abstract_signal = metadata.get("abstract_signal_audit") or {}
        if not isinstance(abstract_signal, dict):
            abstract_signal = {}
        abstract_signal_terms = [
            str(term)
            for term in abstract_signal.get("route_signal_terms") or []
            if str(term).strip()
        ]
        relevance = _closure_lead_route_relevance(
            source_title=source_title,
            journal=str(metadata.get("journal") or ""),
            query=str(metadata.get("query") or ""),
            abstract_signal_terms=abstract_signal_terms,
            target=target,
            requirement_id=requirement_id,
        )
        source = {
            "schema_version": "statin_closure_pubmed_lead_source.v1",
            "evidence_ref": evidence_ref,
            "source_type": "pubmed" if pmid else "literature_lead",
            "pmid": pmid,
            "source_record_id": str(getattr(card, "source_record_id", "") or (f"pubmed:{pmid}" if pmid else "")),
            "source_title": source_title,
            "source_url": source_url,
            "doi": str(getattr(card, "doi", "") or ""),
            "journal": str(metadata.get("journal") or ""),
            "pubdate": str(metadata.get("pubdate") or ""),
            "query": str(metadata.get("query") or ""),
            "abstract_signal_status": str(abstract_signal.get("route_signal_status") or ""),
            "abstract_signal_terms": abstract_signal_terms,
            "abstract_available": bool(abstract_signal.get("abstract_available")),
            "abstract_text_char_count": int(abstract_signal.get("abstract_text_char_count") or 0),
            **relevance,
            "full_text_audit_status": "pending_full_text_or_curator_audit",
            "not_template_support": True,
            "not_lab_procedure": True,
        }
        sources.append(source)
    return sources


def _closure_lead_route_relevance(
    *,
    source_title: str,
    journal: str,
    query: str,
    abstract_signal_terms: list[str],
    target: StatinPanelTarget | None,
    requirement_id: str,
) -> dict[str, Any]:
    title_text = str(source_title or "").lower()
    journal_text = str(journal or "").lower()
    abstract_terms = {str(term or "").lower() for term in abstract_signal_terms if str(term or "").strip()}
    evidence_text = " ".join([title_text, journal_text, " ".join(sorted(abstract_terms))])
    title_and_journal = " ".join([title_text, journal_text])
    target_tokens: set[str] = set()
    if target:
        target_tokens = {
            str(target.safe or "").lower(),
            str(target.name or "").lower(),
            str(target.family_bucket or "").lower().replace("_", " "),
        }

    strong_signals = sorted(
        phrase for phrase in ROUTE_RELEVANCE_STRONG_PHRASES
        if phrase in evidence_text
    )
    weak_signals = sorted(
        term for term in ROUTE_RELEVANCE_WEAK_TERMS
        if term in evidence_text and term not in strong_signals
    )
    context_guard_signals = sorted(
        guard for guard in ROUTE_RELEVANCE_CONTEXT_GUARDS
        if guard in title_and_journal
    )
    target_or_family_in_title = (
        any(token and token in title_text for token in target_tokens)
        or "statin" in title_text
        or "compactin" in title_text
    )
    route_title_anchor = any(
        phrase in title_text
        for phrase in {
            "process chemistry",
            "chemical synthesis",
            "total synthesis",
            "synthetic route",
            "synthesis of",
            "semisynthesis",
            "semi-synthesis",
            "biotransformation",
            "fermentation",
            "intermediate",
            "intermediates",
            "side chain",
            "side-chain",
            "preparation of",
            "salt preparation",
            "scale-up",
            "crystallization",
            "resolution",
        }
    )
    if strong_signals and not target_or_family_in_title and not route_title_anchor:
        context_guard_signals = sorted({*context_guard_signals, "off_target_title"})

    score = len(strong_signals) * 3
    score += len(weak_signals)
    strong_abstract_terms = abstract_terms.intersection({
        "synthesis",
        "semisynthesis",
        "semi-synthesis",
        "fermentation",
        "biotransformation",
        "intermediate",
        "intermediates",
        "process chemistry",
        "side chain",
        "side-chain",
        "scale-up",
    })
    score += len(strong_abstract_terms) * 2
    if target and any(token and token in title_text for token in target_tokens):
        score += 1
    if requirement_id == "endpoint_identity_and_salt_state_audit" and abstract_terms.intersection({"salt", "lactone"}):
        score += 1
    if requirement_id == "hazard_regulatory_and_withdrawn_context_audit" and "impurity" in abstract_terms:
        score += 1
    score -= min(len(context_guard_signals), 4) * 2

    if strong_signals and score >= 5 and not context_guard_signals:
        status = "route_relevant_strong"
    elif strong_signals and score >= 4:
        status = "route_relevant_guarded"
    elif context_guard_signals:
        status = "non_route_context_suspected"
    elif weak_signals:
        status = "weak_route_signal_only"
    else:
        status = "no_route_signal"

    return {
        "lead_relevance_schema": "statin_pubmed_lead_route_relevance.v1",
        "lead_relevance_status": status,
        "route_relevance_score": int(score),
        "route_relevance_strong_signals": strong_signals,
        "route_relevance_weak_signals": weak_signals,
        "route_context_guard_signals": context_guard_signals,
        "route_relevance_guard": (
            "route-related source candidate"
            if status == "route_relevant_strong"
            else "source requires curator triage before route use"
        ),
    }


def _closure_followup_query_variants(target: StatinPanelTarget, requirement_id: str) -> list[str]:
    safe = target.safe
    name = target.name or safe
    generic = [
        f"{safe} synthesis",
        f"{safe} synthetic route",
        f"{safe} intermediates synthesis",
        f"{name} synthesis",
    ]
    if target.family_bucket == "natural_statin":
        family_specific = [
            f"{safe} fermentation",
            f"{safe} semisynthesis",
            f"{safe} biotransformation",
            "statin fermentation semisynthesis",
        ]
    else:
        family_specific = [
            f"{safe} process synthesis",
            f"{safe} side chain synthesis",
            f"{safe} intermediate synthesis",
            "synthetic statin synthesis intermediate",
        ]
    by_requirement = {
        "condition_and_workup_evidence_audit": [
            f"{safe} synthesis conditions",
            f"{safe} process route",
        ],
        "terminal_stock_or_source_audit": [
            f"{safe} precursor synthesis",
            f"{safe} starting material intermediate",
        ],
        "endpoint_identity_and_salt_state_audit": [
            f"{safe} salt synthesis",
            f"{safe} lactone acid synthesis",
        ],
        "route_graph_leaf_closure_audit": [
            f"{safe} total synthesis",
            f"{safe} complete synthesis",
        ],
        "hazard_regulatory_and_withdrawn_context_audit": [
            f"{safe} process impurity synthesis",
            f"{safe} impurity synthesis",
        ],
        "withdrawn_drug_context_guard": [
            "cerivastatin synthesis",
            "cerivastatin process impurity",
        ],
    }
    return [
        *by_requirement.get(requirement_id, []),
        *family_specific,
        *generic,
    ]


def _flatten_trace_refs(traces: list[dict[str, Any]], key: str) -> list[str]:
    return sorted({
        str(ref)
        for trace in traces
        for ref in trace.get(key) or []
        if str(ref)
    })


def _closure_acceptance_signal(requirement_id: str) -> str:
    signals = {
        "full_text_route_step_audit": "full-text or curator-verified sequence with step/intermediate mapping",
        "condition_and_workup_evidence_audit": "condition/workup/isolation evidence tied to the member route",
        "terminal_stock_or_source_audit": "all terminal leaves have source, stock, fermentation, or precursor evidence",
        "endpoint_identity_and_salt_state_audit": "explicit product identity, stereochemistry, acid/lactone/salt state evidence",
        "route_graph_leaf_closure_audit": "no unresolved advanced leaves or fake terminal stock closures remain",
        "hazard_regulatory_and_withdrawn_context_audit": "hazard, impurity, scale, and regulatory context are documented",
        "withdrawn_drug_context_guard": "cerivastatin route planning remains separated from recommendation/use claims",
    }
    return signals.get(requirement_id, "traceable route-closure evidence")


def _validate_fullflow_dossier(dossier: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if dossier.get("schema_version") != STATIN_FULLFLOW_DOSSIER_SCHEMA:
        reasons.append("invalid_dossier_schema")
    if not (dossier.get("target") or {}).get("target_smiles"):
        reasons.append("missing_target_smiles")
    if dossier.get("route_status") == "solved":
        reasons.append("dossier_must_not_claim_solved")
    if not dossier.get("not_lab_procedure"):
        reasons.append("missing_not_lab_procedure_guard")
    if len(dossier.get("synthesis_stages") or []) < 3:
        reasons.append("insufficient_fullflow_stage_count")
    if not dossier.get("evidence_refs"):
        reasons.append("missing_evidence_refs")
    if not dossier.get("primary_template_sources"):
        reasons.append("missing_primary_template_sources")
    route_template = dossier.get("route_template") or {}
    reasons.extend(_route_template_reasons(route_template))
    closure_audit = dossier.get("route_closure_audit") or {}
    closure_validation = _validate_route_closure_audit(closure_audit)
    if not closure_validation.get("accepted"):
        reasons.extend(f"route_closure:{reason}" for reason in closure_validation.get("reasons") or [])
    blueprint = dossier.get("fullflow_blueprint") or {}
    if blueprint.get("schema_version") != "statin_member_fullflow_blueprint.v1":
        reasons.append("invalid_fullflow_blueprint_schema")
    if len(blueprint.get("member_specific_route_outline") or []) < 3:
        reasons.append("insufficient_member_specific_route_outline")
    if len(blueprint.get("key_intermediate_roles") or []) < 2:
        reasons.append("insufficient_key_intermediate_roles")
    escalation = dossier.get("automatic_literature_escalation") or {}
    if escalation.get("schema_version") != "statin_automatic_literature_escalation.v1":
        reasons.append("invalid_automatic_literature_escalation_schema")
    if len(escalation.get("difficulty_queries") or []) < 3:
        reasons.append("insufficient_automatic_literature_queries")
    query_traces = escalation.get("query_execution_traces") or []
    if len(query_traces) < len(escalation.get("difficulty_queries") or []):
        reasons.append("missing_automatic_literature_query_traces")
    trace_summary = escalation.get("query_trace_summary") or {}
    if int(trace_summary.get("accepted_query_trace_count") or 0) < len(escalation.get("difficulty_queries") or []):
        reasons.append("automatic_literature_traces_not_all_accepted")
    if int(trace_summary.get("template_supported_query_trace_count") or 0) < len(escalation.get("difficulty_queries") or []):
        reasons.append("automatic_literature_traces_missing_template_support")
    for index, trace in enumerate(query_traces, start=1):
        if trace.get("schema_version") != STATIN_LITERATURE_QUERY_TRACE_SCHEMA:
            reasons.append(f"literature_query_trace_{index}_invalid_schema")
        if not trace.get("difficulty"):
            reasons.append(f"literature_query_trace_{index}_missing_difficulty")
        if not trace.get("query"):
            reasons.append(f"literature_query_trace_{index}_missing_query")
        if trace.get("execution_status") != "covered_by_validated_literature_search":
            reasons.append(f"literature_query_trace_{index}_not_covered")
        if not trace.get("task_ref"):
            reasons.append(f"literature_query_trace_{index}_missing_task_ref")
        if not trace.get("report_ref"):
            reasons.append(f"literature_query_trace_{index}_missing_report_ref")
        if int(trace.get("hit_count") or 0) < 1:
            reasons.append(f"literature_query_trace_{index}_missing_hits")
        if int(trace.get("validated_evidence_count") or 0) < 1:
            reasons.append(f"literature_query_trace_{index}_missing_validated_evidence")
        if not trace.get("supporting_evidence_refs"):
            reasons.append(f"literature_query_trace_{index}_missing_supporting_evidence_refs")
        if not trace.get("template_supporting_evidence_refs"):
            reasons.append(f"literature_query_trace_{index}_missing_template_supporting_evidence_refs")
        if trace.get("quality_gate_status") == "lead_only_requires_full_text_or_curator_audit":
            reasons.append(f"literature_query_trace_{index}_lead_only_not_template_promotable")
    quality = dossier.get("template_quality") or {}
    if not quality.get("expected_template_hit"):
        reasons.append("expected_template_missing")
    if not quality.get("required_candidate_kinds_hit"):
        reasons.append("required_candidate_kinds_missing")
    if quality.get("warnings"):
        reasons.append("dossier_has_template_warnings")
    return {
        "schema_version": "statin_fullflow_dossier_validation.v1",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
    }


def validate_statin_fullflow_dossier(dossier: dict[str, Any]) -> dict[str, Any]:
    """Validate a statin fullflow dossier as a typed planning artifact."""
    return _validate_fullflow_dossier(dict(dossier or {}))


def validate_statin_fullflow_overview(overview: dict[str, Any]) -> dict[str, Any]:
    """Validate the nine-statin delivery overview as a typed planning artifact."""
    return _validate_fullflow_overview(dict(overview or {}))


def validate_statin_route_template(route_template: dict[str, Any]) -> dict[str, Any]:
    reasons = _route_template_reasons(dict(route_template or {}))
    return {
        "schema_version": "statin_member_route_template_validation.v1",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
    }


def validate_statin_route_closure_audit(audit: dict[str, Any]) -> dict[str, Any]:
    return _validate_route_closure_audit(dict(audit or {}))


def validate_statin_route_closure_matrix(matrix: dict[str, Any]) -> dict[str, Any]:
    return _validate_route_closure_matrix(dict(matrix or {}))


def validate_statin_closure_lead_curation_packet(packet: dict[str, Any]) -> dict[str, Any]:
    return _validate_closure_lead_curation_packet(dict(packet or {}))


def validate_statin_closure_curation_result_set(result_set: dict[str, Any]) -> dict[str, Any]:
    return _validate_closure_curation_result_set(dict(result_set or {}))


def valid_statin_field_resolution_candidate_status(status: str) -> bool:
    return str(status or "") in STATIN_FIELD_RESOLUTION_CANDIDATE_STATUSES


def _validate_route_closure_audit(audit: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if audit.get("schema_version") != STATIN_ROUTE_CLOSURE_AUDIT_SCHEMA:
        reasons.append("invalid_route_closure_audit_schema")
    if not audit.get("target_safe"):
        reasons.append("route_closure_missing_target_safe")
    if audit.get("route_status") == "solved":
        reasons.append("route_closure_must_not_start_from_solved_status")
    if audit.get("readiness_status") != "not_ready_for_solved_status":
        reasons.append("route_closure_invalid_readiness_status")
    if audit.get("solved_claim_allowed"):
        reasons.append("route_closure_unproven_solved_claim_allowed")
    if not audit.get("not_lab_procedure"):
        reasons.append("route_closure_missing_not_lab_procedure_guard")
    passed = audit.get("passed_requirements") or []
    blockers = audit.get("blocking_requirements") or []
    queue = audit.get("automatic_followup_literature_queue") or []
    if len(passed) < 3:
        reasons.append("route_closure_insufficient_passed_preconditions")
    if len(blockers) < 4:
        reasons.append("route_closure_missing_blockers")
    if len(queue) < len(blockers):
        reasons.append("route_closure_followup_queue_incomplete")
    execution = audit.get("followup_execution") or {}
    if execution.get("schema_version") != "statin_route_closure_followup_execution.v1":
        reasons.append("route_closure_invalid_followup_execution_schema")
    if int(execution.get("queue_count") or 0) != len(queue):
        reasons.append("route_closure_followup_execution_queue_count_mismatch")
    if execution.get("policy") not in {"queued_only", "pubmed_lead_search"}:
        reasons.append("route_closure_invalid_followup_execution_policy")
    if int(execution.get("resolved_limit") or 0) > len(queue):
        reasons.append("route_closure_followup_execution_limit_exceeds_queue")
    if execution.get("full_queue_requested"):
        if int(execution.get("trace_count") or 0) != len(queue):
            reasons.append("route_closure_followup_full_queue_missing_trace")
        if execution.get("policy") == "pubmed_lead_search" and int(execution.get("executed_trace_count") or 0) != len(queue):
            reasons.append("route_closure_followup_full_queue_missing_execution")
    if execution.get("policy") == "pubmed_lead_search":
        if int(execution.get("executed_trace_count") or 0) < 1:
            reasons.append("route_closure_followup_execution_missing_executed_trace")
    for index, item in enumerate(passed, start=1):
        if not item.get("requirement_id"):
            reasons.append(f"route_closure_passed_{index}_missing_requirement_id")
        if item.get("status") != "passed":
            reasons.append(f"route_closure_passed_{index}_invalid_status")
        if not item.get("evidence_refs"):
            reasons.append(f"route_closure_passed_{index}_missing_evidence_refs")
    for index, item in enumerate(blockers, start=1):
        if not item.get("requirement_id"):
            reasons.append(f"route_closure_blocker_{index}_missing_requirement_id")
        if item.get("status") != "blocked":
            reasons.append(f"route_closure_blocker_{index}_invalid_status")
        if not item.get("blocker"):
            reasons.append(f"route_closure_blocker_{index}_missing_blocker")
        if not item.get("followup_query"):
            reasons.append(f"route_closure_blocker_{index}_missing_followup_query")
    for index, item in enumerate(queue, start=1):
        if not item.get("requirement_id"):
            reasons.append(f"route_closure_queue_{index}_missing_requirement_id")
        if not item.get("query"):
            reasons.append(f"route_closure_queue_{index}_missing_query")
        if not item.get("acceptance_signal"):
            reasons.append(f"route_closure_queue_{index}_missing_acceptance_signal")
    for index, trace in enumerate(execution.get("traces") or [], start=1):
        if trace.get("schema_version") != "statin_route_closure_followup_trace.v1":
            reasons.append(f"route_closure_followup_trace_{index}_invalid_schema")
        if not trace.get("requirement_id"):
            reasons.append(f"route_closure_followup_trace_{index}_missing_requirement_id")
        if not trace.get("query"):
            reasons.append(f"route_closure_followup_trace_{index}_missing_query")
        if not trace.get("execution_status"):
            reasons.append(f"route_closure_followup_trace_{index}_missing_execution_status")
        if not trace.get("not_template_support"):
            reasons.append(f"route_closure_followup_trace_{index}_must_not_be_template_support")
        lead_sources = [source for source in trace.get("lead_sources") or [] if isinstance(source, dict)]
        if int(trace.get("route_relevant_lead_source_count") or 0) > len(lead_sources):
            reasons.append(f"route_closure_followup_trace_{index}_route_relevant_count_exceeds_sources")
        if int(trace.get("route_context_guarded_source_count") or 0) > len(lead_sources):
            reasons.append(f"route_closure_followup_trace_{index}_context_guard_count_exceeds_sources")
        for source_index, source in enumerate(lead_sources, start=1):
            if source.get("lead_relevance_schema") != "statin_pubmed_lead_route_relevance.v1":
                reasons.append(f"route_closure_followup_trace_{index}_source_{source_index}_missing_relevance_schema")
            if source.get("lead_relevance_status") not in {
                "route_relevant_strong",
                "route_relevant_guarded",
                "weak_route_signal_only",
                "non_route_context_suspected",
                "no_route_signal",
            }:
                reasons.append(f"route_closure_followup_trace_{index}_source_{source_index}_invalid_relevance_status")
            if "route_relevance_score" not in source:
                reasons.append(f"route_closure_followup_trace_{index}_source_{source_index}_missing_relevance_score")
            if not source.get("route_relevance_guard"):
                reasons.append(f"route_closure_followup_trace_{index}_source_{source_index}_missing_relevance_guard")
        if str(trace.get("execution_status") or "").startswith("pubmed_followup_executed"):
            if "pubmed_followup_esearch" not in set(trace.get("search_sources") or []):
                reasons.append(f"route_closure_followup_trace_{index}_missing_pubmed_esearch")
            if int(trace.get("query_attempt_count") or 0) < 1:
                reasons.append(f"route_closure_followup_trace_{index}_missing_query_attempt_count")
            if (
                trace.get("execution_status") == "pubmed_followup_executed_with_leads"
                and int(trace.get("hit_count") or 0) < 1
            ):
                reasons.append(f"route_closure_followup_trace_{index}_missing_hits")
            if trace.get("execution_status") == "pubmed_followup_executed_with_leads" and not trace.get("resolved_query"):
                reasons.append(f"route_closure_followup_trace_{index}_missing_resolved_query")
            if not trace.get("abstract_signal_audit_requested"):
                reasons.append(f"route_closure_followup_trace_{index}_missing_abstract_signal_audit_request")
            if not trace.get("abstract_signal_status"):
                reasons.append(f"route_closure_followup_trace_{index}_missing_abstract_signal_status")
            if trace.get("abstract_signal_status") == "abstract_route_signal_detected":
                if int(trace.get("abstract_signal_hit_count") or 0) < 1:
                    reasons.append(f"route_closure_followup_trace_{index}_missing_abstract_signal_hit")
                if not trace.get("abstract_signal_terms"):
                    reasons.append(f"route_closure_followup_trace_{index}_missing_abstract_signal_terms")
            for audit_row in ((trace.get("report") or {}).get("abstract_signal_audit") or []):
                if "abstract_text" in audit_row:
                    reasons.append(f"route_closure_followup_trace_{index}_stored_abstract_text")
    return {
        "schema_version": "statin_route_closure_audit_validation.v1",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
    }


def _validate_route_closure_matrix(matrix: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if matrix.get("schema_version") != STATIN_ROUTE_CLOSURE_MATRIX_SCHEMA:
        reasons.append("invalid_route_closure_matrix_schema")
    if matrix.get("skipped"):
        return {
            "schema_version": "statin_route_closure_matrix_validation.v1",
            "accepted": not reasons,
            "reasons": sorted(set(reasons)),
        }
    rows = list(matrix.get("rows") or [])
    targets = {str(row.get("target_safe") or "") for row in rows if row.get("target_safe")}
    if int(matrix.get("target_count") or 0) != 9:
        reasons.append("route_closure_matrix_not_all_nine_targets")
    if len(targets) != 9:
        reasons.append("route_closure_matrix_rows_not_all_nine_targets")
    if int(matrix.get("blocker_count") or 0) != len(rows):
        reasons.append("route_closure_matrix_blocker_count_mismatch")
    if len(rows) < 9 * 4:
        reasons.append("route_closure_matrix_insufficient_blocker_rows")
    if int(matrix.get("queued_blocker_count") or 0) != len(rows):
        reasons.append("route_closure_matrix_not_all_blockers_queued")
    if bool(matrix.get("full_trace_coverage")) != (
        bool(rows) and int(matrix.get("trace_link_count") or 0) == len(rows)
    ):
        reasons.append("route_closure_matrix_full_trace_coverage_mismatch")
    if bool(matrix.get("full_execution_coverage")) != (
        bool(rows) and int(matrix.get("executed_trace_count") or 0) == len(rows)
    ):
        reasons.append("route_closure_matrix_full_execution_coverage_mismatch")
    route_relevant_trace_count = sum(
        1 for row in rows
        if int(row.get("route_relevant_source_count") or 0) > 0
    )
    route_context_guarded_trace_count = sum(
        1 for row in rows
        if int(row.get("route_context_guarded_source_count") or 0) > 0
    )
    if int(matrix.get("route_relevant_trace_count") or 0) != route_relevant_trace_count:
        reasons.append("route_closure_matrix_route_relevant_count_mismatch")
    if int(matrix.get("route_context_guarded_trace_count") or 0) != route_context_guarded_trace_count:
        reasons.append("route_closure_matrix_context_guard_count_mismatch")
    if int(matrix.get("route_relevant_trace_count") or 0) > int(matrix.get("lead_trace_count") or 0):
        reasons.append("route_closure_matrix_route_relevant_count_exceeds_leads")
    if int(matrix.get("route_context_guarded_trace_count") or 0) > int(matrix.get("lead_trace_count") or 0):
        reasons.append("route_closure_matrix_context_guard_count_exceeds_leads")
    if int(matrix.get("unresolved_blocker_count") or 0) != len(rows):
        reasons.append("route_closure_matrix_unresolved_count_mismatch")
    if int(matrix.get("solved_claim_allowed_count") or 0) != 0:
        reasons.append("route_closure_matrix_allows_solved_claim")
    for index, row in enumerate(rows, start=1):
        if row.get("schema_version") != "statin_route_closure_matrix_row.v1":
            reasons.append(f"route_closure_matrix_row_{index}_invalid_schema")
        if not row.get("target_safe"):
            reasons.append(f"route_closure_matrix_row_{index}_missing_target")
        if not row.get("requirement_id"):
            reasons.append(f"route_closure_matrix_row_{index}_missing_requirement_id")
        if row.get("closure_status") != "blocked":
            reasons.append(f"route_closure_matrix_row_{index}_not_blocked")
        if not row.get("blocker"):
            reasons.append(f"route_closure_matrix_row_{index}_missing_blocker")
        if not row.get("followup_query"):
            reasons.append(f"route_closure_matrix_row_{index}_missing_followup_query")
        if not row.get("acceptance_signal"):
            reasons.append(f"route_closure_matrix_row_{index}_missing_acceptance_signal")
        if not row.get("queue_present"):
            reasons.append(f"route_closure_matrix_row_{index}_missing_queue")
        if row.get("solved_claim_allowed"):
            reasons.append(f"route_closure_matrix_row_{index}_allows_solved_claim")
        if not row.get("next_action"):
            reasons.append(f"route_closure_matrix_row_{index}_missing_next_action")
        if row.get("trace_present") and not row.get("not_template_support"):
            reasons.append(f"route_closure_matrix_row_{index}_trace_promotes_template")
        lead_sources = [source for source in row.get("lead_sources") or [] if isinstance(source, dict)]
        if int(row.get("route_relevant_source_count") or 0) > len(lead_sources):
            reasons.append(f"route_closure_matrix_row_{index}_route_relevant_count_exceeds_sources")
        if int(row.get("route_context_guarded_source_count") or 0) > len(lead_sources):
            reasons.append(f"route_closure_matrix_row_{index}_context_guard_count_exceeds_sources")
        for source_index, source in enumerate(lead_sources, start=1):
            if source.get("lead_relevance_schema") != "statin_pubmed_lead_route_relevance.v1":
                reasons.append(f"route_closure_matrix_row_{index}_source_{source_index}_missing_relevance_schema")
            if source.get("lead_relevance_status") not in {
                "route_relevant_strong",
                "route_relevant_guarded",
                "weak_route_signal_only",
                "non_route_context_suspected",
                "no_route_signal",
            }:
                reasons.append(f"route_closure_matrix_row_{index}_source_{source_index}_invalid_relevance_status")
            if "route_relevance_score" not in source:
                reasons.append(f"route_closure_matrix_row_{index}_source_{source_index}_missing_relevance_score")
            if not source.get("route_relevance_guard"):
                reasons.append(f"route_closure_matrix_row_{index}_source_{source_index}_missing_relevance_guard")
        if str(row.get("execution_status") or "").startswith("pubmed_followup_executed"):
            if int(row.get("query_attempt_count") or 0) < 1:
                reasons.append(f"route_closure_matrix_row_{index}_missing_query_attempt_count")
            if "pubmed_followup_esearch" not in set(row.get("search_sources") or []):
                reasons.append(f"route_closure_matrix_row_{index}_missing_pubmed_esearch")
            if not row.get("abstract_signal_status"):
                reasons.append(f"route_closure_matrix_row_{index}_missing_abstract_signal_status")
            if row.get("abstract_signal_status") == "abstract_route_signal_detected":
                if int(row.get("abstract_signal_hit_count") or 0) < 1:
                    reasons.append(f"route_closure_matrix_row_{index}_missing_abstract_signal_hit")
                if not row.get("abstract_signal_terms"):
                    reasons.append(f"route_closure_matrix_row_{index}_missing_abstract_signal_terms")
    return {
        "schema_version": "statin_route_closure_matrix_validation.v1",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
    }


def _validate_closure_curation_result_set(result_set: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if result_set.get("schema_version") != STATIN_CLOSURE_CURATION_RESULT_SET_SCHEMA:
        reasons.append("invalid_closure_curation_result_set_schema")
    if result_set.get("skipped"):
        return {
            "schema_version": "statin_closure_curation_result_set_validation.v1",
            "accepted": not reasons,
            "reasons": sorted(set(reasons)),
        }
    results = list(result_set.get("results") or [])
    targets = {str(result.get("target_safe") or "") for result in results if result.get("target_safe")}
    if int(result_set.get("target_count") or 0) != 9:
        reasons.append("closure_curation_result_set_not_all_nine_targets")
    if len(targets) != 9:
        reasons.append("closure_curation_result_set_results_not_all_nine_targets")
    if int(result_set.get("task_count") or 0) != len(results):
        reasons.append("closure_curation_result_set_task_count_mismatch")
    if int(result_set.get("result_count") or 0) != len(results):
        reasons.append("closure_curation_result_set_result_count_mismatch")
    if int(result_set.get("template_promotion_allowed_count") or 0) != 0:
        reasons.append("closure_curation_result_set_allows_template_promotion")
    if int(result_set.get("solved_claim_allowed_count") or 0) != 0:
        reasons.append("closure_curation_result_set_allows_solved_claim")
    if result_set.get("candidate_template_gate_status") != "blocked_pending_full_text_or_curator_records":
        reasons.append("closure_curation_result_set_invalid_candidate_gate_status")
    if not result_set.get("production_write_blocked"):
        reasons.append("closure_curation_result_set_production_write_not_blocked")
    if not result_set.get("not_lab_procedure"):
        reasons.append("closure_curation_result_set_missing_not_lab_procedure_guard")
    execution_summary = result_set.get("open_gap_search_execution") or {}
    if execution_summary.get("schema_version") != "statin_open_gap_search_execution_summary.v1":
        reasons.append("closure_curation_result_set_invalid_open_gap_search_execution_summary")
    if execution_summary.get("policy") not in {"queued_only", "pubmed_open_gap_field_search"}:
        reasons.append("closure_curation_result_set_invalid_open_gap_search_policy")
    access_execution_summary = result_set.get("open_gap_full_text_access_execution") or {}
    if access_execution_summary.get("schema_version") != "statin_open_gap_full_text_access_execution_summary.v1":
        reasons.append("closure_curation_result_set_invalid_full_text_access_execution_summary")
    if access_execution_summary.get("policy") not in {
        "queued_only",
        "ncbi_pubmed_to_pmc_access_metadata_probe",
    }:
        reasons.append("closure_curation_result_set_invalid_full_text_access_policy")

    lead_backed_count = sum(1 for result in results if result.get("evidence_lead_refs"))
    route_relevant_count = sum(
        1 for result in results
        if int((result.get("source_selection_summary") or {}).get("selected_route_source_count") or 0) > 0
    )
    context_guarded_count = sum(
        1 for result in results
        if int((result.get("source_selection_summary") or {}).get("context_guarded_source_count") or 0) > 0
    )
    needs_better_count = sum(
        1 for result in results
        if result.get("curation_result_status") == "needs_better_route_specific_leads_or_manual_curator_source"
    )
    curator_record_supported_count = sum(
        1 for result in results
        if int(result.get("validated_route_field_count") or 0) > 0
    )
    validated_route_field_count = sum(
        int(result.get("validated_route_field_count") or 0)
        for result in results
    )
    audited_gap_route_field_count = sum(
        int(result.get("audited_gap_route_field_count") or 0)
        for result in results
    )
    missing_route_field_count = sum(
        int(result.get("missing_route_field_count") or 0)
        for result in results
    )
    open_route_field_count = sum(
        int(result.get("open_route_field_count") or 0)
        for result in results
    )
    open_gap_followup_count = sum(
        len(result.get("open_gap_followup_tasks") or [])
        for result in results
    )
    open_gap_review_ready_count = sum(
        1
        for result in results
        for followup in result.get("open_gap_followup_tasks") or []
        if (followup.get("literature_triage") or {}).get("triage_status") == "selected_source_review_ready"
    )
    open_gap_search_required_count = sum(
        1
        for result in results
        for followup in result.get("open_gap_followup_tasks") or []
        if (followup.get("literature_triage") or {}).get("triage_status") == "route_specific_search_required"
    )
    open_gap_curator_review_draft_count = sum(
        1
        for result in results
        for followup in result.get("open_gap_followup_tasks") or []
        if (followup.get("literature_triage") or {}).get("curator_review_draft")
    )
    open_gap_selected_source_review_draft_count = sum(
        1
        for result in results
        for followup in result.get("open_gap_followup_tasks") or []
        if (
            (followup.get("literature_triage") or {}).get("curator_review_draft") or {}
        ).get("draft_status") == "metadata_ready_pending_full_text_or_curator_confirmation"
    )
    open_gap_search_execution_package_count = sum(
        1
        for result in results
        for followup in result.get("open_gap_followup_tasks") or []
        if (followup.get("literature_triage") or {}).get("search_execution_package")
    )
    open_gap_search_trace_count = sum(
        1
        for result in results
        for followup in result.get("open_gap_followup_tasks") or []
        if (
            (followup.get("literature_triage") or {})
            .get("search_execution_package") or {}
        ).get("execution_trace")
    )
    open_gap_search_executed_count = sum(
        1
        for result in results
        for followup in result.get("open_gap_followup_tasks") or []
        if str(
            (
                (
                    (followup.get("literature_triage") or {})
                    .get("search_execution_package") or {}
                ).get("execution_trace") or {}
            ).get("execution_status") or ""
        ).startswith("pubmed_open_gap_search_executed")
    )
    open_gap_search_lead_count = sum(
        1
        for result in results
        for followup in result.get("open_gap_followup_tasks") or []
        if int(
            (
                (
                    (followup.get("literature_triage") or {})
                    .get("search_execution_package") or {}
                ).get("execution_trace") or {}
            ).get("hit_count") or 0
        ) > 0
    )
    open_gap_search_selected_source_count = sum(
        1
        for result in results
        for followup in result.get("open_gap_followup_tasks") or []
        if (
            (
                (
                    (followup.get("literature_triage") or {})
                    .get("search_execution_package") or {}
                ).get("execution_trace") or {}
            ).get("selected_route_source_refs") or []
        )
    )
    open_gap_full_text_access_probe_count = len(_open_gap_full_text_access_probe_rows(results))
    open_gap_full_text_access_executed_count = sum(
        1
        for probe in _open_gap_full_text_access_probe_rows(results)
        if str(probe.get("execution_status") or "").startswith("full_text_access_probe_executed")
    )
    open_gap_full_text_access_executed_or_failed_count = sum(
        1
        for probe in _open_gap_full_text_access_probe_rows(results)
        if str(probe.get("execution_status") or "") in {
            "full_text_access_probe_executed",
            "full_text_access_probe_failed",
        }
    )
    open_gap_full_text_access_candidate_count = _open_gap_full_text_access_candidate_count(results)
    open_gap_resolution_candidate_count = _open_gap_field_resolution_candidate_count(results)
    open_gap_self_evo_inbox_count = sum(
        1
        for result in results
        for followup in result.get("open_gap_followup_tasks") or []
        if followup.get("self_evo_inbox_entry")
    )
    full_text_required_count = sum(
        1 for result in results
        if result.get("curation_result_status") in {
            "awaiting_full_text_route_extraction",
            "partial_curator_record_applied_pending_remaining_audit",
        }
    )
    blocked_count = sum(
        1 for result in results
        if result.get("candidate_template_gate_status") != "promotion_allowed"
    )
    if int(result_set.get("lead_backed_result_count") or 0) != lead_backed_count:
        reasons.append("closure_curation_result_set_lead_backed_count_mismatch")
    if int(result_set.get("route_relevant_result_count") or 0) != route_relevant_count:
        reasons.append("closure_curation_result_set_route_relevant_count_mismatch")
    if int(result_set.get("context_guarded_result_count") or 0) != context_guarded_count:
        reasons.append("closure_curation_result_set_context_guarded_count_mismatch")
    if int(result_set.get("needs_better_lead_count") or 0) != needs_better_count:
        reasons.append("closure_curation_result_set_needs_better_count_mismatch")
    if int(result_set.get("curator_record_supported_result_count") or 0) != curator_record_supported_count:
        reasons.append("closure_curation_result_set_curator_record_supported_count_mismatch")
    if int(result_set.get("validated_route_field_count") or 0) != validated_route_field_count:
        reasons.append("closure_curation_result_set_validated_route_field_count_mismatch")
    if int(result_set.get("audited_gap_route_field_count") or 0) != audited_gap_route_field_count:
        reasons.append("closure_curation_result_set_audited_gap_route_field_count_mismatch")
    if int(result_set.get("missing_route_field_count") or 0) != missing_route_field_count:
        reasons.append("closure_curation_result_set_missing_route_field_count_mismatch")
    if int(result_set.get("open_route_field_count") or 0) != open_route_field_count:
        reasons.append("closure_curation_result_set_open_route_field_count_mismatch")
    if int(result_set.get("open_gap_followup_count") or 0) != open_gap_followup_count:
        reasons.append("closure_curation_result_set_open_gap_followup_count_mismatch")
    if int(result_set.get("open_gap_review_ready_count") or 0) != open_gap_review_ready_count:
        reasons.append("closure_curation_result_set_open_gap_review_ready_count_mismatch")
    if int(result_set.get("open_gap_search_required_count") or 0) != open_gap_search_required_count:
        reasons.append("closure_curation_result_set_open_gap_search_required_count_mismatch")
    if int(result_set.get("open_gap_curator_review_draft_count") or 0) != open_gap_curator_review_draft_count:
        reasons.append("closure_curation_result_set_open_gap_curator_review_draft_count_mismatch")
    if int(result_set.get("open_gap_selected_source_review_draft_count") or 0) != open_gap_selected_source_review_draft_count:
        reasons.append("closure_curation_result_set_open_gap_selected_source_review_draft_count_mismatch")
    if int(result_set.get("open_gap_search_execution_package_count") or 0) != open_gap_search_execution_package_count:
        reasons.append("closure_curation_result_set_open_gap_search_execution_package_count_mismatch")
    if int(result_set.get("open_gap_search_trace_count") or 0) != open_gap_search_trace_count:
        reasons.append("closure_curation_result_set_open_gap_search_trace_count_mismatch")
    if int(result_set.get("open_gap_search_executed_count") or 0) != open_gap_search_executed_count:
        reasons.append("closure_curation_result_set_open_gap_search_executed_count_mismatch")
    if int(result_set.get("open_gap_search_lead_count") or 0) != open_gap_search_lead_count:
        reasons.append("closure_curation_result_set_open_gap_search_lead_count_mismatch")
    if int(result_set.get("open_gap_search_selected_source_count") or 0) != open_gap_search_selected_source_count:
        reasons.append("closure_curation_result_set_open_gap_search_selected_source_count_mismatch")
    if int(result_set.get("open_gap_full_text_access_probe_count") or 0) != open_gap_full_text_access_probe_count:
        reasons.append("closure_curation_result_set_open_gap_full_text_access_probe_count_mismatch")
    if int(result_set.get("open_gap_full_text_access_executed_count") or 0) != open_gap_full_text_access_executed_count:
        reasons.append("closure_curation_result_set_open_gap_full_text_access_executed_count_mismatch")
    if int(result_set.get("open_gap_full_text_access_candidate_count") or 0) != open_gap_full_text_access_candidate_count:
        reasons.append("closure_curation_result_set_open_gap_full_text_access_candidate_count_mismatch")
    if int(result_set.get("open_gap_resolution_candidate_count") or 0) != open_gap_resolution_candidate_count:
        reasons.append("closure_curation_result_set_open_gap_resolution_candidate_count_mismatch")
    if int(result_set.get("open_gap_self_evo_inbox_count") or 0) != open_gap_self_evo_inbox_count:
        reasons.append("closure_curation_result_set_open_gap_self_evo_inbox_count_mismatch")
    if open_gap_review_ready_count + open_gap_search_required_count != open_gap_followup_count:
        reasons.append("closure_curation_result_set_open_gap_triage_count_mismatch")
    if open_gap_curator_review_draft_count != open_gap_followup_count:
        reasons.append("closure_curation_result_set_open_gap_review_draft_coverage_mismatch")
    if open_gap_search_execution_package_count != open_gap_followup_count:
        reasons.append("closure_curation_result_set_open_gap_search_package_coverage_mismatch")
    if open_gap_search_trace_count != open_gap_followup_count:
        reasons.append("closure_curation_result_set_open_gap_search_trace_coverage_mismatch")
    executed_or_failed_trace_count = sum(
        1
        for result in results
        for followup in result.get("open_gap_followup_tasks") or []
        if str(
            (
                (
                    (followup.get("literature_triage") or {})
                    .get("search_execution_package") or {}
                ).get("execution_trace") or {}
            ).get("execution_status") or ""
        ) in {
            "pubmed_open_gap_search_executed_with_leads",
            "pubmed_open_gap_search_executed_no_hits",
            "pubmed_open_gap_search_failed",
        }
    )
    if int(execution_summary.get("open_gap_followup_count") or 0) != open_gap_followup_count:
        reasons.append("closure_curation_result_set_open_gap_search_execution_followup_count_mismatch")
    if int(execution_summary.get("candidate_search_required_count") or 0) > open_gap_followup_count:
        reasons.append("closure_curation_result_set_open_gap_search_execution_candidate_count_invalid")
    if int(execution_summary.get("runnable_search_required_count") or 0) > int(execution_summary.get("candidate_search_required_count") or 0):
        reasons.append("closure_curation_result_set_open_gap_search_execution_runnable_count_invalid")
    if int(execution_summary.get("carried_forward_search_trace_count") or 0) > executed_or_failed_trace_count:
        reasons.append("closure_curation_result_set_open_gap_search_execution_carry_forward_invalid")
    if int(result_set.get("open_gap_carried_forward_search_trace_count") or 0) != int(execution_summary.get("carried_forward_search_trace_count") or 0):
        reasons.append("closure_curation_result_set_open_gap_search_execution_carry_forward_count_mismatch")
    if int(execution_summary.get("executed_count") or 0) > int(execution_summary.get("resolved_limit") or 0):
        reasons.append("closure_curation_result_set_open_gap_search_execution_executed_exceeds_limit")
    if int(access_execution_summary.get("review_ready_followup_count") or 0) != open_gap_review_ready_count:
        reasons.append("closure_curation_result_set_full_text_access_review_ready_count_mismatch")
    if int(access_execution_summary.get("runnable_probe_followup_count") or 0) > open_gap_review_ready_count:
        reasons.append("closure_curation_result_set_full_text_access_runnable_count_invalid")
    if int(access_execution_summary.get("carried_forward_probe_count") or 0) > open_gap_full_text_access_executed_or_failed_count:
        reasons.append("closure_curation_result_set_full_text_access_carry_forward_invalid")
    if int(result_set.get("open_gap_carried_forward_full_text_access_probe_count") or 0) != int(access_execution_summary.get("carried_forward_probe_count") or 0):
        reasons.append("closure_curation_result_set_full_text_access_carry_forward_count_mismatch")
    if int(result_set.get("full_text_extraction_required_count") or 0) != full_text_required_count:
        reasons.append("closure_curation_result_set_full_text_required_count_mismatch")
    if int(result_set.get("blocked_result_count") or 0) != blocked_count:
        reasons.append("closure_curation_result_set_blocked_count_mismatch")

    result_ids: set[str] = set()
    for index, result in enumerate(results, start=1):
        result_id = str(result.get("result_id") or "")
        if result.get("schema_version") != "statin_closure_curation_result.v1":
            reasons.append(f"closure_curation_result_{index}_invalid_schema")
        if not result_id:
            reasons.append(f"closure_curation_result_{index}_missing_result_id")
        elif result_id in result_ids:
            reasons.append(f"closure_curation_result_{index}_duplicate_result_id")
        result_ids.add(result_id)
        if not result.get("task_id"):
            reasons.append(f"closure_curation_result_{index}_missing_task_id")
        if not result.get("target_safe"):
            reasons.append(f"closure_curation_result_{index}_missing_target")
        if not result.get("requirement_id"):
            reasons.append(f"closure_curation_result_{index}_missing_requirement_id")
        if result.get("curation_result_status") not in {
            "awaiting_full_text_route_extraction",
            "needs_better_route_specific_leads_or_manual_curator_source",
            "queued_without_literature_leads",
            "partial_curator_record_applied_pending_remaining_audit",
            "local_curator_record_applied_pending_promotion_review",
        }:
            reasons.append(f"closure_curation_result_{index}_invalid_status")
        if result.get("candidate_template_gate_status") != "blocked_pending_full_text_or_curator_records":
            reasons.append(f"closure_curation_result_{index}_invalid_candidate_gate_status")
        if result.get("template_promotion_allowed"):
            reasons.append(f"closure_curation_result_{index}_allows_template_promotion")
        if result.get("solved_claim_allowed"):
            reasons.append(f"closure_curation_result_{index}_allows_solved_claim")
        if not result.get("not_template_support"):
            reasons.append(f"closure_curation_result_{index}_must_not_be_template_support")
        if not result.get("not_lab_procedure"):
            reasons.append(f"closure_curation_result_{index}_missing_not_lab_procedure_guard")
        selection = result.get("source_selection_summary") or {}
        if selection.get("schema_version") != "statin_closure_source_selection_summary.v1":
            reasons.append(f"closure_curation_result_{index}_invalid_source_selection_summary")
        selected_count = int(selection.get("selected_route_source_count") or 0)
        source_count = int(selection.get("source_count") or 0)
        context_count = int(selection.get("context_guarded_source_count") or 0)
        weak_count = int(selection.get("weak_or_metadata_only_source_count") or 0)
        if selected_count > source_count:
            reasons.append(f"closure_curation_result_{index}_selected_source_count_exceeds_sources")
        if context_count > source_count:
            reasons.append(f"closure_curation_result_{index}_context_guard_count_exceeds_sources")
        if weak_count > source_count:
            reasons.append(f"closure_curation_result_{index}_weak_source_count_exceeds_sources")
        if selected_count != len(selection.get("selected_route_source_refs") or []):
            reasons.append(f"closure_curation_result_{index}_selected_source_ref_count_mismatch")
        if context_count != len(selection.get("context_guarded_source_refs") or []):
            reasons.append(f"closure_curation_result_{index}_context_guard_ref_count_mismatch")
        required_fields = [str(field) for field in result.get("required_route_fields") or [] if str(field).strip()]
        route_field_audit = [row for row in result.get("route_field_audit") or [] if isinstance(row, dict)]
        validated_fields = [
            row for row in route_field_audit
            if row.get("status") in {"validated_local_curator_record", "not_applicable_local_curator_record"}
        ]
        audited_gap_fields = [
            row for row in route_field_audit
            if row.get("status") == "audited_gap_local_curator_record"
        ]
        missing_fields = [
            row for row in route_field_audit
            if row.get("status") == "missing_full_text_or_curator_record"
        ]
        open_fields = [
            row for row in route_field_audit
            if row.get("resolution_required_before_promotion")
        ]
        if not required_fields:
            reasons.append(f"closure_curation_result_{index}_missing_required_route_fields")
        if int(result.get("validated_route_field_count") or 0) != len(validated_fields):
            reasons.append(f"closure_curation_result_{index}_validated_route_field_count_mismatch")
        if int(result.get("audited_gap_route_field_count") or 0) != len(audited_gap_fields):
            reasons.append(f"closure_curation_result_{index}_audited_gap_route_field_count_mismatch")
        if int(result.get("missing_route_field_count") or 0) != len(missing_fields):
            reasons.append(f"closure_curation_result_{index}_missing_route_field_count_mismatch")
        if int(result.get("open_route_field_count") or 0) != len(open_fields):
            reasons.append(f"closure_curation_result_{index}_open_route_field_count_mismatch")
        if len(route_field_audit) < len(required_fields):
            reasons.append(f"closure_curation_result_{index}_route_field_audit_incomplete")
        open_field_names = {str(row.get("field") or "") for row in open_fields if row.get("field")}
        open_gap_followups = [row for row in result.get("open_gap_followup_tasks") or [] if isinstance(row, dict)]
        followup_field_names = {str(row.get("field") or "") for row in open_gap_followups if row.get("field")}
        if len(open_gap_followups) != len(open_fields):
            reasons.append(f"closure_curation_result_{index}_open_gap_followup_count_mismatch")
        if followup_field_names != open_field_names:
            reasons.append(f"closure_curation_result_{index}_open_gap_followup_field_mismatch")
        for followup_index, followup in enumerate(open_gap_followups, start=1):
            field = str(followup.get("field") or "")
            if followup.get("schema_version") != "statin_closure_open_gap_followup_task.v1":
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_invalid_schema")
            if not followup.get("followup_id"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_missing_id")
            if field not in open_field_names:
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_field_not_open")
            if followup.get("status") != "queued_for_full_text_or_curator_record":
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_invalid_status")
            if not followup.get("followup_query"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_missing_query")
            if not followup.get("source_requirement"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_missing_source_requirement")
            if not followup.get("acceptance_signals"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_missing_acceptance_signals")
            if followup.get("template_promotion_allowed"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_allows_template_promotion")
            if followup.get("solved_claim_allowed"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_allows_solved_claim")
            if not followup.get("production_write_blocked"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_production_not_blocked")
            if not followup.get("not_template_support"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_promotes_template")
            if not followup.get("not_lab_procedure"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_missing_not_lab_procedure_guard")
            if followup.get("source_gate") not in {
                "review_selected_route_sources_first",
                "find_route_specific_full_text_or_curator_source",
            }:
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_invalid_source_gate")
            triage = followup.get("literature_triage") or {}
            triage_status = triage.get("triage_status")
            if triage.get("schema_version") != "statin_closure_open_gap_literature_triage.v1":
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_invalid_triage_schema")
            if triage_status not in {"selected_source_review_ready", "route_specific_search_required"}:
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_invalid_triage_status")
            if not triage.get("query_variants"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_missing_query_variants")
            if not triage.get("full_text_or_curator_record_required"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_triage_missing_full_text_gate")
            if not triage.get("not_template_support"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_triage_promotes_template")
            if not triage.get("not_lab_procedure"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_triage_missing_not_lab_procedure_guard")
            if followup.get("source_gate") == "review_selected_route_sources_first":
                if triage_status != "selected_source_review_ready":
                    reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_source_gate_triage_mismatch")
                if not triage.get("selected_source_summaries"):
                    reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_missing_selected_source_summaries")
            if followup.get("source_gate") == "find_route_specific_full_text_or_curator_source":
                if triage_status != "route_specific_search_required":
                    reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_source_gate_triage_mismatch")
                if not triage.get("route_specific_source_required"):
                    reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_missing_search_required_flag")
            resolution_schema = triage.get("curator_resolution_schema") or {}
            if resolution_schema.get("schema_version") != "statin_open_gap_curator_resolution_schema.v1":
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_invalid_resolution_schema")
            forbidden_fields = set(resolution_schema.get("forbidden_fields") or [])
            if not {"abstract_text", "raw_reaction"}.issubset(forbidden_fields):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_resolution_schema_missing_forbidden_fields")
            review_draft = triage.get("curator_review_draft") or {}
            if review_draft.get("schema_version") != "statin_open_gap_curator_review_draft.v1":
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_invalid_review_draft_schema")
            if review_draft.get("field") != field:
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_review_draft_field_mismatch")
            if review_draft.get("candidate_field_resolution") != "still_blocked":
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_review_draft_resolves_field")
            if review_draft.get("resolution_confidence") != "not_resolved_metadata_only":
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_review_draft_invalid_confidence")
            if not review_draft.get("curator_questions"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_review_draft_missing_questions")
            if not review_draft.get("evidence_gap_statement"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_review_draft_missing_gap_statement")
            if review_draft.get("promotion_allowed"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_review_draft_allows_promotion")
            if not review_draft.get("production_write_blocked"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_review_draft_production_not_blocked")
            if not review_draft.get("not_template_support"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_review_draft_promotes_template")
            if not review_draft.get("not_lab_procedure"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_review_draft_missing_not_lab_procedure_guard")
            if not {"abstract_text", "raw_reaction"}.issubset(set(review_draft.get("forbidden_fields") or [])):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_review_draft_missing_forbidden_fields")
            access_package = review_draft.get("full_text_access_package") or {}
            if access_package.get("schema_version") != "statin_open_gap_full_text_access_package.v1":
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_invalid_full_text_access_package")
            if access_package.get("execution_status") not in {
                "source_required_before_access_probe",
                "queued_for_full_text_access_probe",
                "full_text_access_probe_executed_with_open_access_candidate",
                "full_text_access_probe_executed_metadata_only",
                "full_text_access_probe_failed",
            }:
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_invalid_full_text_access_status")
            if access_package.get("full_text_content_stored"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_stores_full_text")
            if not access_package.get("not_template_support"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_access_package_promotes_template")
            if not access_package.get("not_lab_procedure"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_access_package_missing_not_lab_procedure_guard")
            if not {"abstract_text", "raw_reaction", "full_text_body"}.issubset(set(access_package.get("forbidden_fields") or [])):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_access_package_missing_forbidden_fields")
            probes = [probe for probe in access_package.get("probes") or [] if isinstance(probe, dict)]
            if int(access_package.get("source_ref_count") or 0) != len(probes):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_access_probe_count_mismatch")
            if triage_status == "selected_source_review_ready" and not probes:
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_missing_access_probes")
            for probe_index, probe in enumerate(probes, start=1):
                if probe.get("schema_version") != "statin_open_gap_full_text_access_probe.v1":
                    reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_access_probe_{probe_index}_invalid_schema")
                if probe.get("execution_status") not in {
                    "queued_not_executed",
                    "full_text_access_probe_executed",
                    "full_text_access_probe_failed",
                }:
                    reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_access_probe_{probe_index}_invalid_execution_status")
                if probe.get("full_text_access_status") not in {
                    "not_probed",
                    "pmc_open_access_link_available",
                    "doi_or_pubmed_access_metadata_available",
                    "source_metadata_only_no_access_link",
                    "access_probe_failed",
                }:
                    reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_access_probe_{probe_index}_invalid_access_status")
                if probe.get("full_text_content_stored"):
                    reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_access_probe_{probe_index}_stores_full_text")
                if not probe.get("source_ref"):
                    reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_access_probe_{probe_index}_missing_source_ref")
                if not probe.get("pmid") and not probe.get("doi") and not probe.get("source_url"):
                    reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_access_probe_{probe_index}_missing_locator")
                if probe.get("full_text_access_status") == "pmc_open_access_link_available":
                    if not probe.get("pmcids") or not probe.get("pmc_urls"):
                        reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_access_probe_{probe_index}_missing_pmc_locator")
                if not probe.get("not_template_support"):
                    reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_access_probe_{probe_index}_promotes_template")
                if not probe.get("not_lab_procedure"):
                    reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_access_probe_{probe_index}_missing_not_lab_procedure_guard")
                if not {"abstract_text", "raw_reaction", "full_text_body"}.issubset(set(probe.get("forbidden_fields") or [])):
                    reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_access_probe_{probe_index}_missing_forbidden_fields")
            field_resolution_candidate = review_draft.get("field_resolution_candidate") or {}
            if field_resolution_candidate.get("schema_version") != "statin_open_gap_field_resolution_candidate.v1":
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_invalid_field_resolution_candidate")
            if not valid_statin_field_resolution_candidate_status(
                str(field_resolution_candidate.get("candidate_status") or "")
            ):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_invalid_field_resolution_candidate_status")
            if field_resolution_candidate.get("candidate_field_resolution") != "still_blocked":
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_field_resolution_candidate_resolves_field")
            if field_resolution_candidate.get("resolution_confidence") != "not_resolved_metadata_only":
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_field_resolution_candidate_invalid_confidence")
            if field_resolution_candidate.get("promotion_allowed"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_field_resolution_candidate_allows_promotion")
            if not field_resolution_candidate.get("production_write_blocked"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_field_resolution_candidate_production_not_blocked")
            if not field_resolution_candidate.get("not_template_support"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_field_resolution_candidate_promotes_template")
            if not field_resolution_candidate.get("not_lab_procedure"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_field_resolution_candidate_missing_not_lab_procedure_guard")
            if not {"abstract_text", "raw_reaction", "full_text_body"}.issubset(set(field_resolution_candidate.get("forbidden_fields") or [])):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_field_resolution_candidate_missing_forbidden_fields")
            if triage_status == "selected_source_review_ready":
                if review_draft.get("draft_status") != "metadata_ready_pending_full_text_or_curator_confirmation":
                    reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_review_draft_status_mismatch")
                if not review_draft.get("source_refs_to_review"):
                    reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_review_draft_missing_source_refs")
            if triage_status == "route_specific_search_required":
                if review_draft.get("draft_status") != "source_required_before_curator_confirmation":
                    reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_review_draft_status_mismatch")
            search_package = triage.get("search_execution_package") or {}
            if search_package.get("schema_version") != "statin_open_gap_search_execution_package.v1":
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_invalid_search_package_schema")
            if search_package.get("field") != field:
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_search_package_field_mismatch")
            if search_package.get("execution_status") != "ready_for_pubmed_or_manual_search":
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_invalid_search_package_status")
            if not search_package.get("query_variants"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_search_package_missing_queries")
            if not search_package.get("source_acceptance_filters"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_search_package_missing_filters")
            if not search_package.get("capture_fields"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_search_package_missing_capture_fields")
            if not search_package.get("not_template_support"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_search_package_promotes_template")
            if not search_package.get("not_lab_procedure"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_search_package_missing_not_lab_procedure_guard")
            if not {"abstract_text", "raw_reaction"}.issubset(set(search_package.get("forbidden_fields") or [])):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_search_package_missing_forbidden_fields")
            trace = search_package.get("execution_trace") or {}
            trace_status = str(trace.get("execution_status") or "")
            if trace.get("schema_version") != "statin_open_gap_search_execution_trace.v1":
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_invalid_search_trace_schema")
            if trace.get("field") != field:
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_search_trace_field_mismatch")
            if trace_status not in {
                "queued_not_executed",
                "pubmed_open_gap_search_executed_with_leads",
                "pubmed_open_gap_search_executed_no_hits",
                "pubmed_open_gap_search_failed",
            }:
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_invalid_search_trace_status")
            if not trace.get("not_template_support"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_search_trace_promotes_template")
            if not trace.get("not_lab_procedure"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_search_trace_missing_not_lab_procedure_guard")
            if trace_status.startswith("pubmed_open_gap_search_executed"):
                if int(trace.get("query_attempt_count") or 0) < 1:
                    reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_search_trace_missing_query_attempt_count")
                if "pubmed" not in str(trace.get("backend_resolved") or ""):
                    reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_search_trace_missing_pubmed_backend")
                if trace_status == "pubmed_open_gap_search_executed_with_leads":
                    if int(trace.get("hit_count") or 0) < 1:
                        reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_search_trace_missing_hits")
                    if not trace.get("evidence_lead_refs"):
                        reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_search_trace_missing_evidence_refs")
                for source_index, source in enumerate(trace.get("lead_sources") or [], start=1):
                    if not isinstance(source, dict):
                        reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_search_trace_source_{source_index}_invalid")
                        continue
                    if source.get("lead_relevance_schema") != "statin_pubmed_lead_route_relevance.v1":
                        reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_search_trace_source_{source_index}_missing_relevance_schema")
                    if source.get("lead_relevance_status") not in {
                        "route_relevant_strong",
                        "route_relevant_guarded",
                        "weak_route_signal_only",
                        "non_route_context_suspected",
                        "no_route_signal",
                    }:
                        reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_search_trace_source_{source_index}_invalid_relevance_status")
                    if "route_relevance_score" not in source:
                        reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_search_trace_source_{source_index}_missing_relevance_score")
            inbox = followup.get("self_evo_inbox_entry") or {}
            if inbox.get("schema_version") != "statin_closure_open_gap_self_evo_inbox_entry.v1":
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_invalid_self_evo_inbox_schema")
            if inbox.get("field") != field:
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_self_evo_field_mismatch")
            if inbox.get("allowed_layer") != "candidate_only":
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_self_evo_not_candidate_only")
            if inbox.get("promotion_allowed"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_self_evo_allows_promotion")
            if not inbox.get("production_write_blocked"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_self_evo_production_not_blocked")
            if not inbox.get("not_template_support"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_self_evo_promotes_template")
            if not inbox.get("not_lab_procedure"):
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_self_evo_missing_not_lab_procedure_guard")
            if inbox.get("required_evidence_gate") != "full_text_or_curator_field_resolution":
                reasons.append(f"closure_curation_result_{index}_open_gap_followup_{followup_index}_self_evo_missing_field_gate")
        for field_index, field_row in enumerate(route_field_audit, start=1):
            if not field_row.get("field"):
                reasons.append(f"closure_curation_result_{index}_field_{field_index}_missing_field")
            if field_row.get("status") not in {
                "missing_full_text_or_curator_record",
                "validated_local_curator_record",
                "audited_gap_local_curator_record",
                "not_applicable_local_curator_record",
            }:
                reasons.append(f"closure_curation_result_{index}_field_{field_index}_unexpected_status")
            if field_row.get("status") == "missing_full_text_or_curator_record" and field_row.get("evidence_refs"):
                reasons.append(f"closure_curation_result_{index}_field_{field_index}_unverified_evidence_refs")
            if (
                field_row.get("status") == "missing_full_text_or_curator_record"
                and not field_row.get("resolution_required_before_promotion")
            ):
                reasons.append(f"closure_curation_result_{index}_field_{field_index}_missing_resolution_guard")
            if field_row.get("status") == "validated_local_curator_record":
                if not field_row.get("evidence_refs"):
                    reasons.append(f"closure_curation_result_{index}_field_{field_index}_missing_curator_evidence_refs")
                record_refs = [str(ref) for ref in field_row.get("curator_record_refs") or [] if str(ref).strip()]
                if not record_refs:
                    reasons.append(f"closure_curation_result_{index}_field_{field_index}_missing_curator_record_refs")
                if any(not ref.startswith("local_curator:") for ref in record_refs):
                    reasons.append(f"closure_curation_result_{index}_field_{field_index}_invalid_curator_record_ref")
            if field_row.get("status") == "not_applicable_local_curator_record":
                if field_row.get("resolution_required_before_promotion"):
                    reasons.append(f"closure_curation_result_{index}_field_{field_index}_not_applicable_requires_resolution")
                if not field_row.get("curator_record_refs"):
                    reasons.append(f"closure_curation_result_{index}_field_{field_index}_missing_not_applicable_record_ref")
            if field_row.get("status") == "audited_gap_local_curator_record":
                if not field_row.get("resolution_required_before_promotion"):
                    reasons.append(f"closure_curation_result_{index}_field_{field_index}_audited_gap_missing_resolution_guard")
                if not field_row.get("evidence_refs"):
                    reasons.append(f"closure_curation_result_{index}_field_{field_index}_audited_gap_missing_evidence_refs")
                record_refs = [str(ref) for ref in field_row.get("curator_record_refs") or [] if str(ref).strip()]
                if not record_refs:
                    reasons.append(f"closure_curation_result_{index}_field_{field_index}_audited_gap_missing_record_ref")
                if any(not ref.startswith("local_curator:") for ref in record_refs):
                    reasons.append(f"closure_curation_result_{index}_field_{field_index}_audited_gap_invalid_record_ref")
        for ref in result.get("full_text_or_curator_record_refs") or []:
            if not str(ref).startswith("local_curator:"):
                reasons.append(f"closure_curation_result_{index}_unverified_full_text_refs")
        if not result.get("promotion_blockers"):
            reasons.append(f"closure_curation_result_{index}_missing_promotion_blockers")
        candidate = result.get("self_evo_template_candidate") or {}
        if candidate.get("schema_version") != "statin_closure_self_evo_template_candidate.v1":
            reasons.append(f"closure_curation_result_{index}_invalid_self_evo_candidate_schema")
        if not candidate.get("candidate_id"):
            reasons.append(f"closure_curation_result_{index}_missing_self_evo_candidate_id")
        if candidate.get("candidate_status") not in {
            "blocked_pending_full_text_audit",
            "blocked_pending_route_specific_lead",
            "blocked_pending_remaining_route_field_audit",
            "blocked_pending_promotion_review",
        }:
            reasons.append(f"closure_curation_result_{index}_invalid_self_evo_candidate_status")
        if candidate.get("allowed_layer") != "candidate_only":
            reasons.append(f"closure_curation_result_{index}_self_evo_candidate_not_candidate_only")
        if candidate.get("promotion_allowed"):
            reasons.append(f"closure_curation_result_{index}_self_evo_candidate_allows_promotion")
        if not candidate.get("production_write_blocked"):
            reasons.append(f"closure_curation_result_{index}_self_evo_candidate_production_not_blocked")
        if not candidate.get("promotion_blockers"):
            reasons.append(f"closure_curation_result_{index}_self_evo_candidate_missing_blockers")
        if not candidate.get("not_template_support"):
            reasons.append(f"closure_curation_result_{index}_self_evo_candidate_promotes_template")
        if not candidate.get("not_lab_procedure"):
            reasons.append(f"closure_curation_result_{index}_self_evo_candidate_missing_not_lab_procedure_guard")
        if _contains_key_recursive(result, "abstract_text"):
            reasons.append(f"closure_curation_result_{index}_stored_abstract_text")
        if _contains_key_recursive(result, "raw_reaction"):
            reasons.append(f"closure_curation_result_{index}_stored_raw_reaction")
    return {
        "schema_version": "statin_closure_curation_result_set_validation.v1",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
    }


def _validate_closure_lead_curation_packet(packet: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if packet.get("schema_version") != STATIN_CLOSURE_LEAD_CURATION_PACKET_SCHEMA:
        reasons.append("invalid_closure_lead_curation_packet_schema")
    if packet.get("skipped"):
        return {
            "schema_version": "statin_closure_lead_curation_packet_validation.v1",
            "accepted": not reasons,
            "reasons": sorted(set(reasons)),
        }
    tasks = list(packet.get("tasks") or [])
    targets = {str(task.get("target_safe") or "") for task in tasks if task.get("target_safe")}
    if int(packet.get("target_count") or 0) != 9:
        reasons.append("closure_lead_curation_packet_not_all_nine_targets")
    if len(targets) != 9:
        reasons.append("closure_lead_curation_packet_tasks_not_all_nine_targets")
    if int(packet.get("task_count") or 0) != len(tasks):
        reasons.append("closure_lead_curation_packet_task_count_mismatch")
    if int(packet.get("blocker_count") or 0) != len(tasks):
        reasons.append("closure_lead_curation_packet_not_one_task_per_blocker")
    if int(packet.get("ready_for_curator_count") or 0) != len(tasks):
        reasons.append("closure_lead_curation_packet_tasks_not_ready_for_curator")
    if int(packet.get("template_promotion_allowed_count") or 0) != 0:
        reasons.append("closure_lead_curation_packet_allows_template_promotion")
    if packet.get("full_execution_coverage") and int(packet.get("lead_backed_task_count") or 0) != len(tasks):
        reasons.append("closure_lead_curation_packet_full_execution_missing_leads")
    if int(packet.get("source_metadata_task_count") or 0) > int(packet.get("lead_backed_task_count") or 0):
        reasons.append("closure_lead_curation_packet_source_metadata_count_exceeds_leads")
    if packet.get("full_execution_coverage") and int(packet.get("fully_traceable_task_count") or 0) != len(tasks):
        reasons.append("closure_lead_curation_packet_full_execution_missing_source_metadata")
    route_relevant_task_count = sum(
        1 for task in tasks
        if int(task.get("route_relevant_source_count") or 0) > 0
    )
    route_context_guarded_task_count = sum(
        1 for task in tasks
        if int(task.get("route_context_guarded_source_count") or 0) > 0
    )
    if int(packet.get("route_relevant_task_count") or 0) != route_relevant_task_count:
        reasons.append("closure_lead_curation_packet_route_relevant_count_mismatch")
    if int(packet.get("route_context_guarded_task_count") or 0) != route_context_guarded_task_count:
        reasons.append("closure_lead_curation_packet_context_guard_count_mismatch")
    if int(packet.get("route_relevant_task_count") or 0) > int(packet.get("lead_backed_task_count") or 0):
        reasons.append("closure_lead_curation_packet_route_relevant_count_exceeds_leads")
    if int(packet.get("route_context_guarded_task_count") or 0) > int(packet.get("lead_backed_task_count") or 0):
        reasons.append("closure_lead_curation_packet_context_guard_count_exceeds_leads")
    task_ids: set[str] = set()
    for index, task in enumerate(tasks, start=1):
        task_id = str(task.get("task_id") or "")
        if task.get("schema_version") != "statin_closure_lead_curation_task.v1":
            reasons.append(f"closure_lead_curation_task_{index}_invalid_schema")
        if not task_id:
            reasons.append(f"closure_lead_curation_task_{index}_missing_task_id")
        elif task_id in task_ids:
            reasons.append(f"closure_lead_curation_task_{index}_duplicate_task_id")
        task_ids.add(task_id)
        if not task.get("target_safe"):
            reasons.append(f"closure_lead_curation_task_{index}_missing_target")
        if not task.get("requirement_id"):
            reasons.append(f"closure_lead_curation_task_{index}_missing_requirement_id")
        if not task.get("blocker"):
            reasons.append(f"closure_lead_curation_task_{index}_missing_blocker")
        if not task.get("followup_query"):
            reasons.append(f"closure_lead_curation_task_{index}_missing_followup_query")
        if task.get("curation_status") != "pending_full_text_or_curator_audit":
            reasons.append(f"closure_lead_curation_task_{index}_invalid_curation_status")
        if task.get("template_promotion_allowed"):
            reasons.append(f"closure_lead_curation_task_{index}_allows_template_promotion")
        if task.get("solved_claim_allowed"):
            reasons.append(f"closure_lead_curation_task_{index}_allows_solved_claim")
        if not task.get("not_template_support"):
            reasons.append(f"closure_lead_curation_task_{index}_must_not_be_template_support")
        if not task.get("not_lab_procedure"):
            reasons.append(f"closure_lead_curation_task_{index}_missing_not_lab_procedure_guard")
        schema = task.get("extraction_schema") or {}
        if schema.get("schema_version") != "statin_closure_lead_extraction_schema.v1":
            reasons.append(f"closure_lead_curation_task_{index}_invalid_extraction_schema")
        if not schema.get("required_source_fields"):
            reasons.append(f"closure_lead_curation_task_{index}_missing_required_source_fields")
        if not schema.get("required_route_fields"):
            reasons.append(f"closure_lead_curation_task_{index}_missing_required_route_fields")
        if "abstract_text" not in set(schema.get("forbidden_fields") or []):
            reasons.append(f"closure_lead_curation_task_{index}_missing_abstract_text_forbidden_field")
        if not task.get("acceptance_criteria"):
            reasons.append(f"closure_lead_curation_task_{index}_missing_acceptance_criteria")
        if not task.get("rejection_rules"):
            reasons.append(f"closure_lead_curation_task_{index}_missing_rejection_rules")
        if task.get("abstract_signal_status") == "abstract_route_signal_detected" and not task.get("abstract_signal_terms"):
            reasons.append(f"closure_lead_curation_task_{index}_missing_abstract_signal_terms")
        if str(task.get("execution_status") or "").startswith("pubmed_followup_executed"):
            if not task.get("evidence_lead_refs"):
                reasons.append(f"closure_lead_curation_task_{index}_missing_evidence_leads")
            if "pubmed_followup_esearch" not in set(task.get("search_sources") or []):
                reasons.append(f"closure_lead_curation_task_{index}_missing_pubmed_esearch")
        lead_refs = [str(ref) for ref in task.get("evidence_lead_refs") or [] if str(ref).strip()]
        lead_sources = [source for source in task.get("lead_sources") or [] if isinstance(source, dict)]
        checklist = task.get("source_checklist") or {}
        if int(task.get("route_relevant_source_count") or 0) > len(lead_sources):
            reasons.append(f"closure_lead_curation_task_{index}_route_relevant_count_exceeds_sources")
        if int(task.get("route_context_guarded_source_count") or 0) > len(lead_sources):
            reasons.append(f"closure_lead_curation_task_{index}_context_guard_count_exceeds_sources")
        if lead_refs:
            if checklist.get("schema_version") != "statin_closure_source_checklist.v1":
                reasons.append(f"closure_lead_curation_task_{index}_invalid_source_checklist")
            if int(checklist.get("evidence_lead_count") or 0) != len(lead_refs):
                reasons.append(f"closure_lead_curation_task_{index}_source_checklist_lead_count_mismatch")
            if int(checklist.get("source_metadata_count") or 0) != len(lead_sources):
                reasons.append(f"closure_lead_curation_task_{index}_source_checklist_metadata_count_mismatch")
            if int(checklist.get("route_relevant_source_count") or 0) != int(task.get("route_relevant_source_count") or 0):
                reasons.append(f"closure_lead_curation_task_{index}_source_checklist_route_relevant_count_mismatch")
            if int(checklist.get("route_context_guarded_source_count") or 0) != int(task.get("route_context_guarded_source_count") or 0):
                reasons.append(f"closure_lead_curation_task_{index}_source_checklist_context_guard_count_mismatch")
            if not checklist.get("all_leads_have_source_metadata"):
                reasons.append(f"closure_lead_curation_task_{index}_missing_source_metadata_for_leads")
            if not checklist.get("all_leads_have_route_relevance_audit"):
                reasons.append(f"closure_lead_curation_task_{index}_missing_route_relevance_audit_for_leads")
            source_refs = {str(source.get("evidence_ref") or "") for source in lead_sources}
            for ref in lead_refs:
                if ref not in source_refs:
                    reasons.append(f"closure_lead_curation_task_{index}_lead_missing_source:{ref}")
        for source_index, source in enumerate(lead_sources, start=1):
            if source.get("schema_version") != "statin_closure_pubmed_lead_source.v1":
                reasons.append(f"closure_lead_curation_task_{index}_source_{source_index}_invalid_schema")
            if not source.get("evidence_ref"):
                reasons.append(f"closure_lead_curation_task_{index}_source_{source_index}_missing_evidence_ref")
            if not (source.get("pmid") or source.get("doi")):
                reasons.append(f"closure_lead_curation_task_{index}_source_{source_index}_missing_pmid_or_doi")
            if not source.get("source_url"):
                reasons.append(f"closure_lead_curation_task_{index}_source_{source_index}_missing_source_url")
            if not source.get("source_title"):
                reasons.append(f"closure_lead_curation_task_{index}_source_{source_index}_missing_source_title")
            if source.get("lead_relevance_schema") != "statin_pubmed_lead_route_relevance.v1":
                reasons.append(f"closure_lead_curation_task_{index}_source_{source_index}_missing_relevance_schema")
            if source.get("lead_relevance_status") not in {
                "route_relevant_strong",
                "route_relevant_guarded",
                "weak_route_signal_only",
                "non_route_context_suspected",
                "no_route_signal",
            }:
                reasons.append(f"closure_lead_curation_task_{index}_source_{source_index}_invalid_relevance_status")
            if "route_relevance_score" not in source:
                reasons.append(f"closure_lead_curation_task_{index}_source_{source_index}_missing_relevance_score")
            if not source.get("route_relevance_guard"):
                reasons.append(f"closure_lead_curation_task_{index}_source_{source_index}_missing_relevance_guard")
            if not source.get("not_template_support"):
                reasons.append(f"closure_lead_curation_task_{index}_source_{source_index}_promotes_template")
            if not source.get("not_lab_procedure"):
                reasons.append(f"closure_lead_curation_task_{index}_source_{source_index}_missing_not_lab_procedure_guard")
        if _contains_key_recursive(task, "abstract_text"):
            reasons.append(f"closure_lead_curation_task_{index}_stored_abstract_text")
    return {
        "schema_version": "statin_closure_lead_curation_packet_validation.v1",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
    }


def _contains_key_recursive(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return any(str(item_key) == key or _contains_key_recursive(item_value, key) for item_key, item_value in value.items())
    if isinstance(value, list):
        return any(_contains_key_recursive(item, key) for item in value)
    return False


def _route_template_reasons(route_template: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if route_template.get("schema_version") != STATIN_MEMBER_ROUTE_TEMPLATE_SCHEMA:
        reasons.append("invalid_route_template_schema")
    if not route_template.get("target_safe"):
        reasons.append("route_template_missing_target_safe")
    if not route_template.get("expected_reaction_class"):
        reasons.append("route_template_missing_expected_reaction_class")
    if not route_template.get("expected_family_id"):
        reasons.append("route_template_missing_expected_family_id")
    if not route_template.get("evidence_refs"):
        reasons.append("route_template_missing_evidence_refs")
    if not route_template.get("template_sources"):
        reasons.append("route_template_missing_template_sources")
    if not route_template.get("not_lab_procedure"):
        reasons.append("route_template_missing_not_lab_procedure_guard")
    template_steps = route_template.get("template_steps") or []
    if len(template_steps) < 3:
        reasons.append("insufficient_route_template_steps")
    for index, step in enumerate(template_steps, start=1):
        if step.get("schema_version") != STATIN_ROUTE_TEMPLATE_STEP_SCHEMA:
            reasons.append(f"route_template_step_{index}_invalid_schema")
        if not step.get("step_id"):
            reasons.append(f"route_template_step_{index}_missing_step_id")
        if not step.get("template_role"):
            reasons.append(f"route_template_step_{index}_missing_template_role")
        if not step.get("route_role"):
            reasons.append(f"route_template_step_{index}_missing_route_role")
        if not step.get("evidence_refs"):
            reasons.append(f"route_template_step_{index}_missing_evidence_refs")
        if not step.get("template_sources"):
            reasons.append(f"route_template_step_{index}_missing_template_sources")
        if step.get("requires_literature_evidence") and not step.get("difficulty_queries"):
            reasons.append(f"route_template_step_{index}_missing_difficulty_queries")
        if step.get("requires_literature_evidence") and not step.get("literature_trace_refs"):
            reasons.append(f"route_template_step_{index}_missing_literature_trace_refs")
        for trace_index, trace in enumerate(step.get("literature_trace_refs") or [], start=1):
            if trace.get("schema_version") != STATIN_LITERATURE_QUERY_TRACE_SCHEMA:
                reasons.append(f"route_template_step_{index}_trace_{trace_index}_invalid_schema")
            if trace.get("execution_status") != "covered_by_validated_literature_search":
                reasons.append(f"route_template_step_{index}_trace_{trace_index}_not_covered")
            if not trace.get("task_ref") or not trace.get("report_ref"):
                reasons.append(f"route_template_step_{index}_trace_{trace_index}_missing_refs")
            if int(trace.get("validated_evidence_count") or 0) < 1:
                reasons.append(f"route_template_step_{index}_trace_{trace_index}_missing_validated_evidence")
            if not trace.get("template_supporting_evidence_refs"):
                reasons.append(f"route_template_step_{index}_trace_{trace_index}_missing_template_supporting_evidence")
        if not step.get("acceptance_criteria"):
            reasons.append(f"route_template_step_{index}_missing_acceptance_criteria")
        if not step.get("rejection_rules"):
            reasons.append(f"route_template_step_{index}_missing_rejection_rules")
        if not step.get("self_evo_tags"):
            reasons.append(f"route_template_step_{index}_missing_self_evo_tags")
        if not step.get("not_lab_procedure"):
            reasons.append(f"route_template_step_{index}_missing_not_lab_procedure_guard")
    return reasons


def _render_fullflow_dossier_md(dossier: dict[str, Any]) -> str:
    target = dossier.get("target") or {}
    blueprint = dossier.get("fullflow_blueprint") or {}
    escalation = dossier.get("automatic_literature_escalation") or {}
    lines = [
        f"# {target.get('name')} fullflow synthesis dossier",
        "",
        f"- route_status: `{dossier.get('route_status')}`",
        f"- planning_status: `{dossier.get('planning_status')}`",
        f"- not_lab_procedure: `{bool(dossier.get('not_lab_procedure'))}`",
        f"- expected template sources: `{', '.join(dossier.get('primary_template_sources') or [])}`",
        "",
        "## Synthesis Stages",
    ]
    for stage in dossier.get("synthesis_stages") or []:
        lines.append(f"- `{stage.get('stage_id')}` {stage.get('title')}: {stage.get('notes')}")
    lines.extend(["", "## Step-Level Route Template"])
    for step in (dossier.get("route_template") or {}).get("template_steps") or []:
        difficulties = ", ".join(
            str(item.get("difficulty") or "")
            for item in step.get("difficulty_queries") or []
            if item.get("difficulty")
        )
        lines.append(
            "- `{step_id}` role `{role}` sources `{sources}` difficulties `{difficulties}`".format(
                step_id=step.get("step_id"),
                role=step.get("template_role"),
                sources=", ".join(step.get("template_sources") or []),
                difficulties=difficulties,
            )
        )
    lines.extend(["", "## Member-Specific Blueprint"])
    for item in blueprint.get("member_specific_route_outline") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Key Intermediate Roles"])
    for item in blueprint.get("key_intermediate_roles") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Endpoint Audit"])
    for item in blueprint.get("endpoint_audit") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Difficulty Escalation"])
    for item in dossier.get("difficulty_escalation") or []:
        lines.append(f"- `{item.get('difficulty')}` -> {item.get('automatic_action')}")
    lines.extend(["", "## Automatic Literature Escalation"])
    trace_summary = escalation.get("query_trace_summary") or {}
    lines.append(
        "- trace_summary: accepted `{accepted}` / total `{total}` via `{backend}`".format(
            accepted=trace_summary.get("accepted_query_trace_count") or 0,
            total=trace_summary.get("query_trace_count") or 0,
            backend=trace_summary.get("backend_resolved") or "",
        )
    )
    for item in escalation.get("difficulty_queries") or []:
        lines.append(
            "- `{difficulty}` query `{query}` -> {signal}".format(
                difficulty=item.get("difficulty"),
                query=item.get("query"),
                signal=item.get("acceptance_signal"),
            )
        )
    lines.extend(["", "## Automatic Literature Trace"])
    for trace in escalation.get("query_execution_traces") or []:
        lines.append(
            "- `{difficulty}` status `{status}` quality `{quality}` hits `{hits}` validated `{validated}` "
            "template_support `{support}` external_leads `{leads}` task `{task}` report `{report}`".format(
                difficulty=trace.get("difficulty"),
                status=trace.get("execution_status"),
                quality=trace.get("quality_gate_status"),
                hits=trace.get("hit_count"),
                validated=trace.get("validated_evidence_count"),
                support=trace.get("template_supporting_evidence_count") or 0,
                leads=trace.get("external_literature_lead_count") or 0,
                task=trace.get("task_ref"),
                report=trace.get("report_ref"),
            )
        )
    lines.extend(["", "## Literature Acceptance Criteria"])
    for item in escalation.get("acceptance_criteria") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Literature Rejection Rules"])
    for item in escalation.get("rejection_rules") or []:
        lines.append(f"- {item}")
    closure = dossier.get("route_closure_audit") or {}
    lines.extend(["", "## Route Closure Audit"])
    lines.append(f"- readiness_status: `{closure.get('readiness_status')}`")
    lines.append(f"- solved_claim_allowed: `{bool(closure.get('solved_claim_allowed'))}`")
    lines.append(
        "- passed/blockers: `{passed}` / `{blocked}`".format(
            passed=len(closure.get("passed_requirements") or []),
            blocked=len(closure.get("blocking_requirements") or []),
        )
    )
    lines.extend(["", "### Closure Blockers"])
    for item in closure.get("blocking_requirements") or []:
        lines.append(f"- `{item.get('requirement_id')}` {item.get('title')}: {item.get('blocker')}")
    lines.extend(["", "### Automatic Follow-Up Literature Queue"])
    for item in closure.get("automatic_followup_literature_queue") or []:
        lines.append(
            "- `{requirement}` query `{query}` -> {signal}".format(
                requirement=item.get("requirement_id"),
                query=item.get("query"),
                signal=item.get("acceptance_signal"),
            )
        )
    execution = closure.get("followup_execution") or {}
    lines.extend(["", "### Follow-Up Literature Execution"])
    lines.append(
        "- policy `{policy}` requested `{requested}` executed `{executed}` lead_traces `{leads}` abstract_signal_traces `{signals}` full_exec `{full_exec}`".format(
            policy=execution.get("policy"),
            requested=bool(execution.get("requested")),
            executed=execution.get("executed_trace_count") or 0,
            leads=execution.get("lead_trace_count") or 0,
            signals=execution.get("abstract_signal_trace_count") or 0,
            full_exec=bool(execution.get("full_execution_coverage")),
        )
    )
    for trace in execution.get("traces") or []:
        lines.append(
            "- `{requirement}` status `{status}` hits `{hits}` attempts `{attempts}` fallback `{fallback}` abstract_signal `{signal}` query `{query}`".format(
                requirement=trace.get("requirement_id"),
                status=trace.get("execution_status"),
                hits=trace.get("hit_count"),
                attempts=trace.get("query_attempt_count") or 0,
                fallback=bool(trace.get("fallback_used")),
                signal=trace.get("abstract_signal_status") or "",
                query=trace.get("query"),
            )
        )
        if trace.get("resolved_query"):
            lines.append(f"  - resolved_query: `{trace.get('resolved_query')}`")
        if trace.get("abstract_signal_terms"):
            lines.append(f"  - abstract_signal_terms: `{', '.join(trace.get('abstract_signal_terms') or [])}`")
    lines.extend([
        "",
        "## Contract",
        "",
        str(dossier.get("status_contract") or ""),
        "",
    ])
    return "\n".join(lines)


def _read_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _evidence_quality_summary(
    evidence_cards: list[dict[str, Any]],
    *,
    evidence_refs: list[str],
) -> dict[str, Any]:
    cards_by_id = {
        str(card.get("evidence_id") or ""): dict(card)
        for card in evidence_cards
        if isinstance(card, dict)
    }
    template_refs: list[str] = []
    external_leads: list[str] = []
    lead_only_refs: list[str] = []
    pubmed_refs: list[str] = []
    for ref in evidence_refs:
        card = cards_by_id.get(str(ref)) or {}
        if _is_external_literature_lead(card):
            external_leads.append(str(ref))
        if _is_lead_only_evidence(card):
            lead_only_refs.append(str(ref))
        if _is_pubmed_evidence(card):
            pubmed_refs.append(str(ref))
        if _is_template_promotable_evidence(card):
            template_refs.append(str(ref))
    return {
        "schema_version": "statin_literature_evidence_quality_summary.v1",
        "evidence_ref_count": len(evidence_refs),
        "template_promotable_refs": template_refs,
        "template_promotable_count": len(template_refs),
        "external_literature_lead_refs": external_leads,
        "external_literature_lead_count": len(external_leads),
        "lead_only_refs": lead_only_refs,
        "lead_only_count": len(lead_only_refs),
        "pubmed_summary_only_refs": pubmed_refs,
        "pubmed_summary_only_count": len(pubmed_refs),
        "template_promotion_guard": (
            "summary-only PubMed leads are review leads; route template promotion requires "
            "curated local/manual/full-text evidence."
        ),
    }


def _is_pubmed_evidence(card: dict[str, Any]) -> bool:
    return str((card.get("source_metadata") or {}).get("backend") or "").lower() == "pubmed"


def _is_external_literature_lead(card: dict[str, Any]) -> bool:
    source_type = str(card.get("source_type") or "").lower()
    return source_type == "literature" or _is_pubmed_evidence(card)


def _is_lead_only_evidence(card: dict[str, Any]) -> bool:
    limitations = {str(item).lower() for item in card.get("limitations") or []}
    return (
        "pubmed_summary_only" in limitations
        or "requires_full_text_route_audit" in limitations
        or (
            _is_pubmed_evidence(card)
            and "full_text_route_audited" not in limitations
        )
    )


def _is_template_promotable_evidence(card: dict[str, Any]) -> bool:
    if not card:
        return False
    if _is_lead_only_evidence(card):
        return False
    relation = str(card.get("target_relation") or "")
    route_role = str(card.get("route_role") or "")
    if relation == "analogy_only":
        return False
    return route_role in {"strategic_disconnection", "route_anchor", "condition_hint"}


def _family_bucket(safe: str) -> str:
    if safe in NATURAL_STATINS:
        return "natural_statin"
    if safe in SYNTHETIC_STATINS:
        return "synthetic_statin"
    return "statin"


def _family_hint(target: StatinPanelTarget) -> str:
    if target.family_bucket == "natural_statin":
        return (
            f"{target.name}, natural statin, fermentation core, semisynthesis, "
            "lovastatin, simvastatin, pravastatin, mevastatin, decalin lactone"
        )
    return (
        f"{target.name}, synthetic statin, syn-3,5-dihydroxy acid side chain, "
        f"HWE, Wittig, convergent side-chain, {_synthetic_core_hint(target.safe)}"
    )


def _synthetic_core_hint(safe: str) -> str:
    return {
        "atorvastatin": "pyrrole core, Paal-Knorr, Hantzsch, diaryl amide",
        "fluvastatin": "indole core, aldol, cryogenic stereoselective reduction",
        "pitavastatin": "quinoline core, Suzuki coupling, hydroboration coupling, cyclopropyl",
        "rosuvastatin": "pyrimidine core, Wittig olefination, biocatalytic ketoreduction",
        "cerivastatin": "pyridine core, shared synthetic statin side-chain convergence",
    }.get(safe, "heteroaryl core, side-chain convergence")


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))
