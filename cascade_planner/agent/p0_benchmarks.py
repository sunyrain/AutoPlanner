"""Frozen P0 benchmark cases for SMILES-first literature workflow."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cascade_planner.agent.smiles_first import SmilesFirstWorkflowConfig, run_smiles_first_workflow


P0_BENCHMARK_SCHEMA = "p0_smiles_first_benchmark.v1"
P0_BENCHMARK_REPORT_SCHEMA = "p0_smiles_first_benchmark_report.v1"


@dataclass
class P0BenchmarkCase:
    case_id: str
    target_name: str
    target_smiles: str
    family_hint: str
    frontier_smiles: str = ""
    expected_reaction_classes: list[str] = field(default_factory=list)
    expected_route_status: str = "partial_anchor"
    expected_candidate_kinds: list[str] = field(default_factory=lambda: [
        "exact_fragment_retro",
        "forward_surrogate",
        "route_anchor",
    ])
    case_category: str = "literature_known_strategic_disconnection"
    schema_version: str = "p0_benchmark_case.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def frozen_p0_benchmark_cases() -> list[P0BenchmarkCase]:
    """Return named literature-known cases used for P0 regression replay."""
    return [
        P0BenchmarkCase(
            case_id="bufotalin_like_c17_pyrone",
            target_name="Bufotalin-like bufadienolide frontier",
            target_smiles="CC(C)CCCC(C)C1CCC2C3CCC4CC(O)CCC4(C)C3CCC12C",
            family_hint="bufotalin, bufadienolide, steroid, C17-pyrone",
            expected_reaction_classes=["C_C_coupling"],
            case_category="NP_like_frontier",
        ),
        P0BenchmarkCase(
            case_id="macro_lactone_macrolactonization",
            target_name="Macrolactone advanced intermediate",
            target_smiles="O=C1CCCCCCCCCCCCO1",
            family_hint="macrocycle, macrolactonization, polyketide",
            expected_reaction_classes=["macrolactonization"],
            case_category="semisynthesis_anchor",
        ),
        P0BenchmarkCase(
            case_id="phenolic_glycoside_glycosylation",
            target_name="Phenolic O-glycoside advanced intermediate",
            target_smiles="Oc1ccccc1OC1COC(O)C(O)C1O",
            family_hint="glycoside, sugar, glycosylation",
            expected_reaction_classes=["glycosylation"],
            case_category="literature_known_strategic_disconnection",
        ),
        P0BenchmarkCase(
            case_id="paclitaxel_taxane_semisynthesis",
            target_name="Paclitaxel / taxane semisynthesis",
            target_smiles="CC(=O)OC1CC(O)C2(C)C(OC(=O)c3ccccc3)C3OC3C(O)C12",
            family_hint="paclitaxel, taxane, baccatin, 10-deacetylbaccatin III",
            expected_reaction_classes=["taxane_side_chain_acylation"],
            case_category="semisynthesis_anchor",
        ),
        P0BenchmarkCase(
            case_id="artemisinin_peroxide_anchor",
            target_name="Artemisinin peroxide late-stage anchor",
            target_smiles="CC(C)C1OC2OOCC1CC2=O",
            family_hint="artemisinin, sesquiterpene peroxide, dihydroartemisinic acid, photooxidation",
            expected_reaction_classes=["late_stage_peroxide_formation"],
            case_category="NP_like_frontier",
        ),
        P0BenchmarkCase(
            case_id="lovastatin_semisynthesis_core",
            target_name="Natural statin fermentation-core semisynthesis",
            target_smiles="CC(=O)OC1CCOC(=O)C1",
            family_hint="lovastatin, simvastatin, natural statin, fermentation core, semisynthesis",
            expected_reaction_classes=["statin_semisynthesis"],
            case_category="semisynthesis_anchor",
        ),
        P0BenchmarkCase(
            case_id="corey_lactone_prostaglandin",
            target_name="Corey-lactone prostaglandin intermediate",
            target_smiles="CCCCCCCC=CC1CCC(=O)O1",
            family_hint="prostaglandin, eicosanoid, Corey lactone, cyclopentane side chain",
            expected_reaction_classes=["corey_lactone_sidechain_installation"],
            case_category="literature_known_strategic_disconnection",
        ),
    ]


def run_p0_benchmark_pack(
    *,
    output_root: str | Path,
    cases: list[P0BenchmarkCase] | None = None,
    query_budget: int = 8,
    literature_backend: str = "local",
) -> dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case in cases or frozen_p0_benchmark_cases():
        case_dir = root / case.case_id
        result = run_smiles_first_workflow(
            SmilesFirstWorkflowConfig(
                target_smiles=case.target_smiles,
                target_name=case.case_id,
                family_hint=case.family_hint,
                frontier_smiles=case.frontier_smiles or case.target_smiles,
                output_dir=case_dir,
                query_budget=query_budget,
                literature_backend=literature_backend,
            )
        )
        rows.append(_evaluate_case(case, result))
    report = {
        "schema_version": P0_BENCHMARK_REPORT_SCHEMA,
        "benchmark_schema": P0_BENCHMARK_SCHEMA,
        "case_count": len(rows),
        "passed": sum(1 for row in rows if row["passed"]),
        "failed": sum(1 for row in rows if not row["passed"]),
        "cases": rows,
        "hard_gates": {
            "all_cases_accept_validation": all(row["validation_accepted"] for row in rows),
            "all_cases_have_expected_templates": all(row["expected_templates_hit"] for row in rows),
            "no_case_claims_solved": all(not row["claims_solved"] for row in rows),
            "all_cases_have_route_map": all(row["route_map_exists"] for row in rows),
        },
    }
    (root / "p0_benchmark_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _evaluate_case(case: P0BenchmarkCase, result: dict[str, Any]) -> dict[str, Any]:
    artifacts = result.get("artifacts") or {}
    package = _read_json(artifacts.get("hybrid_route_package"))
    validation = _read_json(artifacts.get("validation"))
    candidates = list(package.get("literature_candidates") or [])
    classes = sorted({str(item.get("reaction_class") or "") for item in candidates if item.get("reaction_class")})
    kinds = sorted({str(item.get("candidate_kind") or "") for item in candidates if item.get("candidate_kind")})
    expected_classes = set(case.expected_reaction_classes)
    expected_kinds = set(case.expected_candidate_kinds)
    expected_templates_hit = expected_classes.issubset(set(classes))
    expected_kinds_hit = expected_kinds.issubset(set(kinds))
    validation_accepted = bool(validation.get("accepted"))
    route_status_ok = str(validation.get("route_status") or package.get("route_status")) == case.expected_route_status
    claims_solved = str(validation.get("route_status") or package.get("route_status")) == "solved"
    route_map_exists = bool(artifacts.get("route_map") and Path(artifacts["route_map"]).exists())
    reasons = []
    if not validation_accepted:
        reasons.append("validation_not_accepted")
    if not expected_templates_hit:
        reasons.append("expected_reaction_class_missing")
    if not expected_kinds_hit:
        reasons.append("expected_candidate_kind_missing")
    if not route_status_ok:
        reasons.append("unexpected_route_status")
    if claims_solved:
        reasons.append("p0_claimed_solved")
    if not route_map_exists:
        reasons.append("missing_route_map")
    return {
        "case_id": case.case_id,
        "target_name": case.target_name,
        "case_category": case.case_category,
        "passed": not reasons,
        "reasons": reasons,
        "validation_accepted": validation_accepted,
        "route_status": str(validation.get("route_status") or package.get("route_status") or ""),
        "claims_solved": claims_solved,
        "route_map_exists": route_map_exists,
        "expected_templates_hit": expected_templates_hit,
        "expected_candidate_kinds_hit": expected_kinds_hit,
        "expected_reaction_classes": sorted(expected_classes),
        "observed_reaction_classes": classes,
        "observed_candidate_kinds": kinds,
        "output_dir": str(Path(result.get("output_dir") or "")),
    }


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))
