"""Build a closed 20-step Bufotalin route with independent proof semantics.

The route is connected to a benchmark-search stock leaf, so graph/search
closure is true.  That does not make the route process-ready: the five
historical planner steps remain advisory L0 hypotheses, while the accepted
paper visual chain is represented as source-reported, structurally
materialized L1 evidence pending exact curator binding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.application.reaction_condition_records import (  # noqa: E402
    audit_condition_completeness,
)
from cascade_planner.application.route_workbench import (  # noqa: E402
    compile_route_workbench,
)
from cascade_planner.harness.deterministic_literature_registry import (  # noqa: E402
    source_amount_reagent_names,
)
from cascade_planner.harness.source_condition_extraction import (  # noqa: E402
    extract_source_conditions,
)
from cascade_planner.harness.v4_route_workbench import (  # noqa: E402
    compile_v4_route_forest,
    render_v4_route_workbench_html,
)
from cascade_planner.interfaces.literature_authorized_source import (  # noqa: E402
    materialize_authorized_publisher_json,
)
from cascade_planner.interfaces.literature_evidence import (  # noqa: E402
    BuiltinLiteratureEvidenceConfig,
)
from cascade_planner.interfaces.literature_materialization import (  # noqa: E402
    materialize_candidate,
)


DOI = "10.1016/j.tet.2025.134610"
SOURCE_REF = f"doi:{DOI}"
PAPER_TITLE = (
    "Total synthesis of bufotalin by a direct C14 beta-hydroxylation strategy"
)
EXPECTED_FORMULAS = {
    "11": "C19H26O2",
    "24": "C21H30O3",
    "25": "C21H32O3",
    "23": "C21H34O3",
    "26": "C21H33BrO3",
    "27": "C21H32O3",
    "28": "C19H28O2",
    "19": "C25H42O2Si",
    "20": "C25H42O3Si",
    "14": "C25H43IO2Si",
    "22": "C30H46O4Si",
    "30": "C30H46O5Si",
    "31": "C33H54O5Si2",
    "32": "C33H56O5Si2",
    "33": "C35H58O6Si2",
    "bufotalin": "C26H36O6",
}
PAPER_SYNTHESIS_LABELS = [
    "24",
    "25",
    "23",
    "26",
    "27",
    "28",
    "19",
    "20",
    "14",
    "22",
    "30",
    "31",
    "32",
    "33",
    "bufotalin",
]


def main() -> int:
    args = _parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    structured_path = Path(args.structured_json).expanduser().resolve()
    pdf_path = Path(args.pdf).expanduser().resolve()
    visual_path = Path(args.visual_chain).expanduser().resolve()
    upstream_path = Path(args.upstream_routes).expanduser().resolve()
    for path in (structured_path, pdf_path, visual_path, upstream_path):
        if not path.is_file():
            raise ValueError(f"bufotalin_showcase_input_missing:{path}")

    acquisition = _materialize_sources(
        output_dir=output_dir,
        structured_path=structured_path,
        pdf_path=pdf_path,
    )
    structured = acquisition["structured"]
    pdf = acquisition["pdf"]
    structured_labels = [
        str(row.get("label") or "") for row in structured["procedure_inventory"]
    ]
    pdf_labels = [str(row.get("label") or "") for row in pdf["procedure_inventory"]]
    expected_source_labels = ["1" if value == "bufotalin" else value for value in PAPER_SYNTHESIS_LABELS]
    if structured_labels != expected_source_labels or pdf_labels != expected_source_labels:
        raise ValueError(
            "bufotalin_source_procedure_sequence_mismatch:"
            f"structured={structured_labels}:pdf={pdf_labels}"
        )

    visual_chain = _visual_chain(_read_json(visual_path))
    paper_steps, molecule_rows, formula_audit = _paper_steps(
        visual_chain,
        structured["procedure_inventory"],
        structured_sha=str(structured["source_fulltext_sha256"]),
        pdf_sha=str(pdf["source_pdf_sha256"]),
    )
    upstream_steps, upstream_molecules = _upstream_steps(
        _read_json(upstream_path),
        terminal_smiles=next(
            row["reactant_smiles"]
            for row in visual_chain
            if row["reactant_label"] == "11"
        ),
    )
    molecule_rows.update(upstream_molecules)
    steps = [*upstream_steps, *paper_steps]
    if len(steps) != 20:
        raise ValueError(f"bufotalin_showcase_step_count_invalid:{len(steps)}")
    _assert_chain(steps)

    graph, portfolio = _graph_and_portfolio(
        steps,
        molecule_rows=molecule_rows,
        structured=structured,
        pdf=pdf,
    )
    workbench = compile_route_workbench(
        graph,
        portfolio,
        campaign_summary={
            "gates": {
                "B0_blind_input": True,
                "B1_global_multi_route": True,
                "B2_host_validated_routes": False,
                "B3_exact_multi_source": False,
                "B4_stock_boundary": False,
                "B5_configured_portfolio_acceptance": False,
            },
            "highest_contiguous_gate": "B1_global_multi_route",
            "model_cost": {"model_invocations": 0},
            "resource_envelope": {"within_budget": True},
            "counts": {
                "displayed_route_count": 1,
                "displayed_step_count": 20,
                "paper_reported_step_count": 15,
                "planner_hypothesis_step_count": 5,
                "source_procedure_count": 15,
            },
            "claim": {
                "status": "route_closed_proof_unresolved",
                "solved": False,
                "configured_boundary_closed": True,
                "closure_profile": "exploration_closed",
            },
            "current_disposition": {
                "route_is_visible": True,
                "low_confidence_edges_are_warning_encoded": True,
                "full_synthesis_claim": False,
            },
        },
    )
    forest = compile_v4_route_forest(workbench)
    dossier = _dossier(
        steps,
        structured=structured,
        pdf=pdf,
        formula_audit=formula_audit,
        visual_path=visual_path,
        upstream_path=upstream_path,
        workbench=workbench,
        forest=forest,
    )
    _write_json(output_dir / "source_acquisition.json", acquisition)
    _write_json(output_dir / "bufotalin_20_step_dossier.json", dossier)
    _write_json(output_dir / "canonical_hypergraph.json", graph)
    _write_json(output_dir / "proof_portfolio.json", portfolio)
    _write_json(output_dir / "route_workbench.json", workbench)
    _write_json(output_dir / "explored_route_forest.json", forest)
    (output_dir / "route_forest.html").write_text(
        render_v4_route_workbench_html(workbench),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "accepted": True,
                "claim_status": dossier["claim_status"],
                "step_count": len(steps),
                "paper_reported_step_count": len(paper_steps),
                "planner_hypothesis_step_count": len(upstream_steps),
                "structured_procedure_count": len(structured_labels),
                "pdf_procedure_count": len(pdf_labels),
                "formula_match_count": sum(
                    row["formula_match"] is True for row in formula_audit
                ),
                "workbench": str(output_dir / "route_forest.html"),
                "dossier": str(output_dir / "bufotalin_20_step_dossier.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _materialize_sources(
    *,
    output_dir: Path,
    structured_path: Path,
    pdf_path: Path,
) -> dict[str, Any]:
    candidate = {
        "doi": DOI,
        "title": PAPER_TITLE,
    }
    request = {
        "target_name": "bufotalin",
        "edges": [{"edge_id": "paper-reported-route"}],
        "source_tasks": [{"query": "bufotalin total synthesis"}],
    }
    config = BuiltinLiteratureEvidenceConfig(
        cache_dir=output_dir / "acquisition-cache",
        enable_structured_fulltext_first=False,
        max_sources=1,
        max_fulltext_sections=32,
        max_visual_pages=6,
    )
    structured_content = structured_path.read_bytes()
    structured = materialize_authorized_publisher_json(
        candidate,
        request=request,
        source_ref=SOURCE_REF,
        source_dir=output_dir / "acquisition" / "structured",
        fulltext_cache_dir=output_dir / "acquisition-cache" / "structured",
        config=config,
        artifact={
            "provider": "legacy_publisher_spider",
            "structured_path": str(structured_path),
            "structured_sha256": hashlib.sha256(structured_content).hexdigest(),
        },
    )
    pdf = materialize_candidate(
        {**candidate, "local_pdf": str(pdf_path)},
        request=request,
        output_dir=output_dir / "acquisition" / "pdf",
        config=config,
        fetch=lambda *_args: (_ for _ in ()).throw(
            AssertionError("bufotalin showcase must not fetch network data")
        ),
        proxy_root=output_dir / "empty-authorized-proxy",
    )
    return {
        "schema_version": "bufotalin_source_acquisition.v1",
        "source_ref": SOURCE_REF,
        "structured": structured,
        "pdf": pdf,
        "agreement": {
            "procedure_labels_match": [
                row.get("label") for row in structured["procedure_inventory"]
            ]
            == [row.get("label") for row in pdf["procedure_inventory"]],
            "structured_procedure_count": len(structured["procedure_inventory"]),
            "pdf_procedure_count": len(pdf["procedure_inventory"]),
        },
    }


def _visual_chain(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = dict(document.get("result") or {})
    quality = dict(result.get("candidate_quality") or {})
    if document.get("accepted") is not True or result.get("accepted") is not True:
        raise ValueError("bufotalin_visual_chain_not_accepted")
    if quality.get("accepted") is not True:
        raise ValueError("bufotalin_visual_chain_quality_not_accepted")
    if int(quality.get("extraction_gap_count") or 0) != 0:
        raise ValueError("bufotalin_visual_chain_has_open_extraction_gaps")
    if quality.get("missing_expected_labels"):
        raise ValueError("bufotalin_visual_chain_missing_expected_labels")
    parsed = dict(result.get("parsed_output") or {})
    rows = [dict(value) for value in parsed.get("chain") or [] if isinstance(value, Mapping)]
    if len(rows) != 15:
        raise ValueError(f"bufotalin_visual_chain_step_count_invalid:{len(rows)}")
    return rows


def _paper_steps(
    retro_rows: list[dict[str, Any]],
    procedures: list[dict[str, Any]],
    *,
    structured_sha: str,
    pdf_sha: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    procedure_by_label = {
        str(row.get("label") or ""): dict(row) for row in procedures
    }
    molecules: dict[str, dict[str, Any]] = {}
    formula_rows: dict[str, dict[str, Any]] = {}
    synthesis_rows: list[dict[str, Any]] = []
    for retro in reversed(retro_rows):
        product_label = str(retro["product_label"])
        reactant_label = str(retro["reactant_label"])
        product_id = _paper_molecule_id(product_label)
        reactant_id = _paper_molecule_id(reactant_label)
        product_smiles = _canonical_smiles(str(retro["product_smiles"]))
        reactant_smiles = _canonical_smiles(str(retro["reactant_smiles"]))
        molecules[product_id] = _molecule(product_label, product_smiles)
        molecules[reactant_id] = _molecule(reactant_label, reactant_smiles)
        for label, smiles in (
            (product_label, product_smiles),
            (reactant_label, reactant_smiles),
        ):
            observed = _formula(smiles)
            expected = EXPECTED_FORMULAS[label]
            formula_rows[label] = {
                "label": label,
                "expected_formula": expected,
                "observed_formula": observed,
                "formula_match": observed == expected,
                "authority": "formula_cross_check_only",
                "grants_exact_stereochemistry": False,
            }
        source_label = "1" if product_label == "bufotalin" else product_label
        procedure = procedure_by_label[source_label]
        procedure_text = str(procedure.get("procedure_excerpt") or "")
        conditions = extract_source_conditions(
            procedure_text,
            source_amount_names=source_amount_reagent_names(procedure_text),
        )
        completeness = audit_condition_completeness(conditions)
        stage = len(synthesis_rows) + 6
        synthesis_rows.append(
            {
                "stage": stage,
                "step_id": f"paper:{product_label}",
                "edge_id": f"edge:paper:{product_label}",
                "product_label": product_label,
                "product_molecule_id": product_id,
                "product_smiles": product_smiles,
                "reactant_label": reactant_label,
                "reactant_molecule_id": reactant_id,
                "reactant_smiles": reactant_smiles,
                "segment": "paper_reported_15_step_total_synthesis",
                "origin_kind": "literature_visual_extraction",
                "source_ref": SOURCE_REF,
                "source_location": f"Experimental section · {procedure.get('name')}",
                "source_artifact_sha256": structured_sha,
                "source_pdf_sha256": pdf_sha,
                "source_procedure_excerpt": procedure_text,
                "conditions": conditions,
                "condition_completeness": completeness,
                "scheme_condition_summary": str(retro.get("condition") or ""),
                "confidence": "medium",
                "proof_level": 1,
                "reaction_validated": False,
                "exact_structure_bound": False,
                "reported_in_source": True,
                "visual_extraction_status": "accepted_materialized_candidate",
                "warning_codes": [
                    "current_host_reaction_validation_missing",
                    "visual_structure_candidate_not_curator_exact",
                ],
            }
        )
    formula_audit = [formula_rows[label] for label in ["11", *PAPER_SYNTHESIS_LABELS]]
    if not all(row["formula_match"] for row in formula_audit):
        raise ValueError("bufotalin_visual_formula_audit_failed")
    return synthesis_rows, molecules, formula_audit


def _upstream_steps(
    document: Mapping[str, Any],
    *,
    terminal_smiles: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    routes = [dict(value) for value in document.get("routes") or [] if isinstance(value, Mapping)]
    selected = next(
        (row for row in routes if int(row.get("route_rank") or -1) == 6),
        None,
    )
    if selected is None:
        raise ValueError("bufotalin_upstream_route_rank_6_missing")
    retro_steps = [
        dict(value) for value in selected.get("steps") or [] if isinstance(value, Mapping)
    ]
    if len(retro_steps) != 5:
        raise ValueError("bufotalin_upstream_step_count_invalid")
    verifier_findings = [
        dict(value)
        for value in dict(
            dict(selected.get("metrics") or {}).get("cascade_verifier") or {}
        ).get("findings")
        or []
        if isinstance(value, Mapping)
    ]
    findings_by_raw_index: dict[int, list[dict[str, Any]]] = {}
    for finding in verifier_findings:
        findings_by_raw_index.setdefault(
            int(finding.get("step_index") or 0), []
        ).append(finding)
    molecules: dict[str, dict[str, Any]] = {}
    node_by_smiles: dict[str, str] = {}

    def node(smiles: str, label: str) -> str:
        canonical = _canonical_smiles(smiles)
        if canonical in node_by_smiles:
            return node_by_smiles[canonical]
        molecule_id = f"m:planner:{len(node_by_smiles) + 1}"
        node_by_smiles[canonical] = molecule_id
        molecules[molecule_id] = _molecule(label, canonical)
        return molecule_id

    rows: list[dict[str, Any]] = []
    for synthesis_index, raw in enumerate(reversed(retro_steps), start=1):
        raw_index = int(raw.get("index") or 0)
        validation_findings = [
            {
                "finding_code": str(finding.get("reason") or "validation_failure"),
                "severity": "blocker"
                if float(finding.get("severity") or 0.0) >= 1.0
                else "warning",
                "message": str(finding.get("message") or ""),
                "evidence": dict(finding.get("evidence") or {}),
                "required_action": (
                    "Add the missing atom-contributing reactant(s), or replace "
                    "this edge with an atom-balanced literature precedent."
                ),
                "source": "historical_cascade_verifier",
            }
            for finding in findings_by_raw_index.get(raw_index, [])
        ]
        reactant_smiles = _canonical_smiles(str(raw.get("main_reactant") or ""))
        precursor_id = node(
            reactant_smiles,
            "planner stock hit · p-cresol" if synthesis_index == 1 else f"planner U{synthesis_index - 1}",
        )
        if raw_index == 0:
            product_id = "m:paper:11"
            product_smiles = _canonical_smiles(terminal_smiles)
            product_label = "11 · androstenedione"
        else:
            product_smiles = _canonical_smiles(str(raw.get("product") or ""))
            product_label = f"planner U{synthesis_index}"
            product_id = node(product_smiles, product_label)
        rows.append(
            {
                "stage": synthesis_index,
                "step_id": f"planner:{synthesis_index}",
                "edge_id": f"edge:planner:{synthesis_index}",
                "product_label": product_label,
                "product_molecule_id": product_id,
                "product_smiles": product_smiles,
                "reactant_label": str(molecules[precursor_id]["label"]),
                "reactant_molecule_id": precursor_id,
                "reactant_smiles": reactant_smiles,
                "segment": "historical_planner_upstream_hypothesis",
                "origin_kind": "chemenzy",
                "source_ref": "",
                "conditions": {},
                "confidence": "low",
                "proof_level": 0,
                "reaction_validated": False,
                "exact_structure_bound": False,
                "validation_findings": validation_findings,
                "planner_score": float(dict(raw.get("scores") or {}).get("confidence") or 0.0),
                "warning_codes": sorted(
                    {
                        "planner_hypothesis_not_literature_reported",
                        "current_host_reaction_validation_missing",
                        *(
                            ["historical_atom_balance_violation"]
                            if raw_index in {2, 4}
                            else []
                        ),
                        *(
                            ["stereochemical_representation_bridge_to_compound_11"]
                            if raw_index == 0
                            else []
                        ),
                    }
                ),
            }
        )
    return rows, molecules


def _graph_and_portfolio(
    steps: list[dict[str, Any]],
    *,
    molecule_rows: dict[str, dict[str, Any]],
    structured: Mapping[str, Any],
    pdf: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    route_id = "route:bufotalin-20-step-reported-candidate"
    observation_records: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    edge_proofs: dict[str, dict[str, Any]] = {}
    for step in steps:
        edge_id = str(step["edge_id"])
        is_paper = step["segment"] == "paper_reported_15_step_total_synthesis"
        observation_ids: list[str] = []
        if is_paper:
            observation_id = f"observation:{step['product_label']}"
            observation_ids = [observation_id]
            observation_records[observation_id] = {
                "schema_version": "source_reported_procedure_observation.v1",
                "record_id": observation_id,
                "source_ref": SOURCE_REF,
                "location_refs": [str(step["source_location"])],
                "source_artifact_sha256": str(step["source_artifact_sha256"]),
                "source_pdf_sha256": str(step["source_pdf_sha256"]),
                "conditions": dict(step["conditions"]),
                "condition_completeness": dict(step["condition_completeness"]),
                "procedure_excerpt": str(step["source_procedure_excerpt"]),
                "authority_scope": "source_reported_procedure_observation",
                "grants_exact_structure_identity": False,
                "grants_reaction_validation": False,
            }
        edges[edge_id] = {
            "product_molecule_id": str(step["product_molecule_id"]),
            "precursor_molecule_ids": [str(step["reactant_molecule_id"])],
            "origin_records": [
                {
                    "origin_kind": str(step["origin_kind"]),
                    "proposal_id": str(step["step_id"]),
                }
            ],
            "reaction_proofs": [
                {
                    "accepted": False,
                    "authority": "displayed_low_confidence_candidate",
                    "reason_codes": list(step["warning_codes"]),
                }
            ],
            "source_observation_record_ids": observation_ids,
            "validation_findings": list(step.get("validation_findings") or []),
        }
        edge_proofs[edge_id] = {
            "edge_id": edge_id,
            "achieved_level": 1 if is_paper else 0,
            "accepted": False,
            "reaction_validated": False,
            "exact_source_bound": False,
            "source_binding_ids": ["source:bufotalin-paper"] if is_paper else [],
            "exact_record_ids": [],
            "source_observation_record_ids": observation_ids,
            "independent_source_groups": [SOURCE_REF] if is_paper else [],
            "conflict_ids": [],
            "reasons": list(step["warning_codes"]),
        }
    leaf_id = str(steps[0]["reactant_molecule_id"])
    stock_observation_id = "stock:planner:p-cresol-benchmark-hit"
    for molecule_id, molecule in molecule_rows.items():
        molecule["is_leaf"] = molecule_id == leaf_id
        molecule["stock_closed"] = molecule_id == leaf_id
        molecule["stock_observation_ids"] = (
            [stock_observation_id] if molecule_id == leaf_id else []
        )
        if molecule_id == leaf_id:
            molecule["active_stock_observation_id"] = stock_observation_id
    graph_core = {
        "schema_version": "canonical_retrosynthesis_hypergraph.v1",
        "run_id": "bufotalin-v4-reported-candidate-20-step",
        "target_name": "bufotalin",
        "target_molecule_id": "m:paper:bufotalin",
        "revision": 1,
        "molecules": molecule_rows,
        "edges": edges,
        "source_bindings": {
            "source:bufotalin-paper": {
                "source_kind": "paper_si",
                "source_ref": SOURCE_REF,
                "title": PAPER_TITLE,
                "artifact_sha256": str(structured["source_fulltext_sha256"]),
                "source_pdf_sha256": str(pdf["source_pdf_sha256"]),
                "authority_scope": "reported_route_and_procedure_order_only",
            }
        },
        "exact_records": {},
        "source_observation_records": observation_records,
        "procedure_records": {},
        "stock_observations": {
            stock_observation_id: {
                "stock_observation_id": stock_observation_id,
                "canonical_smiles": str(
                    molecule_rows[leaf_id]["canonical_smiles"]
                ),
                "supplier": "historical planner benchmark stock set",
                "catalog_number": "benchmark:p-cresol",
                "authority_scope": "benchmark_search_stock_observation",
                "accepted": True,
            }
        },
        "conflicts": {},
        "hypotheses": {},
        "delta": {
            "rejected": [
                {
                    "kind": "route_acceptance",
                    "reasons": [
                        "paper_visual_structures_pending_curator_review",
                        "upstream_planner_edges_not_host_validated",
                    ],
                    "route_preserved_for_display": True,
                }
            ]
        },
    }
    graph_core["scientific_sha256"] = _digest(graph_core)
    route = {
        "schema_version": "proof_stitched_route.v1",
        "route_id": route_id,
        "route_family_id": "family:bufotalin-reported-plus-upstream",
        "strategy": "蟾毒灵 20 步 · 15 文献 + 5 规划补全",
        "edge_ids": [str(step["edge_id"]) for step in steps],
        "leaf_molecule_ids": [leaf_id],
        "root_edge_ids": [str(steps[-1]["edge_id"])],
        "module_selections": {},
        "minimum_edge_proof_level": 0,
        "all_edges_proven": False,
        "unproven_edge_ids": [str(step["edge_id"]) for step in steps],
        "stock_closure_rate": 1.0,
        "all_leaves_stock_closed": True,
        "open_leaf_molecule_ids": [],
        "independent_source_groups": [],
        "source_independence_met": False,
        "source_independence_required": True,
        "conflict_ids": [],
        "length": 20,
        "convergence_score": 0.0,
        "risk_score": 1.0,
        "complete": True,
        "selected": True,
        "pareto_optimal": True,
        "reported_in_source": True,
        "reported_source_refs": [SOURCE_REF],
        "reported_step_count": 15,
        "planner_hypothesis_step_count": 5,
        "semantics": {
            "weakest_edge_controls_route": True,
            "configured_boundary_closure_is_independent_of_edge_proof": True,
            "reported_segment_survives_unresolved_edges_for_display": True,
            "whole_20_step_route_is_not_fully_literature_reported": True,
            "full_synthesis_claim": False,
        },
    }
    portfolio_core = {
        "schema_version": "proof_stitched_route_portfolio.v1",
        "graph_revision": 1,
        "graph_scientific_sha256": graph_core["scientific_sha256"],
        "evidence_revision": 1,
        "proof_policy": {
            "stock_boundary": "benchmark_search",
            "minimum_edge_proof_level": 2,
        },
        "edge_proofs": edge_proofs,
        "leaf_proofs": {
            leaf_id: {
                "accepted": True,
                "active_stock_observation_id": stock_observation_id,
            }
        },
        "route_candidates": [route],
        "selected_routes": [route],
        "route_modules": [],
        "deficits": [
            {
                "deficit_id": "deficit:bufotalin-paper-structure-curation",
                "route_id": route_id,
                "kind": "source_visual_structure_curation",
                "edge_ids": [str(step["edge_id"]) for step in steps[5:]],
            },
            {
                "deficit_id": "deficit:bufotalin-upstream-validation",
                "route_id": route_id,
                "kind": "reaction_validation",
                "edge_ids": [str(step["edge_id"]) for step in steps[:5]],
            },
        ],
        "metrics": {
            "selected_route_count": 1,
            "complete_route_count": 1,
            "mean_length": 20.0,
            "paper_reported_step_count": 15,
            "planner_hypothesis_step_count": 5,
        },
        "closeout": {
            "schema_version": "retrosynthesis_closeout.v1",
            "decision": "route_closed_proof_unresolved",
            "accepted": False,
            "complete_route_count": 1,
            "selected_route_count": 1,
            "deficit_count": 2,
            "reasons": [
                "paper_visual_structures_pending_curator_review",
                "upstream_planner_edges_not_host_validated",
            ],
        },
        "accepted": False,
        "semantics": {
            "configured_boundary_route_is_closed": True,
            "reported_route_is_displayed_even_when_unresolved": True,
            "display_does_not_grant_solved_status": True,
        },
    }
    portfolio_core["content_sha256"] = _digest(portfolio_core)
    return graph_core, portfolio_core


def _dossier(
    steps: list[dict[str, Any]],
    *,
    structured: Mapping[str, Any],
    pdf: Mapping[str, Any],
    formula_audit: list[dict[str, Any]],
    visual_path: Path,
    upstream_path: Path,
    workbench: Mapping[str, Any],
    forest: Mapping[str, Any],
) -> dict[str, Any]:
    branch = next(iter(forest.get("branches") or []), {})
    visual_result = dict(_read_json(visual_path).get("result") or {})
    visual_quality = dict(visual_result.get("candidate_quality") or {})
    visual_smiles = dict(visual_quality.get("smiles_precheck") or {})
    return {
        "schema_version": "bufotalin_20_step_reported_candidate_dossier.v1",
        "target": {
            "name": "bufotalin",
            "canonical_smiles": str(dict(workbench.get("target") or {}).get("canonical_smiles") or ""),
            "formula": EXPECTED_FORMULAS["bufotalin"],
        },
        "claim_status": "route_closed_proof_unresolved",
        "solved": False,
        "full_synthesis_claim": False,
        "route": {
            "direction": "synthesis_precursor_to_target",
            "step_count": 20,
            "paper_reported_step_count": 15,
            "planner_hypothesis_step_count": 5,
            "proof_distribution": {
                "L0_rejected": sum(
                    "historical_atom_balance_violation"
                    in set(row.get("warning_codes") or [])
                    for row in steps
                ),
                "L0_advisory": sum(
                    row.get("proof_level") == 0
                    and "historical_atom_balance_violation"
                    not in set(row.get("warning_codes") or [])
                    for row in steps
                ),
                "L1_source_reported": sum(
                    row.get("proof_level") == 1 for row in steps
                ),
            },
            "displayed_despite_unresolved_edges": True,
            "configured_boundary_closed": True,
            "closure_profile": "exploration_closed",
            "evidence_complete": False,
            "workbench_branch_kind": str(branch.get("kind") or ""),
            "workbench_proof_tier": str(branch.get("proof_tier") or ""),
            "sequence": [steps[0]["reactant_label"], *[row["product_label"] for row in steps]],
        },
        "source_evidence": {
            "source_ref": SOURCE_REF,
            "title": PAPER_TITLE,
            "structured_json_sha256": str(structured["source_fulltext_sha256"]),
            "structured_procedure_count": len(structured["procedure_inventory"]),
            "pdf_sha256": str(pdf["source_pdf_sha256"]),
            "pdf_text_sha256": str(pdf["fulltext_text_sha256"]),
            "pdf_procedure_count": len(pdf["procedure_inventory"]),
            "procedure_sequences_agree": [
                row.get("label") for row in structured["procedure_inventory"]
            ]
            == [row.get("label") for row in pdf["procedure_inventory"]],
            "visual_extraction": {
                "accepted": visual_result.get("accepted") is True
                and visual_quality.get("accepted") is True,
                "candidate_step_count": int(
                    visual_result.get("candidate_step_count") or 0
                ),
                "extraction_gap_count": int(
                    visual_quality.get("extraction_gap_count") or 0
                ),
                "missing_expected_labels": list(
                    visual_quality.get("missing_expected_labels") or []
                ),
                "rdkit_valid_smiles_count": int(
                    visual_smiles.get("valid_smiles_count") or 0
                ),
                "rdkit_invalid_smiles_count": int(
                    visual_smiles.get("invalid_smiles_count") or 0
                ),
            },
        },
        "formula_audit": formula_audit,
        "steps": steps,
        "input_artifacts": {
            "visual_structure_draft": str(visual_path),
            "historical_upstream_planner_routes": str(upstream_path),
        },
        "remaining_deficits": [
            "Curator approval of the 15 visual structure translations and stereochemistry",
            "Current-host reaction validation of all 20 edges",
            "Replacement of the five weak upstream planner hypotheses with chemically defensible precedent",
            "Procurement or in-house stock proof for every terminal leaf",
        ],
        "semantics": {
            "route_closure_is_independent_of_proof_completion": True,
            "paper_reported_route_order_is_preserved": True,
            "low_confidence_steps_are_visible_and_warning_encoded": True,
            "formula_match_does_not_grant_exact_structure": True,
            "source_conditions_do_not_grant_reaction_validation": True,
            "historical_false_positive_solved_verdict_not_reused": True,
        },
    }


def _assert_chain(steps: list[dict[str, Any]]) -> None:
    for left, right in zip(steps, steps[1:], strict=False):
        if str(left["product_molecule_id"]) != str(right["reactant_molecule_id"]):
            raise ValueError(
                "bufotalin_20_step_chain_disconnected:"
                f"{left['step_id']}->{right['step_id']}"
            )


def _paper_molecule_id(label: str) -> str:
    return f"m:paper:{label.casefold()}"


def _molecule(label: str, smiles: str) -> dict[str, Any]:
    return {
        "label": "Bufotalin" if label == "bufotalin" else f"Compound {label}",
        "canonical_smiles": smiles,
        "formula": _formula(smiles),
        "is_leaf": False,
        "stock_closed": False,
        "stock_observation_ids": [],
    }


def _canonical_smiles(value: str) -> str:
    molecule = Chem.MolFromSmiles(str(value or ""))
    if molecule is None:
        raise ValueError(f"bufotalin_showcase_smiles_invalid:{value}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _formula(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("bufotalin_showcase_formula_smiles_invalid")
    return str(rdMolDescriptors.CalcMolFormula(molecule))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"bufotalin_showcase_json_invalid:{path}")
    return dict(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/shared/bufotalin_v4_reported_candidate_20260715",
    )
    parser.add_argument(
        "--structured-json",
        default=(
            "results/.autoplanner/bufotalin-v4-canonical/evidence/"
            "local_pdf_proxy/publisher-provider/"
            "pdfreq_bufotalin-v4-canonical_0c64ec8be58cd7ee/article/"
            "article-data.json"
        ),
    )
    parser.add_argument(
        "--pdf",
        default="1-s2.0-S0040402025001668-main.pdf",
    )
    parser.add_argument(
        "--visual-chain",
        default=(
            "results/shared/bufotalin_agentic_blackboard_full_retry_v5_20260609/"
            "r3_extract_visual_literature_chain_"
            "visual_literature_chain_extraction_result_v1.json"
        ),
    )
    parser.add_argument(
        "--upstream-routes",
        default=(
            "results/shared/bufotalin_full_exact_stitch_rerun_20260622_073847/"
            "route_expansion_subgoals/"
            "01_source_detail_literature_terminal_raw_result.json"
        ),
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
