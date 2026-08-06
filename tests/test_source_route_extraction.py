from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisRunBudget,
)
from cascade_planner.application.run_kernel import RunLimits, RunSpec
from cascade_planner.harness.source_route_extraction import (
    _clean_ingredient_name,
    compile_deterministic_source_route_observation,
)
from cascade_planner.interfaces.target_solver_stages import (
    materialize_discovered_source_routes,
)
from cascade_planner.orchestration.retrosynthesis_service import (
    RetrosynthesisCampaignService,
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def test_source_ingredient_name_drops_numbered_preparation_qualifier() -> None:
    assert _clean_ingredient_name(
        "the cyclohexylfulvene as synthesized above (1)"
    ) == "cyclohexylfulvene"


def test_source_route_compiler_keeps_target_connected_branching_dag() -> None:
    names = {
        "dimethyl ether": "COC",
        "acetonitrile": "CC#N",
    }

    def resolve(name: str) -> str:
        if name not in names:
            raise RuntimeError("not found")
        return names[name]

    document = {
        "source_ref": "patent:TEST1",
        "source_pdf_sha256": "a" * 64,
        "source_artifact_sha256": "a" * 64,
        "procedures": [
            {
                "label": "1",
                "name": "ethanol",
                "page_number": 1,
                "procedure": "A flask was charged with dimethyl ether (4.6 g, 100 mmol). The reaction was stirred for 2 h.",
                "structure_parse_accepted": True,
                "canonical_smiles": "CCO",
            },
            {
                "label": "2",
                "name": "ethylamine",
                "page_number": 2,
                "procedure": "A flask was charged with acetonitrile (4.1 g, 100 mmol). The reaction was stirred for 3 h.",
                "structure_parse_accepted": True,
                "canonical_smiles": "CCN",
            },
            {
                "label": "3",
                "name": "2-ethoxyethanamine",
                "page_number": 3,
                "procedure": "Ethanol (4.6 g, 100 mmol) and ethylamine (4.5 g, 100 mmol) were added. The reaction mixture was stirred for 4 h.",
                "structure_parse_accepted": True,
                "canonical_smiles": "CCOCCN",
            },
            {
                "label": "4",
                "name": "cyclohexane",
                "page_number": 4,
                "procedure": "A reaction mixture was stirred for 1 h.",
                "structure_parse_accepted": True,
                "canonical_smiles": "C1CCCCC1",
            },
        ],
    }
    observation = compile_deterministic_source_route_observation(
        document,
        structure_resolver=resolve,
        anchor_smiles=["CCOCCN"],
    )

    assert observation["proposal_count"] == 3
    assert observation["unconnected_proposal_count"] == 0
    assert {
        row["source_location"]["label"] for row in observation["proposals"]
    } == {"1", "2", "3"}
    target = next(
        row for row in observation["proposals"] if row["product_smiles"] == "CCOCCN"
    )
    assert set(target["precursor_smiles"]) == {"CCN", "CCO"}
    assert target["semantics"]["proposal_only"] is True


def test_source_route_recovers_narrative_product_and_source_abbreviation() -> None:
    names = {
        "ethyl acetate": "CCOC(C)=O",
        "ethanol": "CCO",
        "acetyl chloride": "CC(=O)Cl",
    }

    def resolve(name: str) -> str:
        value = names.get(name.casefold())
        if not value:
            raise RuntimeError("not found")
        return value

    document = {
        "source_ref": "patent:NARRATIVE",
        "source_artifact_sha256": "b" * 64,
        "source_name_aliases": {"dmb-s-mmp": "acetyl chloride"},
        "procedures": [
            {
                "label": "0106",
                "name": (
                    "Whole-cell catalytic synthesis of ethyl acetate from "
                    "ethanol and DMB-S-MMP was performed as described"
                ),
                "page_number": 20,
                "procedure": "The reaction mixture was stirred and isolated.",
            }
        ],
    }

    observation = compile_deterministic_source_route_observation(
        document,
        structure_resolver=resolve,
        anchor_smiles=["CCOC(C)=O"],
    )

    assert observation["proposal_count"] == 1
    proposal = observation["proposals"][0]
    assert proposal["product_structure_recovery_mode"] == (
        "narrative_product_name_advisory"
    )
    assert set(proposal["precursor_smiles"]) == {"CCO", "CC(=O)Cl"}


def test_source_route_expands_leading_source_alias_with_chemical_suffix() -> None:
    names = {
        "simvastatin acid": "CCOC(C)=O",
        "monacolin j acid": "CCO",
        "acetyl chloride": "CC(=O)Cl",
    }

    def resolve(name: str) -> str:
        value = names.get(name.casefold())
        if not value:
            raise RuntimeError("not found")
        return value

    document = {
        "source_ref": "patent:SOURCE-ALIASES",
        "source_artifact_sha256": "c" * 64,
        "source_name_aliases": {
            "mj": "Monacolin J",
            "dmb-s-mmp": "acetyl chloride",
        },
        "procedures": [
            {
                "label": "90",
                "name": (
                    "Whole-cell synthesis of simvastatin acid from MJ acid "
                    "and DMB-S-MMP was performed as described"
                ),
                "page_number": 21,
                "procedure": "The reaction was monitored and isolated.",
            }
        ],
    }

    observation = compile_deterministic_source_route_observation(
        document,
        structure_resolver=resolve,
        anchor_smiles=["CCOC(C)=O"],
    )

    assert observation["proposal_count"] == 1
    proposal = observation["proposals"][0]
    assert set(proposal["reactant_names"]) == {"MJ acid", "DMB-S-MMP"}
    assert set(proposal["precursor_smiles"]) == {"CCO", "CC(=O)Cl"}


def test_source_route_connects_source_hydroxy_acid_to_target_lactone() -> None:
    names = {
        "4-hydroxybutanoic acid": "OCCCC(=O)O",
        "4-bromobutanoic acid": "O=C(O)CCCBr",
        "water": "O",
    }

    def resolve(name: str) -> str:
        value = names.get(name.casefold())
        if not value:
            raise RuntimeError("not found")
        return value

    document = {
        "source_ref": "doi:10.1000/FORM-BRIDGE",
        "source_artifact_sha256": "d" * 64,
        "procedures": [
            {
                "label": "1",
                "name": (
                    "Synthesis of 4-hydroxybutanoic acid from "
                    "4-bromobutanoic acid and water was performed"
                ),
                "page_number": 1,
                "procedure": "The reaction mixture was isolated.",
            }
        ],
    }

    observation = compile_deterministic_source_route_observation(
        document,
        structure_resolver=resolve,
        anchor_smiles=["O=C1OCCC1"],
    )

    assert observation["proposal_count"] == 2
    bridge = next(
        row
        for row in observation["proposals"]
        if row["proposal_id"].startswith("source-form-bridge:")
    )
    assert bridge["product_smiles"] == "O=C1CCCO1"
    assert bridge["precursor_smiles"] == ["O=C(O)CCCO"]
    assert bridge["semantics"]["structural_form_equivalence_only"] is True


def test_discovered_source_route_uses_canonical_route_and_edge_ingestion(
    tmp_path: Path,
) -> None:
    spec = RunSpec(
        run_id="source-route",
        target_name="target",
        target_smiles="CCOCCN",
        created_at="2026-07-14T00:00:00Z",
        limits=RunLimits(
            model=RetrosynthesisRunBudget(
                max_model_invocations=0,
                max_accepted_expansions=8,
                max_attempt_runs=12,
            ),
            max_total_tasks=32,
        ),
    )
    service = RetrosynthesisCampaignService.create(
        tmp_path / "runtime",
        tmp_path / "run",
        spec=spec,
    )
    body = {
        "schema_version": "deterministic_source_route_observation.v1",
        "source_ref": "patent:TEST1",
        "source_artifact_sha256": "a" * 64,
        "route_family": {
            "route_family_id": "source-route-family:test",
            "family_key": "source-route-family:test",
            "strategy": "source DAG",
            "selected": True,
        },
        "proposal_count": 2,
        "proposals": [
            {
                "proposal_id": "source-route:upstream",
                "product_smiles": "CCO",
                "precursor_smiles": ["COC"],
                "source_ref": "patent:TEST1",
                "source_location": {"kind": "pdf_page", "page_number": 1},
                "origin_kind": "literature_source_route",
            },
            {
                "proposal_id": "source-route:target",
                "product_smiles": "CCOCCN",
                "precursor_smiles": ["CCO", "CCN"],
                "source_ref": "patent:TEST1",
                "source_location": {"kind": "pdf_page", "page_number": 2},
                "origin_kind": "literature_source_route",
            },
        ],
        "diagnostics": [],
        "resolver_attempt_count": 0,
        "resolved_procedure_count": 2,
        "unconnected_proposal_count": 0,
        "semantics": {"proposals_grant_no_exact_or_reaction_proof": True},
    }
    observation = {**body, "content_sha256": _digest(body)}
    discovery = {
        "sources": [
            {
                "source_ref": "patent:TEST1",
                "source_route_observation": observation,
            }
        ]
    }

    result = materialize_discovered_source_routes(service, discovery)
    graph = service.graph_store.load()

    assert result["status"] == "completed"
    assert result["proposal_count"] == 2
    assert result["materialization_command_count"] == 2
    assert len(result["materialized_edge_ids"]) == 2
    assert len(graph["route_families"]) == 1
    route = next(iter(graph["route_families"].values()))
    assert len(route["edge_ids"]) == 2
    assert all(
        edge["origin_records"][0]["origin_kind"] == "literature_source_route"
        for edge in graph["edges"].values()
    )
    assert service.kernel.state.accepted_expansion_count == 2

    replay = materialize_discovered_source_routes(service, discovery)
    assert replay["status"] == "reused_or_empty"
    assert replay["materialization_command_count"] == 0
    assert replay["materialized_edge_ids"] == result["materialized_edge_ids"]
    assert service.kernel.state.accepted_expansion_count == 2


def test_empty_source_route_observation_is_not_reported_as_host_rejection(
    tmp_path: Path,
) -> None:
    spec = RunSpec(
        run_id="empty-source-route",
        target_name="target",
        target_smiles="CCO",
        created_at="2026-07-16T00:00:00Z",
        limits=RunLimits(
            model=RetrosynthesisRunBudget(
                max_model_invocations=0,
                max_accepted_expansions=2,
                max_attempt_runs=4,
            ),
            max_total_tasks=8,
        ),
    )
    service = RetrosynthesisCampaignService.create(
        tmp_path / "runtime",
        tmp_path / "run",
        spec=spec,
    )
    body = {
        "schema_version": "deterministic_source_route_observation.v1",
        "source_ref": "patent:EMPTY",
        "source_artifact_sha256": "b" * 64,
        "route_family": {
            "route_family_id": "source-route-family:empty",
            "family_key": "source-route-family:empty",
            "strategy": "source DAG",
            "selected": True,
        },
        "proposal_count": 0,
        "proposals": [],
        "diagnostics": [],
        "resolver_attempt_count": 0,
        "resolved_procedure_count": 0,
        "unconnected_proposal_count": 0,
        "semantics": {"proposals_grant_no_exact_or_reaction_proof": True},
    }
    observation = {**body, "content_sha256": _digest(body)}

    result = materialize_discovered_source_routes(
        service,
        {
            "sources": [
                {
                    "source_ref": "patent:EMPTY",
                    "source_route_observation": observation,
                }
            ]
        },
    )

    assert result["status"] == "not_needed"
    assert result["proposal_count"] == 0
    assert result["rejected_observations"] == []
