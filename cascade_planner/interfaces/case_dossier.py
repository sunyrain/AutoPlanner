"""Compile a concise evidence dossier into one deterministic replay pack."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.harness.reaction_step_verifier import verify_reaction_step

from .case_dossier_compiler import (
    canonical_smiles,
    compile_inventory,
    compile_routes,
    dataclass_dict,
    global_plan,
    source_coverage,
)
from .case_dossier_contract import (
    CASE_DOSSIER_SCHEMA,
    CaseDossierError,
    json_copy,
    load_case_dossier,
    with_case_dossier_digest,
)
from .replay_contract import (
    ReplayPackError,
    dataclass_value,
    validate_replay_pack,
    with_replay_pack_digest,
)


ReactionMapper = Callable[[list[str]], list[str]]


def compile_case_dossier(
    value: str | Path | Mapping[str, Any],
    *,
    reaction_mapper: ReactionMapper | None = None,
) -> dict[str, Any]:
    dossier = load_case_dossier(value)
    for key in (
        "case_id",
        "target",
        "acceptance",
        "budget",
        "routes",
        "sources",
        "inventory",
    ):
        if key not in dossier:
            raise CaseDossierError(f"case_dossier_required_field_missing:{key}")
    target = dict(dossier["target"])
    target_smiles = canonical_smiles(target.get("smiles"))
    if not str(target.get("name") or "").strip() or not target_smiles:
        raise CaseDossierError("case_dossier_target_invalid")
    target["smiles"] = target_smiles
    acceptance = dataclass_value(RetrosynthesisAcceptanceSpec, dossier["acceptance"])
    budget = dataclass_value(RetrosynthesisRunBudget, dossier["budget"])
    if budget.max_model_invocations or budget.max_visual_invocations:
        raise CaseDossierError("case_dossier_must_be_model_free")

    routes, edges, leaves = compile_routes(dossier["routes"], target_smiles)
    if acceptance.require_distinct_edge_sets:
        edge_sets = [frozenset(route["edge_digests"]) for route in routes]
        if len(set(edge_sets)) != len(edge_sets):
            raise CaseDossierError("case_dossier_route_edge_sets_not_distinct")
    source_edges, source_groups = source_coverage(dossier["sources"])
    missing_source = sorted(set(edges) - source_edges)
    if missing_source:
        raise CaseDossierError(
            "case_dossier_reaction_without_exact_source:" + ",".join(missing_source)
        )
    if len(routes) < acceptance.minimum_complete_routes:
        raise CaseDossierError("case_dossier_route_count_below_acceptance")
    if len(source_groups) < acceptance.minimum_independent_source_groups:
        raise CaseDossierError("case_dossier_source_diversity_below_acceptance")
    inventory = compile_inventory(dossier["inventory"], leaves)

    missing_maps = [
        edge["reaction_smiles"]
        for edge in edges.values()
        if not edge["mapped_reaction_smiles"]
    ]
    mapped_values: list[str] = []
    if missing_maps:
        if reaction_mapper is None:
            raise CaseDossierError("case_dossier_atom_mapping_required")
        mapped_values = list(reaction_mapper(missing_maps))
        if len(mapped_values) != len(missing_maps):
            raise CaseDossierError("case_dossier_atom_mapper_count_mismatch")
    mapped_iter = iter(mapped_values)
    reactions: list[dict[str, Any]] = []
    for edge_digest, edge in sorted(edges.items()):
        mapped = edge["mapped_reaction_smiles"] or next(mapped_iter)
        proof = verify_reaction_step(
            {
                "product_smiles": edge["product_smiles"],
                "reactant_smiles": edge["reactant_smiles"],
                "mapped_reaction_smiles": mapped,
            },
            source_supported_multicentre=True,
        )
        if proof.get("accepted") is not True:
            raise CaseDossierError(
                "case_dossier_reaction_validation_failed:"
                + edge_digest
                + ":"
                + ",".join(proof.get("reasons") or [])
            )
        reactions.append(
            {
                "edge_digest": edge_digest,
                "product_smiles": edge["product_smiles"],
                "reactant_smiles": edge["reactant_smiles"],
                "mapped_reaction_smiles": mapped,
                "mapping_authority": (
                    "dossier_supplied_host_replayed_mapping"
                    if edge["mapped_reaction_smiles"]
                    else "local_rxnmapper_with_deterministic_atom_conservation_audit"
                ),
                "reaction_step_verifier_version": proof["validator_version"],
            }
        )

    if budget.max_accepted_expansions < len(reactions):
        raise CaseDossierError("case_dossier_expansion_budget_too_small")
    minimum_attempts = 2 * len(reactions) + 2 * len(dossier["sources"]) + 1
    if budget.max_attempt_runs < minimum_attempts:
        raise CaseDossierError("case_dossier_attempt_budget_too_small")
    pack = with_replay_pack_digest(
        {
            "schema_version": "retrosynthesis_replay_pack.v1",
            "case_id": str(dossier["case_id"]),
            "created_at": str(dossier.get("created_at") or ""),
            "target": target,
            "acceptance": dataclass_dict(acceptance),
            "budget": dataclass_dict(budget),
            "global_plan": global_plan(dossier, routes, target_smiles),
            "sources": json_copy(dossier["sources"]),
            "reactions": reactions,
            "inventory": inventory,
            "expected": {
                "accepted": True,
                "complete_route_count": len(routes),
                "selected_route_count": len(routes),
                "hyperedge_count": len(reactions),
                "validated_edge_count": len(reactions),
                "exact_record_count": sum(
                    len(source.get("rows") or []) for source in dossier["sources"]
                ),
                "stock_terminal_count": len(leaves),
                "independent_source_groups": sorted(source_groups),
                "accepted_expansion_count": len(reactions),
                "model_invocations": 0,
                "visual_invocations": 0,
            },
            "provenance": {
                "compiled_from": CASE_DOSSIER_SCHEMA,
                "dossier_sha256": str(dossier["content_sha256"]),
                "compiler": "autoplanner.case_dossier.v1",
            },
        }
    )
    try:
        validate_replay_pack(pack)
    except ReplayPackError as exc:
        raise CaseDossierError(str(exc)) from exc
    return pack


def local_rxnmapper(reactions: list[str]) -> list[str]:
    """Map a bounded dossier locally; never call a hosted model or network."""

    try:
        from rxnmapper import RXNMapper
    except ImportError as exc:
        raise CaseDossierError("rxnmapper_not_installed") from exc
    mapper = RXNMapper()
    return [
        str(row.get("mapped_rxn") or "")
        for row in mapper.get_attention_guided_atom_maps(reactions)
    ]


def write_compiled_replay_pack(
    dossier: str | Path | Mapping[str, Any],
    output: str | Path,
    *,
    map_missing: bool = False,
) -> dict[str, Any]:
    pack = compile_case_dossier(
        dossier,
        reaction_mapper=local_rxnmapper if map_missing else None,
    )
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(pack, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return {
        "schema_version": "retrosynthesis_case_compile_result.v1",
        "case_id": pack["case_id"],
        "output": str(destination),
        "pack_sha256": pack["content_sha256"],
        "route_count": pack["expected"]["complete_route_count"],
        "reaction_count": len(pack["reactions"]),
        "source_count": len(pack["sources"]),
        "stock_terminal_count": pack["expected"]["stock_terminal_count"],
        "model_invocations": 0,
    }


__all__ = [
    "CASE_DOSSIER_SCHEMA",
    "CaseDossierError",
    "compile_case_dossier",
    "load_case_dossier",
    "local_rxnmapper",
    "with_case_dossier_digest",
    "write_compiled_replay_pack",
]
