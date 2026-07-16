#!/usr/bin/env python3
"""Build and offline-replay the real EP3381900A1 Vismodegib edge gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.application.reaction_condition_records import (
    audit_condition_completeness,
)
from cascade_planner.application.route_workbench import compile_route_workbench
from cascade_planner.harness.deterministic_literature_registry import (
    PARSER_AUTHORITY_ID,
    build_deterministic_literature_resolvers,
    compile_deterministic_literature_step_registry,
)
from cascade_planner.harness.source_patent_xml import (
    materialize_primary_patent_xml,
)
from cascade_planner.interfaces.patent_source_discovery import (
    fetch_bounded_bytes,
)
from cascade_planner.harness.v4_route_workbench import (
    render_v4_route_workbench_html,
)


PUBLICATION = "EP3381900A1"
SOURCE_REF = f"patent:{PUBLICATION}"
SOURCE_URL = (
    "https://data.epo.org/publication-server/rest/v1.2/patents/"
    "EP3381900NWA1/document.xml"
)
TARGET_NAME = "Vismodegib"
PRODUCT_SMILES = "CS(=O)(=O)c1ccc(C(=O)Nc2ccc(Cl)c(-c3ccccn3)c2)c(Cl)c1"
REACTANT_SMILES = (
    "Nc1ccc(Cl)c(-c2ccccn2)c1",
    "CS(=O)(=O)c1ccc(C(=O)Cl)c(Cl)c1",
)
SOURCE_STRUCTURE_NAMES = (
    "Vismodegib",
    "4-chloro-3-(pyridin-2-yl)aniline",
    "2-chloro-4-(methylsulfonyl)benzoyl chloride",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the official EPO ST.36 XML once, pin deterministic "
            "name resolution, and prove two network-free registry replays agree."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results"
            / ".autoplanner"
            / "patent-real-case-gates"
            / "vismodegib-ep3381900a1"
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="refuse all source/resolver network access and replay cached CAS inputs",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    content = _source_bytes(output_dir, offline=args.offline)
    materialization = materialize_primary_patent_xml(
        content=content,
        publication=PUBLICATION,
        source_ref=SOURCE_REF,
        source_url=SOURCE_URL,
        output_dir=output_dir / "source",
        target_terms=[
            TARGET_NAME,
            "Synthesis of Vismodegib",
            "intermediate 3",
            "intermediate 4",
            *SOURCE_STRUCTURE_NAMES[1:],
        ],
    )
    if materialization.get("status") != "completed":
        raise RuntimeError(
            "official_epo_xml_materialization_failed:"
            + ",".join(materialization.get("reasons") or [])
        )

    expected_product = _canonical_smiles(PRODUCT_SMILES)
    expected_reactants = sorted(_canonical_smiles(value) for value in REACTANT_SMILES)
    snapshot = _resolver_snapshot(
        output_dir,
        expected_product=expected_product,
        expected_reactants=expected_reactants,
        offline=args.offline,
    )
    resolve_structure, resolve_names = _snapshot_resolvers(snapshot)
    step = {
        "step_id": "vismodegib-final-amide",
        "product_smiles": expected_product,
        "reactant_smiles": expected_reactants,
        "source_ref": SOURCE_REF,
        "source_text_companions": [dict(materialization["companion"])],
    }
    audits = [
        compile_deterministic_literature_step_registry(
            [step],
            registry_path=output_dir / f"offline-registry-{label}.json",
            structure_resolver=resolve_structure,
            candidate_name_resolver=resolve_names,
        )
        for label in ("a", "b")
    ]
    registries = [
        json.loads(
            (output_dir / f"offline-registry-{label}.json").read_text(
                encoding="utf-8"
            )
        )
        for label in ("a", "b")
    ]
    bindings = [dict(audit["records"][0].get("binding") or {}) for audit in audits]
    conditions = dict(bindings[0].get("source_conditions") or {})
    completeness = audit_condition_completeness(conditions)
    registry_digests = [str(row.get("content_sha256") or "") for row in registries]
    binding_ids = [str(row.get("binding_id") or "") for row in bindings]
    parser_audit = dict(bindings[0].get("parser_audit") or {})
    location = dict(bindings[0].get("source_location") or {})
    accepted = bool(
        all(audit.get("approved_binding_count") == 1 for audit in audits)
        and len(set(registry_digests)) == 1
        and len(set(binding_ids)) == 1
        and bindings[0].get("source_artifact_kind") == "xml"
        and location.get("kind") == "xml_element_range"
        and parser_audit.get("source_text_authority") == "hash_bound_primary_xml"
        and completeness.get("complete") is True
        and conditions.get("yield_percent") == 52.0
        and conditions.get("time") == "17h"
        and "tetrahydrofuran" in (conditions.get("solvent") or [])
        and "triethylamine" in (conditions.get("base") or [])
    )
    acceptance: dict[str, Any] = {
        "schema_version": "real_patent_procedure_gate.v1",
        "case_id": "blind-vismodegib-01",
        "target_name": TARGET_NAME,
        "publication": PUBLICATION,
        "accepted": accepted,
        "official_source": {
            "authority": "European Patent Office Publication Server",
            "format": "epo_st36_xml.v1",
            "source_url": SOURCE_URL,
            "artifact_path": str(materialization["artifact_path"]),
            "artifact_sha256": str(materialization["artifact_sha256"]),
            "element_count": int(materialization["element_count"]),
            "selected_element_count": int(
                materialization["selected_element_count"]
            ),
            "sections": list(materialization["sections"]),
        },
        "exact_edge": {
            "product_smiles": expected_product,
            "reactant_smiles": expected_reactants,
            "reaction_digest": str(bindings[0].get("reaction_digest") or ""),
            "binding_id": binding_ids[0],
            "source_location": location,
            "procedure_text_sha256": str(
                parser_audit.get("procedure_text_sha256") or ""
            ),
            "reactant_match_modes": list(
                parser_audit.get("reactant_match_modes") or []
            ),
        },
        "procedure": {
            "conditions": conditions,
            "condition_completeness": completeness,
        },
        "offline_replay": {
            "network_used_during_registry_replays": False,
            "resolver_snapshot_path": str(
                output_dir / "resolver-snapshot.json"
            ),
            "resolver_snapshot_sha256": str(snapshot["content_sha256"]),
            "registry_content_sha256": registry_digests,
            "registry_digests_equal": len(set(registry_digests)) == 1,
            "binding_ids_equal": len(set(binding_ids)) == 1,
            "model_invocations": 0,
            "visual_invocations": 0,
        },
        "semantics": {
            "search_metadata_grants_no_authority": True,
            "full_xml_and_selected_range_are_digest_bound": True,
            "source_names_are_resolved_before_offline_snapshot": True,
            "missing_conditions_are_not_inferred": True,
            "one_real_case_does_not_satisfy_the_three_case_gate": True,
        },
    }
    acceptance["showcase"] = _write_workbench(
        output_dir,
        binding=bindings[0],
        completeness=completeness,
        materialization=materialization,
    )
    acceptance["content_sha256"] = _digest(acceptance)
    _write_json_atomic(output_dir / "acceptance.json", acceptance)
    print(json.dumps(acceptance, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if accepted else 2


def _source_bytes(output_dir: Path, *, offline: bool) -> bytes:
    manifest_path = output_dir / "source" / "primary-patent-xml-materialization.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        path = Path(str(manifest.get("artifact_path") or ""))
        expected = str(manifest.get("artifact_sha256") or "")
        if path.is_file():
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() == expected:
                return content
    if offline:
        raise RuntimeError("offline_epo_xml_cache_missing_or_invalid")
    return fetch_bounded_bytes(SOURCE_URL, 30.0, 20_000_000)


def _write_workbench(
    output_dir: Path,
    *,
    binding: Mapping[str, Any],
    completeness: Mapping[str, Any],
    materialization: Mapping[str, Any],
) -> dict[str, str]:
    source_binding_id = str(binding.get("binding_id") or "")
    location = dict(binding.get("source_location") or {})
    location_ref = (
        f"{PUBLICATION}:xml:{location.get('start_element_id') or ''}-"
        f"{location.get('end_element_id') or ''}"
    )
    procedure_id = "procedure:vismodegib-final-amide"
    record_id = "exact:vismodegib-final-amide"
    graph_digest = "vismodegib-ep3381900a1-graph-v1"
    product = _canonical_smiles(PRODUCT_SMILES)
    reactants = [_canonical_smiles(value) for value in REACTANT_SMILES]
    graph = {
        "schema_version": "canonical_retrosynthesis_hypergraph.v1",
        "run_id": "vismodegib-ep3381900a1-procedure-showcase",
        "target_name": TARGET_NAME,
        "target_molecule_id": "m:vismodegib",
        "revision": 1,
        "scientific_sha256": graph_digest,
        "molecules": {
            "m:vismodegib": {
                "canonical_smiles": product,
                "is_leaf": False,
                "stock_observation_ids": [],
            },
            "m:aniline": {
                "canonical_smiles": reactants[0],
                "is_leaf": True,
                "stock_observation_ids": [],
                "stock_closed": False,
            },
            "m:acid-chloride": {
                "canonical_smiles": reactants[1],
                "is_leaf": True,
                "stock_observation_ids": [],
                "stock_closed": False,
            },
        },
        "edges": {
            "edge:vismodegib-final-amide": {
                "product_molecule_id": "m:vismodegib",
                "precursor_molecule_ids": ["m:aniline", "m:acid-chloride"],
                "origin_records": [
                    {
                        "origin_kind": "literature_source_route",
                        "source_ref": SOURCE_REF,
                    }
                ],
                "reaction_proofs": [
                    {
                        "accepted": True,
                        "proof_digest": str(binding.get("reaction_digest") or ""),
                    }
                ],
                "procedure_record_ids": [procedure_id],
            }
        },
        "source_bindings": {
            source_binding_id: {
                "source_binding_id": source_binding_id,
                "source_kind": "patent",
                "source_ref": SOURCE_REF,
                "title": "New synthetic path to pharmaceutically acceptable Vismodegib",
                "artifact_sha256": str(materialization.get("artifact_sha256") or ""),
                "source_url": SOURCE_URL,
                "source_format": "epo_st36_xml.v1",
                "independence_group": f"publication:{PUBLICATION}",
            }
        },
        "exact_records": {
            record_id: {
                "record_id": record_id,
                "source_ref": SOURCE_REF,
                "source_binding_id": source_binding_id,
                "location_ref": location_ref,
                "location_refs": [location_ref],
                "evidence_refs": [
                    f"xml_sha256:{materialization.get('artifact_sha256') or ''}",
                    "procedure-text-sha256:"
                    + str(
                        dict(binding.get("parser_audit") or {}).get(
                            "procedure_text_sha256"
                        )
                        or ""
                    ),
                ],
            }
        },
        "procedure_records": {
            procedure_id: {
                "procedure_record_id": procedure_id,
                "exact_record_id": record_id,
                "source_ref": SOURCE_REF,
                "location_refs": [location_ref],
                "conditions": dict(binding.get("source_conditions") or {}),
                "procedure_authority_scope": "source_exact_reaction_procedure",
                "procedure_status": "condition_complete",
                "condition_completeness": dict(completeness),
                "source_fragment": {
                    "digest_kind": "procedure-text-sha256",
                    "procedure_text_sha256": str(
                        dict(binding.get("parser_audit") or {}).get(
                            "procedure_text_sha256"
                        )
                        or ""
                    ),
                    "source_artifact_sha256": str(
                        materialization.get("artifact_sha256") or ""
                    ),
                    "procedure_text_stored": False,
                },
            }
        },
        "stock_observations": {},
        "conflicts": {},
        "hypotheses": {},
        "delta": {"rejected": []},
    }
    portfolio = {
        "schema_version": "proof_stitched_route_portfolio.v1",
        "graph_revision": 1,
        "graph_scientific_sha256": graph_digest,
        "content_sha256": "vismodegib-ep3381900a1-portfolio-v1",
        "proof_policy": {
            "stock_boundary": "benchmark_search",
            "minimum_edge_proof_level": 3,
        },
        "selected_routes": [
            {
                "route_id": "route:ep3381900a1-procedure-slice",
                "route_family_id": f"publication:{PUBLICATION}",
                "strategy": "EP3381900A1 final-amide evidence slice",
                "edge_ids": ["edge:vismodegib-final-amide"],
                "root_edge_ids": ["edge:vismodegib-final-amide"],
                "leaf_molecule_ids": ["m:aniline", "m:acid-chloride"],
                "module_selections": {},
                "minimum_edge_proof_level": 3,
                "all_edges_proven": True,
                "stock_closure_rate": 0.0,
                "independent_source_groups": [f"publication:{PUBLICATION}"],
                "risk_score": 0.35,
                "convergence_score": 0.0,
                "complete": False,
                "pareto_optimal": True,
            }
        ],
        "edge_proofs": {
            "edge:vismodegib-final-amide": {
                "achieved_level": 3,
                "accepted": True,
                "reaction_validated": True,
                "exact_source_bound": True,
                "source_binding_ids": [source_binding_id],
                "exact_record_ids": [record_id],
                "procedure_record_ids": [procedure_id],
                "conflict_ids": [],
                "reasons": [],
            }
        },
        "leaf_proofs": {
            "m:aniline": {"accepted": False},
            "m:acid-chloride": {"accepted": False},
        },
        "route_modules": [],
        "deficits": [
            {
                "kind": "stock_boundary",
                "route_id": "route:ep3381900a1-procedure-slice",
                "reason": "starting_material_procurement_not_audited_in_this_case_gate",
            }
        ],
        "metrics": {},
        "closeout": {
            "decision": "continue",
            "complete_route_count": 0,
            "note": "single real procedure gate; not the three-case blind acceptance",
        },
        "accepted": False,
    }
    projection = compile_route_workbench(graph, portfolio)
    json_path = output_dir / "route_workbench.json"
    html_path = output_dir / "route_workbench.html"
    _write_json_atomic(json_path, projection)
    _write_text_atomic(html_path, render_v4_route_workbench_html(projection))
    return {
        "route_workbench_json": str(json_path),
        "route_workbench_html": str(html_path),
        "portfolio_accepted": False,
        "display_scope": "single_real_procedure_gate",
    }


def _resolver_snapshot(
    output_dir: Path,
    *,
    expected_product: str,
    expected_reactants: list[str],
    offline: bool,
) -> dict[str, Any]:
    path = output_dir / "resolver-snapshot.json"
    if path.is_file():
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        supplied = str(snapshot.get("content_sha256") or "")
        body = {key: value for key, value in snapshot.items() if key != "content_sha256"}
        if supplied == _digest(body):
            _validate_snapshot(
                snapshot,
                expected_product=expected_product,
                expected_reactants=expected_reactants,
            )
            return snapshot
    if offline:
        raise RuntimeError("offline_resolver_snapshot_missing_or_invalid")
    resolve_structure, resolve_names = build_deterministic_literature_resolvers(
        timeout_s=30.0
    )
    structures = {
        name: _canonical_smiles(resolve_structure(name))
        for name in SOURCE_STRUCTURE_NAMES
    }
    all_smiles = [expected_product, *expected_reactants]
    names = {smiles: list(resolve_names(smiles)) for smiles in all_smiles}
    snapshot: dict[str, Any] = {
        "schema_version": "deterministic_name_resolution_snapshot.v1",
        "authority_id": PARSER_AUTHORITY_ID,
        "structures": structures,
        "candidate_names": names,
        "semantics": {
            "snapshot_is_content_addressed": True,
            "snapshot_replay_uses_no_network": True,
            "snapshot_does_not_replace_source_text": True,
        },
    }
    snapshot["content_sha256"] = _digest(snapshot)
    _validate_snapshot(
        snapshot,
        expected_product=expected_product,
        expected_reactants=expected_reactants,
    )
    _write_json_atomic(path, snapshot)
    return snapshot


def _validate_snapshot(
    snapshot: Mapping[str, Any],
    *,
    expected_product: str,
    expected_reactants: list[str],
) -> None:
    structures = dict(snapshot.get("structures") or {})
    expected = dict(
        zip(
            SOURCE_STRUCTURE_NAMES,
            [
                expected_product,
                *[_canonical_smiles(value) for value in REACTANT_SMILES],
            ],
            strict=True,
        )
    )
    if sorted(
        expected[name] for name in SOURCE_STRUCTURE_NAMES[1:]
    ) != sorted(expected_reactants):
        raise RuntimeError("resolver_snapshot_expected_reactants_mismatch")
    actual = {
        name: _canonical_smiles(structures.get(name)) for name in SOURCE_STRUCTURE_NAMES
    }
    if actual != expected:
        raise RuntimeError("resolver_snapshot_structure_mismatch")


def _snapshot_resolvers(snapshot: Mapping[str, Any]):
    structures = {
        str(name).casefold(): str(smiles)
        for name, smiles in dict(snapshot.get("structures") or {}).items()
    }
    names = {
        _canonical_smiles(smiles): [str(value) for value in values or []]
        for smiles, values in dict(snapshot.get("candidate_names") or {}).items()
    }

    def resolve_structure(value: str) -> str:
        return structures.get(str(value).strip().casefold(), "")

    def resolve_names(value: str) -> list[str]:
        return list(names.get(_canonical_smiles(value), []))

    return resolve_structure, resolve_names


def _canonical_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or ""))
    if molecule is None:
        return ""
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
