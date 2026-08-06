"""Compile current-parser source bindings into a stock-closed route portfolio."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.legacy.application_runtime.retrosynthesis_acceptance import (  # noqa: E402
    evaluate_retrosynthesis_acceptance,
)
from cascade_planner.application.retrosynthesis_run_contract import (  # noqa: E402
    RetrosynthesisAcceptanceSpec,
)
from cascade_planner.legacy.application_runtime.route_portfolio import solve_diverse_routes  # noqa: E402
from cascade_planner.harness.deterministic_literature_registry import (  # noqa: E402
    PARSER_AUTHORITY_ID,
)
from cascade_planner.providers.stock import (  # noqa: E402
    SnapshotStockProvider,
    canonicalize_stock_snapshot,
    replay_stock_provider_result,
    stock_snapshot_sha256,
)
from cascade_planner.providers.contracts import ProviderContext  # noqa: E402
from cascade_planner.routes.domain import MoleculeIdentity  # noqa: E402
from cascade_planner.routes.overlay import build_route_hypergraph_v2_overlay  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--stock-snapshots", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = compile_source_route_portfolio(
        candidate_manifest_path=args.candidate_manifest,
        registry_root=args.registry_root,
        stock_snapshots_path=args.stock_snapshots,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def compile_source_route_portfolio(
    *,
    candidate_manifest_path: Path,
    registry_root: Path,
    stock_snapshots_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build and persist one hard-accepted portfolio without model calls."""

    candidate_manifest = _read_object(candidate_manifest_path)
    bindings, audit_refs = _load_current_parser_bindings(registry_root)
    graph, expected_step_ids = _build_consensus_graph(
        candidate_manifest,
        bindings=bindings,
    )
    missing = sorted(expected_step_ids - set(bindings))
    if missing:
        raise SystemExit(
            "Current deterministic parser did not approve required steps: "
            + ", ".join(missing)
        )
    overlay = build_route_hypergraph_v2_overlay(graph)
    if (overlay.get("validation") or {}).get("valid") is not True:
        raise SystemExit(
            "Route overlay validation failed: "
            + ", ".join((overlay.get("validation") or {}).get("errors") or [])
        )

    stock_ids, stock_bindings, stock_audit = _load_stock_bindings(
        stock_snapshots_path,
        overlay=overlay,
    )
    edge_levels = {
        str(edge.get("hyperedge_id") or ""): {
            "achieved_proof_level": 3,
            "proof_level": "L3_precedent_supported",
            "authority": PARSER_AUTHORITY_ID,
        }
        for edge in overlay.get("reaction_hyperedges") or []
        if isinstance(edge, dict) and str(edge.get("hyperedge_id") or "")
    }
    portfolio = solve_diverse_routes(
        overlay,
        stock_molecule_ids=stock_ids,
        edge_proof_levels=edge_levels,
        stock_bindings=stock_bindings,
        top_k=5,
        min_reaction_proof_level=3,
    ).to_dict()
    acceptance_spec = RetrosynthesisAcceptanceSpec(
        minimum_complete_routes=2,
        minimum_edge_proof_level=3,
        require_all_selected_leaves_stock_closed=True,
        stock_boundary="procurement",
        minimum_independent_source_groups=2,
        require_distinct_edge_sets=True,
    )
    acceptance = evaluate_retrosynthesis_acceptance(
        route_portfolio=portfolio,
        acceptance_spec=acceptance_spec,
    )

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "route_consensus_graph.json": graph,
        "route_hypergraph_overlay.json": overlay,
        "route_portfolio_bindings.json": {
            "schema_version": "source_route_portfolio_bindings.v1",
            "edge_proof_levels": edge_levels,
            "stock_molecule_ids": stock_ids,
            "stock_bindings": stock_bindings,
            "parser_audit_refs": audit_refs,
        },
        "route_portfolio.json": portfolio,
        "stock_replay_audit.json": stock_audit,
        "retrosynthesis_acceptance_report.json": acceptance,
    }
    for name, payload in artifacts.items():
        _write_json(output_dir / name, payload)
    summary = {
        "schema_version": "source_route_portfolio_compile_result.v1",
        "accepted": acceptance.get("accepted") is True,
        "approved_source_step_count": len(bindings),
        "hyperedge_count": len(overlay.get("reaction_hyperedges") or []),
        "complete_route_count": int(portfolio.get("complete_candidate_count") or 0),
        "selected_route_count": int(acceptance.get("selected_route_count") or 0),
        "independent_support_groups": list(
            acceptance.get("independent_support_groups") or []
        ),
        "stock_terminal_count": len(stock_ids),
        "model_invocations": 0,
        "output_dir": str(output_dir),
        "reasons": list(acceptance.get("reasons") or []),
    }
    _write_json(output_dir / "compile_summary.json", summary)
    return summary


def _load_current_parser_bindings(
    registry_root: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    bindings: dict[str, dict[str, Any]] = {}
    audit_refs: list[dict[str, str]] = []
    for audit_path in sorted(
        registry_root.resolve().glob("*/deterministic_literature_registry_audit.json")
    ):
        audit = _read_object(audit_path)
        authority = dict(audit.get("authority") or {})
        if (
            audit.get("schema_version")
            != "deterministic_literature_registry_audit.v1"
            or authority.get("id") != PARSER_AUTHORITY_ID
        ):
            continue
        supplied_audit_digest = str(audit.get("content_sha256") or "")
        audit_payload = dict(audit)
        audit_payload.pop("content_sha256", None)
        if supplied_audit_digest != _digest(audit_payload):
            raise SystemExit(f"Parser audit digest replay failed for {audit_path}")
        registry_path = Path(str(audit.get("registry_path") or ""))
        if not registry_path.is_file() or _sha256_file(registry_path) != str(
            audit.get("registry_sha256") or ""
        ):
            raise SystemExit(f"Registry digest replay failed for {audit_path}")
        registry = _read_object(registry_path)
        supplied_registry_digest = str(registry.get("content_sha256") or "")
        registry_payload = dict(registry)
        registry_payload.pop("content_sha256", None)
        if (
            registry.get("schema_version")
            != "trusted_literature_step_registry.v1"
            or supplied_registry_digest != _digest(registry_payload)
        ):
            raise SystemExit(
                f"Registry content replay failed for {registry_path}"
            )
        registry_bindings = {
            str(row.get("binding_id") or ""): dict(row)
            for row in registry.get("bindings") or []
            if isinstance(row, dict) and str(row.get("binding_id") or "")
        }
        audit_refs.append(
            {
                "audit_path": str(audit_path),
                "audit_sha256": _sha256_file(audit_path),
                "registry_path": str(registry_path),
                "registry_sha256": _sha256_file(registry_path),
            }
        )
        for raw in audit.get("records") or []:
            if not isinstance(raw, dict) or raw.get("accepted") is not True:
                continue
            binding = dict(raw.get("binding") or {})
            binding_authority = dict(binding.get("authority") or {})
            if (
                binding.get("status") != "approved"
                or binding_authority.get("id") != PARSER_AUTHORITY_ID
            ):
                continue
            binding_id = str(binding.get("binding_id") or "")
            if not binding_id or registry_bindings.get(binding_id) != binding:
                raise SystemExit(
                    f"Audit binding is not exactly present in registry: {audit_path}"
                )
            step_id = str(raw.get("step_id") or "")
            if step_id:
                bindings[step_id] = binding
    if not bindings:
        raise SystemExit("No current-parser approved bindings found")
    return bindings, audit_refs


def _build_consensus_graph(
    manifest: dict[str, Any],
    *,
    bindings: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], set[str]]:
    target = MoleculeIdentity(str(manifest.get("target_smiles") or ""))
    if target.validate():
        raise SystemExit("candidate manifest target_smiles is invalid")
    route_rows = [
        dict(row)
        for row in manifest.get("routes") or []
        if isinstance(row, dict)
    ]
    expected = {
        str(step_id)
        for route in route_rows
        for step_id in route.get("step_ids") or []
        if str(step_id or "")
    }
    nodes: dict[str, dict[str, Any]] = {}
    steps: list[dict[str, Any]] = []
    step_nodes: dict[str, set[str]] = {}
    for step_id in sorted(expected):
        binding = bindings.get(step_id)
        if not binding:
            continue
        projection = dict(binding.get("synthesis_projection") or {})
        product = MoleculeIdentity(str(projection.get("product_smiles") or ""))
        precursors = tuple(
            MoleculeIdentity(str(item))
            for item in projection.get("reactant_smiles") or []
        )
        if product.validate() or not precursors or any(row.validate() for row in precursors):
            raise SystemExit(f"Invalid current-parser chemistry for {step_id}")
        for molecule in (product, *precursors):
            nodes[molecule.molecule_id] = {
                "node_id": molecule.molecule_id,
                "canonical_isomeric_smiles": molecule.canonical_isomeric_smiles,
                "smiles": molecule.canonical_isomeric_smiles,
            }
        source_ref = str(binding.get("source_ref") or "")
        steps.append(
            {
                "step_id": step_id,
                "product_node_id": product.molecule_id,
                "product_smiles": product.canonical_isomeric_smiles,
                "precursor_node_ids": [row.molecule_id for row in precursors],
                "precursor_smiles": [
                    row.canonical_isomeric_smiles for row in precursors
                ],
                "reaction_family": "source_exact_precedent",
                "source_channels": ["literature_exact"],
                "independent_support_groups": [source_ref],
                "proposal_ids": [str(binding.get("binding_id") or "")],
                "rank_score": 1.0,
                "conditions": [],
                "required_validation": [],
            }
        )
        step_nodes[step_id] = {
            product.molecule_id,
            *(row.molecule_id for row in precursors),
        }
    routes = [
        {
            "route_id": str(route.get("route_id") or ""),
            "retrosynthetic_step_ids": [
                str(item) for item in route.get("step_ids") or []
            ],
            "node_ids": sorted(
                {
                    node_id
                    for step_id in route.get("step_ids") or []
                    for node_id in step_nodes.get(str(step_id), set())
                }
            ),
            "frontier": [],
            "rank_score": 1.0,
        }
        for route in route_rows
    ]
    return {
        "schema_version": "route_consensus_graph.v1",
        "case_id": "nirmatrelvir-source-route-replay",
        "root_node_id": target.molecule_id,
        "nodes": sorted(nodes.values(), key=lambda row: row["node_id"]),
        "steps": steps,
        "route_hypotheses": routes,
        "semantics": {
            "current_parser_bindings_only": True,
            "model_output_is_not_proof": True,
        },
    }, expected


def _load_stock_bindings(
    path: Path,
    *,
    overlay: dict[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    payload = _read_object(path)
    snapshots = [
        dict(row)
        for row in payload.get("snapshots") or []
        if isinstance(row, dict)
    ]
    canonical_snapshots: list[tuple[dict[str, Any], str]] = []
    for snapshot in snapshots:
        canonical = canonicalize_stock_snapshot(snapshot)
        digest = stock_snapshot_sha256(canonical)
        if digest != str(snapshot.get("snapshot_sha256") or ""):
            raise SystemExit("stock snapshot digest mismatch")
        canonical_snapshots.append((canonical, digest))
    provider = SnapshotStockProvider(
        trusted_snapshots=[row for row, _ in canonical_snapshots]
    )
    trusted_providers = {provider.descriptor.provider_id: provider}
    context = ProviderContext(
        run_id="nirmatrelvir-v3-deterministic-replay",
        case_id="nirmatrelvir-source-route-replay",
        target_smiles=str(
            next(
                (
                    row.get("canonical_isomeric_smiles")
                    for row in overlay.get("molecules") or []
                    if isinstance(row, dict)
                    and row.get("molecule_id") == overlay.get("root_molecule_id")
                ),
                "",
            )
        ),
    )
    molecule_by_smiles = {
        str(row.get("canonical_isomeric_smiles") or ""): str(
            row.get("molecule_id") or ""
        )
        for row in overlay.get("molecules") or []
        if isinstance(row, dict)
    }
    bindings: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for canonical, digest in canonical_snapshots:
        smiles = str(canonical.get("canonical_smiles") or "")
        molecule_id = molecule_by_smiles.get(smiles)
        if not molecule_id:
            continue
        provider_result = provider.invoke(
            {
                "schema_version": "stock_lookup_request.v1",
                "smiles": smiles,
                "offers": [{**canonical, "snapshot_sha256": digest}],
            },
            context=context,
        ).to_dict()
        replay_binding, replay_reasons = replay_stock_provider_result(
            provider_result,
            expected_smiles=smiles,
            trusted_provider_instances=trusted_providers,
            context=context,
        )
        provider_payload = dict(
            (replay_binding.get("provider_result") or {}).get("payload") or {}
        )
        if replay_reasons or not replay_binding or provider_payload.get("accepted") is not True:
            raise SystemExit(
                "stock provider replay failed for "
                f"{canonical.get('catalog_number')}: "
                + ",".join(replay_reasons or ("boundary_not_accepted",))
            )
        binding = {
            "schema_version": "source_route_stock_binding.v1",
            "molecule_id": molecule_id,
            "canonical_smiles": smiles,
            "boundary_type": str(provider_payload.get("boundary_type") or ""),
            "commercial_orderability_claimed": bool(
                provider_payload.get("accepted") is True
                and provider_payload.get("boundary_type") == "commercially_orderable"
            ),
            "snapshot_digest_replayed": True,
            "snapshot_sha256": digest,
            "supplier": str(canonical.get("supplier") or ""),
            "catalog_number": str(canonical.get("catalog_number") or ""),
            "checked_at": str(canonical.get("checked_at") or ""),
            "source_url": str(canonical.get("source_url") or ""),
            "host_provider_replay": replay_binding,
        }
        binding["binding_sha256"] = _digest(binding)
        bindings[molecule_id] = binding
        records.append(binding)
    audit = {
        "schema_version": "source_route_stock_replay_audit.v1",
        "accepted": bool(records),
        "snapshot_file": str(path.resolve()),
        "snapshot_file_sha256": _sha256_file(path.resolve()),
        "matched_terminal_count": len(records),
        "bindings": records,
        "semantics": {
            "snapshot_digest_replayed": True,
            "host_owned_provider_replayed": True,
            "live_availability_not_implied_beyond_checked_at": True,
        },
    }
    audit["content_sha256"] = _digest(audit)
    return sorted(bindings), bindings, audit


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Unable to read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON must be an object: {path}")
    return payload


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    main()
