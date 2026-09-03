#!/usr/bin/env python3
"""Replay generic route-innovation discovery on the Bufotalin dossier.

All chemistry matching lives in the application discovery layer and the
versioned capability catalog.  This target-named script is only a benchmark
adapter: it converts the frozen dossier into a canonical graph projection and
loads a replayable external mechanism proposal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.application.route_innovation_discovery import (  # noqa: E402
    canonical_innovation_batch,
    discover_route_innovations,
)


DEFAULT_DOSSIER = (
    ROOT
    / "results"
    / "shared"
    / "bufotalin_v4_reported_candidate_20260715"
    / "bufotalin_20_step_dossier.json"
)
DEFAULT_OUTPUT = DEFAULT_DOSSIER.with_name("route_innovation_review.json")
DEFAULT_BATCH_OUTPUT = DEFAULT_DOSSIER.with_name(
    "route_innovation_ingestion_batch.json"
)
DEFAULT_CAPABILITY_CATALOG = ROOT / "config" / "route_innovation_capabilities.v1.json"
DEFAULT_MECHANISM_PROPOSALS = (
    ROOT / "benchmarks" / "bufotalin_route_innovation_proposals.v1.json"
)
ROUTE_ID = "route:bufotalin-20-step-reported-candidate"
ROUTE_FAMILY_ID = "family:bufotalin-reported-plus-upstream"


def build_review(
    dossier: Mapping[str, Any],
    *,
    capability_catalog: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    mechanism_proposals: Iterable[Mapping[str, Any]] | None = None,
    input_artifact: str = "",
) -> dict[str, Any]:
    """Adapt the dossier, then delegate all discovery and admission policy."""

    graph, route = _graph_and_route(dossier)
    catalog = (
        capability_catalog
        if capability_catalog is not None
        else _read_json(DEFAULT_CAPABILITY_CATALOG)
    )
    proposals = (
        list(mechanism_proposals)
        if mechanism_proposals is not None
        else list(_read_json(DEFAULT_MECHANISM_PROPOSALS).get("proposals") or [])
    )
    discovery = discover_route_innovations(
        graph,
        route,
        capabilities=catalog,
        mechanism_proposals=proposals,
        max_window_steps=8,
        max_candidates=24,
    )
    candidates = list(discovery["candidates"])
    savings = [
        int(dict(value.get("route_innovation") or {}).get("step_savings") or 0)
        for value in candidates
    ]
    review = {
        "schema_version": "bufotalin_route_innovation_review.v2",
        "route_id": ROUTE_ID,
        "route_family_id": ROUTE_FAMILY_ID,
        "input_artifact": input_artifact,
        "input_content_sha256": _digest(dossier),
        "capability_catalog_sha256": _digest(catalog),
        "discovery_content_sha256": discovery["content_sha256"],
        "baseline": {
            "physical_step_count": len(dossier.get("steps") or []),
            "paper_reported_step_count": int(
                dict(dossier.get("route") or {}).get("paper_reported_step_count") or 0
            ),
            "planner_hypothesis_step_count": int(
                dict(dossier.get("route") or {}).get("planner_hypothesis_step_count") or 0
            ),
            "candidate_innovation_count": len(candidates),
            "maximum_single_window_step_savings": max(savings, default=0),
        },
        "triage": {
            "ready_for_enzyme_screen": [
                value["candidate_id"]
                for value in candidates
                if value["review_status"] == "ready_for_enzyme_screen"
            ],
            "mechanism_review_only": [
                value["candidate_id"]
                for value in candidates
                if value["review_status"] == "mechanism_review_only"
            ],
            "requires_boundary_materialization": [
                value["candidate_id"]
                for value in candidates
                if value["review_status"] == "requires_boundary_materialization"
            ],
            "canonical_edges_created": 0,
        },
        "candidates": candidates,
        "rejected": list(discovery.get("rejected") or []),
        "program_draft_candidate_ids": list(
            discovery.get("program_draft_candidate_ids") or []
        ),
        "ingestion_hypotheses": list(discovery.get("ingestion_hypotheses") or []),
        "semantics": {
            **dict(discovery.get("semantics") or {}),
            "benchmark_adapter_contains_no_chemistry_match_rules": True,
            "low_confidence_candidates_remain_visible": True,
            "screen_results_are_required_before_route_closure": True,
        },
    }
    review["content_sha256"] = _digest(review)
    return review


def build_ingestion_batch(review: Mapping[str, Any]) -> dict[str, Any]:
    return canonical_innovation_batch(
        {
            "route_id": review.get("route_id"),
            "program_draft_candidate_ids": review.get("program_draft_candidate_ids"),
            "ingestion_hypotheses": review.get("ingestion_hypotheses"),
        }
    )


def _graph_and_route(
    dossier: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    steps = [dict(value) for value in dossier.get("steps") or []]
    if not steps:
        raise ValueError("route_innovation_review_steps_missing")
    if dict(dossier.get("source_evidence") or {}).get("procedure_sequences_agree") is not True:
        raise ValueError("route_innovation_review_source_sequence_unresolved")
    molecules: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    for step in steps:
        precursor_id = str(step.get("reactant_molecule_id") or "")
        product_id = str(step.get("product_molecule_id") or "")
        edge_id = str(step.get("edge_id") or "")
        if not precursor_id or not product_id or not edge_id:
            raise ValueError("route_innovation_review_step_identity_missing")
        molecules[precursor_id] = {
            "canonical_smiles": str(step.get("reactant_smiles") or ""),
            "label": str(step.get("reactant_label") or precursor_id),
        }
        molecules[product_id] = {
            "canonical_smiles": str(step.get("product_smiles") or ""),
            "label": str(step.get("product_label") or product_id),
        }
        edges[edge_id] = {
            "precursor_molecule_ids": [precursor_id],
            "product_molecule_id": product_id,
            "innovation_boundary_proof_level": int(step.get("proof_level") or 0),
        }
    return (
        {"molecules": molecules, "edges": edges},
        {
            "route_id": ROUTE_ID,
            "route_family_id": ROUTE_FAMILY_ID,
            "edge_ids": [str(value.get("edge_id") or "") for value in steps],
            "reported_source_refs": [
                str(dict(dossier.get("source_evidence") or {}).get("source_ref") or "")
            ],
        },
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"route_innovation_input_not_object:{path}")
    return dict(value)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dossier", type=Path, default=DEFAULT_DOSSIER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-output", type=Path, default=DEFAULT_BATCH_OUTPUT)
    parser.add_argument(
        "--capability-catalog", type=Path, default=DEFAULT_CAPABILITY_CATALOG
    )
    parser.add_argument(
        "--mechanism-proposals", type=Path, default=DEFAULT_MECHANISM_PROPOSALS
    )
    args = parser.parse_args()

    dossier = _read_json(args.dossier)
    catalog = _read_json(args.capability_catalog)
    proposal_pack = _read_json(args.mechanism_proposals)
    review = build_review(
        dossier,
        capability_catalog=catalog,
        mechanism_proposals=proposal_pack.get("proposals") or [],
        input_artifact=str(args.dossier),
    )
    batch = build_ingestion_batch(review)
    for path, value in ((args.output, review), (args.batch_output, batch)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "batch_output": str(args.batch_output),
                "candidates": len(review["candidates"]),
                "ready_for_enzyme_screen": len(
                    review["triage"]["ready_for_enzyme_screen"]
                ),
                "maximum_single_window_step_savings": review["baseline"][
                    "maximum_single_window_step_savings"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
