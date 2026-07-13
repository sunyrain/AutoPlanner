"""Append-only admission journal for non-Codex route hyperedges.

The Codex campaign commit log and the controller's fused route graph have
different lifetimes.  This module persists the narrow, authority-free part of
an externally fused graph that the campaign must recover: one exact
product/precursor hyperedge.  Journal events cannot carry reaction proof,
stock, or completion authority.  Those decisions remain current-host replay
concerns in the proof state and frontier ledger.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from rdkit import Chem, RDLogger

from cascade_planner.application.frontier_ledger import exact_edge_signature
from cascade_planner.routes.admission import audit_retrosynthetic_candidate
from cascade_planner.routes.admission_receipts import (
    issue_external_hyperedge_provenance_receipt,
    validate_external_hyperedge_provenance_receipt,
)
from cascade_planner.routes.consensus import fuse_route_candidates
from cascade_planner.routes.graph import make_route_consensus_expansion


RDLogger.DisableLog("rdApp.*")

ADMITTED_HYPEREDGE_EVENT_SCHEMA = "admitted_external_hyperedge_event.v2"
ADMITTED_HYPEREDGE_STEP_PROJECTION_SCHEMA = (
    "admitted_external_hyperedge_step_projection.v1"
)
ADMITTED_HYPEREDGE_PROPOSAL_PROJECTION_SCHEMA = (
    "admitted_external_hyperedge_proposal_projection.v1"
)
ADMITTED_HYPEREDGE_ADMISSION_SCHEMA = "admitted_external_hyperedge_admission.v1"
ADMITTED_HYPEREDGE_JOURNAL_REPORT_SCHEMA = "admitted_external_hyperedge_journal.v1"

_EVENT_KEYS = {
    "schema_version",
    "event_id",
    "event_identity_sha256",
    "recorded_at",
    "case_id",
    "canonical_target_smiles",
    "campaign_identity_sha256",
    "campaign_policy_sha256",
    "product_depth",
    "exact_edge_signature",
    "exact_edge",
    "step_projection",
    "source_step_content_sha256",
    "proposal_projection",
    "source_proposal_content_sha256",
    "provenance_material",
    "provenance_material_sha256",
    "provenance_receipt",
    "provenance_receipt_sha256",
    "admission",
    "semantics",
    "content_sha256",
}
_EXACT_EDGE_KEYS = {
    "product_smiles",
    "precursor_smiles",
    "canonical_graph_step_id",
    "canonical_graph_signature",
}
_STEP_PROJECTION_KEYS = {
    "schema_version",
    "source_step_schema_version",
    "source_step_id",
    "source_signature",
    "product_smiles",
    "precursor_smiles",
    "proposal_ids",
    "reaction_family",
    "source_channels",
    "source_refs",
    "evidence_refs",
    "conditions",
    "catalysts",
    "enzymes",
    "limitations",
    "required_validation",
    "rank_score",
    "producer_confidence",
}
_PROPOSAL_PROJECTION_KEYS = {
    "schema_version",
    "producer_proposal_ids",
    "product_smiles",
    "precursor_smiles",
    "reaction_family",
    "transformation_rationale",
    "producer_source_channels",
    "source_refs",
    "evidence_refs",
    "conditions",
    "catalysts",
    "enzymes",
    "limitations",
    "required_validation",
}
_ADMISSION_KEYS = {"schema_version", "policy", "audit", "audit_sha256"}
_STEP_LIST_FIELDS = {
    "precursor_smiles",
    "proposal_ids",
    "source_channels",
    "source_refs",
    "evidence_refs",
    "conditions",
    "catalysts",
    "enzymes",
    "limitations",
    "required_validation",
}
_PROPOSAL_LIST_FIELDS = {
    "producer_proposal_ids",
    "precursor_smiles",
    "producer_source_channels",
    "source_refs",
    "evidence_refs",
    "conditions",
    "catalysts",
    "enzymes",
    "limitations",
    "required_validation",
}


class AdmittedHyperedgeJournalError(ValueError):
    """Raised when a journal event or campaign binding cannot be replayed."""


def canonical_graph_step_signature(
    product_smiles: Any,
    precursor_smiles: Iterable[Any],
) -> str:
    """Return the graph-v1 signature after exact structure canonicalization."""

    product = _canonical_smiles(product_smiles)
    precursors = sorted(
        value
        for value in (_canonical_smiles(item) for item in precursor_smiles)
        if value
    )
    return f"{product}<-{'.'.join(precursors)}" if product and precursors else ""


def canonical_graph_step_id(
    product_smiles: Any,
    precursor_smiles: Iterable[Any],
) -> str:
    """Return the stable graph-v1 id for one exact retrosynthetic edge."""

    signature = canonical_graph_step_signature(product_smiles, precursor_smiles)
    return (
        "step:" + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:24]
        if signature
        else ""
    )


def graph_exact_edge_signatures(graph: Mapping[str, Any] | None) -> set[str]:
    """Project a graph into collision-resistant exact edge identities."""

    if not isinstance(graph, Mapping):
        return set()
    return {
        signature
        for raw in graph.get("steps") or []
        if isinstance(raw, Mapping)
        and (
            signature := exact_edge_signature(
                raw.get("product_smiles"),
                raw.get("precursor_smiles") or [],
            )
        )
    }


def record_external_hyperedges(
    journal_root: str | os.PathLike[str],
    graph: Mapping[str, Any],
    *,
    case_id: str,
    target_smiles: str,
    campaign_identity_sha256: str,
    campaign_policy_sha256: str,
    known_exact_edge_signatures: Iterable[str] = (),
    admission_receipts: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Atomically classify and append current-host admitted external edges."""

    root = Path(journal_root).expanduser().resolve()
    with _exclusive_journal_lock(root / ".admission-journal.lock"):
        return _record_external_hyperedges_unlocked(
            root,
            graph,
            case_id=case_id,
            target_smiles=target_smiles,
            campaign_identity_sha256=campaign_identity_sha256,
            campaign_policy_sha256=campaign_policy_sha256,
            known_exact_edge_signatures=known_exact_edge_signatures,
            admission_receipts=admission_receipts,
        )


def _record_external_hyperedges_unlocked(
    journal_root: str | os.PathLike[str],
    graph: Mapping[str, Any],
    *,
    case_id: str,
    target_smiles: str,
    campaign_identity_sha256: str,
    campaign_policy_sha256: str,
    known_exact_edge_signatures: Iterable[str] = (),
    admission_receipts: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Append every newly observed exact edge before queue/proof mutation.

    ``known_exact_edge_signatures`` must come from replayed Codex commits and
    previously validated journal events, never a mutable controller graph.
    Every other edge requires complete source material in
    ``admission_receipts``.  The host replays that material and persists only
    a narrow, authority-free receipt; graph source labels are never trusted.
    """

    canonical_target = _canonical_smiles(target_smiles)
    _validate_campaign_binding(
        case_id=case_id,
        target_smiles=canonical_target,
        campaign_identity_sha256=campaign_identity_sha256,
        campaign_policy_sha256=campaign_policy_sha256,
    )
    if not isinstance(graph, Mapping) or graph.get("schema_version") != (
        "route_consensus_graph.v1"
    ):
        raise AdmittedHyperedgeJournalError(
            "external hyperedge admission requires route_consensus_graph.v1"
        )
    graph_case = str(graph.get("case_id") or "")
    graph_target = _canonical_smiles(graph.get("target_smiles"))
    if graph_case != str(case_id) or graph_target != canonical_target:
        raise AdmittedHyperedgeJournalError(
            "external hyperedge graph campaign identity mismatch"
        )

    existing = load_external_hyperedge_events(
        journal_root,
        case_id=case_id,
        target_smiles=canonical_target,
        campaign_identity_sha256=campaign_identity_sha256,
        campaign_policy_sha256=campaign_policy_sha256,
    )
    known = {str(item) for item in known_exact_edge_signatures if str(item)}
    known.update(str(row["exact_edge_signature"]) for row in existing)
    receipt_materials = _receipt_materials(admission_receipts)
    node_depths = {
        str(row.get("node_id") or ""): max(0, int(row.get("min_depth") or 0))
        for row in graph.get("nodes") or []
        if isinstance(row, Mapping) and str(row.get("node_id") or "")
    }
    recorded: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    rejected_materials: list[dict[str, Any]] = []
    for raw_step in graph.get("steps") or []:
        if not isinstance(raw_step, Mapping):
            raise AdmittedHyperedgeJournalError(
                "external hyperedge graph contains a non-object step"
            )
        step = dict(raw_step)
        edge_signature = exact_edge_signature(
            step.get("product_smiles"),
            step.get("precursor_smiles") or [],
        )
        if not edge_signature:
            raise AdmittedHyperedgeJournalError(
                "external hyperedge graph contains an invalid exact edge"
            )
        if edge_signature in known:
            continue
        materials = receipt_materials.get(edge_signature) or []
        if not materials:
            quarantined.append(
                _quarantined_edge(
                    step,
                    edge_signature=edge_signature,
                    reasons=["current_host_provenance_receipt_missing"],
                )
            )
            continue
        issued_receipts: list[tuple[dict[str, Any], dict[str, Any]]] = []
        receipt_reasons: list[str] = []
        for material_index, material in enumerate(materials):
            try:
                receipt, reasons = issue_external_hyperedge_provenance_receipt(
                    material,
                    expected_product_smiles=str(step.get("product_smiles") or ""),
                    expected_precursor_smiles=step.get("precursor_smiles") or [],
                )
            except Exception as exc:
                receipt = {}
                reasons = [
                    "current_host_provenance_material_replay_error:"
                    f"{type(exc).__name__}"
                ]
            if reasons:
                receipt_reasons.extend(
                    f"material:{material_index}:{reason}" for reason in reasons
                )
                rejected_materials.append(
                    _rejected_provenance_material(
                        step,
                        edge_signature=edge_signature,
                        material_index=material_index,
                        material=material,
                        reasons=reasons,
                    )
                )
            elif receipt:
                issued_receipts.append((receipt, dict(material)))
        if not issued_receipts:
            quarantined.append(
                _quarantined_edge(
                    step,
                    edge_signature=edge_signature,
                    reasons=receipt_reasons or ["no_valid_receipt"],
                )
            )
            continue
        receipt, provenance_material = min(
            issued_receipts,
            key=lambda row: str(row[0].get("receipt_id") or ""),
        )
        product_depth = node_depths.get(str(step.get("product_node_id") or ""), 0)
        event = _event_from_step(
            step,
            case_id=str(case_id),
            target_smiles=canonical_target,
            campaign_identity_sha256=campaign_identity_sha256,
            campaign_policy_sha256=campaign_policy_sha256,
            product_depth=product_depth,
            provenance_receipt=receipt,
            provenance_material=provenance_material,
        )
        stored = _publish_event(Path(journal_root), event)
        recorded.append(stored)
        known.add(edge_signature)

    events = load_external_hyperedge_events(
        journal_root,
        case_id=case_id,
        target_smiles=canonical_target,
        campaign_identity_sha256=campaign_identity_sha256,
        campaign_policy_sha256=campaign_policy_sha256,
    )
    return _journal_report(
        journal_root=Path(journal_root),
        events=events,
        newly_recorded=recorded,
        quarantined=quarantined,
        rejected_materials=rejected_materials,
    )


def load_external_hyperedge_events(
    journal_root: str | os.PathLike[str],
    *,
    case_id: str,
    target_smiles: str,
    campaign_identity_sha256: str,
    campaign_policy_sha256: str,
) -> list[dict[str, Any]]:
    """Load every immutable event still admissible on the current host.

    Integrity or campaign-binding failures abort replay.  A previously valid
    search-only event whose source adapter or structural admission policy has
    since changed is instead made inactive and audited.  Such an event carries
    no reaction-proof, stock, or completion authority, so dropping it from the
    active projection is the fail-closed behavior; aborting the entire durable
    campaign would incorrectly couple parser upgrades to search identity.
    """

    canonical_target = _canonical_smiles(target_smiles)
    _validate_campaign_binding(
        case_id=case_id,
        target_smiles=canonical_target,
        campaign_identity_sha256=campaign_identity_sha256,
        campaign_policy_sha256=campaign_policy_sha256,
    )
    event_root = _event_root(Path(journal_root))
    if not event_root.exists():
        return []
    if not event_root.is_dir():
        raise AdmittedHyperedgeJournalError(
            "external hyperedge journal event root is not a directory"
        )
    events: list[dict[str, Any]] = []
    inactive_events: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_exact_edges: set[str] = set()
    for path in sorted(event_root.glob("*/*.json")):
        event = _load_event(
            path,
            case_id=str(case_id),
            target_smiles=canonical_target,
            campaign_identity_sha256=campaign_identity_sha256,
            campaign_policy_sha256=campaign_policy_sha256,
        )
        event_id = str(event["event_id"])
        if event_id in seen_ids:
            raise AdmittedHyperedgeJournalError(
                "external hyperedge journal contains a duplicate event id"
            )
        seen_ids.add(event_id)
        authority_drift_reasons = list(
            event.pop("_current_host_authority_drift_reasons", []) or []
        )
        if authority_drift_reasons:
            inactive_events.append(
                {
                    "event_id": event_id,
                    "event_ref": str(event.get("event_ref") or path),
                    "exact_edge_signature": str(
                        event.get("exact_edge_signature") or ""
                    ),
                    "reasons": authority_drift_reasons,
                    "semantics": {
                        "immutable_event_preserved": True,
                        "excluded_from_current_search_projection": True,
                        "cannot_mutate_queue_proof_stock_or_completion": True,
                    },
                }
            )
            continue
        edge_signature = str(event.get("exact_edge_signature") or "")
        if edge_signature in seen_exact_edges:
            raise AdmittedHyperedgeJournalError(
                "external hyperedge journal contains a duplicate exact edge"
            )
        seen_exact_edges.add(edge_signature)
        events.append(event)
    _write_event_replay_report(
        Path(journal_root),
        active_events=events,
        inactive_events=inactive_events,
    )
    return sorted(events, key=lambda row: str(row["event_id"]))


def expansions_from_external_hyperedge_events(
    events: Iterable[Mapping[str, Any]],
    *,
    case_id: str,
) -> list[dict[str, Any]]:
    """Rebuild advisory graph expansions from validated event projections."""

    expansions: list[dict[str, Any]] = []
    for raw in events:
        event = dict(raw)
        proposal = dict(event.get("proposal_projection") or {})
        exact = dict(event.get("exact_edge") or {})
        candidate = {
            "schema_version": "retrosynthesis_candidate.v1",
            "candidate_id": "external-admission:" + str(
                event.get("event_identity_sha256") or ""
            )[:24],
            "product_smiles": str(exact.get("product_smiles") or ""),
            "precursor_smiles": list(exact.get("precursor_smiles") or []),
            "reaction_family": str(
                proposal.get("reaction_family") or "externally admitted hyperedge"
            ),
            "transformation_rationale": str(
                proposal.get("transformation_rationale")
                or "host-admitted external fused hyperedge"
            ),
            # Persisting an edge is not authority to replay the producer's
            # confidence/evidence label.  The original labels remain hashes
            # and advisory refs on the event; graph authority starts at L0.
            "source_channel": "other",
            "source_refs": list(proposal.get("source_refs") or []),
            "evidence_refs": list(proposal.get("evidence_refs") or []),
            "evidence_level": "model_only",
            "confidence": "low",
            "conditions": list(proposal.get("conditions") or []),
            "catalyst": str((proposal.get("catalysts") or [""])[0] or ""),
            "enzyme": str((proposal.get("enzymes") or [""])[0] or ""),
            "limitations": sorted(
                {
                    *[str(item) for item in proposal.get("limitations") or []],
                    "durable_external_admission_is_not_reaction_proof",
                    "producer_authority_not_replayed_by_admission_journal",
                }
            ),
            "required_validation": sorted(
                {
                    *[
                        str(item)
                        for item in proposal.get("required_validation") or []
                    ],
                    "materialize_and_verify_exact_reaction_edge",
                    "audit_precursors_with_current_stock_provider",
                }
            ),
            "report_ref": _event_ref(event),
            "no_solved_claim": True,
            "not_parent_route_proof": True,
        }
        consensus = fuse_route_candidates(
            [candidate],
            case_id=str(case_id),
            target_smiles=str(exact.get("product_smiles") or ""),
            allow_trusted_validated_evidence=False,
            allow_trusted_literature_exact_evidence=False,
        )
        if consensus.get("accepted") is not True:
            raise AdmittedHyperedgeJournalError(
                "validated external hyperedge event no longer passes route consensus"
            )
        expansions.append(
            make_route_consensus_expansion(
                consensus,
                requested_product_smiles=str(exact.get("product_smiles") or ""),
                consensus_ref=_event_ref(event),
                agent_run_ref="host-admitted-external-hyperedge-journal",
                depth=max(0, int(event.get("product_depth") or 0)),
            )
        )
    return expansions


def _event_from_step(
    step: Mapping[str, Any],
    *,
    case_id: str,
    target_smiles: str,
    campaign_identity_sha256: str,
    campaign_policy_sha256: str,
    product_depth: int,
    provenance_receipt: Mapping[str, Any],
    provenance_material: Mapping[str, Any],
) -> dict[str, Any]:
    product = _canonical_smiles(step.get("product_smiles"))
    precursors = sorted(
        value
        for value in (
            _canonical_smiles(item) for item in step.get("precursor_smiles") or []
        )
        if value
    )
    signature = canonical_graph_step_signature(product, precursors)
    step_id = canonical_graph_step_id(product, precursors)
    if (
        not product
        or not precursors
        or str(step.get("step_id") or "") != step_id
        or str(step.get("signature") or "") != signature
    ):
        raise AdmittedHyperedgeJournalError(
            "external hyperedge step id/signature is not canonical"
        )
    admission = audit_retrosynthetic_candidate(product, precursors)
    if admission.get("accepted") is not True:
        raise AdmittedHyperedgeJournalError(
            "external hyperedge failed host structural admission:"
            + ",".join(str(item) for item in admission.get("reasons") or [])
        )
    step_projection = {
        "schema_version": ADMITTED_HYPEREDGE_STEP_PROJECTION_SCHEMA,
        "source_step_schema_version": str(step.get("schema_version") or ""),
        "source_step_id": step_id,
        "source_signature": signature,
        "product_smiles": product,
        "precursor_smiles": precursors,
        "proposal_ids": _texts(step.get("proposal_ids") or []),
        "reaction_family": str(step.get("reaction_family") or "unspecified"),
        "source_channels": _texts(step.get("source_channels") or []),
        "source_refs": _texts(step.get("source_refs") or []),
        "evidence_refs": _texts(step.get("evidence_refs") or []),
        "conditions": _texts(step.get("conditions") or []),
        "catalysts": _texts(step.get("catalysts") or []),
        "enzymes": _texts(step.get("enzymes") or []),
        "limitations": _texts(step.get("limitations") or []),
        "required_validation": _texts(step.get("required_validation") or []),
        "rank_score": _bounded_score(step.get("rank_score")),
        "producer_confidence": str(step.get("confidence") or ""),
    }
    proposal_projection = {
        "schema_version": ADMITTED_HYPEREDGE_PROPOSAL_PROJECTION_SCHEMA,
        "producer_proposal_ids": list(step_projection["proposal_ids"]),
        "product_smiles": product,
        "precursor_smiles": precursors,
        "reaction_family": step_projection["reaction_family"],
        "transformation_rationale": "; ".join(
            _texts(step.get("rationales") or [])
        ),
        "producer_source_channels": list(step_projection["source_channels"]),
        "source_refs": list(step_projection["source_refs"]),
        "evidence_refs": list(step_projection["evidence_refs"]),
        "conditions": list(step_projection["conditions"]),
        "catalysts": list(step_projection["catalysts"]),
        "enzymes": list(step_projection["enzymes"]),
        "limitations": list(step_projection["limitations"]),
        "required_validation": list(step_projection["required_validation"]),
    }
    step_sha256 = _digest(step_projection)
    proposal_sha256 = _digest(proposal_projection)
    edge_signature = exact_edge_signature(product, precursors)
    receipt = _json_value(dict(provenance_receipt))
    receipt_reasons = validate_external_hyperedge_provenance_receipt(
        receipt,
        expected_exact_edge_signature=edge_signature,
    )
    if receipt_reasons:
        raise AdmittedHyperedgeJournalError(
            "external hyperedge provenance receipt is invalid:"
            + ",".join(receipt_reasons)
        )
    receipt_sha256 = str(receipt.get("content_sha256") or "")
    material = _json_value(dict(provenance_material))
    replayed_receipt, material_reasons = _replay_provenance_material(
        material,
        product_smiles=product,
        precursor_smiles=precursors,
    )
    if material_reasons or replayed_receipt != receipt:
        raise AdmittedHyperedgeJournalError(
            "external hyperedge provenance material replay is invalid:"
            + ",".join(material_reasons or ["receipt_replay_mismatch"])
        )
    material_sha256 = str(material.get("content_sha256") or "")
    identity_payload = {
        "schema_version": ADMITTED_HYPEREDGE_EVENT_SCHEMA,
        "case_id": case_id,
        "canonical_target_smiles": target_smiles,
        "campaign_identity_sha256": campaign_identity_sha256,
        "campaign_policy_sha256": campaign_policy_sha256,
        "exact_edge_signature": edge_signature,
        "source_step_content_sha256": step_sha256,
        "source_proposal_content_sha256": proposal_sha256,
        "provenance_material_sha256": material_sha256,
        "provenance_receipt_sha256": receipt_sha256,
    }
    event_identity = _digest(identity_payload)
    event = {
        "schema_version": ADMITTED_HYPEREDGE_EVENT_SCHEMA,
        "event_id": f"admitted-hyperedge:sha256:{event_identity}",
        "event_identity_sha256": event_identity,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "canonical_target_smiles": target_smiles,
        "campaign_identity_sha256": campaign_identity_sha256,
        "campaign_policy_sha256": campaign_policy_sha256,
        "product_depth": max(0, int(product_depth)),
        "exact_edge_signature": edge_signature,
        "exact_edge": {
            "product_smiles": product,
            "precursor_smiles": precursors,
            "canonical_graph_step_id": step_id,
            "canonical_graph_signature": signature,
        },
        "step_projection": step_projection,
        "source_step_content_sha256": step_sha256,
        "proposal_projection": proposal_projection,
        "source_proposal_content_sha256": proposal_sha256,
        "provenance_material": material,
        "provenance_material_sha256": material_sha256,
        "provenance_receipt": receipt,
        "provenance_receipt_sha256": receipt_sha256,
        "admission": {
            "schema_version": ADMITTED_HYPEREDGE_ADMISSION_SCHEMA,
            "policy": "shared_retrosynthetic_candidate_admission.v1",
            "audit": admission,
            "audit_sha256": _digest(admission),
        },
        "semantics": {
            "search_admission_only": True,
            "advisory_hyperedge_only": True,
            "reaction_proof_authority": "none",
            "stock_authority": "none",
            "completion_authority": "none",
            "provenance_material_authority": "comparison_input_only",
        },
    }
    event["content_sha256"] = _digest(event)
    return _json_value(event)


def _publish_event(root: Path, event: dict[str, Any]) -> dict[str, Any]:
    identity = str(event.get("event_identity_sha256") or "")
    path = _event_root(root) / identity[:2] / f"{identity}.json"
    if path.exists():
        return _load_event(
            path,
            case_id=str(event["case_id"]),
            target_smiles=str(event["canonical_target_smiles"]),
            campaign_identity_sha256=str(event["campaign_identity_sha256"]),
            campaign_policy_sha256=str(event["campaign_policy_sha256"]),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            event,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A hard-link publishes a fully written inode without replacing an
            # existing immutable object.  A crash before this call leaves only
            # an ignored temporary file; a crash after it leaves a valid event.
            os.link(temporary_name, path)
        except FileExistsError:
            pass
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return _load_event(
        path,
        case_id=str(event["case_id"]),
        target_smiles=str(event["canonical_target_smiles"]),
        campaign_identity_sha256=str(event["campaign_identity_sha256"]),
        campaign_policy_sha256=str(event["campaign_policy_sha256"]),
    )


def _load_event(
    path: Path,
    *,
    case_id: str,
    target_smiles: str,
    campaign_identity_sha256: str,
    campaign_policy_sha256: str,
) -> dict[str, Any]:
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AdmittedHyperedgeJournalError(
            f"external hyperedge journal event is unreadable:{path}"
        ) from exc
    if not isinstance(raw, dict):
        raise AdmittedHyperedgeJournalError(
            "external hyperedge journal event is not an object"
        )
    event = _json_value(raw)
    reasons = _event_replay_reasons(
        event,
        path=path,
        case_id=case_id,
        target_smiles=target_smiles,
        campaign_identity_sha256=campaign_identity_sha256,
        campaign_policy_sha256=campaign_policy_sha256,
    )
    authority_drift_reasons = sorted(
        reason for reason in reasons if _current_host_authority_drift_reason(reason)
    )
    integrity_reasons = sorted(
        reason for reason in reasons if reason not in authority_drift_reasons
    )
    if integrity_reasons:
        raise AdmittedHyperedgeJournalError(
            "invalid external hyperedge journal event:"
            + ",".join(sorted(set(integrity_reasons)))
        )
    event["event_ref"] = str(path)
    if authority_drift_reasons:
        event["_current_host_authority_drift_reasons"] = authority_drift_reasons
    return event


def _current_host_authority_drift_reason(reason: str) -> bool:
    return bool(
        str(reason).startswith("event_provenance_material_")
        or str(reason) == "event_host_admission_replay_failed"
    )


def _write_event_replay_report(
    journal_root: Path,
    *,
    active_events: Iterable[Mapping[str, Any]],
    inactive_events: Iterable[Mapping[str, Any]],
) -> None:
    active = [dict(row) for row in active_events]
    inactive = [dict(row) for row in inactive_events]
    payload = {
        "schema_version": "admitted_external_hyperedge_replay_report.v1",
        "active_event_count": len(active),
        "inactive_event_count": len(inactive),
        "active_event_ids": sorted(str(row.get("event_id") or "") for row in active),
        "inactive_events": sorted(
            inactive,
            key=lambda row: str(row.get("event_id") or ""),
        ),
        "semantics": {
            "integrity_or_campaign_binding_failure_aborts": True,
            "authority_drift_excludes_search_only_event": True,
            "immutable_event_objects_are_never_rewritten": True,
            "current_source_material_can_publish_a_replacement_event": True,
        },
    }
    payload["content_sha256"] = _digest(payload)
    root = journal_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / "replay_report.json"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=root,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _event_replay_reasons(
    event: dict[str, Any],
    *,
    path: Path,
    case_id: str,
    target_smiles: str,
    campaign_identity_sha256: str,
    campaign_policy_sha256: str,
) -> list[str]:
    reasons: list[str] = []
    if set(event) != _EVENT_KEYS:
        # Exact schemas are deliberate here.  In particular, an event cannot
        # add a producer-controlled ``proof``, ``stock``, ``solved`` or
        # similarly named authority field and make it valid merely by
        # recomputing its content digest.
        reasons.append("event_fields_invalid")
    payload = dict(event)
    recorded_content = str(payload.pop("content_sha256", ""))
    payload.pop("event_ref", None)
    if event.get("schema_version") != ADMITTED_HYPEREDGE_EVENT_SCHEMA:
        reasons.append("event_schema_invalid")
    if not recorded_content or recorded_content != _digest(payload):
        reasons.append("event_content_digest_invalid")
    if (
        event.get("case_id") != case_id
        or _canonical_smiles(event.get("canonical_target_smiles")) != target_smiles
        or event.get("campaign_identity_sha256") != campaign_identity_sha256
        or event.get("campaign_policy_sha256") != campaign_policy_sha256
    ):
        reasons.append("event_campaign_binding_mismatch")
    raw_step = event.get("step_projection")
    raw_proposal = event.get("proposal_projection")
    raw_exact = event.get("exact_edge")
    step = dict(raw_step) if isinstance(raw_step, Mapping) else {}
    proposal = dict(raw_proposal) if isinstance(raw_proposal, Mapping) else {}
    exact = dict(raw_exact) if isinstance(raw_exact, Mapping) else {}
    if not isinstance(raw_step, Mapping):
        reasons.append("event_step_projection_not_object")
    if not isinstance(raw_proposal, Mapping):
        reasons.append("event_proposal_projection_not_object")
    if not isinstance(raw_exact, Mapping):
        reasons.append("event_exact_edge_not_object")
    if set(step) != _STEP_PROJECTION_KEYS:
        reasons.append("event_step_projection_fields_invalid")
    if set(proposal) != _PROPOSAL_PROJECTION_KEYS:
        reasons.append("event_proposal_projection_fields_invalid")
    if set(exact) != _EXACT_EDGE_KEYS:
        reasons.append("event_exact_edge_fields_invalid")
    if any(not _is_string_list(step.get(field)) for field in _STEP_LIST_FIELDS):
        reasons.append("event_step_projection_list_fields_invalid")
    if any(
        not _is_string_list(proposal.get(field))
        for field in _PROPOSAL_LIST_FIELDS
    ):
        reasons.append("event_proposal_projection_list_fields_invalid")
    if step.get("schema_version") != ADMITTED_HYPEREDGE_STEP_PROJECTION_SCHEMA:
        reasons.append("event_step_projection_schema_invalid")
    if proposal.get("schema_version") != (
        ADMITTED_HYPEREDGE_PROPOSAL_PROJECTION_SCHEMA
    ):
        reasons.append("event_proposal_projection_schema_invalid")
    if event.get("source_step_content_sha256") != _digest(step):
        reasons.append("event_step_projection_digest_invalid")
    if event.get("source_proposal_content_sha256") != _digest(proposal):
        reasons.append("event_proposal_projection_digest_invalid")
    raw_receipt = event.get("provenance_receipt")
    receipt = dict(raw_receipt) if isinstance(raw_receipt, Mapping) else {}
    if not isinstance(raw_receipt, Mapping):
        reasons.append("event_provenance_receipt_not_object")
    raw_material = event.get("provenance_material")
    material = dict(raw_material) if isinstance(raw_material, Mapping) else {}
    if not isinstance(raw_material, Mapping):
        reasons.append("event_provenance_material_not_object")
    raw_exact_precursors = exact.get("precursor_smiles")
    if not _is_string_list(raw_exact_precursors):
        reasons.append("event_exact_edge_precursors_invalid")
        raw_exact_precursors = []
    product = _canonical_smiles(exact.get("product_smiles"))
    precursors = sorted(
        value
        for value in (
            _canonical_smiles(item) for item in raw_exact_precursors
        )
        if value
    )
    graph_signature = canonical_graph_step_signature(product, precursors)
    graph_step_id = canonical_graph_step_id(product, precursors)
    edge_signature = exact_edge_signature(product, precursors)
    if (
        not product
        or not precursors
        or exact.get("product_smiles") != product
        or raw_exact_precursors != precursors
        or exact.get("canonical_graph_signature") != graph_signature
        or exact.get("canonical_graph_step_id") != graph_step_id
        or event.get("exact_edge_signature") != edge_signature
        or step.get("product_smiles") != product
        or step.get("precursor_smiles") != precursors
        or step.get("source_step_id") != graph_step_id
        or step.get("source_signature") != graph_signature
        or proposal.get("product_smiles") != product
        or proposal.get("precursor_smiles") != precursors
    ):
        reasons.append("event_exact_edge_binding_invalid")
    receipt_reasons = validate_external_hyperedge_provenance_receipt(
        receipt,
        expected_exact_edge_signature=edge_signature,
    )
    reasons.extend(
        f"event_{reason}" for reason in receipt_reasons
    )
    if event.get("provenance_receipt_sha256") != str(
        receipt.get("content_sha256") or ""
    ):
        reasons.append("event_provenance_receipt_digest_binding_invalid")
    replayed_receipt, material_reasons = _replay_provenance_material(
        material,
        product_smiles=product,
        precursor_smiles=precursors,
    )
    reasons.extend(f"event_provenance_material_{reason}" for reason in material_reasons)
    if replayed_receipt != receipt:
        reasons.append("event_provenance_material_receipt_replay_mismatch")
    if event.get("provenance_material_sha256") != str(
        material.get("content_sha256") or ""
    ):
        reasons.append("event_provenance_material_digest_binding_invalid")
    raw_admission = event.get("admission")
    admission = dict(raw_admission) if isinstance(raw_admission, Mapping) else {}
    if not isinstance(raw_admission, Mapping):
        reasons.append("event_admission_not_object")
    if set(admission) != _ADMISSION_KEYS:
        reasons.append("event_admission_fields_invalid")
    recomputed_admission = audit_retrosynthetic_candidate(product, precursors)
    if (
        admission.get("schema_version") != ADMITTED_HYPEREDGE_ADMISSION_SCHEMA
        or admission.get("policy")
        != "shared_retrosynthetic_candidate_admission.v1"
        or admission.get("audit") != recomputed_admission
        or admission.get("audit_sha256") != _digest(recomputed_admission)
        or recomputed_admission.get("accepted") is not True
    ):
        reasons.append("event_host_admission_replay_failed")
    identity_payload = {
        "schema_version": ADMITTED_HYPEREDGE_EVENT_SCHEMA,
        "case_id": event.get("case_id"),
        "canonical_target_smiles": event.get("canonical_target_smiles"),
        "campaign_identity_sha256": event.get("campaign_identity_sha256"),
        "campaign_policy_sha256": event.get("campaign_policy_sha256"),
        "exact_edge_signature": event.get("exact_edge_signature"),
        "source_step_content_sha256": event.get("source_step_content_sha256"),
        "source_proposal_content_sha256": event.get(
            "source_proposal_content_sha256"
        ),
        "provenance_material_sha256": event.get(
            "provenance_material_sha256"
        ),
        "provenance_receipt_sha256": event.get(
            "provenance_receipt_sha256"
        ),
    }
    expected_identity = _digest(identity_payload)
    if (
        event.get("event_identity_sha256") != expected_identity
        or event.get("event_id")
        != f"admitted-hyperedge:sha256:{expected_identity}"
        or path.stem != expected_identity
        or path.parent.name != expected_identity[:2]
    ):
        reasons.append("event_identity_invalid")
    raw_semantics = event.get("semantics")
    semantics = dict(raw_semantics) if isinstance(raw_semantics, Mapping) else {}
    if not isinstance(raw_semantics, Mapping):
        reasons.append("event_semantics_not_object")
    if semantics != {
        "search_admission_only": True,
        "advisory_hyperedge_only": True,
        "reaction_proof_authority": "none",
        "stock_authority": "none",
        "completion_authority": "none",
        "provenance_material_authority": "comparison_input_only",
    }:
        reasons.append("event_authority_semantics_invalid")
    product_depth = event.get("product_depth")
    if (
        not isinstance(product_depth, int)
        or isinstance(product_depth, bool)
        or product_depth < 0
    ):
        reasons.append("event_product_depth_invalid")
    try:
        timestamp = datetime.fromisoformat(
            str(event.get("recorded_at") or "").replace("Z", "+00:00")
        )
        if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(
            timestamp
        ):
            reasons.append("event_timestamp_not_utc")
    except ValueError:
        reasons.append("event_timestamp_invalid")
    return reasons


def _journal_report(
    *,
    journal_root: Path,
    events: list[dict[str, Any]],
    newly_recorded: list[dict[str, Any]],
    quarantined: list[dict[str, Any]],
    rejected_materials: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema_version": ADMITTED_HYPEREDGE_JOURNAL_REPORT_SCHEMA,
        "journal_root": str(journal_root.resolve()),
        "event_count": len(events),
        "new_event_count": len(newly_recorded),
        "quarantined_edge_count": len(quarantined),
        "quarantined_edges": quarantined,
        "rejected_material_count": len(rejected_materials),
        "rejected_materials": rejected_materials,
        "exact_edge_signatures": sorted(
            {str(row.get("exact_edge_signature") or "") for row in events}
            - {""}
        ),
        "event_refs": sorted(_event_ref(row) for row in events),
        "semantics": {
            "append_only_content_addressed_events": True,
            "events_are_search_admission_not_proof": True,
            "campaign_restart_replay_required": True,
            "unreceipted_edges_are_quarantined_not_durable": True,
            "one_valid_independent_receipt_is_sufficient_for_search_admission": True,
            "invalid_sibling_materials_never_gain_authority": True,
        },
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def _quarantined_edge(
    step: Mapping[str, Any],
    *,
    edge_signature: str,
    reasons: Iterable[Any],
) -> dict[str, Any]:
    return {
        "schema_version": "external_hyperedge_quarantine.v1",
        "step_id": str(step.get("step_id") or ""),
        "exact_edge_signature": str(edge_signature or ""),
        "product_smiles": _canonical_smiles(step.get("product_smiles")),
        "precursor_smiles": sorted(
            value
            for value in (
                _canonical_smiles(item)
                for item in step.get("precursor_smiles") or []
            )
            if value
        ),
        "reasons": sorted({str(reason) for reason in reasons if str(reason)}),
        "semantics": {
            "caller_advisory_only": True,
            "not_persisted": True,
            "cannot_mutate_queue_or_ledger": True,
        },
    }


def _rejected_provenance_material(
    step: Mapping[str, Any],
    *,
    edge_signature: str,
    material_index: int,
    material: Mapping[str, Any],
    reasons: Iterable[Any],
) -> dict[str, Any]:
    return {
        "schema_version": "external_hyperedge_provenance_material_rejection.v1",
        "step_id": str(step.get("step_id") or ""),
        "exact_edge_signature": str(edge_signature or ""),
        "material_index": max(0, int(material_index)),
        "material_schema_version": str(material.get("schema_version") or ""),
        "material_sha256": str(material.get("content_sha256") or ""),
        "reasons": sorted({str(reason) for reason in reasons if str(reason)}),
        "semantics": {
            "audit_only": True,
            "not_persisted_as_event": True,
            "cannot_veto_valid_independent_receipt": True,
        },
    }


def _event_ref(event: Mapping[str, Any]) -> str:
    return str(event.get("event_ref") or event.get("event_id") or "")


def _event_root(root: Path) -> Path:
    return root.expanduser().resolve() / "events" / "sha256"


@contextmanager
def _exclusive_journal_lock(lock_path: Path) -> Iterator[None]:
    """Serialize load/classify/publish across processes for exact-edge uniqueness."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}:{time.time_ns()}"
    deadline = time.monotonic() + 30.0
    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                stale = time.time() - lock_path.stat().st_mtime > 120.0
            except FileNotFoundError:
                continue
            if stale:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise AdmittedHyperedgeJournalError(
                    "external hyperedge admission journal lock timeout"
                )
            time.sleep(0.01)
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(token)
            handle.flush()
            os.fsync(handle.fileno())
        break
    try:
        yield
    finally:
        try:
            if lock_path.read_text(encoding="utf-8") == token:
                lock_path.unlink()
        except FileNotFoundError:
            pass


def _receipt_materials(
    value: Mapping[str, Iterable[Mapping[str, Any]]] | None,
) -> dict[str, list[dict[str, Any]]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise AdmittedHyperedgeJournalError(
            "external hyperedge admission receipts must be a mapping"
        )
    rows: dict[str, list[dict[str, Any]]] = {}
    for raw_signature, raw_materials in value.items():
        signature = str(raw_signature or "")
        if not signature.startswith("edge:sha256:"):
            raise AdmittedHyperedgeJournalError(
                "external hyperedge admission receipt key is invalid"
            )
        if not isinstance(raw_materials, (list, tuple)):
            raise AdmittedHyperedgeJournalError(
                "external hyperedge admission receipt material list is invalid"
            )
        materials: list[dict[str, Any]] = []
        for raw_material in raw_materials:
            if not isinstance(raw_material, Mapping):
                raise AdmittedHyperedgeJournalError(
                    "external hyperedge admission receipt material is not an object"
                )
            materials.append(_json_value(dict(raw_material)))
        if materials:
            rows[signature] = materials
    return rows


def _replay_provenance_material(
    material: Mapping[str, Any],
    *,
    product_smiles: str,
    precursor_smiles: Iterable[Any],
) -> tuple[dict[str, Any], list[str]]:
    try:
        return issue_external_hyperedge_provenance_receipt(
            material,
            expected_product_smiles=product_smiles,
            expected_precursor_smiles=precursor_smiles,
        )
    except Exception as exc:
        return {}, [
            "current_host_provenance_material_replay_error:"
            f"{type(exc).__name__}"
        ]


def _validate_campaign_binding(
    *,
    case_id: str,
    target_smiles: str,
    campaign_identity_sha256: str,
    campaign_policy_sha256: str,
) -> None:
    if not str(case_id or "").strip() or not target_smiles:
        raise AdmittedHyperedgeJournalError(
            "external hyperedge journal campaign identity is incomplete"
        )
    if not _valid_sha256(campaign_identity_sha256) or not _valid_sha256(
        campaign_policy_sha256
    ):
        raise AdmittedHyperedgeJournalError(
            "external hyperedge journal campaign digests are invalid"
        )


def _canonical_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return ""
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _texts(values: Iterable[Any] | Any) -> list[str]:
    if isinstance(values, (str, bytes)):
        values = [values]
    elif not isinstance(values, Iterable):
        values = [values]
    return sorted({str(item).strip() for item in values if str(item or "").strip()})


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _bounded_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    if not math.isfinite(score):
        score = 0.0
    return round(max(0.0, min(1.0, score)), 6)


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


def _valid_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _json_value(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _reject_duplicate_keys(values: list[tuple[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in values:
        if key in row:
            raise ValueError(f"duplicate JSON key:{key}")
        row[key] = value
    return row


def _reject_nonfinite(value: str) -> Any:
    raise ValueError(f"non-finite JSON value:{value}")
