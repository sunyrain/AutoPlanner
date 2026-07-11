"""Adapters that bring blackboard route channels into the canonical consensus domain."""
from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Mapping
from typing import Any

from cascade_planner.application.frontier_ledger import exact_edge_signature
from cascade_planner.harness.stitched_route import (
    is_materialized_source_bound_literature_step,
    is_validated_source_detail_literature_step,
)
from cascade_planner.harness.route_verifier import (
    replay_route_proof_bank_entry,
    validate_route_proof_bank,
)
from cascade_planner.routes.consensus import fuse_route_candidates
from cascade_planner.routes.admission_receipts import (
    make_chemenzy_admission_material,
    make_exact_literature_admission_material,
    make_materialized_literature_search_admission_material,
)
from cascade_planner.routes.domain import canonicalize_smiles, stable_content_hash, stable_domain_id
from cascade_planner.routes.graph import assemble_route_consensus_graph, make_route_consensus_expansion


def rebuild_consensus_graph_from_blackboard(
    blackboard: Mapping[str, Any],
    *,
    max_depth: int = 2,
) -> dict[str, Any]:
    """Fuse every provider within its canonical product neighborhood.

    ``route_consensus.v1`` is intentionally a one-product contract.  The old
    adapter sent every blackboard record through one root-target invocation,
    which quarantined valid intermediate ChemEnzy, literature, and legacy
    proposals as target mismatches.  This rebuild keeps the v1 boundary but
    invokes it once per canonical product before assembling the route graph.
    """
    board = dict(blackboard)
    target_profile = dict(board.get("target_profile") or {})
    target_smiles = str(
        target_profile.get("target_smiles")
        or target_profile.get("isomeric_smiles")
        or target_profile.get("canonical_smiles")
        or (board.get("target") or {}).get("smiles")
        or ""
    )
    canonical_target = canonicalize_smiles(target_smiles)
    case_id = str(board.get("case_id") or "")
    team = dict(board.get("codex_agent_team") or {})
    prior_expansions = [
        dict(row)
        for row in team.get("route_consensus_expansions") or []
        if isinstance(row, Mapping)
    ]
    chemenzy_candidates, chemenzy_proof_bank_audits, chemenzy_receipts = (
        _candidates_from_chemenzy_proof_banks(
            board.get("chemenzy_route_proof_banks") or [],
        )
    )
    exact_candidates, exact_receipts = _candidates_from_exact_rows(
        (board.get("literature_evidence") or {}).get("exact_rows")
        or (board.get("literature_evidence") or {}).get("exact_literature_rows")
        or [],
        target_smiles=canonical_target or target_smiles,
    )

    candidates: list[dict[str, Any]] = [
        *_candidates_from_consensus(dict(board.get("route_consensus") or {})),
        *[
            candidate
            for expansion in prior_expansions
            for candidate in _candidates_from_consensus(dict(expansion.get("route_consensus") or {}))
        ],
        *_candidates_from_legacy_proposals(
            board.get("retrosynthetic_proposals") or [],
            target_smiles=canonical_target or target_smiles,
        ),
        *exact_candidates,
        *chemenzy_candidates,
    ]
    candidates = _dedupe_candidates(candidates)
    buckets, unbucketed = _bucket_candidates_by_product(candidates)
    consensuses = {
        product: fuse_route_candidates(
            rows,
            case_id=case_id,
            target_smiles=product,
            # Trust is candidate-scoped.  The deterministic exact-row adapter
            # below attaches the private host binding; replayed consensus,
            # Codex, and legacy rows cannot inherit a global exact-literature
            # permission merely because they share this product bucket.
            allow_trusted_literature_exact_evidence=False,
        )
        for product, rows in sorted(buckets.items())
    }
    if canonical_target not in consensuses:
        consensuses[canonical_target] = fuse_route_candidates(
            [],
            case_id=case_id,
            target_smiles=canonical_target or target_smiles,
            allow_trusted_literature_exact_evidence=False,
        )
    consensus = consensuses[canonical_target]

    expansion_metadata = _expansion_metadata_by_product(prior_expansions)
    product_depths = _product_depths(consensuses, root_product=canonical_target)
    coordinator_ref = str((team.get("coordinator") or {}).get("run_record_ref") or "")
    ordered_products = sorted(
        consensuses,
        key=lambda product: (
            product != canonical_target,
            int(product_depths.get(product, max_depth + 1)),
            product,
        ),
    )
    expansions: list[dict[str, Any]] = []
    for product in ordered_products:
        metadata = expansion_metadata.get(product) or {}
        expansions.append(
            make_route_consensus_expansion(
                consensuses[product],
                requested_product_smiles=product,
                consensus_ref=str(
                    metadata.get("consensus_ref")
                    or f"blackboard:route_consensus_neighborhood:{stable_domain_id('product', product)}"
                ),
                agent_run_ref=str(metadata.get("agent_run_ref") or coordinator_ref),
                depth=int(product_depths.get(product, metadata.get("depth", max_depth + 1))),
            )
        )
    graph = assemble_route_consensus_graph(
        expansions,
        case_id=case_id,
        target_smiles=canonical_target or target_smiles,
        max_depth=max(1, int(max_depth or 1)),
    )
    product_neighborhoods = [
        _product_neighborhood_summary(product, consensuses[product])
        for product in ordered_products
    ]
    aggregate_source_summary = _aggregate_source_summary(consensuses)
    return {
        "schema_version": "blackboard_route_consensus_rebuild.v1",
        "accepted": bool(consensus.get("accepted")),
        "candidate_count": len(candidates),
        "unbucketed_candidate_count": len(unbucketed),
        "unbucketed_candidates": unbucketed,
        "consensus": consensus,
        "consensus_by_product": consensuses,
        "product_neighborhoods": product_neighborhoods,
        "source_summary": aggregate_source_summary,
        "chemenzy_proof_bank_audits": chemenzy_proof_bank_audits,
        "admission_receipts": _merge_admission_receipts(
            exact_receipts,
            chemenzy_receipts,
        ),
        "expansions": expansions,
        "graph": graph,
        "semantics": {
            "advisory_only": True,
            "no_solved_claim": True,
            "deterministic_parent_proof_required": True,
            "fusion_scope": "canonical_product_neighborhood",
            "codex_roles_share_one_support_group": True,
            "v1_consensus_contract_preserved": True,
            "chemenzy_candidates_require_current_host_proof_bank_replay": True,
            "external_admission_requires_current_host_provenance_receipt": True,
        },
    }


def _dedupe_candidates(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        if key in seen:
            continue
        seen.add(key)
        rows.append(dict(value))
    return rows


def _bucket_candidates_by_product(
    values: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unbucketed: list[dict[str, Any]] = []
    for index, row in enumerate(values):
        product = canonicalize_smiles(row.get("product_smiles"))
        if not product:
            unbucketed.append(
                {
                    "index": index,
                    "candidate_id": str(row.get("candidate_id") or ""),
                    "reasons": ["invalid_product_smiles"],
                }
            )
            continue
        candidate = dict(row)
        candidate["product_smiles"] = product
        buckets[product].append(candidate)
    return dict(buckets), unbucketed


def _expansion_metadata_by_product(
    expansions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for expansion in expansions:
        consensus = dict(expansion.get("route_consensus") or {})
        product = canonicalize_smiles(
            expansion.get("requested_product_smiles") or consensus.get("target_smiles")
        )
        if not product:
            continue
        candidate = {
            "depth": max(0, int(expansion.get("depth") or 0)),
            "consensus_ref": str(expansion.get("consensus_ref") or ""),
            "agent_run_ref": str(expansion.get("agent_run_ref") or ""),
        }
        existing = rows.get(product)
        if existing is None or int(candidate["depth"]) < int(existing.get("depth") or 0):
            rows[product] = candidate
    return rows


def _product_depths(
    consensuses: Mapping[str, Mapping[str, Any]],
    *,
    root_product: str,
) -> dict[str, int]:
    if not root_product:
        return {}
    depths = {root_product: 0}
    queue: deque[str] = deque([root_product])
    while queue:
        product = queue.popleft()
        next_depth = depths[product] + 1
        for proposal in (consensuses.get(product) or {}).get("proposals") or []:
            if not isinstance(proposal, Mapping):
                continue
            for precursor in proposal.get("precursor_smiles") or []:
                canonical = canonicalize_smiles(precursor)
                if not canonical or canonical not in consensuses:
                    continue
                if next_depth < depths.get(canonical, 1_000_000):
                    depths[canonical] = next_depth
                    queue.append(canonical)
    return depths


def _product_neighborhood_summary(product: str, consensus: Mapping[str, Any]) -> dict[str, Any]:
    proposals = [dict(row) for row in consensus.get("proposals") or [] if isinstance(row, Mapping)]
    payload: dict[str, Any] = {
        "schema_version": "route_consensus_product_neighborhood.v2",
        "neighborhood_id": stable_domain_id("consensus-neighborhood", product),
        "product_smiles": product,
        "proposal_ids": sorted(str(row.get("consensus_id") or "") for row in proposals),
        "proposal_count": len(proposals),
        "candidate_count": int((consensus.get("source_summary") or {}).get("candidate_count") or 0),
        "rejected_count": int((consensus.get("source_summary") or {}).get("rejected_count") or 0),
        "source_channels": sorted(
            {
                str(channel)
                for row in proposals
                for channel in row.get("source_channels") or []
                if str(channel or "").strip()
            }
        ),
        "independent_support_groups": sorted(
            {
                str(group)
                for row in proposals
                for group in row.get("independent_support_groups") or []
                if str(group or "").strip()
            }
        ),
        "support_group_scope": "union_across_competing_proposals",
        "multi_source_proposal_ids": sorted(
            str(row.get("consensus_id") or "")
            for row in proposals
            if int(row.get("source_diversity") or 0) > 1
        ),
        "max_independent_support_group_count": max(
            (int(row.get("source_diversity") or 0) for row in proposals),
            default=0,
        ),
        "advisory_only": True,
        "no_solved_claim": True,
        "not_parent_route_proof": True,
    }
    payload["content_hash"] = stable_content_hash(str(payload["schema_version"]), payload)
    return payload


def _aggregate_source_summary(consensuses: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    channel_counts: dict[str, int] = defaultdict(int)
    candidate_count = 0
    rejected_count = 0
    proposal_count = 0
    multi_source_proposals = 0
    authority_capped_candidate_count = 0
    normalization_record_count = 0
    for consensus in consensuses.values():
        summary = dict(consensus.get("source_summary") or {})
        candidate_count += int(summary.get("candidate_count") or 0)
        rejected_count += int(summary.get("rejected_count") or 0)
        proposal_count += int(summary.get("proposal_count") or 0)
        multi_source_proposals += int(summary.get("multi_source_proposals") or 0)
        authority_capped_candidate_count += int(
            summary.get("authority_capped_candidate_count") or 0
        )
        normalization_record_count += int(
            summary.get("normalization_record_count") or 0
        )
        for channel, count in (summary.get("channel_counts") or {}).items():
            channel_counts[str(channel)] += int(count or 0)
    return {
        "product_neighborhood_count": len(consensuses),
        "candidate_count": candidate_count,
        "rejected_count": rejected_count,
        "proposal_count": proposal_count,
        "multi_source_proposals": multi_source_proposals,
        "authority_capped_candidate_count": authority_capped_candidate_count,
        "normalization_record_count": normalization_record_count,
        "channel_counts": dict(sorted(channel_counts.items())),
    }


def _candidates_from_consensus(consensus: dict[str, Any]) -> list[dict[str, Any]]:
    if consensus.get("schema_version") != "route_consensus.v1":
        return []
    candidates: list[dict[str, Any]] = []
    for proposal in consensus.get("proposals") or []:
        if not isinstance(proposal, Mapping):
            continue
        proposal = dict(proposal)
        source_records = [dict(row) for row in proposal.get("source_records") or [] if isinstance(row, Mapping)]
        if not source_records:
            source_records = [
                {
                    "candidate_id": str(proposal.get("consensus_id") or "consensus"),
                    "source_channel": "other",
                    "evidence_level": str(proposal.get("evidence_level") or "model_only"),
                    "confidence": str(proposal.get("confidence") or "low"),
                    "source_refs": _as_text_list(proposal.get("source_refs")),
                    "evidence_refs": _as_text_list(proposal.get("evidence_refs")),
                    "report_ref": "",
                }
            ]
        for record in source_records:
            source_channel = str(record.get("source_channel") or "other").strip().lower().replace("-", "_")
            candidates.append(
                _candidate(
                    candidate_id=str(record.get("candidate_id") or proposal.get("consensus_id") or "consensus"),
                    product_smiles=str(proposal.get("product_smiles") or ""),
                    precursor_smiles=list(proposal.get("precursor_smiles") or []),
                    reaction_family=str(proposal.get("reaction_family") or "unspecified"),
                    rationale=" | ".join(str(value) for value in proposal.get("rationales") or []),
                    source_channel=source_channel,
                    source_refs=_as_text_list(record.get("source_refs")),
                    evidence_refs=_as_text_list(record.get("evidence_refs")),
                    evidence_level=_safe_evidence_level(
                        record.get("evidence_level"),
                        source_channel=source_channel,
                    ),
                    confidence=str(record.get("confidence") or proposal.get("confidence") or "low"),
                    producer_evidence_level=str(
                        record.get("producer_evidence_level")
                        or record.get("evidence_level")
                        or "model_only"
                    ),
                    producer_evidence_level_raw=str(
                        record.get("producer_evidence_level_raw")
                        or record.get("producer_evidence_level")
                        or record.get("evidence_level")
                        or "model_only"
                    ),
                    producer_confidence=str(
                        record.get("producer_confidence")
                        or record.get("confidence")
                        or proposal.get("confidence")
                        or "low"
                    ),
                    producer_confidence_raw=str(
                        record.get("producer_confidence_raw")
                        or record.get("producer_confidence")
                        or record.get("confidence")
                        or proposal.get("confidence")
                        or "low"
                    ),
                    conditions=_as_text_list(proposal.get("conditions")),
                    catalyst="",
                    enzyme="",
                    limitations=_as_text_list(proposal.get("limitations")),
                    required_validation=_as_text_list(proposal.get("required_validation")),
                    report_ref=str(record.get("report_ref") or ""),
                )
            )
    return candidates


def _candidates_from_legacy_proposals(values: Any, *, target_smiles: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, raw in enumerate(values if isinstance(values, list) else []):
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        if str(row.get("source_type") or "") == "multi_source_consensus" or str(row.get("proposal_id") or "").startswith("consensus:"):
            continue
        precursor = row.get("precursor_smiles") or row.get("precursor_set_smiles") or row.get("precursors")
        if not precursor:
            continue
        producer_source_channel = _legacy_source_channel(row)
        producer_evidence_level = _legacy_evidence_level(
            row,
            source_channel=producer_source_channel,
        )
        candidates.append(
            _candidate(
                candidate_id=str(row.get("proposal_id") or f"legacy:{index}"),
                product_smiles=str(row.get("product_smiles") or row.get("target_smiles") or target_smiles),
                precursor_smiles=precursor if isinstance(precursor, list) else [str(precursor)],
                reaction_family=str(row.get("proposal_label") or row.get("reaction_family") or row.get("proposal_type") or "unspecified"),
                rationale=str(row.get("transformation_idea") or row.get("transformation_rationale") or ""),
                # Legacy blackboard labels are producer metadata, not a host
                # capability.  In particular, a row cannot gain ChemEnzy
                # authority merely by naming ``chem_enzy_adapter``.
                source_channel="other",
                source_refs=_as_text_list(row.get("source_refs")),
                evidence_refs=_as_text_list(row.get("evidence_refs")),
                evidence_level="model_only",
                confidence="low",
                producer_evidence_level=producer_evidence_level,
                producer_evidence_level_raw=str(
                    row.get("evidence_level") or producer_evidence_level
                ),
                producer_confidence=str(row.get("confidence") or "low"),
                producer_confidence_raw=str(row.get("confidence") or "low"),
                conditions=_as_text_list(row.get("conditions")),
                catalyst=str(row.get("catalyst") or ""),
                enzyme=str(row.get("enzyme") or ""),
                limitations=_as_text_list(
                    [
                        *(_as_text_list(row.get("risk_flags") or row.get("limitations"))),
                        "legacy_source_authority_unbound",
                        f"producer_source_channel:{producer_source_channel}",
                    ]
                ),
                required_validation=_as_text_list(
                    row.get("required_verification") or row.get("required_validation")
                ),
                report_ref=str(row.get("artifact_ref") or ""),
            )
        )
    return candidates


def _candidates_from_exact_rows(
    values: Any,
    *,
    target_smiles: str,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    candidates: list[dict[str, Any]] = []
    receipts: dict[str, list[dict[str, Any]]] = {}
    for index, raw in enumerate(values if isinstance(values, list) else []):
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        validation_status = str(row.get("validation_status") or "").strip().lower()
        if (
            row.get("accepted") is not True
            and row.get("validated") is not True
            and validation_status not in {"accepted", "validated", "accepted_by_verifier"}
        ):
            continue
        product = row.get("product_smiles") or row.get("products") or row.get("product")
        reactants = (
            row.get("reactant_smiles")
            or row.get("precursor_smiles")
            or row.get("reactants")
            or row.get("main_reactant_smiles")
        )
        if isinstance(product, list):
            product = product[0] if product else ""
        if not product or not reactants:
            continue
        refs = _as_text_list(
            [
                row.get("source_ref"),
                *_as_text_list(row.get("source_refs")),
                *_as_text_list(row.get("evidence_refs")),
            ]
        )
        if not refs:
            continue
        strict_exact = is_validated_source_detail_literature_step(row)
        source_bound_search_claim = (
            is_materialized_source_bound_literature_step(row)
        )
        evidence_level = "literature_exact" if strict_exact else "analogy"
        source_channel = "literature_exact" if strict_exact else "literature_analogy"
        limitations = _as_text_list(row.get("limitations"))
        required_validation = _as_text_list(
            row.get("required_validation") or ["parent_route_connection"]
        )
        if not strict_exact:
            limitations.append("untrusted_exact_literature_claim_downgraded_to_analogy")
            required_validation.extend(
                ["trusted_source_detail_step_binding", "deterministic_reaction_revalidation"]
            )
            if source_bound_search_claim:
                limitations.append(
                    "source_bound_search_admission_is_not_literature_precedent"
                )
                required_validation.append(
                    "trusted_precedent_registry_binding_for_l3"
                )
        candidate = _candidate(
                candidate_id=str(row.get("step_id") or row.get("row_id") or f"exact:{index}"),
                product_smiles=str(product or target_smiles),
                precursor_smiles=reactants if isinstance(reactants, list) else [str(reactants)],
                reaction_family=str(row.get("reaction_family") or row.get("reaction_class") or "literature exact step"),
                rationale=(
                    "validated exact literature row"
                    if strict_exact
                    else "unverified literature claim; analogy only"
                ),
                source_channel=source_channel,
                source_refs=refs,
                evidence_refs=_as_text_list(row.get("evidence_refs")),
                evidence_level=evidence_level,
                confidence="high" if strict_exact else "low",
                conditions=_as_text_list(row.get("conditions")),
                catalyst=str(row.get("catalyst") or ""),
                enzyme=str(row.get("enzyme") or ""),
                limitations=sorted(set(limitations)),
                required_validation=sorted(set(required_validation)),
                report_ref=str(row.get("artifact_ref") or ""),
                host_authority_binding=(
                    "validated_source_detail_literature_step"
                    if strict_exact
                    else ""
                ),
            )
        candidates.append(candidate)
        if strict_exact:
            material = make_exact_literature_admission_material(row)
            _append_admission_material(
                receipts,
                material,
                product_smiles=candidate["product_smiles"],
                precursor_smiles=candidate["precursor_smiles"],
            )
        elif source_bound_search_claim:
            material = make_materialized_literature_search_admission_material(
                row
            )
            _append_admission_material(
                receipts,
                material,
                product_smiles=candidate["product_smiles"],
                precursor_smiles=candidate["precursor_smiles"],
            )
    return candidates, receipts


def _candidates_from_chemenzy_proof_banks(
    values: Any,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    """Replay durable ChemEnzy proof banks into canonical route candidates.

    Serialized provider labels are deliberately ignored.  A ChemEnzy step
    receives computational-source authority only when the complete proof bank
    validates and every selected entry reproduces exactly under the currently
    imported host verifier.  One invalid entry rejects the whole bank so a
    partially tampered route cannot leak individual edges into the graph.
    """

    raw_values = (
        list(values)
        if isinstance(values, list)
        else [values]
        if isinstance(values, Mapping)
        else []
    )
    candidates: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    receipts: dict[str, list[dict[str, Any]]] = {}
    for wrapper_index, raw_wrapper in enumerate(raw_values):
        if not isinstance(raw_wrapper, Mapping):
            continue
        wrapper = dict(raw_wrapper)
        bank_value = (
            wrapper
            if wrapper.get("schema_version") == "route_proof_bank.v1"
            else wrapper.get("route_proof_bank")
        )
        artifact_ref = str(wrapper.get("artifact_ref") or "")
        bank = dict(bank_value) if isinstance(bank_value, Mapping) else {}
        target = canonicalize_smiles(bank.get("target_smiles"))
        bank_hash = str(bank.get("content_hash") or "")
        reasons = (
            validate_route_proof_bank(bank, expected_target_smiles=target)
            if bank and target
            else ["route_proof_bank_missing_or_target_invalid"]
        )
        entries = [
            dict(row)
            for row in bank.get("entries") or []
            if isinstance(row, Mapping)
        ]
        replayed_entries: list[dict[str, Any]] = []
        if not reasons:
            for entry in entries:
                proof_id = str(entry.get("proof_id") or "")
                replay = replay_route_proof_bank_entry(
                    bank,
                    proof_id=proof_id,
                    expected_target_smiles=target,
                )
                if replay.get("accepted") is not True:
                    reasons.extend(
                        f"entry:{proof_id}:{reason}"
                        for reason in replay.get("reasons") or [
                            "route_proof_bank_entry_replay_failed"
                        ]
                    )
                    continue
                replayed_entries.append(entry)

        bank_candidates: list[dict[str, Any]] = []
        if not reasons and len(replayed_entries) != len(entries):
            reasons.append("route_proof_bank_replay_coverage_mismatch")
        if not reasons:
            for entry in replayed_entries:
                proof_id = str(entry.get("proof_id") or "")
                steps = [
                    dict(step)
                    for step in (
                        (entry.get("materialized_route") or {}).get("steps") or []
                    )
                    if isinstance(step, Mapping)
                ]
                for step_index, step in enumerate(steps):
                    product = canonicalize_smiles(
                        step.get("product_smiles") or step.get("product")
                    )
                    raw_reactants = (
                        step.get("reactant_smiles")
                        or step.get("precursor_smiles")
                        or step.get("reactants")
                        or []
                    )
                    reactants = (
                        list(raw_reactants)
                        if isinstance(raw_reactants, (list, tuple))
                        else [raw_reactants]
                    )
                    canonical_reactants = [
                        canonicalize_smiles(value) for value in reactants
                    ]
                    if (
                        not product
                        or not canonical_reactants
                        or any(not value for value in canonical_reactants)
                    ):
                        reasons.append(
                            f"entry:{proof_id}:step:{step_index}:materialized_step_invalid"
                        )
                        continue
                    source_refs = _as_text_list(
                        [
                            artifact_ref,
                            f"route-proof-bank:sha256:{bank_hash}" if bank_hash else "",
                        ]
                    )
                    candidate = _candidate(
                            candidate_id=(
                                f"chemenzy:{proof_id}:step:{step_index}"
                            ),
                            product_smiles=product,
                            precursor_smiles=canonical_reactants,
                            reaction_family=str(
                                step.get("reaction_family")
                                or step.get("reaction_class")
                                or step.get("reaction_type")
                                or "ChemEnzy materialized step"
                            ),
                            rationale=(
                                "current-host replayed ChemEnzy route proof bank step"
                            ),
                            source_channel="chem_enzy",
                            source_refs=source_refs,
                            evidence_refs=_as_text_list(
                                [
                                    proof_id,
                                    str(entry.get("content_hash") or ""),
                                ]
                            ),
                            evidence_level="computational",
                            confidence="medium_high",
                            conditions=_as_text_list(step.get("conditions")),
                            catalyst=str(step.get("catalyst") or ""),
                            enzyme=str(step.get("enzyme") or ""),
                            limitations=[
                                "route_proof_bank_is_advisory_until_edge_and_frontier_closure"
                            ],
                            required_validation=[
                                "current_host_edge_reaction_verification",
                                "frontier_and_stock_audit",
                            ],
                            report_ref=artifact_ref,
                            host_authority_binding="deterministic_chemenzy_adapter",
                        )
                    bank_candidates.append(candidate)
                    material = make_chemenzy_admission_material(
                        bank,
                        source_entry_id=proof_id,
                        source_step_index=step_index,
                        artifact_ref=artifact_ref,
                    )
                    _append_admission_material(
                        receipts,
                        material,
                        product_smiles=product,
                        precursor_smiles=canonical_reactants,
                    )
        accepted = not reasons and bool(bank_candidates)
        if accepted:
            candidates.extend(bank_candidates)
        audits.append(
            {
                "schema_version": "chemenzy_proof_bank_replay_audit.v1",
                "wrapper_index": wrapper_index,
                "artifact_ref": artifact_ref,
                "target_smiles": target,
                "bank_content_hash": bank_hash,
                "accepted": accepted,
                "entry_count": len(entries),
                "replayed_entry_count": len(replayed_entries),
                "candidate_count": len(bank_candidates) if accepted else 0,
                "reasons": sorted(set(str(reason) for reason in reasons if str(reason))),
                "no_solved_claim": True,
            }
        )
    return candidates, audits, receipts


def _append_admission_material(
    receipts: dict[str, list[dict[str, Any]]],
    material: Mapping[str, Any],
    *,
    product_smiles: Any,
    precursor_smiles: Any,
) -> None:
    if not material:
        return
    signature = exact_edge_signature(product_smiles, precursor_smiles or [])
    if not signature:
        return
    rows = receipts.setdefault(signature, [])
    normalized = dict(material)
    if normalized not in rows:
        rows.append(normalized)


def _merge_admission_receipts(
    *values: Mapping[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {}
    for value in values:
        for signature, materials in value.items():
            for material in materials:
                rows = merged.setdefault(str(signature), [])
                normalized = dict(material)
                if normalized not in rows:
                    rows.append(normalized)
    return {key: rows for key, rows in sorted(merged.items())}


def _candidate(**values: Any) -> dict[str, Any]:
    candidate = {
        "schema_version": "retrosynthesis_candidate.v1",
        "candidate_id": str(values["candidate_id"]),
        "product_smiles": str(values["product_smiles"]),
        "precursor_smiles": list(values["precursor_smiles"]),
        "reaction_family": str(values["reaction_family"]),
        "transformation_rationale": str(values["rationale"]),
        "source_channel": str(values["source_channel"]),
        "source_refs": list(values["source_refs"]),
        "evidence_refs": list(values["evidence_refs"]),
        "evidence_level": str(values["evidence_level"]),
        "confidence": str(values["confidence"]),
        "conditions": list(values["conditions"]),
        "catalyst": str(values["catalyst"]),
        "enzyme": str(values["enzyme"]),
        "limitations": list(values["limitations"]),
        "required_validation": list(values["required_validation"]),
        "report_ref": str(values["report_ref"]),
        "no_solved_claim": True,
        "not_parent_route_proof": True,
    }
    for field in (
        "producer_evidence_level",
        "producer_evidence_level_raw",
        "producer_confidence",
        "producer_confidence_raw",
    ):
        if field in values:
            candidate[field] = str(values[field])
    if values.get("host_authority_binding"):
        # This private capability is emitted only after the deterministic
        # source-detail predicate above succeeds.  It is deliberately absent
        # from Codex child and legacy proposal schemas.
        candidate["_host_authority_binding"] = str(values["host_authority_binding"])
    return candidate


def _legacy_source_channel(row: dict[str, Any]) -> str:
    text = " ".join(
        [
            str(row.get("source_type") or ""),
            str(row.get("proposal_type") or ""),
            str(row.get("origin") or ""),
        ]
    ).lower()
    if "chem" in text or "enzyme" in text:
        return "chem_enzy"
    if "literature" in text or "exact" in text:
        return "literature_analogy"
    if "template" in text:
        return "template"
    if "stock" in text:
        return "stock"
    if "human" in text:
        return "human"
    return "other"


def _legacy_evidence_level(row: dict[str, Any], *, source_channel: str) -> str:
    if source_channel == "chem_enzy":
        return "computational"
    if source_channel == "template":
        return "analogy"
    if source_channel == "literature_analogy":
        return "analogy"
    return "model_only"


def _safe_evidence_level(value: Any, *, source_channel: str = "") -> str:
    text = str(value or "model_only")
    # Rebuilding is not a validation authority.  A historical Codex record is
    # always replayed at the unbound-model floor; its producer value is copied
    # separately for advisory/audit display by the caller above.
    if source_channel.startswith("codex_"):
        return "model_only"
    return "computational" if text == "validated" else text


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set)) else [value]
    seen: set[str] = set()
    rows: list[str] = []
    for item in values:
        if isinstance(item, (list, tuple, set)):
            nested = _as_text_list(item)
        else:
            text = str(item or "").strip()
            nested = [text] if text else []
        for text in nested:
            if text in seen:
                continue
            seen.add(text)
            rows.append(text)
    return rows
