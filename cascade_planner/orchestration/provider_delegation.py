"""Complete provider metadata from a Codex-selected canonical skeleton step."""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence


Canonicalizer = Callable[[Any], str]


def complete_chemenzy_delegation(
    *,
    skeletons: Sequence[Mapping[str, Any]],
    shared_intermediates: Sequence[Mapping[str, Any]] = (),
    frontier_priorities: Sequence[Mapping[str, Any]],
    campaign_target: str,
    canonicalize: Canonicalizer,
    max_requests: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Complete Codex-selected local-provider requests without changing chemistry.

    Models commonly refer to a shared intermediate by ``intermediate_id`` in a
    frontier priority.  Treat that as equivalent to selecting a skeleton step:
    both already carry a model-proposed structure and this helper adds only the
    deterministic scheduling metadata required by the host.
    """

    priorities = [dict(value) for value in frontier_priorities]
    candidates: dict[str, dict[str, Any]] = {}
    for skeleton in skeletons:
        family_id = str(skeleton.get("route_family_id") or "")
        for value in skeleton.get("steps") or []:
            if not isinstance(value, Mapping):
                continue
            step_id = str(value.get("step_id") or "")
            product = canonicalize(value.get("product_smiles"))
            if step_id and product and product != campaign_target:
                candidates[step_id] = {
                    **dict(value),
                    "canonical_product": product,
                    "route_family_ids": [family_id] if family_id else [],
                    "candidate_kind": "skeleton_step",
                }
    for value in shared_intermediates:
        if not isinstance(value, Mapping):
            continue
        intermediate_id = str(value.get("intermediate_id") or "")
        product = canonicalize(value.get("smiles"))
        if intermediate_id and product and product != campaign_target:
            candidates[intermediate_id] = {
                **dict(value),
                "canonical_product": product,
                "route_family_ids": [
                    str(family_id)
                    for family_id in value.get("route_family_ids") or []
                    if str(family_id)
                ],
                "candidate_kind": "shared_intermediate",
            }

    repairs: list[dict[str, Any]] = []
    for index, priority in enumerate(priorities):
        providers = {
            str(value).strip().lower()
            for value in priority.get("provider_preferences") or []
            if str(value).strip()
        }
        target = canonicalize(priority.get("target_smiles"))
        if "chemenzy" not in providers or target != campaign_target:
            continue
        proposal_id = str(priority.get("proposal_id") or "")
        candidate = candidates.get(proposal_id)
        if candidate is not None:
            priorities[index]["target_smiles"] = candidate["canonical_product"]
            priorities[index]["route_family_ids"] = list(candidate["route_family_ids"])
            reason = "campaign_target_provider_rebound_to_selected_non_root_candidate"
            replacement = candidate["canonical_product"]
        else:
            priorities[index]["provider_preferences"] = [
                value
                for value in priority.get("provider_preferences") or []
                if str(value).strip().lower() != "chemenzy"
            ]
            reason = "campaign_target_provider_downgraded_to_host_priority"
            replacement = campaign_target
        repairs.append(
            {
                "schema_version": "global_campaign_contract_repair.v1",
                "field": "frontier_priorities.provider_delegation",
                "priority_id": str(priority.get("priority_id") or ""),
                "proposal_id": proposal_id,
                "reason": reason,
                "replacement_canonical_smiles": replacement,
                "semantics": {
                    "chemistry_unchanged": True,
                    "host_priority_preserved": True,
                    "campaign_target_not_delegated": candidate is None,
                    "normal_validation_still_required": True,
                },
            }
        )
    completed = sum(
        _is_complete_chemenzy_request(row, campaign_target, canonicalize)
        for row in priorities
    )
    ranked = sorted(
        enumerate(priorities),
        key=lambda item: (-_priority(item[1]), item[0]),
    )
    for index, priority in ranked:
        if completed >= max(0, int(max_requests)):
            break
        if _is_complete_chemenzy_request(priority, campaign_target, canonicalize):
            continue
        proposal_id = str(priority.get("proposal_id") or "")
        candidate = candidates.get(proposal_id)
        if candidate is None:
            continue
        hints = [
            str(value).strip()
            for value in candidate.get("source_hints") or []
            if str(value).strip()
        ]
        hypothesis = str(
            candidate.get("transformation_hypothesis")
            or candidate.get("strategic_role")
            or ""
        ).strip()
        if hypothesis:
            hints.append(hypothesis)
        priorities[index].update(
            {
                "target_smiles": candidate["canonical_product"],
                "provider_preferences": ["chemenzy"],
                "retron_hints": list(dict.fromkeys(hints))[:4],
                "route_family_ids": list(candidate["route_family_ids"]),
            }
        )
        completed += 1
        repairs.append(
            {
                "schema_version": "global_campaign_contract_repair.v1",
                "field": "frontier_priorities.provider_delegation",
                "priority_id": str(priority.get("priority_id") or ""),
                "proposal_id": proposal_id,
                "reason": (
                    "provider_metadata_completed_from_codex_selected_"
                    + str(candidate["candidate_kind"])
                ),
                "replacement_canonical_smiles": candidate["canonical_product"],
                "semantics": {
                    "chemistry_unchanged": True,
                    "codex_frontier_selection_preserved": True,
                    "scheduling_metadata_only": True,
                    "normal_validation_still_required": True,
                },
            }
        )
    return priorities, repairs


def _is_complete_chemenzy_request(
    row: Mapping[str, Any], campaign_target: str, canonicalize: Canonicalizer
) -> bool:
    providers = {
        str(value).strip().lower()
        for value in row.get("provider_preferences") or []
    }
    target = canonicalize(row.get("target_smiles"))
    return "chemenzy" in providers and bool(target) and target != campaign_target


def _priority(row: Mapping[str, Any]) -> float:
    try:
        return float(row.get("priority") or 0.0)
    except (TypeError, ValueError):
        return 0.0


__all__ = ["complete_chemenzy_delegation"]
