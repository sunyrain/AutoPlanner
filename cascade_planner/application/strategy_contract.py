"""Durable, structure-aware strategy contracts.

StrategyCards used to be prompt-local metadata.  This module gives them one
canonical representation that can cross the worker, materializer and
canonical-graph boundaries without turning evidence fields into design
constraints.  The contract is deliberately hypothesis-only: a digest proves
identity/replayability, never reaction feasibility or literature support.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping


STRATEGY_CARD_SCHEMA = "strategy_card.v1"
REACTION_EDIT_SIGNATURE_SCHEMA = "reaction_edit_signature.v1"

_TEXT_FIELDS = (
    "scaffold_motif",
    "key_forward_transformation",
    "protection_policy",
    "stereochemical_plan",
    "convergence_plan",
    "skeleton_change_class",
    "expected_complexity_drop",
    "orthogonality_basis",
    "strategy_signature",
)

# The worker schema keeps every possible ReactionJSON field nullable because
# strict structured outputs require one fixed object shape.  Normalize each
# primitive back to its semantic field set before computing a digest or
# replaying it; otherwise harmless ``null``-schema filler (for example
# ``order`` on ``break_bond``) is rejected as an unknown replay field.
_REACTION_OPERATION_FIELDS = {
    "break_bond": {"op", "map_a", "map_b"},
    "add_bond": {"op", "map_a", "map_b", "order"},
    "change_bond_order": {"op", "map_a", "map_b", "delta"},
    "change_atom": {"op", "map_idx", "atomic_num", "element", "formal_charge", "isotope"},
    "set_explicit_h": {"op", "map_idx", "count", "no_implicit"},
    "add_group": {"op", "map_idx", "fragment_smiles"},
    "remove_group": {"op", "map_indices"},
    "invert_stereocenter": {"op", "map_idx"},
    "clear_stereocenter": {"op", "map_idx"},
    "set_bond_stereo": {"op", "map_a", "map_b", "stereo", "stereo_atom_maps"},
}


def normalize_strategy_card(
    value: Mapping[str, Any] | None,
    *,
    strategy_id: str = "",
    route_family_id: str = "",
    reaction_operations: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return the canonical card and stable digest for a model hypothesis."""

    raw = dict(value or {})
    operations = normalize_reaction_operations(reaction_operations)
    operation_digest = reaction_edit_digest(operations) or str(
        raw.get("reaction_edit_digest") or ""
    )
    edit_signature = (
        reaction_edit_signature(operations)
        if operations
        else dict(raw.get("reaction_edit_signature") or {})
    )
    card: dict[str, Any] = {
        "schema_version": STRATEGY_CARD_SCHEMA,
        **{
            field: str(raw.get(field) or "").strip()
            for field in _TEXT_FIELDS
        },
        "key_bond_changes": _string_list(raw.get("key_bond_changes")),
        "functional_group_conflicts": _string_list(
            raw.get("functional_group_conflicts")
        ),
        "strategic_step_count": _bounded_int(raw.get("strategic_step_count"), 1, 2),
        "execution_domain": _execution_domain(raw),
        "route_family_id": str(route_family_id or raw.get("route_family_id") or ""),
        "reaction_edit_digest": operation_digest,
        "reaction_edit_signature": edit_signature,
    }
    card["key_bond_signature"] = key_bond_signature(card["key_bond_changes"])
    card["topology_signature"] = _normalized_text(
        [card["scaffold_motif"], card["skeleton_change_class"]]
    )
    card["convergence_signature"] = _normalized_text([card["convergence_plan"]])
    card["stereochemical_signature"] = _normalized_text(
        [card["stereochemical_plan"]]
    )
    structural = {
        "key_bond_signature": card["key_bond_signature"],
        "reaction_edit_digest": operation_digest,
        "topology_signature": card["topology_signature"],
        "convergence_signature": card["convergence_signature"],
        "stereochemical_signature": card["stereochemical_signature"],
        "execution_domain": card["execution_domain"],
    }
    card["structural_signature"] = _digest(structural)
    body = {
        key: value
        for key, value in card.items()
        if key not in {"strategy_id", "strategy_digest", "route_family_id"}
    }
    digest = _digest(body)
    card["strategy_digest"] = digest
    card["strategy_id"] = str(strategy_id or raw.get("strategy_id") or f"strategy:{digest[:24]}")
    card["semantics"] = {
        "hypothesis_only": True,
        "evidence_independent_identity": True,
        "reaction_edit_digest_is_not_reaction_proof": True,
    }
    card["content_sha256"] = _digest(card)
    return card


def normalize_reaction_operations(
    operations: Iterable[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for raw in operations or ():
        if not isinstance(raw, Mapping):
            continue
        row = {
            str(key): value
            for key, value in dict(raw).items()
            if value not in (None, "", [], {})
        }
        kind = str(row.get("op") or "").strip().lower()
        allowed = _REACTION_OPERATION_FIELDS.get(kind)
        if allowed is not None:
            row = {key: value for key, value in row.items() if key in allowed}
        row["op"] = kind
        if row.get("op"):
            rows.append(row)
    return tuple(rows)


def reaction_edit_signature(
    operations: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    rows = normalize_reaction_operations(operations)
    edits = []
    for row in rows:
        edits.append(
            {
                "op": str(row.get("op") or ""),
                "map_a": _maybe_int(row.get("map_a")),
                "map_b": _maybe_int(row.get("map_b")),
                "map_idx": _maybe_int(row.get("map_idx")),
                "order": row.get("order"),
                "delta": row.get("delta"),
                "atomic_num": _maybe_int(row.get("atomic_num")),
                "element": str(row.get("element") or ""),
                "count": _maybe_int(row.get("count")),
                "stereo": str(row.get("stereo") or ""),
                "stereo_atom_maps": sorted(
                    _maybe_int(value)
                    for value in row.get("stereo_atom_maps") or []
                    if _maybe_int(value) is not None
                ),
            }
        )
    return {
        "schema_version": REACTION_EDIT_SIGNATURE_SCHEMA,
        "operations": edits,
        "changed_map_pairs": sorted(
            {
                tuple(sorted((int(row["map_a"]), int(row["map_b"]))))
                for row in edits
                if row.get("map_a") is not None and row.get("map_b") is not None
            }
        ),
        "topology_edit_classes": sorted(
            {
                _edit_class(str(row.get("op") or ""))
                for row in edits
                if row.get("op")
            }
        ),
    }


def reaction_edit_digest(operations: Iterable[Mapping[str, Any]] = ()) -> str:
    rows = reaction_edit_signature(operations)
    return _digest(rows) if rows["operations"] else ""


def key_bond_signature(values: Iterable[Any]) -> tuple[str, ...]:
    """Normalize key-bond descriptions while ignoring prose labels."""

    signatures: set[str] = set()
    for value in values or ():
        text = _normalized_text([str(value)])
        if not text:
            continue
        maps = re.findall(r"(?:map|atom|c|n|o|s)?\s*#?\s*(\d+)\s*[-:=/>]+\s*(?:map|atom|c|n|o|s)?\s*#?\s*(\d+)", text)
        if maps:
            signatures.update("map_pair:" + ":".join(sorted(pair)) for pair in maps)
        else:
            signatures.add(text)
    return tuple(sorted(signatures))


def strategy_structural_signature(
    card: Mapping[str, Any] | None,
    *,
    reaction_operations: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    normalized = normalize_strategy_card(
        card,
        reaction_operations=reaction_operations,
    )
    return {
        "structural_signature": normalized["structural_signature"],
        "key_bond_signature": normalized["key_bond_signature"],
        "reaction_edit_digest": normalized["reaction_edit_digest"],
        "topology_signature": normalized["topology_signature"],
        "convergence_signature": normalized["convergence_signature"],
        "stereochemical_signature": normalized["stereochemical_signature"],
        "execution_domain": normalized["execution_domain"],
    }


def strategy_cards_conflict(
    candidate: Mapping[str, Any] | None,
    prior: Mapping[str, Any] | None,
) -> bool:
    """Detect structural duplication, independent of model-authored names."""

    left = normalize_strategy_card(candidate)
    right = normalize_strategy_card(prior)
    if left["strategy_digest"] == right["strategy_digest"]:
        return True
    if (
        left["reaction_edit_digest"]
        and right["reaction_edit_digest"]
        and left["reaction_edit_digest"] == right["reaction_edit_digest"]
    ):
        return True
    if left["key_bond_signature"] and left["key_bond_signature"] == right["key_bond_signature"]:
        if left["topology_signature"] == right["topology_signature"]:
            return True
    return bool(
        left["structural_signature"] == right["structural_signature"]
        and left["execution_domain"] == right["execution_domain"]
    )


def strategy_execution_domain(card: Mapping[str, Any] | None) -> str:
    return _execution_domain(dict(card or {}))


def strategy_card_has_content(card: Mapping[str, Any] | None) -> bool:
    row = dict(card or {})
    return any(
        str(row.get(field) or "")
        for field in (
            "scaffold_motif",
            "key_forward_transformation",
            "key_bond_signature",
            "reaction_edit_digest",
        )
    )


def _execution_domain(raw: Mapping[str, Any]) -> str:
    explicit = str(raw.get("execution_domain") or "").strip().lower().replace("-", "_")
    if explicit in {"chemical", "enzymatic", "whole_cell", "hybrid", "mechanistic"}:
        return explicit
    text = " ".join(str(raw.get(key) or "") for key in _TEXT_FIELDS).lower()
    if any(token in text for token in ("whole cell", "whole_cell")):
        return "whole_cell"
    if any(token in text for token in ("hybrid", "chemoenzymatic")):
        return "hybrid"
    if any(token in text for token in ("enzyme", "enzymatic", "synthase", "biocatal")):
        return "enzymatic"
    if "mechanism" in text:
        return "mechanistic"
    return "chemical"


def _edit_class(op: str) -> str:
    if op in {"add_bond", "break_bond", "change_bond_order"}:
        return "bond_edit"
    if op in {"add_group", "remove_group", "change_atom"}:
        return "functional_group_edit"
    if op in {"invert_stereocenter", "clear_stereocenter", "set_bond_stereo"}:
        return "stereochemical_edit"
    return "atom_state_edit"


def _normalized_text(values: Iterable[Any]) -> str:
    text = " ".join(str(value or "") for value in values)
    return re.sub(r"[^a-z0-9#:=/> -]+", " ", text.lower()).strip()


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, Iterable):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _bounded_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(maximum, int(value)))
    except (TypeError, ValueError):
        return minimum


def _maybe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


__all__ = [
    "REACTION_EDIT_SIGNATURE_SCHEMA",
    "STRATEGY_CARD_SCHEMA",
    "key_bond_signature",
    "normalize_reaction_operations",
    "normalize_strategy_card",
    "reaction_edit_digest",
    "reaction_edit_signature",
    "strategy_cards_conflict",
    "strategy_card_has_content",
    "strategy_execution_domain",
    "strategy_structural_signature",
]
