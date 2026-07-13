"""Immutable contract and host validation for scientific replay packs."""
from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.application.retrosynthesis_workers import (
    VERSIONED_INVENTORY_ARTIFACT_SCHEMA,
    normalize_source_binding,
)
from cascade_planner.harness.reaction_step_verifier import verify_reaction_step
from cascade_planner.routes.admission import audit_retrosynthetic_candidate


REPLAY_PACK_SCHEMA = "retrosynthesis_replay_pack.v1"
REPLAY_RESULT_SCHEMA = "retrosynthesis_replay_result.v1"
REPLAY_STAGES = ("plan", "materialization", "evidence", "validation", "stock")


class ReplayPackError(ValueError):
    """A replay pack is invalid or does not reproduce its contract."""


def load_replay_pack(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        pack = json_copy(value)
    else:
        path = Path(value).expanduser().resolve()
        pack = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(pack, dict):
        raise ReplayPackError("replay_pack_must_be_an_object")
    if pack.get("schema_version") != REPLAY_PACK_SCHEMA:
        raise ReplayPackError("replay_pack_schema_invalid")
    supplied = str(pack.get("content_sha256") or "").lower()
    payload = {key: item for key, item in pack.items() if key != "content_sha256"}
    if supplied != digest(payload):
        raise ReplayPackError("replay_pack_digest_invalid")
    validate_replay_pack(pack)
    return pack


def with_replay_pack_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    pack = json_copy(value)
    pack.pop("content_sha256", None)
    pack["content_sha256"] = digest(pack)
    return pack


def validate_replay_pack(pack: Mapping[str, Any]) -> None:
    required = (
        "case_id",
        "target",
        "acceptance",
        "budget",
        "global_plan",
        "sources",
        "reactions",
        "inventory",
        "expected",
    )
    if any(key not in pack for key in required):
        raise ReplayPackError("replay_pack_required_field_missing")
    target = dict(pack["target"])
    if not str(target.get("name") or "") or not str(target.get("smiles") or ""):
        raise ReplayPackError("replay_pack_target_invalid")
    acceptance = dataclass_value(RetrosynthesisAcceptanceSpec, pack["acceptance"])
    budget = dataclass_value(RetrosynthesisRunBudget, pack["budget"])
    if budget.max_model_invocations or budget.max_visual_invocations:
        raise ReplayPackError("replay_pack_must_be_model_free")
    if budget.max_accepted_expansions < len(pack["reactions"]):
        raise ReplayPackError("replay_pack_expansion_budget_too_small")
    if not pack["sources"] or not pack["reactions"]:
        raise ReplayPackError("replay_pack_facts_missing")
    for source in pack["sources"]:
        binding = normalize_source_binding(source["binding"])
        if (
            binding["usable_for_extraction"] is not True
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(binding.get("artifact_sha256") or "")
            )
            or not source.get("rows")
        ):
            raise ReplayPackError("replay_pack_source_invalid")
    edge_digests: set[str] = set()
    for reaction in pack["reactions"]:
        audit = audit_retrosynthetic_candidate(
            reaction.get("product_smiles"), reaction.get("reactant_smiles") or []
        )
        if (
            audit.get("accepted") is not True
            or reaction.get("edge_digest") != audit.get("edge_digest")
        ):
            raise ReplayPackError("replay_pack_reaction_identity_invalid")
        proof = verify_reaction_step(
            {
                "product_smiles": reaction["product_smiles"],
                "reactant_smiles": reaction["reactant_smiles"],
                "mapped_reaction_smiles": reaction.get("mapped_reaction_smiles"),
            }
        )
        if proof.get("accepted") is not True:
            raise ReplayPackError("replay_pack_reaction_validation_failed")
        edge_digests.add(str(reaction["edge_digest"]))
    if len(edge_digests) != len(pack["reactions"]):
        raise ReplayPackError("replay_pack_reaction_duplicate")
    artifact = dict(dict(pack["inventory"]).get("artifact") or {})
    if artifact.get("schema_version") != VERSIONED_INVENTORY_ARTIFACT_SCHEMA:
        raise ReplayPackError("replay_pack_inventory_invalid")
    route_count = len(pack["global_plan"].get("route_families") or [])
    if acceptance.minimum_complete_routes > route_count:
        raise ReplayPackError("replay_pack_route_count_below_acceptance")


def dataclass_value(cls: Any, value: Mapping[str, Any]) -> Any:
    names = {
        field.name
        for field in fields(cls)
        if field.init and field.name != "schema_version"
    }
    return cls(**{key: item for key, item in dict(value).items() if key in names})


def json_copy(value: Any) -> Any:
    return json.loads(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )


def digest(value: Any) -> str:
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
    "REPLAY_PACK_SCHEMA",
    "REPLAY_RESULT_SCHEMA",
    "REPLAY_STAGES",
    "ReplayPackError",
    "dataclass_value",
    "digest",
    "load_replay_pack",
    "with_replay_pack_digest",
]
