"""ChemEnzy proposal request and deterministic content contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping

from cascade_planner.application.blind_benchmark_contract import canonical_smiles


@dataclass(frozen=True, slots=True)
class ChemEnzyProposalRequest:
    """One seed or frontier-guided request; never a second search queue."""

    target_smiles: str
    target_name: str = ""
    mode: str = "seed"
    frontier_smiles: tuple[str, ...] = ()
    route_family_ids: tuple[str, ...] = ()
    retron_hints: tuple[str, ...] = ()
    forbidden_smiles: tuple[str, ...] = ()
    limits: Mapping[str, Any] = field(default_factory=dict)
    stop_conditions: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "chemenzy_proposal_request.v2"

    def __post_init__(self) -> None:
        if self.mode not in {"seed", "guided_frontier"}:
            raise ValueError("invalid_chemenzy_proposal_mode")
        if not self.target_smiles.strip():
            raise ValueError("chemenzy_target_smiles_required")
        if self.mode == "guided_frontier" and not self.frontier_smiles:
            raise ValueError("guided_chemenzy_frontier_required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "target_name": self.target_name,
            "target_smiles": self.target_smiles,
            "frontier_smiles": list(self.frontier_smiles),
            "route_family_ids": list(self.route_family_ids),
            "retron_hints": list(self.retron_hints),
            "forbidden_smiles": list(self.forbidden_smiles),
            "limits": dict(self.limits),
            "stop_conditions": dict(self.stop_conditions),
            "semantics": {
                "canonical_frontier_is_authoritative": True,
                "provider_has_no_private_expansion_state": True,
                "result_is_proposal_only": True,
            },
        }


def _json_safe_copy(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
    )


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_safe_copy(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _opaque_target_name(target_smiles: str) -> str:
    canonical = canonical_smiles(str(target_smiles or ""))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"target-{digest[:8]}"


def _result(status: str, **values: Any) -> dict[str, Any]:
    extra_semantics = dict(values.pop("semantics", {}) or {})
    return {
        "schema_version": "v4_chemenzy_proposal_stage.v1",
        "stage": "chemenzy_baseline",
        "status": status,
        **values,
        "semantics": {
            "proposal_only": True,
            "canonical_host_admission_required": True,
            "raw_backend_solved_is_not_route_proof": True,
            "codex_receives_proposals_through_shared_hypergraph": True,
            **extra_semantics,
        },
    }


__all__ = ["ChemEnzyProposalRequest"]
