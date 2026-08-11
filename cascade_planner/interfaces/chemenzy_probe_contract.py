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
    random_seed: int = 0
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
        if isinstance(self.random_seed, bool) or not 0 <= int(self.random_seed) <= 2**32 - 1:
            raise ValueError("invalid_chemenzy_random_seed")
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "random_seed": int(self.random_seed),
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
def provider_invocation_binding(
    request: Mapping[str, Any],
    *,
    random_seed: int,
    raw_proposal_sha256: str,
    raw_result_sha256: str,
    runtime_preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind replay inputs and receipts without claiming backend determinism."""
    preflight = dict(runtime_preflight or {})
    capability = dict(preflight.get("capability_probe") or {})
    limits = dict(request.get("limits") or {})
    runtime = {
        "env_prefix_selection_source": str(preflight.get("env_prefix_selection_source") or ""),
        "python_executable": str(preflight.get("python_executable") or ""),
        "vendor_root": str(preflight.get("vendor_root") or ""),
        "requested_one_step_models": list(preflight.get("requested_one_step_models") or []),
        "model_override_digest": str(preflight.get("model_override_digest") or ""),
        "model_content_binding_sha256": str(capability.get("model_content_binding_sha256") or preflight.get("model_content_binding_sha256") or ""),
        "model_content_identity_complete": (capability.get("model_content_identity_complete", preflight.get("model_content_identity_complete")) is True),
        "model_path_checks": list(capability.get("model_path_checks") or []),
        "stock_path_checks": list(capability.get("stock_path_checks") or []),
        "requested_stock_names": list(limits.get("stock_names") or []),
        "requested_stock_paths": dict(limits.get("stock_paths") or {}),
    }
    runtime_sha256 = _content_sha256(runtime)
    replay_key_sha256 = _content_sha256({
        "schema_version": "chemenzy_provider_replay_key.v1", "request_sha256": _content_sha256(request),
        "random_seed": int(random_seed), "runtime_binding_sha256": runtime_sha256,
    })
    return {
        "schema_version": "chemenzy_provider_invocation_binding.v1",
        "request_sha256": _content_sha256(request),
        "replay_key_sha256": replay_key_sha256,
        "random_seed": int(random_seed),
        "raw_proposal_sha256": str(raw_proposal_sha256 or ""),
        "raw_result_sha256": str(raw_result_sha256 or ""),
        "runtime_binding_sha256": runtime_sha256,
        "runtime_binding": runtime,
        "semantics": {
            "raw_proposal_sha_excludes_operational_metadata": True,
            "raw_result_sha_binds_complete_operational_receipt": True,
            "replay_key_detects_input_or_runtime_conflict_only": True,
            "binding_does_not_fabricate_backend_determinism": True,
            "full_model_file_content_identity_is_not_proven": not runtime["model_content_identity_complete"],
        },
    }
__all__ = ["ChemEnzyProposalRequest", "provider_invocation_binding"]
