from __future__ import annotations

import hashlib
import json

import pytest

from cascade_planner.application.fact_lifecycle import (
    build_fact_lifecycle_event,
    fact_lifecycle_state,
    summarize_fact_lifecycle,
    validate_fact_lifecycle_event,
)
from cascade_planner.application.proof_policy import (
    ProofPolicy,
    stitch_edge_proof,
    stitch_leaf_stock_proof,
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _with_digest(value: dict) -> dict:
    row = dict(value)
    row["content_sha256"] = _digest(row)
    return row


def _policy() -> ProofPolicy:
    return ProofPolicy(
        minimum_edge_proof_level=3,
        minimum_independent_source_groups=1,
        require_stock_for_every_selected_leaf=True,
        stock_boundary="procurement",
    )


def _proof_graph() -> tuple[dict, dict[str, str]]:
    reaction_proof = {
        "schema_version": "reaction_step_proof.v1",
        "accepted": True,
        "proof_level": "L2_reaction_validated",
    }
    reaction_proof["proof_digest"] = _digest(reaction_proof)
    source = _with_digest(
        {
            "source_binding_id": "source:one",
            "independence_group": "patent:one",
        }
    )
    exact = _with_digest(
        {
            "record_id": "exact:one",
            "edge_digest": "ester",
            "source_binding_id": "binding:one",
            "independence_group": "patent:one",
        }
    )
    procedure = _with_digest(
        {
            "procedure_record_id": "procedure:one",
            "exact_record_id": "exact:one",
            "edge_digest": "ester",
            "source_binding_id": "binding:one",
            "condition_completeness": {"complete": True},
        }
    )
    edge = _with_digest(
        {
            "edge_id": "edge:ester",
            "edge_digest": "ester",
            "product_molecule_id": "molecule:product",
            "precursor_molecule_ids": ["molecule:leaf"],
            "reaction_proofs": [reaction_proof],
            "exact_record_ids": ["exact:one"],
            "procedure_record_ids": ["procedure:one"],
        }
    )
    graph = {
        "edges": {"edge:ester": edge},
        "source_bindings": {"source:one": source},
        "source_aliases": {"binding:one": "source:one"},
        "exact_records": {"exact:one": exact},
        "procedure_records": {"procedure:one": procedure},
        "fact_lifecycle_events": {},
    }
    ids = {
        "proof": reaction_proof["proof_digest"],
        "source_digest": source["content_sha256"],
    }
    return graph, ids


def test_lifecycle_event_is_digest_bound_and_restore_is_append_only() -> None:
    revoked = build_fact_lifecycle_event(
        subject_kind="source_binding",
        subject_id="source:one",
        subject_content_sha256="a" * 64,
        action="revoke",
        effective_at="2026-07-15T12:00:00+00:00",
        reason_codes=["publisher_retraction"],
    )
    assert validate_fact_lifecycle_event(revoked) == []
    state = fact_lifecycle_state(
        {revoked["event_id"]: revoked},
        subject_kind="source_binding",
        subject_id="source:one",
        subject_content_sha256="a" * 64,
    )
    assert state["status"] == "revoked"
    assert state["active"] is False

    restored = build_fact_lifecycle_event(
        subject_kind="source_binding",
        subject_id="source:one",
        subject_content_sha256="a" * 64,
        action="restore",
        effective_at="2026-07-15T13:00:00Z",
        reason_codes=["retraction_withdrawn"],
        supersedes_event_id=revoked["event_id"],
    )
    restored_state = fact_lifecycle_state(
        {revoked["event_id"]: revoked, restored["event_id"]: restored},
        subject_kind="source_binding",
        subject_id="source:one",
        subject_content_sha256="a" * 64,
    )
    assert restored_state["active"] is True
    assert restored_state["event_count"] == 2

    with pytest.raises(ValueError, match="authority_scope_invalid"):
        build_fact_lifecycle_event(
            subject_kind="source_binding",
            subject_id="source:one",
            subject_content_sha256="a" * 64,
            action="revoke",
            effective_at="2026-07-15T12:00:00Z",
            reason_codes=["untrusted_request"],
            authority_scope="model_claimed_revocation",
        )


def test_revoked_source_downgrades_edge_without_deleting_audit_facts() -> None:
    graph, ids = _proof_graph()
    revoke = build_fact_lifecycle_event(
        subject_kind="source_binding",
        subject_id="source:one",
        subject_content_sha256=ids["source_digest"],
        action="revoke",
        effective_at="2026-07-15T12:00:00Z",
        reason_codes=["source_retracted"],
    )
    graph["fact_lifecycle_events"] = {revoke["event_id"]: revoke}

    proof = stitch_edge_proof(graph, "edge:ester", policy=_policy())

    assert proof["reaction_validated"] is True
    assert proof["exact_source_bound"] is False
    assert proof["achieved_level"] == 2
    assert proof["accepted"] is False
    assert proof["exact_record_ids"] == []
    assert graph["exact_records"]["exact:one"]
    assert graph["procedure_records"]["procedure:one"]
    assert proof["inactive_facts"] == [
        {
            "subject_kind": "source_binding",
            "subject_id": "source:one",
            "status": "revoked",
            "lifecycle_event_id": revoke["event_id"],
            "effective_at": "2026-07-15T12:00:00Z",
            "reason_codes": ["source_retracted"],
            "authority_scope": "source_fact_lifecycle_authority",
        }
    ]

    summary = summarize_fact_lifecycle(graph)
    assert summary["event_count"] == 1
    assert summary["revoked_fact_count"] == 1


def test_expired_stock_observation_cannot_close_procurement() -> None:
    observation = _with_digest(
        {
            "stock_observation_id": "stock:leaf",
            "molecule_id": "molecule:leaf",
            "accepted": True,
            "audited_as_of": "2026-07-15T00:00:00Z",
            "provider_result": {
                "payload": {"boundary_type": "commercially_orderable"}
            },
        }
    )
    molecule = _with_digest(
        {
            "molecule_id": "molecule:leaf",
            "canonical_smiles": "CCO",
            "active_stock_observation_id": "stock:leaf",
            "stock_observation_ids": ["stock:leaf"],
            "inactive_stock_observation_ids": [],
        }
    )
    expired = build_fact_lifecycle_event(
        subject_kind="stock_observation",
        subject_id="stock:leaf",
        subject_content_sha256=observation["content_sha256"],
        action="expire",
        effective_at="2026-07-16T00:00:00Z",
        reason_codes=["offer_expired"],
    )
    graph = {
        "molecules": {"molecule:leaf": molecule},
        "stock_observations": {"stock:leaf": observation},
        "fact_lifecycle_events": {expired["event_id"]: expired},
    }

    proof = stitch_leaf_stock_proof(graph, "molecule:leaf", policy=_policy())

    assert proof["accepted"] is False
    assert "stock_observation_expired:stock:leaf" in proof["reasons"]
    assert proof["inactive_fact_count"] == 1
