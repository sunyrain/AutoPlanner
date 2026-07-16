"""HTTP routes for Program innovation review, stores, and experience memory."""

from __future__ import annotations

from typing import Any, Callable

from flask import Blueprint, jsonify, request

from cascade_planner.web.v4_program_payload import program_innovation_payload


def register_program_innovation_routes(
    blueprint: Blueprint, factory: Callable[[], Any]
) -> None:
    prefix = "/api/v4/runs/<run_id>/programs/innovations"

    @blueprint.post(prefix)
    def route_program_innovations(run_id: str):
        return jsonify(factory().route_program_innovations(
            run_id,
            **program_innovation_payload(_payload(), allow_reported_candidates=True),
        ))

    @blueprint.get(prefix + "/store")
    def biocatalytic_program_store(run_id: str):
        return jsonify(factory().biocatalytic_program_store(run_id))

    @blueprint.post(prefix + "/admit")
    def admit_route_program_innovations(run_id: str):
        payload = _payload()
        result = factory().admit_route_program_innovations(
            run_id,
            **program_innovation_payload(payload),
            enable_biocatalytic_program_admission=(
                payload.get("enable_biocatalytic_program_admission") is True
            ),
        )
        return jsonify(result), 201 if result.get("created") is True else 200

    @blueprint.get(prefix + "/mechanisms/store")
    def mechanism_program_store(run_id: str):
        return jsonify(factory().mechanism_program_store(run_id))

    @blueprint.post(prefix + "/mechanisms/admit")
    def admit_route_mechanism_programs(run_id: str):
        payload = _payload()
        result = factory().admit_route_mechanism_programs(
            run_id,
            **program_innovation_payload(payload),
            enable_mechanism_program_admission=(
                payload.get("enable_mechanism_program_admission") is True
            ),
        )
        return jsonify(result), 201 if result.get("created") is True else 200

    @blueprint.get(prefix + "/claims/store")
    def experimental_claim_store(run_id: str):
        return jsonify(factory().experimental_claim_store(run_id))

    @blueprint.post(prefix + "/claims/admit")
    def admit_route_experimental_claims(run_id: str):
        payload = _payload()
        result = factory().admit_route_experimental_claims(
            run_id,
            **program_innovation_payload(payload),
            enable_experimental_claim_admission=(
                payload.get("enable_experimental_claim_admission") is True
            ),
        )
        return jsonify(result), 201 if result.get("created") is True else 200

    @blueprint.get(prefix + "/experience")
    def program_experience(run_id: str):
        return jsonify(factory().program_experience(run_id))

    @blueprint.post(prefix + "/experience/learn")
    def learn_program_experience(run_id: str):
        payload = _payload()
        return jsonify(factory().learn_program_experience(
            run_id,
            enable_program_experience_learning=(
                payload.get("enable_program_experience_learning") is True
            ),
        ))


def _payload() -> dict[str, Any]:
    value = request.get_json(force=False, silent=False)
    if not isinstance(value, dict):
        raise ValueError("request_body_must_be_an_object")
    return value


__all__ = ["register_program_innovation_routes"]
