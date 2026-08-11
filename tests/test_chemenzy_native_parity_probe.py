from __future__ import annotations

from scripts.run_chemenzy_native_parity_probe import compile_native_parity_report


def _route(reactant: str) -> dict:
    return {
        "routes": [
            {
                "steps": [
                    {
                        "product_smiles": "CCO",
                        "reactant_smiles": [reactant],
                    }
                ]
            }
        ]
    }


def test_native_parity_report_accepts_equal_proposals_not_receipts() -> None:
    report = compile_native_parity_report(
        request={"target_smiles": "CCO", "chemenzy_seed": 0},
        stage={
            "request_sha256": "a" * 64,
            "replay_key_sha256": "b" * 64,
            "provider_invocation_binding": {
                "runtime_binding": {
                    "model_content_binding_sha256": "c" * 64,
                    "model_content_identity_complete": True,
                }
            },
        },
        embedded_raw={**_route("CC"), "elapsed_s": 1.0},
        standalone_raw={**_route("CC"), "elapsed_s": 9.0},
        embedded_elapsed_s=1.0,
        standalone_elapsed_s=9.0,
    )

    assert report["raw_proposal_digest_equal"] is True
    assert report["route_fingerprint_rows_equal"] is True
    assert report["embedded"]["raw_result_sha256"] != report["standalone"]["raw_result_sha256"]
    assert report["parity_accepted"] is True


def test_native_parity_report_rejects_changed_proposal() -> None:
    report = compile_native_parity_report(
        request={"target_smiles": "CCO", "chemenzy_seed": 0},
        stage={
            "provider_invocation_binding": {
                "runtime_binding": {
                    "model_content_identity_complete": True,
                }
            }
        },
        embedded_raw=_route("CC"),
        standalone_raw=_route("C"),
        embedded_elapsed_s=1.0,
        standalone_elapsed_s=1.0,
    )

    assert report["raw_proposal_digest_equal"] is False
    assert report["parity_accepted"] is False


def test_native_parity_report_rejects_vacuous_empty_proposals() -> None:
    report = compile_native_parity_report(
        request={"target_smiles": "CCO", "chemenzy_seed": 0},
        stage={
            "provider_invocation_binding": {
                "runtime_binding": {
                    "model_content_identity_complete": True,
                }
            }
        },
        embedded_raw={"routes": []},
        standalone_raw={"routes": []},
        embedded_elapsed_s=1.0,
        standalone_elapsed_s=1.0,
    )

    assert report["raw_proposal_digest_equal"] is True
    assert report["route_fingerprint_rows_equal"] is True
    assert report["nonempty_route_set_observed"] is False
    assert report["parity_accepted"] is False


def test_native_parity_report_rejects_backend_failure_even_with_routes() -> None:
    report = compile_native_parity_report(
        request={"target_smiles": "CCO", "chemenzy_seed": 0},
        stage={
            "provider_invocation_binding": {
                "runtime_binding": {
                    "model_content_identity_complete": True,
                }
            }
        },
        embedded_raw={**_route("CC"), "backend_failures": [{"message": "boom"}]},
        standalone_raw=_route("CC"),
        embedded_elapsed_s=1.0,
        standalone_elapsed_s=1.0,
    )

    assert report["nonempty_route_set_observed"] is True
    assert report["backend_failure_free"] is False
    assert report["parity_accepted"] is False
