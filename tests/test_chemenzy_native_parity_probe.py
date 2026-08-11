from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.run_chemenzy_native_parity_probe import compile_native_parity_report
from scripts.run_chemenzy_native_parity_probe import _stock_content_binding


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
        ],
        "raw_backend_metadata": {
            "cascade_expansion_trace": {
                "rows": [{"parent_mol": "CCO", "reactants": [reactant]}]
            }
        },
    }


def _stock_binding() -> dict:
    return {
        "schema_version": "chemenzy_native_parity_stock_binding.v1",
        "identity_complete": True,
        "stocks": [{"stock_name": "stock", "sha256": "d" * 64}],
        "content_sha256": "e" * 64,
    }


def test_stock_content_binding_hashes_the_selected_file(tmp_path: Path) -> None:
    stock = tmp_path / "stock.sqlite3"
    stock.write_bytes(b"complete stock contents")

    binding = _stock_content_binding(
        stock_names=["RetroStar-stock"],
        stock_paths={"RetroStar-stock": str(stock)},
    )

    assert binding["identity_complete"] is True
    assert binding["stocks"] == [
        {
            "stock_name": "RetroStar-stock",
            "path": str(stock.resolve()),
            "size_bytes": stock.stat().st_size,
            "sha256": hashlib.sha256(stock.read_bytes()).hexdigest(),
        }
    ]


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
        stock_content_binding=_stock_binding(),
    )

    assert report["raw_proposal_digest_equal"] is True
    assert report["route_fingerprint_rows_equal"] is True
    assert report["search_trace_digest_equal"] is True
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
        stock_content_binding=_stock_binding(),
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
        stock_content_binding=_stock_binding(),
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
        stock_content_binding=_stock_binding(),
    )

    assert report["nonempty_route_set_observed"] is True
    assert report["backend_failure_free"] is False
    assert report["parity_accepted"] is False


def test_native_parity_report_rejects_changed_search_trace() -> None:
    embedded = _route("CC")
    standalone = _route("CC")
    standalone["raw_backend_metadata"]["cascade_expansion_trace"]["rows"] = [
        {"parent_mol": "CCN", "reactants": ["CC"]}
    ]
    report = compile_native_parity_report(
        request={"target_smiles": "CCO", "chemenzy_seed": 0},
        stage={
            "provider_invocation_binding": {
                "runtime_binding": {
                    "model_content_identity_complete": True,
                }
            }
        },
        embedded_raw=embedded,
        standalone_raw=standalone,
        embedded_elapsed_s=1.0,
        standalone_elapsed_s=1.0,
        stock_content_binding=_stock_binding(),
    )

    assert report["raw_proposal_digest_equal"] is True
    assert report["search_trace_digest_equal"] is False
    assert report["parity_accepted"] is False
