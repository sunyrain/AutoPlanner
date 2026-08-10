from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from typing import Any, Mapping

from cascade_planner.interfaces.live_stock import (
    FrozenBenchmarkStockIndex,
    FrozenInventorySnapshotBuilder,
    build_pubchem_vendor_catalog,
    load_versioned_inventory_snapshot,
)
from cascade_planner.interfaces.target_solver_stages import (
    _assert_stock_oracle_builder_binding,
    _selected_stock_audit_molecules,
)
from cascade_planner.application.unified_campaign_spec import (
    StockOracleReference,
    UnifiedCampaignSpec,
    stock_oracle_reference_from_builder,
)


def _requester(
    method: str,
    url: str,
    *,
    timeout_s: float,
    data: Mapping[str, Any] | None = None,
) -> tuple[int, bytes, Mapping[str, Any]]:
    del timeout_s
    if method == "POST":
        assert data == {"smiles": "CCO"}
        value: Mapping[str, Any] = {
            "PropertyTable": {"Properties": [{"CID": 702, "ConnectivitySMILES": "CCO"}]}
        }
    else:
        assert url.endswith("/702/JSON")
        value = {
            "SourceCategories": {
                "Categories": [
                    {
                        "Category": "Chemical Vendors",
                        "Sources": [
                            {
                                "SID": 10,
                                "SourceName": "Vendor B",
                                "RegistryID": "B-1",
                                "SourceRecordURL": "https://vendor.invalid/B-1",
                            },
                            {
                                "SID": 9,
                                "SourceName": "Vendor A",
                                "RegistryID": "A-1",
                                "SourceRecordURL": "https://vendor.invalid/A-1",
                            },
                        ],
                    }
                ]
            }
        }
    content = json.dumps(value, sort_keys=True).encode()
    return 200, content, value


def test_pubchem_vendor_adapter_freezes_bounded_benchmark_only_catalog() -> None:
    catalog = build_pubchem_vendor_catalog(
        ["CCO", "OCC"],
        requester=_requester,
        retrieved_at="2026-07-14T00:00:00Z",
        max_vendors_per_molecule=1,
    )

    assert catalog["queried_molecule_count"] == 1
    assert len(catalog["members"]) == 1
    member = catalog["members"][0]
    assert member["canonical_smiles"] == "CCO"
    assert member["vendor_count"] == 2
    assert member["vendors"][0]["supplier"] == "Vendor A"
    assert catalog["source"]["boundary"] == "benchmark_search"
    assert catalog["semantics"]["not_procurement_authority"] is True
    assert len(catalog["content_sha256"]) == 64


def test_versioned_inventory_loader_is_bounded_and_schema_typed(tmp_path: Path) -> None:
    path = tmp_path / "inventory.json"
    expected = {
        "schema_version": "versioned_inventory_snapshot.v1",
        "adapter_version": "tests.inventory.v1",
        "inventory_version": "snapshot-1",
        "retrieved_at": "2026-07-14T00:00:00Z",
        "offers": [],
    }
    path.write_text(json.dumps(expected), encoding="utf-8")

    assert load_versioned_inventory_snapshot(path) == expected

    builder = FrozenInventorySnapshotBuilder(path)
    binding = builder.stock_oracle_binding
    assert builder(["CCO"]) == expected
    assert binding["kind"] == "frozen_inventory_snapshot"
    assert binding["snapshot_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(binding["content_sha256"]) == 64


def test_frozen_benchmark_stock_index_is_hashed_read_only_membership(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stock.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE stock (canonical_smiles TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("schema_version", "frozen_benchmark_stock_index.v1"),
                ("catalog_name", "unit-test-stock"),
                ("source_sha256", "a" * 64),
                ("member_count", "1"),
                ("complete", "true"),
                ("created_at", "2026-07-23T00:00:00Z"),
                ("rdkit_version", "test"),
            ],
        )
        connection.execute(
            "INSERT INTO stock(canonical_smiles) VALUES (?)",
            ("CCO",),
        )
    expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    builder = FrozenBenchmarkStockIndex(
        path,
        expected_sha256=expected_sha256,
    )
    catalog = builder(["OCC", "CCN"])

    assert catalog["catalog_name"] == "unit-test-stock"
    assert catalog["source"]["index_sha256"] == expected_sha256
    assert catalog["source"]["immutable_content_addressed"] is True
    assert [row["canonical_smiles"] for row in catalog["members"]] == ["CCO"]
    assert catalog["members"][0]["membership_verified"] is True
    assert len(catalog["members"][0]["membership_proof_sha256"]) == 64
    assert [row["canonical_smiles"] for row in catalog["misses"]] == ["CCN"]
    assert catalog["semantics"]["not_a_reaction_or_route_provider"] is True


def test_selected_stock_leaves_above_one_batch_are_not_globally_rejected() -> None:
    leaf_ids = [f"molecule:{index:02d}" for index in range(25)]
    graph = {
        "target_molecule_id": "molecule:target",
        "molecules": {
            molecule_id: {"canonical_smiles": f"C{'C' * index}"}
            for index, molecule_id in enumerate(leaf_ids, start=1)
        },
        "edges": {},
        "route_families": {
            "route-family:test": {
                "selected": True,
                "leaf_molecule_ids": leaf_ids,
                "edge_ids": [],
            }
        },
    }

    selection = _selected_stock_audit_molecules(graph, max_molecules=24)

    assert selection["limit_exceeded"] is False
    assert selection["batching_required"] is True
    assert selection["audit_batch_limit"] == 24
    assert selection["leaf_molecule_ids"] == leaf_ids
    assert selection["stock_candidate_molecule_ids"] == leaf_ids


def test_stock_stage_rejects_a_runtime_resolver_different_from_run_spec() -> None:
    def configured_resolver(_smiles, **_kwargs):
        return {}

    def substituted_resolver(_smiles, **_kwargs):
        return {"substituted": True}

    campaign_spec = UnifiedCampaignSpec(
        target_smiles="CCO",
        stock_oracle=stock_oracle_reference_from_builder(
            configured_resolver,
            boundary="benchmark_search",
        ),
    )
    service = SimpleNamespace(
        kernel=SimpleNamespace(spec=SimpleNamespace(campaign_spec=campaign_spec))
    )

    _assert_stock_oracle_builder_binding(
        service,
        builder=configured_resolver,
        boundary="benchmark_search",
    )
    try:
        _assert_stock_oracle_builder_binding(
            service,
            builder=substituted_resolver,
            boundary="benchmark_search",
        )
    except ValueError as exc:
        assert str(exc) == "stock_oracle_runtime_binding_mismatch"
    else:
        raise AssertionError("a substituted stock resolver was accepted")

    legacy = SimpleNamespace(
        kernel=SimpleNamespace(
            spec=SimpleNamespace(
                campaign_spec=UnifiedCampaignSpec(
                    target_smiles="CCO",
                    stock_oracle=StockOracleReference.compatibility_unbound(
                        boundary="benchmark_search"
                    ),
                )
            )
        )
    )
    _assert_stock_oracle_builder_binding(
        legacy,
        builder=substituted_resolver,
        boundary="benchmark_search",
    )
