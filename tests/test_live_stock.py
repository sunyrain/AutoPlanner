from __future__ import annotations

import json
from typing import Any, Mapping

from cascade_planner.interfaces.live_stock import build_pubchem_vendor_catalog


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
