from __future__ import annotations

from cascade_planner.providers import (
    ProviderContext,
    SnapshotStockProvider,
    StockOffer,
    stock_snapshot_sha256,
)


def _snapshot(
    *,
    supplier: str,
    catalog_number: str,
    smiles: str = "CCO",
    available: bool = True,
    price: float | None = None,
    lead_time_days: int | None = None,
) -> dict:
    return {
        "schema_version": "stock_offer_snapshot.v1",
        "supplier": supplier,
        "catalog_number": catalog_number,
        "smiles": smiles,
        "checked_at": "2026-07-10T00:00:00Z",
        "available": available,
        "price": price,
        "currency": "USD" if price is not None else "",
        "lead_time_days": lead_time_days,
    }


def _request_offer(snapshot: dict) -> dict:
    return {**snapshot, "snapshot_sha256": stock_snapshot_sha256(snapshot)}


def test_snapshot_stock_provider_preserves_supplier_alternatives_and_ranking() -> None:
    slow = _snapshot(
        supplier="slow",
        catalog_number="S-1",
        price=5.0,
        lead_time_days=20,
    )
    fast = _snapshot(
        supplier="fast",
        catalog_number="F-1",
        price=4.0,
        lead_time_days=2,
    )
    provider = SnapshotStockProvider(trusted_snapshots=[slow, fast])
    context = ProviderContext(run_id="run", case_id="case", target_smiles="CCO")
    result = provider.invoke(
        {
            "schema_version": "stock_lookup_request.v1",
            "smiles": "OCC",
            "offers": [
                _request_offer(slow),
                _request_offer(fast),
            ],
        },
        context=context,
    )

    assert result.accepted is True
    assert result.payload["boundary_type"] == "commercially_orderable"
    assert [row["supplier"] for row in result.payload["offers"]] == ["fast", "slow"]
    assert all(row["offer_id"].startswith("offer:") for row in result.payload["offers"])
    assert all(row["snapshot_verified"] is True for row in result.payload["offers"])


def test_offer_requires_snapshot_and_timestamp() -> None:
    try:
        StockOffer(
            supplier="supplier",
            catalog_number="sku",
            canonical_smiles="CCO",
            checked_at="",
            snapshot_sha256="",
            available=True,
        )
    except ValueError as exc:
        assert "identity" in str(exc) or "snapshot" in str(exc)
    else:
        raise AssertionError("invalid offer was accepted")


def test_mismatched_offer_cannot_close_stock_boundary() -> None:
    wrong = _snapshot(supplier="wrong", catalog_number="W-1", smiles="CCC")
    result = SnapshotStockProvider(trusted_snapshots=[wrong]).invoke(
        {
            "schema_version": "stock_lookup_request.v1",
            "smiles": "CCO",
            "offers": [
                _request_offer(wrong)
            ],
        },
        context=ProviderContext(run_id="run", case_id="case", target_smiles="CCO"),
    )

    assert result.accepted is False
    assert "offer:0:molecule_mismatch" in result.reasons


def test_untrusted_or_availability_tampered_offer_cannot_close_boundary() -> None:
    unavailable = _snapshot(
        supplier="supplier",
        catalog_number="U-1",
        available=False,
    )
    digest = stock_snapshot_sha256(unavailable)
    provider = SnapshotStockProvider(trusted_snapshots=[unavailable])

    tampered = provider.invoke(
        {
            "schema_version": "stock_lookup_request.v1",
            "smiles": "CCO",
            "offers": [{**unavailable, "available": True, "snapshot_sha256": digest}],
        },
        context=ProviderContext(run_id="run", case_id="case", target_smiles="CCO"),
    )
    assert tampered.accepted is False
    assert "offer:0:snapshot_content_mismatch" in tampered.reasons

    invented = _snapshot(supplier="attacker", catalog_number="FAKE-1", available=True)
    untrusted = provider.invoke(
        {
            "schema_version": "stock_lookup_request.v1",
            "smiles": "CCO",
            "offers": [_request_offer(invented)],
        },
        context=ProviderContext(run_id="run", case_id="case", target_smiles="CCO"),
    )
    assert untrusted.accepted is False
    assert "offer:0:untrusted_snapshot" in untrusted.reasons


def test_snapshot_hash_and_timestamp_are_strictly_validated() -> None:
    snapshot = _snapshot(supplier="supplier", catalog_number="S-1")
    provider = SnapshotStockProvider(trusted_snapshots=[snapshot])
    malformed = provider.invoke(
        {
            "schema_version": "stock_lookup_request.v1",
            "smiles": "CCO",
            "offers": [{**snapshot, "snapshot_sha256": "g" * 64}],
        },
        context=ProviderContext(run_id="run", case_id="case", target_smiles="CCO"),
    )
    assert malformed.accepted is False
    assert "offer:0:invalid_snapshot_sha256" in malformed.reasons

    bad_time = {**snapshot, "checked_at": "2026-07-10 00:00:00"}
    try:
        stock_snapshot_sha256(bad_time)
    except ValueError as exc:
        assert "timezone" in str(exc)
    else:
        raise AssertionError("timezone-free stock snapshot was accepted")
