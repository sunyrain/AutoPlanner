from cascade_planner.cascadeboard.vendor_stock import (
    VendorStockChecker,
    build_vendor_stock_index,
    stock_query_variants,
)


def test_vendor_stock_index_matches_neutralized_variant(tmp_path):
    csv_path = tmp_path / "stock.csv"
    sqlite_path = tmp_path / "stock.sqlite"
    csv_path.write_text("smiles\nCC(=O)O\n", encoding="utf-8")

    report = build_vendor_stock_index(csv_path=csv_path, sqlite_path=sqlite_path)
    checker = VendorStockChecker(sqlite_path)

    assert report["row_count"] == 1
    assert checker("CC(=O)[O-]") is True
    assert checker("CCN") is False


def test_stock_query_variants_include_canonical_and_neutralized_forms():
    variants = stock_query_variants("CC(=O)[O-]")

    assert "CC(=O)[O-]" in variants
    assert "CC(=O)O" in variants
