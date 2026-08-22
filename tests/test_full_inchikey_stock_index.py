from pathlib import Path
import sqlite3

import pytest

from rdkit import Chem

from cascade_planner.baselines.chem_enzy_adapter import _SqliteStockMembership
from cascade_planner.interfaces.live_stock import FrozenBenchmarkStockIndex
from scripts.build_full_inchikey_stock_index import (
    build_composite_index,
    build_inchikey_composite_index,
    build_index,
)
from scripts.run_chem_enzy_plan_for_web import _smiles_in_stock_file


def test_full_inchikey_index_is_shared_by_host_and_chemenzy(tmp_path: Path) -> None:
    ethanol_key = Chem.MolToInchiKey(Chem.MolFromSmiles("CCO"))
    water_key = Chem.MolToInchiKey(Chem.MolFromSmiles("O"))
    source = tmp_path / "stock.txt"
    source.write_text(f"{ethanol_key}\n{water_key}\n", encoding="utf-8")
    index = tmp_path / "stock.sqlite3"
    built = build_index(
        [source],
        index,
        column="",
        catalog_name="test ZINC+eMolecules",
        expected_count=2,
        batch_size=1,
    )

    host = FrozenBenchmarkStockIndex(
        index,
        expected_sha256=built["index_sha256"],
    )
    result = host(["CCO", "CC"], max_molecules=2)
    assert [row["canonical_smiles"] for row in result["members"]] == ["CCO"]
    provider = _SqliteStockMembership(index)
    assert "CCO" in provider
    assert "CC" not in provider
    assert _smiles_in_stock_file("CCO", index) is True
    assert _smiles_in_stock_file("CC", index) is False


def test_composite_full_inchikey_index_unions_hdf_and_smiles_sqlite(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("tables")
    ethanol_key = Chem.MolToInchiKey(Chem.MolFromSmiles("CCO"))
    water_key = Chem.MolToInchiKey(Chem.MolFromSmiles("O"))
    methane_key = Chem.MolToInchiKey(Chem.MolFromSmiles("C"))
    hdf = tmp_path / "zinc.hdf5"
    pd.DataFrame({"inchi_key": [ethanol_key, water_key]}).to_hdf(
        hdf, key="table", mode="w"
    )
    smiles = tmp_path / "emolecules.sqlite3"
    connection = sqlite3.connect(smiles)
    connection.execute(
        "CREATE TABLE stock (canonical_smiles TEXT PRIMARY KEY) WITHOUT ROWID"
    )
    connection.executemany(
        "INSERT INTO stock(canonical_smiles) VALUES (?)",
        [("C",), ("CCO",)],
    )
    connection.commit()
    connection.close()
    output = tmp_path / "combined.sqlite3"

    built = build_composite_index(
        inchikey_hdf=hdf,
        smiles_sqlite=smiles,
        output=output,
        hdf_key="table",
        hdf_column="inchi_key",
        sqlite_table="stock",
        sqlite_column="canonical_smiles",
        catalog_name="test ZINC+eMolecules",
        expected_count=3,
        batch_size=1,
        workers=1,
        resume=False,
    )

    assert built["member_count"] == 3
    connection = sqlite3.connect(output)
    keys = {
        row[0] for row in connection.execute("SELECT full_inchikey FROM stock")
    }
    connection.close()
    assert keys == {ethanol_key, water_key, methane_key}


def test_composite_full_inchikey_index_uses_provider_csv_keys_directly(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("tables")
    ethanol_key = Chem.MolToInchiKey(Chem.MolFromSmiles("CCO"))
    water_key = Chem.MolToInchiKey(Chem.MolFromSmiles("O"))
    methane_key = Chem.MolToInchiKey(Chem.MolFromSmiles("C"))
    hdf = tmp_path / "zinc.hdf5"
    pd.DataFrame({"inchi_key": [ethanol_key, water_key]}).to_hdf(
        hdf, key="table", mode="w"
    )
    provider_csv = tmp_path / "emolecules.csv"
    provider_csv.write_text(
        "mol,inchi_key\n"
        f"C,{methane_key}\n"
        f"CCO,{ethanol_key}\n"
        "invalid,not-an-inchikey\n",
        encoding="utf-8",
    )
    output = tmp_path / "combined.sqlite3"

    built = build_inchikey_composite_index(
        inchikey_hdf=hdf,
        inchikey_csv=provider_csv,
        output=output,
        hdf_key="table",
        hdf_column="inchi_key",
        csv_column="inchi_key",
        catalog_name="test ZINC+eMolecules",
        expected_count=3,
        batch_size=1,
        resume=False,
    )

    assert built["member_count"] == 3
    connection = sqlite3.connect(output)
    keys = {
        row[0] for row in connection.execute("SELECT full_inchikey FROM stock")
    }
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    connection.close()
    assert keys == {ethanol_key, water_key, methane_key}
    assert metadata["inchikey_csv_processed_rows"] == "3"
    assert metadata["inchikey_csv_valid_rows"] == "2"
