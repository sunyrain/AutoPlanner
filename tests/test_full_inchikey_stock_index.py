from pathlib import Path

from rdkit import Chem

from cascade_planner.baselines.chem_enzy_adapter import _SqliteStockMembership
from cascade_planner.interfaces.live_stock import FrozenBenchmarkStockIndex
from scripts.build_full_inchikey_stock_index import build_index


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
