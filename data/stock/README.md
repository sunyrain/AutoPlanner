# Pinned benchmark stock catalogs

## Basic chemical supplement

`basic_chemicals.csv` is an explicit, manually selected list of 50 common
chemical feedstocks and reagents. It includes water, hydrogen, oxygen,
ethylene, ethanol, methanol, acetaldehyde, hydrogen peroxide, common acids,
bases and salts. Each SMILES/full-InChIKey pair was checked against PubChem
on 2026-09-17; the CSV links to the corresponding identity record.

This list defines an assumed starting-material boundary for synthesis planning.
It is not a download of all PubChem compounds or a live supplier inventory.
PubChem identity alone never admits an arbitrary compound to the stock: only
the explicitly listed 50 identities are added. No atom-count, molecular-weight,
or structural-complexity bypass is used. Radical, isotope, salt and stereochemical
variants require their own exact identity match.

Supplier-category observations collected alongside the identities are in
`results/discussion/basic-chemicals-20260917/source-observations.json`. They
are sourcing leads, not verified product offers: category links can include
related substances and formulations. These observations do not automatically
select compounds for this list. The sulfuric-acid category request timed out;
its identity was verified and its inclusion is an explicit commodity assumption.

The supplement does not specify concentration, purity, hydration or delivery
form. For example, a formaldehyde or hydrogen chloride hit does not establish
that an anhydrous feed or a particular solution is available. Reaction conditions
must still specify a compatible form; a stock hit supplies no reaction proof.

The first supplement was a **new** union of the original 17,422,831-member ZINC
index and this list, named `ZINC + basic chemicals`. There are 49 new identities
and 17,422,880 total members. Original ZINC and previous experiment artifacts
are unchanged. eMolecules is not part of this default. Existing runs that
explicitly bind the old ZINC index remain reproducible with that index.

Build from the existing local ZINC index:

```powershell
python scripts/build_full_inchikey_stock_index.py `
  --base-index data_external/synthatlas/zinc_17_04_20_full_inchikey.sqlite3 `
  --input data/stock/basic_chemicals.csv --column full_inchikey `
  --catalog-name "ZINC + basic chemicals" `
  --output data_external/synthatlas/zinc_basic_chemicals_20260917_full_inchikey.sqlite3
```

The command refuses to overwrite an existing index. The build reports the hash
to bind through the existing `benchmark_stock_index`,
`benchmark_stock_index_sha256` and `benchmark_stock_name` options. A rebuilt
SQLite file may have different bytes across SQLite versions; use its reported
hash. No new runtime field or online lookup is needed.

## Common reagents supplement (current default)

`common_reagents_20260917.csv` adds 44 explicitly selected common reagents,
solvents, salts and elements. Every SMILES/full-InChIKey pair was checked
against PubChem on 2026-09-17. Of these, 28 already occur in the first union;
16 are new, giving **17,422,896** exact identities. New members include
methanesulfonyl chloride, methanesulfonic acid, tosyl chloride, sulfuryl chloride,
three chlorosilanes and common inorganic reagents. Identity responses are
saved in `results/discussion/statin-serial-20260917/common-reagents-identity-check.json`.

These are explicit planning-stock assumptions, not live vendor offers. PubChem
was used to check identity, not to infer buyability. Direct TCI product-page
requests returned HTTP 403, so no product, purity, pack size or formulation
verification is claimed. No eMolecules or benchmark reference-route leaves
were merged. Stock queries remain local; no worker search tools were enabled.

The new default index is
`data_external/synthatlas/zinc_basic_reagents_20260917_full_inchikey.sqlite3`,
SHA-256 `2b88171fe43726d7d47539a794a7ef8e008a28d1a2a85ed81f6c30d529ebf272`.
Both previous indexes and the eight experiment configurations remain unchanged.

```powershell
python scripts/build_full_inchikey_stock_index.py `
  --base-index data_external/synthatlas/zinc_basic_chemicals_20260917_full_inchikey.sqlite3 `
  --input data/stock/common_reagents_20260917.csv --column full_inchikey `
  --catalog-name "ZINC + basic chemicals" `
  --output data_external/synthatlas/zinc_basic_reagents_20260917_full_inchikey.sqlite3
```

### What other planners use

Primary project sources checked on 2026-09-17:

| System | Stock boundary in the checked implementation | Implication here |
| --- | --- | --- |
| [AiZynthFinder](https://github.com/MolecularAI/aizynthfinder/blob/master/aizynthfinder/tools/download_public_data.py) | Public downloader supplies `zinc_stock.hdf5`; [custom queries](https://github.com/MolecularAI/aizynthfinder/blob/master/docs/stocks.rst) can use other databases | Retain ZINC and union explicit supplements |
| [ASKCOS](https://github.com/ASKCOS/ASKCOS/blob/master/makeit/utilities/buyable/pricer.py) | Local file or MongoDB buyables, with price and source filtering | A reagent/building-block catalog is useful; this code alone does not grant access to its vendor data |
| [Syntheseus](https://github.com/microsoft/syntheseus/blob/main/syntheseus/search/mol_inventory.py) | Explicit purchasable SMILES inventory; `SmilesListInventory` is the common form | Keep material identities explicit and configurable |
| [PaRoutes](https://github.com/MolecularAI/PaRoutes/blob/main/README.md) | n1/n5 stock lists define the benchmark stopping criterion | Keep as separate benchmark stock; do not label it a supplier inventory |
| [Retro*](https://github.com/binghong-ml/retro_star/blob/master/README.md) | Separately distributed building-block molecules | Stock is an input dataset, not all reaction-database reactants |

## PaRoutes n1

`paroutes_n1.csv` is a repository-contained copy of the PaRoutes n1 building
block catalog distributed with ChemEnzyRetroPlanner commit
`8a59127c736a65e1f21f7ec8f4dafbcbd7717f37`.

- Source: <https://github.com/wangxr0526/ChemEnzyRetroPlanner>
- SHA-256: `d7593e5ba19a2b780236bf8eab3629cac75a20611804fbb32ca5d1b95ef2145f`
- License: MIT; see `LICENSE-CHEMENZY.txt`.
- Authority: reproducible benchmark membership only.

This catalog is intentionally not a supplier catalog. A hit does not establish
commercial availability, price, lead time, stock quantity, or procurement
readiness. Those claims require a separately configured, host-replayed trusted
stock snapshot.
