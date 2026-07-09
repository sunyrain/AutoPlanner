# Tetrahedron 2025 Bufotalin Route Audit

Source: `/root/autodl-tmp/AutoPlanner/1-s2.0-S0040402025001668-main.pdf`

DOI: `10.1016/j.tet.2025.134610`

## Status

The local PDF is readable and contains the main route sequence, schemes, conditions, yields, and characterization. ScienceDirect direct access is blocked from this environment, so the full-text source used here is the user-provided local PDF.

This source can support a bufotalin retrosynthesis handoff, but executable `source_detail_route_step` / `one_step_row` records still require RDKit-valid SMILES for each reported compound. The PDF images provide structures, but they are not machine-readable molfile/SDF data.

## Reported Route

The paper reports total synthesis of bufotalin from androstenedione `(11)`:

`11 -> 24 -> 25 -> 23 -> 26 -> 27 -> 28 -> 19 -> 20 -> 14 -> 22 -> 30 -> 31 -> 32 -> 33 -> bufotalin (1)`

Important late-stage sequence from Scheme 4:

| Step | Reported transformation | Conditions | Yield/status |
|---|---|---|---|
| `22 -> 30` | epoxidation of 16,17-double bond | `m-CPBA`, `Na2CO3`, `DCM`, `-78 C -> -20 C` | 70% |
| `30 -> 31` | Lewis-acid epoxide rearrangement to C16 ketone | `TMSOTf`, `2,6-lutidine`, `DCM`, `-78 C -> rt` | 68% |
| `31 -> 32` | stereospecific C16 ketone reduction | `NaBH4`, `MeOH/THF`, `0 C` | 90% |
| `32 -> 33` | C16 alcohol acetylation | `Ac2O`, `pyridine`, rt | 90% |
| `33 -> bufotalin (1)` | deprotection of silyl ethers | `HF-pyridine`, `THF`, rt, 160 h | 93% |

Key upstream route:

| Step | Reported transformation | Conditions | Yield/status |
|---|---|---|---|
| `11 -> 24` | C17 ketal protection | ethylene glycol, `p-TsOH`, rt | 93% |
| `24 -> 25` | C4/C5 hydrogenation | `Pd/C`, `H2`, 4-methylpyridine, rt | 92%, 1.5:1 dr |
| `25 -> 23` | C3 ketone reduction | `K-selectride`, `THF`, `-20 C -> -5 C` | 72%, 3:1 dr |
| `23 -> 26` | C16 bromination | pyridinium perbromide, `THF`, `0 C -> rt` | 87% |
| `26 -> 27` | elimination to D-ring olefin | `t-BuOK`, `DMSO`, `70 C` | 73% |
| `27 -> 28` | deketalization/enone formation | `p-TsOH`, acetone, `60 C` | 56% |
| `28 -> 19` | C3 TBS protection | `TBSCl`, imidazole, `DMF`, `70 C` | 74% |
| `19 -> 20` | Shibuya allylic oxidation, C14-beta-OH | `SeO2`, formic acid, dioxane/water, `125 C` | 52% |
| `20 -> 14` | hydrogenation, hydrazone formation, vinyl iodide formation | `Pd/C/H2`; hydrazine/Et3N; `I2/Et3N` | 73% over two steps |
| `14 -> 22` | Stille coupling with pyrone partner `(21)` | `Pd(PPh3)4`, `CuI`, `LiCl`, `DMSO/THF`, `60 C` | 62% |

## Relation To Current Harness Frontier

The current ChemEnzy rejected terminal is the unprotected deacetylated bufotalin-like triol:

`C[C@]12CC[C@H](O)C[C@H]1CC[C@@H]1[C@@H]2CC[C@]2(C)[C@@H](c3ccc(=O)oc3)[C@@H](O)C[C@]12O`

The Tetrahedron 2025 route does not use this exact unprotected triol as the immediate reported precursor. Its late-stage reported precursor to bufotalin is protected compound `33`; compound `32` is the protected C16 alcohol that is acetylated to `33`, followed by silyl deprotection to bufotalin.

Therefore the literature connection should be:

`androstenedione (11) -> ... -> 32 -> 33 -> bufotalin (1)`

not:

`current unprotected triol -> bufotalin (1)`

The one-step ChemEnzy rescue `unprotected triol + Ac2O -> bufotalin` is chemically plausible as an advisory hypothesis, but it is not the exact route supported by this PDF.

## Anchor Interpretation

The previous five anchors were route-expansion priors:

- Androstenedione
- DHEA
- Estrone
- Pregnenolone
- Progesterone

Only androstenedione is directly supported as the starting material in this PDF. The other four are broader steroid/chiral-pool semisynthesis anchors and should not consume priority over source-detail extraction for the reported bufotalin route.

## Required Next Step

To promote this PDF route into executable one-step rows:

1. Convert compound structures `11, 24, 25, 23, 26, 27, 28, 19, 20, 14, 22, 30, 31, 32, 33, 1` into RDKit-valid isomeric SMILES.
2. Build `source_detail_curator_records.v1` with one `source_detail_route_step.v1` per reported step.
3. Run `resolve_source_detail_extraction_pack` and `compile_downstream_consumables`.
4. Rerun route expansion with the source-detail one-step rows enabled.

No production KB promotion should happen until the SMILES are validated and the one-step rows pass harness validation.
