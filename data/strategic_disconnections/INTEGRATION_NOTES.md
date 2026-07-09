# Integration Notes

The first version is advisory data. The safest integration path is incremental:

## 0. Agentic Blackboard Literature Strategy

When a user provides a new SMILES and no local target-specific data exists, use
the agentic blackboard workflow rather than forcing ordinary templates to solve
every advanced natural-product core.

Canonical workflow:

- parse the input SMILES and identify scaffold, ring systems, functional groups,
  and candidate strategic bonds;
- run ordinary ChemEnzy/template planning for normal functional-group and
  small-fragment steps;
- record unresolved advanced-frontier molecules and material-gate reasons;
- search literature for the family-level key construction and reviewed anchors;
- instantiate the literature strategy as three separate output types:
  `exact_fragment_retro`, `forward_surrogate`, and `route_anchor`;
- render the hybrid route and validate all non-empty SMILES/rxn SMILES.

Detailed runbook:

- `docs/AGENTIC_BLACKBOARD_MAINLINE_2026-06-24.md`

Important boundary: a `forward_surrogate` is model-facing planning material, not
an extracted experimental procedure. A `route_anchor` is not ordinary stock and
must not close a route unless a separate reviewed anchor policy says so.

## 1. Audit/Rerank Only

Use the records to add route audit tags and reranking features:

- bonus: route exposes a named strategic disconnection,
- bonus: route reaches a reviewed strategic anchor with explicit upstream status,
- penalty: product-like terminal with same scaffold and no strategic progress,
- penalty: repeated protecting-group growth without ring/fragment simplification.

For the deacetylbufotalin failure, the relevant hard signal is:

- target family: `bufadienolide_steroid`
- expected move: `bufadienolide_c17_pyrone_installation`
- observed bad terminal: same 5-ring bufadienolide analogue, larger than target

## 2. Proposal Source

After manual review, a strategic record can generate high-level candidate
actions. These should be marked as `strategic_disconnection` and carry:

- `strategy_id`
- `precursor_roles`
- `evidence`
- `requires_manual_structure_instantiation`
- `stock_policy`
- `candidate_kind`: one of `exact_fragment_retro`, `forward_surrogate`,
  `route_anchor`
- `confidence`
- `not_lab_procedure`: required for surrogate reactions

They should not be marked solved unless all terminals are accepted by the normal
stock checker or a separately reviewed anchor policy.

## 3. Anchor Policy

Strategic anchors are not normal stock. Promote an anchor only if:

- it is named and source-supported,
- it is materially upstream from the target,
- the route explains the transformation from anchor to target,
- product-like analogues are excluded by similarity/heavy/ring guards.

For bufadienolides, acceptable anchors are steroid chiral-pool materials such as
androstenedione/DHEA-like precursors, not arbitrary acetylated bufadienolide
analogues from vendor stock.

## 4. Bufotalin Lessons

The Bufotalin simulation established the expected behavior for hard natural
product cases:

- Step-level clarity matters. The terminal O-acetylation is an ordinary step;
  the strategic step is C17-2-pyrone installation.
- Deacetylbufotalin is an advanced frontier, not a solved upstream route.
- The exact C17-pyrone disconnection should be represented as a dummy-fragment
  `retro_rxn_smiles`.
- Stille/Suzuki C17-pyrone records can be useful `forward_surrogate` entries,
  but they must be labeled as surrogates unless extracted from the exact SI.
- Androstenedione-like steroid chiral-pool material is a reviewed anchor, not a
  single-step predecessor and not automatic stock.

Reusable output package from the simulation:

- `results/shared/bufotalin_hybrid_literature_20260603/`

## 5. Convert PPTX Schemes Later

Many statin schemes in `合作课题.pptx` are embedded OLE objects. The current
database uses their text/table context only. Before creating reaction templates:

- extract or redraw the chemical scheme,
- assign reactant/product SMILES,
- check atom mapping and stereochemistry,
- tag hazards and process constraints from the PPTX tables.
