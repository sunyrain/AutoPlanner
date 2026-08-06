# `cascade_planner.cascade_search`

This package is the research interface for cascade-aware search. It is not the
default Web route generator; the Web path currently uses ChemEnzy native search
through `scripts/run_chem_enzy_plan_for_web.py`.

## Current Role

`cascade_search` is retained for:

- cascade-state/search contracts
- rule / learned verifier value hooks
- product-audit features
- subgoal proposal experiments
- provider-neutral model injection and proposal ablations

## Important Dependencies

- `action_value.py` owns only the current evidence-supported subgoal hint
  scorer. Historical action-value checkpoints, transition-value adapters, and
  adjacent-step pair scorers live under
  `cascade_planner.legacy.cascade_search_runtime`.
- `value.py` owns the current heuristic and verifier-augmented value models.
  The historical `LearnedCascadeValueModel` checkpoint adapter lives under
  `cascade_planner.legacy.cascade_search_runtime.value`.
  The obsolete `RuleCascadeValueModel` experiment alias is deleted.
- `subgoal_evidence_contract.py` owns the evidence, fragment, fingerprint, and
  serialized feature contract used by runtime subgoal hints and archived
  trainers.
- Historical V4 product-value route encoders and checkpoint loaders live under
  `cascade_planner.legacy.cascade_search_runtime.v4_product_value`; this package
  no longer exports them.

## Promotion Rule

A `cascade_search` feature becomes runtime-promoted only after:

1. it has a named report artifact,
2. it beats the ChemEnzy-native or ChemEnzy-plus-audit baseline on the relevant
   split,
3. product-audit and stock-closure guardrails remain acceptable,
4. focused tests pass.

## Verifier-First Boundary

Verifier-first work is now the main research direction. Rule verifier signals
may be used as conservative route metrics or hard gates. Learned verifier
signals must remain behind an explicit flag until a held-out benchmark shows
that reranking improves route quality rather than only changing the order. The
current learned verifier artifact includes a calibrated feasible threshold; use
the calibrated conservative policy by default and reserve raw learned-score
sorting for ablation only.

When a route lacks an explicit `stage_partition`, treat it as a sequential
stepwise synthesis. Only use a single-stage / one-pot assumption when the input
route explicitly says so or an experiment flag requests it.
