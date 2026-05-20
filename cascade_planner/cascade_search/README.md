# `cascade_planner.cascade_search`

This package is the research interface for cascade-aware search. It is not the
default Web route generator; the Web path currently uses ChemEnzy native search
through `scripts/run_chem_enzy_plan_for_web.py`.

## Current Role

`cascade_search` is retained for:

- cascade-state/search contracts
- action-value hooks
- v4 product-value and product-audit features
- subgoal proposal experiments
- search-time rerank/proposal ablations

## Important Dependencies

- `action_value.py` imports helpers from `cascade_planner.eval.train_cascade_action_value`.
- `proposals.py` imports helpers from `cascade_planner.eval.train_cascade_subgoal_scorer`.
- `v4_product_value.py` consumes product-audit features produced by
  `cascade_planner.eval.product_route_feasibility_audit`.

Do not archive these eval scripts while this package remains active.

## Promotion Rule

A `cascade_search` feature becomes runtime-promoted only after:

1. it has a named report artifact,
2. it beats the ChemEnzy-native or ChemEnzy-plus-audit baseline on the relevant
   split,
3. product-audit and stock-closure guardrails remain acceptable,
4. focused tests pass.

