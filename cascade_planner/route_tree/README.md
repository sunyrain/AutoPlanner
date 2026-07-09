# Route Tree Planner

This package implements the unified route-level planner. One-step engines
remain proposal tools only: they return candidate actions for one product
molecule, and the route-tree controller decides which leaf and which action to
expand.

The active controller contract is:

- node policy: choose the open leaf to expand
- action policy: choose the candidate action
- calibrated value: search backup is disabled unless checkpoint metadata marks
  the value calibration as frozen
- bottleneck head: classify failed route states
- budget head: recommend source allocation for proposal budgets
- verifier: reject type, EC, condition, cycle, and route-progress violations

## Runtime

Use the route-tree search mode from live benchmark:

```bash
AUTOPLANNER_ENABLE_ROUTE_TREE_PLANNER=1 \
PYTHONPATH=. python cascade_planner/cascadeboard/live_benchmark.py \
  --search-mode route_tree
```

The default controller checkpoint is
`results/shared/controller_v2_20260512/fullrun/train_v8/open_leaf_policy/stock_aware_search_policy.pt`.
Override it with:

```bash
AUTOPLANNER_ROUTE_TREE_POLICY=/path/to/search_policy.pt
```

The proposal layer can load trained source allocation and source-specific
candidate rerankers while still returning candidates only:

```bash
AUTOPLANNER_SOURCE_GATE=results/shared/proposal_rankers/full_20260508/source_gate.pt \
AUTOPLANNER_ENABLE_PROPOSAL_RANKERS=1 \
AUTOPLANNER_PROPOSAL_RANKER_DIR=results/shared/proposal_rankers/full_20260508 \
AUTOPLANNER_ENABLE_V3_RETRIEVAL_PROPOSALS=1
```

If the env flag is not enabled or the checkpoint cannot be loaded, the planner
still runs with deterministic heuristic node/action scores. That fallback is
for smoke tests and data bootstrapping, not the target production setting.
Uncalibrated checkpoints may score node/action choices, but their value logits
are not used for route-value backup.

## Trace Collection

Collect expansion-level traces:

```bash
PYTHONPATH=. python cascade_planner/eval/collect_route_tree_traces.py \
  --bench data/benchmark_v2_100.json \
  --output results/shared/route_tree_traces/traces.jsonl \
  --check-stock \
  --use-route-model
```

Each JSONL row stores the target context plus one search expansion event:

- full partial route state
- open leaves
- expanded leaf
- normalized candidate actions
- selected action key
- model scores
- final search outcome

## Training

Train the unified policy/value/bottleneck controller directly from traces:

```bash
PYTHONPATH=. python cascade_planner/eval/train_vnext_from_pack.py \
  --task policy \
  --route-tree-traces results/shared/route_tree_traces/traces.jsonl \
  --output-dir results/shared/vnext
```

This produces a `search_policy.pt` checkpoint that `RouteTreeRuntime` can load
directly. The policy loss supervises selected actions, and the route heads
supervise solved, stock-closed, progressive, compatibility, and bottleneck
labels.

## New Data Requirements

For each target, prefer collecting both successful and failed route-tree search
traces. The most useful rows include diverse candidate pools per expansion,
stock availability, final route metrics, and failure outcome labels. Balanced
coverage across chemistry-only, enzyme-only, and mixed chemoenzymatic routes is
more valuable than only adding easy solved cases.
