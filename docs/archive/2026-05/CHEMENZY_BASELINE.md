# ChemEnzy Baseline

ChemEnzyRetroPlanner is treated as an external vendor baseline. Its source and
models stay outside the AutoPlanner package under `vendor/ChemEnzyRetroPlanner/`
and are ignored by git.

## Scope

First-stage reproduction is limited to external-baseline route search:

- RetroRollout*/MCTS-style multi-source organic + enzymatic route search
- normalized `RouteCandidate` JSON under `results/shared/chem_enzy_baseline/`
- smoke runs on simple direct targets and `data/benchmark_cascade_gold_smoke_v1.json`
- optional post-search condition/enzyme annotation with ChemEnzy metadata

Out of scope for this stage:

- Llama 70B agent
- web UI
- EasIFA/AlphaFold active-site annotation in routine smoke runs
- full paper-scale USPTO/PaRoutes/natural-product benchmarks

## Setup

```bash
bash scripts/setup_chem_enzy_vendor.sh
bash scripts/setup_chem_enzy_runtime.sh --scope core
```

The runtime script sources `/etc/network_turbo` when present and keeps the
packed ChemEnzy environment under `/root/autodl-tmp/chem_enzy_runtime` instead
of filling the root conda env directory. Use `--scope full` to add condition,
enzyme, rxn-filter, Parrot, and EasIFA metadata.

Observed full-scope disk footprint on this machine:

- `/root/autodl-tmp/chem_enzy_runtime`: about 16G
- `vendor/ChemEnzyRetroPlanner`: about 7.6G
- `results/shared/chem_enzy_baseline`: about 25M

## Smoke

Dry-run without importing the heavy vendor stack:

```bash
PYTHONPATH=. python scripts/run_chem_enzy_smoke.py --dry-run --limit 2
```

Real core-search run:

```bash
PYTHONPATH=. /root/autodl-tmp/chem_enzy_runtime/envs/retro_planner_env/bin/python \
  scripts/run_chem_enzy_smoke.py \
  --benchmark data/benchmark_cascade_gold_smoke_v1.json \
  --stock BioNav-stock \
  --iterations 10 \
  --max-depth 6 \
  --expansion-topk 20 \
  --output results/shared/chem_enzy_baseline/gold_smoke_bionav_core_reuse.json
```

Full condition/enzyme annotation run:

```bash
PYTHONPATH=. /root/autodl-tmp/chem_enzy_runtime/envs/retro_planner_env/bin/python \
  scripts/run_chem_enzy_smoke.py \
  --benchmark data/benchmark_cascade_gold_smoke_v1.json \
  --stock BioNav-stock \
  --iterations 10 \
  --max-depth 6 \
  --expansion-topk 20 \
  --enable-condition-prediction \
  --enable-enzyme-assignment \
  --output results/shared/chem_enzy_baseline/gold_smoke_bionav_full_attr.json
```

The smoke runner reuses one initialized ChemEnzy planner for all targets with
the same stock/model/search settings. Pass `--no-reuse-planner` to reproduce
the older per-target initialization behavior.

The adapter disables the agent, web UI, and EasIFA by default. Condition and
enzyme flags call ChemEnzy's post-search `predict_rxn_attributes()` path and
normalize the returned pandas JSON tables into `RouteCandidate.steps`.

## Comparison

```bash
PYTHONPATH=. python -m cascade_planner.eval.compare_chem_enzy_baseline \
  --benchmark data/benchmark_cascade_gold_smoke_v1.json \
  --chem-enzy results/shared/chem_enzy_baseline/gold_smoke_bionav_full_attr.json \
  --route-tree results/shared/chem_enzy_baseline/route_tree_gold_smoke.json \
  --output results/shared/chem_enzy_baseline/gold_smoke_full_attr_vs_route_tree_comparison.json
```

Same-target route-tree baseline:

```bash
PYTHONPATH=. python scripts/run_route_tree_gold_smoke.py \
  --check-stock \
  --output results/shared/chem_enzy_baseline/route_tree_gold_smoke.json
```

Metrics are baseline diagnostics only: solved rate, route count, enzymatic-step
presence, GT step overlap, condition recovery, average search time, and failure
categories. Do not use these smoke results for SOTA claims.

Observed 10-target smoke results:

| Run | Solved/plan rate | Avg routes | GT exact | GT partial | Condition | Enzyme presence | Avg search s |
|---|---:|---:|---:|---:|---:|---:|---:|
| ChemEnzy core reuse | 1.00 | 56.3 | 0.10 | 0.50 | 0.00 | 0.00 | 5.09 |
| ChemEnzy full attr | 1.00 | 56.3 | 0.10 | 0.50 | 1.00 | 1.00 | 5.06 |
| route_tree same-target | 0.30 | 1.1 | 0.30 | 0.30 | 0.30 | n/a | 3.25 |

For the full-attr ChemEnzy run, summed per-target elapsed time was about 77.9s,
including about 25.4s of post-search annotation. The core reuse run wall time
was about 60s for 10 targets on this machine.

## Absorptive Refactor Target

After the external baseline produces real route-search results, the main
AutoPlanner-Cascade path should absorb ChemEnzy-style multi-source expansion
instead of post-processing separate planners.

Planned `CascadeSearchState` fields:

- route graph and open retrosynthetic leaves
- step annotations: source model, reaction type, EC, catalyst, evidence
- condition envelopes per step and compatibility between adjacent steps
- stage partition: one-pot, telescoped, isolated, or unknown transfer
- cofactor/redox ledger with required, regenerated, and unclosed species
- evidence confidence for reaction, enzyme, condition, and literature support

Planned unified cost features:

- retrosynthesis proposal score
- stock/reachability score
- reaction plausibility
- enzyme/EC evidence confidence
- condition compatibility
- cofactor/redox closure
- stage complexity penalty
- uncertainty penalty

Current `route_tree` and `vnext` assets remain regression and ablation
baselines until this cascade-native state/cost contract is implemented and
validated on the v3 gold benchmark.

The first repo-owned contract skeleton now lives under
`cascade_planner/cascade_search/`:

- `state.py`: `CascadeSearchState`, `StepAnnotation`, `ConditionEnvelope`,
  and `CofactorLedger`
- `cost.py`: lower-is-better unified cost with stock, enzyme, condition,
  cofactor, stage-complexity, and uncertainty terms

This is intentionally not wired into the production planner yet; it is the
tested handoff point for the next absorption step.
