# Next Iteration Plan - 2026-05-12

## Objective

Do not jump directly to a monolithic final value model.

The next iteration should improve the controller in the order exposed by the
coverage audit:

```text
candidate coverage -> source scheduling -> open-leaf policy -> cascade priority
-> state/action value -> process repair
```

The current bottleneck is still search access and stock-aware control, not a
lack of another route-level scorer.

## Phase 1: Source Scheduling

### Problem

The full100 audit still contains:

```text
intermediate_not_reached
queried_budget_too_small
selector_missed_candidate
generated_ranked_out
stock_dead_end
```

These are controller failures. A final value model cannot fix a source that was
never queried or a leaf that was never expanded.

### Data

Use production route-tree traces and v4 traces, excluding full100 from training
when reporting final metrics.

Inputs:

- parent search state
- open leaf features
- current unresolved failure labels
- route depth and remaining budget
- existing route domain and step types
- source call history
- candidate yield by source
- stock status and stock frontier features

Labels:

- which source recovered a GT-like candidate
- which source produced useful stock-closing candidates
- whether the state needed more budget or a different source
- whether the leaf was a dead end under current budget

### Model

Train `CascadeSourcePolicy`.

Output:

```text
source family probabilities
budget per source
retry / relax / switch-leaf decision
```

Source families:

```text
ChemEnzy native core
organic proposal source
enzymatic proposal source
template/retrieval source
cascade-aware retrieval/provider
stock-closing fallback
```

### Acceptance

On full100:

| Metric | Gate |
| --- | ---: |
| `plan_rate` | >= 0.95 |
| `candidate_gt_reactant_in_pool` | >= 0.58 |
| `candidate_exact_reaction_in_pool` | >= 0.40 |
| `strict_stock_solve_any` | first target >= 0.45, later >= 0.60 |
| `avg_time_per_target_s` | <= 30.0 effect-first gate; 16.0 remains a speed-optimized reference |

## Phase 2: Stock-Aware Open-Leaf Policy

### Problem

Coverage improved but strict stock solve regressed. The policy must learn when
to deepen a chemically relevant intermediate and when to close a purchasable
terminal branch.

### Data

Use traces from:

- `results/shared/coverage_fix_20260511/coverage_fix_full100_trace.jsonl`
- v4 training traces after full100 exclusion
- stock closure reports and candidate miss audits

### Label

Replace pure expansion imitation with utility:

```text
leaf_utility =
  route outcome
  + candidate yield
  + GT/intermediate proximity
  + adjacent pair formation
  + stock closure progress
  - dead-end risk
  - repeated low-yield source calls
```

### Acceptance

The policy should improve strict stock solve without erasing the coverage gains.
If candidate coverage drops, the policy is over-closing and should not be
promoted.

## Phase 3: Cascade Priority As Child-State Delta

Pairwise and fragment cascade scorers should be used only when a newly added
reaction forms a valid adjacent reaction pair.

Correct placement:

```text
apply action -> child state
find new adjacent reaction pair
if pair exists: add cascade delta to child-state priority
if no pair exists: no cascade reward, no default constant
```

Do not treat pairwise cascade scoring as a single-step action score.

## Phase 4: State/Action Value Model

Train this only after source scheduling and stock-aware leaf expansion are
stable.

Target:

```text
Q(state, action) = whether this action moves the partial program toward a
feasible cascade route under budget
```

Labels:

- final route success
- stock closure
- cascade fragment quality
- condition conflict count
- cofactor debt
- stage complexity
- evidence sufficiency
- remaining depth

Preferred loss:

- pairwise/listwise ranking within each expanded state
- auxiliary typed-failure heads
- rollout-value regression

Do not train it as:

```text
candidate == GT step
record is gold/silver
```

## Phase 5: Process Repair Actions

After the controller can reach useful routes, add repair actions:

- cofactor regeneration
- stage split
- buffer exchange
- solvent switch
- isolation/workup
- immobilized-enzyme compartment
- enzyme/evidence retrieval

These actions should be triggered by typed failures, not only reported after
search.

## Reporting Rule

Every result report must separate:

1. raw v4 data size
2. trace-candidate size
3. traced target count
4. model supervision count
5. final evaluation target count

Do not infer training-data sufficiency from a small trace pack.
