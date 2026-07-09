# Guarded CCTS Decision - 2026-05-19

## Decision

Do not promote the adjacent-step CCTS pair reward as the main model contribution.

Keep it as a safe diagnostic / tie-break feature, but stop spending primary
effort on tuning this reward alone.

## Evidence

### 1. Adjacent-step pair scorer learns the synthetic compatibility task

Current guarded pipeline artifact:

```text
results/shared/model_strengthening_20260519_guarded_v4_pipeline/models/cascade_pair_scorer.pt
results/shared/model_strengthening_20260519_guarded_v4_pipeline/reports/cascade_pair_scorer.md
```

Result:

```text
test pairwise_group_accuracy: 0.998688
rule baseline: 0.984252
```

Interpretation: the model can learn local adjacent-step compatibility labels.

### 2. Runtime ChemEnzy hard-negative ranking has only modest lift

Report:

```text
results/shared/model_strengthening_20260519_transition_hardneg/transition_hardneg_summary.md
```

Result:

```text
ChemEnzy block-supported MRR: 0.391640
selected residual blend MRR:  0.423058
delta:                       +0.031418
```

Caveat:

```text
nonlearned retrieval blend block MRR: 0.422418
```

Interpretation: v4 evidence helps, but the learned component is not clearly
stronger than retrieval similarity.

### 3. Live guarded search is safe but weak

Report:

```text
results/shared/model_strengthening_20260519_guarded_v4_pipeline/smoke/guarded_pair_limit20_weight_tie_sweep.md
```

Result:

```text
baseline                         solved 0.9  stock 1.0  exact 0.05  GT 0.35
guarded eps0   weight 0.005      solved 0.9  stock 1.0  exact 0.05  GT 0.35  changed 1/20
guarded eps.03 weight 0.005      solved 0.9  stock 1.0  exact 0.05  GT 0.35  changed 1/20
guarded eps.03 weight 0.05       solved 0.9  stock 1.0  exact 0.05  GT 0.35  changed 1/20
```

Interpretation: guarded CCTS enters live search and preserves guardrails, but it
does not improve aggregate quality in the current policy.

### 4. Block coherence classifiers do not transfer cleanly to route recovery

Synthetic/context block scorer:

```text
results/shared/cascadebench_strict_20260516/block_hard_scorer_runtime_evidence/cascade_block_coherence_report.md
```

Best classifier-level result:

```text
structure_plus_context AUC: 0.942667
full AUC:               0.917131
```

Route-level heldout block recovery:

```text
results/shared/cascadebench_strict_20260516/block_runtime_model_compare/runtime_model_compare_summary.md
```

Result:

```text
structure_plus_context analog@1: 0.08
full analog@1:                 0.06
transform-consistent@10:       0
```

Interpretation: context/block classifiers can separate constructed negatives,
but this does not yet translate to strong route-level recovery.

## Diagnosis

The current CCTS signal is too local.

It answers:

```text
Are two adjacent steps process-compatible?
```

But the route-quality problem asks:

```text
Does this candidate route/block move the synthesis toward a useful,
cascade-consistent, non-artifact route?
```

Those are related but not equivalent. A safe adjacent-step tie-break can preserve
guardrails while still failing to change meaningful top-K route quality.

## Next Direction

Move from adjacent-step CCTS to route/block outcome scoring.

Priority:

1. Use ChemEnzy as generator.
2. Build route/block candidates from the same ChemEnzy pools.
3. Use v4 evidence to label or weakly supervise:
   - block recovery
   - route fragment analog support
   - hidden-intermediate preservation
   - nontrivial stock closure
   - product-audit artifact rejection
4. Compare:
   - ChemEnzy native rank
   - retrieval-only evidence rank
   - learned route/block scorer
   - learned scorer plus guard
5. Promote only if learned scorer beats retrieval-only on held-out routes while
   preserving stock/no-failure.

## Immediate No-Go Rule

Do not claim publication-level improvement from the adjacent-pair reward.

Acceptable wording:

```text
The adjacent-step scorer is a safe search-time diagnostic and tie-break feature.
```

Not acceptable:

```text
The adjacent-step scorer solves cascade-conditioned route selection.
```
