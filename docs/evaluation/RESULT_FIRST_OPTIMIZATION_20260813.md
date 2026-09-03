# Result-first optimization (2026-08-13)

## Decision

The current product objective is route output first. In adaptive runs, the
priority order is now:

1. generate candidate routes;
2. materialize and host-validate reaction edges;
3. close route leaves against the configured stock boundary (B4);
4. only then spend budget on exact evidence, condition completion, conflict
   resolution, and enzyme/mechanism Program work.

This is a scheduling policy, not a scientific claim. Lower-level candidates
remain labelled at their actual proof level.

## Why the previous 50-target result was misleadingly low

The frozen raw run remains unchanged under
`results/shared/synthatlas50_unified_adaptive_20260813`.

- 1,998 evidence actions and 731 Program actions were attempted while only 242
  ChemEnzy actions were attempted. The scheduler therefore spent most of its
  finite task budget downstream before stock closure.
- 202 guided requests were fixed at only 8 iterations / 90 seconds and yielded
  no accepted proposals.
- Five concurrent shards with two workers each created a ten-target pressure
  wave; four targets ended in `MemoryError`.
- A provider-parameter audit was a fail-closed admission gate. Eight targets
  were marked `chemenzy_parameter_binding_mismatch`; five of those raw results
  already contained 57 routes in total. Target 001 alone had `ok=true`, five
  returned routes, and a native stock-closed result, but was discarded because
  the launcher receipt omitted `stock_names`.

The reported low closure rate therefore mixed search quality with host control
debt. It must not be interpreted as a clean measurement of either ChemEnzy or
the unified planner.

## Implemented changes

- Adaptive scheduling defers exact-evidence, condition, conflict, and Program
  actions until B4. Reaction validation stays early because stock-closed route
  selection depends on it.
- Initial evidence prefetch and pre-B4 Program signalling are disabled in the
  adaptive path.
- ChemEnzy condition/enzyme sidecars are disabled during adaptive route search;
  they can run after route closure.
- Fast/standard/proof provider profiles are now respectively 100/500/1500
  iterations with 300/1200/1800 second limits. Guided probes inherit the same
  declared profile instead of being silently promoted by unrelated host task
  budgets.
- The host portfolio default is 16 routes. Guided search receives two bounded
  frontiers.
- Parameter-binding mismatches remain digest-bound audit warnings but no longer
  discard structurally admissible provider routes.
- Event replay and tail repair are streamed instead of loading the complete
  JSONL log into memory.
- Target runtime concurrency is capped at two independent initial producers;
  validation/proof cohorts are no longer launched concurrently.
- Frozen procurement snapshots use a 365-day default reuse window, preventing
  a multi-batch audit from invalidating its own snapshot after 30 days.
- A post-loop replan signal is now consumed once through the same unified
  runtime instead of being published and left pending.

## Verification

- Ruff and Python compilation: passed.
- Focused regression suite: 127 passed.
- Additional profile and parameter-admission tests: 26 passed.
- A live two-target, single-worker fast-profile canary completed under
  `results/shared/synthatlas_result_first_canary_v2_20260813`:
  - target 018: B1/B2, two target-rooted routes, one host-validated route,
    B4 not reached, 919.5 seconds;
  - target 024: B1/B2/B4/B5, three target-rooted routes, three host-validated
    routes, one stock-closed route, B4 first reached at 245.5 seconds and the
    run closed at 303.5 seconds;
  - target 024 previously stopped at B2 after 663.2 seconds. The paired canary
    therefore changed the same target from non-B4 to B4/B5 while halving final
    wall time;
  - target 018 used 14 actions versus 68 previously. Its pre-B4 action log has
    no evidence, condition, or Program execution;
  - target 024 used two Codex-selected guided frontiers. They each returned one
    host-admissible ChemEnzy route in 13.3 and 28.7 seconds. B4 was achieved
    before credibility/Program work became eligible.

## Interpretation boundary

The code changes repair priority inversion and route rejection, and the paired
canary shows a real B4 gain. Two targets are not a benchmark success-rate
estimate. The next valid measurement is a frozen 20-target result-first panel
using the canary profile. Only after that panel should publication-scale
credibility enrichment or a larger benchmark be resumed.
