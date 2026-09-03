# Result-first 20-target V2 invalidation (2026-08-14)

## Decision

`results/shared/synthatlas_result_first20_v2_20260814` was intentionally
stopped before any target completed. It is a runtime diagnostic root, not a
benchmark result, and must not be resumed or merged into V3.

## Trigger

Target 001 reproduced the earlier provider timeout pattern:

- target-level ChemEnzy ran to the full 300-second parent timeout;
- no result artifact was written, so any successful route found before process
  termination was unavailable to the host;
- only after the hard timeout did guided search begin.

The provider request asked for 16 output routes, but the launcher derived a
128-route successful-search reserve. With `keep_search=true`, this made the
iteration and wall budgets work targets rather than maxima for many targets.

## Replacement requirement

V3 may start only after a real target-001 canary verifies both behaviors:

1. native MCTS receives an internal soft wall-time limit below the existing
   external hard limit and returns routes already found at that boundary;
2. the successful-route reserve is a small bounded multiple of the actual host
   output portfolio, not eight times that portfolio.

The external 300-second resource ceiling remains unchanged. The internal
deadline only reserves time for route extraction, host audit, JSON serialization,
and atomic result publication.
