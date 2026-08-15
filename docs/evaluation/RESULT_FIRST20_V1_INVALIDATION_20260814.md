# Result-first 20-target V1 invalidation (2026-08-14)

## Decision

`results/shared/synthatlas_result_first20_v1_20260813` was intentionally
stopped after three completed targets and one partial fourth target. The root is
preserved as read-only diagnostic evidence and is excluded from benchmark
performance claims.

## Observed result

- Completed: 3/20; queued/running at stop: 16/1 in the last runner-written
  status snapshot.
- B1: 3/3 completed; B2: 0/3; B4: 0/3.
- Provider-integration loss diagnostic: 2/3 completed targets.
- Target 002 contained a raw ChemEnzy guided route with seven steps and
  canonical materialized edges, but `canonical_route_ids` was empty. Target
  003 exhibited the same disposition class.

The 0/3 B4 observation is therefore not an interpretable search-quality
estimate: routes found by the provider could exist in the graph without being
part of the target-rooted route family used for scoring.

## Root cause

Guided ChemEnzy hypotheses retained their temporary provider alias, while the
canonical parent route-family identity was not propagated through the
materialization command, worker result, and candidate-ingestion boundary. A
frontier may belong to multiple route families, but the old handoff retained at
most one parent and could retain none after alias resolution.

## General fix

- Carry `canonical_route_family_ids` as a plural, digest-bound host field from
  guided hypothesis through materialization.
- Intersect supplied parent IDs with route families already present in the
  canonical graph; the provider cannot invent or hijack canonical parents.
- Attach admitted hypotheses and edges to every valid parent family.
- Add a regression test proving that a guided stock-recovery edge becomes part
  of each requested parent portfolio and can close B4.
- Add `provider_integration_loss_target_count` and lineage disposition counts to
  the read-only panel summary so this failure cannot remain silent.

## Verification

- Guided integration, canonical hypergraph, and ChemEnzy probe slice: 41 passed.
- Summary, canonical hypergraph, and guided target slice: 28 passed.
- Ruff on all changed implementation and test files: passed.

The replacement panel must use a fresh output root and denominator. No records
from V1 may be copied into it.
