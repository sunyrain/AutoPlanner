# Frozen V4 B4 failures: result-first first-loss audit

Status: frozen-data diagnosis complete; implementation canary-ready

## Scope

This audit reads the completed 20-target V4 panel at
`results/shared/synthatlas_result_first20_v4_20260814`. It does not rerun a
provider, mutate raw outputs, or rewrite the official B4 result of 12/20.

The question is deliberately narrow: for each of the eight B4-open targets,
where is the earliest observed result loss or remaining route-opening event?
Credibility, condition completion, and release engineering are downstream and
do not participate in this classification.

## Per-target diagnosis

| Target | Frozen provider observation | Root state | First useful classification | Generic recovery status |
|---|---|---|---|---|
| 006 | native 0; first guided found 4 but gate kept 0; one atom-balance-only stock-closed quarantine | one selected route retained one open leaf | frontier soft-gate loss, followed by a zero-result frontier | structural-hard/stoichiometry-soft gate implemented; matched canary pending |
| 008 | native 0; guided 0 | one host-validated Codex route, one open leaf | true provider zero-result on attempted target/frontier | zero-result frontier is now sent to one failure-aware Codex replan |
| 011 | native 0; guided attempts eventually admitted 2 proposals | best long route had one open leaf | local guided success did not close the root; continuation policy stopped too early | no-gain closes only the attempted molecule; distinct leaves remain eligible |
| 012 | native returned a stock-closed route but soft gate/import path lost it; guided also found routes | frozen root remained open | direct soft-gate plus complete-route truncation | fresh canary verified B4 after full 9-step import |
| 013 | native 0; both guided searches admitted routes | best long route retained two open leaves | local guided success without root closure | adaptive distinct-frontier continuation implemented; matched canary pending |
| 014 | native 0; both guided searches admitted routes | best route retained one open complex intermediate | local guided success without root closure | adaptive distinct-frontier continuation implemented; matched canary pending |
| 015 | native 0; both tried guided frontiers 0 | two Codex routes retained different open intermediates | true provider zero-result on all tried frontiers | failure-aware global replan and molecule-identity dedup implemented |
| 016 | native found 18; gate kept 0; 16 atom-balance-only stock-closed quarantines | frozen root open | direct rule-gate loss | fresh soft-gate canary verified B4 with 3 stock-closed routes |

## What the earlier audit got wrong

The earlier provider-integration audit proved that a route-level lineage row
survived several boundaries. It did not prove complete step topology
conservation. A nine-step provider route imported as six steps still had a raw
route, normalized route, selected route family, and non-empty canonical IDs.
The aggregation then used any step overlap as a bound route, so partial import
was reported as integration success.

Four process errors followed from that contract gap:

1. audit work optimized trust and deliverability before B4 first-loss recall;
2. aggregate target counters hid partial loss inside a nominally present route;
3. fixtures did not contain a successful route longer than the search-depth
   setting, so search depth and ingestion depth could drift undetected;
4. two independent actionable-event lists allowed the outer replan gate to
   accept an event that the Director silently ignored.

The corrected first integration invariant is per selected provider route:

`provider steps == imported proposals == canonical-bound steps`

Any overlap below equality is partial binding, not complete integration. An
explicit topology loss remains visible even if another route reaches B4.

## Implemented result-path changes

- Complete provider routes are never sliced by the search-depth setting.
- Provider topology conservation is recorded per route and is the first loss
  boundary before generic canonical ingestion/materialization labels.
- A no-gain guided call closes only that molecule; it cannot suppress a
  different untried root open leaf.
- Guided dedup uses canonical molecule identity and is reconstructed from
  persisted action stages, so graph revision changes do not repeat a call.
- A real zero-result search emits one actionable replan event with the failed
  frontier. Preflight/runtime unavailability emits no search-exhaustion event.
- The signal gate and Director consume one canonical actionable-event set.
- Concurrent provider/LLM computation remains parallel, while graph reads and
  writes are serialized through the canonical store lock.
- All eight frozen B4 failures exposed an additional host control break: the
  loop terminated as `no_action` while its own next-action projection selected
  an eligible `recompute_route_closure` action. The compiler already emitted
  the action, but the target runtime had no handler. The generic handler now
  invokes the canonical derived-state recomputation path; an unchanged
  projection settles as bounded no-gain instead of fabricating progress. A
  read-only scheduler projection over each frozen graph found at least one
  untried guided frontier after the stranded closure action for every failed
  target, so this was a search-truncation boundary, not merely a cosmetic final
  projection.

Focused acceptance after these changes: 171 tests passed. This proves the
offline production path and the frozen-data diagnosis. It does not claim a new
panel rate. The next scientific state is `canary_ready`, not `scale_ready`.
