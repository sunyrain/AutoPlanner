# Result-first integration fix canaries

Status: in progress

## Why the earlier audits missed the dominant failure

The previous embedding audit checked route-level presence and several adjacent
boundaries: raw provider route, normalized route, host portfolio selection,
canonical lineage, materialization, and stock projection.  It did not assert
step-level topology conservation across the complete route.  Consequently a
nine-step successful provider route could be imported as six steps while still
reporting one raw route, one normalized route, one selected route family, and
non-empty canonical lineage.

The truncation was introduced by treating the provider search-depth contract
as a post-search ingestion slice.  `max_steps=6` correctly constrained search,
but an already returned complete route was incorrectly cut after step six.
This is a semantic contract error, not a chemistry or evidence failure.

The new first integration gate is therefore per-route topology conservation:
provider step count, imported proposal count, and bound canonical step count
must agree for every selected complete route.  Credibility and operational
receipt checks remain downstream of this result-preservation gate.

This was an audit-design failure. Route-level counters were treated as a
proxy for result preservation, and the paired replay fixtures did not contain
a successful route longer than the configured search depth. Claims that the
provider-to-host chain no longer silently lost routes were therefore broader
than the invariant actually tested.

## Observed canaries

### Target 016: soft reaction warning gate

The route-generation verifier now blocks only structural contract failures
such as invalid SMILES, product mismatch, or route-order mismatch.  Atom
balance and missing-condition findings remain visible warnings and grant no
reaction proof, but no longer delete structural routes.  The fresh target-016
canary reached B4 with four root routes and three stock-closed routes.

### Target 012: complete provider route import

The frozen 20-target panel recorded target 012 as B4-open.  Inspection showed
that ChemEnzy had returned a nine-step stock-closed route and every terminal
leaf was present in the frozen benchmark SQLite stock.  The host imported only
the first six steps.

After removing the ingestion truncation, a fresh target-012 run reached B4:

- B1: true;
- B4: true;
- imported ChemEnzy proposals: 13 across native and guided calls;
- target-rooted canonical routes: 3;
- strict stock-closed routes: 1;
- end-to-end elapsed time: 253.162 seconds.

This canary validates the causal fix.  It does not rewrite the frozen official
20-target score of 12/20 and does not claim a new aggregate score without a
fresh matched panel.

## Result-aware provider stop canary

In the frozen panel, target 001 found its first native success at about 15.9
seconds but the provider continued to about 185.0 seconds while filling a raw
route reserve. Result mode now audits each newly successful native route with
the same host structural gate used at final materialization and stops on the
first admitted route. If a route is rejected, search continues under the
original iteration and wall-time limits.

The fresh target-001 canary retained B4=true. ChemEnzy stopped after three
iterations in 21.438 seconds instead of 30 iterations in 184.969 seconds, an
88.4% provider-time reduction. The stop reason was
`host_admitted_success_route_found`.

End-to-end time changed only from 243.606 to 235.793 seconds. The provider
finished at offset 29.362 seconds, while the initial Codex peer finished at
233.269 seconds; stable whole-cohort delivery therefore imposed 203.907
seconds of peer wait. This is now the leading latency target: a completed,
usable provider result must be materialized and stock-checked without waiting
for a slower peer. The frozen V4 panel remains 12/20 and is not rewritten by
these targeted canaries.

## Progressive delivery implementation

The initial cohort still starts ChemEnzy and Codex concurrently, but no longer
uses whole-cohort completion as a publication barrier. When the ChemEnzy
action completes, the same campaign runtime records it immediately and may
schedule only the deterministic delivery chain: route materialization followed
by stock audit. No second scheduler, queue, graph, or trajectory is created.

If and only if that chain closes the root-target B4 stock boundary, the target
solver signals cooperative cancellation to the default Codex worker. The
worker terminates its process tree and the director settles the action as
`cancelled_after_delivery`; this is an expected post-delivery settlement, not
a scientific or worker failure. If B4 remains open, the cancellation signal is
not set and the Codex peer completes normally.

Focused verification now covers both branches and the production wiring:

- 154/154 runtime, worker-process, director, and target-solver tests passed;
- 31/31 ChemEnzy probe and V4 architecture tests passed;
- route topology helper and built-in provider runtime are separately included
  in the V4 dependency and line-budget audit.

This slice is **canary-ready**, not yet scale-ready. The 203.907 seconds above
is the observed historical peer wait and therefore an upper-bound opportunity,
not a measured speedup from the new implementation. A fresh matched provider
canary is required before reporting an end-to-end latency improvement. The
official frozen panel remains 12/20.

## Budget-derived guided frontier continuation

Guided ChemEnzy no longer defaults to a fixed two-frontier policy. When the
operator does not provide a compatibility cap, the maximum number of frontier
calls is inherited from the unified native-search/attempt resource envelope.
An explicitly supplied frontier limit remains a hard upper bound for replay
compatibility.

Continuation is result-aware rather than call-count-aware. After every guided
provider result, AutoPlanner blocks another guided dispatch until the result is
materialized and the required stock audit is complete. It then compares the
best target-rooted parent route's open stock leaves before and after the call.
A decrease permits another frontier call; root B4, no decrease, native-search
budget exhaustion, or the unified run deadline stops it. A locally stock-closed
subtarget route is excluded unless it actually reduces the root parent's open
leaf count.

Offline target-solver integration tests cover both decisive branches:

- two parent leaves decrease `2 -> 1 -> 0`, causing two guided calls and then
  an immediate root-B4 stop;
- a no-result call leaves the count at `1 -> 1`, causing no second guided call.

Together with the progress-unit, CLI/API/Web, V4 architecture, and legacy Web
entrypoint coverage, 102 focused tests passed. This change is canary-ready; it is not counted as a
new provider success until a fresh matched real-provider run is recorded.

## Distinct-frontier continuation and zero-result replanning

The frozen failures showed that `1 -> 1` progress on one guided molecule was
being interpreted as a global stop. It now closes only the attempted frontier;
other root-route open leaves stay eligible. Attempt identity is the canonical
frontier molecule rather than an action ID, so a later graph revision cannot
cause the same zero-result molecule to be searched again. Persisted action
stages rebuild the attempted set on resume.

A second hidden contract drift existed in replanning. The target-level signal
gate and the Director maintained different actionable-event lists. A provider
zero-result event could pass the outer gate and then be ignored internally as
`no_material_replan_trigger`. Both consumers now share one event authority.
The replan context contains the failed frontier, target/frontier scope, provider
status, invocation count, and failure reasons.

Only an actually executed provider call can emit this event. Built-in runtime
preflight failure, disabled providers, and duplicate-suppressed calls report
zero provider invocations and do not spend a replan call. Tests cover both
initial completion orders (ChemEnzy first and Codex first), exactly one event
replan, two distinct guided leaves after a no-gain call, and no retry of the
same molecule after graph revision changes.

The completion-order tests also exposed a canonical graph publication race:
one producer could observe the kernel revision after another producer updated
it but before the new graph pointer was published. Graph load/apply now share
the same reentrant store lock. The full result-path focused suite passed
171/171 tests. No fresh provider run was started, so the frozen B4 score remains
12/20.
