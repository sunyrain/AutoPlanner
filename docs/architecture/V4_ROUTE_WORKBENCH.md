# V4 route workbench

## Contract boundary

The browser is a read client. `application.route_workbench` joins one canonical
hypergraph revision to one proof portfolio revision and emits at most five
selected routes. It cannot promote proof, infer stock, or turn aggregate counts
into completion.

```text
canonical hypergraph + proof portfolio
  -> retrosynthesis_route_workbench.v1
       -> hypotheses
       -> expanded routes
       -> reaction-validated routes
       -> stock-closed routes
       -> proof/evidence/stock/conflict/provenance inspectors
  -> retrosynthesis_route_workbench_delta.v1
       -> entity upserts
       -> entity removals
       -> revision-bound metadata
```

A delta is accepted only against its exact `base_sha256`. A missed or reordered
delta falls back to a full snapshot. Stable canonical IDs preserve selection and
camera state when surviving entities are updated.

## Presentation semantics

- Edge color is a display of the stitched L0-L4 proof result; it is never proof
  authority itself.
- Badges identify proposal origin, exact-source kind, reaction validation,
  conflict, Pareto membership, and stock closure on separate axes.
- L0 disconnection hypotheses have their own view and cannot appear in the
  stock-closed count.
- Shared intermediates use one canonical graph node. Alternative reactions are
  modules expanded on demand, not duplicated full routes.
- The default canvas is one readable portfolio route. Shared and portfolio
  overviews are capped at five until the operator explicitly requests more
  historical exploration content.

## Product profiles and proof vector

Workbench routes now compile six named product profiles from one read-only,
six-axis proof vector:

```text
identity / reaction / conditions / sources / stock / process
  -> exploration_closed
  -> reaction_validated
  -> literature_grounded
  -> condition_complete
  -> procurement_closed
  -> process_ready
```

The profiles are not inferred from L0-L4 color. `benchmark_hit` and
`offer_verified` are separate stock states; exact structure observations and
exact, complete procedures are separate condition records. A stronger profile
requires its named axes and fails closed when an older projection lacks the
versioned proof vector.

The route navigator exposes overlapping, authority-bound views for hypotheses,
fully expanded routes, host-validated reactions, literature-grounded routes,
condition-complete routes, configured-boundary closure, real procurement
closure, and process-ready candidates. The reaction and route inspectors show
all six axes independently; the older numeric trust vector remains a display
aid rather than product-completion authority.

## Source procedures and condition gaps

Exact structure observations and exact reaction procedures are separate
canonical entities. A `source_reaction_procedure_record.v1` must bind an edge,
an exact structure record, a source, a location, and a SHA-256 digest of the
source-authored procedure fragment. Conditions without that binding remain an
unverified compatibility projection and cannot satisfy `condition_complete`.

The reaction inspector lists every bound procedure with its status, location,
fragment digest, normalized condition fields, and missing operational groups.
When several procedures cover one edge, the UI shows the smallest current gap
as a work hint while preserving every record; it does not select a scientific
winner or hide conflicting conditions.

## Fact lifecycle and route degradation

Canonical facts are append-only even when their authority changes. A
`canonical_fact_lifecycle_event.v1` binds the subject kind, canonical ID,
original digest, host authority scope, effective time, and reason codes.
`revoke` and `expire` remove authority without deleting the original record;
`restore` is a later causal event that names the inactive predecessor.

Reaction validation, exact sources, procedures, and stock proofs filter these
events before stitching. The dependency index recomputes only affected edges,
leaves, routes, and deficits, while the full-recompute oracle remains the
correctness gate. The inspector displays every active lifecycle impact under
“失效事实”; the old record remains visible for audit but cannot satisfy a
proof-vector axis or product profile.

## Camera and rendering

The SVG element is a fixed viewport. One `.graph-world` transform owns both pan
and zoom:

```text
screen = translate(panX, panY) * scale(zoom) * world
```

Pointer capture starts at pointer-down. Pointer moves only update the latest
coordinates; one animation-frame callback commits the camera. No layout,
innerHTML replacement, or depiction generation occurs during drag.

Large views use semantic zoom, cached structure SVGs, cached stable graph
models, and viewport culling. V4 logical layers are prepared by the backend, so
the browser interaction thread does not perform heavy graph layout. The runtime
probe at `window.__AUTOPLANNER_ROUTE_PERF__.snapshot()` reports camera-frame
duration/delay, dropped frames, graph update time, rendered and culled object
counts, heap usage when supported, and the current camera.

## Regression gate

`tests/test_route_ui_runtime.py` includes static invariants and a real headless
Chromium interaction run. It verifies drag, zoom-anchor preservation, fit,
selection, minimap recentering, single-world-transform ownership, and culling on
a 70-step graph. The test skips only when no Chromium-family browser is present;
all non-browser contracts continue to run.
