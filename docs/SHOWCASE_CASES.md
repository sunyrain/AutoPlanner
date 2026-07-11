# Complex-molecule showcase policy

Updated: 2026-07-11

The showcase must distinguish a useful exploration result from a deterministically
closed synthesis route. Replaying the saved complex cases with the current strict
proof contract produced **zero** eligible `solved` cases. The UI and sample menu
therefore label every current complex example as unresolved instead of carrying
forward historical solved claims.

## Recommended cases

| Case | Current projection | Evidence and route diversity | Honest use |
| --- | --- | --- | --- |
| Artemisinin | 51 branches, 35 molecules, 64 reactions, 149 edges | Three DOI-bound PDF/visual chains, eight consensus proposals and ten Codex consensus-graph routes | Primary showcase for literature evidence plus alternatives; unresolved |
| Paclitaxel | 96 branches, 83 molecules, 122 reactions, 337 edges | Two visual chains, 26 consensus proposals and 24 consensus-graph routes; independent multi-source support exists for several graph hyperedges | Architecture V2 and large-graph stress showcase; unresolved |
| Erythromycin A | 67 branches, 80 molecules, 75 reactions | Macrocycle and stereochemistry stress case, but only one independent PDF source in the saved run | Advanced density/performance test, not a primary evidence showcase |
| Atorvastatin | 23 branches, 15 molecules, 23 reactions after current replay | Compact and visually convenient, but the historical route has no strict full atom maps or trusted precedent binding | Migration/rejection audit only; not solved |
| Bufotalin | Current strict replay collapses to two L0 branches | Historical PDF assets remain reusable, but the old stitched route has no current step-level L3 proof | Literature-revalidation case only; not solved |

Artemisinin is the default sample because it is the best balance of readable route
length, multiple literature documents and alternative hypotheses. Paclitaxel stays
available as the deliberately dense expert panorama. Erythromycin is useful for
performance regression, while Atorvastatin and Bufotalin are intentionally absent
from the positive showcase menu until a fresh strict proof run succeeds.

## Display contract

- The initial canvas shows one readable featured branch, not every branch at an
  unreadable 3–12% scale.
- A deterministically solved and executable primary branch always wins. For an
  unresolved run, the featured branch is selected only for display quality using
  route length, evidence sources, confidence, structured-molecule coverage and
  branch type; this does not alter the compiler's scientific `primary_branch_id`.
- Molecule nodes use the compiler's RDKit depiction when available. Trust colors
  remain limited to reactions and dependency edges, so a neutral molecule is never
  misread as an L0 rejection or advisory claim.
- `共享骨架` merges canonical molecules for cross-route inspection. `路线全景`
  preserves all branches as the expert/stress view. Neither view creates an edge,
  replacement or solved claim that is absent from backend artifacts.
- A replacement is shown only after full backend AND/OR route revalidation. Pairwise
  interface similarity remains diagnostic and cannot enable a visual one-step splice.

## Rebuild the local samples

Run artifacts under `results/shared/` are intentionally ignored by Git. Refresh them
from their saved blackboards with the current compiler before a demonstration:

```powershell
$env:AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY = "tests/fixtures/trusted_literature_step_registry.json"

python scripts/refresh_agentic_closeout_artifacts.py results/shared/full_rerun_advisory_visual_20260702/artemisinin

python scripts/refresh_agentic_closeout_artifacts.py results/shared/paclitaxel_architecture_v2_20260710
```

These commands rebuild derived closeout artifacts from saved work. They do not rerun
the expensive Codex teams, model search or literature acquisition.
