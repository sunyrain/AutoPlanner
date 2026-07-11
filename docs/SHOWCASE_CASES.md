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

## Fresh unseen-target audit

Two complex targets whose names, structures and InChIKeys were absent from the local
result tree were run end-to-end on 2026-07-11. These are discovery runs, not curated
fixtures. They exercise direct Codex child-agent launch, consensus rebuild, frontier
scheduling, the deterministic closeout gate and the route-forest projection.

| Fresh target | Agent result | Projected forest | Source result | Verdict |
| --- | --- | --- | --- | --- |
| Strychnine | Four child roles succeeded; seven candidates and six proposals | 16 branches, 10 molecules, 22 reactions, 48 edges | Zero real sources; one placeholder kept separate | Useful failure/complexity diagnostic; not showcase-eligible and not solved |
| Nirmatrelvir | The accepted second attempt had four successful child roles and five proposals | 10 branches, 8 molecules, 10 reactions, 26 edges | One real exact-target DOI (`10.1126/science.abl4784`); no placeholder; Codex-role reports remain one correlated support group | Stronger fresh discovery example, but still not showcase-eligible and not solved |

Both runs stopped at `hypothesis_routes_pending_execution`. Nirmatrelvir produced five
alternative consensus-graph branches and identified target-proximal patent chemistry,
but no branch had stock closure plus reaction proof. It therefore has no proof-eligible
portfolio route, validated full-route replacement or genuine independent multi-source
consensus. This is the intended fail-closed behavior: a useful route proposal is not a
scientifically completed route.

Use three separate labels when evaluating a case:

- **Showcase-eligible** means the evidence and layout are suitable for a positive demo.
- **Scientifically solved** means the deterministic parent-route proof and stock audit
  both close; neither fresh run qualifies.
- **Architecture V2 complete** means full frontier closure, proof-bound portfolio
  selection, validated replacements and independent multi-source coverage; neither
  fresh run qualifies.

The trusted registry under `tests/fixtures/` is test-only proof material. It must not be
used to promote a new scientific run or to describe a fresh target as solved.

## Display contract

- The initial canvas shows one readable featured branch, not every branch at an
  unreadable 3–12% scale.
- A deterministically solved and executable primary branch always wins. When several
  unresolved branches have the same top rank, the compiler records a lexical display
  anchor and the UI opens that same branch. It is labelled `同分候选`, never chemical
  equivalence. Otherwise an unresolved featured branch may be selected for display
  quality without changing scientific proof status.
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
Remove-Item Env:AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY -ErrorAction SilentlyContinue

python scripts/refresh_agentic_closeout_artifacts.py results/shared/full_rerun_advisory_visual_20260702/artemisinin

python scripts/refresh_agentic_closeout_artifacts.py results/shared/paclitaxel_architecture_v2_20260710
```

These commands rebuild derived closeout artifacts from saved work. They do not rerun
the expensive Codex teams, model search or literature acquisition. Keep the test-only
trusted registry unset for scientific demonstrations.

On a clean checkout, the example selector discovers only route HTML files that really
exist under the repository result roots. If none exist, it is disabled and the page
shows the command needed to create a run; no hard-coded missing iframe is loaded.
