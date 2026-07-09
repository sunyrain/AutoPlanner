# Repository Cleanup Report, 2026-05-12

## Policy

The cleanup keeps active code, tests, v4 data, ChemEnzy vendor code, and local
result artifacts in place. Files were only moved when they were root-level
reference inputs with no active import path. This avoids breaking old scripts
that still default to root-level `cascade_dataset_v2.*` and
`cascade_dataset_v3.json`.

## Current Entry Points

- `cascade_planner/route_tree/` - active route-tree controller and proposal
  integration.
- `cascade_planner/cascade_search/` - cascade-native controller contracts and
  search layer.
- `cascade_planner/eval/` - benchmark, trace, audit, and training scripts.
- `cascade_planner/web/` - demo web interface.
- `dataset_v4_release/` - current v4 cascade dataset release.
- `results/shared/coverage_fix_20260511/` - latest full100 coverage-fix result.
- `results/shared/open_leaf_policy_20260511/` - latest learned open-leaf policy.
- `paper/nature_autoplanner_cascade/` - generated Nature-style draft and figure.

## Archived From Root

- `image.png` -> `archive/reference_inputs_2026-05-12/root_image_20260512.png`
- `s41467-025-65898-3 (1).pdf` -> `archive/reference_inputs_2026-05-12/`

## Left In Root Deliberately

- `cascade_dataset_v2*.json` and `cascade_dataset_v3.json`: many historical
  scripts still use these as default paths.
- `templates*.csv.gz` and `ecreact-1.0.csv`: ignored local inputs that may be
  used by older expansion/training scripts.

## Latest Measured State

Compared with the 2026-05-09 full100 route-tree baseline, the 2026-05-11
coverage-fix run improved plan rate, skeleton-type recovery, candidate coverage,
route-pool recovery, condition compatibility, cascade compatibility and runtime.
Strict stock solve regressed and is the main unresolved control problem.
