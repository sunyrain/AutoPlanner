# results/ layout (post 2026-04-23)

| Folder | Dataset | Notes |
|---|---|---|
| `v1/` | `cascade_dataset.normalized.uniprot.json` (1754 records / 6306 steps) | Original benchmark artifacts |
| `v2/` | `cascade_dataset_v2.normalized.json` (2491 records / 8748 steps) | Current focus |
| `shared/` | cross-version assets | atom-map cache, enzymemap templates, demo route JSON |

## Conventions
- Eval scripts default to `CASCADE_VERSION=v2` and write to `results/v2/`. Override with env var:
  ```powershell
  $env:CASCADE_VERSION="v1"
  python -m cascade_planner.eval.hybrid_multi_audited
  ```
- Filenames are NOT prefixed with version — the version lives in the parent folder.
- `hybrid_multi_audited.md` (audited, honest) supersedes `hybrid_multi_report.md` (legacy).
- Any script that still writes to bare `results/foo.csv` is **legacy**; prefer `results/{v1,v2}/foo.csv`.
