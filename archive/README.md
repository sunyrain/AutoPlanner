# archive/

Cold storage. Nothing here is loaded by the active pipeline.

## Layout

| Path | Content |
|---|---|
| `code/eval_v1_superseded/` | v1 evaluators replaced by `eval/hybrid_multi_audited.py`, `eval/condition_diagnosis.py`. Kept for reproducing old reports. |
| `code/data_v1_superseded/` | v1 schema normalizer + audit script, replaced by `data/normalize_v2.py` + `data/strict_filter_v2.py`. |
| `code/data/`, `code/demo/`, `code/training/` | Older retired scripts (pre-2026-04-23). |
| `datasets/` | v1 snapshots (`cascade_full_snapshot_*.json`) + the original `cascade_dataset.json` family. Superseded by `cascade_dataset_v2.json` at repo root. |
| `docs/` | Early planning docs (`FEASIBILITY_ANALYSIS.md`, `CODE_ARCHITECTURE_v3.md`, etc.). Current truth is `STATUS_REPORT.md`. |
| `logs/` | Dated run logs. `logs/2026-04-23/` = logs swept from repo root during the 04-24 cleanup. |
| `migration_2026-04-23/` | GPU cluster migration bundle (`autoplanner_full_2026-04-23.zip`, manifest, cluster readme). |
| `results/` | Retired v1 result CSVs. Current results live at `results/v1|v2|shared/`. |

## Restoring

```powershell
# Restore a single module
Copy-Item archive\code\eval_v1_superseded\hybrid_multi.py cascade_planner\eval\

# Restore the v1 dataset
Copy-Item archive\datasets\cascade_dataset.normalized.uniprot.json .
```
