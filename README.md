# AutoPlanner-Cascade

AutoPlanner-Cascade is the active AutoPlanner research codebase for
process-aware chemoenzymatic retrosynthesis. The current implementation keeps
ChemEnzyRetroPlanner as the mature multi-step planning core, then adds
coverage-aware controller logic, route-tree traces, typed candidate-miss audits,
open-leaf policy training, cascade search contracts, and a web demo layer.

The repository is intentionally organized around the current development path:
improve candidate coverage and search access first, then train source scheduling
and state/action value models, then add process repair actions.

## Current Result Anchor

Current project summary:

- [Current state, cleanup, and next step](docs/CURRENT_STATE_2026-05-19.md)
- [Codebase status and cleanup guard](docs/CODEBASE_STATUS_2026-05-19.md)
- [Model strengthening plan](docs/MODEL_STRENGTHENING_PLAN_2026-05-19.md)
- [Phase I closeout report](docs/PHASE1_RESEARCH_CLOSURE_2026-05-15.md)
- [Phase I cleanup manifest](docs/PHASE1_CLEANUP_MANIFEST_2026-05-15.md)
- [Phase II completion audit](docs/PHASE2_COMPLETION_AUDIT_2026-05-15.md)

Latest audited artifacts:

- `results/shared/phase2_20260515/full100_abcd_gate30/reports/comparison.md`
- `results/shared/phase2_20260515/quality_filter_ablation_gate30/reports_quality/comparison.md`

Current conclusion:

AutoPlanner is currently best read as a ChemEnzy-backed route generation and
quality-control system. Student-only control remains below baseline; the usable
path is to preserve strong ChemEnzy proposal/search capability, then add
AutoPlanner-side queueing, stock policy, material-sanity audit, rejected-route
traceability, and cascade-aware ranking/search hooks.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `cascade_planner/route_tree/` | Active route-tree controller, proposal adapters, features, traces |
| `cascade_planner/cascade_search/` | Cascade-native state/search contracts and controller layer |
| `cascade_planner/vnext/` | Feature schemas and model-facing route/action representations |
| `cascade_planner/eval/` | Benchmark, trace, audit, and training scripts |
| `cascade_planner/web/` | Local demo web interface |
| `dataset_v4_release/` | Current v4 cascade dataset release |
| `data/` | Frozen benchmark inputs and small curated datasets |
| `results/shared/` | Local benchmark outputs, traces, checkpoints, caches; ignored by git |
| `docs/` | Current architecture notes, cleanup report, and postmortems |
| `paper/nature_autoplanner_cascade/` | Nature-style manuscript draft, main figure, and PDF export |
| `archive/` | Retired docs, old snapshots, and reference inputs |
| `vendor/` | Local ChemEnzyRetroPlanner/vendor runtime; ignored by git |
| `AI_OS_AutoResearch/` | Optional external integration checkout; ignored by this repo |

## Paper Draft

A draft manuscript and generated Figure 1 are available under:

```text
paper/nature_autoplanner_cascade/
```

Build manually when a TeX toolchain is available:

```bash
cd paper/nature_autoplanner_cascade
python scripts/make_main_figure.py
pdflatex -interaction=nonstopmode -halt-on-error -output-directory build main.tex
cp build/main.pdf build/autoplanner_cascade_nature_draft.pdf
```

## Quick Checks

Focused checks for the latest policy/training path:

```bash
PYTHONPATH=. python tests/test_vnext_pack_and_training.py -v
```

Full test discovery is useful before commits, but some historical tests may
track older source-order expectations. Review failures before assuming a
runtime regression.

## Cleanup Notes

The root-level `cascade_dataset_v2*.json`, `cascade_dataset_v3.json`,
`templates*.csv.gz`, and `ecreact-1.0.csv` files remain in place because older
scripts use them as default paths. Root-level reference image/PDF files were
moved to `archive/reference_inputs_2026-05-12/`.

Superseded 2026-05-14 phase-I drafts were removed after the closeout and
their content was folded into the 2026-05-15 report and cleanup manifest.

Generated AI_OS integration bundles from the Web demo work were archived under
`archive/code/generated_patches_2026-05-19/`. Keep `AI_OS_AutoResearch/` as its
own git checkout rather than adding it to this repository.
