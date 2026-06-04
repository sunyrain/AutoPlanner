# AutoPlanner-Cascade

AutoPlanner-Cascade is the active AutoPlanner research codebase for
process-aware chemoenzymatic retrosynthesis. The current mainline is ChemEnzy-
backed multi-step search plus conservative material/route audit, SMILES-first
literature research, strategic disconnection cards, and validated literature
templates that may enter ChemEnzy only as audited one-step proposal sources.

Old ranker, CCTS, v4 learned-value, and fallback lines are frozen or archived.
They remain in the tree for reproducibility, but they are not the default
runtime path or the next training target unless an active checklist explicitly
promotes them.

## Current Anchor

Start here:

- [AutoPlanner mainline](docs/MAINLINE.md)

Supporting docs:

- [Docs index](docs/README.md)
- [SMILES-first literature strategic workflow](docs/SMILES_FIRST_LITERATURE_STRATEGIC_WORKFLOW_2026-06-03.md)
- [EvoChemEnzy code delivery checklist](docs/EvoChemEnzy_Code_Delivery_Checklist_2026-06-03.md)
- [Literature-to-executable template checklist](docs/LITERATURE_TO_EXECUTABLE_TEMPLATE_CHECKLIST_2026-06-04.md)
- [Codebase hygiene audit](docs/CODEBASE_HYGIENE_AUDIT_2026-06-04.md)

Current conclusion:

AutoPlanner is best read as a ChemEnzy-backed route generation and quality-
control system. The usable path is to keep strong ChemEnzy proposal/search
capability, add material-sanity and route audit, and allow literature-derived
templates into search only through deterministic retron/applicability/
reconstruction gates. LLM work remains outside the ChemEnzy inner loop.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `cascade_planner/cascade_search/` | Cascade-native state/search contracts, verifier hooks, proposal providers |
| `cascade_planner/route_tree/` | Older route-tree controller and compatibility helpers |
| `cascade_planner/agent/` | Episode-level agents, artifact schemas, literature workflow, route audit, policy gates |
| `cascade_planner/baselines/` | ChemEnzy adapter, one-step providers, plugin wrappers, baseline bridges |
| `cascade_planner/vnext/` | Feature schemas and model-facing route/action representations |
| `cascade_planner/eval/` | Benchmark, trace, audit, and training scripts |
| `cascade_planner/web/` | Local demo web interface |
| `dataset_v4_release/` | Current v4 cascade dataset release |
| `data/` | Frozen benchmark inputs and small curated datasets |
| `results/shared/` | Local benchmark outputs, traces, checkpoints, caches; ignored by git |
| `docs/` | Current source-of-truth docs, hygiene report, archive index, static showcase outputs |
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

Focused checks for the current literature-template and ChemEnzy bridge path:

```bash
pytest -q tests/test_literature_template_cards.py \
  tests/test_template_applicability.py \
  tests/test_executable_template_validation.py \
  tests/test_literature_one_step_plugin.py \
  tests/test_literature_template_plugin_benchmark.py

pytest -q tests/test_agent_artifact_contracts.py \
  tests/test_literature_evidence_cards.py \
  tests/test_smiles_first_workflow.py \
  tests/test_chem_enzy_onestep.py \
  tests/test_chem_enzy_native_chemical_plugin.py
```

Full test discovery is useful before commits, but some historical tests may
track older source-order expectations. Review failures before assuming a
runtime regression.

## Cleanup Notes

Top-level docs are intentionally small. Historical reports are under
`docs/archive/`, and root-level presentation/reference drops belong under
`docs/archive/2026-06/reference_materials/`.

Large local artifacts such as `*.mar`, `data_external/`, `vendor/`,
`AI_OS_AutoResearch/`, `results/shared/`, and generated `results/v2/` outputs
are ignored or treated as local runtime state. Do not promote them into source
control unless a release manifest explicitly calls for it.
