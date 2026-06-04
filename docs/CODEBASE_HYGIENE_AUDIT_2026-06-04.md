# Codebase Hygiene Audit

Date: 2026-06-04.

Purpose: record the current repository boundary after the SMILES-first
literature workflow and literature-to-executable-template work. This file is a
cleanup contract, not a new research plan.

## Current Source Of Truth

Active implementation docs:

- `docs/MAINLINE.md`
- `docs/SMILES_FIRST_LITERATURE_STRATEGIC_WORKFLOW_2026-06-03.md`
- `docs/EvoChemEnzy_Code_Delivery_Checklist_2026-06-03.md`
- `docs/LITERATURE_TO_EXECUTABLE_TEMPLATE_CHECKLIST_2026-06-04.md`
- `docs/EvoChemEnzy_Agentic_CASP_Plan_2026-06-03.md`
- `docs/EvoChemEnzy_Plan_Feasibility_Audit_2026-06-04.md`

`docs/MAINLINE.md` is the first entry point and authority index. Historical
documents are provenance only unless one of the active documents explicitly
references them.

## Cleanup Completed

- Updated root `README.md` away from stale May 2026 top-level doc links.
- Updated `docs/README.md` and `docs/archive/README.md` to include the
  literature-to-executable-template checklist and this audit.
- Moved root-level project support/reference drops into
  `docs/archive/2026-06/reference_materials/`.
- Removed generated `image.png` screenshots from the repository root and
  `cascade_planner/agent/`.
- Removed generated Python and pytest caches:
  `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`.
- Added root-level `/*.pdf` and `/*.pptx` ignore rules so new presentation drops
  do not clutter the source root.
- Archived unused top-level `results/shared/` outputs under
  `results/shared/archive/2026-06-04_cleanup/`.

## Active Runtime Surface

The current active path is:

```text
ChemEnzy native planner
-> route/material audit
-> trigger gate for literature mode
-> typed evidence/strategy/template artifacts
-> deterministic retron/applicability/reconstruction validation
-> literature_template_plugin one-step proposals
-> ChemEnzy route search and route audit
```

Implementation anchors:

- `cascade_planner/agent/literature_templates.py`
- `cascade_planner/agent/template_applicability.py`
- `cascade_planner/agent/executable_template_validation.py`
- `cascade_planner/baselines/literature_one_step_plugin.py`
- `cascade_planner/baselines/chem_enzy_adapter.py`
- `cascade_planner/baselines/chem_enzy_onestep.py`
- `cascade_planner/agent/chem_enzy_policy.py`
- `cascade_planner/agent/smiles_first.py`
- `scripts/run_literature_template_plugin_benchmark.py`

## Code Audit

The repository contains a large historical research surface. A conservative
text/import scan found about 90 Python modules with no direct in-tree import
reference. Most are historical CLIs, benchmark entry points, or frozen research
modules, not safe deletion candidates without reproducing old reports.

Categories to treat as legacy unless an active doc names them:

- `cascade_planner/demo/`
- `cascade_planner/conditions/`
- older `cascade_planner/expand/*` reranker/debug utilities
- older `cascade_planner/cascadeboard/*benchmark.py` and supervision helpers
- older CCTS/v4/ranker/reservoir scripts under `cascade_planner/eval/`
- strict review, route-block value, and old model-review scripts under
  `scripts/`

Decision: no source modules were deleted in this cleanup pass. Deleting them
would be a behavioral change and needs a separate migration manifest with
import checks and any reproduced report dependencies.

## Local Artifact Boundary

Keep these out of source control unless a release manifest explicitly promotes
them:

- `*.mar`
- `data_external/`
- `vendor/ChemEnzyRetroPlanner/`
- `AI_OS_AutoResearch/`
- `results/shared/`
- generated `results/v2/` route figures and reports
- root-level `cascade_dataset_v2*.json` and `cascade_dataset_v3.json`

Small curated inputs under `data/`, generated showcase pages under
`docs/statins/` and `docs/bufotalin/`, and active docs under `docs/` remain
allowed repository artifacts.

## Results Shared Cleanup

`results/shared/` was cleaned with a conservative policy:

- kept entries that are explicitly referenced by `README.md`, `docs/`,
  `scripts/`, `cascade_planner/`, or `tests/`;
- kept current literature/P0 outputs:
  `smiles_first_p0_e2e_20260604`,
  `p0_smiles_first_benchmark_20260604`,
  `p0_literature_fullflow_20260604`, and
  `bufotalin_hybrid_literature_20260603`;
- moved unreferenced top-level old runs, probes, smoke outputs, and stale local
  logs into `results/shared/archive/2026-06-04_cleanup/`;
- moved stale runtime state files such as `heartbeat.json` into
  `results/shared/archive/2026-06-04_cleanup/runtime_state_files/`.

Archive manifests:

- `results/shared/archive/2026-06-04_cleanup/manifest.json`
- `results/shared/archive/2026-06-04_cleanup/runtime_state_files/manifest.json`

Cleanup result:

```text
Moved unreferenced top-level results/shared entries: 138
Kept referenced/current top-level results/shared entries: 109
Moved stale runtime state files from kept entries: 1
```

## Verification

Focused verification run before this audit:

```bash
pytest -q tests/test_literature_template_cards.py tests/test_template_applicability.py tests/test_executable_template_validation.py tests/test_literature_one_step_plugin.py tests/test_literature_template_plugin_benchmark.py
pytest -q tests/test_agent_artifact_contracts.py tests/test_literature_evidence_cards.py tests/test_smiles_first_workflow.py tests/test_chem_enzy_onestep.py tests/test_chem_enzy_native_chemical_plugin.py
pytest -q tests/test_strategic_candidate_generation.py tests/test_smiles_first_workflow.py tests/test_p0_literature_benchmark_pack.py tests/test_literature_template_cards.py tests/test_template_applicability.py tests/test_executable_template_validation.py tests/test_literature_one_step_plugin.py tests/test_literature_template_plugin_benchmark.py
python -m py_compile cascade_planner/agent/literature_templates.py cascade_planner/agent/template_applicability.py cascade_planner/agent/executable_template_validation.py cascade_planner/baselines/literature_one_step_plugin.py cascade_planner/baselines/chem_enzy_adapter.py cascade_planner/baselines/chem_enzy_onestep.py cascade_planner/agent/chem_enzy_policy.py cascade_planner/agent/smiles_first.py scripts/run_literature_template_plugin_benchmark.py
```

Post-cleanup verification should rerun at least the first pytest group after
any further source edits.
