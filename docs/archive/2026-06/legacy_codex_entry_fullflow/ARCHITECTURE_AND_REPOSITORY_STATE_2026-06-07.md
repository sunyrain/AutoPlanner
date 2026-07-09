# AutoPlanner Architecture And Repository State

Last update: 2026-06-07.

This document is the current repository architecture and cleanup record after
the bufotalin exact-compound-11 rerun. It is meant to be the first file to read
before changing the harness, report generation, or ChemEnzy integration.

## Current Position

AutoPlanner is now organized around a Codex-entry route-control harness:

```text
target input
-> deterministic preflight
-> deterministic workflow plan or Codex-selected tool plan
-> local tools execute ChemEnzy, source-detail extraction, route audit, and validation
-> Codex may research and draft structured artifacts
-> deterministic validators decide solved, fake_closed_rejected, partial, or needs_followup
```

ChemEnzy is treated as a route generator and proposal source, not as the final
authority. The final authority is the deterministic verifier/auditor layer.

## Cleanup Performed

Safe generated files removed in this pass:

- all `__pycache__/` directories
- `.pytest_cache/`
- `*.pyc` / `*.pyo`
- `.DS_Store` and editor backup files matching `*~`

Files intentionally not removed:

- `results/shared/**`: run outputs, raw route JSON, verifier JSON, logs, and
  report inputs are audit evidence.
- `docs/bufotalin/report_20260607/**`: generated report, route images, and
  exact-route figures are the current bufotalin deliverables.
- current dirty source/doc deletions and modifications in git status: these
  pre-existed this cleanup pass and should be reviewed as an explicit
  repository-hygiene decision, not silently reverted or deleted.
- vendor/runtime directories: live ChemEnzy requires local vendor code and an
  external runtime environment.

## Active Repository Surface

```text
cascade_planner/
  agent/              Codex-facing schemas, policies, route audit, literature artifacts.
  baselines/          ChemEnzy adapter, route contracts, rescue/plugin logic.
  harness/            Codex-entry controller, tool wrappers, source-detail compiler,
                      raw route verifier, final verdict machinery.
  cascade_search/     Route-tree and cascade-aware search components.
  cascade_verifier/   Rule and learned verifier features.
  route_tree/         Route-tree runtime, proposals, reservoir, verifier.
  web/                Local WebUI and API endpoints.

scripts/
  run_codex_entry_controller.py
  run_chem_enzy_plan_for_web.py
  run_open_structure_template_agent.py
  run_statin_panel_literature_self_evo.py

docs/
  MAINLINE.md
  CODEX_ENTRY_HARNESS_AND_PROMPT_2026-06-06.md
  BUFOTALIN_V0_FULLFLOW_HARNESS_2026-06-06.md
  bufotalin/report_20260607/

tests/
  test_codex_entry_harness_contract.py
  test_open_research_experience.py
  targeted agent/web/literature contract tests

results/shared/
  local run artifacts; ignored by git; keep as evidence, not source code
```

## Main Data Flow

### 1. Target Intake

Primary entry:

```text
scripts/run_codex_entry_controller.py
```

Important harness modules:

- `cascade_planner/harness/preflight.py`
- `cascade_planner/harness/codex_plan.py`
- `cascade_planner/harness/runner.py`
- `cascade_planner/harness/tools.py`
- `cascade_planner/harness/schemas.py`

The controller writes a run directory containing:

- `target_input.json`
- `preflight.json`
- `codex_workflow_plan.json`
- `decision_trace.jsonl`
- `tool_calls.jsonl`
- `artifact_bundle.json`
- `final_verdict.json`
- `progress_panel.html`

### 2. ChemEnzy Native Search

Primary script:

```text
scripts/run_chem_enzy_plan_for_web.py
```

Key adapter:

```text
cascade_planner/baselines/chem_enzy_adapter.py
```

Expected request/result files:

- `chemenzy_request.json`
- `chemenzy_native_raw_result.json`
- `route_verifier_report.json`

The harness restricts default live searches to small/strict stock unless broad
stock is explicitly allowed.

### 3. Raw Route Verification

Primary verifier:

```text
cascade_planner/harness/route_verifier.py
```

The current verifier checks:

- hidden non-stock reactants
- large atom jumps
- advanced same-scaffold terminal leaves
- exact target equivalence through canonical isomeric SMILES and InChIKey
- route product target match audit

This layer exists because `raw_solved=true` from ChemEnzy is not sufficient.

### 4. Literature And Source-Detail Flow

Primary modules:

- `cascade_planner/harness/open_research_contract.py`
- `cascade_planner/harness/open_research_experience.py`
- `cascade_planner/harness/downstream_compiler.py`
- `cascade_planner/harness/source_detail_chain_builder.py`
- `cascade_planner/harness/source_detail_resolution.py`
- `cascade_planner/harness/literature_pdf_extraction.py`
- `cascade_planner/harness/visual_structure_extraction.py`

Outputs are compiled into deterministic consumables:

- source-detail route steps
- executable one-step rows
- literature template plugin flags
- route-expansion child targets
- advisory anchors
- self-evo staging memory

Raw LLM reaction injection remains forbidden. Literature artifacts must pass
schema, structure, source, and applicability checks before they influence
search.

## Bufotalin Current State

The bufotalin route is now a two-part closure:

```text
ChemEnzy verifier closure -> exact compound 11
compound 11 -> 24 -> 25 -> 23 -> 26 -> 27 -> 28 -> 19 -> 20
-> 14 -> 22 -> 30 -> 31 -> 32 -> 33 -> bufotalin
```

Important artifacts:

- `docs/bufotalin/report_20260607/bufotalin_retrosynthesis_report_20260607.pdf`
- `docs/bufotalin/report_20260607/bufotalin_fully_connected_route_graph_20260607.png`
- `results/shared/bufotalin_compound11_exact_chemenzy_rerun_20260607_035728/route_expansion_subgoals/01_compound_11_exact_tet2025_terminal_verifier.json`
- `results/shared/bufotalin_compound11_exact_chemenzy_rerun_20260607_035728/route_expansion_subgoals/01_compound_11_exact_tet2025_terminal_route_audit_compact.json`

Exact compound 11 rerun result:

```text
raw_solved: true
raw_route_count: 59
verifier_accepted: true
accepted_route_count: 15
target_match: true
InChIKey: AEMFNILZOJDQLW-MGGVOPSTSA-N
```

Important caveat: the ChemEnzy terminal closure is planner-level evidence, not
an experimental protocol. The experimental-strength part of the route is the
source-detail literature chain from compound 11 to bufotalin.

## Repository Hygiene State

The working tree is intentionally not force-cleaned. It currently contains a
large pre-existing migration/refocus state:

- many tracked legacy files are marked deleted
- active harness files are untracked under `cascade_planner/harness/`
- current bufotalin report artifacts are untracked under `docs/bufotalin/report_20260607/`
- many historical docs and scripts are removed from the active surface
- `results/`, vendor code, local external workspaces, caches, and archives are
  ignored by `.gitignore`

Before committing, make an explicit staging decision:

1. Keep the Codex-entry harness and its tests.
2. Keep current bufotalin report artifacts or move generated reports to a
   release/export area.
3. Decide whether legacy deleted modules are intentionally removed or should be
   moved to `archive/`.
4. Avoid committing `results/shared/**`, runtime logs, vendor checkouts, or
   credentials.

## Biggest Problems And Blockers

### 1. Scientific route acceptance is still weaker than target closure

The strict verifier now catches target identity, hidden non-stock leaves,
advanced same-scaffold terminals, and large atom jumps. It still cannot prove
that an accepted ChemEnzy route is chemically realistic. The exact compound 11
closure passed the current verifier, but the upstream closure remains a
planner-level hypothesis.

Required next work:

- add reaction-class plausibility scoring for stock-to-steroid construction
- add template provenance and precedent checks for accepted routes
- add a stricter terminal closure tier: commercial stock, known chiral-pool
  material, or literature-backed precursor only

### 2. Report/documentation is slightly stale after the exact 11 rerun

The PDF report was generated before the strict exact compound 11 rerun. The new
fully connected route graph exists, and the exact rerun artifacts exist, but the
main PDF text still reflects the earlier state where terminal exact closure was
not yet complete.

Required next work:

- regenerate the bufotalin report with the exact 11 closure section
- include the fully connected graph in the PDF
- explicitly separate literature-proven route steps from ChemEnzy planner
  closure steps

### 3. Worktree hygiene needs a deliberate commit/archival pass

The repo has a broad refocus in progress: active harness files are untracked
and many legacy modules/scripts/docs are deleted. This is not a cache problem;
it is a product-surface decision.

Required next work:

- stage the active harness/doc/test files together
- either commit tracked deletions as intentional cleanup or move old material
  into ignored archive paths
- keep generated results out of git unless they are curated report exports
- remove or archive residual references to deleted legacy modules before
  calling the deletion pass complete. Current known references include
  `cascade_planner.cascadeboard` and `cascade_planner.eval` imports of
  `cascade_planner.expand`, `cascade_planner.multistep`, and
  `cascade_planner.conditions`; `cascade_planner/web/app.py` also has a
  best-effort route-SVG import of the deleted
  `scripts/render_linear_route_schemes.py`.

### 4. Live ChemEnzy remains environment-bound

The working setup depends on:

- `vendor/ChemEnzyRetroPlanner`
- `/root/autodl-tmp/chem_enzy_runtime/envs/retro_planner_env`
- local model archives and stock files

The latest exact 11 run completed despite a missing EC-context checkpoint
warning, but this means the live runtime is not fully portable.

Required next work:

- add a runtime preflight report listing missing optional and required models
- separate required core-search dependencies from optional enzyme/context
  enrichment dependencies
- add a reproducible environment manifest

### 5. Artifact sprawl is now significant

`results/shared/` contains many bufotalin and atorvastatin probes. They are
useful evidence, but hard to navigate without a manifest.

Required next work:

- create a curated artifact manifest per case
- mark superseded runs explicitly
- keep raw run outputs local but expose only final curated artifacts through
  docs/reports

## Near-Term Action Plan

1. Regenerate the bufotalin PDF with exact compound 11 closure and the fully
   connected graph.
2. Add a strict route-plausibility tier beyond target/stock closure.
3. Create `docs/bufotalin/report_20260607/MANIFEST.md` linking final report,
   exact rerun, graph, and source-detail chain.
4. Decide staging for the Codex-entry harness and tracked legacy deletions.
5. Add ChemEnzy runtime preflight checks and a reproducibility manifest.

## Verification Commands

Targeted harness tests:

```bash
pytest -q tests/test_codex_entry_harness_contract.py tests/test_open_research_experience.py
```

Broad collection check:

```bash
pytest -q
```

ChemEnzy exact compound 11 rerun artifacts:

```text
results/shared/bufotalin_compound11_exact_chemenzy_rerun_20260607_035728/
```
