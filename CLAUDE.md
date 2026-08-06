# Repository Guidance

This repository is a bounded V4 retrosynthesis system. The canonical operator
surface is `python -m cascade_planner`; the active architecture is documented
in `docs/MAINLINE.md`, `docs/RUNBOOK.md`, and
`docs/architecture/CURRENT_ARCHITECTURE_STATUS.md`.

## Canonical Commands

```bash
# Inspect the current V4 surface
python -m cascade_planner --help
python -m cascade_planner audit

# Create and operate one bounded run
python -m cascade_planner run --run-id target-001 --target-name TARGET --target-smiles SMILES
python -m cascade_planner status target-001
python -m cascade_planner validate target-001
python -m cascade_planner replay target-001
python -m cascade_planner export target-001 --output-dir local-export

# Fresh target-only campaign
python -m cascade_planner solve-target --target-name TARGET --target-smiles SMILES --run-id target-blind-001

# Serve the canonical Web/API surface
python -m cascade_planner serve
```

V4 owns one `RunKernel`, one canonical retrosynthesis hypergraph, one deficit
frontier, one proof/acceptance pipeline, and one Workbench projection. CLI,
API, Web, replay, and export are adapters over those state owners; they must
not create parallel chemistry state.

## Package Boundaries

- `cascade_planner/application/`: canonical run, graph, frontier, proof,
  acceptance, and program services.
- `cascade_planner/interfaces/`: evidence, stock, source, target, and delivery
  contracts.
- `cascade_planner/orchestration/`: the V4 global campaign director.
- `cascade_planner/route_tree/`: bounded search, candidate admission, and
  route-tree runtime contracts.
- `cascade_planner/cascade_search/`: reusable proposal, evidence, and model
  contracts used by the current search path.
- `cascade_planner/cascadeboard/`: current route-scoring, skeleton, and
  benchmark adapters that remain explicitly consumed by V4.
- `cascade_planner/eval/`: current V4 data/pack builders, artifact trainers,
  benchmark gates, and conservative route audits. See its README for the
  retained CLI contract list.
- `cascade_planner/research/`: non-authoritative current research workers.
- `cascade_planner/legacy/`: frozen V3, K2, Phase-II, report, replay, and
  compatibility code. It is never imported by a fresh V4 run.

## Legacy Rules

Historical entrypoints are physically moved under explicit legacy namespaces;
old import paths are not kept as shims. Direct execution of archived research
requires:

```powershell
$env:AUTOPLANNER_ALLOW_LEGACY_RESEARCH = "1"
```

Use the old paths only to reproduce named reports or inspect saved runs. The
old `expand`, `conditions`, `multistep`, and K2 report-card packages are not
current modules. Their archived entrypoints live under
`cascade_planner.legacy.eval_runtime`.

## Verification

Focused contract checks:

```bash
python -m pytest tests/test_legacy_namespace.py tests/test_runtime_model_contracts.py -q
ruff check cascade_planner/eval --isolated --select E4,E7,E9,F
```

Canonical V4 import must leave both `cascade_planner.legacy` and
`cascade_planner.research` absent from `sys.modules`. New features must enter
through current V4 contracts and tests, not by reviving archived package paths.
