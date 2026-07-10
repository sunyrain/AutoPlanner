# Repository Surface and Hygiene

Last audited: 2026-07-10.

This page defines where new work belongs. It is a boundary map, not a claim
that every historical package has already been migrated.

## Surface classification

| Class | Current paths | Policy |
| --- | --- | --- |
| Active mainline | `cascade_planner/agent/`, `cascade_planner/harness/`, `cascade_planner/orchestration/`, `cascade_planner/routes/`, `cascade_planner/baselines/`, `cascade_planner/web/`, current launchers in `scripts/`, and `tests/` | New Codex-driven retrosynthesis, evidence fusion, validation, route-forest, and UI work belongs here. Entry points must be documented and covered by a contract test. |
| Maintained compatibility and research support | `cascade_planner/cascadeboard/`, `cascade_planner/route_tree/`, `cascade_planner/vnext/`, `cascade_planner/cascade_search/`, `cascade_planner/cascade_verifier/`, and `cascade_planner/eval/` | These packages still supply deterministic tools, models, or older route contracts. Do not call them the controller mainline. Remove or relocate them only after imports, saved-artifact readers, and regression coverage have been audited. |
| Opt-in legacy experiment | `AUTOPLANNRELLM/` | Independent DeepSeek-era experiment retained for compatibility and provenance. It is not the Codex/blackboard mainline; new orchestration must not be added here. |
| Tracked archive | `docs/archive/`, `scripts/archive/`, and `archive/code/frozen_research_2026-05-20/` | Historical evidence only. Active code must not import from these locations. Fix factual provenance in place, but put current guidance in the active docs. |
| Local runtime artifacts | `results/`, `workspace/`, `releases/`, local `vendor/` checkouts, `data_external/`, model weights, root-level PDFs, `.mar` files, caches, and generated datasets | Git-ignored and reproducible or externally sourced. Never make runtime success depend on an unrecorded local artifact; record acquisition/configuration instructions instead. |

The authoritative controller direction remains [MAINLINE.md](MAINLINE.md).
The lists above deliberately preserve compatibility surfaces until callers are
measured; they are not permission to grow a second orchestration stack.

## Audit snapshot

- The tracked repository root contained seven small text/configuration files
  before this audit. `requirements-dev.txt` is the only new root file.
- Root-level PDFs, model archives, generated datasets, credentials, and
  `results/` are ignored. This cleanup did not delete or relocate any of them.
- `pytest.ini` scopes discovery to `tests/`, adds the repository root to the
  import path, and excludes archive, result, vendor, and external-data trees.
- The Parquet-backed enzyme retrieval modules import `pyarrow` at module load;
  it is therefore declared in `requirements.txt` rather than treated as an
  undocumented optional package.
- Historical documentation is the largest tracked area (about 40 MB in this
  checkout), including PDFs, slide decks, and verification renders. Those
  files were left intact because provenance and downstream links need an audit
  before any storage migration.
- Python bytecode and pytest caches are disposable. This audit removed 38 such
  directories. Each normalized absolute path was verified to be below the
  repository root, Git-ignored, non-symlinked, and free of tracked files before
  recursive deletion.

## Contribution rules

1. Keep source, tests, documentation, and runtime output separate. Write run
   output below `results/shared/`, not beside source files.
2. Never commit tokens, credentials, browser profiles, local PDFs, model
   checkpoints, vendor repositories, or generated benchmark output.
3. Add a documented launcher only when it represents a supported workflow.
   One-off inspection and migration scripts should be retired to an archive
   after their result is captured.
4. Prefer one typed artifact contract per concept and adapters at legacy
   boundaries. Do not duplicate controller state across packages.
5. Install `requirements.txt` plus `requirements-dev.txt` in an isolated Python
   3.11/3.12 environment, then invoke tests with `python -m pytest`.

## Staged migration backlog

These are follow-up migrations, not safe mechanical cleanup:

1. Build an import/call graph for the compatibility packages and assign each
   one an owner: active deterministic tool, read-only artifact adapter, or
   removable legacy implementation.
2. Inventory launchers against `scripts/README.md`; add deprecation notices for
   superseded entry points before moving them to `scripts/archive/`.
3. Split heavyweight and platform-specific inference dependencies into named
   profiles, then generate tested lock/constraint files for CPU and CUDA
   environments.
4. Decide whether large tracked historical binaries remain in Git, move to Git
   LFS, or become checksummed release assets. Preserve links and hashes during
   that migration.
5. Retire `AUTOPLANNRELLM/` only after its route-tree imports and saved results
   have explicit compatibility coverage.
