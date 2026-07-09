# Codex WellAU Streaming Runbook

Last update: 2026-06-05.

This is the fixed contract for open Codex structure/template agents that use
WellAU through the Codex CLI. It records the successful Bufotalin run and the
wrong modes that were cleaned out after the run.

## Canonical Contract

Use `scripts/run_open_structure_template_agent.py` and keep these invariants:

- Provider config must be explicit:
  - `model_provider = "wellau"`
  - `[model_providers.wellau]`
  - `base_url = "https://api.wellau.com/v1"`
  - `env_key = "OPENAI_API_KEY"`
  - `wire_api = "responses"`
- The child environment must remove ambient `OPENAI_BASE_URL`.
- The Codex CLI must run with `exec --json`, producing `codex_events.jsonl`.
- Usage is read from `codex_events.jsonl` at `turn.completed.usage`.
- On this host, use `--sandbox bypassed`; `workspace-write` hits bwrap
  namespace errors before local tools can execute.

Do not conflate these two layers:

- `wire_api = "responses"` is the provider/API transport.
- `codex exec --json` is the Codex CLI event stream to local JSONL.

Both are required for the audited streaming run record.

## Canonical Success Run

Retained canonical run:

```text
results/shared/bufotalin_open_structure_template_agent_stream_fixed_20260605_044433
```

Required success signals from that run:

- `open_agent_run_record.json` has `exit_code: 0`.
- `open_agent_run_record.json` has `streaming.mode = "codex_exec_jsonl"`.
- `codex_events.jsonl` contains `turn.completed`.
- `turn.completed.usage` exists.
- Final bundle files exist:
  - `structure_template_report.md`
  - `structure_template_candidates.json`
  - `evidence/literature_sources.json`
  - `evidence/pubchem_validated_compounds.json`
  - `validated_compounds.smi`
  - `open_agent_audit.json`

The Bufotalin chemistry verdict remains partial planning material only:
`AutoPlanner solved = false`, no raw reaction SMILES promoted, and no production
KB promotion.

## Cleaned Wrong Practices

The previous wrong or superseded runs were removed from `results/shared` and
quarantined under:

```text
results/quarantine/codex_wellau_wrong_practices_20260605
```

Manifest:

```text
results/quarantine/codex_wellau_wrong_practices_20260605/cleanup_manifest.json
```

Cleaned categories:

- `workspace-write` sandbox run that failed on bwrap namespace creation.
- Bypassed sandbox run without Codex CLI JSONL streaming.
- Incomplete repro setup without exit code or usage.
- Early JSONL attempt without `turn.completed.usage`.
- Temporary WellAU stream-key and responses-stream smoke probes that were not
  the Codex CLI audited event stream.

## Command Pattern

The launcher now defaults to JSONL streaming. This is the preferred pattern:

```bash
python scripts/run_open_structure_template_agent.py \
  --output-dir results/shared/<target>_open_structure_template_agent_<stamp> \
  --target-name <TargetName> \
  --target-smiles '<SMILES>' \
  --frontier-smiles '<SMILES>' \
  --context-root results/shared/<context_run> \
  --timeout-s 1800 \
  --sandbox bypassed
```

Do not add `--no-stream-jsonl` for audited runs. That disables the event stream
where `turn.completed.usage` is recorded.

## Post-Run Checks

Use these checks before accepting a run:

```bash
grep -n 'turn.completed\|usage' <run_dir>/codex_events.jsonl
python -m json.tool <run_dir>/open_agent_run_record.json >/dev/null
python -m json.tool <run_dir>/structure_template_candidates.json >/dev/null
```

Expected:

- `turn.completed.usage` is present.
- `open_agent_run_record.json` records `wire_api = "responses"`.
- `open_agent_run_record.json` records `stream_jsonl = true`.
- `open_agent_run_record.json` records `streaming.event_summary.turn_completed = true`.

If any of those fail, treat the run as a wrong practice and do not leave it in
`results/shared`.
