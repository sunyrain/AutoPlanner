# Codex Entry Harness And Prompt - 2026-06-06

This document summarizes the current AutoPlanner Codex-entry harness, the open-structure research prompt contract, and the downstream handoff path used in the bufotalin audit.

## Scope

The harness has two Codex roles:

1. `codex_plan.py`: a bounded workflow planner. It returns only a JSON workflow plan.
2. `run_open_structure_template_agent.py`: an open research/extraction agent. It writes audited structure/template/downstream artifacts under one run directory.

Deterministic validators, not Codex, decide final route status.

## Controller Flow

Entry point:

```text
scripts/run_codex_entry_controller.py
```

Core runner:

```text
cascade_planner/harness/runner.py
```

High-level sequence:

1. Write `target_input.json` and `budget.json`.
2. Run preflight and write `preflight.json`.
3. Obtain workflow plan:
   - live Codex planner via `plan_workflow_with_codex`, or
   - deterministic offline plan via `deterministic_workflow_plan`.
4. Execute planned local tools in order.
5. Validate artifact bundle.
6. Emit final verdict.
7. Write progress panel and final run artifacts.

Default deterministic hard-case sequence:

```text
run_chemenzy
audit_route_and_extract_frontier
run_smiles_first_literature_workflow
run_open_structure_research_agent
run_guided_chemenzy_rerun
run_route_expansion_subgoal_search
run_self_evo_replay_gate
validate_artifact_bundle
emit_final_verdict
```

The final verdict is emitted from `artifact_bundle`, route audits, route verifier reports, and bundle validations. Codex planner/open-agent output cannot directly claim `solved`.

## Planner Prompt

File:

```text
cascade_planner/harness/codex_plan.py
```

Planner prompt role:

- Choose strategy and ordered local tools only.
- Return exactly one JSON object.
- Do not solve chemistry.
- Do not emit raw reaction SMILES.
- Do not claim solved verdict.
- Use only allowed local tools.

Allowed strategies:

```text
chem_enzy_first
literature_first
hybrid
reject_invalid_input
```

Allowed tools:

```text
run_chemenzy
audit_route_and_extract_frontier
run_smiles_first_literature_workflow
run_open_structure_research_agent
run_guided_chemenzy_rerun
run_route_expansion_subgoal_search
run_self_evo_replay_gate
validate_artifact_bundle
emit_final_verdict
```

Planner schema:

```text
codex_entry_workflow_plan.v1
```

Required keys:

```text
schema_version
case_id
recommended_strategy
planned_tools
rationale
risk_flags
expected_verdict_floor
```

## Open-Structure Prompt

File:

```text
scripts/run_open_structure_template_agent.py
```

Prompt builder:

```text
_read_or_build_prompt()
```

The open agent is instructed to produce downstream-consumable planning assets, not just narrative reports.

Target output classes:

- structure/template candidates
- guided ChemEnzy rerun requests
- literature template cards
- literature route segments
- executable template extraction tasks
- `source_detail_route_steps`
- route expansion tasks
- self-evolution candidates

Core prompt constraints:

- Do not read the large raw ChemEnzy route dump by default.
- Use route verifier/audit/failure feedback summaries for route status.
- Read `open_research_manifest.json` first.
- Treat the manifest as the binding search budget and source-order contract.
- Use harness-owned retrieval artifacts as the default retrieval boundary.
- Shell/Python are allowed only for local deterministic transformation, JSON/schema checks, and RDKit validation over local/manifest-listed data.
- Do not use shell HTTP retrieval (`curl`, `wget`, `urllib`, `requests`, `httpx`) for PubChem, CrossRef, PubMed, patent, DOI, or web retrieval.
- Do not run environment discovery/probing commands.
- Do not store full text, copied procedures, raw reactions, or production KB writes.
- Do not mark solved.
- Do not promote production KB.

Required open-agent artifacts:

```text
structure_template_report.md
structure_template_candidates.json
downstream_consumables.json
evidence/literature_sources.json
evidence/pubchem_validated_compounds.json
validated_compounds.smi
open_agent_audit.json
```

Required JSON schemas:

```text
open_structure_template_candidates.v1
open_downstream_consumables.v1
open_literature_sources.v1
open_pubchem_validated_compounds.v1
open_structure_agent_audit.v1
```

## Source-Detail Extraction

The key path for turning literature into executable rows is:

```text
source_detail_curator_records.json
  -> source_detail_resolution_pack.json
  -> downstream_consumables.source_detail_route_steps
  -> compiled one_step_rows
  -> ChemEnzy literature_template_plugin
```

Codex may write:

```text
evidence/source_detail_curator_records.json
```

Supported schema:

```text
source_detail_curator_records.v1
```

Supported record styles:

- `source_detail_curator_record.v1` with nested `steps`
- direct `source_detail_route_step.v1` shaped records
- Codex source-text translations with `provenance: codex_source_text_translation`

Required fields for exact source-detail steps:

```text
step_id
segment_id
product_smiles
reactant_smiles
source_ref
evidence_refs
relation_type=exact
applicability.product_reconstruction_passed
condition_candidate
```

For `codex_source_text_translation`, additional fields are required:

```text
structure_derivation
source_excerpt
```

The resolver rejects records that:

- contain raw reaction strings
- store full text or procedure text
- request production writes
- lack product/reactant SMILES
- fail RDKit SMILES validation
- lack source refs, evidence refs, or source-grounded condition fields

## Retrieval And Manifest Boundary

Open research run writes/uses:

```text
open_research_manifest.json
evidence/harness_retrieval_prefetch.json
evidence/source_detail_extraction_pack.json
evidence/source_detail_resolution_pack.json
evidence/source_material_locator_pack.json
```

The manifest includes:

- target identity
- local context summaries
- route failure feedback
- runtime capabilities
- shell/retrieval policies
- query plan
- source-detail extraction queue
- source-material locator hints

Important boundary rule:

Metadata-only DOI/publisher/SI/material URLs are not route evidence. They are pointers for structured extraction into `source_detail_curator_records.v1`.

## Downstream Compiler

File:

```text
cascade_planner/harness/downstream_compiler.py
```

Input:

```text
downstream_consumables.json
```

Compiled artifacts:

```text
compiled_downstream_consumables.json
compiled_guided_chemenzy_requests.json
compiled_route_expansion_tasks.json
compiled_literature_template_plugin.json
compiled_executable_template_maturity.json
self_evo_staging_kb.json
```

Compiler outputs:

- guided ChemEnzy search policies
- route expansion tasks and child targets
- literature template plugin flags
- one-step rows when exact executable candidates exist
- executable template maturity report
- self-evo staging KB
- follow-up actions
- rejected items

Important maturity gate:

`one_step_rows` are emitted only when product/reactant structures are source-grounded, RDKit-valid, exact relation, and pass product reconstruction/applicability checks.

## Handoff Fix

A key bufotalin issue was that downstream consumers read the harness wrapper:

```text
compiled_downstream_harness_result.v1
```

instead of the actual payload:

```text
compiled_downstream_consumables.v1
```

This caused false skips:

- `guided_policy_missing`
- `route_expansion_child_targets_missing`
- `self_evo_staging_missing`

The current helper now unwraps `artifact_refs.compiled_downstream_consumables` and falls back to:

```text
open_structure_research/compiled_downstream_consumables.json
```

After the fix, old bufotalin compiled assets can be consumed:

```json
{
  "compiled_schema": "compiled_downstream_consumables.v1",
  "template_card_count": 6,
  "one_step_row_count": 0,
  "staging_candidate_count": 6,
  "route_expansion_target_count": 1
}
```

## Tool Behavior

### `run_chemenzy`

Runs ChemEnzy native/backend search and writes raw result plus route verifier data. Large/advanced hard cases use strict building-block stock defaults.

### `audit_route_and_extract_frontier`

Audits raw routes, rejects fake closure, writes:

```text
route_verifier_report.json
route_audit.json
route_failure_feedback.json
```

### `run_smiles_first_literature_workflow`

Runs local SMILES-first/literature workflow and emits partial anchors/candidates.

### `run_open_structure_research_agent`

Launches the open Codex research agent in an isolated run directory. It:

- writes/validates required open-research artifacts
- extracts open-research experience
- compiles downstream consumables
- allows continuation when the open agent fails but compiled downstream handoff is accepted

### `run_guided_chemenzy_rerun`

Reads compiled guided policy and optional literature template plugin flags, merges route failure feedback blacklist, and reruns ChemEnzy.

### `run_route_expansion_subgoal_search`

Reads compiled route expansion child targets. If no explicit child targets exist, it can use accepted task `frontier_smiles` as a stuck-node fallback target.

### `run_self_evo_replay_gate`

Reads compiled self-evo staging KB and writes replay report plus reusable memory. Production remains blocked for target runs.

### `validate_artifact_bundle`

Rejects unsafe bundles:

- fake closure evidence
- open research failures
- guided verifier rejections
- unsafe self-evo production writes
- raw reaction injection

## Bufotalin Current Status

The bufotalin fullflow run remains not solved:

```text
verdict: fake_closed_rejected
solved: false
stock_audit_passed: false
```

Root cause:

- native ChemEnzy returned routes
- verifier accepted zero routes
- closure used an advanced same-scaffold Deacetylbufotalin-like terminal

Downstream assets are useful but advisory:

```json
{
  "guided_policy_count": 2,
  "template_card_count": 6,
  "route_expansion_task_count": 1,
  "self_evo_staging_candidate_count": 6,
  "one_step_row_count": 0
}
```

Interpretation:

- guided policies can rerun ChemEnzy with better priors and terminal blacklist
- template cards are `advisory_only`
- route expansion can rerun the stuck frontier
- self-evo can keep staging candidates
- no executable reaction rows exist yet

## Field Extraction Probe

A short bufotalin Codex field-extraction probe was run after the handoff fix.

Result:

- Codex did not generate `evidence/source_detail_curator_records.json` within the short timeout.
- `source_detail_route_step_count` remained `0`.
- It preserved source-detail gaps and did not fabricate SMILES.
- It again triggered a `large_raw_artifact_dump` boundary issue by reading a large case bundle.

Conclusion:

The source-detail extraction path is implemented and tested, but bufotalin needs a narrower source-detail worker/prompt. Letting the open agent freely inspect broad case artifacts is too slow and can violate context-boundary rules.

## Recommended Next Prompt Shape

For bufotalin source-detail extraction, use a narrower worker than the general open agent.

Input only:

```text
open_research_manifest.json
evidence/source_detail_extraction_pack.json
evidence/source_material_locator_pack.json
selected source_material_cache/*.json
route_failure_feedback.json
```

Explicitly forbid:

```text
case_bundle.json
chemenzy_native_raw_result.json
large harness_retrieval_prefetch.json dumps
recursive file discovery
raw HTTP retrieval
```

Worker output should be only:

```text
evidence/source_detail_curator_records.json
source_detail_extraction_report.md
```

Target task:

1. Pick one DOI/material record.
2. Determine whether exact product/reactant structures are actually present.
3. If yes, emit one `source_detail_curator_record.v1`.
4. If no, emit a structured gap with precise missing fields.
5. Stop.

This keeps Codex in field-extraction mode instead of broad research mode.

## Current Verification

Relevant tests passing:

```text
tests/test_codex_entry_harness_contract.py: 43 passed, 2 skipped
tests/test_open_research_experience.py: 40 passed
```
