# AutoPlanner Mainline

Last update: 2026-06-05.

This file is the active repository authority for the next implementation phase.

## Main Conclusion

The next mainline is a Codex-entry harness:

```text
target input
-> deterministic preflight
-> Codex controller decides the workflow
-> local tools execute ChemEnzy, route audit, frontier extraction, structure checks
-> Codex performs open literature/source reasoning
-> deterministic validators emit final verdict
```

ChemEnzy is a strong tool, not the top-level authority. The Bufotalin replay
showed why: ChemEnzy can emit many raw routes while the audited state remains
`fake_closed_rejected`. Codex should see the target and preflight context early
enough to choose ChemEnzy-first, literature-first, or hybrid execution.

## Hard Boundaries

- Codex may choose tools and write draft artifacts.
- Codex may research literature and classify evidence.
- ChemEnzy may generate route candidates.
- Deterministic validators decide `solved`, `partial`, `rejected`, or
  `needs_followup`.
- No LLM-only solved claim.
- No raw LLM reaction injection.
- No production KB promotion without schema, structure, route, and evidence
  validation.

## Active Code Surface

Primary implementation anchors:

- `cascade_planner/agent/codex_controller.py`
- `cascade_planner/agent/codex_worker.py`
- `cascade_planner/agent/artifact_schemas.py`
- `cascade_planner/agent/artifact_validators.py`
- `cascade_planner/agent/case_blackboard.py`
- `cascade_planner/agent/evidence_cards.py`
- `cascade_planner/agent/literature_research.py`
- `cascade_planner/agent/literature_segments.py`
- `cascade_planner/agent/literature_templates.py`
- `cascade_planner/agent/smiles_first.py`
- `cascade_planner/agent/route_auditor.py`
- `cascade_planner/agent/target_profile.py`
- `cascade_planner/agent/template_applicability.py`
- `cascade_planner/agent/executable_template_validation.py`
- `cascade_planner/baselines/chem_enzy_adapter.py`
- `cascade_planner/baselines/route_contract.py`
- `cascade_planner/web/app.py`
- `scripts/run_open_structure_template_agent.py`
- `scripts/run_bufotalin_fullflow_wellau.py`
- `scripts/run_smiles_first_literature_workflow.py`
- `scripts/run_chem_enzy_plan_for_web.py`

Strategic-disconnection curated sources remain active under:

```text
data/strategic_disconnections/
```

## Harness To Build Next

Create a Codex-entry controller harness with fixed contracts:

```text
scripts/run_codex_entry_controller.py
cascade_planner/harness/
tests/test_codex_entry_harness_contract.py
```

Expected run artifacts:

- `decision_trace.jsonl`
- `codex_events.jsonl`
- `tool_calls.jsonl`
- `route_audit.json`
- `frontier_report.json`
- `artifact_bundle.json`
- `final_verdict.json`
- `progress_panel.html`

The first regression targets should include Bufotalin as a hard negative:
expected verdict `partial_anchor_only_not_solved`, not solved.

## Supporting Docs

- `docs/CODEX_WELLAU_STREAMING_RUNBOOK_2026-06-05.md`
- `docs/SMILES_FIRST_LITERATURE_STRATEGIC_WORKFLOW_2026-06-03.md`
- `docs/LITERATURE_TO_EXECUTABLE_TEMPLATE_CHECKLIST_2026-06-04.md`

Older plans, paper materials, statin panel runs, ONMT/training scripts, route
renderers, and benchmark scaffolds are outside the current harness direction
unless explicitly restored.

## Minimum Verification

Before pushing harness edits, run:

```bash
PYTHONPATH=. pytest --collect-only -q
python -m py_compile scripts/run_open_structure_template_agent.py \
  scripts/run_smiles_first_literature_workflow.py \
  scripts/run_chem_enzy_plan_for_web.py
```

Run targeted tests when touching their surfaces:

```bash
pytest -q tests/test_codex_worker_controller_evolution.py \
  tests/test_agent_artifact_contracts.py \
  tests/test_agent_route_auditor_condition.py \
  tests/test_literature_evidence_cards.py \
  tests/test_smiles_first_workflow.py \
  tests/test_web_app.py
```
