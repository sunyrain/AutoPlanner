# AutoPlanner Mainline

Last update: 2026-06-24.

The active mainline is the policy-driven agentic blackboard controller.

```text
target input
-> deterministic preflight
-> blackboard state
-> Codex chooses a compact typed action batch
-> deterministic validator checks safety, budget, binding, and proof boundaries
-> local tools execute approved actions
-> blackboard records typed summaries and artifact refs
-> deterministic parent proof emits the final verdict
```

Codex can plan, search, rank, and draft typed artifacts. It cannot directly mark
a case solved, inject raw reactions, write production KB entries, or promote a
child route into a parent solution.

## Active Entry

```bash
python scripts/run_codex_entry_agentic_blackboard.py \
  --target-name NAME \
  --target-smiles SMILES \
  --codex-action-planner
```

Use `--stop-on-problem` for debugging runs where planner fallback, invalid
action batches, rejected actions, or stale actions should stop immediately for
inspection.

## Active Code Surface

- `scripts/run_codex_entry_agentic_blackboard.py`
- `cascade_planner/harness/agentic_blackboard_controller.py`
- `cascade_planner/harness/agent_action_planner.py`
- `cascade_planner/harness/codex_action_planner.py`
- `cascade_planner/harness/agentic_blackboard.py`
- `cascade_planner/harness/route_objectives.py`
- `cascade_planner/harness/analogical_reaction_templates.py`
- `cascade_planner/harness/parent_route_proof.py`
- `cascade_planner/agent/codex_worker.py`
- `cascade_planner/agent/action_contracts.py`

## Current Prompt Rule

Codex action planner output must be compact. It should emit action skeletons,
not full downstream policies. Local repair/builders complete:

- source acquisition policy;
- guided ChemEnzy policy;
- child target policy;
- analogical template safety policy;
- stitch/proof payload boundaries.

This avoids structured-output truncation and keeps final authority deterministic.

## Chemenzy Boundary

Simple targets may run immediate baseline Chemenzy. Complex steroid,
polycyclic, or natural-product-like targets require blackboard signal before a
full guided rerun. A first-round complex target probe is allowed only when
explicitly bounded as an initial probe.

## Supporting Docs

- `docs/AGENTIC_BLACKBOARD_MAINLINE_2026-06-24.md`
- `docs/CODEX_WELLAU_STREAMING_RUNBOOK_2026-06-05.md`
- `docs/archive/2026-06/legacy_codex_entry_fullflow/`

Older fixed-chain fullflow and SMILES-first documents are historical context
only.

## Minimum Verification

Before pushing controller or prompt edits, run:

```bash
python -m pytest tests/test_agentic_blackboard_controller.py -q
python -m pytest tests/test_codex_entry_harness_contract.py -q
```

When touching proof/objective/template surfaces, also run:

```bash
python -m pytest tests/test_route_objectives.py tests/test_parent_route_proof.py -q
```
