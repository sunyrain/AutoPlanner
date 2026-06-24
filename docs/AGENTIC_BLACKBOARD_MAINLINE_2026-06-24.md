# Agentic Blackboard Mainline

Last update: 2026-06-24.

This is the active AutoPlanner route-control architecture.

## Current Flow

```text
target SMILES
-> deterministic preflight and target profile
-> agentic blackboard state
-> Codex action planner emits a short typed action batch
-> deterministic validator checks safety, budget, source binding, and proof boundaries
-> local tools execute selected actions
-> blackboard records typed summaries and artifact refs
-> final verdict comes only from deterministic parent proof and stock audit
```

The system is policy-driven, not a fixed `failure -> scout -> extract -> rerun`
chain. Codex chooses the next bounded actions from the blackboard; local code
normalizes payloads and enforces proof gates.

## Active Entry Point

```bash
python scripts/run_codex_entry_agentic_blackboard.py \
  --target-name NAME \
  --target-smiles SMILES \
  --codex-action-planner \
  --local-pdf-search-dir /path/to/pdf/cache
```

Useful debugging flags:

- `--stop-on-problem`: stop on planner fallback, invalid batch, rejected action,
  or stale/no-useful-artifact action.
- `--exhaust-round-budget`: continue until the configured round budget when no
  stop condition fires.
- `--codex-action-planner-tools`: controls audited planner search tools.

## Prompt Contract

Codex action planner output must stay compact:

- at most three actions per round;
- typed actions only, no solved verdicts;
- no raw reaction strings or production KB writes;
- brief `rationale`, `expected_artifact`, and `success_condition`;
- skeletal payloads only.

Local repair/builders fill heavy payloads such as source acquisition policies,
guided ChemEnzy policies, child-target policies, and analogical template safety
policies. This avoids long structured-output failures and keeps final authority
out of the prompt.

## Chemenzy Policy

ChemEnzy can run immediately for simple targets as a baseline. For complex
polycyclic, steroid, or natural-product-like targets, a first-round ChemEnzy call
must either be supported by blackboard signal or explicitly be a bounded probe:

```text
initial_probe=true
max_steps <= 6
chem_enzy_iterations <= 10
chem_enzy_expansion_topk <= 20
timeout_s <= 180
max_candidates <= 5
```

Full guided reruns should consume bridge tasks, exact rows, target-side
hypotheses, broad templates, selected analogical templates, or route-objective
signals.

## Literature And PDFs

Online/Codex search is primary. Local PDFs are a cache and fallback after the
agent discovers matching DOI, title, PII, URL, or user-provided source metadata.
If online and local access fail, the system records placeholders or PDF proxy
requests instead of pretending evidence exists.

PDF structure extraction is separate from visual-model budget. Only actual
visual chain extraction and visual structure resolution consume visual calls.

## Solved Boundary

The only path to `solved` is deterministic:

- target equivalence passed;
- parent route verifier accepted;
- stock audit passed;
- no unexplained large atom jump;
- child route, if any, is connected to the parent bridge;
- exact literature segment, if any, is connected to the parent route;
- analogy remains rationale, not proof.

Child-target success never promotes the parent target to solved by itself.

## Core Files

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

## Regression Set

```bash
python -m pytest tests/test_agentic_blackboard_controller.py -q
python -m pytest tests/test_codex_entry_harness_contract.py -q
python -m pytest tests/test_route_objectives.py tests/test_parent_route_proof.py -q
```
