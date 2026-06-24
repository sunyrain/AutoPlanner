# Bufotalin v0 Fullflow Harness Run - 2026-06-06

## Summary

This run is the first end-to-end v0 harness replay for Bufotalin after tightening the raw route verifier.

Run directory:

```text
results/shared/bufotalin_v0_fullflow_harness_20260606_073155
```

Final verdict:

```json
{
  "verdict": "fake_closed_rejected",
  "route_status": "fake_closed_rejected",
  "solved": false,
  "stock_audit_passed": false
}
```

The run did not solve Bufotalin. That is the expected safe behavior for this hard case: ChemEnzy produced route candidates, but the harness rejected all stock-closed claims as fake closure due to advanced same-scaffold terminal use and large atom jumps.

## Input

Target:

```text
bufotalin_v0_fullflow_20260606
```

Target SMILES:

```text
CC(=O)O[C@H]1C[C@@]2([C@@H]3CC[C@@H]4C[C@H](CC[C@@]4([C@H]3CC[C@@]2([C@H]1C5=COC(=O)C=C5)C)C)O)O
```

Family hint:

```text
bufotalin, bufadienolide, steroid, C17 2-pyrone, bufalin
```

Preflight:

- accepted: `true`
- canonical SMILES: `CC(=O)OC1CC2(O)C3CCC4CC(O)CCC4(C)C3CCC2(C)C1c1ccc(=O)oc1`
- InChIKey: `VOZHMAYHYHEWBW-NVOOAVKYSA-N`
- heavy atoms: `32`
- initial risk flags: `polycyclic_or_steroid_like`

## Command

```bash
python scripts/run_codex_entry_controller.py \
  --target-name bufotalin_v0_fullflow_20260606 \
  --target-smiles 'CC(=O)O[C@H]1C[C@@]2([C@@H]3CC[C@@H]4C[C@H](CC[C@@]4([C@H]3CC[C@@]2([C@H]1C5=COC(=O)C=C5)C)C)O)O' \
  --family-hint 'bufotalin, bufadienolide, steroid, C17 2-pyrone, bufalin' \
  --output-dir results/shared/bufotalin_v0_fullflow_harness_20260606_073155 \
  --offline-planner \
  --timeout-s 1800 \
  --chem-enzy-timeout-s 1200 \
  --open-research-timeout-s 900 \
  --guided-chemenzy-timeout-s 900 \
  --smiles-first-timeout-s 600 \
  --max-route-expansion-subgoal-runs 2
```

Planner mode:

- `--offline-planner`
- deterministic v0 workflow plan
- local tools still executed live

Planned strategy:

```text
hybrid
```

Planned tools:

1. `run_chemenzy`
2. `audit_route_and_extract_frontier`
3. `run_smiles_first_literature_workflow`
4. `run_open_structure_research_agent`
5. `run_guided_chemenzy_rerun`
6. `run_route_expansion_subgoal_search`
7. `run_self_evo_replay_gate`
8. `validate_artifact_bundle`
9. `emit_final_verdict`

## Tool Results

| # | Tool | Status | Elapsed | Key result |
|---|---|---:|---:|---|
| 1 | `run_chemenzy` | accepted | 275.293s | returned `213` routes, verifier accepted `0` |
| 2 | `audit_route_and_extract_frontier` | accepted | 1.449s | raw route claims rejected as fake closure |
| 3 | `run_smiles_first_literature_workflow` | accepted | 0.375s | produced partial anchor / literature package |
| 4 | `run_open_structure_research_agent` | rejected | 825.239s | boundary violation: large raw artifact dump; nonzero exit |
| 5 | `run_guided_chemenzy_rerun` | accepted | 0.0s | skipped: `guided_policy_missing` |
| 6 | `run_route_expansion_subgoal_search` | accepted | 0.0s | skipped: `route_expansion_child_targets_missing` |
| 7 | `run_self_evo_replay_gate` | accepted | 0.0s | skipped: `self_evo_staging_missing`; production write blocked |
| 8 | `validate_artifact_bundle` | rejected | 0.044s | fake closure evidence and open research boundary failure |

## ChemEnzy Native Search

Request settings:

- backend: `chem_enzy_native`
- device: `cpu`
- search preset: `thorough`
- max steps: `20`
- iterations: `50`
- expansion top-k: `100`
- stock mode: `building-block`
- effective stock: `PaRotes_n1-stock`

Native result:

- `ok`: `true`
- `n_results`: `213`
- depth attempt status: `solved`
- first successful route appeared in native backend, but harness verification rejected all solved claims.

Raw route verifier:

```json
{
  "accepted": false,
  "route_status": "fake_closed_rejected",
  "route_count": 213,
  "accepted_route_count": 0,
  "rejected_route_count": 213,
  "reasons": [
    "advanced_same_scaffold_terminal",
    "large_atom_jump",
    "no_verifier_accepted_stock_closed_route"
  ]
}
```

Important detail after the verifier fix:

- `hidden_nonstock_count` is `0` for the inspected rejected routes.
- The harness no longer rejects normal internally generated intermediates as hidden non-stock reactants.
- Rejection is now driven by true terminal and structural checks.

Top rejected route:

```json
{
  "route_rank": 0,
  "score": 0.9,
  "n_steps": 1,
  "terminal_count": 2,
  "reasons": ["advanced_same_scaffold_terminal"]
}
```

Rejected advanced terminal:

```text
C[C@]12CC[C@H](O)C[C@H]1CC[C@@H]1[C@@H]2CC[C@]2(C)[C@@H](c3ccc(=O)oc3)[C@@H](O)C[C@]12O
```

Terminal summary:

- heavy atoms: `29`
- target similarity: `0.7656`
- reason: `advanced_same_scaffold_terminal`

This is consistent with the Bufotalin hard-case pattern: the native route closes by treating a Deacetylbufotalin-like advanced same-scaffold precursor as if it were an acceptable terminal.

## Route Audit

Route audit:

```json
{
  "route_status": "fake_closed_rejected",
  "stock_audit_passed": false,
  "fake_closure_rejected": true,
  "target_match": true,
  "step_structural_audit": "failed",
  "condition_status": "condition_gap",
  "next_action": "diagnose_failure_and_blacklist_terminal"
}
```

Route audit reasons:

- `advanced_same_scaffold_terminal`
- `large_atom_jump`
- `no_verifier_accepted_stock_closed_route`
- `route_verifier_rejected_raw_routes`

## Route Failure Feedback

The harness compiled route failure feedback successfully.

Source route status:

```text
fake_closed_rejected
```

Active failure modes:

- `advanced_same_scaffold_terminal`
- `large_atom_jump`
- `no_verifier_accepted_stock_closed_route`

Terminal blacklist:

```text
C[C@]12CC[C@H](O)C[C@H]1CC[C@@H]1[C@@H]2CC[C@]2(C)[C@@H](c3ccc(=O)oc3)[C@@H](O)C[C@]12O
```

Query hints:

- terminal blacklist intermediate search
- upstream intermediate synthesis / large skeleton construction search

This feedback is suitable for guiding later runs away from treating the same advanced Bufotalin-like precursor as a terminal stock closure.

## Smiles-First Literature Workflow

`run_smiles_first_literature_workflow` completed successfully and emitted a guarded partial-anchor package.

Validation:

- accepted: `true`
- route status: `partial_anchor`
- guards:
  - `forward_surrogate_not_lab_procedure`
  - `p0_not_solved_without_stock_audit`
  - `route_anchor_not_stock`

Artifacts include:

- `smiles_first_literature_workflow/summary.md`
- `smiles_first_literature_workflow/frontier_report.json`
- `smiles_first_literature_workflow/validation.json`
- `smiles_first_literature_workflow/bufotalin_v0_fullflow_20260606_hybrid_retrosynthesis_route.json`
- `smiles_first_literature_workflow/bufotalin_v0_fullflow_20260606_strategic_disconnection_cards.jsonl`

Interpretation:

- The literature workflow can describe strategic anchors for Bufotalin-like chemistry.
- It does not provide stock-closed executable synthesis.
- It correctly remains below a solved verdict.

## Open Structure Research

`run_open_structure_research_agent` produced useful downstream artifacts, but the tool was rejected by boundary validation.

Status:

```json
{
  "accepted": false,
  "status": "failed",
  "reasons": [
    "open_agent_boundary_violation:context_boundary:large_raw_artifact_dump",
    "open_structure_research_nonzero_exit"
  ]
}
```

The agent completed a handoff and wrote outputs, but violated the context boundary by reading a large raw artifact dump:

```text
sed -n '1,280p' evidence/harness_retrieval_prefetch.json
```

Compiled downstream summary was still available:

```json
{
  "guided_policy_count": 2,
  "template_card_count": 6,
  "route_expansion_task_count": 1,
  "self_evo_staging_candidate_count": 6,
  "one_step_row_count": 0
}
```

The open agent conclusion was conservative:

- not solved
- no production KB promotion
- no executable route/template rows emitted
- exact reactant/product SMILES were not source-grounded enough

The boundary failure prevented these outputs from being trusted as accepted workflow artifacts.

## Guided Rerun and Expansion

Guided ChemEnzy rerun:

```json
{
  "status": "skipped",
  "route_status": "unresolved",
  "solved": false,
  "reasons": ["guided_policy_missing"]
}
```

Route expansion subgoal search:

```json
{
  "status": "skipped",
  "solved": false,
  "subgoal_count": 0,
  "reasons": ["route_expansion_child_targets_missing"]
}
```

Self-evolution replay gate:

```json
{
  "status": "skipped",
  "production_write_blocked": true,
  "production_promoted_count": 0,
  "reasons": ["self_evo_staging_missing"]
}
```

These skips are acceptable for v0 because the upstream open-structure research artifact was not accepted.

## Artifact Bundle Validation

Artifact bundle validation failed, as expected, because the run contained fake closure evidence and an open research boundary failure.

Validation reasons:

- `fake_closure_evidence_present`
- `open_agent_boundary_violation:context_boundary:large_raw_artifact_dump`
- `open_structure_research_nonzero_exit`

Artifact keys present:

- `chemenzy`
- `route_verifier`
- `route_audit`
- `route_failure_feedback`
- `smiles_first`
- `open_structure_research`
- `compiled_downstream`
- `guided_chemenzy`
- `route_expansion_subgoal_search`
- `self_evo_replay`
- `frontier_smiles`

## Final Verdict

Final verdict:

```text
fake_closed_rejected
```

Reasons:

- `advanced_same_scaffold_terminal`
- `fake_closure_evidence_present`
- `large_atom_jump`
- `no_verifier_accepted_stock_closed_route`
- `open_agent_boundary_violation:context_boundary:large_raw_artifact_dump`
- `open_structure_research_nonzero_exit`
- `route_verifier_rejected_raw_routes`

This is a correct v0 outcome:

- The harness did not accept ChemEnzy native solved claims at face value.
- Advanced same-scaffold terminal closure was rejected.
- Large atom jumps were rejected.
- No production self-evolution promotion occurred.
- No solved route was emitted.

## What This Run Demonstrates

1. The v0 fullflow harness is operational end to end.
2. The raw route verifier now distinguishes generated intermediates from true terminal leaves.
3. Bufotalin native ChemEnzy routes are correctly rejected as fake closure under strict stock validation.
4. Literature and open-structure components can produce advisory/partial-anchor material, but they do not override route verification.
5. Artifact validation blocks solved claims when fake closure or open research boundary violations are present.

## Known Issues Exposed

### 1. Open research context boundary violation

The open research agent read too much raw prefetch context. The boundary audit correctly rejected this:

```text
open_agent_boundary_violation:context_boundary:large_raw_artifact_dump
```

Fix direction:

- expose compact `route_verifier_report`, `route_failure_feedback`, and retrieval summaries
- hide or prohibit direct large raw artifact reads
- make the open agent consume bounded summaries instead of raw dumps

### 2. Guided rerun did not consume compiled downstream outputs

Although open research generated compiled downstream artifacts, the top-level open research tool was rejected. As a result:

- guided rerun skipped with `guided_policy_missing`
- route expansion skipped with `route_expansion_child_targets_missing`

Fix direction:

- allow explicitly validated curator-augmented downstream packets to be used only if boundary audit passes
- keep rejected open-agent outputs blocked from production route claims

### 3. Bufotalin remains unresolved

The current system can reject bad closure, but it still does not solve Bufotalin from small stock.

Needed next capability:

- source-grounded synthesis of the advanced steroid/bufadienolide core
- executable upstream route for the Deacetylbufotalin-like terminal
- validated transformations with exact reactant/product SMILES, not just advisory template cards

## Key Files

Root run:

- `results/shared/bufotalin_v0_fullflow_harness_20260606_073155/final_verdict.json`
- `results/shared/bufotalin_v0_fullflow_harness_20260606_073155/run_summary.json`
- `results/shared/bufotalin_v0_fullflow_harness_20260606_073155/tool_calls.jsonl`
- `results/shared/bufotalin_v0_fullflow_harness_20260606_073155/artifact_bundle.json`
- `results/shared/bufotalin_v0_fullflow_harness_20260606_073155/artifact_bundle_validation.json`

Route verification:

- `results/shared/bufotalin_v0_fullflow_harness_20260606_073155/chemenzy_native_raw_result.json`
- `results/shared/bufotalin_v0_fullflow_harness_20260606_073155/route_verifier_report.json`
- `results/shared/bufotalin_v0_fullflow_harness_20260606_073155/route_audit.json`
- `results/shared/bufotalin_v0_fullflow_harness_20260606_073155/route_failure_feedback.json`

Literature / research:

- `results/shared/bufotalin_v0_fullflow_harness_20260606_073155/smiles_first_literature_workflow/summary.md`
- `results/shared/bufotalin_v0_fullflow_harness_20260606_073155/smiles_first_literature_workflow/validation.json`
- `results/shared/bufotalin_v0_fullflow_harness_20260606_073155/open_structure_research_result.json`
- `results/shared/bufotalin_v0_fullflow_harness_20260606_073155/open_structure_research/structure_template_report.md`
- `results/shared/bufotalin_v0_fullflow_harness_20260606_073155/open_structure_research/downstream_consumables.json`

## Recommended Next Step

Fix the open-structure research boundary behavior first. The route verifier is now doing the right thing for this run, but the open research agent needs a stricter context contract so it can produce accepted downstream guidance without reading large raw route or retrieval dumps.
