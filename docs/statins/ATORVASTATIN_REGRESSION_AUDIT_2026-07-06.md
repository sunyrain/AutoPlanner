# Atorvastatin Regression Audit - 2026-07-06

## Scope

This audit answers why atorvastatin was previously a stable ChemEnzy retrosynthesis case but the July 2026 online/blackboard reruns stopped at hypothesis routes or `no_route_found`.

## Finding 1: July runs used the wrong target molecule

The June solved runs targeted atorvastatin free acid:

- SMILES: `CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4`
- InChIKey: `XUKUURHRXDUEBC-KAYWLYCHSA-N`

The July online runs were named atorvastatin, but preflight recorded:

- `results/shared/atorvastatin_online_zero_20260704_202940/preflight.json`
- `results/shared/atorvastatin_online_anchor_prompt_20260705_rerun/preflight.json`
- InChIKey: `OYBJITKZFDHGHP-SVBPBHIXSA-N`

That is a different connected structure/regioisomer. The old preflight only checked RDKit validity, so it accepted a valid molecule with the wrong identity.

Impact: downstream literature hints, route forest display, ChemEnzy requests, and verifier gates were all trying to solve a molecule that was not atorvastatin while the UI still called it atorvastatin.

## Finding 2: Current local ChemEnzy runtime was not equivalent to the old successful runtime

Old successful evidence:

- `results/shared/atorvastatin_chemenzy_strict_deep15_probe_20260606/final_verdict.json`
- Verdict: `solved`
- `results/shared/atorvastatin_chemenzy_strict_deep15_probe_20260606/chemenzy_native_raw_result.json`
- `ok: true`
- `n_results: 26`

Fresh parity check with the correct June target on current Windows py312:

- `results/shared/atorvastatin_parity_old_deep15_20260706/chemenzy_native_raw_result.json`
- `ok: false`
- `n_results: 0`
- `search_status.status: failed`
- `failure_diagnosis: no_route_found`

The current runtime is:

- Python: `3.12.13`
- torchtext: `0.17.2+cpu`

The July guided ChemEnzy stdout logs show the ONMT proposer failing during import/legacy API access:

- `No module named 'torchtext.data.field'`
- `onmt_models.bionav_native_one_step`

After adding partial legacy import shims, a fresh parity run still failed and exposed the next incompatible legacy attribute:

- `results/shared/atorvastatin_parity_after_identity_runtime_audit_20260706/chemenzy_native_raw_result.json`
- `ok: false`
- `n_results: 0`
- stdout: `'Vocab' object has no attribute 'stoi'`

The packed environment at `D:/Autoplanner/chem_enzy_runtime/envs/retro_planner_env` is a Linux conda environment (`bin`, `x86_64-conda-linux-gnu`) and has no Windows `python.exe`. WSL and Docker are not currently available, so this machine cannot directly execute the old Linux/Python3.8 ChemEnzy environment.

Root cause inside the Windows py312 runtime:

- Modern `torchtext` removed the legacy `torchtext.data` API used by vendored OpenNMT.
- Vendored OpenNMT also unpickles old checkpoint `Vocab` objects that carry Python `stoi/itos` state, while modern `torchtext.vocab.Vocab` expects a C++ `vocab` object.
- The first shim only let imports proceed; it did not restore enough old `Vocab`, `Field`, `Dataset`, `Iterator`, and `Batch` behavior for inference.

Fix:

- `cascade_planner/baselines/chem_enzy_adapter.py` now installs a scoped legacy torchtext compatibility layer before importing ChemEnzy.
- It supports old checkpoint `Vocab` state (`stoi`, `itos`, `freqs`, `len`, lookup methods) and the minimal old `torchtext.data` path used by ONMT inference.

Post-fix evidence:

- Direct ONMT checkpoint load: `model_step_30000.pt` loads as `Translator`.
- Direct ONMT inference on aspirin returns a reactant proposal (`O=C(O)c1ccccc1O`) with a score.
- Fresh deep15 parity:
  - `results/shared/atorvastatin_parity_after_onmt_inference_patch_deep15_20260706/chemenzy_native_raw_result.json`
  - `ok: true`
  - `search_status.status: solved`
  - `n_results: 639`
  - `native_raw_n_routes: 827`
  - `best_depth: 15`
  - `time_s: 240.516`

Impact: the local Windows py312 ChemEnzy path is now able to reproduce and exceed the old stock-closed atorvastatin solve when given the correct atorvastatin free-acid target.

## Finding 3: The blackboard integration misread mixed verifier reports

The July final verdicts are stricter:

- `atorvastatin_online_zero_20260704_202940`: `hypothesis_route_proposed`, `route_status: hypothesis_route_execution_partial`
- `atorvastatin_online_anchor_prompt_20260705_rerun`: `hypothesis_route_proposed`, `route_status: hypothesis_routes_pending_execution`

Those gates are behaving defensibly: a literature/process anchor can produce an advisory route, but without a deterministic parent-route proof or accepted ChemEnzy route, the system should not claim solved.

The stricter gate explains why the run stops at hypothesis, but it does not explain why ChemEnzy stopped being stable. The two root causes above do.

After the ChemEnzy runtime was restored, a full blackboard rerun still failed once:

- `results/shared/atorvastatin_blackboard_correct_identity_runtime_fixed_round5_20260706`
- Guided ChemEnzy verifier: `accepted: true`, `route_status: solved`, `accepted_route_count: 636`
- Rejected sibling routes: `rejected_route_count: 5`, reason `large_atom_jump`
- Final verdict: `hypothesis_route_proposed`

That showed a third integration bug: several blackboard proof gates treated the presence of any rejected sibling route with `large_atom_jump` as proof that the entire guided result was fake-closed. The correct interpretation is: if the verifier has accepted stock-closed routes for the requested target, rejected sibling routes are diagnostics, not blockers for the accepted route.

Fix:

- The guided ChemEnzy wrapper now only hard-blocks `large_atom_jump` when no verifier route is accepted.
- Direct parent-route readiness, blackboard next-action bias, and parent-route proof compilation now use `accepted_route_count > 0` to distinguish accepted routes from rejected siblings.

Post-fix full blackboard evidence:

- `results/shared/atorvastatin_blackboard_correct_identity_runtime_fixed_parent_proof_20260706/final_verdict.json`
- Verdict: `solved`
- `route_status: solved`
- `stock_audit_passed: true`
- Guided ChemEnzy verifier: `accepted: true`, `solved: true`, `accepted_route_count: 637`, `rejected_route_count: 3`
- Round 4 executed `stitch_parent_route`, producing a deterministic parent-route proof.

## Fixes Applied

1. Added known-target identity auditing in `cascade_planner/harness/preflight.py`.
   - Exact atorvastatin/Lipitor cases now require InChIKey `XUKUURHRXDUEBC-KAYWLYCHSA-N`.
   - Wrong connected isomer `OYBJITKZFDHGHP-SVBPBHIXSA-N` is rejected before any live Codex/ChemEnzy call.
   - `atorvastatin_like` remains allowed as an analog/test case and does not force exact identity.

2. Corrected atorvastatin constants in focused tests to use the real free-acid SMILES.

3. Added ChemEnzy runtime diagnostics for guided runs.
   - Logs containing `torchtext.data.field`, `Vocab.stoi`, `Vocab.itos`, `Vocab.vocab`, or ONMT recursion are surfaced as `one_step_model_runtime_error`.
   - These diagnostics are blackboard failure feedback only and are not allowed to prove a parent route.

4. Restored ChemEnzy ONMT inference on Windows py312.
   - Added scoped legacy `torchtext.vocab.Vocab` behavior for old checkpoint state.
   - Added minimal legacy `torchtext.data` `Field`, `RawField`, `Dataset`, `Example`, `Batch`, `Iterator`, and batching behavior needed by vendored OpenNMT.
   - Verified direct ONMT inference and full atorvastatin deep15 parity.

5. Fixed mixed-route verifier handling in the blackboard integration.
   - Accepted verifier routes now remain accepted even when rejected sibling routes carry `large_atom_jump` diagnostics.
   - Parent-route proof compilation treats `accepted_route_count > 0` as the proof subject and keeps rejected sibling routes as diagnostics.

## Verification

Targeted tests:

```text
D:\conda\envs\py312\python.exe -m pytest \
  tests/test_codex_entry_harness_contract.py::CodexEntryHarnessContractTest::test_preflight_rejects_known_target_name_smiles_mismatch \
  tests/test_agentic_blackboard_controller.py::AgenticBlackboardControllerTest::test_guided_chemenzy_detects_onmt_runtime_error_from_logs \
  tests/test_agentic_blackboard_controller.py::AgenticBlackboardControllerTest::test_guided_chemenzy_timeout_is_blackboard_feedback_not_tool_failure \
  tests/test_smiles_first_workflow.py::SmilesFirstWorkflowTest::test_separate_statin_aromatic_rings_are_not_polycyclic_steroid_hint \
  tests/test_chem_enzy_adapter_model_availability.py \
  tests/test_route_objectives.py \
  tests/test_route_forest.py
```

Result before adding the lightweight Vocab regression test: `21 passed`.

Final focused regression suite after the verifier/proof fix:

```text
D:\conda\envs\py312\python.exe -m pytest \
  tests/test_chem_enzy_adapter_model_availability.py \
  tests/test_codex_entry_harness_contract.py::CodexEntryHarnessContractTest::test_preflight_rejects_known_target_name_smiles_mismatch \
  tests/test_agentic_blackboard_controller.py::AgenticBlackboardControllerTest::test_guided_chemenzy_detects_onmt_runtime_error_from_logs \
  tests/test_agentic_blackboard_controller.py::AgenticBlackboardControllerTest::test_guided_chemenzy_timeout_is_blackboard_feedback_not_tool_failure \
  tests/test_agentic_blackboard_controller.py::AgenticBlackboardControllerTest::test_guided_chemenzy_large_atom_jump_overrides_backend_solved_when_no_route_accepted \
  tests/test_agentic_blackboard_controller.py::AgenticBlackboardControllerTest::test_guided_chemenzy_preserves_solved_verifier_with_rejected_sibling_routes \
  tests/test_agentic_blackboard_controller.py::AgenticBlackboardControllerTest::test_direct_parent_verifier_drives_deterministic_stitch_fast_path \
  tests/test_smiles_first_workflow.py::SmilesFirstWorkflowTest::test_separate_statin_aromatic_rings_are_not_polycyclic_steroid_hint \
  tests/test_route_objectives.py \
  tests/test_route_forest.py \
  tests/test_parent_route_proof.py
```

Result: `36 passed`.

Direct preflight sanity check:

- `atorvastatin_latest_small_stock_depth20_real` with the wrong regioisomer is rejected.
- `atorvastatin_like` with `CCO` remains accepted as an analog/test probe.
- correct atorvastatin free acid yields `XUKUURHRXDUEBC-KAYWLYCHSA-N`.

Runtime parity:

- Old June deep15: `ok: true`, `n_results: 26`, `native_raw_n_routes: 27`, `best_depth: 15`.
- Fresh post-fix deep15: `ok: true`, `n_results: 639`, `native_raw_n_routes: 827`, `best_depth: 15`.

Full blackboard closure:

- `results/shared/atorvastatin_blackboard_correct_identity_runtime_fixed_parent_proof_20260706/final_verdict.json`
- `verdict: solved`
- `solved: true`
- `stock_audit_passed: true`

Do not use the July `atorvastatin_online_*` outputs as success/failure evidence because they were run with the wrong target InChIKey.
