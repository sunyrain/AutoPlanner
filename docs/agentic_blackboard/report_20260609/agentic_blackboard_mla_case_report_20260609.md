# Agentic Blackboard Case Report

- Generated: 2026-06-09T04:38:12.354881+00:00
- Case run: `/root/autodl-tmp/AutoPlanner/docs/agentic_blackboard/report_20260609/case_run`
- Target: MLA-like alkaloid case (25 heavy atoms, 5 rings)
- Test gate: pytest -q: 279 passed, 2 skipped, 7 warnings in 76.17s
- Final verdict: `fake_closed_rejected` / route_status `fake_closed_rejected`

## Round Decisions

### Round 1: `generate_disconnection_hypotheses`
- Rationale: Initial MLA-like target needs target-side handles before any rerun.
- Expected artifact: target_side_disconnection_hypotheses.v1
- Success condition: Aryl ester, imide, cage, and amine advisory tasks appear.
- Result: status `accepted`, useful_artifact `True`

### Round 1: `build_failure_critic_report`
- Rationale: Prior route verifier shows large atom jump and advanced terminal; normalize that into bridge tasks.
- Expected artifact: failure_critic_report.v1
- Success condition: Target bridge, terminal blacklist, and next-action bias are recorded.
- Result: status `accepted`, useful_artifact `True`

### Round 1: `search_literature`
- Rationale: Bridge tasks need target-proximal source candidates before exact replay.
- Expected artifact: literature_scout_report.v1
- Success condition: Scout emits source candidates and extraction recommendations.
- Result: status `accepted`, useful_artifact `True`

### Round 2: `compile_exact_literature_rows`
- Rationale: Mock source-detail row is available; compile it as exact evidence, not as proof.
- Expected artifact: compiled exact literature rows
- Success condition: One exact row enters literature_evidence.exact_rows.
- Result: status `accepted`, useful_artifact `True`

### Round 2: `rank_analogical_hypotheses`
- Rationale: Rank advisory hypotheses after exact-row context is present.
- Expected artifact: analogical_hypothesis_ranking.v1
- Success condition: Selected hypotheses carry required verification and no_solved_claim.
- Result: status `accepted`, useful_artifact `True`

### Round 2: `stop_unresolved`
- Rationale: No stitched parent proof exists; stop without solved claim.
- Expected artifact: unresolved stop marker
- Success condition: Final verdict stays unresolved or partial, never solved.
- Result: status `accepted`, useful_artifact `False`

## Architecture Guards

- Planner selects typed actions only; it cannot emit route SMILES or solved verdicts.
- Action batches are schema-validated, budgeted, and stale-action checked before execution.
- Failure critic turns route-verifier and plugin-runtime failures into bridge tasks.
- Exact literature rows and analogy are separated; analogy only changes priorities.
- Final solved requires stitched_parent_route_proof.v1, not backend or child solved flags.

## Final Gate

No stitched parent proof exists in this case, so the final verdict remains non-solved.
