# Bufotalin stitched full-flow expert report

- generated_at_utc: `2026-06-08T04:36:36.355994+00:00`
- run_dir: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345`
- final verdict: `solved`
- solved: `True`
- stitched route: accepted=`True`, status=`solved`

## Architecture

The current controller treats ChemEnzy, retrieval, PDF extraction, route expansion, and self-evolution as typed tools behind deterministic gates. A solved claim must come from deterministic validators, not from raw planner text or raw route-generator status.

## Bufotalin example

- live planner accepted: `True`; run_semantics=`canonical_agent_controller`; changed=`False`.
- native ChemEnzy: elapsed `303.173` s, route_count `179`, accepted routes `0`.
- local PDF: `/root/autodl-tmp/AutoPlanner/1-s2.0-S0040402025001668-main.pdf`; rendered pages `4`, crops `3`.
- source-detail chain: `15` steps, terminal reached `True`.
- subgoal verifier: route_count `56`, accepted `20`, best rank `2`.
- stitched route: `24` total steps.
- patched guided rerun: elapsed `327.683` s, route_status `fake_closed_rejected`.

## Key artifacts

- `final_verdict`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/final_verdict.json`
- `pdf_evidence`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/literature_pdf_structure_extraction/literature_pdf_structure_evidence.json`
- `planner_record`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/codex_planner_run_record.json`
- `route_expansion_subgoals`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/route_expansion_subgoals`
- `source_detail_chain_route`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/source_detail_chain_route/source_detail_route_chain_audit.json`
- `source_detail_curator_records`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/open_structure_research/evidence/source_detail_curator_records.json`
- `stitched_route`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/stitched_semisynthesis_route/stitched_semisynthesis_route.json`
- `visual_candidate_chain`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/visual_candidate_chain/visual_structure_candidate_chain.json`
- `visual_chain_validation`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/literature_intermediate_chain_validation/visual_structure_chain_validation.json`
- `patched_guided_tool_call`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/guided_chemenzy_patched_tool_call.json`
- `guided_route_verifier`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/guided_route_verifier_report.json`
- `artifact_bundle_validation`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/artifact_bundle_validation.json`
- `summary`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/bufotalin_stitched_fullflow_summary.json`
