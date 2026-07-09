# Bufotalin 中文流程汇报

## 一句话结论

bufotalin 本轮最终结论是 `solved`。它不是 ChemEnzy 原生路线直接成功，而是：文献 source-detail 链闭合到 compound 11，compound 11 子目标被 ChemEnzy/verifier 证明可由 stock 闭合，最后由 stitch 工具做身份审计后拼成完整半合成路线。

## 关键数字

- 文献链：15 步，terminal reached=True
- 子目标：20 / 56 条路线被 verifier 接受
- 拼接路线：24 步
- 原生 ChemEnzy：0 条 verifier accepted
- guided ChemEnzy 补跑：fake_closed_rejected

## 审计入口

- final_verdict: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/final_verdict.json`
- pdf_evidence: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/literature_pdf_structure_extraction/literature_pdf_structure_evidence.json`
- planner_record: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/codex_planner_run_record.json`
- route_expansion_subgoals: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/route_expansion_subgoals`
- source_detail_chain_route: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/source_detail_chain_route/source_detail_route_chain_audit.json`
- source_detail_curator_records: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/open_structure_research/evidence/source_detail_curator_records.json`
- stitched_route: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/stitched_semisynthesis_route/stitched_semisynthesis_route.json`
- visual_candidate_chain: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/visual_candidate_chain/visual_structure_candidate_chain.json`
- visual_chain_validation: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/literature_intermediate_chain_validation/visual_structure_chain_validation.json`
- patched_guided_tool_call: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/guided_chemenzy_patched_tool_call.json`
- guided_route_verifier: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/guided_route_verifier_report.json`
- artifact_bundle_validation: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/artifact_bundle_validation.json`
- summary: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_stitched_fullflow_existing_pdf_20260608_0345/bufotalin_stitched_fullflow_summary.json`
