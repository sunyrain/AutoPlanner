# Bufotalin strict visual final corrected report

## 最终结论

- verdict: `solved`
- solved: `True`
- route_status: `solved`
- stock_audit_passed: `True`
- combined steps: `23` = `8` subgoal + `15` literature

## 关键修正

- strict visual terminal 11 与旧 hard-coded compound 11 不是同一立体异构体；本报告使用 strict terminal 重新跑 exact subgoal search。
- 旧 `final_verdict_strict_visual.json` 被旧 guided/raw artifacts 污染而 rejected；clean bundle 只保留 strict visual deterministic artifacts，最终 verdict 为 solved。

## 审计产物

- `artifact_bundle`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_fullflow_fresh_visual_existing_pdf_20260608_065053/artifact_bundle_strict_visual.json`
- `clean_artifact_bundle`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_fullflow_fresh_visual_existing_pdf_20260608_065053/artifact_bundle_strict_visual_clean.json`
- `clean_artifact_bundle_validation`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_fullflow_fresh_visual_existing_pdf_20260608_065053/artifact_bundle_validation_strict_visual_clean.json`
- `clean_final_verdict`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_fullflow_fresh_visual_existing_pdf_20260608_065053/final_verdict_strict_visual_clean.json`
- `final_verdict`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_fullflow_fresh_visual_existing_pdf_20260608_065053/final_verdict_strict_visual.json`
- `route_expansion_subgoals`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_fullflow_fresh_visual_existing_pdf_20260608_065053/route_expansion_subgoals`
- `source_detail_chain_route`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_fullflow_fresh_visual_existing_pdf_20260608_065053/source_detail_chain_route_strict_visual/source_detail_route_chain_audit.json`
- `source_detail_curator_records`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_fullflow_fresh_visual_existing_pdf_20260608_065053/open_structure_research_strict_visual/evidence/source_detail_curator_records.json`
- `stitched_route`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_fullflow_fresh_visual_existing_pdf_20260608_065053/stitched_semisynthesis_route_strict_visual/stitched_semisynthesis_route.json`
- `strict_visual_candidate_chain`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_fullflow_fresh_visual_existing_pdf_20260608_065053/visual_literature_chain_extraction_strict_visual/visual_structure_candidate_chain.json`
- `strict_visual_terminal_audit`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_fullflow_fresh_visual_existing_pdf_20260608_065053/visual_literature_chain_extraction_strict_visual/strict_visual_terminal_audit.json`
- `visual_chain_validation`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_fullflow_fresh_visual_existing_pdf_20260608_065053/literature_intermediate_chain_validation_strict_visual/visual_structure_chain_validation.json`
- `clean_summary`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_fullflow_fresh_visual_existing_pdf_20260608_065053/bufotalin_strict_visual_continuation_clean_summary.json`
- `strict_visual_clean_verdict`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_fullflow_fresh_visual_existing_pdf_20260608_065053/final_verdict_strict_visual_clean.json`
- `strict_visual_clean_bundle`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_fullflow_fresh_visual_existing_pdf_20260608_065053/artifact_bundle_strict_visual_clean.json`
- `strict_visual_subgoal_verifier`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_fullflow_fresh_visual_existing_pdf_20260608_065053/route_expansion_subgoals/01_strict_visual_terminal_11_verifier.json`
