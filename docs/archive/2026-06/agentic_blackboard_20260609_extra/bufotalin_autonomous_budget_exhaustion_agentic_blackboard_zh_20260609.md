# Bufotalin Agentic Blackboard 自主预算耗尽中文报告

- 生成时间：2026-06-09T12:30:25.724985+00:00
- run 目录：`/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_agentic_blackboard_full_retry_v7_20260609`
- 本地 PDF：`/root/autodl-tmp/AutoPlanner/1-s2.0-S0040402025001668-main.pdf`，exists=True，size=878443 bytes
- 最终结论：`unresolved` / route_status `child_solved_parent_unresolved` / solved `False`
- 停止原因：`max_round_budget_exhausted`，rounds `8/8`
- action 选择：默认 `agent_action_planner.v1` 依据 blackboard 状态自主选择；未传入按轮次写死的 action planner。
- 测试摘要：full retry v7 running; focused tests passed: python -m pytest tests/test_agentic_blackboard_controller.py tests/test_failure_critic.py tests/test_parent_route_proof.py -q

## 为什么这一轮没有直接停止

本次使用 controller 的 `exhaust_round_budget=True`，action batch 仍由默认 blackboard planner 动态选择。该模式只改变停止策略：当普通策略会因连续无新 artifact 而停下时，planner 必须先尝试 blackboard 中仍未耗尽、未 stale 的替代方向；只有父路线 proof 接受或轮次预算耗尽才收口。

## Action 执行轨迹

### 第 1 轮：生成目标侧断键假设 (`generate_disconnection_hypotheses`)
- rationale：target-side bridge tasks are missing
- expected_artifact：target_side_disconnection_hypotheses.v1
- success_condition：at least one advisory hypothesis and bridge task
- status：`accepted`，useful_artifact=`True`，reasons：无

### 第 1 轮：检索目标近端文献 (`search_literature`)
- rationale：blackboard lacks target-proximal literature/source evidence
- expected_artifact：literature_scout_report.v1
- success_condition：source candidate or extraction recommendation generated
- status：`accepted`，useful_artifact=`True`，reasons：无

### 第 2 轮：渲染/索引本地 PDF 结构证据 (`extract_pdf_literature_structures`)
- rationale：local PDF source is available and must be converted into current-run visual evidence
- expected_artifact：literature_pdf_structure_evidence.v1
- success_condition：rendered pages or indexed images are available for visual extraction
- status：`accepted`，useful_artifact=`True`，reasons：无

### 第 2 轮：排序类比假设 (`rank_analogical_hypotheses`)
- rationale：advisory target-side hypotheses can be ranked before rerun selection
- expected_artifact：analogical_hypothesis_ranking.v1
- success_condition：ranked advisory hypotheses with no solved claim
- status：`accepted`，useful_artifact=`True`，reasons：无

### 第 3 轮：视觉抽取文献结构链 (`extract_visual_literature_chain`)
- rationale：source candidates exist but exact rows are missing
- expected_artifact：visual_literature_chain/exact rows artifact
- success_condition：validated source-detail chain or extraction failure reason
- status：`accepted`，useful_artifact=`True`，reasons：visual_literature_chain_extraction_gaps, visual_literature_chain_has_no_steps, visual_literature_chain_missing_expected_labels

### 第 4 轮：视觉抽取文献结构链 (`extract_visual_literature_chain`)
- rationale：prior visual extraction left target-relevant source-detail gaps; repair those gaps before compiling exact rows
- expected_artifact：visual_literature_chain/exact rows artifact
- success_condition：missing source-detail labels are either filled or explicitly rejected
- status：`accepted`，useful_artifact=`True`，reasons：visual_literature_chain_extraction_gaps

### 第 5 轮：编译 exact 文献行 (`compile_exact_literature_rows`)
- rationale：visual chain needs exact source-detail rows for plugin use
- expected_artifact：compiled exact literature rows
- success_condition：one or more exact row summaries
- status：`accepted`，useful_artifact=`True`，reasons：无

### 第 6 轮：运行 guided ChemEnzy (`run_guided_chemenzy`)
- rationale：bridge tasks and search hints are available for one guided rerun
- expected_artifact：guided_chemenzy_result plus verifier report
- success_condition：route verifier accepts or returns actionable failure evidence
- status：`accepted`，useful_artifact=`True`，reasons：高级同骨架终端被拒绝, 出现无法解释的大重原子跳跃, 文献模板插件未被后端调用, no_verifier_accepted_stock_closed_route

### 第 7 轮：扩展上游子目标 (`expand_child_target`)
- rationale：advanced terminal/upstream bridge task exists
- expected_artifact：route_expansion_subgoal_search_result.v1
- success_condition：child target verifier result is recorded without parent solved claim
- status：`accepted`，useful_artifact=`True`，reasons：没有子目标被 verifier 证明闭合

### 第 8 轮：拼接父路线证明 (`stitch_parent_route`)
- rationale：guided/child/literature artifacts need deterministic parent connectivity proof
- expected_artifact：stitched_parent_route_proof.v1
- success_condition：parent proof accepted or explicit connectivity rejection
- status：`accepted`，useful_artifact=`True`，reasons：子目标路线未连到父路线桥接点, exact 文献片段未连到父路线, 父路线 verifier 未接受, 库存审计未通过, unexplained_large_atom_jump

## 为什么这次失败，而之前 bufotalin 成功

- 之前成功 run：`/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_fullflow_fresh_visual_existing_pdf_20260608_065053`。
- 之前 clean verdict：`solved` / solved `True`。
- 之前 strict visual chain：accepted `True`，steps `15`，terminal `strict visual terminal 11`。
- 之前 source-detail exact chain：accepted `True`，one_step_rows `15`，terminal_reached `True`。
- 之前 stitched route：accepted `True`，stock_audit `True`。

反思：之前的 solved 不是 ChemEnzy 对 bufotalin 原生闭合，而是 `stock -> compound 11 -> bufotalin` 的拼接半合成证明：strict visual/source-detail 文献链把 bufotalin 连到 terminal 11，子目标搜索再把 terminal 11 从库存闭合，最后 stitch proof 通过。本轮自主 blackboard 运行没有重新生成 accepted exact literature chain，`compile_exact_literature_rows` 因缺少可用 source-detail rows 得到 0 行；后续 guided ChemEnzy/child expansion 即使产生候选，也缺少和父目标相连的 exact 文献段，因此 parent proof 不能通过。

## Blackboard 和最终门禁

- bridge_tasks：6
- exact_rows：3
- selected_analogies：3
- parent_route_proof：`child_solved_parent_unresolved`

结论：没有 deterministic parent proof 时，final verdict 只能保持 unresolved/partial；类比、子目标或后端 solved flag 都不能升级为父目标 solved。
