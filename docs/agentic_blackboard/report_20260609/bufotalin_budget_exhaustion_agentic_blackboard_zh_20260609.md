# Bufotalin Agentic Blackboard 预算耗尽中文报告

- 生成时间：2026-06-09T05:28:13.448534+00:00
- run 目录：`/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_agentic_blackboard_budget_exhaustion_valid_policy_20260609`
- 本地 PDF：`/root/autodl-tmp/AutoPlanner/1-s2.0-S0040402025001668-main.pdf`，exists=True，size=878443 bytes
- 最终结论：`unresolved` / route_status `child_solved_parent_unresolved` / solved `False`
- 停止原因：`max_round_budget_exhausted`，rounds `5/5`
- 测试摘要：python -m pytest tests/test_agentic_blackboard_controller.py tests/test_failure_critic.py tests/test_parent_route_proof.py tests/test_open_research_experience.py tests/test_codex_entry_harness_contract.py -q: 139 passed, 2 skipped in 23.01s

## 为什么这一轮没有直接停止

本次使用的是预算耗尽演示 planner。它没有选择 `stop_unresolved`，而是按 blackboard 状态持续探索：目标侧断键、文献 scout、PDF 渲染、视觉链抽取、exact rows 编译、类比排序、guided ChemEnzy、子目标扩展和父路线拼接。最终停止来自 `max_round_budget_exhausted`，不是 agent 过早放弃。

## Action 执行轨迹

### 第 1 轮：生成目标侧断键假设 (`generate_disconnection_hypotheses`)
- rationale：先从 bufotalin 目标本身生成可审计断键/桥接假设，避免直接把远端文献或类比路线当成证明。
- expected_artifact：target_side_disconnection_hypotheses.v1
- success_condition：形成 target-proximal bridge tasks，且不输出 reaction SMILES。
- status：`accepted`，useful_artifact=`True`，reasons：无

### 第 1 轮：检索目标近端文献 (`search_literature`)
- rationale：blackboard 缺少目标近端文献证据；agent 因此请求文献 scout，本地 PDF 在此时作为候选源提供。
- expected_artifact：literature_scout_report.v1
- success_condition：source_candidates 中包含本地 PDF 和后续抽取建议。
- status：`accepted`，useful_artifact=`True`，reasons：无

### 第 2 轮：渲染/索引本地 PDF 结构证据 (`extract_pdf_literature_structures`)
- rationale：scout 已经把本地 PDF 标为候选源；先把 PDF 渲染/索引为当前 run 的视觉证据。
- expected_artifact：literature_pdf_structure_evidence.v1
- success_condition：生成 rendered_pages/indexed_images/scheme_crops，供视觉链工具使用。
- status：`accepted`，useful_artifact=`True`，reasons：无

### 第 2 轮：排序类比假设 (`rank_analogical_hypotheses`)
- rationale：已有目标侧假设，可以先排序类比探索方向；排序只影响策略，不作为 proof。
- expected_artifact：analogical_hypothesis_ranking.v1
- success_condition：selected_hypotheses 全部带 no_solved_claim 和 required_verification。
- status：`accepted`，useful_artifact=`True`，reasons：无

### 第 3 轮：视觉抽取文献结构链 (`extract_visual_literature_chain`)
- rationale：PDF 结构证据已产生；尝试用真实视觉链 agent 从当前 PDF 图片抽取 source-detail 结构序列。
- expected_artifact：visual_literature_chain_extraction_result.v1
- success_condition：获得候选 source-detail chain，或记录真实失败原因。
- status：`accepted`，useful_artifact=`True`，reasons：visual_literature_chain_extraction_gaps

### 第 3 轮：编译 exact 文献行 (`compile_exact_literature_rows`)
- rationale：视觉链若可用，则把 source-detail validated steps 编译为 exact rows；不可用时保留失败证据。
- expected_artifact：compiled exact literature rows
- success_condition：exact_rows 进入 literature_evidence，或写明为什么无法编译。
- status：`accepted`，useful_artifact=`False`，reasons：no_compiled_downstream_assets, missing_one_step_row_for_product, no_chain_unrolled

### 第 4 轮：运行 guided ChemEnzy (`run_guided_chemenzy`)
- rationale：已有 bridge tasks、类比排序和可能的 exact rows；执行一次真实 guided ChemEnzy，但结果仍必须过 verifier。
- expected_artifact：guided_chemenzy_result.v1
- success_condition：verifier 接受路线，或把失败反馈写回 blackboard。
- status：`accepted`，useful_artifact=`True`，reasons：高级同骨架终端被拒绝, 出现无法解释的大重原子跳跃, no_verifier_accepted_stock_closed_route

### 第 4 轮：扩展上游子目标 (`expand_child_target`)
- rationale：若 failure critic 或 exact rows 暗示高级终端/桥接子目标，则尝试上游子目标搜索；子目标 solved 不能升级父目标。
- expected_artifact：route_expansion_subgoal_search_result.v1
- success_condition：记录子目标 verifier 结果，或记录没有可扩展子目标。
- status：`accepted`，useful_artifact=`True`，reasons：没有子目标被 verifier 证明闭合

### 第 5 轮：拼接父路线证明 (`stitch_parent_route`)
- rationale：最后一轮尝试 deterministic parent-route proof；没有连通 proof 时只能保持 unresolved/partial。
- expected_artifact：stitched_parent_route_proof.v1
- success_condition：证明目标等价、父路线 verifier、stock audit、child/文献连通性，或明确拒绝。
- status：`accepted`，useful_artifact=`True`，reasons：子目标路线未连到父路线桥接点, exact 文献片段未连到父路线, 父路线 verifier 未接受, 库存审计未通过, unexplained_large_atom_jump

### 第 5 轮：构建失败批判报告 (`build_failure_critic_report`)
- rationale：预算耗尽前整理 guided/插件/verifier 失败，把原因转成下一轮可读 blackboard 状态。
- expected_artifact：failure_critic_report.v1
- success_condition：产生 route_failures、blocked_directions、bridge_tasks 或 no_failure_evidence。
- status：`accepted`，useful_artifact=`True`，reasons：无

## Blackboard 和最终门禁

- bridge_tasks：5
- exact_rows：0
- selected_analogies：3
- parent_route_proof：`child_solved_parent_unresolved`

结论：没有 deterministic parent proof 时，final verdict 只能保持 unresolved/partial；类比、子目标或后端 solved flag 都不能升级为父目标 solved。
