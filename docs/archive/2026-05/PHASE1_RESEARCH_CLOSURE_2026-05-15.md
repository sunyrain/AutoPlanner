# Phase I Research Closure Report

Date: 2026-05-15

## 1. 收束结论

第一阶段已经从“训练一个大一统 retrosynthesis generator”收束为“级联搜索控制层吸收强 proposal/search provider”的路线。当前最强、最可复现的系统是 hybrid cascade：

```text
AutoPlanner route-tree search
  + source/budget gate
  + existing proposal providers
  + ChemEnzy native broad route reservoir, bounded top-k
  + cost/rank based route selection
  + stock-closed alternative audit
```

现阶段不能声称 student-only controller 已经替代 ChemEnzy native search。核心证据很明确：C(student-only) 基本等同或略低于 baseline；D(hybrid) 的提升主要来自 bounded native reservoir。AutoPlanner 的贡献不是把 ChemEnzy 压成小模型，而是把强原生路线池放进级联约束、预算控制、多源 proposal、质量过滤和审计框架里。

阶段状态：

| 判定项 | 结论 |
| --- | --- |
| 可复现 hybrid prototype | yes |
| ChemEnzy native reservoir 接入 | yes |
| cost/rank scoring 替代拍脑袋 reward | yes |
| GT 作为 reference recall 而不是唯一正确路线 | yes |
| stock-closed alternative audit | yes |
| student-only 蒸馏成功 | no |
| 发表级证据 | no |

## 2. 研究思路演变

| 阶段 | 架构 | 当时假设 | 结果 |
| --- | --- | --- | --- |
| Stage 0: AutoPlanner baseline | Route-tree + heuristic source gate + existing providers | 级联搜索本身可提升路线覆盖 | 能跑通，但 stock closure 不强 |
| Stage 1: source/stock controller | source policy、stock-aware leaf、stock terminal fixes | 学控制策略就能显著提升 closure | 有局部收益，但稳定性不够，coverage 易波动 |
| Stage 2: v4/action/value scorers | action/source/transition/pair/fragment scorers | 离线监督可以改善每步选择 | 训练和接口跑通，但 off-policy scorer 没形成 full100 稳定增益 |
| Stage 3: reservoir teacher | AutoPlanner D + ChemEnzy native broad routes + rank_plus_stock top-5 | 原生 ChemEnzy route pool 可作为 teacher ceiling | 成立，stock closure 和 route recall 明显提升 |
| Stage 4: distilled controller | shared encoder + source/budget/leaf/action/stock/rerank heads | student 学会问谁、问多少、保留什么 | adapter 可用，但 student-only 没学到 native search 的路线池能力 |
| Stage 5: direct native pool integration | bounded reservoir fallback/top-k append | 不蒸馏 search，先吃到强路线池 | 当前最有效，D 接近 B teacher |
| Stage 6: scoring correction | cost/rank scoring + GT diagnostic + alternative audit | GT 只是多条可行逆合成路线之一 | 成立，非 GT stock-closed route 需要审计而不是直接判错 |
| Stage 7: quality filter | reservoir quality filter / top-10 ablation | 去掉可疑 native shortcut 会提升可信度 | 质量明显变干净，但 stock/coverage 有代价 |

## 3. 尝试过的方法和结果

### 3.1 Route-tree baseline 与 source/stock gate

最早的 AutoPlanner D baseline 是 route-tree 展开加多源 proposal provider。之后尝试过 source policy、stock-aware open-leaf、stock terminal heuristic、late stock rescue 等控制项。

结果：这些控制项让系统能稳定产出 route JSON 和 trace，但没有单独解决 stock closure。早期 debug 产物已经从结果目录清理，保留完整矩阵和报告用于追溯。

相关保留代码：

- `cascade_planner/route_tree/search.py`
- `cascade_planner/route_tree/source_gate.py`
- `cascade_planner/route_tree/runtime.py`
- `cascade_planner/eval/controller_v2_reports.py`

### 3.2 v4 数据、action/source/transition/value/fragment scorer

尝试把 route-tree trace 转成 action/value/source/transition/pair/fragment 监督数据，训练多个小 scorer。这个阶段的重要教训是：小 scorer 可以学到局部偏好，但离线标签与在线搜索分布不一致，容易在 full100 上出现 coverage 下滑或路线数减少。

结果：作为诊断和后续 on-policy 研究基础保留，不作为当前主线 claim。

相关保留代码：

- `cascade_planner/eval/build_v4_trace_benchmark.py`
- `cascade_planner/eval/train_cascade_action_value.py`
- `cascade_planner/eval/train_cascade_pair_scorer.py`
- `cascade_planner/eval/train_cascade_transition_value.py`
- `cascade_planner/cascade_search/`

### 3.3 Reservoir distillation pack

新增 distillation pack builder，把 AutoPlanner route-tree traces、ChemEnzy native routes、reservoir synthesized payload 合并成统一 pack。full100 行标记为 `eval_only=true`，不进入训练。

结果：数据管线可复现，能够训练多头 controller；但数据本身不足以让 student-only 复刻 ChemEnzy native search。

相关保留代码和 artifact：

- `cascade_planner/eval/build_reservoir_distill_pack.py`
- `cascade_planner/eval/build_native_route_replay_pack.py`
- `results/shared/reservoir_distill_20260513/reservoir_distill_manifest.json`
- `results/shared/reservoir_distill_20260513/teacher_report.json`

### 3.4 Reservoir-distilled controller

实现了 `ReservoirDistilledController`，包含 source group、budget、leaf value、action value、stock risk、route rerank、latency cost 等 head。运行时只做 soft priority 和预算建议，checkpoint 失败会回退 heuristic。

结果：工程接口成立，但 C(student-only) 没有超过 baseline。问题不在 adapter，而在 student 没有直接拥有 ChemEnzy native graphfp/onmt search 产生的高质量路线池。

相关保留代码：

- `cascade_planner/route_tree/reservoir_distilled.py`
- `cascade_planner/eval/train_reservoir_distilled_controller.py`
- `tests/test_reservoir_distilled_controller.py`

### 3.5 ChemEnzy native bounded reservoir

接入 ChemEnzy native broad route reservoir，默认 `rank_plus_stock` top-5。这里的 reservoir 不是模型 checkpoint，而是 ChemEnzy 原生 search/proposal 结果的 bounded replay/integration。

结果：这是第一阶段最有效的增强。B teacher 和 D hybrid 都显著提高 stock closure；D 相比 B route count 更低，coverage 略低。

相关保留代码：

- `cascade_planner/route_tree/bounded_reservoir.py`
- `cascade_planner/eval/reservoir_distill_matrix.py`
- `cascade_planner/eval/reservoir_acceptance_manifest.py`
- `scripts/run_chem_enzy_smoke.py`

### 3.6 Scoring overhaul

旧版 teacher label 中 `exact full route +1.00`、`GT reactant +0.55`、`stock closed +0.45` 等分值过于主观，也错误暗示 GT 是唯一正确路线。当前改成接近 ChemEnzy/MolStar 风格的 cost/rank 口径：优先看 stock closure、route cost、reaction plausibility、重复、低进展和无效路线惩罚；GT exact/reactant 只作为 reference recall 诊断。

结果：指标解释更合理，但也暴露 baseline 在当前口径下 coverage 比历史矩阵低。这个变化是必要的，因为逆合成不是单答案任务。

相关保留代码：

- `cascade_planner/route_tree/search.py`
- `cascade_planner/route_tree/cascade_oracle.py`
- `cascade_planner/eval/reservoir_publication_readiness.py`

### 3.7 Stock-closed alternative audit 和 quality filter

对 stock-closed 但未命中 reference GT reactants 的路线做 case study，区分 plausible alternative、needs condition review、weak alternative、suspicious shortcut、invalid/open route。

结果：大量非 GT stock-closed route 是合理替代路线，不应直接算错；但确有可疑 shortcut 和开放路线，需要质量过滤。quality filter 明显提高审计通过率，但会牺牲 stock/coverage。

相关保留代码：

- `cascade_planner/eval/audit_stock_closed_alternatives.py`
- `cascade_planner/eval/reservoir_completion_audit.py`
- `cascade_planner/eval/reservoir_publication_readiness.py`

### 3.8 AUTOPLANNRELLM 探索支线

新建了 `AUTOPLANNRELLM/`，让 LLM 参与 leaf/action 选择，并向候选池追加一条 DeepSeek 建议。默认不开启，缺少 `DEEPSEEK_API_KEY` 时回退 AutoPlanner。

结果：fallback、单目标真实 key smoke、以及 full100 真实 key paired run 都已验证。full100 使用 current A baseline 的标准配置：`n_results=5`、`skeleton_samples=2`、`check_stock`、同一 source/open-leaf policy 和 native payload。DeepSeek 实际完成 873 条有效缓存调用，其中 `leaf_selection=293`、`reaction_suggestion=288`、`action_selection=292`。

full100 结果为负：`LLM_BOTH` 的 `plan_rate=0.62`、`strict_stock_solve_any=0.36`、`exact_reaction_in_route_pool=0.15`、`gt_reactant_in_route_pool=0.24`、`avg_time_per_target_s=21.735`、`avg_route_count=1.15`。相比 A baseline，plan 少 14 个目标，stock 少 10 个百分点，route GT 少 15 个百分点。LLM proposal 进入路线 38 步、覆盖 31 个目标，但同时产生 92 个 self-loop invalid candidate，并显著压缩 route-tree 多样性。它不纳入第一阶段主结论，也不能 promotion。

随后按“LLM 可选 1-3 个分支，并关闭 route-tree soft/hard timeout”重跑 full100。`LLM_TOP3_NO_TIMEOUT` 把 plan 和路线数恢复到 baseline 水平，并把 stock 从 A 的 `0.46` 提到 `0.53`，但 route GT 仍为 `0.38`，略低于 A 的 `0.39`，平均耗时升到 `64.802s/target`。这说明旧 LLM 失败有明显 timeout/branch-collapse 成分，但新的收益主要来自更深搜索，不是 LLM ranker 已经优于 AutoPlanner。该配置只作为 LLM 上限诊断，不作为默认系统。

相关保留代码：

- `AUTOPLANNRELLM/`
- `tests/test_autoplannrellm.py`

## 4. 关键实验结果

### 4.1 历史完整 full100 A-D 矩阵

Artifact: `results/shared/reservoir_distill_20260513/full100_acceptance_real_v2/reports/comparison.md`

| Label | plan | stock | cand GT | route exact | route GT | avg seconds | avg routes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A baseline | 0.9700 | 0.4900 | 0.5800 | 0.3600 | 0.5500 | 6.3450 | 4.2000 |
| B teacher top-5 | 1.0000 | 0.9100 | 0.5800 | 0.4000 | 0.6300 | 6.5360 | 9.1900 |
| C student-only | 0.9700 | 0.4300 | 0.5800 | 0.3600 | 0.5500 | 6.3880 | 4.1700 |
| D hybrid | 1.0000 | 0.9100 | 0.5800 | 0.4000 | 0.6300 | 6.4610 | 7.9300 |
| D_APPEND | 1.0000 | 1.0000 | 0.5800 | 0.4000 | 0.6300 | 6.3880 | 9.1300 |

结论：native reservoir/hybrid 显著提升 stock closure；student-only 没超过 baseline。

### 4.2 当前 cost-scoring full100 A/B/C/D, 30s gate

Artifact: `results/shared/phase2_20260515/full100_abcd_gate30/reports/comparison.md`

| Label | plan | stock | cand GT | route exact | route GT | avg seconds | avg routes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A baseline | 0.7600 | 0.4600 | 0.4200 | 0.2400 | 0.3900 | 3.0100 | 3.4100 |
| B teacher top-5 | 1.0000 | 0.9300 | 0.4200 | 0.3100 | 0.5500 | 3.2460 | 8.0700 |
| C student-only | 0.7600 | 0.4600 | 0.4100 | 0.2300 | 0.3800 | 3.0810 | 3.4300 |
| D hybrid | 1.0000 | 0.9300 | 0.4100 | 0.2900 | 0.5200 | 3.2690 | 7.1700 |

结论：在当前口径下，D 相比 A 提升 `+0.47` stock、`+0.05` route exact、`+0.13` route GT，但 C 仍不成立。D 略低于 B，说明 controller/rerank 并未完全保留 teacher route pool。

### 4.3 Quality-filter ablation

Artifact: `results/shared/phase2_20260515/quality_filter_ablation_gate30/reports_quality/comparison.md`

| Label | plan | stock | cand GT | route exact | route GT | avg seconds | avg routes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D | 1.0000 | 0.9300 | 0.4200 | 0.3100 | 0.5300 | 3.2830 | 7.1300 |
| D_FILTER | 0.9800 | 0.8500 | 0.4100 | 0.2800 | 0.4700 | 3.3100 | 5.1500 |
| D_TOP10_FILTER | 0.9800 | 0.8500 | 0.4100 | 0.2800 | 0.4700 | 3.3410 | 5.1300 |

Audit change:

| Label | reviewed non-reference stock targets | review-pass | plausible-alternative | critical/suspicious best-route |
| --- | ---: | ---: | ---: | ---: |
| D | 42 | 0.8571 | 0.5952 | 0.1429 |
| D_FILTER | 41 | 0.9756 | 0.6585 | 0.0244 |

结论：quality filter 是正确方向，但不能直接默认开启为最终系统，因为它把 stock 从 0.93 降到 0.85。下一阶段需要把 filter 变成 calibrated rerank 或 condition-aware penalty，而不是硬删除。

### 4.4 外部 benchmark

已把 PaRoutes n1/n5、USPTO-190、BioNavi-like 从早期 smoke 扩展到 full-input asset、USPTO-190 full C/D、以及 PaRoutes/BioNavi-like full-input sampled shards。代表性 artifact：

- `results/shared/reservoir_distill_20260513/external_publication_matrix_limit10_ua_20260514/`
- `results/shared/phase2_20260515/external_full_input_assets_20260515/`
- `results/shared/phase2_20260515/external_bionavi_full_input_20260515/`
- `results/shared/phase2_20260515/external_full_input_sample_aggregate_20260515/external_smoke_aggregate.md`
- `results/shared/phase2_20260515/external_cross_dataset_bionavi_full_20260515/external_smoke_aggregate.md`
- `results/shared/phase2_20260515/external_uspto_full_20260515/`
- `results/shared/phase2_20260515/external_uspto_full_cd_summary_20260515/external_smoke_summary.json`
- `results/shared/phase2_20260515/uspto190_cache_20260515/uspto190_cache_report.md`

External cross-dataset aggregate:

| Dataset | Config | n | plan | stock | route exact | route GT | avg seconds | avg routes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BioNavi-like | C | 373 | 0.8391 | 0.1769 | 0.0322 | 0.0992 | 4.3360 | 2.2950 |
| BioNavi-like | D | 373 | 0.9866 | 0.8123 | 0.1850 | 0.3619 | 40.7730 | 5.8100 |
| PaRoutes n1 | C | 30 | 0.9667 | 0.8000 | 0.3333 | 0.6333 | 21.0083 | 2.6667 |
| PaRoutes n1 | D | 30 | 1.0000 | 0.8667 | 0.4333 | 0.7667 | 33.1407 | 4.1667 |
| PaRoutes n5 | C | 30 | 1.0000 | 0.7333 | 0.3333 | 0.7333 | 19.0240 | 2.8000 |
| PaRoutes n5 | D | 30 | 1.0000 | 0.7333 | 0.3333 | 0.8000 | 27.8527 | 3.7000 |
| USPTO-190 | C | 190 | 0.9000 | 0.6316 | 0.1789 | 0.4158 | 15.0060 | 2.5630 |
| USPTO-190 | D | 190 | 0.9579 | 0.7632 | 0.3474 | 0.5684 | 37.2500 | 4.5580 |

Full input status:

- PaRoutes n1/n5 input assets each contain 10000 rows; evaluated evidence currently covers sampled full-input shards, not all rows.
- USPTO-190 C/D is complete over 190 targets; D_FILTER full was interrupted and is only available on cached30.
- BioNavi-like full-input C/D is complete over 373 rows.

结论：D 在外部证据上通常提高 stock closure 和 reference route recall，但运行成本明显增加。PaRoutes n5 sampled aggregate 仍在 30s gate 内；BioNavi-like full-input、PaRoutes n1 sampled、USPTO-190 full 的 D 平均时间超过 30s。D_FILTER 在 full100 上提高路线审计质量，但在 USPTO cached30 和 PaRoutes shards 上过度过滤，不适合作为默认系统。外部结果已经足够支撑“hybrid 值得继续”，但还不足以支撑完整发表级 benchmark claim。

### 4.5 AUTOPLANNRELLM full100 real-key run

Artifact: `results/shared/autoplannrellm_20260515/full100_publication_real_key_20260515/`

| Label | plan | stock | cand GT | route exact | route GT | avg seconds | avg routes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A baseline | 0.7600 | 0.4600 | 0.4200 | 0.2400 | 0.3900 | 3.0100 | 3.4100 |
| D hybrid | 1.0000 | 0.9300 | 0.4100 | 0.2900 | 0.5200 | 3.2690 | 7.1700 |
| LLM_BOTH | 0.6200 | 0.3600 | 0.3800 | 0.1500 | 0.2400 | 21.7350 | 1.1500 |

Paired vs A:

| Metric | gain | loss | same positive | same negative |
| --- | ---: | ---: | ---: | ---: |
| plan | 0 | 14 | 62 | 24 |
| stock | 4 | 14 | 32 | 50 |
| route exact | 1 | 10 | 14 | 75 |
| route GT | 2 | 17 | 22 | 59 |

结论：LLM 分支真实可跑，但当前架构下不是发表级增强。问题不是 API failure，而是 LLM control/proposal 过强，导致有效路线池缩水、hard timeout 增多、route diversity 降低。

### 4.6 AUTOPLANNRELLM top-3 no-timeout run

Artifact: `results/shared/autoplannrellm_20260515/full100_top3_no_timeout_w4_20260515/`

| Label | plan | stock | cand GT | route exact | route GT | avg seconds | avg routes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A baseline | 0.7600 | 0.4600 | 0.4200 | 0.2400 | 0.3900 | 3.0100 | 3.4100 |
| D hybrid | 1.0000 | 0.9300 | 0.4100 | 0.2900 | 0.5200 | 3.2690 | 7.1700 |
| LLM_BOTH | 0.6200 | 0.3600 | 0.3800 | 0.1500 | 0.2400 | 21.7350 | 1.1500 |
| LLM_TOP3_NO_TIMEOUT | 0.7600 | 0.5300 | 0.4300 | 0.2400 | 0.3800 | 64.8020 | 3.5700 |

Paired vs A:

| Metric | gain | loss | same positive | same negative |
| --- | ---: | ---: | ---: | ---: |
| plan | 0 | 0 | 76 | 24 |
| stock | 7 | 0 | 46 | 47 |
| route exact | 3 | 3 | 21 | 73 |
| route GT | 3 | 4 | 35 | 58 |

LLM usage: DeepSeek cache rows `2667`，其中 `leaf_selection=834`、`reaction_suggestion=827`、`action_selection=1006`；route steps from `llm_deepseek=122` across 60 targets；route count histogram `{0:24, 1:1, 2:3, 3:4, 4:2, 5:66}`。

结论：让 LLM 选 1-3 个分支解决了“路线太少”的直接问题，也证明关闭 route-tree timeout 后能恢复 baseline plan。但是 runtime 远超当前 gate，且 route GT 没有超过 A，更远低于 D hybrid。这个结果支持后续把 LLM 当作外部慢速 oracle 或 case-study 工具，而不是主线 controller。

## 5. 当前架构

当前架构不是单一 wrapper，也不是单模型 generator。它是级联控制层：

```text
target
  -> route-tree open-leaf selection
  -> source/budget allocation
  -> proposal providers
       - ChemEnzy native graphfp/onmt route pool
       - RetroChimera
       - Enzyformer
       - template/retrieval providers
       - optional LLM candidate branch, disabled by default
  -> child/action cost scoring
  -> bounded native reservoir fallback
  -> route-level rerank
  -> stock and route-quality audit
  -> ranked route set
```

级联特色体现在：

- 控制“何时问、问谁、问多少、保留什么”，而不是只生成一步反应。
- route-tree 能在 open leaf、child state、route stock closure 三个层面做排序。
- ChemEnzy native reservoir 是 bounded safety net，不允许无限追加。
- GT 指标改名义上应视为 reference recall；路线是否可用主要看 stock closure、成本、反应可信度和审计分类。
- 可以容纳多个 provider，包括 ChemEnzy、RetroChimera、Enzyformer、template/retrieval、LLM branch。

## 6. 代码和文件整理

### 6.1 保留的核心代码

| Area | Paths |
| --- | --- |
| route-tree runtime | `cascade_planner/route_tree/search.py`, `runtime.py`, `source_gate.py`, `proposals.py`, `trace.py` |
| reservoir integration | `cascade_planner/route_tree/bounded_reservoir.py`, `reservoir_distilled.py`, `cascade_oracle.py` |
| cascade scorer research | `cascade_planner/cascade_search/`, `cascade_planner/eval/train_cascade_*.py`, `build_cascade_*_pack.py` |
| distillation | `cascade_planner/eval/build_reservoir_distill_pack.py`, `train_reservoir_distilled_controller.py`, `build_native_route_replay_pack.py` |
| evaluation/reporting | `reservoir_acceptance_manifest.py`, `reservoir_distill_matrix.py`, `reservoir_completion_audit.py`, `reservoir_publication_readiness.py`, `audit_stock_closed_alternatives.py` |
| external benchmark | `build_external_reservoir_smokes.py`, `targets_n1.txt`, `targets_n5.txt`, `ref_routes_n1.json`, `ref_routes_n5.json` |
| LLM branch | `AUTOPLANNRELLM/` |

### 6.2 保留的关键 artifact

| Purpose | Path |
| --- | --- |
| historical complete A-D/D_APPEND | `results/shared/reservoir_distill_20260513/full100_acceptance_real_v2/` |
| current full100 A/B/C/D gate30 | `results/shared/phase2_20260515/full100_abcd_gate30/` |
| quality-filter ablation | `results/shared/phase2_20260515/quality_filter_ablation_gate30/` |
| reservoir distill pack/checkpoints | `results/shared/reservoir_distill_20260513/` |
| external limit10/smoke evidence | `results/shared/reservoir_distill_20260513/external_publication_matrix_limit10_ua_20260514/` |
| external PaRoutes shard evidence | `results/shared/phase2_20260515/external_paroutes_shard10_manifest_20260515/` |
| external USPTO cached30 evidence | `results/shared/phase2_20260515/external_uspto_cached30_only_20260515/` |
| external full-input assets | `results/shared/phase2_20260515/external_full_input_assets_20260515/`, `results/shared/phase2_20260515/external_bionavi_full_input_20260515/` |
| external BioNavi full-input C/D | `results/shared/phase2_20260515/external_bionavi_full_cd_20260515/` |
| external full-input sampled aggregate | `results/shared/phase2_20260515/external_full_input_sample_aggregate_20260515/` |
| external USPTO-190 full C/D | `results/shared/phase2_20260515/external_uspto_full_20260515/`, `results/shared/phase2_20260515/external_uspto_full_cd_summary_20260515/` |
| external cross-dataset aggregate | `results/shared/phase2_20260515/external_cross_dataset_bionavi_full_20260515/` |
| ChemEnzy baseline/reservoir payloads | `results/shared/chem_enzy_baseline/` |
| AUTOPLANNRELLM real-key smoke | `results/shared/autoplannrellm_20260515/smoke_real_key_20260515_row1/` |
| AUTOPLANNRELLM full100 real-key paired run | `results/shared/autoplannrellm_20260515/full100_publication_real_key_20260515/` |

### 6.3 已删除内容

清理记录见 `docs/PHASE1_CLEANUP_MANIFEST_2026-05-15.md`。本次实际删除：

- Python `__pycache__/` 和 `.pytest_cache/`
- 中止的 `results/shared/phase2_20260515/external_expanded_limit20_cd/`
- `results/shared/controller_v2_20260512/**/debug*` 调试目录
- 未触发 route-tree 的 AUTOPLANNRELLM 首次 real-key smoke 和裸 DeepSeek client cache

未删除：

- checkpoint、pack、manifest
- 完整 full100 矩阵
- 当前 gate30 full100 和 quality-filter 对照
- 用户提供的 PaRoutes n1/n5 target/reference 文件
- `vendor/` ChemEnzy runtime

## 7. 距离“可使用”和“可发表”的差距

内部可使用：可以作为 research prototype 使用，尤其是 D hybrid 配置。它能稳定调用 bounded native reservoir，并有 fallback、trace、report 和 audit。

论文可发表：还不够。主要缺口：

1. 外部 benchmark 仍未到完整发表规模；USPTO-190 C/D 和 BioNavi-like C/D 已完整，但 PaRoutes n1/n5 目前还是 sampled full-input evidence。
2. student-only 不成立，不能写成“成功蒸馏 ChemEnzy search”。
3. quality filter 还在 hard-filter 阶段，提升可信度但损失 coverage。
4. D hybrid 在多个外部 aggregate 上超过 30s，若把 gate 放到 20-30s 仍不能无条件 promotion。
5. 非 GT 可行路线需要更系统的人审或自动反应可行性验证。
6. AUTOPLANNRELLM full100 真实 key 比较实验已经完成，结果为负；还不能作为质量增强 claim。
7. 需要统计显著性、失败案例、数据泄漏检查和 benchmark protocol。

## 8. 下一阶段建议

Phase II 主线应改为：

```text
Calibrated Hybrid Cascade:
  preserve strong native route pools
  learn when/how to call them
  calibrate quality and stock-risk reranking
  report GT/reference recall separately from route usability
```

优先级：

1. 扩大外部 benchmark：补 PaRoutes n1/n5 和 BioNavi-like 更大分片或全量运行，补 USPTO-190 D_FILTER 或明确放弃 hard-filter 默认。
2. 把 quality filter 从硬过滤改成 calibrated rerank，避免 stock 从 0.93 掉到 0.85。
3. 固定报告口径：`stock_solve`、`reference_recall_exact`、`reference_recall_reactant`、`route_quality_audit` 分开。
4. 只在有 on-policy trace 后再继续蒸馏 controller，不再用离线 GT reward 硬压。
5. 如果要蒸馏 ChemEnzy proposal model，作为独立线做，不混入当前 controller 成败判断。
