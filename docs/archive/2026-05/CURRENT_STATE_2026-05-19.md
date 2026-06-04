# Current State - 2026-05-19

## Status

`整理中，主线清晰`.

当前仓库的主线已经收束成一个可运行、可审计的 ChemEnzy-backed Web 工作流：

```text
ChemEnzy native route search
  -> AutoPlanner WebUI queue/cancel
  -> material sanity product-audit
  -> raw / rejected sidecar artifacts
  -> step-level provenance display
```

## 本轮整理做了什么

- 把外部集成工作区 `AI_OS_AutoResearch/` 标记为独立 nested git repo，不再纳入本仓库提交。
- 把四个临时导出物归档到 `archive/code/generated_patches_2026-05-19/`：
  - `AI_OS_AutoResearch_autoplanner_integration_2a61a5f.bundle`
  - `AI_OS_AutoResearch_autoplanner_integration_2a61a5f.patch`
  - `AI_OS_AutoResearch_railway_autoplanner_demo_846f8ed.bundle`
  - `AI_OS_AutoResearch_railway_autoplanner_demo_846f8ed.patch`
- WebUI 保留了队列、终止、条件展示、raw/rejected sidecar、proposal provenance。
- 增加了 `scripts/monitor_autoplanner_web.py`，可从终端实时查看服务健康、GPU、任务队列和 latest artifacts。

## 当前可用能力

- ChemEnzy native search 仍是主生成器。
- RetroChimera 已作为 proposal sidecar 接入 `cascade_search`，可通过
  `--use-retrochimera-proposals` 打开，用来补齐候选反应物。
- `product-audit` 会过滤明显不守恒/不可信的路线，避免把 artifact 当结果展示。
- RouteSelector-v0 工具链已能从 ChemEnzy/CCTS route pools 构建 pack、训练
  pairwise ranker、输出 A/B/C/D 风格 ablation。
- `cascade_search` 已补出统一的 route-level `condition_state` 摘要，能汇总
  stage 级条件冲突、温度/pH 跨度、缺失条件和 same-pot 风险。
- 新任务会同时保存：
  - 主结果
  - raw sidecar
  - rejected sidecar
- 每一步 proposal 现在能看到来源、打分、条件、原子变化和证据摘要。
- 缺失 condition envelope 的 step 会被显式审计为 cascade condition conflict，
  不会被误当成可级联条件。
- 运行中可用 `PYTHONPATH=. python scripts/monitor_autoplanner_web.py --url http://127.0.0.1:7991 --once`
  做一次状态快照，或去掉 `--once` 持续监控。

## 当前仍保留但不默认推广的研究代码

- `cascade_planner/cascade_search/`
- `cascade_planner/eval/` 中的 CCTS / subgoal / value / rerank / ablation 脚本
- `AUTOPLANNRELLM/` 支线

这些代码保留是为了实验和回放，不代表已经收束为主结论。

## 当前模型结论

最新大规模 pack：

```text
results/shared/model_strengthening_20260519_v4_ccts_routepool/
```

结果摘要：

- 32,528 条去重路线，346 个 target，train/val/test 来自已有 v4
  ChemEnzy/CCTS route-pool split。
- `audit_guard` 仍是最强当前 guard：test MRR 1.0000，R@1 0.9192。
- `all_pairwise` 达到同样结果，但主要因为吃到了 audit-derived 特征，不能作为独立模型提升。
- 去掉 audit 泄漏后，`cascade_only_pairwise` 相比 native rank 有弱正信号：
  R@3 0.8485 vs 0.8182，但 R@1 0.7172 vs 0.7273。
- 结论：级联证据有信号，但还不够强，不能推进 search-time promotion。

更有价值的级联一致性模型结果：

```text
results/shared/model_strengthening_20260519_pair_scorer/
```

- 从 `dataset_v4_release/cascade_v4_high_quality.jsonl` 构建 adjacent-step
  pair pack，并排除了 `data/benchmark_v2_100.json` overlap。
- pack 规模：9,552 行，其中 3,184 个真实相邻 step 对、6,368 个 hard negatives。
- 训练集/验证集/测试集：7,509 / 978 / 1,065。
- learned pair scorer test pairwise group accuracy: 0.998688。
- rule pair baseline test pairwise group accuracy: 0.988189。
- 这是当前最接近论文主线的模型结果：它学习的是 adjacent-step cascade
  compatibility，而不是 route-level product-audit 标签。
- 但是 full100 旧 trace replay 显示，直接把 learned pair reward 加进排序会降低
  stock/no-failure/oracle-child-quality guardrail：
  `learned_w0p005` top1 stock/no-failure 0.8214 vs base 0.8452。
  因此暂不允许无约束 search-time promotion。
- guarded tie-break replay 是更安全的接入形态：
  `learned_guarded_w0p005_eps0` 保持 top1 stock/no-failure 0.8452，与 base
  持平；在 pair-informative events 上 mean top1 child quality 从 1.5059
  小幅提高到 1.5130。
- guarded tie-break 已接入 live `CascadeProgramSearch`：
  `CascadeSearchConfig(pair_reward_mode=\"guarded_tie_break\", pair_reward_tie_epsilon=...)`
  会在 base 分数近似并列且不回退 stock/no-failure 时才应用 learned pair reward。

已有 ChemEnzy runtime hard-negative transition 结果也已整理：

```text
results/shared/model_strengthening_20260519_transition_hardneg/transition_hardneg_summary.md
```

- 候选池来自 ChemEnzy runtime candidates，而不是人工构造的 pair negatives。
- 规模：train/val/test 为 173,217 / 31,267 / 29,564 candidate rows，
  对应 3,184 / 617 / 620 groups。
- test 上 ChemEnzy 原始 block-supported MRR 为 0.391640。
- 选中的 residual CCTS blend block-supported MRR 为 0.423058，
  delta 为 +0.031418。
- exact MRR 从 0.335809 提高到 0.375055。
- 但 nonlearned retrieval blend 已达到 block-supported MRR 0.422418，
  几乎追平 selected learned blend。因此这证明了 v4 evidence 对候选排序有用，
  但还没有证明 learned model 明显超过 retrieval similarity。
- 该结果只能作为 CCTS-v1 的候选级 scorer 证据，不能作为 search promotion。

追加的 no-human runtime probe 也已完成：

```text
results/shared/model_strengthening_20260519_transition_hardneg_nohuman_probe/
  runtime_hardneg_nohuman_probe.json
  runtime_hardneg_nohuman_probe.md
```

结果：

```text
retrieval control test block MRR: 0.422418
best material/product-sanity blend test block MRR: 0.422418
best delta vs retrieval: 0.000000
required delta: 0.030000
HGB runtime probe selected alpha: 0.0
```

结论：在当前固定 runtime hard-negative candidate cache 上，简单加入
material/product sanity 特征或非线性 HGB ranker 都不能超过 retrieval-only。
下一步需要换 no-human 标签/候选构造，或者直接训练 retrieval-control residual
错误，而不是继续调这组 runtime 特征。

## 下一步

1. 在同一 ChemEnzy runtime candidate pool 上比较 ChemEnzy rank、retrieval-only
   blend、learned CCTS final rerank、learned CCTS guarded in-search scoring。
2. 如果 learned CCTS 不能稳定超过 retrieval-only blend，下一步训练目标要改成
   context-shuffle / block-order / hidden-intermediate hard negatives，而不是继续堆模型。
3. 在 statin panel 上做 qualitative case study，重点看是否降低明显不兼容的
   consecutive steps，而不是只看 stock closure。

## Guarded Pipeline Status

用于下一轮 guarded CCTS 的命令 manifest 已生成：

```text
results/shared/model_strengthening_20260519_guarded_v4_pipeline/v4_full_training_pipeline_manifest.json
results/shared/model_strengthening_20260519_guarded_v4_pipeline/v4_full_training_commands.sh
```

已实跑不依赖 ChemEnzy trace 的两步：

```text
index 5: build_pair_pack
index 6: train_pair_scorer
```

输出：

```text
results/shared/model_strengthening_20260519_guarded_v4_pipeline/models/cascade_pair_scorer.pt
results/shared/model_strengthening_20260519_guarded_v4_pipeline/reports/cascade_pair_scorer.md
results/shared/model_strengthening_20260519_guarded_v4_pipeline/PROGRESS.md
```

本次 pipeline pair pack 为 6,368 行，测试 pairwise group accuracy 为
0.998688，rule baseline 为 0.984252。尚未运行 ChemEnzy-dependent trace、
action/source/transition 训练和 locked full100 guarded eval。

已额外完成一个 live guarded CCTS smoke：

```text
results/shared/model_strengthening_20260519_guarded_v4_pipeline/smoke/SMOKE_REPORT.md
```

- full100 前 5 个 target、ChemEnzy iterations=2、topk=10、cascade budget=20。
- ChemEnzy solved/cascade solved/stock closed 都是 5/5。
- pair scorer 在 live trace 中被调用：12 个 candidate applicable。
- guarded reward 实际 applied 2 次，10 次被 `outside_base_tie_window` 阻止。
- 这个结果只证明 live integration 生效，不是性能结论。

随后又跑了 full100 前 20 个 target 的 baseline vs guarded 小对照：

```text
results/shared/model_strengthening_20260519_guarded_v4_pipeline/smoke/live_guarded_pair_limit20_comparison.md
```

- aggregate 指标一致：cascade solved 0.9、stock closed 1.0、result exact 0.05、
  result GT reactant 0.35。
- guarded trace 中 pair scorer applicable 72 次，reward applied 23 次。
- top route 改变 1/20：`C=CC(=O)OC` 从一跳路线变成两步 stock-closed 路线。
- 结论：guarded mode 在小样本上安全，但还没有证明性能提升。

`tie_epsilon` 小 sweep 也已完成：

```text
results/shared/model_strengthening_20260519_guarded_v4_pipeline/smoke/guarded_pair_limit20_tie_sweep.md
```

- `eps=0.0` 与 `eps=0.03` 的 aggregate 指标完全一致。
- reward applied 从 23 增至 27，但 top-route changed 仍为 1/20。
- 结论仍是：integration 和 safety 成立，quality lift 还没有出现。
- 该对照现在由 `cascade_planner.eval.compare_cascade_search_runs` 生成，
  可复用于后续 50/100 target 或 retrieval/final-rerank 对照。

追加 weight sweep：

```text
results/shared/model_strengthening_20260519_guarded_v4_pipeline/smoke/guarded_pair_limit20_weight_tie_sweep.md
```

- `eps=0.03, weight=0.005` 与 `eps=0.03, weight=0.05` 的结果完全一致。
- top-route changed 仍为 1/20，aggregate 指标无变化。
- 这说明当前弱效应不只是权重太小；更可能是 candidate/search 状态中可被
  adjacent-pair scorer 影响的位置太少，或者 pair scorer 目标不直接对应 route
  quality。

阶段决策已写入：

```text
docs/GUARDED_CCTS_DECISION_2026-05-19.md
```

结论：adjacent-step CCTS pair reward 保留为安全诊断/tie-break，不作为主模型贡献；
下一步转向 route/block outcome scoring，并且必须证明 learned scorer 超过
retrieval-only evidence rank。

route/block 方向的当前证据已固化成一个机器可读 summary：

```text
results/shared/model_strengthening_20260519_route_block_summary/route_block_strengthening_summary.json
results/shared/model_strengthening_20260519_route_block_summary/route_block_strengthening_summary.md
```

当前门控结论：

```text
status: do_not_promote_yet
route-pool learned vs native: pass
route-pool learned vs retrieval-only: fail
runtime hard-negative learned vs ChemEnzy: pass
runtime hard-negative learned vs retrieval-only: fail
runtime no-human product/HGB probe vs retrieval-only: fail
guarded live search no-regression: pass
guarded live search quality lift: pass via no-label product-audit final rerank
```

这说明 route/block 方向比 adjacent-pair 更接近主线，但 learned scorer 仍不能
promotion。当前 live 小样本提升来自无专家标签的 product-audit conservative
final rerank，不是 learned route/block scorer；剩余 blocker 是 learned scorer
必须在 runtime hard-negative pool 上稳定超过 retrieval-only control。

`route_block_value_pack_v1` 的第一版已生成：

```text
results/shared/model_strengthening_20260519_route_block_value_v1/route_block_value_pack.jsonl
results/shared/model_strengthening_20260519_route_block_value_v1/route_block_value_pack_report.json
```

规模：

```text
rows: 32528
targets: 346
train / val / test: 18998 / 3946 / 9584
pair_context_evidence_any: 8211
strong_route_evidence: 787
reject_artifact: 4616
```

这个 pack 只分离 feature groups 和 weak label tasks，不产生手写综合分数。

第一版 route/block value model trainer 已加入并完成三组 pilot：

```text
cascade_planner/eval/train_route_block_value_model.py
results/shared/model_strengthening_20260519_route_block_value_v1/models/
```

当前结论：

```text
strong_route_evidence with retrieval: model MRR 1.0000, retrieval MRR 1.0000
strong_route_evidence without retrieval/audit/CCTS: model MRR 0.6531, retrieval MRR 1.0000
reviewable_vs_reject without audit: model MRR 0.8699, native MRR 0.8511, retrieval MRR 0.7908, audit_guard MRR 1.0000
```

也就是说，`strong_route_evidence` 不能作为主标签，因为它基本就是 retrieval
阈值；`reviewable_vs_reject_no_audit` 有弱模型信号，但仍未超过 audit guard。
下一步需要更好的 outcome label，而不是继续围绕现有 retrieval-derived label
调模型。

已检查现有 route-pool review 标签：

```text
human CSV review accepted_rows: 0 / 51
self-review calibration accepted_rows: 36
self-review usable_positive_rows: 5
self-review usable_negative_rows: 31
```

这些 self-review 标签只能做 calibration/label-design 参考，不能作为主监督数据。

已把它们规范化成 calibration-only pack：

```text
results/shared/model_strengthening_20260519_route_block_review_labels/route_block_review_label_pack.jsonl
results/shared/model_strengthening_20260519_route_block_review_labels/route_block_review_label_pack_report.json
```

报告确认：36 行全部来自 self-review，5 个 usable positive、31 个 usable
negative，`sufficient_for_main_training=false`。

为补这个标签缺口，已从 20/full100/statin 三个 route-pool evidence audit
生成 150 行扩展审阅 worklist：

```text
results/shared/model_strengthening_20260519_review_expansion/route_pool_evidence_review_expansion_worklist.csv
results/shared/model_strengthening_20260519_review_expansion/route_pool_evidence_review_expansion_worklist.jsonl
results/shared/model_strengthening_20260519_review_expansion/route_pool_evidence_review_expansion_worklist_report.json
```

组成：

```text
any_analog_supported: 75
multistep_without_observed_pair: 74
same_pair_analog_supported: 1
20/full100/statin pools: 51 / 49 / 50
```

附带 transform sanity 诊断显示 `85/150` 行有 transform-label warning，
因此该 worklist 只能作为人工/LLM 审阅入口，不能直接训练。

已用现有 review pipeline 做 dry-run 验证：

```text
results/shared/model_strengthening_20260519_review_expansion/dryrun_pipeline/
prompt_rows: 150
written_rows: 150
usable_positive_rows: 0
usable_negative_rows: 0
ready_for_training: false
```

这只证明 prompt/response/label/gate 流程打通；dry-run 标签全是 placeholder，
不能用于训练。

review label pack builder 已加 `placeholder_review` 防护，并用 dry-run labels
验证：

```text
results/shared/model_strengthening_20260519_review_expansion/dryrun_pipeline/expansion_dryrun_review_label_pack_report.json
placeholder_review: 150
usable_positive_rows: 0
usable_negative_rows: 0
```

阶段 completion audit 已写入：

```text
docs/MODEL_STRENGTHENING_COMPLETION_AUDIT_2026-05-19.md
```

结论：计划尚未完成，主要缺口是真实非 placeholder route/block outcome labels
以及 learned scorer 超过 retrieval/audit controls 的证据。

真实 LLM 审阅 runner 已准备好：

```text
scripts/run_route_block_review_expansion_real.sh
```

运行前需要设置 `DEEPSEEK_API_KEY`。脚本会执行非 dry-run review pipeline，
并自动生成 `expansion_real_review_label_pack` 与 promotion gate。
脚本也会从 `.env.local` 或 `.env` 读取 `DEEPSEEK_API_KEY`，避免在命令行中暴露 key。
仓库提供 `.env.local.example` 占位模板；复制成 `.env.local` 后再填入真实 key。

## 2026-05-19 Strict Review Update

后续方向已经从 150 行通用 expansion worklist 收束到 strict
model-control disagreement review：

```text
results/shared/model_strengthening_20260519_strict_model_review_worklist/
  strict_model_control_disagreement_review.jsonl
  strict_model_control_disagreement_review.csv
  strict_model_control_disagreement_prompts.jsonl

results/shared/model_strengthening_20260519_strict_model_review_worklist_extended/
  strict_model_control_disagreement_review_300.jsonl
  strict_model_control_disagreement_review_300.csv
  strict_model_control_disagreement_prompts_300.jsonl
```

当前准备状态：

```text
120-row strict worklist: ready, 47 targets
300-row fallback worklist: ready, 78 targets
human/external packet archive: ready
strict packet CSVs include immutable target_id, route_id, source_value_pack, and value_split context columns
readiness validates packet metadata, context columns, archive checksum, and 11 required archive members
route_id / target_id / value_split preserved through prompt -> ingest -> label pack -> merge
human CSV ingest requires route_id on reviewed rows and preserves it for filled packet merge
human CSV ingest and label-pack reports expose value_split counts for pre-merge split-balance audit
dry-run merge: 120/120 route ids matched, 0 usable labels because placeholder
mock non-dry-run smoke: 120/120 route ids matched, 120 positive / 0 negative, training refused
human CSV positive-path smoke: synthetic filled CSV matched 6/6 route ids and passed merge gate under test thresholds
strict continuation wrappers: incomplete expert-training gates exit 4 unless ALLOW_NOT_READY_EXIT_ZERO=1
deepseek placeholder guard: wrappers, strict review runner, shared client, and agent prior/benchmark/CLI paths reject or mark quoted/padded template values unusable
pytest.ini scoped project tests: 472 passed, 4 warnings
```

可交付 packet：

```text
results/shared/model_strengthening_20260519_strict_review_packets.tar.gz
results/shared/model_strengthening_20260519_strict_review_packets.tar.gz.sha256
sha256: 070279a0e9b1faed5a7f6bfcb614bf50298b7e9c92d0488622f5ece4eb4f62c1
```

机器可读 readiness：

```text
results/shared/model_strengthening_20260519_strict_review_readiness.json

strict_120_ready_for_real_review: true
strict_300_fallback_ready: true
external_packet_ready: true
real_review_can_run_now: false
filled_expert_csv_available: false
filled_expert_csv_rows: 0
ready_for_expert_value_training: false
```

`filled_expert_csv_rows` counts rows with at least one filled expert decision
field: `expert_route_plausible`, `expert_block_transform_correct`,
`expert_support_precedent_relevant`, `expert_cascade_coherent`, or
`expert_priority`; comments-only rows do not count.

主要脚本：

```text
scripts/run_strict_model_review_real.sh
scripts/run_strict_model_review_real_extended.sh
scripts/run_strict_review_full_after_key.sh
scripts/run_strict_review_from_filled_csv.sh
scripts/train_strict_model_review_value_models.sh
cascade_planner/eval/merge_route_block_review_labels.py
cascade_planner/eval/check_strict_review_pipeline_readiness.py
```

Expert-review 分支当前状态：

```text
DEEPSEEK_API_KEY is not configured
no filled expert CSV rows are available
no merged review value pack has passed expert-training gate
```

该分支保留为可审计 fallback，不再作为主线阻塞项。原因是后续不会依赖专家
标签；不能再把 `ready_for_expert_value_training=false` 当作整体研究目标未完成
的唯一判断。

## No-Human Route/Block Training

当前主线改为不需要专家标签的 route/block weak supervision。runtime
train-provenance value pack 已扩展出以下自动标签任务：

```text
no_human_route_positive
no_human_route_negative
no_human_consensus_positive
no_human_consensus_negative
```

当前 pack：

```text
results/shared/model_strengthening_20260519_route_block_value_runtime_train_provenance/
```

关键计数：

```text
rows: 32528
train/val/test: 18998 / 3946 / 9584
no_human_consensus_positive: 7765
no_human_consensus_negative: 4616
retrieval provenance: verified_or_no_retrieval_features
```

no-human ablation 已训练完成：

```text
results/shared/model_strengthening_20260519_no_human_route_block_value_models/
```

test MRR：

```text
native rank baseline: 0.707357
retrieval-only baseline: 0.761806
no_human_all_features: 1.000000
no_human_no_audit: 0.797227
no_human_no_audit_no_retrieval: 0.782899
no_human_no_audit_no_cascade: 0.816142
no_human_route_no_audit_no_retrieval: 0.871659
```

结论：固定 route-pool 内的 no-human weak-supervision strict gate 已通过；最佳
无专家控制模型是 `no_human_route_no_audit_no_retrieval`，比 retrieval-only 高
`0.080862` MRR。但 search-time promotion 仍为 `False`；下一步必须证明 audit
control 与 guarded live-search aggregate quality lift，而不是继续等待 expert CSV。

final-rerank replay 已生成：

```text
results/shared/model_strengthening_20260519_no_human_route_block_value_models/
  no_human_route_no_audit_no_retrieval_final_rerank_replay.json
  no_human_route_no_audit_no_retrieval_final_rerank_replay.md
```

test split 结果：

```text
route_block_value_model MRR: 0.871659
native rank MRR: 0.851071
retrieval-only MRR: 0.790797
audit-guard MRR: 1.000000
model - retrieval: +0.080862
model - audit: -0.128341
top route changed vs native: 51 / 99 groups
```

guarded live-search sweep 已补齐 `tie_epsilon=0.08`，并加入 additive 探针：

```text
baseline:              top exact 0.00 / top GT 0.30 / any exact 0.05 / any GT 0.35 / changed 0
guarded eps0 w0.005:   top exact 0.00 / top GT 0.30 / any exact 0.05 / any GT 0.35 / changed 1 / applied 23
guarded eps0.03 w0.005 top exact 0.00 / top GT 0.30 / any exact 0.05 / any GT 0.35 / changed 1 / applied 27
guarded eps0.08 w0.005 top exact 0.00 / top GT 0.30 / any exact 0.05 / any GT 0.35 / changed 1 / applied 27
guarded eps0.03 w0.05  top exact 0.00 / top GT 0.30 / any exact 0.05 / any GT 0.35 / changed 1 / applied 27
additive w0.05:        top exact 0.00 / top GT 0.30 / any exact 0.05 / any GT 0.35 / changed 1 / applied 72
route/block final:     top exact 0.00 / top GT 0.25 / any exact 0.05 / any GT 0.35 / changed 7 / enabled 20
product-audit final:   top exact 0.00 / top GT 0.35 / any exact 0.05 / any GT 0.35 / changed 4 / enabled 20
```

结论更新：无标签 product-audit conservative final rerank 在 20-target smoke 上
带来小幅 top-GT 提升（0.30 -> 0.35），且不降低 solved/stock/any-result 指标；
但 learned route/block value final rerank 仍是负向证据（0.30 -> 0.25）。因此
当前可推广的是 no-expert product-audit guard/rerank 思路，不能推广 learned
route/block scorer。
