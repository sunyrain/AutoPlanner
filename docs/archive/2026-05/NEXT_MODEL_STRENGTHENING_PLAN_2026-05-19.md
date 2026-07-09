# 下一阶段模型强化计划 - 2026-05-19

## 当前判断

当前系统不能再继续沿着“训练一个替代 ChemEnzy 的 planner/student”走。
已经验证过的事实是：

- ChemEnzy native route pool 是目前最强 generator。
- AutoPlanner/CCTS 的价值不在生成新反应，而在筛选、解释、审计和轻量影响搜索。
- learned scorer 目前能打过 ChemEnzy 原始排序的一些场景，但没有稳定打过
  retrieval-only / audit guard，因此不能宣称模型已经学到独立的路线价值函数。
- guarded in-search tie-break 已经能安全接入，但在 20-target 小样本上没有质量提升。

所以 Phase II 的主线应收束为：

```text
ChemEnzy native route pool
  -> strict provenance route/block features
  -> no-human route/block weak supervision
  -> learned route/block value model
  -> compare against native / retrieval-only / audit guard
  -> only then guarded in-search integration
```

## 目标

下一阶段只追求一个核心问题：

```text
在固定 ChemEnzy route pool 的前提下，
我们能否训练出一个比 native rank、retrieval-only 和 product-audit guard
更可靠的 route/block value model？
```

如果不能做到这一点，就不要继续把 scorer 接进 search 内部，也不要把它作为论文主贡献。

## 当前可用资产

严格 provenance route/block value pack：

```text
results/shared/model_strengthening_20260519_route_block_value_runtime_train_provenance/
  route_block_value_pack.jsonl
  route_block_value_pack_report.json
  strict_route_block_value_ablation_summary.json
  strict_route_block_value_ablation_summary.md
```

关键信息：

```text
rows: 32528
targets: 346
split: train 18998 / val 3946 / test 9584
evidence provenance: verified_or_no_retrieval_features
runtime_retrieval_only: true
no_human_consensus_positive: 7765
no_human_consensus_negative: 4616
```

no-human ablation 已训练完成：

```text
results/shared/model_strengthening_20260519_no_human_route_block_value_models/
```

当前 test MRR：

```text
native rank baseline: 0.707357
retrieval-only baseline: 0.761806
no_human_all_features: 1.000000
no_human_no_audit: 0.797227
no_human_no_audit_no_retrieval: 0.782899
no_human_no_audit_no_cascade: 0.816142
no_human_route_no_audit_no_retrieval: 0.871659
```

这说明固定池训练 strict gate 已过；最佳无专家控制模型是
`no_human_route_no_audit_no_retrieval`，比 retrieval-only 高 `0.080862` MRR。
但 search-time promotion 仍是 `False`。

严格 model-control disagreement review worklist：

```text
results/shared/model_strengthening_20260519_strict_model_review_worklist/
  strict_model_control_disagreement_review.jsonl
  strict_model_control_disagreement_review.csv
  strict_model_control_disagreement_prompts.jsonl
  strict_model_control_disagreement_review_report.json
```

关键信息：

```text
selected rows: 120
targets: 47
purpose: 挑出 model / retrieval / audit / native 排序不一致的多步路线
```

扩展 fallback worklist 已预生成：

```text
results/shared/model_strengthening_20260519_strict_model_review_worklist_extended/
  strict_model_control_disagreement_review_300.jsonl
  strict_model_control_disagreement_review_300.csv
  strict_model_control_disagreement_prompts_300.jsonl
  strict_model_control_disagreement_review_300_report.json
```

关键信息：

```text
selected rows: 300
targets: 78
purpose: 如果 120 条真实 review 不够，直接扩展到 300 条 disagreement samples
```

人工/外部 review packet 也已预生成，但现在只作为 fallback/审计路径，不再是
主线阻塞项：

```text
results/shared/model_strengthening_20260519_strict_model_review_packet/
  route_pool_evidence_review_calibration_subset_TO_FILL.csv
  route_pool_review_calibration_packet.json
  README.md

results/shared/model_strengthening_20260519_strict_model_review_packet_300/
  route_pool_evidence_review_calibration_subset_TO_FILL.csv
  route_pool_review_calibration_packet.json
  README.md
```

用途：

如果当前机器没有 DEEPSEEK_API_KEY，可以把 TO_FILL.csv 交给外部 LLM
或人工审阅；填完后可用封装脚本导回、按 `route_id` merge，并只在 expert
gate 通过时训练：

CSV 里带有不可改动的上下文字段 `target_id`, `route_id`,
`source_value_pack`, `value_split`。外部审阅时保留这些列不变，并用
`value_split` 保证 train/val/test 中都有 positive 与 negative。

```bash
PYTHONPATH=. scripts/run_strict_review_from_filled_csv.sh
PACKET_SIZE=300 PYTHONPATH=. scripts/run_strict_review_from_filled_csv.sh
```

如果 merge 后 expert-training gate 仍未通过，封装脚本退出 `4`；仅做检查且不想
让 shell 命令失败时才设置 `ALLOW_NOT_READY_EXIT_ZERO=1`。

当前 dry-run review 只能验证流程，不能训练：

```text
usable_positive_rows: 0
usable_negative_rows: 0
label_positive_counts.placeholder_review: 120
```

## 当前阻塞

主线阻塞不是专家标签，而是 learned scorer 还没有通过 search-time promotion：

```text
runtime hard-negative learned scorer does not clear retrieval-only control
guarded live search has no aggregate quality lift
```

下一步不再等待 `DEEPSEEK_API_KEY` 或 filled expert CSV。应直接围绕
no-human scorer 做更严格的 audit-control 和 live-search replay。

## 阶段 1：No-Human 固定池训练

主训练任务不再依赖专家标签。当前 value pack 已经内置 no-human weak label：

```text
positive:
  no_human_consensus_positive
  no_human_route_positive

negative:
  no_human_consensus_negative
  no_human_route_negative
```

这些标签只来自可复现信号：stock closure、product-audit class、large atom gain、
generic template、route/block evidence 和 runtime retrieval provenance。它们不是
人工或 LLM 评分，也不需要未来补专家 CSV。

当前固定池训练命令：

```bash
PYTHONPATH=. scripts/train_no_human_route_block_value_models.sh
```

脚本训练并汇总 consensus-task 与 route-task 两组 ablation：

```text
no_human_consensus_*
no_human_route_*
```

判读时以 `*_no_audit_no_retrieval` 为关键控制项；只有这个控制模型超过
retrieval-only 且 margin 达到阈值，才说明 scorer 不只是复制 audit 或 retrieval
proxy。

## 阶段 2：Audit-Control Replay

每次训练后必须同时报告：

```text
ChemEnzy native rank MRR
retrieval-only baseline MRR
product-audit guard selected model
learned model without audit features
learned model without retrieval features
learned model without audit/retrieval/cascade features
```

当前 no-human 固定池结果：

```text
native rank baseline: 0.707357
retrieval-only baseline: 0.761806
no_human_all_features: 1.000000
no_human_no_audit: 0.797227
no_human_no_audit_no_retrieval: 0.782899
no_human_no_audit_no_cascade: 0.816142
no_human_route_no_audit_no_retrieval: 0.871659
```

结论：固定 route pool 上 no-human strict gate 已通过，但这还不足以推广到
search-time。

final-rerank replay 已确认固定池重排正信号：

```text
route_block_value_model MRR: 0.871659
native rank MRR: 0.851071
retrieval-only MRR: 0.790797
audit-guard MRR: 1.000000
model - retrieval: +0.080862
model - audit: -0.128341
top route changed vs native: 51 / 99 groups
```

但 live final rerank 不能 promotion：`route_block_final_rerank` 在 full100
前 20 个 target 上启用 20/20、改变 7/20 个 top route，但 top GT reactant 从
baseline 的 0.30 降到 0.25，any-result GT 仍是 0.35。

同一 20-target smoke 上，新增的无专家标签 product-audit conservative final
rerank 启用 20/20、改变 4/20 个 top route，把 top GT reactant 从 0.30 提到
0.35，any-result GT 仍为 0.35。这说明可继续使用 no-human guard/rerank 信号；
但它不是 learned route/block scorer 的 search-time promotion 证据。

因此下一步不是继续证明 fixed-pool，也不是放宽 gate，而是解决 learned scorer
在 runtime hard-negative pool 上只比 retrieval-only 高 0.00064 的问题。

已追加验证的无专家标签 probe：

```text
results/shared/model_strengthening_20260519_transition_hardneg_nohuman_probe/runtime_hardneg_nohuman_probe.md
```

结论：material/product sanity pairwise 特征和 HGB runtime ranker 都没有超过
retrieval-only；最佳 blend test block MRR 仍为 0.422418，delta 为 0.0。因此
下一步必须改训练目标，而不是继续调同一组 runtime features。

## 阶段 3：Promotion Gate

只有同时满足以下条件，才进入 search-time：

```text
learned MRR >= retrieval-only MRR + 0.03
learned MRR >= audit-guard MRR + 0.01
top1 reject_artifact_rate 下降
top3 reviewable_route_count 上升
stock_closed_reviewable_rate 不下降超过 0.03
bootstrap CI lower bound > 0 for learned-vs-retrieval
```

如果只满足 learned > native，但不满足 learned > retrieval/audit，结论是：

```text
级联 evidence 有用，但当前学习器没有证明超过简单检索或规则审计。
```

## 阶段 4：Search-Time 接入

满足 promotion gate 后，再做最小 search-time integration。

优先顺序：

```text
1. final route rerank
2. guarded tie-break in partial route ranking
3. child transition soft bias
4. open-leaf soft priority
```

不允许一开始硬剪枝 ChemEnzy candidates。初始配置：

```text
mode: guarded_tie_break
weight sweep: 0.005 / 0.02 / 0.05 / 0.10
tie_epsilon sweep: 0.0 / 0.03 / 0.08
runtime gate: 20-30s per target 可接受
```

Search-time promotion 必须证明：

```text
in-search > final rerank only
no stock/solve guardrail regression
top route actually changes on enough targets
quality lift is visible, not only applied-count 增加
```

## 阶段 5：Statin Panel

他汀类药物只作为应用案例，不用于特异性训练。

要报告：

```text
raw route count
kept route count
rejected route count
top1/top3 no-human weak-label class
main rejection reasons
condition / EC confidence
stock shortcut risk
是否出现可解释的核心 disconnection
```

目标不是刷 GT exact，而是判断产品场景下路线是否可审阅、是否明显荒谬、是否比
ChemEnzy 原始排序更适合交给化学家看。

## 不再继续的方向

以下方向暂时停止：

- student-only controller 替代 ChemEnzy。
- 用 full100 GT exact/GT reactant 作为主训练目标。
- 把 gold/silver 当 route preference。
- 继续用 dry-run 或 placeholder review 标签训练。
- 只靠 rule-post 分数宣称模型贡献。
- 在 learned scorer 没打过 retrieval-only 之前推进 search-time promotion。

## 下一步命令

主线下一步是继续 no-human scorer 的 audit-control 和 live-search replay：

```bash
PYTHONPATH=. scripts/train_no_human_route_block_value_models.sh
```

刷新总汇总：

```bash
PYTHONPATH=. python -m cascade_planner.eval.summarize_route_block_strengthening \
  --route-pool-report results/shared/cascadebench_strict_20260516/route_pool_pairwise_ranker_v4_structured_train200_test100/route_pool_pairwise_ranker_report.json \
  --route-block-value-report results/shared/model_strengthening_20260519_route_block_value_runtime_train_provenance/route_block_value_pack_report.json \
  --no-human-ablation-summary results/shared/model_strengthening_20260519_no_human_route_block_value_models/no_human_route_block_value_ablation_summary.json \
  --ablation-summary results/shared/cascadebench_strict_20260516/route_pool_pairwise_ranker_ablation_v4_structured_train200_test100/ablation_summary.json \
  --bootstrap-stability results/shared/cascadebench_strict_20260516/route_pool_pairwise_ranker_ablation_v4_structured_train200_test100/bootstrap_stability.json \
  --transition-hardneg-summary results/shared/model_strengthening_20260519_transition_hardneg/transition_hardneg_summary.json \
  --guarded-search-comparison results/shared/model_strengthening_20260519_guarded_v4_pipeline/smoke/guarded_pair_limit20_weight_tie_sweep.json \
  --output-json results/shared/model_strengthening_20260519_route_block_summary/route_block_strengthening_summary.json \
  --output-md results/shared/model_strengthening_20260519_route_block_summary/route_block_strengthening_summary.md
```

随后只在 fixed-pool gate 和 replay 都清楚通过时，才进入 search-time sweep。
当前不是专家标签阻塞，而是以下 promotion gate 仍未通过：

```text
runtime hard-negative learned scorer does not clear retrieval-only control
guarded live search has no aggregate quality lift
```

expert-review 脚本只保留为 fallback/审计路径，不作为主线前置条件：

```bash
PYTHONPATH=. scripts/run_strict_review_from_filled_csv.sh
PACKET_SIZE=300 PYTHONPATH=. scripts/run_strict_review_from_filled_csv.sh
PYTHONPATH=. scripts/run_strict_model_review_real_extended.sh
```
