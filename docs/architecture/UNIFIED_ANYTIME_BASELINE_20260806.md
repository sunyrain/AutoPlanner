# Unified Anytime 基线与 Checkpoint A 现场证据

日期：2026-08-06
状态：RetroStar-190 目标 001 单目标成对复现完成；不得外推为 190 全量结论。

## 1. 目的

本记录验证两个独立问题：

1. AutoPlanner V4 嵌入是否改变了 ChemEnzy 在相同高级配置下生成的路线集合；
2. B4 库存闭合是否仍被当成 benchmark 专用终态，从而形成两套 solver 的审稿风险。

本轮只运行一个真实目标，不调用 Codex 模型，不重复执行全套 benchmark。

## 2. 冻结输入

- Case：`retrostar190-001-ddec287bee`
- Target：`C[C@H](c1ccccc1)N1C[C@]2(C(=O)OC(C)(C)C)C=CC[C@@H]2C1=S`
- Stock index SHA-256：`30c828d6780e534d8368f4eb74f844c889683453080d44053ba298a7bebdd79c`
- ChemEnzy environment：`D:\conda\envs\py312`
- Search preset：`standard`
- Maximum steps：14
- Iterations：60
- Expansion top-k：100
- Provider route reserve：16
- Host route portfolio：8
- Timeout：600 s
- Model invocation budget：0

旧成功运行：

`results/.autoplanner/benchmark190-optimized-20260727/retrostar001-search-only-v2/`

新统一运行：

`results/.autoplanner/unified-anytime-checkpoint-a-20260806/retrostar001-v2/`

## 3. 结果

| 指标 | 旧成功运行 | 新统一运行 |
| --- | ---: | ---: |
| ChemEnzy stage elapsed | 207.659 s | 182.889 s |
| 主 proposal routes | 16 | 16 |
| host-search eligible | 16 | 16 |
| host portfolio selected | 8 | 8 |
| target-rooted measured routes | 4 | 4 |
| B4 stock-closed routes | 3 | 3 |
| Codex model calls | 0 | 0 |

新比较器输出：

`results/.autoplanner/unified-anytime-checkpoint-a-20260806/retrostar001-v2/chemenzy-embedding-comparison.json`

核心结论：

- `normalized_route_multiset_equal = true`；
- standalone 和 embedded 均得到 32 个可追踪 provider rows，其中包含主 routes 与 quarantined advisory rows；
- 8 条进入 host portfolio；
- 8 条 selected routes 均进入 canonical route family 并完成物化；
- 其中 3 条达到冻结库存 B4；
- raw result 文件摘要不同，但规范化路线 multiset 完全相同。raw payload 包含运行时元数据，因此 raw 文件摘要不同不能证明化学候选不同。

## 4. 首次损失边界

本轮没有发现 standalone ChemEnzy 路线在 embedded normalization 之前丢失。

候选缩减发生在明确且可审计的边界：

- host portfolio 只选择 8 条主路线；
- 其余 eligible 主路线因 portfolio budget 暂不进入生产选择；
- quarantined routes 只保留有限 advisory 展示；
- 8 条 selected/materialized routes 中有 3 条库存闭合，另外 5 条停在 stock-open。

因此目标 001 的证据不支持“V4 嵌入改坏 ChemEnzy 原生生成器”。此前差结果更可能来自：

- 低配搜索预算或不同 ChemEnzy 环境；
- host portfolio 截断；
- stock/validation/acceptance 投影差异；
- 旧 benchmark 专用控制流掩盖了真实科学状态。

该判断目前只对目标 001 成立。仍需对 standalone 成功而 embedded 失败的真实目标执行相同 lineage diff。

## 5. 统一控制流验证

新运行不再在 B4 处走 benchmark 专用 finalize：

- B4 仍为 `true`，3 条库存闭合路线完整保留；
- B2、B3、B5 为 `false`；
- disposition 为 `stock_closed_proof_open`；
- `objective_mode` 只作为兼容输出元数据；
- B4 milestone 不再决定 kernel completion；
- 同一运行可继续 reaction validation、exact evidence、conditions 和 Program 工作。

这消除了“benchmark 达到 B4 即采用另一套 solver 终止”的直接代码和运行证据。

## 6. 同步发现并修复的问题

统一执行原先 benchmark 禁用的 condition enrichment 后，暴露了 condition worker 在新 graph revision
重放相同预测时复用旧 reservation idempotency key 的冲突。command identity 现已绑定 input revision 和
dependency revisions，避免同一逻辑动作在不同 canonical revision 上产生账本冲突。

模型调用预算为 0 时，Global Director reservation 原先会抛出未捕获的 `RunKernelBudgetError`。现在该情况
转为具名 `skipped` DirectorOutcome，后续 deterministic ChemEnzy、stock 和验证工作可以继续。

## 7. 下一门

Checkpoint A 尚未完全关闭。下一项必须选择一个“standalone 成功、embedded 失败”的真实目标，使用相同工具输出：

- normalized proposal parity；
- host selection loss；
- canonical admission/materialization loss；
- stock closure loss；
- 首个确定性拒绝理由。

在找到该失败目标的首个损失边界前，不依据 RetroStar-190 测试目标调整 scheduler 权重。
