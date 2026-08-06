# W6 ChemEnzy 嵌入首损边界与减负报告

日期：2026-08-06  
验证提交：`3dfcd75`  
状态：W6 定因与通用修复完成；尚未进入 RetroStar-190 全量运行。

## 1. 问题与约束

本轮回答三个问题：

1. ChemEnzy 独立运行成功、嵌入 AutoPlanner V4 后表现下降，是否由 ChemEnzy 本体退化造成；
2. 路线第一次在哪个可审计边界丢失；
3. 在不按目标、分子或 benchmark 分组、不接入隔离模型、不放宽科学 validator 的前提下，能否恢复 RetroStar 可比的 B4 指标并消除运行空转。

本轮使用 Nirmatrelvir 的既有成功高级配置与缓存 provider 输出，所有 current-V4 replay 均为零新模型调用。RetroStar 可比主指标按冻结库存下至少一条 target-rooted、host-selected、终端叶全部 stock-closed 的路线计算，即 B4；B2/B3/B5 继续作为独立科学成熟度轴报告。

## 2. 冻结输入

- Standalone 成功运行：`results/.autoplanner/runs/v4-proof-nirmatrelvir-20260717-repair1--8ca91baaa17f`
- Current replay：`results/.autoplanner/w6-nirmatrelvir-current-replay-20260806-e`
- Current report：`results/.autoplanner/w6-nirmatrelvir-current-replay-20260806-e/runs/w6-nirmatrelvir-current-replay-20260806-e--9efddbe08f8f/target-only-solve-report.json`
- 逐路线比较：`results/.autoplanner/w6-nirmatrelvir-current-replay-20260806-e/chemenzy-embedding-comparison.json`
- Frozen stock：RetroStar-190 eMolecules，约 2308 万条
- Stock index SHA-256：`30c828d6780e534d8368f4eb74f844c889683453080d44053ba298a7bebdd79c`
- Provider route reserve：2
- Host route portfolio：2
- 单次 stock batch 上限：24
- 新模型调用：0

## 3. 逐层结果

| 边界 | 数量 | 结论 |
| --- | ---: | --- |
| Standalone ChemEnzy routes | 39 | 原始成功候选集 |
| Embedded ChemEnzy routes | 39 | 无嵌入前丢失 |
| Raw provider digest parity | true | 缓存 payload 完全一致 |
| Normalized route multiset parity | true | 规范化路线集合完全一致 |
| Host-selected routes | 2 | 由统一 portfolio budget 截断 |
| Fully materialized routes | 2 | selected 路线均进入 canonical graph |
| Host-validated routes | 0 | 严格 reaction validator 未闭合整条路线 |
| Stock-closed routes | 1 | B4 恢复 |

首损计数：

- `host_portfolio_budget_truncated`：37；
- `materialized_not_host_validated`：2。

因此，ChemEnzy 本体没有退化。39 条路线在 raw 与 normalized 两层均保持完全一致；生产轨迹中的主要缩减发生在显式、统一、可审计的 host portfolio budget，而不是 provider 嵌入或目标特判。

## 4. B4 下降的真实原因

此前 RetroStar 类结果下降的直接原因不是缺少“全局视野”，而是 stock action 与 materialization 边界错位：

- stock frontier 曾对未入选或尚未物化的路线叶子发动作；
- 超过单批上限时曾把整条路线视为不可审，而不是拆成 bounded batches；
- canonical graph 没有始终作为最终 selected/materialized leaf boundary 的唯一投影来源。

修复后：

- `max_live_stock_molecules` 只表示单次 Action 批量上限；
- stock audit 只处理当前 selected、materialized leaf boundary；
- 最终 stock closure 由 canonical graph 统一投影；
- stock action 从早期轨迹的 25 次降至 2 次；
- B4 从 false 恢复为 true。

## 5. Director 与 validation 减负

### 5.1 Initial Director

旧 replay 中 `codex_global_architecture` 被调度 32 次。第一次 Director 已返回 accepted，但所有 proposal 被 host admission gate 拒绝，图中没有 `codex_global_director` 来源 hypothesis；旧 deficit 编译器因此把“没有 Codex hypothesis”误判为“初始架构从未尝试”。

提交 `3dfcd75` 将一次终结的 initial Director attempt 记录为 canonical operational state。即使 accepted plan 的所有 proposal 都被 host gate 拒绝，初始架构也只运行一次；后续由新 canonical event 触发的 event replan 不受影响。

| Action kind | 修复前 replay `-d` | 修复后 replay `-e` |
| --- | ---: | ---: |
| `chemenzy_target_expand` | 1 | 1 |
| `codex_global_architecture` | 32 | 1 |
| `host_materialize` | 30 | 30 |
| `reaction_validate` | 30 | 30 |
| `stock_audit` | 2 | 2 |
| 总计 | 95 | 64 |

本次通用修复减少 31 个无收益 Action，Action 总量下降约 32.6%，模型调用仍为 0。

### 5.2 Reaction validation

当前 30 个物化边中 13 个 accepted、17 个 rejected。主要拒绝诊断为：

- `reaction_edit_budget_exceeded`：9；
- `bond_change_audit_not_eligible_for_transform_reapply`：7；
- `mapping_audit_not_eligible_for_transform_reapply`：6；
- `mapping_consistent_without_trusted_transform_or_precedent`：4；
- `reaction_centre_not_in_deterministic_transform_registry`：4；
- `product_heavy_atom_without_reactant_provenance`：3。

同一 validator 版本已有负验证、且没有新证据时不再重复验证。本轮不放宽 validator 来换取 B2；B2 保持 false 是当前严格科学边界，不能与 RetroStar 的 B4 搜索指标混报。

## 6. 最终 Gates

| Gate | 结果 |
| --- | --- |
| B0 blind input | true |
| B1 global multi-route | true |
| B2 host-validated routes | false |
| B3 exact multi-source | false |
| B4 frozen-stock boundary | true |
| B5 configured scientific acceptance | false |

结论：当前架构已经恢复 RetroStar 可比的库存闭合能力，同时保留严格 reaction/evidence acceptance，不再通过 benchmark 专用 solver、目标分组或 validator 放宽制造虚假提升。

## 7. 验证与 W7 后续状态

- 新增回归覆盖：Director accepted、所有 proposal rejected 后 initial architecture 不再生成；后续 graph revision 不会恢复该 deficit；event replan 仍可进入 frontier。
- Ruff：通过。
- 聚焦回归：44 passed。
- 当时的扩展回归为 67 passed、7 failed；同一 7 项曾在未修改的 `d581f66` 基线逐项复现，因此不是 `3dfcd75` 引入。
- W7 已清除上述 7 项，扩展 W7 集合达到 74 passed。
- 除已批准延期的超行数预算测试外，完整离线套件达到 2632 passed、3 skipped、1 deselected、2 subtests passed。
- RetroStar-190 fresh blind preflight 已达到 190/190；正式全量运行仍未开始。

## 8. W7 进入条件

W7 收束状态：

1. 已清理现有 7 个基线失败；
2. 已生成 `benchmarks/retrostar190_w7_freeze_20260806.json`；
3. 已通过除批准延期项外的完整离线门；
4. 已保存 RetroStar-001/002/003 既有阳性 smoke，避免无必要重跑 001 parity；
5. 尚待一次最终 Nirmatrelvir zero-model replay；
6. 该 replay 通过后才允许启动统一配置的 RetroStar-190 与全目标消融。

完整 W7 证据见 `docs/architecture/W7_FREEZE_AND_PREFLIGHT_20260806.md`。
