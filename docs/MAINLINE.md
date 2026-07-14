# V4 主线架构

更新：2026-07-13

## 理想状态

Codex 是 campaign 级全局 Director，而不是单步模板。一次调用读取有界的完整上下文：
目标、路线族、多步骨架、共享中间体、规范超图、来源/库存事实、冲突、失败、frontier、
当前 Pareto 组合和剩余预算。它可以同时替换几条路线、合并共享上游、淘汰支配路线，
或改变证据优先级；输出仍然只是 proposal。

主机侧把全局策略升级为事实：

```text
target + acceptance + hard budgets
                |
                v
     Codex GlobalCampaignDirector
   (initial / event replan / final)
                | proposal only
                v
 canonical admission + early chemistry gates
                v
 materialize → validate → exact evidence → stock audit
                |                         |
                +--- one deficit frontier+
                v
 proof stitcher → small diverse portfolio → acceptance
                v
 bounded incremental workbench
```

所有结果写回同一个 `RunKernel` 和 canonical hypergraph。CLI、API、Web、恢复和导出
只是适配器，不持有化学状态。

## 为什么 Blackboard 不再是核心

Blackboard 适合协作记录，不适合表达 AND/OR 反应超图。旧流程把 expansion、proof、
stock 和 UI 状态复制到多个字典与队列，导致同一条边在一个模块已关闭、另一个模块
仍开放。更多模型调用只会增加 L0 文本，不能修补最弱的验证或库存缺口。

V4 只保留：

- 运行与预算：`RunKernel`
- 化学拓扑：`canonical_retrosynthesis_hypergraph.v1`
- 下一项工作：`deficit_frontier.v1`
- 证明与选择：`proof_stitched_route_portfolio.v1`
- 展示：由上述状态派生的 `retrosynthesis_route_workbench.v1`

## 全局 Director 的硬边界

- 默认每个 campaign 最多 1 次初始架构、2 次物质事件重规划、1 次最终组合总结；
- 相同上下文和配置只执行一次，其余读取不可变缓存；
- 只有关键拒绝、新精确证据、库存边界变化、共享瓶颈、新路线族或真实停滞可触发重规划；
- 单边物化、验证、抽取和库存任务不得隐式调用 Codex；
- 调用、token、上下文字节、墙钟时间和 attempts 全部计入同一 ledger；
- 预算耗尽只能产生具名 `budget_exhausted`，不能放宽证明要求。

CLI 和 golden replay 默认模型预算为 0。

## 调度原则

1. 身份、元素、原子跳跃、重复和祖先循环在昂贵工作前拒绝。
2. 已发现来源先抽取 exact rows；确定性缺口优先于新模型提议。
3. `attempt_count` 与 `accepted_expansion_count` 分开计费，一条规范边只接受一次。
4. 所有 proposal producer 共用规范入口，不拥有旁路路线图。
5. 物质事件恢复同一 campaign，不创建第二个 expansion loop。
6. 图按脏实体增量重算，并用 full-recompute oracle 校验。

专利来源遵循同一条不可逆降级链：官方完整 HTML → 未闭合边的 PDF 原生文本 →
低文本页本地 OCR → 显式准入的视觉 L0 候选。上一级已闭合的边不会进入下一级；
搜索摘要、视觉识别和 Codex 转述都不能直接授予 exact-source authority。

## 多信源、可替换路线与 UI

每条边分别记录 proposal origin、反应验证、精确来源、独立来源组和冲突。多个 Codex
child 都属于同一模型来源，不能伪装成多篇独立文献。Portfolio 默认只展示 2–5 条
边集或关键模块不同的路线，共享中间体只渲染一次。

Workbench 严格分开四个视图：断键假设、已展开路线、反应验证路线、库存闭合路线。
颜色由 L0–L4 proof 决定；分支数量从不表达完成度。

## 完成定义

默认要求至少 2 条不同完整路线、每条所选边 L3+、至少 2 个独立来源组、全部所选叶
达到采购库存边界，并满足路线多样性。只有 acceptance contract 能把 run 标为完成。

只有建议、只有物化边、只有 mapping、一部分库存、Agent 自称完成或预算耗尽都必须
显示为 `unresolved` 或 `budget_exhausted`。
