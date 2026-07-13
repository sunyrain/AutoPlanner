# V4 主线架构

更新：2026-07-13

## 理想状态

Codex 是 campaign 级全局 director，不是被循环调用的单步模板。它一次接收一个有界的 `campaign_context.v1`：目标结构、全局反应超图、路线族、共享中间体、已有证据、库存观察、冲突、frontier 和剩余预算。输出是少量多步路线骨架、路线族优先级、风险判断和工作分配，而不是“这个分子下一步断哪里”的孤立答案。

确定性主机负责把全局策略变成可验证事实：

```text
target + acceptance + hard budget
              |
              v
       global Codex director  ---- proposal only
              |
              v
 canonical ingestion / early chemistry gates
              |
              v
 materialize -> reaction validate -> exact evidence -> stock audit
              |                         |
              +---- one deficit frontier+
              |
              v
 proof stitcher -> small diverse route portfolio -> acceptance/closeout
              |
              v
 bounded incremental workbench
```

所有箭头最终写入同一个 `RunKernel` 和 canonical hypergraph。CLI、API、Web、恢复和导出只是适配器，不拥有化学状态。

## 为什么不再使用 Blackboard 作为核心

Blackboard 适合租约、事件记录、恢复和人工协作，但不适合表达 AND/OR 反应超图。旧流程把 expansion、proof、stock 和 UI 状态复制到多个键和队列，导致同一条边在一个模块已关闭、另一个模块仍开放。更多模型调用只会增加 L0 文本，不会补上最弱的证据或库存缺口。

V4 只保留一种状态：

- 运行和预算：`RunKernel`；
- 化学拓扑：`canonical_retrosynthesis_hypergraph.v1`；
- 下一项工作：`deficit_frontier.v1`；
- 证明和选择：`proof_stitched_route_portfolio.v1`；
- 展示：从上述状态派生的 `retrosynthesis_route_workbench.v1`。

旧 Blackboard 仍可读取历史运行，但不能写入 V4 科学状态。

## 调度原则

1. 元素守恒、大原子跃迁、身份反应和祖先循环在搜索前端拒绝。
2. 已发现来源先抽取 exact rows；确定性缺口始终排在新的模型提议前。
3. `attempt_count` 和 `accepted_expansion_count` 分开计费；一个 child 只产生一次接受记账。
4. ChemEnzy、模板、文献和 Codex 都进入相同 ingestion，不拥有旁路图。
5. 新证据到达后恢复同一个 campaign，而不是启动第二个 expansion loop。
6. 仅当 frontier 是 proposal/diversity 缺口且确定性工作无法推进时，global director 才可再次调用。
7. 上下文使用 delta 和固定大小的 portfolio，不随全历史无限增长。

## 多信源与可替换路线

每条反应边分别记录 proposal origin、确定性验证、精确来源和冲突。多个 Codex child 属于同一个相关模型信源，不能伪装成独立文献。路线 portfolio 默认选择 2–5 条边集合不同或关键模块不同的候选；共享中间体只渲染一次，可替换模块显式标注，不能通过复制节点制造“路线更多”的错觉。

## 完成定义

默认接受要求：至少两条完整路线、每条反应边 L3+、至少两个独立来源组、所有被选叶节点达到配置的库存边界，并且路线边集合满足多样性。只有 acceptance contract 能把 run 标为完成。

以下情况必须显示为 unresolved：只有断键建议、只有已物化边、只有 mapping 一致、只有部分叶节点库存、预算耗尽、Agent 说完成、或 UI 分支很多。

## 性能边界

- CLI 默认 0 次模型调用；
- optional director 有全运行调用/token/墙钟/上下文硬上限；
- dirty-subgraph 增量重算，并用 full-recompute oracle 校验；
- workbench 默认最多 5 条路线，camera 只变换一个 world layer；
- 拖拽由 pointer capture + requestAnimationFrame 合并更新；
- 大图做 viewport culling 和 semantic zoom；
- 所有缓存均可删除并由权威事件/CAS 重建。

## 当前迁移边界

P0–P8 已建立 kernel、global director、worker、canonical graph、proof portfolio、单一 service 和 V4 workbench。P9 已把新 CLI/API/Web 接到统一 gateway，并清理主线文档和生成报告。旧 V3 campaign、一次性研究工具和历史 saved-run reader 作为冻结兼容代码保留到 P10 golden replay 完成；任何新功能不得进入这些模块。
