# Canonical V4 当前运行主线

更新：2026-08-10

状态：这是当前已经实现的运行主干，不是下一代 GRIA 已完成声明。实现边界与迁移状态以
[CURRENT_ARCHITECTURE_STATUS.md](architecture/CURRENT_ARCHITECTURE_STATUS.md) 为准；
目标 program 级架构见
[GENERAL_RETROSYNTHESIS_INNOVATION_ARCHITECTURE.md](architecture/GENERAL_RETROSYNTHESIS_INNOVATION_ARCHITECTURE.md)。

## 当前统一方向

Codex 是 campaign 级全局 Director，而不是单步模板；ChemEnzy 是同一 campaign 中的原生路线
producer，而不是 Codex 的从属实现。所有目标使用同一个 target-blind anytime 控制面。Codex 调用读取
有界的完整上下文：
目标、路线族、多步骨架、共享中间体、规范超图、来源/库存事实、冲突、失败、frontier、
当前 Pareto 组合和剩余预算。它可以同时替换几条路线、合并共享上游、淘汰支配路线，
或改变证据优先级；输出仍然只是 proposal。

主机侧把全局策略升级为事实：

```text
UnifiedCampaignSpec(target + immutable stock oracle + constraints + budgets)
                   |
                   v
              one RunKernel
                   |
        +----------+-----------+
        |          |           |
 ChemEnzy seed  Codex global  source prefetch
 native search  architecture   / deterministic facts
        |          |           |
        +----------+-----------+
                   | proposal only
                   v
 canonical admission + one hypergraph
                   v
 one deficit frontier → target-blind Action opportunities
                   v
 deterministic scheduler selects next highest-value action
   | materialize | validate | stock | evidence | conditions |
   | ChemEnzy expand | Codex replan | Program review         |
                   v
 recompute proof / stock / portfolio / full quality state + B0-B5 snapshot
                   v
 budget / no action / low gain / cancellation / unrecoverable error stops the loop
```

所有结果写回同一个 `RunKernel` 和 canonical hypergraph。CLI、API、Web、恢复和导出
只是适配器，不持有化学状态。

当前迁移状态：benchmark B4 专用提前结束和功能禁用已删除；旧 `objective_mode` 只保留带弃用收据的兼容展示，并计划于 2026-10-01 从新请求契约移除。Canonical Web 已不再发送该字段。saved-run 恢复会写入 `saved_run_objective_compatibility.v1`：旧 checkpoint/report 字段仅作为只读 provenance；字段存在、未知或缺失均使用当前统一 state/budget 产生相同 Action 决策。恢复后的 Action stage 从历史最大序号继续，不能被 stage 去重覆盖。
Director 的运行 task/context/plan ID 仍作为操作收据保留，但不进入 canonical scientific identity。Director proposal 的来源现在使用 `director_plan:<digest>`：摘要只覆盖计划的科学内容，剔除 plan/run/context/revision 等运行绑定字段；因此同一计划在 replay、resume 或 provider 并发完成顺序变化时保持同一科学来源身份，而真正的 transformation 内容变化仍会改变摘要。
新运行写入 `autoplanner_run_spec.v2`，嵌入不含名称、dataset、objective 或 acceptance 的
`unified_campaign_spec.v1`；v1 只在原摘要验证后兼容读取。每个快照同时输出八轴
`campaign_quality_state.v1`，acceptance 只是审计投影，不会让统一 Action loop 提前返回。完整契约见
[UNIFIED_CAMPAIGN_CONTRACT.md](architecture/UNIFIED_CAMPAIGN_CONTRACT.md)。
ChemEnzy route lineage、统一 Action opportunity、target-blind scheduler decision 和 anytime trajectory
已经进入主报告。生产 `target_solver` 只调用一次 `CampaignActionRuntime.run_anytime()`；原手写阶段只投影统一 trajectory/backlog，不再执行第二套 scheduler。CLI/API/Web 仍有参数装配与历史展示兼容层，但不拥有运行时序或化学写入权威。
新生产快照使用 `campaign_anytime_snapshot.v2`：同一内容寻址记录同时包含 RunKernel event/graph
revision、累计 campaign wall time、全部资源维度、唯一 Action execution 计数、路线计数、紧凑 Pareto
archive、B0–B5/Program milestones，以及控制面源码、配置、统一输入、stock oracle 和 provider/model
绑定。`campaign_trajectory.v2` 从这些快照导出 first-route、B1、first-host-valid-route、B2–B5 和
Program 首达时间、binding epochs 与资源曲线；resume 沿用累计 RunKernel 时间，不能生成新的零点。
旧 v1 snapshot 仍可读，但缺失时间/绑定保持未知。快照自身是权威 CAS 内容，不能被 stage 展示限长器裁剪。
评测入口不再向 solver 注入结果视图；`campaign_trajectory_cutoff_projection.v1` 只在求解完成后按冻结的
累计资源坐标读取最后一个合法 v2 snapshot，并将 cutoff 后的 B0–B5、路线和成本全部截断。resume
导致时间或任一受约束资源回退时，投影显式不可用，不能退回最终报告补数。
最终 Workbench 在当前 canonical 最优路线之外并列展示 `workbench_trajectory_history.v1`：每个 B0–B5、
首条路线、首条 host-valid 路线和 Program 里程碑明确区分“当前成立”“历史达到但当前失效”“从未达到”，
并保留首达 snapshot 与资源曲线。历史记录是只读审计信息，不能恢复已撤销的 proof。
Gateway export 同时从摘要验证通过的 target report 生成 `campaign_review_bundle.v1`，并拆分写出 Action trace、
失败 trace、provider/canonical route lineage 和累计资源曲线。四个组件各有独立 SHA-256；报告或 trajectory
摘要损坏时相应内容失败关闭。开放的 evidence/condition/Program 门不会被错误归入运行失败。
初始 ChemEnzy 与 Codex provider 计算仍从同一 frozen revision 并发启动；同一 cohort 的 canonical
admission 固定为 ChemEnzy 后 Codex，避免线程完成先后泄漏到 graph/frontier 和后续 Action trace。
安全的 evidence prefetch 也作为 canonical `evidence` Action signal 进入同一 start cohort，不再拥有独立
executor。后续 exact evidence 与 reaction validation 通过 `CampaignActionDeferredHandler` 对同一 frozen
revision 并行 prepare：connector acquisition 和 validation WorkerResult 都不在 worker 线程发布 canonical
revision；barrier 后由同一 Action runtime 按稳定 Action 顺序 commit。`campaign_action_concurrent_cohort.v1`
最多使用 4 个 runtime-owned workers，同 resource class 排他，并在 wrapper task 容量不足时整体回退单
Action 调度；prepare/commit failure 可重放且不取消 peer。该机制继续只使用一个 anytime loop、一个
RunKernel in-flight registry 和一个 canonical graph，不创建后台 scheduler、Blackboard 或 phase queue。
Web 运行中心每 2.5 秒读取同一状态链生成 `campaign_action_timeline.v1`：已结算项来自 target
checkpoint，正在执行项只来自 RunKernel in-flight Action wrapper；child tasks 不重复成行。ChemEnzy、
Codex、证据、验证、条件、库存及 Program/实验均在一个时间线上显示，时间线既不调度也不授予科学权威。

科学层 focused gate 已确认：B4 首次出现后，同一 trajectory 继续调度 replan、conditions 与 Program
review；缺 exact evidence 时 B3/B5 保持 false。官方 EPO 三案例和 enzyme positive/no-applicable
controls 的验证范围与限制见
[SCIENTIFIC_LAYER_REGRESSION_20260809.md](architecture/SCIENTIFIC_LAYER_REGRESSION_20260809.md)。

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

GRIA Phase-1 迁移另有一条默认关闭的 shadow path：canonical edge 可投影为
`ChemicalState` / `TransformationProgram` / `OperationNode`，并在显式启用时追加到可重放
Program store。它只验证新旧表示等价，不属于上述生产权威，也不能改变 proof、ranking、
route completion 或 acceptance。

## 全局 Director 的硬边界

- runtime 契约最多容纳 1 次初始架构、2 次物质事件重规划、1 次最终组合总结；当前
  target-only 生产编排进一步收紧为至多 1 次重规划；
- 相同上下文和配置只执行一次，其余读取不可变缓存；
- 只有关键拒绝、新精确证据、未闭合库存边界、新增且经宿主验证的 provider 边、共享瓶颈或
  新路线族可触发重规划；“路线尚未闭合/portfolio stagnation”本身只形成 deficit，不花第二次
  模型调用；
- 重规划前后的 molecule、edge 与 route-family ID 必须通过集合单调性审计；第二轮只能追加或
  强化 proof，不能替换第一轮已有路线；
- 每次实际执行的 replan 必须记录 gate/路线计数增量与模型调用、token、墙钟增量；`no_gain`
  保留为回归信号，不倒推删除已有低可信路线；
- 单边物化、验证、抽取和库存任务不得隐式调用 Codex；
- 调用、token、上下文字节、墙钟时间和 attempts 全部计入同一 ledger；evidence、stock、validation、
  Program 与 experiment 具有独立 task cap，native target/frontier 具有独立 reservation 规则；
- `campaign_task_budget.v1` 同时计算 settled 与 in-flight reserved 容量，resume/replay 不得重置
  Program/experiment 上限；
- `campaign_action.v2` 在 reservation 前声明类级预计资源；handler 子任务继承 Action execution ID，
  settlement 从事件链记录实际资源与 variance，并发 Action 不共享归因上下文；
- loop 终态以 `campaign_unexecuted_action_set.v1` 保留最终 revision 的所有未执行候选和具名原因；
  ChemEnzy timeout 不会取消同 cohort Codex，也不会吞掉后续 validation 状态；
  显式 RunKernel cancel 会在长期 loop 边界停止新 Action 派发，并把标准取消原因与操作者原因写入未执行 Action 集；
- 预算耗尽只能产生具名 `budget_exhausted`，不能放宽证明要求。

CLI 和 golden replay 默认模型预算为 0。

## 调度原则

1. 身份、元素、原子跳跃、重复和祖先循环在昂贵工作前拒绝。
   `campaign_action_preflight.v1` 只读取同一个 Action opportunity set，并只在 canonical materialization 当前可处理时激活：模型、证据、条件与 Program/实验动作等待 cheap gate；validation、stock 和 route closure 只等待同 route 候选。空图初始 ChemEnzy/Codex discovery、handler/resource blocker 与 round-robin 顺序均不被绕过。
2. 已发现来源先抽取 exact rows；确定性缺口优先于新模型提议。
3. `attempt_count` 与 `accepted_expansion_count` 分开计费，一条规范边只接受一次。
4. 所有 proposal producer 共用规范入口，不拥有旁路路线图。
5. 物质事件恢复同一 campaign，不创建第二个 expansion loop。
6. 图按脏实体增量重算，并用 full-recompute oracle 校验。
7. dataset ID、target index、target name 和旧 objective 标签不得进入 Action 排序。
   Director prompt、prompt 内 run identity 和 ChemEnzy request 只使用结构派生 opaque identity；结构名称解析延后到 evidence Action，检索名必须来自 exact-InChIKey 观察，初始 ChemEnzy/Codex 不等待身份网络。Director task/context ID 只用于运行审计；canonical proposal origin 使用剔除运行绑定字段的 Director plan provenance digest。
8. B4 是同一 trajectory 上的库存 milestone，不是 benchmark solver 的专用终态。
9. Codex 指导只能追加候选和优先级，不能删除 ChemEnzy native frontier。
10. 缺 evidence、conditions 或 Program validation 只开放对应 proof axis，不能擦除合法路线拓扑。

专利来源遵循同一条不可逆降级链：官方完整 HTML → 未闭合边的 PDF 原生文本 →
低文本页本地 OCR → 显式准入的视觉 L0 候选。上一级已闭合的边不会进入下一级；
搜索摘要、视觉识别和 Codex 转述都不能直接授予 exact-source authority。

专利 self-evolution 是 proposal memory，不是第二套搜索状态。只有同时绑定可重放 patent
exact row 和当前版本 accepted reaction proof 的边才会抽取局部 reaction-center SMARTS；
抽取结果必须先重放原例，才以 digest-bound 记录写入仓库外模板库。下一 campaign 启动时，
Codex 一次性看到这些候选并可在全局路线中组合；host 也可零模型地应用到 target/open leaves，
但结果始终从 L0 重新进入统一 admission、mapping 和 reaction validation。失败复用会按规范边
digest 去重回写，损坏的库则 fail closed，模板本身永不授予 L2/L3 或库存权威。
为防止 blind benchmark 退化成答案记忆，模板不会应用到其 exact 训练样例产物；零成功且
累计三条不同验证失败边的模板会自动隔离，只有新的已验证结果才能改变其统计。

## 多信源、可替换路线与 UI

每条边分别记录 proposal origin、反应验证、精确来源、独立来源组和冲突。多个 Codex
child 都属于同一模型来源，不能伪装成多篇独立文献。Portfolio 默认只展示 2–5 条
边集或关键模块不同的路线，共享中间体只渲染一次。

Workbench 严格分开四个视图：断键假设、已展开路线、反应验证路线、库存闭合路线。
颜色由 L0–L4 proof 决定；分支数量从不表达完成度。

## 完成与停止定义

默认要求至少 2 条不同完整路线、每条所选边 L3+、至少 2 个独立来源组、全部所选叶
达到采购库存边界，并满足路线多样性。Acceptance contract 可以在快照上标记配置要求已经满足，
但不会终止当前 Action loop；核心停止只读取预算、可执行动作、连续低收益、取消和不可恢复错误。
产品若希望首次闭合即停，由外部 milestone 订阅器显式取消。

只有建议、只有物化边、只有 mapping、一部分库存、Agent 自称完成或预算耗尽都必须
显示为 `unresolved` 或 `budget_exhausted`。
