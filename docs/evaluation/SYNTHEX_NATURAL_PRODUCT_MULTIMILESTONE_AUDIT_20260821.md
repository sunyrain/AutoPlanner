# SynthEx 天然产物：单战略、路线内多战略与可选酶臂审计

日期：2026-08-21  
状态：三目标匹配实验已收束；多里程碑绑定缺陷已修复并完成确定性重放验证

## 1. 执行结论

本轮在同一组 3 个 SynthEx Figure 1 复杂天然产物上比较了三种配置：

1. 论文等价单战略：3 条独立根战略，每条只声明一个关键断键；
2. 路线内多战略：仍为 3 条独立臂，但允许一条路线携带 3 个连续战略里程碑；
3. 可选酶臂：2 条纯化学臂 + 1 条化学/生物混合臂，混合臂允许纯化学回退，**没有强制任何路线使用生物步骤**。

以 `paper-equivalent solved`（完整拓扑连通，所有叶节点精确命中同一冻结库存）计：

| 目标 | 论文等价单战略 | 路线内多战略（修复后） | 可选酶臂 | 观察 |
|---|---:|---:|---:|---|
| Traversiadiene | **通过**；7 步 | 未通过；候选深度 3/3/4 | 未通过；候选深度 2/11；保留真酶步 0 | 单战略已经足够，额外里程碑没有带来收益 |
| Dibohemamine A | 未通过；4 条候选均约 1 步 | **通过**；命中路线 8 步、4 个库存叶 | 未通过；候选深度 4；保留真酶步 0 | 多战略显著改变了可达区域，修复前曾被错误记为未解 |
| Cyclopiamine B | 未通过；候选深度 1/2/1/1 | 未通过；候选深度 13/3 | 未通过；候选深度 11/3/7；保留真酶步 1 | 生物步骤进入了连续路线，但没有使全部叶节点闭合 |

汇总：

- 论文等价单战略：1/3；
- 路线内多战略：1/3；
- 可选酶臂：0/3；
- 单战略与多战略的目标级 oracle 并集：2/3，但这代表两套预算的事后并集，不能报告为单一方法的公平成功率；
- 样本量仅为 3，结果只用于工程诊断和方法方向选择，不能据此声称总体统计优势。

因此，多步战略当前最准确的结论不是“提高了总体 solved rate”，而是：**它对单步战略具有目标级互补性，首次得到 Dibohemamine A 的 8 步库存闭合路线；合理下一步是条件触发多战略，而不是对所有目标无条件加倍计算。**

## 2. 冻结口径与论文对照

目标及协议冻结于：

- `benchmarks/synthex_figure1_head_to_head_3.v1.json`
- `benchmarks/synthex_figure1_head_to_head_3.protocol.json`

论文等价主口径：

- 3 条真正独立的根战略臂；
- 每臂最多 25 个连续 LLM 节点扩展；
- 每个不同开放叶使用 500 iterations / 1200 s / depth 6 的标准短尾；
- 叶节点使用同一冻结 ZINC + eMolecules 精确库存；
- 完整路线后执行 RouteJSON Critic–Editor，最多 6 轮；
- `paper-equivalent solved` 与反应验证、证据、条件完整度分开报告。

SynthEx arXiv v1 对 1,098 个目标报告的三种设置为：模板基线 151/1,098（13.8%）、战略规划 275/1,098（25.0%）、路线拼接 702/1,098（63.9%）。项目 README 存在版本数字漂移（20.8% / 67.2%），本审计固定使用 arXiv v1，避免跨版本混报。

本轮新增的“路线内多战略”不是论文原配置：原论文的一个 StrategyCard 对应一个关键断键，并没有显式要求一条路线预先声明多个战略里程碑。因此，它是我们的扩展实验，不应被包装为论文复现。

资料：

- 论文：https://arxiv.org/abs/2608.07454
- SynthAtlas：https://synthatlas.epfl.ch/
- 本地论文卡：`docs/evaluation/SYNTHEX_SYNTHATLAS_PAPER_CARD_20260812.md`
- 冻结来源包：`docs/evaluation/evidence/synthex_2608_07454/source_bundle.json`

## 3. 计算预算与候选深度

| 目标 | 单战略：模型调用 / 输入 / 输出 / 模型墙钟秒 | 多战略：模型调用 / 输入 / 输出 / 模型墙钟秒 | 可选酶臂：模型调用 / 输入 / 输出 / 模型墙钟秒 |
|---|---|---|---|
| Traversiadiene | 18 / 283,751 / 38,983 / 593.8 | 34 / 536,988 / 68,919 / 885.2 | 25 / 389,199 / 54,691 / 984.0 |
| Dibohemamine A | 41 / 680,777 / 79,973 / 1,544.3 | 34 / 544,987 / 69,834 / 869.4 | 35 / 580,589 / 70,968 / 1,109.8 |
| Cyclopiamine B | 32 / 510,050 / 67,358 / 1,069.7 | 38 / 631,305 / 81,330 / 1,430.0 | 50 / 835,796 / 115,012 / 868.0* |

\* Cyclopiamine B 可选酶臂为故障恢复结果：39 条原始 Strategy/Node 记录被确定性回放，新增 7 次 Critic 与 4 次 Editor。该结果可用于后执行评估与故障修复验证，不等价于新的独立盲测重复。

所有多战略和混合实验都调用了配置内的标准短尾。Cyclopiamine B 混合臂的 4 次短尾都返回 `paper_short_tail_no_complete_stock_closed_route`；短尾分别产生 3/1/5/1 个接受候选，但都未形成完整库存闭合路线。这里不存在“漏跑短尾”，而是复杂开放叶在 depth 6 预算内没有闭合。

## 4. 两个会扭曲结论的工程缺陷

### 4.1 路线内里程碑被误判为策略替换

旧的 canonical route family 只冻结根 StrategyCard。Director 产生的后续合法里程碑虽然在 RouteJSON 中连续，却在 canonical materialization 时被标记为 `strategy_replacement_conflict`。Cyclopiamine B 的 21 个步骤中有 5 个因此被隔离；Dibohemamine A 的库存闭合路线也被截断，造成假阴性。

修复位于 `cascade_planner/application/canonical_hypergraph.py`：

- route family 同时绑定根卡和声明的 `strategy_milestone_cards`；
- 只有声明集合内的里程碑可以进入同一路线族；
- 未声明的静默 StrategyCard 替换仍然拒绝；
- 新增回归测试，确保合法里程碑不再被当作替换。

真实 Cyclopiamine B plan 重放后，21/21 个初始 hypothesis 均成为 `frontier_candidate`，生成 21 条 materialization command 和 21 条 canonical edge，路线边数恢复为 11/7/3，且无 delta rejection。

Dibohemamine A 使用同一冻结 39,478,827-member full-InChIKey 库存完成确定性后修复重放：`B4=true`，命中 1 条 8 步路线、4 个叶节点，4/4 精确命中库存。严格反应证明仍单独为 false，不影响 paper-equivalent 指标。

### 4.2 Critic prompt 上限导致整条 Director 结果被晚期丢弃

Cyclopiamine B 的原始混合轨迹已完成 9 个 StrategyCard 和 30 个节点调用，但完整路线 Critic prompt 超过字节守卫，旧控制流把整条 Director 结果清空。这是后执行解释/组装缺陷，不是模型没有产生路径。

按照防御控制债务的处置框架，该事件属于影响等级 C：计算已付出、结果已生成，失败发生在后执行解释阶段。原有“全局终止”动作与风险不匹配。修复为：

- progressive compact Critic prompt，优先保留全部结构身份；
- 单分支 Critic 不可用只降级该分支，不再清空整个 Director；
- durable worker journal；
- 仅在显式 `critic_prompt_compaction_v1` 恢复模式中对 StrategyCard/Node 做逻辑别名回放，Critic/Editor 必须重新运行；
- 无法满足冻结 manifest 的恢复预检和一次无效代码快照恢复均排除出正式结果。

修复涉及 `cascade_planner/orchestration/sequential_strategy_director.py` 与 `cascade_planner/interfaces/target_solver.py`。

## 5. 路线价值与酶臂结论

### Traversiadiene

当前最佳是原论文式单战略路线：7 步、4 个精确库存叶，paper-equivalent solved。可视化中绿色叶表示精确库存命中，琥珀色反应表示反应证明仍开放。它证明了拓扑和库存闭合，不应被表述为已经获得全部实验条件或严格反应验证。

路线图：

- `docs/evaluation/figures/traversiadiene_stock_closed_route.png`
- `docs/evaluation/figures/traversiadiene_stock_closed_route.svg`
- `docs/evaluation/figures/traversiadiene_stock_closed_route.pdf`
- `docs/evaluation/figures/traversiadiene_stock_closed_route.tiff`
- `docs/evaluation/figures/traversiadiene_stock_closed_route.source.json`

### Dibohemamine A

这是本轮多战略扩展的主要正结果。单战略只达到浅层候选；路线内 3 个里程碑把搜索带到 8 步且全部 4 个叶节点命中库存。它说明对需要多个骨架构建阶段的复杂天然产物，一个根断键不足以持续塑造搜索。

### Cyclopiamine B

混合路线保留了 1 个真生物步骤（whole-cell P450 双环氧化）和 6 个化学步骤，但最终开放叶未全部闭合，Critic 也没有接受完整路线。它证明“可选生物动作能进入连续路线”，但尚未证明酶臂提高 solved rate。

匹配三目标结果：

- 酶臂 B4 增益：0/3；
- 最终保留真酶步：Traversiadiene 0、Dibohemamine A 0、Cyclopiamine B 1；
- 主要瓶颈是 `enzyme intent → 精确可执行生物 ReactionJSON → 被选择的连通路线 → 库存闭合`，不是是否把生物步骤设为必选；
- 先前 Traversiadiene 的 AaTS/AaT09930 环化酶匹配证明系统有提出酶战略的能力，但本轮匹配配置没有把它保留到最终库存闭合路线，暴露的是选择与奖励问题。

## 6. 当前不足与优先级

### P0：条件式策略组合，而非固定增加臂数

先运行论文等价单战略；只有在库存进展停滞、开放叶复杂度高、或骨架明显需要多个构建阶段时，才升级为路线内多战略。这样可以尝试获得当前单战略/多战略 2/3 的互补覆盖，而不对每个目标支付两套完整预算。

触发信号至少包括：连续若干节点无精确库存距离改善、开放叶环系/立体中心复杂度、根断键后的前体复杂度不降、以及短尾重复返回复杂未闭合叶。

### P0：搜索奖励直接观察“可闭合性”

MCTS/UCB 的价值不能主要来自路线深度、里程碑完成或语言评分。每次 materialize 后应立即计算：

- 精确库存叶比例；
- 未闭合叶的复杂度与模板可达性；
- 距离完整库存闭合还差多少个独立前沿；
- 反应身份是否合法、是否引入更难前体。

一旦存在精确库存闭合路线，立即进入 validation，不再继续搜索空前沿。

### P0：增强臂保留多个 ReactionJSON OR 候选

论文等价臂继续保持每节点 1 个候选以保证可比性；增强臂允许同一开放节点保留多个合法 ReactionJSON，并使用 progressive widening、访问计数与回溯，避免一个局部坏动作锁死整条战略。路线内里程碑只定义搜索方向，不应替代 OR 状态。

### P0：便宜的逐节点化学否决 + 路线末端 Critic–Editor

论文中的 Critic–Editor 仍应保留在完整 RouteJSON 之后；但在每个节点接受前增加低成本的结构守恒、价态、反应中心、官能团把手与明显机理禁区检查。后者不是完整路线 Critic 的替代，而是避免花费十多个节点后才发现早期断键化学不成立。

### P1：酶臂作为真正可选的第四/伴随臂

保持纯化学臂完全不受生物要求影响。混合臂只在局部选择性、晚期氧化/还原、动态动力学拆分、糖基化等具有明确优势时提出酶步骤，并允许化学回退。奖励应关注选择性收益、可检索酶家族/底物相容性和最终保留的真生物边，而不是仅出现生物措辞。

### P1：区分“短尾问题”和“战略问题”

depth 6 短尾适合已经足够简单的开放叶。若叶节点仍包含复杂稠环、多手性中心或缺少明显合成把手，应触发战略重规划，而不是对同一复杂叶重复增加短尾次数。

## 7. 验证与边界

代码验证：

```text
228 passed, 7 skipped, 9 warnings in 105.73s
```

覆盖：canonical hypergraph、global/sequential Director、target solver、blind panel、SynthEx protocol preflight 和 AiZynthFinder ReactionJSON expansion。

结论边界：

- `paper-equivalent solved` 只证明拓扑连通和冻结库存闭合；
- B2、反应证明、来源证据和条件完整度继续单独报告；
- Dibohemamine A 是旧盲跑动作的确定性后修复重放，不冒充新的独立盲测；
- Cyclopiamine B 混合结果是 journal 恢复后的后执行评估，不冒充完整新重复；
- 下一轮发表级结论必须在代码冻结后进行独立、预注册、无恢复路径的新鲜评测。

## 8. 收束判断

本轮已经回答三个问题：

1. 多步战略是否改善单步战略：**在一个目标上产生决定性改善，但 N=3 总 solved rate 持平；表现为互补，不是稳定统治。**
2. 酶步是否提升：**本组没有 B4 提升；只在一个未闭合候选中保留了真酶步。**
3. 为什么旧结果偏差：**canonical 里程碑绑定和晚期 Critic 全局终止制造了假阴性，均已修复并回归验证。**

正式下一步应是“论文等价单战略 + 条件触发多战略 + 可选酶伴随臂”的冻结实现，并在至少 20–50 个复杂天然产物上报告命中率、成本、路线深度、库存闭合、反应验证和酶步保留率。
