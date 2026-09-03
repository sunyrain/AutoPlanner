# SynthEx 与 AutoPlanner 当前架构深审：相同主干、不同有效搜索语义

> Source coverage: SynthEx arXiv v1 full paper (41 PDF pages), official SynthEx repository, local AutoPlanner source, and frozen bufotalin v13 run artifacts  
> Extraction confidence: High for published methods and local behavior; medium-to-low for unpublished SynthEx implementation details  
> Locator mode: page-grounded  
> Primary analytical lens: Methods  
> Secondary analytical lens: Resource  
> Context verification: Official arXiv PDF, visually inspected Figures 1/4/5, official repositories, and frozen run report  
> Card completeness: Complete for the questions addressed here  
> Comparison date: 2026-08-20

术语账本：`战略臂`指由一个 StrategyCard/strategy sentence 驱动的一棵独立搜索树；`ReactionJSON` 指在映射产物上确定性执行的有序图编辑；`OR 候选`指同一 MCTS 状态下可由 UCB 选择和回溯的替代反应动作；`paper-equivalent solved` 指至少一条目标根连通路线的全部叶节点命中同一冻结 ZINC + eMolecules 库存，不要求条件、证据或实验验证；`多里程碑`指一条路线内两个以上相互依赖但不必相邻的战略构建目标。

## 01 基本信息

- [Paper] Daniel Armstrong 等，*Strategy-first synthesis planning for complex natural products*，arXiv:2608.07454v1，2026-08-07。[Paper: PDF p. 1]
- [Paper] 论文默认每个目标生成三个独立战略，每个战略作为 steering query 进入后续搜索；系统是 AiZynthFinder MCTS 的子类，神经模板扩展 policy 被 LLM-guided policy 替代。[Paper: PDF pp. 22–23]
- [Paper] 主干模型是 `gemini-3.1-pro-preview`；每臂 Phase-1 上限 25 次扩展，三臂合计硬预算 75 次 LLM policy 调用。[Paper: PDF pp. 23, 36]
- [External] 官方 SynthEx 仓库截至本次核对仍明确写着 “The code is not here yet”，只发布 README 和图片，论文声称的精确 `synthelite.gemini3_1.code_ops.yml` 尚不可独立检查。
- [Analysis] 当前 AutoPlanner 冻结运行是 `bufotalin-chemoenzymatic-fusion-formal-v13`：`gpt-5.6-terra` medium、3 个配置战略臂、每臂 25 次上限、`strategy_tree_engine=aizynthfinder_mcts`、每节点最多 1 个 ReactionJSON 候选。

## 02 一句话总结

[Analysis] AutoPlanner 已经实现了论文所称的主干——每个战略进入独立 AiZynthFinder MCTS，Codex 在树内逐节点提出并由宿主 replay ReactionJSON；当前低效不是“没有 MCTS”，而是正式配置和控制流把这棵树压成了近似单链：每状态只给一个动作、只处理第一个开放叶、Critic 在未闭合路线阶段即可删掉分支，且蟾毒灵实验把全部战略强制成化学–生物融合。

## 03 研究问题

1. [Analysis] AutoPlanner 当前的 AiZ MCTS 是否真正拥有可供 UCB 比较的同节点 OR 候选？
2. [Analysis] 为什么设置了每臂 25 次，实际只发生 3 次和 7 次 LLM policy 调用？
3. [Paper] SynthEx 是否显式解决了复杂天然产物“一条路线需要多个战略里程碑”的问题？
4. [Analysis] 如何在保留论文等价主臂的同时，把多里程碑与酶法变成 AutoPlanner 的真实增益，而不是额外约束？

## 04 研究背景与发展路径

1. [Paper] SynthEx 认为复杂天然产物的主要瓶颈不是搜索次数，而是 USPTO/专利模板反应空间缺少稀有成环、重排、级联与收敛化学。[Paper: PDF pp. 2–6]
2. [Paper] 它用三个竞争战略避免对一个根断键过早承诺，再让 LLM policy 用 ReactionJSON 扩展反应空间。[Paper: PDF pp. 5–6, 22–23]
3. [Paper] 未购买叶由短程 AiZynthFinder 完成，限制为 6 transforms、500 iterations、1200 s；完整路线全部叶命中库存才计 solved。[Paper: PDF pp. 24–25]
4. [Paper] Route Builder 形成一份完整、可序列化的 RouteJSON 后，进入 Critic–Editor 局部修复；这里的“完整”指路线文档已经形成，不等同于论文 solved 判据所要求的“全部叶已命中库存”。[Paper: PDF pp. 6, 18–19, 24]
5. [Analysis] AutoPlanner 后续已经补上独立 AiZ MCTS、宿主 ReactionJSON replay、full-InChIKey 库存与 paper-equivalent 指标；旧审计中“上游仍是自定义 scheduler 而非 AiZ MCTS”的描述不再适用于当前代码。

## 05 论文识别的核心痛点

| 痛点 | SynthEx 的处理 | 当前 AutoPlanner 的状态 |
|---|---|---|
| 模板空间不含战略反应 | LLM 直接写 ReactionJSON | 已实现；Codex 动作经宿主 replay 后注入 AiZ |
| 单一根战略失败 | 默认三个独立战略 | 配置有三臂；v13 最终只保留两个路线家族 |
| 自由 SMILES 不可靠 | 映射产物上的十类图编辑 | 已实现，并有结构审计 |
| 搜索需替代动作与回溯 | AiZ MCTS/UCB 管理搜索状态 | 引擎存在；v13 的每节点候选上限为 1，实际树近似单链 |
| 路线文档需局部修复 | Route Builder 形成完整 RouteJSON 后进入 Critic–Editor；与短尾的严格先后未披露 | 当前对目标根战略 RouteJSON 运行本身不一定错误，但 Critic 可硬阻断输出 |
| 完整目标模板搜索低效 | 仅对降复杂叶做短尾 | 已有短尾，但本次两条高级甾体叶均未闭合 |

## 06 核心思想

[Paper] SynthEx 的关键组合是：战略假设产生反应空间方向，AiZ MCTS 管理逐步搜索，ReactionJSON 提供可执行动作，库存提供终止判据，短尾完成常规化学，Critic–Editor 在路线形成后改善内部一致性。[Paper: PDF pp. 5–6, 18, 22–24]

[Analysis] AutoPlanner 已复现这些名义模块，但“有 MCTS”不等于“有有效树”。AiZ 的一个节点只执行一次 `expand()`；如果 expansion policy 只返回一个动作，该状态就只有一个孩子，之后无法靠 UCB 产生未返回的替代反应。当前测试确实证明 `max_candidates_per_call=3` 时能保留两个根动作并回溯，但 v13 正式配置使用的是 1。

## 07 方法概览

### SynthEx

- [Paper] Strategy Generator：固定检查 scaffold、关键成键、官能团/保护冲突和立体中心，默认输出三个一语句战略；每个战略要求可由一至两步关键反应实现。[Paper: PDF p. 22]
- [Paper] Route Builder：每个战略作为 steering query，LLM policy 逐节点写 ReactionJSON；每臂 step limit 25。[Paper: PDF pp. 22–23]
- [Paper] Search：AiZynthFinder MCTS 子类；LLM-guided policy 替代神经模板扩展 policy。[Paper: PDF p. 22]
- [Paper] Tail：每个未购买叶单独运行 6/500/1200 的 AiZynthFinder 短搜索并 stitch。[Paper: PDF p. 24]
- [Paper] Critic–Editor：在 Route Builder 已完成并序列化的 RouteJSON 上重排、插入、删除或替换局部步骤，保留核心战略；论文没有明确交代该循环相对 AiZ 叶级短尾 stitching 的严格先后。[Paper: PDF pp. 6, 18–19, 24]

### AutoPlanner 当前实现

- [Analysis] 每个 StrategyCard 启动独立 `AiZynthFinder.MctsSearchTree`；AiZ 拥有 UCB、cycle pruning、stock termination 与 back-propagation。
- [Analysis] Codex 接收节点局部上下文，返回 ReactionJSON；宿主 replay 出映射前体并构造 `SmilesBasedRetroReaction`，再交回 AiZ。
- [Analysis] v13 的两个有效策略树分别为：3 次 policy call / 3 个树节点 / 2 个 accepted actions / selected depth 2；以及 7 次 policy call / 7 个树节点 / 7 个 accepted actions / selected depth 5。两棵树都运行到 125 MCTS iterations，但均未 solved。
- [Analysis] 当前 handler 总是从 `expandable_smiles` 取第一个分子，且节点提示在内部比较三个局部断键后只返回配置上限允许的候选；v13 上限为 1。
- [Analysis] MCTS 后无条件进入 Critic–Editor；最终 `usable` 列表排除 Critic 状态为 `reject` 的分支，即 Critic 事实上会影响 paper-equivalent 主链保留。

## 08 核心模块拆解

| 模块 | 论文公开方法 | AutoPlanner 当前事实 | 判断 |
|---|---|---|---|
| 独立战略树 | 默认 3 个战略分别搜索 | 已有独立 AiZ MCTS | 主干已对齐 |
| LLM 节点 policy | ReactionJSON，3×25 hard budget | Codex ReactionJSON，配置同为 3×25 | 接口已对齐 |
| 同节点动作数 | 论文未披露 | v13 固定为 1 | 无法宣称复现；本地 OR 宽度被关闭 |
| 开放叶选择 | 论文未披露 | 总是选择状态中的第一个开放叶 | AiZ 原生“对状态内多个可扩分子返回动作”的空间被缩小 |
| UCB/回溯 | 论文声明由 AiZ MCTS 承担 | 代码和单测均存在 | 只有出现兄弟动作时才产生实际价值 |
| 战略约束 | 一句话 steering query；保留 key strategy | 精确 StrategyCard、映射键签名、immutable 约束 | AutoPlanner 更硬，可能提高一致性，也可能抑制可恢复变体 |
| 生物路线 | 未提供专门酶臂 | v13 的 fusion profile 强制每臂同时含化学和生物步骤 | 不是论文对齐项，且本次明显压缩战略多样性 |
| Critic 时序 | Route Builder 形成完整 RouteJSON 后、Analyst/展示前运行；与短尾先后未明确 | 在战略 RouteJSON 后运行，但对未达到库存闭合的分支拥有硬淘汰权 | 运行位置可能接近论文；真正不一致是硬门控及提前固定预留预算 |
| 短尾 | 每个未购买叶一次完整解搜索 | 已实现 6/500/1200 | 参数接近；输入叶质量不足 |
| solved | 任一路线全部叶命中库存 | 已独立实现 B4/paper-equivalent | 指标已对齐 |

## 09 必要公式与符号

- [Paper] `paper_solved = exists(route): target_rooted(route) and all(leaf in frozen_stock)`。[Paper: PDF p. 24]
- [Paper] `blocking_rate = blocking_steps / route_steps`，仅表示同类 LLM Critic 的内部收敛，不是实验失败率。[Paper: PDF pp. 18, 24]
- [Analysis] 对当前树，实际有效分支因子可写成 `b_eff = accepted_sibling_actions / expanded_states`。v13 每状态最多一个候选，因此 `b_eff <= 1`；在这种情况下 UCB 只能沿唯一孩子前进，不能比较替代反应。
- [Hypothesis] 多里程碑可用潜势差进行软引导：`r' = r_stock + lambda * (Phi(s_next) - Phi(s))`。`r_stock` 仍是唯一 solved 权威，`Phi` 只反映已完成里程碑与复杂度下降，避免把战略符合度误当库存闭合。

## 10 实验设计与证据链

### 论文结果

| 证据 | 结果 | 结论边界 |
|---|---:|---|
| AiZ intact baseline | 151/1098 = 13.8% | 模板空间基线，不是等算力比较 |
| SynthEx strategic only | 275/1098 = 25.0% | LLM 战略层可直接闭合部分目标 |
| SynthEx stitched | 702/1098 = 63.9% | 战略降复杂与短尾高度互补 |
| LLM policy calls | 每目标均值/中位数均为 75 | 论文运行能持续使用三臂各 25 次调用 |
| 共同 solved 的 134 个目标 | SynthEx 中位 5 步，AiZ 中位 11 步 | 只适用于共同解决集 |

[Paper: PDF pp. 24, 36–37]

### 当前 v13 证据

| 观测 | 数值 | 含义 |
|---|---:|---|
| 配置战略臂 | 3 | 名义与论文一致 |
| 最终路线家族 | 2 | 一臂未形成可保留家族 |
| 上游 LLM policy calls | 3 + 7 = 10 | 远低于配置的 75 次上限 |
| MCTS iterations | 125 + 125 | 迭代次数很多，但没有转化为新 policy 节点 |
| 每节点候选上限 | 1 | 没有同节点 ReactionJSON 兄弟动作 |
| 全部模型调用 | 29 | 含 6 次 StrategyCard、16 次 Proposal、7 次 Critic |
| 模型 token | 475,538 input / 68,524 output | 预算主要花在少量保留路线和 Critic–Editor 上 |
| 目标根路线 / stock closed | 2 / 0 | paper reach=true，paper-equivalent solved=false |

[Analysis] 这说明当前失败不是 AiZ 未启动，也不是 25 次上限太小，而是搜索树在 3/7 次后没有新的可扩 OR 状态；循环仍跑到 125 iterations。把“MCTS iterations”当成“LLM 连续扩展次数”会严重高估实际搜索。

### 完整图表证据覆盖

- [Paper] Figure 1 给出 Strategy Generator → Route Builder/ReactionJSON → Critic–Editor → Analyst/SynthAtlas 的总体架构，并展示一个战略可包含两个紧密衔接的关键事件。[Paper: PDF p. 5]
- [Paper] Figure 2 用 Okaramine M、Melonine 和 Chanoclavine/Lysergol 个案展示战略推理；它们是案例证据，不是 Strategy Generator 可靠率统计。[Paper: PDF pp. 7–11]
- [Paper] Figure 3 与 Table 1 比较 SynthEx 与 USPTO 的反应空间、成环/收敛特征和反应识别率；偏离专利分布不等同于实验正确。[Paper: PDF pp. 12–14]
- [Paper] Figure 4 汇总子集 solve rate、分子量趋势、专家关键步骤评分和与文献路线长度比较；战略价值是文献步骤仍占优势的评分轴。[Paper: PDF pp. 15–17]
- [Paper] Figure 5 展示 Critic–Editor 六轮内的 blocking-rate 下降与 Monascuspirolide A 局部编辑案例；这是内部一致性，不是湿实验验证。[Paper: PDF pp. 18–19]
- [Paper] Figure 6 报告 AiZ intact、LLM-only 与 stitched pipeline 的 policy-call 量；不同 policy call 不是相同计算单位。[Paper: PDF pp. 36–37]
- [Paper] Figure 7 在双方共同解决的 134 个目标上比较路线长度，SynthEx 中位 5 步、AiZ 中位 11 步。[Paper: PDF p. 37]
- [Paper] Figure 8、Figure 9、Figure 10 和 Figure 11 分别从研究组、项目重叠、评分分布和 rater 敏感性角度检查专家评分的稳健性；这些分析不能消除仅十名评审和较低绝对一致性的限制。[Paper: PDF pp. 37–40]
- [Paper] Table 3 做逐一移除评审者的效应量分析，战略价值方向在每次移除后仍偏向文献路线。[Paper: PDF p. 41]

## 11 对结论的正确解释

1. [Analysis] 用户的纠正成立：AutoPlanner 已经有每战略独立 AiZ MCTS 和逐节点 LLM ReactionJSON，旧的“我们只有 AiZ 短尾”判断是过时的。
2. [Analysis] 当前仍不能说与论文实现等价，因为论文的 75 次 policy 调用全部发生，而 v13 只发生 10 次；当前每状态一动作使 UCB 的探索项几乎没有对象。
3. [Paper] SynthEx 并未显式建模一条天然产物路线中的多个远距离战略里程碑。默认战略仍围绕一个可由一至两步完成的关键构建。[Paper: PDF p. 22]
4. [Paper] 论文偶尔承认一个路线关键设计可由两个紧密耦合步骤组成，例如收敛汇合后接级联成环；这仍不是多里程碑 DAG 或分阶段战略控制。[Paper: PDF p. 35]
5. [Paper] 作者明确承认专家比较只说明“找到可行战略以后”的化学质量，不能说明 Strategy Generator 多可靠；战略价值也是文献路线仍占优势的评分轴。[Paper: PDF pp. 17, 21, 41]
6. [Analysis] 因而多里程碑不是为了“补回论文已有能力”，而是 AutoPlanner 可以超越论文、尤其适配蟾毒灵这类复杂二级代谢物的核心创新方向。

## 12 作者明确承认的局限

| 局限 | 作者边界 | 来源 |
|---|---|---|
| 无湿实验验证 | 报告 reach 与纸面质量，不是实验可行性 | [Paper: PDF p. 21] |
| 立体结果未验证 | 专家仍发现选择性与可行性错误 | [Paper: PDF p. 21] |
| 战略可靠性未被专家实验完整覆盖 | 比较条件化于共享战略框架 | [Paper: PDF pp. 16, 21] |
| Critic 不是外部 oracle | 修复和评分共享同类 backbone | [Paper: PDF pp. 18, 21, 24] |
| 成本不公平 | 论文比较 reach，不比较等 compute | [Paper: PDF pp. 24–25, 36] |
| 条件与实验闭环缺失 | 被列为后续 frontier | [Paper: PDF p. 21] |
| 精确实现当前不可复核 | 官方仓库尚未发布代码/config | [External: official SynthEx repository, checked 2026-08-20] |

## 13 批判性分析与当前问题定位

### 13.1 核心不是“有没有 MCTS”，而是 MCTS 是否有选择

[Analysis] AiZ 节点的 `expand()` 只执行一次并把 expansion policy 返回的全部 actions 固化为 children。当前 v13 每次最多返回一个 candidate，所以一个已展开状态不会在未来访问时再向 Codex 索取第二个候选。现有回溯测试使用 `max_candidates_per_call=3` 才证明了高 prior 死路与低 prior 可解路之间的回溯；该能力没有被正式配置启用。

### 13.2 状态含多个开放叶时，当前只扩第一个

[Analysis] AiZ 把一个状态的全部 `expandable_mols` 交给 policy，本来可以返回针对不同叶的动作；当前 host handler 取第一个合法索引，并只让 Codex处理该分子。这样 UCB 既不能选择同一叶的替代反应，也不能选择先扩哪一个开放叶。对于收敛路线或一次断键产生多个高级片段的情况，这会显著缩窄搜索。

### 13.3 Critic–Editor 的主要问题是淘汰权，不是简单的“位于短尾之前”

[Analysis] 当前 Director 在上游树结束并得到目标根路线文档后运行 `_run_codex_critics`，这一位置可以与论文“Route Builder 后、Analyst 前”的第三阶段相容；论文并未证明 Critic 必须晚于短尾。问题在于随后 `usable` 会过滤掉 Critic=`reject` 的分支。v13 在没有 stock-closed 路线时执行了 7 次 Critic、5 次 Editor，并为这些调用预留上游预算。论文六轮后仍有约 0.06 的 blocking rate，说明“仍有 blocking step”并不等同于路线必须从资源或 reach 统计中消失。[Paper: PDF pp. 18–19, 24] 因此当前应保留 Critic 的中间修复作用，但取消其对 paper-equivalent 候选的删除权，并且只在确有连通、可序列化 RouteJSON 时才分配调用。

### 13.4 v13 的组合臂不是论文对齐臂

[Analysis] `chemoenzymatic_fusion` profile 只有一个 mandate，并通过取模赋给所有分支；每个 StrategyCard 都被要求同时包含化学骨架形成与生物局部修饰。结果六次 StrategyCard 调用都围绕 hybrid/P450，最终两条路线共享高级 P450 氧化边界。论文没有专门酶臂，更没有强制所有战略含生物步骤。该配置同时牺牲了纯化学对照、根战略多样性和可解释增益。

### 13.5 单锚点对蟾毒灵类目标不够

[Hypothesis] 蟾毒灵需要同时处理甾体骨架来源/成环、多个立体中心与氧化态、C17 不饱和内酯安装以及可能的后期选择性官能团化。只固定一个 1–2 步锚点，后续节点容易局部合理但全局停留在“另一个高级甾体前体”。这正是 v13 两个叶仍高度复杂、短尾无能为力的化学原因。论文默认单锚点设计没有解决这一点。

## 14 学到的知识

- [Analysis] 当前 AutoPlanner 的主要缺口已从“搜索引擎不存在”收敛为“动作集合和阶段控制使真实搜索宽度不足”。
- [Analysis] ReactionJSON replay 保证动作可执行，但不能自动产生动作多样性；UCB 只有在一个状态存在多个可行孩子时才有意义。
- [Analysis] 25 是调用上限，不是自动保证；树过早无可扩节点时，MCTS 可以跑很多 iteration 而不再调用模型。
- [Analysis] 更强的 Codex 无法补偿 `candidate_limit=1`、first-leaf-only 和 Critic 提前淘汰这三个系统约束。
- [Paper] SynthEx 的高总体 solved rate主要来自“战略层把叶降复杂 + 短尾闭合”，并不证明它已经可靠发现天然产物的完整多阶段战略。[Paper: PDF pp. 21, 24, 36]
- [Analysis] 酶法应是相对于同一化学骨架的可测增益，不应成为所有战略的入场条件。

## 15 与 AutoPlanner 现有资产的连接

[Analysis] 应保留并复用的现有资产包括：AiZ sidecar 隔离、真实 MCTS/UCB、宿主 ReactionJSON replay、exact full-InChIKey stock、独立 B2/B4、canonical hypergraph、候选生命周期、短尾 6/500/1200、条件/证据层与酶步骤 contract。

[Analysis] 需要重新排列热路径，而不是推倒重写：

`Strategy portfolio → independent AiZ MCTS → multi-action ReactionJSON expansion → host structural replay → coherent strategic RouteJSON → non-destructive Critic/Editor → stock check and one short tail per resulting open leaf → stitched RouteJSON → optional final integrated Critic → paper-equivalent selection → B2/conditions/evidence → optional enzyme proof`

[Analysis] 其中宿主结构 replay 必须留在树动作入口。第一次 Critic 可处在中间过程，用来修复已形成的战略 RouteJSON；第二次 Critic 是 AutoPlanner 可增加的 stitched-route 全局一致性检查。两者都可以标注风险或触发局部编辑，但不应在 paper-equivalent reach/B4 计算前拥有删除原始候选的权力。

## 16 研究设想

统一字段：[Analysis] 创新状态为“部分基础已实现、核心增量未验证”；如何验证应采用同靶标、同库存、同模型和同调用预算的消融；可能失败包括候选同质化、reward shaping 误导、输出 token 膨胀和酶步骤缺乏精确底物范围。

### 设想 A：先修复真实 OR 搜索，不改论文主干

- [Hypothesis] 将 paper-matched 臂设为纯化学 3×25，`max_reactionjson_candidates_per_node` 从 1 调为 3，并要求候选在断键拓扑或反应家族上真正不同。
- [Hypothesis] 对每个状态保留全部合法 sibling actions；失败后由 AiZ UCB 回溯。若一次返回 3 个仍不够，再实现 progressive widening：节点访问达到阈值时追加一次带 blacklist 的 Codex 候选，而不是重建整棵树。
- [Hypothesis] 扩展 provider 应能对多个 `expandable_mols` 返回动作，或明确把“选哪个开放叶”建成可搜索决策，取消 first-leaf-only。
- [Analysis] 终止条件应区分 `policy_calls_exhausted`、`no_selectable_frontier` 与 `mcts_iteration_cap`；不能再用 125 次 iteration 掩盖只有 3/7 次有效调用。
- [Analysis] 验证：同一 5–8 target canary、同库存、同模型、同总调用预算，对比 K=1 与 K=3；报告有效分支因子、policy calls、树节点数、最大深度、首次 B4 时间和 solved 数。

### 设想 B：天然产物的多里程碑 RouteArchitectureCard

- [Hypothesis] 对高环数/高立体复杂度目标，让外层规划器输出 2–4 个带依赖关系的里程碑，而不是只有一个不可变根断键。例如：核心骨架构建、关键稠环/桥环闭合、侧链或杂环收敛安装、晚期选择性氧化/还原。
- [Hypothesis] 里程碑描述“必须达成的结构变化和顺序约束”，不预先锁定完整前体 SMILES；内层仍是每策略独立 AiZ MCTS，ReactionJSON 仍逐节点生成。
- [Hypothesis] 用 `milestone_progress` 和复杂度下降做软 reward shaping，库存闭合作为唯一 solved 权威。连续若干节点无进展时允许回溯或触发外层战略替换。
- [Analysis] 这不是 SynthEx 的已实现模块，而是论文自己暴露的战略可靠性边界，也是 AutoPlanner 最有发表价值的算法增量之一。

### 设想 C：六个入口、渐进分配，而不是六条都跑满

- [Hypothesis] 对复杂天然产物先廉价生成 6 个架构候选：4 个纯化学、2 个可选化学–生物融合。每个先给 3–5 个节点做预检，再将剩余总预算分配给前 3 个有真实复杂度下降且保留 OR 宽度的候选。
- [Hypothesis] 这样可以在不把总调用从 75 放大到 150 的情况下增加根战略覆盖；被提升的分支再连续扩展到 25 次上限。
- [Analysis] 论文等价报告仍单独保留固定 3×25 arm，渐进六入口作为 AutoPlanner-enhanced arm，避免把算法增益和预算增益混在一起。

### 设想 D：酶法作为 companion/overlay，而不是全臂强制

- [Hypothesis] 先让纯化学架构形成可比较路线，再在明确 substrate→product 边界上允许酶步骤替代局部化学片段；保留同一路线的化学 fallback span，计算净步骤节省。
- [Hypothesis] hybrid 分支只有在出现可信的酶类/底物范围、辅因子与选择性假设时才晋级；否则降级为纯化学或明确失败，不允许靠“P450”标签满足战略合同。
- [Analysis] 分别报告 B4、路线长度、酶替代的化学步骤数、精确底物–产物证据和实验验证状态；这样酶法才是独立增益，而不是造成全部战略同质化的约束。

### 推荐执行顺序

1. [Analysis] P0：paper-matched arm 取消强制生物；保留 Route Builder 后的中间 Critic–Editor，但只对连通可序列化 RouteJSON 运行，不得删除 paper-equivalent/B4 候选；短尾 stitch 后可选再运行一次全路线 Critic。
2. [Analysis] P0：候选上限改为 3，补多开放叶动作与真实 sibling/backtrack 集成测试。
3. [Analysis] P0：跑 1 个单靶 canary，要求每臂产生多个 sibling，且 policy calls 不再停在 3/7；再跑 5–8 targets。
4. [Analysis] P1：加入多里程碑 RouteArchitectureCard 与等调用预算消融。
5. [Analysis] P2：加入化学/酶 companion arm，单独证明酶法增益。

最终判断：[Analysis] AutoPlanner 不需要回退或替换 AiZynthFinder；当前 AiZ 主干是正确资产。需要修的是“如何给树动作、何时停止、何时允许 Critic 淘汰、怎样表达天然产物的全局战略”，而不是再换一个规划器名称。
