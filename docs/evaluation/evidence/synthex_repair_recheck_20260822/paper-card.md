# SynthEx 原论文复核与 AutoPlanner 修复依据

> Source coverage: Full paper
> Extraction confidence: High for the cited method and result sections; mixed for unpublished implementation details
> Locator mode: page-grounded
> Primary analytical lens: Methods
> Secondary analytical lens: None
> Context verification: Targeted external check
> Card completeness: Complete relative to supplied source

输入边界：arXiv:2608.07454v1 的 41 页完整 PDF；重新生成并验证了 PDF 页码证据包。外部核对仅限官方 SynthEx 仓库截至 2026-08-22 的公开状态。

术语账本：`Strategy`/`steering query` 指每支搜索的自然语言战略先验；`ReactionJSON` 指原子映射上的有序图编辑；`RouteJSON` 指可被 Critic–Editor 就地修改的完整路线文档；`blocking reaction` 指 Critic 判定为按当前结构与条件不可执行的步骤；`strategy_anchor_progress` 是 AutoPlanner 当前额外引入的键对进度表示，论文没有报告这一对象。

## 01 基本信息

- 题目：*Strategy-first synthesis planning for complex natural products*。
- 作者：Daniel Armstrong、Xuan-Vu Nguyen 等；通讯作者 Philippe Schwaller。[Paper: PDF p. 1]
- 载体：arXiv:2608.07454v1，2026-08-07；方法/系统论文。[Paper: PDF p. 1]
- 核心系统：SynthEx；公开路线资源：SynthAtlas。
- 官方实现状态：`[External]` 截至 2026-08-22，官方仓库仍写明 “The code is not here yet”，并把 planner、ReactionJSON/RouteJSON 规范与复现配置列为待发布；因此精确提示词、隐藏重试和节点过滤规则不可核验。
- 与本项目关系：用于界定 `paper_matched_reach` 应复制的公开算法语义，以及哪些机制增强必须另立实验模式。

## 02 一句话总结

SynthEx 用三条自然语言战略分别引导 LLM–MCTS 逐节点生成 ReactionJSON，再把完整 RouteJSON 交给最多六轮 Critic–Editor 内部修复，在库存闭合指标上扩大复杂天然产物的可达范围，但不把图重放、LLM 自评或库存闭合解释为实验可行性。[Paper: PDF pp. 6, 15, 18, 21–24]

## 03 研究问题

- 具体问题：固定模板或语料分布限制的规划器难以表达复杂天然产物需要的低频、上下文依赖型关键断键。[Paper: PDF pp. 3–4]
- 方法问题：能否让 LLM 先给出多个战略假设，再直接写原子级图编辑扩展搜索，并在不重跑整棵搜索树的前提下修补完整路线？[Paper: PDF pp. 4–6, 18]
- 对本次修复最关键的问题：公开论文把哪些判断交给 Strategy、host、Critic 和 Editor，各层是否拥有“战略已实现”的权威？

## 04 研究背景与发展路径

`[Paper-framed; external verification not performed]`

1. 模板/神经扩展策略：搜索稳定，但反应空间受历史语料约束。[Paper: PDF pp. 2–4]
2. Synthelite 等 LLM policy：可用自然语言提出下一步，但仍由固定模板库落地。[Paper: PDF p. 3]
3. SynthEx：LLM 直接写 ReactionJSON，host 确定性得到前体；随后用 RouteJSON 支持路线级编辑。[Paper: PDF pp. 4, 6, 23]
4. Critic–Editor：以同 backbone 的前向化学审计和就地编辑降低内部 blocking rate，但不是独立实验 oracle。[Paper: PDF pp. 18, 24]

## 05 论文识别的核心痛点

| 痛点 | 表现 | 作者解释 | 证据 |
|---|---|---|---|
| 固定反应空间 | 复杂天然产物无法被模板搜索闭合 | 关键成环、重排和级联在语料中稀疏 | [Paper: PDF pp. 3–4, 12–15] |
| 过早承诺单一路线 | 搜索缺乏战略多样性 | 应先生成多个战略假设再探索 | [Paper: PDF p. 6] |
| 自由 SMILES 不稳定 | LLM 直接重画分子容易失真 | 用 atom-map keyed graph edits 确定生成前体 | [Paper: PDF pp. 6, 23] |
| 初稿路线含 blocking steps | 条件、官能团、选择性或反应结构不相容 | 初始 Route Builder 只产出草稿，需 Critic–Editor 循环 | [Paper: PDF pp. 18–20] |
| 无真实化学 oracle | 同类 LLM 可共享盲点 | 真正可行性仍需实验 | [Paper: PDF pp. 18, 21, 24] |

## 06 核心思想

1. 表层方法：`Strategy Generator → LLM-guided route search → RouteJSON → Critic ↔ Editor → Analyst → short-tail completion`。[Paper: PDF pp. 5–6, 18–24]
2. 核心洞见：Strategy 是搜索的语义先验，ReactionJSON 是确定性结构执行语言，Critic–Editor 是化学一致性修复层；这三种职责没有被压缩成一个不完整的 Strategy→bond-pair compiler。
3. `[Analysis]` 对 AutoPlanner 的直接启示：`paper_matched_reach` 不应让 `required_map_pairs` 获得生成约束或战略完成权威；若未来要由 host 判断机制事件完成，必须另建完整的 StrategicEvent 表示和独立实验模式。

## 07 方法概览

- 输入：目标结构；可选自然语言约束或必需起始物。[Paper: PDF p. 22]
- Strategy Generator：默认一次产生三条独立的一句话策略，经 scaffold、key bond-forming reaction、FG/protection 和 stereocenters 四点分析；每条策略面向一至两个关键反应，并作为后续搜索的 `steering query`。[Paper: PDF p. 22]
- Route Builder：AiZynthFinder MCTS 的 LLM-guided expansion policy，逐节点写 ReactionJSON；公开论文只给出步数上限 25，没有公开精确 node prompt、候选宽度或隐藏过滤规则。[Paper: PDF pp. 22–23]
- ReactionJSON：十类 primitive；把操作应用到 mapped product 后确定性得到 mapped precursors。[Paper: PDF p. 23]
- Critic–Editor：对完整 RouteJSON 前向审计；Editor 可重排、插入、删除、改条件、改官能团，并尽量保留 key disconnection 与 overall strategy。[Paper: PDF pp. 18, 24]
- 停止：无 blocking step 或达到迭代上限；solve 则独立定义为完整路线所有叶节点精确命中 stock。[Paper: PDF p. 24]

流程：

```text
target
→ one Strategy call / three steering queries
→ one independent LLM–MCTS search per strategy
→ node-local ReactionJSON → deterministic precursor replay
→ complete RouteJSON
→ Critic(all steps) → Editor(full route) → Critic ... ≤ 6 edits
→ Analyst
→ optional short AiZynthFinder leaf completion
```

## 08 核心模块拆解

| 模块 | 公开论文中的职责 | 输入/输出 | 不应拥有的权威 | 本项目修复含义 |
|---|---|---|---|---|
| Strategy Generator | 生成三条战略 steering queries | target → natural-language strategies | 不证明路线或机制已实现 | 保留完整语义卡，删除 hard checklist 语义 |
| Route Builder | 根据 Strategy 和当前结构写下一步 graph edits | mapped leaf → ReactionJSON | 初稿化学不自动可信 | 允许 supporting transformations 和 tactical freedom |
| host replay | 确定性应用 graph edits | product + ops → precursors | 不判 named mechanism 为真 | 只保留 schema、map、sanitization、continuity 等结构不变量 |
| Critic | 前向模拟每一步并标 blocking | complete RouteJSON → all step annotations | 不授予实验真值 | 当前能力基本保留；必须把全部 blocker 交给 Editor |
| Editor | 就地修路线并保持总体战略 | route + all annotations → revised RouteJSON | 不必保持原终端叶或原路线长度 | 默认输出完整修订 RouteJSON；允许新手柄和步骤 |
| Analyst | 对修完路线评分和列风险 | finished route → analysis | 不影响 stock solved | 与结构 solve 分轴报告 |

## 09 关键公式与符号

论文无核心算法公式。与修复相关的两个定义是：

- `blocking rate = Critic 标为 blocking 的步骤数 / 路线总步骤数`；这是内部一致性指标，不是实验失败率。[Paper: PDF pp. 18, 24]
- `solved = 完整路线的所有 leaves 以 full InChIKey 精确命中 ZINC + eMolecules stock`；不要求 Critic 通过或实验验证。[Paper: PDF p. 24]

## 10 实验设计与证据链

| 实验/图表 | 论点 | 结果或作用 | 正确边界 | 来源 |
|---|---|---|---|---|
| Figure 1 | 系统模块和 Strategy/Route Builder/Critic–Editor 分工 | 三策略、逐步路线、迭代编辑 | 架构示意，不公开精确 prompt | [Paper: PDF p. 5] |
| Figure 2 | 三个复杂目标案例 | 展示战略关键步 | 案例，不是随机可靠率 | [Paper: PDF pp. 7–10] |
| Figure 3 + Table 1 | 输出反应空间不同于 USPTO | 成环步骤识别率低、结构更构建性 | 不证明反应正确 | [Paper: PDF pp. 11–14] |
| Figure 4 + Table 2 | reach 与专家 key-step 评价 | 13.8%/25.0%/63.9%；47 个 strategy-congruent targets 进入盲评 | 盲评条件于 shared strategic frame | [Paper: PDF pp. 15–17] |
| Figure 5 | Critic–Editor 是否能降低内部 blockers | blocking rate 约 0.27→0.06（六轮）；Editor 加 MgBr、插保护步骤、重排路线 | 同 backbone 内部收敛，不是实验验证 | [Paper: PDF pp. 18–19] |
| Figure 6 | LLM 与模板查询量 | 描述不同搜索阶段的调用形态 | 不是等成本比较 | [Paper: PDF p. 36] |
| Figure 7 | 共享求解集的路线长度 | 描述 SynthEx 与基线的路线长度 | 只覆盖双方均求解的目标 | [Paper: PDF p. 37] |
| Figure 8 | 专家评分的组别效应 | 展示不同研究组的偏好差异 | 不能代表全部合成群体 | [Paper: PDF p. 39] |
| Figure 9 | 组间 item overlap | 检查不同组评价项目的重叠 | 是稳健性诊断，不扩展总体 | [Paper: PDF p. 39] |
| Figure 10 | 评分尺度差异 | 展示不同组使用评分量表的差异 | 支持以 rater 聚类处理 | [Paper: PDF p. 40] |
| Figure 11 | 逐 rater 异质性 | 展示个体效应差异 | 十位专家，仍有抽样边界 | [Paper: PDF p. 40] |
| Table 3 | 专家评分补充统计 | 汇总各轴与敏感性统计 | 不改变 shared-strategy 条件选择 | [Paper: PDF pp. 39–41] |

本次最强直接证据是 Figure 5：原始 R1 把 `CH4` 当亲核体，Critic 指出缺 `MgBr`；Editor 在 ReactionJSON 中加入 `add_group ... *MgBr`，并同时插入保护/脱保护、移动 HWE、重排 Friedel–Crafts 与 spiroketalization。[Paper: PDF p. 19]

## 11 对结论的正确解释

- `[Paper]` Strategy 是 steering query，但系统仍要求路线围绕 key disconnection，并要求 Editor 保留 key disconnection 和 overall strategy。[Paper: PDF pp. 5–6, 22, 24]
- `[Analysis]` 因此正确修复不是“完全取消 Strategy 约束”，而是取消 host 的 bond-pair completion checklist，让约束恢复为 LLM 语义先验和 Critic 的 strategy-adherence 判断。
- `[Paper]` 0.27→0.06 证明广义 blocking reactions 在同模型循环中下降；不能证明这些 blocker 多数属于缺反应手柄或 Strategy/ReactionJSON 编译错误。[Paper: PDF pp. 18, 24]
- `[Paper]` 专家盲评只回答“找到共享战略后，关键步质量如何”，不回答 Strategy 找到率或任意 Route Builder 初稿的正确率。[Paper: PDF pp. 16, 21]
- `[Analysis]` 当前 AutoPlanner 的主要非论文行为不是“Builder 会犯错”，而是 host 用不完整键对表示宣布 strategy fulfilled，同时 Editor 又被禁止做论文示例中的结构性修复。

## 12 作者明确承认的局限

| 局限 | 表现 | 作者方向 | 来源 |
|---|---|---|---|
| 未做实验验证 | route 是纸面假设 | 未来接入实验闭环 | [Paper: PDF p. 21] |
| 未验证立体结果 | 仍有 selectivity/feasibility errors | 以实验事实评估 | [Paper: PDF p. 21] |
| 专家比较有条件筛选 | 仅共享战略的 47 个 target | 更直接研究战略可靠性/人类 steering | [Paper: PDF pp. 16, 21] |
| Critic 非独立 oracle | Critic、Editor、Analyst 共享 backbone | 不能把 blocking rate 当实验真值 | [Paper: PDF pp. 18, 24] |
| 模型与运行不可完全复现 | 无 seed、非零温度、报告来自单次全基准运行 | 论文未给消除方案 | [Paper: PDF pp. 23, 27] |

## 13 批判性分析

| `[Analysis]` 观察 | 影响 | 如何验证 | 依据 |
|---|---|---|---|
| `required_map_pairs` 是不完整的机制投影 | 诱发 reaction-label compliance 和 host 假阳性 | paper-matched A/B：hard progress vs hidden diagnostic-only | [Paper: PDF pp. 22–24] + 当前 smoke |
| 当前 Editor 只收到首个完整 blocker 的精简评估 | 容易被最难 anchor 卡死，忽略可先修的 Mg/保护/顺序问题 | 将全部 blocking assessments 放入一次 Editor prompt | [Paper: PDF pp. 18, 24] |
| 当前 prompt 要求保持 exact target-to-terminal-leaf boundary | 与论文允许改变官能团、插删步骤和替换 disconnection 冲突 | 只冻结 target root 和 route continuity，允许 terminal leaves 改变 | [Paper: PDF pp. 6, 18–19, 24] |
| paper-matched Editor 默认另一个 `route_patch` DSL | 增加论文没有描述的语义层和协调负担 | 让完整 revised RouteJSON 成为默认，patch 仅作简单可选优化 | [Paper: PDF pp. 18, 24] |
| Figure 5 的新增 MgBr 无全局原子映射 | 当前 host 要求新增 fragment 每个原子预先映射，比论文展示更严格 | 由 host 为未映射新增原子确定性分配新 map，再进行 replay | [Paper: PDF p. 19] |

## 14 学到的知识

### Agent-derived knowledge candidates

- 语义 steering 与符号 completion 是两种不同权威；不完整 compiler 比无 compiler 更危险。
- 路线初稿错误不是系统事故，而是 SynthEx 设计中由 Critic–Editor 吸收的正常方法结果。
- 可编辑 RouteJSON 的价值在于允许终端叶、官能团状态、步骤数和顺序变化，而不仅是“同一边界换条件”。
- host 应精确保证 graph/replay，Critic 应判断 chemistry；两者的结果必须分轴，不互相冒充。

## 15 与现有知识和本项目的连接

- `[User]` 用户提供的“semantic steering + graph execution + LLM chemical auditing”概括与论文公开架构一致。
- `[Analysis]` 当前 smoke 的 Branch 1 几乎复现了 Figure 5 的错误类型：中性丙烯被当作 organomagnesium equivalent；论文 Editor 会加金属手柄，而当前 Editor 因边界约束弃权。
- `[Analysis]` Branch 3 的 missing alkyne/CO carbonyl 更接近需要替换 dependency neighborhood 的复杂 blocker；论文允许 replace a disconnection，但未公开具体 prompt 或成功率，不能保证一次 Editor 就能修好。
- `[External]` 官方实现仍未发布，因此精确复现只能针对论文公开协议，不能声称复现隐藏实现。

## 16 研究设想

### Agent-derived research candidates

**A. Paper-faithful steering ablation**

- 起点：当前 hard anchor checklist 与 Methods 4.2 不一致。
- 假设：隐藏 `strategy_anchor_progress`、只保留 Strategy 语义 steering，会降低“只断目标键、不造反应手柄”的比例。
- 差异：representation/feedback 从 hard symbolic completion 改为 diagnostic-only。
- Validation / 验证：同一三目标、同模型、同 token/调用预算；比较 Critic blocking rate、strategy adherence、真实手柄缺失率、stock closure。
- 证伪：hard/soft 两臂的 blocker 类型和比率无差异，或 soft arm 显著丢失战略多样性。
- Possible failure modes / 可能失败模式：模型完全忽略 steering；单次运行方差大。
- 创新状态：`unverified`。

**B. Figure-5 Editor conformance test**

- 起点：论文展示 MgBr、保护步骤和重排，而当前 Editor 全部弃权。
- 假设：完整 blocker 集 + full RouteJSON 默认输出 + 可改变 terminal leaves，能使 Editor 完成同类结构修复。
- 差异：Editor action/representation contract。
- Validation / 验证：固定三类合成缺陷（missing organometallic handle、FG incompatibility、wrong order），要求 route replay、再次 Critic、非阻断步骤保留。
- 证伪：Editor 仍高比例弃权，或修复后 chain/replay 失败率上升。
- Possible failure modes / 可能失败模式：完整 RouteJSON token 成本高；同 backbone 共同盲点。
- 创新状态：`unverified`。

**C. 完整 StrategicEvent 增强模式**

- 起点：若未来希望 host 判定机制实现，单纯 target bond pairs 不够。
- 假设：显式编码 mechanism、precursor-only bonds、required handles、temporary groups、stereo provenance 和 sequencing，能减少 Critic 的 strategy mismatch。
- 差异：新增完整语义表示；不得放进 `paper_matched_reach`。
- Validation / 验证：与 paper-faithful soft steering 在预算匹配下比较机制假阳性、Critic agreement 和可重放率。
- 证伪：事件 compiler 错误率抵消收益，或显著压缩搜索多样性。
- Possible failure modes / 可能失败模式：本体覆盖不全；host 规则过度保守。
- 创新状态：`unverified; prior-art search required`。
