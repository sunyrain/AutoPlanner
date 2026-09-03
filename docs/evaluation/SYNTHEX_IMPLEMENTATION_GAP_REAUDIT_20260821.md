# SynthEx 原论文—AutoPlanner 复现差距再审计（2026-08-21）

> Source coverage: Full paper  
> Extraction confidence: High  
> Locator mode: page-grounded  
> Primary analytical lens: methods / executable-contract reproduction  
> Secondary analytical lens: implementation and experiment-ledger audit  
> Context verification: Targeted external check  
> Card completeness: Complete relative to supplied source

输入范围：SynthEx arXiv v1 全文、本仓库当前实现、`synthexfig1-001-paper-matched-reach-v5` 的真实付费运行报告与 worker journal。论文类型：方法论文。本文中的 `[Paper]` 是论文陈述，`[External]` 是官方仓库或 arXiv 元数据，`[Analysis]` 是基于论文与本地执行证据的判断。

## 01 基本信息

| 字段 | 内容 |
|---|---|
| 标题 | SynthEx（以 arXiv v1 全文标题为准） |
| 作者与单位 | 见论文首页；本卡不重抄作者列表 |
| 年份 / 载体 | 2026，arXiv:2608.07454 v1 |
| 论文类型 | LLM 引导的多步逆合成规划方法 |
| 领域 | 计算化学、逆合成、LLM agent、MCTS |
| 代码 | 官方仓库存在，但截至 2026-08-21 README 仍称实现代码尚未发布 |
| 数据 / 库存 | 论文使用 ZINC + eMolecules 合并库存，按完整 InChIKey 精确命中 |
| 阅读日期 | 2026-08-21 |
| 对本项目的定位 | AutoPlanner 的化学主链应先建立一个最小、可核验的 SynthEx 等价基线，再叠加酶法、条件、证据与审计层 |

[Paper: PDF p. 1, title and abstract] [Paper: PDF p. 24, Section 4.6]

## 02 一句话总结

SynthEx 用 LLM 先生成三条简短战略，再让每条战略进入独立的 AiZynthFinder MCTS、以 ReactionJSON 逐节点扩展完整路线，并在 RouteJSON 完成后进行 Critic–Editor 修复，最终用同一商业库存和短尾搜索统计闭合率；论文报告在 1,098 个目标上由 AiZynthFinder exhaustive 的 13.8% 提升到 stitched 的 63.9%，但没有公开足以直接重建执行行为的完整代码、prompt 和配置。[Paper: PDF p. 6, Figure 2 and overview] [Paper: PDF p. 15, Table 2] [Paper: PDF pp. 22–24, Methods]

## 03 研究问题

- 具体问题：纯模板搜索难以提出有价值的高层战略，纯 LLM 路线又慢、少且可能化学不可靠。
- 重要性：复杂天然产物的成功率主要受战略断键和长程路线一致性限制，而不只是单步模板覆盖。
- 既有方案不足：标准 AiZynthFinder 的局部模板扩展缺少全局战略；一次性 LLM 文本路线缺少树搜索、库存闭合和可执行结构表示。[Paper: PDF pp. 4–6, Introduction and Figures 1–2]
- 精确问题：能否让 LLM 作为 MCTS 的战略与单步动作策略，并用结构化图编辑和路线级修复，在固定库存边界上提高复杂目标的可达率？

## 04 研究背景与发展路径

| 阶段 | 代表方式 | 优点 | 局限 | 论文声称的位置 |
|---|---|---|---|---|
| 模板 / 神经策略搜索 | AiZynthFinder MCTS | 搜索、库存和路径评分成熟 | 高层战略弱，复杂目标可达率低 | 保留其搜索环境与库存语义 |
| LLM 直接路线规划 | 文本或一次性完整路线 | 能表达战略与反应知识 | 路线少、慢，结构和局部一致性弱 | 不采用一次性文字路线作为唯一计划 |
| SynthEx | LLM Strategy + LLM ReactionJSON policy + AiZ MCTS + RouteJSON repair | 战略、树搜索、结构化动作和短尾组合 | 真实执行契约和代码未完全公开 | 论文主张的混合架构 |

这一路径主要来自论文自身叙述，未在本卡中独立完成整个领域的系统综述。[Paper: PDF pp. 4–6, Introduction]

## 05 论文识别的核心痛点

| 痛点 | 表现 | 原因或作者解释 | 论文证据 |
|---|---|---|---|
| 局部搜索缺少战略 | 复杂骨架无法被模板搜索有效拆解 | 模板策略偏局部，不显式表达路线级合成逻辑 | [Paper: PDF pp. 4–6, Figures 1–2] |
| LLM 路线稀疏且昂贵 | 一次生成的候选有限，推理成本高 | 缺少树搜索复用与受控逐节点扩展 | [Paper: PDF p. 22, Section 4.2] |
| LLM 反应难以执行 | 文本反应可能结构不一致 | 需要 ReactionJSON 图编辑与宿主重放 | [Paper: PDF pp. 22–23, Sections 4.2–4.3] |
| 完整路线含局部缺陷 | 单步看似合理但路线级顺序、官能团或条件冲突 | 需要完整 RouteJSON 后的 Critic–Editor | [Paper: PDF p. 18, Figure 5] |
| 搜索叶接近库存但未闭合 | LLM 路线留下短尾 | 用深度 6 的传统搜索拼接 | [Paper: PDF pp. 15, 24, Table 2 and Section 4.6] |

## 06 核心思想

1. 表层方法：三条战略、三个独立 MCTS、逐节点 ReactionJSON、完整 RouteJSON、Critic–Editor、短尾拼接和统一库存。
2. 核心洞见：让 LLM 决定“往哪里断、下一步做什么”，让成熟搜索框架管理 OR/AND 状态、回溯、库存和路线选择；两者不是串行的“LLM 先写骨架、传统工具再补尾巴”。[Paper: PDF pp. 22–24, Methods]
3. `[Analysis]` 可迁移教训：论文图中的模块名称不是可执行合同；若 prompt、动作接纳、计数、停止和修复语义不闭合，同一框图可以实现出完全不同的算法。

## 07 方法总览

| 项目 | 论文做法 |
|---|---|
| 输入 | 目标分子结构、固定库存、LLM 与 AiZynthFinder 环境 |
| 输出 | 一组完整或可短尾闭合的 RouteJSON 路线 |
| 战略 | 默认 3 条、每条一句话，分析骨架、关键成键、官能团冲突/保护和立体中心；战略目标应可在 1–2 个关键步骤实现 |
| 搜索 | 每条战略进入独立 AiZynthFinder MCTS，阶段一每臂最多 25 次 LLM 扩展 |
| 动作 | LLM 产生结构化 ReactionJSON，由宿主执行图编辑 |
| 修复 | RouteJSON 完整后，Critic 前向检查，Editor 作局部插入/删除/重排/条件或官能团修复，最多 6 轮 |
| 短尾 | 每个不同开放叶使用 max depth 6、500 iterations、1200 s 的传统搜索 |
| solved | 所有叶都命中同一冻结库存 |

文本流程：`target → 3 strategies → 3 independent AiZ MCTS trees → target-rooted complete RouteJSON → Critic/Editor → open-leaf short tail → exact stock closure → solved`。[Paper: PDF pp. 22–24, Sections 4.2–4.6]

## 08 核心模块拆解

| 模块 | 功能 | 必要性 | 输入与输出 | 支持证据 | 移除后的影响 |
|---|---|---|---|---|---|
| Strategy generator | 产生互异高层战略 | 给搜索全局方向 | target → 3 one-sentence strategies | [Paper: PDF p. 22, Section 4.2] | `[Expected]` 回到局部模板搜索 |
| AiZ MCTS | 管理树、节点、回溯与路径 | 避免单链贪心 | state + actions → search tree | [Paper: PDF p. 22, Section 4.2] | `[Expected]` 候选更少、无法正常回溯 |
| ReactionJSON policy | 对当前节点提出可重放图编辑 | 把 LLM 化学提议变成结构动作 | molecule/context → ReactionJSON | [Paper: PDF pp. 22–23, Sections 4.2–4.3] | 无法可靠物化前体 |
| Route Builder | 将逐节点结果组成完整路线 | 建立路线级对象 | accepted nodes → RouteJSON | [Paper: PDF p. 6, Figure 2] | 无法做路线级修复 |
| Critic–Editor | 完整路线后发现并修复阻断缺陷 | 减少路线级不一致 | RouteJSON → critique/patch | [Paper: PDF p. 18, Figure 5] | 论文内部阻断率下降收益消失 |
| Short-tail stitcher | 闭合接近库存的开放叶 | 利用模板搜索强项 | open leaf → stock-closed tail | [Paper: PDF p. 24, Section 4.6] | stitched solved 率下降 |
| Exact stock oracle | 定义可比 solved | 防止购买边界漂移 | leaf InChIKey → membership | [Paper: PDF p. 24, Section 4.6] | 指标不可比 |

## 09 必要公式与符号

论文的关键结果不依赖新公式；理解复现只需以下预算与判定：

- LLM 策略搜索预算：`3 strategies × 25 expansions = 75 policy calls / target`。论文的均值和中位数均为 75，说明基准运行通常用满该上限。[Paper: PDF pp. 36–37, Figure 6]
- paper-equivalent solved：`solved = all(route leaves ∈ same frozen stock)`。[Paper: PDF p. 24, Section 4.6]
- stitched 短尾：对每个不同开放叶运行 `depth ≤ 6, iterations ≤ 500, time ≤ 1200 s`。[Paper: PDF p. 24, Section 4.6]
- MCTS 的具体 UCB 公式和战略层超参数在论文正文中没有足够细节支撑逐参数复现，因此这里标为 `Not assessable from the supplied paper`。

## 10 实验设计与证据链

论文设置：1,098 个目标；固定 ZINC + eMolecules 库存；AiZynthFinder exhaustive、strategic-only 和 stitched 三个层次；baseline 为最多 25 transforms、1500 iterations、1800 s、模板累计概率 0.995、最多 50 个动作，论文报告 baseline exploration constant 1.4；LLM 是 `gemini-3.1-pro-preview`，strategy temperature 0.1、next-step temperature 0.3、每次调用 600 s timeout、最多重试 3 次、无工具/网络。[Paper: PDF pp. 15, 23–24, Table 2 and Methods]

| 实验 | 被检验主张 | 比较与条件 | 结果 | 支持的结论 | 不支持的更强结论 | 来源 |
|---|---|---|---|---|---|---|
| Exhaustive baseline | 传统搜索覆盖率 | AiZ exhaustive，同库存 | 151/1098 = 13.8% | 当前 baseline 在该设置下可解 13.8% | 不能代表所有 AiZ 配置上限 | [Paper: PDF p. 15, Table 2] |
| Strategic only | LLM 战略搜索增益 | 3×25，不拼短尾 | 275/1098 = 25.0% | 战略层提高拓扑可达 | 不能证明路线实验可行 | [Paper: PDF p. 15, Table 2] |
| Stitched | 短尾组合增益 | strategic route + depth-6 tail | 702/1098 = 63.9% | 组合在相同库存指标上显著增益 | 不能与不同库存/模型直接比较 | [Paper: PDF p. 15, Table 2] |
| Critic–Editor | 路线内部阻断缺陷减少 | 最多 6 轮内部评价 | blocking rate 约 0.27 降至约 0.06 | 内部 Critic 标记减少 | 不是独立实验验证率 | [Paper: PDF p. 18, Figure 5] |

### 主图表覆盖清单

- Figure 1：三阶段 agentic planning pipeline 与 SynthAtlas 资源。[Paper: PDF p. 5, Figure 1]
- Figure 2：Okaramine M、Melonine 等战略推理案例。[Paper: PDF p. 8, Figure 2]
- Figure 3：SynthEx 与 USPTO/template 方法的反应空间差异及环构建富集。[Paper: PDF p. 13, Figure 3]
- Table 1：5,318 个成环步骤和 27,827 个非成环步骤的反应命名器识别情况。[Paper: PDF p. 14, Table 1]
- Figure 4：基准子集 solve rate、分子量退化与专家评分等总体结果。[Paper: PDF p. 17, Figure 4]
- Figure 5：Critic–Editor 迭代与阻断反应率变化。[Paper: PDF p. 19, Figure 5]
- Figure 6：Gemini policy query 与 AiZ 模板调用的预算说明。[Paper: PDF p. 36, Figure 6]
- Figure 7：共同解出的 134 个目标上，SynthEx 路线中位长度 5、AiZ 中位长度 11。[Paper: PDF p. 37, Figure 7]
- Figures 8–11：按研究组、重叠条目、原始评分分布和逐评分者异质性展开专家评价的稳健性与边界。[Paper: PDF p. 39, Figure 8] [Paper: PDF p. 39, Figure 9] [Paper: PDF p. 40, Figure 10] [Paper: PDF p. 40, Figure 11]
- Table 3：leave-one-rater-out 的 Cliff's delta 稳健性分析。[Paper: PDF p. 41, Table 3]

### 本地最新付费运行对照

| 维度 | 论文合同 | 本地 v5 实际 | 判定 |
|---|---|---|---|
| 战略臂 | 3 | 3 | 对齐 |
| 阶段一 LLM policy 调用 | 最多 75，论文统计通常 75 | journal 中真实节点调用 61（21/20/20），但 sidecar 报告 75 | **硬错误：代理回调被错计为实际调用** |
| 路线拓扑 | 完整 RouteJSON | 3 条 target-rooted 路线，2/6/6 步，各剩 1 个开放叶 | `paper_reach=true`，未 solved |
| Critic–Editor | 每条完整路线后，最多 6 轮 | 仅 2 次 Critic、1 次真实 Editor；没有完整闭环 | **不对齐** |
| 短尾 | 每个开放叶 6/500/1200 | 三个叶均真实运行；分别约 2332、3028、2013 节点，均剩 1 个未入库前体 | 对齐执行，但搜索失败 |
| 库存 | full-InChIKey ZINC+eMolecules | 本地冻结 union 39,478,827 个唯一有效键；论文原始记录数 39,684,411 | 语义基本对齐，计数差来自去重/无效行处理，非主因 |
| solved | 所有叶入库 | 0/3 stock closed | `paper_equivalent_solved=false` |

本地模型账本为 67 次总调用、约 1,151,997 input tokens、120,801 output tokens；其中 3 次 StrategyCard、61 次 route node、2 次 Critic、1 次 Editor。每个 route node 平均约 17,283 input tokens。因此按当前适配器，75 次节点调用本身约需 1.296M input tokens，尚未计算战略与修复；用户此前设定的 1.2M 总输入上限数学上无法容纳论文式 75 次实际节点调用。[Analysis: local worker journal and run budget]

## 11 正确解读结论

- 论文解决的是固定库存下的计算拓扑可达与拼接成功，不等于每个反应经实验或文献证据验证。
- `strategic-only`、`stitched`、反应验证、条件完整度与证据强度必须分别报告；AutoPlanner 更严格的后验验证不能用来解释 paper-equivalent solved 失败。
- 论文的 75 个 LLM calls 与模板 baseline 的 75 次模板动作不是等价算力；论文也明确提示调用成本不能直接类比。[Paper: PDF pp. 36–37, Figure 6]
- 模型依赖明显：论文使用 Gemini 预览模型；本地选择 GPT-5.6-terra medium 是一个新的 backbone，不能只凭主观能力排序推断一定优于论文。
- 本地 v5 的 0/3 solved 是真实的库存闭合失败；但它还不是有效的“论文等价 3×25”实验，因为真实 policy 调用只有 61，且 Critic–Editor 未完整执行。

有界重述：论文证明其未公开完整实现，在自身模型、prompt、库存和计算预算上达到了表 2 的覆盖率；我们目前只复现了主要模块形状和部分运行语义，尚未复现完整执行合同，不能据当前结果判断 AutoPlanner 或 Codex 的算法上限。

## 12 作者明确承认的局限

论文没有用一个集中 `Limitations` 表完整枚举所有工程复现缺口，因此不把我们的推断冒充作者声明。

### 作者提到的相关约束（非正式 limitations 表）

| 约束 | 具体表现 | 作者提出或暗示的处理 | 来源 |
|---|---|---|---|
| LLM 与模板调用不可视为等价计算 | 调用成本和信息量不同 | 分开报告预算 | [Paper: PDF pp. 36–37, Figure 6] |
| Critic–Editor 指标是内部判据 | 阻断率下降不等于实验验证 | 把它作为路线内部改进指标 | [Paper: PDF p. 18, Figure 5] |
| 结果依赖固定库存和配置 | solved 边界由库存定义 | 统一 full-InChIKey 库存 | [Paper: PDF p. 24, Section 4.6] |

## 13 批判性分析

| `[Analysis]` 观察 | 潜在问题或替代解释 | 为什么重要 | 如何检验 | 依据 |
|---|---|---|---|---|
| 论文公开不等于实现公开 | prompt、代码、Action admission、空调用、停止与 patch 语义未发布 | 只能做合同重建，不能逐行复刻 | 等官方发布后做 commit-pinned differential test | 官方仓库状态 + 论文 Methods |
| 本地把 provider callback 当实际 LLM call | budget 拒绝时返回空候选但计数已增加 | 伪造“25/臂”完成态 | 加入 callback=25、journal=20 的失败测试 | 本地接口与 journal |
| 1.2M token cap 与当前 schema 冲突 | 61 节点已经消耗约 1.05M input | 不是并行能解决的吞吐问题 | 压缩 schema/context 后做 75-call dry ledger | 本地账本 |
| RouteJSON 顺序合同自相矛盾 | 宿主要 target-rooted retrosynthetic order；Critic 被要求“forward-simulate”却没说明应逆序执行依赖 | Editor 会把正确存储顺序重排后又被宿主拒绝 | 构造三步路线，断言 critic 不因存储方向报错 | 本地 prompt、diagnostic、critic 输出 |
| Critic–Editor 预算调度不公平 | 早期分支为未来分支预留预算，反而先被拒绝 | 不是论文的逐路线修复闭环 | 每臂独立 repair budget + 最多 6 轮状态机测试 | 本地预算逻辑 |
| 本地战略 schema 过重 | 20+ 字段、跨臂禁用与替代比较远超论文四点一句话战略 | 可能抑制直观战略并浪费上下文 | A/B：paper-minimal prompt vs current rich prompt | 论文 p.22 与本地 schema |
| Traversiadiene 战略不匹配论文展示 | 论文示例强调两步环化后 Grob fragmentation；本地三臂未稳定提出该家族 | 说明战略 prompt 尚未达到示例能力 | 固定同一 target/model，比较 strategy recall 与 route closure | 论文 Figure 1、本地路线 |
| 测试主要覆盖 mock happy path | 真实 `codex_cli` backend、预算耗尽、回调/实际计数分离未覆盖 | 单测通过但真实路径失败 | 增加 ledger、order、repair state-machine 集成测试 | 本地 tests 与 journal |

## 14 学到的知识

### Agent-derived knowledge candidates

- “架构复现”至少有四层：模块拓扑、数据/库存、执行语义、预算/停止语义；只对齐第一层不足以复现结果。
- 对 LLM 搜索，计费/模型调用 journal 才是实际动作预算的真值，provider callback 只是请求数。
- 路线存储方向与化学执行方向可以相反，Critic prompt 必须显式定义两者。
- 短尾搜索的价值建立在前缀质量上；三个尾巴都只差一个叶并不意味着再加时间一定能闭合，也可能是前缀把问题推到了模板/库存不可覆盖区域。
- 更严格的证据、条件和反应验证层应作为独立投影，不能污染论文 solved 指标，也不能用来掩盖 paper-equivalent 主链问题。

## 15 与现有知识的联系

- 与 AiZynthFinder 的联系：SynthEx 不是抛弃 AiZynthFinder，而是把 LLM 作为战略和逐节点策略嵌入其 MCTS，再使用传统搜索闭合短尾。[Paper: PDF p. 22, Section 4.2]
- 与本项目早期 ChemEnzy 路线的冲突：若 ChemEnzy 只作为尾部 provider，而上游仍是单链 Director，就没有复现论文的“每个策略一棵独立 MCTS”。
- 与 AutoPlanner 的优势互补：酶法、条件、证据和 UI 是潜在论文增量，但必须在化学 paper-equivalent 基线稳定后作为独立臂或后验层加入，避免破坏基线可比性。
- `[External]` 官方 GitHub 当前仍把 planner、ReactionJSON/RouteJSON reference implementation 和 reproduction configs 列为待发布，因此现阶段不能把本地失败简单归因于“照着公开代码也没做好”。

## 16 研究想法

### Agent-derived research candidates

#### 候选 A：Executable-contract SynthEx clone

- 起源：公开论文缺少完整执行语义，本地多个代理指标失真。
- 核心假设：先建立仅含 3×25、AiZ MCTS、ReactionJSON、RouteJSON、repair、short-tail、stock 的小内核，可显著减少故障面。
- 相对论文增量：不是算法创新，而是可审计复现工程。
- 初步方法：所有状态转移写入单一 append-only ledger；derived counters 只从 ledger 计算。
- 验证（Validation）：同一 target 的 callback、accepted action、actual LLM calls、tree nodes、repair rounds 必须可逐条对账。
- 失败模式（Failure modes）：仍受未公开 prompt 和模型差异影响。
- 创新状态：`unverified`。

#### 候选 B：Compact ReactionJSON policy artifact

- 起源：当前每节点约 17.3k input tokens，1.2M 无法容纳 75 次节点调用。
- 核心假设：将节点输出缩为必需图编辑、简短理由与置信度，完整 RouteJSON 由宿主累计，可在相同或更低 token 预算下完成 75 次调用。
- 相对论文增量：强化宿主/模型职责分离与账本可审计性。
- 初步方法：删除每节点重复的全 StrategyCard、全 route、条件、证据和酶法字段；按哈希引用不可变上下文。
- 验证（Validation）：对同一 5-target canary 比较 action validity、closure、token/call 和 wall time。
- 失败模式（Failure modes）：上下文过短造成重复或失去长程一致性。
- 创新状态：`unverified`。

#### 候选 C：Paper-baseline + optional biocatalytic arm

- 起源：AutoPlanner 的酶法优势不应强制污染所有化学战略。
- 核心假设：保留 3 条论文等价化学臂，再增加独立可选酶/融合臂，可提高天然产物覆盖且不降低化学基线。
- 相对论文增量：战略组合中加入可选择的生物转化，而非每臂必带酶步骤。
- 初步方法：共同 MCTS/库存接口、领域特定 ReactionJSON action、统一 RouteJSON，指标按 chemical-only、hybrid、union 分报。
- 验证（Validation）：matched targets 上比较 union gain、独立新增 solved、路线长度、条件可执行性。
- 失败模式（Failure modes）：生物转化动作库和可购买辅因子/底物边界不足。
- 创新状态：`unverified`。

---

## 本地复现的最低验收顺序（实施审计附录，不新增 Paper Card 章节）

1. 冻结最小 paper-equivalent 主链，酶法、证据、条件和严格验证全部作为独立后验投影。
2. 以 worker journal 的实际成功返回为唯一 LLM policy-call 计数；空回调不得计入 25。
3. 压缩逐节点 schema/context，或明确取消论文没有的 1.2M 总 token 限制；先证明可真实容纳 3×25。
4. 明确 RouteJSON 为 target-rooted retrosynthetic storage order；Critic 对每步做 forward chemistry simulation，但按反向依赖顺序检查可执行性，不能因此改写存储方向。
5. 每条完成路线拥有独立 Critic–Editor 状态机和预算，最多 6 轮；每次 patch 后必须重放、接纳并 re-critic。
6. 用 Traversiadiene 单目标验收：3 条战略、真实调用账本、路线顺序、修复闭环、三个短尾和 exact-stock 判定全部通过后，再跑 Dibohemamine A 与 Cyclopiamine B。
