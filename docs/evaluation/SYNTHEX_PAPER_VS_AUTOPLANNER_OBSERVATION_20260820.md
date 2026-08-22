# SynthEx 原论文与 AutoPlanner 当前正式运行对照观察

> Source coverage: Full paper, including Methods and Supplementary figures/tables
> Extraction confidence: High for paper protocol and frozen local run artifacts
> Locator mode: page-grounded
> Primary analytical lens: Methods
> Secondary analytical lens: Resource
> Context verification: Official arXiv PDF plus frozen local protocol and run report
> Card completeness: Complete relative to the paper and the single observed target
> Local run coverage: One of three frozen Figure 1 targets; no population-level claim

术语账本：本文把 `SynthEx` 限定为论文中的策略优先规划系统，把 `SynthAtlas` 限定为其公开路线资源；`ReactionJSON` 是对映射产物执行的有序原子级图编辑，`RouteJSON` 是可局部修改的反应步骤序列；`paper reach` 表示存在目标根连通路线，`paper solved` 表示该路线的全部叶节点精确命中同一个 ZINC + eMolecules 库存；`B2` 是 AutoPlanner 的宿主反应验证轴，`B4` 是库存闭合轴，两者相互独立。

| 名称 | 本文固定含义 |
|---|---|
| 论文战略层 | 三个默认战略，每个战略由 LLM 引导最多 25 次逐步扩展 |
| 论文短尾 | 对战略层未购买叶运行 AiZynthFinder，depth 6、500 iterations、1200 s |
| 当前正式运行 | `synthexfig1-001-paper-3x25-v1` 中唯一已运行目标 `synthexfig1-001-9c1f431594a7` |
| 论文等价终点 | 至少一条目标根连通路线的全部叶节点按 full InChIKey 命中冻结库存；不要求条件、证据或 B2 |

## 01 基本信息

- 论文：Daniel Armstrong 等，*Strategy-first synthesis planning for complex natural products*，arXiv:2608.07454v1，2026-08-07。[Paper: PDF p. 1]
- 类型：方法论文兼开放路线资源论文，主题为复杂天然产物逆合成、LLM agent、ReactionJSON 和模板短尾闭合。[Paper: PDF pp. 2–6]
- 论文主干模型为 `gemini-3.1-pro-preview`；LLM 推理阶段不使用搜索 grounding 或网络工具，只查询冻结的本地模板库和库存。[Paper: PDF p. 23]
- 当前对照对象是 AutoPlanner 的单目标冻结运行，模型为 `gpt-5.6-terra`、medium，输入仅含盲化名称和目标 SMILES。

## 02 一句话总结

SynthEx 的优势不是“让 LLM 一次写完整路线”，而是把三个战略假设嵌入 AiZynthFinder 的树搜索语义中，用 ReactionJSON 连续扩展，再把已显著降复杂度的开放叶交给短程 AiZynthFinder；当前 AutoPlanner 已具备确定性图编辑、库存和短尾基础设施，但这次运行的逐节点调用没有稳定转化为可回溯的 OR 路线状态，因此仍未达到论文等价闭合。[Paper: PDF pp. 6, 15, 22–24]

## 03 研究问题

论文研究的是：当复杂天然产物需要的关键断键不在 USPTO 模板空间时，能否先让 LLM 生成多个高层战略，再以可执行图编辑逐步搜索，从而把难目标变成模板规划器可以完成的简单叶节点？[Paper: PDF pp. 3–6]

对 AutoPlanner 的直接问题则是：我们是否真正复现了这种“战略种子 + 树状态 + 逐节点动作 + 叶级闭合”，还是只复现了外观相似的三个提示和 ReactionJSON 字段？本次观察支持后者仍有明显差距。

## 04 研究背景与发展路径

1. 专利模板规划器在同分布基准上接近饱和，但复杂天然产物需要稀有成环、级联、重排和收敛断键。[Paper: PDF pp. 2–4]
2. 单纯扩大模板搜索预算无法产生模板库之外的反应空间；论文的近穷举 AiZynthFinder 在 1,098 个目标上仍仅解决 13.8%。[Paper: PDF pp. 14–15]
3. 直接生成反应 SMILES 不稳定，因此论文让 LLM 输出对映射产物执行的 ReactionJSON 图编辑。[Paper: PDF pp. 6, 23]
4. 完整路线被序列化为 RouteJSON，Critic 和 Editor 可在不放弃总体战略的前提下局部重排、插入、删除或修改步骤。[Paper: PDF pp. 18, 24]
5. 未购买叶由短程 AiZynthFinder 完成，形成“LLM 扩反应空间、模板搜索做常规闭合”的分工。[Paper: PDF pp. 15, 24]

## 05 论文识别的核心痛点

| 痛点 | 论文处理 | 对当前运行的启示 |
|---|---|---|
| 模板反应空间有限 | LLM 直接提出图编辑 | Codex 的价值应落在战略断键和新动作，不是文献检索 |
| 单一路线过早承诺 | 默认三个独立战略 | 三臂只是入口，还必须保留各臂内部的多状态搜索 |
| 自由 SMILES 难可靠落地 | ReactionJSON 由宿主确定性执行 | 当前宿主 replay 是已实现的优势 |
| 整路重生成难局部修复 | RouteJSON + Critic/Editor | 必须保证 canonical graph 与展示 RouteJSON 只有一个权威状态 |
| 模板搜索在完整目标上浪费 | 只对降复杂后的叶做短尾 | 不能把失败的短尾部分路线无限递归当成论文等价流程 |

## 06 核心思想

论文真正的算法组合是“战略优先的反应空间扩张 + 搜索状态管理 + 局部模板闭合”。三个战略提供宏观多样性；AiZynthFinder 的 Monte-Carlo tree search 框架提供节点、替代动作和回溯语义；ReactionJSON 提供可执行动作；Critic/Editor 提供完成路线后的局部修改；库存提供 solved 的终止判据。[Paper: PDF pp. 5–6, 22–24]

因此，模型强弱并不能自动推出系统强弱。即使 Codex 单步化学判断优于 Gemini，只要合法候选没有进入可回溯 OR 状态、开放叶提前失活、或拼接只保留局部片段，最终 reach 仍会更差。

## 07 方法概览

- Strategy Generator：默认提出三个彼此独立的一句话战略，提示固定检查 scaffold、关键成键、官能团冲突/保护和立体中心；战略预期由一至两步关键反应实现。[Paper: PDF p. 22]
- Route Builder：每个战略作为 steering query，LLM policy 逐步产生 ReactionJSON；Phase 1 guided search 的 step limit 为 25。[Paper: PDF pp. 22–23]
- ReactionJSON：十类图编辑原语，应用到映射产物后确定性生成映射前体。[Paper: PDF p. 23]
- RouteJSON：完成路线的线性 ReactionJSON 序列；Critic 标记 blocking reaction，Editor 最多迭代到无阻断或达到上限。[Paper: PDF pp. 18, 24]
- 搜索与库存：论文实现是 AiZynthFinder MCTS 的子类，LLM-guided policy 取代神经模板扩展 policy；full InChIKey 精确命中 39,684,411 条 ZINC + eMolecules 库存记录。[Paper: PDF pp. 22, 24]
- 短尾：每个未购买叶使用 AiZynthFinder，maximum 6 transforms、500 iterations、1200 s；返回完整路线才计 solved。[Paper: PDF pp. 24–25]

## 08 核心模块拆解

| 模块 | 论文职责 | 当前实现状态 | 关键差异 |
|---|---|---|---|
| Strategy Generator | 三个独立战略 | 三个独立 StrategyCard | 数量接近，提示与模型不同 |
| 搜索状态 | AiZynthFinder MCTS 中的 LLM policy | 自定义 campaign scheduler + canonical hypergraph | 不是论文同一搜索算法 |
| ReactionJSON | 每次逐节点动作 | 宿主 replay 与审计 | 已可确定性执行，但公开 profile 仍是项目侧解释 |
| RouteJSON | 完成路线的唯一可编辑对象 | Director skeleton、canonical graph、portfolio 多投影 | 当前存在权威状态分裂迹象 |
| Critic/Editor | 完整路线后局部修复 | 配置六轮；本目标无闭合路线，未发生有效 repair | 不能用“配置了六轮”代替实际执行 |
| AiZ 短尾 | 对每个未购买叶返回首个完整解 | 运行 11 个 native search，并会物化最佳部分路线 | 比论文流程更递归、更宽松，不是严格 matched arm |
| Solved metric | 一条路线全部叶命中库存 | `paper_equivalent_solved` 已独立计算 | 这一口径已对齐 |

## 09 必要公式与符号

论文没有理解方法所必需的新方程。必要指标只有：

- `solve rate = solved targets / 1098`；`solved` 要求存在一条完整目标根路线且全部叶节点命中冻结库存。[Paper: PDF p. 24]
- `blocking rate = blocking steps / route steps`；该值是同类 LLM Critic 的内部一致性指标，不是实验失败率。[Paper: PDF pp. 18, 24]
- 当前单目标只能报告 `0/1 paper-equivalent solved`，不能与论文 `702/1098` 做显著性或总体性能比较。

## 10 实验设计与证据链

论文 benchmark 为 NPAtlas 2024_09 中 1,098 个目标：852 个 complex-large、123 个 complexity-dense、123 个 control。[Paper: PDF p. 22]

| 证据 | 结果 | 能支持什么 | 不能支持什么 |
|---|---|---|---|
| Table 1 | 5,318 个成环步骤中，专利派生分类器识别率很低，而 NameRXN 为 66.8% | SynthEx 输出偏离 USPTO 频率空间 | 不证明实验可行 |
| Table 2 | AiZ 单独 151/1098=13.8%；战略层 275/1098=25.0%；stitched 702/1098=63.9% | 战略降复杂与短尾互补 | 不构成等算力比较 |
| Figure 1 | Strategy Generator → Route Builder → Critic/Editor → Analyst/SynthAtlas | 论文完整系统结构 | 不给逐目标成功证明 |
| Figure 2 | Okaramine M、Melonine、Chanoclavine/Lysergol 案例 | 展示战略推理质量 | 是案例而非总体统计 |
| Figure 3 | 反应空间与成环/收敛统计 | 输出与 USPTO 空间有分布差异 | 分布差异不等于化学正确 |
| Figure 4 | 子集 solve rate、分子量趋势、专家评分 | 复杂度越高，SynthEx 相对优势越明显 | 专家比较只覆盖共享战略框架 |
| Figure 5 | 六轮后 blocking rate 约由 0.27 降至 0.06 | 同 backbone Critic 判据内收敛 | 不是独立验证或湿实验结果 |
| Figure 6 | AiZ intact target 平均/中位调用 24,218/28,802；LLM policy 均为 75；full SynthEx 平均/中位总调用 3,499/3,176 | 搜索形态不同 | LLM 调用与模板调用不是同一算力单位 |
| Figure 7 | 共同解决的 134 个目标中，SynthEx 中位 5 步、AiZ 11 步；105/134 更短 | 在共同解决集上路线较短 | 不能推广到所有失败目标 |
| Figure 8、Figure 9、Figure 10、Figure 11 与 Table 3 | 分组、项目重叠、评分尺度、rater 异质性和逐人剔除稳健性 | 说明专家评分需以 rater 聚类处理 | 仅十名 rater，且仍有组间差异 |

当前运行的冻结协议只选择 Figure 1 三个展示目标中的一个，不是论文 1,098 靶标总体样本。当前结果为：6 条 target-rooted 路线投影、1 条 host-validated 路线、0 条 stock-closed 路线；因此 paper reach 为真，paper solved 与 B4 为假。

## 11 对结论的正确解释

1. 论文的 63.9% 是结构拓扑和库存叶闭合，不要求条件、精确文献证据、B2 或实验结果。[Paper: PDF pp. 15, 24]
2. 当前失败不只是“我们更严格”：本次独立的 paper-equivalent 指标同样为 false。
3. 当前 B2=true 说明至少一条路线投影通过宿主反应验证，但 B2 不能替代 B4；反过来，论文 B4 也不证明反应可执行。
4. Figure 1 的 Traversiadiene 只展示关键战略步骤；公开冻结协议记录该组三个目标、九条参考路线的自报 stock-solved route count 为 0，因此不能拿论文总体 63.9% 当作这个单靶的已知阳性标签。
5. 论文 Figure 1 对 Traversiadiene 强调“还原性 ketyl–olefin 自由基成环后接 Grob fragmentation”；当前三臂分别是醇引发阳离子多烯环化、RCM/选择性加氢思路、以及晚期 C(sp3)–C(sp3) 拼接/脱氢/5-exo 自由基成环。第三臂只在“自由基成环”层面部分接近，没有复现论文的 ketyl–olefin + Grob 战略。[Paper: PDF p. 5]

## 12 作者明确承认的局限

| 局限 | 作者边界 | 来源 |
|---|---|---|
| 未做实验验证 | 报告 reach 和纸面步骤质量，不是实验可行性 | [Paper: PDF p. 21] |
| 未验证立体化学结果 | 专家审查仍发现选择性与可行性错误 | [Paper: PDF p. 21] |
| 专家比较有条件选择 | 只比较双方战略一致的 47/70 个目标 | [Paper: PDF pp. 16, 21] |
| Critic/Editor 非独立 oracle | 修复者与评分者共享同类模型，blocking rate 只表示内部收敛 | [Paper: PDF pp. 18, 21, 24] |
| LLM 成本高 | 论文明确说比较 reach 而非 compute cost | [Paper: PDF pp. 21, 24–25, 36] |
| 条件和实验闭环仍未完成 | 反应条件整合和实验 closed loop 被列为后续方向 | [Paper: PDF p. 21] |

## 13 批判性分析与当前运行观察

| 维度 | 论文 | 当前运行 | 判断 |
|---|---|---|---|
| 规模 | 1,098 目标 | 1 个冻结目标 | 只能做流程 canary，不能比较 solve rate |
| 模型 | Gemini 3.1 Pro preview | GPT-5.6 Terra medium | 模型不同，不是严格复现；强模型不自动弥补搜索语义 |
| 战略臂 | 默认 3 | 3 | 表面一致 |
| 逐臂展开 | MCTS 中每臂 hard limit 25；Figure 6 的均值和中位均为 75 次 LLM policy 调用/目标 | 三臂实际 compact calls 为 19、15、7，共 41；总模型调用 48 | 未吃满论文 policy 预算，且调用角色计数也不完全同义 |
| 有效连续路线 | 树搜索保留搜索状态 | 三臂最终展示为 1、1、3 个 host-compiled 节点 | 逐节点转化效率很差，是主瓶颈 |
| Canonical 权威 | 论文描述单一 RouteJSON | Director skeleton 共显示 5 个 replay-complete 步骤，但 lifecycle 中仅 3 个 Codex 候选 canonical admission=true，另 2 个因 `ancestor_or_target_cycle` 被拒 | 存在 RouteJSON 展示与 canonical graph 权威分裂，应视为代码/报告硬伤 |
| 短尾 | 每个战略未购买叶独立跑一次短搜索，完整解才 stitch | 共提交 11 次 AiZ native search；41 个 AiZ 部分步骤被接纳并递归产生新叶 | enhanced repair 可以这样做，但 paper-matched arm 不应把失败部分路线继续级联 |
| 库存 | 声明 39,684,411 entries | 39,478,827 unique full InChIKeys；差额已由重复/无效 eMolecules 行精确对账 | 成员集合语义可比，计数语义不同 |
| 终点 | 任一路线全叶命中 | 6 target-rooted、1 host-validated、0 stock-closed | 当前确实 unsolved，不是被条件/证据拖死 |

运行消耗为 48 次模型调用、843,053 input tokens、106,033 output tokens、2,473.328 s 模型 wall time，总 elapsed 2,966.556 s。折算到最终展示的 5 个逐节点步骤，约为每个保留步骤 168,611 input tokens 和 495 s 模型时间，效率明显不适合扩展到 1,098 目标。并行只能减少部分墙钟时间，不能修复这种低转化率。

候选统计也容易误读：`accepted_expansion_count=44` 并不代表 44 个 Codex 战略节点，其中 lifecycle 实际是 41 个 AiZ 候选和 3 个 Codex 候选被 canonical admission；另外 5 个候选被拒。报告应拆成 `accepted_llm_nodes`、`accepted_short_tail_edges` 和 `selected_route_nodes`。

运行最终为 terminal `unresolved`，原因是 `acceptance_not_met_and_no_registered_eligible_action`，而不是 token、时间或 AiZ 预算耗尽；native search 只使用 11/256。与此同时，报告顶层 `next_action` 仍保留一个 `eligible=true` 的旧 recompute action，与终态 stop decision 文义冲突。这是 closeout 投影的可读性缺陷，应把它标为 historical last action 或在终态清空。

## 14 学到的知识

- 与论文接近的关键不是换成 AiZynthFinder 这个名字，而是让 LLM 动作真正进入带替代分支和回溯的搜索树。
- ReactionJSON replay 解决的是“动作可执行”，RouteJSON/canonical graph 一致性解决的是“动作能否持续组成路线”；两者缺一不可。
- 短尾最有效的前提是战略层已经把叶显著降复杂；对失败的短尾部分路线连续级联可能增加 reach，但应作为增强消融，不应混入论文等价主臂。
- 条件、证据与宿主反应验证仍有论文之外的发表价值，但当前应在 B4 之后报告，不能遮蔽路线搜索本身的效果。
- 本次主要损失发生在“模型调用 → canonical 可继续节点 → selected RouteJSON”的转化漏斗，而不是库存 oracle 或 AiZ 参数缺失。

## 15 与 AutoPlanner 现有资产的连接

AutoPlanner 已具备论文没有明确提供的 canonical hypergraph、候选生命周期、独立 B2/B4、条件/证据分层以及酶步骤扩展。这些资产不应删除，但论文等价主臂应收敛为更短的热路径：

`3 StrategyCards → 每臂 OR-tree 逐节点 ReactionJSON → 宿主 replay → canonical admission → stock check → 每个原始开放叶一次 AiZ short-tail → solved route stitch → 可选六轮 Critic/Editor`

严格层应在这条热路径之后运行：

`B2 reaction validation → conditions → exact evidence → enzyme companion arm → experimental program`

这既保留项目优势，也避免工程审计再次压过最先需要展示的 reach。

## 16 研究设想

### 设想 A：论文匹配 OR-tree 主臂

- 创新状态：partially checked；ReactionJSON replay 已存在，真正按分支保留替代节点与回溯仍未在正式结果中证明。
- 核心假设：在相同 3×25 LLM policy 预算下，允许每个开放节点保留多个 canonical 合法动作、失败回溯到兄弟节点，会显著提高 `selected_route_nodes / model_calls` 和 B4。
- 如何验证：固定同一 8-target canary、模型、库存和 token 包络，对比 sequential top-1 与 OR/UCB 两臂；报告 LLM 节点接受率、最长连续深度、首次 B4 时间和 solved count。
- 可能失败：候选多样性不足；UCB 只在重复近似动作间分配预算；canonical 去重把有效立体异构动作错误合并。

### 设想 B：严格 paper-tail 与 recursive-tail 分臂

- 创新状态：unverified。
- 核心假设：论文匹配臂只 stitch 完整 AiZ 解可得到可解释基线；允许物化失败搜索的最佳部分路线并递归短尾，可能提高 reach，但增加调用并改变算法。
- 如何验证：每个原始战略叶只允许一次 6/500/1200 搜索，与 recursive partial-tail 最多六轮局部修复对照；分别报告每初始叶调用数、B4、路径长度和模板调用量。
- 可能失败：recursive arm 只把难叶向后推移、耗尽预算而无增益；严格 arm 样本量太小导致方差大。

### 设想 C：酶步骤作为独立增益臂

- 创新状态：partially checked；基础设施存在，但尚无与修复后化学主臂匹配的正式增益结论。
- 核心假设：在相同目标、库存和 LLM 节点预算下，酶战略可以用高选择性氧化还原、动力学拆分或级联替代若干化学保护/去保护和立体控制步骤，从而提高 B4 后的路线质量或缩短步骤。
- 如何验证：先冻结达到 B4 的化学主臂，再只改变 `enzyme_advantage` 战略提供者；报告 paper-equivalent B4、严格底物–产物酶证据、步骤节省和条件完整度，避免把无证据酶猜想计为增益。
- 可能失败：酶底物特异性证据不足；库存闭合不变；所谓步骤缩短只来自把多操作抽象成一个未验证生物转化。
