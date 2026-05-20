# AutoPlanner · 级联催化逆合成 PPT 全脚本

> **目标场次**：2026-05-20 评审 demo · 25 min 主讲 + 10 min Q&A
> **配套报告**：[AutoPlanner_Final_Report_2026-05-19.md](AutoPlanner_Final_Report_2026-05-19.md)
> **文件命名**：`AutoPlanner_2026-05-20_demo_v1.pptx`
> **总页数**：39 张正片 + 8 张 Backup
> **三大重点**（按用户指令）：① 级联催化逆合成的关键问题 · ② 各设计阶段及方案问题 · ③ 当前架构 + 未来架构 + 国际 SOTA 对照

---

## 全局排版规范

| 项 | 规范 |
| --- | --- |
| 模板色 | 主蓝 `#1F4E79` · 强调红 `#C00000`（仅用于重点数字）· 中性灰 `#595959` |
| 字体 | 中文 Source Han Sans / 英文 Arial · 标题 28pt · Takeaway 22pt · 正文 18pt · 图注 14pt |
| 单页结构 | 标题（顶）/ Takeaway（顶下方一行）/ 正文 bullets（左 60%）/ 图（右 40%）/ 念稿（PPT 备注） |
| 每页时长 | ≤ 45 秒；Live demo 与结论页各 +30 秒 |
| 资源目录 | `paper/figures/`、`results/shared/cascade_verifier_proof_20260519/`、`paper/scheme_route_01.pdf` |

---

## 全局目录（六大部分）

| 部分 | 页数 | 主题 | 时长 |
| --- | ---: | --- | ---: |
| 0 · 开场 | 3 | 封面 / 大纲 / 一页结论 | 2 min |
| 1 · 关键问题 | 6 | 级联催化逆合成为什么难 | 5 min |
| 2 · 国际 SOTA | 5 | 单步 / 多步 / 酶催化 / 国际 cascade 工作的差距 | 4 min |
| 3 · 设计阶段与问题 | 11 | Phase A→D 全部尝试与每一种方案的失败/边界 | 8 min |
| 4 · 当前架构 | 7 | Hybrid D + Verifier-first 飞轮 | 4 min |
| 5 · 未来架构 + 路线图 | 5 | 目标架构 / DPO / SOTA 接入 / 30·90 天计划 | 2 min |
| 6 · 收尾 + Live demo + Q&A | 2 | demo + Go/Kill + 边界 | 5 min |
| **正片合计** | **39** | | **30 min** |
| Backup | 8 | Q&A 召唤 | — |

---

# 第 0 部分 · 开场（3 张）

---

## Slide 01 · 封面

| 区块 | 内容 |
| --- | --- |
| 标题 | **AutoPlanner：化学–酶级联催化逆合成 Planner** |
| 副标题 | 面向多步级联反应的"可验证、可解释、可迭代"逆合成路线生成系统 |
| 作者 / 时间 / 单位 | 项目组 · 2026-05-20 |
| 图 | 系统总览缩略图（painter→ChemEnzy fill→scorer→verifier 飞轮），居中放大 |

**念稿**：今天汇报的是一个面向"级联催化"场景的逆合成 planner。和单步、单酶、单化学步的工作不同，我们处理的是一条反应链上多步连锁、化学–酶交替的整体路线设计。

---

## Slide 02 · 大纲

| 区块 | 内容 |
| --- | --- |
| Takeaway | 用 30 分钟讲清楚"难在哪 · 谁在做 · 我们怎么做 · 接下来怎么做" |
| 正文 | 1. 关键问题（5 min）<br/>2. 国际 SOTA Landscape（4 min）<br/>3. 我们走过的设计阶段（8 min）<br/>4. 当前架构 Hybrid D + Verifier-first（4 min）<br/>5. 目标架构 + 30/90 天路线图（2 min）<br/>6. Live Demo + 结论（5 min） |
| 图 | 横向 6 段时间线，对应上述 6 部分；下方一条"Q&A backup 8 页"提示条 |

**念稿**：先从问题本质讲起，再看国际 SOTA 现在能做到哪一步，然后看我们四代尝试的得失，给出当前可交付系统和未来主线，最后用一个 17 步级联案例收尾。

---

## Slide 03 · 一页结论（Executive Summary）

| 区块 | 内容 |
| --- | --- |
| Takeaway | **当前最强可交付系统：Hybrid D；最强未来主线：Verifier-first DPO 飞轮接入 frozen ChemEnzy** |
| 正文 |  · **问题**：级联催化逆合成是"路线池 × 顺序 × 条件包络 × 辅因子台账 × 库存闭合"五维联合约束<br/> · **数据**：v4 release 高质量 cascade 3,744 条 / 8,609 步，比通用单步少两个量级<br/> · **设计阶段**：4 个 Phase / 14 个子方案，其中 12 个被诚实证伪或冻结<br/> · **当前 SOTA（自家）**：Hybrid D · plan 1.00 · stock 0.93 · route GT 0.52（full100 cost-scoring 30s）<br/> · **下一里程碑**：Verifier-first DPO（30,556 perturb pair 已就绪，learned acc 0.91 / 8-类 macro-F1 0.97）|
| 图 | 4 行迷你 KPI 卡片：① 数据 8.6K → ② 设计 14/12 freeze → ③ 当前 0.93/0.52 → ④ 飞轮 30K pair |

**念稿**：一页话讲完整盘。我们最重要的判断不是"做出来一个强模型"，而是"承认这个问题不能用一个端到端大模型解决"，并把研究重心转到 verifier-first 这条飞轮上。

---

# 第 1 部分 · 关键问题（6 张）

> **重点一**：级联催化逆合成的关键问题（用户要求重点之一）

---

## Slide 04 · 什么是级联催化（定义 + 经典示例）

| 区块 | 内容 |
| --- | --- |
| Takeaway | 级联催化 = 一锅 / 一线 多个催化剂顺序作用，**中间体不分离纯化** |
| 正文 |  · 化学–化学级联（如多步金属催化）<br/> · 酶–酶级联（如多酶辅因子再生）<br/> · **化学–酶混合级联**（本项目焦点，最难也最重要）<br/> · 工业价值：原子经济、节能、缩短工艺，对应"绿色合成"国家战略<br/> · 典型 cascade 长度：3–5 步，每步换一个催化剂或酶 |
| 图 | 一条 4 步级联示例反应图（target 化合物 → 中间体 a → b → c → 试剂），下方标注每步催化剂/酶 + T/pH/溶剂 |

**念稿**：级联催化的核心物理特征是中间体不分离。这意味着前一步的"溶剂/pH/温度/抑制物"必须能被下一步催化剂容忍。这就是为什么不能把它当成 N 步独立反应来做。

---

## Slide 05 · 为什么"级联" ≠ "逐步逆合成"——7 个独特挑战

| 区块 | 内容 |
| --- | --- |
| Takeaway | 把单步模型串起来 ≠ 级联 planner，**会在 7 个维度集体崩盘** |
| 正文 |  1. **顺序约束**：催化剂 A 必须在 B 之前/之后<br/> 2. **条件包络（envelope）**：所有步的 T/pH/溶剂必须相互兼容<br/> 3. **辅因子台账（cofactor ledger）**：NAD+/NADH/ATP 等必须收支平衡<br/> 4. **酶毒性 / 抑制物链**：上一步产物不能毒死下一步酶<br/> 5. **原子守恒 × 立体化学**：cascade 内不能有 atom-balance gap<br/> 6. **中间体不分离**：禁止 silica column、禁止换溶剂等隐含约束<br/> 7. **库存闭合**：最终原料必须落在可商购的 building block |
| 图 | 7 宫格 icon，每格一个挑战，**红色高亮第 2/3/4** 三项（我们的 verifier 8 类规则正对应这些） |

**念稿**：这 7 条不是教科书定义，是我们一年踩过的坑。当中 2-3-4 三条是"级联专属"，单步模型完全不会涉及，这就是为什么 ChemEnzy / AiZynthFinder 直接用都不行。

---

## Slide 06 · 5 条硬约束的数学表述

| 区块 | 内容 |
| --- | --- |
| Takeaway | 级联路线必须**同时**满足 5 个约束方程，缺一不可 |
| 正文 |  $$\text{Route 合法} = \text{AtomBalance} \cap \text{Envelope} \cap \text{Cofactor} \cap \text{Order} \cap \text{StockClosure}$$<br/> · AtomBalance：每步 $\sum \text{原子}_\text{reactant} = \sum \text{原子}_\text{product}$<br/> · Envelope：$\bigcap_{i} [T_i^{\min}, T_i^{\max}] \cap [\text{pH}_i^{\min}, \text{pH}_i^{\max}] \neq \emptyset$<br/> · Cofactor：每个步骤的辅因子净流量在全链上 = 0（或有再生酶）<br/> · Order：拓扑序与各反应温度敏感性一致<br/> · StockClosure：所有叶节点 ∈ commercial building blocks |
| 图 | Venn 图 5 圈交集，中心一个绿色 `legal cascade`；右侧表格列每个约束对应的我们 8 类 verifier rule |

**念稿**：这页的关键信息是这五个约束**乘起来**，单步合规不蕴含 cascade 合规——这是我们把 verifier 拎出来当一等公民的原因。

---

## Slide 07 · 数据墙（cascade 数据为什么稀缺）

| 区块 | 内容 |
| --- | --- |
| Takeaway | cascade 高质量标注 **比通用单步少 2 个量级**，且条件标注更稀缺 |
| 正文 |  · USPTO-full：~1,000,000 反应 · Reaxys：> 50M · Pistachio：~3M<br/> · 公开 cascade 数据集**几乎不存在**（ECReact、RetroBioCat、ReactZyme 都是单步酶反应）<br/> · 我们自建 `cascade_dataset_v4_release/`：3,744 条 cascade / 8,609 步 / 三档 quality（gold 2,885 · silver 859 · bronze 60）<br/> · 这一规模 ↔ 模型容量上限：单一 monolithic 大模型必然欠拟合（参见 §3 v20 失败案）<br/> · → 策略只能是：**冻结强单步模型，让 cascade 数据训练 planner / verifier / scorer 这种"任务结构层"** |
| 图 | 对数纵轴柱状图：USPTO 1M · Pistachio 3M · Reaxys 50M · BioNavi 100K · cascade v4 8.6K（最右侧红色） |

**念稿**：这是整套架构选择的底层原因。3K 量级的 cascade 数据训不动一个端到端大模型，但足够训控制策略 / verifier / scorer 这些薄层。

---

## Slide 08 · 把通用 retro 模型直接用在 cascade 上的失败案

| 区块 | 内容 |
| --- | --- |
| Takeaway | AiZynthFinder + USPTO 直接跑 cascade：**plan 0.66 · GT@5 0.20**，几乎等于 random |
| 正文 |  · 案例：他汀类侧链 4 步级联（GT 已知）<br/> · 失败模式 1：第 2 步选了高温反应，但第 1 步在水相 → envelope 冲突<br/> · 失败模式 2：把酶催化步退化成"通用还原" → 丢掉立体选择性<br/> · 失败模式 3：建议引入 NADPH，但全链无再生酶 → cofactor 台账崩 <br/> · 失败模式 4：终点不闭合到 stock，给出"伪叶节点" |
| 图 | 案例 4 步反应图，**红色叉号标出 4 个失败点**；下方一行小字给出 USPTO 模型 top-5 与 GT 的逐步类型对比表 |

**念稿**：这页就是问题陈述的"实证锚点"。直接用通用 retro 模型不仅打不到 GT，连化学/物理常识都过不去。

---

## Slide 09 · 我们对"级联问题"的 4 个新定义

| 区块 | 内容 |
| --- | --- |
| Takeaway | 把 cascade planning 从"找到一条路"改写为"找到一条**可验证**通过 5 项约束的路" |
| 正文 |  1. **从 plan rate → 可验证 plan rate**：无 verifier 通过的路线不算<br/> 2. **从单尺 GT@5 → 双尺**：skeleton GT@K（结构）+ stock-closure（可制造性）独立汇报<br/> 3. **从 ranker-shaped → verifier-shaped**：把信号源从"哪个路线更好"改成"哪个路线为什么不行"<br/> 4. **从 monolithic learning → 任务分解 + 冻结强 prior**：painter 学骨架 / fill 用 ChemEnzy / scorer 学排序 / verifier 学拒绝 |
| 图 | 左旧右新两栏对照：旧 "Plan rate / GT@K / End-to-end" vs 新 "Verifier-pass rate / 双尺 / 任务分解" |

**念稿**：第 1 部分总结。这 4 个重新定义直接决定了后面架构的形态——尤其是"verifier-shaped 而不是 ranker-shaped"这一条。

---

# 第 2 部分 · 国际 SOTA Landscape（5 张）

---

## Slide 10 · SOTA 单步逆合成模型（USPTO-50K leaderboard）

| 区块 | 内容 |
| --- | --- |
| Takeaway | 单步 SOTA 已饱和在 **top-1 ≈ 55% / top-10 ≈ 84%**，没有量级突破 |
| 正文 | （表格） |
| 图 | 横向表，5 列：模型 / 年份 / 类型 / top-1 / top-10 |

**表内容**：

| 模型 | 年份 | 类型 | top-1 | top-10 |
| --- | --- | --- | ---: | ---: |
| GLN | 2019 | template logic | 52.5% | 79.1% |
| LocalRetro | 2021 | local template + GNN | 53.4% | 77.4% |
| Graph2Edits / G2Retro | 2022 | graph edits | 55.1% | 80.0% |
| Chemformer | 2022 | BART seq2seq 预训 | 53.6% | 80.0% |
| R-SMILES | 2023 | aug seq2seq | 56.3% | 80.5% |
| **RetroChimera (Chimera)** | 2024 | template + transformer **ensemble** | **≈ 56%** | **≈ 84%** |
| RetroDiff / DiffRetro | 2024 | 扩散生成 | 54% | 82% |

**念稿**：单步模型已经接近天花板，再卷 1-2 个点意义不大。我们的策略是**直接吃这些强模型**，而不是再训一个。

---

## Slide 11 · SOTA 多步 planner（搜索框架）

| 区块 | 内容 |
| --- | --- |
| Takeaway | 多步 planner 都假定"单步模型 + tree search"，**没有 cascade-stateful 设计** |
| 正文 | （表格） |
| 图 | 4 列对照表：planner / 搜索算法 / 单步 backbone / 是否 cascade-aware |

**表内容**：

| Planner | 搜索算法 | 单步 backbone | cascade 感知 |
| --- | --- | --- | :-: |
| AiZynthFinder (AZ) | MCTS | USPTO template MLP | ✗ |
| Retro\* (Chen 2020) | best-first A* | LocalRetro | ✗ |
| PDVN (Liu 2023) | proof-DAG value net | learned policy | ✗ |
| Spaya (Iktos) | proprietary | ensemble | ✗ |
| Syntheseus (MSR 2024) | 通用 search 框架 | 可插拔 | ✗ |
| **ChemEnzy** (Tu 2024) | retro-star + 7 backbone ensemble | template + ONMT | **部分**（酶 backbone 独立） |
| BioNavi (Zheng 2022) | MCTS + 酶 prior | 自训 ONMT | 部分 |
| **AutoPlanner (本工作)** | route-tree + bounded reservoir + verifier | 冻结 ChemEnzy 7 backbone | **完整** |

**念稿**：现有 planner 把 single-step + search 拼起来，但 cascade 的"条件包络/辅因子/顺序"这些约束**没有任何 planner 把它当一等公民**——这正是我们的差异化定位。

---

## Slide 12 · SOTA 酶催化逆合成（enzymatic）

| 区块 | 内容 |
| --- | --- |
| Takeaway | 酶 retro 都是**单步**工作，没有 cascade-aware 的酶路线 planner |
| 正文 |  · **RetroBioCat (2021)**：rule-based 酶模板库，~150 反应类型，单步<br/> · **EnzymeMap (Heid 2023)**：把酶反应 mapping 化，提供 ML-ready 数据，单步<br/> · **ReactZyme (2024)**：~300K 酶反应 SMILES 数据，单步分类 + 生成<br/> · **ECReact (Probst 2022)**：EC class → reaction SMILES，单步<br/> · **EnzyKR / Enzyformer**：ESM-2 编码酶 + seq2seq，单步<br/> · **结论**：酶 retro 整个领域**没有 multi-step / cascade planner**，AutoPlanner 是唯一系统化做这件事的 |
| 图 | 时间线（2019→2024），上方 6 个酶 retro 模型节点；下方一条粗线"cascade enzymatic planner（空白）"，我们的 logo 落在 2025/2026 节点 |

**念稿**：这一页传达"赛道空白"。我们不是和这些模型竞争，而是把它们作为 backbone 候选挂进我们的 fill 层。

---

## Slide 13 · 国际工作对 cascade 的处理（最相关的 4 篇）

| 区块 | 内容 |
| --- | --- |
| Takeaway | 业界对 cascade 的处理仍停留在**数据层 / 单步层**，没人做端到端 planner |
| 正文 | （4 项工作对照） |
| 图 | 4 行表：作者 / 方法 / cascade 处理深度 / 与我们的差距 |

**表内容**：

| 工作 | 方法 | cascade 处理 | 与我们的差距 |
| --- | --- | --- | --- |
| ChemEnzy (Tu 2024, *Nature Comm.*) | 7-backbone ensemble + retro-star | 在 backbone 维度区分化学/酶；搜索不感知 cascade | 我们以它为 fill 层基底 |
| BioNavi-NP (Zheng 2022) | 天然产物逆合成 MCTS | 不区分单步与级联 | 我们的 v4 cascade 数据完全跨入 |
| AlphaSynthesis (DeepMind WIP) | RL + chemistry foundation model | 闭源，未公开 cascade 评测 | 我们公开复现 + verifier |
| EnzymeFlow (2024 preprint) | flow-matching 生成酶反应 | 单步生成 | 不涉及 multi-step |

**念稿**：这页直接告诉评委——cascade-aware 端到端 planner 在公开文献里**还是空白**，我们的工作具有原创性。

---

## Slide 14 · 国际 SOTA 与我们的"能力矩阵"

| 区块 | 内容 |
| --- | --- |
| Takeaway | 我们独占的能力是 **cascade verifier + 双尺评测 + cascade-stateful route audit** |
| 正文 | （能力矩阵表） |
| 图 | 6×5 矩阵 ✓/✗：能力 × 系统 |

**表内容**：

| 能力 \ 系统 | AZ | Retro\* | ChemEnzy | Spaya | **AutoPlanner** |
| --- | :-: | :-: | :-: | :-: | :-: |
| 多步 search | ✓ | ✓ | ✓ | ✓ | ✓ |
| 化学–酶混合 backbone | ✗ | ✗ | ✓ | ✗ | ✓ |
| stock-closure 审计 | △ | △ | ✓ | ✓ | ✓ |
| cascade 顺序/包络约束 | ✗ | ✗ | △ | ✗ | **✓** |
| cofactor 台账 verifier | ✗ | ✗ | ✗ | ✗ | **✓** |
| 双尺评测（skeleton + stock） | ✗ | ✗ | ✗ | ✗ | **✓** |
| Verifier-first DPO 飞轮 | ✗ | ✗ | ✗ | ✗ | **✓**（proof 就绪）|

**念稿**：4 个 ✓ 在最右一列加粗——这些是我们的差异化护城河，全部都和"级联专属约束"相关。

---

# 第 3 部分 · 设计阶段与各方案问题（11 张）

> **重点二**：级联催化逆合成模型的设计阶段和各方案问题（用户要求重点之二）

---

## Slide 15 · 全部 4 个 Phase / 14 个子方案总览

| 区块 | 内容 |
| --- | --- |
| Takeaway | **14 个尝试 / 12 个证伪或冻结 / 2 个生还**——这是诚实研究路径，不是堆方案 |
| 正文 | （全景表）|
| 图 | 4 列垂直 Gantt：Phase A 单步生成器 / Phase B Planner / Phase C Skeleton / Phase D Ranker；右侧用绿/红/灰标注每个子方案"生还/证伪/冻结" |

**全景表**：

| Phase | # | 子方案 | 状态 |
| --- | --- | --- | --- |
| **A · 单步生成器** | A1 | EnzExpand 模板 MLP | 冻结（被 ChemEnzy 覆盖）|
| | A2 | Dual-tower DRFP↔ESM-2 | 证伪（contrastive 不生成）|
| | A3 | Enzyformer v2→v5 | **生还**（作为 fill 层之一）|
| | A4 | 冻结 ChemEnzy 7 backbone | **生还**（fill 层主干）|
| **B · Planner 架构** | B1 | CascadeBoardTransformer v20 monolithic 3.92M | 证伪（GT@5 24%，任务干扰）|
| | B2 | Particle + 规则能量 | 证伪（rule-based 太脆）|
| | B3 | two-stage search | 冻结（边界条件窄）|
| | B4 | cc_aostar AO\* 原型 | 冻结（LLM 重排序成本高）|
| | B5 | DESP bridge | 冻结（外部依赖）|
| | B6 | Reservoir distillation 学生 | 证伪（C 不超 baseline）|
| | B7 | Hybrid D = distill + bounded native reservoir | **生还**（**当前主线**）|
| **C · Skeleton-based** | C1 | OA-ARM Skeleton Inpainter | **生还**（最强 L1 painter，未挂入 Hybrid D）|
| **D · Ranker** | D1–D9 | 9 次 ranker / DPO 试验 | 8 证伪 / 1 生还（learned verifier）|

**念稿**：这张是 §3 的索引页。后面 10 张分别拉开 A / B / C / D 四个 Phase 讲清楚谁活下来谁死掉为什么。

---

## Slide 16 · Phase A · 单步生成器的 4 代

| 区块 | 内容 |
| --- | --- |
| Takeaway | A 代的核心教训：**不要在 3K 数据上自训单步生成器，去冻结业界已有的强模型** |
| 正文 |  · **A1 EnzExpand**（Morgan2-2048 → template-id MLP）：天花板被模板覆盖率限制，long-tail 反应全砸<br/> · **A2 Dual-tower**（DRFP↔ESM-2 对比学习）：能"匹配酶"但不能"生成 reactants"，contrastive ≠ generative<br/> · **A3 Enzyformer v2→v5**：seq2seq + ESM-2 condition，v5 在酶反应单步可达 top-10 = 0.71，但 cascade 内 enzymatic recall 仍不够<br/> · **A4 冻结 ChemEnzy 7 backbone**：直接吃 USPTO/Pistachio/Reaxys/BKMS 等大规模 prior，**节省 4 个月训练**<br/> · 结论：**A4 当 fill 层主干 + A3 当 enzymatic 候选**，A1/A2 冻结 |
| 图 | 4 代折线图：横轴时间，纵轴 enzymatic top-10 recall（A1 ~0.3 → A4 ~0.65） |

**念稿**：A 代最大的反思是——我们一开始想"重造单步轮子"，半年后才意识到应该直接吃 ChemEnzy 的 7 个 backbone。

---

## Slide 17 · Phase B · Planner 架构 7 个子方案

| 区块 | 内容 |
| --- | --- |
| Takeaway | B 代核心教训：**monolithic 端到端 planner 必败，路线必须分解到 search + 强 prior + scorer + verifier** |
| 正文 | （表） |
| 图 | 7 段横向 timeline，每段一个 mini KPI（plan / stock / route GT） |

**表内容**：

| 子方案 | 设计假设 | 实际结果 | 失败/边界原因 |
| --- | --- | --- | --- |
| B1 CascadeBoardTransformer v20 | 3.92M 单体多 head 端到端 | plan 1.00 但 GT@5 = 24%（≈ random 19%） | 任务干扰；3K 数据训不动 |
| B2 Particle + 规则能量 | 粒子滤波 + 手写能量项 | smoke 跑通，规模化能量失真 | rule-based 不可学 |
| B3 two-stage search | 化学 MCTS + 酶 expand | 部分 cascade 有效 | 切分点选择脆弱 |
| B4 cc_aostar | AO\* + LLM 重排序 | two-target 通过 | 推理慢、成本高 |
| B5 DESP bridge | 接 DESP 外部 planner | 接口跑通 | 外部依赖不稳定 |
| B6 Reservoir distillation (C) | 学 ChemEnzy native search | plan 0.76、stock 0.46 不超 baseline | student 没学到原生 route pool |
| **B7 Hybrid D** | distill + bounded ChemEnzy native reservoir + cost/rank + audit | **plan 1.00 · stock 0.93 · route GT 0.52** | 当前最强 |

**念稿**：6 个证伪、1 个生还。这条路径告诉我们："**学搜索"是错的；"用学习去引导强搜索 + 强 prior"才是对的**。这是后面所有架构判断的根。

---

## Slide 18 · Phase B 焦点 · CascadeBoardTransformer v20 为什么证伪

| 区块 | 内容 |
| --- | --- |
| Takeaway | v20 的 "plan rate 100%" 是**无弃权**指标的假象，真信号 GT@5 = 24% 几乎等于 random 19% |
| 正文 | （指标诚实表） |
| 图 | 4 行表：指标 / 真实含义 / v20 / OA-ARM |

**指标诚实表**：

| 指标 | 真实含义 | v20 | OA-ARM |
| --- | --- | ---: | ---: |
| Plan rate | 不弃权率（恒等于 1，无信号） | 1.00 | 0.99 |
| Random baseline | 无信息基线 | 0.19 | 0.19 |
| **GT@5 (real signal)** | 真实 top-5 命中 | **0.24** | **0.75** |
| Lift over random | 真实增量 | ×1.3 | **×3.9** |

**结构性失败原因**：① 任务干扰：edit policy + inpainting + real-label 3 head 在 3.92M 上争容量 · ② 重复造 ChemEnzy 的轮子 · ③ 缺"任务分解先验"。

**结论**：v20 = 论文 ablation baseline，**证伪了 monolithic 路线**。这条结论直接催生了 Phase C 的 OA-ARM。

**念稿**：必须把"plan rate 100%" 这个 misleading 指标讲清楚——它只是"系统不弃权"，没有任何能力信号。GT@5 才是真信号。

---

## Slide 19 · Phase B 焦点 · Hybrid D 为什么生还

| 区块 | 内容 |
| --- | --- |
| Takeaway | Hybrid D 的增益**几乎全部来自 bounded ChemEnzy native reservoir**，而不是 student 学到了什么 |
| 正文 |  · Stage 3 reservoir teacher：B 配置 stock 0.93、route GT 0.55（teacher 上界）<br/> · Stage 4 distill 学生 only（C）：stock 0.46 ≈ baseline，**学不到 native search 的路线池**<br/> · Stage 5 直接接入 native bounded reservoir（D）：stock 0.93、route GT 0.52，**接近 teacher**<br/> · 关键诚实：D 的胜出**不是"学 ChemEnzy 压成小模型"，而是"把 ChemEnzy 原生强 proposal 圈进我们级联约束 + audit 框架"**<br/> · cost/rank scoring 取代主观 reward；stock-closed alternative audit 把非 GT 合法路线归正 |
| 图 | 4 列柱状（A / B / C / D）× 3 指标（plan / stock / route GT），D 接近 B，C 接近 A，用箭头标"D 的增量来自 reservoir" |

**念稿**：这页是 Hybrid D 的"诚实账单"。我们没有击败 ChemEnzy，我们把 ChemEnzy 装进了 cascade-aware 的盒子里。

---

## Slide 20 · Phase C · OA-ARM Skeleton Inpainter（最强 painter，但不是替代品）

| 区块 | 内容 |
| --- | --- |
| Takeaway | OA-ARM = 当前最强 L1 skeleton painter，但**单独跑无法替代 ChemEnzy（自废武功）** |
| 正文 |  · 架构：6 层 Transformer decoder · d=256 · 8 head · 6.5M 参数<br/> · 训练：v4 3,810 cascade + 随机置换 → 50K 训练样本<br/> · 验证：rtype_acc 92.4% · ec1_acc 94.8% · T MAE 0.74°C<br/> · 100-target audited：**plan 99% · skeleton GT@5 75% · ×3.9 vs random**<br/> · **关键边界**：OA-ARM 只输出骨架（rxn type + EC + T + pH），**不生成分子、不做 stock 闭合**<br/> · → 正确定位：**L1 painter，提供 prior/mask 给 ChemEnzy 的 fill 层** |
| 图 | OA-ARM 架构图（左）+ 与 ChemEnzy 7 backbone 的能力对比表（右） |

**对比表**：

| 维度 | OA-ARM 单独 | ChemEnzy (Hybrid D) |
| --- | --- | --- |
| Backbone 数 | 1（6.5M） | 7（百万级训练）|
| 反应覆盖 | cascade 内 ~50 类 | Reaxys/Pistachio 全谱 |
| 分子级生成 | ✗ | ✓ |
| stock 闭合 | ✗ | 0.93 |
| 外部 benchmark | 0 | PaRoutes / USPTO-190 / BioNavi |

**念稿**：必须把这页讲清楚——OA-ARM 是 painter，不是 planner。"用 OA-ARM 替代 ChemEnzy" = 自废武功。

---

## Slide 21 · Phase D · 9 次 Ranker 试错

| 区块 | 内容 |
| --- | --- |
| Takeaway | 9 个 ranker / DPO / contrastive 试验 / **8 证伪 / 1 生还（learned verifier）**——彻底证伪 "ranker-shaped 信号 + 小数据" |
| 正文 | （9 行表） |
| 图 | 9 行小表，最后一行 learned verifier 绿色高亮 |

**9 次试错**：

| # | 方案 | 信号源 | 数据量 | 结果 |
| --- | --- | --- | --- | --- |
| D1 | LightGBM LambdaRank（reranker_v2） | GT pair-wise | 8K | 边界微涨，无 lift |
| D2 | Listwise NDCG XGBoost | top-k 排序 | 8K | 无显著增益 |
| D3 | ESM-based dual-tower rerank | 酶相似度 | 3K | OOD 完全失效 |
| D4 | Route-level scorer (Transformer 0.9M) | route_score | v3 路线 | 训出来但 calibration 差 |
| D5 | DPO on routes | preference pair | 1.2K | 没有可靠 pair label |
| D6 | Contrastive route encoder | InfoNCE | 4K | overfit 严重 |
| D7 | Energy-based scorer | rule | 0 | 不可学 |
| D8 | RL on plan rate | sparse reward | — | reward hacking |
| **D9** | **Learned Cascade Verifier** | **8 类失败原因 + 30K perturb pair** | **30K** | **✓ acc 0.91 / F1 0.97** |

**念稿**：8 次失败的统一原因是 **ranker-shaped 信号在 3K cascade 上学不出来**——pair label 噪声、preference 不可靠、reward 稀疏。D9 把信号从"哪个更好"换成"为什么不行"，立刻就成立。

---

## Slide 22 · 失败模式根因诊断

| 区块 | 内容 |
| --- | --- |
| Takeaway | 12 个证伪方案归一到 2 个根因：**任务结构不对** + **信号源不对** |
| 正文 |  · **根因 1 任务结构错误**（A1/A2/B1/B2/C 学单步等）<br/> 　→ 试图让一个模型同时学"骨架 + 分子 + 条件 + 排序"<br/> 　→ 修正：**任务分解**（painter / fill / scorer / verifier 四层）<br/> · **根因 2 信号源错误**（D1–D8）<br/> 　→ ranker-shaped 信号需要大量 preference pair，cascade 数据不够<br/> 　→ 修正：**verifier-shaped 信号**——用 8 类失败规则 + perturbation 生成无限 pair<br/> · 修正后存活方案：A3 / A4 / B7 / C1 / D9 |
| 图 | 双根因鱼骨图：左根因 → 失败方案集合；右修正 → 存活方案集合 |

**念稿**：这一页是 §3 的总结。我们用 14 次尝试换来了两个根因诊断，这两个诊断直接定义了下一阶段（§4 / §5）的所有动作。

---

## Slide 23 · 范式迁移：从 ranker-shaped 到 verifier-shaped

| 区块 | 内容 |
| --- | --- |
| Takeaway | **核心范式迁移**：不再问"哪条路线更好"，而问"哪条路线为什么不合法" |
| 正文 |  · **Ranker-shaped**（旧）：需要 preference pair (A>B)，要人标，数据少，pair 主观<br/> · **Verifier-shaped**（新）：需要 fail label (route, reason)，可由 8 条规则自动生成无限<br/> · 数据放大：8.6K cascade → 通过 perturbation 生成 **30,556 pair**<br/> · 信号密度：每条 route 同时给 0/1 + 8 类原因 vector<br/> · 工程收益：DPO pair 自动产生，无需人标 preference |
| 图 | 左右对照：左 ranker-shaped（人标 pair, 数据 1.2K）vs 右 verifier-shaped（规则生成 pair, 数据 30K），中间一个大箭头"信号源换轨" |

**念稿**：这是整个项目最重要的范式迁移。它把"标注瓶颈"换成了"规则工程"——而规则工程可以由化学家直接介入，这才是可持续路径。

---

## Slide 24 · 8 类 Cascade Verifier 规则定义

| 区块 | 内容 |
| --- | --- |
| Takeaway | 8 条规则覆盖 cascade 的 5 项硬约束，**learned verifier macro-F1 = 0.9653** |
| 正文 | （表） |
| 图 | 8 行表 + 8x8 confusion matrix 热图（右） |

**8 类规则**：

| # | 规则 ID | 检测内容 | 对应硬约束 |
| --- | --- | --- | --- |
| 1 | atom_balance_violation | 原子数不守恒 | AtomBalance |
| 2 | product_mismatch | 产物 SMILES 与下一步底物不一致 | Order |
| 3 | temperature_conflict | T 包络冲突 | Envelope |
| 4 | ph_conflict | pH 包络冲突 | Envelope |
| 5 | solvent_conflict | 溶剂不兼容 | Envelope |
| 6 | enzyme_toxicity | 上一步产物毒害下一步酶 | Envelope |
| 7 | cofactor_ledger_gap | 辅因子收支不平衡 | Cofactor |
| 8 | route_order_mismatch | 拓扑顺序与温敏性冲突 | Order |

**念稿**：8 条规则不是拍脑袋——是 cascade 5 项硬约束的工程化表达。每一条都对应 verifier 一个 head + DPO 时的一个 firewall。

---

## Slide 25 · Phase D 唯一生还：Learned Verifier 证据

| 区块 | 内容 |
| --- | --- |
| Takeaway | Learned verifier 在 30K perturb pair 上 **acc 0.9094 · macro-F1 0.9653**，在 592 条真实路线上 **label acc 1.0000** |
| 正文 |  · 模型：DictVectorizer + LogisticRegression · 51 维特征<br/> · 训练数据：30,556 perturbation pair（per-route 自动生成 + 规则标签）<br/> · 规则一致性：rule acc = 0.9964<br/> · 真实数据零样本：592 条真实 cascade，label acc = **1.0000**<br/> · DPO readiness：29,079 pair 已就绪，可直接做 ChemEnzy adapter |
| 图 | 8×8 reason confusion matrix 热图（行真实 / 列预测） + 右侧 KPI 卡（acc / F1 / 真实零样本 acc / DPO pair 数） |

**念稿**：这是 §3 最后的"生还者证据页"。learned verifier 不仅在合成 perturb 上准，在真实路线上零样本 acc = 1.0，意味着规则覆盖足够正交。

---

# 第 4 部分 · 当前架构 Hybrid D + Verifier-first（7 张）

> **重点三**：级联催化逆合成模型目前设计架构（用户要求重点之三）

---

## Slide 26 · 当前系统总览（Hybrid D + Verifier 飞轮）

| 区块 | 内容 |
| --- | --- |
| Takeaway | 4 层管线 + 1 个飞轮：**Frozen Backbones → Search → Score/Audit → Verifier → DPO 回流** |
| 正文 |  · **L1 Frozen Backbones**：ChemEnzy 7 backbone + Enzyformer v4（fill 层）<br/> · **L2 Cascade-aware Search**：AutoPlanner route-tree + source/budget gate + bounded native reservoir<br/> · **L3 Score & Audit**：cost/rank scoring + stock-closed alternative audit + quality filter<br/> · **L4 Cascade Verifier**：8 类规则 + learned head + 真实数据零样本 acc 1.0<br/> · **🌀 Flywheel**：verifier 标注 → DPO pair → 反哺 ChemEnzy（防火墙隔离 GT∈pool / 可恢复 / 不可达 三桶）|
| 图 | 系统架构图（横版）：左 user target → L1 7 backbone → L2 search → L3 scorer/audit → L4 verifier → 右 DPO pair → 飞轮箭头回到 L1 |

**念稿**：这张是整个系统的"骨架图"。后面 6 张分别拉开 L1/L2/L3/L4/飞轮，每层一页。

---

## Slide 27 · L1 · Frozen ChemEnzy 7 Backbone

| 区块 | 内容 |
| --- | --- |
| Takeaway | 不重训单步、不蒸馏单步，**直接吃 7 个百万级训练的强 prior** |
| 正文 | （表）|
| 图 | 7 行 backbone 表 + 一行 Enzyformer v4（自家酶 backbone） |

**Backbone 表**：

| 插槽 | Backbone | 训练数据 | 角色 |
| --- | --- | --- | --- |
| `onmt_models` | bionav_one_step | USPTO-NPL + BioChem | 天然产物通用 |
| `template_relevance` | USPTO-full_remapped | USPTO 全谱 | 通用化学 |
| `template_relevance` | pistachio | Pistachio | 商业化学 |
| `template_relevance` | pistachio_ringbreaker | Pistachio ring | 杂环 |
| `template_relevance` | reaxys | Reaxys | 商业化学 |
| `template_relevance` | reaxys_biocatalysis | Reaxys 酶 | 酶催化 |
| `template_relevance` | bkms_metabolic | BKMS | 代谢通路 |
| `+` Enzyformer v4 | ESM-2 + seq2seq | v4 酶反应 | cascade-酶补充 |

**念稿**：这一层我们一行代码都没动 ChemEnzy。我们的贡献从 L2 开始。

---

## Slide 28 · L2 · Cascade-aware Search

| 区块 | 内容 |
| --- | --- |
| Takeaway | route-tree + source/budget gate + bounded native reservoir = 让强 backbone 在 cascade 约束下展开 |
| 正文 |  · `cascade_planner/route_tree/search.py`：自家 route-tree 展开<br/> · `source_gate.py`：source/budget policy，控制每个 backbone 调用预算<br/> · `bounded_reservoir.py`：ChemEnzy native broad route 池 bounded top-k 回放<br/> · `reservoir_distilled.py`：distilled controller 给出 soft priority（不强制裁剪，失败回退 heuristic）<br/> · cascade-aware fixed fields：在 search 内强制 order / envelope mask |
| 图 | route-tree 展开示意（带 source-gate 和 reservoir merge）；右下角标 D 配置 stock 0.93 |

**念稿**：L2 的关键不是"我们写了一个 search"，而是"我们让 search 能感知 cascade 约束"。这就是 Hybrid D 比单独 ChemEnzy 多出来的部分。

---

## Slide 29 · L3 · Score & Audit

| 区块 | 内容 |
| --- | --- |
| Takeaway | 取代主观 reward，用 **cost/rank scoring + stock-closed alternative audit + quality filter** |
| 正文 |  · 旧版主观 reward：`exact +1.00 / GT reactant +0.55 / stock +0.45` → 误导 "GT 是唯一正确路线"<br/> · 新版 cost/rank（ChemEnzy/MolStar 风格）：route cost + stock closure + reaction plausibility + 低进展/无效路线惩罚<br/> · **Stock-closed alternative audit**（`audit_stock_closed_alternatives.py`）：把非 GT 但 stock 闭合的路线区分为 plausible / needs review / weak / suspicious / invalid<br/> · D_FILTER quality filter：把审计通过率从 0.857 → 0.976，suspicious 从 0.143 → 0.024 |
| 图 | 旧 reward → 新 cost/rank 的对照表；下方一张审计五分类饼图（D vs D_FILTER） |

**念稿**：这里要强调一个研究观念修正——逆合成不是单答案任务，GT 只是参考召回，stock-closed 的合法替代必须用 audit 而不是直接判错。

---

## Slide 30 · L4 · Cascade Verifier（8 类规则 + learned）

| 区块 | 内容 |
| --- | --- |
| Takeaway | Verifier = 系统的"刹车 + 教练" · 既拒绝坏路线，又为 DPO 提供高质量 pair |
| 正文 |  · 8 类规则定义（见 Slide 24）<br/> · learned verifier：DictVectorizer + LogisticRegression，51 维<br/> · 性能：rule acc 0.9964 / learned acc 0.9094 / reason macro-F1 0.9653 / 真实零样本 acc 1.0000<br/> · 工件目录：`results/shared/cascade_verifier_proof_20260519/`<br/> · 接入：与 L2 search 并行——search 出 candidate → verifier 标注 → audit 修订 → 输出 route + 失败原因向量 |
| 图 | Verifier 流程图：candidate route → 8 规则 → fail vector → learned head → final accept/reject + reason；右侧 KPI 卡（4 个指标） |

**念稿**：verifier 是整个未来飞轮的支点。它现在能做 acc 0.91 + reason F1 0.97，30K pair 已经产出，DPO 就剩接口对接。

---

## Slide 31 · 飞轮 · Verifier → DPO → 反哺 ChemEnzy（含防火墙）

| 区块 | 内容 |
| --- | --- |
| Takeaway | Verifier-first 飞轮 = 我们最有把握的下一里程碑，**proof 全就绪**，差最后一公里 adapter manifest |
| 正文 |  · Step 1 routes → verifier 8 规则标注（自动）<br/> · Step 2 perturbation 产生失败变体 → 30,556 pair<br/> · Step 3 **三桶防火墙**（关键）：<br/> 　- 桶 A：GT ∈ proposal pool → 正常 DPO<br/> 　- 桶 B：GT 可 retrieval 召回 → 先抬召回再 DPO<br/> 　- 桶 C：GT 不可达 → **不进 DPO**（防止 model 学到 "pool top = correct"）<br/> · Step 4 DPO 后冻结 ChemEnzy adapter，回灌 L2 search<br/> · ChemEnzy 状态：`ready_for_supervised_adapter_manifest_not_direct_dpo`，LoRA / 损失尚未对接 |
| 图 | 飞轮图（四阶段循环）+ 三桶分流示意 + 防火墙伪代码 5 行 |

**念稿**：这是整个项目的下一里程碑。防火墙这块必须讲——没有它，DPO 会学到错误的捷径。

---

## Slide 32 · 当前性能 KPI（full100 + 4 个外部基准）

| 区块 | 内容 |
| --- | --- |
| Takeaway | Hybrid D 在 5 个 benchmark 上一致显示**接近 teacher 上界**，外部数据集证据完整 |
| 正文 | （表） |
| 图 | 5 行表 × 4 列指标，最右一列"和 A baseline 的 lift" |

**性能表**：

| Benchmark | 配置 | plan | stock | route GT | s/target |
| --- | --- | ---: | ---: | ---: | ---: |
| full100 (30s gate) | A baseline | 0.76 | 0.46 | 0.39 | 3.0 |
| full100 (30s gate) | **D hybrid** | **1.00** | **0.93** | **0.52** | 3.3 |
| PaRoutes n1 (30) | D | 1.00 | 0.87 | 0.77 | 33 |
| PaRoutes n5 (30) | D | 1.00 | 0.73 | 0.80 | 28 |
| USPTO-190 full | D | 0.96 | 0.76 | 0.57 | 37 |
| BioNavi-like (373) | D | 0.99 | 0.81 | 0.36 | 41 |

**念稿**：Hybrid D 不是只在自家 benchmark 强，4 个外部数据集都跑过——这是发表级证据的雏形。

---

# 第 5 部分 · 未来架构 + 路线图（5 张）

---

## Slide 33 · 目标架构：Painter + Frozen Fill + Scorer + Verifier 四层

| 区块 | 内容 |
| --- | --- |
| Takeaway | 把 OA-ARM 接入 Hybrid D 的 fill 输入端做 prior，**而不是替代 ChemEnzy** |
| 正文 |  · **L1 Painter (新)**：OA-ARM 6.5M Transformer，输入 target，输出 K=5 候选骨架（rxn type + EC + T + pH）<br/> · **L1.5 Prior → mask**：骨架转成 fill 层的 hard/soft mask（type mask、enzyme class mask、condition envelope mask）<br/> · **L2 Fill (frozen)**：ChemEnzy 7 backbone + Enzyformer v4，在 mask 下生成 SMILES 候选<br/> · **L3 Scorer (4-layer Transformer 0.9M)**：route-level multi-task ranking<br/> · **L4 Verifier (DPO-ready)**：8 类规则 + learned head + DPO pair 反哺 ChemEnzy<br/> · 这套架构在 §6 已写入正式 ADR |
| 图 | 目标架构总图（4 层 + 1 飞轮）；右侧标注每层的输入/输出 schema |

**念稿**：这是我们对评委的"未来主线一图"。OA-ARM 不消失，而是上升为 L1 painter，给 ChemEnzy 提供 cascade-aware prior。

---

## Slide 34 · 接入新 SOTA Backbone 计划（3 个优先级）

| 区块 | 内容 |
| --- | --- |
| Takeaway | ChemEnzy 接口可插拔，**3 个 P0/P1 候选 backbone 1 个月内可上线** |
| 正文 | （表） |
| 图 | 3 行候选 × 4 列（类型 / 状态 / 报价指标 / 预计增量） |

**接入候选**：

| # | Backbone | 类型 | 仓库现状 | 预计 ChemEnzy 增量 |
| --- | --- | --- | --- | --- |
| **P0-1** | **RetroChimera (Chimera 2024)** | template + transformer ensemble | 权重已在 `data_external/retrochimera_model/` | top-10 pool +5~8pt · stock +2~4pt |
| **P0-2** | **Chemformer (AstraZeneca BART)** | 预训 seq2seq | 公开 checkpoint 可下 | OOD 目标 +10pt 量级 |
| **P0-3** | **RetroBioCat / ReactZyme retro head** | 酶模板 + 酶 seq2seq | 数据已下 `data_external/retrobiocat/` `reactzyme/` | enzymatic GT-in-pool +10~20pt |

**接入步骤**：① 写 `predict(target, topk)` wrapper → ② 注册到 `one_step_model_configs` → ③ 加进 `selected_one_step_opt` → ④ Recall@K bucket A/B/C 复测

**念稿**：单步层我们不再自研，**而是把业界 SOTA 接进来**。3 个候选中 RetroChimera 是已经在仓里的，最容易上。

---

## Slide 35 · 30 天路线图

| 区块 | 内容 |
| --- | --- |
| Takeaway | 5 个 milestone，**每个都有 Go/Kill 准则**，不允许"做着看" |
| 正文 | （Gantt） |
| 图 | 5 行 Gantt（横轴 Day 1→30），每行标 Owner + 验收 KPI |

**30 天 milestone**：

| # | 内容 | Owner | 验收 KPI | Kill 准则 |
| --- | --- | --- | --- | --- |
| M1 | RetroChimera 挂入 ChemEnzy onmt_models | 工程 | Recall@10 bucket A +5pt | <2pt 撤回 |
| M2 | Recall@K bucket A/B/C 诊断脚本 `scripts/recall_ceiling_v4_frozen.py` | 工程 | 三桶比例稳定输出 | 跑不通撤回 |
| M3 | Verifier → DPO pair 接入 ChemEnzy adapter manifest | 训练 | 29K pair 可序列化注入 | manifest schema 失败撤回 |
| M4 | OA-ARM 输出 prior mask 注入 ChemEnzy fill | 工程 | full100 GT route +3pt | <1pt 撤回 |
| M5 | full100 + 4 外部基准回归测试 | 评测 | 全部 ≥ 当前数字 | 任一下滑 >2pt 撤回 |

**念稿**：每个 milestone 一周节奏，前 4 个独立可并行，M5 收尾验证。

---

## Slide 36 · 90 天路线图

| 区块 | 内容 |
| --- | --- |
| Takeaway | 90 天目标 = **第一篇 cascade verifier-first paper 投稿可读稿** |
| 正文 | 6 个 milestone |
| 图 | 6 节点时间线（Week 4 / 6 / 8 / 10 / 12 / 13） |

**90 天 milestone**：

| Week | 内容 |
| --- | --- |
| W4 | RetroChimera + Chemformer 双 backbone 上线，外部基准回归 |
| W6 | RetroBioCat / ReactZyme 酶 backbone 上线，enzymatic recall +10pt |
| W8 | DPO 首轮 close loop，verifier-pass rate 收敛 |
| W10 | OA-ARM painter 全 pipeline 联调，full100 GT route ≥ 0.60 |
| W12 | Paper §1–§6 草稿 + 全部 figure 定稿 |
| W13 | 投稿可读稿（目标：JACS Au / Nature Comm. / Chem. Sci.）|

**念稿**：90 天目标只有一个——把项目变成可投稿的故事。

---

## Slide 37 · Go / Kill 准则（避免做着看）

| 区块 | 内容 |
| --- | --- |
| Takeaway | 每个方向都有**单一数值门**和**回滚版本**，避免技术债 |
| 正文 |  · 单步 backbone：Recall@10 bucket A 不涨 2pt → 撤回<br/> · DPO：verifier-pass rate 不涨 5pt → 回滚到 supervised adapter<br/> · OA-ARM 接入：full100 GT route 不涨 3pt → 退回 prior-only 模式<br/> · 任意外部基准回归下滑 >2pt → 24h 内回滚<br/> · 所有 freeze 决策写入 `docs/ADR/` |
| 图 | 决策树（菱形节点 + 单一数值门），右下角"回滚 = 12h 内可执行" |

**念稿**：这一页给评委传达"我们有纪律不烧时间"。

---

# 第 6 部分 · 收尾（2 张正片）

---

## Slide 38 · Live Demo · 17 步级联路线

| 区块 | 内容 |
| --- | --- |
| Takeaway | 用一张 17 步级联路线图证明系统不是 PPT-only |
| 正文 |  · Target：他汀类侧链关键中间体（hash `3764f7`）<br/> · 系统：Hybrid D + Enzyformer v4<br/> · 输出：`paper/scheme_route_01.pdf`（17 步连续顺合成路线图）<br/> · 命中：stock 闭合 ✓ / 4 步酶催化 ✓ / 条件包络通过 ✓ / verifier 8 类规则 0 警告 |
| 图 | 整页插入 `paper/scheme_route_01.pdf` 全图（横版）；下方一行小字注明 demo 命令：<br/>`python -m cascade_planner.multistep.plan_route --target "<SMILES>" --enable-cascade-source-policy` |

**念稿**：这页念稿很短——"图自己说话"。如果时间允许，现场敲一次命令重生成。

---

## Slide 39 · 结论 + 可以说 / 不能说 / 暂停说 边界

| 区块 | 内容 |
| --- | --- |
| Takeaway | **可以说**：Hybrid D + verifier proof · **不能说**：DPO 已 close loop · **暂停说**：单步全面 SOTA |
| 正文 | （三分屏）|
| 图 | 三分屏 ✓ / ⚠ / ⏸ |

**三分屏**：

| ✓ 可以说 | ⚠ 暂停说 | ⏸ 不能说 |
| --- | --- | --- |
| Hybrid D = 当前 cascade 最强可交付系统 | 单步层全面超过 SOTA（我们用了 SOTA backbone）| DPO 已 close loop（manifest 未对接）|
| 4 个外部基准 stock/GT 一致提升 | OA-ARM 已上线为 painter（30 天目标）| Cascade Verifier 解决所有 cascade 问题（8 类规则只覆盖 5 项硬约束）|
| Cascade Verifier 8 规则 + learned acc 0.91 / F1 0.97 / 真实零样本 1.0 | 30K DPO pair 已就绪可灌 | 自研单步模型超过 RetroChimera（我们冻结它）|
| Cascade-aware planner 在公开文献是空白 | 90 天可投稿 | 已在 JACS / Nature 接收（未投）|

**念稿**：最后一页要说的就是边界——评委最反感的是 "what cannot you do you don't say"。我们主动给。

---

# Backup Slides（Q&A 召唤，8 张）

| # | 标题 | 召唤场景 |
| --- | --- | --- |
| B1 | OA-ARM 消融表（无 painter / w/ painter） | "OA-ARM 真的有用吗" |
| B2 | Verifier 8 规则完整定义 + 失败案例 | "verifier 怎么写的" |
| B3 | v4 release tier 详细统计（gold/silver/bronze） | "数据质量怎么保证" |
| B4 | DPO pair 构造算法 + 三桶防火墙伪代码 | "DPO 怎么防止 shortcut" |
| B5 | Live demo target full trace（17 步） | "随便挑一步讲讲" |
| B6 | ChemEnzy 7 backbone 调用细节 + 接口签名 | "怎么和 ChemEnzy 对接" |
| B7 | 14 个子方案完整失败原因表 | "为什么不试 X" |
| B8 | 国际 SOTA leaderboard 完整对照（30 篇 reference） | "和 X 论文比怎么样" |

---

# 念稿节奏（25 min 主讲建议）

| 段 | 页数 | 时长 | 重点提醒 |
| --- | --- | --- | --- |
| 开场 | 1–3 | 2 min | 一页结论一定要念全 |
| 关键问题 | 4–9 | 5 min | Slide 7 数据墙 + Slide 8 失败案放慢 |
| SOTA Landscape | 10–14 | 4 min | Slide 14 能力矩阵停 30s 让评委看 |
| 设计阶段 | 15–25 | 8 min | Slide 17 / 18 / 22 / 23 是高密度，必须停 |
| 当前架构 | 26–32 | 4 min | Slide 31 飞轮 + Slide 32 KPI 表是重头 |
| 未来 + 路线图 | 33–37 | 2 min | Slide 35 Gantt 一带而过即可 |
| Live demo + 结论 | 38–39 | 5 min | Slide 38 留 90s 给现场命令 |

---

# 资源对照（PPT 制作时取用）

| 内容 | 文件 |
| --- | --- |
| Final Report（数据源） | [docs/AutoPlanner_Final_Report_2026-05-19.md](AutoPlanner_Final_Report_2026-05-19.md) |
| 17 步级联路线图 | `paper/scheme_route_01.pdf` |
| Verifier proof 工件 | `results/shared/cascade_verifier_proof_20260519/` |
| Hybrid D 性能 | `results/shared/phase2_20260515/full100_abcd_gate30/reports/comparison.md` |
| 外部基准 | `results/shared/phase2_20260515/external_*` |
| 数据集 v4 release | `dataset_v4_release/manifest.json` |
| OA-ARM 训练入口 | `cascade_planner/cascadeboard/skeleton_inpainter.py` |
| ChemEnzy 接口 | `vendor/ChemEnzyRetroPlanner/retro_planner/common/prepare_utils.py` |

---

**END · 39 张正片 + 8 张 Backup · 30 min · 2026-05-20 demo**
