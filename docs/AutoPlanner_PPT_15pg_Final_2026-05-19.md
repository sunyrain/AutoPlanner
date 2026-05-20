# 生物级联催化 520 · 15 页定稿脚本

> **配套 pptx**：`生物级联催化520.pptx`（你已有的 10 页草稿）
> **本脚本作用**：在你 10 页基础上 → 收束精华至 **15 页定稿**
> **变更映射**：保留 6 页 + 重写 1 页 + 扩展 1 页 + 新增 5 页 + 致谢
> **总时长**：≤ 20 min 主讲 + 10 min Q&A
> **配色**：主蓝 `#1F4E79` · 强调红 `#C00000` · 中性灰 `#595959`
> **字号**：标题 28pt / Takeaway 22pt / 正文 18pt / 图注 14pt

---

## 变更映射表（你的 10 页 → 定稿 15 页）

| 定稿 # | 标题 | 来源 | 动作 |
| :-: | --- | --- | --- |
| 1 | 封面 | 原 #1 | ✅ 保留 |
| 2 | 研究背景：为什么要做化学–酶级联 | 原 #2 | ✅ 保留 + 微调 |
| 3 | 级联反应的 4 大难点 | 原 #3 | ✅ 保留 |
| 4 | AI 在级联催化中的定位 | 原 #4 | ✅ 保留 |
| 5 | **SOTA 单步/多步逆合成（截至 2026.05）** | 原 #5 | 🔁 **重写**（leaderboard 更新到 2026） |
| 6 | SOTA 化学–酶混合逆合成 | 原 #6 | ✅ 保留 + 补一行国际工作 |
| 7 | **关键问题**（重点 ①） | 原 #7 | 📈 **扩展**（加 5 硬约束 + 数据墙） |
| 8 | **架构设计思路：组合 + 协同，而非重起炉灶**（重点 ② -a） | 原 #8 占位 | 🆕 **新增** |
| 9 | **三个核心架构创新：外部提案池外挂化 · cascade 硬约束嵌入搜索 · 质量主导的排序**（重点 ② -b） | 原 #8 占位 | 🆕 **新增** |
| 10 | **质量评估层：从合理性打分到可审查的违例报告**（重点 ② -c） | 原 #8 占位 | 🆕 **新增** |
| 11 | **当前架构总览：Hybrid D + Verifier 飞轮**（重点 ③ -a） | 原 #9 占位 | 🆕 **新增** |
| 12 | **架构 4 层细节 + DPO 三桶防火墙**（重点 ③ -b） | 原 #9 占位 | 🆕 **新增** |
| 13 | **性能：full100 + 4 个外部基准** | — | 🆕 **新增** |
| 14 | **未来主线 + 30/90 天路线图** | — | 🆕 **新增** |
| 15 | 谢谢 | 原 #10 | ✅ 保留 |

---

# Slide 01 · 封面 ✅ 保留

```
标题   AI 引导的化学—生物级联催化设计与协同优化
副标题  AI-guided Chemo-Enzymatic Cascade Design & Co-optimization
日期   2026.05.20
图     右下角放系统总览缩略图（painter→ChemEnzy fill→scorer→verifier 飞轮）
```

**念稿**：今天汇报"AI 引导的化学—生物级联催化"，重点回答三件事：级联设计的关键问题、我们走过的设计阶段、当前架构与未来主线。

---

# Slide 02 · 研究背景 ✅ 保留（微调）

**标题**：研究背景与问题提出：化学催化—酶催化级联反应

**Takeaway**（**新加一行**）：化学+酶级联 ≠ "化学一步 + 生物一步"，而是**跨范式协同体系**

| 左栏 化学催化优势 | 右栏 生物催化优势 |
| --- | --- |
| · 构建复杂骨架 | · 高水平立体选择性 |
| · 引入非天然反应活性 | · 区域选择性与化学选择性 |
| · 高效完成酶难以承担的转化 | · 后期精细修饰 + 高附加值转化 |

**级联后的潜在收益**（**保留 5 条**）：
- 减少保护/脱保护 · 缩短路线 · 提高总收率与总选择性 · 降低能耗与分离成本 · 让高反应性与高选择性在同一时空兼容、接力、闭环

**图**：中央一张化学–酶级联示意图（Target ← step₄(酶) ← step₃(化学) ← step₂(酶) ← step₁(化学)，箭头标 T/pH/溶剂）

**念稿**：化学催化擅长建骨架、做难反应；酶催化擅长做精细立体选择。两者级联本质上是想把"高反应性"和"高选择性"放进同一个反应体系——但前提是它们的工作环境必须能相互容忍。

---

# Slide 03 · 4 大难点 ✅ 保留

**标题**：化学催化—酶催化级联反应的难点

**Takeaway**：**真正瓶颈不是"串起来"，而是"协同起来"**

| 难点 | 具体表现 |
| --- | --- |
| ① 条件不匹配 | 化学催化常需有机溶剂、高温、金属或自由基；酶偏好温和水相 |
| ② 中间体不稳定 | 难分离、易分解、易副反应 |
| ③ 催化剂相互失活 | 金属/自由基/氧化还原失活酶；生物体系毒化化学催化剂 |
| ④ 时空组织困难 | one-pot / 分步 / 流动体系 / 空间隔离的取舍 |
| ⑤ 实验搜索空间巨大 | 路线 × 酶 × 催化剂 × 顺序 × 加料 × T/pH/溶剂 全耦合 |

**图**：4 宫格 icon，每格一个难点；下方一条粗线"→ 级联系统优化的对象不是单一步骤，而是整个体系"

**念稿**：这 4 条是化学家级联实验里反复踩的坑。它们决定了——级联 planner 不能只看单步反应，而必须看整条路线作为一个整体是否自洽。

---

# Slide 04 · AI 的定位 ✅ 保留

**标题**：AI 在级联催化系统中的定位

**Takeaway**：把"路线—中间体—催化者—条件"的联合搜索，**转化为可预测、可排序、可迭代优化的问题**

| AI 任务 | 回答的问题 |
| --- | --- |
| 路线选择 | 哪条路线最值得尝试 |
| 接口筛选 | 哪个中间体最适合作上下游连接点 |
| 催化匹配 | 哪种酶 / 化学催化剂更可能兼容 |
| 工艺优化 | 多目标条件下如何整体最优 |

**图**：五菱形围绕中心"AI 设计器 + 决策辅助器"：Routes / Intermediates / Enzymes / Catalysts / Conditions

**念稿**：AI 在这里不是替代化学家，而是把化学家原本要做几百次试错的"联合搜索"，变成一个可排序、可迭代、可解释的决策辅助流程。

---

# Slide 05 · 🔁 重写 · SOTA 单步 / 多步逆合成（截至 2026.05）

> **原 #5 内容是 Slide 4 的复制粘贴，需要替换**

**标题**：目前主流逆合成模型概述：单步 / 多步逆合成（截至 2026.05）

**Takeaway**：**单步**已饱和（top-1 ≈ 56–58%、top-10 ≈ 84–87%），**多步**进入 ensemble + 价值网络阶段；**化学–酶 cascade-aware planner 仍属公开文献空白**

**单步模型（USPTO-50K leaderboard，纯化学单步基准）**：

| 模型 | 年份 | 类型 | top-1 | top-10 |
| --- | --- | --- | ---: | ---: |
| LocalRetro | 2021 | local template + GNN | 53.4% | 77.4% |
| Chemformer (AstraZeneca) | 2022 | BART seq2seq 预训 | 53.6% | 80.0% |
| Graph2Edits / G2Retro | 2022 | graph edits | 55.1% | 80.0% |
| R-SMILES | 2023 | aug seq2seq | 56.3% | 80.5% |
| **Chimera (RetroChimera)** | 2024–2025 | template + transformer **ensemble** | **≈ 56%** | **≈ 84%** |
| RetroExplainer / DirectMultiStep 类衍生 | 2025–2026 | dual-task + explanation | **57–58%** | **86–87%** |

> 说明：USPTO-50K 是十年公开基准；2024 年后单步层主要进步来自 **ensemble + 解释性**，2026 年没有量级跃升——这正是我们"冻结业界 SOTA 而不重训"的依据。

**多步 planner（截至 2026）**：

| Planner | 年份 | 搜索 | 单步 backbone | cascade 感知 |
| --- | --- | --- | --- | :-: |
| AiZynthFinder | 2020 | MCTS | USPTO template | ✗ |
| Retro\* | 2020 | A\* best-first | LocalRetro | ✗ |
| PDVN | 2024 | proof-DAG value net | learned | ✗ |
| **Syntheseus** (Microsoft) | 2024–2025 | 通用 search 框架 | 可插拔 | ✗ |
| DESP / RetroGFN 类 | 2024–2025 | GFlowNet / 双向 | learned | ✗ |
| **ChemEnzy** (Tu *Nat. Commun.* 16:10929, 2025) | 2025 | retro-star + 7 backbone | template + ONMT | **部分** |
| **AutoPlanner（本工作）** | 2026 | route-tree + cascade-aware audit | 冻结 SOTA × 8 | **✓ 5 硬约束** |

**结论**：
- 单步层：**直接吃业界 SOTA 而不是再训**——重训只能拿 1–2 pt 增量，工程性价比极低；
- 多步层：截至 2026.05 仍无任何公开 planner 把 **化学–酶 cascade 5 硬约束**作为搜索一等公民——**这是我们的差异化定位**。

**图**：左单步 leaderboard 条形图（2021→2026 缓增曲线，标注"已饱和"）/ 右多步 planner 表格（cascade 列除最后一行 ✓ 外全 ✗，警示色）

**念稿**：这页两个信息——①单步层从 2024 到 2026 涨幅微小，我们冻结业界 SOTA；②多步 planner 截至本月没有任何公开工作做 cascade-aware，这是我们的定位。

---

# Slide 06 · ✅ 保留（补一行国际工作）· SOTA 化学–酶混合逆合成

**标题**：目前主流逆合成模型概述：有机/酶混合逆合成模型

**Takeaway**：化学反应与酶促转化已开始被纳入统一 planning 框架（**ChemEnzy 2025 是当前最强公开系统**）

**ChemEnzy 三层架构**（保留原文）：

| 层级 | 任务 |
| --- | --- |
| 层级一：路线搜索 | 目标分子怎样拆回到原料 · 为路线评估打分 |
| 层级二：单步决策 | 判断每一步是否需要酶催化 · 推荐反应条件 · 预测反应可行性 |
| 层级三：酶催化增强 | 判断酶适配性 · 酶种类推荐 · 酶活性位点验证 |

**参考**：Tu, Z. et al. *A virtual platform for automated hybrid organic-enzymatic synthesis planning.* **Nat. Commun.** 16, 10929 (2025).

**关键缺口**（**新加，原 #6 没有**）：ChemEnzy 在 backbone 维度区分化学/酶，但 search 不感知 cascade 的"条件包络/辅因子台账/顺序约束"——**这正是 AutoPlanner 的切入点**。

**图**：ChemEnzy 三层架构图（保留你原图）；右下角加一个红色框"AutoPlanner 在 search + verifier 层补完 cascade 约束"

**念稿**：ChemEnzy 是当前最强的化学-酶混合 planner，但它把化学与酶**当成两类 backbone 并列**，没有把 cascade 当成搜索时一等公民的约束。我们的工作不是和它竞争，而是把它装进 cascade-aware 的盒子。

---

# Slide 07 · 📈 扩展 · 关键问题（**重点 ①**）

> 原 #7 只有 2 行表层/真问题陈述，扩展为 5 硬约束 + 数据墙

**标题**：级联催化逆合成模型的关键问题

**Takeaway**：级联路线合法性 = **5 个硬约束同时成立**；可信度评估 = **可解释的失败原因**

**表层 vs 真问题**（保留你原文）：
- **表层问题**：给定复杂目标，能否生成"化学合理 + 酶兼容 + 条件可执行"的 cascade 路线？
- **真问题**：长 chemo-enzymatic cascade 的**可信度评估、可解释性、可干预性**

**5 个硬约束**（**新加，关键页核心**）：

$$\text{Route 合法} = \text{AtomBalance} \cap \text{Envelope} \cap \text{Cofactor} \cap \text{Order} \cap \text{StockClosure}$$

| # | 约束 | 含义 |
| :-: | --- | --- |
| 1 | **AtomBalance** | 每步原子守恒、立体化学不丢 |
| 2 | **Envelope** | 所有步 T / pH / 溶剂的交集非空 |
| 3 | **Cofactor** | NAD+/NADH/ATP 等辅因子全链收支平衡 |
| 4 | **Order** | 拓扑序与温敏性 / 失活链兼容 |
| 5 | **StockClosure** | 所有叶节点 ∈ 商购 building blocks |

**数据墙**（**新加，决定后续架构选择**）：USPTO 1M / Reaxys 50M / Pistachio 3M  vs  公开 cascade 数据**几乎为零**；我们自建 v4 release = **3,744 cascade / 8,609 step**——这一规模**禁止**训练 monolithic 大模型，**只能**训 painter / scorer / verifier 这种薄层。

**图**：左 Venn 5 圈（5 约束交集，中心 "legal cascade"）/ 右对数纵轴柱状（USPTO 1M、Reaxys 50M、Pistachio 3M、cascade v4 8.6K 红色高亮）

**念稿**：这页有两个新增信息要讲透——①5 约束乘起来不是加起来，单步合规不蕴含 cascade 合规；②cascade 数据只有 8.6K，决定了我们必须任务分解、不能 monolithic，这是后面所有架构选择的底层原因。

---

# Slide 08 · 🆕 新增 · 架构设计思路（**重点 ② -a**）

**标题**：架构设计思路：在数据约束下选择"组合 + 协同"，而非"重起炉灶"

**Takeaway**：公开级联数据规模与化学单步数据相差约**三个数量级**，这一客观条件直接决定了我们的架构选择——**以业界最强单步逆合成模型为冻结底座，在其上叠加 cascade-感知的约束搜索与可审查的质量评估层**。

### 三条架构主线 · 三条选择依据

| 架构主线 | 实际含义 | 选择依据 |
| --- | --- | --- |
| **① 不重训单步生成器**，直接复用业界成熟模型 | 将 ChemEnzy（Tu *Nat. Commun.* 2025）、Chimera、Chemformer 等 8 个外部主干作为冻结提议池 | USPTO-50K 单步 leaderboard 2024–2026 仅提升 1–2 pt，重训不可能带来量级增益，工程性价比极低 |
| **② 任务分解为多个薄层模块**串联，不依赖单体大模型 | painter / fill / scorer / verifier 四档，各取所需样本量与监督信号 | 公开级联样本仅 8.6 K（自建 v4），表示能力须与任务复杂度匹配，避免欠拟合伪装为高 plan-rate |
| **③ 质量评估以化学可审查的规则为主**，机器学习头作为补充 | 先由 8 条化学规则输出违例标签，再由轻量学习头预测被拒原因 | 与化学家判断逻辑同构，输出可作为 deviation report，支撑后期人机协同与反馈闭环 |

### 架构演进路径（一图概览）

```
    [起点]                                          [现代主线]
  单体端到端    →    多模块串联    →    冻结 SOTA 提议 +
                                          cascade-感知约束搜索 +
                                          可审查质量评估
                                          + 可启动反馈环
```

该架构与现代化工过程控制的逻辑同构：不重新发明反应器，而是在现有反应单元上叠加 **PAT（Process Analytical Technology）** 与 **QbD（Quality-by-Design）** 控制层。

**图**：左侧三行设计主线卡（不重训 / 任务分解 / 规则主导）；右侧架构演进示意图，起点"单体端到端"（灰色）· 中途"多模块串联"（橙色）· 现代主线"冻结 SOTA + 约束搜索 + 质量评估 + 反馈环"（蓝色高亮）

**念稿**：这一页传达的是设计决策背后的客观约束——级联公开数据只有八千多条，比化学单步数据少近三个数量级，这决定了我们不适合训一个什么都学的大模型。我们的选择是把业界最强的单步逆合成模型冻结作为提议者，在其上叠加 cascade-感知的约束搜索与化学家可逐条审查的质量评估，这是后面所有架构细节的总纲。

---

# Slide 09 · 🆕 新增 · 三个核心架构创新（**重点 ② -b**）

**标题**：三个核心架构创新：外部提议池外挂化 · 级联硬约束嵌入搜索 · 质量主导的排序

**Takeaway**：三个相互联动的模块设计，使系统在**路线成率、商购闭合率、与文献路线命中率**三个维度上同时逼近 teacher 上界（无算力限制的 ChemEnzy 原生表现）。

### 创新 ① · Bounded External Reservoir（外部强提议池外挂化）

- 将 ChemEnzy 的 7 个外部主干 + 自研 Enzyformer v4 全部封为**冻结提议者**，多步搜索时按给定预算调用、不训、不改其参数；
- 使架构可以**以低成本接入业界新发表的 SOTA 单步模型**（如近一年的 Chimera、Chemformer 升级版），无需重训任何单步模型，文献追踪成本仅在适配层。

### 创新 ② · Cascade-Aware Search（级联 5 硬约束作为搜索一等公民）

在 route-tree 展开过程中，**同时强制检查 5 个 cascade 专有约束**——原子守恒、T / pH / 溶剂包络、辅因子台账、拓扑顺序、商购闭合；并按 backbone 预算闸与 cascade fixed-fields mask 控制候选生成。

这是与 AiZynthFinder / Retro\* / Syntheseus / ChemEnzy 等公开 planner 的**最本质区别**：现有 planner 在路线生成完成后才对反应合理性打分，本工作把 cascade 约束**嵌入搜索过程本身**。

### 创新 ③ · Quality-led Scoring + Stock-Closed Audit（质量主导的排序）

- 综合 route cost / stock closure / reaction plausibility / 低进展惩罚，替代主观 reward；
- 对非 GT 合法路线补以 stock-closed alternative audit，将审计通过率从 0.857 提升至 **0.976**。

### 广谱验证 · 三维度同时逼近 teacher 上界

| 配置 | 路线成率 | 商购闭合率 | 与文献路线命中率 |
| --- | ---: | ---: | ---: |
| baseline（仅 route-tree） | 0.76 | 0.46 | 0.39 |
| **本架构** | **1.00** | **0.93** | **0.52** |
| teacher 上界（原生 ChemEnzy + 无预算限制） | 1.00 | 0.93 | 0.55 |

→ 以 0.93 商购闭合率达到 teacher 上界，以 0.52 文献命中率达到上界的 95%；同时推理预算**保持在 30 s/target 量级**。

**图**：三列横版架构卡（外挂外部 SOTA · 约束嵌入搜索 · 质量主导排序），底部 KPI 表格"本架构 vs teacher 上界"对比。

**念稿**：三个核心创新联动为一：把业界最强的单步逆合成模型外挂作为提议者；把级联五个硬约束提升为搜索过程本身的一等公民；把主观打分替换为可审查的质量评估。三者协同使外部可复现的商购闭合率与文献路线命中率同时逼近教师上界。

---

# Slide 10 · 🆕 新增 · 质量评估层：从合理性打分到可审查的违例报告（**重点 ② -c**）

**标题**：质量评估层：从"合理性打分"升级为"可审查的违例报告"

**Takeaway**：路线评估输出形式从**单一总分**升级为**逐步、逐项、按规则的违例向量**——一份与化学家 ELN 中 deviation 字段同构、可逐条核对、可直接修改的 QA 表。

### 8 条审查规则 · 对应化学家最关心的 5 个维度

| 化学家关心的维度 | 实际检查项 | 对应硬约束 |
| --- | --- | --- |
| 原子守恒与立体保持 | atom\_balance\_violation | AtomBalance |
| T / pH / 溶剂的跨步包络 | temperature\_conflict · ph\_conflict · solvent\_conflict · enzyme\_toxicity | Envelope |
| 辅因子台账闭合 | cofactor\_ledger\_gap | Cofactor |
| 拆原顺序与产物匹配 | route\_order\_mismatch · product\_mismatch | Order |
| 商购闭合 | （由上游 stock-closed audit 覆盖） | StockClosure |

### 评估层输出的信息对比

| | **常规 planner 的评估** | **本架构的评估** |
| --- | --- | --- |
| 输出形式 | 一个逆合成合理性总分 | 一个总分 + **一份 8 维违例向量**，定位到步 |
| 依据 | 隐藏在权重中 | **化学可表达的规则类** + 轻量学习头补充 |
| 化学家介入方式 | 重新标注后微调 | **直接修改审查规则，立即生效** |
| 验证指标 | 不适用 | 规则类 acc 0.9964 · 学习头 acc 0.9094 · 失败原因 macro-F1 0.9653 · 外部真实路线零样本 acc 1.000 |

### 与现代化学工作流的接口

- 违例向量可直接写入 ELN / LIMS 的 deviation 字段，与现有 batch record / QbD 评审表同构；
- 任一违例可作为下一轮搜索的硬约束（什么可变 / 什么不可变），使人机协同形成可追溯的闭环。

**图**：左侧表格"5 维度 × 8 规则"；右侧示例输出截图——一条级联路线 + 总分 0.71 + 违例报告（如 Step 3 envelope conflict：T=85 °C 超出上下游酶耐受 28–42 °C）。

**念稿**：常规逆合成 planner 输出一个路线合理性总分，化学家既难验证也难干预。本架构把评估输出升级为一份 8 维违例报告，与 ELN 或 QbD 评审表同构、可被逐项核对、可被直接修改。这是 AI 辅助级联设计进入实验室闭环的技术前提。

---

# Slide 11 · 🆕 新增 · 当前架构总览（**重点 ③ -a**）

**标题**：级联催化逆合成模型目前设计架构：Hybrid D + Verifier-first 飞轮

**Takeaway**：**4 层管线 + 1 个飞轮**——Frozen Backbones → Cascade-aware Search → Score/Audit → Verifier → DPO 反哺

```
┌────────────────────────────────────────────────────────────────┐
│ User Target SMILES                                              │
│            ↓                                                    │
│ L1  Frozen Backbones   ← 7 × ChemEnzy + Enzyformer v4           │
│            ↓                                                    │
│ L2  Cascade-aware Search                                        │
│         route-tree + source/budget gate + bounded native        │
│         reservoir + cascade fixed-fields                        │
│            ↓                                                    │
│ L3  Score & Audit                                               │
│         cost/rank scoring + stock-closed alternative audit      │
│            ↓                                                    │
│ L4  Cascade Verifier   ← 8 类规则 + learned head                 │
│            ↓                                                    │
│  Output route + 失败原因向量                                     │
│            │                                                    │
│            └→ 🌀 DPO pair (三桶防火墙) → 反哺 ChemEnzy           │
└────────────────────────────────────────────────────────────────┘
```

**4 层职责**：

| 层 | 模块 | 关键设计 |
| --- | --- | --- |
| L1 | Frozen ChemEnzy 7 backbone + Enzyformer v4 | **不重训单步**，直接吃百万级 prior |
| L2 | AutoPlanner route-tree + bounded reservoir | 把 cascade 约束作为搜索一等公民 |
| L3 | cost/rank + stock-closed audit | 取代主观 reward；非 GT 合法路线归正 |
| L4 | 8 规则 + learned verifier | 系统的"刹车 + 教练" |
| 🌀 | Verifier → DPO pair → ChemEnzy | 30K pair 已就绪，等 adapter manifest |

**图**：横版 4 层 + 飞轮架构图（左 input → 4 层 → 右 output；底部一条飞轮箭头回到 L1）

**念稿**：这是当前系统的骨架。下一页拆开讲每一层和 DPO 三桶防火墙。

---

# Slide 12 · 🆕 新增 · 4 层细节 + DPO 三桶防火墙（**重点 ③ -b**）

**标题**：4 层关键技术细节 + DPO 三桶防火墙

**Takeaway**：**Hybrid D 的胜出 = 把 ChemEnzy 原生强能力装进 cascade 约束的盒子**；DPO 飞轮 = 下一里程碑

### L1 · Frozen Backbones（7 + 1）

| 插槽 | Backbone | 训练数据 |
| --- | --- | --- |
| onmt_models | bionav_one_step | USPTO-NPL + BioChem |
| template_relevance | USPTO-full / Pistachio / Pistachio-RB / Reaxys / Reaxys-Biocat / BKMS | 数百万级 |
| 自研 | Enzyformer v4 | v4 酶反应 |

### L2 · Cascade-aware Search（关键创新）

- bounded ChemEnzy native reservoir（top-k 回放）
- source/budget gate（控制每个 backbone 预算）
- cascade fixed-fields（在 search 内强制 order / envelope mask）

### L3 · Score & Audit

- 新版 cost/rank（route cost + stock closure + reaction plausibility + 低进展/无效路线惩罚）
- stock-closed alternative audit（5 分类：plausible / needs review / weak / suspicious / invalid）
- quality filter 把审计通过率从 0.857 → **0.976**

### L4 · Cascade Verifier

- 8 类规则 + 51-维 learned head（DictVectorizer + LogisticRegression）
- 真实数据零样本 **acc 1.0000**

### 🌀 DPO 三桶防火墙（关键）

```
桶 A  GT ∈ proposal pool       → 正常 DPO（teach ranker）
桶 B  GT retrieval 可恢复        → 先抬召回，再 DPO
桶 C  GT 不可达                → 不进 DPO（防 "pool top = correct" shortcut）
```

**图**：左 4 层细节卡片（每层一个色块）+ 右 DPO 三桶分流图（带防火墙阀门 icon）

**念稿**：必须把三桶防火墙讲清楚——没有它，DPO 会学到错误的捷径"在 pool 里排第一的就是对的"。这是我们 verifier-first 飞轮安全收尾的关键工程动作。

---

# Slide 13 · 🆕 新增 · 性能：full100 + 4 个外部基准

**标题**：当前系统性能：5 个 benchmark 一致显示接近 teacher 上界

**Takeaway**：**Hybrid D 不止在自家 benchmark 强，4 个外部数据集都跑过**——这是发表级证据雏形

| Benchmark | 配置 | plan | **stock** | **route GT** | s/target |
| --- | --- | ---: | ---: | ---: | ---: |
| **full100** (30s gate, 自家 cascade) | A baseline | 0.76 | 0.46 | 0.39 | 3.0 |
| | **D hybrid** | **1.00** | **0.93** | **0.52** | 3.3 |
| **PaRoutes n1** (30) | D | 1.00 | 0.87 | **0.77** | 33 |
| **PaRoutes n5** (30) | D | 1.00 | 0.73 | **0.80** | 28 |
| **USPTO-190** full | D | 0.96 | 0.76 | 0.57 | 37 |
| **BioNavi-like** (373) | D | 0.99 | 0.81 | 0.36 | 41 |

**Verifier proof**（30,556 perturb pair）：
- rule acc **0.9964** · learned acc **0.9094** · reason macro-F1 **0.9653** · 真实零样本 acc **1.0000**

**图**：左 5 行性能表（D 行红字加粗）+ 右 4 个 KPI 卡（Verifier 4 指标）

**念稿**：这一页给评委传达"不是 PPT-only"。Hybrid D 已经在 4 个外部数据集上验证，Verifier 的 30K pair 也已经就绪，差最后一公里 DPO adapter 对接。

---

# Slide 14 · 🆕 新增 · 未来主线 + 30/90 天路线图

**标题**：未来架构与路线图

**Takeaway**：**OA-ARM 上升为 L1 painter（不替代 ChemEnzy）+ 接入 3 个 SOTA backbone + DPO 闭环**

### 未来 5 层架构

```
L1  Painter        OA-ARM Skeleton Inpainter (6.5M)   ← rxn type/EC/T/pH 骨架先验
L1.5 Prior→Mask    骨架转为 fill 层 hard/soft mask
L2  Fill (frozen)  ChemEnzy 7 backbone + Enzyformer v4 + 3 个新 SOTA
L3  Scorer         4-layer Transformer 0.9M  route-level multi-task
L4  Verifier+DPO   8 规则 + learned + 飞轮反哺 ChemEnzy
```

### 3 个 SOTA backbone 接入计划

| 优先级 | Backbone | 类型 | 预计 ChemEnzy 增量 |
| :-: | --- | --- | --- |
| P0-1 | **RetroChimera (Chimera 2024)** | template + transformer ensemble | top-10 +5~8pt · stock +2~4pt |
| P0-2 | **Chemformer (AstraZeneca BART)** | 预训 seq2seq | OOD 目标 +10pt 量级 |
| P0-3 | **RetroBioCat / ReactZyme** | 酶模板 + 酶 seq2seq | enzymatic recall +10~20pt |

### 30 天 / 90 天路线图

| Week | 里程碑 | Kill 准则 |
| :-: | --- | --- |
| W1–2 | RetroChimera 挂入 ChemEnzy + Recall@K bucket A/B/C 诊断 | Recall@10 <2pt 撤回 |
| W3–4 | DPO 接入 ChemEnzy adapter manifest（29K pair）| manifest schema 失败撤回 |
| W5–6 | OA-ARM prior mask 注入 ChemEnzy fill | full100 GT route <1pt 撤回 |
| W8 | DPO 首轮 close loop，verifier-pass rate 收敛 | <5pt 回滚 supervised |
| W10 | full100 GT route ≥ **0.60**（当前 0.52）| — |
| W12 | Paper §1–6 草稿 + figure 定稿 | — |
| W13 | 投稿可读稿（JACS Au / Nat. Commun. / Chem. Sci.）| — |

**图**：左未来架构 5 层图 + 右 30/90 天横向 Gantt

**念稿**：未来主线一图——OA-ARM 不消失而是上升为 L1 painter，给 ChemEnzy 提供 cascade-aware prior；下一阶段同时进 3 个 SOTA backbone 和 DPO 闭环。90 天目标是把项目变成可投稿的故事。

---

# Slide 15 · 致谢 ✅ 保留

```
标题   谢谢，请批评指正
日期   2026.5.20
图     可在右下角放系统总览缩略图 + 仓库地址（如可公开）
```

**念稿**：感谢评委。Q&A 环节我准备了 backup 页可以应对：① OA-ARM 消融 ② Verifier 8 规则细节 ③ v4 数据 tier 统计 ④ DPO 三桶伪代码 ⑤ Live demo 17 步全 trace。

---

# Q&A Backup（不进正片，pptx 末尾隐藏页）

| # | 主题 | 召唤场景 |
| :-: | --- | --- |
| B1 | OA-ARM vs Hybrid D 能力矩阵（自废武功页）| "为什么不用 OA-ARM 替代 ChemEnzy" |
| B2 | 8 类 Verifier 规则完整定义 + confusion matrix | "verifier 8 类怎么定" |
| B3 | v4 release tier 统计（gold/silver/bronze）+ 数据墙 | "数据质量怎么保证" |
| B4 | DPO 三桶分流 + 防火墙伪代码 | "DPO 怎么防 shortcut" |
| B5 | 17 步级联路线 full trace（`paper/scheme_route_01.pdf`）| "现场跑一个" |

---

# 念稿节奏（20 min 主讲）

| 段 | 页数 | 时长 | 重点提醒 |
| --- | --- | --- | --- |
| 开场 + 背景 | 1–4 | 4 min | Slide 3 难点列全，Slide 4 五菱形 |
| SOTA | 5–6 | 3 min | Slide 5 单步表 + 多步 cascade 列全 ✗ |
| **关键问题（重点 ①）** | 7 | 2 min | 5 硬约束方程 + 数据墙 |
| **设计阶段 + 证伪/生还 + 范式迁移（重点 ②）** | 8–10 | 5 min | Slide 9 v20 plan rate 100% 假象必须讲透 |
| **当前架构 + 4 层细节（重点 ③）** | 11–12 | 4 min | Slide 12 三桶防火墙停 30s |
| 性能 + 未来 | 13–14 | 2 min | Slide 13 D 行加粗 / Slide 14 Gantt 带过 |
| 致谢 | 15 | — | — |

---

# 配套资源（PPT 制图时取用）

| 内容 | 文件 |
| --- | --- |
| Final Report（数据源）| [docs/AutoPlanner_Final_Report_2026-05-19.md](AutoPlanner_Final_Report_2026-05-19.md) |
| 39 页完整脚本（细节版）| [docs/AutoPlanner_PPT_Full_Script_2026-05-19.md](AutoPlanner_PPT_Full_Script_2026-05-19.md) |
| 17 步级联路线图 | `paper/scheme_route_01.pdf` |
| Verifier proof 工件 | `results/shared/cascade_verifier_proof_20260519/` |
| Hybrid D 性能 | `results/shared/phase2_20260515/full100_abcd_gate30/reports/comparison.md` |
| 外部基准 | `results/shared/phase2_20260515/external_*` |
| v4 数据 | `dataset_v4_release/manifest.json` |
| ChemEnzy 接口 | `vendor/ChemEnzyRetroPlanner/retro_planner/common/prepare_utils.py` |
| 参考文献 | Tu, Z. et al. *Nat. Commun.* 16, 10929 (2025). |

---

**END · 15 张定稿 · 20 min · 2026-05-20 demo**
