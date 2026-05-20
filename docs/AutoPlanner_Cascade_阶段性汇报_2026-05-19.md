# AutoPlanner-Cascade 阶段性汇报材料

日期：2026-05-19

用途：面向化学信息学、合成化学、酶催化和自动规划方向专家的 PPT 汇报底稿。

---

## 一页摘要

我们这轮工作的核心不是简单证明“模型能不能生成路线”，而是把问题推进到更实际的一层：

> 对复杂 statin-like 目标分子，系统能生成大量长路线；真正的问题是如何判断这些长路线是否具有化学意义、哪些只是搜索伪影、哪些值得专家继续看。

本轮关键结论：

| 项目 | 结果 |
| --- | ---: |
| 原始路线数 | 906 |
| 原始唯一路线签名 | 393 |
| 原始 post-filter 保留路线 | 640 |
| 当前规则重新审计后 `triage_fragment` | 360 |
| 当前规则重新审计后 `needs_chemist_review` | 232 |
| 当前规则重新审计后 `reject_artifact` | 48 |
| 条件审计 `high` | 193 |
| 条件审计 `warn` | 447 |

本轮最重要的认知修正：

1. **大 terminal 不一定是高级产物片段。**  
   例如 Wittig / HWE 试剂本身很大，但其中大量原子只是载体或离去部分，真正引入产物的可能只是一个小片段。

2. **反应物 SMILES 中没有出现的元素，不一定是凭空出现。**  
   很多路线记录把 `POCl3`、卤化剂、氧化剂、还原剂、酸碱、盐等放在 condition/reagent 字段里，而不是放进 reaction SMILES 的 reactants 侧。

3. **因此路线审计不能只看 heavy atom count。**  
   更合理的框架应该是：显式反应物 + 条件试剂 + 反应角色 + 原子贡献。

4. **条件预测不能等同于已验证工艺条件。**  
   目前图中的温度、溶剂、试剂来自逐步条件预测模型。它们可用于提示“可能需要低温、强碱、加热偶联或分步操作”，但不能直接作为级联一锅条件或实验方案。

本轮新增产物：

| 产物 | 路径 | 说明 |
| --- | --- | --- |
| 当前规则重新审计 JSON | `results/v2/ui_chem_enzy_plan_20260519_032819_3764f7_reaudited_current.json` | 640 条路线的最新审计字段 |
| 论文式顺合成路径图索引 | `results/v2/route_schemes_3764f7_top10_current/index.html` | top10 paper-style synthesis scheme |
| 论文式顺合成路径图 PDF | `results/v2/route_schemes_3764f7_top10_current/scheme_route_01.pdf` 至 `scheme_route_10.pdf` | 推荐用于 PPT/专家汇报 |
| 论文式顺合成路径图 SVG | `results/v2/route_schemes_3764f7_top10_current/scheme_route_01.svg` 至 `scheme_route_10.svg` | 矢量图，可继续精修 |
| 连续路线树索引 | `results/v2/route_trees_3764f7_top10_current/index.html` | route topology 检查 |
| 路线树渲染脚本 | `scripts/render_route_trees.py` | Graphviz + RDKit 路线图 |
| 顺合成 scheme 渲染脚本 | `scripts/render_linear_route_schemes.py` | RDKit + 自定义路线级排版 |
| 路线池重新审计脚本 | `scripts/reaudit_route_pool.py` | 规则更新后刷新 product/condition audit |
| 单步面板渲染脚本 | `scripts/render_route_figures.py` | 附录式 step panel |

验证：

```bash
python -m pytest tests/test_reaudit_route_pool.py tests/test_render_linear_route_schemes.py tests/test_web_product_audit_filter.py tests/test_render_route_trees.py tests/test_render_route_figures.py tests/test_route_plausibility.py tests/test_product_route_feasibility_audit.py
```

结果：

```text
21 passed
```

---

## 1. 我们目前经历了哪些阶段

### 阶段 1：从“no route”问题定位到路线池问题

最开始的问题是：目标分子在界面或汇报中被理解为 `no route`。但实际检查结果文件后发现，这不是路线不存在，而是路线生成、过滤、展示和评价之间的语义没有对齐。

目标分子：

```text
CC(C)C1=NC(=NC(=C1/C=C/[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)N(C)S(=O)(=O)C
```

对应结果文件中实际存在：

| 统计项 | 数值 |
| --- | ---: |
| raw routes | 906 |
| raw unique signatures | 393 |
| kept routes | 640 |
| kept unique signatures | 243 |
| rejected routes | 266 |
| rejected unique signatures | 150 |

这说明问题不是“完全搜不到路线”，而是：

- 哪些路线是真正有意义的？
- 哪些路线只是 stock closure 或模型伪影？
- 哪些路线应当作为专家审查候选？
- 哪些路线可以进入后续训练和 reranking？

### 阶段 2：建立路线可用性分层

我们没有把所有 stock-closed route 都算作成功，而是分成几类：

| 类别 | 含义 |
| --- | --- |
| `triage_fragment` | 有可讨论的片段连接、芳基偶联、侧链构建等路线思路 |
| `needs_chemist_review` | 可能有参考价值，但证据不足或风险较高 |
| `reject_artifact` | 有明显原子来源、结构闭合或模型伪影问题 |
| autonomous candidate | 更严格的可执行候选，目前没有路线达到这个级别 |

第一版审计帮助我们识别了很多明显 artifact，但也暴露出一个严重问题：它把“分子很大”过度等同于“高级中间体”。

### 阶段 3：用户指出 route5 案例，推动审计逻辑修正

640 条保留路线中的第 5 条是关键案例。它包含一个 25-heavy-atom 的大 terminal：

```text
CCOC(=O)C=P(c1ccccc1)(c1ccccc1)c1ccccc1
```

第一版审计把它标成：

```text
advanced_or_product_like_terminal
large_polycyclic_terminal
product_like_terminal
```

但化学上这个判断不严谨。该分子是 ethyl 2-(triphenylphosphoranylidene)acetate，一类稳定 Wittig 试剂。它的三个 phenyl 不是最终产物骨架的一部分，而是反应载体/离去结构。它大，但不等于高级产品片段。

同一条路线还出现了 `Cl` 在产物中出现但不在显式 reactants 中出现的问题。进一步检查条件预测发现：

```text
Reagent: O=P(Cl)(Cl)Cl
```

也就是 `POCl3`。因此这个 `Cl` 不是一定凭空出现，而是可以由 condition reagent 解释。

### 阶段 4：修正 product audit 与 route plausibility

我们做了两类修正。

第一类：terminal profile 不再只看原始 heavy atom count。

现在同时记录：

```text
max_terminal_heavy_atoms
effective_max_terminal_heavy_atoms
carrier_reagents
product_like_terminal
large_polycyclic_terminal
```

其中 carrier reagent 会从 product-like 判定中剥离出来。

第二类：route plausibility 不再只看 reaction SMILES 的 reactants。

现在区分：

```text
raw_element_gains
condition_supported_element_gains
unexplained_element_gains
unexplained_new_elements
```

只有在显式 reactants 和 condition reagent 都无法解释时，才标记为 unsupported element source。

### 阶段 5：重新审计 640 条路线

用当前规则重新审计后：

| 当前审计类别 | 数量 |
| --- | ---: |
| `triage_fragment` | 360 |
| `needs_chemist_review` | 232 |
| `reject_artifact` | 48 |

新增 condition audit 后：

| 条件审计类别 | 数量 |
| --- | ---: |
| `high` | 193 |
| `warn` | 447 |
| `ok` | 0 |

解释：

- 这不是说 640 条路线都不可用；
- 它说明当前 `condition_predictions` 还只是逐步预测，不能当作级联工艺条件；
- `warn` 通常表示低温有机金属/还原剂、加热偶联、低条件分数、弱酶证据和有机溶剂等；
- `high` 表示存在更强的条件风险，例如极端高温、强酶证据与温度窗口冲突，或路线温度跨度过大。

这比第一版更合理：

- 不再因为 Wittig/HWE carrier reagent 大而误伤大量路线。
- 仍然能识别真正没有元素来源支持的路线。
- route5 被恢复为合理的 `triage_fragment` 候选。
- 同时不再把预测条件静默展示为“已验证条件”，而是单独打 `condition_audit` 标记。

route5 当前审计摘要：

```text
route_class: triage_fragment
issues: condition_high_risk
tags: acylating_piece_present, aryl_coupling_hint, carrier_reagent_terminal
product_like_terminal: false
large_polycyclic_terminal: false
effective_max_terminal_heavy_atoms: 10
route_plausibility_passed: true
condition_audit.route_risk: high
condition_audit.high_risk_step_count: 1
condition_audit.warning_step_count: 9
condition_audit.temperature_span_c: 199.434
```

这里的 `condition_high_risk` 不是 material artifact，而是条件层风险：Route 5 中存在 LDA、DIBAL-H、硼氢化物/硼试剂还原、加热 Suzuki 偶联等单步条件，且温度跨度从约 -70 °C 到 130 °C。它更像分步合成路线候选，不应汇报为一锅级联条件。

### 阶段 6：生成连续顺合成路线图

此前的路线图是单步面板式，不适合论文或专家 PPT。现在新增了两类渲染：

```text
scripts/render_route_trees.py
scripts/render_linear_route_schemes.py
```

其中 `render_linear_route_schemes.py` 是当前推荐给专家汇报使用的版本，更接近论文中的 synthesis scheme 风格：

- 按最终合成方向展示，而不是逆合成方向；
- 主链分子连续横向展开，长路线按行自然延续；
- 对由关键组件分别制备再汇合的情况，辅助组件会以小型 branch scheme 显示在对应箭头上方；
- 分子结构不加外框；
- 反应条件写在箭头线上方/下方；
- 辅助反应物尽量以缩小结构显示，而不是 SMILES；
- 分子结构下方不再写 SMILES；
- 图中不写 `template`、`planner` 等内部模型来源；
- 条件中的常见模型 SMILES 会转成 NaOH、POCl3、DIBAL-H、LDA、THF、toluene、Pd catalyst 等合成方案常用写法；
- 简单无机酸碱、盐、催化剂优先写入条件，不作为大号辅助结构干扰主线。
- 条件审计风险用 `?` / `!` 标记：`?` 表示需要专家关注的逐步条件风险，`!` 表示更高风险；
- 图下方说明条件是模型逐步预测假设，不是已验证工艺条件。

当前推荐用于 PPT 的输出：

```text
results/v2/route_schemes_3764f7_top10_current/index.html
results/v2/route_schemes_3764f7_top10_current/scheme_route_01.pdf
results/v2/route_schemes_3764f7_top10_current/scheme_route_05.pdf
```

连续树输出保留用于查看 route topology，不作为主 PPT 图首选：

```text
results/v2/route_trees_3764f7_top10_current/index.html
results/v2/route_trees_3764f7_top10_current/route_tree_01.svg
results/v2/route_trees_3764f7_top10_current/route_tree_01.pdf
...
results/v2/route_trees_3764f7_top10_current/route_tree_10.svg
results/v2/route_trees_3764f7_top10_current/route_tree_10.pdf
```

这些图是连续 retrosynthesis tree：

- 目标产物在左侧；
- 前体向右展开；
- 每个节点是 RDKit 分子结构；
- 边表示对应反应步骤；
- 边标签包含 step、reaction type、EC、温度等信息；
- terminal materials 单独以 terminal 节点展示。

关于标准作图工具：目前没有一个开源工具可以直接把任意 AI 生成的 18 步 route JSON 自动排成论文级 synthesis scheme。常用工具分工通常是：

| 工具 | 适合做什么 | 局限 |
| --- | --- | --- |
| ChemDraw / BioRender / Adobe Illustrator | 人工精修论文 scheme | 不适合批量自动生成 top10 |
| RDKit MolDraw2D | 自动画分子结构 | 不负责整条路线排版 |
| CDK DepictionGenerator | 自动画分子/反应 | Python 项目集成成本较高 |
| Indigo Toolkit | 分子和反应描绘 | 路线级排版仍需额外逻辑 |
| Open Babel depict | 分子格式转换和描绘 | 论文 scheme 排版能力有限 |
| Graphviz | 自动图布局 | 化学分子结构仍需 RDKit/CDK 等提供 |

因此我们采用的方案是：用 RDKit 负责标准分子结构描绘，用自定义 layout 负责 route scheme 排版。这不是从零画化学结构，而是在标准分子绘图基础上做路线级自动排版。当前版本已能批量生成 top10 的 SVG/PDF，后续可在 ChemDraw 或 Illustrator 中做最后的人工美化。

---

## 2. 我们选择了哪些方法

### 2.1 路线池，而不是只看 top-1

复杂 cascade 路线不能只看 top-1。长路线中单步小错误会累积，top-1 的分数也不等于整条路线可执行。

因此我们采用：

```text
生成路线池 -> 去重 -> 审计 -> 分层 -> rerank -> shortlist -> 专家可视化
```

路线池的价值在于：

- 可以比较不同 disconnection；
- 可以找出重复模式；
- 可以建立 hard negatives；
- 可以用无专家标签的规则信号训练 ranker。

### 2.2 Product audit，而不是 stock closure

stock-closed 不等于路线可用。

一条路线可能因为以下原因 stock-closed：

- 把高级中间体当成 stock；
- 把 carrier reagent 误当成产品片段；
- reaction SMILES 省略了关键 reagent；
- 单步模型凭空引入元素或官能团；
- route search 用低可信步骤强行闭合。

所以我们引入：

```text
terminal_profile
reaction_profile
route_plausibility
condition-supported atom source
route_class
risk_order
```

### 2.3 无专家标签训练

用户已经明确：现在没有专家标签，未来也不会有专家标签。因此训练不能依赖 expert-labeled good/bad route。

我们应使用弱监督和自监督信号：

| 信号 | 来源 | 是否需要专家标签 |
| --- | --- | --- |
| 原子来源一致性 | RDKit / atom count / condition reagent | 否 |
| route consistency | 上一步产物是否对应下一步前体 | 否 |
| stock realism | stock map / terminal complexity | 否 |
| 条件支持 | condition_predictions | 否 |
| 酶证据 | EC confidence / enzyme annotation | 否 |
| hard negative | 人工扰动、删除 reagent、引入 unsupported element | 否 |
| pairwise preference | 低风险路线优于高风险路线 | 否 |
| route diversity | reaction signature clustering | 否 |

这更接近：

```text
rule-derived weak supervision + hard-negative training + route-pool reranking
```

而不是传统专家标注分类。

### 2.4 通用方向：atom contribution profile

当前对 carrier reagent 的识别只是第一层修正，不能停留在白名单。

更通用的下一步是对每个 terminal/reagent 计算：

```text
terminal_smiles
raw_heavy_atoms
atoms_retained_in_product
retained_fraction
target_coverage
role_guess
role_confidence
evidence
```

这里比例可以作为特征，但不应作为唯一判据。

合理判定应结合：

- 哪些原子进入最终产物；
- 哪些原子只是 carrier / leaving group / protecting group；
- 该分子是 reactant、reagent、catalyst、solvent 还是 salt；
- 条件预测是否支持该元素或基团转移；
- 反应类型是否与该转移相符。

---

## 3. 国际前沿怎么做

### 3.1 Neural-symbolic retrosynthesis / MCTS

Segler、Preuss、Waller 在 Nature 2018 提出将神经网络策略与符号模板、Monte Carlo Tree Search 结合，用于 retrosynthesis planning。

代表特点：

- 单步反应模板预测；
- MCTS 或启发式搜索；
- 目标是递归断键到 purchasable building blocks；
- 主要优化 search success 和 stock closure。

局限：

- 更关注“能不能断到 stock”；
- route-level condition compatibility 较弱；
- 对酶催化、级联兼容性、cofactor、pH 等支持不足。

### 3.2 AiZynthFinder

AiZynthFinder 是目前非常常用的开源 retrosynthesis planner，基于神经网络模板策略与 MCTS。

优点：

- 开源、稳定；
- route search 工程实现成熟；
- 可接入 stock；
- 可作为 CASP baseline。

局限：

- 标准任务仍是 small-molecule retrosynthesis；
- 对 chemoenzymatic cascade 的条件状态建模有限；
- stock closure 不等于级联路线可执行。

### 3.3 ASKCOS

ASKCOS 体系将 retrosynthesis、条件推荐、反应可行性和自动实验平台连接起来。Science 2019 的 AI planning + flow synthesis 是代表性工作。

优点：

- 更接近实验执行；
- 同时考虑 retrosynthesis 和 forward validation；
- 有条件推荐和反应可行性模块。

局限：

- 主要面向有机小分子合成；
- cascade 中跨步骤兼容、酶催化和多步状态传递仍不是核心。

### 3.4 RDChiral / template stereochemistry

RDChiral 解决 template extraction/application 中 stereochemistry 的一致性问题，是 template-based retrosynthesis 的重要基础工具。

它能解决：

- 反应模板提取；
- 立体化学处理；
- SMARTS 应用一致性。

但它不解决：

- route-level 级联条件；
- enzyme evidence；
- condition reagent atom provenance；
- 长路线全局可执行性。

### 3.5 RetroBioCat

RetroBioCat 是更接近我们问题的系统，因为它面向 biocatalytic reactions and cascades。

它的优势：

- 明确考虑 biocatalysis；
- 支持 enzyme reaction rules；
- 更接近 cascade planning。

但仍有挑战：

- 与常规有机 CASP 的统一仍困难；
- enzyme substrate scope 和 condition compatibility 需要高质量知识库；
- 长 chemoenzymatic route 的全局评分仍很难。

### 3.6 RAscore / SCScore 等快速可合成性评分

这类模型用于快速估计 synthetic accessibility 或 retrosynthetic accessibility。

优点：

- 快；
- 适合大规模筛选；
- 可作为 route reranking 特征。

局限：

- 通常不是逐步路线验证；
- 不能解释 condition reagent；
- 不能保证 cascade 可执行；
- 不能替代 route audit。

---

## 4. 为什么国际前沿仍难处理 cascade

### 4.1 传统 CASP 多数是 step-wise，而 cascade 是 stateful

传统 retrosynthesis 通常把路线看成一系列单步断键。

但 cascade synthesis 需要维护状态：

```text
solvent
pH
temperature
redox state
cofactor
enzyme compatibility
intermediate stability
isolation/no-isolation
workup compatibility
```

这些状态会跨步骤传递，不是每步独立打分能解决的。

### 4.2 Reaction SMILES 通常不完整

公开数据和模型输出里，reaction SMILES 往往只记录主反应物和产物。

常被省略：

- 酸碱；
- 氧化剂/还原剂；
- 卤化剂；
- 保护/脱保护试剂；
- 盐；
- 水；
- cofactor；
- catalyst；
- solvent。

这会导致两个相反问题：

1. 真正合理的步骤被误判为原子来源错误；
2. 真正不合理的步骤因为没有做 role-aware 检查而被放过。

### 4.3 酶标签不是酶证据

EC number 只是酶反应类别，不是具体酶、序列、活性、底物范围或实验条件。

当前很多路线的 enzyme confidence 在：

```text
0.07 - 0.20
```

这只能作为 hypothesis，不能作为强证据。

### 4.4 长路线误差会累积

17-20 步路线里，每一步即使只有小概率错误，整条路线也容易变得不可执行。

因此必须做：

- route-level audit；
- step consistency check；
- condition compatibility check；
- terminal role assignment；
- route tree visualization；
- expert triage。

### 4.5 化学 CASP 与生物催化规划割裂

有机小分子 CASP 强在模板、stock 和反应数据；
生物催化规划强在 enzyme rule 和 pathway/cascade 思维。

我们的目标是 chemoenzymatic cascade，必须同时处理：

```text
chemical transformation
enzymatic transformation
condition prediction
enzyme annotation
stock closure
route-level compatibility
```

这也是为什么单纯套用现有 CASP 框架不足。

我们目前已经把 per-step 条件风险推进到 route-level `condition_state`
摘要：它可以统一给出 stage 内冲突、温度/pH 跨度、缺失条件和 same-pot 风险，
作为后续 search-time 级联条件建模的共享输入。

---

## 5. 我们目前如何处理 cascade

当前系统的处理流程：

```text
ChemEnzy route generation
  -> route pool
  -> product audit
  -> plausibility audit
  -> condition-supported atom source
  -> carrier/terminal role correction
  -> route-class reranking
  -> top route visualization
```

已具备能力：

| 能力 | 当前状态 |
| --- | --- |
| 生成长路线池 | 已有，当前 target raw 906 routes |
| route 去重 | 已有 reaction signature 方式 |
| product-aware audit | 已有 |
| condition reagent 支持原子来源 | 第一版已有 |
| condition audit | 已有，区分条件预测风险与物料闭合 artifact |
| carrier reagent terminal 修正 | 第一版已有 |
| reject artifact | 已有 |
| top10 连续路径图 | 已有 |
| 无专家标签训练方向 | 已确定 |

### 5.1 RetroChimera 已接入 proposal sidecar

为了把 proposal 层和级联条件问题拆开处理，我们已经把 RetroChimera 接进
`cascade_search` 的 proposal provider 链路中。它现在的角色是：

```text
RetroChimera proposal sidecar
  -> candidate reactant generation
  -> cascade search state
  -> condition-aware audit
```

这件事的含义不是“再加一个生成器”，而是：

- 让 RetroChimera 作为候选补全器，补 ChemEnzy 生成池里缺失的 proposal；
- 让级联条件约束继续留在 search state 和 audit 层，而不是被 proposal 层吃掉；
- 让我们可以区分“反应物候选是否找得到”和“这条路线能否级联”两个问题。

当前已经确认：

- RetroChimera provider 可以从本地模型目录正常加载；
- `CCO` 这类输入可以返回候选反应物；
- benchmark runner 已可通过 `--use-retrochimera-proposals` 打开该 sidecar；
- 缺失 condition envelope 的 step 不会被当作可级联条件。

所以现在的主线已经不是继续堆单纯 proposal 训练，而是：

```text
proposal completion
  + condition-aware state
  + route-level audit
  + cascade compatibility scoring
```

还不能声称：

- 不能声称路线已经实验可执行；
- 不能声称酶步骤已经验证；
- 不能声称 360 条 `triage_fragment` 都是高质量路线；
- 不能声称当前 carrier 逻辑已经完全通用。
- 不能声称图中的预测条件已经经过实验或级联兼容性验证。

---

## 6. 代表性高有效性长路线

### 6.1 Route 1：高分长路线代表

推荐展示图：

```text
results/v2/route_schemes_3764f7_top10_current/scheme_route_01.pdf
results/v2/route_schemes_3764f7_top10_current/scheme_route_01.svg
```

摘要：

```text
n_steps: 17
score: 0.0112
route_class: triage_fragment
tags: acylating_piece_present, aryl_coupling_hint, carrier_reagent_terminal
```

适合 PPT 展示点：

- 展示系统确实能生成长路线；
- 展示 aryl coupling 和侧链构建；
- 展示长路线必须经 audit/rerank；
- 展示该路线仍是 triage route，不是 autonomous route。

### 6.2 Route 5：审计逻辑修正的关键案例

推荐展示图：

```text
results/v2/route_schemes_3764f7_top10_current/scheme_route_05.pdf
results/v2/route_schemes_3764f7_top10_current/scheme_route_05.svg
```

摘要：

```text
n_steps: 18
score: 0.00131
route_class: triage_fragment
issues: condition_high_risk
effective_max_terminal_heavy_atoms: 10
condition route risk: high
condition warnings: 9
condition high-risk steps: 1
temperature span: about 199 °C
```

为什么重要：

- 旧规则误把 Wittig phosphorane 当成高级 product-like terminal；
- 新规则识别其为 `carrier_reagent_terminal`；
- step 中出现的 `Cl` 可由 condition reagent `POCl3` 支持；
- 路线图现在将 LDA、DIBAL-H、NaBH4/EtOBEt2、Pd coupling 等条件标记为 `?`；
- 这说明路线审计必须 role-aware、condition-aware，并且要把“原子来源合理”和“条件可执行/可级联”分开。

PPT 讲法建议：

> 这条路线不是严格意义的可执行工艺，但它是一个非常好的系统能力案例：它暴露了 naive audit 的误伤，也说明我们需要从 heavy atom count 转向 atom provenance、reagent role 和 condition audit。Route 5 的 disconnection 有讨论价值，但当前条件更像分步合成条件集合，不是可直接执行的一锅级联。

---

## 7. 还需要加强的部分

### 7.1 从 carrier 白名单升级为通用 atom contribution

当前 carrier reagent 修正仍然偏规则化。

下一步应该做：

```text
atom_contribution_profile
```

对每个 terminal/reagent 计算：

| 字段 | 含义 |
| --- | --- |
| `raw_heavy_atoms` | 原始分子 heavy atom 数 |
| `atoms_retained_in_product` | 进入产物的原子数 |
| `retained_fraction` | 该分子有多少比例进入产物 |
| `target_coverage` | 该分子贡献了目标产物多少比例 |
| `role_guess` | building block / carrier / reagent / catalyst / solvent 等 |
| `evidence` | atom mapping、MCS、condition、reaction class |

判断逻辑不应是单阈值，而应是：

```text
原子贡献 + 化学角色 + 条件证据 + 反应类型
```

### 7.2 引入 atom mapping / MCS

推荐技术路径：

1. 优先使用 atom-mapped reaction；
2. 没有 mapping 时使用 RXNMapper 或同类工具；
3. mapping 失败时用 RDKit MCS 近似；
4. 对明显 carrier / protecting / oxidant / reductant / halogenation reagent 用 role prior。

### 7.3 condition reagent role assignment

现在只是读取 `condition_predictions[].Reagent`。

下一步要区分：

| 来源 | 可信度 |
| --- | --- |
| explicit reactants | 最高 |
| small stoichiometric reagent | 高 |
| catalyst | 低，不能默认提供原子 |
| solvent | 很低，通常不能默认提供骨架原子 |
| unknown condition text | 需要 warning |

### 7.4 enzyme evidence calibration

需要建立酶证据分层：

| 证据 | 解释 |
| --- | --- |
| EC top-1 confidence | 当前已有，但偏弱 |
| substrate scope similarity | 需要补 |
| enzyme sequence / UniProt evidence | 需要补 |
| cofactor compatibility | 需要补 |
| pH/temperature compatibility | 需要补 |
| cascade compatibility | 需要补 |

### 7.5 cascade state model

需要为路线引入状态变量：

```text
route_state = {
  solvent,
  pH,
  temperature,
  redox_state,
  cofactor,
  isolation_required,
  intermediate_stability,
  enzyme_compatibility
}
```

每一步不只是生成产物，还要更新 route state。

### 7.6 条件模型校准与显示策略

当前条件模型的温度、溶剂、试剂是逐步预测结果，不能直接作为工艺条件。产品层需要继续强化：

| 模块 | 当前处理 | 下一步 |
| --- | --- | --- |
| reagent as atom source | 可解释 POCl3 等元素来源 | 继续区分 stoichiometric/catalytic/solvent |
| temperature | 仅做风险标记 | 用反应类别和文献先验校准 |
| enzyme condition | 低置信度 EC 不硬判工艺冲突 | 引入 BRENDA/UniProt/序列证据 |
| route compatibility | 识别温度跨度和分步需求 | 建立 route-state transition model |
| visualization | 用 `?` / `!` 标记预测条件风险 | 支持隐藏/展开条件风险层 |

### 7.7 级联条件状态要真正进入 search

这是当前最需要继续强化的部分。

现在我们已经能做：

- 生成 proposal；
- 做 route pool；
- 识别 carrier reagent；
- 标记条件风险；
- 渲染连续路线图。

但如果 search state 里没有真正维护下列状态变量，级联问题就还只是事后审计：

```text
solvent
pH
temperature
redox_state
cofactor
enzyme_compatibility
isolation_required
intermediate_stability
```

因此下一步重点不是再堆一个新的 proposal 模型，而是把这些状态变量变成 search 的一等公民，让 proposal、条件、酶证据和 route audit 统一进一个 stateful planner。

### 7.8 更适合 PPT 的路线压缩

当前 paper-style scheme 已可用于主文 PPT，但 17-18 步路线仍然很长。后续还可以继续做压缩版：

- 隐藏小分子 salts/acid/base；
- 合并 reagent preparation subtree；
- 只保留关键 C-C / C-N / side-chain forming steps；
- terminal materials 单独放在右侧；
- 反应条件保留在箭头附近，但对重复条件做合并。

---

## 8. 建议 PPT 结构

### Slide 1：问题定义

标题：从路线生成到级联路线可信度评估

要点：

- 目标不是“有没有路线”，而是“长路线是否可信”。
- chemoenzymatic cascade 比普通 retrosynthesis 更难。

### Slide 2：目标分子与初始误解

展示：

- target structure；
- 原始 `no route` 误解；
- 实际 raw 906 routes。

### Slide 3：路线池结果

表格：

```text
raw: 906
unique raw: 393
kept: 640
current triage_fragment: 360
needs_chemist_review: 232
reject_artifact: 48
```

### Slide 4：为什么 stock closure 不够

列出四类误差：

- advanced terminal；
- carrier reagent；
- condition reagent omitted；
- unsupported element source。

### Slide 5：国际前沿

展示：

- Segler/Waller neural-symbolic retrosynthesis；
- AiZynthFinder；
- ASKCOS；
- RDChiral；
- RetroBioCat。

### Slide 6：为什么它们处理 cascade 仍困难

重点：

- step-wise vs stateful；
- condition/pH/cofactor；
- enzyme evidence；
- long-route error compounding。

### Slide 7：我们的 CascadePlanner 流程

画流程图：

```text
route generation -> route pool -> audit -> rerank -> visualization -> weak supervision
```

### Slide 8：route5 案例

放：

```text
scheme_route_05.pdf
```

讲：

- old audit 为什么误伤；
- Wittig carrier 为什么不应按高级 terminal 处理；
- `POCl3` 如何解释 `Cl` 来源；
- 新审计如何修正。

### Slide 9：代表路线

放：

```text
scheme_route_01.pdf
```

讲：

- 系统能生成 17 步长路线；
- 路线有 fragment/coupling 价值；
- 仍需专家审查和酶证据加强。

### Slide 10：无专家标签训练路线

展示：

```text
rule-derived weak supervision
hard negatives
pairwise preferences
route-pool reranking
self-consistency
```

### Slide 11：下一步

重点：

- atom contribution profile；
- condition role assignment；
- enzyme calibration；
- cascade state model；
- PPT-friendly compressed route diagrams。

---

## 9. 对专家汇报时可以说与不能说

可以说：

- 系统已经能为复杂 target 生成大量长路线候选；
- 仅以 stock closure 作为成功标准是不够的；
- 我们已经建立了 product-aware audit 和 route-level triage；
- 我们发现并修正了 naive terminal-heavy-atom 规则的误伤；
- 我们开始处理 condition reagent 对原子来源的解释；
- 已生成 top10 顺合成 synthesis scheme 和连续路线树用于专家审查。

不能说：

- 不能说已经得到可直接实验执行的合成工艺；
- 不能说酶步骤已经验证；
- 不能说所有 `triage_fragment` 都可用；
- 不能说当前 carrier/condition 逻辑已经完全通用；
- 不能说无需专家后续审查。

---

## 10. 参考文献与来源

1. Segler, M. H. S.; Preuss, M.; Waller, M. P. Planning chemical syntheses with deep neural networks and symbolic AI. Nature 555, 604-610 (2018). https://www.nature.com/articles/nature25978
2. Genheden, S. et al. AiZynthFinder: a fast, robust and flexible open-source software for retrosynthetic planning. Journal of Cheminformatics 12, 70 (2020). https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00472-1
3. Coley, C. W. et al. A robotic platform for flow synthesis of organic compounds informed by AI planning. Science 365, eaax1566 (2019). https://doi.org/10.1126/science.aax1566
4. Coley, C. W.; Green, W. H.; Jensen, K. F. RDChiral: An RDKit Wrapper for Handling Stereochemistry in Retrosynthetic Template Extraction and Application. Journal of Chemical Information and Modeling 59, 2529-2537 (2019). https://pubs.acs.org/doi/10.1021/acs.jcim.9b00286
5. Finnigan, W.; Hepworth, L. J.; Flitsch, S. L.; Turner, N. J. RetroBioCat as a computer-aided synthesis planning tool for biocatalytic reactions and cascades. Nature Catalysis 4, 98-104 (2021). https://www.nature.com/articles/s41929-020-00556-z
6. Thakkar, A. et al. Retrosynthetic accessibility score: rapid machine learned synthesizability classification from AI driven retrosynthetic planning. Chemical Science 12, 3339-3349 (2021). https://pubs.rsc.org/en/content/articlehtml/2021/sc/d0sc05401a
7. ACS Green Chemistry Institute, Process Mass Intensity materials. https://www.acs.org/green-chemistry-sustainability/green-chemistry-nexus/articles/process-mass-intensity-calculation-tool.html
8. ACS GCI Pharmaceutical Roundtable, Process Mass Intensity Metric. https://learning.acsgcipr.org/guides-and-metrics/metrics/process-mass-intensity-metric/
9. Sheldon, R. A. The E factor at 30: a passion for pollution prevention. Green Chemistry 25, 1704 (2023). https://pubs.rsc.org/fa/content/articlelanding/2023/gc/d2gc04747k
