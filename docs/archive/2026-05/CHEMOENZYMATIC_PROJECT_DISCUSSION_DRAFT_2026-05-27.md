# 化学-酶催化逆合成项目讨论文稿

日期：2026-05-27

本文稿用于内部和专家讨论。目标不是给出已经封闭的结论，而是系统整理目前项目中的困惑、证据、失败原因、可能转向方向，以及需要专家进一步判断的问题。

## 1. 当前项目走到的位置

我们最初希望做一个面向化学-酶催化级联反应的逆合成规划系统。早期叙事主要集中在：

1. 多步逆合成。
2. 化学步骤和酶催化步骤混合。
3. one-pot 或 cascade 条件兼容性。
4. 对路线做条件、酶、辅因子、库存、可行性审计。

经过多轮实验后，一个比较明确的现实是：如果把创新点主要放在 one-pot 条件兼容、pH、温度、溶剂冲突、条件审计上，这条线很难成立为强创新。原因是现实中的 chemo-enzymatic synthesis 很多并不是严格 one-pot，更多是序贯操作、分步纯化或条件切换。此时条件兼容性当然有用，但不一定是主问题。

另一方面，如果完全放弃级联，只做普通多步逆合成，项目又会失去辨识度。现在主流 retrosynthesis 工具有很多，我们目前的系统在普通多步逆合成上没有明显优势。单纯堆搜索深度、迭代次数和 proposal 模型，并没有稳定转化成更好的长路线质量。

因此当前真正需要讨论的是：

**级联是否真的无意义，还是我们之前定义错了级联的核心问题？**

## 2. 已经观察到的关键事实

### 2.1 单步 proposal 是瓶颈之一

我们反复观察到：如果真实反应物不在候选池里，搜索再深也不可能选到正确路线。多步搜索的上限首先由单步 proposal 覆盖率决定。

近期对 `benchmark_v2_100` final/top-level GT step 的 one-step 评估显示：

| proposal 来源 | Top1 exact reactant | Top5 exact reactant | Top16 exact reactant | 平均耗时 |
|---|---:|---:|---:|---:|
| ChemEnzy native 两模型 | 5/100 | 12/100 | 17/100 | 1.4903 s |
| AiZynthFinder uspto | 8/100 | 16/100 | 17/100 | 0.2760 s |
| RetroKNN external150k | 16/100 | 17/100 | 18/100 | 0.1096 s |

组合后的 Top16 union：

| 组合 | Top16 覆盖 |
|---|---:|
| native | 17/100 |
| native + AiZ | 21/100 |
| native + RetroKNN | 30/100 |
| native + AiZ + RetroKNN | 33/100 |

这说明新增 proposal 不是完全无效。它确实补到了 native 模型没有覆盖的候选，尤其 RetroKNN 对真实反应库的检索有明显补缺口作用。

### 2.2 直接 ensemble 会污染搜索

但是，将 AiZynthFinder 和 RetroKNN 直接作为全节点常开 proposal 放进多步搜索后，效果并没有改善，反而出现搜索膨胀。

前 5 个目标小样本测试：

| 配置 | exact reaction in route pool | gt reactant in route pool | 平均 cascade search 时间 |
|---|---:|---:|---:|
| baseline | 0.2 | 0.6 | 0.0278 s |
| AiZ + RetroKNN top16 | 0.0 | 0.6 | 27.57 s |
| AiZ + RetroKNN top4 | 0.0 | 0.6 | 12.9732 s |

provider 调用量也显著膨胀：

| 配置 | provider 调用 | 返回候选 |
|---|---:|---:|
| baseline | static 17 次 | 166 |
| AiZ + RetroKNN top16 | static/aiz/retroknn 各 389 次 | 约 8900 |
| AiZ + RetroKNN top4 | static/aiz/retroknn 各 359 次 | 约 2890 |

结论是：

**proposal 覆盖提升是真实的，但 naive ensemble 不等于更好的多步搜索。必须做 gated proposal routing，不能所有节点都开所有模型。**

### 2.3 当前酶步预测明显不足

目前系统对“什么酶催化什么底物”“什么底物需要什么酶催化”“某个底物是否处在某类酶可接受的底物空间内”的建模远远不够。

表现为：

1. 酶步经常只是形式上给出 EC 或 enzyme candidate，但缺少可信 substrate-enzyme 匹配证据。
2. 对底物结构微小变化、手性、官能团位置、反应中心的识别不够。
3. 酶催化 proposal 很容易变成泛化过度的结构相似检索。
4. 当前系统更像是在“给化学路线附加酶标签”，而不是在真正判断酶催化机会。

这可能才是 chemo-enzymatic planning 的真正缺口。

## 3. 目前最大的困惑

### 3.1 级联到底是不是有意义？

如果“级联”被理解为 one-pot 条件兼容，意义可能有限。因为现实工艺可以分步、换溶剂、换 buffer、纯化中间体。此时 pH/温度/溶剂兼容不是不存在，但不是最核心的瓶颈。

但如果“级联”被重新理解为：

**化学步骤和酶催化步骤之间的可接续性，以及中间体进入酶底物空间的机会识别。**

那么级联仍然有意义，而且可能是项目最有差异化的地方。

也就是说，问题不是：

> 这些步骤能不能 one-pot 放在一起？

而是：

> 一个化学步骤的产物，是否可以成为某类酶催化步骤的底物？
> 一个酶催化步骤的产物，是否可以作为后续化学步骤的中间体？
> 哪些化学中间体其实提供了切换到生物催化的机会？

这个问题比普通多步逆合成更有特色。

### 3.2 虚拟级联数据是否合理？

专家提出的关键观点是：

**化学步产物可以作为酶步的反应物，这个事实可以构建大量虚拟级联数据。**

这点非常重要。因为我们没有专家标签，未来也不会有大规模专家标签。真实高质量 chemo-enzymatic cascade 文献数据数量有限，直接训练 cascade generator 或 ranker 很难。

但如果用化学反应库和酶反应库做连接，可以构造大量 weakly supervised 数据：

1. 化学反应产物匹配已知酶反应底物。
2. 酶反应产物匹配化学反应底物。
3. 化学反应产物与酶底物高度相似，构成潜在连接。
4. 酶产物与已知合成中间体相似，构成后续化学转化机会。

这类数据不应该被称为“真实工艺路线正样本”。更准确地说，它们是：

**chemo-enzymatic connectivity weak labels**

即化学和酶反应空间之间的可连接信号。

这可能是突破数据稀缺的关键。

### 3.3 虚拟数据的风险是什么？

虚拟级联数据也有明显风险：

1. 结构相似不等于酶可催化。
2. EC 类别相同不代表底物可接受。
3. 文献底物空间可能很窄，模型容易过度泛化。
4. 化学产物作为酶底物在理论上可连接，不代表真实体系可操作。
5. 反应中心匹配、辅因子、立体选择性、区域选择性都可能成为失败原因。

因此虚拟数据不能直接当作路线 ground truth。它更适合用于：

1. 预训练 enzyme-substrate representation。
2. 训练 enzyme bridge retriever。
3. 训练 enzyme feasibility verifier。
4. 生成 hard negatives。
5. 支持搜索中的 gated enzyme proposal。

## 4. 重新定义项目创新点

现在不建议继续把项目定义为：

> one-pot cascade retrosynthesis planner

也不建议退化成：

> generic multi-step retrosynthesis planner

更合适的定义是：

> enzyme-aware chemo-enzymatic route bridging for hard-target retrosynthesis

或者：

> substrate-enzyme-aware proposal routing for chemo-enzymatic synthesis planning

核心问题从“条件兼容性”转为：

1. 化学和酶反应空间如何连接？
2. 哪些化学中间体可以进入酶底物空间？
3. 哪些目标结构适合引入酶催化步骤？
4. 如何预测底物需要什么酶，酶能催化什么底物？
5. 如何在多步搜索中只在合适位置调用酶 proposal，而不是全节点污染搜索？

这个方向仍然保留“级联”的思想，但不再局限于 one-pot。

## 5. 当前技术路线的重新组织

### 5.1 第一层：普通化学 proposal

保留 ChemEnzy native 两个模型作为基础：

1. `graphfp_models.USPTO-full_remapped`
2. `onmt_models.bionav_one_step`

新增的 AiZynthFinder 和 RetroKNN 不应替代 native，而是作为补充。

当前证据表明：

1. AiZynthFinder 提升 Top1/Top5 排序，且常驻后速度较快。
2. RetroKNN 能补真实反应库中的缺口。
3. 二者全节点展开会严重增加搜索噪声。

因此它们应该进入 gated sidecar：

1. root/top-level 优先调用。
2. native 候选不足时调用。
3. frontier 卡死时调用。
4. audit 判断当前候选质量很差时调用。
5. 每个目标限制最大调用次数。

### 5.2 第二层：化学-酶桥接数据

构建 `virtual_chemoenzymatic_bridge_pack`。

数据来源可以包括：

1. 化学反应库：USPTO、Pistachio、Reaxys、ChemEnzy 内部反应。
2. 酶反应库：Rhea、EnzymeMap、ECREACT、RetroBioCat、BRENDA、MetaCyc。
3. 代谢通路库：KEGG、MetaCyc、BiGG。

连接类型：

1. chemical product = enzymatic substrate。
2. enzymatic product = chemical substrate。
3. chemical product 与 enzymatic substrate 相似。
4. enzymatic product 与 chemical substrate 相似。
5. 反应中心一致或反应类型兼容。

每条连接需要记录：

1. connector molecule。
2. chemical reaction id。
3. enzyme reaction id。
4. EC number / enzyme family。
5. reaction class。
6. substrate similarity。
7. reaction center consistency。
8. stereochemistry consistency。
9. evidence source。
10. confidence tier。

### 5.3 第三层：酶-底物预测模块

这是目前最大缺口。

需要训练或构建：

1. `EnzymeSubstrateRetriever`
   - 输入底物或中间体。
   - 输出相似酶反应、可能 EC、可能 enzyme family。

2. `EnzymeFeasibilityVerifier`
   - 输入 substrate、reaction type、EC/enzyme candidate。
   - 输出可行性分数和失败原因。

3. `EnzymeReactionProposer`
   - 输入目标或中间体。
   - 输出可能的酶催化逆合成拆解。

4. `ChemoEnzymaticBridgeScorer`
   - 判断某个化学中间体是否值得切换到酶催化。

训练信号不依赖专家标签：

1. 真实酶反应作为正样本。
2. 错误 EC 替换作为 hard negative。
3. 相似但反应中心不匹配的底物作为 hard negative。
4. 同一官能团但不同位点/立体构型作为 hard negative。
5. 化学库与酶库连接产生 weak positive。
6. 无连接或冲突连接产生 weak negative。

### 5.4 第四层：搜索策略

当前教训是：不能把所有 proposal 全节点打开。

搜索状态需要包含：

1. 当前分子。
2. 当前路线中是否已经出现酶步。
3. 当前分子是否接近酶底物空间。
4. native proposal 是否充分。
5. 当前 frontier 是否卡死。
6. 是否存在高置信 enzyme bridge。

搜索策略从“所有 provider 同时展开”改成：

1. native chemical proposal 先行。
2. 如果 chemical proposal 足够强，暂不调用酶模型。
3. 如果目标或中间体落入酶底物空间，调用 enzyme bridge。
4. 如果 RetroKNN 命中真实相似反应，加入 evidence-supported proposal。
5. 如果 enzyme verifier 低分，则酶 proposal 不进入主搜索，只保留为候选证据。

## 6. 应该如何评价新方向

不能只看传统 solved rate。当前 benchmark 没有 starting materials，且普通 stock-closed 指标容易误导。

建议建立四类指标。

### 6.1 Proposal coverage 指标

1. GT reactant in proposal pool。
2. exact reaction in proposal pool。
3. top-k reactant recall。
4. union coverage。
5. coverage gain over native。

### 6.2 酶催化预测指标

1. EC top-k recall。
2. enzyme family top-k recall。
3. substrate feasibility AUC。
4. enzyme-substrate pair classification F1。
5. product prediction accuracy。
6. reaction center consistency。

### 6.3 化学-酶桥接指标

1. chemical product to enzymatic substrate hit rate。
2. enzymatic product to chemical substrate hit rate。
3. bridge confidence calibration。
4. expert-reviewed bridge plausibility。
5. heldout literature bridge recovery。

### 6.4 多步搜索指标

1. evidence-supported route count。
2. enzyme step precision。
3. route plausibility audit pass rate。
4. search cost per useful route。
5. long-route diversity。
6. whether enzyme step replaces chemically difficult transformation。

## 7. 需要专家重点讨论的问题

### 7.1 关于级联定义

1. 如果不强调 one-pot，而强调化学-酶步骤可接续性，专家是否认可这是有意义的 chemo-enzymatic planning 问题？
2. “化学步产物作为酶步底物”的虚拟连接，是否符合领域直觉？
3. 这种连接应该被称为 cascade、hybrid route、chemo-enzymatic bridge，还是其他更准确的术语？

### 7.2 关于虚拟数据

1. 哪些连接可以认为是强 weak label？
2. 哪些连接只能作为低置信候选？
3. 结构相似到什么程度才可以认为进入酶底物空间？
4. 是否必须匹配反应中心？
5. 是否必须考虑立体化学？
6. 是否需要按 EC 大类分别建模？

### 7.3 关于酶-底物预测

1. 专家认为酶步预测最关键的输入是什么？
   - substrate 结构。
   - product 结构。
   - reaction class。
   - EC。
   - enzyme sequence。
   - active site。
   - cofactors。

2. 在没有 enzyme sequence 或具体酶名时，仅预测 EC/subclass 是否有用？
3. 对合成规划来说，预测“某类酶可能催化”是否已经足够，还是必须具体到 enzyme candidate？
4. 当前最需要避免的假阳性是哪类？

### 7.4 关于实验验证

1. 是否应该建立一个专家小规模评审集？
2. 专家更关心路线完整性，还是单个 enzyme bridge 的合理性？
3. 如果先展示 top-level enzyme bridge，而不是完整路线，是否更容易获得认可？
4. 哪类药物或天然产物最适合作为 case study？

## 8. 当前可选战略

### 方案 A：放弃级联，转做复杂分子逆合成候选池增强

优点：

1. 当前 AiZ + RetroKNN 数据支持 proposal coverage 提升。
2. 问题清晰：候选池覆盖不足和搜索噪声。
3. 工程可控。

缺点：

1. 与普通 retrosynthesis 差异化不足。
2. 很难解释为什么这是 chemo-enzymatic 项目。
3. 酶催化特色会被弱化。

### 方案 B：继续做 one-pot cascade condition planner

优点：

1. 保留原始级联叙事。
2. 条件、辅因子、酶毒化等规则可以继续使用。

缺点：

1. 专家已经认为现实意义有限。
2. 数据少，训练难。
3. 很容易变成事后审计，而不是生成能力。

### 方案 C：转向 enzyme-aware chemo-enzymatic bridge

优点：

1. 保留级联思想，但避开 one-pot 争议。
2. 与专家指出的虚拟级联数据思路一致。
3. 酶-底物预测是明确缺口。
4. 可以用 weak supervision 扩大数据。
5. 和普通多步逆合成有差异化。

缺点：

1. 需要重新组织数据和 benchmark。
2. 需要真正做 enzyme-substrate 模块。
3. 虚拟数据质量需要严格分层，否则会引入大量假阳性。

当前建议优先选择方案 C，并把方案 A 作为支撑技术。

## 9. 建议下一阶段工作

### 9.1 立即停止或降级的方向

1. 不再把 one-pot 条件兼容作为主创新。
2. 不再继续训练仅基于 3K cascade 正样本的 ranker。
3. 不再无差别打开所有 proposal provider。
4. 不再只用 solved rate 判断模型好坏。

### 9.2 立即启动的方向

1. 构建 `virtual_chemoenzymatic_bridge_pack`。
2. 统计化学产物和酶底物的可连接规模。
3. 构造 enzyme-substrate positive/negative pairs。
4. 建立 enzyme-substrate benchmark。
5. 训练或实现 `EnzymeBridgeRetriever`。
6. 训练或实现 `EnzymeFeasibilityVerifier`。
7. 将 enzyme proposal 作为 gated sidecar 接入搜索。

### 9.3 第一阶段可交付结果

1. 数据报告：
   - 化学反应数。
   - 酶反应数。
   - exact bridge 数。
   - similarity bridge 数。
   - EC 分布。
   - bridge confidence 分布。

2. 模型报告：
   - enzyme-substrate top-k。
   - EC top-k。
   - substrate feasibility AUC。
   - hard negative 表现。

3. 搜索报告：
   - native baseline。
   - native + gated enzyme bridge。
   - native + ungated sidecar。
   - search cost 对比。
   - bridge-supported route examples。

4. 专家评审材料：
   - 20 到 50 个 enzyme bridge case。
   - 每个 case 展示 substrate、product、EC、相似文献反应、反应中心。
   - 让专家判断是否“值得进入搜索”。

## 10. 当前暂定结论

1. 级联不是无意义，但 one-pot 条件兼容不是最有力的创新点。
2. 项目真正有价值的问题是化学和酶催化反应空间之间的桥接。
3. 酶-底物预测是目前最大短板，也是最可能形成创新的缺口。
4. 虚拟级联数据是可行方向，但必须定义为 weak connectivity data，而不是工艺路线 ground truth。
5. AiZynthFinder 和 RetroKNN 已证明能补 proposal 覆盖，但必须 gated 使用。
6. 下一步应从“级联条件规划器”转为“enzyme-aware chemo-enzymatic bridge planner”。

可以带给专家的核心问题是：

**如果我们把级联重新定义为化学中间体与酶底物空间之间的可连接性，而不是 one-pot 条件兼容，这是否是一个有意义、有创新性、且可通过虚拟数据训练的问题？**
