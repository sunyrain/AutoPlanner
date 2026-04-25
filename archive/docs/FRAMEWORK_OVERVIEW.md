# 工艺感知化学-酶逆合成规划：完整框架说明

> **日期**：2026-04-20  
> **目的**：从零开始讲清楚我们到底要做什么、为什么这么做、怎么做

---

## 第一部分：现有系统做了什么

### 1.1 逆合成规划的本质

给你一个目标分子，找到一条从商业可得原料合成它的路线。

```
目标分子: 一种药物中间体
    ↓ 拆解
中间体 A + 试剂 B
    ↓ 继续拆
原料 C + 原料 D (都能买到 → 完成)
```

这是一棵"逆向"的树：从产物往回拆，直到每个叶子都是能买到的原料。

### 1.2 ChemEnzyRetroPlanner 现有系统

已发表在 Nature Communications 2025，核心架构：

```
输入: 目标分子 SMILES
        ↓
┌───────────────────────────────────────────────────────┐
│  MCTS* 搜索引擎                                       │
│                                                       │
│  每一轮迭代:                                           │
│    1. 选择: 找到最有希望的未展开分子节点                  │
│    2. 展开: 调用 one-step 模型 → 得到候选反应            │
│    3. 评估: 计算代价，回传更新树                         │
│    4. 检查: 所有叶子都是可买原料? → 找到一条路线!         │
│                                                       │
│  one-step 模型（可同时用多个）:                          │
│    · template_relevance (pistachio/reaxys/bkms/...)   │
│    · ONMT Transformer (BioNav训练)                     │
│    · Graph retrosyn (D-MPNN)                          │
│    · MLP retrosyn                                     │
│                                                       │
│  每个模型返回:                                          │
│    · reactants: ["A.B", "C.D", ...]  ← 反应物SMILES   │
│    · scores:    [0.95, 0.87, ...]    ← 概率分数        │
│    · template:  ["SMARTS", None, ...] ← 反应模板(如有) │
│                                                       │
│  搜索只看: SMILES + 分数 + 原料是否可买                  │
│  搜索不看: 这步是有机还是酶、条件是什么、能不能级联       │
└───────────────────────────────────────────────────────┘
        ↓
输出: 多条成功路线 (按 pathway ranker 排序)
        ↓
┌───────────────────────────────────────────────────────┐
│  后处理标注 (全部在搜索完成之后)                         │
│                                                       │
│  对每条路线的每一步反应:                                 │
│    ① 有机/酶分类器 → "这步是有机反应 (95%)" 或          │
│                      "这步是酶催化反应 (87%)"           │
│    ② EC号预测器 → "推荐酶: EC 1.1.1.1 (ADH)"          │
│    ③ 条件预测器 (RCR/Parrot) → "T=65°C, 溶剂=DMF"     │
│    ④ EAsIFA → 酶活性位点预测                            │
└───────────────────────────────────────────────────────┘
        ↓
最终输出: 带标注的合成路线
```

### 1.3 关键事实

**搜索和标注是完全分离的。**

搜索过程中，`ReactionNode` 只携带：
- `cost`（-log(score)，纯数值）
- `template`（SMARTS 模板字符串，seq2seq 模型没有，为 None）
- `children`（反应物分子节点）

搜索**不知道**这步反应是有机还是酶催化的，不知道需要什么条件，不知道相邻两步能不能放在一个锅里。它只关心"拆解分数高不高"和"原料能不能买到"。

所有化学知识（有机/酶分类、条件、EC号）都是**搜索完成后贴上去的标签**。

---

## 第二部分：问题在哪

### 2.1 一个真实场景

假设搜索找到了这条 3 步路线：

```
目标分子
  ← Step 3: CALB 酶催化酯化 (水相, pH 7.5, 37°C)
    ← Step 2: Pd-催化 Suzuki 偶联 (DMF, 110°C, 惰性气氛)
      ← Step 1: ADH 酶催化还原 (水相, pH 7.0, 30°C, NADH)
```

现有系统会说："找到了！3 步，分数不错。" 然后后处理标注条件。

但化学家看到这条路线会立刻问：
- Step 1 和 Step 3 都是水相酶反应，条件接近，**能不能合并到一个锅里**？
- Step 2 是有机金属催化，110°C，DMF 溶剂，**跟酶反应完全不兼容**
- 最优操作方案是什么？一锅到底？分两锅？三锅各做？

**现有系统对此完全沉默。**

### 2.2 空白在哪

所有已有的逆合成规划系统（ASKCOS、AiZynthFinder、Levin/Coley 2022、ACERetro 2025、我们自己的 ChemEnzyRetroPlanner）都止步于：

```
"这是一条路线" + "每步的推荐条件"
```

没有任何系统回答：
- 哪些步骤可以合并操作？
- 合并时用什么操作模式？（同时反应？分阶段加料？空间隔离？）
- 整条路线需要几个"锅"？
- 不同路线之间，哪条更适合级联操作？

**这就是我们要填的空白。**

---

## 第三部分：我们要做什么

### 3.1 一句话定义

> **在现有逆合成规划系统的基础上，增加"路线级工艺评估"能力：自动判断相邻步骤的操作模式，划分最优级联分组，计算 pot-economy。**

### 3.2 我们不做什么

| 不做的事 | 为什么不做 |
|---------|-----------|
| ❌ 重写逆合成搜索引擎 | 已有 MCTS* 够用，不是我们的创新点 |
| ❌ 训练新的 one-step 逆合成模型 | 赛道太拥挤（RSGPT, ReactionT5, Enzyformer），且不是我们的优势 |
| ❌ 发明新的操作模式分类法 | 文献已有成熟共识（concurrent/sequential/compartmentalized） |
| ❌ 做端到端黑盒条件预测 | 数据不够，且规则更可靠可解释 |

### 3.3 核心产出

```
输入: MCTS* 搜索输出的一条路线
      （每步有: reactants SMILES, product SMILES, template, 来源模型）

输出:
  ① 每对相邻步骤的操作模式标签:
     concurrent / sequential / compartmentalized / incompatible
  
  ② 最优 pot 分组方案:
     [Step 1 + Step 2 → Pot 1 (concurrent)] → [Step 3 → Pot 2]
  
  ③ Pot-economy 分数:
     PotEconomy = 1 - num_pots / num_steps = 1 - 2/3 = 0.33
  
  ④ 每个判断的理由 (可解释):
     "Step 1-2: 同为水相酶催化, pH差0.5, T差7°C → concurrent"
     "Step 2-3: Pd催化(DMF,110°C) vs 酶(水相,37°C) → incompatible"
```

---

## 第四部分：怎么做（技术方案）

### 4.1 总体流程

```
                    现有系统（不改动）
                    ─────────────────
Step 0              目标分子
  │                     ↓
  │              MCTS* 搜索 + 多模型展开
  │                     ↓
  │              多条成功路线
  │                     ↓
  │              后处理标注（有机/酶分类、条件预测）
                    ─────────────────
                        ↓
                    我们新增的部分
                    ─────────────────
Step 1 ·        信息提取（从已有输出中提取）
  │                     ↓
Step 2 ·        规则引擎判断（操作模式预测）
  │                     ↓
Step 3 ·        Pot 分组优化
  │                     ↓
Step 4 ·        路线重排序（按 pot-economy）
                    ─────────────────
                        ↓
                    输出: 带工艺评估的路线排名
```

### 4.2 Step 1 — 信息提取：从哪里来？

这是你之前问的关键问题。让我逐项说明每个信息的来源：

```
每步反应已有的信息                来源                      可靠度
─────────────────────────────────────────────────────────────────
reactants SMILES                搜索输出                   ✅ 确定
product SMILES                  搜索输出                   ✅ 确定
template SMARTS                 搜索输出 (seq2seq为None)   ✅ 确定
来源模型名                       搜索输出                   ✅ 确定
  └─ "bkms_metabolic"           → 酶催化
  └─ "reaxys_biocatalysis"      → 酶催化
  └─ "pistachio"                → 有机
  └─ "bionav_one_step"          → 可能是酶
有机/酶分类                      现有后处理分类器            ✅ 已有
EC号 (如果是酶)                  现有后处理EC预测器          ✅ 已有
条件预测 (T, 溶剂, 催化剂)       现有 RCR/Parrot            ⚠️ 粗糙
```

**但这些信息对操作模式判断够不够？**

分两层看：

#### Layer A — 催化剂类型冲突检测（已有信息足够）

不需要任何额外预测，只需要：
- 有机/酶分类结果 → 已有
- 模板来源 → 已有（bkms = 酶, pistachio = 有机）
- 模板 SMARTS 中的特征 → 可解析（含 `[Pd]` → 钯催化）

就能做到：

```python
# 伪代码
if step_i.is_enzyme and step_j.uses_heavy_metal:
    return "incompatible"  # 重金属毒害酶

if step_i.is_grignard and step_j.is_aqueous_enzyme:
    return "incompatible"  # 无水 vs 水相

if step_i.is_enzyme and step_j.is_enzyme:
    return "likely_concurrent"  # 同类催化剂，可能兼容
```

这层判断**零额外成本**，只用现有输出。

#### Layer B — 条件兼容性精细判断（需要查找表）

要判断 concurrent vs sequential vs compartmentalized，需要知道大致条件范围：

```
"CALB 酯化"    → 水相/有机相均可, pH 6-8, T 30-60°C
"ADH 还原"     → 水相, pH 7-8, T 25-37°C, 需要 NADH
"Suzuki 偶联"  → 有机溶剂, T 80-120°C, 需要 Pd 催化剂, 惰性气氛
"Grignard"     → 无水 THF, T -78 到 0°C
```

这些信息从哪来？**反应类型 → 典型条件查找表（我们需要手工构建）。**

具体来说：

```
信息来源                    怎么获得                    工作量
──────────────────────────────────────────────────────────────
模板 → 反应类型名称         SMARTS 解析 + 命名规则库     中 (已有工具)
反应类型 → 典型条件范围     人工编写查找表               ~50-100条, 1-2周
酶 EC号 → 典型条件范围     BRENDA 数据库查询            可自动化
```

查找表示例：

```yaml
# reaction_type_conditions.yaml （我们要构建的核心资产之一）

CALB_transesterification:
  solvent_class: [organic, aqueous_organic]
  pH_range: [6.0, 9.0]
  T_range: [25, 70]
  atmosphere: ambient
  incompatible_with: [strong_acid, strong_base, heavy_metal_homogeneous]

Suzuki_coupling:
  solvent_class: [organic]
  T_range: [60, 130]
  atmosphere: inert
  requires: [Pd_catalyst, base]
  incompatible_with: [aqueous_enzyme, thiol_compounds]

ADH_reduction:
  solvent_class: [aqueous]
  pH_range: [6.5, 8.5]
  T_range: [20, 40]
  requires: [NADH_or_NADPH]
  cofactor_regen: [GDH, FDH, IPA]
  incompatible_with: [organic_solvent_high_conc, heavy_metal, strong_oxidant]
```

#### Layer C — ML 条件预测（锦上添花）

现有的 RCR/Parrot 条件预测器可以提供更精确的数值估计，作为 Layer B 判断的补充证据。但它**不是前提**。

### 4.3 Step 2 — 规则引擎：怎么判断操作模式

拿到 Layer A + B 的信息后，对每对相邻步骤 (i, j) 做如下判断：

```
                    Step i 条件范围
                    Step j 条件范围
                          ↓
              ┌──── 绝对不兼容？────┐
              │  (重金属+酶,         │
              │   无水+水相,         │
              │   强酸+酶, etc.)     │
              │                     │
             YES                   NO
              ↓                     ↓
         INCOMPATIBLE       ┌── 条件高度重叠？──┐
                            │  pH差≤1, T差≤10°C, │
                            │  同溶剂类, 催化剂   │
                            │  无互相失活         │
                            │                    │
                           YES                  NO
                            ↓                    ↓
                       CONCURRENT         ┌── 条件可调节？──┐
                                          │  pH差≤3, T差≤30°C,│
                                          │  或可中间加料/     │
                                          │  调pH/换气         │
                                          │                   │
                                         YES                 NO
                                          ↓                   ↓
                                     SEQUENTIAL        ┌── 有隔离方案？──┐
                                                       │  可双相? 可膜?   │
                                                       │  可包埋? 可流动?  │
                                                       │                  │
                                                      YES                NO
                                                       ↓                  ↓
                                                COMPARTMENTALIZED    INCOMPATIBLE
```

**操作模式定义（3+1，回归文献共识）：**

| 模式 | 含义 | 判据 |
|------|------|------|
| **Concurrent** | 所有催化剂和底物同时存在，条件不变 | 条件高度重叠，无互相失活 |
| **Sequential** | 同锅，完成一步后加料/调条件再下一步 | 条件有差异但可调节桥接 |
| **Compartmentalized** | 同体系内空间隔离（双相/膜/包埋/flow） | 催化剂不兼容但底物可跨相传递 |
| **Incompatible** | 必须分离纯化 | 无工程策略可弥合 |

### 4.4 Step 3 — Pot 分组优化

一条 N 步路线有 N-1 个相邻对。每对有一个操作模式标签。

```
例: 4 步路线
  Step 1 ←(concurrent)→ Step 2 ←(incompatible)→ Step 3 ←(sequential)→ Step 4

最优分组:
  Pot 1: [Step 1 + Step 2] (concurrent)
  Pot 2: [Step 3 + Step 4] (sequential)
  
  PotEconomy = 1 - 2/4 = 0.5
```

分组规则：
- concurrent/sequential/compartmentalized 的相邻步骤可以合并到同一个 pot
- incompatible 的相邻步骤必须分开
- 目标：最小化 pot 数量（最大化 pot-economy）

### 4.5 Step 4 — 路线重排序

MCTS* 通常找到多条路线。现有系统按"路径代价"排序。

我们增加一个维度：**在代价相近时，优先推荐 pot-economy 更高的路线。**

```
Route A: 5 步, 代价 3.2, 3 pots, pot-economy = 0.40
Route B: 4 步, 代价 3.5, 2 pots, pot-economy = 0.50  ← 可能更实用
Route C: 3 步, 代价 4.1, 3 pots, pot-economy = 0.00  ← 每步都分离
```

---

## 第五部分：数据从哪来

### 5.1 构建查找表（最核心的工作）

```
反应类型 → 典型条件范围 → 不兼容列表
```

来源：
- **BRENDA**: 85,601 酶反应的 pH/T/cofactor（可自动提取统计值）
- **文献综述**: RSC review (d3cs00595j) 中 400+ 级联案例的条件总结
- **化学常识**: Grignard 无水、Suzuki 惰性气氛等（教科书知识）

工作量：~50-100 条核心规则，1-2 周。

### 5.2 验证数据集（评估系统准确率）

就是文献提取团队已经提供的数据：
- 255-364 条级联反应案例
- 94% 有兼容性信息
- 需要重新映射为 4 类操作模式

用途：
- **定量评估**: 对已知级联案例，系统能否正确预测操作模式？
- **消融实验**: Layer A only vs A+B vs A+B+C 的准确率对比
- **论文 benchmark**: 这是第一个带操作模式标注的级联反应数据集

### 5.3 不需要的数据

| 不需要 | 为什么 |
|--------|--------|
| 训练逆合成模型的数据 | 不重写逆合成引擎 |
| RXNGraphormer 预训练数据 | 如果不训练 embedding，不需要 |
| 大规模 ML 兼容性训练集 | MVP 用规则，不用 ML |

---

## 第六部分：开发计划

### Phase 0 · 基础设施 (Week 1-2)

```
目标: 能跑通现有系统，拿到搜索结果
  ├─ 搭建开发环境
  ├─ 跑通 MCTS* 搜索 + 后处理标注
  └─ 验证: 输入一个目标分子 → 拿到带标注的路线
```

### Phase 1 · 规则引擎 MVP (Week 2-4)

```
目标: 对一条路线，输出操作模式 + pot 分组
  ├─ 构建 reaction_type → conditions 查找表 (50-100条)
  ├─ 实现 Layer A: 催化剂类型冲突检测
  ├─ 实现 Layer B: 条件范围兼容性判断
  ├─ 实现操作模式决策树
  ├─ 实现 pot 分组算法
  └─ 验证: 用提取团队的高质量案例 (score≥60) 测试准确率
```

### Phase 2 · 集成与评估 (Week 4-6)

```
目标: 端到端系统 + 定量评估
  ├─ 将规则引擎接入现有系统的后处理流程
  ├─ 路线重排序 (加入 pot-economy 维度)
  ├─ 在验证集上测准确率 (操作模式预测 accuracy)
  ├─ 消融实验: with/without 各层级
  └─ 与 ChemEnzyRetroPlanner 对比: 同一目标分子，路线质量差异
```

### Phase 3 · 搜索集成 (Week 6-8)

```
目标: 将级联信号回灌搜索 (cascade-aware MCTS)
  ├─ 扩展搜索状态: state = (mol_tree, pot_context)
  ├─ 在展开时计算 cascade_bonus (用规则, ~ms级)
  ├─ soft reward: value = base_value + λ * cascade_bonus
  └─ 实验: cascade-aware vs 纯后处理 的路线质量对比
```

### Phase 4 · 论文 (Week 8-10)

```
目标: 写论文
  ├─ 正式化操作模式预测任务定义
  ├─ 发布 CascadeCompat 数据集
  ├─ 实验: 操作模式预测准确率 + pot-economy 提升
  └─ 案例: 文献已知级联路线的重新发现
```

---

## 第七部分：论文卖什么

**不卖**:
- ~~"我们发明了操作模式分类"~~ — 文献已有
- ~~"我们构建了第一个混合逆合成系统"~~ — 已有很多

**卖**:
1. **首次将操作模式预测形式化为 CASP 预测目标** — 无人做过
2. **首个带操作模式标注的级联反应 benchmark 数据集** — 数据贡献
3. **规则 + 条件查找表 → 操作模式预测系统** — 方法贡献
4. **Cascade-aware MCTS 搜索** — 搜索贡献
5. **Pot-economy 作为路线质量新维度** — 评估贡献

---

## 第八部分：和之前的文档对比，砍掉了什么

| 之前方案中有、现在砍掉的 | 为什么砍 |
|------------------------|---------|
| RXNGraphormer embedding | MVP 不需要反应 embedding，规则引擎直接用催化剂类型+条件查找表 |
| ConditionHead ML 训练 | 现有 RCR/Parrot 已有粗糙预测，查找表更可靠；ML版降级为可选增强 |
| ReactionEvaluator 多头网络 | 过度工程，MVP 用规则 |
| TypeHead / FeasibilityHead | 现有分类器已有，不需要重训 |
| Enzyformer A/B test | 不做 embedding 就不需要 |
| "Telescoped" 操作模式 | 数据为零，文献不单独列出，定义模糊 |
| "5 类操作模式" | 回归文献共识 3+1 |

**最小化原则：只做必须做的，其他全部砍掉或降级为 future work。**

---

## 附录：一个完整的端到端例子

```
用户输入: 目标分子 SMILES = "CC(=O)OC1CCCCC1"  (环己基乙酸酯)

=== 现有系统输出 ===

Route 1 (3 steps, cost=2.8):
  Step 3: 目标 ← CALB酯化(环己醇 + 乙酸乙烯酯)     [bkms模板, 酶]
  Step 2: 环己醇 ← ADH还原(环己酮)                   [bkms模板, 酶]
  Step 1: 环己酮 ← PCC氧化(环己醇)                   [USPTO模板, 有机]
  后处理标注:
    Step 1: 有机反应(99%), T=25°C, solvent=DCM
    Step 2: 酶反应(95%), EC 1.1.1.1, T=30°C, pH=7.0
    Step 3: 酶反应(97%), EC 3.1.1.3, T=37°C, pH=7.5

=== 我们新增的输出 ===

操作模式分析:
  Step 1→2: PCC氧化(DCM,25°C) → ADH还原(水相,pH7,30°C)
            → 有机氧化剂(Cr⁶⁺) + 酶 → INCOMPATIBLE ❌
            理由: 重金属Cr⁶⁺对酶有毒

  Step 2→3: ADH还原(水相,pH7,30°C) → CALB酯化(水相/有机相,pH7.5,37°C)
            → 两步均为水相酶, pH差0.5, T差7°C → CONCURRENT ✅
            理由: 条件高度兼容, 可同锅同时反应

Pot 分组:
  Pot 1: [Step 1] (PCC氧化, 有机相)
  Pot 2: [Step 2 + Step 3] (ADH还原 + CALB酯化, 水相级联)
  
  PotEconomy = 1 - 2/3 = 0.33
  
  工艺建议:
    "Step 2+3 可设计为 one-pot concurrent 酶级联。
     需要 NADH 再生系统（推荐 GDH + glucose）。
     Step 1 使用 Cr 基氧化剂，必须与酶步骤完全分离。
     建议考虑替代 Step 1 为 Oppenauer 氧化或生物氧化以实现全级联。"
```
