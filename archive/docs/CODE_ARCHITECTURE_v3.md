# 级联兼容性逆合成规划器 · 代码架构方案 v3.1

> **版本**：v3.1 — 统一模型 + 全新代码  
> **日期**：2026-04-19  
> **替代**：v3.0（复制旧代码方案，废弃）、v2.0（RGFM 大模型方案，废弃）  
> **与 ChemEnzyRetroPlanner 关系**：继承科学思路和预训练权重，代码全部重写  

---

## 0. 为什么要全部重写

### 0.1 ChemEnzyRetroPlanner 的代码问题

| 问题 | 具体表现 |
|------|---------|
| **碎片化** | 9 个独立 `setup.py`、各自的依赖树，包之间无共享 |
| **微服务沼泽** | parrot 要单独起 TorchServe Docker，template_relevance 要 HTTP 调 9410 端口，条件预测可选 RCR 或 parrot 两套完全不同的接口 |
| **模型割裂** | 有机/酶分类器、EC 推荐器、条件预测器、可行性过滤器——4 个独立模型分别训练，不共享任何表示 |
| **搜索与表征脱耦** | MCTS 搜索完毕后才做反应类型分类和条件预测——搜索根本不知道它在规划一条酶催化路线 |
| **无级联概念** | 路线只是"一棵树"，完全没有步骤间兼容性的数据结构 |

**复制这些代码 = 继承全部技术债。**

### 0.2 v2.0 RGFM 的教训

v2.0 想用一个 150M 的基础模型统一一切（包括单步逆合成），问题是：
- 在 USPTO-50K Top-1 上和 LocalRetro/Graphormer 硬拼没有差异化
- 训练成本高（4×A100 几个月），但不产生独特价值
- "反应基础模型"听起来很大，却没有回答"你到底比别人好在哪？"

**但 v2.0 有一个对的想法**：把多个独立模型统一到一个共享骨干 + 多头的架构中。这个思路本身没问题，问题是它试图连单步逆合成都自己做。

### 0.3 v3.1 的定位

**保留 v2.0 "统一模型"的正确思路，砍掉 retro head，加上级联兼容性。**

```
v2.0 RGFM: 共享骨干 + [Retro, Forward, Filter, Condition, Enzyme, Value] ← 6 headv3.1 URC:  共享骨干 + [Type, Condition, Enzyme, Feasibility, Cascade, Value] ← 6 heads
                                                              ↑ 去掉了 Retro/Forward
                                                              ↑ 加了 Cascade（核心创新）
```

单步逆合成？不是我们的问题。用现有模型就好。  
级联兼容性？全世界没人做。这才是我们的问题。

---

## 1. 系统三层架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Layer 3 · API & Orchestration                     │
│    CascadePlanner: 编排搜索→表征→兼容性→评分的完整管线               │
└────────────────────────────────────┬────────────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌─────────────────┐   ┌──────────────────────────┐   ┌──────────────────┐
│ Layer 2 · Search │   │  Layer 1 · Model Layer   │   │  Layer 1 · Data  │
│                  │   │                          │   │                  │
│ MCTS* 搜索引擎  │──▶│ URC (统一反应表征器)      │   │ CascadeDB        │
│ - 级联感知       │   │  ├─ Type Head            │   │ - 级联文献数据    │
│ - value 指引     │   │  ├─ Condition Head       │   │ - 步骤对训练集    │
│ - 约束剪枝       │   │  ├─ Enzyme Head          │   │ - 规则知识库      │
│                  │   │  ├─ Feasibility Head     │   │                  │
│                  │   │  ├─ Cascade Feature Head  │   │ StockDB          │
│                  │   │  └─ Value Head           │   │ - 可购买分子库    │
│                  │   │                          │   │                  │
│                  │   │ Compatibility Model       │   │                  │
│                  │   │  = f(cascade_feat_i,      │   │                  │
│                  │   │      cascade_feat_j)      │   │                  │
│                  │   │                          │   │                  │
│                  │   │ SingleStepExpander        │   │                  │
│                  │   │  (可插拔后端)              │   │                  │
└─────────────────┘   └──────────────────────────┘   └──────────────────┘
```

**三个设计原则**：
1. **化学理解留给统一模型**——一个骨干学所有反应知识，不再 4 个模型各自为战
2. **搜索留给经典算法**——MCTS* 清晰可控，但从一开始就感知级联兼容性
3. **单步逆合成是可插拔的**——我们不和 LocalRetro 竞争，用现成的就行

---

## 2. Layer 1a · URC（统一反应表征器）

**这是取代 ChemEnzyRetroPlanner 5 个独立模型的核心组件。**

### 2.1 为什么需要统一

ChemEnzyRetroPlanner 的 5 个模型各自为政：

```
旧系统：
  rxn_smiles → OrganicEnzymeRXNClassifier (RXNFP-BERT)  → type
  rxn_smiles → ConditionPredictor (MLP, 16384-FP)       → conditions
  rxn_smiles → EnzymeRXNClassifier (RXNFP-BERT)         → EC number
  rxn_smiles → FilterModel (MLP, 2048-FP)               → feasibility
  mol_smiles → ValueMLP (MLP, 2048-FP)                  → synthesis distance
  
  5 个模型，5 种输入编码，5 个训练流程，零知识共享。
```

新系统：

```
URC：
  rxn_smiles → Shared Reaction Encoder → reaction_repr (d=512)
                   ├── TypeHead(reaction_repr)      → organic/enzymatic
                   ├── ConditionHead(reaction_repr)  → T, pH, solvent, cofactors
                   ├── EnzymeHead(reaction_repr)     → EC number (hierarchical)
                   ├── FeasibilityHead(reaction_repr)→ feasibility score
                   ├── CascadeHead(reaction_repr)    → cascade_feature (d=128)
                   └── ValueHead(mol_repr)           → synthesis distance
  
  1 个编码器，1 次前向传播，6 组输出，全部任务共享化学知识。
```

### 2.2 Shared Reaction Encoder 架构

**选择：D-MPNN on Reaction Graph**（而非 SMILES Transformer）

理由：
- 反应的本质是原子和键的变化（断键/成键），图是自然表示
- D-MPNN 在 MoleculeNet/反应预测上有坚实的 track record（Chemprop/DMPNN 系列）
- 相比 SMILES Transformer（把分子当字符串处理），图模型有更好的归纳偏置
- 比 Graphormer 轻量得多（不需要全注意力），适合我们的规模

```python
class ReactionEncoder(nn.Module):
    """
    输入：reactants>>product（SMILES）
    处理：
      1. 分别构建 reactant 分子图和 product 分子图
      2. D-MPNN 消息传递 → 原子级表示
      3. 反应差异池化：product_repr - reactant_repr（捕获化学变化）
      4. 可选：拼接全局 Morgan FP 作为补充特征
    输出：reaction_repr (batch, d_model)，d_model=512
    """
    def __init__(self, d_atom=128, d_bond=64, n_layers=4, d_model=512):
        self.atom_encoder = AtomFeaturizer()      # 原子特征：元素、电荷、杂化等
        self.bond_encoder = BondFeaturizer()      # 键特征：类型、环内/外、立体等
        self.mpnn = DMPNN(d_atom, d_bond, n_layers)  # 消息传递
        self.diff_pool = ReactionDiffPool(d_model)    # 反应差异池化
        self.fp_proj = nn.Linear(2048, d_model)       # Morgan FP 投影（可选）
```

**参数规模估计**：~20-30M（backbone）+ ~5M×6（heads）≈ **50M 总参**

### 2.3 六个任务头

| Head | 输入 | 输出 | 损失函数 | 训练数据 |
|------|------|------|---------|---------|
| **TypeHead** | reaction_repr | P(organic), P(enzymatic) | BCE | USPTO(organic) + ECReact/BRENDA(enzymatic), ~2M |
| **ConditionHead** | reaction_repr | T(regression), pH(regression), solvent(multi-label), catalyst(multi-label) | MSE + BCE | ORD ~2M（含条件标注） |
| **EnzymeHead** | reaction_repr | EC level 1→2→3→4 (hierarchical) | Hierarchical CE | BRENDA + ECReact ~250K |
| **FeasibilityHead** | reaction_repr | P(feasible) | BCE | 正例=真实反应，负例=random template → ~2M |
| **CascadeHead** | reaction_repr | cascade_feature (d=128) | 对比学习 + 下游兼容性任务 | 级联提取数据 + 规则增强 |
| **ValueHead** | mol_repr (从 product 图提取) | synthesis_distance (regression) | MSE | PaRoutes/RetroStar solved routes ~100K |

**关键设计**：CascadeHead 输出的不是最终分数，而是一个稠密特征向量。兼容性预测由 Compatibility Model 完成（见 §3）。这样设计的好处：
- CascadeHead 学习的是"描述这个反应在级联中重要特性"的表示（pH 要求、溶剂耐受、辅因子需求等）
- 两个这样的表示可以被 Compatibility Model 组合来判断兼容性
- 表示可以被缓存、复用、可视化

### 2.4 多任务训练策略

```
Stage 1 · 自监督预训练 on 大规模反应数据（~1 周，2×A100）
  - 任务：反应中心定位（掩码原子/键恢复）、化学合法性判别
  - 数据：USPTO-Full + ORD + ECReact 合并去重 ~4M 反应
  - 目的：让骨干学会通用化学知识

Stage 2 · 多任务有监督微调（~3 天，2×A100）
  - 6 个 head 联合训练，动态权重（GradNorm 或 uncertainty weighting）
  - Type/Condition/Feasibility 用大数据集先热身
  - Enzyme head 数据少，给更高权重
  - CascadeHead 此阶段用规则增强数据（真实级联数据可能尚未到位）

Stage 3 · 级联数据微调（数据到位后，~1 天，1×GPU）
  - 冻结骨干，只微调 CascadeHead + Compatibility Model
  - 真实级联提取数据 + 规则增强
```

### 2.5 与 RGFM v2.0 的关键区别

| | RGFM v2.0 | URC v3.1 |
|---|---|---|
| 做单步逆合成？ | ✅ Retro head + Forward head | ❌ 不做，用现有模型 |
| 参数量 | 80-150M | ~50M |
| 训练成本 | 4×A100 × 7 周 | 2×A100 × 2 周 |
| 在 USPTO-50K 刷榜？ | 目标 Top-1≥60% | 不刷，没意义 |
| 级联兼容性？ | ❌ 没有 | ✅ 核心创新 |
| 何时可用？ | ~3 个月（纯训练） | ~3 周（含数据准备） |

---

## 3. Layer 1b · Cascade Compatibility Model

**这是整个项目的学术贡献核心。**

### 3.1 任务定义

```
输入：
  (step_i, step_j) — 路线中两个相邻反应步骤
  每个 step 已通过 URC 获得 cascade_feature (d=128) + 条件预测 + 类型分类

输出：
  - compatible: bool          — 是否可 one-pot
  - score: float [0,1]        — 兼容概率
  - failure_modes: Dict[str, float]  — 5 大失败模式各自概率
      pH_conflict, solvent_incompatibility, mutual_inactivation,
      cofactor_competition, product_inhibition
```

### 3.2 模型

```python
class CascadeCompatibilityModel(nn.Module):
    """
    Siamese 架构：两个步骤共享同一个 URC 编码器，
    然后通过交互层预测兼容性。
    """
    def __init__(self, d_cascade=128, d_cond=64, d_hidden=256):
        # 条件编码器：把结构化条件（T, pH, solvent...）编码为向量
        self.cond_encoder = ConditionEncoder(d_cond)
        
        # 交互层
        self.interaction = nn.Sequential(
            nn.Linear(2 * (d_cascade + d_cond) + (d_cascade + d_cond), d_hidden),
            #         concat(i, j)           + |i - j|           → 融合
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
        )
        
        # 输出头
        self.score_head = nn.Linear(d_hidden, 1)       # sigmoid → 兼容概率
        self.failure_head = nn.Linear(d_hidden, 5)     # sigmoid × 5 → 失败模式概率
    
    def forward(self, cascade_feat_i, cond_i, cascade_feat_j, cond_j):
        z_i = torch.cat([cascade_feat_i, self.cond_encoder(cond_i)], dim=-1)
        z_j = torch.cat([cascade_feat_j, self.cond_encoder(cond_j)], dim=-1)
        z = torch.cat([z_i, z_j, torch.abs(z_i - z_j)], dim=-1)
        h = self.interaction(z)
        score = torch.sigmoid(self.score_head(h))
        failure = torch.sigmoid(self.failure_head(h))
        return score, failure
```

**参数量**：~2M。极轻量——真正的化学理解在 URC 骨干里，这里只做兼容性判断。

### 3.3 训练数据

| 来源 | 类型 | 规模估计 |
|------|------|---------|
| 级联文献提取（CASCADE_EXTRACTION） | 正例（成功 one-pot 步骤对） | ~2000-5000 |
| 文献报告的失败尝试 | 负例（明确不兼容） | ~500-1000 |
| **规则增强负例** | 确定性不兼容（Pd+酶、THF+酶、pH 差>3 等） | **~10000+** |
| **条件冲突负例** | URC 预测的条件互相矛盾 | **~5000** |
| RetroBioCat 52 条级联 | 补充正例 | ~100 |

规则增强是关键：化学常识告诉我们很多确定性不兼容组合，无需等待文献提取就能开始训练。

### 3.4 规则基线（硬编码知识）

即使 ML 模型训不好，硬编码规则也提供有意义的输出（现有工具的零级联分析 → 规则打分已是巨大进步）：

```python
# 确定性不兼容规则（置信度 > 0.8）
HARD_INCOMPATIBLE = {
    ("metal_catalyst:Pd|Ni|Ru", "enzyme:*"):       "mutual_inactivation",
    ("pH:<4", "enzyme:*"):                          "ph_conflict",
    ("pH:>10", "enzyme:*"):                         "ph_conflict",
    ("solvent:THF|DCM|hexane", "enzyme:*"):         "solvent_incompatibility",
    ("reagent:LDA|BuLi|NaH", "enzyme:*"):          "mutual_inactivation",
    ("temp:>80", "enzyme:most"):                    "mutual_inactivation",
}

# 已知兼容组合（置信度 > 0.8）
KNOWN_COMPATIBLE = {
    ("enzyme:KRED", "enzyme:GDH"):                  "cofactor_recycling_NADPH",
    ("enzyme:transaminase", "enzyme:LDH"):          "cofactor_recycling_PLP",
    ("enzyme:lipase", "enzyme:lipase"):             "same_family_compatible",
    ("enzyme:laccase", "chemical:mild_oxidation"):  "mild_conditions_ok",
}
```

---

## 4. Layer 1c · SingleStepExpander（可插拔单步逆合成）

**我们不重写单步逆合成模型，定义干净的接口，接入现有工具。**

### 4.1 统一接口

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ExpansionResult:
    """单步逆合成的返回结果"""
    reactants: list[str]        # 每个元素是 "smi1.smi2" 格式的反应物组合
    scores: list[float]         # 对应每组反应物的置信度
    templates: list[str | None] # 对应的模板（有的模型没有模板）
    reaction_smiles: list[str]  # "reactants>>product" 格式

class SingleStepExpander(ABC):
    @abstractmethod
    def expand(self, product_smi: str, topk: int = 50) -> ExpansionResult:
        """给定产物 SMILES，返回 topk 组反应物"""
        ...
```

### 4.2 可选后端

| 后端 | 说明 | 数据覆盖 | 集成方式 |
|------|------|---------|---------|
| **TemplateBased** | MLP/GNN 模板匹配 | 有机为主（USPTO 模板） | 加载预训练权重 |
| **Seq2Seq** | Transformer 翻译 | 酶催化（BioNavi 数据） | 加载 ONMT 权重 |
| **RetroRules** | 117万酶促反应模板 | 酶催化 | REST API |
| **ExternalAPI** | ASKCOS / AiZynthFinder | 有机为主 | HTTP 调用 |
| **EnsembleExpander** | 组合多个后端 | 有机 + 酶催化 | 加权融合 |

**MVP 路径**：先用 `TemplateBased`（加载 ChemEnzyRetroPlanner 已训好的权重）+ `Seq2Seq`（加载已训好的 ONMT）。代码是新的，但模型权重复用——这不是复制代码，是使用研究产物。

**长期目标**：训一个统一的单步模型，合并有机 + 酶催化模板空间（这是一个额外的论文贡献，但不是 MVP 阶段的焦点）。

---

## 5. Layer 2 · 级联感知搜索引擎

**全新实现 MCTS*，但从一开始就理解级联概念。**

### 5.1 与旧搜索器的区别

| | ChemEnzyRetroPlanner 旧搜索 | 新搜索 |
|---|---|---|
| 反应节点 | 只存 cost 和 template | 存 reaction_type, conditions, cascade_features |
| 搜索目标 | 找到路线就行 | 找到**级联友好**的路线 |
| Value 函数 | 纯深度启发式 | URC ValueHead（可选：加级联兼容性估计） |
| 扩展后处理 | 无 | 每个新步骤立即做 URC 推理，获得类型+条件+级联特征 |
| 路线评分 | 找到就是赢 | cascade_score 加权排序 |

### 5.2 搜索流程

```
1. target_mol → SingleStepExpander.expand() → 候选 reactant_lists
2. 对每个候选反应：
   rxn_smiles → URC.encode() → reaction_repr
   reaction_repr → URC.type_head() → organic/enzymatic
   reaction_repr → URC.condition_head() → T, pH, solvent...
   reaction_repr → URC.cascade_head() → cascade_feature
3. 比较新步骤的 cascade_feature 与父节点路径上的已有步骤
   → CascadeCompatibilityModel.predict() → 兼容性估计
4. 调整 reaction_node.cost：
   adjusted_cost = base_cost - α * compatibility_bonus
   （兼容的路径成本更低，MCTS 优先探索）
5. 正常 MCTS 回溯 + value 更新
6. 搜索完成后：CascadeRouteScorer 对所有成功路线做精确评分 + pot 分配
```

### 5.3 节点设计

```python
@dataclass
class MolNode:
    smiles: str
    value: float
    is_known: bool           # 是否可购买
    depth: int
    children: list['ReactionNode']

@dataclass
class ReactionNode:
    rxn_smiles: str
    cost: float
    template: str | None
    # ---- 以下是新增的，旧系统没有 ----
    reaction_type: str       # "organic" | "enzymatic"
    conditions: dict         # {T, pH, solvent, catalyst, cofactors}
    cascade_feature: Tensor  # URC CascadeHead 输出 (d=128)
    ec_number: str | None    # 酶步的 EC 号
    compatibility_with_parent: float  # 与路径上前一步的兼容性分数
    children: list[MolNode]
```

### 5.4 路线评分与 pot 优化

```python
class CascadeRouteScorer:
    def score(self, route: SynRoute) -> CascadeScore:
        """
        1. 提取路线中所有反应步骤（线性化）
        2. 对每对相邻步骤计算精确兼容性分数
        3. 动态规划求最优 pot 划分（最小化 pot 数，满足兼容性阈值）
        4. 输出综合评分
        """
        steps = route.linearize()  # 会有分支的情况，需要处理
        pairs = [(steps[i], steps[i+1]) for i in range(len(steps)-1)]
        
        pairwise = [self.compatibility_model(s_i, s_j) for s_i, s_j in pairs]
        pot_assignment = self.optimize_pots(pairwise, threshold=0.5)
        
        return CascadeScore(
            cascade_score=...,       # 综合分
            num_pots=max(pot_assignment) + 1,
            pot_assignment=pot_assignment,
            pairwise=pairwise,
            bottleneck=min(pairwise, key=lambda x: x.score),
        )
```

---

## 6. Layer 3 · API 编排

```python
class CascadePlanner:
    """统一入口。取代 ChemEnzyRetroPlanner 的 RSPlanner。"""
    
    def __init__(self, config: dict):
        self.urc = UnifiedReactionCharacterizer.from_config(config)
        self.compatibility_model = CascadeCompatibilityModel.from_config(config)
        self.expander = build_expander(config)    # 按配置选择后端
        self.searcher = MCTSSearcher(config, self.urc, self.expander)
        self.scorer = CascadeRouteScorer(self.compatibility_model)
        self.stock = StockDB.from_config(config)
    
    def plan(self, target_smi: str, constraints: dict = None) -> PlanResult:
        """
        完整规划流程：
        1. 搜索 → 多条候选路线
        2. 每条路线的每步已经有 URC 表征（搜索中完成）
        3. CascadeRouteScorer 精确评分
        4. 按 cascade_score 排序
        5. 返回
        """
        search_result = self.searcher.search(
            target_smi, 
            known_mols=self.stock,
            constraints=constraints,
        )
        
        scored_routes = []
        for route in search_result.all_routes:
            cascade_score = self.scorer.score(route)
            scored_routes.append((route, cascade_score))
        
        scored_routes.sort(key=lambda x: x[1].cascade_score, reverse=True)
        
        return PlanResult(
            success=len(scored_routes) > 0,
            routes=scored_routes,
            search_stats=search_result.stats,
        )
```

---

## 7. 目录结构

```
CascadePlanner/
├── pyproject.toml                      # 单一包，不要 9 个 setup.py
├── environment.yml
├── README.md
├── configs/
│   ├── default.yaml                    # 默认配置
│   ├── model/                          # 模型配置
│   └── experiment/                     # 实验配置
│
├── cascade_planner/                    # 主包（唯一的包）
│   ├── __init__.py
│   ├── api.py                          # CascadePlanner 主入口
│   │
│   ├── models/                         # 所有模型在一个模块下
│   │   ├── __init__.py
│   │   ├── reaction_encoder.py         # D-MPNN 反应编码器（共享骨干）
│   │   ├── urc.py                      # UnifiedReactionCharacterizer（6 heads）
│   │   ├── compatibility.py            # CascadeCompatibilityModel
│   │   ├── heads/                      # 任务头
│   │   │   ├── type_head.py
│   │   │   ├── condition_head.py
│   │   │   ├── enzyme_head.py
│   │   │   ├── feasibility_head.py
│   │   │   ├── cascade_head.py
│   │   │   └── value_head.py
│   │   └── featurizer.py              # 原子/键/分子 特征化
│   │
│   ├── search/                         # 搜索引擎（全新实现）
│   │   ├── __init__.py
│   │   ├── mcts.py                     # MCTS* 主循环
│   │   ├── tree.py                     # AND-OR 树
│   │   ├── nodes.py                    # MolNode + ReactionNode
│   │   ├── route.py                    # SynRoute 表示
│   │   └── constraints.py             # 约束 DSL（可选）
│   │
│   ├── expansion/                      # 单步逆合成（可插拔）
│   │   ├── __init__.py
│   │   ├── interface.py                # SingleStepExpander ABC
│   │   ├── template_based.py           # 模板匹配后端
│   │   ├── seq2seq.py                  # Transformer 后端
│   │   ├── api_backend.py              # 外部 API 后端
│   │   └── ensemble.py                 # 集成后端
│   │
│   ├── scoring/                        # 路线评分
│   │   ├── __init__.py
│   │   ├── cascade_scorer.py           # CascadeRouteScorer
│   │   ├── pot_optimizer.py            # pot 划分（DP/贪心）
│   │   └── rules.py                    # 硬编码兼容性规则
│   │
│   ├── data/                           # 数据处理
│   │   ├── __init__.py
│   │   ├── cascade_db.py              # 级联数据存储 (SQLite)
│   │   ├── stock_db.py                # 可购买分子库
│   │   ├── reaction_dataset.py         # 训练数据集
│   │   └── pair_generator.py           # 步骤对生成（训练用）
│   │
│   └── utils/                          # 工具
│       ├── __init__.py
│       ├── chem.py                     # SMILES 处理、规范化
│       ├── mol_graph.py                # 分子图构建（for D-MPNN）
│       └── viz.py                      # 路线可视化
│
├── training/                           # 训练脚本
│   ├── train_urc.py                    # 多任务训练 URC
│   ├── train_compatibility.py          # 兼容性模型训练
│   ├── pretrain_encoder.py             # 自监督预训练
│   ├── data_prep/
│   │   ├── prepare_reaction_data.py    # USPTO/ORD/ECReact 清洗合并
│   │   ├── generate_rule_negatives.py  # 规则增强负例
│   │   └── import_cascade_data.py      # 导入文献提取数据
│   └── configs/                        # 训练配置
│
├── evaluation/                         # 评估
│   ├── eval_characterizer.py           # URC 各头评估
│   ├── eval_compatibility.py           # 兼容性预测评估
│   ├── eval_routes.py                  # 路线规划端到端评估
│   └── benchmarks/                     # 基准数据集
│
└── tests/                              # 单元测试
    ├── test_reaction_encoder.py
    ├── test_urc.py
    ├── test_compatibility.py
    ├── test_mcts.py
    ├── test_expander.py
    └── test_scorer.py
```

**对比旧系统**：
- 旧：9 个 `setup.py` + 主包 → 10 个安装目标
- 新：**1 个 `pyproject.toml`**，一次 `pip install -e .` 搞定
- 旧：packages 之间通过文件系统路径互相 import
- 新：统一的 `cascade_planner.*` 命名空间

---

## 8. 训练数据规划

### 8.1 URC 预训练 + 微调数据

| 数据集 | 规模 | 用于 | 获取难度 |
|--------|------|------|---------|
| USPTO-Full | ~1.8M 反应（有机） | Stage 1 预训练 + Type/Feasibility | ✅ 公开 |
| ORD (Open Reaction Database) | ~2M 反应（含条件标注） | Stage 1 + Condition head | ✅ 公开 |
| ECReact | ~250K 酶反应（带 EC） | Type + Enzyme head | ✅ 公开 |
| BRENDA 反应数据 | ~100K（带 EC + 部分条件） | Enzyme + Condition head | ✅ 公开 |
| RetroRules（Selenzyme） | ~117万 酶促反应模板 | Enzyme head 补充 | ✅ 公开 |
| PaRoutes solved routes | ~100K 路线 | Value head | ✅ 公开 |

### 8.2 级联兼容性数据

| 来源 | 规模 | 用于 |
|------|------|------|
| 文献提取（CASCADE_EXTRACTION） | ~2000-5000 步骤对 | Cascade head + Compatibility model 正例/负例 |
| 规则增强负例 | ~10000+ 合成对 | Compatibility model 负例 |
| URC 条件预测 → 条件冲突对 | ~5000 合成对 | Compatibility model 负例 |
| RetroBioCat 52 条级联 | ~100 步骤对 | 评估集 |

---

## 9. 开发路线图

### Phase 0 · 基础设施（第 1 周）
- [ ] 创建 CascadePlanner 仓库 + `pyproject.toml`
- [ ] 实现 `utils/chem.py`（SMILES 处理，canonicalize）
- [ ] 实现 `utils/mol_graph.py`（分子图构建，原子/键特征化）
- [ ] 实现 `data/stock_db.py`（可购买分子库加载）
- [ ] 实现 `expansion/interface.py`（SingleStepExpander ABC）
- [ ] 搭建测试框架

### Phase 1 · 搜索引擎（第 2-3 周）
- [ ] 实现 `search/nodes.py`（MolNode + ReactionNode，含级联字段）
- [ ] 实现 `search/tree.py`（AND-OR 树）
- [ ] 实现 `search/mcts.py`（MCTS* 主循环）
- [ ] 实现 `search/route.py`（SynRoute + route_to_dict + 线性化）
- [ ] 实现 `expansion/template_based.py`（加载已有权重验证搜索工作）
- [ ] 端到端验证：target → expand → search → routes

### Phase 2 · URC 统一模型（第 3-5 周）
- [ ] 实现 `models/featurizer.py`（原子/键特征化）
- [ ] 实现 `models/reaction_encoder.py`（D-MPNN 骨干）
- [ ] 实现 6 个 heads
- [ ] 实现 `models/urc.py`（组装）
- [ ] `training/data_prep/prepare_reaction_data.py`（USPTO+ORD+ECReact 清洗）
- [ ] `training/pretrain_encoder.py`（Stage 1 自监督）
- [ ] `training/train_urc.py`（Stage 2 多任务微调）

### Phase 3 · 级联兼容性（第 5-7 周，与数据提取并行）
- [ ] 实现 `scoring/rules.py`（硬编码规则基线）
- [ ] 实现 `data/cascade_db.py` + `data/pair_generator.py`
- [ ] 实现 `models/compatibility.py`
- [ ] 实现 `scoring/cascade_scorer.py` + `scoring/pot_optimizer.py`
- [ ] `training/data_prep/generate_rule_negatives.py`
- [ ] `training/train_compatibility.py`

### Phase 4 · 集成 + 评估（第 7-9 周）
- [ ] 实现 `api.py`（CascadePlanner 编排）
- [ ] 将兼容性信号集成到 MCTS 搜索中
- [ ] 基准评估
- [ ] 消融实验（rules only vs URC only vs URC+rules）

---

## 10. 预训练权重复用策略

**代码全部重写，但不需要所有模型从零训练。**

| 组件 | 权重来源 | 说明 |
|------|---------|------|
| ReactionEncoder 骨干 | **从零训练** | 这是我们的核心创新之一 |
| URC heads | **从零训练** | 多任务联合训练 |
| CascadeCompatibilityModel | **从零训练** | 核心创新 |
| SingleStepExpander (template) | **加载 ChemEnzyRetroPlanner 权重** | 非我们的创新点，用训好的 |
| SingleStepExpander (seq2seq) | **加载已有 ONMT 权重** | 同上 |

这就像用 ImageNet 预训练的 ResNet 做迁移学习——你不需要重新训练 ResNet，但你的任务代码是全新的。

---

## 11. 论文贡献定位

1. **Primary**：级联兼容性预测（首次提出任务 + 数据集 + 模型 + 评估）
2. **Secondary**：URC 统一反应表征器（多任务，有机+酶催化统一，替代碎片化方案）
3. **Tertiary**：级联感知 MCTS 搜索（在搜索阶段就考虑兼容性，而非事后标注）
4. **Dataset**：首个结构化化学-酶法级联催化数据集（从文献提取）

---

## 12. 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| D-MPNN 反应编码器从零训练效果差 | 中 | 回退到 RXNFP 作为骨干（预训练好的 BERT encoder），只需改 featurizer |
| 多任务训练一个 head 拖后腿 | 中 | GradNorm 动态权重 + 可随时冻结/解冻单个 head |
| 级联数据提取量不足 (<500) | 中 | 规则增强 + 规则基线先发 method paper，数据补齐后发 data+model paper |
| MCTS 重写 bug 多 | 低 | 算法成熟，有大量参考实现；Phase 1 充分测试 |
| 单步扩展模型加载权重格式不兼容 | 低 | 新的 Expander 只定义接口，具体加载逻辑封装在 adapter 中 |

---

## 13. 不做清单

- ❌ 在 USPTO-50K 上刷单步逆合成 SOTA（不是我们的战场）
- ❌ 训自己的 LLM（Copilot 用现成的，Phase 2+）
- ❌ 蛋白质结构预测 / 酶从头设计（交给 ESM/RFdiffusion）
- ❌ 建立大规模分子数据库基础设施（SQLite 足够）
- ❌ 复制 ChemEnzyRetroPlanner 的代码
# 级联兼容性逆合成规划器 · 代码架构方案 v3.0

> **版本**：v3.0 — 级联兼容性聚焦版  
> **日期**：2026-04-18  
> **替代**：v2.0 RGFM 版（已废弃，见 NEXT_GEN_ARCHITECTURE_PLAN.md）  
> **承前**：ChemEnzyRetroPlanner (Nature Comms 2025)  
> **核心定位**：化学-酶法级联催化路线规划 + 级联兼容性预测（现有工具的唯一空白）

---

## 0. 定位重申

经过诚实的文献调研和反复讨论，我们放弃了 RGFM 基础模型路线（在单步逆合成上与 LocalRetro/Graphormer 正面竞争没有差异化优势），聚焦于**世界上唯一尚无人做的问题**：

> **给定一条多步合成路线，预测其中哪些步骤可以 one-pot 级联执行，哪些必须分步操作，以及失败的原因是什么。**

五大失败模式（Sheldon & Woodley, Chem Rev 2018）：
1. **pH 冲突** — 酶最适 pH 与化学步 pH 不兼容
2. **溶剂不相容** — 有机溶剂使酶失活
3. **互相失活** — 金属催化剂毒害酶，或反之
4. **辅因子竞争** — 两个酶步需要同一辅因子
5. **产物抑制** — 中间体或副产物抑制下一步催化剂

**没有任何开源工具做这件事。** RetroBioCat 有 52 条级联数据但无兼容性字段；RetroRules 有模板但无级联信息；ASKCOS/AiZynthFinder 完全不考虑级联。

---

## 1. 系统功能需求（从用户视角）

**输入**：目标分子 SMILES + 约束（可选）  
**输出**：
1. top-N 条合成路线（含化学步 + 酶步混合）
2. 每条路线的**级联兼容性分析**：
   - 哪些连续步骤可以 one-pot（兼容性分数 + 置信度）
   - 不兼容的步骤对及其原因（5 大失败模式分类）
   - 推荐的 pot 划分方案（最少分段数）
3. 每步的反应条件（温度、溶剂、pH、催化剂/酶）
4. 每步的有机/酶催化分类 + EC 号推荐

---

## 2. 组件清单与复用决策矩阵

### 2.1 总览

| # | 组件 | 功能 | 来源决策 | 工作量 |
|---|------|------|----------|--------|
| C1 | MCTS* 搜索引擎 | 多步逆合成路线搜索 | **复用** ChemEnzyRetroPlanner | 小（接口适配） |
| C2 | 单步逆合成模型 | product → reactants | **复用** ChemEnzyRetroPlanner（MLP/Graph/ONMT/TemplateRel） | 零 |
| C3 | 有机/酶催化分类器 | 判断每步是化学还是酶催化 | **复用** ChemEnzyRetroPlanner | 零 |
| C4 | EC 号推荐器 | 为酶催化步推荐 EC 号 | **复用** ChemEnzyRetroPlanner | 零 |
| C5 | 反应条件预测器 | 预测溶剂/温度/试剂 | **复用** ChemEnzyRetroPlanner (RCR/Parrot) | 零 |
| C6 | 反应可行性过滤器 | 过滤不可行的逆合成提议 | **复用** ChemEnzyRetroPlanner | 零 |
| C7 | 可购买原料库 | 判断分子是否可购买 | **复用** ChemEnzyRetroPlanner (5 个 CSV stocks) | 零 |
| C8 | **级联兼容性预测模型** | **核心创新**：预测步骤对是否兼容 | **自建** | **大** |
| C9 | **级联数据库 & 数据管道** | 存储/查询提取的级联数据 | **自建** | **中** |
| C10 | **级联感知路线评分器** | 综合兼容性给路线排序 | **自建** | **中** |
| C11 | **级联感知搜索调度** | 搜索时集成兼容性信号 | **自建**（扩展 C1） | **中** |
| C12 | 值函数 | MCTS 节点估值 | **复用+扩展** ChemEnzyRetroPlanner ValueMLP | 小 |
| C13 | API 编排层 | 组装全流程管线 | **复用+扩展** RSPlanner | 中 |
| C14 | Web 前端 | 用户交互界面 | **复用+扩展** webapp/ | 低优先 |
| C15 | 分子指纹工具 | SMILES → fingerprint | **复用** smiles_to_fp.py + RDKit | 零 |
| C16 | 路线可视化 | 路线树/DAG 渲染 | **复用** viz_utils/ + SynRoute | 零 |

### 2.2 复用详情

#### ✅ 直接复用（零改动）

| 组件 | 来源文件 | 说明 |
|------|----------|------|
| C2 单步模型 | `packages/mlp_retrosyn/`, `graph_retrosyn/`, `onmt/`, `template_relevance/` | 4 种单步模型，接口统一 `run(smi, topk)` |
| C3 有机/酶分类 | `packages/organic_enzyme_rxn_classifier/` | RXNFP-based，`predict(rxns)` → labels |
| C4 EC 推荐 | 同 C3 的 `EnzymeRXNClassifier` | `predict(rxns, topk)` → EC numbers |
| C5 条件预测 | `packages/condition_predictor/` | `get_n_conditions(rxn, n)` → 溶剂/温度/试剂 |
| C6 过滤器 | `packages/rxn_filter/` | `is_feasible(rxn)` → bool |
| C7 原料库 | `common/prepare_utils.py` → `prepare_starting_molecules()` | 5 个 CSV stocks |
| C15 指纹 | `common/smiles_to_fp.py` | Morgan FP 2048-bit |
| C16 可视化 | `viz_utils/`, `search_frame/mcts_star/syn_route.py` | `route_to_dict()` + viz |

#### 🔧 复用+适配（小量改动）

| 组件 | 来源 | 需要的改动 |
|------|------|-----------|
| C1 MCTS* | `search_frame/mcts_star/molmcts_star.py` + `mol_tree.py` | 在 `expand()` 后插入兼容性打分钩子；`ReactionNode` 添加 `reaction_type` 和 `compatibility_score` 属性 |
| C12 值函数 | `packages/value_function/` | 可选：训新的 ValueMLP 加入级联特征（如路线中已有的酶步数、pH 范围估计） |
| C13 RSPlanner | `api.py` | 在 `plan()` 后添加 `evaluate_cascade_compatibility()` 调用；在 `_predict_rxn_attribute` 中添加兼容性属性 |

#### 🆕 需要自建（核心创新）

详见下文第 3 节。

---

## 3. 自建组件详细设计

### 3.1 C8 · 级联兼容性预测模型

**这是整个项目的核心技术贡献，也是唯一需要大量研究投入的组件。**

#### 3.1.1 任务定义

```
输入：(step_i, step_j) — 两个相邻反应步骤的结构化描述
      step = {rxn_smiles, catalyst/enzyme, conditions(pH, temp, solvent, cofactors)}
输出：
  - compatibility_score: float [0, 1]  — 一锅兼容概率
  - failure_modes: Dict[str, float]    — 5 大失败模式各自的概率
  - explanation: str                   — 可解释性输出（Phase 2）
```

#### 3.1.2 模型架构选项（按优先级排序）

**方案 A：双塔 + 交互 MLP（MVP 首选）**
```
Step_i  →  [RXNFP(rxn_i) ⊕ CondFP(cond_i)]  →  MLP_encoder  →  repr_i
Step_j  →  [RXNFP(rxn_j) ⊕ CondFP(cond_j)]  →  MLP_encoder  →  repr_j
[repr_i ⊕ repr_j ⊕ |repr_i - repr_j|]  →  Interaction_MLP  →  (score, failure_modes)
```
- RXNFP：复用 C3 中已有的 rxnfp BERT 模型（256-dim）
- CondFP：条件编码 = concat(pH_onehot, temp_normalized, solvent_fp, cofactor_fp)
- 参考 `rxn_filter/FilterModel` 的双线性打分架构
- 参数量：~5M，单 GPU 几小时可训

**方案 B：图神经网络（如果 A 不够）**
- 将两步反应构建为联合反应图
- 用 D-MPNN（参考 `graph_retrosyn/`）编码
- 更强的表达能力，但需要更多数据

**方案 C：微调 LLM（如果数据很少 <500 条）**
- 将步骤对描述为文本模板，微调 Llama-3-8B 做分类
- Few-shot 友好，但推理慢

**MVP 决策：先走方案 A，数据足够（>1000 对）时直接有效。**

#### 3.1.3 训练数据来源

| 来源 | 正例/负例 | 估计规模 |
|------|----------|---------|
| CASCADE_EXTRACTION 提取的级联文献 | 正例（成功的 one-pot 步骤对） | ~2000-5000 对 |
| 同一文献中报告的失败尝试 | 负例（明确不兼容的组合） | ~500-1000 对 |
| RetroBioCat 52 条级联 | 正例（补充） | ~100 对 |
| 规则增强负例 | pH>2 单位差、有机溶剂+酶 等确定性不兼容 | ~5000 对 |
| 药物化学常识负例 | Pd 催化 + 任意酶 → 不兼容 | ~2000 对 |

规则增强负例是关键——我们已知的化学常识可以大量生成高置信度负例（如：Pd/Ni 金属催化剂 + 酶 = 不兼容；强酸/强碱条件 + 酶 = 不兼容），弥补正例数据不足。

#### 3.1.4 代码结构
```python
# cascade_compatibility/
#   __init__.py
#   model.py           — CascadeCompatibilityModel(nn.Module)
#   features.py         — 步骤对特征编码（RXNFP + CondFP）
#   inference_api.py    — CascadeCompatibilityPredictor (加载+推理接口)
#   train.py            — 训练脚本
#   data_utils.py       — 数据加载、规则增强负例生成
#   failure_modes.py    — 5 大失败模式分类头
#   rules.py            — 硬编码化学兼容性规则（作为基线+数据增强）
```

---

### 3.2 C9 · 级联数据库 & 数据管道

**存储从文献提取的级联催化数据（见 CASCADE_EXTRACTION_RESPONSE.md 的 5 层 JSON schema）。**

#### 3.2.1 存储方案

**SQLite + JSON 字段（MVP）**，不上 MongoDB/PostgreSQL：
- 级联数据量级 ~5000 条，SQLite 完全够
- JSON 字段存嵌套结构（步骤详情、兼容性信息）
- 后期如需扩展，迁移到 PostgreSQL 成本很低

```sql
CREATE TABLE cascades (
    id TEXT PRIMARY KEY,          -- CASCADE_2026_00001
    doi TEXT NOT NULL,
    title TEXT,
    cascade_type TEXT,            -- sequential/concurrent/parallel
    num_steps INTEGER,
    overall_yield REAL,
    overall_ee REAL,
    steps_json TEXT,              -- Layer B：各步详情 JSON array
    conditions_json TEXT,         -- Layer C：全局条件 JSON
    compatibility_json TEXT,      -- Layer D：兼容性信息 JSON
    substrate_scope_json TEXT,    -- Layer E：底物范围 JSON
    source TEXT,                  -- 提取来源（WoS batch/manual etc）
    quality_score REAL,           -- 提取质量评分
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE step_pairs (
    id INTEGER PRIMARY KEY,
    cascade_id TEXT REFERENCES cascades(id),
    step_i_idx INTEGER,
    step_j_idx INTEGER,
    compatible BOOLEAN,           -- 是否兼容
    compatibility_score REAL,     -- 如有定量数据
    failure_modes TEXT,           -- JSON: {"pH": 0.8, "solvent": 0, ...}
    evidence TEXT,                -- 原文引用
    rxn_smiles_i TEXT,
    rxn_smiles_j TEXT,
    conditions_i TEXT,            -- JSON
    conditions_j TEXT             -- JSON
);
```

#### 3.2.2 代码结构
```python
# cascade_db/
#   __init__.py
#   schema.py           — SQLAlchemy / dataclass 模型
#   loader.py            — 从提取 JSON → 入库
#   query.py             — 查询接口（按类型、按兼容性、按酶等）
#   pair_generator.py    — 从 cascades 表生成 step_pairs 训练数据
#   stats.py             — 数据统计与质量检查
```

---

### 3.3 C10 · 级联感知路线评分器

**对 MCTS* 搜索输出的每条完整路线，计算级联兼容性综合得分。**

```python
class CascadeRouteScorer:
    """
    输入：dict_route（ChemEnzyRetroPlanner 的 route_to_dict() 输出格式）
    输出：
      - cascade_score: float        — 路线级联整体兼容性 [0, 1]
      - pot_assignment: List[int]   — 每步属于哪个 pot（最优分段方案）
      - pairwise_scores: List[dict] — 每对相邻步的兼容性详情
      - num_pots: int               — 最少需要多少个 pot
    """
    def __init__(self, compatibility_model, condition_predictor, rxn_classifier):
        ...

    def score_route(self, dict_route: dict) -> dict:
        # 1. 遍历 dict_route 树，提取所有反应步骤（线性化）
        # 2. 对每步预测条件（C5）和有机/酶分类（C3）
        # 3. 对每对相邻步骤调用 C8 计算兼容性
        # 4. 用动态规划求最优 pot 划分（最小化分段数，满足兼容性阈值）
        # 5. 综合得分 = f(num_pots, avg_compatibility, min_compatibility)
        ...
    
    def optimal_pot_assignment(self, pairwise_scores: List[float], threshold: float = 0.5) -> List[int]:
        # 贪心/DP：尽量把兼容的连续步骤放同一 pot
        ...
```

#### 代码结构
```python
# cascade_scoring/
#   __init__.py
#   route_scorer.py      — CascadeRouteScorer
#   pot_optimizer.py      — pot 划分优化算法（贪心 + DP）
#   route_linearizer.py   — dict_route 树 → 线性步骤序列
```

---

### 3.4 C11 · 级联感知搜索调度

**两种集成策略（选一或组合）：**

**策略 A · 后处理排序（MVP，低风险）**
```
MCTS* → all_succ_routes → CascadeRouteScorer.score_route() → 按cascade_score排序
```
- 优点：不改搜索引擎，完全解耦
- 缺点：搜索本身不考虑级联，可能找不到最优级联路线

**策略 B · 搜索内集成（Phase 2，高收益）**
```
MCTS* expand → 对候选 reactant_lists 加入级联兼容性估计 → 调整 cost/value → 引导搜索优先探索级联友好方向
```
- 在 `mol_tree.expand()` 后、`reaction_node` 创建时，查询当前路径上已有步骤，用 C8 快速估计新步骤与前一步的兼容性
- 兼容性高的方向获得更低的 cost，引导 MCTS 优先探索
- 改动点：`molmcts_star.py` 的 expand 逻辑 + `reaction_node.py` 增加属性

**MVP 决策：先走策略 A，验证 cascade_score 有区分度后再集成到搜索内。**

---

## 4. 目录结构

```
CascadePlanner/                          # 新仓库
├── setup.py
├── environment.yml
├── README.md
├── config/
│   └── config.yaml                      # 全局配置（继承 ChemEnzyRetroPlanner 格式）
│
├── cascade_planner/                     # 主包
│   ├── __init__.py
│   ├── api.py                           # CascadePlanner 主入口（扩展 RSPlanner）
│   │
│   ├── common/                          # 通用工具（从 ChemEnzyRetroPlanner 复制）
│   │   ├── __init__.py
│   │   ├── utils.py
│   │   ├── smiles_to_fp.py
│   │   ├── prepare_utils.py             # 工厂函数（扩展注册新组件）
│   │   └── parse_args.py
│   │
│   ├── search/                          # 搜索引擎（从 ChemEnzyRetroPlanner 复制+扩展）
│   │   ├── __init__.py
│   │   ├── molmcts_star.py              # MCTS* 主循环（加 cascade 钩子）
│   │   ├── mol_tree.py
│   │   ├── mol_node.py
│   │   ├── reaction_node.py             # 扩展：reaction_type, compatibility_score
│   │   └── syn_route.py
│   │
│   ├── cascade_compatibility/           # 🆕 C8：核心创新
│   │   ├── __init__.py
│   │   ├── model.py                     # CascadeCompatibilityModel(nn.Module)
│   │   ├── features.py                  # 步骤对特征编码
│   │   ├── inference_api.py             # CascadeCompatibilityPredictor
│   │   ├── train.py                     # 训练脚本
│   │   ├── data_utils.py                # 数据加载 + 规则增强
│   │   ├── failure_modes.py             # 5 大失败模式分类
│   │   └── rules.py                     # 硬编码规则基线
│   │
│   ├── cascade_db/                      # 🆕 C9：级联数据库
│   │   ├── __init__.py
│   │   ├── schema.py                    # 数据模型
│   │   ├── loader.py                    # JSON → DB
│   │   ├── query.py                     # 查询接口
│   │   ├── pair_generator.py            # 生成训练用步骤对
│   │   └── stats.py                     # 数据统计
│   │
│   ├── cascade_scoring/                 # 🆕 C10：路线评分
│   │   ├── __init__.py
│   │   ├── route_scorer.py              # CascadeRouteScorer
│   │   ├── pot_optimizer.py             # pot 划分 DP
│   │   └── route_linearizer.py          # dict_route → 线性步骤
│   │
│   ├── packages/                        # 复用的子包（符号链接或直接复制）
│   │   ├── condition_predictor/         # ← ChemEnzyRetroPlanner
│   │   ├── rxn_filter/                  # ← ChemEnzyRetroPlanner
│   │   ├── mlp_retrosyn/               # ← ChemEnzyRetroPlanner
│   │   ├── graph_retrosyn/             # ← ChemEnzyRetroPlanner
│   │   ├── onmt/                        # ← ChemEnzyRetroPlanner
│   │   ├── template_relevance/          # ← ChemEnzyRetroPlanner
│   │   ├── organic_enzyme_rxn_classifier/  # ← ChemEnzyRetroPlanner
│   │   ├── easifa/                      # ← ChemEnzyRetroPlanner
│   │   └── value_function/              # ← ChemEnzyRetroPlanner（可选扩展）
│   │
│   ├── utils/                           # 日志等（复用）
│   │   ├── __init__.py
│   │   └── logger.py
│   │
│   └── viz_utils/                       # 可视化（复用+扩展级联视图）
│       ├── chem.py
│       ├── image.py
│       └── route_tree.py
│
├── data/                                # 数据目录
│   ├── cascades/                        # 提取的级联数据 JSON
│   ├── stocks/                          # 可购买分子库（链接 ChemEnzyRetroPlanner）
│   ├── models/                          # 预训练模型权重
│   └── rules/                           # 兼容性规则文件
│
├── scripts/                             # 训练/评估脚本
│   ├── train_compatibility_model.py
│   ├── evaluate_compatibility.py
│   ├── generate_rule_negatives.py       # 规则增强负例生成
│   ├── import_cascade_data.py           # 导入提取数据
│   └── benchmark_routes.py              # 路线评估基准
│
├── tests/                               # 测试
│   ├── test_compatibility_model.py
│   ├── test_cascade_db.py
│   ├── test_route_scorer.py
│   └── test_pot_optimizer.py
│
└── webapp/                              # Web 前端（Phase 2，复用+扩展）
    └── ...
```

---

## 5. 开发优先级与路线图

### Phase 0 · 脚手架（第 1 周）
- [ ] 创建 `CascadePlanner` 仓库
- [ ] 从 ChemEnzyRetroPlanner 复制 C1-C7 + C15-C16（搜索、模型包、工具）
- [ ] 验证复制后的代码能独立运行 `plan(target_mol)` 并产出 dict_route
- [ ] 建立 `config.yaml` 和 `prepare_utils.py` 的扩展注册点

### Phase 1 · 级联数据基础（第 2-3 周，与数据提取并行）
- [ ] C9：实现 `cascade_db/` 全套（schema + loader + query）
- [ ] 实现 `pair_generator.py`：从导入的级联数据生成步骤对
- [ ] 实现 `rules.py`：硬编码 ~20 条确定性兼容性规则
- [ ] 实现 `data_utils.py`：规则增强负例生成

### Phase 2 · 兼容性模型 MVP（第 4-6 周）
- [ ] C8：实现方案 A（双塔 + 交互 MLP）
- [ ] 实现 `features.py`：RXNFP + CondFP 特征编码
- [ ] 实现 `model.py` + `train.py`：训练流程
- [ ] 用规则增强数据做初步训练，验证模型架构
- [ ] 数据到位后用真实数据训练

### Phase 3 · 路线评分集成（第 7-8 周）
- [ ] C10：实现 `CascadeRouteScorer`
- [ ] C11 策略 A：后处理排序集成到 `api.py`
- [ ] 端到端测试：target_mol → plan → cascade_score → 排序

### Phase 4 · 搜索内集成 + 评估（第 9-12 周）
- [ ] C11 策略 B：兼容性信号反馈到 MCTS（如 Phase 3 验证有效）
- [ ] 基准评估：RetroBioCat 52 条级联 + 自建评估集
- [ ] 论文写作

---

## 6. 关键接口约定

### 6.1 步骤描述格式（Step Description）

所有组件间传递步骤信息使用统一格式：

```python
StepDesc = {
    "rxn_smiles": "reactants>>product",          # 必需
    "reaction_type": "enzymatic" | "organic",     # C3 预测
    "ec_number": "1.1.1.1" | None,                # C4 预测（酶步）
    "conditions": {                                # C5 预测
        "temperature": 37.0,                       # °C
        "ph": 7.4,                                 # 仅酶步
        "solvent": "water",                        # SMILES 或名称
        "cofactors": ["NADH", "FAD"],              # 仅酶步
        "catalyst": "Pd(PPh3)4" | None,            # 仅化学步
        "reagents": ["Et3N"]                        # 仅化学步
    },
    "confidence": 0.85                             # 模型置信度
}
```

### 6.2 兼容性预测输出格式

```python
CompatibilityResult = {
    "step_i_idx": 0,
    "step_j_idx": 1,
    "compatible": True | False,
    "score": 0.82,                                  # [0,1]
    "failure_modes": {
        "ph_conflict": 0.05,
        "solvent_incompatibility": 0.03,
        "mutual_inactivation": 0.08,
        "cofactor_competition": 0.01,
        "product_inhibition": 0.02
    },
    "confidence": 0.90
}
```

### 6.3 路线评分输出格式

```python
CascadeRouteScore = {
    "route_id": "...",
    "cascade_score": 0.75,                          # 路线级综合分
    "num_pots": 2,                                   # 最少 pot 数
    "pot_assignment": [1, 1, 2, 2, 2],              # 每步属于哪个 pot
    "pairwise_compatibility": [                      # 相邻步骤对
        CompatibilityResult,
        ...
    ],
    "bottleneck": {                                  # 最弱环节
        "step_pair": [1, 2],
        "score": 0.3,
        "reason": "ph_conflict"
    }
}
```

---

## 7. 与 ChemEnzyRetroPlanner 的代码关系

### 7.1 复用策略

**不用 fork，用复制 + 选择性导入**，理由：
- ChemEnzyRetroPlanner 有大量 Docker/Singularity 部署代码、parrot 微服务等与级联无关的内容
- 我们只需要核心算法代码，不需要整个部署栈
- 复制后可以自由修改接口，不受上游 breaking changes 影响

**具体操作**：
```bash
# 复制搜索引擎
cp -r ChemEnzyRetroPlanner/retro_planner/search_frame/mcts_star/* CascadePlanner/cascade_planner/search/

# 复制通用工具
cp ChemEnzyRetroPlanner/retro_planner/common/*.py CascadePlanner/cascade_planner/common/

# 复制 packages（保持目录结构）
cp -r ChemEnzyRetroPlanner/retro_planner/packages/{condition_predictor,rxn_filter,mlp_retrosyn,graph_retrosyn,onmt,template_relevance,organic_enzyme_rxn_classifier,easifa,value_function} CascadePlanner/cascade_planner/packages/

# 复制 viz_utils
cp -r ChemEnzyRetroPlanner/retro_planner/viz_utils/* CascadePlanner/cascade_planner/viz_utils/

# 复制 api.py 作为扩展基础
cp ChemEnzyRetroPlanner/retro_planner/api.py CascadePlanner/cascade_planner/api.py
```

### 7.2 需要修改的复用代码

| 文件 | 修改内容 |
|------|---------|
| `api.py` | 添加 `evaluate_cascade_compatibility()` 方法；在 `plan()` 返回中添加 `cascade_scores` |
| `reaction_node.py` | 添加 `reaction_type`, `conditions`, `compatibility_score` 属性 |
| `syn_route.py` | `route_to_dict()` 输出扩展 `cascade_info` 字段 |
| `prepare_utils.py` | 添加 `prepare_cascade_compatibility_model()` 工厂函数 |
| `config.yaml` | 添加 `cascade_compatibility` 配置段 |

---

## 8. 外部依赖

### 8.1 已有依赖（从 ChemEnzyRetroPlanner 继承）
- `torch >= 1.12`
- `rdkit >= 2022.03`
- `rxnfp` — BERT-based 反应指纹（C3/C4/C8 都用）
- `numpy`, `pandas`, `scipy`
- `dgl` / `torch_geometric` — 图神经网络（C2 graph_retrosyn）
- `rdchiral` — 模板应用
- `flask`, `celery` — webapp

### 8.2 新增依赖
- `sqlalchemy` — 级联数据库 ORM（轻量，不需要单独的数据库服务）
- `scikit-learn` — 基线模型、评估指标
- 无其他新依赖

---

## 9. 规则基线（C8 的兜底方案）

即使机器学习模型训练不理想，我们仍然可以用**硬编码规则**提供有价值的级联兼容性分析：

```python
# rules.py 示例规则

INCOMPATIBILITY_RULES = [
    # (条件A, 条件B, 失败模式, 置信度)
    ("metal_catalyst:Pd", "enzyme:*", "mutual_inactivation", 0.95),
    ("metal_catalyst:Ni", "enzyme:*", "mutual_inactivation", 0.90),
    ("metal_catalyst:Cu", "enzyme:*", "mutual_inactivation", 0.80),
    ("ph:<4", "enzyme:*", "ph_conflict", 0.90),
    ("ph:>10", "enzyme:*", "ph_conflict", 0.90),
    ("solvent:DMSO>50%", "enzyme:lipase", "solvent_incompatibility", 0.60),
    ("solvent:THF", "enzyme:*", "solvent_incompatibility", 0.85),
    ("solvent:DCM", "enzyme:*", "solvent_incompatibility", 0.90),
    ("cofactor:NADH", "cofactor:NADH", "cofactor_competition", 0.70),
    ("reagent:LDA", "enzyme:*", "mutual_inactivation", 0.95),
    ("reagent:NaBH4", "enzyme:KRED", "mutual_inactivation", 0.75),
    ("temp:>80", "enzyme:*", "mutual_inactivation", 0.85),
    # ... 更多规则从文献知识编码
]

COMPATIBILITY_RULES = [
    # 已知兼容的组合
    ("enzyme:KRED", "enzyme:GDH", "cofactor_recycling", 0.95),  # NADPH 循环
    ("enzyme:lipase", "enzyme:lipase", "compatible", 0.90),
    ("enzyme:transaminase", "enzyme:alanine_dehydrogenase", "cofactor_recycling", 0.90),
    # ...
]
```

这些规则：
1. 可以作为**数据增强**生成高置信度负例
2. 可以作为**规则基线模型**直接使用（论文中与 ML 模型对比）
3. 可以作为**特征**输入 ML 模型（rule_match_count 等）
4. 数据不足时是**兜底方案**，仍然比现有工具（零级联分析）好得多

---

## 10. 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| 级联数据提取量不足 (<500 条) | 中 | 规则增强负例 + 规则基线兜底；先发 method paper 后补数据 |
| 兼容性模型区分度不够 | 中 | 方案 A → B → C 逐步升级；规则基线保底 |
| RXNFP 对酶催化反应效果差 | 低-中 | 替换为 DRFP 或 Morgan difference FP |
| 复用的 ChemEnzyRetroPlanner 代码有 bug | 低 | Phase 0 做完整回归测试 |
| 搜索内集成（C11 策略 B）导致搜索变慢 | 中 | 兼容性预测用轻量规则先行，ML模型作后处理 |

---

## 11. 立即行动清单

### 本周
1. [ ] 创建 `CascadePlanner` 仓库，建立目录结构
2. [ ] 从 ChemEnzyRetroPlanner 复制可复用代码（§7.1 的命令）
3. [ ] 验证复制后的基础流程可运行
4. [ ] 实现 `cascade_compatibility/rules.py` — 硬编码 20+ 条规则

### 下周
5. [ ] 实现 `cascade_db/schema.py` + `loader.py`
6. [ ] 实现 `cascade_scoring/route_linearizer.py` — dict_route → 步骤序列
7. [ ] 用规则基线实现 `CascadeRouteScorer` v0.1
8. [ ] 端到端测试：target_mol → plan → rule-based cascade_score

### 数据到位后
9. [ ] 实现 `features.py` + `model.py`（方案 A 架构）
10. [ ] 实现 `data_utils.py`（规则增强 + 数据加载）
11. [ ] 训练 + 评估
