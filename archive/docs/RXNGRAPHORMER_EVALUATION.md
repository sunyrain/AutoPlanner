# RXNGraphormer 深度评估：架构、性能、与我们系统的整合可行性

> **日期**：2026-04-19
> **核心问题**：RXNGraphormer 能在我们的系统中扮演什么角色？能否处理级联/非级联逆合成？

---

## 0. 结论先行

| 问题 | 结论 |
|------|------|
| RXNGraphormer 能做逆合成吗？ | ✅ 能，但只是**单步逆合成**（RXNG2Sequencer），且性能仅中等 |
| 能替代我们现有的逆合成模型？ | ❌ **不能直接替代**，推理速度太慢，且不支持酶反应 |
| 能处理级联？ | ❌ **完全不能**，没有任何级联/多步概念 |
| 能产生有用的反应 embedding？ | ✅ 能，`RXNEMB` 可以作为反应特征提取器 |
| **最佳定位** | **反应 embedding 提供者** + 可选的**辅助逆合成候选生成器** |

**关键认知纠正**：级联兼容性只是我们问题的一部分。我们的核心问题是**逆合成路线规划**（包括级联和非级联）。RXNGraphormer 对这个核心问题的贡献是**有限的**。

---

## 1. 架构深度分析

### 1.1 整体架构

```
RXNGraphormer 是一个 **三合一** 框架：

                    ┌── RXNGClassifier    (分类，用于预训练)
RXNGraphormer ──────┼── RXNGRegressor     (回归，yield/selectivity)
  (Factory)         └── RXNG2Sequencer    (序列生成，retro/forward-synthesis)

共享骨干：
  Input (SMILES) → RDKit Parse → Molecular Graph
                                    ↓
                             GNN Encoder (GCN/GIN/GAT)
                             ┌─ rct_encoder (反应物)
                             └─ pdt_encoder (产物)
                                    ↓
                             Transformer Layer (self-attention)
                                    ↓
                             Graph Pooling (attention)
                                    ↓
                           Fixed-length Reaction Embedding
                                    ↓
                        Task-specific Head (分类/回归/序列)
```

### 1.2 关键参数（从 config 文件）

| 参数 | 预训练模型 | 典型微调模型 |
|------|----------|------------|
| gnn_type | gcn | gcn / gin / gat |
| gnn_num_layer | 4 (~5 typical) | 3-5 |
| emb_dim | 256 | 256-768 |
| trans_num_layer | 4 (~8 typical) | 3-8 |
| nhead | 2 (~8 typical) | 4-8 |
| trans_ff_dim | 768 | 768 |
| graph_pooling | attention | attention / mean |
| output_num_layer | 3 | 3 |
| drop_ratio | 0.0 | 0.1-0.2 |

**注意**：DeepWiki 列出了两组数字——预训练配置（较小）和典型微调配置（较大）。实际模型大小取决于具体任务。

### 1.3 三种模型变体

| 变体 | 用途 | 编码器 | 解码器 | 输出 |
|------|------|--------|--------|------|
| RXNGClassifier | 预训练 + 虚假反应识别 | 双编码器 (rct+pdt) | MLP | 2 类概率 |
| RXNGRegressor | yield/selectivity 预测 | 双编码器 + 可选 mid_encoder | MLP + 缩放 | 连续值 |
| RXNG2Sequencer | retro/forward synthesis | 单编码器 (product 或 reactant) | **Transformer decoder + beam search** | SMILES 序列 |

### 1.4 模型大小

| 模型 | 文件大小 | 估算参数量 |
|------|---------|-----------|
| pretrained_classification_model | ~500 MB | ~数十 M |
| 回归模型 (各) | ~300-400 MB | ~数十 M |
| 序列生成模型 (各) | ~400-600 MB | ~数十 M |
| 全部模型 | ~5-10 GB | — |

---

## 2. 逆合成性能评估

### 2.1 RXNGraphormer 在 USPTO-50K 上的性能

**Top-1 准确率：56.8%**（来源：我们之前的调研 NEXT_GEN_ARCHITECTURE_PLAN.md）

对比其他逆合成方法：

| 方法 | 类型 | USPTO-50K Top-1 | 年份 |
|------|------|----------------|------|
| Molecular Transformer | Seq2Seq (Transformer) | ~53.5% (w/ class) | 2019 |
| LocalRetro | Template-based (Graph) | 53.4% | 2021 |
| RetroXpert | Template-free (Graph) | ~50.4% | 2020 |
| Graph2SMILES | Seq2Seq (Graph encoder) | ~52.9% | 2021 |
| **RXNGraphormer** | **Seq2Seq (GNN+Transformer)** | **56.8%** | **2025** |
| SOTA 专用模型 | 各种 | 55-65%+ | 2023-2025 |

**评价**：
- 56.8% 作为**统一模型**的逆合成性能是**不错的**——比专用逆合成模型低几个点
- 但它不是逆合成 SOTA——专门设计的逆合成模型（如 R-SMILES、FusionRetro, RetroBridge 等）可以做到 60%+
- **它的卖点不是逆合成性能最优，而是"一个模型多任务"**

### 2.2 推理速度——⚠️ 严重瓶颈

| 任务 | 100 条反应 | 速率 (CPU) | 速率 (GPU, 估算) |
|------|-----------|-----------|-----------------|
| 回归 (yield/selectivity) | ~25 sec | ~4 rxn/sec | ~20-40 rxn/sec |
| **序列生成 (retro/forward)** | **~40 sec** | **~2.5 rxn/sec** | **~12-25 rxn/sec** |

**在树搜索中的影响**：

```
MCTS 搜索典型参数：
  iterations = 500-5000
  expansion_topk = 50 candidates per molecule
  总推理次数 = iterations × topk_per_expansion ≈ 25K-250K

RXNGraphormer seq gen 速度（GPU）: ~25 rxn/sec
  → 25K 次推理 = 1000 sec ≈ 17 min
  → 250K 次推理 = 10000 sec ≈ 167 min

对比现有系统：
  Template-based (MLP/Graph topk=50): ~10ms per target (batch template lookup + RDChiral)
  → 25K 次推理 ≈ 250 sec ≈ 4 min
  → 5000 次扩展 × 1 target each ≈ 50 sec
```

**RXNGraphormer 做逆合成在搜索中实在太慢了。** 它不能替代 template-based 模型在搜索循环中的角色。TeTemplte-based 模型每秒能评估数百个分子，RXNGraphormer 只能处理 ~25 个（GPU）。

### 2.3 酶反应支持——❌ 完全缺失

| 维度 | RXNGraphormer | 我们的需求 |
|------|--------------|-----------|
| 预训练数据 | 13M 有机反应 | 有机 + 酶反应 |
| 逆合成训练 | USPTO-50K/480K/full（全部有机） | 有机 + ECReact 酶反应 (~250K) |
| 酶反应识别 | ❌ | ✅ 必须 |
| 酶推荐 | ❌ | ✅ 必须 |
| 辅因子预测 | ❌ | ✅ 必须 |

**RXNGraphormer 完全不知道酶反应的存在。** 要用它做酶反应逆合成，需要用 ECReact 数据重新微调 RXNG2Sequencer——但这就是"在别人的框架上训自己的模型"，不如自己设计。

---

## 3. 级联处理能力——❌ 零支持

RXNGraphormer **没有任何级联相关功能**：

| 级联需求 | RXNGraphormer 支持 |
|---------|-------------------|
| 多步路线规划 | ❌（只做单步预测） |
| 步骤间兼容性判断 | ❌ |
| 条件冲突检测 | ❌ |
| One-pot 可行性评估 | ❌ |
| 级联感知搜索 | ❌ |
| 溶剂/pH/温度兼容 | ❌ |

**原因**：RXNGraphormer 的设计目标是"对单个反应做预测"（分类/回归/序列），不是"规划多步合成路线"。级联是路线级的问题，不是反应级的问题。

**结论**：级联相关功能 100% 需要我们自己构建，和选什么骨干无关。

---

## 4. RXNGraphormer 能为我们做什么？（正面评估）

### 4.1 反应 Embedding（最有价值）

```python
from rxngraphormer.rxn_emb import RXNEMB

rxnemb = RXNEMB(pretrained_model_path="./pretrained_classification_model",
                model_type="classifier")
embeddings = rxnemb.gen_rxn_emb(["reactants>>product", ...])
# → (N, emb_dim) numpy array
```

**用途**：
1. **反应类型聚类/可视化**：论文已展示可以用 embedding 区分 USPTO-50K 的 10 类反应
2. **下游分类/回归头的输入**：冻结骨干，在上面接轻量 head
3. **反应相似性计算**：比较两个反应的 embedding 距离
4. **虚假反应检测**：`RXNClassifier` 已经可以判断反应是否虚构

**速度**：embedding 生成用的是编码器部分（不需要 beam search），速度比序列生成快很多（约 ~4 rxn/sec CPU → ~20+ rxn/sec GPU）。

### 4.2 虚假反应过滤器（有用但非核心）

```python
from rxngraphormer.rxn_emb import RXNClassifier
classifier = RXNClassifier(pretrained_model_path, random_init=False)
predictions, confidences = classifier.rxn_pred(["reactants>>product", ...])
# predictions: 0=虚假, 1=真实
```

可以用来过滤逆合成模型产生的无效候选反应。

### 4.3 辅助逆合成候选（锦上添花）

在 MCTS 搜索**之外**，可以用 RXNGraphormer 为最终路线候选生成额外的逆合成建议：
- 不放在搜索循环里（太慢）
- 在找到几条路线后，对个别关键节点用 RXNGraphormer 验证或补充候选
- 作为 MultiOneStepRunWrapper 中的一个低权重备选模型

---

## 5. 训练时间评估

### 5.1 预训练（我们不需要）

- 数据：13M 反应
- 设备：Multi-GPU DDP
- 时间：论文未公开，估算 **数天到一周**（batch=5120, 20 epochs）
- **我们直接加载预训练权重，跳过此步**

### 5.2 微调逆合成（如果想用其做逆合成）

- 数据：USPTO-50K (~50K 反应)
- 估算：单 GPU, 20-50 epochs → **数小时到半天**
- 但要加酶反应 (ECReact ~250K) 需要更久

### 5.3 微调回归/分类（如果想用于 yield 预测等）

- 小数据集 (几千到几万条)
- 单 GPU, 很快 → **1-2 小时**

### 5.4 如果我们用 RXNEMB 做 embedding + 自训 heads

| 组件 | 训练时间 | 设备 |
|------|---------|------|
| 加载预训练权重 | 0（已有） | — |
| 生成 2M 反应的 embedding | ~14 小时 (CPU) / ~2-3 小时 (GPU) | 1×A100 |
| 训 5 个 heads | ~数小时 | 1×A100 |
| **总计** | **< 1 天** | 1×A100 |

---

## 6. 整合方案分析

### 方案对比

| 方案 | 描述 | 开发量 | 风险 | 贡献度 |
|------|------|--------|------|--------|
| **A. 只用 embedding** | RXNEMB 提取特征 → 接 heads | 低 (1-2 天) | 低 | 中（论文引用 Nature MI） |
| **B. embedding + 辅助逆合成** | A + 用 RXNG2Sequencer 作为备选候选生成器 | 中 (3-5 天) | 中 | 中 |
| **C. 重训骨干做逆合成** | 在 RXNGraphormer 上微调，加入酶反应数据 | 高 (2-3 周) | 高 | 低（重复造轮子） |
| **D. 不用 RXNGraphormer** | 完全自建或用 Chemprop | 高 (3-4 周) | 高 | 低（无强引用） |

### 推荐：方案 A（高性价比）+ 保持现有逆合成模型

**理由**：
1. 我们现有的逆合成模型（template_relevance + ONMT + MLP + Graph）**已经很好用**
2. RXNGraphormer 的逆合成**不能替代**它们（速度太慢、不支持酶）
3. RXNGraphormer 的 **embedding** 是有价值的——13M 反应预训练的表示
4. 级联功能不管用什么骨干都要自建

---

## 7. 重新审视系统架构

既然"逆合成是核心，级联兼容性是子问题"，我们应该这样定位各组件：

```
我们的新系统架构（修正版）：

┌─────────────────────────────────────────────────────┐
│                    搜索引擎 (MCTS*)                    │
│  ┌──────────────────────────────────────────────┐   │
│  │        SingleStepExpander (多模型协作)         │   │
│  │                                              │   │
│  │  ① template_relevance → 快速 top-50 候选     │   │  ← 主力（现有模型，速度快）
│  │  ② ONMT transformer → seq2seq 候选补充       │   │  ← 辅助（现有模型）
│  │  ③ [可选] RXNG2Sequencer → 额外候选          │   │  ← 锦上添花（新增，低权重）
│  │                                              │   │
│  │  MultiOneStepRunWrapper 合并排序              │   │
│  └──────────────────────────────────────────────┘   │
│           ↓ 候选反应                                  │
│  ┌──────────────────────────────────────────────┐   │
│  │        ReactionEvaluator (NEW)                │   │
│  │                                              │   │
│  │  RXNGraphormer RXNEMB → 反应 embedding        │   │  ← embedding 提取（新增）
│  │       ↓                                      │   │
│  │  ┌─ TypeHead (有机/酶分类)                    │   │
│  │  ├─ FeasibilityHead (反应可行性)              │   │  ← 自训 heads（新增）
│  │  ├─ ValueHead (路线价值估计)                  │   │
│  │  └─ ConditionHead (条件预测)                  │   │
│  └──────────────────────────────────────────────┘   │
│           ↓ 搜索完成，得到多条路线                      │
│  ┌──────────────────────────────────────────────┐   │
│  │        CascadeAnalyzer (NEW, 后处理)          │   │
│  │                                              │   │
│  │  对完整路线做级联分析：                        │   │
│  │  ├─ 规则引擎（溶剂/催化剂/pH 冲突检测）       │   │
│  │  ├─ 条件兼容性评分                            │   │
│  │  ├─ One-pot 分组建议                          │   │
│  │  └─ [未来] ML 兼容性模型                      │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 谁做逆合成？ | **保持现有模型** (template_relevance, ONMT, MLP/Graph) | 速度快、支持酶反应、已经验证 |
| RXNGraphormer 做什么？ | **反应 embedding 提供者** | 高质量预训练表示，零训练成本 |
| 级联怎么处理？ | **搜索后处理** | 不在搜索循环中做兼容性判断（太慢），对搜索结果做后处理 |
| 酶反应逆合成？ | **ONMT (BioNav 训练) + template_relevance (bkms/reaxys_biocatalysis)** | 已有支持 |
| 级联感知怎么进搜索？ | **分离式 → 后续集成式** | MVP 先后处理，效果验证后再考虑在线评估 |

---

## 8. 对"我们的关键问题是逆合成"的回应

完全同意。让我重新定位问题：

### 8.1 我们实际要解决的问题

```
输入：目标分子 SMILES
输出：合成路线（可能包含有机步骤 + 酶催化步骤 + one-pot 级联段落）

具体要求：
1. 能规划到可购买原料（多步逆合成）           ← 核心（已有基础）
2. 有机 + 酶催化反应都要支持                   ← 已有（ONMT + bkms）
3. 识别哪些步骤可以级联（one-pot）             ← 新增（级联分析）
4. 对路线进行条件预测和评分                     ← 新增（条件/价值头）
5. 路线要尽量考虑级联效率                       ← 高级（级联感知搜索）
```

### 8.2 各组件对核心问题的贡献

| 组件 | 对逆合成的贡献 | 对级联的贡献 | RXNGraphormer 参与 |
|------|--------------|------------|-------------------|
| SingleStepExpander | **核心** | 间接（产生候选） | 可选辅助 |
| MCTS* Search | **核心** | 间接（搜索策略） | ❌ 不参与 |
| 反应 Embedding | 间接（价值估计） | 间接（兼容性特征） | ✅ RXNEMB |
| TypeHead | 辅助（有机/酶分类） | 直接（级联前提） | ✅ 消费 embedding |
| ConditionHead | 辅助（条件预测） | **直接** | ✅ 消费 embedding |
| ValueHead | 辅助（搜索引导） | 间接 | ✅ 消费 embedding |
| CascadeAnalyzer | ❌ 不参与逆合成 | **核心** | ❌ 不参与 |

### 8.3 开发优先级（修正）

```
Phase 0 · 基础设施 + 搜索引擎（最优先）
  - 复用/重新实现 MCTS* 搜索
  - 适配现有 SingleStepExpander 模型
  - 确保有机 + 酶反应都能搜索
  → 这时已经能做逆合成了（核心功能完成）

Phase 1 · 反应评估增强
  - 集成 RXNGraphormer RXNEMB
  - 训练 TypeHead, FeasibilityHead, ValueHead
  - 用 ValueHead 改进搜索质量
  → 搜索质量提升

Phase 2 · 级联分析（后处理）
  - 实现规则引擎（条件冲突检测）
  - 实现 ConditionHead 条件预测
  - 路线评分 + one-pot 分组建议
  → 级联功能 MVP

Phase 3 · 级联感知搜索（可选进阶）
  - 在搜索中在线评估兼容性
  - 级联偏好搜索策略
  → 高级功能
```

---

## 9. RXNGraphormer 环境兼容性

### 9.1 依赖

```
Python 3.8
PyTorch 1.12.1+cu113
PyG (torch-geometric) 2.3.1
RDKit
```

### 9.2 与 ChemEnzyRetroPlanner 的兼容性

ChemEnzyRetroPlanner 的 `environment.yml` 可能使用更新的 Python/PyTorch 版本。需要检查：
- Python 3.8 vs 我们的版本
- PyTorch 1.12 vs 我们的版本
- PyG 2.3.1 vs 我们的版本

**缓解**：
1. RXNGraphormer 有 [pytorch2 分支](https://github.com/licheng-xu-echo/RXNGraphormer/tree/pytorch2)
2. 可以作为独立 conda 环境，通过子进程/API 调用
3. 如果只用 embedding，可以预计算好存文件，完全避免运行时依赖

---

## 10. 总结

### RXNGraphormer 是什么

- 一个**统一的反应预测框架**（GNN + Transformer，13M 预训练）
- 在 yield/selectivity 预测 + 逆/正合成方面都达到了**有竞争力的**性能
- Nature Machine Intelligence 2025，有分量的引用

### RXNGraphormer 不是什么

- ❌ 不是逆合成 SOTA（56.8% vs 60%+ 专用模型）
- ❌ 不是快速推理引擎（序列生成太慢，不适合搜索循环）
- ❌ 不支持酶反应（纯有机训练）
- ❌ 不能做多步规划或级联分析
- ❌ 不能替代我们现有的、已经工作的逆合成模型

### 我们应该怎么用它

1. **用 RXNEMB 提取反应 embedding**（高质量、零训练成本、有 Nature MI 引用）
2. **在上面训轻量 heads**（TypeHead, ConditionHead, ValueHead, FeasibilityHead）
3. **保持现有逆合成模型**（template_relevance, ONMT, Graph, MLP）
4. **级联功能自建**（规则引擎 + 条件预测 + 后处理）
5. **可选**：把 RXNG2Sequencer 作为 MultiOneStepRunWrapper 中的低权重备选

### 修正了之前可行性分析的过度乐观

之前 FEASIBILITY_ANALYSIS.md 中将 RXNGraphormer 定位为"骨干替代品"，暗示可以用它替代整个反应编码器。**这个定位需要修正**：

- ✅ 正确的：用 RXNEMB 做 embedding → 接 heads
- ❌ 不正确的：用 RXNGraphormer 替代现有逆合成模型
- ❌ 不正确的：暗示 RXNGraphormer 能处理级联

**RXNGraphormer 是一个好的配角（embedding 提供者），但不是主角（逆合成引擎）。我们的主角仍然是现有的 template-based + seq2seq 逆合成模型 + 自建的 MCTS* 搜索。**
