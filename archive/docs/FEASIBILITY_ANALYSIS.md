# v3.1 架构方案可行性分析

> **日期**：2026-04-19  
> **目的**：逐条验证 CODE_ARCHITECTURE_v3.md 中的技术假设，标注风险等级，提出修正建议  
> **评估标准**：每个组件给出 ✅ 可行 / ⚠️ 有风险 / ❌ 不可行，附证据和替代方案

---

## 1. 反应编码器骨干

### 1.1 v3.1 声称

> "D-MPNN on Reaction Graph"作为 Shared Reaction Encoder，参数量 ~20-30M。

### 1.2 事实核查：RXNGraphormer (Nature Machine Intelligence, 2025)

v3.1 方案参照的 CGR-GCNN (Heid & Green, JCIM 2022) 已不再是 SOTA。**RXNGraphormer** (Xu et al., Nature Machine Intelligence, 2025) 是目前与我们 URC 构想最接近的已发表工作：

| 维度 | RXNGraphormer | CGR-GCNN (2022) | v3.1 原提议 |
|------|--------------|-----------------|-------------|
| 架构 | GCN (4层) + Transformer (4层, 2 heads, ff=768) | D-MPNN (3层) | D-MPNN (4层) |
| emb_dim | 256 | 300 | 512 |
| 预训练数据 | **13M 反应** | 无预训练 | 4M |
| 预训练任务 | 反应类型分类 (cross-entropy) | — | 掩码原子恢复 |
| 跨任务 | ✅ yield, selectivity, retro/forward-synthesis | ❌ 单任务 | ✅ 6 heads |
| 迁移学习 | 分类器预训练 → 回归/seq2seq 微调 | — | Stage 1→2→3 |
| 推理速度 | ~4 rxn/sec (回归, AMD CPU), ~2.5 rxn/sec (seq gen) | 未报告 | 未估计 |
| 参数量 | ~数 M (GCN 4层×256 + Transformer 4层, ff=768) | ~378K (默认) | 20-30M |
| License | MIT | MIT (Chemprop) | — |
| 发表 | **Nature Machine Intelligence 2025** | JCIM 2022 | — |
| 代码+权重 | ✅ GitHub (41 stars) + Figshare 预训练权重 | ✅ Chemprop | 不存在 |

**核心发现**：
1. RXNGraphormer 在 **8 个基准 + 3 个外部验证集** 上达到 SOTA，且已经是「统一预训练 + 跨任务微调」范式——和 URC 的思路高度一致
2. 它用 **GCN + Transformer** 而非纯 D-MPNN：GCN 编码分子内信息，Transformer 捕获分子间关系（反应物↔产物）
3. 预训练在 **13M 反应** 上做分类任务（远超我们 4M 计划），微调只需几百到几万条数据就能下游 SOTA
4. 已有 **反应 embedding API** (`RXNEMB`)，可直接输出固定长度向量，无需自己训骨干
5. 训练超参：batch=5120, 20 epochs, AdamW, noamlr scheduler (warmup 60K steps)

### 1.3 风险评级：⚠️→✅ 如果正确利用 RXNGraphormer

**原方案（自写 D-MPNN, 20-30M）的风险是过度设计 + 重复造轮子。**

### 1.4 修正建议

**三个可选方案，按推荐顺序：**

**方案 A（推荐）：直接使用 RXNGraphormer 预训练权重提取 embedding**

```python
from rxngraphormer.rxn_emb import RXNEMB

# 加载 13M 反应预训练模型，直接获取反应 embedding
rxnemb = RXNEMB(pretrained_model_path="./model_path/pretrained_classification_model",
                model_type="classifier")
reaction_embeddings = rxnemb.gen_rxn_emb(["reactants>>product", ...])
# → (N, emb_dim) 固定长度向量，可直接输入我们的 task heads
```

优势：
- **零骨干训练成本**：13M 反应预训练已完成，我们只需训 heads
- Nature MI 2025 论文背书，reviewer 不会质疑骨干质量
- MIT 协议，可自由使用和修改
- 论文可写"基于 RXNGraphormer 预训练表示，新增级联兼容性预测"——合理增量创新

**方案 B：在 RXNGraphormer 骨干上微调**

加载其 GCN+Transformer 权重，冻结/微调下层，在上面接我们的 heads。需要改动 RXNGraphormer 代码但仍以其架构为基础。

**方案 C（保底）：使用 Chemprop v2 作为骨干**

Chemprop v2.2.3 (2.3k stars, MIT) 的 D-MPNN 也可用，但劣势明显：无预训练权重、无 Transformer 层、JCIM 2022 发表。

**建议**：先用方案 A 快速验证（1-2 天）——如果 embedding 质量足以支撑 heads 性能，就不需要训骨干；如果不够再用方案 B 微调。

**注意事项**：RXNGraphormer 基于 PyTorch 1.12.1 + PyG 2.3.1 (Python 3.8)，版本较旧。可查看其 pytorch2 分支，或必要时用 Chemprop v2 方案 C 作为兜底。

---

## 2. 多任务训练的数据不平衡

### 2.1 v3.1 声称

> 6 个 head 联合训练，动态权重（GradNorm 或 uncertainty weighting）。

### 2.2 数据规模矩阵

| Head | 数据量 | 相对 Cascade 比例 |
|------|--------|-------------------|
| TypeHead | ~2M | 400× |
| ConditionHead | ~2M | 400× |
| FeasibilityHead | ~2M | 400× |
| EnzymeHead | ~250K | 50× |
| ValueHead | ~100K | 20× |
| **CascadeHead** | **~5K-20K** | **1×** |

**数据量跨度 3 个数量级**。这不是"Type 比 Enzyme 大 10 倍"的常规不平衡，而是 CascadeHead 比最大 head 小 **100-400 倍**。

### 2.3 风险评级：⚠️ 高风险（CascadeHead 被淹没）

GradNorm 的设计目的是让各任务以相似速率学习，但它的假设是各任务都有充足的数据。当一个 task 的数据比其他 task 小两个数量级时：

- 每个 epoch，CascadeHead 看到 ~5K 样本，TypeHead 看到 ~2M 样本
- 骨干的梯度将被大任务主导
- GradNorm 可以提高 CascadeHead 的 loss 权重，但提高 400 倍的权重会导致训练不稳定

### 2.4 修正建议

**v3.1 的 Stage 3 冻结骨干策略本身是正确的，但应该做得更彻底：**

```
实际可行的训练方案：

Stage 1 · 自监督预训练
  - 数据：USPTO + ORD + ECReact (~4M)
  - 任务：掩码原子恢复
  - 正常训练即可
  
Stage 2 · 多任务微调（5 个大数据 head）
  - 训练 TypeHead, ConditionHead, EnzymeHead, FeasibilityHead, ValueHead
  - **不训 CascadeHead**（数据太少，此阶段放进去是制造噪音）
  
Stage 3 · 级联专项微调
  - **冻结骨干和前 5 个 head**
  - 只训 CascadeHead + Compatibility Model
  - 小学习率 (1e-5)，可以微调骨干最后 1-2 层
```

**关键调整**：不要在 Stage 2 就把 CascadeHead 扔进去联合训。这是"一碗水里加两滴墨"，对骨干没贡献，对 CascadeHead 也学不到什么。

---

## 3. CascadeHead 与兼容性模型的数据充分性

### 3.1 v3.1 声称

> 正例 ~2000-5000（文献提取），负例 ~10000+（规则增强），条件冲突负例 ~5000。

### 3.2 分析

**正例**：
- 2000-5000 个成功 one-pot 步骤对，**这是乐观估计**
- CASCADE_EXTRACTION_RESPONSE.md 的搜索策略建议从 ~5000 篇论文中提取
- 实际提取率（论文中含有结构化可提取的级联步骤对）通常 20-40%
- 现实估计：**1000-2000 个正例**

**负例**：
- 规则增强负例 (~10K)：本质是"知道 Pd+酶不兼容就生成 Pd+酶的配对"
- 条件冲突负例 (~5K)：本质是"URC 预测的条件参数差距大就标为不兼容"
- **问题**：这些负例是确定性规则的投射，不是经验型的"试过不行"

**数据结构**：
```
训练集构成：
  正例：~1000-2000 真实级联步骤对，标注 "兼容"
  负例：~15000 规则生成的步骤对，标注 "不兼容"
  比例：约 1:10，严重不平衡，且负例高度同质
```

### 3.3 风险评级：⚠️ 高风险（模型可能退化为规则的近似器）

如果训练集中 85%+ 的负例来自规则增强，那模型学到的是"复述规则"，而不是"从数据中发现新模式"。这样的模型对 reviewer 没有说服力——"你的 ML 模型跟规则的区别是什么？"

### 3.4 修正建议

**接受现实，分两步走：**

**Step 1（rules-first paper，可以先发）**：
- 不训 ML 兼容性模型
- 用硬编码规则 + URC 条件预测的冲突检测
- 论文贡献：(1) 首次定义级联兼容性预测任务，(2) 首个结构化数据集，(3) 规则基线系统，(4) 集成到搜索中
- 这已经是一篇有意义的系统论文

**Step 2（data-driven paper，数据充分后）**：
- 当正例积累到 5000+ 时，训 ML 模型
- 消融实验：rules only vs ML only vs rules+ML
- 证明 ML 模型能发现规则未覆盖的兼容/不兼容模式

**v3.1 文档 §12 已经提到了这个策略（"规则基线先发 method paper，数据补齐后发 data+model paper"），确认这是正确的保底方案。**

---

## 4. ORD 条件标注质量

### 4.1 v3.1 声称

> ORD ~2M 反应用于 ConditionHead 训练（T, pH, solvent, catalyst 等）。

### 4.2 分析

ORD (Open Reaction Database) 确实有 ~2M 反应，但字段完整度参差不齐：

| 字段 | 估计覆盖率 | 说明 |
|------|-----------|------|
| temperature | ~60-70% | 很多条目只记录了"room temperature"或缺失 |
| solvent | ~70-80% | 相对完整 |
| catalyst | ~30-50% | 很多反应未标注催化剂 |
| **pH** | **<10%** | pH 几乎只出现在生化反应中，有机反应极少标注 |
| **cofactors** | **<5%** | ORD 以有机反应为主，辅因子信息极少 |

**pH 和 cofactors 恰恰是级联兼容性最关键的参数**，但在 ORD 中覆盖率最低。

### 4.3 风险评级：⚠️ 中等风险

ConditionHead 对温度和溶剂的预测可能靠谱，但**对 pH 和辅因子的预测将严重缺乏训练信号**。

### 4.4 修正建议

1. **补充 BRENDA/ECReact 的酶反应条件**：BRENDA 对 pH、温度、辅因子有较好的标注（这些就是酶学数据库的核心字段）
2. **ConditionHead 分两块设计**：
   - 有机条件（T, solvent, catalyst）：用 ORD 训练，覆盖率够
   - 酶条件（pH, cofactors, 最适温度）：用 BRENDA + ECReact 训练
3. **必须在训练前做一次 ORD 字段覆盖率审计**，统计各字段的非空率，再决定训练策略

---

## 5. 酶反应的原子映射 (Atom Mapping)

### 5.1 v3.1 声称

> D-MPNN 反应编码使用"反应差异池化：product_repr - reactant_repr（捕获化学变化）"。

### 5.2 分析

CGR 方法的核心前提是**正确的原子到原子映射 (atom-to-atom mapping, AAM)**。没有 AAM：
- CGR 无法构建（不知道反应物和产物中哪些原子对应）
- "反应差异"无法计算（减法没有意义）

**有机反应 AAM 状态**：RXNMapper (Schwaller et al., 2021) 在 USPTO 上准确率 >97%，成熟可用。

**酶反应 AAM 状态**：
- ECReact 的反应已经有 AAM 标注，但只有 ~250K 反应
- RXNMapper 是在有机反应上训练的，对酶催化反应的准确率**未经系统验证**
- 酶反应中常涉及大分子底物（如糖苷、多肽片段），图规模更大，MPNN 消息传递可能受限
- Heid & Green (2022) 论文明确指出：*"Like the dual GCNN architecture, it relies on correct atom mapping of reactions, which increases the work load on preprocessing steps of databases significantly. Incorrect atom mappings add noise to the data."*

### 5.3 风险评级：⚠️ 中等风险

如果酶反应的 AAM 质量差，骨干在酶反应上的表征质量就会差，进而影响 EnzymeHead 和 CascadeHead（级联涉及大量酶步骤）。

### 5.4 修正建议

1. **先用 ECReact 已有 AAM 的数据**（~250K），不要自己跑 RXNMapper 在酶反应上
2. **对 RXNMapper 在 ECReact 子集上做一次准确率验证**（随机抽 1000 条，与已有 AAM 比较）
3. **备选方案**：如果 AAM 不靠谱，CascadeHead 可以用 Morgan FP difference 代替 CGR，牺牲部分精度但避免 AAM 依赖。Heid & Green 论文已证明 Morgan FP 在某些任务上性能也不差（只是不如 CGR GCNN）

---

## 6. 搜索延迟

### 6.1 v3.1 声称

> 每个新步骤立即做 URC 推理，获得类型+条件+级联特征。

### 6.2 分析

搜索中每次扩展需要：
1. SingleStepExpander 生成 ~50 候选反应
2. 对每个候选做 URC 前向传播

**URC 推理延迟估算**（使用 Chemprop 规模的模型，~400K-3M params）：

| 操作 | 单条延迟 | 50 条批量 |
|------|---------|-----------|
| SMILES → 分子图构建 | ~1-5ms | ~50-250ms (可批量) |
| D-MPNN 前向传播 | ~5-20ms | ~50-100ms (GPU 批量) |
| 6 个 head 推理 | ~1ms | ~5ms |
| **合计** | ~10-25ms | **~100-350ms** |

**搜索规模**：
- 典型 MCTS 搜索：500-5000 次扩展
- 每次扩展 50 候选 → 25K-250K 次 URC 推理
- 批量处理（每 50 条一批）→ 500-5000 个批次
- **总 URC 推理时间：50s - 880s**

**对比 ChemEnzyRetroPlanner 现状**：
- 旧系统搜索不做 URC 推理，搜索本身耗时 ~30-60s
- 加上 URC 后可能增加 2-15 倍

### 6.3 风险评级：⚠️ 中等风险（可控）

### 6.4 缓解方案

1. **缓存**：同一个反应 SMILES 不需要重复推理。加入 LRU cache (maxsize=100000)，实际计算量可降低 30-60%
2. **延迟评估**：搜索阶段只对 top-10 候选（而非全部 50）做 URC 推理，其余用快速启发式筛掉
3. **分离式评估**：搜索阶段不做兼容性判断，只记录 reaction_smiles；搜索完成后对成功路线做 batch URC 推理 + 兼容性评分
4. **GPU 批量优化**：合理使用 DataLoader + pinned memory + GPU batch

**建议 MVP 方案**：先采用方案 3（分离式评估），验证端到端效果后再考虑方案 1+2（在线兼容性指导搜索）。这样搜索速度不降级，同时仍能提供级联评分。

---

## 7. 训练资源估计

### 7.1 v3.1 声称

> Stage 1: ~1 周, 2×A100; Stage 2: ~3 天, 2×A100; Stage 3: ~1 天, 1×GPU

### 7.2 验证

**参考数据点**：
1. **RXNGraphormer** (Xu et al., Nat. MI 2025)：13M 反应预训练 20 epochs, batch=5120, multi-GPU
2. **CGR-GCNN** (Heid & Green, JCIM 2022)：30-100 epochs on ≤1.07M reactions, ~378K params

**修正后方案（使用 RXNGraphormer 预训练骨干）**：

| Stage | 数据 | 操作 | 预估时间 |
|-------|------|------|----------|
| Stage 1 预训练 | — | **跳过！直接用 RXNGraphormer 预训练权重** | **0 天** |
| Stage 2 多任务 | 2M+ 反应 × 30 epochs | 冻结骨干，训 5 个 heads | 1-2 天 (1×A100) |
| Stage 3 级联 | 20K 对 × 100 epochs | 训 CascadeHead + Compat (~2M) | 数小时 |
| **总计** | | | **2-3 天** |

**对比原方案**：
- v3.1 原方案（自训骨干 50M params, 2×A100）：9-13 天
- 之前修正方案（Chemprop 骨干 ~2-3M, 2×A100）：4-6 天
- **当前方案（RXNGraphormer 预训练权重）：2-3 天 (1×A100)**

使用 RXNGraphormer 预训练权重可节省约 **1-2 周训练时间**，且只需 1 张 A100。

### 7.3 风险评级：✅ 可行，且大幅优于原方案

---

## 8. 骨干选型：RXNGraphormer vs Chemprop v2 vs 自写

### 8.1 三方对比

| 维度 | 自己写 D-MPNN | Chemprop v2 | **RXNGraphormer** |
|------|-------------|-------------|-------------------|
| 开发时间 | 2-3 周 | 1-3 天 | **1-2 天（直接用 embedding API）** |
| bug 风险 | 高 | 低 | **极低（直接调 API）** |
| 论文引用 | "our custom D-MPNN"（弱） | Chemprop (Yang 2019; Heid 2024)（中） | **RXNGraphormer (Xu et al., Nat. Mach. Intell. 2025)**（强） |
| 预训练权重 | ❌ 无 | ❌ 无 | **✅ 13M 反应预训练** |
| 跨任务验证 | ❌ | 单任务 | **✅ 8 基准 + 3 外部集 SOTA** |
| 反应 embedding | 需自己设计 | CGR 方案 | **✅ `RXNEMB` API 直接输出** |
| 架构 | D-MPNN only | D-MPNN only | **GCN + Transformer（更强表达力）** |
| License | — | MIT | **MIT** |
| 发表 | — | JCIM 2022 | **Nature Machine Intelligence 2025** |

### 8.2 风险评级：✅ 最佳方案已明确

### 8.3 建议

**优先使用 RXNGraphormer 作为骨干（方案 A）。**

```python
# 方案 A：直接使用 RXNGraphormer embedding
from rxngraphormer.rxn_emb import RXNEMB

class ReactionEncoder:
    """基于 RXNGraphormer 预训练模型的反应编码器"""
    def __init__(self, model_path):
        self.rxnemb = RXNEMB(
            pretrained_model_path=model_path,
            model_type="classifier"  # 使用分类预训练模型
        )
    
    def encode(self, rxn_smiles_list: list[str]):
        # 直接获取 13M 反应预训练的 embedding
        return self.rxnemb.gen_rxn_emb(rxn_smiles_list)
        # → (N, emb_dim) tensor, 可直接输入我们的 heads

# 方案 B：加载骨干权重后微调
# 需要 fork RXNGraphormer 代码，在其 GCN+Transformer 后接自定义 heads
# 更灵活但开发量更大
```

**好处**：
- **零骨干训练成本**：13M 反应预训练已完成，我们只训 heads
- 论文中引用 Nature MI 2025，比引用 JCIM 2022 更有说服力
- GCN + Transformer 捕获分子内+分子间信息，表达力强于纯 D-MPNN
- 开发时间从"写骨干 2-3 周"缩短到"封装 adapter 1-2 天"

**注意**：如果 RXNGraphormer 的 PyTorch 1.12.1 / PyG 2.3.1 依赖与其他组件冲突，可回退到 Chemprop v2 方案。

---

## 9. SingleStepExpander 权重复用

### 9.1 v3.1 声称

> MVP: 加载 ChemEnzyRetroPlanner 的模板模型权重 + 已训好的 ONMT 权重。

### 9.2 分析

- ChemEnzyRetroPlanner 的 `template_relevance` 模型用的是 MLP + Morgan FP，权重格式通常是 `.pt` 或 `.pth`
- ONMT 权重是 OpenNMT 格式
- 新 Expander 接口 (`SingleStepExpander ABC`) 只需封装加载逻辑

### 9.3 风险评级：✅ 低风险

权重本身是训好的研究产物，加载到新 Expander 中只是工程问题。唯一风险是依赖版本（PyTorch/ONMT 版本变化可能导致权重格式不兼容），但可以通过 `torch.load` 的兼容性参数解决。

---

## 10. MCTS* 重写

### 10.1 v3.1 声称

> 全新实现 MCTS*，加入级联感知。

### 10.2 分析

MCTS* (Retro*) 是经过充分验证的算法：
- Chen et al. (2020) Retro*: Top-down search
- 已有多个开源实现（RetroStar, AiZynthFinder, ASKCOS）
- ChemEnzyRetroPlanner 本身就有一个实现 (`search_frame/mcts_star/`)

新增的级联感知功能（在节点中存储 cascade_feature、兼容性评分）是对数据结构的扩展，不改变算法核心。

### 10.3 风险评级：✅ 低风险

算法成熟，参考实现丰富。新增级联字段是增量修改。

---

## 11. 综合可行性评估

### 11.1 组件风险总览

| 组件 | 风险 | 核心问题 | 建议调整 |
|------|------|---------|---------|
| 反应编码器骨干 | ⚠️→✅ | 20-30M 过大，自写重复造轮子 | **用 RXNGraphormer (Nat. MI 2025) 预训练权重**，零训练成本 |
| 多任务训练 | ⚠️ | CascadeHead 数据量差 100-400× | Stage 2 不训 CascadeHead，Stage 3 专门训 |
| 级联数据充分性 | ⚠️ | 正例可能只有 1000-2000 | 规则基线先行，ML 模型等数据充足后再发 |
| ORD 条件质量 | ⚠️ | pH/cofactor 覆盖 <10% | 补充 BRENDA 酶条件数据，分有机+酶两块训 |
| 酶反应原子映射 | ⚠️ | RXNMapper 在酶反应上未验证 | 先用 ECReact 已有 AAM，做 RXNMapper 准确率评估 |
| 搜索延迟 | ⚠️ | URC 在线推理可能增加 2-15× 搜索时间 | MVP 用分离式评估，不在搜索中做在线 URC |
| 训练资源 | ✅ | 合理 | 小模型 1 周，大模型 2 周 |
| Expander 权重复用 | ✅ | 工程问题 | 正常处理 |
| MCTS* 重写 | ✅ | 算法成熟 | 正常实现 |

### 11.2 没有 ❌ 不可行项

**所有组件经过调整后都是可行的。** 没有"走不通"的致命缺陷。但有 5 个 ⚠️ 需要认真对待，不能按原方案硬上。

### 11.3 关键调整总结

| 原方案 | 调整后方案 | 理由 |
|--------|-----------|------|
| 自写 D-MPNN, 20-30M | **RXNGraphormer 预训练骨干** (Nat. MI 2025) | 13M 反应预训练 + 8 基准 SOTA + MIT 开源 + 零训练成本 |
| 6 heads 同时联合训练 | Stage 2 只训 5 heads, Stage 3 单独训 Cascade | 数据量差距太大 |
| ML 兼容性模型为核心贡献 | 规则基线为 MVP 核心贡献 | 正例不足，规则已有价值 |
| URC 在线指导搜索 | 先分离式评估，后集成 | 降低搜索延迟风险 |
| ConditionHead 用 ORD 统一训 | 有机/酶条件分别训练 | pH/cofactor 在 ORD 中覆盖极低 |

---

## 12. 修正后的开发优先级

```
Phase 0 · 基础设施（第 1 周）
  - 创建仓库 + pyproject.toml
  - 安装 RXNGraphormer 作为依赖（备选 Chemprop v2）
  - 实现 chem utils, mol_graph, stock_db
  - 实现 SingleStepExpander ABC + template_based adapter（加载旧权重）
  
Phase 1 · 搜索引擎（第 2-3 周）
  - 实现 MCTS*（不含级联感知，先跑通基本搜索）
  - 端到端验证：target → expand → search → routes
  - ← 这里已经可以复现 ChemEnzyRetroPlanner 的搜索能力
  
Phase 2 · URC（第 3-5 周）
  - 基于 RXNGraphormer 预训练权重封装 ReactionEncoder（方案 A / B）
  - 实现 5 个 heads（不含 CascadeHead）
  - Stage 1 预训练 + Stage 2 微调
  - 验证各 head 性能：TypeHead acc, EnzymeHead top-k, etc.
  
Phase 3a · 规则基线 + 评分（第 5-7 周）
  - 实现硬编码规则系统（rules.py）
  - 实现路线级联评分（cascade_scorer.py + pot_optimizer.py）
  - 将 URC 条件预测接入规则系统
  - ← 这里可出第一篇论文（系统论文：任务定义 + 数据集 + 规则基线 + 搜索集成）
  
Phase 3b · ML 兼容性模型（数据充足后）
  - 实现 CascadeHead + Compatibility Model
  - Stage 3 微调
  - 消融实验
  - ← 第二篇论文（模型论文：ML 超越规则基线）
```

---

## 13. 对论文贡献的影响

修正后的贡献层次：

1. **Primary**：级联兼容性预测任务定义 + 首个结构化数据集 + 评估基准 → **不依赖 ML 模型是否成功**
2. **Primary**：级联感知合成路线规划（规则基线 + MCTS 集成）→ **现有工具的唯一空白**
3. **Secondary**：URC 统一反应表征器（RXNGraphormer 预训练骨干 + 5 heads）→ **替代碎片化方案**
4. **Secondary/Tertiary**：ML 级联兼容性模型 → **data-dependent，可能升为 primary 如果数据充足**

**即使 ML 兼容性模型效果不好，1+2 已经是一篇够格的论文。** 这是可行性分析的核心结论：我们的贡献不依赖于最难实现的组件。

---

## 14. 结论

**v3.1 架构在概念层面是正确的**——统一模型替代碎片化、级联兼容性作为核心差异化、可插拔单步逆合成。但在以下 5 个实现细节上需要调整：

1. **用 RXNGraphormer (Xu et al., Nat. Mach. Intell. 2025) 预训练权重作为骨干**，而非自写 D-MPNN
2. CascadeHead 单独训练而非联合训练
3. 规则基线先行，ML 模型视数据定
4. 搜索先用分离式评估
5. 条件预测分有机/酶两路

调整后的方案在 **1×A100 上 2-3 天**即可完成训练（骨干预训练直接跳过），代码量减少 ~50%（不用写反应编码器骨干），且论文贡献不依赖于最高风险组件（ML 兼容性模型）。
