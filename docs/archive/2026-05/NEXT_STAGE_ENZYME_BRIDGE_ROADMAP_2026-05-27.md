# 下一阶段路径：Enzyme-aware Chemo-enzymatic Bridge Planning

日期：2026-05-27

本文档用于把专家反馈、内部实验结果和当前项目困惑整理成下一阶段可执行路线。核心前提是：**没有持续专家标签，但项目仍要做成可训练、可验证、可扩展。**

## 0. 一句话结论

项目不应继续以 one-pot cascade 条件兼容作为主创新，也不应退化成普通多步逆合成。下一阶段应重构为：

> **Enzyme-aware chemo-enzymatic bridge planning：识别化学中间体与酶底物空间之间的可连接性，并通过 weak labels、hard negatives、enzyme-substrate verifier 和 gated search 控制酶步假阳性。**

更直接地说，项目要回答的问题从：

> 这些步骤能不能 one-pot 级联？

改为：

> 一个化学中间体是否值得进入酶催化空间？如果值得，它可能对应什么 EC / enzyme family / substrate-product transform？这个判断有什么证据？

## 1. 专家思路的核心梳理

### 1.1 级联不是无意义，而是定义错了

专家反馈的关键不是“级联无意义”，而是原来把级联理解成 one-pot 条件兼容太窄。现实中 chemo-enzymatic synthesis 经常是序贯操作、分步纯化、buffer/溶剂切换。此时 pH、温度、溶剂兼容仍有价值，但更适合作为 route audit，而不是路线生成的核心创新。

新的级联定义应是：

1. 化学步骤产物是否可以成为酶步底物。
2. 酶步产物是否可以成为后续化学步骤底物。
3. 某个中间体是否进入已知酶底物空间。
4. 某个酶催化机会是否有底物、反应中心、立体化学、EC/family、precedent 和可选 3D/active-site 证据。

因此，“级联”的新表达建议使用：

1. `chemo-enzymatic bridge`
2. `hybrid route bridge`
3. `enzyme-aware retrosynthetic proposal`
4. `substrate-enzyme-aware route planning`

避免继续主打 `one-pot cascade planner`。

### 1.2 当前真正缺口是酶-底物判断

目前系统对酶步的处理仍然偏浅：

1. 更像给路线贴 EC 标签，而不是判断酶是否能催化该底物。
2. 缺少 enzyme-substrate-product triad 判断。
3. 缺少反应中心、区域选择性、立体化学、cofactor/common metabolite 假连接过滤。
4. 缺少 hard-negative-trained verifier。
5. 缺少“什么时候调用酶 proposal”的 gated policy。

专家建议实际指向一个更具体的问题：

> **什么酶催化什么底物，以及什么底物需要什么酶催化。**

这是项目下一阶段最值得投入的核心。

### 1.3 没有专家标签不是死局

没有专家持续标注时，不能训练“专家偏好路线模型”，但可以构造自动证据栈：

1. 真实酶反应正样本。
2. 化学产物与酶底物 exact bridge。
3. 化学产物与酶底物 similarity bridge。
4. 酶产物与化学底物 bridge。
5. 反应中心一致性。
6. 立体化学一致性。
7. EC/family/reaction class 一致性。
8. cofactor/common metabolite 排除。
9. hard negatives。
10. heldout literature recovery。

因此，问题不是“数据完全不够”，而是：

> **高质量正样本少，弱标签很多，负样本构造更关键。**

## 2. 内部实验给出的约束

### 2.1 Proposal 覆盖可以提升

`benchmark_v2_100` final/top-level GT step one-step 评估：

| proposal 来源 | Top1 exact reactant | Top5 exact reactant | Top16 exact reactant | 平均耗时 |
|---|---:|---:|---:|---:|
| ChemEnzy native 两模型 | 5/100 | 12/100 | 17/100 | 1.4903 s |
| AiZynthFinder uspto | 8/100 | 16/100 | 17/100 | 0.2760 s |
| RetroKNN external150k | 16/100 | 17/100 | 18/100 | 0.1096 s |

组合 Top16 union：

| 组合 | Top16 覆盖 |
|---|---:|
| native | 17/100 |
| native + AiZ | 21/100 |
| native + RetroKNN | 30/100 |
| native + AiZ + RetroKNN | 33/100 |

结论：

1. 新增 proposal 确实能补 native 候选池缺口。
2. RetroKNN 对真实反应物进入候选池最有帮助。
3. AiZynthFinder 对 Top1/Top5 排序和模板补充有价值。

### 2.2 Naive ensemble 会污染搜索

前 5 个目标多步搜索小样本：

| 配置 | exact reaction in route pool | gt reactant in route pool | 平均 cascade search 时间 |
|---|---:|---:|---:|
| baseline | 0.2 | 0.6 | 0.0278 s |
| AiZ + RetroKNN top16 | 0.0 | 0.6 | 27.57 s |
| AiZ + RetroKNN top4 | 0.0 | 0.6 | 12.9732 s |

provider 调用量：

| 配置 | provider 调用 | 返回候选 |
|---|---:|---:|
| baseline | static 17 次 | 166 |
| AiZ + RetroKNN top16 | static/aiz/retroknn 各 389 次 | 约 8900 |
| AiZ + RetroKNN top4 | static/aiz/retroknn 各 359 次 | 约 2890 |

结论：

1. 候选池增强有效，但不能全节点常开。
2. 搜索需要 gated routing、provider budget 和 verifier。
3. 下一阶段不应继续“所有模型全开”，而应研究“何时调用、如何过滤、如何校准置信度”。

## 3. 下一阶段总体技术路线

推荐路线：

```text
数据体检
→ virtual chemoenzymatic bridge pack
→ hard negative pool
→ enzyme substrate/product retriever
→ enzyme feasibility verifier
→ bridge scorer
→ gated route search
→ 3D/active-site reranking for top bridges
```

阶段优先级：

```text
数据工程 50%
verifier / retriever 30%
搜索策略 15%
大模型 5%
```

不建议马上做：

1. 端到端 chemo-enzymatic route generator。
2. 1B+ reaction foundation model。
3. 全节点 3D/docking。
4. 把 virtual bridge 当真实路线 ground truth。
5. 继续仅基于少量 cascade 正样本训练 route ranker。

## 4. 数据路线

### 4.1 数据层级

| 数据类型 | 作用 | 第一阶段处理重点 |
|---|---|---|
| 普通化学反应 | 产生 chemical product pool，支撑 RetroKNN/chemical proposer | 去重、产物抽取、反应中心、模板/类别 |
| 酶反应 | enzyme substrate/product/reaction class/EC 来源 | 去 cofactors、balanced、atom mapping、EC 分层 |
| 酶-底物-产物对 | 训练 enzyme verifier | 构造 triad、正负样本、split |
| 化学-酶 bridge | 新主线核心弱标签 | exact/similarity/reaction-center/stereo 分层 |
| 酶序列 | sequence-aware verifier / candidate enzyme | 只使用 EC-linked / reaction-linked 子集 |
| 酶结构 | top bridge 验证 | 不进主搜索，只做 rerank/validation |

### 4.2 virtual_chemoenzymatic_bridge_pack

目标文件：

```text
chemical_product_pool.parquet
enzyme_substrate_product_pool.parquet
exact_bridge.parquet
similarity_bridge_raw.parquet
similarity_bridge_filtered.parquet
hard_negative_pool.parquet
bridge_confidence_tiers.parquet
```

bridge 方向：

1. `chemical_product -> enzymatic_substrate`
2. `enzymatic_product -> chemical_substrate`
3. `chemical_product ~ enzymatic_substrate`
4. `enzymatic_product ~ chemical_substrate`

bridge 字段：

```text
bridge_id
bridge_direction
connector_molecule_smiles
connector_inchikey
chemical_reaction_id
enzyme_reaction_id
EC / enzyme_family / sequence_id
reaction_class
match_type
2D similarity
reaction_center_similarity
stereo_consistency
cofactor_artifact_flag
common_metabolite_flag
evidence_source
confidence_tier
negative_flags
```

### 4.3 Confidence tiers

| Tier | 定义 | 用途 |
|---|---|---|
| Tier 1 | exact/stereo-aware match + 非 cofactor/common metabolite + 反应中心兼容 | strong weak positive |
| Tier 2 | high similarity + reaction center 一致 + stereochemistry 不冲突 | medium positive |
| Tier 3 | high similarity 但 reaction center 不确定 | retriever candidate，不直接监督 verifier |
| Tier 4 | 只有 EC/官能团/粗相似 | hard negative 或低权重弱正 |
| Tier 5 | cofactor/common metabolite 导致连接 | 剔除或 artifact negative |

第一阶段不要追求百万 bridge。更合理目标是：

```text
exact bridge: 1K-10K
high-confidence similarity bridge: 10K-30K
hard negatives: 0.5M-2M raw pool
```

### 4.4 Hard negatives

hard negatives 是没有专家标签时最关键的质量来源。

必须构造：

| 类型 | 目的 |
|---|---|
| 同 EC，不同底物空间 | 防止模型只看 EC |
| 相似底物，反应中心不同 | 防止全局结构相似误判 |
| 同官能团，不同反应位点 | 检查区域选择性 |
| 同骨架，不同手性 | 检查立体选择性 |
| cofactor/common metabolite 连接 | 防止 ATP/NADH/H2O/CoA 等假 bridge |
| 化学可反应但无酶 precedent | 防止有机反应被误迁移成酶反应 |
| 相似产物但不同 enzyme family | 防止 scaffold leakage |

推荐 batch 构成：

```text
1 positive
1 easy negative
1 EC hard negative
1 reaction-center hard negative
1 stereochemistry hard negative
1 common-metabolite artifact negative
```

## 5. 模型路线

不要训练一个单体大模型。拆成 7 个模块。

### 5.1 Chemical proposer

状态：

1. 保留 ChemEnzy native。
2. 保留 AiZynthFinder ONNX。
3. 保留 RetroKNN。
4. 不优先重训普通化学大模型。

下一步：

1. 做 gated sidecar。
2. 做 proposal quality gate。
3. 只在 root/top-level、native 弱、frontier stuck 时调用 sidecar。

### 5.2 Enzyme reaction proposer

作用：给 product/intermediate 提出可能的酶催化逆合成拆解。

第一阶段不追求强生成模型，可用：

1. EnzymeMap/ECREACT/Rhea retrieval。
2. EC-conditioned 模板/transformer 小模型。
3. substrate-product precedent matching。

推荐数据量：

| 阶段 | 酶反应数据 |
|---|---:|
| P0 | 20K-50K |
| P1 | 50K-100K |
| P2 | 100K-300K |

### 5.3 EnzymeSubstrateRetriever

作用：输入中间体，找相似酶底物、反应、EC、enzyme family。

版本：

1. v0：ECFP/MAP4 + FAISS，无训练。
2. v1：dual encoder，5M-30M。
3. v2：加入 reaction-center contrastive learning。

指标：

```text
Recall@k
MRR
family recall
hard-negative recall
retrieval latency
```

### 5.4 EnzymeFeasibilityVerifier

这是最关键模块。

输入：

```text
substrate
product or reaction center
EC / enzyme class
enzyme sequence optional
cofactor optional
known precedent optional
2D features
3D features optional
```

输出：

```text
feasibility score
confidence tier
failure reason
```

训练规模：

| 阶段 | 正样本 | 负样本池 |
|---|---:|---:|
| P0 | 50K-200K | 0.5M-2M |
| P1 | 200K-1M | 5M-20M |
| P2 | 1M+ | 20M-100M |

模型建议：

| 阶段 | 模型 |
|---|---|
| P0 | fingerprint + EC embedding + MLP/GBDT，1M-10M |
| P1 | molecule encoder + EC/family embedding + cross-attention，20M-80M |
| P1+ | frozen ESM2 embedding + molecule encoder，30M-120M trainable |
| P2 | 2D + sequence + active-site/3D joint verifier |

### 5.5 ChemoEnzymaticBridgeScorer

作用：判断当前化学中间体是否值得触发 enzyme proposal。

输入：

```text
molecule features
retrieval hit quality
reaction center match
EC/family consistency
stereo consistency
cofactor artifact flags
route state
```

输出：

```text
bridge_score
call_enzyme_proposal?
confidence_tier
```

模型规模：5M-30M，或先用 GBDT/MLP。

### 5.6 Route policy / gated router

作用：决定当前节点调用哪个 provider。

输入：

```text
node molecule
route depth
native proposal count/score/diversity
frontier stagnation
stock proximity
bridge_score
has_enzyme_step?
budget used
```

输出：

```text
call_native?
call_AiZ?
call_RetroKNN?
call_enzyme_retriever?
call_verifier?
top_k budget
```

第一版可以完全规则化，不必训练。

### 5.7 3D validator

定位：top bridge reranker / report validator，不进全节点搜索。

阶段：

1. 3D-S0：ligand-only conformer + shape/pharmacophore。
2. 3D-S1：reaction-center-constrained 3D similarity。
3. 3D-S2：docking / PLIF。
4. 3D-S3：AlphaFold3/cofolding。
5. 3D-S4：MD/QM-MM，仅最终 case。

第一阶段只做 3D-S0/S1 的设计，不作为主线阻塞项。

## 6. 搜索路线

核心原则：

> 快生成、强过滤、受控调用。

### 6.1 Gated routing v0

规则：

1. native chemical proposal 先行。
2. root/top-level target 允许 sidecar 调用一次。
3. native 候选数量少、分数低、重复高、forward/audit 失败率高时调用 sidecar。
4. frontier 连续 N 次卡死时调用 sidecar。
5. 只有 bridge scorer 高于阈值，才调用 enzyme retriever/proposer。
6. enzyme proposal 必须经过 feasibility verifier。
7. 低分酶候选不进入主搜索，只保留为 audit evidence。
8. 每个目标设置 provider budget：
   - max AiZ calls
   - max RetroKNN calls
   - max enzyme retriever calls
   - max verifier calls
   - max 3D validations

### 6.2 搜索指标

不再只看 solved rate。重点看：

```text
search cost per useful bridge
search cost per useful route
enzyme step precision
false enzyme proposal rate
evidence-supported route count
route plausibility pass rate
GT reactant in proposal pool
exact reaction in route pool
```

### 6.3 速度原则

多步搜索不是单步越快越好。更关键是有效 branching factor。

| 层级 | 模块 | 目标耗时 |
|---|---|---:|
| S0 | canonicalization / fingerprint / stock check | <10 ms |
| S1 | FAISS retriever / cached sidecar | 10-50 ms |
| S2 | native proposer / enzyme retriever | 50-500 ms |
| S3 | verifier / reranker | 0.1-2 s |
| S4 | 3D shape / conformer rerank | 1-30 s，仅 top candidates |
| S5 | docking / AF3 / MD | 离线验证 |

## 7. 评价体系

### 7.1 P0 必须输出的评价

数据体检：

```text
chemical reaction count
unique chemical product count
enzyme reaction count
unique enzyme substrate/product count
exact bridge count
similarity bridge count
cofactor/common metabolite artifact rate
EC distribution
reaction class distribution
confidence tier distribution
```

模型体检：

```text
retriever Recall@k
bridge precision@k
hard negative rejection rate
EC top-k recall
reaction-center consistency
stereo conflict rate
calibration curve
```

搜索体检：

```text
native baseline
native + ungated sidecar
native + gated sidecar
native + gated enzyme bridge
native + gated enzyme bridge + verifier
```

### 7.2 专家不可持续时的替代标准

每个 enzyme bridge 生成 evidence card：

```text
1. exact/similarity bridge evidence
2. enzyme reaction precedent
3. EC / family consistency
4. reaction center consistency
5. stereochemistry consistency
6. cofactor/common metabolite filtering
7. substrate-product pair verifier score
8. sequence/structure evidence if available
9. forward plausibility
10. route-level value: 是否替代困难化学步骤
```

没有专家时，Tier 1 bridge 至少应满足：

```text
exact or near-exact substrate precedent
reaction center matched
stereochemistry not conflicting
not cofactor artifact
verifier high score
source >= 1 curated database
```

## 8. 阶段路线

### P0：数据体检与 bridge v0（2-3 周）

目标：

证明是否有足够高置信 bridge 数据支撑项目继续。

交付：

1. `chemical_product_pool.parquet`
2. `enzyme_substrate_product_pool.parquet`
3. `exact_bridge.parquet`
4. `similarity_bridge_raw.parquet`
5. `similarity_bridge_filtered.parquet`
6. `hard_negative_pool.parquet`
7. `bridge_confidence_tiers.parquet`
8. P0 数据体检报告。

判定标准：

1. high-confidence bridge 达到 10K+ 为理想。
2. exact bridge 至少达到 1K+。
3. cofactor/common metabolite 假连接比例可控。
4. EC/reaction class 分布不是极端单一。

如果 P0 数据体检失败，优先补酶数据和过滤规则，不进入训练。

### P1：Retriever + verifier v0（3-5 周）

目标：

证明模型能区分“值得进入酶空间”和“相似但错误”的候选。

交付：

1. EnzymeSubstrateRetriever v0。
2. BridgeScorer v0。
3. EnzymeFeasibilityVerifier v0。
4. hard-negative benchmark。
5. calibration report。

判定标准：

1. hard-negative false positive rate 明显低于相似度 baseline。
2. bridge precision@k 可解释。
3. failure reason 能覆盖主要错误类型。
4. verifier 能压低 common metabolite/cofactor artifact。

### P2：Gated search v0（3-4 周）

目标：

把 enzyme bridge 变成搜索中的受控能力，而不是污染搜索。

交付：

1. GatedSidecarRouter。
2. EnzymeBridgeProposalProvider。
3. Provider budget。
4. native vs ungated vs gated ablation。

判定标准：

1. search cost per useful bridge 下降。
2. enzyme proposal false positive rate 下降。
3. useful bridge count 上升。
4. route plausibility 不低于 baseline。

### P3：Enzyme proposer 与中等模型（4-8 周）

目标：

从 retrieval/verifier 进化到可提出酶逆合成候选。

交付：

1. EC-conditioned enzyme proposer。
2. enzyme-substrate-product verifier v1。
3. sequence-aware feature 或 frozen ESM2 embedding。
4. route-level gated enzyme proposal ablation。

判定标准：

1. enzyme reaction proposer 的 top-k 不低于 retrieval baseline。
2. verifier 对 hard negatives 仍保持低 FPR。
3. gated route 中 enzyme step precision 上升。

### P4：3D validator（并行低优先级，后置）

目标：

对 top enzyme bridge 做更强证据验证，而不是作为主搜索循环。

交付：

1. ligand-only 3D shape/pharmacophore scorer。
2. reaction-center-constrained 3D similarity。
3. 少量 active-site-aware case。
4. evidence card 可视化。

判定标准：

1. 3D score 能提高 top bridge precision 或专家可读性。
2. 成本可控。
3. 不阻塞 P0-P2。

### P5：展示与论文/汇报材料

目标：

形成专家可讨论的系统材料。

交付：

1. 10-20 条 evidence-supported chemo-enzymatic bridge routes。
2. 每条路线附 evidence card。
3. native / ungated / gated / gated+verifier 对照。
4. hard negative failure analysis。
5. 资源、局限和下一步实验筛选建议。

## 9. 算力与工程资源

### P0 最小配置

| 资源 | 建议 |
|---|---|
| GPU | 1 × RTX 4090 / A5000 / A6000 / L40S |
| 显存 | 24GB 起步 |
| CPU | 32 cores |
| RAM | 128GB |
| 存储 | 4TB NVMe |

可完成：

1. RDKit preprocessing。
2. bridge exact/similarity matching。
3. FAISS index。
4. MLP/GBDT verifier。
5. 小型 transformer fine-tune。
6. 小规模 search ablation。

### P1 推荐配置

| 资源 | 建议 |
|---|---|
| GPU | 4 × A100 80GB / 4 × L40S / 4 × RTX 4090 |
| CPU | 64-128 cores |
| RAM | 256-512GB |
| 存储 | 10TB NVMe + 归档盘 |
| 数据库 | DuckDB/Parquet + FAISS；必要时 PostgreSQL |

可完成：

1. frozen ESM2 embedding precompute。
2. dual encoder retriever。
3. cross-attention verifier。
4. enzyme proposer fine-tune。
5. large hard-negative sampling。
6. gated search policy learning。

## 10. 立即执行清单

### 第 1 周

1. 冻结旧 one-pot cascade 叙事。
2. 整理现有化学 reaction metadata。
3. 整理现有 Rhea/EnzymeMap/ECREACT/BRENDA 资源状态。
4. 定义 bridge schema。
5. 定义 cofactor/common metabolite blacklist。
6. 写 `build_virtual_chemoenzymatic_bridge_pack.py` 骨架。

### 第 2 周

1. 生成 chemical product pool。
2. 生成 enzyme substrate/product pool。
3. exact bridge matching。
4. 统计 exact bridge 数量和 EC 分布。
5. 构造第一版 hard negatives。
6. 输出 P0 数据体检报告。

### 第 3-4 周

1. similarity bridge 检索。
2. reaction-center filter。
3. stereo filter。
4. BridgeScorer v0。
5. EnzymeSubstrateRetriever v0。
6. hard-negative benchmark。

### 第 5-8 周

1. FeasibilityVerifier v0。
2. calibration report。
3. GatedSidecarRouter。
4. EnzymeBridgeProposalProvider。
5. native vs ungated vs gated ablation。

## 11. 停止事项

下一阶段不再优先做：

1. one-pot 条件兼容作为主创新。
2. 基于少量 cascade 正样本训练 route ranker。
3. 全节点开启 AiZ/RetroKNN/enzyme proposal。
4. 直接扩大多步搜索深度/迭代数。
5. 直接训练端到端 cascade generator。
6. 把 similarity bridge 当强正样本。
7. 全节点 docking / AF3 / 3D validation。
8. 只用 solved rate 做主要指标。

## 12. 对外/对专家表述

建议表述：

> 我们不再把级联限定为 one-pot 工艺，而是研究有机化学步骤和酶催化步骤之间的可接续性。项目核心是识别哪些化学中间体进入酶底物空间、哪些酶促转化有底物-酶-反应中心证据，并在多步逆合成搜索中受控触发酶 proposal。

更短版本：

> 我们做的不是普通多步逆合成，而是 enzyme-aware chemo-enzymatic bridge planning：在化学路线中识别可切换到酶催化的高证据中间体。

## 13. 成功判据

P0 成功：

1. 构建出可用 bridge pack。
2. high-confidence bridge 数量达到可训练量级。
3. hard negative 类型覆盖主要假阳性。
4. 数据体检能说明项目可继续。

P1 成功：

1. retriever 能召回合理 enzyme precedents。
2. verifier 能拒绝相似但错误的酶步。
3. bridge precision 明显优于纯相似度。

P2 成功：

1. gated search 保留 sidecar coverage gain。
2. 搜索成本显著低于 ungated ensemble。
3. enzyme proposal 假阳性下降。
4. 能生成 evidence-supported bridge route examples。

最终成功：

1. 不是简单 solved rate 更高。
2. 而是系统能高精度识别“化学中间体 -> 酶催化机会”的桥接点。
3. 每个酶步都能给出 substrate、EC/family、reaction center、precedent、verifier score 和可选 3D evidence。

## 14. 参考资源

1. Rhea reaction knowledgebase: https://www.rhea-db.org/
2. BRENDA Enzyme Database: https://www.brenda-enzymes.org/
3. ECREACT / RXN biocatalysis model: https://github.com/rxn4chemistry/biocatalysis-model
4. EnzymeMap: https://zenodo.org/records/8254726
5. ACERetro / SPScore: https://pubs.rsc.org/en/content/articlelanding/2025/dd/d5dd00008d
6. TTLAB: https://pubs.rsc.org/en/content/articlehtml/2024/sc/d4sc02408g
7. ChemEnzyRetroPlanner: https://www.nature.com/articles/s41467-025-65898-3
8. AlphaFold: https://deepmind.google/science/alphafold/
