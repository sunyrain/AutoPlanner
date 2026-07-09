# AutoPlanner-Cascade 现状总报告

日期：2026-05-21
版本：v1
用途：统一说明当前框架、问题、结果、冻结边界与下一阶段工作思路。

## 1. 总结

当前项目的主线不是单个 scorer、ranker 或 fallback，而是：

**ChemEnzy 原生多步搜索 + 保守物料/条件审计 + rule verifier gate/annotation + condition / stage / cofactor state-aware search**

训练侧的有效方向是：

**chosen-only ChemEnzy ONMT adapter / context proposal / verifier-derived preference**

但到目前为止，这些训练与 scorer 信号仍主要停留在离线指标或局部路由变化，没有形成稳定的 route-level 提升。

核心判断：**系统不是“完全没信号”，而是“信号没有穿透到候选覆盖、搜索保留和最终路线质量”**。

---

## 2. 当前框架

### 2.1 Runtime 主线

- ChemEnzy native multi-step search
- product / material sanity audit
- rule cascade verifier gate
- learned verifier annotation only
- condition / stage / cofactor search state
- static route rendering and web display

### 2.2 训练侧当前可用线

- chosen-only supervised ChemEnzy ONMT adapter
- context-mode proposal sidecar
- verifier preference derived supervision
- legality / validity filtering of proposals

### 2.3 已冻结 / 归档

- CCTS v0/v1/v2/v3
- route-pool LambdaRank / old ranker
- adjacent-step pair scorer
- block-coherence / block-hard line
- v4 product-value / action-source / provider-retrieval lineage
- expert CSV / LLM review fallback
- learned verifier default rerank

这些方向保留为历史复现和诊断资产，不再作为默认主线。

---

## 3. 目前结果

### 3.1 静态展示与物料审计

statin / 半合成展示曾暴露出几个关键化学误判点：

- 大 terminal 不等于高级产物片段
- 条件试剂可解释反应物里没有的元素
- 不能用 heavy-atom 比例做粗暴硬杀
- 短路线大跳、dummy atom、carrier/protecting group 误判会严重影响展示可信度

对应的结论已经写进主报告和 cleanup 文档。
见 [MAINLINE_CLEANUP_2026-05-20.md](/root/autodl-tmp/AutoPlanner/docs/MAINLINE_CLEANUP_2026-05-20.md)

### 3.2 Learned verifier

learned verifier 已经做到较强的离线 annotation / gate 信号，但没有成为默认 reranker。

关键点：
- 30K perturbation pack 上 stage-aware learned verifier 有明显离线提升
- static statin showcase 上它没有带来 audit-rank 的实质改善
- 所以它现在仍应作为 conservative gate / annotation，而不是主排序器

### 3.3 ChemEnzy adapter / proposal sidecar

chosen-only context adapter 确实学到了东西，但还没有转成 route-level promotion。

关键证据：
- full chosen-only context run 在 valid/test 上 exact recall 有提升
- 但 route-level A/B 没有出现实质提升
- 100-target 对照里 top route 改动很多，但质量几乎没变，且耗时明显上升

对应摘要：
- [chosen-only context adapter training summary](/root/autodl-tmp/AutoPlanner/results/shared/chem_enzy_adapter_mainline_20260521/chosen_v4_1477_context_adapter_training_summary.md)
- [100-target comparison](/root/autodl-tmp/AutoPlanner/results/shared/chem_enzy_adapter_mainline_20260521/context_preference_scorer_route_ab_limit100/comparison.md)

### 3.4 100-target route-level 对照

比较结果说明 preference scorer 不是当前主解：

- `top_result_exact` 仅从 `0.04` 到 `0.05`
- `top_result_gt_reactant` 仍是 `0.17`
- `candidate_exact_reaction_in_pool = 0.13`
- `candidate_gt_reactant_in_pool = 0.37`
- 平均时间从 `0.0023s` 升到 `0.1661s`

结论：**改路了，但没明显提质，而且变慢很多**。

---

## 4. 目前问题

### 4.1 第一问题：候选覆盖不足

真实候选没有稳定进入池。
这意味着 rerank、verifier、DPO 都无法补救“根本没生成到”的情况。

### 4.2 第二问题：候选保留不足

即使正确或半正确候选出现，也可能被：

- top-rank 截断
- branch factor 剪掉
- source bias 压掉
- 短路线大跳路线挤掉

### 4.3 第三问题：搜索和化学语义没完全对齐

当前系统能做审计，但没有把以下信息稳定变成主搜索约束：

- 条件冲突
- stage 兼容性
- cofactor 闭环
- 短路线伪影
- carrier / protecting-group 语义

### 4.4 第四问题：训练目标还不够贴近任务

目前的 proposal 训练仍偏 product→reactants 翻译式 SFT。
它能提升局部 exact recall，但还不足以保证：

- 更好的 route-level 覆盖
- 更高的正确候选保留率
- 更合理的长路线生成

### 4.5 第五问题：评估归因不够细

现在还不够回答一个失败样本到底属于：

- candidate missing
- present but pruned
- present but ranked low
- selected but chemically weak

没有这个归因，训练和搜索就会继续空转。

---

## 5. 委员会判断

### 5.1 总工程师

主线应收敛为：

**ChemEnzy native search + 审计 + verifier + state-aware search**

下一阶段的工程优先级：
1. route-level 失败归因面板
2. candidate retention search
3. proposal 训练目标重构

### 5.2 化学专家

最大问题不是“大 terminal”，而是：

- 短路线伪影
- 大结构跳跃
- 条件与反应类型脱节
- dummy atom / 泛化片段污染

需要保守的规则：

- 保护基、载体、转移试剂、cofactor 不可按 heavy-atom 硬杀
- 只有“凭空造复杂骨架且无条件解释”的情况才强拒绝

### 5.3 计算机专家

最优技术路径不是继续训新 scorer，而是：

**route-level candidate ledger + retention search**

要求每个候选记录：

- provider
- raw rank
- canonical reactants
- validity
- preference score
- verifier annotation
- 是否被 dedupe
- 是否被扩展
- 是否进入最终 route

---

## 6. 冻结边界

以下方向不再作为主线投入：

- CCTS v0/v1/v2/v3
- route-pool LambdaRank / old ranker
- adjacent-step pair scorer
- block-coherence / block-hard pack
- v4 product-value / action-source / provider-retrieval lineage
- expert CSV / LLM review fallback
- learned verifier 默认 rerank
- preference scorer 直接 promotion

这些方向可以保留为诊断和复现资产，但不应继续占用主线算力。

---

## 7. 建议下一步

### 7.1 先做失败归因面板

统一 benchmark 输出，至少拆成四类：

- `candidate_missing`
- `present_but_pruned`
- `present_but_ranked_low`
- `selected_but_chemically_weak`

### 7.2 再做 candidate retention search

保留策略应覆盖：

- native ChemEnzy 候选
- context adapter 候选
- known-good / legal corpus 候选
- verifier-feasible 候选
- 长路线 / 短路线分桶候选

### 7.3 然后重做 proposal 训练目标

优先方向：

- reactant-set completion
- route-targeted positives
- legality-aware decoding / filtering

### 7.4 固定实验矩阵

每轮只跑四组：

- baseline
- proposal-only
- search-only
- proposal+search

每组先 `limit20` 快筛，再 `limit100` 定论。

---

## 8. 参考结果

- [Mainline cleanup](/root/autodl-tmp/AutoPlanner/docs/MAINLINE_CLEANUP_2026-05-20.md)
- [Current state](/root/autodl-tmp/AutoPlanner/docs/CURRENT_STATE_2026-05-19.md)
- [Codebase status](/root/autodl-tmp/AutoPlanner/docs/CODEBASE_STATUS_2026-05-19.md)
- [Chosen-only context adapter training summary](/root/autodl-tmp/AutoPlanner/results/shared/chem_enzy_adapter_mainline_20260521/chosen_v4_1477_context_adapter_training_summary.md)
- [100-target comparison](/root/autodl-tmp/AutoPlanner/results/shared/chem_enzy_adapter_mainline_20260521/context_preference_scorer_route_ab_limit100/comparison.md)
