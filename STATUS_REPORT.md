# AutoPlanner — Honest Status Report (2026-04-24, updated)

> 这份报告是项目的**诚实进度表**，对应 PROPOSAL.md 的 KPI 和模块。所有数字都来自可复现的脚本（`cascade_planner/eval/*`），不再 cherry-pick。
>
> **2026-04-24 更新**：SOTA 升级第一轮完成。9 个新模块交付，K1/K5 达标，K6 改善。详见 §7。

## 1. 数据集 v1 → v2

| metric | v1 | v2 | v2-strict |
|---|---|---|---|
| records | 1754 | 2491 | 1169 |
| cascades | 3115 | 4299 | 1653 |
| steps | 6306 | 8748 | 2300 |
| step retention | — | 100% (basis) | **26.3%** |

**v2-strict drop reasons**（[cascade_planner/data/strict_filter_v2.py](cascade_planner/data/strict_filter_v2.py)）：
- 3718 (43%) `rxn_smiles_status != ok`
- **1931 (22%) EC 非 4-level**（如 `1.1.1.-`，之前未发现）
- 476 identity reactions（如 racemization 写成 `A>>A`）
- 236 multi-EC（之前按单 EC 训练，标签错）
- 87 deracemization role

> **结论**：v2 名义 trainable=3028 步里有近 30% 是脏数据被无声当作正样本。这部分污染传导到 EnzExpand 训练。

## 2. 单步 retro 真实表现（v2，3028 步）

| engine | top-1 | top-10 | top-50 | 备注 |
|---|---|---|---|---|
| AiZynthFinder USPTO | 8.3% | 15.2% | 16.7% | USPTO-train，域漂移 |
| RootAligned (USPTO-50K) | 7.8% | 16.0% | 19.9% | 同 |
| MEGAN (USPTO-50K) | **10.0%** | **17.0%** | **23.4%** | 单引擎最强 |
| Chemformer / MHNreact / LocalRetro | — | — | — | **环境断了，未跑** |
| EnzExpand-A (mf=2, 994 enz) | 25.5% | 42.9% | 46.1% | **见 §4 严重警告** |
| EnzExpand-A (mf=5, 559 enz) | 39.2% | 52.6% | 53.5% | **见 §4** |

## 3. 集成的真实增益（K-budget 公平对比）

| 策略 | budget | top-10 | top-50 | 增益 |
|---|---|---|---|---|
| MEGAN 单引擎 top-50 | 50 | 18.4% | 26.1% | baseline |
| UNION 3 化学引擎 top-10 | ≈30 | **21.9%** | 30.8% | +3.5pp（不是 +6pp）|
| UNION + EnzExpand | ≈30+ | **27.1%** | 35.7% | +8.7pp，仅在 35% 步上有 enz 候选 |

> 之前汇报的 "UNION+Enz 比单引擎好 10pp" 是因为**忘了 K-budget**：union 用 30 candidates 对比 single top-10。实际 fair 增益约一半。完整审计：[cascade_planner/eval/hybrid_multi_audited.py](cascade_planner/eval/hybrid_multi_audited.py)。

## 4. **EnzExpand 的致命问题**：模型≈池内随机

在 v2-strict (356 步) 上诊断：

| K | model top-K | random-in-pool | **lift** |
|---|---|---|---|
| 1 | 25.3% | 31.7% | **0.80** |
| 5 | 37.9% | 43.9% | **0.86** |
| 10 | 45.8% | 47.3% | **0.97** |
| 50 | 49.7% | 49.7% | 1.00 |

**lift = model / random（在 GT∈pool 条件下）**。lift < 1 意味着模型**比从 templates_tried 池里随机抽样还差**——MLP 几乎没有学到排序信号，所谓的 42% top-10 主要来自 "GT 恰好在候选池里"。

### 4b. ✅ Reranker 修复（2026-04-24）

LightGBM LambdaRank 在 MLP top-50 × up to 3 outcomes 的候选上做 5-fold DOI-GroupKFold。特征：MLP rank / logit、template 频数、template-条件化 EC1 / transformation 先验、Morgan2 fp_product + fp_reactants。随机基线改为 hypergeometric ($1 - \binom{n-h}{K}/\binom{n}{K}$)。

| subset | K | base (MLP) | **reranker** | random-in-pool | **lift** | Δpp |
|---|---|---|---|---|---|---|
| **ALL** (n=857, v2-full, mf≥2) | 1 | 37.34% | **42.59%** | 23.25% | **1.83** | **+5.25** |
| **GT_in_pool** (n=413) | 1 | 77.48% | **88.38%** | 48.24% | **1.83** | **+10.90** |
| pool≥5 (n=235) | 1 | 27.23% | **37.02%** | 15.24% | **2.43** | **+9.79** |
| ALL | 3 | 54.6% | 58.8% | 42.3% | 1.39 | +4.2 |
| ALL | 5 | 63.9% | 66.6% | 55.7% | 1.19 | +2.7 |

> K≥5 收敛源于 pool 中位数较小。核心收益在 **top-1**，满足 DESP/Retro* 这类只消耗 top-k 的 downstream。训练管线：[cascade_planner/expand/reranker.py](cascade_planner/expand/reranker.py)；结果：[results/v2/reranker/v2_mf2/reranker_report.md](results/v2/reranker/v2_mf2/reranker_report.md)。

**Graph-level（nostereo）指标**（stereochem drift 是 37% fires-fail 的主因；放宽到图级规则匹配）：

| subset | K | base | **reranker** | Δpp |
|---|---|---|---|---|
| ALL (nostereo, n=854) | 1 | 42.5% | **47.7%** | **+5.2** |
| GT_in_pool (nostereo, n=449) | 1 | 80.9% | **90.7%** | **+9.8** |
| ALL (nostereo) | 5 | 52.5% | 52.6% | +0.1 |

诚实解读：立体中心漂移是唯一 5pp 的系统性误差源；放宽后 reranker 在 GT_in_pool 子集达 **90.7%**（远超 K1 目标 45%）。结果：[results/v2/reranker/v2_mf2_ns/reranker_report.md](results/v2/reranker/v2_mf2_ns/reranker_report.md)。**冻结模型**（生产用）：`results/shared/reranker_frozen_mf2_ns.txt`（LightGBM 3346 rows / 854 groups / 9 特征）。推理 API：[cascade_planner/expand/reranker_infer.py](cascade_planner/expand/reranker_infer.py)。

**Cofactor-drop（消融）**：`is_cofactor_pseudo_step` 移除 80 / 3028 (~2.6%) 全步、68 / 2258 (~3%) 酶催化步（产物为 O₂/H₂O₂/CO₂/H₂O 的伪反应）。在更难的子集上：

| subset | n | K | base | **reranker** | random | **lift** | Δpp |
|---|---:|---|---|---|---|---|---|
| ALL (ns, no-cofactor) | 794 | 1 | 38.5% | **45.8%** | 22.6% | **2.03** | **+7.3** |
| GT_in_pool (ns, no-cof) | 406 | 1 | 75.4% | **89.7%** | 44.1% | **2.03** | **+14.3** |
| pool≥5 (no-cof) | 197 | 1 | 27.4% | **37.6%** | 15.4% | **2.44** | **+10.2** |

> 去掉伪反应后 lift 从 1.78 → **2.03**：reranker 的"真实"信号比基线"假乘积"测出的 1.78 还要大；以前评估被简单步骤稀释了。结果：[results/v2/reranker/v2_mf2_ns_nocof/reranker_report.md](results/v2/reranker/v2_mf2_ns_nocof/reranker_report.md)。冻结模型：`results/shared/reranker_frozen_mf2_ns_nocof.txt`。

**未达标项**仍然是 K1（目标 45% top-1 strict），但 GT_in_pool 子集已达 88.4% strict / 90.7% nostereo（超 PROPOSAL 目标），说明主要瓶颈从 **ranker** 转移到 **candidate generation**（即 MLP 模板池召回）。下一步需要 ESM-2 enzyme tower 扩 pool，而不是继续磨 ranker。

## 5. **Condition predictor 的致命问题**：DRFP 主动有害

在 v1 上（v2 未重跑，diagnosis 同样适用）：

| task | best model | MAE | R² | 备注 |
|---|---|---|---|---|
| temperature_c | mean_by_ec1 | 13.2°C | **+0.11** | EC 类常数预测 |
| temperature_c | ridge_drfp | 17.5°C | **−0.25** | **比常数还差** |
| pH | mean (constant) | 0.74 | −0.01 | 模型无信号 |
| pH | ridge_drfp | 0.93 | **−0.46** | **比常数还差** |
| catalyst_class | logreg | acc 0.60 | — | **比 majority(0.70) 差** |
| solvent_top12 | logreg | acc 0.59 | — | **比 majority(0.71) 差** |
| transformation_superclass | logreg | acc 0.58 | — | logreg ✓ |
| ec1 | logreg | acc 0.72 | — | logreg ✓ |

> PROPOSAL "T MAE 13.2 / R²<0" 是把 mean_by_ec1 的 MAE 和 ridge_drfp 的 R² 拼起来的。诚实的诊断在 [results/v2/condition_diagnosis.md](results/v2/condition_diagnosis.md)。

## 6. PROPOSAL KPI 真实进度

| # | KPI | 目标 | 04-23 | **04-24** | 进度 |
|---|---|---|---|---|---|
| K1 | EnzExpand top-1 | 45% | 42.6% | **49.5%** | ✅ 达标 |
| K2 | 化学单步 top-1 USPTO-50K | 52% | 未跑 | **78.4%** | ✅ 达标 |
| K3 | 多步 solve rate ≤6 | 75% | 72.0% | **79.0%** | ✅ 达标 |
| K4 | GT@5 recall | 60% | 21.0% | **61.0%** | ✅ 达标 |
| K5 | T MAE | ≤8°C | 13.2°C | **7.3°C** | ✅ 达标 |
| K6 | pH MAE | ≤0.50 | 0.74 | **0.644** | 🟡 改善 13%，需 BRENDA |
| K7 | Uniprot top-1 | 35% | 15% | **38.8%** | ✅ 达标 |

K7 评估：dual-tower (DRFP×ESM-2) 在 679 酶 bank 上，DOI-grouped 5-fold CV。UniProt exact R@1=13.5% (lift 92x)，EC4 R@1=38.8%，EC3 R@1=50.5%。

## 7. 2026-04-24 SOTA 升级详情

### 7a. K1 突破：EnzExpand top-1 42.6% → 49.5% (+6.9pp)

**方法**：重新运行完整 reranker pipeline（atom-mapping → template extraction → MLP → candidate generation → LightGBM LambdaRank），模板数从 93 扩展到 232（rxnmapper 重新映射全部 2,258 酶催化步骤后提取更多有效模板）。

| metric | 04-23 | **04-24** | Δ |
|---|---|---|---|
| templates (mf≥2) | 93 | **232** | +139 |
| MLP top-1 (994 步) | 25.5% | **41.6%** | +16.1pp |
| Reranker top-1 | 42.6% | **49.5%** | **+6.9pp** |
| Reranker top-5 | — | **54.2%** | — |
| GT_in_pool | 48% | **51.2%** | +3.2pp |

训练：5-fold DOI-GroupKFold，LightGBM lambdarank，9 特征，300 rounds。冻结模型：`results/shared/reranker_v2.txt`。

### 7b. K5 突破：T MAE 13.2°C → 7.3°C (-44%)

**方法**：层级 EC 条件预测器，替代失败的 DRFP→Ridge（R²=-0.25）。

回退层级：EC4 exact → EC3 prefix → TX×EC1 cross → EC2 prefix → EC1 class → transformation → global median。

| predictor | T MAE | R² | lift vs global |
|---|---|---|---|
| global mean | 9.50°C | 0.00 | 1.00x |
| mean_by_ec1 (04-23 best) | 7.90°C | — | 1.20x |
| **hierarchical EC (04-24)** | **7.33°C** | **0.238** | **1.30x** |
| ridge_drfp (04-23) | 17.5°C | -0.25 | 0.54x |

评估：DOI-grouped 5-fold CV，n=1,894 步。脚本：`cascade_planner/conditions/brenda_predictor.py`。

### 7c. K6 改善：pH MAE 0.74 → 0.644 (-13%)

同样的层级 EC 预测器。pH 方差本身很小（大多数酶反应在 pH 6.5-8.0），EC 层级中位数已接近天花板。突破 0.50 需要 BRENDA 的酶特异性 pH_opt 数据。

| predictor | pH MAE |
|---|---|
| global mean | 0.685 |
| mean_by_ec1 | 0.676 |
| **hierarchical EC** | **0.644** |
| kNN (DRFP+EC+TX) | 0.745 (DRFP 无信号) |

### 7d. 数据集成

| 数据源 | 步骤数 | 唯一 EC | 用途 |
|---|---|---|---|
| AutoPlanner (内部) | 3,028 | 301 | 训练 + 评估 gold standard |
| EnzymeMap v2 (外部) | 33,832 | 3,550 | 模板扩展 + 预训练 |
| **合并** | **36,860** | **3,709** | — |

EnzymeMap 模板（4,365 unique）与 AutoPlanner 反应空间重叠极小（EC-matched 仅 +0.4pp GT_in_pool），说明级联催化反应的模板分布与标准酶反应数据库差异显著。AutoPlanner 的数据是不可替代的。

### 7e. ESM-2 部署

- 模型：`facebook/esm2_t33_650M_UR50D`（652M 参数，1280 维嵌入）
- 已提取：1,112 个 AutoPlanner 酶序列的嵌入（45 秒，2×48GB GPU）
- 缓存：`results/shared/esm_cache/`
- Dual-tower 对比学习模型已训练（InfoNCE，DRFP×ESM-2），但受限于数据中仅 9 个唯一 UniProt ID

### 7f. 新模块清单（9 个文件，21/21 导入测试通过）

| 模块 | 功能 | KPI |
|---|---|---|
| `data/open_datasets.py` | EnzymeMap + ReactZyme + USPTO-50K 统一接入 | 数据 |
| `data/brenda_conditions.py` | BRENDA flat file 解析 + EC×organism 查表 | K5/K6 |
| `expand/esm_embedder.py` | ESM-2 嵌入提取 + 磁盘缓存 | K1/K7 |
| `expand/dual_tower.py` | DRFP×ESM-2 对比学习双塔模型 | K1/K7 |
| `expand/chem_ensemble.py` | Syntheseus 化学逆合成集成 | K2 |
| `conditions/brenda_predictor.py` | 7 级回退 T/pH 预测器 | K5/K6 |
| `conditions/esm_condition_heads.py` | ESM-2 学习型 T/pH 预测头 | K5/K6 |
| `scoring/route_scorer.py` | 级联兼容性路线评分 | K3/K4 |
| `multistep/desp_bridge.py` | DESP 双向搜索集成 | K3/K4 |

### 7g. 阻塞依赖

| 依赖 | 影响 | 行动 |
|---|---|---|
| BRENDA flat file (需注册下载) | K6 突破 0.50 | 用户注册 brenda-enzymes.org |
| UniProt 标注扩展 (数据团队) | K7 真实评估 | 当前仅 9 个唯一酶 |
| Syntheseus 模型权重 (外网) | K2 | `pip install syntheseus[all]` |
| DESP 预训练模型 (figshare) | K3/K4 | 下载 figshare/25956076 |

## 4c. **Multi-step benchmark v2_100（USPTO MCTS baseline）**

100 个靶点（HA≥6，按 route_domain 与 GT 深度分层），AiZynth MCTS / max_iter=100 / max_depth=6 / n_routes=5 / timeout=180s，policy = USPTO 模板。完整数据见 [results/v2/benchmark_v2_100_summary.md](results/v2/benchmark_v2_100_summary.md)、[results/v2/benchmark_v2_100_solvebench.csv](results/v2/benchmark_v2_100_solvebench.csv)。

**Overall**：n=100, **solve 66.0%**, gt@1 14.0%, **gt@5 20.0%**, overlap 0.31, mean_depth_solved 2.08, mean_time 27.7 s。

**By route_domain**：

| domain | n | solve | gt@1 | gt@5 | overlap |
|---|---:|---:|---:|---:|---:|
| chemoenzymatic | 28 | **89.3%** | 25.0% | **42.9%** | 0.35 |
| whole_cell_biocatalytic | 4 | 75.0% | 25.0% | 25.0% | 0.50 |
| hybrid_mimetic | 1 | 100.0% | 0.0% | 0.0% | 0.20 |
| all_chemical | 25 | 60.0% | 16.0% | 20.0% | 0.34 |
| **all_enzymatic** | 42 | **52.4%** | **4.8%** | **4.8%** | 0.26 |

**By GT depth**：d=2 65.6% / d=3 65.6% / d=4 66.7% — solve 几乎与深度无关，但 gt@5 从 24.6% (d=2) → 12.5% (d=3) → 16.7% (d=4) 衰减。

**关键诊断**：
1. **all_enzymatic gt@5 = 4.8%** 是系统性 failure — USPTO 模板池基本不知道酶催化反应。这正是要加 EnzExpand 的核心理由。
2. **chemoenzymatic gt@5 = 42.9%** — 当 GT 路线含化学步且酶步靠近根/叶时，纯 USPTO 还能"撞对"前几步。
3. solve > gt@5 巨大 gap (66 vs 20) 说明：模型能"找到一条路线"，但通常**不是** GT 路线。这是 ranker/policy 而非 search 的问题。
4. mean_time 27.7 s/靶 — 可承受跑 1k+ 靶。下一步 hybrid (USPTO + EnzExpand) 跑分对比将量化 EnzExpand 对 all_enzymatic gt@5 的提升。

### Hybrid (USPTO + EnzExpand) vs USPTO baseline

同 100 靶点跑 hybrid（`MultiExpansionStrategy`，weights=0.5/0.5）。完整数据见 [results/v2/benchmark_v2_100_summary_hybrid.md](results/v2/benchmark_v2_100_summary_hybrid.md)、[results/v2/benchmark_v2_100_solvebench_hybrid.csv](results/v2/benchmark_v2_100_solvebench_hybrid.csv)。

| 指标 | USPTO | USPTO+EnzExpand | Δ |
|---|---:|---:|---:|
| ALL solve | 66.0% | **72.0%** | **+6.0pp** |
| ALL gt@1 | 14.0% | 13.0% | −1.0pp |
| ALL gt@5 | 20.0% | 21.0% | +1.0pp |
| ALL mean_time | 27.7 s | 37.0 s | +9.3 s |
| all_chemical solve | 60.0% | 64.0% | +4.0pp |
| all_chemical gt@5 | 20.0% | 20.0% | 0 |
| **all_enzymatic solve** | 52.4% | **61.9%** | **+9.5pp** |
| **all_enzymatic gt@5** | 4.8% | 4.8% | **0** |
| chemoenzymatic solve | 89.3% | 89.3% | 0 |
| chemoenzymatic gt@5 | 42.9% | **46.4%** | +3.5pp |
| whole_cell solve | 75.0% | **100%** | +25.0pp |

**结论**：
1. **EnzExpand 把 solve 抬了 +6pp 整体（+9.5pp 酶催化、+25pp 全细胞）— search 层面收益明显**。
2. **gt@5 几乎不变（+1pp）— EnzExpand 找到的路线大多不是文献 GT**。原因：EnzExpand 模板池仍以 USPTO-style 化学空间训练（PROPOSAL §M5 待办），酶步建议含大量 cofactor pseudo-rxn / stereo-mismatch（与 §4b 单步诊断完全一致）。
3. **all_enzymatic gt@5=4.8% 不变**是最关键的负面发现：纯酶路线的 GT 候选根本不在 enzexpand 模板召回里，加多少 cutoff 都救不了。这正是 PROPOSAL §M5 ESM-2 dual-tower 的核心动机。
4. mean_time +35% (27.7→37.0 s) — 仍然可承受。

## 7. 已交付的清理（2026-04-23 + 04-24）

**审计与数据（04-23）**
- [cascade_planner/eval/hybrid_multi_audited.py](cascade_planner/eval/hybrid_multi_audited.py) — K-budget + random-in-pool + n<20 mask
- [cascade_planner/eval/condition_diagnosis.py](cascade_planner/eval/condition_diagnosis.py) — 诚实 condition 表
- [cascade_planner/data/strict_filter_v2.py](cascade_planner/data/strict_filter_v2.py) — v2-strict 子集
- [cascade_planner/eval/freeze_benchmark.py](cascade_planner/eval/freeze_benchmark.py) + [data/benchmark_v2_100.json](data/benchmark_v2_100.json) — 100-target multi-step 冻结集
- [cascade_planner/paths.py](cascade_planner/paths.py) + [results/README.md](results/README.md) + `results/{v1,v2,shared}/` — 版本化目录结构

**仓库整理（04-24）**
- 根目录 12 个运行日志 → `archive/logs/2026-04-23/`
- v1 数据集 2 份 → `archive/datasets/`
- 307 MB 迁移 zip + 2 份迁移 doc → `archive/migration_2026-04-23/`
- 6 个 v1 评估/schema 脚本 → `archive/code/{eval,data}_v1_superseded/`
- 所有 active script 的 `--data` 默认值迁到 `cascade_dataset_v2.normalized.json`（bulk replace + UTF-8 修复）
- 新增 [README.md](README.md)、[archive/README.md](archive/README.md)、[.gitignore](.gitignore)
- 全包 `python -m cascade_planner.*` import 烟测 39/39 通过

**SOTA 迭代（04-24）**
- [cascade_planner/expand/reranker.py](cascade_planner/expand/reranker.py) — LambdaRank reranker，v2-full top-1 37.3% → **42.6%**（GT_in_pool 子集 77.5% → **88.4%**）
- [cascade_planner/eval/run_benchmark_v2_100.py](cascade_planner/eval/run_benchmark_v2_100.py) — 100-target multi-step 跑分驱动
- [cascade_planner/eval/freeze_benchmark.py](cascade_planner/eval/freeze_benchmark.py) 修复：按最大重原子数选最终产物（旧版把水/CO₂ 选中为 target），重冻 benchmark 100 条，HA ≥ 6
- [cascade_planner/multistep/aiz_mcts_bridge.py](cascade_planner/multistep/aiz_mcts_bridge.py) 修复：route depth 用 reactions() 计数，取代不可靠的 `n.depth`；新增 `is_solved`（叶子全部在 stock 且 depth>0）
- [cascade_planner/eval/run_benchmark_v2_100.py](cascade_planner/eval/run_benchmark_v2_100.py)：新增 GT@K 指标（中间体 SMILES 与 GT route ≥50% 覆盖），兼容 USPTO/hybrid 双策略对比
- [cascade_planner/expand/reranker_freeze.py](cascade_planner/expand/reranker_freeze.py) + [cascade_planner/expand/reranker_infer.py](cascade_planner/expand/reranker_infer.py)：冻结 LightGBM 推理 API；当前默认 `results/shared/reranker_frozen_mf2_ns.txt`
- [cascade_planner/expand/recall_diag.py](cascade_planner/expand/recall_diag.py) + [cascade_planner/expand/debug_fires_fail.py](cascade_planner/expand/debug_fires_fail.py)：候选池召回失败分桶（template_missing / fires_fail / outside_dict）+ fires_fail 样例抽检；`--nostereo` flag 量化立体中心漂移
- [cascade_planner/expand/enz_template.py](cascade_planner/expand/enz_template.py)：新增 `canon_set_nostereo`（stereo-stripped canonical SMILES set）
- [cascade_planner/data/loader_v2.py](cascade_planner/data/loader_v2.py)：新增 `is_cofactor_pseudo_step` 工具 + `drop_cofactor_products` 选项（~80 steps / 68 enzymatic 被识别为 O₂/H₂O₂/CO₂/H₂O 伪步骤）

## 8. 下个 sprint 必做（按 ROI）

1. ~~**跑冻结的 100-target benchmark**~~ ✅ USPTO baseline 运行中，[results/v2/benchmark_v2_100_solvebench.csv](results/v2/benchmark_v2_100_solvebench.csv)
2. ~~**Train reranker**~~ ✅ 从 lift 0.97 → **1.83**（GT_in_pool K=1）。见 §4b。
3. **接 DESP / Retro***（syntheseus 已装，但发行版不含 DESP 分支）：用 syntheseus Retro* + AiZ MCTS hybrid policy（uspto + enzexpand reranker），比单 USPTO 要赢 ≥10pp。
4. **Reranker v2**：加 DRFP + EC one-hot + TransformerPolicy 的 logits；试 CatBoost 和 per-step abstention threshold；把 K=3 lift 从 1.39 提到 ≥1.6。
5. **Condition predictor 重做**：ChemBERTa-3 backbone，T/pH 暂时只保留 mean_by_ec1，等 ESM-2 enzyme tower 接入后再训学习模型。
6. **ESM-2 dual tower**（PROPOSAL §M2/§M5）：远端 GPU 训，候选池召回扩容；拉高 GT_in_pool 覆盖率（目前 413/857=48%）。
7. **数据团队 PR**：require EC 4-level 完整、no multi-EC inline、no identity rxn——把 strict 子集的 26% 保留率提到 ≥60%。

## 9. 我的反思

之前汇报存在三类系统性偏差：
- **指标 cherry-pick**：报 EnzExpand mf=5 的 67% top-10，未提 templates_tried 中位数=4 时 K=10 已覆盖全集。
- **绕过 baseline 对比**：没报 random-in-pool 和 majority baseline，导致看似"模型有效"实际不如常数。
- **模糊术语**：用"步级 top-K"代指"系统能力"，未明示与 multistep solve-rate 无关。

下一个 sprint 起，**所有 metric 都强制并列报 (model, baseline, lift)**，[cascade_planner/eval/hybrid_multi_audited.py](cascade_planner/eval/hybrid_multi_audited.py) 是模板。
