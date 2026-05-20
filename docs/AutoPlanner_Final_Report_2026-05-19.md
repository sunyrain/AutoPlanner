# AutoPlanner-Cascade 最终汇报文稿

日期：2026-05-19
版本：v1（合并 6 份现役文档）
用途：面向化学信息学、合成化学、酶催化、自动规划方向专家的 PPT 汇报底稿；
亦作为下一阶段研究主线的唯一权威说明。

合并自：
- `docs/AutoPlanner_Cascade_阶段性汇报_2026-05-19.md`
- `docs/CASCADE_VERIFIER_PROOF_REPORT_2026-05-19.md`
- `docs/CURRENT_STATE_2026-05-19.md`
- `docs/CASCADE_TARGET_ARCHITECTURE.md`
- `docs/GUARDED_CCTS_DECISION_2026-05-19.md`
- `docs/CODEBASE_STATUS_2026-05-19.md`

> **📑 PPT 全脚本**：完整 39 张正片 + 8 张 Backup 的逐页文案与图描述见 [docs/AutoPlanner_PPT_Full_Script_2026-05-19.md](AutoPlanner_PPT_Full_Script_2026-05-19.md)（覆盖关键问题 / 设计阶段 / 当前架构 / 国际 SOTA / 未来主线）。

---

## 目录

0. [一页执行摘要](#一页执行摘要)
1. [问题定义](#1-问题定义)
2. [我们经历的六个阶段](#2-我们经历的六个阶段)
3. [当前可用能力（demo 主线）](#3-当前可用能力demo-主线)
4. [当前模型证据：什么有信号、什么被冻结](#4-当前模型证据什么有信号什么被冻结)
5. [新方向 proof：Cascade Verifier-first 飞轮](#5-新方向-proofcascade-verifier-first-飞轮)
6. [目标架构（cascade-stateful planner）](#6-目标架构cascade-stateful-planner)
7. [国际前沿对比 + 为何 cascade 仍难](#7-国际前沿对比--为何-cascade-仍难)
8. [仓库主线 vs 研究残骸 vs 已归档](#8-仓库主线-vs-研究残骸-vs-已归档)
9. [下一阶段路线图（30 / 90 天）](#9-下一阶段路线图30--90-天)
10. [建议的 12-slide PPT 结构](#10-建议的-12-slide-ppt-结构)
11. [可以说 / 不能说 边界](#11-可以说--不能说-边界)
12. [参考文献](#12-参考文献)

---

## 一页执行摘要

我们这一轮工作的核心**不是**简单证明"模型能不能生成路线"，而是把问题推进到下一层：

> 对复杂 statin-like 目标分子，系统能生成大量长 cascade 路线候选；
> 真正的问题是如何判断这些长路线是否具有化学意义、
> 哪些只是搜索伪影、哪些值得专家继续看、
> 以及——在 3K 量级真实 cascade 数据下，如何**继续教模型变得更好**。

### 关键数字一览（statin 目标，target hash 3764f7）

| 项目 | 数值 |
| --- | ---: |
| ChemEnzy 原始路线 | 906 |
| 原始唯一路线签名 | 393 |
| Post-filter 保留路线 | 640 |
| 当前规则重新审计 `triage_fragment` | 360 |
| `needs_chemist_review` | 232 |
| `reject_artifact` | 48 |
| Condition audit `high` 风险 | 193 |
| Condition audit `warn` | 447 |

### 三个关键认知修正（本轮新增）

1. **大 terminal ≠ 高级产物片段**——Wittig phosphorane、HWE 试剂这种 25+ heavy-atom 分子，其中大部分原子只是 carrier / leaving group。第一版审计因此误伤了 route 5 类路线。
2. **反应物 SMILES 中没有的元素 ≠ 凭空出现**——`POCl3` / 卤化剂 / 氧化剂 / 还原剂 / 酸碱常被放在 condition/reagent 字段，不在 reaction SMILES 的 reactants 侧。
3. **条件预测 ≠ 已验证工艺条件**——逐步条件预测可作为"该步可能需要 LDA、低温、Pd 偶联"的提示，**不能直接作为一锅 cascade 条件方案**。

### 本轮新方向（已跑通 proof，明天展望主线）

**Cascade Verifier-first 飞轮**——把"3K 真实 cascade 正样本太少所以学不动"的死结，用**无限规则负样本 → verifier → DPO preference → ChemEnzy 微调**绕开：

| 指标 | 数值 |
| --- | ---: |
| Perturbation 样本（30K structured v4） | 30,556 |
| 规则 verifier label accuracy | **0.9964** |
| 规则 verifier expected-reason coverage | 0.9962 |
| Learned verifier feasibility test acc | 0.9094 |
| Learned verifier reason macro F1 | **0.9653** |
| 真实路线池 592 条 label accuracy | 1.0000 |
| DPO preference pairs 就绪 | **29,079** |

### 本轮 freeze 的方向（写进 ADR）

| 方向 | 结论 |
| --- | --- |
| Adjacent-step CCTS pair reward | 仅留作 search-time safe diagnostic / tie-break，不再当主模型贡献 |
| Block-coherence classifier | AUC 0.94 但 route-level analog@1 仅 0.06–0.08，不 promote |
| Runtime hard-negative learned blend | MRR 0.423 ≈ retrieval-only 0.422，不 promote |
| Learned route/block value scorer | live search top-GT 反而 0.30→0.25，不 promote |
| Expert CSV expansion 路线 | 维持为 fallback，不阻塞主线 |

---

## 1. 问题定义

### 1.1 表层问题

> 给定一个 statin-like 复杂目标分子，能不能生成一条**化学合理 + 酶催化兼容 + 条件可执行**的 cascade 路线？

### 1.2 真问题（深一层）

`stock closure` 不等于成功。一条路线可能因以下原因被"假闭合"：

- 把高级中间体当成 stock；
- 把 carrier reagent 误当成产品片段；
- reaction SMILES 省略了关键 reagent；
- 单步模型凭空引入元素/官能团；
- route search 用低可信步骤强行闭合到 stock。

因此真问题是：**长 chemoenzymatic cascade 路线的可信度评估、可解释性、可干预性**。

### 1.3 数据约束（决定性的）

- 真实 cascade 正样本 ≈ **3K 条**（`cascade_dataset_v3.json` 3,810 cascades / 8,753 steps，`cascade_v4_high_quality.jsonl` 量级类似）；
- 没有专家标签，**未来也不会有**专家标签（用户明确表态）；
- 任何"训一个 cascade-level scorer"的方案都被这 3K 卡住。

### 1.4 由此推出的研究纪律

- 不再追求 expert-labeled good/bad route；
- 改用**弱监督 + 自监督 + 规则可造负样本**；
- 必须区分 raw dataset size、traced target count、可用 supervised pair count，不能用"原始反应数"掩盖正样本稀缺。

---

## 2. 我们经历的六个阶段

### 阶段 1：从 `no route` 误解定位到路线池问题

最初的报告是 `no route`。检查 artifact 后发现路线存在：raw 906 / unique 393 / kept 640 / rejected 266。问题不是搜不到，而是**生成—过滤—展示—评价之间语义没对齐**。

目标分子：

```text
CC(C)C1=NC(=NC(=C1/C=C/[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)N(C)S(=O)(=O)C
```

### 阶段 2：建立路线可用性分层

| 类别 | 含义 |
| --- | --- |
| `triage_fragment` | 有可讨论的片段连接、芳基偶联、侧链构建思路 |
| `needs_chemist_review` | 可能有参考价值，但证据不足或风险较高 |
| `reject_artifact` | 有明显原子来源、结构闭合或模型伪影问题 |
| autonomous candidate | 更严格的可执行候选，**目前没有路线达到这个级别** |

第一版审计帮助筛掉明显 artifact，但也暴露问题：它把"分子很大"过度等同于"高级中间体"。

### 阶段 3：Route 5 案例推动审计逻辑修正

640 条保留路线中的 route 5 含一个 25-heavy-atom 大 terminal：

```text
CCOC(=O)C=P(c1ccccc1)(c1ccccc1)c1ccccc1
```

旧规则标成 `advanced_or_product_like_terminal` + `large_polycyclic_terminal` + `product_like_terminal`。

化学上不严谨：这是 ethyl 2-(triphenylphosphoranylidene)acetate，稳定 Wittig 试剂，三个苯基是载体不是骨架。

同条路线 `Cl` 在产物中出现但不在显式 reactants 中。检查 condition 发现：

```text
Reagent: O=P(Cl)(Cl)Cl   # POCl3
```

→ 不是凭空出现，是 condition reagent 提供。

### 阶段 4：修正 product audit + route plausibility

**Terminal profile 改造**（不再只看 raw heavy atom count）：

```text
max_terminal_heavy_atoms
effective_max_terminal_heavy_atoms
carrier_reagents
product_like_terminal
large_polycyclic_terminal
```

**Route plausibility 改造**（不再只看 reaction SMILES 的 reactants 侧）：

```text
raw_element_gains
condition_supported_element_gains
unexplained_element_gains
unexplained_new_elements
```

只有在 explicit reactants 和 condition reagent 都无法解释时，才标 unsupported element source。

### 阶段 5：重新审计 640 条路线

| 当前审计类别 | 数量 |
| --- | ---: |
| `triage_fragment` | 360 |
| `needs_chemist_review` | 232 |
| `reject_artifact` | 48 |

新增 condition audit：

| 条件审计类别 | 数量 |
| --- | ---: |
| `high` | 193 |
| `warn` | 447 |
| `ok` | 0 |

`high` ≠ 物料 artifact，而是**条件层风险**（强酶 + 温度窗口冲突、跨度过大等）。Route 5 当前审计：

```text
route_class: triage_fragment
issues: condition_high_risk
tags: acylating_piece_present, aryl_coupling_hint, carrier_reagent_terminal
product_like_terminal: false
effective_max_terminal_heavy_atoms: 10
route_plausibility_passed: true
condition_audit.route_risk: high
condition_audit.high_risk_step_count: 1
condition_audit.warning_step_count: 9
condition_audit.temperature_span_c: 199.434
```

→ 物料合理、但条件更像分步合成方案集合而非一锅 cascade。

### 阶段 6：生成连续顺合成路线图（PPT 主图）

新增两类渲染：

```text
scripts/render_route_trees.py            # 连续 retrosynthesis tree
scripts/render_linear_route_schemes.py   # paper-style synthesis scheme（推荐 PPT 用）
```

**推荐 PPT 直接用：**

```text
results/v2/route_schemes_3764f7_top10_current/index.html
results/v2/route_schemes_3764f7_top10_current/scheme_route_01.pdf   # 长路线代表
results/v2/route_schemes_3764f7_top10_current/scheme_route_05.pdf   # 审计修正案例
```

Scheme 风格约定：

- 按合成方向（左→右），不是逆合成方向；
- 反应条件在箭头上下，不写 `template`/`planner` 等内部来源；
- POCl3、LDA、DIBAL-H、Pd catalyst 等用合成方案常用写法；
- 分子下方不写 SMILES（PPT 干净）；
- condition 风险用 `?`（warn）/ `!`（high）标记；
- 图下方注明：**条件是模型逐步预测假设，不是已验证工艺条件**。

---

## 3. 当前可用能力（demo 主线）

### 3.1 端到端 runtime pipeline

```text
ChemEnzy native route search
  → AutoPlanner WebUI queue/cancel
  → material sanity product-audit
  → raw / rejected sidecar artifacts
  → step-level provenance display
```

### 3.2 关键入口

| 项 | 路径 |
| --- | --- |
| 启动 web 服务 | `PYTHONPATH=. python scripts/run_autoplanner_web_waitress.py` |
| 状态监控（一次） | `PYTHONPATH=. python scripts/monitor_autoplanner_web.py --url http://127.0.0.1:7991 --once` |
| 状态监控（持续） | 同上去掉 `--once` |
| 现成路线 JSON | `results/v2/ui_chem_enzy_plan_20260519_032819_3764f7.json` |
| Top-10 SVG 索引 | `results/v2/route_figures_3764f7_top10_current/index.html` |
| Paper-style scheme | `results/v2/route_schemes_3764f7_top10_current/index.html` |
| 路线 shortlist | `results/v2/ui_chem_enzy_plan_20260519_032819_3764f7_route_shortlist.md` |

### 3.3 Runtime 代码主线

| 模块 | 文件 |
| --- | --- |
| Web server | `cascade_planner/web/app.py` |
| Web UI | `cascade_planner/web/static/{app.js,index.html,styles.css}` |
| Web runner | `scripts/run_chem_enzy_plan_for_web.py`、`scripts/run_autoplanner_web_waitress.py` |
| Web monitor | `scripts/monitor_autoplanner_web.py` |
| ChemEnzy adapter | `cascade_planner/baselines/chem_enzy_adapter.py`、`route_contract.py` |
| Material sanity | `cascade_planner/baselines/route_plausibility.py`、`cascade_planner/eval/product_route_feasibility_audit.py` |
| Route selector pack/training | `cascade_planner/eval/build_route_pool_selector_pack.py`、`train_route_selector_v0.py` |
| Web tests | `tests/test_web_app.py`、`test_web_product_audit_filter.py`、`test_chem_enzy_web_payload.py`、`test_route_plausibility.py` |

### 3.4 当前能力清单

| 能力 | 状态 |
| --- | --- |
| 长路线池生成 | 已有（target raw 906 routes） |
| Route 去重（reaction signature） | 已有 |
| Product-aware audit | 已有 |
| Condition reagent → 原子来源解释 | 已有（v1） |
| Condition audit（区分预测风险 vs 物料 artifact） | 已有 |
| Carrier reagent terminal 修正 | 已有（v1） |
| Reject artifact 分层 | 已有 |
| Top-10 paper-style scheme | 已有 |
| 无专家标签训练方向 | 已确定 |

### 3.5 历史最强单模型（被问到再提）

**OA-ARM Skeleton Inpainter**（100-target benchmark）：

| 指标 | 值 |
| --- | ---: |
| Plan rate | 99% |
| GT@5 | 75% |
| Lift over random | 4.0× |
| Avg time/target | 0.81 s |

入口：`python -m cascade_planner.cascadeboard.skeleton_inpainter predict --target "<SMILES>" --n-steps 3 --k 5`

---

## 4. 当前模型证据：什么有信号、什么被冻结

> **本节 = PPT 第 4–16 页**。13 张幻灯片，按 4 个 Phase 串起 5 个月的全部模型工作。
> 每张含【一句话结论 / 数字 / 为什么 / 教训 / 下一步】。

---

### 4.0 一页索引：4 个 Phase × 13 张幻灯片（PPT Slide A · Master Index）

我们的工作分四代：**单步生成器** → **Planner 架构** → **Skeleton-based pipeline** → **Ranker 试错**。每代都留下了"什么被冻结、什么被否决、什么生还"。

| Phase | Slide | 内容 | 当前状态 |
| :-: | :-: | --- | --- |
| **A · 单步生成器世代** | B | EnzExpand template-MLP（Morgan2→template）| 冻结·DESP fallback |
| | C | Dual-tower contrastive（DRFP↔ESM-2）| 冻结·研究侧 |
| | D | Enzyformer v2 → v5（5 代 transformer 单步模型）| 冻结·v4 best |
| | E | **冻结决定**：以 ChemEnzy 7-model ensemble 作为生成主线 | **当前 frozen runtime** |
| **B · Planner 架构世代** | F | CascadeBoardTransformer v20（3.92M params，edit policy + inpainting + real-label） | legacy·保留权重 |
| | G | 粒子精化 planner + Energy API（rule-based 评分） | legacy·保留 |
| | H | two-stage search / cc_aostar / DESP bridge / Reservoir distillation | 各自的边界条件 |
| **C · Skeleton-based pipeline** | I | **OA-ARM Skeleton Inpainter（"painter"）+ Planner fill + Learned Route Scorer** | **当前最强单模型** |
| **D · Ranker 试错 9 连发** | J | 尝试 1–2：Route-pool LambdaRank → cascade_only_pairwise（leakage 教训） | 否决 |
| | K | 尝试 3：Adjacent-step CCTS（0.9987 离线 / live 0 lift） | **ADR 冻结** |
| | L | 尝试 4–5：Runtime hard-neg + no-human probe（打平 retrieval = 0 分） | 否决 |
| | M | 尝试 6–7：Block coherence + value model（分类强 ≠ 路线还原；离线强 ≠ live 强） | 否决 |
| | N | 尝试 9（唯一生还）：Product-audit conservative rerank（live top-GT +5pt） | **唯一推广** |

**贯穿 4 代的根因**：**真实 cascade 正样本 ≈ 3K，不够支撑 cascade-level objective**——这条结论是从 Phase D 反推出来的，但回头看 Phase A/B/C 也都被它限制了。这直接导出第 5 节的 verifier-first 飞轮。

---

## Phase A · 单步生成器世代（4.1 – 4.4）

### 4.1 Slide B · EnzExpand template-MLP（项目最早的酶单步模型）

**做了什么**
- 输入：Morgan2-2048 产物指纹；输出：template-id 分类（softmax over template 库）
- 训练：v2 normalized 8,748 step 中 3,028 可训练子集；GroupKFold by DOI
- 模块：`cascade_planner/expand/enz_template.py` · 模型 `TemplateMLP`

**当时数字（v2 数据）**
- Top-1 template hit ≈ 20–30%（具体数字见 `results/v2/enzexpand_*/`）
- 速度：CPU 上 100ms/query

**为什么不再是主线**
- **template 库覆盖天花板**：训练集没出现过的反应类型完全召不回
- 后来证实在 cascade 多步路线中，单步 top-1 hit 20–30% 会被指数级放大成 routes 几乎不可能匹配 GT
- 但作为 **"反应类型先验"** 仍然有用：DESP 仍把它作为 backward model fallback 注入

**当前定位**
- ✅ 保留在 `cascade_planner/multistep/desp_bridge.py` 作为 EnzExpand backward model
- ✅ 在 source_gate 里仍然是合法 enz 源之一
- ❌ 不作为主线生成器声称

---

### 4.2 Slide C · Dual-tower 对比学习（DRFP × ESM-2）

**做了什么**
- 两塔：反应塔（DRFP fingerprint）+ 酶塔（ESM-2 序列 embedding）
- 对比目标：让真实 (反应, 酶) 对比随机配对更近
- 模块：`cascade_planner/expand/dual_tower.py`，权重 `results/shared/dual_tower_v2.pt`

**当时数字**
- 在 v2 数据上 enzyme retrieval top-5 ≈ 0.4–0.5（粗略，记忆中）
- 需要 GPU + ESM-2 embedding cache（`results/shared/esm_cache/`）

**为什么不再是主线**
- 反应方向的 DRFP 表示太宽——对 cascade 内细微的酶切换不敏感
- ESM-2 序列侧已经在 EnzExpand 之外有更便宜的 EC1/EC3 先验可用
- 主要价值是验证了"酶序列 embedding 可以加进 retro pipeline"这件事

**当前定位**
- ✅ 保留权重作为研究侧 baseline，论文 §"enzymatic retrieval" 引用
- ❌ 不进入 frozen runtime

---

### 4.3 Slide D · Enzyformer v2 → v5（5 代酶单步 Transformer）

**做了什么**
- 多代演进的酶单步 Transformer，retro 方向（产物 → 底物 + 酶）
- 权重序列：`enzyformer_retro_finetuned.pt` / `_v2.pt` / `_v3.pt` / `_v4.pt` / `_v5.pt`（v4 是 best）
- 与 EnzExpand 不同：seq2seq 风格生成，而非 template 分类

**当时数字**
- v4 在 100-target 基准上参与 fill 层，配合 OA-ARM skeleton 拿到 **plan rate 99% / GT@5 75%**（见 4.6 Slide I）
- 单步 enzymatic top-1 ≈ 30–40%（v4，记忆中）

**为什么停在 v5（仍是 frozen 状态）**
- 与 ChemEnzy 7-model（pistachio + reaxys_biocatalysis 已经覆盖了酶反应）部分重叠
- 边际收益递减：v4→v5 提升小于实验噪声
- 训练资源向 OA-ARM skeleton 转移

**当前定位**
- ✅ Enzyformer v4 仍是 Skeleton fill 层的 enz 路径之一（`cascade_planner/cascadeboard/live_retro.py`）
- ✅ source_gate 合法 enz 源
- ❌ 不在主路线声明 Enzyformer 为模型贡献

---

### 4.4 Slide E · 冻结决定：以 ChemEnzy 7-model ensemble 为生成主线

**关键决策**
- vendor `ChemEnzyRetroPlanner`（7 个 backbone：USPTO-full_remapped / bionav_one_step / bkms_metabolic / pistachio / pistachio_ringbreaker / reaxys / reaxys_biocatalysis）
- **状态**：`ready_for_supervised_adapter_manifest_not_direct_dpo`——LoRA NOT ready，DPO loss NOT detected
- 我们**不重训 ChemEnzy**，只能在它输出之上做 rerank/verify/escalation

**为什么把整个 Phase A 工作冻结到 ChemEnzy**
1. ChemEnzy 7-model 在 chem 路径上覆盖远大于我们自训
2. 自训单步在 3K cascade 上必然 overfit（见 Phase A 各代天花板）
3. 我们的差异化应当在 **cascade-level 工作**（fill + score + verify），不在重造 single-step

**约束**
- Phase A 所有自训模型作为 **fallback / diagnostic / research baseline** 保留
- 真正的 single-step 主线 = frozen ChemEnzy
- 任何 cascade-level 提升必须在"不能改 ChemEnzy 权重"前提下成立

**下一步**：进入 Phase B / C / D，研究在 frozen ChemEnzy 之上做什么

---

## Phase B · Planner 架构世代（4.5 – 4.7）

### 4.5 Slide F · CascadeBoardTransformer v20——"monolithic 路线编辑器"为何被证伪

> **诚实说明**：v20 不是"曾经的强模型"，而是"在 OA-ARM 之前唯一在跑的 monolithic 尝试"。我们用它**证伪了"单大模型端到端做路线"这条路**——这一节是 negative result，不是 achievement。

**做了什么**
- `cascade_planner/cascadeboard/route_encoder.py`，3.92M 参数 Transformer
- 三个 head 共享主干：edit policy（怎么改路线）+ inpainting（填空缺 slot）+ real-label heads（评估真实标签）
- 一个模型试图端到端做完所有事——典型 monolithic 路线

**100-target benchmark 真实数字**
| 指标 | 真实含义 | v20 | （后来）OA-ARM |
| --- | --- | :-: | :-: |
| Plan rate 100% | 系统**不弃权**，恒等于 100%——**无信号指标** | 100% | 99% |
| **GT@5 24%** | 真实命中率——**才是真正信号** | **24%** | **75%** |
| Avg time | 推理成本 | ~10s | **0.81s** |

> ⚠ "Plan rate 100%" 在历史文档里被反复引用，但它只是"系统不抛 exception"的同义词，**不是模型强度指标**。真信号是 GT@5。

**为什么被证伪（结构性原因）**
1. **任务干扰**：三 head 在 3.92M 共享主干上争容量，3K cascade 不够分；edit/inpaint/label 互相 regress
2. **重复造轮子**：v20 自己学生成分子细节，等于和 frozen ChemEnzy 7-model 抢同一件事且做得更差
3. **缺少任务分解先验**：没有"painter 学骨架 + frozen generator 填分子 + scorer 排序"这个分工假设

**OA-ARM 用同样数据把 GT@5 从 24% 推到 75%（×3.1）——证明问题不在数据，在架构假设。**

**当前定位**
- ✅ `cli.py` cache mode 仍调用，作为 legacy 路径保留
- ✅ 论文 §"ablation" 作为 monolithic baseline 对照——给 OA-ARM 的任务分解提供反证
- ❌ **不作为模型贡献声称**

**教训（这是项目最重要的一条架构观察）**
> 在 3K cascade + frozen strong generator 的局面下，**任务分解 + 借用 frozen baseline 永远打过 monolithic 端到端**。
> 这条结论直接指导了 Phase C 的 OA-ARM 设计。

---

### 4.6 Slide G · 粒子精化 planner + Energy API（rule-based 评分）

**做了什么**
- `cascadeboard/planner.py` — **粒子滤波风格** 的路线精化（candidate hypergraph 上采样多条路线，按 energy 加权重采样）
- `cascadeboard/energy_api.py` — rule-based energy 评分（cascade 一致性、温度/pH 冲突、cofactor ledger gap 的简单加权）
- 二者搭配是 v20 的搜索时组件

**当时数字**
- 粒子精化对短 cascade（≤3 步）能把 GT@5 提 5–10pt；长 cascade（≥4 步）几乎无效
- Energy API 给出的分数和 product audit 强相关（这一观察直接催生 Slide N 的 product-audit rerank）

**为什么不再扩展**
- 粒子滤波在小 candidate pool 上退化为暴力枚举
- Energy API 的规则数量有限（~10 条），覆盖不了 8 类失败原因里的大部分（见第 5 节）
- 升级方向 = 把 Energy API 的"评分"换成 Verifier 的"硬拒绝"——这就是第 5 节的设计动机来源

**当前定位**
- ✅ 保留作 v20 legacy 搜索组件
- ✅ Energy API 的规则被吸收进 cascade_verifier 的 8 类规则中
- ❌ 不再单独扩展

---

### 4.7 Slide H · 其余 planner 试验：two-stage / cc_aostar / DESP bridge / Reservoir distillation

5 个月里我们还做了这些**架构尝试**，分别拿到了不同的局部结论：

| 名字 | 做什么 | 结论 | 当前 |
| --- | --- | --- | --- |
| `multistep/two_stage_search.py` | chem MCTS（AiZynthFinder）+ enz expansion 串联 | 100-target benchmark plan 66% / GT@5 20%；比 CascadeBoard v20 低，但作为外部对照有意义 | 保留作 MCTS-USPTO baseline |
| `cascadeboard/cc_aostar` | cascade-constrained AO\* 原型，LLM 重排序 skeleton | two-target smoke 通过；full100 等效合并完成 | 论文 §"cc_aostar" 引用，未推广 |
| `multistep/desp_bridge.py` | DESP（学术 search 框架）注册 EnzExpand 作为 backward model | 集成完成，验证 EnzExpand 可作为外部 backward 使用 | 保留作 external integration 证据 |
| Reservoir distillation（Phase I） | student-only 蒸馏 + 多个外部数据集（PaRoutes/USPTO-190/BioNavi-like/USPTO-50K）equal-benchmark | 73 tests 通过；D_FILTER/D_TOP10_FILTER 完成；coverage 多但 runtime 严格门下失败 | reproducibility 保留，不作主结论 |
| AUTOPLANNRELLM（DeepSeek controller） | LLM 控制 stock checker / route-tree / reservoir / cost scoring | 端到端可跑，作为 LLM-augmented planner 原型 | 实验侧保留 |

**Phase B 的总教训**
1. **planner 架构种类很多，但都被 single-step 召回天花板卡死**——任何下游 planner 在 ChemEnzy 不召回的 step 上都无能为力
2. 这条结论与第 5 节"verifier-first"的转向直接相关：**问题在 pool 召回 + 拒绝，不在 planner 算法**

---

## Phase C · Skeleton-based pipeline（4.8 当前最强单模型）

### 4.8 Slide I · OA-ARM Skeleton Inpainter + Planner Fill + Learned Route Scorer

**这是当前 SOTA 单模型，是明天 demo 第二屏要展示的东西。**

**三层结构**

| Layer | 模型 | 参数 | 训练 | 关键指标 |
| --- | --- | --- | --- | --- |
| L1 Skeleton | **OA-ARM Transformer "painter"** | 6.5M（6-layer decoder d=256 8heads） | v3 3,810 routes + 随机置换增广 → 50K 样本 | val rtype_acc **92.4%** / ec1_acc **94.8%** / T MAE 0.74°C |
| L2 Fill | RetroChimera（chem）+ EnzExpand/Enzyformer（enz）| frozen | — | greedy fill + diagnosis 驱动 refine |
| L3 Scoring | 4-layer Transformer encoder（d=128, 4heads, 0.9M） | multi-task：route_score + compat + opmode + issues + yield/ee | 排候选路线 |

**100-target 审计基准（Method E）**

| System | Plan rate | GT@5 | Lift vs random | Fill 质量 | Avg time |
| --- | --- | --- | --- | --- | --- |
| **OA-ARM + Enzyformer v4** | **99%** | **75%** | **4.0×**（random=19%）| **99%** | **0.81s** |
| CascadeBoard v20（legacy） | 100% | 24% | — | — | ~10s |
| MCTS-USPTO | 66% | 20% | — | — | ~30s |

**Statin 侧链 cascade**：3/4 变体通过（skeleton 100% match + valid fill + stock-available endpoint）

**为什么这一代 work**
1. **任务分离**：painter 只学"骨架长什么样"，fill 把分子细节交给已经强大的 frozen generator，scorer 只学"排序"
2. **OA-ARM 随机置换训练**：让 painter 在任意条件子集下都能生成——下游 fill 失败一个 slot 不会阻塞整条路线
3. **6.5M / 0.9M 参数 + 50K 增广样本**：参数和数据规模匹配，没有过参数化

**约束（必须说清楚）**
- L1 painter 的"94.8% ec1_acc" 是在 val（v3 hold-out）上，不是 live 100-target
- L2 fill 的 99% 质量是"通过简单 sanity"，不是"生化可行"
- L3 scorer 排到 GT@5 75% 已经是**当前最高记录**，但这 75% 里多少是"召回到 + 排到"vs"召回到 + 偶然排进 top5"还需进一步消融——这是 30 天 roadmap 的工作

**入口**
```bash
python -m cascade_planner.cascadeboard.skeleton_inpainter predict \
  --target "<SMILES>" --n-steps 3 --k 5
```

**当前定位**
- ✅ **当前最强单模型，明天 demo 第二屏 main exhibit**
- ✅ 论文 §"Skeleton inpainter" 主结果
- ⚠ 仍受 single-step 召回天花板限制——见第 5 节为何还要做 verifier-first

---

## Phase D · Ranker 试错 9 连发（4.9 – 4.13）

> 这是之前那 9 个 ranker 尝试，对应 PPT Slide J–N。结构与之前完全一致，只是从 PPT 角度重新编号。

---

### 4.9 Slide J · 尝试 1–2：Route-pool LambdaRank 与"audit guard 泄漏"教训

**做了什么**
- 在 32,528 条候选路线 / 346 个 target 上训 LightGBM LambdaRank
- 第一版加 `audit_guard` 特征——把 product-audit 的几个布尔信号喂给 ranker

**离线数字**
- `audit_guard` 版本：test **MRR 1.0 / R@1 0.92**——看起来"赢麻了"

**为什么失败**
- `audit_guard` 特征本身就是从 audit 派生的，包含了"这条路线最终是不是 GT"的间接信息——**典型数据泄漏**
- 剥掉 audit 特征训第二版 `cascade_only_pairwise`：R@3 0.85 vs 朴素 retrieval 0.82，R@1 反而 **0.72 vs 0.73**——加了 ranker 比不加还差

**教训**
1. 凡是用 GT 派生的标签构造的特征都不能进入 ranker——必须做 leakage audit
2. R@K 离线指标在小数据上极易过拟合到 pool 结构
3. **"打平 retrieval"必须当作"无信号"，而不是"持平"**——retrieval 是免费 baseline

**下一步**：放弃"在固定 pool 上学排序"的 framing，改为"先扩 pool，再让 verifier 做硬过滤"（第 5 节）。

---

### 4.10 Slide K · 尝试 3：Adjacent-step CCTS——离线 0.9987 / 全局影响微弱

**做了什么**
- 把 cascade 拆成 **相邻两步对**（adjacent step pair），训 pairwise compatibility scorer
- 数据：3,184 真实正 pair + 6,368 hard negative

**离线数字**
- Test **pairwise group acc 0.9987**

**为什么 live 没用**
- CCTS 信号**太局部**：只看相邻两步，看不到"整条路线"
- 接进 search rerank 后，aggregate top-GT 没变化，solved 没变化
- 独立 ADR：[GUARDED_CCTS_DECISION_2026-05-19.md](GUARDED_CCTS_DECISION_2026-05-19.md)

**安全边界（写论文必须照搬）**
> ✅ 可以说：adjacent-step scorer is a safe search-time **diagnostic and tie-break**.
> ❌ 不能说：adjacent-step scorer **solves** cascade-conditioned route selection.

**教训**
1. 局部任务高分 ≠ 全局指标提升
2. 评估必须**离线 + live 两条腿**

**下一步**：CCTS 留作 search-time diagnostic / tie-break；卡片化进入第 5 节 verifier 的 8 类规则之一。

---

### 4.11 Slide L · 尝试 4–5：Runtime hard-negative + no-human probe——"打平 retrieval = 0 分"

**做了什么**
- **尝试 4**：ChemEnzy runtime top-K 作为 hard negative，训 reranker（173k/31k/29k 行）
- **尝试 5**：去掉所有 human-curated 特征，只用 material-sanity / product-sanity + HGB

**离线数字**
| 方案 | MRR |
| --- | --- |
| Retrieval-only baseline | **0.4224** |
| Learned hard-neg blend | 0.4231 (+0.0007) |
| No-human HGB | 0.4224 (delta **0.0**) |

**为什么失败**
- ChemEnzy 7-model ensemble 已经把弱信号吃光了
- 额外的 sanity 信号和 retrieval 排序高度相关，不带新信息

**教训**
1. 在 frozen 强 generator 之后做 rerank，必须先证明 generator 没把信号吃光
2. ΔMRR < 0.005 默认判"零信号"

**下一步**：从"在 ChemEnzy 输出之后接 reranker"转向"**绕开 ChemEnzy 没召回到的部分**"——即 Recall@K ceiling + retrieval-augmented expander（30 天 roadmap）。

---

### 4.12 Slide M · 尝试 6–7：Block coherence + value model——"分类强 ≠ 路线还原；离线强 ≠ live 强"

**做了什么**
- **尝试 6**：Block coherence classifier（连续 N 步工艺自洽性二分类）
- **尝试 7**：Route/block value model，5 个 ablation（含/不含 human / audit / retrieval / route 特征）

**离线数字**
| 任务 | 离线 | 路线还原 / live |
| --- | --- | --- |
| Block coherence 二分类 | AUC **0.94** | route-level analog@1 **0.06–0.08** |
| Value model（最佳 ablation） | MRR **0.872** vs retrieval 0.762 (+0.08) | live top-GT 0.30 → **0.25** ⬇ |

**两个独立的坑**
1. **分类强 ≠ 路线还原强**：AUC 0.94 在路线 analog@1 只有 6–8%
2. **离线强 ≠ live 强**：value model 离线 +0.08，live 反掉 5pt——扰动了 expansion 顺序

**教训（项目最深一次）**
1. 凡不在 live search 闭环里训的 scorer，**默认与 search policy 不兼容**
2. value model 必须用 **on-policy data**（search 自己 rollout）训

**下一步**：放弃"训会评分的 value model"，转向"训会**拒绝**的 verifier"——拒绝是 off-policy-safe 的。

---

### 4.13 Slide N · 尝试 9（唯一生还）：Product-audit conservative final rerank

**做了什么**
- **完全不训模型**：ChemEnzy 输出之上跑 product-audit 规则保守重排
- product 与 query 化学式/MW/原子组成对不上 → 降权；否则维持
- 完全 rule-based，**无专家标签依赖**

**离线 → live 数字（20-target smoke）**
| 指标 | Baseline | + Product-audit rerank |
| --- | --- | --- |
| top-GT | 0.30 | **0.35** (+5 pt) |
| solved rate | 不降 | 不降 |
| stock rate | 不降 | 不降 |

**为什么这个 work**
- 不试图给 candidate 打质量分，只做确定性的事：**杀掉产物对不上的 candidate**
- 与 search policy 正交——纯 post-hoc 剪枝，不扰动 expansion 顺序
- 规则化 → 不会小样本 overfit

**教训**
1. 小数据 + strong frozen baseline 局面下，**确定性规则常比学习模型更安全**
2. 与其训"什么是好"，不如先精确定义"什么显然是坏"

**这是项目目前唯一可写进 paper 的 lift。**

**下一步 = 第 5 节**：把这个洞察推广 → verifier-first 飞轮 = 8 类规则 + 无限负样本 + DPO。

---

### 4.14 Slide O · 串起来：4 代工作的因果链 + 转向陈述

**因果链（明天 PPT 念稿可直接照搬）**

```
Phase A：自训 4 代 single-step 生成器 → 看到 3K 数据天花板
   ↓
冻结 Phase A，迁到 frozen ChemEnzy 7-model
   ↓
Phase B：5 种 planner 架构尝试 → 都被 single-step 召回天花板卡死
   ↓
Phase C：Skeleton-based pipeline（painter + fill + scorer）→ GT@5 75% 当前 SOTA
   ↓
Phase D：在 frozen ChemEnzy + 强 SOTA 之上继续做 ranker → 9 连发，只有规则化的 product-audit 拿到 live lift
   ↓
根因诊断：3K 真实正 cascade + 无限可造负样本 = verifier-shaped 问题，不是 ranker-shaped 问题
   ↓
转向：第 5 节 Verifier-first 飞轮
```

**对外念稿（30 秒版）**

> 我们经历了 4 代工作：自训 4 个单步生成器，再做 5 种 planner 架构，再做 skeleton-based pipeline 拿到 GT@5 75% 的当前最强单模型，最后在它之上做了 9 次 ranker 尝试。
> 唯一拿到 live aggregate lift 的不是任何学习模型，而是规则化的 product-audit 重排。
> 由此得出根因：**正样本稀缺、负样本无限**——这是一个 verifier-shaped 问题，不是 ranker-shaped 问题。
> 下一节展示我们如何把这个洞察规模化。

---

## 5. 新方向 proof：Cascade Verifier-first 飞轮

### 5.1 为什么转向 Verifier-first

把上一节所有失败串起来，根本原因不是模型架构，而是**真实 cascade 正样本 ≈ 3K，没法支撑 cascade-level objective**。

观察：**负样本可以无限造**——一条合规 cascade 改一处原子守恒 / 改一处温度 / 打乱顺序，就是确定性的失败样本。

由此推出新工作流：

```text
真实 cascade（3K，只作正样本来源）
   ↓ 规则扰动（21 类，无限）
30K 标注样本
   ↓ 训 rule verifier + learned verifier
高准确度失败诊断器
   ↓ chosen=正 / rejected=扰动
29K DPO preference pairs
   ↓ DPO / supervised continue-train
ChemEnzy 微调成 cascade-aware
```

**自我强化防火墙（写死的纪律）**：

- Verifier-passed 的 generator output **永远不能**进 supervised positives；
- 只能作为 verifier 的 negative mining 或 DPO 的相对偏好信号；
- 真实 3K cascade 仅用于最终 DPO + 外部 holdout，**禁止**进 verifier 训练集；
- 必须 rule + learned verifier 共存，永远不能让 learned 单独 gate。

### 5.2 已落成的产物

| 类型 | 路径 |
| --- | --- |
| Verifier schema | `cascade_planner/cascade_verifier/schema.py` |
| 规则 verifier（8 类失败原因） | `cascade_planner/cascade_verifier/rules.py` |
| Perturbation pack 生成 | `scripts/build_cascade_perturbation_pack.py` |
| Verifier pack 评估 | `scripts/evaluate_cascade_verifier_pack.py` |
| Learned verifier 训练 | `scripts/train_cascade_verifier_from_pack.py` |
| DPO preference pack | `scripts/build_cascade_verifier_preference_pack.py` |
| ChemEnzy DPO readiness 检查 | `scripts/check_chem_enzy_dpo_readiness.py` |
| ChemEnzy OpenNMT cascade corpus | `scripts/build_chem_enzy_cascade_onmt_corpus.py` |
| 单元/端到端测试 | `tests/test_cascade_verifier.py`、`tests/test_chem_enzy_dpo_readiness.py`、`tests/test_build_chem_enzy_cascade_onmt_corpus.py` |
| Proof artifact 目录 | `results/shared/cascade_verifier_proof_20260519/` |

### 5.3 失败原因 schema（8 类）

| Reason | 来源约束 |
| --- | --- |
| `atom_balance_violation` | 物料守恒 |
| `product_mismatch` | step 产物 ≠ 下一步前体 |
| `temperature_conflict` | 一锅段内 T 窗口冲突 |
| `ph_conflict` | 一锅段内 pH 窗口冲突 |
| `solvent_conflict` | solvent 不兼容 |
| `enzyme_toxicity` | 酶毒化（有机溶剂/重金属/极端 pH） |
| `cofactor_ledger_gap` | cofactor 缺口/不闭合 |
| `route_order_mismatch` | 路线顺序错误 |

与 `cascade_planner/cascade_search/state.py` 内 `CascadeFailureKind` 枚举一一对齐，确保 verifier 输出可直接进 search state。

### 5.4 评估结果

**Top-12 路线 proof（用当前路线池构造）**：

| 指标 | 数值 |
| --- | ---: |
| examples | 84 |
| seed positives | 12 |
| rule negatives | 72 |
| label accuracy | **1.0000** |
| expected-reason coverage | 1.0000 |

**592 条真实路线池扩展（同一 artifact）**：

| 指标 | 数值 |
| --- | ---: |
| routes used | 592 |
| examples | 4,144 |
| seed positives | 592 |
| rule negatives | 3,552 |
| label accuracy | **1.0000** |
| expected-reason coverage | 1.0000 |

**30K structured v4 评估**（来自 `cascadebench_strict_20260516/splits_structured_v1/`）：

| 指标 | 数值 |
| --- | ---: |
| routes used | 1,477 |
| examples | **30,556** |
| seed positives | 1,477 |
| rule negatives | 29,079 |
| label accuracy | **0.9964** |
| expected-reason coverage | 0.9962 |

失败原因覆盖分布（30K）：

| Reason | count |
| --- | ---: |
| temperature_conflict | 13,052 |
| route_order_mismatch | 1,184 |
| atom_balance_violation | 592 |
| cofactor_ledger_gap | 592 |
| enzyme_toxicity | 592 |
| ph_conflict | 592 |

剩余 111 条放过的负样本主要集中在 `route_order` 类，**说明顺序错误仍需更强的 route-continuity 规则或结构化约束**。

### 5.5 Learned Verifier Baseline

`DictVectorizer + LogisticRegression`，51 维特征，按 v4 原始 split：

| 指标 | 数值 |
| --- | ---: |
| examples | 30,556 |
| train / val / test | 21,350 / 4,506 / 4,700 |
| feature dim | 51 |
| feasibility test accuracy | **0.9094** |
| reason micro F1 | 0.9689 |
| reason macro F1 | **0.9653** |

每个 reason head：

| Reason | precision | recall | F1 |
| --- | ---: | ---: | ---: |
| atom_balance_violation | 0.9590 | 0.9956 | 0.9769 |
| temperature_conflict | 0.8979 | 0.9679 | 0.9316 |
| ph_conflict | 0.9288 | 0.9969 | 0.9617 |
| solvent_conflict | 0.9436 | 0.9977 | 0.9699 |
| enzyme_toxicity | 1.0000 | 1.0000 | 1.0000 |
| cofactor_ledger_gap | 1.0000 | 1.0000 | 1.0000 |
| route_order_mismatch | 0.8723 | 0.9673 | 0.9174 |

**重要边界**：binary feasibility 在 test 上 acc 0.91，但 feasible precision 仅 ~0.34。因此目前**只适合作 search value / rerank 辅助信号，不适合作硬 gate**。

### 5.6 Search 接入（已落地）

- `VerifierAugmentedCascadeValueModel`（`cascade_planner/cascade_search/value.py`）：包装现有 heuristic value，把 rule verifier `score` 软融合进 state value，把完整 report 写进 value metadata。
- `LoadedLearnedVerifierValueModel`：直接加载 `learned_verifier_v4_30k.joblib`，给 search runtime 提供 learned feasibility + reason probabilities。
- 仍**保留** heuristic / base value 作为底座，避免 learned model 单独接管 search。

### 5.7 DPO Preference Pack

| Artifact | 路径 |
| --- | --- |
| preference JSONL | `results/shared/cascade_verifier_proof_20260519/verifier_dpo_pairs_v4_30k.jsonl` |
| summary | 同目录 `*_summary.json` |

| 指标 | 数值 |
| --- | ---: |
| groups | 1,477 |
| preference pairs | **29,079** |
| source examples | 30,556 |

主要 rejected reason 分布：cofactor_ledger_gap 5,908 / atom_balance 4,431 / enzyme_toxicity 4,431 / ph_conflict 4,230 / temperature 4,230 / route_order 3,029 / solvent 2,820。

### 5.8 ChemEnzy DPO Readiness（真实 vendor 检查）

| 项 | 结果 |
| --- | --- |
| overall_status | **`ready_for_supervised_adapter_manifest_not_direct_dpo`** |
| configured model count | 7 |
| configured families | `graphfp_models`、`onmt_models`、`template_relevance` |
| preference pairs available | 29,079 |
| supervised vendor training ready | true |
| **direct DPO ready** | **false**（vendor 缺 DPO loss / trainer） |
| **LoRA ready** | **false**（vendor 缺 PEFT/LoRA adapter） |

可严谨表述：

> verifier preference 数据已经准备好；ChemEnzy 具备 supervised 训练入口；
> 直接 DPO/LoRA 还需要新增训练目标和 adapter 代码。

### 5.9 ChemEnzy OpenNMT Cascade Corpus（supervised continue-train 入口）

| Mode | train | valid | test | total |
| --- | ---: | ---: | ---: | ---: |
| `plain` | 1,588 | 414 | 421 | **2,423** |
| `context` | 1,980 | 433 | 437 | **2,850** |

- `plain` = product → reactants，最接近现有 ChemEnzy ONMT char-tokenized checkpoint。**已做 smoke**：`preprocess.py` ✓、`train.py -train_steps 1` ✓、`train_from model_step_100000.pt -reset_optim all` ✓。
- `context` = 加 stage / step idx / 温度桶 / pH 桶 / 溶剂 / EC 前缀 / target token，**真正 cascade-conditioned**，但引入新 token，需要 vocab/model adaptation。

### 5.10 当前 proof 的边界

- verifier 是规则 baseline，不是神经 verifier；
- 评估是规则扰动恢复，不是专家路线可行性评估；
- seed positive 来自当前路线池，是工程 smoke，不等同于真实文献级正样本；
- 真实训练必须按来源分层：`real_literature_cascade` / `synthetic_cascade` / `metabolic_pathway` / `planner_seed_route` **不能混成同一种正样本**。

---

## 6. 目标架构（cascade-stateful planner）

### 6.1 Thesis

> AutoPlanner-Cascade should plan **catalytic cascade processes**,
> not merely multi-step chemoenzymatic routes.

中心问题：

> Can these catalytic steps form a feasible **one-pot, sequential one-pot,
> telescoped, or staged cascade** under shared or segmented conditions?

### 6.2 Target Pipeline

```text
Target molecule + constraints
        ↓
Retrosynthetic proposal layer
  organic / enzymatic / retrieval / template proposals
        ↓
Catalyst and enzyme evidence layer
  EC, enzyme candidate, organism, sequence, substrate similarity, literature
        ↓
Condition envelope layer
  pH, temperature, solvent, buffer, salt, metal, oxygen, redox, cofactors
        ↓
Cascade-state route-tree search
  add reaction step / assign catalyst/enzyme / assign condition window
  merge step into stage / split stage
  insert buffer exchange / quench / isolation
  add cofactor regeneration
        ↓
Cascade feasibility model
  step plausibility / catalyst match / condition likelihood
  pairwise compatibility / cofactor closure / global cascade value / uncertainty
        ↓
Independent cascade verifier  ← 第 5 节已落地 v1
  atom/product sanity / EC/type sanity / pH/T overlap / solvent conflict
  enzyme stability / catalyst poisoning / cofactor/redox balance
  one-pot vs sequential feasibility
        ↓
Ranked cascade plans
  route, stage partition, shared conditions, evidence pack, risks,
  uncertainty, and recommended validation experiments
```

### 6.3 状态对象

**ConditionEnvelope**：`pH_min/max`、`temperature_min/max`、`solvent_class`、`organic_cosolvent_fraction`、`buffer`、`salt/metal`、`oxygen_requirement`、`oxidant/reductant`、`cofactor/cosubstrate`、`water_activity`。已实现：`cascade_planner/cascade_search/state.py` 内含 `overlaps()` / `merged_with()`。

**StagePartition**：`stages = [[step_1, step_2], [step_3]]`，`operation_type ∈ {one_pot, sequential_one_pot, telescoped, isolation}`，`required_operations ∈ {buffer_exchange, quench, extraction, immobilization}`。

**CascadeLedger**：cofactor_balance、redox_conflicts、pH_overlap、temperature_overlap、solvent_risks、metal_enzyme_conflicts、buffer_conflicts、reactive_intermediate_risks、inhibition_flags、evidence_level、uncertainty。

**EvidencePack**：reaction / enzyme / condition / compatibility / negative evidence、model-only assumptions、recommended validation experiment。

### 6.4 设计原则（写死）

> Condition and compatibility must become **search state** and **scoring features**.
> They must not remain only post-hoc metrics like
> `condition_window_success_any` or `cascade_compatibility_success_any`.

---

## 7. 国际前沿对比 + 为何 cascade 仍难

### 7.1 国际前沿

| 系统 | 强 | 弱 |
| --- | --- | --- |
| Segler/Preuss/Waller neural-symbolic + MCTS (Nature 2018) | 模板 + MCTS 搭配，能断到 stock | route-level condition、cascade、cofactor 弱 |
| AiZynthFinder | 开源稳定、工程成熟、stock 可接入 | 仍是 small-molecule retrosynthesis；cascade condition state 有限 |
| ASKCOS（Coley et al., Science 2019） | retrosynthesis + 条件推荐 + flow synthesis 闭环 | 主要面向有机小分子；跨步 cascade 兼容不是核心 |
| RDChiral | template stereochemistry 一致性、SMARTS 应用 | 不解决 route-level cascade / enzyme / condition |
| RetroBioCat (Nat Catal 2021) | 明确面向 biocatalysis + cascade，最接近我们 | enzyme substrate scope / condition compatibility 仍需高质量 KB；长 chemoenzymatic route 评分难 |
| RAscore / SCScore | 快速 synthetic accessibility 排序 | 非逐步验证；不解释 condition；不能替代 audit |

### 7.2 为什么国际前沿仍难处理 cascade（五条）

1. **传统 CASP 是 step-wise，cascade 是 stateful**——cascade 需要跨步传递 solvent / pH / T / redox / cofactor / enzyme 兼容 / 中间体稳定性 / isolation 与否 / workup 兼容性。
2. **Reaction SMILES 通常不完整**——酸碱、氧化剂、卤化剂、保护试剂、盐、水、cofactor、catalyst、solvent 常被略——既会误判合理步骤为原子来源错，又会让真正不合理的步骤蒙混过关。
3. **酶标签不是酶证据**——EC number 只是类，不是具体酶、序列、活性、底物范围、实验条件。当前 EC top-1 confidence 通常 0.07–0.20，只能作 hypothesis。
4. **长路线误差累积**——17–20 步 cascade，每步小概率错也会让整条路线不可执行。
5. **化学 CASP 与生物催化规划割裂**——chemoenzymatic cascade 必须同时处理 chemical transformation、enzymatic transformation、condition prediction、enzyme annotation、stock closure、route-level compatibility。

### 7.3 我们的差异化（一句话）

> 我们不和国际前沿在"能不能搜到 stock"这件事上拼；我们把工作重心放在**长 chemoenzymatic cascade 的可信度评估、可解释失败原因、以及在 3K 真实正样本下还能继续教模型变好**这件事上。Cascade Verifier-first 飞轮就是这件事的第一份可运行 proof。

---

## 8. 仓库主线 vs 研究残骸 vs 已归档

### 8.1 Runtime 主线（不可未经替换删除）

| 区域 | 文件 |
| --- | --- |
| Web server | `cascade_planner/web/app.py` |
| Web UI | `cascade_planner/web/static/{app.js,index.html,styles.css}` |
| Web runner | `scripts/run_chem_enzy_plan_for_web.py`、`scripts/run_autoplanner_web_waitress.py` |
| Web monitor | `scripts/monitor_autoplanner_web.py` |
| ChemEnzy adapter | `cascade_planner/baselines/chem_enzy_adapter.py`、`route_contract.py` |
| Material sanity | `cascade_planner/baselines/route_plausibility.py`、`cascade_planner/eval/product_route_feasibility_audit.py` |
| Route selector pack/training | `cascade_planner/eval/build_route_pool_selector_pack.py`、`train_route_selector_v0.py`、`train_route_pool_ranker.py` |
| Cascade Verifier (v1, NEW) | `cascade_planner/cascade_verifier/{schema,rules}.py`、`cascade_planner/cascade_search/value.py` 内 `VerifierAugmentedCascadeValueModel` / `LoadedLearnedVerifierValueModel` |
| Web tests | `tests/test_web_app.py`、`test_web_product_audit_filter.py`、`test_chem_enzy_web_payload.py`、`test_route_plausibility.py` |
| Verifier tests | `tests/test_cascade_verifier.py`、`test_chem_enzy_dpo_readiness.py`、`test_build_chem_enzy_cascade_onmt_corpus.py`、`test_cascade_search_contract.py` |

### 8.2 Active Research（保留但非默认）

| 区域 | 状态 |
| --- | --- |
| `cascade_planner/cascade_search/` | Active research：cascade search、subgoal proposals、action value hooks、v4 product value |
| `cascade_planner/eval/` 中的 CCTS / subgoal / value / rerank / ablation 脚本 | Research-only：用于 ablation 和 replay，不是 Web default |
| `train_cascade_subgoal_scorer.py` 等 | Research-only，但被 `cascade_search/proposals.py` import |
| `train_cascade_action_value.py` | Research-only，但被 `cascade_search/action_value.py` import |
| `cascade_planner/cascadeboard/` | 含 3 套并行 planner（v20 legacy / OA-ARM skeleton / cc_aostar），demo 后再拆 |

### 8.3 Frozen（写进 ADR，不再上）

按 [GUARDED_CCTS_DECISION_2026-05-19.md](GUARDED_CCTS_DECISION_2026-05-19.md)：

- 所有 `train_ccts_v0/v1/v2/v3_*`
- 所有 `train_cba_v0_*`
- 所有 `cascade_pair_scorer` 变体
- 所有 `block_coherence` / `block_hard_pack`
- 所有 `reservoir_distill` / `bounded_reservoir`
- `controller_v2` lineage
- `route_pool_evidence_review_*` LLM pipeline

### 8.4 Historical（已归档）

| 区域 | 状态 |
| --- | --- |
| Phase I reservoir/student-only distillation | 仅保留 reproducibility，不当作当前主结论 |
| `AUTOPLANNRELLM/` | 可选 LLM 支线，非主框架 |
| `AI_OS_AutoResearch/` | 外部 nested git，本仓库忽略 |
| AI_OS 导出 patch/bundle | `archive/code/generated_patches_2026-05-19/` |
| `releases/autoplanner_cascade_fixed_20260517/` | 冻结 demo 包 |
| 17 份过时 docs | `docs/archive/2026-05/` |
| 20 份 `training_summary_*.json` | `results/shared/archive/legacy_training_summaries/` |
| 39 个 dated runs | `results/shared/archive/dated_runs/` |

### 8.5 已识别但未清的代码 debt（demo 后再处理）

- `cascade_planner/route_tree/verifier.py` 与新 `cascade_verifier/` 命名冲突——需要重命名；
- frozen 训练脚本被 runtime 反向 import（`_baseline_scores`、`_fp`、`_feature_row` 等）——需要下沉到 `cascade_planner/_shared/` 再 freeze；
- `cascadeboard/` 拆分 `runtime/`/`legacy/`/`benchmarks/`。

---

## 9. 下一阶段路线图（30 / 90 天）

### 9.1 30 天（demo 之后立即做）

1. **加强 `route_order` 特征 + 扰动生成**，把当前 30K pack 中 111 条放过的 false-positive 负样本降到 <30。
2. **校准 binary feasibility head**，提高 feasible precision（当前 ~0.34）至 >0.6，才能从 rerank 升级为软 gate。
3. **`context` corpus 的 vocab/model adaptation 方案**——验证 cascade state token 能否进入 ChemEnzy/OpenNMT 训练而不破坏现有 checkpoint。
4. **30 条真实文献 chemoenzymatic cascade benchmark**（外部 holdout）——按 scaffold / source-paper / EC-class 分层切分，禁止随机切分造成泄漏。
5. **Verifier 接入 search 作 value/guard**（已落 `VerifierAugmentedCascadeValueModel` 骨架，需在 statin panel 上跑端到端 A/B）。

### 9.2 90 天（论文级目标）

6. **新增 DPO / pairwise loss wrapper**——在 vendor OpenNMT/template 通路上加 DPO loss 与 LoRA/PEFT adapter，把 `chem_enzy_dpo_readiness.json` 从 `ready_for_supervised_adapter_manifest_not_direct_dpo` 升级到 `dpo_ready`。
7. **正式 ChemEnzy 微调**——用 29K verifier-derived DPO pairs 跑实战微调，外部 holdout 验证。
8. **Atom contribution profile** 升级 carrier 白名单为通用规则：raw heavy atoms / atoms retained in product / retained fraction / target coverage / role guess / role confidence / evidence。
9. **Atom mapping / MCS 引入**：优先 atom-mapped reaction → RXNMapper → RDKit MCS → role prior。
10. **Condition reagent role assignment**：explicit reactants / small stoichiometric reagent / catalyst / solvent / unknown 分层。
11. **Enzyme evidence calibration**：补 substrate scope similarity / UniProt sequence / cofactor / pH-T compatibility / cascade compatibility（接 BRENDA）。
12. **CascadeLedger 作为 search state**——每步不只生成产物，还更新 route_state。
13. **Tiered positive labels 严格分层**：L1 synthetic cascade（50K–200K，verifier pretrain only）、L2 biological（Rhea release 140，~10K）、L3 gold（3K real + 30 literature，finetune/DPO/holdout only），**跨层混合禁止**。

### 9.3 Go / Kill 准则（写死）

- 如果 90 天内 ChemEnzy + DPO 微调 在 30 条真实文献 cascade holdout 上**不能稳定超过 native ChemEnzy + product-audit rerank**，则 verifier-DPO 路线本身也要 freeze，转向 condition/state-aware search 直接做 search-time 集成。
- 任何 learned scorer 想做硬 gate，必须先证 feasible precision >0.6 + retrieval-only control 不打平 + live search aggregate quality lift。

---

## 10. 标准化 PPT 大纲（22 张正片 + 5 张 backup）

> **使用方法**：每张 slide 严格 4 个区块——【标题 / Takeaway（顶部一行结论）/ 正文（左侧 bullets）/ 图表（右侧）】，外加底部讲解念稿（speaker notes）。学术汇报标准布局。

> **总时长**：22 张 × ~45 秒 = 17 分钟主讲 + 8 分钟 Q&A backup。

---

### Slide 1 · 封面

| 区块 | 内容 |
| --- | --- |
| 标题 | **AutoPlanner: Chemo-Enzymatic Cascade Retrosynthesis Planner** |
| 副标题 | *From single-step ranking to verifier-first cascade planning* |
| 作者/单位/日期 | （自填）· 2026-05-20 |
| 图 | 1 张系统总览示意：左 target 分子骨架 → 中 cascade 路线树 → 右 verifier 飞轮 |

**念稿**："今天汇报我们 5 个月在 chemo-enzymatic cascade 路线规划上的工作。我们会展示当前最强单模型、9 次 ranker 试错的诚实记录、以及下一阶段的 verifier-first 飞轮设计。"

---

### Slide 2 · 汇报大纲（Outline）

| 区块 | 内容 |
| --- | --- |
| Takeaway | 5 部分 · 17 分钟 |
| 正文 | 1. 问题定义与难度<br/>2. 相关工作与差异化<br/>3. 我们做过的 4 代工作（A/B/C/D）<br/>4. 当前最强单模型 + 唯一 live lift<br/>5. 新方向 Verifier-first + 路线图 |
| 图 | 一条横向时间线：Phase A（generator）→ B（planner）→ C（skeleton SOTA）→ D（ranker 试错）→ E（verifier-first 飞轮·future） |

**念稿**："5 个部分。中间 3 部分都是诚实的 lessons learned，包括我们冻结了哪些路线、为什么冻结。"

---

### Slide 3 · 问题定义：什么是 chemo-enzymatic cascade retrosynthesis

| 区块 | 内容 |
| --- | --- |
| Takeaway | 不是"找一条路线"，是"找一条**长 cascade、跨化学+酶、条件自洽**的可信路线" |
| 正文 | • 输入：target SMILES + 可选约束（one-pot / green / EC 限制）<br/>• 输出：N 步 cascade 路线，每步含 substrate / product / enzyme or catalyst / T / pH / solvent<br/>• 约束：相邻步条件兼容、cofactor 平衡、原子守恒、立体保持<br/>• 数据天花板：v4 release **3,744 高质量 cascade / 8,609 steps**（gold 2,885 + silver 859） |
| 图 | 一张示意路线：4 步 cascade，前 2 步酶催化 + 后 2 步化学合成，标出 condition envelope（T/pH/solvent 一致性框） |

**念稿**："注意输出格式——不只是反应箭头，每一步都带条件。这是和普通 retrosynthesis 的本质区别。"

---

### Slide 4 · 为什么 cascade 比普通 retrosynthesis 难（5 条结构性原因）

| 区块 | 内容 |
| --- | --- |
| Takeaway | 5 条结构性难度叠加，是当前 SOTA（AiZynthFinder / ASKCOS）也没 cover 的部分 |
| 正文 | 1. **Stateful conditions**：T/pH/solvent 在相邻步必须兼容，不是 step-wise 独立<br/>2. **SMILES 不完整**：cofactor / 立体 / 同位素信息常缺失<br/>3. **EC ≠ enzyme evidence**：EC 1.1.1.1 对应数千酶，真实酶来源/活性需独立证据<br/>4. **长路线累积误差**：单步 80% 召回 → 5 步 32% 联合召回<br/>5. **化学 CASP 与 biocatalysis 数据生态割裂**：USPTO 50K vs Rhea/BRENDA 互不通用 |
| 图 | 5 个 icon 横排：温度计 / SMILES 字符串残缺 / EC 树状图 / 衰减曲线 / 两个不相交的圆 |

**念稿**："这 5 条决定了我们不能照搬 AiZynthFinder 直接得到 cascade 结果——必须额外建模条件和酶证据。"

---

### Slide 5 · 相关工作与 SOTA Landscape

| 区块 | 内容 |
| --- | --- |
| Takeaway | 我们不与单步 SOTA 比单步，我们与 cascade 评估和 condition prediction 比 |
| 正文 | 单步：Segler 2018 / AiZynthFinder / ASKCOS / RDChiral templates<br/>多步搜索：MCTS / AO\* / DESP<br/>评分：RAscore / SCScore<br/>酶侧：RetroBioCat / EnzymeMap / ReactZyme / RetroChimera |
| 图 | 一张 2×3 网格表，行 = single-step / multi-step / scoring，列 = 化学 / 酶；填入对应工作 |

**念稿**："SOTA 在化学单步上已经很强；我们不挑战这一层。我们的工作集中在 cascade-level 条件兼容、酶证据校准和长路线可信度。"

---

### Slide 6 · 我们的系统总览（架构图）

| 区块 | 内容 |
| --- | --- |
| Takeaway | 三层 + 一个未来环：painter → fill → score → (verifier 飞轮) |
| 正文 | **L1 Skeleton Painter**（OA-ARM Transformer 6.5M）<br/>**L2 Fill**（frozen ChemEnzy 7-model + Enzyformer v4）<br/>**L3 Route Scorer**（4-layer Transformer 0.9M）<br/>**(Future) L4 Verifier**（8 类规则 + learned）→ DPO → 反哺 ChemEnzy |
| 图 | 系统流程图：target → L1 skeleton → L2 fill → L3 score → top-K routes；下方虚线框 L4 verifier→DPO→ChemEnzy（标"30/90 天"）|

**念稿**："这是系统全景。前三层是今天能演示的，最后一层是下一阶段。我们刻意把任务分解开——这是从 v20 monolithic 失败中学到的最重要架构原则。"

---

### Slide 7 · Phase A 总结：单步生成器世代（4 代后冻结）

| 区块 | 内容 |
| --- | --- |
| Takeaway | 自训 4 代，发现 3K 数据不够；冻结到 frozen ChemEnzy 7-model |
| 正文 | • EnzExpand template-MLP：top-1 hit 20–30%，template 库覆盖天花板<br/>• Dual-tower DRFP×ESM-2：粗粒度，对 cascade 内细微切换不敏感<br/>• Enzyformer v2→v5：seq2seq，v4 best，仍 ≤40% top-1<br/>• **冻结**：迁到 vendor ChemEnzy 7-model（USPTO/pistachio/reaxys/bkms 等）|
| 图 | 折线图：x 轴 4 代模型，y 轴 single-step top-1 hit；横线标 ChemEnzy ensemble 的水平 |

**念稿**："我们试了 4 代单步模型，每代提升越来越小。在 3K cascade 数据下，单步质量被数据卡死。所以我们做了关键决定：冻结自训，迁到 ChemEnzy。"

---

### Slide 8 · Phase B 总结：Planner 架构世代（5 个尝试）

| 区块 | 内容 |
| --- | --- |
| Takeaway | 5 个 planner 架构都被同一个天花板卡死——下游再聪明也补不回上游的召回 |
| 正文 | • CascadeBoardTransformer v20（monolithic，**证伪**）<br/>• 粒子精化 planner + Energy API（短路线 +5pt，长路线无效）<br/>• two-stage search（chem MCTS + enz expansion，plan 66%）<br/>• cc_aostar（LLM 重排 skeleton 原型）<br/>• Reservoir distillation（73 tests pass，runtime 严格门下失败） |
| 图 | 横向条形图：5 个 planner 在 100-target benchmark 上的 GT@5；与 retrieval-only baseline 一根参考线 |

**念稿**："Phase B 的核心发现是：planner 算法种类很多，但只要 single-step 召回不到 GT 反应物，下游再聪明也只能在错的池子里挑。"

---

### Slide 9 · Phase C 核心：OA-ARM Skeleton Inpainter（架构）

| 区块 | 内容 |
| --- | --- |
| Takeaway | 任务分解 + 借用 frozen baseline = 当前最强单模型 |
| 正文 | • **架构**：6-layer Transformer decoder · d=256 · 8 heads · 6.5M 参数<br/>• **训练目标**：order-agnostic autoregressive，随机排列 slot 顺序<br/>• **数据**：v3 3,810 routes × 增广 → 50K 训练样本<br/>• **输出**：每个 slot 的 reaction type + EC1 + T + pH 分布<br/>• **验证集精度**：rtype_acc **92.4%** · ec1_acc **94.8%** · T MAE **0.74°C** |
| 图 | 模型结构示意：左 target FP + 约束 → encoder → 6 层 decoder → N 个 slot 输出；标注 OA-ARM 随机排列 mask 模式 |

**念稿**："核心创新是 painter 只学骨架——反应类型、EC 类、条件分布。具体分子细节交给已经很强的 frozen generator。这是 OA-ARM 任务分解原则。"

---

### Slide 10 · Phase C 结果：100-target audited benchmark

| 区块 | 内容 |
| --- | --- |
| Takeaway | OA-ARM 在所有维度全面超过 monolithic v20 与 MCTS-USPTO |
| 正文（表格化） | （见右图） |
| 图 | 表格 4 行 × 5 列：<br/>| System | Plan rate | **GT@5** | Lift vs random | Avg time |<br/>| OA-ARM + Enzyformer v4 | 99% | **75%** | **4.0×** | 0.81s |<br/>| CascadeBoard v20 (legacy) | 100% | 24% | 1.3× | ~10s |<br/>| MCTS-USPTO | 66% | 20% | 1.1× | ~30s |<br/>| Random baseline | — | 19% | — | — |<br/>**重点标红 GT@5 75% 和 4.0× lift** |

**念稿**："75% 的 GT@5 是从随机 baseline 19% 拉上来的 4 倍提升。Plan rate 100% vs 99% 不要看——那只是'系统不弃权'的指标，没意义。**真信号是 GT@5**。"

---

### Slide 11 · Phase D 概览：在 frozen-ChemEnzy 之上的 9 次 ranker 试错

| 区块 | 内容 |
| --- | --- |
| Takeaway | 9 个 ranker 方向、8 个被否决、1 个生还——根因诊断在下一页 |
| 正文 | （见右表） |
| 图 | 表格 9 行 × 3 列：<br/>| # | 方向 | 状态 |<br/>| 1 | Route-pool LambdaRank + audit_guard | ❌ 数据泄漏 |<br/>| 2 | cascade_only_pairwise | ❌ 信号弱 |<br/>| 3 | Adjacent-step CCTS | ⚠ ADR 冻结 |<br/>| 4 | Runtime hard-neg blend | ❌ ΔMRR=0.0007 |<br/>| 5 | No-human HGB probe | ❌ Δ=0 |<br/>| 6 | Block coherence classifier | ❌ AUC 0.94 / analog@1 6% |<br/>| 7 | Route/block value model | ❌ 离线↑ live↓ |<br/>| 8 | Expert CSV expansion | ⏸ 不阻塞主线 |<br/>| **9** | **Product-audit conservative rerank** | ✅ **唯一生还** | |

**念稿**："5 个月、9 次尝试、1 个生还。下一页解释为什么生还的不是学习模型。"

---

### Slide 12 · Phase D 唯一生还者：Product-audit conservative rerank

| 区块 | 内容 |
| --- | --- |
| Takeaway | 唯一拿到 live lift 的不是学习模型，是规则化的产物审计——top-GT **0.30→0.35** |
| 正文 | • 做法：在 frozen ChemEnzy top-K 输出之上做规则重排<br/>• 规则：candidate product 与 target 化学式/MW/原子组成对不上 → 降权<br/>• 完全 rule-based，**无专家标签**<br/>• 20-target smoke：top-GT 0.30→**0.35** (+5pt)；solved/stock 不降 |
| 图 | 柱状对比图：左 Baseline 0.30；右 +Product-audit 0.35；标 "+5pt" 红色箭头 |

**念稿**："为什么这个 work 而前 8 个不 work？因为它不试图给 candidate 打质量分，只杀掉确定性错的。这件事和 search policy 正交，不会扰动 expansion 顺序，也不会小样本 overfit。"

---

### Slide 13 · 根因诊断：为什么 ranker 路线必然失败

| 区块 | 内容 |
| --- | --- |
| Takeaway | 3K 真实 cascade 正样本，不足以支撑 cascade-level 排序学习——这是数据约束，不是架构问题 |
| 正文 | • **正样本稀缺**：v4 仅 ~3K 真实 cascade routes<br/>• **负样本无限**：任意改一处温度/原子/顺序就是确定性的失败<br/>• **学习排序需要正样本量级 ≫ 负样本**——我们正好反过来<br/>• **诊断**：这不是 ranker-shaped 问题，是 **verifier-shaped** 问题 |
| 图 | 一张对比示意：左侧"ranker 范式"（正样本少→ overfit）；右侧"verifier 范式"（负样本无限→ scalable）|

**念稿**："这是项目最关键的一次认知转变。Ranker 需要'什么好'，verifier 只需要'什么显然坏'。在我们的数据约束下，后者可以无限造样本。"

---

### Slide 14 · Paradigm Shift：Verifier-first 飞轮（设计图）

| 区块 | 内容 |
| --- | --- |
| Takeaway | 把 ranker 任务降级为"拒绝 + 排序"——拒绝层用规则 + learned，排序层只在通过的池子里做 |
| 正文 | • **8 类规则**：atom_balance / product_mismatch / temperature / pH / solvent / enzyme_toxicity / cofactor_ledger / route_order<br/>• **真实 cascade → 规则扰动 → verifier 训练** → **DPO pairs** → **ChemEnzy 微调**<br/>• **自我强化防火墙**：verifier-passed generator 输出**永不**进入 supervised positives |
| 图 | 飞轮图：真实 cascade → 规则扰动 → verifier → DPO pair → ChemEnzy → 新 cascade（回到起点）；防火墙标在"新 cascade → 真实"的入口位置 |

**念稿**："这是新的工作流。注意防火墙——这是防止 verifier 闭环正反馈污染的关键设计。"

---

### Slide 15 · Verifier proof：实证数字

| 区块 | 内容 |
| --- | --- |
| Takeaway | 在 30,556 扰动样本上验证：规则 acc 0.9964 / learned acc 0.9094 / reason F1 0.9653 / 真实 592 routes acc 1.0000 |
| 正文 | • 扰动 pack：30,556 个 (cascade, perturbation_kind, label) 样本<br/>• 规则 verifier：accuracy **0.9964**<br/>• Learned verifier（DictVectorizer + LR，51 维）：accuracy **0.9094** · reason macro F1 **0.9653**<br/>• 真实 592 routes hold-out：label accuracy **1.0000** |
| 图 | 一张表 + 一张混淆矩阵热图：8 类 reason × 8 类 reason，对角线深色 |

**念稿**："这些数字证明 verifier-first 范式在我们的数据上是可学的。reason F1 0.9653 意味着不仅判'坏'，还能解释'为什么坏'——这是 DPO 的前提条件。"

---

### Slide 16 · DPO Readiness：29,079 pairs 就绪，flywheel 待启动

| 区块 | 内容 |
| --- | --- |
| Takeaway | DPO 训练 pair 数据集已就绪，但 vendor ChemEnzy 暂缺 DPO loss / LoRA——supervised 入口可走 |
| 正文 | • **DPO pairs**：**29,079** 对（chosen / rejected）<br/>• 覆盖 8 类 reason 全部<br/>• 平均 chosen-rejected verifier score gap：（自填）<br/>• ChemEnzy readiness：supervised 入口 ✅ · direct DPO ❌（vendor 缺 DPO loss / LoRA）<br/>• 落地路径：先 supervised continue-train，再等 vendor adapter |
| 图 | 一个进度条：v3-only proof ✅ → 30K perturbation pack ✅ → learned verifier ✅ → DPO pair ✅ → **ChemEnzy DPO 训练（等 vendor）** → live A/B（30 天）|

**念稿**："DPO pair 已经造完。下一步是和 vendor 协调 adapter，或先走 supervised continue-train 拿一个保守版本。"

---

### Slide 17 · Live Demo / Route Exhibit

| 区块 | 内容 |
| --- | --- |
| Takeaway | 系统能为复杂 target 生成长 cascade 路线候选，并已加入 condition audit 标记 |
| 正文 | • Demo target：（自选，建议 statin 侧链 cascade）<br/>• 路线长度：17-step<br/>• 通过 verifier：✓ / 标记的风险点：carrier reagent / unsupported element source |
| 图 | **核心**：放 `scheme_route_01.pdf`（17-step cascade scheme）。建议占整页 70%，左侧只放 4–5 个 caption（target / steps / verifier verdict / risk flags） |

**念稿**："这是系统能生成的代表路线。注意每一步都标了 enzyme/catalyst 和条件。橙色框是 verifier 标记的需要专家审查的点——这正是 verifier-first 的最直接价值。"

---

### Slide 18 · SOTA 对比表

| 区块 | 内容 |
| --- | --- |
| Takeaway | 我们不和 ASKCOS/AiZynthFinder 比单步召回；我们和它们比 cascade-level 评估 + condition audit + verifier-first |
| 正文 | （见右表） |
| 图 | 表格 6 行 × 4 列：<br/>| 系统 | 单步 | Cascade 评估 | Verifier-first |<br/>| Segler 2018 | ✓ | ✗ | ✗ |<br/>| AiZynthFinder | ✓ | △（template only） | ✗ |<br/>| ASKCOS | ✓ | △ | ✗ |<br/>| RDChiral | ✓（template）| ✗ | ✗ |<br/>| RetroBioCat | △ | ✓（biocatalysis only） | ✗ |<br/>| **AutoPlanner** | **借用 ChemEnzy** | **✓ +condition audit** | **✓（30K perturbations）** | |

**念稿**："我们的差异化不在单步算法——那一层已经被 SOTA 解决了。我们在 cascade 评估、条件审计、verifier 飞轮这三件事上是当前唯一系统化的方案。"

---

### Slide 19 · 30 天路线图（Sprint 1）

| 区块 | 内容 |
| --- | --- |
| Takeaway | 5 件事，全部可量化、有 Go/Kill |
| 正文 | 1. **Recall@K ceiling 诊断**：frozen ChemEnzy 在 v4 gold/silver 上的 recall@10/50/200，分 tier 输出<br/>2. **DPO pair 分桶重做**：按 GT∈pool / retrieval-only / 不可达分三桶<br/>3. **route_order 规则加强**：覆盖更多顺序冲突模式<br/>4. **30 条文献 benchmark**：建立 holdout 评测集<br/>5. **Verifier 接入 search guard**：A/B on statin panel |
| 图 | 5 行 Gantt 图：横轴 30 天，每行一个任务条 |

**念稿**："5 件事都是直接服务于 verifier-first 飞轮的真实落地。第 1 件是最高优先级——没有 ceiling 数字，所有架构辩论都是猜。"

---

### Slide 20 · 90 天路线图（Sprint 2/3）

| 区块 | 内容 |
| --- | --- |
| Takeaway | 与 vendor 协调 DPO adapter，落地 ChemEnzy 微调闭环 |
| 正文 | • Vendor DPO/LoRA adapter（与 ChemEnzy 团队协调）<br/>• ChemEnzy DPO 正式微调（用 29K pairs）<br/>• Atom contribution / atom mapping / MCS 进入 verifier 特征<br/>• Condition reagent role 建模（区分 carrier vs reactant）<br/>• Enzyme evidence calibration（EC + 实际酶来源）<br/>• CascadeLedger as search state |
| 图 | 6 个里程碑节点的 timeline，每个节点标 "deliverable" + "verification metric" |

**念稿**："90 天目标是把今天展示的 verifier 数据真的喂回 ChemEnzy 拿到 live 提升。这需要和 vendor 在 adapter 层面协作。"

---

### Slide 21 · Go / Kill 准则

| 区块 | 内容 |
| --- | --- |
| Takeaway | 给所有方向预设 kill criteria，避免无限延展 |
| 正文 | • **Verifier-first Kill**：90 天内 ChemEnzy+DPO 在 30 条真实 holdout 上**无法稳定超过** native+product-audit rerank → freeze verifier-DPO 路线<br/>• **Skeleton pipeline Kill**：6 个月内 GT@5 不能进一步 +5pt → 转纯 retrieval-augmented<br/>• **Recall ceiling Kill**：若发现 recall@200 < 50% → 强制转范式（forward search）<br/>• **每月 milestone review**，所有 metric 写进 `results/shared/quarterly_review/` |
| 图 | 决策树：3 个 Kill 分支，每个分支末端写 "fallback plan" |

**念稿**："这页是给评委看的——我们对每条路线都预设了放弃条件。这不是悲观，是研究纪律。"

---

### Slide 22 · 结论与可声明边界

| 区块 | 内容 |
| --- | --- |
| Takeaway | 3 件可声明、2 件待 prove、1 件待 vendor 协作 |
| 正文 | **可声明（写论文 / 汇报安全）** ✅<br/>1. OA-ARM skeleton pipeline 在 100-target benchmark 上 GT@5 75% / plan rate 99%<br/>2. Product-audit conservative rerank 在 20-target smoke 上 top-GT +5pt（live lift）<br/>3. Verifier-first 飞轮 proof：30K perturbations / rule acc 0.9964 / 29K DPO pairs 就绪<br/><br/>**待 prove**（30/90 天）⚠<br/>4. Recall@K ceiling 量化与分桶 DPO 修正<br/>5. Verifier-as-search-guard live A/B<br/><br/>**待 vendor 协作** ⏸<br/>6. ChemEnzy DPO/LoRA adapter |
| 图 | 三分屏：✅ 已证明 / ⚠ 进行中 / ⏸ 待协作；每屏 2–3 行 |

**念稿**："这是诚实的边界。我们不夸大也不藏短——这正是评委希望看到的研究态度。谢谢，欢迎提问。"

---

## 10A. Backup Slides（Q&A 备用，按问题召唤）

| Backup # | 内容 | 触发问题 |
| :-: | --- | --- |
| B1 | OA-ARM 详细消融（去掉随机排列 / 去掉增广 / 改 head 数） | "为什么 OA-ARM 比 v20 强 3 倍？" |
| B2 | Verifier 8 类规则定义详细表 | "verifier 到底在判什么？" |
| B3 | 数据分 tier 详细统计（gold/silver/bronze） | "v4 数据质量如何？" |
| B4 | DPO pair 构造算法 + 自我强化防火墙伪代码 | "为什么 verifier 不会闭环污染？" |
| B5 | Live demo target 的 full trace（每步条件、verifier verdict） | "能不能跑一个端到端？" |

---

## 10B. PPT 制作工程建议

| 项 | 建议 |
| --- | --- |
| 模板 | 学术蓝灰系（避免炫彩商业模板） |
| 字体 | 标题：Source Han Sans Heavy / Arial Bold；正文：Source Han Sans Regular / Arial |
| 字号 | 标题 28pt / Takeaway 22pt / 正文 18pt / 图例 14pt |
| 配色 | 主色 #1F4E79（深蓝）/ 强调 #C00000（红，仅用于"重点数字"）/ 中性 #595959（深灰） |
| 图表 | 全部矢量（SVG/PDF 导入）；柱状图/折线图用 matplotlib 或 mermaid 预生成 |
| 命名 | 文件名 `AutoPlanner_2026-05-20_demo_v1.pptx`；图片资源放 `paper/figures/` |
| Speaker notes | 每张 slide 念稿写进 PPT 的"备注"区，不展示给观众 |
| 时间控制 | 每张 ≤ 45 秒；Slide 17（demo）+ Slide 22（结论）可各延 30 秒 |

---

## 11. 可以说 / 不能说 边界

### 可以说

- 系统已经能为复杂 target 生成大量长路线候选；
- 仅以 stock closure 作为成功标准是不够的；
- 已经建立 product-aware audit 和 route-level triage（360 + 232 + 48 分层）；
- 已发现并修正 naive terminal-heavy-atom 规则的误伤（route 5 案例）；
- 已开始处理 condition reagent 对原子来源的解释；
- 已生成 top-10 顺合成 synthesis scheme + 连续路线树供专家审查；
- 已完成 Cascade Verifier v1：30K 规则扰动、规则准确率 0.9964、learned acc 0.9094、29K DPO pairs；
- ChemEnzy supervised continue-train 入口已 smoke 通过；
- adjacent-step CCTS pair reward 作为 safe search-time diagnostic 已落地。

### 不能说

- **不能说**已经得到可直接实验执行的合成工艺；
- **不能说**酶步骤已经验证；
- **不能说**所有 `triage_fragment` 都是高质量路线；
- **不能说**当前 carrier / condition 逻辑已经完全通用；
- **不能说**无需专家后续审查；
- **不能说**ChemEnzy DPO/LoRA 已完成（vendor 缺 loss/adapter）；
- **不能说**adjacent-step scorer "solves" cascade-conditioned route selection；
- **不能说**verifier-passed 路线就是化学家可接受路线（verifier 是规则 baseline）；
- **不能说**learned route/block value model 在 live search 上稳定胜过 baseline（**反例已有**：top-GT 0.30→0.25）。

---

## 12. 参考文献

1. Segler, M. H. S.; Preuss, M.; Waller, M. P. Planning chemical syntheses with deep neural networks and symbolic AI. *Nature* **555**, 604–610 (2018). https://www.nature.com/articles/nature25978
2. Genheden, S. *et al.* AiZynthFinder: a fast, robust and flexible open-source software for retrosynthetic planning. *J. Cheminformatics* **12**, 70 (2020). https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00472-1
3. Coley, C. W. *et al.* A robotic platform for flow synthesis of organic compounds informed by AI planning. *Science* **365**, eaax1566 (2019). https://doi.org/10.1126/science.aax1566
4. Coley, C. W.; Green, W. H.; Jensen, K. F. RDChiral: An RDKit wrapper for handling stereochemistry in retrosynthetic template extraction and application. *J. Chem. Inf. Model.* **59**, 2529–2537 (2019). https://pubs.acs.org/doi/10.1021/acs.jcim.9b00286
5. Finnigan, W.; Hepworth, L. J.; Flitsch, S. L.; Turner, N. J. RetroBioCat as a computer-aided synthesis planning tool for biocatalytic reactions and cascades. *Nat. Catal.* **4**, 98–104 (2021). https://www.nature.com/articles/s41929-020-00556-z
6. Thakkar, A. *et al.* Retrosynthetic accessibility score: rapid machine learned synthesizability classification from AI driven retrosynthetic planning. *Chem. Sci.* **12**, 3339–3349 (2021). https://pubs.rsc.org/en/content/articlehtml/2021/sc/d0sc05401a
7. ACS Green Chemistry Institute, Process Mass Intensity materials. https://www.acs.org/green-chemistry-sustainability/green-chemistry-nexus/articles/process-mass-intensity-calculation-tool.html
8. ACS GCI Pharmaceutical Roundtable, Process Mass Intensity Metric. https://learning.acsgcipr.org/guides-and-metrics/metrics/process-mass-intensity-metric/
9. Sheldon, R. A. The E factor at 30: a passion for pollution prevention. *Green Chem.* **25**, 1704 (2023). https://pubs.rsc.org/fa/content/articlelanding/2023/gc/d2gc04747k

---

**End of document.**  
本文件为单一权威汇报底稿；如发现与子文档冲突，以本文件为准。
