# AutoPlanner Docs

Last update: 2026-05-19. 历史文档已移至 [archive/2026-05/](archive/2026-05/)。

> **👉 PPT / 汇报请直接读：[AutoPlanner_Final_Report_2026-05-19.md](AutoPlanner_Final_Report_2026-05-19.md)**
> 单一权威合并文稿（合并 6 份现役文档），含一页摘要、6 个阶段、所有尝试与结果、新方向 proof、目标架构、30/90 天路线图、12-slide PPT 结构、可说/不可说边界、参考文献。

---

## 一、明天 Demo 主线（best model right now）

### 主展品 1：可工作的端到端系统 — ChemEnzy + WebUI + Product Audit

唯一一个**能现场跑、有 UI、有 SVG 路线图、有 provenance** 的 demo。

| 项 | 路径 |
| --- | --- |
| 启动 web 服务 | `PYTHONPATH=. python scripts/run_autoplanner_web_waitress.py` |
| 状态监控 | `PYTHONPATH=. python scripts/monitor_autoplanner_web.py --url http://127.0.0.1:7991 --once` |
| 现成路线 JSON | `results/v2/ui_chem_enzy_plan_20260519_032819_3764f7.json` (statin 类目标，640 条 kept routes) |
| Top-10 路线 SVG 索引 | `results/v2/route_figures_3764f7_top10_current/index.html` |
| **重点 case study** | `results/v2/route_figures_3764f7_top10_current/route_05.svg`（Wittig phosphorane + POCl3，演示 audit 修正） |
| 路线 shortlist | `results/v2/ui_chem_enzy_plan_20260519_032819_3764f7_route_shortlist.md` |
| 入口代码 | `cascade_planner/web/app.py`、`cascade_planner/baselines/chem_enzy_adapter.py` |

### 主展品 2：新方向的可运行 proof — Cascade Verifier v1（今天刚跑通）

把 "3K 正样本 → verifier-first 飞轮" 从口头收紧成代码 + 数字。详见
[CASCADE_VERIFIER_PROOF_REPORT_2026-05-19.md](CASCADE_VERIFIER_PROOF_REPORT_2026-05-19.md)。

| 关键数字 | 值 |
| --- | --- |
| Perturbation 样本 | 30,556 |
| 规则 verifier label accuracy | **0.9964** |
| 规则 verifier expected-reason coverage | **0.9962** |
| Learned verifier feasibility test acc | 0.9094 |
| Learned verifier reason macro F1 | **0.9653** |
| 真实路线池 592 条 label accuracy | 1.0000 |
| DPO preference pairs 就绪 | 29,079 |

| 产物 | 路径 |
| --- | --- |
| Schema | `cascade_planner/cascade_verifier/schema.py` |
| Rule verifier | `cascade_planner/cascade_verifier/rules.py` |
| Perturbation pack builder | `scripts/build_cascade_perturbation_pack.py` |
| Verifier evaluation | `scripts/evaluate_cascade_verifier_pack.py` |
| Proof artifact dir | `results/shared/cascade_verifier_proof_20260519/` |
| Tests | `tests/test_cascade_verifier.py` |

### 历史最强单模型（如被问到再提，不是 demo 主线）

**OA-ARM Skeleton Inpainter**（100-target benchmark）：plan rate 99%，GT@5 75%，
4.0× lift over random，0.81 s/target。
入口：`python -m cascade_planner.cascadeboard.skeleton_inpainter predict --target "<SMILES>" --n-steps 3 --k 5`。

---

## 二、当前战略方向（一页）

1. **退路：ranker / LambdaRank / adjacent-pair scorer 全线 freeze。** 详见
   [GUARDED_CCTS_DECISION_2026-05-19.md](GUARDED_CCTS_DECISION_2026-05-19.md)。
   学不动的根因不是模型架构，而是 **真实 cascade 正样本 ≈ 3K，无法支撑 cascade-level objective**。
2. **进路：Verifier-first cascade 数据飞轮。**
   - 规则 verifier 用化学/酶学约束，**无限制造负样本**（已实现，30K 扰动跑通）；
   - learned verifier 在 rule-derived perturbations 上训练，给出可解释失败原因；
   - verifier 输出 → DPO preference pairs（29K 已就绪）→ 微调 ChemEnzy；
   - 真实 3K cascade **仅用于最终 DPO + 外部 holdout**，禁止进 verifier 训练集；
   - verifier 自举正样本 **明确禁止**——避免自我强化漂移。
3. **目标架构：condition / stage / cofactor 进入 search state**，详见
   [CASCADE_TARGET_ARCHITECTURE.md](CASCADE_TARGET_ARCHITECTURE.md)。
   骨架已在 `cascade_planner/cascade_search/state.py`。
4. **下一步**：(a) 把 DPO 接到 ChemEnzy 的 OpenNMT/template 通路；(b) 在 30 条真实
   文献 chemo-enzymatic cascade benchmark 上验证；(c) cascade verifier 接入 search
   作 value/guard。

---

## 三、当前活跃文档（只剩这 5 份 + 本 README）

| 文档 | 用途 |
| --- | --- |
| [CURRENT_STATE_2026-05-19.md](CURRENT_STATE_2026-05-19.md) | 仓库主线 + 模型结论快照 |
| [CASCADE_VERIFIER_PROOF_REPORT_2026-05-19.md](CASCADE_VERIFIER_PROOF_REPORT_2026-05-19.md) | 新方向的 proof 报告 |
| [CASCADE_TARGET_ARCHITECTURE.md](CASCADE_TARGET_ARCHITECTURE.md) | Cascade 目标架构（ConditionEnvelope / StagePartition / CascadeLedger） |
| [GUARDED_CCTS_DECISION_2026-05-19.md](GUARDED_CCTS_DECISION_2026-05-19.md) | "ranker 路线 freeze" 的决策记录（ADR） |
| [CODEBASE_STATUS_2026-05-19.md](CODEBASE_STATUS_2026-05-19.md) | runtime 主线 vs 研究残骸 vs 已归档 的边界 |

历史文档（22 份）已移至 [archive/2026-05/](archive/2026-05/)。其中包括：
`AUTOPLANNER_CASCADE_PROGRESS_REPORT_2026-05-19`、`CHEMENZY_BASELINE`、
`CLEANUP_PROGRESS_AUDIT_*`、`CURRENT_STATE_2026-05-09/12`、`DELIVERY_INDEX_*`、
所有 `MODEL_STRENGTHENING_*` / `NEXT_*` / `PHASE1_*` / `PHASE2_*` / `POSTMORTEM_*`。

## Expert Feedback

- [expert_feedback/prof_2026-05-09.md](expert_feedback/prof_2026-05-09.md)

## Documentation Rules

- Keep current claims tied to benchmark artifacts.
- Separate raw dataset size, trace-candidate size, traced target count, and
  action-value supervision count. Do not infer dataset size from a small trace
  pack.
- Mark all unpassed architecture claims as `未收束`.
- Do not describe the system as international SOTA without public benchmark
  comparisons, locked validation, calibration, and blind-test evidence.
- Use `results/v2/` for concise curated reports; keep heavy JSON/checkpoints in
  ignored local artifact folders.
- Treat `AI_OS_AutoResearch/` as a separate integration checkout, not as source
  under this AutoPlanner repository.

## Superseded Drafts

The following duplicate phase-I drafts were removed during the closeout and
should not be restored:

- `PHASE1_RESEARCH_CLOSURE_2026-05-14.md`
- `PHASE1_CLEANUP_MANIFEST_2026-05-14.md`

## Archived On 2026-05-09

- `archive/obsolete_2026-05-09/docs/ARCHITECTURE_CURRENT_VS_TARGET.md`
- `archive/obsolete_2026-05-09/docs/KPI.md`
- `archive/obsolete_2026-05-09/docs/CHEMENZY_BASELINE_EVALUATION_2026-05-09.md`
