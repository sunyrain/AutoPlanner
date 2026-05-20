# Cascade Verifier Proof Report

生成时间：2026-05-19

## 结论

本轮把“先做 verifier schema + 扰动生成 spec”的方案落成了可运行的 proof：

- 新增规则 verifier schema：`cascade_verifier.v1`
- 新增 perturbation pack schema：`cascade_perturbation_pack.v1`
- 新增规则 verifier baseline，覆盖物料守恒、产物不匹配、温度冲突、pH 冲突、溶剂冲突、酶毒化、cofactor 缺口、路线顺序错误
- 新增 perturbation pack 生成脚本和评估脚本
- 在当前路线 artifact 上完成一次小规模评估

这一步验证的是：我们可以把“无专家标签”的问题转成可控的 verifier 规则扰动任务。它不是最终 ChemEnzy 微调，也不是专家可行性判定。

## 关键产物

| 类型 | 路径 |
| --- | --- |
| Verifier schema | `cascade_planner/cascade_verifier/schema.py` |
| 规则 verifier | `cascade_planner/cascade_verifier/rules.py` |
| Perturbation pack 生成 | `scripts/build_cascade_perturbation_pack.py` |
| Verifier pack 评估 | `scripts/evaluate_cascade_verifier_pack.py` |
| 单元/端到端测试 | `tests/test_cascade_verifier.py` |
| Proof pack | `results/shared/cascade_verifier_proof_20260519/perturbation_pack_top12.json` |
| Proof 评估 JSON | `results/shared/cascade_verifier_proof_20260519/verifier_eval_top12.json` |
| Proof 评估 Markdown | `results/shared/cascade_verifier_proof_20260519/verifier_eval_top12.md` |

## 当前评估结果

### Top12 proof

数据来源：

`results/v2/ui_chem_enzy_plan_20260519_032819_3764f7_reaudited_current.json`

构造方式：

- 取前 12 条非 reject artifact 路线
- 每条路线保留 1 个 seed positive
- 每条路线生成 6 类规则负样本
- 默认给没有 stage partition 的 seed 路线补 `stepwise` stage partition
- 温度/pH 冲突负样本会被故意压回同一 stage，以测试一锅兼容性冲突

评估读数：

| 指标 | 数值 |
| --- | ---: |
| examples | 84 |
| seed positives | 12 |
| rule negatives | 72 |
| label accuracy | 1.0000 |
| expected-reason coverage | 1.0000 |
| expected reason hits | 72 / 72 |

Verifier 输出的失败原因计数：

| reason | count |
| --- | ---: |
| temperature_conflict | 336 |
| route_order_mismatch | 24 |
| atom_balance_violation | 12 |
| cofactor_ledger_gap | 12 |
| enzyme_toxicity | 12 |
| ph_conflict | 12 |

### 当前路线池扩展评估

同一输入 artifact 上，扩大到非 `reject_artifact` 路线池：

| 指标 | 数值 |
| --- | ---: |
| routes used | 592 |
| examples | 4144 |
| seed positives | 592 |
| rule negatives | 3552 |
| label accuracy | 1.0000 |
| expected-reason coverage | 1.0000 |
| expected reason hits | 3552 / 3552 |

扩展评估 artifacts：

- `results/shared/cascade_verifier_proof_20260519/perturbation_pack_all640.json`
- `results/shared/cascade_verifier_proof_20260519/verifier_eval_all640.json`
- `results/shared/cascade_verifier_proof_20260519/verifier_eval_all640.md`

扩展评估失败原因计数：

| reason | count |
| --- | ---: |
| temperature_conflict | 13052 |
| route_order_mismatch | 1184 |
| atom_balance_violation | 592 |
| cofactor_ledger_gap | 592 |
| enzyme_toxicity | 592 |
| ph_conflict | 592 |

### 30K structured v4 评估

来源：

`results/shared/cascadebench_strict_20260516/splits_structured_v1/v4_trace_all_structured.json`

构造方式：

- 使用 2230 条 structured v4 cascade row
- 其中 1477 条 seed 通过 verifier 预检
- 每条 seed 生成 21 类扰动
- 默认补 `stepwise` stage partition

评估读数：

| 指标 | 数值 |
| --- | ---: |
| routes used | 1477 |
| skipped seed verifier fail | 753 |
| examples | 30556 |
| seed positives | 1477 |
| rule negatives | 29079 |
| label accuracy | 0.9964 |
| expected-reason coverage | 0.9962 |
| expected reason hits | 28968 / 29079 |

扩展评估 artifacts：

- `results/shared/cascade_verifier_proof_20260519/perturbation_pack_v4_structured_30k.json`
- `results/shared/cascade_verifier_proof_20260519/verifier_eval_v4_structured_30k.json`
- `results/shared/cascade_verifier_proof_20260519/verifier_eval_v4_structured_30k.md`

关键结论：

- 这已经是接近 30K 级别的 verifier-first 训练/评估 pack。
- 仍有 111 个负样本被放过，主要集中在 route_order 类扰动，说明“顺序错误”仍然需要更强的结构化约束或更严格的 route continuity 规则。
- atom-balance、温度、pH、solvent、cofactor、enzyme toxicity 这些负样本覆盖已经稳定。

解释：

- top12 和扩展路线池中的规则负样本全部被打回，且预期失败原因全部覆盖。
- seed positive 在 stepwise stage partition 下全部通过。
- 温度冲突计数很高，说明当前路线条件跨度大；若误把全路线当一锅 cascade，会产生大量 condition conflict。因此 stage partition 不是展示细节，而是 verifier/search 的必要状态字段。

## 重要边界

本轮结果不能解释为“模型已经学会 cascade”：

- 目前 verifier 是规则 baseline，不是神经 verifier。
- 当前评估是规则扰动恢复，不是专家路线可行性评估。
- seed positive 来自当前路线池，只能作为工程 smoke，不等同于真实文献级正样本。
- 真实训练时必须按来源分层：`real_literature_cascade`、`synthetic_cascade`、`metabolic_pathway`、`planner_seed_route` 不能混成同一种正样本。

## Learned Verifier Baseline

在 30K structured v4 pack 上训练了一个轻量 learned verifier：

| Artifact | 路径 |
| --- | --- |
| model | `results/shared/cascade_verifier_proof_20260519/learned_verifier_v4_30k.joblib` |
| report JSON | `results/shared/cascade_verifier_proof_20260519/learned_verifier_v4_30k_report.json` |
| report Markdown | `results/shared/cascade_verifier_proof_20260519/learned_verifier_v4_30k_report.md` |

训练方式：

- `DictVectorizer + LogisticRegression`
- 输入不包含 `perturbation_type`，避免直接泄漏标签
- 按 v4 原始 `split` 字段切分：train / val / test
- 同时训练 binary feasibility head 和 multi-label reason heads

评估结果：

| 指标 | 数值 |
| --- | ---: |
| examples | 30556 |
| train / val / test | 21350 / 4506 / 4700 |
| feature dim | 51 |
| feasibility test accuracy | 0.9094 |
| reason micro F1 | 0.9689 |
| reason macro F1 | 0.9653 |

Reason head：

| reason | precision | recall | F1 |
| --- | ---: | ---: | ---: |
| atom_balance_violation | 0.9590 | 0.9956 | 0.9769 |
| temperature_conflict | 0.8979 | 0.9679 | 0.9316 |
| ph_conflict | 0.9288 | 0.9969 | 0.9617 |
| solvent_conflict | 0.9436 | 0.9977 | 0.9699 |
| enzyme_toxicity | 1.0000 | 1.0000 | 1.0000 |
| cofactor_ledger_gap | 1.0000 | 1.0000 | 1.0000 |
| route_order_mismatch | 0.8723 | 0.9673 | 0.9174 |

解释：

- reason classifier 已经能从规则扰动数据中学到稳定的失败原因信号。
- binary feasibility 的 test accuracy 可用，但 feasible precision 只有约 0.34，说明它还不适合作为最终硬 gate；目前更适合作为 search value / rerank 辅助信号。
- route_order 仍然是最弱 reason，需要更强的结构化 route graph/ledger 特征。

## Search 接入

已新增 search-facing adapter：

- `VerifierAugmentedCascadeValueModel`
- 实现位置：`cascade_planner/cascade_search/value.py`
- 导出位置：`cascade_planner/cascade_search/__init__.py`
- 测试：`tests/test_cascade_search_contract.py::test_verifier_augmented_value_model_exposes_report`

作用：

- 包装现有 `HeuristicCascadeValueModel`
- 将 rule verifier 的 `score` 软融合进 state value
- 将完整 `verifier_report` 写入 value metadata
- 对 `cofactor_ledger_gap`、`enzyme_toxicity` 等失败原因同步压低相应概率

这一步让 search 可以先消费 rule verifier signal；下一节的 learned runtime adapter 则负责加载 joblib 模型。当前仍没有替换 ChemEnzy proposal。

## Learned Runtime 接入

已新增 joblib-backed runtime adapter：

- `LoadedLearnedVerifierValueModel`
- 实现位置：`cascade_planner/cascade_search/value.py`
- 导出位置：`cascade_planner/cascade_search/__init__.py`
- 测试覆盖：`tests/test_cascade_verifier.py::test_learned_verifier_training_smoke`

作用：

- 直接加载 `learned_verifier_v4_30k.joblib`
- 用训练时相同的特征函数做 runtime 推理
- 将 learned feasible probability 和 reason probabilities 写进 search value metadata
- 仍保留 heuristic/base value 作为底座，避免 learned model 单独接管 search

## Verifier-Derived Preference Pack

已将 verifier perturbation pack 转成 DPO/reranking 可消费的成对偏好数据：

| Artifact | 路径 |
| --- | --- |
| preference JSONL | `results/shared/cascade_verifier_proof_20260519/verifier_dpo_pairs_v4_30k.jsonl` |
| summary JSON | `results/shared/cascade_verifier_proof_20260519/verifier_dpo_pairs_v4_30k_summary.json` |

构造方式：

- chosen：clean seed cascade
- rejected：同一 seed 的 rule-derived perturbation negative
- preference source：`verifier_perturbation`
- 这些不是专家偏好标签，而是 verifier-derived preferences

结果：

| 指标 | 数值 |
| --- | ---: |
| groups | 1477 |
| preference pairs | 29079 |
| source examples | 30556 |

主要 rejected reason：

| reason | count |
| --- | ---: |
| cofactor_ledger_gap | 5908 |
| atom_balance_violation | 4431 |
| enzyme_toxicity | 4431 |
| ph_conflict | 4230 |
| temperature_conflict | 4230 |
| route_order_mismatch | 3029 |
| solvent_conflict | 2820 |

## ChemEnzy Adapter/DPO Readiness

已新增 ChemEnzy 微调/DPO readiness 检查，避免把“有 preference pack”误报为“已经完成 DPO/LoRA 微调”。

| Artifact | 路径 |
| --- | --- |
| readiness JSON | `results/shared/cascade_verifier_proof_20260519/chem_enzy_dpo_readiness.json` |
| readiness Markdown | `results/shared/cascade_verifier_proof_20260519/chem_enzy_dpo_readiness.md` |
| readiness 脚本 | `scripts/check_chem_enzy_dpo_readiness.py` |
| 测试 | `tests/test_chem_enzy_dpo_readiness.py` |

真实 vendor 检查结果：

| 指标 | 结果 |
| --- | --- |
| overall_status | `ready_for_supervised_adapter_manifest_not_direct_dpo` |
| configured model count | 7 |
| configured families | `graphfp_models`, `onmt_models`, `template_relevance` |
| preference pairs | 29079 |
| preference groups | 1477 |
| supervised vendor training ready | true |
| direct DPO ready | false |
| LoRA ready | false |

解释：

- `onmt_models.bionav_one_step` 有 OpenNMT checkpoint 和 `preprocess.py` / `train.py`，可以作为 supervised continue-train 的第一候选。
- `graphfp_models.USPTO-full_remapped` 有本地模板分类器 checkpoint 和训练脚本，可以做 template classifier retrain，但不是 cascade-conditioned DPO。
- `template_relevance.*` 在当前 vendor 中是外部服务配置，不能从本 repo 直接训练。
- vendor tree 没有检测到 DPO loss/trainer，也没有 LoRA/PEFT adapter。因此目前不能宣称 ChemEnzy DPO/LoRA 已完成。

当前可被严谨表述为：

> verifier preference 数据已经准备好；ChemEnzy 具备 supervised 训练入口；直接 DPO/LoRA 还需要新增训练目标和 adapter 代码。

## ChemEnzy OpenNMT Cascade Corpus

已将 verifier 预检通过的 v4 seed cascade 转成 ChemEnzy/OpenNMT 可消费的 supervised 语料：

| Artifact | 路径 |
| --- | --- |
| corpus builder | `scripts/build_chem_enzy_cascade_onmt_corpus.py` |
| corpus test | `tests/test_build_chem_enzy_cascade_onmt_corpus.py` |
| corpus manifest JSON | `results/shared/cascade_verifier_proof_20260519/chem_enzy_onmt_corpus_v4_30k/manifest.json` |
| corpus manifest Markdown | `results/shared/cascade_verifier_proof_20260519/chem_enzy_onmt_corpus_v4_30k/manifest.md` |
| corpus output dir | `results/shared/cascade_verifier_proof_20260519/chem_enzy_onmt_corpus_v4_30k/` |

构造方式：

- 输入：`perturbation_pack_v4_structured_30k.json`
- 只使用 `label=1` 的 seed cascade，不把扰动负样本当 supervised 正例。
- 每个 step 转成 one-step retrosynthesis pair：`product -> reactants`。
- 输出两套语料：
  - `plain`：只含 product SMILES，最接近现有 ChemEnzy ONMT char-tokenized checkpoint。
  - `context`：加入 stage、step index、温度桶、pH 桶、溶剂、EC 前缀、target/product token，是真正 cascade-conditioned 的 supervised 目标。

语料规模：

| mode | train | valid | test | total |
| --- | ---: | ---: | ---: | ---: |
| context | 1980 | 433 | 437 | 2850 |
| plain | 1588 | 414 | 421 | 2423 |

重要边界：

- `plain` 可以作为最保守的 ChemEnzy ONMT continue-train 输入，但它不是 cascade-aware。
- `context` 才体现 cascade state，但引入了新 token；不能无条件套用旧 checkpoint vocab，需要做 vocab/model adaptation 或重新预训练。
- 本轮完成 corpus/manifest 和 `plain` smoke，没有做正式训练，也不宣称 ChemEnzy 已微调。

### OpenNMT smoke

对 `plain` corpus 已完成实际 smoke：

- `preprocess.py` 成功产出 `plain.train.0.pt`、`plain.valid.0.pt` 和 `plain.vocab.pt`
- `train.py -train_steps 1 -valid_steps 1 -gpu_ranks 0` 成功完成 1-step 训练并保存 checkpoint
- `train.py -train_from vendor/.../model_step_100000.pt` 首次失败于 optimizer state 形状不兼容
- 加上 `-reset_optim all` 后，`train_from` resume smoke 成功完成 1-step 训练并保存 checkpoint

这说明：

1. `plain` 语料链路是可运行的。
2. 现有 ChemEnzy/OpenNMT checkpoint 需要显式 reset optimizer 才能继续训练。
3. `context` 语料仍需做 vocab / model adaptation，当前还不应表述为已可直接无缝续训。

## 下一阶段建议

1. 加强 route_order 特征和扰动生成，降低当前 111 个 false-positive negatives。
2. 校准 binary feasibility head，尤其提高 feasible precision；当前更适合 value/rerank，不适合作硬 gate。
3. 对 `context` corpus 做 vocab/model adaptation 方案，验证 cascade state token 能否进入 ChemEnzy/OpenNMT 训练。
4. 在 held-out real cascade 上做 scaffold/source-paper/EC-class 分层评估，禁止随机切分造成泄漏。
5. 新增 DPO/pairwise loss wrapper 后，再进入正式 ChemEnzy adapter/DPO。

## 完成审计

| 要求 | 当前证据 | 状态 |
| --- | --- | --- |
| 设计 verifier 失败原因 schema | `VerifierFailureReason` 覆盖 8 类失败原因 | 完成 |
| 生成 30K 扰动对的脚本和数据 | `build_cascade_perturbation_pack.py`；`perturbation_pack_v4_structured_30k.json` 含 30556 examples | 完成 |
| 规则 verifier baseline | `verify_cascade_route()` 输出 feasible、score、findings、reason_counts | 完成 |
| 可复现评估 | `evaluate_cascade_verifier_pack.py` 生成 JSON/Markdown 指标 | 完成 |
| 当前 artifact 小规模结果 | top12：84 examples，accuracy 1.0，reason coverage 1.0 | 完成 |
| 当前路线池扩展结果 | 592 routes，4144 examples，accuracy 1.0，reason coverage 1.0 | 完成 |
| 30K structured v4 结果 | 1477 routes，30556 examples，accuracy 0.9964，reason coverage 0.9962 | 完成 |
| learned verifier baseline | feasibility accuracy 0.9094；reason micro/macro F1 0.9689/0.9653 | 完成 |
| verifier 接入 search value | `VerifierAugmentedCascadeValueModel` + contract test | 完成 |
| learned verifier runtime 接入 | `LoadedLearnedVerifierValueModel` + smoke test | 完成 |
| verifier-derived DPO/preference pack | 29079 pairs，summary artifact 已生成 | 完成 |
| ChemEnzy adapter/DPO readiness | `check_chem_enzy_dpo_readiness.py` 生成真实 vendor manifest；确认 supervised 入口可用、DPO/LoRA 阻塞 | 完成 readiness，DPO/LoRA 未完成 |
| ChemEnzy OpenNMT corpus | `plain` 2423 examples；`context` 2850 examples；manifest 和命令提示已生成 | 完成 corpus，已做 plain smoke |
| ChemEnzy OpenNMT plain smoke | `preprocess.py` 成功；`train.py` 1-step 成功；`train_from model_step_100000.pt -reset_optim all` 成功 | 完成 smoke，非正式训练 |
| 测试覆盖 | `tests/test_build_chem_enzy_cascade_onmt_corpus.py`、`tests/test_chem_enzy_dpo_readiness.py`、`tests/test_cascade_verifier.py` 等通过 | 完成 |
| ChemEnzy 微调 | 本轮未做；当前只完成 readiness 和 preference 输入准备 | 未开始 |
| ChemEnzy DPO/LoRA | readiness 确认 vendor 缺少 DPO loss/trainer 和 LoRA/PEFT adapter | 未完成 |
| KEGG/MetaCyc/合成 cascade 导入 | 本轮未做 | 未开始 |
| 真实 30 个文献 cascade benchmark | 本轮未做 | 未开始 |
