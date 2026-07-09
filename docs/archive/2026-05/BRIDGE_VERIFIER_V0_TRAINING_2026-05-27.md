# Bridge Verifier v0 训练结果

## 训练状态

已完成 `bridge verifier v0` 训练。

| 项 | 值 |
|---|---:|
| train rows | 456,579 |
| valid rows | 56,839 |
| test rows | 53,791 |
| feature count | 6,165 |
| model | LightGBM binary classifier |
| best iteration | 429 |
| output dir | `results/shared/bridge_verifier_v0_20260527` |

## 训练数据

输入数据来自 `data/bridge_pack_v0/verifier_train/valid/test.parquet`。

正样本：

- exact strict bridge：37,209
- similarity bridge：30,000
- total positives：67,209

负样本：

- hard negatives：500,000
- 类型包括 same-EC wrong molecule、near-size wrong molecule、common/cofactor artifact、near-similarity below threshold、random easy negative。

## 总体指标

| Split | ROC-AUC | PR-AUC | Precision@0.5 | Recall@0.5 | F1@0.5 | Tanimoto-only PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| valid | 0.9998 | 0.9986 | 0.9821 | 0.9927 | 0.9874 | 0.8303 |
| test | 0.9998 | 0.9984 | 0.9809 | 0.9908 | 0.9858 | 0.8277 |

关键解释：

单独使用 Tanimoto 的 recall 很高，但 precision 只有约 0.36；verifier v0 把 test precision 提到 0.981，同时保持 recall 0.991。这说明模型不只是学会“相似就接受”，也学到了一部分拒绝 hard negative 的规则。

## Test 集 hard negative 拒绝率

| Negative type | Rows | Mean score | Reject <0.5 | Reject <0.2 |
|---|---:|---:|---:|---:|
| common_or_cofactor_artifact | 11,939 | 0.0000 | 1.0000 | 1.0000 |
| near_similarity_below_positive_threshold | 526 | 0.0087 | 0.9905 | 0.9848 |
| near_size_wrong_molecule | 11,910 | 0.0015 | 0.9982 | 0.9981 |
| random_easy_negative | 12,026 | 0.0003 | 0.9996 | 0.9996 |
| same_ec_wrong_molecule | 11,007 | 0.0081 | 0.9917 | 0.9901 |

## Test 集正样本召回

| Positive type | Rows | Mean score | Recall >=0.5 | Recall >=0.2 |
|---|---:|---:|---:|---:|
| tier1_strict_exact_substrate_bridge | 1,064 | 0.9999 | 1.0000 | 1.0000 |
| tier2_strict_exact_product_bridge | 2,624 | 0.9999 | 1.0000 | 1.0000 |
| tier3_high_similarity_nonexact_bridge | 2,695 | 0.9318 | 0.9781 | 0.9840 |

## 排序指标

| Split | Groups | MRR | R@1 | R@3 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|
| valid | 2,789 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| test | 2,747 | 0.9996 | 0.9993 | 1.0000 | 1.0000 | 1.0000 |

## 产物

| Artifact | Path |
|---|---|
| model | `results/shared/bridge_verifier_v0_20260527/bridge_verifier_v0_lgbm.joblib` |
| report JSON | `results/shared/bridge_verifier_v0_20260527/bridge_verifier_v0_report.json` |
| report MD | `results/shared/bridge_verifier_v0_20260527/bridge_verifier_v0_report.md` |
| feature names | `results/shared/bridge_verifier_v0_20260527/feature_names.json` |
| valid scores | `results/shared/bridge_verifier_v0_20260527/valid_scores.parquet` |
| test scores | `results/shared/bridge_verifier_v0_20260527/test_scores.parquet` |
| stress audit | `results/shared/bridge_verifier_v0_20260527/bridge_verifier_v0_stress_audit.md` |
| scorer smoke input | `results/shared/bridge_verifier_v0_20260527/scoring_smoke_input.jsonl` |
| scorer smoke output | `results/shared/bridge_verifier_v0_20260527/scoring_smoke_output.jsonl` |

## Stress / Leakage Audit

已补充 stress audit，结果为 `pass=True`。

关键 gate：

| Gate | Value | Pass |
|---|---:|---:|
| test PR-AUC >= 0.95 | 0.9984 | true |
| test precision >= 0.95 | 0.9809 | true |
| test recall >= 0.95 | 0.9908 | true |
| non-exact PR-AUC >= 0.95 | 0.9892 | true |
| similarity-boundary PR-AUC >= 0.95 | 0.9999 | true |
| tanimoto >= 0.80 precision >= 0.95 | 0.9818 | true |
| chemical train/test overlap = 0 | 0 | true |

Stress subsets：

| Subset | Rows | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| all test | 53,791 | 0.9984 | 0.9809 | 0.9908 | 0.9858 |
| non-exact only | 50,103 | 0.9892 | 0.9554 | 0.9781 | 0.9666 |
| similarity boundary | 3,221 | 0.9999 | 0.9981 | 0.9781 | 0.9880 |
| tanimoto >= 0.80 | 11,421 | 0.9985 | 0.9818 | 0.9908 | 0.9863 |
| novel chemical scaffold | 3,052 | 0.9989 | 0.9834 | 0.9807 | 0.9820 |

Leakage notes：

- chemical connector train/test overlap 为 0，说明当前主 split 没有 chemical connector 泄漏。
- enzyme molecule train/test overlap 为 20,956，这是当前 weak-label 数据的残余风险；v0 暂接受，v1 需要 enzyme/scaffold/source holdout。
- test 中仅有 2 行的 EC 对 train 完全 novel，说明当前数据还不能证明新 EC 泛化。

## 推荐部署阈值

从 valid set 选择阈值，并在 test set 验证：

| Target valid precision | Threshold | Test precision | Test recall | Test F1 |
|---:|---:|---:|---:|---:|
| 0.98 | 0.3318 | 0.9793 | 0.9926 | 0.9859 |
| 0.99 | 0.8410 | 0.9893 | 0.9563 | 0.9725 |
| 0.995 | 0.9331 | 0.9965 | 0.8899 | 0.9402 |

建议：

- route-search gate 默认用 `0.8410`，偏高精度。
- 若用于离线候选召回/人工复核，可用 `0.3318` 提高 recall。
- 若用于自动接受强过滤，可用 `0.9331`。

## Scoring 接口

已提供独立打分脚本：

```bash
python scripts/score_bridge_verifier_v0.py \
  --input candidates.jsonl \
  --output scored_candidates.jsonl \
  --model results/shared/bridge_verifier_v0_20260527/bridge_verifier_v0_lgbm.joblib \
  --threshold 0.8409896871324669
```

输入行至少需要：

```json
{"chemical_smiles": "...", "enzyme_smiles": "..."}
```

可选字段：

```text
chemical_inchikey
enzyme_inchikey
bridge_direction
enzyme_ec_sample_json / enzyme_ec / ec
```

scoring smoke 已通过：

- input rows: 6
- positive rows passed: 3/3
- negative rows passed: 0/3

## 边界和下一步

这个结果证明了 P0 weak-label 体系可以训练出一个能拒绝 hard negatives 的 verifier v0。但它还不是最终生产级 enzyme feasibility model。

下一步建议：

1. 做 leakage/stress audit：按 scaffold、EC、source 拆分，而不仅是 chemical connector split。
2. 增加 reaction-center features，减少模型对 Tanimoto 和 same-InChIKey 的依赖。
3. 接入 route search gate：只在 enzyme bridge/retriever 候选进入搜索前调用 verifier。
4. 训练 v1 时引入 sequence-linked triads 和更强 hard negatives。
