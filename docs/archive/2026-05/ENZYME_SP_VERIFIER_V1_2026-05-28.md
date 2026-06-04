# Enzyme-Substrate-Product Verifier v1 阶段结果

生成时间：2026-05-28

## 目标

本阶段完成两个交付：

1. 构建 `enzyme-substrate-product-EC` 三元组 verifier v1 数据集。
2. 训练一个轻量 verifier v1 小模型，用于判断酶步 `substrate -> product + EC` 是否可信。

这不是原来的 bridge verifier v0。v0 判断“化学产物是否接近酶底物/产物空间”；v1 判断一个具体酶催化转化三元组是否合理。

## 数据集

输出目录：

`data/enzyme_sp_verifier_v1/`

核心文件：

- `train.parquet`
- `valid.parquet`
- `test.parquet`
- `manifest.json`
- `dataset_report.md`

数据来源：

- 正样本来自 `data/bridge_pack_v0/enzyme_reaction_pool.parquet`
- 负样本由规则生成，包括 same-EC wrong product、same-EC wrong substrate、random wrong product、common/cofactor artifact

数据规模：

| Split | Rows | Positives | Negatives |
|---|---:|---:|---:|
| train | 320,901 | 80,073 | 240,828 |
| valid | 39,963 | 10,020 | 29,943 |
| test | 40,084 | 10,144 | 29,940 |
| total | 400,948 | 100,237 | 300,711 |

负样本类型：

| Label type | Rows |
|---|---:|
| same_ec_wrong_product | 75,693 |
| same_ec_wrong_substrate | 75,159 |
| common_or_cofactor_wrong_product | 75,574 |
| random_wrong_product | 74,285 |

## 模型

输出目录：

`results/shared/enzyme_sp_verifier_v1_20260528/`

核心文件：

- `enzyme_sp_verifier_v1_lgbm.joblib`
- `enzyme_sp_verifier_v1_report.json`
- `enzyme_sp_verifier_v1_report.md`
- `feature_schema.json`
- `valid_scores.parquet`
- `test_scores.parquet`

模型形式：

- LightGBM binary classifier
- Morgan fingerprint：substrate、product、shared bits
- 数值特征：heavy atom delta、component count、ring/hetero delta、substrate-product Tanimoto、EC count
- EC 特征：EC1 one-hot

## 指标

验证集选择阈值：

`0.363312`

选择逻辑：

在验证集上满足 target precision 0.95，并最大化 recall。

测试集结果：

| Metric | Value |
|---|---:|
| ROC-AUC | 0.9980 |
| PR-AUC | 0.9940 |
| Precision | 0.9443 |
| Recall | 0.9858 |
| F1 | 0.9646 |
| Accuracy | 0.9817 |
| TP | 10,000 |
| FP | 590 |
| FN | 144 |
| TN | 29,350 |

测试集按负样本类型的拒绝率：

| Label type | Rejection rate |
|---|---:|
| common_or_cofactor_wrong_product | 0.9988 |
| random_wrong_product | 0.9930 |
| same_ec_wrong_product | 0.9653 |
| same_ec_wrong_substrate | 0.9642 |

## 当前结论

step1/2 已经完成：我们现在有了第一个可训练、可复现、可审计的酶步三元组 verifier v1。

它已经能有效拒绝明显错误和 same-EC 难负样本，说明数据构造方向是成立的。当前最重要的价值不是“证明酶步都对”，而是给后续 gated search 提供一个低成本过滤器，减少错误酶 proposal 污染多步搜索。

## 限制

该模型仍然是 weak-label verifier，不是专家真值模型。

主要限制：

- 正样本来自公开酶反应池，偏代谢/数据库反应，不完全等同合成生物催化。
- 负样本是规则构造，测试指标会高于真实专家评审场景。
- 当前特征没有显式反应中心、原子映射、酶序列、结构或 active-site 信息。
- EC 只用了浅层 EC1 特征，后续需要加入 EC2/EC3、反应中心、substrate scope。
- 可逆反应和多底物辅因子反应仍可能引入标签噪声。

## 验证命令

```bash
python -m py_compile scripts/build_enzyme_sp_verifier_v1_pack.py scripts/train_enzyme_sp_verifier_v1.py tests/test_enzyme_sp_verifier_v1.py
pytest -q tests/test_enzyme_sp_verifier_v1.py
python scripts/build_enzyme_sp_verifier_v1_pack.py --input-pack-dir data/bridge_pack_v0 --output-dir data/enzyme_sp_verifier_v1 --negatives-per-positive 3 --max-negatives 330000
python scripts/train_enzyme_sp_verifier_v1.py --data-dir data/enzyme_sp_verifier_v1 --output-dir results/shared/enzyme_sp_verifier_v1_20260528
```

## 下一步

下一阶段应把 verifier v1 接入 gated search，但不要全局强行 rerank 所有路线。推荐只在 enzyme proposal 或 bridge-supported step 上调用：

1. 搜索节点触发 enzyme proposal。
2. 生成 candidate enzyme step。
3. verifier v1 过滤低可信 `substrate-product-EC`。
4. 只把高置信酶步放回 route search。

评估组应包含：

- baseline
- baseline + enzyme proposal 全开
- baseline + bridge gate
- baseline + bridge gate + verifier v1

关键指标不是单纯 solved rate，而是错误酶步率、useful enzyme step 数、搜索节点数、top-k route plausibility。
