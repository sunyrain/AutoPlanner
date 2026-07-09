# Bridge Pack v0 P0 补齐结果

## 本轮补齐结论

本轮已经把 `data/bridge_pack_v0` 从“exact bridge 正样本底座”推进到 P0 可训练数据包。

核心新增：

| 数据产品 | 当前数量 |
|---|---:|
| enzyme_reaction_pool | 109,857 unique reactions |
| enzyme_sequence_pool | 190,750 UniProt sequences |
| exact strict positives | 37,209 |
| similarity bridge positives | 30,000 |
| total verifier positives | 67,209 |
| hard negatives | 500,000 |
| verifier total rows | 567,209 |
| verifier train | 456,579 rows |
| verifier valid | 56,839 rows |
| verifier test | 53,791 rows |

## 当前文件

| 文件 | 说明 |
|---|---|
| `chemical_product_pool.parquet` | 化学 product pool，89,387 unique products |
| `enzyme_substrate_product_pool.parquet` | 酶底物/产物 pool，82,454 unique molecules |
| `enzyme_reaction_pool.parquet` | 反应级酶反应池，109,857 unique reactions |
| `enzyme_sequence_pool.parquet` | UniProt/EC/Rhea 序列池，190,750 条 |
| `exact_bridge_strict.parquet` | exact bridge 保守正样本，37,209 条 |
| `similarity_bridge_filtered.parquet` | high-similarity non-exact bridge，30,000 条 |
| `similarity_bridge_near_negative_candidates.parquet` | 近似但未过正样本阈值的候选，18,143 条 |
| `hard_negative_pool.parquet` | hard negative pool，500,000 条 |
| `verifier_train.jsonl/parquet` | train split |
| `verifier_valid.jsonl/parquet` | valid split |
| `verifier_test.jsonl/parquet` | test split |

## 与原 P0/P1 规划对比

| 数据项 | P0 目标 | P1 目标 | 当前 | P0 状态 | P1 差距 |
|---|---:|---:|---:|---|---:|
| cleaned chemical reactions/products | 300K-500K rows | 0.8M-1.5M rows | 109,183 chemical rows / 89,387 products | 未达标 | 少约 691K-1.39M rows |
| cleaned enzyme reactions | 20K-50K unique | 50K-100K unique | 109,857 unique | 已超过 | 已超过 |
| enzyme substrate/product molecules | 10K-30K unique | 50K+ unique | 82,454 | 已超过 | 已超过 |
| exact bridge links | 1K-10K | 10K-50K | 37,209 strict | 已超过 | 已达 P1 区间 |
| high-confidence similarity bridge | 10K-30K | 50K-200K | 30,000 | 已达 P0 上限 | 少 20K 到 P1 下限 |
| verifier positives | 50K-200K | 200K-1M | 67,209 | 已达 P0 | 少 132,791 到 P1 下限 |
| hard negatives | 0.5M-2M | 5M-20M | 500,000 | 达到 P0 下限 | 少 4.5M 到 P1 下限 |
| route/search states | 50K-200K | 0.5M-5M | 未产品化 | 未达标 | 待补 |
| sequence-linked enzyme candidates | P0 可选 | 100K-500K | 190,750 | 已超过 P0 | 已达 P1 区间 |
| 3D/structure-supported cases | P0 可选 | top bridge 子集 | 0 | 未开始 | 待补 |

## Verifier 数据组成

### 正样本

| 来源 | 数量 | 权重 |
|---|---:|---:|
| exact bridge strict | 37,209 | 1.0 |
| similarity bridge filtered | 30,000 | 0.65 |
| 合计 | 67,209 | - |

### 负样本

| 类型 | 数量 |
|---|---:|
| random_easy_negative | 126,726 |
| near_size_wrong_molecule | 125,655 |
| common_or_cofactor_artifact | 125,576 |
| same_ec_wrong_molecule | 116,035 |
| near_similarity_below_positive_threshold | 6,008 |
| 合计 | 500,000 |

### Splits

| Split | Rows | Positives | Negatives |
|---|---:|---:|---:|
| train | 456,579 | 54,099 | 402,480 |
| valid | 56,839 | 6,727 | 50,112 |
| test | 53,791 | 6,383 | 47,408 |

Split 按 `chemical_inchikey` hash 切分，避免同一 chemical connector 同时出现在 train/test。

## 仍需注意的质量边界

当前已经达到 P0 规模，但仍不是最终高质量 P1 数据集。主要限制：

1. `similarity_bridge_filtered` 使用 Morgan fingerprint + 稀有 bit 倒排召回 + Tanimoto 阈值，不等价于 reaction-center 证明。
2. hard negative 是自动构造弱标签，适合训练 verifier 的拒绝能力，但仍需 label smoothing 或 sample weighting。
3. chemical product pool 仍偏小，只有 109K source rows，需要 USPTO full/Pistachio/Reaxys 或更多内部反应补齐。
4. sequence pool 已达数量级，但还不是完整 substrate-product-enzyme triads；缺少 active-site/kinetic/substrate scope 级证据。
5. 3D/structure validator 尚未开始，只能作为 P1/P2 的 top bridge 验证层。

## 下一步

现在可以进入 verifier v0 训练：

```text
input:
  verifier_train.parquet/jsonl
  verifier_valid.parquet/jsonl
  verifier_test.parquet/jsonl

baseline:
  Morgan fingerprint pair features
  EC overlap/features
  heavy atom/formula delta
  label_weight

model:
  GBDT / MLP v0

primary metrics:
  ROC-AUC / PR-AUC
  hard negative rejection by type
  precision@k for bridge retrieval
  exact vs similarity positive calibration
```

不建议马上做 route-level DPO 或大模型微调；应先证明 verifier 能拒绝 same-EC wrong molecule、common/cofactor artifact、near-size wrong molecule。
