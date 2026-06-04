# Bridge Pack v0 与原数据规划对比

## 当前实际产物

数据包位置：`data/bridge_pack_v0`

| 项目 | 当前数量 |
|---|---:|
| chemical source rows | 109,183 |
| chemical unique products | 89,387 |
| enzyme source rows, raw combined | 766,013 |
| enzyme unique substrate/product molecules | 82,454 |
| common/cofactor blacklist | 308 |
| exact bridge all | 38,133 |
| exact bridge filtered | 37,939 |
| strict exact bridge positives | 37,209 |
| audit-only exact bridge | 924 |
| hard negatives | 0 |
| similarity bridge | 0 |
| reaction-center filtered bridge | 0 |
| stereochemistry-filtered bridge | 0 |

Raw enzyme source rows include EnzymeMap 349,458 rows, ECREACT 62,222 rows, and enzymatic_retro train/val 354,333 rows. These are not yet deduplicated into a formal `enzyme_reaction_pool.parquet`.

## 对照原 P0/P1 规划

| 数据项 | P0 目标 | P1 目标 | 当前 | P0 差距 | P1 差距 | 判断 |
|---|---:|---:|---:|---:|---:|---|
| cleaned chemical reactions/products | 300K-500K reactions | 0.8M-1.5M reactions | 109,183 chemical rows / 89,387 unique products | 少约 190K-391K rows | 少约 691K-1.39M rows | 不足；只能做 v0 |
| cleaned enzyme reactions | 20K-50K unique reactions | 50K-100K unique reactions | raw rows 766,013；unique reactions 未统一 dedupe | 原始覆盖足够 | 原始覆盖可能足够 | 需要 reaction pool 标准化 |
| enzyme substrate/product molecules | 10K-30K unique molecules | 50K+ unique molecules | 82,454 | 已超过 | 已超过 | 达标 |
| exact bridge links | 1K-10K | 10K-50K | 37,209 strict / 37,939 filtered | 已超过 | 达到 P1 中段 | 达标 |
| high-confidence similarity bridge | 10K-30K | 50K-200K | 0 | 少 10K-30K | 少 50K-200K | 未开始 |
| verifier positives | 50K-200K | 200K-1M | 37,209 strict exact positives | 少 12,791 到 P0 下限 | 少 162,791 到 P1 下限 | 不足，需要 similarity/triads 扩增 |
| hard negatives | 0.5M-2M raw pool | 5M-20M raw pool | 0 | 少 0.5M-2M | 少 5M-20M | 未开始 |
| route/search states | 50K-200K | 0.5M-5M | 旧搜索日志分散，未产品化 | 未产品化 | 未产品化 | 待做 |
| sequence-linked enzyme triads | P0 可选 | 100K-500K | 本地 sequence TSV 1,116 | 不足 | 少约 99K+ | 明显不足 |
| 3D/structure-supported cases | P0 可选 | top bridge 子集 | 0 | 未开始 | 未开始 | 暂不优先 |

## 关键解释

目前最大的进展是 exact bridge 已经从 0 变成 37,209 条保守正样本，已经超过 P0，并达到 P1 的 exact bridge 规模要求。

但这还不能支撑完整 verifier，因为 verifier 训练需要的不只是正样本，还需要 hard negatives。现在 hard negative 仍为 0，因此还不能宣称 enzyme feasibility verifier 数据集已经完成。

当前数据缺口按优先级排序：

1. `hard_negative_pool.parquet`：至少 0.5M 条 P0 raw negatives。
2. `similarity_bridge_filtered.parquet`：至少 10K-30K 条 P0 high-confidence similarity bridge。
3. `enzyme_reaction_pool.parquet`：把 EnzymeMap/ECREACT/enzymatic_retro/Rhea/ReactZyme 统一去重成 reaction-level 表。
4. `reaction_center_filter`：给 exact/similarity bridge 加反应中心一致性字段。
5. `stereo_filter`：标注立体冲突，不直接删除。
6. `verifier_train/valid/test.jsonl`：正负样本分层、source split、EC split。
7. UniProt reviewed enzyme sequence 扩展：当前 1,116 条太小，P1 至少需要 100K 级 sequence-linked triads 或 reviewed enzyme candidates。

## 当前能立即训练什么

可以训练：

- exact bridge retriever baseline
- bridge presence classifier smoke
- common/cofactor artifact rejector
- EC hard-negative 原型，如果先从现有 enzyme pool 采样

不建议马上训练：

- 完整 enzyme feasibility verifier
- sequence-aware enzyme verifier
- route-level policy
- DPO/RL 类 route generator

原因是 hard negative 和 similarity/reaction-center 证据还没形成。

## 下一步数据目标

最小 P0 补齐目标：

| 数据 | 下一步目标 |
|---|---:|
| similarity bridge filtered | 10K-30K |
| hard negative raw pool | 0.5M |
| verifier positives | 从 37,209 扩到 50K+ |
| verifier splits | train/valid/test 三套 |
| enzyme reaction pool | 统一 dedupe 后输出 |
| reaction-center flags | 覆盖 exact bridge 和 similarity bridge |

完成这些以后，才进入 verifier v0 训练。
