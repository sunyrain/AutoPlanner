# Enzyme Candidate Quality Audit v0

日期：2026-05-28

## 目的

本轮不是继续训练 verifier，而是补齐 enzyme proposal coverage 的质量证据链。核心问题是：

> 新增酶候选源能不能提供可注入搜索的候选，而不是只增加噪声。

因此新增候选级审计脚本：

- `scripts/audit_enzyme_candidate_quality_v0.py`

它在同一批目标上比较：

- `enzyme_precedent`：109,857 条酶反应 precedent 的大池检索；
- `v3_retrieval`：原有酶检索源；
- 可选 `chem_enzy_onestep`：原生 ChemEnzy one-step proposer 参考。

## 当前主流程与增强位置

真实 Web/native 路径目前仍以 ChemEnzy 原生搜索为主：

- `graphfp_models.USPTO-full_remapped`
- `onmt_models.bionav_one_step`

本轮增强不直接替换 ChemEnzy 主搜索，而是先增强 route-tree / sidecar 层的酶候选能力：

```text
bridge evidence
  -> enzyme_precedent retrieval
  -> SP-v1 substrate-product-EC verifier
  -> quality/risk tier
  -> 后续再决定是否注入主搜索
```

这样避免把未门控的酶候选直接污染多步搜索。

## 质量规则

候选被标记为 `search_ready` 必须满足：

1. 是酶源候选；
2. 不是 SP-v1 reject；
3. 有 EC evidence；
4. 主反应物/产物不是 common metabolite/cofactor artifact；
5. 有 bridge hit 或显式 EC context 触发。

特别注意：

- `aux_common_or_cofactor` 不是硬拒绝。O2、水、NAD(P)、ATP 等辅助物可以解释条件/辅助试剂引入的元素。
- `large_heavy_atom_delta_review_only` 也不是硬拒绝。大辅助试剂有时只是引入基团，不能用原子数比例简单误杀。
- 没有 bridge/EC 触发证据的高分酶候选降为 `ungated_review`，不进入 `search_ready`。

## 10 正 / 10 负审计结果

输出目录：

- `results/shared/enzyme_candidate_quality_audit_v0_20260528_10p10n_gated/`

总体：

| 指标 | 数值 |
|---|---:|
| targets | 20 |
| positives | 10 |
| negatives | 10 |
| candidate rows | 606 |
| unique candidates | 392 |
| search-ready candidates | 161 |
| strong candidates | 150 |

按 source：

| source | rows | unique | ready | strong | positive ready recall | negative ready rate | mean quality | mean SP-v1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| enzyme_precedent | 329 | 242 | 155 | 148 | 1.0000 | 0.0000 | 2.432 | 0.529 |
| v3_retrieval | 277 | 150 | 6 | 2 | 0.2000 | 0.0000 | -0.038 | 0.107 |

目标级：

| split | targets | with bridge | ready targets | strong targets | mean ready |
|---|---:|---:|---:|---:|---:|
| all | 20 | 10 | 10 | 10 | 8.05 |
| positive | 10 | 10 | 10 | 10 | 16.10 |
| negative | 10 | 0 | 0 | 0 | 0.00 |

主要风险标签：

| flag | count |
|---|---:|
| sp_v1_reject | 389 |
| low_product_similarity | 329 |
| aux_common_or_cofactor | 240 |
| no_bridge_or_ec_trigger_for_injection | 177 |
| bridge_ec_mismatch | 63 |
| main_common_or_cofactor | 27 |
| large_heavy_atom_delta_review_only | 24 |
| no_ec_evidence | 7 |

## 结论

本轮证明了一个关键点：

> 大酶反应 precedent 源确实能显著提高酶候选覆盖，但必须由 bridge/EC trigger + SP-v1 + artifact gate 共同约束。

在 10p10n 小样本上：

- `enzyme_precedent` 对 bridge-positive 目标保持 1.0 target recall；
- 对无 bridge 负例，直接 search-ready rate 被压到 0；
- 原有 `v3_retrieval` 在相同规则下仅提供少量可注入候选；
- 这说明新增 source 的价值主要是 enzyme proposal coverage，而不是单纯 rerank 原有候选。

## 相对原生 ChemEnzy 的增强

原生 ChemEnzy 主 one-step 当前仍主要解决普通化学拆解：

- `graphfp_models.USPTO-full_remapped`
- `onmt_models.bionav_one_step`

本轮增强补的是原生 ChemEnzy 较弱的部分：

1. 从酶反应 precedent 大池中检索酶催化前体；
2. 用 bridge evidence 决定什么时候触发酶候选；
3. 用 SP-v1 判断 substrate-product-EC 三元组是否可信；
4. 把 cofactor/common metabolite artifact 和大辅助试剂风险分开处理；
5. 输出候选级证据，支持后续和 ChemEnzy native 做 route-level 对比。

## 原生 ChemEnzy one-step 最小候选对照

输出目录：

- `results/shared/enzyme_candidate_quality_audit_v0_20260528_native_probe_1p1n/`

运行设置：

```bash
python scripts/audit_enzyme_candidate_quality_v0.py \
  --positives 1 --negatives 1 --max-targets 2 \
  --top-k 3 --bridge-top-k 4 --max-bridge-ec-contexts 1 \
  --sources enzyme_precedent,v3_retrieval \
  --include-native-chem-enzy
```

结果：

| source | rows | unique | ready | strong | positive ready recall | negative ready rate |
|---|---:|---:|---:|---:|---:|---:|
| enzyme_precedent | 9 | 7 | 4 | 4 | 1.0000 | 0.0000 |
| v3_retrieval | 9 | 6 | 0 | 0 | 0.0000 | 0.0000 |
| chem_enzy_onestep | 6 | 6 | 0 | 0 | 0.0000 | 0.0000 |

解释：

- `chem_enzy_onestep` 在这里作为原生化学 proposer 参考，不用 `search_ready` 酶候选口径评价；
- 它确实能返回普通化学拆解候选；
- `enzyme_precedent` 提供的是原生 ChemEnzy one-step 不负责的酶 precedent 候选；
- 因此当前增强是互补 proposal coverage，而不是替代 ChemEnzy 化学单步模型。

## 当前仍未完成

1. 还没有把 `search_ready` 酶候选默认注入 Web/native ChemEnzy 主搜索。
2. 原生 ChemEnzy one-step 的直接候选级对比还需要用 `--include-native-chem-enzy` 单独跑；这会初始化较重 vendor 模型。
3. 当前 retrieval 仍主要是 product-side similarity，不是严格 reaction-center-aware retrieval。
4. SP-v1 是 weak-label verifier，不是专家真值；高分候选仍需要代表性人工复核。
5. 需要把 10p10n 扩大到更稳定的 100/500 target benchmark，并报告 route-level solved / route quality / latency。

## 下一步

推荐顺序：

1. 把 `enzyme_precedent + quality gate` 接入 route-tree 搜索的受控分支；
2. 跑 `native ChemEnzy` vs `AutoPlanner enhanced route-tree` 的固定 benchmark；
3. 再跑 `--include-native-chem-enzy` 候选级对照，明确原生 single-step 在同一批 target 上提供了哪些化学候选；
4. 继续把 `enzyme_precedent` 从 product similarity 升级成 substrate-product pair / reaction-center-aware retrieval。
