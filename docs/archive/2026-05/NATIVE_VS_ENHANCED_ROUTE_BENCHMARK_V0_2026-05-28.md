# Native ChemEnzy vs Enhanced Route-Tree 阶段报告

日期：2026-05-28

## 目的

本报告记录当前“增强 AutoPlanner route-tree”相对原生 ChemEnzy 的真实进展。重点不是宣称多步性能已经超过原生系统，而是把增强发生在哪一层、证据是什么、还卡在哪里讲清楚。

## 当前单步/多步模型形态

### 原生 ChemEnzy 主路径

真实 Web/native 服务仍以 ChemEnzy 原生搜索为主：

- `graphfp_models.USPTO-full_remapped`：普通有机化学 one-step proposer；
- `onmt_models.bionav_one_step`：BioNav/ChemEnzy seq2seq one-step proposer；
- 默认 stock：`Zinc_Fix-stock`；
- 搜索：ChemEnzy vendor MCTS/planner。

vendor 配置中还注册了 `template_relevance.pistachio / pistachio_ringbreaker / reaxys / bkms_metabolic / reaxys_biocatalysis`，但这些不是当前默认全开主线。此前测试显示 naive ensemble 会显著增加噪声和耗时，因此后续原则是 gated 调用，而不是所有 proposer 全开。

### 当前增强路径

增强路线目前在 AutoPlanner route-tree / sidecar 层实现，核心新增能力包括：

- `enzyme_precedent`：基于 109,857 条酶反应 precedent 的检索源；
- `BridgeAwareSourceGate`：只有目标或中间体有 bridge/EC 证据时才更积极触发酶源；
- `SP-v1 enzyme-substrate-product verifier`：对酶源 proposal 做 substrate-product-EC 判别过滤；
- `bridge_ec_context`：当 bridge 命中 EC1 时，用该 EC 上下文额外触发酶 proposal；
- `bridge enzyme bonus`：对有 bridge 证据的酶动作给搜索采纳 bonus，用于验证“候选有了但搜索不选”的问题；
- `timeout frontier fallback`：搜索超时且已有部分状态时，返回前沿 partial routes，避免把“已采纳但未闭合”的路线误报为完全无结果。

## 已完成的代码改动

主要文件：

- `cascade_planner/route_tree/search.py`
  - 接入 `enzyme_sp_verifier` runtime gate；
  - 增加 bridge EC context proposals；
  - 增加 bridge-supported enzyme selection bonus；
  - 增加 timeout frontier fallback，输出仍标注为 `timeout_frontier` partial，不伪装成 solved。
- `cascade_planner/route_tree/schema.py`
  - 将 selected action metadata 中的 `enzyme_sp_verifier_v1`、`source_gate`、`source_provenance` 写入 `Slot.evidence`；
  - 解决原先“搜索时 SP-v1 已接受，但导出路线看不到 accepted 证据”的问题。
- `cascade_planner/route_tree/proposals.py`
  - 修复 bridge gate 与 source budget floor 的优先级冲突；
  - 当 bridge gate 判定某节点没有 bridge 证据并 suppress enzymatic 时，`enzyme_precedent` / `v3_retrieval` 等酶侧 source 不能被 `AUTOPLANNER_ROUTE_TREE_SOURCE_MIN_BUDGETS` 重新打开。
- `cascade_planner/cascadeboard/route_export.py`
  - route export 现在在 step 顶层和 evidence 中都保留 `enzyme_sp_verifier_v1`；
  - benchmark 可以区分“酶源步骤”和“SP-v1 accepted 酶步骤”。
- `scripts/run_native_vs_enhanced_route_benchmark.py`
  - 新增 native-vs-enhanced route-level benchmark；
  - 修正目标选择，保证 `--positives 1 --max-targets 1` 不会被负例覆盖；
  - 新增 enzyme proposal calls/candidates 指标；
  - 新增 `sp_v1_accepted_enzyme_routes` 和 summary 中的 `targets_with_sp_v1_accepted_enzyme_route`；
  - 新增 `--bridge-enzyme-bonus`、`--enzyme-precedent-min-budget`、`--enhanced-stock zinc|none`；
  - enhanced route-tree 默认接 ZINC stock checker，使其与 native ChemEnzy 的 stock 设定更可比。
- `scripts/run_enzyme_precursor_expansion_benchmark.py`
  - 新增 accepted enzyme precursor continuation benchmark；
  - 从 route rows 抽取 SP-v1 accepted 酶步骤的上游前体；
  - 只用 chemical-only engine 继续展开这些前体，判断瓶颈是在酶步候选还是酶步之后的化学闭合。
  - 支持多个 route-row 文件输入；
  - 默认排除 CoA/NAD/ATP/核苷酸/多磷酸 carrier-like accepted precursors，避免把辅因子/载体结构当普通化学前体强拆；
  - 支持 `--include-carrier-like` 做审计复现；
  - 支持 `--chem-template-max-per-query` / `--chem-template-max-templates` 控制 chemical template 扫描成本。
- `cascade_planner/route_tree/search.py`
  - 新增可开关的 exact-stock action bonus；
  - 环境变量 `AUTOPLANNER_ROUTE_TREE_EXACT_STOCK_REACTANT_BONUS` / `AUTOPLANNER_ROUTE_TREE_FULL_STOCK_ACTION_BONUS` 默认关闭，不改变旧行为；
  - 打开后，搜索会对 reactants 精确命中 stock 的动作给软 bonus，并把 `stock_closure_bonus` 和 `stock_closure_diagnostics` 写入 selection trajectory；
  - 目的不是把小分子 heuristic 当 stock，而是让 continuation 阶段更偏向真实库存闭合。
- `cascade_planner/cascadeboard/live_retro.py`
  - 新增 `build_chemical_retro_engine()`；
  - continuation benchmark 现在可以只加载 chemical sources，避免为了 chemical-only 测试加载酶模型。
- `tests/test_native_vs_enhanced_route_benchmark.py`
  - 覆盖目标选择、source budget 参数、stock checker 关闭路径。
- `tests/test_enzyme_precursor_expansion_benchmark.py`
  - 覆盖 SP-v1 accepted 酶前体抽取和 continuation summary 统计。
- `tests/test_route_tree_planner.py`
  - 覆盖 timeout frontier fallback：硬超时后能返回已探索的 partial route，并标注 `timeout_frontier`。
  - 覆盖 bridge gate suppress 后 enzyme_precedent floor 不再泄漏。

## 验证

已通过：

```bash
python -m py_compile scripts/run_native_vs_enhanced_route_benchmark.py tests/test_native_vs_enhanced_route_benchmark.py
pytest -q tests/test_native_vs_enhanced_route_benchmark.py tests/test_enzyme_sp_verifier_gate.py tests/test_enzyme_candidate_quality_audit.py
# 8 passed

python -m py_compile cascade_planner/route_tree/search.py tests/test_route_tree_planner.py
pytest -q tests/test_route_tree_planner.py -k 'frontier_fallback or contrast_fallbacks or result_pool_larger'
# 3 passed, 56 deselected

python -m py_compile cascade_planner/route_tree/schema.py cascade_planner/cascadeboard/route_export.py scripts/run_native_vs_enhanced_route_benchmark.py scripts/run_enzyme_precursor_expansion_benchmark.py tests/test_enzyme_sp_verifier_gate.py tests/test_native_vs_enhanced_route_benchmark.py tests/test_enzyme_precursor_expansion_benchmark.py
pytest -q tests/test_enzyme_sp_verifier_gate.py tests/test_native_vs_enhanced_route_benchmark.py tests/test_enzyme_precursor_expansion_benchmark.py
# 10 passed

python -m py_compile cascade_planner/route_tree/proposals.py tests/test_route_tree_planner.py
pytest -q tests/test_route_tree_planner.py -k 'bridge_aware_source_gate_blocks_enzyme_precedent_floor_without_bridge_hit or bridge_aware_source_gate_blocks_enzymatic_fallback_without_bridge_hit or bridge_aware_source_gate_suppresses_enzymatic_budget_without_bridge_hit'
# 3 passed, 57 deselected

python -m py_compile cascade_planner/route_tree/search.py scripts/run_enzyme_precursor_expansion_benchmark.py tests/test_route_tree_planner.py
pytest -q tests/test_route_tree_planner.py -k 'stock_closure_bonus or action_cost_treats_stock_closure or action_delta_penalizes_nonstock_small_reactants or bridge_supported_enzyme_bonus'
# 4 passed, 57 deselected

python -m py_compile scripts/run_enzyme_precursor_expansion_benchmark.py tests/test_enzyme_precursor_expansion_benchmark.py cascade_planner/cascadeboard/retrorules_applicator.py
pytest -q tests/test_enzyme_precursor_expansion_benchmark.py
# 6 passed
```

## 小规模 route-level smoke 结果

### baseline enhanced smoke，不加采纳 bonus

输出：

`results/shared/native_vs_enhanced_route_benchmark_20260528_enhanced_positive_fixed_v2/`

结果：

| 指标 | 数值 |
|---|---:|
| positive targets | 1 |
| returned routes | 0 |
| enzyme proposal calls | 8 |
| enzyme proposal candidates | 11 |
| SP-v1 scored | 11 |
| SP-v1 rejected | 3 |
| stop reason | hard_timeout |

解释：酶候选已经进入真实搜索统计，但当时没有返回 partial route，因此表现为 `0 route`。

### bonus + enzyme_precedent budget，无 stock checker 旧行为

输出：

`results/shared/native_vs_enhanced_route_benchmark_20260528_enhanced_positive_bonus2/`

结果：

| 指标 | 数值 |
|---|---:|
| returned routes | 0 |
| enzyme proposal calls | 18 |
| enzyme proposal candidates | 28 |
| SP-v1 scored | 35 |
| SP-v1 rejected | 6 |
| stop reason | hard_timeout |

解释：预算和 bonus 明显增加了酶候选覆盖，但仍不能闭合路线。

### bonus + enzyme_precedent budget + ZINC stock + timeout frontier fallback

输出：

`results/shared/native_vs_enhanced_route_benchmark_20260528_enhanced_positive_bonus2_zinc_frontier_v2/`

结果：

| 指标 | 数值 |
|---|---:|
| positive targets | 1 |
| returned routes | 2 |
| solved routes | 0 |
| progressive routes | 0 |
| enzyme route targets | 1 |
| enzyme proposal calls | 10 |
| enzyme proposal candidates | 15 |
| SP-v1 scored | 18 |
| SP-v1 rejected | 5 |
| timeout frontier partial routes | 2 |
| stop reason | hard_timeout |

返回的两条路线都是 `timeout_frontier` partial route，不是 solved/progressive route。其中一条包含酶源步骤：

```text
source = enzexpand
EC = 1.x
product = CC/C=C\C/C=C\CC(/C=C/C=C\C/C=C\C/C=C\CCC(=O)[O-])OO
main_reactant = CC/C=C\C/C=C\CC(/C=C/C=C\C/C=C\C/C=C\C[C@H](N)C(=O)[O-])OO
```

因此当前可以证明的是：增强分支已经能在路线层采纳酶动作，并能把酶候选作为 partial route 返回；还不能证明它已经提高 solved route 性能。

### SP-v1 trace export 修正版

输出：

`results/shared/native_vs_enhanced_route_benchmark_20260528_enhanced_positive_sptrace/`

结果：

| 指标 | 数值 |
|---|---:|
| positive targets | 1 |
| returned routes | 2 |
| solved routes | 0 |
| progressive routes | 0 |
| enzyme route targets | 1 |
| SP-v1 accepted enzyme route targets | 1 |
| enzyme proposal calls | 10 |
| enzyme proposal candidates | 15 |
| SP-v1 scored | 18 |
| SP-v1 rejected | 5 |
| timeout frontier partial routes | 2 |
| stop reason | hard_timeout |

这次导出已经能在 selected step 上看到 SP-v1 accepted 证据。代表性酶步骤：

```text
source = enzexpand
EC = 1.x
SP-v1 score = 0.8113
SP-v1 threshold = 0.3633
product = CC/C=C\C/C=C\CC(/C=C/C=C\C/C=C\C/C=C\CCC(=O)[O-])OO
main_reactant = CC/C=C\C/C=C\CC(/C=C/C=C\C/C=C\C/C=C\C[C@H](N)C(=O)[O-])OO
```

解释：现在已经能证明增强路线里有被 SP-v1 接受的 selected enzyme action，而不是只在候选池里出现酶 proposal。但它仍是 partial route，没有 stock-closed。

## 酶步前体 continuation benchmark

输出：

`results/shared/enzyme_precursor_expansion_benchmark_20260528_sptrace/`

目的：把上面这条 SP-v1 accepted 酶步骤的上游前体单独拿出来，只用 chemical-only route-tree 继续展开，验证“酶步之后是否能继续闭合到库存”。

运行设置：

| 项 | 值 |
|---|---:|
| subgoals | 1 |
| chemical sources | `retrochimera`, `chemtemplates` |
| stock | ZINC |
| max depth | 3 |
| branch factor | 6 |
| expansion budget | 20 |
| timeout | 45 s |

结果：

| 指标 | 数值 |
|---|---:|
| extracted SP-v1 accepted enzyme precursors | 1 |
| initially in stock | 0 |
| subgoals with routes | 1 |
| solved subgoals | 0 |
| progressive subgoals | 0 |
| timeout-frontier routes | 2 |
| proposal source outputs, chemtemplates | 24 |
| proposal source outputs, retrochimera | 20 |
| selected step sources, retrochimera | 2 |
| stop reason | hard_timeout |
| runtime bottlenecks | source_timeout, proposal_slow, no_route_returned |

解释：

1. 这个 accepted 酶前体不是 ZINC stock。
2. chemical-only search 能继续产生 partial routes，但没有向更简单或库存方向取得有效 progress。
3. `chemtemplates` 实际有输出，但最终采纳的 partial route 仍来自 `retrochimera`；模板输出没有转化为有效 route progress。
4. 因此当前瓶颈已经进一步定位为：**SP-v1 accepted enzyme action 之后，上游化学 continuation/stock closure 仍不足**。

## continuation chemical proposer ablation

为了验证“是不是因为 continuation 阶段没有接 ChemEnzy chemical proposer”，我把 continuation benchmark 改成真正的 chemical-only engine，并做了同一 accepted enzyme precursor 的对照。

baseline 输出：

`results/shared/enzyme_precursor_expansion_benchmark_20260528_gatefloor_baseline/`

ChemEnzy one-step 补源输出：

`results/shared/enzyme_precursor_expansion_benchmark_20260528_gatefloor_chemenzy_onestep/`

设置相同：

| 项 | 值 |
|---|---:|
| accepted enzyme precursors | 1 |
| stock | ZINC |
| max depth | 3 |
| branch factor | 6 |
| expansion budget | 20 |
| timeout | 45 s |

结果：

| 指标 | baseline chemical-only | + ChemEnzy one-step |
|---|---:|---:|
| loaded chemical sources | `retrochimera`, `chemtemplates` | `retrochimera`, `chemtemplates`, `chem_enzy_onestep` |
| subgoals with routes | 1/1 | 1/1 |
| solved subgoals | 0/1 | 0/1 |
| progressive subgoals | 0/1 | 0/1 |
| returned routes | 2 | 2 |
| mean steps | 2.0 | 1.5 |
| selected `chem_enzy_graphfp` steps | 0 | 1 |
| selected `uspto_template` steps | 2 | 2 |
| selected `retrochimera` steps | 2 | 0 |
| stop reason | hard_timeout | hard_timeout |

解释：

1. ChemEnzy one-step 确实加载并输出了候选，也有 1 个 `chem_enzy_graphfp` step 被 route-tree 采纳。
2. 但它没有把 accepted 酶前体闭合到 stock，也没有产生 progressive route。
3. 因此当前瓶颈不是“没有接入 ChemEnzy chemical proposer”这么简单，而是 continuation 阶段缺少 stock-aware/action-value 的选择能力；候选存在，但选择后仍走向保护基/衍生化式大前体。
4. 下一步不应全局放宽 proposer，而应在 accepted enzyme precursor continuation 节点做 gated stock-aware rescue/action rerank。

## stock-aware action rerank smoke

为了验证“route-tree 选错候选”是否能通过更强 stock-aware 排序缓解，我新增了 exact-stock action bonus，并在同一 accepted enzyme precursor 上打开该开关重跑。

输出：

`results/shared/enzyme_precursor_expansion_benchmark_20260528_stockaware_smoke/`

运行设置：

| 项 | 值 |
|---|---:|
| accepted enzyme precursors | 1 |
| chemical sources | `retrochimera`, `chemtemplates`, `chem_enzy_onestep` |
| stock | ZINC |
| max depth | 3 |
| branch factor | 6 |
| expansion budget | 20 |
| timeout | 45 s |
| exact stock reactant bonus | 1.0 |
| full stock action bonus | 2.0 |

结果：

| 指标 | + ChemEnzy one-step | + ChemEnzy one-step + stock-aware action rerank |
|---|---:|---:|
| subgoals with routes | 1/1 | 1/1 |
| solved subgoals | 0/1 | 0/1 |
| progressive subgoals | 0/1 | 0/1 |
| returned routes | 2 | 2 |
| mean steps | 1.5 | 1.5 |
| selected `chem_enzy_graphfp` steps | 1 | 1 |
| selected `uspto_template` steps | 2 | 2 |
| stop reason | hard_timeout | hard_timeout |

解释：

1. stock-aware action bonus 已经进入 route-tree cost model，并由单元测试覆盖；
2. 但在这个 accepted enzyme precursor 上，打开该 bonus 后仍然没有 solved/progressive route；
3. 这说明当前样本不是简单“stock 命中候选被排序埋掉”，而是候选池里缺少能把该高级前体继续连接到 stock 的有效拆解；
4. 下一步要扩大 accepted enzyme precursor continuation benchmark，确认这个结论在更多酶步前体上是否稳定；如果稳定，就必须优先增强 chemical precursor proposal / route value，而不是继续只调 rerank 权重。

## multi-row accepted precursor continuation

为了避免单样本误判，我把 continuation benchmark 扩展为支持多个 route-row 文件输入，并合并去重 SP-v1 accepted 酶前体。用现有 3 个 enhanced route-row 文件共 7 条 parent rows，抽到了 4 个唯一主前体。

输出：

`results/shared/enzyme_precursor_expansion_benchmark_20260528_multirow_stockaware/`

输入文件：

- `results/shared/native_vs_enhanced_route_benchmark_20260528_1p1n_gatefloor_fix/native_vs_enhanced_route_rows.jsonl`
- `results/shared/native_vs_enhanced_route_benchmark_20260528_1p1n_sptrace_compare/native_vs_enhanced_route_rows.jsonl`
- `results/shared/native_vs_enhanced_route_benchmark_20260528_enhanced_positive_sptrace/native_vs_enhanced_route_rows.jsonl`

设置：

| 项 | 值 |
|---|---:|
| parent rows | 7 |
| unique accepted enzyme precursors | 4 |
| chemical sources | `retrochimera`, `chemtemplates`, `chem_enzy_onestep` |
| stock | ZINC |
| max depth | 3 |
| branch factor | 6 |
| expansion budget | 20 |
| timeout | 30 s / subgoal |
| stock-aware action rerank | on |

结果：

| 指标 | 数值 |
|---|---:|
| subgoals | 4 |
| initially in stock | 0 |
| subgoals with routes | 4 |
| subgoals solved | 1 |
| subgoals progressive | 1 |
| total routes | 7 |
| solved routes | 2 |
| timeout-frontier routes | 4 |
| mean elapsed | 26.039 s |

逐前体结果：

| 前体类型 | 酶步来源 | SP-v1 score | 结果 |
|---|---|---:|---|
| 长链多烯氨基酸/过氧化前体 | `enzexpand` | 0.811 | 2 条 timeout-frontier，0 solved |
| 长链多烯羧酸前体 | `enzyme_precedent`, EC 1.13.11.12 | 0.987 | 1 条 dead-end，0 solved |
| 长链多烯羟基过氧化前体 | `enzyme_precedent` | 0.816 | 2 条 timeout-frontier，0 solved |
| 二氯苯亚胺环己二烯酮前体 | `enzyme_precedent`, EC 1.1.3.15 | 0.544 | 2 条 stock-closed，2 solved |

解释：

1. accepted enzyme precursor 并非全部不可继续拆；4 个中有 1 个能由 chemical-only route-tree 闭合到 stock。
2. 失败的 3 个集中在长链多烯/过氧化/羧酸型高级前体；这类结构对当前 chemical proposer 和 stock closure 明显困难。
3. `chem_enzy_onestep`、`chemtemplates`、`retrochimera` 都有输出，selected step 包含 `chem_enzy_graphfp`、`retrochimera`、`uspto_template`，但多数没有形成可闭合路径。
4. 因此现在的瓶颈要更精确地表述为：**酶步 accepted 后，部分前体可闭合；长链脂质/多烯/过氧化类前体的化学续拆覆盖不足，是当前主要失败簇。**
5. 下一步需要扩大 SP-v1 accepted precursor pool，而不是继续围绕 4 个样本调权重。

## 10p10n enhanced precursor pool

为了扩大 accepted enzyme precursor pool，我跑了一批 enhanced-only 目标：

输出：

`results/shared/native_vs_enhanced_route_benchmark_20260528_enhanced_10p10n_precursor_pool/`

设置：

| 项 | 值 |
|---|---:|
| positives / negatives | 10 / 10 |
| targets | 20 |
| native | skipped |
| max depth | 3 |
| branch factor | 6 |
| expansion budget | 20 |
| n results | 2 |
| timeout | 25 s / target |
| bridge enzyme bonus | 2 |
| enzyme_precedent min budget | 2 |
| stock | ZINC |

route-level 结果：

| 指标 | 数值 |
|---|---:|
| targets with routes | 19/20 |
| solved targets | 6/20 |
| progressive targets | 1/20 |
| enzyme route targets | 5/20 |
| SP-v1 accepted enzyme route targets | 5/20 |
| enzyme proposal candidates | 101 |
| SP-v1 rejections | 28 |
| negative enzyme proposal rate | 0.0 |
| negative enzyme target rate | 0.0 |
| timeout-frontier partial routes | 24 |

解释：

1. bridge gate + SP-v1 gate 在这批 10 个负例上没有触发错误酶 route，负例酶污染仍为 0；
2. 10 个正例中有 5 个目标采纳了 SP-v1 accepted 酶步骤，说明 enzyme proposal/gate 已经能稳定进入路线；
3. enhanced route-tree 自身有 6/20 solved，但这些 solved 不能直接等同于“优于 ChemEnzy”，因为本批没有跑 native 对照；
4. 这批数据的主要用途是扩大 accepted enzyme precursor pool，用于 continuation 诊断。

从该批 route rows 抽到 7 个唯一 SP-v1 accepted 主前体：

- 5 个长链多烯/脂质/过氧化或羟基脂肪酸型前体；
- 1 个蒽醌/多酚样前体；
- 1 个 CoA/核苷酸辅因子样前体。

加入 carrier-aware extraction 后，默认 continuation 前体数从 7 降到 6；被排除的是 CoA/核苷酸/多磷酸样前体，原因包括：

```text
coa_thioester_motif
known_carrier_fragment
nucleotide_phosphate
polyphosphate
```

这符合之前的化学判断：这类结构常常来自酶反应载体/辅因子，不应被后续普通化学逆合成当作主前体强拆。

## 10p10n accepted precursor continuation

输出：

`results/shared/enzyme_precursor_expansion_benchmark_20260528_10p10n_stockaware/`

设置：

| 项 | 值 |
|---|---:|
| accepted enzyme precursors | 7 |
| chemical sources | `retrochimera`, `chemtemplates`, `chem_enzy_onestep` |
| stock | ZINC |
| max depth | 3 |
| branch factor | 6 |
| expansion budget | 20 |
| timeout | 25 s / subgoal |
| stock-aware action rerank | on |

结果：

| 指标 | 数值 |
|---|---:|
| subgoals with routes | 7/7 |
| stock-closed subgoals | 0/7 |
| progressive subgoals | 1/7 |
| timeout-frontier routes | 14 |
| total routes | 14 |
| mean steps | 1.643 |
| mean elapsed | 28.175 s |

逐前体结论：

| 前体簇 | 数量 | 结果 |
|---|---:|---|
| 长链多烯/脂质/过氧化/羟基脂肪酸 | 5 | 0 solved，1 progressive partial |
| 蒽醌/多酚样前体 | 1 | 0 solved |
| CoA/核苷酸辅因子样前体 | 1 | 0 solved，应视为辅因子/条件相关结构，不能简单当普通化学前体强拆 |

解释：

1. 新池 7 个 accepted enzyme precursors 全部能产生 partial routes，但没有一个闭合到 stock；
2. `chem_enzy_onestep`、`chemtemplates`、`retrochimera` 都有输出，selected step 中也有 `chem_enzy_graphfp`、`retrochimera`、`uspto_template`，所以失败不是“完全没有候选”；
3. 失败主要集中在长链脂质/多烯类结构，这类结构对当前 chemical proposal 和 stock database 都不友好；
4. CoA/核苷酸样前体暴露出另一个问题：有些酶 reaction precedent 会把辅因子/载体结构带入 main precursor，这不应该被后续化学逆合成当作普通目标硬拆；
5. carrier-aware extraction 已经加入，默认排除 CoA、NAD、ATP、核苷酸、多磷酸 carrier-like 前体；后续报告应同时给出“默认前体池”和“include-carrier-like 审计池”；
6. 下一步需要增强脂质/长链前体的 chemical proposal/stock mapping。

## 1p1n gate-floor 修复对照

修复前对照输出：

`results/shared/native_vs_enhanced_route_benchmark_20260528_1p1n_sptrace_compare/`

关键现象：

| 指标 | native | enhanced |
|---|---:|---:|
| targets | 2 | 2 |
| targets with routes | 2 | 2 |
| solved targets | 2 | 1 |
| progressive targets | 2 | 0 |
| enzyme route targets | 0 | 2 |
| SP-v1 accepted enzyme route targets | 0 | 2 |
| negative enzyme proposal rate | 0.0 | 1.0 |
| negative enzyme target rate | 0.0 | 1.0 |

解释：增强分支在 positive 和 negative/control 两个目标上都选到了 SP-v1 accepted enzyme step。这说明 `SP-v1 accepted` 还不能单独等同于“路线合理”，并且当时存在 gate 泄漏：`enzyme_precedent_min_budget=2` 会在 bridge gate 无命中时重新打开酶侧 source。

修复后 enhanced-only 输出：

`results/shared/native_vs_enhanced_route_benchmark_20260528_1p1n_gatefloor_fix/`

结果：

| 指标 | 修复后 enhanced |
|---|---:|
| targets | 2 |
| targets with routes | 2 |
| solved targets | 0 |
| progressive targets | 1 |
| enzyme route targets | 1 |
| SP-v1 accepted enzyme route targets | 1 |
| positive enzyme proposal recall | 1.0 |
| positive enzyme target recall | 1.0 |
| negative enzyme proposal rate | 0.0 |
| negative enzyme target rate | 0.0 |
| enzyme proposal candidates | 11 |
| SP-v1 rejects | 3 |

解释：

1. 修复后 positive 目标仍保留 1 条 SP-v1 accepted 酶 partial route。
2. negative/control 目标的 `enzyme_proposal_calls=0`、`enzyme_routes=0`，不再被 enzyme floor 污染。
3. 这不是 solved-rate 提升，但它是精度提升：减少错误酶步进入搜索主线。
4. 代价是增强分支在这个小对照里没有 solved target；说明下一步要做 continuation/closure，而不是继续放宽酶 gate。

## 与原生 ChemEnzy 的关系

当前不是替换原生 ChemEnzy one-step 模型，而是在其外侧补一个更谨慎的 chemo-enzymatic route-tree 分支。对比逻辑如下：

| 层级 | 原生 ChemEnzy | 当前增强 |
|---|---|---|
| chemical one-step | `graphfp USPTO-full_remapped` | 保留/参考，不直接替换 |
| bio/enzyme one-step | `bionav_one_step` | 外加 `enzyme_precedent`、`v3_retrieval`、`enzexpand`、`enzyformer` |
| enzyme feasibility | 原生有 enzyme assign / classifier，但不是我们的主判别器 | SP-v1 substrate-product-EC verifier |
| source routing | vendor planner 内部策略 | BridgeAwareSourceGate + budget floors + bridge EC context |
| route result | native MCTS routes | route-tree solved/partial routes，区分 `timeout_frontier` |
| stock | `Zinc_Fix-stock` | benchmark 现在默认接 ZINC checker |

当前证据更强的是 proposal/candidate 层，而不是 solved-route 层。

## 2p2n native 对照与 SP-v1 selection bonus

原始 native+enhanced 对照输出：

`results/shared/native_vs_enhanced_route_benchmark_20260528_current_2p2n_stockaware/`

设置：2 个 bridge-positive、2 个 negative/control，`max_depth=3`，`branch_factor=6`，`expansion_budget=20`，enhanced 启用 ZINC stock、bridge enzyme bonus、stock-aware action rerank，但未启用 SP-v1 accepted selection bonus。

结果：

| 指标 | native ChemEnzy | enhanced route-tree |
|---|---:|---:|
| targets | 4 | 4 |
| targets with routes | 4 | 4 |
| solved targets | 4 | 0 |
| progressive targets | 4 | 0 |
| enzyme route targets | 0 | 1 |
| SP-v1 accepted enzyme route targets | 0 | 1 |
| positive enzyme proposal recall | 0.0 | 1.0 |
| positive enzyme target recall | 0.0 | 0.5 |
| negative enzyme proposal rate | 0.0 | 0.0 |
| negative enzyme target rate | 0.0 | 0.0 |
| timeout-frontier routes | 0 | 8 |
| mean elapsed | 171.282 s | 54.495 s |

解释：native 路线闭合能力仍明显更强，但不产生酶路线；enhanced 能提出酶候选且没有负例酶污染，但只有一半 positive 被选中为酶路线，且都是 partial。

新增搜索层增强：

- `AUTOPLANNER_ROUTE_TREE_ENZYME_SP_ACCEPTED_BONUS`
- `AUTOPLANNER_ROUTE_TREE_ENZYME_SP_SCORE_BONUS`
- `AUTOPLANNER_ROUTE_TREE_ENZYME_SP_BONUS_CAP`

这些参数默认关闭。打开后，SP-v1 已接受的酶候选不只用于过滤错误 proposal，也进入 selection cost，帮助高置信酶步在候选池中被选中。

SP bonus enhanced-only 输出：

`results/shared/native_vs_enhanced_route_benchmark_20260528_enhanced_2p2n_spbonus/`

设置：同一组 2p2n targets，`--skip-native`，额外启用 `--enzyme-sp-accepted-bonus 0.75 --enzyme-sp-score-bonus 1.0`。

| 指标 | enhanced 无 SP selection bonus | enhanced 有 SP selection bonus |
|---|---:|---:|
| targets | 4 | 4 |
| targets with routes | 4 | 4 |
| solved targets | 0 | 0 |
| progressive targets | 0 | 1 |
| enzyme route targets | 1 | 2 |
| SP-v1 accepted enzyme route targets | 1 | 2 |
| positive enzyme proposal recall | 1.0 | 1.0 |
| positive enzyme target recall | 0.5 | 1.0 |
| negative enzyme proposal rate | 0.0 | 0.0 |
| negative enzyme target rate | 0.0 | 0.0 |
| enzyme proposal candidates | 14 | 23 |
| SP-v1 rejects | 6 | 8 |
| timeout-frontier routes | 8 | 4 |
| mean elapsed | 54.495 s | 33.137 s |

解释：

1. SP-v1 accepted bonus 让两个 positive target 都选中了 SP-v1 accepted enzyme route；
2. negative/control target 仍然没有酶 proposal 污染；
3. 其中一个 negative/control 产生了 3 步 progressive chemical partial route，但不是 enzyme route；
4. enhanced 仍然 0 solved，说明当前瓶颈已经从“酶候选是否能进路线”转到“酶步之后/之前的 continuation 与 stock closure”；
5. 这个增强不能被表述为整体超过 native ChemEnzy，只能表述为：酶步采纳率和酶步选择精度改善。

## 当前结论

1. 单步主模型仍是 ChemEnzy 默认的 `graphfp USPTO-full_remapped` + `bionav_one_step`。
2. 酶步判别器 SP-v1 已经明显强于初版，适合作为第一阶段 gate。
3. 新增 `enzyme_precedent` 明显增强了酶候选覆盖；候选审计中 10 个 bridge-positive 目标 ready recall 为 1.0，negative ready rate 为 0.0。
4. route-level smoke 显示增强分支能产生并采纳 SP-v1 accepted 酶动作，但当前只到 partial route，还没有形成稳定 solved route。
5. 酶步前体 continuation benchmark 说明：目前主要瓶颈不是“完全没有酶候选”，而是酶候选被接受后，上游化学前体继续展开和 stock 闭合失败。
6. gate-floor 修复后，negative/control 上错误酶步污染从 1.0 降到 0.0；这说明增强方向开始具备“只在有 bridge 证据时调用酶源”的可控性。
7. ChemEnzy one-step 作为 continuation 补源能被调用和采纳，但没有解决 stock closure；这说明下一步重点应转向 stock-aware continuation rerank/rescue。
8. stock-aware action rerank 已经实现并通过 smoke；4 个历史唯一 accepted 前体中 1 个可闭合，但新 10p10n 池的 7 个前体 0 个可闭合，说明 closure rate 强依赖前体结构分布。
9. 当前主要失败簇是长链脂质/多烯/过氧化前体，以及 cofactor/carrier 样酶前体错误进入普通化学续拆。
10. SP-v1 accepted selection bonus 已经把 positive target 的 enzyme route selection 从 0.5 提到 1.0，同时保持 negative enzyme rate 为 0.0；这说明 verifier 现在可以作为 selection signal，而不只是 filter。
11. 当前 route-level 证据仍不能宣称整体性能超过 ChemEnzy；能宣称的是增强系统已经从候选层推进到 route-selected enzyme action 层，并且瓶颈被定位到 continuation/closure 的具体结构类型。
12. 2026-05-28 后续修正了 bridge-derived EC context 的 source floor：此前 EC context 下 `enzyme_precedent` 可能被 `v3_retrieval/enzyformer/enzexpand` 挤出预算；现在 EC context 默认保留 `enzyme_precedent` 预算。

### EC context enzyme_precedent floor smoke

输出：

`results/shared/native_vs_enhanced_route_benchmark_20260528_enhanced_2p2n_ec_precedent_floor/`

设置：2 正 / 2 负，enhanced-only，不使用手动 `--enzyme-precedent-min-budget`，保留 SP-v1 selection bonus。

| 指标 | SP bonus 手动 floor 版本 | EC-context 自动 floor 版本 |
|---|---:|---:|
| targets | 4 | 4 |
| targets with routes | 4 | 4 |
| solved targets | 0 | 0 |
| progressive targets | 1 | 1 |
| enzyme route targets | 2 | 2 |
| SP-v1 accepted enzyme route targets | 2 | 2 |
| positive enzyme target recall | 1.0 | 1.0 |
| negative enzyme target rate | 0.0 | 0.0 |
| enzyme proposal candidates | 23 | 20 |
| SP-v1 rejects | 8 | 6 |
| timeout-frontier routes | 4 | 4 |
| mean elapsed | 33.137 s | 32.045 s |

预算验证：

```text
EC context top_k=6:
enzyme_precedent=2
v3_retrieval=2
enzyformer=1
enzexpand=1
chemical sources=0
```

解释：这次改动没有解决 solved route，但消除了一个真实集成问题：`enzyme_precedent` 作为目前质量最高的酶候选来源，现在在 bridge/EC 命中时不再依赖手动 floor 才能进入搜索。负例仍保持酶源预算为 0，说明 bridge gate 的抑制逻辑没有被破坏。

### Accepted enzyme precursor closure：normalized stock / no-progress smoke

目标：验证 accepted enzyme step 之后的 chemical continuation 是否只是 stock 盐型/电荷状态或假进展排序问题。

新增默认关闭的 route-tree selection 信号：

- `AUTOPLANNER_ROUTE_TREE_NORMALIZED_STOCK_REACTANT_BONUS`
- `AUTOPLANNER_ROUTE_TREE_NORMALIZED_STOCK_FULL_ACTION_BONUS`
- `AUTOPLANNER_ROUTE_TREE_NO_PROGRESS_SINGLE_REACTANT_PENALTY`

这些信号只影响 action selection，不改变 strict stock solved 定义。

验证输出：

- baseline stock rescue: `results/shared/enzyme_precursor_expansion_benchmark_20260528_spbonus_closure_rescue_v2/`
- normalized-stock: `results/shared/enzyme_precursor_expansion_benchmark_20260528_spbonus_normstock_v1/`
- normalized-stock + no-progress penalty: `results/shared/enzyme_precursor_expansion_benchmark_20260528_spbonus_normstock_noprogress_v1/`

| 指标 | baseline rescue | normalized stock | normalized + no-progress |
|---|---:|---:|---:|
| subgoals | 4 | 4 | 4 |
| initial stock | 1 | 1 | 1 |
| solved by route | 0 | 0 | 0 |
| closed by stock or route | 1 | 1 | 1 |
| progressive subgoals | 0 | 0 | 0 |
| timeout-frontier routes | 6 | 6 | 6 |
| stock rescue retries | 7 | 7 | 7 |
| mean elapsed | 25.203 s | 24.043 s | 23.813 s |

候选选择变化：

| source | baseline rescue | normalized + no-progress |
|---|---:|---:|
| chem_enzy_graphfp | 2 | 1 |
| retrochimera | 2 | 4 |
| uspto_template | 2 | 1 |

解释：

1. normalized-stock bonus 正确处理中和形式，例如 `CC(=O)[O-] -> CC(=O)O`，但没有带来新闭合；
2. no-progress penalty 能改变候选选择，减少部分“单反应物、重原子数不变、非 stock”的假进展；
3. 但三组实验均没有新增 solved/progressive route，说明当前 continuation 瓶颈不是单纯 rerank，而是化学 proposal 覆盖和搜索运行时；
4. accepted enzyme precursors 主要落在长链多烯/羧酸盐/芳香多酚盐结构，现有 chemical proposer 能给一跳变体，但很少给出可继续闭合到 stock 的有效拆解。

## 下一步建议

优先做三件事：

1. **提高化学前体 continuation 能力**：ChemEnzy one-step、exact/normalized stock bonus 和 no-progress penalty 已经验证为“可接/可用但不足以闭合全部前体”，下一步应增加能覆盖高级酶前体的 chemical proposal 数据和 stock-aware route value，而不是只增加候选源数量。
2. **继续优化 chemical template runtime / source gating**：已加入 rdchiral reaction cache 和模板扫描上限参数，但真实 route-tree 仍会在多个前沿节点首次编译大量模板；需要进一步做候选缓存、预筛选或按结构簇启用模板。
3. **cofactor/carrier policy 已加入，下一步接入主 benchmark 默认路径**：默认排除 carrier-like 前体，保留 `--include-carrier-like` 审计。
4. **按结构簇诊断 continuation**：当前失败集中在长链多烯/脂质/过氧化前体，应单独统计这类结构的 proposal coverage、stock proximity 和 route value。
5. **扩大 accepted enzyme precursor pool**：当前已有 7 个新前体，但还不足以训练 route value；目标至少 20-50 个唯一 accepted 前体后再训练/校准 continuation scorer。
6. **继续 native vs enhanced 对比**：在相同 ZINC stock、相同深度/时间预算下跑更大样本，报告 solved、partial、enzyme-selected、SP-v1 accepted、runtime，并分开报告“路线闭合能力”和“酶步覆盖能力”。

不建议马上做：

- 不建议把 template relevance 7 个模型全开；
- 不建议继续只堆 verifier 指标；
- 不建议把 timeout partial route 当 solved route 宣传；
- 不建议在 route-level 未闭合前宣称整体性能超过 ChemEnzy。

## Enzyme Precedent v2 Route Smoke

新增目的：验证主组分感知 `enzyme_precedent` 是否会降低路线层酶步采纳。

关键修正：

- `enzyme_precedent` 内部排序现在使用 `retrieval_rank_score`；
- route-tree 可见的 `score` 恢复为 0-1 product similarity；
- 否则 `score > 1` 会被 `_probability_from_score()` 当成无效概率，导致酶候选在 route selection 中被错误降权。

输出：

`results/shared/native_vs_enhanced_route_benchmark_20260528_enhanced_2p2n_main_component_v2_scorefix/`

设置：

| 项 | 值 |
|---|---:|
| positives / negatives | 2 / 2 |
| max depth | 3 |
| iterations | 80 |
| branch factor | 6 |
| expansion budget | 20 |
| enhanced stock | ZINC |
| stock-aware rerank | on |
| enzyme SP bonus | on |
| bridge enzyme bonus | 2.0 |
| native | skipped |

结果：

| 指标 | v2 scorefix |
|---|---:|
| targets with routes | 4/4 |
| solved targets | 0/4 |
| progressive targets | 1/4 |
| enzyme route targets | 2/4 |
| SP-v1 accepted enzyme route targets | 2/4 |
| positive enzyme target recall | 1.0000 |
| negative enzyme target rate | 0.0000 |
| enzyme proposal candidates | 18 |
| SP-v1 rejections | 6 |
| timeout-frontier routes | 4 |
| mean elapsed | 35.905 s |

与此前 native 对照的关系：

`results/shared/native_vs_enhanced_route_benchmark_20260528_current_2p2n_stockaware/`

| 指标 | native ChemEnzy | enhanced route-tree v2 scorefix |
|---|---:|---:|
| targets with routes | 4/4 | 4/4 |
| solved targets | 4/4 | 0/4 |
| progressive targets | 4/4 | 1/4 |
| enzyme route targets | 0/4 | 2/4 |
| SP-v1 accepted enzyme route targets | 0/4 | 2/4 |
| positive enzyme target recall | 0.0000 | 1.0000 |
| negative enzyme target rate | 0.0000 | 0.0000 |

结论：

增强分支在“酶步发现/酶步采纳”上继续优于原生 ChemEnzy，但整体路线闭合仍明显弱于原生 ChemEnzy。当前不能宣称多步逆合成性能超过原生；只能宣称新增模块提高了酶反应覆盖，并且通过 SP-v1/gate 控制了负例酶污染。

## Transition Signature v1 Route Smoke

新增目的：在 route-tree 中保留 enzyme precedent 的主底物-主产物 transition evidence，尤其处理“辅助反应物/条件可以解释元素变化”的情况。

输出：

`results/shared/native_vs_enhanced_route_benchmark_20260528_enhanced_2p2n_transition_v1/`

结果：

| 指标 | main-component v2 scorefix | transition v1 |
|---|---:|---:|
| targets with routes | 4/4 | 4/4 |
| solved targets | 0/4 | 0/4 |
| progressive targets | 1/4 | 1/4 |
| enzyme route targets | 2/4 | 2/4 |
| SP-v1 accepted enzyme route targets | 2/4 | 2/4 |
| positive enzyme target recall | 1.0000 | 1.0000 |
| negative enzyme target rate | 0.0000 | 0.0000 |
| enzyme proposal candidates | 18 | 23 |
| SP-v1 rejections | 6 | 6 |
| timeout-frontier routes | 4 | 4 |
| mean elapsed | 35.905 s | 36.951 s |

与原生 ChemEnzy 对照：

| 指标 | native ChemEnzy | enhanced transition v1 |
|---|---:|---:|
| solved targets | 4/4 | 0/4 |
| progressive targets | 4/4 | 1/4 |
| enzyme route targets | 0/4 | 2/4 |
| SP-v1 accepted enzyme route targets | 0/4 | 2/4 |
| positive enzyme target recall | 0.0000 | 1.0000 |

解释：

1. transition v1 没有损害酶步采纳，2/2 正例仍有 SP-v1 accepted enzyme route；
2. 负例酶污染仍为 0；
3. selected enzyme precedent step 中已经能看到 `transition_flags=[auxiliary_explains_element_gain]`，说明 O2/H2O 等辅助反应物可以解释元素变化；
4. route closure 仍未改善，native ChemEnzy 在 solved/progressive 上继续明显更强；
5. 因此增强分支当前应定位为“酶反应覆盖和证据增强层”，不是完整替代原生 ChemEnzy 的多步搜索。

## Native ChemEnzy Subplanner for Accepted Enzyme Precursors

问题：

前面的 continuation benchmark 已经说明：增强 route-tree 能发现并采纳酶步，但酶步上游前体继续拆到 ZINC stock 的能力弱。为了判断这是“前体本身不可拆”还是“route-tree chemical continuation 太弱”，新增 native subplanner probe：把 SP-v1 accepted enzyme precursor 直接交给原生 ChemEnzy 作为子目标搜索。

新增：

- `scripts/run_enzyme_precursor_expansion_benchmark.py`
  - `--native-subplanner`
  - `--native-subplanner-iterations`
  - `--native-subplanner-max-depth`
  - `--native-subplanner-expansion-topk`
  - `--native-subplanner-models`
  - `--native-subplanner-gpu`
- 输出：
  - `native_subplanner_summary`
  - `hybrid_summary`
  - `native_subplanner_outcomes`
  - `enzyme_precursor_native_subplanner_rows.jsonl`

验证：

```text
pytest -q tests/test_enzyme_precursor_expansion_benchmark.py tests/test_native_vs_enhanced_route_benchmark.py
14 passed
```

smoke 输出：

`results/shared/enzyme_precursor_expansion_benchmark_20260528_native_subplanner_transition_v1_smoke_v2/`

输入：

- route rows: `results/shared/native_vs_enhanced_route_benchmark_20260528_enhanced_2p2n_transition_v1/native_vs_enhanced_route_rows.jsonl`
- accepted enzyme precursors: 2
- route-tree continuation:
  - max depth 3
  - branch factor 6
  - expansion budget 20
  - ChemEnzy one-step proposal enabled
  - ZINC stock
- native subplanner:
  - iterations 80
  - max depth 3
  - expansion topk 50
  - models: `graphfp_models.USPTO-full_remapped`, `onmt_models.bionav_one_step`
  - stock: `Zinc_Fix-stock`

结果：

| 指标 | route-tree chemical-only | native ChemEnzy subplanner |
|---|---:|---:|
| accepted enzyme precursors | 2 | 2 |
| with routes | 2/2 | 2/2 |
| solved / stock-closed | 0/2 | 2/2 |
| total routes | 4 | 12 |
| mean steps | 1.25 | 1.0 |
| mean elapsed | 31.027 s | 9.173 s |
| failure categories | hard_timeout/source_timeout | none |

逐前体：

| precursor | route-tree solved | native solved | native routes | native elapsed |
|---|---:|---:|---:|---:|
| long polyene carboxylate | 0 | 1 | 9 | 9.006 s |
| hydroperoxy polyene carboxylate | 0 | 1 | 3 | 9.339 s |

结论：

这组 smoke 很关键：accepted enzyme precursor 并不是本质上不可拆；原生 ChemEnzy 可以在同样深度 3 下快速闭合。当前失败主要来自增强 route-tree 的 chemical continuation/search value，而不是酶步候选本身。

因此下一阶段架构应调整为：

```text
Bridge/EC trigger
→ enzyme_precedent + SP-v1 + transition evidence 发现可信酶步
→ accepted enzyme precursor 交给 native ChemEnzy chemical subplanner 闭合
→ 合并为 hybrid chemo-enzymatic route
```

这比“用 route-tree 全面替代 ChemEnzy”更现实，也更符合当前证据：增强模块负责原生 ChemEnzy 缺少的酶步覆盖，原生 ChemEnzy 负责它已经很强的化学闭合。

注意：

native subplanner 初始化会重新加载 ZINC stock 和 GraphFP/ONMT 模型，单次脚本有明显冷启动成本。若进入主流程，应把 ChemEnzy planner 作为常驻服务或复用进程，而不是每个请求重建。
