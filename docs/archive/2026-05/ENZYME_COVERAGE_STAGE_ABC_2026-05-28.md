# Enzyme Coverage Stage A/B/C

日期：2026-05-28

## 背景判断

当前 SP-v1 verifier 已能过滤明显错误酶步，但 verifier 不能生成新候选。继续只强化判别器会让系统更保守，不能解决真实酶反应物不在候选池的问题。

因此本阶段切换到 enzyme coverage 主线：

1. 阶段 A：审计酶数据与 proposal 覆盖；
2. 阶段 B：接入更大的 enzyme precedent retrieval proposal 源；
3. 阶段 C：把 bridge 命中的 EC context 传进 proposal，而不是只用 bridge gate 做预算分配。

## 阶段 A：覆盖审计

新增脚本：

- `scripts/audit_enzyme_coverage_v0.py`

小规模审计输出：

- `results/shared/enzyme_coverage_audit_v0_20260528_5p5n/enzyme_coverage_audit_report.json`
- `results/shared/enzyme_coverage_audit_v0_20260528_5p5n/enzyme_coverage_audit_report.md`

数据现状：

| 数据项 | 数量 |
|---|---:|
| enzyme reaction precedents | 109857 |
| enzyme substrate/product molecules | 82454 |
| chemical product molecules | 89387 |
| scored bridge candidates | 67209 |

5 正例 / 5 负例小样本覆盖：

| retrieval | targets with hits |
|---|---:|
| v3 retrieval | 10 / 10 |
| large enzyme precedent retrieval | 10 / 10 |

注意：这个指标只说明“能返回候选”，不说明候选一定化学合理。

## 阶段 B：大酶反应池 proposal 源

新增模块：

- `cascade_planner/cascadeboard/enzyme_precedent_retrieval.py`

新增 route-tree source：

- `enzyme_precedent`

数据来源：

- `data/bridge_pack_v0/enzyme_reaction_pool.parquet`

行为：

- 输入 product SMILES；
- 可选 EC1/EC prefix；
- 在 109857 条 enzyme reaction precedents 上按 product-side Morgan fingerprint similarity 检索；
- 输出 `CandidateAction` 兼容的 retrosynthetic proposal；
- evidence 中保留 reaction_id、precedent reaction、EC、Rhea、source_counts、example_ids。

快速验证：

- `results/shared/live_proposal_replay_pack_v1_20260528_enzyme_precedent_only_2p2n/manifest.json`

结果：

| source | actions |
|---|---:|
| enzyme_precedent | 28 |

说明新 source 已能实际进入 proposal replay pack。

## 阶段 C：Bridge EC Context 注入

修改位置：

- `cascade_planner/route_tree/source_gate.py`
- `cascade_planner/route_tree/search.py`
- `cascade_planner/route_tree/proposals.py`

核心变化：

1. BridgeAwareSourceGate 命中 bridge 后，不只设置 `bridge_gate_hits`，还记录：
   - `bridge_gate_ec_numbers`
   - `bridge_gate_ec1s`
   - `bridge_gate_primary_ec1`
   - `bridge_gate_confidence_tiers`

2. Route-tree proposal 在 root/no-EC context 下，如果发现 bridge EC 信息，会额外用 bridge-derived EC1 重新查询 enzyme sources。

3. `enzyme_precedent`、`v3_retrieval`、`enzyformer` 等可利用 EC context 的 source 可以获得更有方向的候选。

## Replay 结果

只用 `enzyme_precedent` 的 2 正例 / 2 负例 replay：

- pack: `results/shared/live_proposal_replay_pack_v1_20260528_enzyme_precedent_only_2p2n/live_proposal_replay_pack.jsonl`
- replay: `results/shared/enzyme_sp_v1_replay_benchmark_20260528_enzyme_precedent_only_2p2n/enzyme_sp_v1_replay_report.json`

结果摘要：

| policy | false enzyme target rate | recall | mean SP-v1 reject |
|---|---:|---:|---:|
| ungated_all | 1.0000 | 1.0000 | 0.00 |
| bridge_gate_v0_sp_v1_hard | 0.0000 | 1.0000 | 0.50 |

解释：

- 新 source 会带来酶候选覆盖；
- 但 ungated 仍会污染负例；
- bridge gate + SP-v1 仍必须保留为过滤层。

## 当前限制

1. 当前相似度仍是 product-side fingerprint，不是 reaction-center-aware retrieval。
2. `enzyme_precedent` 已有持久化索引，但冷启动读索引仍约 5-6 秒，需要进一步服务常驻缓存。
3. 大池命中率高不等于质量高；必须走 bridge/EC trigger + SP-v1 + artifact gate。
4. 目前还没有把 enzyme_precedent source 默认打开到 web 主搜索，只在 route-tree/sidecar/audit 层验证。

## Stage D：候选质量门控

新增：

- `scripts/audit_enzyme_candidate_quality_v0.py`
- `docs/ENZYME_CANDIDATE_QUALITY_AUDIT_V0_2026-05-28.md`

10 正 / 10 负 gated 审计：

| source | rows | ready | strong | positive ready recall | negative ready rate |
|---|---:|---:|---:|---:|---:|
| enzyme_precedent | 329 | 155 | 148 | 1.0000 | 0.0000 |
| v3_retrieval | 277 | 6 | 2 | 0.2000 | 0.0000 |

最小原生 ChemEnzy one-step 对照：

| source | rows | ready | 说明 |
|---|---:|---:|---|
| enzyme_precedent | 9 | 4 | 酶 precedent 候选 |
| v3_retrieval | 9 | 0 | 原有酶检索 |
| chem_enzy_onestep | 6 | 0 | 原生化学 proposer 参考，不用酶 ready 口径评价 |

结论：`enzyme_precedent` 明确增加了酶候选覆盖，但不能无门控注入。没有 bridge/EC 触发证据的高分候选降为 `ungated_review`，不进入 `search_ready`。

## 下一步

优先级：

1. 已修正 route-tree live search 的 EC-context source floor：bridge-derived EC 查询现在默认保留 `enzyme_precedent` 预算，而不是只给 `v3_retrieval/enzyformer/enzexpand`。
2. 扩大 benchmark：native ChemEnzy vs enhanced route-tree，报告 route-level solved/quality/latency。
3. 将 `enzyme_precedent` 从 product similarity 升级为 substrate-product pair / reaction-center-aware retrieval。

## Stage E：EC context 中保留 enzyme_precedent 预算

问题：

候选质量审计显示 `enzyme_precedent` 是当前最强酶候选来源，但 route-tree 的 bridge-derived EC context floor 只保证了：

```text
v3_retrieval
enzyformer
enzexpand
```

在小预算下，`enzyme_precedent` 可能被挤到 0，导致“审计里有效、真实搜索里不稳定进入”。

修正：

- `cascade_planner/route_tree/proposals.py`
  - EC context floor 加入 `enzyme_precedent=3`
  - `v3_retrieval` floor 从 3 调整为 2
  - `enzyformer` floor 从 2 调整为 1
  - stock-rescue 的 enzymatic route floor 也加入 `enzyme_precedent`
  - request cap 加入 `enzyme_precedent=24`
- `tests/test_route_tree_planner.py`
  - 新增 `test_ec_context_keeps_enzyme_precedent_budget`

验证：

```text
pytest -q tests/test_route_tree_planner.py -k 'ec_context_keeps_enzyme_precedent_budget or bridge_aware_source_gate or enzyme_sp_accepted_bonus or stock_rescue_budget'
9 passed, 55 deselected
```

预算结果：

```text
EC context top_k=6:
enzyme_precedent=2
v3_retrieval=2
enzyformer=1
enzexpand=1
retrorules=0
chemical sources=0
```

route-level smoke：

`results/shared/native_vs_enhanced_route_benchmark_20260528_enhanced_2p2n_ec_precedent_floor/`

结果：

- positive enzyme target recall: 1.0
- negative enzyme target rate: 0.0
- enzyme route targets: 2/4
- SP-v1 accepted enzyme route targets: 2/4
- solved targets: 0/4

结论：

这一步解决的是“高质量酶 precedent 是否能稳定进入搜索预算”，不是 route closure。当前真正瓶颈仍是 accepted enzyme step 之后的 chemical continuation / stock closure。

## Stage F：Accepted enzyme precursor closure 负结果

新增默认关闭的 route-tree selection 信号：

- normalized-stock bonus：奖励中和后命中 stock 的反应物，但不改变 strict stock solved；
- no-progress single-reactant penalty：惩罚单反应物、重原子数不变、且不进 stock 的假进展。

代码位置：

- `cascade_planner/route_tree/search.py`
- `scripts/run_enzyme_precursor_expansion_benchmark.py`
- `tests/test_route_tree_planner.py`
- `tests/test_enzyme_precursor_expansion_benchmark.py`

验证：

```text
pytest -q tests/test_route_tree_planner.py -k 'stock_closure_bonus or no_progress_single_reactant_penalty or stock_rescue_budget or late_stock_rescue'
7 passed, 59 deselected

pytest -q tests/test_enzyme_precursor_expansion_benchmark.py
7 passed
```

closure smoke：

`results/shared/enzyme_precursor_expansion_benchmark_20260528_spbonus_normstock_noprogress_v1/`

结果：

- closed by stock or route: 1/4
- solved by route: 0/4
- progressive: 0/4
- timeout-frontier routes: 6
- stock rescue retries: 7

解释：

normalized-stock 和 no-progress penalty 能改变候选排序，但不能产生新闭合。这说明 accepted enzyme precursor closure 的主因不是 rerank，而是 chemical proposal 覆盖和模板运行时。下一步要把工作重心放到高级化学前体 continuation 的 proposal 覆盖，而不是继续堆 selection bonus。

## 真实任务试运行模式

已新增真实 Web/native 任务的实验 sidecar：

- `cascade_planner/cascade_search/enzyme_coverage_sidecar.py`
- `scripts/run_chem_enzy_plan_for_web.py`
- `cascade_planner/web/static/index.html`
- `cascade_planner/web/static/app.js`

真实任务当前仍以 ChemEnzy native multi-step search 为主，不替换 vendor 搜索内核。开启“酶覆盖 sidecar”后，输出 JSON 会额外包含：

```text
route_set_metrics.enzyme_coverage_sidecar
ui_metadata.enzyme_coverage_sidecar
```

sidecar 内容包括：

- bridge hit 数量；
- bridge-derived EC1；
- `enzyme_precedent` top candidates；
- SP-v1 accepted/rejected 数量；
- accepted candidate evidence。

新增持久化索引：

- `data/bridge_pack_v0/enzyme_precedent_index_v1.joblib`

索引构建结果：

| 项目 | 数值 |
|---|---:|
| precedents | 109857 |
| 首次构建 | 64.52 s |
| 索引大小 | 9.95 MB |
| 冷启动读索引 | 5.67 s |
| sidecar+SP-v1 smoke | 9.56 s |

sidecar smoke：

```json
{
  "bridge_hits": 1,
  "candidates": 10,
  "accepted": 8,
  "rejected": 2,
  "top": "1.13.11.12"
}
```

这说明目前可以在真实任务中试用新酶覆盖模块，但它仍是 annotation/diagnostic sidecar；还不是主搜索 proposal replacement。

## Stage G：主组分感知 enzyme precedent retrieval

问题：

`enzyme_precedent` v1 只按完整 product-side fingerprint 检索，并用底物端“最大重原子组分”作为 `main_reactant`。这在酶数据库里不够稳，因为很多反应包含 ATP/ADP、NAD(P)、CoA、磷酸、水等载体/辅因子；这些组分有时比真实底物更大，容易被错误送进后续 chemical retrosynthesis。

修正：

- `cascade_planner/cascadeboard/enzyme_precedent_retrieval.py`
  - 新增 component blacklist 加载：默认读取 `data/bridge_pack_v0/cofactor_common_metabolite_blacklist.parquet`；
  - 产物相似度优先使用非 blacklist、非 carrier-like 的主产物组分；
  - 底物端 `main_reactant` 优先选择非 blacklist、非 carrier-like 且含碳的真实底物组分；
  - 保留辅助组分为 `aux_reactants`，不把大分子辅助试剂一刀切删除；
  - 新增 evidence：`precedent_product_main_smiles`、`product_full_similarity`、`retrieval_rank_score`、`substrate_component_selection`；
  - 修正 route-tree score 兼容性：返回给搜索的 `score` 保持 0-1 product similarity，`retrieval_rank_score` 只用于 retrieval 内部排序和 evidence，避免 `score > 1` 被 route-tree 当成 0 概率。
- `tests/test_enzyme_precedent_retrieval.py`
  - 覆盖“黑名单大组分不能作为 main_reactant”；
  - 覆盖“产物端带磷酸等组分时仍按主产物计算相似度”；
  - 覆盖返回 `score` 仍在 0-1。

验证：

```text
pytest -q tests/test_enzyme_precedent_retrieval.py tests/test_enzyme_candidate_quality_audit.py
6 passed
```

索引：

- 新索引：`data/bridge_pack_v0/enzyme_precedent_index_v2.joblib`
- precedents: 109857
- 索引大小：18 MB
- 一次性冷构建：285.365 s
- 冷启动读索引：16.243 s
- 单次检索 smoke：0.698 s

说明：v2 构建成本偏高，但这是离线索引成本；真实服务侧应继续使用常驻进程缓存，避免每个请求冷启动。

候选质量审计：

旧 gated 审计：

`results/shared/enzyme_candidate_quality_audit_v0_20260528_10p10n_gated/`

v2 审计：

`results/shared/enzyme_candidate_quality_audit_v0_20260528_main_component_v2_10p10n/`

| 指标 | v1 gated | v2 main-component |
|---|---:|---:|
| candidate rows | 606 | 720 |
| search-ready candidates | 161 | 161 |
| strong candidates | 150 | 150 |
| enzyme_precedent ready | 155 | 155 |
| enzyme_precedent positive ready recall | 1.0000 | 1.0000 |
| enzyme_precedent negative ready rate | 0.0000 | 0.0000 |
| main_common_or_cofactor flags | 27 | 12 |

解释：

1. v2 没有盲目增加 search-ready 数量，ready/strong 总量保持不变；
2. `main_common_or_cofactor` 风险从 27 降到 12，说明主反应物选择更稳；
3. 候选总数增加来自主产物组分 fingerprint 让更多带辅因子/小分子副产物的 precedent 能被检索到，但 SP-v1/gate 仍会压住 search-ready 数量；
4. 这一步增强的是酶候选质量和 evidence，不解决 route closure。

## Stage H：Transition Signature v1

问题：

主组分感知解决了“哪个组分是主底物/主产物”，但还没有回答“这个主底物到主产物的变化是否像一个可解释的酶转化”。直接用 atom mapping 目前成本较高，而且公开酶反应的 mapping 质量不完全稳定，所以先加一个轻量 transition signature。

新增：

- `transition_signature(substrate_main, product_main, substrate_aux, product_aux)`
- evidence 字段：`transition_signature`

记录内容：

- 主底物/主产物 Morgan similarity；
- heavy atom delta；
- element delta；
- auxiliary-explained element gains；
- unexplained element gain review；
- motif delta：carbonyl、carboxyl、ester、hydroxyl、amine、phosphate、thioester；
- transition flags：
  - `auxiliary_explains_element_gain`
  - `unexplained_element_gain_review`
  - `weak_main_transition_similarity`
  - `large_main_transition_delta_review`
  - `main_transition_self_loop`

设计原则：

1. 这些是 soft evidence，不是硬拒绝；
2. O、N、P 等元素变化如果能由辅助反应物解释，会记录为 `auxiliary_explains_element_gain`；
3. 对“条件/辅助试剂可能引入元素”的情况保持保守，不再简单按元素变化误杀；
4. route-tree 可见 `score` 仍保持 0-1，避免打坏 proposal probability。

测试：

```text
pytest -q tests/test_enzyme_precedent_retrieval.py tests/test_enzyme_candidate_quality_audit.py tests/test_native_vs_enhanced_route_benchmark.py
13 passed
```

候选质量审计：

输出：

`results/shared/enzyme_candidate_quality_audit_v0_20260528_transition_v1_10p10n/`

| 指标 | main-component v2 | transition v1 |
|---|---:|---:|
| candidate rows | 720 | 720 |
| search-ready candidates | 161 | 161 |
| strong candidates | 150 | 150 |
| enzyme_precedent ready | 155 | 155 |
| enzyme_precedent positive ready recall | 1.0000 | 1.0000 |
| enzyme_precedent negative ready rate | 0.0000 | 0.0000 |
| main_common_or_cofactor | 12 | 12 |
| auxiliary_explains_element_gain | - | 265 |
| weak_main_transition_similarity | - | 25 |
| main_transition_self_loop | - | 2 |
| unexplained_element_gain_review | - | 4 |
| large_main_transition_delta_review | - | 4 |

解释：

transition v1 没有降低正例 recall，也没有增加负例污染；它把 265 个“元素变化可由辅助物解释”的候选显式标出来，同时把少量弱转化、自循环、大 delta 和未解释元素引入候选降为审计风险。

route smoke：

输出：

`results/shared/native_vs_enhanced_route_benchmark_20260528_enhanced_2p2n_transition_v1/`

| 指标 | 数值 |
|---|---:|
| targets with routes | 4/4 |
| solved targets | 0/4 |
| progressive targets | 1/4 |
| enzyme route targets | 2/4 |
| SP-v1 accepted enzyme route targets | 2/4 |
| positive enzyme target recall | 1.0000 |
| negative enzyme target rate | 0.0000 |
| enzyme proposal candidates | 23 |
| SP-v1 rejections | 6 |
| timeout-frontier routes | 4 |

代表性 selected enzyme step：

```text
EC 1.13.11.12
transition_flags = [auxiliary_explains_element_gain]
transition_quality_score = 0.7944
heavy_atom_delta = 2
```

这说明双加氧/过氧化等酶步中的 O 元素引入被 O2 辅助反应物解释，而不是被误判为不守恒。

## Stage I：Native ChemEnzy Chemical Subplanner Probe

目的：

验证 accepted enzyme precursor 闭合失败是否因为前体不可拆，还是增强 route-tree 的 chemical continuation/search value 不够。

新增：

- `scripts/run_enzyme_precursor_expansion_benchmark.py --native-subplanner`
- 对同一批 SP-v1 accepted enzyme precursors 同时运行：
  - route-tree chemical-only continuation；
  - 原生 ChemEnzy 子目标规划。

smoke 输出：

`results/shared/enzyme_precursor_expansion_benchmark_20260528_native_subplanner_transition_v1_smoke_v2/`

结果：

| 指标 | route-tree chemical-only | native ChemEnzy subplanner |
|---|---:|---:|
| subgoals | 2 | 2 |
| with routes | 2/2 | 2/2 |
| solved | 0/2 | 2/2 |
| total routes | 4 | 12 |
| mean steps | 1.25 | 1.0 |
| mean elapsed | 31.027 s | 9.173 s |

结论：

accepted enzyme precursor 不是不可拆；原生 ChemEnzy 在这两个长链多烯/过氧化前体上能快速闭合。当前增强路线的主要短板是 route-tree chemical continuation，而不是酶候选本身。

建议主线调整：

```text
AutoPlanner enzyme layer:
  bridge/EC trigger
  enzyme_precedent proposal
  SP-v1 substrate-product verifier
  transition evidence

ChemEnzy native layer:
  accepted enzyme precursor chemical subplanning
  ZINC stock closure

Hybrid assembly:
  enzyme step + native chemical subroutes merge
```

这保留了增强分支的核心创新：原生 ChemEnzy 没有稳定酶步发现；同时利用原生 ChemEnzy 已经更强的化学闭合能力。
