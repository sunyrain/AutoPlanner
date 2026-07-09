# Enzyme SP Verifier v1 Gate 接入与 Benchmark

生成时间：2026-05-28

## 目标

本阶段目标是把 `enzyme-substrate-product verifier v1` 接到 route-tree 搜索中，并通过 benchmark 判断是否改善路线质量。

## 已完成接入

新增 runtime scorer：

`cascade_planner/cascade_search/enzyme_sp_verifier_v1.py`

接入点：

`cascade_planner/route_tree/search.py`

接入位置在 route-tree action filter 中：

1. proposal source gate 先决定是否调用 enzyme proposal。
2. proposal provider 生成候选 action。
3. route verifier 做基础结构/条件/合同过滤。
4. v1 gate 对 enzymatic 或 bridge-supported enzyme action 打分。
5. score 低于阈值的酶步被过滤。

默认阈值来自训练产物：

`0.36331207712759417`

## 启用方式

环境变量：

```bash
AUTOPLANNER_ENABLE_ENZYME_SP_VERIFIER_V1_GATE=1
AUTOPLANNER_ENZYME_SP_VERIFIER_V1_MODEL=results/shared/enzyme_sp_verifier_v1_20260528/enzyme_sp_verifier_v1_lgbm.joblib
AUTOPLANNER_ENZYME_SP_VERIFIER_V1_THRESHOLD=0.36331207712759417
```

可选控制：

```bash
AUTOPLANNER_ENZYME_SP_VERIFIER_V1_SCOPE=enzyme_or_bridge_supported_enzyme
AUTOPLANNER_ENZYME_SP_VERIFIER_V1_REJECT_BELOW_THRESHOLD=1
AUTOPLANNER_ENZYME_SP_VERIFIER_V1_FAIL_OPEN=1
```

## 统计字段

route-tree stats 现在包含：

- `enzyme_sp_verifier_calls`
- `enzyme_sp_verifier_scored`
- `enzyme_sp_verifier_rejections`
- `enzyme_sp_verifier_errors`

每个被打分 action 的 metadata 会写入：

- `enzyme_sp_verifier_v1.score`
- `enzyme_sp_verifier_v1.threshold`
- `enzyme_sp_verifier_v1.accepted`
- `enzyme_sp_verifier_v1.substrate_smiles`
- `enzyme_sp_verifier_v1.product_smiles`
- `enzyme_sp_verifier_v1.ec_numbers`

## Controlled Route Benchmark

脚本：

`scripts/run_enzyme_sp_v1_gate_benchmark.py`

输出：

`results/shared/enzyme_sp_v1_gate_benchmark_20260528/`

结果：

| policy | selected enzyme | plausible selected | implausible selected | plausible recall | implausible accept rate | mean v1 rejections |
|---|---:|---:|---:|---:|---:|---:|
| bridge_gate_v0_only | 2 | 1 | 1 | 1.0000 | 1.0000 | 0.00 |
| bridge_gate_v0_plus_enzyme_sp_v1 | 1 | 1 | 0 | 1.0000 | 0.0000 | 0.50 |

结论：

在受控路线 benchmark 中，v1 action gate 能把不可信酶步接受率从 1.0 降到 0.0，同时保留可信酶步 recall 1.0。这证明接入路径有效，且 v1 确实能改变路线级选择。

## Live Provider Minismoke

脚本：

`scripts/run_bridge_live_policy_benchmark_v0.py`

输出：

`results/shared/bridge_live_policy_benchmark_v1_sp_gate_20260528_minismoke/`

设置：

- positives: 1
- negatives: 1
- max_depth: 1
- branch_factor: 4
- expansion_budget: 4
- n_results: 2

结果：

| policy | selected targets | true | false | precision | recall | false rate | mean enzyme calls | mean v1 reject |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native_no_enzyme | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 |
| ungated_default_source_gate | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 2.00 | 0.00 |
| bridge_gate_verifier | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.00 | 0.00 |
| bridge_gate_verifier_sp_v1 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 2.00 | 0.00 |
| bridge_gate_verifier_bonus2 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | 1.00 | 0.00 |

观察：

- `bridge_gate_verifier_sp_v1` 在 positive target 上实际调用了 v1：`enzyme_sp_verifier_calls=2`，`scored=2`。
- 该 minismoke 中 v1 没有产生 rejection。
- 样本太小，不能证明真实 live provider 路线质量提升。
- live provider 启动成本明显偏高，正负各 3、深度 2 的 smoke 运行时间过长，已降级为正负各 1、深度 1。

## 当前结论

本阶段完成了“接入”和“可运行验证”：

1. v1 gate 已经接入 route-tree 搜索。
2. controlled benchmark 证明 v1 能降低错误酶步接受率。
3. live provider minismoke 证明真实 provider 路径可加载和调用 v1。

## Live Provider Reuse Benchmark

脚本：

`scripts/run_bridge_live_policy_benchmark_v0.py --reuse-live-engine`

输出：

`results/shared/bridge_live_policy_benchmark_v1_sp_gate_20260528_reuse_3p3n/`

设置：

- positives: 3
- negatives: 3
- max_depth: 1
- branch_factor: 4
- expansion_budget: 4
- n_results: 2
- reuse_live_engine: true

结果：

| policy | selected targets | true | false | precision | recall | false rate | mean enzyme calls | mean v1 reject | mean elapsed s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native_no_enzyme | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 | 4.11 |
| ungated_default_source_gate | 1 | 1 | 0 | 1.0000 | 0.3333 | 0.0000 | 2.00 | 0.00 | 0.63 |
| bridge_gate_verifier | 1 | 1 | 0 | 1.0000 | 0.3333 | 0.0000 | 1.00 | 0.00 | 0.09 |
| bridge_gate_verifier_sp_v1 | 1 | 1 | 0 | 1.0000 | 0.3333 | 0.0000 | 2.00 | 0.33 | 0.09 |
| bridge_gate_verifier_bonus2 | 3 | 3 | 0 | 1.0000 | 1.0000 | 0.0000 | 1.00 | 0.00 | 0.09 |

观察：

- `bridge_gate_verifier_sp_v1` 与 `bridge_gate_verifier` 都选中 1 个正例 target，没有假阳性。
- v1 gate 在真实 provider 输出上发生了过滤：mean v1 rejection = 0.33。
- 具体有 2 个 positive target 出现 v1 rejection，各自 `enzyme_sp_verifier_calls=2`、`rejections=1`。
- 但在 3p3n、depth=1 的小样本上，v1 没有提升 selected target recall，也没有降低 false target rate，因为 false target rate 原本就是 0。
- `bridge_gate_verifier_bonus2` 通过额外 bridge enzyme bonus 把 positive recall 提到 1.0，但这不是 v1 gate 本身的效果。

但还没有完成“真实路线质量显著提升”的证明。当前 live 样本仍偏小，只能说明 v1 在真实 provider 路径中确实会过滤部分酶候选；尚不能证明整体路线成功率或质量显著提升。

## 还差什么

下一步需要扩大真实 benchmark，但必须控制 provider 启动成本：

1. 固定 live engine，只在同一 engine 上切换 policy，避免每个 policy 重复初始化 rulebase。
2. 缓存每个 target 的 proposal pool，再离线重放 source gate + v1 action gate。
3. 对至少 20-50 个 bridge-positive/negative targets 跑路线级对比。
4. 重点指标：
   - selected enzyme route precision
   - false enzyme target rate
   - v1 rejection count
   - route plausibility pass rate
   - runtime / proposal calls

目前不能声称真实 benchmark 已经提升，只能声称：

> v1 gate 已完成接入；受控路线实验显示能过滤错误酶步；真实 provider smoke 可运行但尚未证明质量提升。

## 验证命令

```bash
python -m py_compile cascade_planner/cascade_search/enzyme_sp_verifier_v1.py cascade_planner/route_tree/search.py tests/test_enzyme_sp_verifier_gate.py scripts/run_enzyme_sp_v1_gate_benchmark.py scripts/run_bridge_live_policy_benchmark_v0.py
pytest -q tests/test_enzyme_sp_verifier_gate.py tests/test_enzyme_sp_verifier_v1.py tests/test_route_tree_planner.py -k 'enzyme_sp or bridge_aware_source_gate or bridge_supported_enzyme_bonus'
python scripts/run_enzyme_sp_v1_gate_benchmark.py --output-dir results/shared/enzyme_sp_v1_gate_benchmark_20260528
python scripts/run_bridge_live_policy_benchmark_v0.py --output-dir results/shared/bridge_live_policy_benchmark_v1_sp_gate_20260528_minismoke --positives 1 --negatives 1 --max-depth 1 --branch-factor 4 --expansion-budget 4 --n-results 2
python scripts/run_bridge_live_policy_benchmark_v0.py --reuse-live-engine --output-dir results/shared/bridge_live_policy_benchmark_v1_sp_gate_20260528_reuse_3p3n --positives 3 --negatives 3 --max-depth 1 --branch-factor 4 --expansion-budget 4 --n-results 2
```
