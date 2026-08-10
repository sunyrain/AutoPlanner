# RetroStar-190 W8 前 20 目标四臂 Pilot 结果

日期：2026-08-08

结论状态：pilot 已完成；可用于工程决策，不可表述为 RetroStar-190 全量结果或 benchmark-wide improvement。

## 1. 运行契约

- 结果根：`results/.autoplanner/retrostar190-w8-pilot-20260807-a`
- 目标：冻结 manifest 顺序的前 20/190；四臂使用完全相同的 case IDs，不按结果挑样。
- 四臂：`chemenzy-only`、`codex-only`、`unified-round-robin`、`unified-adaptive`。
- 每臂：20 completed、0 failed/incomplete；共 80 个完成 case。
- 代码提交：`225df39`。
- Manifest SHA-256：`2d31de46f20cac4dec3c89f822d9059c4fa6ee68f43261929ba5c6b06a4f7623`。
- Stock SHA-256：`30c828d6780e534d8368f4eb74f844c889683453080d44053ba298a7bebdd79c`。
- Base environment SHA-256：`2faccfc0f35149074164bf24038b22b2ed6ca7054002c5cc2864d454afefb516`。
- 模型：`gpt-5.5`，reasoning effort `low`；profile `standard`；每臂 1 worker。
- 主指标：B4，即至少一条 target-rooted、host-admitted 且终端叶全部命中冻结库存的路线。

## 2. 核心结果

| Arm | B4 solved | 结构路线 | B2 host-validated | 预算内 | 平均耗时 | 中位耗时 | 模型调用 | 输入 / 输出 tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ChemEnzy-only | 16/20 (80%) | 17/20 | 1/20 | 20/20 | 484.302 s | 486.899 s | 0 | 0 / 0 |
| Codex-only | 1/20 (5%) | 17/20 | 4/20 | 17/20 | 384.930 s | 390.024 s | 26 | 1,451,287 / 259,956 |
| Unified round-robin | 15/20 (75%) | 19/20 | 4/20 | 18/20 | 753.446 s | 663.539 s | 22 | 977,280 / 175,322 |
| Unified adaptive | 15/20 (75%) | 18/20 | 3/20 | 19/20 | 598.437 s | 561.554 s | 21 | 705,299 / 155,195 |

B3 exact-source 与 B5 configured scientific acceptance 在四臂均为 0/20。它们必须与 B4 分开解释；pilot 只证明库存闭合搜索能力，不证明实验可执行性或文献证据闭合。

## 3. 逐目标配对比较

参考臂为 `unified-adaptive`，置信区间是固定 seed、5,000 次 paired bootstrap 在这 20 个 case 上的区间。

| Comparator | Adaptive − comparator | 95% CI | wins / losses / ties |
| --- | ---: | ---: | ---: |
| ChemEnzy-only | -5 percentage points | [-15, 0] pp | 0 / 1 / 19 |
| Codex-only | +70 percentage points | [50, 90] pp | 14 / 0 / 6 |
| Unified round-robin | 0 percentage points | [-15, 15] pp | 1 / 1 / 18 |

Adaptive 与 round-robin 的 B4 完全持平，但平均总耗时少 155.009 s/target（约 20.6%），总输入 token 少 271,981、模型调用少 1。这个结果支持“adaptive 更省资源”的局部工程判断，但不支持“adaptive 提高成功率”。

## 4. 失败分类

| Arm | solved | provider 无候选 | stock miss | canonical merge/search-depth | 无结构路线未分类 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ChemEnzy-only | 16 | 2 | 1 | 1 | 0 |
| Codex-only | 1 | 0 | 16 | 0 | 3 |
| Unified round-robin | 15 | 1 | 4 | 0 | 0 |
| Unified adaptive | 15 | 1 | 3 | 1 | 0 |

## 5. 可用结论与停止判定

1. 当前 B4 能力主要由 ChemEnzy native search 提供；Codex-only 虽能形成结构路线，却几乎不能在冻结库存上闭合。
2. 在该 20-case pilot 中，加入 Codex 与统一 host workflow 没有超过 ChemEnzy-only：adaptive 少解 1 个，且没有产生任何 adaptive-only 胜例。
3. Adaptive scheduler 相对 round-robin 的价值目前体现在资源效率，不在 B4 成功率；两者各赢 1 个 case，净差为 0。
4. 当前最值得继续优化的是 unified ingestion/stock closure 与 action 资源分配，而不是继续扩大 Codex-only 推理预算。
5. 因为主要排序已经稳定为 ChemEnzy-only 80% > unified 75% >> Codex-only 5%，且 adaptive 对 round-robin 的成功率增益为 0，继续跑到 190 的边际工程信息不足以抵消机器时间。本轮按用户决策在 20/190 停止。

## 6. 论断边界

- 这 20 个目标是 manifest 前缀，不是随机样本；bootstrap CI 只描述该配对 panel，不能外推到全部 190。
- 不得与 ChemEnzy 论文在完整数据集、不同轮数或不同求解协议下的数字作直接胜负比较。
- 不得声称 AutoPlanner 已达到或超过 ChemEnzy 的 full-benchmark success rate。
- 若论文需要 benchmark-wide 主张，仍须重新执行冻结的 190×4 协议；本次停止不改变该科学门，只改变当前工程范围。

## 7. 机器可读产物

- `summary/w8-run-manifest.json`：SHA-256 `02fc42eb6746f685c6274322801c80354fb81b0d4e61a9faa6fd55f3899e89d1`
- `summary/w8-per-target-metrics.json`：SHA-256 `7828685c89421d3c8def6cbe33816f6d2324185b12db305e819e984cf18b7ba6`
- `summary/w8-paired-comparison.json`：SHA-256 `ef96c32393e2652cef3ecb8d9ef11e03e384d8a2a7c46058f12b645947589578`
- `summary/w8-failure-taxonomy.json`：SHA-256 `18a94798884e884902da0c298d1fd32b841da144b63835987bc12ff2489b867a`
- `summary/w8-panel-summaries.json`：SHA-256 `f189513fd3af4ae77d9894deefef3d487fc6362d90371f4f376981608661c10a`

生成命令：

```text
python scripts/summarize_retrostar190_w8.py \
  --w8-root results/.autoplanner/retrostar190-w8-pilot-20260807-a \
  --output-root results/.autoplanner/retrostar190-w8-pilot-20260807-a/summary
```
