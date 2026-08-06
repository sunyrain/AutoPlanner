# W7 RetroStar-190 冻结与预评测门

日期：2026-08-06  
冻结 runtime commit：`1bff6a3`  
状态：W7 完成；冻结清单、完整离线门、190/190 blind preflight 与最终 Nirmatrelvir 零模型回放均已通过。W8 尚未开始。

## 1. 本阶段完成了什么

W7 不运行 RetroStar-190 正式结果，而是先固定正式运行所依赖的代码、输入、库存、配置、调度器、环境与 provider 身份，并证明全部 190 个目标都能通过 fresh-blind 运行前审计。

机器可读冻结清单为：

- `benchmarks/retrostar190_w7_freeze_20260806.json`

它冻结或记录：

- runtime commit/tree 与关键模块 SHA-256；
- 190-target manifest、protocol 和 23,081,629-member stock index；
- `gpt-5.5`、low reasoning、standard profile、单 panel worker、无 visual 的统一配置；
- 所有 190 个 case 相同的预算与 acceptance projection；
- deterministic target-blind scheduler 的全部显式系数、state bonus 与 tie-break；
- host/ChemEnzy Python、RDKit 与 Codex CLI 摘要；
- 离线测试、blind preflight 和三个既有阳性 smoke 的证据位置。

## 2. 可复现性边界

本冻结是诚实的 content-addressed 工程冻结，不虚构外部 provider 的位级确定性：

- host scheduler 不使用 RNG；相同状态、资源和 action set 产生相同排序；
- 当前 panel contract 没有暴露 provider seed，因此没有可声明的冻结 seed 集合；
- 远端模型标识、reasoning effort、执行入口和本地可执行文件已记录；
- 远端模型权重和 provider sampling 不能由本仓库位级冻结。

因此，正式复现应区分两层：host 决策与 canonical ingestion 可由 commit/config/hash 审计；远端生成结果则必须依靠保存 raw provider payload、digest 与完整 action trace 审计，不能承诺逐 token 重现。

## 3. 完整离线门

W7 扩展聚焦集合：

```text
74 passed
```

除用户批准延期的超行数预算测试外，完整离线套件：

```text
2632 passed
3 skipped
1 deselected
2 subtests passed
11 warnings
```

命令：

```text
python -m pytest -q -k "not test_new_focused_modules_stay_within_practical_line_budgets"
```

唯一明确延期项：

```text
tests/test_v4_architecture.py::test_new_focused_modules_stay_within_practical_line_budgets
```

进入 W7 时的 7 个基线失败已全部清除；完整套件在修复 source-route replay 幂等性后通过。

## 4. 190/190 blind preflight

最终 fresh preflight：

- Root：`results/.autoplanner/w7-freeze-preflight-20260806-b`
- Snapshot：`results/.autoplanner/w7-freeze-preflight-20260806-b/snapshots/benchmark-snapshot.json`
- Panel status：`results/.autoplanner/w7-freeze-preflight-20260806-b/panel-status.json`
- 结果：190 passed、0 failed、0 provider runs

关键摘要：

| 项目 | 值 |
| --- | --- |
| Snapshot content SHA-256 | `e06a48597a8b4a1aef8ce453100f587037d8f61df855252bcfbfdb9869cbf938` |
| Snapshot file SHA-256 | `f871f277b3926d0e48fc521f3dab115c10b7dc4c0c13af83ddfee485b8c51d8e` |
| Base environment SHA-256 | `0d5204f3178b5d292c19d51b0114e117774795ab454c25ae1f492abb38e6622f` |
| Target manifest SHA-256 | `2d31de46f20cac4dec3c89f822d9059c4fa6ee68f43261929ba5c6b06a4f7623` |
| Protocol SHA-256 | `5b18d6f96cac04d15e86612f44cb73e0f32d6f588eb82b0e063b71d9e7982fd4` |
| Stock SHA-256 | `30c828d6780e534d8368f4eb74f844c889683453080d44053ba298a7bebdd79c` |

第一次 preflight 的 189/190 不是算法失败，而是 repository narrative 中残留了 target 001 exact SMILES。删除该盲测泄漏后，新目录下的完整 preflight 达到 190/190；这证明 preflight 会真实阻断仓库泄漏，而不是形式化盖章。

## 5. Scheduler 冻结

当前 scheduler 是解释性确定性函数，不是训练模型。正式冻结包括：

- route/proof/diversity gain 系数；
- cost/failure-risk penalty；
- B1/B2/B3/B4 状态 bonus；
- dependency penalty；
- action kind order 与 stable action-ID tie-break；
- scheduler module SHA-256：`cc468dad88373251828f4fcbc326c8cb4a67ceaa4da180c821497676d9cad595`。

这些字段只读取 canonical 状态、action opportunity 和资源可用性，不读取 dataset ID、target index、target name 或 RetroStar 标签。

## 6. 已保存的阳性证据

W7 不重复消耗 RetroStar-001 parity：

- RetroStar-001：`results/.autoplanner/benchmark190-optimized-20260727/retrostar001-search-only-v2`
- RetroStar-002/003：`results/.autoplanner/benchmark190-optimized-20260727/retrostar002-003-search-only`

三个 artifact 的 B4 均为 true。它们是已有阳性 smoke，不应被表述为 W8 全量结果，也不证明 190-target 整体提升。

## 7. 最终零模型回放与 W7 关闭

在后续 W7 修复后的 runtime 上完成了 Nirmatrelvir cached-provider zero-model replay：

- Root：`results/.autoplanner/w6-nirmatrelvir-current-replay-20260806-f`
- Report：`results/.autoplanner/w6-nirmatrelvir-current-replay-20260806-f/runs/w6-nirmatrelvir-current-replay-20260806-f--a71e797af896/target-only-solve-report.json`
- Comparison：`results/.autoplanner/w6-nirmatrelvir-current-replay-20260806-f/chemenzy-embedding-comparison.json`

结果：

| 项目 | 结果 |
| --- | ---: |
| 新模型调用 | 0 |
| Standalone / embedded routes | 39 / 39 |
| Raw provider digest parity | true |
| Normalized route multiset parity | true |
| Host selected / fully materialized | 2 / 2 |
| Host validated | 0 |
| Stock closed | 1 |
| B4 | true |

Action 计数保持为 64：`chemenzy_target_expand=1`、`codex_global_architecture=1`、`host_materialize=30`、`reaction_validate=30`、`stock_audit=2`。严格 B2/B3/B5 仍为 false，没有为了 benchmark 指标放宽 validator 或 evidence policy。

至此 W7 已关闭，可以进入 W8。RetroStar-190 正式运行与全目标消融尚未开始，当前仍不得宣称 benchmark-wide improvement。
