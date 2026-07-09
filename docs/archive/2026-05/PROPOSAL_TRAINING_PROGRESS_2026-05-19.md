# Proposal Training Progress

更新时间：2026-05-19

## 当前训练任务

`chem_enzy_plain_continue_v3_lr5e5_1000`

目标：先做能直接影响 proposal 层的训练，而不是继续做 fixed-pool rerank。该任务使用 ChemEnzy vendored OpenNMT one-step proposer，从现有 `model_step_100000.pt` 继续训练 product -> reactants。

当前候选 checkpoint 分工：

- v3 step700: `results/shared/proposal_training_20260519/chem_enzy_plain_continue_v3_lr5e5_1000/checkpoints/plain_continue_lr5e5_step_700.pt`
- v2 step300: `results/shared/proposal_training_20260519/chem_enzy_plain_continue_v2_lr1e4/checkpoints/plain_continue_lr1e4_step_300.pt`

选择结论：v3 step700 在 one-step exact 与 gold10 route recovery 上最好；v2 step300 在 locked24 上的 GT reaction hit 更好但 solved 略降；base ChemEnzy ONMT ensemble 仍然是 solvability 最稳的配置。当前不能把任何 trained checkpoint 作为无条件替换，应作为 sidecar/rescue proposal 源继续验证。

## 为什么先做这个

- CCTS/reranker 只能在已有候选池内排序，不能生成候选池里没有的真实反应物。
- 当前最直接能改变候选生成分布的入口是 ChemEnzy/OpenNMT one-step proposal model。
- `plain` corpus 已通过 OpenNMT preprocess 和 1-step resume smoke；继续训练需要 `-reset_optim all`。

## 输入

| 项 | 路径 / 数值 |
| --- | --- |
| corpus manifest | `results/shared/cascade_verifier_proof_20260519/chem_enzy_onmt_corpus_v4_30k/manifest.json` |
| train src/tgt | `plain.train.src` / `plain.train.tgt` |
| valid src/tgt | `plain.valid.src` / `plain.valid.tgt` |
| preprocessed prefix | `results/shared/cascade_verifier_proof_20260519/chem_enzy_onmt_corpus_v4_30k/onmt_preprocess_smoke/plain` |
| seed checkpoint | `vendor/ChemEnzyRetroPlanner/retro_planner/packages/onmt/checkpoints/np-like/model_step_100000.pt` |
| source seed routes | 1477 |
| plain examples | 2423 |

## 运行状态

| 时间 | 状态 | 证据 |
| --- | --- | --- |
| 2026-05-19 | 准备启动 | preprocessed ONMT shards 和 checkpoint 均存在 |
| 2026-05-19 | `chem_enzy_plain_continue_v1` 完成 | 300 step 跑通，产出 step100/200/300 checkpoints；默认 LR=1.0，valid 指标波动大 |
| 2026-05-19 | `chem_enzy_plain_continue_v2_lr1e4` 完成 | 300 step 跑通，低 LR 稳定收敛；step300 显著强于 base |
| 2026-05-19 | 训练 checkpoint 接入 proposal provider | `AUTOPLANNER_CHEMENZY_ONMT_MODEL_PATH` 可指定 trained ONMT checkpoint；真实 provider 加载 step700 成功并返回候选 |
| 2026-05-19 | `chem_enzy_plain_continue_v3_lr5e5_1000` 完成 | 1000 step 跑通；step700 exact 最优，step800 后开始出现 exact 退化 |
| 2026-05-19 | locked24 route smoke 完成 | base、v2、v3、base+v2 ensemble 均已跑；结果显示 trained checkpoint 不能直接替换 base ensemble |

## 训练结果

### chem_enzy_plain_continue_v1

目的：验证从 ChemEnzy checkpoint 正式 continue-train 能完整跑通。

命令要点：

- `-train_from vendor/.../model_step_100000.pt`
- `-reset_optim all`
- `-train_steps 300`
- 默认 `learning_rate=1.0`

输出：

| checkpoint | valid ppl | valid acc | 备注 |
| --- | ---: | ---: | --- |
| `plain_continue_step_100.pt` | 685.484 | 11.7403 | 不稳定 |
| `plain_continue_step_200.pt` | 10205.7 | 8.80829 | 不稳定 |
| `plain_continue_step_300.pt` | 21.4248 | 16.3852 | ppl 下降但 acc 仍差 |

结论：链路可跑，但默认学习率过激，不作为当前候选模型。

### chem_enzy_plain_continue_v2_lr1e4

目的：做第一轮有意义的 proposal-level supervised continue-train。

命令要点：

- `-train_from vendor/.../model_step_100000.pt`
- `-reset_optim all`
- `-optim adam`
- `-learning_rate 0.0001`
- `-decay_method none`
- `-max_grad_norm 1`
- `-train_steps 300`
- `-valid_steps 50`
- `-save_checkpoint_steps 100`

输出目录：

`results/shared/proposal_training_20260519/chem_enzy_plain_continue_v2_lr1e4/`

结果：

| step | train acc | train ppl | valid ppl | valid acc |
| ---: | ---: | ---: | ---: | ---: |
| 50 | - | - | 13.0948 | 70.8138 |
| 100 | 76.25 | 3.30 | 3.66853 | 73.1972 |
| 150 | - | - | 2.30718 | 79.0917 |
| 200 | 84.13 | 1.73 | 2.06281 | 81.1033 |
| 250 | - | - | 1.98798 | 81.9201 |
| 300 | 87.48 | 1.48 | 1.9586 | 82.4931 |

当前选择：

`results/shared/proposal_training_20260519/chem_enzy_plain_continue_v2_lr1e4/checkpoints/plain_continue_lr1e4_step_300.pt`

选择理由：验证集 ppl 最低、验证 acc 最高，且训练曲线稳定。

### chem_enzy_plain_continue_v3_lr5e5_1000

目的：在 v2 证明有效后，做更长、更低学习率的 proposal-level continue-train，观察是否还能提高候选生成 exact recall。

命令要点：

- `-train_from vendor/.../model_step_100000.pt`
- `-reset_optim all`
- `-optim adam`
- `-learning_rate 0.00005`
- `-decay_method none`
- `-max_grad_norm 1`
- `-train_steps 1000`
- `-valid_steps 100`
- `-save_checkpoint_steps 100`

输出目录：

`results/shared/proposal_training_20260519/chem_enzy_plain_continue_v3_lr5e5_1000/`

结果：

| step | train acc | train ppl | valid ppl | valid acc |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 75.29 | 7.62 | 10.3653 | 71.2710 |
| 200 | 76.97 | 3.17 | 3.51861 | 73.2338 |
| 300 | 81.75 | 1.99 | 2.25198 | 79.2563 |
| 400 | 84.75 | 1.68 | 2.04717 | 81.0546 |
| 500 | 86.44 | 1.54 | 1.97809 | 82.0603 |
| 600 | 87.87 | 1.45 | 1.95242 | 82.7248 |
| 700 | 88.81 | 1.40 | 1.94834 | 83.2429 |
| 800 | 89.97 | 1.34 | 1.95485 | 83.4197 |
| 900 | 90.92 | 1.30 | 1.98908 | 83.5172 |
| 1000 | 91.62 | 1.27 | 2.01069 | 83.5233 |

结论：token accuracy 到 1000 还在升，但 validation ppl 从 800 开始回升，exact recall 也从 800/1000 开始下降；因此不选最终 step1000，当前选 step700。

## Proposal Inference 对比

为了避免只看 teacher-forced token accuracy，本轮还用 ChemEnzy vendored `onmt.bin.translate` 推理函数，对同一 `plain.valid` 集做了 base vs continue checkpoint 的 canonical reactant exact 对比。评估脚本已固化为：

`scripts/evaluate_chem_enzy_onmt_checkpoint_exact.py`

评估设置：

- valid examples: 414
- canonical exact：对 predicted reactants 和 target reactants 做 RDKit canonical SMILES 后比较，reactant 顺序排序。

beam5/top5 结果：

| model | nonempty | exact top1 | exact top5 | top1 rate | top5 rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| base `model_step_100000.pt` | 411 / 414 | 5 | 22 | 0.0121 | 0.0531 |
| `plain_continue_lr1e4_step300` | 376 / 414 | 39 | 72 | 0.0942 | 0.1739 |
| `plain_continue_lr5e5_step700` | 355 / 414 | 42 | 72 | 0.1014 | 0.1739 |
| `plain_continue_lr5e5_step800` | 356 / 414 | 40 | 68 | 0.0966 | 0.1643 |
| `plain_continue_lr5e5_step1000` | 352 / 414 | 38 | 65 | 0.0918 | 0.1570 |

beam20/top20 结果：

| model | nonempty | exact top1 | exact top20 | top1 rate | top20 rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| base `model_step_100000.pt` | 413 / 414 | 5 | 54 | 0.0121 | 0.1304 |
| `plain_continue_lr1e4_step300` | 404 / 414 | 39 | 100 | 0.0942 | 0.2415 |
| `plain_continue_lr5e5_step700` | 398 / 414 | 42 | 100 | 0.1014 | 0.2415 |

解释：

- 这证明训练不只是跑通了 loss；它确实改变了 proposal 生成，并显著提高了当前 validation corpus 上的 reactant exact recall。
- 从 route-search 候选池角度看，base 的 top20 exact 为 54/414，trained checkpoint 提高到 100/414，约 1.85 倍。
- v3 step700 相比 v2 step300 主要提升 top1 exact；top5/top20 持平，因此它是一跳 exact 最强 checkpoint，但 route-level 仍需按 benchmark 分场景选择。
- 但 absolute recall 仍然不高，尤其 multi-reactant / cofactor-heavy 样本仍弱。
- route-level 已补 gold10 与 locked24；结论是 trained checkpoint 有增益，但不能直接无条件替换 base ensemble。

## Proposal Provider 接入

已完成：

- `cascade_planner/baselines/chem_enzy_adapter.py` 增加 `onmt_model_path` / `AUTOPLANNER_CHEMENZY_ONMT_MODEL_PATH`。
- `cascade_planner/baselines/chem_enzy_onestep.py` 的 `from_env()` 可以读取 trained checkpoint。
- `cascade_planner/eval/audit_chem_enzy_transition_coverage.py` 增加 `--onmt-model-path`，cache key 包含 checkpoint，避免 base/trained 结果混淆。
- 单元测试通过：`test_vendor_config_can_override_onmt_checkpoint_path`、`test_chem_enzy_onestep_provider_reads_trained_checkpoint_env`。
- 真实 provider 验证通过：step700 checkpoint 可加载并对 `OC(O)C1CCCCC1` 返回 5 条候选，`load_error` 为空。

## Route-Level Smoke Benchmark

为了确认训练不只提升 one-step validation，本轮补了一个小规模 route-level smoke。设置如下：

- benchmark: `data/benchmark_cascade_gold_smoke_v1.json`
- targets: 10
- one-step model: `onmt_models.bionav_one_step` only
- iterations: 20
- max depth: 6
- expansion topk: 20
- gpu: 0
- 对比对象：base ONMT、v2 step300、v3 step700

输出：

| run | output |
| --- | --- |
| base | `results/shared/proposal_training_20260519/route_smoke_onmt_base_gold10.json` |
| v2 step300 | `results/shared/proposal_training_20260519/route_smoke_onmt_v2_step300_gold10.json` |
| v3 step700 | `results/shared/proposal_training_20260519/route_smoke_onmt_v3_step700_gold10.json` |
| comparison | `results/shared/proposal_training_20260519/route_smoke_onmt_gold10_comparison.json` |
| comparison md | `results/shared/proposal_training_20260519/route_smoke_onmt_gold10_comparison.md` |

结果：

| run | solved | total routes | avg routes | exact rxn target hit | exact full route | avg best rxn frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 10 / 10 | 1178 | 117.8 | 5 / 10 | 1 / 10 | 0.30 |
| v2 step300 | 10 / 10 | 2381 | 238.1 | 8 / 10 | 1 / 10 | 0.45 |
| v3 step700 | 10 / 10 | 2102 | 210.2 | 8 / 10 | 2 / 10 | 0.50 |

解释：

- 这个 smoke 不能替代完整 benchmark，但它说明 trained proposal 确实进入了 route search，并提高了 gold reaction recovery。
- solved rate 三者都是 10/10，因此 solved 在这个小集合上没有区分度。
- v2 step300 生成路线更多；v3 step700 生成路线略少，但 exact full-route hit 和 avg best rxn fraction 更好。
- 因此在 gold10 上 v3 step700 最强，v2 step300 是候选池扩张 fallback。

### Locked Validation 24

进一步扩到 `data/benchmark_locked_validation_20260508.json` 的 24 个目标，仍使用 ONMT-only、`iterations=20/max_depth=6/expansion_topk=20/gpu=0`。额外增加了一个 `base_plus_v2` 配置：原 ChemEnzy ONMT 4-checkpoint ensemble + v2 step300。

输出：

| run | output |
| --- | --- |
| base | `results/shared/proposal_training_20260519/route_smoke_onmt_base_locked24.json` |
| v2 step300 | `results/shared/proposal_training_20260519/route_smoke_onmt_v2_step300_locked24.json` |
| v3 step700 | `results/shared/proposal_training_20260519/route_smoke_onmt_v3_step700_locked24.json` |
| base + v2 | `results/shared/proposal_training_20260519/route_smoke_onmt_base_plus_v2_locked24.json` |
| comparison | `results/shared/proposal_training_20260519/route_smoke_onmt_locked24_comparison_all_plus_ensemble.json` |
| comparison md | `results/shared/proposal_training_20260519/route_smoke_onmt_locked24_comparison_all_plus_ensemble.md` |

结果：

| run | solved | total routes | avg routes | exact rxn target hit | exact full route | avg best rxn frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 24 / 24 | 3206 | 133.58 | 2 / 24 | 0 / 24 | 0.0417 |
| v2 step300 | 23 / 24 | 2575 | 107.29 | 3 / 24 | 0 / 24 | 0.0625 |
| v3 step700 | 22 / 24 | 2502 | 104.25 | 2 / 24 | 0 / 24 | 0.0417 |
| base + v2 | 24 / 24 | 3807 | 158.63 | 2 / 24 | 0 / 24 | 0.0417 |

解释：

- locked24 与 gold10 的结论不同：v3 step700 不能直接替代 base，solved 从 24/24 降到 22/24，GT recovery 没提升。
- v2 step300 在 locked24 上多恢复了 1 个目标的 GT reaction hit，但 solved 从 24/24 降到 23/24。
- base+v2 ensemble 保住 solved 24/24，但没有提高 GT recovery，且更慢、路线更多。
- 直接替换 checkpoint 或简单拼接 checkpoint 都不是最终产品形态。更合理的下一步是把 trained proposal 作为 sidecar/rescue source，在 base search 失败或 GT-like transformation 缺失时注入，而不是替代 base ONMT。

## Sidecar Route-Pool Merge

根据 locked24 结果，直接替换 checkpoint 会牺牲 solved，简单加入 ONMT ensemble 又不提升 recovery。于是补了一个更贴近产品形态的 sidecar merge：base ChemEnzy 作为 primary route pool，trained checkpoint 作为 sidecar route pool，最后做路线池并集。

新增脚本：

`scripts/merge_chem_enzy_sidecar_routes.py`

### gold10: base + v3 sidecar

输出：

| artifact | path |
| --- | --- |
| merged pool | `results/shared/proposal_training_20260519/route_smoke_onmt_base_with_v3_sidecar_gold10.json` |
| comparison | `results/shared/proposal_training_20260519/route_smoke_onmt_gold10_sidecar_comparison.json` |
| comparison md | `results/shared/proposal_training_20260519/route_smoke_onmt_gold10_sidecar_comparison.md` |

结果：

| run | solved | total routes | avg routes | exact rxn target hit | exact full route | avg best rxn frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 10 / 10 | 1178 | 117.8 | 5 / 10 | 1 / 10 | 0.30 |
| v3 step700 | 10 / 10 | 2102 | 210.2 | 8 / 10 | 2 / 10 | 0.50 |
| base + v3 sidecar | 10 / 10 | 3280 | 328.0 | 8 / 10 | 3 / 10 | 0.55 |

### locked24: base + v2 sidecar

输出：

| artifact | path |
| --- | --- |
| merged pool | `results/shared/proposal_training_20260519/route_smoke_onmt_base_with_v2_sidecar_locked24.json` |
| comparison | `results/shared/proposal_training_20260519/route_smoke_onmt_locked24_sidecar_comparison.json` |
| comparison md | `results/shared/proposal_training_20260519/route_smoke_onmt_locked24_sidecar_comparison.md` |

结果：

| run | solved | total routes | avg routes | exact rxn target hit | exact full route | avg best rxn frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 24 / 24 | 3206 | 133.58 | 2 / 24 | 0 / 24 | 0.0417 |
| v2 step300 | 23 / 24 | 2575 | 107.29 | 3 / 24 | 0 / 24 | 0.0625 |
| base + v2 sidecar | 24 / 24 | 5781 | 240.88 | 3 / 24 | 0 / 24 | 0.0625 |

结论：

- sidecar merge 保留 base 的 solved 稳定性，同时吸收 trained checkpoint 的额外 GT recovery。
- 代价是路线池变大，下一步需要 selector/verifier 做 pool pruning，而不是继续把 trained checkpoint 直接塞进 ChemEnzy ONMT ensemble。

## Sidecar Pool Pruning

为了控制 sidecar route pool 的规模，新增 source-aware quota pruning：先按反应序列去重，然后保留 primary 与 sidecar 各自的前若干路线。

新增脚本：

`scripts/prune_chem_enzy_sidecar_routes.py`

### gold10 pruning

配置：`primary_keep=80`、`sidecar_keep=80`、`max_routes=160`。

输出：

| artifact | path |
| --- | --- |
| pruned pool | `results/shared/proposal_training_20260519/route_smoke_onmt_base_with_v3_sidecar_gold10_pruned_p80_s80.json` |
| comparison | `results/shared/proposal_training_20260519/route_smoke_onmt_gold10_sidecar_pruned_comparison.json` |
| comparison md | `results/shared/proposal_training_20260519/route_smoke_onmt_gold10_sidecar_pruned_comparison.md` |

结果：

| pool | solved | total routes | avg routes | exact rxn target hit | exact full route | avg best rxn frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| merged | 10 / 10 | 3280 | 328.0 | 8 / 10 | 3 / 10 | 0.55 |
| pruned | 10 / 10 | 1551 | 155.1 | 8 / 10 | 3 / 10 | 0.55 |

### locked24 pruning

配置：`primary_keep=120`、`sidecar_keep=80`、`max_routes=200`。

输出：

| artifact | path |
| --- | --- |
| pruned pool | `results/shared/proposal_training_20260519/route_smoke_onmt_base_with_v2_sidecar_locked24_pruned_p120_s80.json` |
| comparison | `results/shared/proposal_training_20260519/route_smoke_onmt_locked24_sidecar_pruned_comparison.json` |
| comparison md | `results/shared/proposal_training_20260519/route_smoke_onmt_locked24_sidecar_pruned_comparison.md` |

结果：

| pool | solved | total routes | avg routes | exact rxn target hit | exact full route | avg best rxn frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| merged | 24 / 24 | 5781 | 240.88 | 3 / 24 | 0 / 24 | 0.0625 |
| pruned | 24 / 24 | 3632 | 151.33 | 3 / 24 | 0 / 24 | 0.0625 |

结论：

- source-aware pruning 能显著降低路线池规模，并保留目前观察到的 recovery 增益。
- 这比直接替换 checkpoint 更安全，也比无剪枝 sidecar 更接近可部署状态。

## 明确边界

- 这不是 DPO。
- 这不是 `context` cascade-aware 训练。
- 这不是最终模型，只是 proposal-level supervised continue-train 的第一轮有效实训。
- 当前 exact 指标是一跳 product -> reactants validation，不等价于 route-level benchmark。
- 已接回 ChemEnzy proposal inference，并完成 gold10 与 locked24 route-level smoke。
- 已在 v2_100 前 20 个目标上验证 pruned sidecar route pool；该批次没有产生新增 GT recovery。
- 下一步扩大到 v2_100 分片验证，重点看 trained sidecar 的增益是否只集中在 gold10/locked24，还是能在更大 benchmark 上稳定出现。

## v2_100 First20 Route Smoke

配置：`benchmark=data/benchmark_v2_100.json`、`limit=20`、`iterations=20`、`max_depth=6`、`expansion_topk=20`、ONMT-only proposal。

输出：

| artifact | path |
| --- | --- |
| base | `results/shared/proposal_training_20260519/route_smoke_onmt_base_v2_100_first20.json` |
| v2 sidecar source | `results/shared/proposal_training_20260519/route_smoke_onmt_v2_step300_v2_100_first20.json` |
| merged pool | `results/shared/proposal_training_20260519/route_smoke_onmt_base_with_v2_sidecar_v2_100_first20.json` |
| pruned pool | `results/shared/proposal_training_20260519/route_smoke_onmt_base_with_v2_sidecar_v2_100_first20_pruned_p120_s80.json` |
| comparison | `results/shared/proposal_training_20260519/route_smoke_onmt_v2_100_first20_sidecar_comparison.json` |
| comparison md | `results/shared/proposal_training_20260519/route_smoke_onmt_v2_100_first20_sidecar_comparison.md` |

结果：

| run | solved | total routes | avg routes | exact rxn target hit | exact full route | avg best rxn frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 20 / 20 | 3009 | 150.45 | 4 / 20 | 0 / 20 | 0.070833 |
| v2 step300 | 20 / 20 | 3182 | 159.10 | 4 / 20 | 0 / 20 | 0.070833 |
| base + v2 sidecar | 20 / 20 | 6191 | 309.55 | 4 / 20 | 0 / 20 | 0.070833 |
| base + v2 sidecar pruned | 20 / 20 | 3527 | 176.35 | 4 / 20 | 0 / 20 | 0.070833 |

结论：

- 这一批次里，v2 checkpoint 没有带来新增 route-level GT reaction recovery。
- sidecar merge/pruning 保住 solved 20/20，但只是增加候选池，没有增加命中。
- 这说明 proposal continue-train 的收益不是全局稳定的；继续扩大训练/评估时，必须同时报告 route-level recovery，而不能只看一跳 exact validation。
