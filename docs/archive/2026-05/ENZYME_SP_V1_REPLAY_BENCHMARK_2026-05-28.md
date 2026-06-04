# Enzyme SP-v1 Replay Benchmark

日期：2026-05-28

## 目的

本轮不是再跑一个完整多步 solved-rate benchmark，而是把 live proposal 候选池先缓存下来，再离线比较不同 gate/verifier 策略。

这样可以拆开两个问题：

1. proposal 候选池里有没有酶候选；
2. bridge gate / enzyme-substrate-product verifier 能不能在同一批候选上减少错误酶步。

## 新增内容

- `scripts/build_live_proposal_replay_pack.py`
  - 一次性调用 live proposal providers；
  - 保存 root/no-EC 候选和 bridge-derived EC context 候选；
  - 输出可复用 JSONL proposal replay pack。

- `scripts/replay_enzyme_sp_gate_policies.py`
  - 在同一候选池上离线比较：
    - `ungated_all`
    - `bridge_gate_v0`
    - `bridge_gate_v0_sp_v1_hard`
    - `bridge_gate_v0_sp_v1_soft`

- `tests/test_replay_enzyme_sp_gate_policies.py`
  - 验证 SP-v1 hard gate 能拒绝低分酶候选；
  - 验证无 bridge hit 时 bridge gate 会拒绝酶候选。

## 小规模结果

输入：

- 3 个 bridge-positive target
- 3 个 chemical negative target
- 每个 source top-k = 6
- 输出目录：`results/shared/live_proposal_replay_pack_v1_20260528_3p3n`

候选池：

| source | actions |
|---|---:|
| retrochimera | 36 |
| chemtemplates | 36 |
| enzyformer | 46 |
| enzexpand | 47 |
| v3_retrieval | 40 |
| chem_enzy_onestep | 0 |
| retrorules | 0 |

总计：

- targets: 6
- contexts: 10
- actions: 205
- live pack 构建耗时：66.222 s

Replay 结果：

| policy | selected targets | true | false | precision | recall | false rate | kept enzyme | SP calls | SP reject |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ungated_all | 6 | 3 | 3 | 0.5000 | 1.0000 | 1.0000 | 22.17 | 0.00 | 0.00 |
| bridge_gate_v0 | 3 | 3 | 0 | 1.0000 | 1.0000 | 0.0000 | 17.17 | 0.00 | 0.00 |
| bridge_gate_v0_sp_v1_hard | 3 | 3 | 0 | 1.0000 | 1.0000 | 0.0000 | 7.17 | 17.17 | 10.00 |
| bridge_gate_v0_sp_v1_soft | 3 | 3 | 0 | 1.0000 | 1.0000 | 0.0000 | 17.17 | 17.17 | 0.00 |

## 结论

在这个小样本 root/frontier replay 中：

- naive ungated enzyme proposal 会污染搜索：3 个负例全部选到了酶候选；
- bridge gate v0 已经能把无 bridge evidence 的负例酶候选筛掉；
- SP-v1 hard gate 在 bridge-positive 上进一步压缩酶候选池，平均每个 target 拒绝 10 个 enzyme SP 不可信候选，同时没有降低这 3 个正例的 root/frontier 酶候选 recall。

这说明当前增强的实际作用不是“创造新的 proposal”，而是：

> 在 proposal 已覆盖的情况下，减少错误酶步进入搜索。

## 限制

这个 benchmark 不是完整路线求解率，不能说明最终路线一定可合成。它只说明在相同 live proposal 候选池上，gate/verifier 对酶候选污染有可测的抑制作用。

下一步应该扩大 proposal replay pack，并把被 SP-v1 拒绝但有争议的 real provider candidates 回灌为 v2 hard negative / audit pool。
