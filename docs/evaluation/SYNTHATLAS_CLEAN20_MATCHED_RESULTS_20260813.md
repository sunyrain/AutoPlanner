# SynthAtlas clean-20 同宿主四臂结果

日期：2026-08-13

状态：v3 已因并发墙钟重复计费而判定方法学失效并主动停止；修复后的 v4 已通过 unified-adaptive target 001 独立烟测，并从四臂 target 001 重新开始正式运行。v3 诊断见 `SYNTHATLAS_CLEAN20_V3_INVALIDATION_20260813.md`。

## 1. 评价问题

本实验不比较谁能生成更多看起来合理的路线，而比较在相同 target、stock、host gates、预算口径和失败分母下，各策略能把候选推进到哪个可审计闭环层级：

- `C0`：target-rooted strategy/route topology；
- `C1`：host-admitted chemical topology；
- `C2`：reaction-feasibility validation；
- `C3`：exact evidence binding；
- `C4`：source-exact procedure / condition closure；
- `C5`：frozen stock closure；
- `C6`：experimental claim closure。没有执行实验时必须记为 `unassessed`，不能记成负结果。

冻结 protocol 的四臂为 `SynthAtlas external snapshot`、`Codex-only`、`ChemEnzy-only` 和 `unified-adaptive`。四个臂中只有后三者需要 live 执行；公开 SynthAtlas snapshot 是固定路线输入的外部参照，不继承 `solved`、条件或 critic verdict 的 host authority。`unified-round-robin` 属于 Retro*-190 主消融，不属于本 clean-20 protocol。

## 2. 冻结身份

- manifest：`benchmarks/synthatlas_strategy_closure_clean20.v1.json`
- 已失效且只读保留的 live root：`results/shared/synthatlas_strategy_closure_clean20_live_contract_v3`
- v4 正式 live root：`results/shared/synthatlas_strategy_closure_clean20_live_contract_v4`
- v4 execution receipt SHA-256：`cb65eb18c8968c12158abd65988084569a7457a4c6d089c6a6f789e751dcf0b2`
- v4 source bundle：817 files
- v4 source bundle SHA-256：`9dd24cfa41d101842505b9a3bb8424c9275fa9a9fcb1391f16468c2f531fc254`
- 每臂固定 cutoff：1800 s、192 settled tasks；相同 frozen benchmark stock 与 leakage-audit pack。

冻结前烟测使用同一个 source bundle。unified-adaptive target 001 的 fixed-cutoff projection 可用，累计 wall time 为 962.303094 s，而并发任务 compute time 为 2436.119782 s；B1/B4 达成，B2/B3/B5 未达，证明修复后的计费没有把并发 compute seconds 再误当作 wall seconds。烟测目录与正式目录分离，target 002 启动后即终止整棵烟测进程树，不进入正式分母。

运行期间只允许修改不属于 source bundle 的测试和文档。严格汇总必须在任何冻结源码修改前生成，并验证 receipt、source bundle、case order、失败分母和每臂 completion。

## 3. 预注册主表

下表只由 `scripts/summarize_strategy_closure_pilot.py` 的 digest-valid paired summary 回填：

| Arm | 完成 targets | C0 | C1 | C2 | C3 | C4 | C5 | C6 | 主要失败类型 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SynthAtlas external snapshot | 20/20 | 待引用已冻结外部汇总 | 待引用已冻结外部汇总 | open | open | open | open | unassessed | external authority stops at snapshot topology/admission |
| Codex-only | 运行中 | 待汇总 | 待汇总 | 待汇总 | 待汇总 | 待汇总 | 待汇总 | unassessed | 待汇总 |
| ChemEnzy-only | 运行中 | 待汇总 | 待汇总 | 待汇总 | 待汇总 | 待汇总 | 待汇总 | unassessed | 待汇总 |
| unified adaptive | 运行中 | 待汇总 | 待汇总 | 待汇总 | 待汇总 | 待汇总 | 待汇总 | unassessed | 待汇总 |

## 4. 必报资源与配对差异

最终报告必须同时给出：

- 每目标 wall time、model calls/input/output tokens、native-search invocations/expansions、host validation、stock/evidence/condition/Program tasks；
- target-level paired differences，而不只给平均值；
- raw → normalized → admitted → validated → stock-closed 的首损边界；
- provider timeout、empty proposal、normalization quarantine、host chemistry rejection、stock miss、budget exhaustion、canonical merge 和 portfolio projection 分开计数；
- 所有失败、timeout、partial 和空结果进入同一分母。

## 5. 解释边界

允许的结论取决于机器结果：

- 某臂 C0/C1 更高，只能说明该固定预算下的拓扑/准入覆盖更高；
- C5 更高，必须确认不是更宽松的 stock oracle 或不同 host gate；
- C3/C4 更高才能支持 evidence/procedure closure 增益；
- unified 对 ChemEnzy-only 无增益时，应报告“协同未在该 panel 转化为闭合增益”，不能用个案故事替代；
- runtime/provider failure 必须作为工程失败报告，不能伪装成化学能力劣势；
- clean-20 只支持有界竞争响应，不支持对 1,098-target SynthAtlas、Retro*-190 全量或实验成功率的总体优越性主张。

## 6. 待生成机器证据

- 修复后新批次的 `paired-summary.json`；不得为 v3 生成性能汇总
- 修复后四臂各自的 `panel-status.json` 与 target reports
- failure taxonomy、resource table 和 paired target rows（以严格汇总实际字段为准）
