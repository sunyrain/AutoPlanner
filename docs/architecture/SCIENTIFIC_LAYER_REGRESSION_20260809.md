# 统一 anytime 科学层回归门（2026-08-09）

## 结论

Checkpoint F 的工程门已闭合：B4 只是一条同轨迹 milestone，达到后不会阻止 evidence、condition、replan 或 Program Action；文献 proof authority、conventional fallback 与 Program shadow/Claim 边界保持不变。

这不是“所有路线都已有科学证明”的声明。回归中特意保留 `B4=true`、`B3=false`、`B5=false` 的状态，以证明库存闭合不会伪造 exact-source 或配置验收。

## 同轨迹集成门

`tests/test_target_solver.py::test_b4_milestone_keeps_scientific_actions_on_the_same_trajectory` 使用一个不携带条件建议的冻结 director fixture，并由统一 Action loop 处理：

1. ChemEnzy target proposal、Codex global architecture、host materialization、validation、evidence discovery 和 stock audit 共用同一个 `RunKernel`；
2. 首个 B4 snapshot 出现后，同一 run 继续执行 `codex_global_replan`、`condition_enrich` 与 `program_review`；
3. replan retention audit 通过，第二次模型调用记入同一预算账本；
4. discovery-only 文献观察不获得 exact-source authority，因此最终仍为 `B3=false`、`B5=false`；
5. Program review 保持只读，shadow store event count 为 0。

本轮同时修复了规划深度/拓扑 replan 信号的时序：信号现在在 initial architecture Action 结算时进入统一循环。旧的后处理投影不再出现“replan signal/budget gate 通过，但没有实际 Action”的假执行状态。

## 真实文献案卷

通用 `scripts/replay_patent_xml_gate_suite.py` 已用当前 `autoplanner.opsin_pubchem_source_text.v14` 重新冻结名称解析快照，并随后以 `--offline` 完成零网络重放：

- Vismodegib / EP3381900A1：acid chloride amidation；
- DMB-S-MMP / EP2483292B1：thioester formation；
- Nirmatrelvir C4 / EP3953330B1：acidic ester hydrolysis。

三例均绑定官方 EPO XML 精确元素范围，条件完整，3/3 accepted；离线重放为 0 模型、0 视觉调用。当前 suite `content_sha256` 为 `29b15c0337633c3d711a61d4ca4869f96351aba9c7d7238897cec78c16fa9370`。

案卷门只证明 exact edge、source location 和 procedure/conditions 可复算；它不授予完整路线、库存或采购闭合。

## 酶正负对照与权限边界

`benchmarks/cross_category_program_regression_set.v1.json` 的两类冻结对照继续通过：

- Bufotalin 阳性：发现 5 个结构适用的 enzyme Program 候选，最大连续区间为 6 个化学步；
- Ibrutinib 负对照：3 条可扫描路线均为 0 个适用酶候选。

阳性在这里指结构/能力匹配阳性候选，不代表 exact-substrate 实验阳性。未验证候选仍是 proposal-only，逐边 conventional route 永久作为 fallback，不能关闭路线。专项 validation、shadow admission 和 experimental Claim 仍由独立 digest/CAS 门控制。

## 验证记录

- 同轨迹 focused tests：3 passed；
- 新的 B4 后科学 Action 集成门：1 passed；
- Program、酶对照、真实 procedure focused tests：22 passed；
- 官方专利案卷在线迁移：3/3 passed；
- 官方专利案卷随后的严格离线 replay：3/3 passed；
- 仓库级离线门：2642 passed、3 skipped、1 个已批准延期的模块行数预算门 deselected，另有 2 个 subtests passed。

仍开放的科学工作是取得真实 exact-substrate 酶实验结果，并据此决定是否显式准入 shadow store；当前回归没有虚构该实验结果。
