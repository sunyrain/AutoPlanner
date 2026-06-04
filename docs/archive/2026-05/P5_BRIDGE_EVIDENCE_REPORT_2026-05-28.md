# P5 桥接酶路线证据报告

日期：2026-05-28

## 结论

P5 的诊断型交付已经完成：我们构建了 verifier-gated bridge 数据、接入 route-tree source gate、完成 controlled/live benchmark、做了 live route quality audit，并打包了 14 条 production-policy 的 partial evidence routes。

但必须明确：当前还没有得到 stock-closed 的生产级完整合成路线。现阶段证据支持的是“bridge verifier + source gate + selection bonus 能减少酶侧污染，并在 bridge-positive 目标上促进真实 live provider 的酶步被选中”，而不是“系统已经能稳定给出完整可采购起始物闭合的化学-酶逆合成路线”。

## 关键交付物

| 交付物 | 路径 |
|---|---|
| P5 进度总表 | `docs/P5_EXECUTION_PROGRESS_2026-05-27.md` |
| live target probe | `results/shared/bridge_gate_ablation_v0_20260527/live_enzyme_bridge_target_probe_report.md` |
| live route evidence | `results/shared/bridge_gate_ablation_v0_20260527/bridge_live_route_evidence_report.md` |
| bonus=2 live route evidence | `results/shared/bridge_gate_ablation_v0_20260527_bonus2/bridge_live_route_evidence_report.md` |
| fair live benchmark depth-1 | `results/shared/bridge_live_policy_benchmark_v0_20260528_depth1_4p4n/bridge_live_policy_benchmark_report.md` |
| route quality audit | `results/shared/bridge_route_quality_audit_v0_20260528/bridge_live_route_quality_audit_report.md` |
| P5 evidence package | `results/shared/p5_bridge_evidence_package_v0_20260528/p5_bridge_evidence_package.md` |

## 主要结果

### 1. Bridge verifier 可以显著降低噪声

候选级 verifier gate：

| Policy | Precision | Recall | FPR | Cost / TP |
|---|---:|---:|---:|---:|
| ungated bridge | 0.1187 | 1.0000 | 1.0000 | 8.43 |
| verifier precision gate | 0.9893 | 0.9563 | 0.0014 | 1.01 |

这说明问题不再是“是否能构造 bridge”，而是如何让 search 只在有证据的位置调用酶侧 proposal。

### 2. Live enzyme provider 不是完全打不到

对 30 个 verifier-pass bridge-positive 目标直接调用 live enzyme providers：

| 指标 | 数值 |
|---|---:|
| probed targets | 30 |
| raw covered targets | 30 |
| usable covered targets | 30 |
| raw enzyme candidates | 199 |
| usable enzyme candidates | 165 |

过滤掉的典型问题包括 self-loop 和 tiny-largest-reactant。结论是 provider 有可用信号，但候选仍需质量控制。

### 3. Bridge evidence 需要进入 route selection

仅做 source gating 时，normal live route evidence 中酶步自然被选中的比例不高：

| 设置 | Targets | targets with enzyme route | selected enzyme routes |
|---|---:|---:|---:|
| normal bridge-gated, bonus=0 | 12 | 3 | 5 |
| normal bridge-gated, bonus=2.0 | 12 | 4 | 8 |

depth-1 fair benchmark 中，bonus 的效果更清楚：

| Policy | True selected | False selected | Recall | FPR | Mean enzyme calls |
|---|---:|---:|---:|---:|---:|
| ungated_default_source_gate | 1 | 0 | 0.25 | 0.00 | 2.00 |
| bridge_gate_verifier | 1 | 0 | 0.25 | 0.00 | 1.00 |
| bridge_gate_verifier_bonus2 | 4 | 0 | 1.00 | 0.00 | 1.00 |

这支持一个明确工程判断：bridge verifier 不应只用于“是否调用酶源”，也应作为 action selection prior。

### 4. 路线质量审计

对 94 条 selected enzyme routes 做质量审计：

| 指标 | 数值 |
|---|---:|
| selected enzyme routes audited | 94 |
| diagnostic-only routes | 72 |
| production-policy partial candidates | 22 |
| stock-closed production candidates | 0 |
| production positive routes | 22 |
| production negative routes | 0 |
| hard artifact flags | 0 |

主要风险：

| 风险 | 数量 |
|---|---:|
| generic EC | 92 |
| missing EC | 8 |
| larger reactant than product | 4 |

没有发现自环、产物不匹配或“小试剂生成大分子”的硬错误。当前失败点是路线没有闭合，且酶证据仍停留在泛化 EC/template 层面。

## P5 Evidence Package

最终打包 14 条 production-policy partial evidence cards：

| 指标 | 数值 |
|---|---:|
| evidence cards | 14 |
| unique targets | 7 |
| stock-closed cards | 0 |
| route-solved cards | 0 |
| hard-flag cards | 0 |
| bridge_gate_verifier_bonus2 cards | 4 |
| normal_bridge_gated cards | 8 |
| bridge_gate_verifier cards | 2 |

这些路线只应作为“模型机制证据”和“下一阶段优化样例”，不能作为最终合成建议展示。

## 当前失败原因

1. 酶候选大多只有 generic EC，例如 `1.x`，缺少具体 enzyme/substrate/product 三元组证据。
2. route-tree 能选中酶步，但后续化学/酶 leaf 不能稳定继续拆到 stock。
3. search ranking 仍偏向化学 provider；bridge-supported bonus 有帮助，但只是浅层改善。
4. EnzExpand/Enzyformer 的候选质量不稳定，需要更强的 enzyme-substrate verifier 或 EC-conditioned proposer。
5. 目前的 stock closure 和 purchasability 约束不足，导致 partial route 多、完整路线少。

## 下一阶段建议

1. 训练或接入 enzyme-substrate-product verifier v1，目标不是预测反应式，而是拒绝错误酶步。
2. 把 bridge evidence、EC evidence、reaction-center evidence 显式纳入 action scoring。
3. 扩大 fair live benchmark 到更大正负集合，并保留 depth-2/3 对比。
4. 对 P5 package 中的 7 个 unique targets 做 case-driven 修复，优先解决 stock closure。
5. 如果要做论文创新点，应聚焦“无专家标签条件下的 chemo-enzymatic bridge weak label + verifier-gated search”，而不是泛泛的多步逆合成。

## P5 状态

诊断型 P5 已完成：有指标、有 live provider 证据、有质量审计、有 10-20 条 evidence-supported partial route cards、有明确失败原因。

生产型 P5 尚未完成：没有 stock-closed 路线，不能宣称已经解决真实合成规划。
