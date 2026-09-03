# P10 展示与验收案例

展示必须来自 V4 canonical run，不能再提交整棵历史路线树或静态截图冒充当前结果。

| 目标 | 输入 | 当前诚实结果 | 用途 |
| --- | --- | --- | --- |
| Nirmatrelvir | `nirmatrelvir_v4_replay_pack.json` | 2 条文献落地且真实采购闭合路线；12 条验证超边；15 条精确结构记录与 15 条独立 procedure records（8 条条件未解析、7 条部分条件、0 条完整）；7 个库存叶；0 条 condition-complete/process-ready；0 模型调用 | 多信源科学 golden、恢复、缓存与 procedure/proof-vector 展示 |
| Nirmatrelvir lifecycle | `scripts/replay_fact_lifecycle_showcase.py` + 上述 replay pack | 撤销专利来源后仍保留 12 条反应验证和 15 条审计记录；有效 exact/procedure 各 8 条；complete route 2→0；只剩 1 个独立来源组；重新打开 evidence deficit；0 模型调用 | 摘要绑定撤销、增量降级、replay lifecycle stage 与“失效事实”展示 |
| 官方专利 procedure 三例门禁 | `scripts/replay_patent_xml_gate_suite.py` | Vismodegib / EP3381900A1 酰胺化、DMB-S-MMP / EP2483292B1 硫酯化、Nirmatrelvir C4 / EP3953330B1 酸性酯水解均绑定官方 EPO ST.36 XML 精确元素范围；3/3 条件完整；每例两次离线 registry digest 与 binding ID 一致；0 模型/视觉调用；结构化来源闭合后 PDF/OCR/vision 调用均为 0 | P3 三独立专利、三反应类型发布门禁；统一审计面板和来源/条件 inspector 展示 |
| Artemisinin | `artemisinin_v4_case_dossier.json` | 2 条完整路线；2 条验证超边；3 条精确来源记录；4 个库存叶；0 模型调用 | 一键案卷编译、采购边界替代和串联氧化验证 |
| Paclitaxel | `paclitaxel_v4_bounded_showcase_plan.json` | 3 条不同 L1 路线，0 条完整；明确 evidence/validation/stock/closure 缺口 | 共享中间体、替代路线和 UI 压力 |
| Lorlatinib | `unseen_v4_baseline_panel.json` | 无本地事实，零模型预算耗尽，diversity 缺口 | 未见收敛药物小分子 |
| Trabectedin | 同上 | 无本地事实，零模型预算耗尽，diversity 缺口 | 稠密天然产物 |
| Voclosporin | 同上 | 无本地事实，零模型预算耗尽，diversity 缺口 | 大环肽 |

Paclitaxel 旧运行包含 96 分支、83 节点、122 步和约 600 MiB 本地生成文件，但 0 条
proof portfolio 路线。它现在只作为仓库外诊断来源；默认展示改为 3 条策略不同、边集
不同的有界路线，并明确停在 L1。分支体量不再制造完成错觉。

Workbench 的四个视图分别是：

1. 断键建议；
2. 已展开路线；
3. 反应验证路线；
4. 库存闭合路线。

V4 proof-vector 展示在此基础上进一步拆分为：断键建议、全路径已展开、反应验证、文献落地、
条件完整、配置边界闭合、真实采购闭合和工艺候选。各视图可以重叠，但只能由当前 canonical
投影的逐边/逐叶 authority 进入；旧 L0-L4 颜色、路线数量和 `benchmark_search` 均不能反推档位。

Nirmatrelvir 与 Artemisinin 的停止条件是全部所选边与叶满足 acceptance；Paclitaxel 和 unseen panel
的停止结果是具名 unresolved/budget-exhausted。任何展示都不能用 Agent 轮数或分支数
代替 completion。

Artemisinin 同时保留两种采购边界：从青蒿酸开始的两步路线，以及直接采购二氢青蒿酸
开始的一步路线。短路线不是长路线的“冗余子集”，因为二者的叶节点、供应风险和执行
边界不同。氧化串联步骤只有在同一规范边存在精确来源记录时才允许使用受限的多中心验证；
同样的无来源碳骨架改写仍会被前端拒绝。

这些闭环案例证明的是“审阅案卷进入统一主线后可快速、确定地验证和展示”，不是声称
系统已能在无来源、无候选生成器的情况下凭空发现任意复杂分子的完整路线。未见目标仍须
由 Codex 全局规划、外部 proposal provider 或人工审阅产生小型路线案卷；主机侧负责
结构、反应、来源、库存和 completion 的逐层硬验收。目录 offer 是带时间戳的供应证据，
不等于实验室当日现货。

三例门禁都只覆盖各专利中的目标 procedure，不冒充完整逆合成路线。它分别绑定 Vismodegib
`h0016-p0046`、DMB-S-MMP `h0019-p0165` 与 Nirmatrelvir C4 `h0012-p0199`；首次联网物化后，
`--offline` 只读取摘要校验后的官方 XML 与 resolver snapshot，并为每例连续编译两次。
各 Workbench 都保留未审计起始原料和 `stock_boundary` deficit，因此来源/条件完整不会自动升级为
采购或全路线闭合。P3 的三个真实专利/反应类型发布门禁现已 3/3 通过。

每个案例至少记录：完整路线数、validated edge、叶库存率、最低 proof、独立来源组、
路线多样性、拒绝原因、墙钟/CPU/内存、artifact 字节、attempt/accepted expansion、
模型调用/token/时间和 UI projection 延迟。
