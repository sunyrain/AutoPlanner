# P10 展示与验收案例

展示案例必须由可重放 V4 run 生成，不再提交静态 PDF/PNG 或复制整棵历史路线树。

| 目标 | 目的 | 期望结果 |
| --- | --- | --- |
| Nirmatrelvir | 多信源、exact step、库存闭合 acceptance | 至少两条独立完整路线；Science/SI、专利和独立工艺来源分别绑定 |
| Paclitaxel | 高分支、共享中间体和 UI 压力 | 默认 2–5 条路线；L0 与已验证路线分离；拖拽/缩放稳定 |
| Artemisinin | 天然产物与半合成路线族 | 化学/生物路线可替换模块和冲突显式显示 |
| 一个复杂药物小分子 | 深度、保护基和工艺选择 | model-free baseline 后再测有界 global director gain |
| 一个仓库无 fixture 的新目标 | 防止记忆/特例伪成功 | 合法闭合，或精确报告未解决 edge/leaf/evidence deficit |

每个案例记录：完整路线数、validated edge、叶节点库存率、最低 proof level、独立来源组、路线多样性、拒绝原因、墙钟/CPU/内存、CAS 字节、增量重算比例、缓存命中、attempt/accepted expansion、模型调用/token/时间，以及 UI update/frame 指标。

停止条件是“所选路线的全部反应边和全部叶节点满足 acceptance”，不是固定 Agent 轮数。任何未闭合案例仍可作为展示，但必须在标题、颜色和 inspector 中标为 unresolved。
