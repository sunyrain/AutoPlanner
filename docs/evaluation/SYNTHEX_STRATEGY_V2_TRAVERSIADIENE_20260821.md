# Strategy-v2 Traversiadiene canary

日期：2026-08-21  
目标：SynthEx Figure 1 第一个目标（仓库盲测名 `opaque case 9c1f431594a7`）  
运行目录：`D:\Autoplanner\canary_runs\synthex-figure1-v2-traversiadiene`

## 配置

- 3 个独立战略分支，策略模式 `autoplanner_strategy_v2`。
- 每个分支独立进入 AiZynthFinder MCTS/UCB；每节点一个 ReactionJSON 候选。
- `gpt-5.6-terra`，reasoning `medium`；无 ChemEnzy、无 web、无证据预取。
- ZINC+eMolecules 冻结库存索引用于最终库存边界判断。
- 1 个 strategic milestone/branch；局部修复上限 6 轮。

## 结果

| 指标 | 结果 |
| --- | ---: |
| StrategyCard | 3/3 接受 |
| 模型调用 | 18 |
| 输入 / 输出 token | 304,080 / 39,340 |
| 目标根路线达到 | 5 条（paper reach） |
| 物化路线 | 2 条主路线，5/4 步 |
| paper-equivalent solved | 0 条 |
| 库存闭合 | 0 条 |
| 反应验证闭合 | 0 条（条件与证据轴独立报告） |
| 终态 | `unresolved`，预算内正常收束 |

## 三个新版战略卡

1. 汇聚片段连接 + 阳离子多烯级联环化。
2. 分子内 α-烷基化 + 羰基编辑，构建关键桥键。
3. 多烯环化–骨架重排级联 + 晚期分子内 C–C 捕获。

第三臂与论文对 Traversiadiene 的公开路线类型（多烯环化后级联/重排）在战略层面一致；这只说明战略假设命中了正确拓扑，不等于路线已被主机验证或库存闭合。

## 失败位置

- 新版战略生成已不再是主要瓶颈：三个分支都给出了明确骨架事件、前体角色和前体专属断键。
- 执行层仍出现局部氧化/还原循环及非库存叶，导致路线不能达到 paper-equivalent solved。
- 报告中的 `paper_reach=true` 只表示生成了目标根连通路线；`paper_equivalent_solved=false` 是因为每个叶必须命中同一冻结库存，而本次没有任何完整路线满足该条件。
- 条件、证据、反应验证没有被拿来否定 paper reach；它们作为独立质量轴继续报告。

## 后续工程动作

1. 在 Route Builder 节点状态加入同一叶节点的反应族/氧化态循环惩罚，并保留可回溯的候选。
2. 让 `--no-chemenzy` 同时把全局架构中的 provider 请求上限置零，并让无结果的短尾阶段按实际引擎标记 provider，避免禁用 ChemEnzy、实际运行 AiZynthFinder 后仍在计划/阶段元数据中出现 `provider_preferences=['chemenzy']` 或 `provider_id='chemenzy'`。
3. 对第三臂的多烯级联候选做一次短的手工/自动结构重放，再决定是否增加候选宽度或战略里程碑数。
