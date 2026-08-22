# SynthEx 单战略与酶法臂根因审计（2026-08-21）

本审计针对 `synthex_figure1_head_to_head_3` 的三个冻结目标：Traversiadiene、Dibohemamine A 和 Cyclopiamine B。它区分三个问题：

1. MCTS 是否真的把连续路径交付给了 Director；
2. 复杂天然产物为什么在单战略臂中停在浅层或不可执行步骤；
3. 酶法应该作为路线搜索中的什么对象，而不是把它当成一个强制分支标签。

## 1. 直接观察到的单战略结果

| target | paper-equivalent | policy calls / selected depth | public skeleton steps | Editor effect |
|---|---:|---:|---:|---|
| Traversiadiene | true | 7 / 7 | 7 | none |
| Dibohemamine A | false | 4 / 1; 20 / 17; 5 / 4 | 1, 1, 1 | 17→1 and 4→1 |
| Cyclopiamine B | false | 7 / 1; 7 / 4; 2 / 1 | 1, 2, 1 | 4→2, then 2→2 |

因此，Dibohemamine A 并不是“没有搜索到 17 步”，而是 Critic/Editor 的旧 surgical 路径把已搜索路线替换成 `prefix + edited_step`，而 AiZynthFinder 分支没有后缀重建器。Cyclopiamine B 同样出现了 4 步搜索、2 步交付的截断。Traversiadiene 之所以成功，是因为它在 Critic 前已经找到 stock-closed descendant，没有经过这条破坏性路径。

## 2. 为什么复杂天然产物的单战略表现差

### 2.1 搜索结果被后处理截短（P0）

`selected_depth` 是 AiZ MCTS 的路径深度，`multi_step_skeleton.steps` 是交付给后续 canonical graph 的路线。旧代码允许二者不一致；最坏情况下 20 次节点调用只留下根步。这个问题与 LLM 能力无关，直接降低 paper-equivalent solved。

本轮已修复：AiZ 分支进入 Critic/Editor 时自动使用完整 RouteJSON 编辑契约；编辑失败或导致拓扑退化时保留原 MCTS 路线，并公开 `editor_execution_notes`。AiZ 侧车还新增：

```text
path_action_count
path_route_step_count
path_route_step_metadata_missing_indices
path_route_projection_complete
```

### 2.2 MCTS 的“连续调用”不等于“完整路线”

当前 paper profile 每节点只允许一个 ReactionJSON 候选（这是为了保持论文 matched arm 的可比性）。因此 UCB 能选择的是连续状态，不是同一节点的多个 OR 反应动作。对复杂目标，如果第一步选择把骨架带入无可执行的局部化学，后续 20 次调用可能只是沿着同一条错误链继续修改，而不是回溯到另一个断键。

### 2.3 策略卡只冻结一个根锚点，不能代替路线级战略里程碑

复杂天然产物通常至少需要“骨架构建 → 立体/官能团整理 → 末端环合或氧化”的多个战略阶段。当前 paper arm 有意把 `max_strategic_milestones_per_branch=1`，因此它是公平的单战略对照，不是完整 AutoPlanner 能力上限。增强臂应在精确映射的上游叶节点上触发第二个 StrategyCard，而不是让所有路线预先声明多个战略，也不能把第二个策略当成对第一个策略的替换。

### 2.4 Route Builder 的局部反应假设缺少前置机制筛查

Dibohemamine A 的 Critic 已指出典型硬伤：无离去基的 C–N 烷基化、DDQ 对未活化烷烃的脱氢、未定义的水合/脱水区域选择性、无活化伙伴的桥环闭合，以及自由胺/硝基/酰亚胺/多羰基在强酸强碱氧化还原条件下的兼容性问题。也就是说，根断键可以是合理的，但逐节点 Route Builder 没有在早期拒绝“反应家族与底物边界不匹配”的候选，导致预算被错误链消耗。

### 2.5 Critic 是一致性轴，不是溶剂/条件证据轴

论文的 Critic/Improvement 主要用于路线内部一致性；它不等于文献证据或湿实验验证。当前 paper profile 也关闭了条件富集。因此“条件为空”不是 paper-equivalent 失败的主因，但它会提示路线仍停留在 hypothesis 层。结构连通、stock closure、反应可行性、条件完整度应分别统计。

## 3. 酶法的正确位置

酶法不应作为“每条路线都必须出现”的 StrategyCard，也不应在目标级别给路线强行加一个 biological edge。正确的对象是**路线局部替代（enzyme companion / overlay）**：

```text
三条独立化学根战略
        │
        ├─ 逐节点 ReactionJSON + AiZ MCTS，先得到完整化学路线/开放叶
        │
        └─ 对每个已物化的局部边运行 enzyme-opportunity scorer
              │
              ├─ 化学原边（永远保留）
              └─ 生物 OR 边（只有边界、底物范围、辅因子和验证计划齐全才进入）
```

一条生物候选必须同时绑定：

- host-replayed ReactionJSON 的精确输入/输出边界；
- enzyme class / EC / candidate 或 whole-cell host；
- 选择性目标和 substrate-scope basis；
- cofactor ledger（需求、再生、cosubstrate）；
- exact-substrate validation plan；
- 明确的化学 fallback span（替代几步，端点完全相同）。

现有 `biocatalytic_step_contract.py` 已经提供这些字段，并明确：策略级 intent 不创建 ReactionJSON 步；生物边的 `validation_gate` 默认关闭；step savings 只有在 fallback span 绑定后才能计算。因此真正缺少的不是 contract，而是把它接入搜索状态的时机：

1. 化学路线先产生局部可复现边；
2. scorer 只在有真实选择性机会时生成生物 OR 候选；
3. 化学边和生物边共享同一前后端点，分别进入 MCTS/stock/validation；
4. 生物边没有 exact proof 时标记 `enzyme_hypothesis`，不能获得 B4 或 step-savings 奖励；
5. 只有生物候选使同端点路线的 `stock closure`、选择性或物理操作数改善时，才进入最终推荐。

对于复杂天然产物，适合优先尝试的局部机会是晚期羟化/环氧化、立体选择性还原/氧化、动态拆分和已知底物范围内的官能团互变；不应让通用 cyclase 从简单底物凭空构造整个多环骨架。

## 4. 与原论文的真正差异

SynthEx 的公开流程是 Strategy Generator → Route Builder（写出完整 RouteJSON）→ Critic/Improvement → Analysis，并把不易采购的叶交给短尾模板搜索。当前实现虽然已经使用 AiZynthFinder MCTS 和逐节点 ReactionJSON，但历史运行仍有四个实质差异：

1. 旧 Editor 不能安全地重建 AiZ 后缀，造成已搜索路线截短；
2. `max_reactionjson_candidates_per_node=1` 使 paper arm 没有多候选 OR 回溯能力；
3. Route Builder 的策略卡只有一个根锚点，复杂目标没有增强臂的中间里程碑；
4. Critic 前没有足够便宜的反应机制/官能团兼容性筛查，错误链会消耗大部分调用。

因此现在不应先换模型或盲目增加酶臂。先重跑修复后的单战略对照，确认 `selected_depth == path_route_step_count == skeleton.steps`；再对增强臂做 `K=3` OR 候选、允许精确叶节点触发第二战略里程碑、以及 enzyme overlay 的独立消融。

## 5. 推荐的后续实验矩阵

| arm | strategy branches | ReactionJSON width | enzyme | purpose |
|---|---:|---:|---|---|
| paper-control | 3 | 1 | off | 与 SynthEx 的公平 reach/stock 对照 |
| OR-ablation | 3 | 3 | off | 测量候选宽度和回溯的真实增益 |
| milestone-ablation | 3 | 3 | optional | 允许精确上游叶触发第二战略卡 |
| enzyme-overlay | 同一 3 化学臂 | chemical route + local OR bio edge | optional | 测量酶法对闭合、选择性和步骤替代的增益 |

每个 arm 都应同时报告：paper-equivalent reach/solved、RouteJSON replay、reaction validation、stock closure、condition completeness、exact evidence，以及 enzyme validation/chemical-step savings；不能用最后一项替代前五项。

参考： [SynthEx arXiv 论文](https://arxiv.org/abs/2608.07454)；本地协议 [synthex_figure1_head_to_head_3.v1.json](../../benchmarks/synthex_figure1_head_to_head_3.v1.json)。
