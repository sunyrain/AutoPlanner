# V9：Strategy-Guided Self-Correcting Search

中文名：策略引导的自纠偏搜索。

状态：**过渡实现**。`v9_smoke` 实验 profile 已接入一次 Strategy portfolio review、每分支至多两次关键事件 audit、完整路线 final audit，以及 V8 兼容的 full-route `Critic -> Editor replace_span -> Host replay -> Critic` 修复循环。`checkpoint_match=false` 若只是误标的准备步骤则延期；若该动作替代、消费或切断 Strategy 必需拓扑，则只回滚该动作并保留父路径。搜索异常 audit、按需 ReactionWitness 和事务式 Path Repair 尚未实现，不能从本文的目标设计反推为现有能力。

基线：paper-aligned V8 已冻结为 Git 引用 `paper-aligned-v8-freeze-20260826`。本文只定义 V9，不改变 V8 的实现事实或论文复现结论。

## 1. 结论

本节描述目标 V9。当前 `v9_smoke` 已实现其中的 Strategy review、关键事件稀疏审查、Host
结构权威和 final audit，但跨步修复仍使用 V8 的完整路线 Editor 合同；目标态 Path Repair
尚未替换这条兼容路径。

V9 只在搜索开始前增加一次紧凑 Strategy portfolio review，不增加逐步 Critic、Structural Auditor Agent、Repair Planner Agent 或新的综合评分器。正常搜索只保留三个常驻责任边界：

1. **Strategist**：一次提出三个相互独立、经过同调用内部比较的策略假设；
2. **Builder**：作为 MCTS 的单节点 policy，每次只写一个当前 ReactionJSON 动作；
3. **Host kernel**：拥有结构回放、原子身份、树搜索、库存、停止和 solved 的唯一事实。

Critic 和 Editor 不再是每条路线必经的流水线阶段，而是一个按事件调用的 **Chemistry Reviewer** 的两种模式：

- `audit`：在有新信息时判断化学；
- `path_repair`：仅在跨多步协调确有必要时规划修复。

核心原则是：

> 正常路径保持短；确定性错误由 Host 当场返回；化学审查只在信息增益高的事件发生；跨步修复是异常处理，不是标准工序。

这不是减少化学能力，而是停止让多个 Agent 重复解释同一条路线。

## 2. 论文边界

SynthEx 论文支持以下事实：

- Strategy Generator 默认在一次调用中给出三个独立的一句话策略，每个策略作为搜索的 steering query。[Paper: PDF p. 22]
- Builder 是 AiZynthFinder MCTS 中的 next-step LLM policy，逐节点输出由十类 primitive 组成的 ReactionJSON；应用图编辑后确定性地产生 mapped precursors。[Paper: PDF pp. 22–23]
- 原论文只在完整路线后运行 Critic–Editor 循环；Critic 正向检查，Editor 可重排、插入、删除或改变条件和官能团。[Paper: PDF pp. 18–19, 24]
- 六轮后 blocking rate 从约 0.27 降至约 0.06，只说明系统对同类 backbone 的 Critic 标准发生内部收敛，不是实验验证。[Paper: PDF pp. 18, 24]
- solved 只由 Host/AiZ 的完整目标根路线和全部叶的精确 full-InChIKey 库存命中决定；short tail 为 6 transforms / 500 iterations / 1200 s 的完整 AiZ solution stitching。[Paper: PDF p. 24]

V9 保留论文的 Strategy、ReactionJSON、MCTS、完整路线审查和跨步编辑能力，但改变审查时机和物化方式。它是以 V8 复现为基线的研究架构，不应冒充论文原实现。

## 3. 对外部报告的处理

| 报告建议 | V9 决定 | 原因 |
| --- | --- | --- |
| 三个 Strategy 在一次调用中内部比较和攻击 | 采纳 | 不增加调用，也减少三个同质策略 |
| Builder 在同一次推理中 propose / challenge / revise | 采纳 | 增强单步质量，但不输出推理过程 |
| Host 只做 replay、valence、map、provenance、cycle、connectivity | 采纳 | 这些是低成本确定性事实 |
| Host 错误回到同一 leaf 的 Builder | 采纳 | 编译失败不是 Editor 任务 |
| Critic 变成三个事件触发的中断 | 采纳 | 避免逐步 Critic 拉长流程 |
| Editor 只处理少见的跨步修复 | 采纳 | 保留论文能力，同时避免整路反复重写 |
| 只保存不能从结构重建的信息 | 采纳 | 防止 ledger、status 和投影副本膨胀 |
| Strategy 输出硬 `key_graph_effect` | 修正 | 只保留自然语言可观察意图，不变成 required-map admission |
| `checkpoint_relation=executes_checkpoint` | 限定采纳 | 仅作 Critic 调度 metadata；必须先经 Host 回放，再由 Reviewer 的 `checkpoint_match` 确认，不能自证关键事件成立 |
| 固定 `base + strategy - failure` 总分 | 拒绝 | 不恢复词典式综合评分或新的 admission 权限 |
| 连续 N 步未完成关键事件就强制扩展或删路线 | 拒绝 | 只能触发一次反思，不能成为硬 gate |
| 完全删除 Editor | 拒绝 | 跨步重排、保护和依赖协调仍是论文的重要能力 |

同一 backbone 的内部自我比较只能提高一致性，不能称为独立验证。需要独立性时，应改变 reviewer 配置或引入人工/实验评价，而不是再堆一个同输入同模型的常驻 Agent。

## 4. 最短热路径

```text
Target
  |
  v
Strategist: one call -> three strategy hypotheses
  |
  v
Strategy Critic: one compact review -> same three-card contract
  |
  +-------------------+-------------------+
  |                   |                   |
  v                   v                   v
MCTS A              MCTS B              MCTS C
  |                   |                   |
  +---- selected leaf + strategy + compact root-to-leaf history
                      |
                      v
                  Builder
             one ReactionJSON action
                      |
                      v
                 Host kernel
      normalize -> reserve maps -> replay -> admit
          |                         |
          | typed structural error  | accepted action
          v                         v
      same-leaf retry             MCTS continues
                                    |
                          event interrupt only when needed
                                    |
                 exact stock -> complete AiZ tail -> stitch
                                    |
                                    v
                              final route audit
```

Strategy 组合只在搜索前审查一次。正常接受一个树动作只需要一次 Builder 调用和零 token 的 Host 检查；没有关键事件时，不调用在线 Critic；没有跨步 blocker 时，不调用 Editor。

## 5. 最小模型合同

### 5.1 Strategist

输入：

- target structure；
- 可选的人类约束或必需起始物；
- 论文中的四点思考方向可作为内部提示，但不要求输出分析表。

同一次调用内部完成：产生候选、相互比较、攻击关键假设、保留三个真正不同的方向。中间推理不输出。

每条策略只输出：

- `strategy`：一句话，包含关键事件及其预期骨架作用，作为 Builder steering query；
- `critical_assumption`：一句话，指出这条路线最值得以后验证的化学假设；
- `critic_checkpoint`：一句话，指出应触发稀疏审查的唯一实际图变换，并明确区别于手柄安装、去保护或其他准备步骤。

Strategy 不输出完整路线、required map pairs、milestone checklist、handoff、stop、solved 或 admission 判断。`critical_assumption` 是审查焦点，不是失败条件。

`v9_smoke` 在 Builder 搜索前运行一次独立但紧凑的 Strategy portfolio review。它读取 target 与三张三句话卡片，输出仍是同一个三卡合同；有效卡必须逐字保留，只能修改具有具体弱点、重复、内部矛盾或与目标骨架明显不相干的方向，且修改时必须保留所有未被质疑的 reactive handle、保护/遮蔽、tether geometry、立体控制与顺序约束。它同时校正 `critic_checkpoint` 是否指向可观察图变换；有效审查结果成为 Builder 唯一看到的 portfolio，无效审查则保留生成器初稿。它不是 admission gate，不产生第二套 verdict、score、ledger 或回滚权限。具体图编辑的化学真实性仍由关键事件 audit 和 final audit 判断。

### 5.2 Builder

Builder 只负责回答一个问题：

> 对当前真实 mapped leaf，下一项最好的逆合成图编辑是什么？

输入只包括：

- target 与当前策略的三个紧凑字段：`strategy_query`、`critical_assumption`、`critic_checkpoint`；
- 当前 `selected_leaf_mapped`；
- 从 target 到该 leaf 的完整但紧凑的已接受反应历史；
- 产生当前 leaf 的父反应及同一 split 的 Host-mapped sibling co-precursors；lineage 优先按 mapped boundary 匹配，不能依赖未映射立体 SMILES 完全相等；
- 当前路径的 ancestor identities；
- 同一 leaf 最近一次 typed rejection（若存在）；
- 尚未解决的紧凑 checkpoint feedback（至多两条原因和一个最小修订建议），跨准备步骤、换叶和超时保留。

历史必须包含已经真正回放的反应及其顺序，不能只给反应名，也不能夹带其他分支、完整运行日志、Critic 长解释或可从结构重建的大表。

Builder 在一次调用内部自行 propose、challenge、mental replay、revise，最终只输出：

- `reaction_intent`：一个简洁化学意图；
- `operations`：论文支持的 ReactionJSON primitive 序列；
- `conditions`：可选的简洁条件。

不要求 `move_role`、`feasibility_check`、大段 rationale、完整路线或多候选。Builder 没有 `fail`、`handoff`、`stop`、`solved`；也不负责认证 Strategy 已完成。

内部 mental replay 只是提高候选质量，真实结构仍必须由 Host 回放。工具是否被模型尝试使用不是整条候选的化学 admission 标准；只有真实输出合同和 Host 回放结果决定该动作能否入树。

### 5.3 Chemistry Reviewer

Reviewer 不是常驻 Agent，而是两个稀疏调用模式。

`audit` 输入：策略、`critical_assumption`、当前完整紧凑路径、相关 ReactionWitness、条件和触发原因。输出只包括：

- `verdict`: `pass | uncertain | reject`；
- `focus_step_id`；
- 一条具体化学理由；
- 若 reject，指出最早需要重新考虑的化学边界。

`uncertain` 永不删除路线，也不自动回滚。`pass` 只代表 Reviewer 标准下未发现 blocker，不代表实验可行。

`path_repair` 只在 reject 涉及跨多步依赖且当前路线值得保留时调用。它看完整紧凑路线，但只输出：

- `repair_intent`；
- `affected_step_ids`；
- `must_preserve`。

它不重写完整 RouteJSON，不分配 atom maps，不声明库存或 solved。Host 计算依赖闭包，再由普通 Builder 合同逐个真实 mapped boundary 物化。

## 6. Host kernel

Host 是唯一事实所有者，但不是化学 oracle。

它负责：

- ReactionJSON schema normalization 和确定性 replay；
- valence、connectivity、component、cycle、ancestor 和 target-root 连通性；
- route-lineage 内单调递增、删除后不复用的 atom-map provenance；
- 在整条 RouteJSON 的保留 namespace 中为新原子分配 map，并在图编辑/价态补全后从当前结构派生 RDKit stereo reference neighbours；模型只表达 stereo intent；
- 从回放前后结构按需派生 ReactionWitness；
- MCTS 节点、分支、widening 和停止；
- 精确库存、short-tail eligibility、完整 tail stitching 和最终 solved；
- typed error 返回、路径事务和原子提交。

它可以拒绝形式上可证明的矛盾，例如非法 map、不可回放操作、孤立分支，或一个机器可读的 intramolecular claim 实际生成两个独立前体。它不能证明 8-endo 动力学可行、阳离子 relay 连续、催化剂有效或命名反应真实成立。

### 6.1 Map provenance

必须保存的不是又一份 molecule ledger，而是无法从最终结构恢复的最小历史：

- 当前 lineage 的 `next_map`；
- map 的 origin step / fragment atom / element；
- tombstoned maps。

`add_group` 的新原子由 Host 分配身份。整条线性或 DAG RouteJSON 共用保留 namespace，删除后的 map 永不复用，避免当前 V8 已观察到的 O→C、O→Br 假 provenance。若 provider 未提供 mapped reaction，Host replay audit 的 mapped product/precursors 是 admission 判断前体原子贡献的直接事实，不要求复制第二份 provider 映射。

### 6.2 ReactionWitness

Witness 是从已接受 ReactionJSON 和回放前后结构生成的只读视图，不是第二套 canonical state。需要时派生：

- 实际 bond additions、deletions 和 order changes；
- component count 与分子内/分子间拓扑；
- added、removed、retained atoms 及 map origins；
- stereo changes；
- intent 中可机器核对的结构主张是否与图编辑一致。

Witness 可缓存，但必须能从 canonical step 和 Map provenance 重建；它没有 admission score，也不证明化学机理。

## 7. 三个 Critic 中断

### 7.1 关键事件候选首次出现

关键词、反应名和共享术语不参与触发。只有 Builder 返回 `checkpoint_relation=executes_checkpoint`，且 Host 已成功回放并观察到真实 mapped 骨架键编辑时，才触发 `audit`。

这是“候选出现”，不是“关键事件已执行”。Reviewer 先返回 `checkpoint_match`：若为 false 且动作只是保留必需拓扑的误标准备步骤，候选按 `preparatory` 保留；若为 false 但动作已经替代、消费或不可逆切断关键事件所需拓扑，Reviewer 可用 `reject + sequence_dependency` 只拒绝该动作并回到父节点；若为 true，再用 pass/uncertain/reject 判断实际图编辑、立体来源和 `critical_assumption` 是否协调。未解决的精确反馈以紧凑状态跨准备步骤、换叶和超时保留，只有真实 checkpoint 通过后才清除。每分支最多两次关键事件审查，避免循环调用。

事件检测只有调度权限，没有 admission 权限。误触发最多多做一次 audit；漏触发仍由完整路线终审兜底，不能因此把 Strategy 标记为完成。

### 7.2 搜索异常

只在已有搜索事实显示异常时触发，例如：

- 同一 node 反复产生相同 normalized net edit 或相同 typed rejection；
- Builder 反复声称已完成不存在的关键步骤；
- 路径持续增加步骤却没有有意义的结构进展；
- 新步骤与已接受路径的化学角色发生明确冲突。

异常检测使用现有 MCTS 统计、normalized action 和 rejection memory，不引入固定的“第 N 步必须做关键反应”规则。触发结果是一次重新定向建议，不是强制扩展、整分支删除或 Strategy 失败。

### 7.3 完整路线形成

Host 完成目标根拼接并核对全部叶后，Reviewer 对整条路线做一次论文式正向终审。只有这里需要完整路线级的条件、官能团兼容性、步骤顺序和累计风险判断。

一条无异常路线通常只经历一次关键事件 audit 和一次 final audit；不会每一步都运行 Critic。

终审不授予或撤销结构/库存意义上的 `solved`。stock closure 与 Reviewer verdict 分轴保存；reject 只会触发局部重开、Path Repair，或保留一条带 blocker 标记的 stock-closed 路线。

## 8. 搜索、反馈与评分

- 三个 Strategy 各自建立搜索树，Strategy 是 steering query，不是 admission contract。
- 保持 AiZ/MCTS 原有 selection 语义；V9 第一版不新增 strategy-progress composite score。
- replay 成功的不同动作作为同一 node 的不同 action 保留；相同 normalized action 去重。
- structural error 不入树，把精确错误返回同一 leaf；这是 Builder retry，不是 Editor。
- Critic reject 只重开最早相关 node 或受影响子树，保留无关兄弟 action 和旧的可回放路线。
- `uncertain` 只进入最终风险报告，不降低 admission。
- 不恢复 `[stock_closed, milestone_count, route_depth, -open_leaf_count]`，不让 `strategy_relation`、`move_role` 或 topology score 决定接纳。
- exact stock 仍是唯一终止事实；provider 的 `provider_solved` 必须经过目标根 stitching、全量 replay 和 Host 库存复核。

## 9. 最小状态

每个搜索节点只持有一份 canonical 状态：

| 状态 | 为什么必须保存 |
| --- | --- |
| mapped molecular frontier | 当前真实搜索对象 |
| accepted root-to-node ReactionJSON refs | 路径程序和依赖事实 |
| strategy ref + critical assumption | 无法从分子结构恢复的搜索意图 |
| map provenance delta | 新原子身份和 tombstone 无法从最终结构恢复 |
| ancestor identities / MCTS stats | cycle 与搜索本身所需 |
| deduplicated node-local rejection memory | 避免重复无效动作并给最近错误反馈 |
| reviewed key-event fingerprint | 避免相同化学候选重复审查 |

以下内容按需派生，不作为新的权威状态：compact path history、ReactionWitness、官能团列表、stereo ledger、strategy progress、route score、stock summary 和 UI status。

模型 I/O journal 继续用于观察和复盘，但不拥有结构、库存、Critic verdict 或 solved。

## 10. Path Repair 事务

只有同时满足以下条件才进入 `path_repair`：

1. Reviewer 给出具体 blocker；
2. blocker 跨越多步，单 leaf 重试不足以解决；
3. 路线已有值得保留的化学和结构资产。

执行顺序：

1. Reviewer 从完整路线给出短 repair intent 和 affected steps；
2. Host 从真实 DAG 计算 dependency slice 和 exact mapped boundaries；
3. Builder 在每个局部边界逐步产生一个 ReactionJSON；
4. Host 回放、重接依赖并更新 map provenance；
5. 所有局部步骤通过后原子提交；任一步失败则旧路线保持权威；
6. 对修复后的完整路线再做一次 audit。

Editor 因而仍能重排、插入、删除和协调多步，但不再复制未修改步骤、手写新 atom maps 或一次生成整条 RouteJSON。是否提交只看完整 replay 和 blocker 是否被解决，不看拓扑总分是否上升。

## 11. 与 SynthEx 论文的同异

| 能力 | SynthEx 论文 | 当前 `v9_smoke` | 目标 V9 |
| --- | --- | --- | --- |
| 模型 | `gemini-3.1-pro-preview`；Strategy 温度 0.1，next-step 0.3；无工具 | `gpt-5.6-sol/high`；结构化 worker，可观察但不靠工具调用决定化学 admission | reviewer 可与生成模型异构，但独立性必须单独验证 |
| Strategy | 一次返回三条独立一句话 steering query | 一次三卡；增加 `critical_assumption`、`critic_checkpoint`，并做一次 portfolio review | 保持当前紧凑合同，不增加第二套 score 或 admission |
| Builder | AiZ MCTS 的 next-step ReactionJSON policy，每次一个 action | 同样逐节点 top-1 action；输入增加真实 mapped leaf、紧凑 root-to-leaf 历史、split siblings 和 typed rejection | 保持单步 policy；正常路径不增加常驻 Agent |
| ReactionJSON / Host | 十类 primitive；应用图编辑后产生 mapped precursors | Host 是结构唯一权威；另维护单调 atom provenance、tombstone、Host-derived stereo reference，并拒绝循环/孤立/错误 lineage | 增加按需派生的 ReactionWitness，但不建立第二套 canonical state |
| 树搜索 | 每条 Strategy 独立 AiZ MCTS；每分支最多 25 次 LLM policy call | 三棵独立 AiZ MCTS；25 是上限，Host/MCTS/库存可更早终止；Builder 无 handoff/stop | 不增加 strategy-progress 综合分或固定步数强制扩展 |
| Critic | 完整 RouteJSON 后正向检查，最多六轮 Critic–Editor | 除 final audit 外，增加 Strategy portfolio review 和最多两次关键事件 audit | 再增加事实触发的搜索异常 audit；仍不逐步调用 Critic |
| Editor | 看完整 RouteJSON 和 Critic annotations，可增删、重排、改条件/官能团 | 看完整路线但 wire output 为 dependency-closed `replace_span`；Host 合并后全量回放；最多六次实际 Editor 调用 | Reviewer 只给 repair intent，Host 计算 dependency slice，Builder 逐边界物化并原子提交 |
| short-tail | depth 6 / 500 iterations / 1200 s；完整 AiZ solution stitching | 相同预算与 AiZ runtime；额外要求真实 mapped open precursor、目标根连通和 Host 库存复核 | 保持不变 |
| solved | 存在目标根完整路线，全部叶 full-InChIKey 命中 ZINC + eMolecules | `paper_equivalent_solved` 保持同口径；reaction validation、evidence、condition 另轴报告 | 保持分轴，不把内部 Critic verdict 变成 solved 权限 |
| Analyst | 对完成路线给出可行性和风险摘要 | 尚未实现论文 Analyst | 只做派生报告，不影响结构或 solved |
| 评价规模 | 1,098 targets；报告 13.8% exhaustive、25.0% strategic、63.9% stitched，并有专家盲评 | 目前只完成 Figure 1 三分子的工程 smoke，不能外推论文 solve rate 或专家质量 | 需要冻结全基准、等库存、成本与独立化学评价后再比较 |

因此，当前系统已经复现 SynthEx 的核心生成骨架，但不是论文的同实现：模型、Prompt、Host
严格度、审查时机、Editor wire contract 和评价规模均不同。新增部分主要针对 V8 暴露的
“命名关键反应与真实图编辑脱节、atom provenance 失真、Editor 整路重写不稳定”；这些
增强结构可审计性，不自动证明化学质量优于 SynthEx。

### 11.1 2026-08-27 三分子 25-step smoke

冻结配置为 `gpt-5.6-sol/high`、三个 Strategy、每分支最多 25 次 Builder policy call、top-1
ReactionJSON、AiZ MCTS、AiZ short-tail、同一 ZINC + eMolecules full-InChIKey oracle：

| Case | target-rooted reach | stock-closed | Host-replayable branches | final Critic |
| --- | ---: | ---: | ---: | --- |
| Figure 1-001 | 3 | 3 | 3/3 | reject / uncertain / uncertain |
| Figure 1-002 | 3 | 2 | 3/3 | reject / uncertain / reject |
| Figure 1-003 | 2 | 1 | 2/3 | uncertain / unavailable / reject |

三个 case 均为 `paper_equivalent_solved=true`，canonical materialization gap 和 false closure
均为 0；但 reaction-validated route 仍为 0，且没有一条 final Critic 为 `pass`。这证明当前
流程已跨过结构物化与库存闭合故障，尚未证明论文级化学质量，更不能解释为实验可行。

## 12. 实施顺序

### Phase 0：基线冻结

已完成。V8 tag 与正式 25-step artifact 保持只读证据。

### Phase 1：Host correctness，零模型调用

状态：**部分完成**。单调 map namespace、tombstone、Host-derived stereo、mapped precursor
identity 与完整 RouteJSON replay 已实现；通用 ReactionWitness 尚未实现。

1. 实现 route-lineage monotonic Map provenance 和 tombstone；
2. 让 `add_group` 的 fresh maps 由 Host 分配；
3. 实现按需 ReactionWitness；
4. 用冻结 artifact 回归 map 复用、IMDA molecularity mismatch 和 disconnected materialization。

Acceptance：离线 replay 正负例通过；不运行 smoke。

### Phase 2：精简模型合同

状态：**已接入 `v9_smoke`**。Strategist/Builder schema、紧凑路径历史、typed compiler
feedback 和 Host-owned precursor 已进入真实 provider 路径。

1. Strategist 收敛为 `strategy + critical_assumption + critic_checkpoint`；
2. Builder 收敛为 `checkpoint_relation + reaction_intent + operations + optional conditions`；
3. Builder 内部自检不进入输出；
4. typed compiler error 原样回到同一 leaf；
5. 删除旧执行语义和重复字段，而不是再加兼容投影。

Acceptance：provider schema 预检、保存的 I/O fixtures、one-step compiler canary。

### Phase 3：稀疏中断

状态：**部分完成**。Strategy portfolio review、关键事件 audit、fingerprint 去重和 final
audit 已实现；通用搜索异常 audit 尚未实现。

1. 实现三类事件检测和 fingerprint 去重；
2. Reviewer `audit` 输出最小 verdict；
3. reject 只重开最早相关子树；
4. uncertain 只报告。

Acceptance：冻结路径分别覆盖 pass、uncertain、reject，证明无逐步 Critic 和无整 Strategy 回滚。

### Phase 4：Path Repair

状态：**未实现**。当前仍使用 V8 兼容的 full-route Editor `replace_span` 与 Host 全量回放；
它是可运行过渡路径，不应称为事务式 Path Repair。

1. Reviewer 输出 repair intent / affected steps / must preserve；
2. Host dependency slice；
3. Builder 局部逐步物化；
4. atomic route transaction。

Acceptance：使用已有失败路线做离线修复；验证成功重接和失败保留旧路线，不重跑 Strategy/MCTS。

### Phase 5：受控实验

按 `schema preflight -> one-step -> frozen-path repair -> 5-step integration -> paper 25-step` 递进。前一层未通过时不消耗后一层模型预算；同一实验固定 target、model、stock、stop rule 和评价指标。

## 13. 评价

不合并成一个总分，分别报告：

- **Reach**：target-rooted routes、stock-closed routes、first-solved calls；
- **Structural integrity**：replay、map provenance、connectivity、materialization retention；
- **Chemical coherence**：Reviewer pass/uncertain/reject、关键事件真实性、跨步兼容性；
- **Efficiency**：calls per accepted action、tokens per materialized step、重复审查命中率、最小重跑单元。

`paper_equivalent_solved=true` 只能表示论文口径的目标根库存闭合，不能单独代表化学可靠性。

## 14. 明确不做

- 不增加第二轮/循环式 Strategy Critic 或逐步 Critic；Strategy 只做一次同合同 portfolio review；
- 不把 Strategy 编译成 required-map checklist；
- 不让 Builder 的 `checkpoint_relation` 成为关键事件执行证明或 admission 权限；
- 不给 Builder 恢复 fail、handoff、stop 或 solved；
- 不要求 Builder 输出完整路线、多候选、机制大表或长解释；
- 不建立 stereo、FG、risk、evidence、handle 等多套 ledger；
- 不把 ReactionWitness 扩成命名反应 hardcode 或化学 oracle；
- 不增加 strategy-progress 综合分、topology rollback score 或新的 admission gate；
- 不用固定步数强迫关键反应发生；
- 不让 Path Repair 重写未修改路线、预猜 maps 或覆盖旧路线后再验证；
- 不因 evaluator、UI 或日志错误重跑已经保存的模型调用。

## 15. 一句话架构

> 一个 Strategist 给方向，一个 Builder 写当前动作，一个 Host 守住事实；Critic 只在有新证据时出现，Editor 只在确需跨步协调时出现。
