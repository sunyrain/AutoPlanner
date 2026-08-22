# AutoPlanner 使用 AiZynthFinder 后仍未闭合的原因与本轮修复

日期：2026-08-20  
范围：当前工作树、`synthexfig1-001-paper-3x25-v1` 冻结报告、无付费回归  
结论状态：历史诊断保留；下述“战略搜索仍为 ChemEnzy”结论已被同日后续接线修复取代

## 后续接线修复（当前权威状态）

`paper_synthex` 的三条战略分支现已进入真实 `AiZynthFinder.MctsSearchTree`。主进程负责 Codex 调用和 ReactionJSON 宿主重放，隔离的 AiZynthFinder 进程负责 UCB 节点选择、候选动作保留、循环剪枝和回传；ChemEnzy best-first 只保留为非论文配置的显式兼容引擎。正式报告会分别记录 `strategy_tree_engine`、`strategy_ucb_active`、policy calls、MCTS iterations 和 selected branch 状态。缺少 AiZ 运行时或冻结 ZINC+eMolecules SQLite 库存时，论文模式在任何付费调用前失败。以下结论与整改项是接线前的历史诊断，用于解释旧运行，不再描述当前热路径。

## 结论

当前 AutoPlanner **使用了 AiZynthFinder，但只把它用作开放叶节点的短尾模板搜索**。三条战略分支的 LLM 逐节点搜索仍由 AutoPlanner Director 与 `ChemEnzyRetroPlanner.MolTree` facade 管理。该 facade 当前采用 `v_target` best-first 选择；vendor 的访问计数和 UCB 代码没有启用，paper 配置又将每个节点的 ReactionJSON 候选宽度固定为 1。因此“启用了 AiZynthFinder”不等于“LLM policy 已经进入 AiZynthFinder MCTS”。

当前热路径实际是：

`3 StrategyCards -> ChemEnzy MolTree best-first ReactionJSON -> host replay -> canonical admission -> stock -> AiZynthFinder short tail`

SynthEx 论文描述的关键路径是：

`3 strategies -> LLM expansion policy inside AiZynthFinder MCTS -> ReactionJSON -> RouteJSON -> Critic/Editor -> AiZynthFinder short tail`

两者在库存、短尾参数和 solved 指标上可以对齐，但在战略搜索状态、访问计数、替代动作和回溯语义上仍不等价。

## 本轮发现的确定性故障

1. Route Builder 在进入 Critic/Editor 前过早调用 `_public_branch()`，删除了私有 OR 树，只留下静态 `reactionjson_or_search` summary。Critic/Editor 改写 `branch["steps"]` 后，旧 `root_solved` 因而无法重算；需要继续展开时还会从目标根重新建树。
2. paper 模式的 AiZ 短尾在没有完整解时会导入“最佳部分路线”，随后递归产生更多开放叶。这属于 AutoPlanner enhanced repair，不属于论文 matched arm。
3. paper 模式的局部 Editor 调用错误增加 `route_call_count`，挤占每条分支的 25 次 Route Builder policy call 配额。
4. kernel 已终态 `unresolved` 后，顶层 `next_action` 仍保留终态前的 eligible scheduler action，造成“已经停止但仍有下一动作”的自相矛盾。
5. `accepted_expansion_count` 混合了 Codex 战略边和 AiZ 短尾边，不能表示 LLM 战略搜索效率。

## 已完成修复

- 私有 OR 树现在一直保留到 Critic/Editor 完成，最终 `_compile_plan()` 才执行公共只读投影。Editor 修改路线后，从 host-replayed 路线行重建一棵新的 OR 状态；旧 `succ/root_solved` 不再复用，并记录一次非权威 reset audit。
- `paper_synthex` 调用 AiZ 短尾时设置严格模式：只导入 `all_leaves_in_provider_stock=true` 的完整路线。部分路线仍保留为普通 AutoPlanner profile 的显式增强能力。
- Editor 使用独立 `editor_call_count`；不再消耗 `route_call_count`。为修复被删除后缀而新发起的 Route Builder 调用仍正常计入 25 次上限。
- 终态报告清空可执行 `next_action`，只保留历史 scheduler decision 的摘要哈希。
- 新增 `expansion_accounting`，分别报告 Director 选中步、canonical candidate、以及按 `origin_kind` 的接纳、物化、验证和 portfolio 接受数。
- RouteJSON 输出新增 `routejson_validation_scope=local_host_replay_only`、`canonical_admission_status=pending_at_director_output`，不再把本地 replay 与下游 canonical admission 混为一谈。
- OR 搜索摘要明确报告 `selection_policy=best_first_v_target`、`ucb_active=false`、`progressive_widening_active=false`。

## 对冻结旧运行的重新计数

对 `synthexfig1-001-paper-3x25-v1` 的既有生命周期只做只读投影，得到：

| 口径 | 数量 |
|---|---:|
| kernel accepted expansions | 44 |
| Director 最终展示步 | 5 |
| canonical candidates | 49 |
| Codex candidates / canonical admitted / materialized | 5 / 3 / 3 |
| AiZ candidates / canonical admitted / materialized | 44 / 41 / 41 |
| portfolio accepted | 0 |

因此旧报告的 44 不能解释成 44 个 Codex 战略节点；真正的 Codex 最终候选只有 5 个，其中 3 个进入 canonical graph。

## 仍需完成的核心能力

### P0：真正的搜索状态

必须二选一，并通过相同预算 A/B，而不能继续用名称推断性能：

1. 实现 AiZynthFinder `ExpansionStrategy`/MCTS 与 Codex ReactionJSON 的动态桥接，让 LLM 动作直接成为 AiZ 节点动作；或
2. 保留 AutoPlanner 自有 OR 树，但补齐访问计数、UCB/PUCT、progressive widening、同一分子节点的多次替代动作以及 canonical rejection 后的原位回溯，并证明它在 matched canary 上优于 AiZ MCTS。

当前实现两者都没有完全满足。ChemEnzy tree facade 能保存单次调用返回的多个 sibling，但 paper width=1 时同一分子节点不会被再次扩展，不能形成论文意义上的分支内 MCTS。

### P0：在线 canonical 反馈

Director 的 host-local ReactionJSON replay 通过后，canonical critic 仍可能因 product binding、cycle 或其他全局图约束拒绝。下一步需要把 canonical admission 变成逐节点提交结果，并将拒绝原因立即反馈给同一搜索节点；不能等整份 Director plan 输出后再发现两步不可拼接。

### P1：正式比较实验

修复后的第一项付费实验应是单目标 paired run，而不是立即扩到 1,098 个目标：

- Arm A：严格 paper width=1、完整 AiZ short-tail only；
- Arm B：相同模型、库存和调用上限，启用真正的分支内替代动作/回溯；
- 报告：LLM policy calls、canonical-admitted Codex nodes、最长连续主链、原始开放叶数、完整 AiZ stitch 数、B4、首个 B4 时间。

在该 paired run 前，不应再宣称“Codex 更强所以最终一定更好”；当前瓶颈主要是搜索状态转换效率，而不是单次化学回答能力。

## 验证

- `tests/test_aizynthfinder_sidecar.py`、`tests/test_chemenzy_reactionjson_expansion.py`、`tests/test_sequential_strategy_director.py`：54 passed。
- `tests/test_target_solver.py`：63 passed。
- 修改文件通过 Ruff 检查。
- 本轮没有启动 Codex 付费实验，也没有修改冻结旧运行。
