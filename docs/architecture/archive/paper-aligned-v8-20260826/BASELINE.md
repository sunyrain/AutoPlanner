# Paper-aligned V8 冻结基线（2026-08-26）

## 冻结身份

- Git 引用：`paper-aligned-v8-freeze-20260826`
- 父提交：`7e162aebf3756df779a4e9b6c58eee46b982423f`
- 父分支：`agent/unify-workspace-entrypoints`
- 冻结对象：当前工作树中的源码、配置、合同、测试和文档；不改变原工作树或其 index。
- 排除对象：`results/`、`output/`、`tmp/`、`archive/` 中的生成物和历史大文件。
- 证据引用：正式运行 `v8-paper25-case1-sol-20260826-014710` 的原始结果仍保留在 `results/.autoplanner/runs/`，不复制进源码快照。

该引用是历史基线，不是运行时 manifest、readiness gate 或新的 solved 权威。查看快照使用 `git show paper-aligned-v8-freeze-20260826:<path>`；需要隔离检出时应新建 worktree，不要覆盖当前工作树。

## 基线定义

这份基线冻结的是论文对齐主流程：

```text
Target
  -> one Strategy call / three steering hypotheses
  -> one independent AiZ MCTS branch per Strategy
  -> Builder emits one node-local ReactionJSON action per call
  -> Host deterministically replays mapped graph edits
  -> complete Host-replayed RouteJSON
  -> Critic / Editor iterative repair
  -> canonical materialization and exact full-InChIKey stock audit
  -> complete-only AiZ short-tail stitching
  -> paper-equivalent reach/solved projection
```

主要实现入口：

- `cascade_planner/orchestration/sequential_strategy_director.py`
- `cascade_planner/interfaces/aizynthfinder_reactionjson_expansion.py`
- `cascade_planner/interfaces/aizynthfinder_strategy_sidecar.py`
- `cascade_planner/application/reactionjson_primitives.py`
- `cascade_planner/application/reactionjson_replay.py`
- `cascade_planner/application/routejson_compiler.py`
- `cascade_planner/application/canonical_hypergraph.py`
- `cascade_planner/application/paper_equivalent_metric.py`
- `cascade_planner/agent/codex_worker.py`
- `cascade_planner/interfaces/target_solver.py`
- `config/aizynthfinder.paper.yml`
- `docs/architecture/SYNTHEX_COMPONENT_CONTRACT.md`

## 与论文一致的边界

1. Strategy 是自然语言 steering query，不是 Host admission contract。
2. Builder 是 MCTS 的逐节点 expansion policy，每次只返回一个当前动作；一次输出一个动作不限制其内部做路线级推演。
3. ReactionJSON 使用论文列出的十类 primitive，Host 负责确定性图编辑和 mapped precursor 生成。
4. Critic 正向模拟完整路线；Editor 可重排、插入、删除或替换步骤，并修改条件和官能团。
5. 三个 Strategy 默认各有 25 次 LLM policy call 上限；short-tail 使用 AiZynthFinder 的 6 transforms / 500 iterations / 1200 s 配置。
6. solved 只要求存在一条目标根连通、全部终端叶命中冻结库存的完整路线；reaction proof、条件和文献证据是独立轴。

## 正式运行证据

正式运行：`v8-paper25-case1-sol-20260826-014710`

- 3 条 target-rooted materialized route；
- 2 条 stock-closed route；
- `paper_reach=true`；
- `paper_equivalent_solved=true`；
- `stock_comparable_to_synthex=true`；
- 69 次模型调用，输入 1,361,148 tokens，输出 433,150 tokens；
- 运行耗时约 10,004 s，未超出声明预算；
- 严格轴为 reaction-validated route 0、evidence-closed route 0，B2/B3/B5 未通过；
- 运行状态为 paused/non-terminal，不得把论文等价 solved 解释成实验可行或严格验收完成。

## 已知局限，随基线一起冻结

1. Branch 1 的阳离子级联没有形成连续的活性中心 relay，且绝对立体来源不足。
2. Branch 2 的所谓 IMDA 回放成两个断开的前体，本质是 intermolecular DA；库存闭合由错误拆分放大。
3. Branch 3 的结构忠实度最好，但核心级联包含高风险的中心 `8-endo` 自由基闭环。
4. `add_group` 目前按“当前分子最大 map + 1”分配新 map；删除原子后可能复用历史 map，造成跨步骤 O→C/O→Br 等假 provenance transmutation。该问题属于 Host map ledger，不应通过降低 admission 标准处理。
5. `strategy_execution_contract_satisfied` 在空 required-map 合同下可能出现 vacuous true；它没有 admission 权限，也不证明 Strategy 被化学实现。
6. Critic/Editor 与生成端共享同类 backbone；blocking-rate 收敛只能说明内部一致性，不能替代实验验证。
7. `sequential_strategy_director.py` 和 `target_solver.py` 均已超过 9000 行，prompt、状态机、搜索适配、投影和兼容逻辑耦合在同一文件，是下一代架构需要拆分的实现债，而不是本基线继续加规则的理由。

## 冻结后的变更政策

- 论文复现问题只能在此 tag 上重现、对照和报告；不再为了追随新架构反向改写其语义。
- 确定的 K0/K1 正确性缺陷（例如 fresh atom-map provenance、RouteJSON replay、精确库存绑定）可以在主线修复，但必须用独立回归说明，不改写历史运行结论。
- 新的反应因果表示、搜索价值、独立 Critic、两阶段 Editor 或实验闭环都进入下一代设计，不写回本冻结合同。
- 历史报告和 paper-equivalent 指标是 K2/History 证据，只在比较与发布边界使用，不得成为开发期 fail-closed 门禁。
