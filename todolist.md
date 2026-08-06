# AutoPlanner V4 统一 Anytime 架构优化 TODO

更新日期：2026-08-06
状态：实施中；W1–W6 已完成，W7 冻结清单、离线门与 190/190 预检已完成，等待最终零模型回放
适用范围：Canonical V4 主线、目标求解入口、ChemEnzy/Codex/文献/验证/Program 协同、RetroStar-190 评测

计划口径：

- 本文件是本轮唯一施工清单；架构文档解释现状，但不另建平行 TODO。
- `[x]` 只表示“实现 + 聚焦验证”均完成；仅有代码、没有验收证据时继续保持 `[ ]`。
- 所有工期均为工程净工时估算，不包含 ChemEnzy 外部模型排队、文献网络访问和 RetroStar-190 全量机器运行时间。
- 不按目标、分子难度、benchmark/scientific 标签分组；所有优化必须能用同一状态规则解释。
- 工作树已有大量归档与主线改动；实施时只做增量修改，不覆盖、不回滚用户现有变更。

## 0. 本轮结论

- [x] 不把目标分成 benchmark、scientific、procurement 等不同求解组。
- [x] 不维护 benchmark solver 和 scientific solver 两套控制流。
- [x] 所有目标进入同一个 `RunKernel`、同一个 canonical hypergraph、同一个 deficit/action scheduler。
- [x] benchmark 只是在固定预算点读取同一条 anytime 运行轨迹的 B4 投影，不是算法模式。
- [x] Codex 保留 campaign 级全局规划、文献假设、失败重规划和酶/机理 Program 设计职责。
- [x] ChemEnzy 保留目标级原生多步搜索能力；Codex 指导只能增量加权或补充候选，不能取代或裁掉原生 frontier。
- [x] proposal、reaction proof、exact evidence、stock、conditions、Program validation 分轴记录；缺少高层证明不能删除低层有效候选。
- [x] 本轮不处理“新聚焦模块超行数预算”测试；除该已批准延期项外，其余架构、功能和评测门必须通过。

## 当前架构蓝图

唯一生产链路必须收束为：

`UnifiedCampaignSpec → RunKernel → canonical hypergraph → deficit_frontier → CampaignActionOpportunity → target-blind scheduler → CampaignActionRuntime(reserve/execute/settle) → host canonical ingestion → campaign trajectory/B0–B5`

边界约束：

- `RunKernel` 只拥有运行、预算、reservation、settlement、恢复和审计权威。
- canonical hypergraph 只拥有化学事实、路线拓扑、proof、evidence、stock 和 Program 绑定权威。
- scheduler 只做 target-blind 排序，不执行能力、不写图、不授予科学真实性。
- Action handler 复用现有 provider/worker/service，不复制 ChemEnzy、Codex、文献、验证或 Program 实现。
- benchmark、CLI、API 和 Web 都只是同一 trajectory 的观察者，不拥有第二套求解控制流。

## 当前执行看板（按顺序推进）

状态口径：`[x]` 表示实现与聚焦验证均已完成；“已实现，待验证”仍保持 `[ ]`，避免把代码存在误报为架构完成。

### P0：封住当前 Action runtime 切片

- [x] 对当前未验证切片执行一次 `py_compile`，只修复真实语法/导入问题，不顺手扩散重构。
- [x] 新增 `tests/test_unified_campaign_runtime.py`，覆盖 action binding identity、RunKernel reserve/settle、cache replay、stale revision、handler unavailable、outcome digest tamper。
- [x] 将 `unified_campaign_runtime.py` 纳入 `tests/test_v4_architecture.py::V4_MODULES`，接受现有 V4 依赖门；超行数预算门继续按批准范围延期。
- [x] 验证 wrapper `other` task 不重复计算 child worker 的 model/search/validation/stock 成本，也不增加 proposal attempt count。
- [x] 验证同一 slice 中失败、无 revision 增益或 cache replay 不会形成无限 action 重试。
- [x] 聚焦集合一次通过：runtime、scheduler、condition、V4 dependency/token gate、ChemEnzy seed、legacy objective invariant、stock recovery 和 replan，共 18 tests。

### P1：让 scheduler 接管确定性主线工作

- [x] 将 post-Director materialization + reaction validation 迁入 `CampaignActionRuntime` slices。
- [x] 保持 validation diagnostics 聚合结构兼容，确保 `repair_rejected_precursor_typos()` 仍能消费拒绝明细。
- [x] guided ChemEnzy、learned template、recovery、replan、source-route/visual evidence validation 均已迁移，并补齐 structured-evidence exact-record-set 强制重验证 deficit。
- [x] stock audit、condition enrichment、evidence acquisition/binding 已注册为 handlers；兼容 stage projection 保留，但执行均经过 RunKernel Action wrapper。
- [x] 每次 Action settlement 后追加 trajectory snapshot，而非只记录 seed/closeout。
- [x] 门：`target_solver.py` 不再直接决定上述能力“何时运行”，只负责兼容输入、handler 装配和报告投影。

### P2：接管生成式与 Program 能力

- [x] target-level ChemEnzy、guided ChemEnzy、Codex global architecture 与 Codex event replan 已注册为独立非确定性 handlers。
- [x] 为 ChemEnzy native target search 保留最低服务预算；最低服务完成前不可借出，完成后只有显式 release/borrow 事件才能让 guided search 使用剩余额度。
- [x] Codex 初始全局视野与 ChemEnzy native search 非阻塞交错；任一失败不取消另一方已完成或在途结果。
- [x] literature hypothesis、enzyme/mechanism/whole-cell Program discovery/review 已接入同一 Action frontier；mechanism hypotheses 可走 canonical ingestion，enzyme/whole-cell/hybrid 保持 proposal-only 与 conventional fallback。
- [x] 门：任意目标只存在一个 action loop、一个 canonical graph、一个 trajectory 和一个终止判定来源。

### P3：完成失败边界与全量评测

- [x] 已以 Nirmatrelvir 完成 raw → normalized → selected → materialized → validated → B4 成对首损报告；确认 ChemEnzy 39/39 parity，B4 下降来自 stock/materialization 边界错位而非 provider 退化。
- [x] 不重复运行已完成 parity 的 RetroStar-001；其证据只作为 ChemEnzy 未被 V4 改坏的阳性基线。
- [x] P0–P2 门通过后已冻结 commit/config/stock/scheduler 权重；正式 RetroStar-190 仍须等待最终零模型回放关闭 W7。
- [ ] 对全部 190 目标运行统一组件消融，不做目标分组、不挑样、不逐目标调参。

## 当前关键路径与工作量预估

以下顺序是依赖顺序，不是按目标分组。W1–W5 完成前不得启动 RetroStar-190 全量评测。

| 工作包 | 目标 | 主要交付物 | 进入门 | 完成门 | 工程净工时 |
|---|---|---|---|---|---:|
| W1 | ChemEnzy target native reserve 与可审计借用 | 预算字段、双 native resource class、reservation/borrow/release/settle 事件、单测 | 当前 Action runtime 稳定 | guided/Codex/evidence 无法挤占 target 最低服务；总上限不突破 | 1.5–2.5 天 |
| W2 | 单一长期 anytime action loop | runtime loop、canonical event deficits、统一终止器、`target_solver` 兼容投影 | W1 | 主线不再由多个 phase-level `execute_action_slice()` 决定能力何时运行 | 3–5 天 |
| W3 | Codex 与 ChemEnzy 非阻塞首轮 | 同 revision 双初始 action、in-flight registry、稳定 merge/replay | W1、W2 | 任一超时/失败不取消另一方；ChemEnzy 首 proposal 不等待 Codex | 2–3 天 |
| W4 | Program validation/实验 feedback Action 化 | `PROGRAM_VALIDATE`、`EXPERIMENT_FEEDBACK_INGEST`、shadow/canonical 权限测试 | W2 | proposal-only、conventional fallback、Claim 与 canonical graph 边界不变 | 2–4 天 |
| W5 | 兼容层与入口收束 | `target_solver_compat`、CLI/API/Web 映射、resume/trajectory 一致性 | W2–W4 | 所有入口只装配同一 runtime；旧 objective 仅为展示兼容 | 2–3 天 |
| W6 | embedded 首损边界定因 | 真实失败目标复现包、逐层 route diff、修复与回归 | W1–W5 | standalone host-audited 路线在 unified trajectory 可追踪或有确定性拒绝证据 | 1–3 天 + 外部运行 |
| W7 | 冻结与预评测门 | commit/config/stock/provider/scheduler manifest、全量离线门 | W6 | 除已批准延期的超行数预算项外全部通过 | 1–2 天 |
| W8 | RetroStar-190 与论文防御包 | 四个全目标消融、paired metrics、失败分类、审稿材料 | W7 | 190 目标无挑样、无逐目标调参、结果可复现 | 2–4 天分析 + 机器运行 |

剩余工程净工时粗估为 **2–4 人日 + 外部运行时间**；W7 只剩最终 Nirmatrelvir 零模型回放与证据登记，之后进入 W8 全量 190 运行和分析。W8 日历时间仍主要受 ChemEnzy/模型运行吞吐影响。

当前验证证据：W1 合并验证 32 passed；W2 生产路径只剩一个 `run_anytime()` 调用；W3 以同 revision cohort 同时 reserve ChemEnzy 与 Codex，peer failure/replay 已验证；W4 新增 `PROGRAM_VALIDATE` 与 `EXPERIMENT_FEEDBACK_INGEST`，统一走 validation resource/RunKernel 账本；W5 已验证 completed checkpoint 新反馈重开、route-family rebound 和 Program ID 对 operational revision 稳定。W6 Nirmatrelvir current replay 为 0 新模型调用、ChemEnzy raw/normalized 39/39 parity、2 条 selected/materialized、1 条 stock closed，B4=true；Action 总量 95→64，其中 initial Director 32→1。W7 扩展集合 74 passed；除批准延期的超行数预算测试外，完整离线套件为 2632 passed、3 skipped、1 deselected、2 subtests passed；fresh blind preflight 为 190/190。RetroStar-190 正式运行尚未开始，本文件不得把 preflight 或 smoke 表述为全系统 benchmark 提升。

## 1. 不可破坏的架构约束

### 2026-08-06 实施进度

- [x] ChemEnzy proposal 新增 digest-bound route lineage，贯通 raw、normalized、host selection、canonical route family 和最终物化/库存状态。
- [x] 新增 `scripts/compare_chemenzy_embedding.py`，可直接比较 standalone provider 输出与 embedded V4 报告并定位首个损失边界。
- [x] 完成目标 001 真实成对复现：规范化路线 multiset parity、3 条 B4、首个 host/materialization/stock 边界均有记录。
- [x] 删除 benchmark B4 专用 early return、专用 finalize 和 objective gate 选择函数。
- [x] benchmark 标签不再关闭 target identity、self-evolution、evidence、condition enrichment 或 replan。
- [x] 旧 objective 标签动态对照测试证明相同输入产生相同 stage trace、gates、模型用量和 ChemEnzy route digest。
- [x] 修复统一流程暴露的 condition worker revision/idempotency 冲突。
- [x] 新增 target-blind `CampaignActionOpportunity`、确定性 `schedule_next_action()` 和 `campaign_trajectory.v1` 基座。
- [x] 将 action schedule、seed/closeout snapshots 和 trajectory 写入 target solve report。
- [x] canonical deficit frontier 采用固定最低 work policy：至少 2 条路线、L3 proof、2 个独立来源组；低配结果视图不能关闭验证和 evidence 工作。
- [x] action core 增加静态禁词门，禁止 benchmark、dataset、objective 等控制 token 进入 opportunity/scheduler 模块。
- [x] revision-bound `CampaignAction`、handler availability、`CampaignActionRuntime` 和 seed materialize/stock action slice 已通过 P0 聚焦验证。
- [x] Action pointer 改为 Windows-safe 的复合摘要短键，并修复 `ArtifactStore.write_pointer()` 对分层 pointer 父目录不创建的问题。
- [x] exact evidence 变化现在生成 canonical forced-revalidation deficit；新 proof 记录所见 exact/procedure/source-binding ID 集，避免无休止重复重验证。
- [x] source-route 与 visual-route 分阶段验证并按 edge ID 收窄归因，避免 literature/visual validation receipt 混算。
- [x] ChemEnzy 拆为 `CHEMENZY_TARGET_EXPAND` 与 `CHEMENZY_FRONTIER_EXPAND`，避免原生目标搜索和 guided subtarget 互相误选。
- [x] canonical frontier 新增 target native-search 与 global-architecture deficits；target-level ChemEnzy、guided ChemEnzy、stock recovery 和 Codex initial architecture 均经过 Action runtime。
- [x] Action outcome 改为持久化完整 canonical handler result，cache replay 不再截断 Director plan。
- [x] Nirmatrelvir 已完成 standalone/embedded 成对首损报告：39/39 route parity，首损为 host portfolio 截断；B4 失败根因是 stock action 与 materialization 边界错位，修复后 B4=true。

### 1.1 单一事实所有者

- [ ] `RunKernel` 继续作为运行身份、任务 reservation、预算、attempt、settlement、恢复和事件审计的唯一权威。
- [ ] `canonical_retrosynthesis_hypergraph.v1` 继续作为分子、反应边、路线拓扑、来源绑定和库存事实的唯一化学权威。
- [ ] `deficit_frontier.v1` 继续作为未完成工作的唯一 canonical 投影；不得新增第二套 queue、blackboard 或 Program frontier 权威。
- [ ] proof portfolio/acceptance 只从 canonical 状态派生，不接受 provider、Codex、UI 或 benchmark harness 的直接“已完成”声明。
- [ ] Program、实验和创新能力继续通过同一 canonical frontier 的只读或显式准入路径工作，不创建第二个 campaign state。

### 1.2 目标盲化

- [ ] scheduler、RunKernel、action scorer 和 provider admission 不得读取 dataset ID、target index、target name、`RetroStar`、`benchmark_search` 或类似标签。
- [ ] target name 只用于显示、日志和目录索引，不参与候选生成、排序、预算和停止。
- [ ] benchmark stock、采购 stock、in-house stock 统一实现为不可变 Stock Oracle 输入；核心只能读取逐分子的可用性、价格或来源事实，不能读取库存的用途标签。
- [x] 为 scheduler 核心目录增加静态禁词测试，阻止以后重新引入 dataset/objective 特判。
- [ ] 同一 canonical 状态、同一预算向量、同一 provider 结果必须产生相同 action 排序，与旧 `objective_mode` 值无关。

### 1.3 单一算法与 anytime 语义

- [x] 所有 proof axes 始终计算和报告，不再按模式禁用 reaction validation、evidence、conditions、self-evolution 或 Program review。
- [x] B0–B5 是同一运行轨迹上的成熟度快照，不是互斥模式。
- [ ] 内核只因预算耗尽、无可执行动作、连续低边际收益、显式用户取消或不可恢复错误而终止。
- [ ] 达到 B4 或 B5 只记录 milestone，不在核心内触发模式专用 return。
- [ ] 产品侧“拿到第一条库存闭合路线后停止/通知”实现为外部订阅或取消策略；论文评测使用固定预算 cutoff。

## 2. 完成定义

本轮只有同时满足以下条件才算完成：

- [ ] 主线不存在 benchmark 专用求解分支、专用 finalize、专用 replan 禁用或专用 action 开关。
- [ ] 每个任意 SMILES 都通过同一个 action loop 调用 ChemEnzy、Codex、文献、验证、库存和 Program 能力。
- [ ] ChemEnzy 在相同请求、随机种子和环境下的原始 proposal 集合与独立运行一致。
- [ ] 嵌入 V4 后，ChemEnzy 已发现且通过统一 host gate 的路线不会因缺证据、缺条件或 Codex 未选择而消失。
- [x] Codex 初始全局规划不阻塞目标级 ChemEnzy 搜索，后续 replan 由状态事件触发而非目标类别触发。
- [ ] 同一运行可连续导出 time-to-first-route、B1、B2、B3、B4、B5 和 Program milestone。
- [ ] RetroStar-190 对 190 个目标使用同一 commit、同一配置、同一 scheduler、同一预算规则和冻结 stock hash。
- [ ] 完成 ChemEnzy-only、Codex-only、固定调度和统一自适应调度的全目标组件消融。
- [x] 除已延期的超行数预算模块外，focused tests、完整离线测试、Ruff、架构门和 `git diff --check` 全部通过。
- [ ] `docs/MAINLINE.md`、当前架构状态、CLI/API/Web 语义和实际代码一致。

## 3. 先冻结基线与变更边界

- [ ] 记录本轮开始时的 branch、HEAD、tracked modifications/deletions、untracked files 和工作树摘要。
- [ ] 明确本轮只修改统一调度相关主线文件、测试、benchmark harness 和文档；不覆盖或回滚现有旧代码归档改动。
- [x] 保存当前三个已成功 RetroStar 目标的原始运行目录、配置、provider 输出、B0–B5、耗时和资源账本。
- [ ] 选择至少一个当前失败目标，保存独立 ChemEnzy 成功而嵌入 V4 失败的成对复现包。
- [ ] 对成对复现包锁定：canonical target、stock hash、ChemEnzy 环境、模型/模板版本、preset、seed、迭代数、top-k、timeout 和 worker 数。
- [ ] 分别记录四个边界的候选数量：ChemEnzy raw、provider normalized、host admitted、B4 stock-closed。
- [ ] 建立“路线在哪里丢失”的逐边差异报告，禁止只比较最终 success boolean。
- [ ] 固定本轮 publication benchmark 配置；调度参数只能在独立开发样本或合成单测中确定，不能依据 190 个测试目标逐个调整。

交付物：

- [x] `docs/architecture/UNIFIED_ANYTIME_BASELINE_20260806.md`
- [x] 机器可读 baseline manifest：`benchmarks/retrostar190_w7_freeze_20260806.json`，包含代码、环境、stock、预算和 provider 摘要。
- [x] standalone ChemEnzy 与 embedded V4 的 route/proposal diff 工具。

## 4. 统一运行契约

### 4.1 拆除 objective 对核心的控制权

- [x] 将 `TargetObjectiveMode` 标记为兼容输入，不再传入 scheduler、replan、acceptance recording 或 stop decision。
- [x] 删除 `_objective_gate_name()` 和 `_objective_gate_achieved()` 对运行控制的作用。
- [x] 删除 `_finalize_benchmark_search_objective()` 及其专用 closeout 路径。
- [ ] 删除 `benchmark_search_can_finish_at_B4` 等会改变控制流的语义字段；B4 改为通用 milestone。
- [x] 删除 `objective_mode != "benchmark_search"` 对 self-evolution、evidence、conditions 和 replan 的开关。
- [x] 删除 `benchmark_search_completed` 作为独立终态；兼容字段只表达 B4 milestone，不再控制 kernel 或 disposition。
- [ ] CLI/API/Web 暂时接受旧 objective 值，但只能转换为展示视图、通知阈值或兼容字段，并记录 deprecation warning。
- [ ] 最终从 V4 新建运行表单移除“求解算法模式”，改为“结果视图/运行预算/库存来源/约束”。

### 4.2 新的通用输入

- [ ] 定义 `UnifiedCampaignSpec`，只包含 target、不可变 stock-oracle reference、约束和多维资源预算。
- [ ] target 约束允许表达禁用试剂、最大步数、允许执行域、安全限制和库存来源，但不得表达 dataset 名称。
- [ ] 将现有 `RetrosynthesisAcceptanceSpec` 改为质量审计/里程碑编译输入，不再直接选择算法分支。
- [ ] 所有 runs 始终产生完整 `CampaignQualityState`：topology、reaction validation、exact evidence、stock、conditions、procurement、Program validation、diversity。
- [ ] acceptance 可以在任意 snapshot 上被判定为 true，但不自动终止 action loop。

### 4.3 多维资源账本

- [ ] 把预算明确拆成 native search、model、evidence、validation、visual、Program/experiment 和 total wall-time 维度。
- [x] 将当前单一 `native_search` resource class 拆为 `native_search_target` 与 `native_search_frontier`；二者共享硬总上限，但拥有不同 reservation 规则。
- [x] 在 `RetrosynthesisRunBudget` 增加 target-native 最低服务、guided 上限/可借额度和借用策略字段；旧配置缺省时由 `max_attempt_runs` 派生兼容值。
- [x] target reserve 只有通过显式 release 事件才改变保护量；guided borrow 在 task reservation 中记录额度、原因和绑定的账本摘要。
- [x] Codex、evidence、validation、visual 和 Program 不写入 native-search 账本，也不会隐式减少 native expansion reserve。
- [ ] 每种 Action 在 reservation 前声明预计资源，在 settlement 后记录实际资源。
- [x] 不允许文献/模型 Token 消耗隐式减少 ChemEnzy expansion 限额。
- [ ] 不允许 ChemEnzy 超时吞掉未启动的 Codex/验证任务状态；所有未执行动作保留明确原因。
- [x] 所有目标使用同一 native-search 借用规则；借用留痕且硬总上限由 RunKernel 强制执行。其他资源维度的通用借用继续在 W2/W4 后扩展。

W1 验收不变量：

- [x] `target_spent + target_reserved + frontier_spent + frontier_reserved <= native_total_limit` 始终成立。
- [x] target 最低服务未满足时，frontier action 即使调度分更高也不能获得该 reserve。
- [x] target 最低服务满足或显式释放后，frontier 才能通过 `borrow_granted` 使用基础配额外容量。
- [x] replay/resume 后 resource ledger、可用额度和借用状态与原运行一致。
- [x] 旧预算对象缺少新字段时从旧 `max_attempt_runs` 派生有限默认值，不会获得无限预算。

## 5. 统一 Action SPI

### 5.1 通用契约

- [x] `CampaignAction` 执行契约的 revision-bound identity、reservation/execute/settle、cache replay 和 fail-closed artifact binding 已通过聚焦验证。
- [ ] 新增 `ActionEstimate`：成功概率区间、预期 route/proof/diversity 增益、依赖解除量、成本和不确定性。
- [ ] 新增 `ActionResult`：不可变工件引用、实际资源、状态、material events、候选/事实增量和失败类型。
- [x] materialization、reaction validation、stock、conditions、evidence acquire/bind、target/guided ChemEnzy、Codex architecture/replan 与 Program discover/review/admit 已通过 RunKernel Action wrapper。
- [x] Program 专项 validation、实验 feedback 与后续新增能力也必须通过 RunKernel reserve → execute → settle，不得绕过账本直接修改 canonical graph。
- [ ] 只有 host canonical ingestion 能把 proposal 提升为图事实；Action producer 永远没有直接 authority。
- [x] stale revision、cache replay、handler failure、CAS 摘要篡改和完整 canonical outcome replay 已由聚焦单测锁定。
- [ ] 部分失败、timeout、取消、资源释放和跨进程恢复策略随 W1–W4 补齐。

### 5.2 适配现有能力

- [x] 将 target-level ChemEnzy proposal 封装为 `CHEMENZY_TARGET_EXPAND`。
- [x] 将 guided ChemEnzy subtarget 展开封装为 `CHEMENZY_FRONTIER_EXPAND`。
- [x] 将 Codex 初始全局架构封装为 `CODEX_GLOBAL_ARCHITECTURE`。
- [x] 将事件重规划封装为 `CODEX_GLOBAL_REPLAN`。
- [x] 将 exact literature/source 获取与绑定拆成可审计 Actions。
- [x] 将 mapping/reaction validation、stock audit、condition prediction 分别封装为确定性 Actions。
- [x] 将 enzyme、whole-cell/hybrid、mechanism Program discovery/review 映射为同一 action 空间中的 Program Actions；专项 validation/实验 feedback 继续使用 shadow frontier。
- [x] 新增 `PROGRAM_VALIDATE`，只消费 Program 专项 validation deficit；无外部结果时保持 pending request，不直接把 enzyme/whole-cell proposal 伪装成普通 reaction edge。
- [x] 新增 `EXPERIMENT_FEEDBACK_INGEST`，反馈重新经过现有 domain validation/Claim oracle；默认不写 shadow store，且永不直接创建 canonical reaction edge。
- [x] 保留现有 provider SPI、worker runtime、experiment dispatch/settlement 与 experimental Claim store；Action 只做适配，不复制执行实现。

建议新增或收束的模块：

- [x] `cascade_planner/application/campaign_actions.py`
- [x] opportunity 编译保留在 `campaign_actions.py`，不额外制造 `action_opportunities.py` 重复所有者。
- [x] `cascade_planner/application/action_scheduler.py`
- [x] `cascade_planner/application/campaign_trajectory.py`
- [x] `cascade_planner/orchestration/unified_campaign_runtime.py` 已实现并纳入 V4 架构门。
- [ ] `cascade_planner/interfaces/target_solver_compat.py`

### 5.3 单一长期 Action Loop

- [x] 将 `target_solver.py` 中 phase-level `execute_action_slice()` 调用收束到一个长期 `CampaignActionRuntime.run_anytime()`；solver 只负责输入兼容、handler 注册、事件订阅和报告投影。
- [x] phase 名称只允许作为 trajectory/view 标签，不能决定注册哪些 action、何时启动 action 或何时停止内核。
- [x] supplemental event deficits（Program discovery/review/admit、Codex replan）现先写为 canonical `action_signals`，由唯一 `deficit_frontier` 投影，执行后显式 resolve；solver 不再临时拼接第二份工作集。
- [x] `CampaignActionRuntime.run_anytime()` 每轮从最新 graph revision 重编译 opportunity set，并具备 bounded no-action/low-gain 收敛；生产调用点已合并为一个。
- [x] W2 同步执行路径只在 RunKernel/runtime 保留 action 状态，不创建第二套 queue；完成结果按 input revision、幂等键、稳定 action ID 合并。真正并发的 in-flight registry 属于 W3。
- [x] 唯一终止来源为预算耗尽、无可执行 action、连续低边际收益收敛、显式取消或不可恢复错误；B4/B5 只记录 milestone。
- [x] 兼容 stage report 从统一循环产生的 trajectory/backlog 投影生成，不能反向驱动 action loop。

W2 验收门：

- [x] 生产路径中只保留一个 scheduler loop 调用点；`target_solver.py` 不再逐阶段调用 ChemEnzy/Codex/evidence/Program runtime。
- [x] canonical action signal 与普通 graph settlement 都会刷新同一 deficit frontier；原 29 个 slice 调用点已改为只读兼容投影。
- [x] 相同初始 state、预算和 action outcomes 在 fresh/replay/resume 下得到相同 action trace 与 canonical digest。
- [x] handler unavailable、no-action 和 repeated no-gain 可有限收敛，不形成 busy loop 或隐式重试。

## 6. 单一 Deficit-Driven Scheduler

### 6.1 输入和禁区

- [ ] scheduler 只读取 canonical state、deficit frontier、route Pareto archive、最近 action outcomes 和剩余预算。
- [ ] scheduler 不读取 objective、dataset、target label、benchmark manifest path 或 UI view。
- [ ] stock oracle 名称在进入 scheduler 前移除，只保留 molecule-level facts 和 oracle digest。
- [ ] scheduler 输出 Action 排序与解释，不直接执行动作或修改图。

### 6.2 通用优先级

- [ ] 昂贵动作前先执行 identity、元素守恒、循环、重复和明显非法结构检查。
- [ ] 没有库存闭合路线时，route discovery/stock closure 获得统一的最低服务保障。
- [ ] 已存在可物化候选时，确定性 materialization/validation 不应被新的模型猜测长期饿死。
- [ ] 搜索停滞、路线族单一、共享瓶颈或关键边反复失败时，提高 Codex 全局重构和替代路线动作价值。
- [ ] 已有完整路线但 proof/evidence/conditions 开放时，逐步提高验证和证据动作价值；这些动作不能反向删除路线拓扑。
- [ ] 常规路线存在高代价连续区间、特定选择性瓶颈或已知能力匹配时，提高 Program discovery/review 价值。
- [ ] 负结果和不确定结果降低精确边界上的重复动作价值，但不能全局禁用 capability。

### 6.3 排序模型

- [ ] 使用可解释的确定性初版，而不是立即训练黑箱 scheduler。
- [ ] 初版分数至少包含 route closure gain、proof gain、diversity gain、dependency unblock、novelty、success likelihood、cost 和 risk。
- [x] 所有权重在运行 RetroStar-190 全集前冻结并写入 manifest。
- [ ] 对相同分数使用稳定 action ID 做 deterministic tie-break。
- [ ] 每次选择记录候选 actions、各分量、被选原因和未选原因。
- [ ] 增加连续低收益检测；达到阈值后记录 `converged_low_marginal_gain`，而不是伪造 acceptance。

### 6.4 公平调度而非样本分组

- [ ] 对 action classes 使用所有目标一致的 minimum service guarantee 和预算借用规则。
- [ ] ChemEnzy、Codex、evidence 和 validation 可并发，但共享同一事件循环、in-flight registry 和 canonical state。
- [x] 初始 ChemEnzy/Codex cohort 结果按 revision、幂等键和稳定 action 顺序合并，cache replay 不重复 reservation/settlement。
- [ ] 任何 action class 都不能因目标来自某个数据集而被开启、关闭或获得额外预算。
- [x] 初始状态同时生成 target-native 与 global-architecture opportunities；同 revision cohort 保证两者都获得启动机会，而不是用目标类别选择先后。
- [ ] 第一版并发只允许 runtime 管理的 bounded workers；禁止为了并发另建后台 scheduler、Blackboard 或 phase queue。

## 7. 单一候选图与 Pareto 保留

- [ ] 所有 ChemEnzy、Codex、文献、模板、人工和 Program 候选进入同一个 canonical ingestion 边界。
- [ ] 在 raw proposal、normalized proposal、host admission、reaction proof、stock closure 之间保留完整 provenance。
- [ ] 明确区分 `rejected-invalid`、`quarantined-reviewable`、`admitted-unproved`、`validated` 和 `accepted`。
- [ ] 缺 exact evidence、条件或采购事实只能降低对应 proof axis，不能删除合法拓扑。
- [ ] Codex 未选中不能成为删除 ChemEnzy 分支的理由。
- [ ] guided ChemEnzy 只能新增局部搜索，不能替换 target-level native route pool。
- [ ] 路线排序使用多维向量/Pareto dominance；不得用一个科学成熟度总分提前淘汰结构上有效的路线。
- [ ] 路线向量至少包含 topology closure、stock closure、reaction feasibility、proof/evidence、conditions、diversity、cost/length 和 Program readiness。
- [ ] 只在所有关键维度被另一候选支配，或明确违反硬化学约束时淘汰候选。
- [ ] 保留 conventional edge route 作为 enzyme/mechanism/whole-cell superstep 的显式 fallback。

## 8. ChemEnzy 性能保护与嵌入不变量

### 8.1 原生搜索不变量

- [ ] 相同 target、stock、preset、seed、迭代、top-k、timeout 和模型文件必须生成相同 raw proposal digest。
- [ ] V4 的 target-level ChemEnzy 调用参数与成功的高级配置逐字段对比并固化。
- [ ] ChemEnzy target search 从运行开始即可执行，不等待 Codex、文献或条件模块。
- [ ] Codex 初始计划失败、超时或合同拒绝不能取消已经运行或已完成的 ChemEnzy 搜索。
- [x] target-level route reserve 与 guided frontier reserve 分开记账；guided search 不得挤占原生搜索保底预算。
- [ ] provider normalization 不改变反应方向、前体 multiplicity、立体化学或路线连通性。

### 8.2 路线丢失定位

- [x] 为每条 ChemEnzy route 生成贯穿 raw → normalized → admitted → materialized → validated → stock-closed 的 trace ID。
- [ ] 对 standalone 成功、embedded 失败目标生成逐步差异和首个丢失边界。
- [ ] 检查过早 product audit、identity gate、atom mapping、库存规范化、路线去重和 portfolio 截断是否误删路线。
- [ ] 把“证明不足”与“化学非法”彻底拆开，只有后者允许在搜索视图隐藏。
- [ ] 建立 host-audited ChemEnzy baseline，避免拿未经统一化学门的 raw success 与 V4 B4 直接比较。

### 8.3 性能验收

- [ ] 对固定 provider 请求实现 raw proposal 100% digest parity。
- [ ] 任何 standalone-host-audited B4 路线都必须在 unified trajectory 中可追踪；若未进入 B4，必须有逐边确定性拒绝证据。
- [ ] 初始 ChemEnzy time-to-first-proposal 不因同步等待 Codex 增加。
- [ ] unified run 的 native-search expansion 数不得低于同预算 ChemEnzy baseline。
- [ ] 在全 190 结果上预注册非劣界；基线测量后冻结阈值，冻结后不得根据测试结果修改。

## 9. Codex 总控的统一职责

### 9.1 初始全局视野

- [x] 每个目标都可获得同一 bounded global architecture action；不得只给 scientific 目标调用。
- [x] 初始 Codex action 与 ChemEnzy target search 通过同 revision cohort 非阻塞并发执行。
- [x] Codex 冻结上下文包括 target、canonical route pool、共享中间体、open leaves、库存事实、失败、proof deficits、Program opportunities 和剩余预算。
- [x] ChemEnzy 尚未返回时允许 Codex 基于冻结的 target preflight/context 形成初始假设；两方结果随后通过同一 canonical graph 增量 union。

### 9.2 状态触发重规划

- [ ] replan 触发器只依赖物质事件和状态变化：关键边拒绝、新路线族、新 exact evidence、库存变化、共享瓶颈、搜索停滞和路线多样性不足。
- [ ] 去除 benchmark 模式对 depth、validation、evidence replan reason 的屏蔽。
- [ ] replan 输出只能追加候选、调整优先级或提出替换 Program；不能删除既有 canonical 路线。
- [ ] 每次 replan 报告前后 route family、edge、stock closure、proof 和资源增量。
- [ ] `no_gain`、失败和超时作为 scheduler 学习/审计信号保留，不倒推污染已有事实。

### 9.3 文献与创新

- [ ] Codex 负责 source-consistent 文献路线假设和检索策略；exact authority 仍由 host 文献 connector 和绑定门授予。
- [ ] Codex 负责识别可被酶、whole-cell/hybrid 或机理一跳替换的长区间；Program 编译和专项验证继续由 host 严格执行。
- [ ] 文献外假设永远标为 hypothesis，不因 Codex 自信表达升级 proof。
- [ ] Program 候选必须精确绑定输入/输出状态、replaced-edge span、capability/precedent、验证计划和 conventional fallback。

## 10. Anytime 轨迹、里程碑与输出

- [x] `campaign_trajectory.v1` 已在每个 Action settlement 后追加 snapshot，并记录 milestone、资源与 action decision。
- [ ] 每个 snapshot 记录 graph revision、wall time、各资源维度、action counts、route counts、Pareto archive 和 B0–B5。
- [ ] 记录 time-to-first-B1、time-to-first-host-valid-route、time-to-first-B4、time-to-B3/B5 和 Program milestones。
- [ ] 每个 snapshot 绑定代码、配置、stock oracle、provider/model 和输入摘要。
- [ ] resume 后继续同一 trajectory，不能从新的计时/预算基线伪造性能。
- [x] W5 允许“已完成 checkpoint 收到新的 canonical action signal/实验反馈”时通过带工作指纹的 `run_reopened` 事件重新进入同一 action loop；不能直接刷新旧终态报告而跳过反馈。
- [ ] benchmark harness 只读取固定 cutoff 的 trajectory projection，不向 solver 传 benchmark mode。
- [ ] Workbench 同时展示当前最优路线和历史上曾达到的 milestone，避免后续 proof 撤销被最终状态掩盖。
- [ ] 导出 action trace、失败 trace、route lineage 和资源曲线，供审稿复核。

## 11. CLI、API 与 Web 迁移

- [ ] CLI 新建运行参数改为 target、stock oracle、constraints、budget 和可选通知条件。
- [ ] 保留旧 `--objective-mode` 一段兼容窗口，但将其转换为 client view，不进入 `solve_target()` 核心。
- [ ] API request schema 将 objective 标为 deprecated；response 返回统一 milestones 和 trajectory links。
- [ ] Web 的“Benchmark 检索闭合/科学证明/采购交付”改为结果查看深度或提醒条件，并明确不会改变算法。
- [ ] Web 可实时显示 ChemEnzy、Codex、evidence、validation、Program actions 在同一时间线中的状态。
- [ ] 所有入口经同一个 `UnifiedCampaignSpec` 和 `unified_campaign_runtime`，不得在 Web/CLI 复制调度逻辑。
- [ ] saved-run 恢复兼容旧 objective 字段，但 replay 后的动作决策必须使用新统一规则。

## 12. 测试设计

### 12.1 架构与静态门

- [ ] scheduler/RunKernel/action modules 禁止出现 dataset 和 objective 特判字符串。
- [ ] `target_solver.py` 不再包含 benchmark 专用 finalize 或 B4 early return。
- [ ] V4 主线不得重新依赖 legacy frontier、Blackboard、旧 route portfolio 或旧 controller。
- [ ] 新 Action modules 不能绕过 RunKernel 或 canonical ingestion。
- [ ] 继续执行现有 V4 dependency gate；超行数预算项按已批准决定单独跳过，不借机扩大豁免范围。

### 12.2 单元测试

- [x] 相同 state/budget、不同旧 objective 值产生相同 action candidates 和排序的 focused test。
- [ ] 相同 state 在 replay、resume 和并发完成顺序变化下产生相同 canonical digest。
- [ ] Pareto archive 保留 topology-valid/evidence-open 路线。
- [ ] 缺证据、条件或 Program validation 不删除 B1/B4 路线。
- [ ] Codex 未选择的 ChemEnzy route 仍可继续 materialize/validate/stock audit。
- [ ] guided ChemEnzy 不替换 target-level route pool。
- [ ] stock oracle digest 改变只更新 stock facts 和后续 deficits，不改变代码路径。
- [ ] budget timeout 和 cancellation 仍需在长期 loop 中闭合；native borrowing、reservation、settlement 已可重放。
- [x] milestone 达成不触发核心 early return。
- [x] `CampaignAction` binding 对同 revision/decision 稳定，对 revision 变化生成新 execution identity。
- [x] `CampaignActionRuntime` reserve/settle、cache replay、stale revision、handler unavailable 和 digest tamper 行为稳定。
- [x] action wrapper 资源账本不与 child worker/provider 成本双计。

### 12.3 集成测试

- [ ] 用旧 benchmark/scientific/procurement 标签启动同一输入，在相同预算前缀内 action trace 完全一致。
- [ ] ChemEnzy 与 Codex 初始动作互不阻塞；任一失败时另一方结果仍可进入 graph。
- [ ] 文献获取、reaction validation、stock audit 和 condition enrichment 都由 open deficits 触发。
- [ ] conventional route 与 enzyme/mechanism Program 在同一 campaign 中共存且 fallback 不丢失。
- [ ] B4 后继续运行可自然获得 B3/B5，不创建第二个 run。
- [ ] resume、checkpoint、API、Web、CLI 和 artifact export 读取同一 trajectory。

### 12.4 回归集

- [ ] 当前成功的 3 个 RetroStar smoke 目标继续成功。
- [x] standalone 成功而 embedded 失败的 Nirmatrelvir 代表目标已完成定因和通用修复。
- [x] Nirmatrelvir current-V4 zero-model replay 保持 raw/normalized parity，B4=true。
- [ ] 至少一个文献驱动真实目标保留 exact-source 路径。
- [ ] 至少一个 enzyme superstep 阳性候选和一个无适用酶负对照保持严格语义。
- [ ] Program shadow store、实验 Claim 和 canonical graph digest 不受统一 scheduler 重构污染。

## 13. RetroStar-190 全量评测

### 13.1 冻结协议

- [x] 继续使用 `retrostar190_v4.protocol.json` 中冻结的 190 targets 和约 23M eMolecules stock，并重新验证 hash。
- [x] 190/190 fresh preflight 证明 planner 输入只含 opaque target SMILES、stock oracle 和统一预算，不暴露 reference route、target index 或 dataset name。
- [x] 在运行全量测试前冻结代码 commit、配置、scheduler 权重、环境和模型摘要；当前 contract 未暴露 provider seed，manifest 已如实记录无冻结 seed 与远端权重非位级冻结边界。
- [ ] 如果因成本分批执行，所有批次必须使用完全相同的配置和规则；批次只用于运行管理，不能形成不同算法组。
- [ ] 所有失败、timeout、空结果和部分结果纳入汇总，禁止只报告成功样本。

### 13.2 对所有 190 个目标执行的组件消融

- [ ] ChemEnzy-only：统一 host admission、validation 和 stock audit，Codex Actions 不注册。
- [ ] Codex-only：ChemEnzy Actions 不注册，其余 host gates 相同。
- [ ] Unified round-robin：所有 Actions 注册，但用固定公平顺序替代价值调度。
- [ ] Unified adaptive：所有 Actions 注册，使用统一 deficit-driven scheduler。
- [ ] 可选机制消融必须同样覆盖全部 190，不允许只给挑选目标启用。
- [ ] 每个消融使用相同 target manifest、stock oracle、报告代码和相应预算说明。

### 13.3 指标

- [ ] RetroStar 可比主指标：固定预算内至少一条 host-admitted、target-rooted、终端叶全部在冻结 stock 的路线。
- [ ] success@固定 wall-time、success@native expansions、success@total resource envelope。
- [ ] time-to-first raw proposal、B1、host-valid route 和 B4。
- [ ] top-k 完整路线数、不同 edge-set/route-family 数、路线长度和共享中间体。
- [ ] raw → normalized → admitted → B4 的召回损失分解。
- [ ] B2 reaction validation、B3 exact evidence、B5 configured scientific acceptance 分开报告，不重定义 RetroStar solved。
- [ ] Codex calls/tokens、ChemEnzy expansions、证据请求、验证次数、wall time、失败率和恢复次数。
- [ ] 逐目标 paired differences、置信区间和失败类型，不只给平均值。

### 13.4 失败分析

- [ ] 将失败归为 provider 无候选、normalization 损失、host chemistry rejection、搜索深度不足、stock miss、budget/timeout、canonical merge、portfolio/展示遗漏。
- [ ] 对每类保留 representative trace，但结论必须基于全量计数。
- [ ] 区分 ChemEnzy 本身搜索失败与 AutoPlanner 嵌入造成的路线丢失。
- [ ] 检查 Codex 是否提高 route diversity、修复停滞或反而消耗无增益资源。
- [ ] 检查统一 scheduler 相对 round-robin 的增益是否来自普遍状态规则，而不是少数目标偶然收益。

## 14. 审稿防御包

- [ ] 发布单一算法图，图中不出现 benchmark/scientific 两条流程。
- [ ] 声明所有目标使用同一代码、scheduler、action space 和预算规则。
- [ ] 发布 solver 可见字段白名单，证明 dataset ID 和 reference route 不可见。
- [x] 发布 stock、target manifest、配置、代码和环境 hash。
- [ ] 发布 Action 决策日志及各资源维度成本。
- [ ] 报告同一 trajectory 的 B1–B5 anytime 曲线，说明 benchmark 只是 B4 评价投影。
- [ ] 消融按组件覆盖全部目标，不做人工目标分组。
- [x] 对 scheduler 权重说明开发/冻结过程；禁止在 RetroStar-190 test targets 上逐目标调参。
- [ ] 明确 Codex 输出只具有 proposal authority，exact evidence 和 reaction proof 由独立 host gate 授予。
- [ ] 明确 enzyme/mechanism Program 是创新候选，未验证时不伪装成普通已证反应边，并保留 conventional fallback。
- [x] 如实报告目前只完成的 smoke 数量；全 190 完成前不得宣称 benchmark 整体提升。

交付物：

- [ ] `docs/evaluation/UNIFIED_ANYTIME_RETROSTAR190_PROTOCOL.md`
- [ ] `docs/evaluation/UNIFIED_ANYTIME_ABLATION_PLAN.md`
- [ ] `docs/evaluation/REVIEWER_DEFENSE_CHECKLIST.md`
- [ ] 机器可读 run manifest、per-target metrics、paired comparison 和 failure taxonomy。

## 15. 文档与清理收束

- [x] 更新 `docs/MAINLINE.md`：从固定 Codex-first 阶段图改为同一事件循环中的全局 Director + native search + unified Action frontier。
- [x] 更新 `CURRENT_ARCHITECTURE_STATUS.md`，明确当前实现与统一架构完成度。
- [ ] 更新 CLI/API/Web 文档，删除“benchmark 到 B4、scientific 到 B5 是两套运行模式”的表述。
- [ ] 更新架构演化时间线，解释为什么从阶段式 objective branching 转向 target-blind anytime scheduler。
- [ ] 归档旧 objective compatibility 说明和迁移期限。
- [ ] 新统一主线稳定后删除临时 shadow scheduler/feature flag；最终仓库只能保留一个生产控制流。
- [ ] 不恢复已迁入 legacy 的旧 frontier scheduler、Blackboard controller 或旧 acceptance 实现。
- [ ] 保持旧代码物理归档边界，不在本轮重新混入主线。

## 16. 推荐实施顺序与检查点

### Checkpoint A：先证明没有改坏 ChemEnzy

- [x] 目标 001 已完成成对复现与 raw/normalized route parity，比较工具和 route lineage 已具备。
- [x] 已定位 standalone/embedded 首个差异边界：39 条 provider route 无损进入 normalization，37 条在统一 host portfolio budget 截断；旧 B4 损失另由 stock/materialization 边界错位造成。
- [x] `docs/architecture/W6_CHEMENZY_EMBEDDING_FIRST_LOSS_20260806.md` 固化逐层证据、Action 减负和严格 validator 边界。
- [x] 门：逐路线 comparison 已记录 normalization、host selection、materialization、validation 与 stock 首损原因。

### Checkpoint B：切断模式分支

- [x] 已引入统一 milestone/trajectory，并在迁移期复用现有顺序 stages 作为兼容投影。
- [x] 已移除 benchmark 专用 early return、feature disabling 和 replan suppression。
- [x] 门：不同旧 objective 标签在同预算前缀内产生相同轨迹。

### Checkpoint C：统一 Action SPI

- [x] `CampaignAction` 与 runtime 基座已包住 materialization、validation、stock、conditions、ChemEnzy、Codex、evidence 和 Program discovery/review/admit。
- [x] Program 专项 validation 与实验 feedback 已迁入同一 Action/RunKernel 账本，Claim/shadow/canonical 权限保持分离。
- [x] 终态 checkpoint 新信号重开已在 W5 收束；timeout/cancel/resource-release 的完整组合回归仍留在后续运行门。
- [ ] 门：无旁路 canonical write，W1–W4 完成后 replay/resume digest 稳定。

### Checkpoint D：接入 deficit-driven scheduler

- [x] 已实现确定性、可解释、无训练的排序器基座和 stable action-ID tie-break。
- [ ] 实现 action-class 公平保障、预算借用和跨 slice 的连续低收益收敛；stable tie-break 已完成。
- [ ] 门：scheduler target-blind 动态与静态测试已通过；生产 action-loop 接管仍未完成。

### Checkpoint E：非阻塞 Codex + ChemEnzy

- [x] 初始全局规划和 target-level native search 通过同 revision cohort 并发启动。
- [x] Codex proposal additive、ChemEnzy frontier monotonic、canonical merge 使用稳定 action identity/order 与唯一 graph union。
- [x] 门：ChemEnzy handler 与 canonical ingestion 不等待 Codex 完成；Codex 失败不会取消已完成的 ChemEnzy action。

### Checkpoint F：科学层与 Program 回归

- [ ] B4 后继续同一 run 完成 evidence、conditions、replan 和 Program 工作。
- [ ] conventional fallback、proof authority 和 Program shadow 边界不变。
- [ ] 门：真实文献案卷与 enzyme positive/negative controls 通过。

### Checkpoint G：全量 190 与论文包

- [ ] 冻结 commit/config 后运行四个全目标组件消融。
- [ ] 汇总 anytime、资源、召回损失和失败类型。
- [ ] 门：结果可复现、无目标特判、无挑样报告、审稿防御材料完整。

## 17. 明确不做

- [ ] 不为 RetroStar-190 编写目标 ID、SMILES、名称或 reference-route 特判。
- [ ] 不按“简单分子/复杂分子”“benchmark/scientific”人工分配不同 planner。
- [ ] 不用 B4 早停掩盖后续科学证明缺口。
- [ ] 不让 evidence/condition 缺失删除结构有效路线。
- [ ] 不让 Codex 计划覆盖 ChemEnzy native frontier。
- [ ] 不把 enzyme/mechanism superstep 伪装成已验证普通 reaction edge。
- [ ] 不通过恢复 legacy scheduler 或 Blackboard 快速绕过统一设计。
- [ ] 不在全 190 test targets 上边看结果边逐目标调参。
- [ ] 不在本轮处理已明确延期的新聚焦模块超行数预算。

## 18. 当前施工序列

严格执行 W1 → W2 → W3/W4 → W5 → W6 → W7 → W8；W3 与 W4 可在 W2 基座稳定后并行设计，但合并与验收仍使用同一 action loop。不得提前启动 RetroStar-190 全量运行。

- [x] 第一刀：锁定并验证 `CampaignActionRuntime` 当前切片。
- [x] 第二刀：迁移 post-Director materialization/validation，并保持 repair diagnostics 兼容。
- [x] 第三刀：迁移 stock、conditions、evidence、guided ChemEnzy、Codex replan 和 Program handlers。
- [x] 第四刀：把 trajectory 扩展到每个 settlement，完成基础 resume/replay 一致性。
- [x] 第五刀（W1）：拆分 target/frontier native resource class，完成 target reserve、显式 release/borrow 和资源审计；W1 合并验证 32 passed。
- [x] 第六刀（W2）：`target_solver.py` 只保留一个生产 `run_anytime()`；原 29 个 phase-level slice 已改为统一 trajectory/backlog 的兼容投影，replan retention/gain 审计也已绑定统一执行。
- [x] 第七刀（W3）：Codex initial architecture 与 target ChemEnzy 已通过同一 runtime 的同 revision cohort 非阻塞启动；RunKernel 持有 durable in-flight reservation，稳定观察与 cache replay 已验证。
- [x] 第八刀（W4）：已注册 `PROGRAM_VALIDATE` 与 `EXPERIMENT_FEEDBACK_INGEST`；前者只形成待外部执行请求，后者复用现有 host gate/Claim store，默认不写 shadow 且不创建 canonical edge。
- [x] 第九刀（W5）：抽离 `target_solver_compat`，统一旧 objective 展示、checkpoint cursor、外部反馈信号和 resume/trajectory 投影；新增 route-family rebound 与 scientific-content-bound Program ID，避免 operational revision 污染 Program 身份。
- [ ] 第十刀（W6–W8）：W6 真实 embedded failure 已定位并修复；W7 冻结清单、完整离线门和 190/190 preflight 已完成，待最终零模型回放后执行 W8 全量 190、全目标组件消融与审稿防御包。
