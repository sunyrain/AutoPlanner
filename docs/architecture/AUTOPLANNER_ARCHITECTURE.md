# AutoPlanner 架构总览

更新：2026-09-03

状态：本文是当前架构的总入口。它只汇总已经存在的运行路径和明确标注的过渡/目标能力，
不替代各组件合同，也不从设计文档反推实现完成。

![AutoPlanner 当前总体架构](../assets/current-architecture/autoplanner-architecture-overview.svg)

面向合成学者的简化说明见
[策略先行的生成式逆合成概念图](../assets/current-architecture/strategy-first-generative-retrosynthesis-zh.png)。
该图是展示投影，不参与架构或运行状态判定。

## 1. 一句话架构

AutoPlanner 是一个由 **Canonical Host** 承载的 anytime 逆合成系统：模型和搜索器只提出
候选，Host 负责编译真实结构、维护运行与预算、发布 canonical reaction graph、查询精确库存，
再把结构闭合、化学验证、证据、条件、专家路线价值和实验状态作为互不替代的轴输出。

系统的科学目标是发现并保留 route-defining synthetic insight；库存闭合只说明叶节点状态，
Host replay 只说明结构动作可执行，两者都不能替代关键反应可信度或合成学家的路线价值判断。

当前只有一个生产架构：**Canonical Host + unified Action loop**。论文复现、增强型搜索和
Program 研究只是同一宿主上的配置或尚未接管主线的研究能力，不能再当成多套版本架构。
增强型研究路径统一命名为 `self_correcting_sequential`，不拥有独立事实、状态机或运行平台。

## 2. 一次请求怎样穿过系统

1. Web、CLI 或 API 只接收目标和操作参数；`solve_target_request()` 将其编译成 target、约束、
   profile、预算、模型/provider 与冻结 stock binding。`benchmark_search` 默认只绑定标准
   `ZINC+eMolecules` full-InChIKey 索引；Builder/MCTS 与 Host 使用同一 oracle，PubChem 不再是主流程库存回退。
2. `solve_target()` 创建唯一 `RunKernel` 和唯一 `CampaignActionRuntime`，恢复、取消、任务、
   execution receipt 与资源消耗都归该 run 管理。
3. 论文/增强型 profile 进入 `SequentialStrategyDirectorRunner`：一次 Strategy 调用产生三张卡；每张卡
   驱动一个独立 AiZ MCTS 分支。
4. AiZ 选择当前 leaf；Builder 只返回一个当前 ReactionJSON 动作。它可以在内部做路线级推演，
   但不能输出 precursor 事实、库存、停止或 solved。
5. `RouteJSONCompiler` 在当前 mapped boundary 上回放动作，派生真实 mapped/unmapped precursors、
   stereo 和 provenance。失败留在当前 leaf；成功动作才进入搜索树。
   每个已完成 worker artifact 同时按稳定 task id 与完整 task-contract digest 写入 durable worker journal；
   分支状态由这些有序记录和 Host compiler 确定性重建，不另存一份可漂移的 branch snapshot。
6. 完整战略段经过 Critic/Editor 审查。full-route Editor 输出 dependency-closed `replace_span`；
   transactional Path Repair Editor 只给 rollback directive，Host 从 replay 派生包含 suffix 入口与原 terminal precursor occurrences 的完整 cut frontier，Builder 逐步重建到该边界，
   suffix 经 exact/stereo-aware 唯一同构边界重接后全量回放。任何经库存审计仍未闭合的真实开放叶，都回到同一套
   Route Builder 单步 ReactionJSON 合同；不存在独立 short-tail 模式、第二套 prompt 或 provider admission。
7. 所有可接纳结果通过一个 canonical ingestion 进入同一 AND/OR hypergraph。`DeficitFrontier`
   从当前 revision 派生下一项 materialize、validate、stock、evidence、condition、Program 或 replan
   Action；同一个 `CampaignActionRuntime` 可在后续 durable deficit 出现后恢复调度，但不存在第二个
   scheduler、ledger 或 Blackboard 权威。
8. 所有 Builder、provider、Editor、replan、条件补全和 Program 写入停止后，`Route Critic Agent`
   对每条最终 target-rooted canonical route 做一次版本绑定审核。审核输入来自 Host 的最终 mapped
   boundary；每个输入步骤只暴露 Host 分配的 `review_slot`，Critic 也只返回 `review_slot`；Host 校验
   slot 集合后回填 canonical `step_id` 与 authoritative digest，因此交换、重复或
   遗漏步骤评价都会失败。结论同时绑定
   `reviewed_route_sha256`；相同路线恢复时直接复用。`reject` 可触发同一 transactional repair；候选先保持 unselected，只有 re-Critic 为 viable/uncertain 且库存闭合不低于原路线时才一次性切换。`uncertain` 是
   已完成的化学审核结论，只有 Critic 不可用才留下可恢复缺口，并且恢复时只补 Critic，不重跑搜索。
9. proof portfolio、paper-equivalent metric、target report、Workbench 和 live UI 都从同一事实链
   派生。terminal report refresh 重新编译后通过唯一 portfolio 持久化函数更新 artifact、pointer、index 和
   `portfolio_ref`，但不重复发布 frontier 或 acceptance。展示与模型日志可以解释运行，但不能创造结构、
   proof、库存或 solved。路线 reconciliation 分别报告 Builder 草稿/接纳历史与最终 canonical edge、
   longest linear sequence、库存闭合；历史 reject 不得覆盖已经闭合的最终路线分类。
10. Web/CLI 主运行和隔离 panel 可拥有不同 `run_index.sqlite3`。需要展示的 registry 由 launcher
   显式注册到 `RunRegistryCatalog`；网站按 `(registry_id, run_id)` 联合身份分页聚合，再回到所属
   `RunIndex`/`RunKernel` 读取真实状态。Catalog 不扫描 `results/**`，也不复制 run status。

取消同样只有一条事实链：Web job 只负责发送协作式取消信号，`CampaignGateway.cancel()` 将终态写入
`RunKernel`。未收到 measured settlement 的 reservation 会留在事件账本中并投影为 `interrupted`，用于
保留迟到的真实 token、耗时和 provider receipt；它们在任何 terminal run 中都不属于 active work，
网页和 Action 时间线不得据此显示“正在思考”。

Provider 瞬时不可用使用同一条恢复链：失败的 worker record 不可重放；已经完成的 Strategy、Builder、
Critic、Editor record 继续有效。Sequential Director 返回当前 Host-replayed prefix 和缺失 task ids；
Global Director 只把它写成仍在 flight 的 RunKernel task checkpoint，不写正式 plan cache、不进入
canonical ingestion，也不结算 Director。显式 resume 恢复原冻结 context，按 journal 重建各分支，并且
只调用没有成功记录的 task。最终 settlement 一次性登记累计真实用量；暂停期间 report/Web 只把最新
in-flight checkpoint 的测量值叠加到 settled totals 作观察投影，任务结算后不再叠加，因此不会重复计费。

## 3. 分层组件图

| 层 | 责任 | 主要实现 | 是否拥有事实 |
| --- | --- | --- | --- |
| 接口层 | Web/CLI/API、任务创建、取消、只读查询 | `web/v4_*`、`interfaces/target_cli.py` | 否 |
| 请求编译层 | 解析 profile、预算、provider、stock 与约束 | `interfaces/target_solve_request.py`、`target_runtime_dependencies.py` | 只拥有已解析运行输入 |
| 运行控制层 | run、恢复、预算、Action reservation/settlement、停止 | `application/run_kernel.py`、`orchestration/unified_campaign_runtime.py` | 是，运行事实 |
| 策略搜索层 | Strategy portfolio、三分支 MCTS、Builder policy、稀疏审查 | `orchestration/sequential_strategy_director.py`、AiZ sidecars | 搜索状态；不拥有化学事实 |
| Host 编译层 | ReactionJSON 规范化、回放、maps、stereo、provenance、拓扑 | `reactionjson_primitives.py`、`reactionjson_replay.py`、`routejson_compiler.py` | 是，结构事实 |
| Canonical 科学层 | identity、reaction edges、route families、admission、reconciliation | `canonical_identity.py`、`canonical_hypergraph.py`、`routes/*` | 是，canonical 化学事实 |
| 决策层 | deficit frontier、proof portfolio、stock closure、各轴质量状态 | `deficit_frontier.py`、`proof_portfolio.py`、`paper_equivalent_metric.py` | 是，派生决策事实 |
| 验证与创新层 | reaction validation、证据、条件、Program、实验 Claim | workers/connectors、Program/Claim stores | 只在各自验证边界拥有观察；不能越权闭合路线 |
| 投影层 | registry discovery、分页任务列表、report、trajectory、Workbench、live model I/O | `runtime/run_registry_catalog.py`、`web/v4_run_catalog.py`、`target_delivery.py`、`target_job_projection.py`、`application/route_workbench*` | 否，只读派生；每个 registry 仍拥有自己的运行状态 |

## 4. Proposal 到事实的唯一通路

```text
Strategy / Builder / Critic / Editor / provider output
                         │
                         │ proposal or advisory
                         ▼
schema + typed runtime checks
                         │
                         ▼
Host graph replay / identity / topology / provenance
                         │
                         ▼
canonical admission + one hypergraph revision
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
       exact stock   validation   evidence/conditions
             │           │           │
             └───────────┼───────────┘
                         ▼
       portfolio / metrics / reports / UI projections
```

任何模型字段、provider 的 `solved=true`、合法 JSON、Critic pass 或 UI 状态都不能跳过这条通路。

## 5. 模型组件的最小职责

| 组件 | 看到什么 | 只输出什么 | 明确不负责 |
| --- | --- | --- | --- |
| Strategist | 初始为 target；后续为当前 mapped leaf + Host lineage context | 初始三张、后续一张紧凑 StrategyCard | 前体、完整路线、库存、停止、solved |
| Strategy review | 与 Strategist 相同的 lineage context + 待审卡 | 同一 StrategyCard 合同的最小修订 | admission、第二套评分/ledger、循环审查 |
| Builder | Strategy、当前 mapped leaf、已回放反应摘要、split context、最近 typed failure；repair 时只增加短 repair goal/constraints 与 Host-derived cut frontier，不重复 Final Critic 全文 | 一个 ReactionJSON expansion + 简洁条件/意图 metadata | precursor SMILES、map 分配、handoff/fail/stop、Strategy 认证 |
| Route Critic Agent | Strategy + Host-replayed 路径 + 触发原因；closeout 前还会看到最终 canonical route revision | 按 Host `review_slot` 返回逐步 pass/uncertain/reject、最早 blocker 与简洁路线整体评价；Host 回填 step/digest 身份，最终结论绑定路线 digest | 改结构、库存、solved、Host reaction proof、实验可行性证明 |
| Full-route Editor | 完整 RouteJSON + Critic annotations + 真实 mapped frontier | dependency-closed `replace_span` | 直接发布 precursor、降低 replay/admission 标准 |
| Path Repair Editor | 完整 RouteJSON + Critic annotations | 两个 step boundary、可选 coupled blocker ids、suffix compatibility、短 repair goal/constraints | ReactionJSON、map、precursor、保留/删除行事实 |
| Host | 所有结构化候选、运行输入和当前事实 | replay、canonical revision、stock/closure 与 typed diagnostics | 用命名反应或模型信心替代化学验证 |

## 6. 四个当前权威

系统只允许四类运行决定拥有独立权威：

1. **RunKernel**：run、task、attempt、恢复、取消、资源、in-flight recovery checkpoint 和 execution receipt；
2. **Canonical hypergraph**：分子 identity、mapped provenance、reaction edge、路线拓扑；
3. **DeficitFrontier + CampaignActionRuntime**：当前还应做什么，以及下一项 Action；
4. **Proof/stock/quality compilers**：在当前 revision 上计算路线闭合、最弱边/叶、库存和产品状态。

模型日志、Run Registry Catalog、Web job projection、trajectory、review bundle、Workbench、历史 baseline 和 GRIA shadow store
都是观察、历史或迁移表面，不能成为第五个事实权威。

隔离运行的展示通路固定为：

```text
panel launcher / CLI --publish-registry
                 │ explicit location binding
                 ▼
        Run Registry Catalog
                 │ (registry_id, run_id)
                 ▼
      owning RunIndex + RunKernel
                 │ status / model I/O / artifacts
                 ▼
        paginated Web projection
```

## 7. 完成状态必须分轴

| 轴 | 回答的问题 | 权威 |
| --- | --- | --- |
| structural | RouteJSON 是否可回放且目标根连通？ | Host compiler + canonical graph |
| stock | 所有 terminal leaves 是否命中绑定库存？ | exact stock oracle + Host stitching |
| paper-equivalent | 是否满足 SynthEx 的目标根库存闭合口径？ | `paper_equivalent_metric` |
| reaction validation | 每条反应是否有独立机理/映射/来源验证？ | validation workers + canonical proof |
| evidence/conditions | 来源和操作条件是否闭合？ | evidence/procedure stores |
| reviewer verdict | 最终 canonical route 是否已由管线内化学审核者检查、是否发现 blocker？ | Route Critic Agent；结论绑定最终路线 digest，独立保存 |
| configured acceptance | 是否达到用户声明的产品质量合同？ | acceptance/quality projection |

`paper_equivalent_solved=true` 不等于 reaction-validated、evidence-closed、process-ready 或实验可行。
Route Critic Agent 是管线内审核者；外部合成专家只评价路线洞察、价值和实验优先级，是独立科学评价轴，
不负责替系统补做内部审核。Critic 读取 Host 已绑定的 mapped boundary 也不会提升原 reaction proof、
放宽 canonical admission 或改变库存闭合。

## 8. Profile 与架构边界

| Profile/架构 | 当前定位 | 与宿主关系 |
| --- | --- | --- |
| `standard` / `proof` | 通用 anytime campaign | 直接使用 Canonical Host、Action loop 与多类 worker |
| `paper_synthex` | 历史兼容的正式论文 profile 名称 | 论文合同；三策略、每分支最多 25 policy calls、六轮 Critic/Editor |
| `paper_matched_reach` | 隔离的论文 reach/缩短 smoke profile | 与论文合同同骨架，可显式降低 Builder call ceiling 做 canary |
| `self_correcting_sequential` | 自纠偏顺序搜索研究 profile | 同一 Host/runtime；增加 Strategy review、证据触发 key-event audit、final audit 与事务式 Path Repair |
| GRIA | Program-first 长期目标 | 当前只有 shadow projection/store，尚未接管生产路线语义 |

## 9. 当前主要架构债

- `target_solver.py` 与 `sequential_strategy_director.py` 仍承载过多 prompt、状态机、兼容投影和搜索
  适配逻辑；下一步应按现有权威边界拆包，而不是增加新的 gate。
- 事务式 Path Repair 现按拓扑和 Critic/Editor 明示的化学耦合统一分区，并在 Builder 前检查 exact suffix compatibility；冻结 `paper_matched_reach` 仍保留论文 control 的 full-route `replace_span` Editor。下一步验证重点是联合 blocker、边界矛盾零调用终止和 uncertain obligation 闭环，而不是增加修复状态机。
- 论文 Analyst 尚未实现；现有 Critic、scoring 或 UI 摘要不能冒充 Analyst。
- 三分子 smoke 已证明 materialization 与 paper-equivalent closure，但 reaction-validated route 仍为 0；
  化学质量是当前首要科学缺口，不应再用结构 gate 数量代替。
- Program/实验能力仍是 shadow/read-only 或独立观察权威，尚未成为 canonical route 的一等
  `program_ids[]` 语义。

## 10. 修改架构时的五条规则

1. 每个新字段先声明唯一 writer、reader、authority 与生命周期；
2. LLM 输出默认是 proposal，只有 Host replay/admission 能升级结构事实；
3. 新审查只有在保护不同不变量时才可阻断，否则只能作为 diagnostic；
4. runtime、canonical graph、frontier、completion 各保留一个 owner，投影不得反向控制执行；
5. 论文复现、当前增强能力和长期 Program 目标分别标注，不能用同一个“已实现”覆盖三种状态。

## 11. 进一步阅读

- 当前实现状态与迁移门：[`CURRENT_ARCHITECTURE_STATUS.md`](CURRENT_ARCHITECTURE_STATUS.md)
- SynthEx 组件合同：[`SYNTHEX_COMPONENT_CONTRACT.md`](SYNTHEX_COMPONENT_CONTRACT.md)
- 当前统一 Action 主线：[`../MAINLINE.md`](../MAINLINE.md)
- 论文复现冻结证据：[`archive/paper-aligned-v8-20260826/BASELINE.md`](archive/paper-aligned-v8-20260826/BASELINE.md)
