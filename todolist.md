# AutoPlanner V4 统一 Anytime 架构优化 TODO

更新日期：2026-08-11
状态：实施中；W1–W7 已完成；W8 当前范围已在前 20/190 四臂 pilot 形成可用工程结论后收口；剩余 170 个目标及 190×4 全量运行不属于本轮待办，不得把 pilot 表述为全量结果
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
- [x] “新聚焦模块超行数预算”不阻断 W7 主线验收；主线闭合后已通过十七个独立维护切片全部偿还，超限模块由 17 个降至 0 个。

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
- [x] 将 `unified_campaign_runtime.py` 纳入 `tests/test_v4_architecture.py::V4_MODULES`，接受现有 V4 依赖门；当时批准延期的超行数预算债务现已全部偿还。
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
- [x] P0–P2 门通过后已冻结 commit/config/stock/scheduler 权重；最终零模型回放已关闭 W7。
- [x] 对 manifest 顺序前 20 个目标完成四臂统一组件 pilot，不做结果后挑样、不逐目标调参，并生成 paired metrics 与 failure taxonomy。
- [x] 当前工程范围在 20/190 四臂 pilot 后按停止判定收口；190×4 publication-scale 评测仅保留为未来独立科学门，不作为本轮未完成项。

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
| W7 | 冻结与预评测门 | commit/config/stock/provider/scheduler manifest、全量离线门 | W6 | 全部通过，包括聚焦模块行数预算门 | 1–2 天 |
| W8-P | RetroStar-190 前 20 目标 pilot | 四臂 pilot、paired metrics、失败分类、停止判定 | W7 | 四臂各 20 completed、结果可复现、论断不外推 190 | 已完成 |
| W8-F | RetroStar-190 publication-scale 防御包 | 四个全目标消融与审稿材料 | W8-P | 190 目标无挑样、无逐目标调参、结果可复现 | 非当前范围；未来独立立项时再评估 |

当前批准范围内的 W8-P 已完成，W8-F 不再计入本轮剩余工作量。若未来为 benchmark-wide 论文主张独立启动 W8-F，必须从新的 fresh root 按原冻结门执行 190×4，不能把 20-target pilot 或任何中断根拼入全量结果。

当前验证证据：W1 合并验证 32 passed；W2 生产路径只剩一个 `run_anytime()` 调用；W3 以同 revision cohort 同时 reserve ChemEnzy 与 Codex，peer failure/replay 已验证；W4 新增 `PROGRAM_VALIDATE` 与 `EXPERIMENT_FEEDBACK_INGEST`，分别进入 Program/experiment 独立 RunKernel 账本；W5 已验证 completed checkpoint 新反馈重开、route-family rebound 和 Program ID 对 operational revision 稳定。W6/W7 最终 Nirmatrelvir replay 为 0 新模型调用、ChemEnzy raw/normalized 39/39 parity、2 条 selected/materialized、1 条 stock closed，B4=true；Action 总量 95→64，其中 initial Director 32→1。W7 完整离线门已通过。W8 四臂和汇总器已实现，`-d` fresh preflight 为 760/760。提交 `a330a88` 收束预算终态后，`-i` 因外部中断保留为非结果审计。提交 `225df39` 支持冻结的 manifest-prefix pilot；`retrostar190-w8-pilot-20260807-a` 四臂各 20 completed、0 failed，B4 分别为 ChemEnzy-only 16/20、Codex-only 1/20、round-robin 15/20、adaptive 15/20。2026-08-09 启动的 `-j` 在 0 个完整 case 时因范围缩减主动停止并排除。Checkpoint F 已补齐 B4 后同轨迹 science Action 集成门，官方 EPO 三案例迁移至 parser v14 后通过严格离线 replay。2026-08-10 已完成 `UnifiedCampaignSpec`/`RunSpec v2`、不可变 Stock Oracle 绑定、八轴 `CampaignQualityState`、B4/B5 非早停、ChemEnzy/Codex 并发计算后的确定性 canonical admission、可配置且可重放的多维资源账本、Action v2 预计/实际资源归因，以及 `campaign_action_estimate.v1`/`campaign_action_result.v1` 完整契约；Director plan provenance 已与运行 task/context receipt 分离，三种旧 objective、交换并发完成顺序、replay 与 resume 的 Action binding/canonical scientific digest 回归通过。2026-08-11 修复证据阶段绕过 `enable_target_identity=False` 的 PubChem 网络旁路，并让 validation fork 复用源运行身份；目标求解集成测试改用确定性结构身份。空 condition prediction 现在只保留在 Action outcome，不再写入 canonical graph 或反复推进 revision；原 110 次同 ID condition Action 已收敛为 1 次。最终仓库门为 2699 passed、3 skipped、1 个已批准延期的行数预算门 deselected、2 subtests passed，总耗时由 500.38 秒降至 178.18 秒。随后开始增量偿还行数预算债务：`visual_evidence.py` 从 1250 行降至 595 行，请求编译/来源相关性迁入 491 行的 `visual_evidence_request.py`，canonical host materialization 迁入 190 行的 `visual_evidence_materialization.py`；facade 继续拥有 provider 执行、预算结算和 observation replay，公共导入保持不变。相关集成回归 115 passed，超预算模块由 17 个降至 16 个；拆分后的完整离线套件再次通过，结果为 2699 passed、3 skipped、1 deselected、2 subtests passed，用时 173.10 秒。第二个维护切片把 `chemenzy_probe.py` 从 1216 行降至 758 行，请求/内容摘要/结果封套迁入 102 行的 `chemenzy_probe_contract.py`，路线归一化、指纹、组合筛选和 provider metadata 迁入 398 行的 `chemenzy_probe_routes.py`；facade 继续拥有 provider 执行与 canonical ingestion，公共导入保持不变。相关集成回归 108 passed，超预算模块由 16 个降至 15 个；完整离线套件为 2699 passed、3 skipped、1 deselected、2 subtests passed，用时 172.02 秒。当前评测结论见 `docs/evaluation/RETROSTAR190_W8_PILOT20_RESULTS_20260808.md`，统一输入契约见 `docs/architecture/UNIFIED_CAMPAIGN_CONTRACT.md`，科学层门见 `docs/architecture/SCIENTIFIC_LAYER_REGRESSION_20260809.md`。

第三个维护切片把 `retrosynthesis_service.py` 从 671 行降至 264 行：Director/context/创新审查与显式实验 Claim 准入迁入 234 行的 `retrosynthesis_service_planning.py`，worker dispatch、预算终态、canonical batch 和 action signals 迁入 207 行的 `retrosynthesis_service_execution.py`。`RetrosynthesisCampaignService` 仍是唯一公开服务并继续持有 RunKernel、canonical graph store 与 WorkerRuntime，拆分没有创建第二写入入口。聚焦回归 80 passed，相关 Gateway/target solver/evidence 集成 114 passed，超预算模块由 15 个降至 14 个；完整离线套件为 2699 passed、3 skipped、1 deselected、2 subtests passed，用时 172.93 秒。

第四个维护切片把 `v4_target_runtime.py` 从 704 行降至 386 行：payload 校验、约束/预算编译和可选 evidence provider 装配迁入 interfaces 层 331 行的 `target_solve_request.py`，runtime 只保留后台 job、自动 continuation、实时进度与历史投影。请求模块放在 interfaces 而非 web，继续满足 V4 禁止依赖 `cascade_planner.web.*` 的所有权门；HTTP 行为、Gateway 调用和 `solve_target_request` facade 导出不变。Web/架构聚焦回归 30 passed，相关 Gateway/target solver/CLI 集成 85 passed，超预算模块由 14 个降至 13 个；完整离线套件为 2699 passed、3 skipped、1 deselected、2 subtests passed，用时 173.53 秒。

第五个维护切片把 `v4_api.py` 从 520 行降至 333 行：同步 target solve、后台 job 启动/列表/状态/删除、objective 兼容警告和 workspace queue visibility 迁入 230 行的 `v4_target_routes.py`。Blueprint 总装配、错误处理、run lifecycle、Program/Workbench/PDF 路由仍留在 facade；注册器通过参数接收原模块的 `_solve_target_request`、`_run_target_job` 与 payload reader，保留现有测试和运行时注入点。Web/架构聚焦回归 30 passed，相关 Web/Workspace/Gateway 集成 69 passed，超预算模块由 13 个降至 12 个；完整离线套件为 2699 passed、3 skipped、1 deselected、2 subtests passed，用时 190.90 秒。

第六个维护切片把 `literature_evidence.py` 从 607 行降至 156 行：并行检索、来源物化、授权浏览器重试、route binding、discovery/receipt 编译迁入 466 行的 `literature_evidence_connector.py`，纯确定性的摘要、discovery 压缩和 binding eligibility 迁入 58 行的 `literature_evidence_contract.py`。facade 继续拥有配置校验、resolver cache 生命周期与依赖注入；`_materialize_candidate` 在每次调用时传入执行器，保留 monkeypatch/自定义物化器行为，原候选辅助兼容导出也已保留。文献/Web/架构聚焦回归 56 passed，相关 literature/target solver/CLI/source-route/visual 集成 122 passed，超预算模块由 12 个降至 11 个；完整离线套件为 2699 passed、3 skipped、1 deselected、2 subtests passed，用时 177.17 秒。

第七个维护切片把 `literature_pdf_materialization.py` 从 266 行降至 157 行：manifest 原子发布、视觉页面选择、source-evidence 行和 target-focus 摘要迁入 159 行的 `literature_pdf_projection.py`。原 materialization 函数继续负责 PDF 大小/页数校验、内容摘要、focus/extraction 调用、全文 UTF-8 与摘要校验、procedure 编译和最终来源记录，实际获取与解析顺序不变。文献/source-route/visual/架构聚焦回归 60 passed，相关 target solver/CLI/Web/OCR/HTML 集成 106 passed，超预算模块由 11 个降至 10 个；完整离线套件为 2699 passed、3 skipped、1 deselected、2 subtests passed，用时 175.43 秒。

第八个维护切片把 `v4_route_evidence_projection.py` 从 283 行降至 171 行：条件证据状态摘要与缺口解析迁入 125 行的 `v4_route_condition_resolution.py`；原 85 行的 `v4_route_condition_projection.py` 继续只负责来源/模型条件行投影，没有混入状态解析职责。evidence projection 保留兼容导出，因此 Workbench、planned route branches 与 PDF 消费端无需改入口。Ruff、compileall、`git diff --check` 均通过，Workbench/Web/架构聚焦回归 57 passed，超预算模块由 10 个降至 9 个；完整离线套件为 2699 passed、3 skipped、1 deselected、2 subtests passed，用时 176.97 秒。

第九个维护切片把 `visual_observation_normalization.py` 从 402 行降至 295 行：RDKit canonicalization、连接性标准化、host 反应物子集准入、spectator 分区和标签映射迁入 131 行的 `visual_observation_chemistry.py`。原 normalization 模块继续拥有 provider digest 校验、链路锚定、条件候选降权、观察协议和内容摘要；`normalize_visual_observation` 公共入口与 visual evidence facade 不变。Ruff、compileall、`git diff --check` 均通过，visual/架构聚焦回归 27 passed，target solver visual 集成 3 passed，超预算模块由 9 个降至 8 个；完整离线套件为 2699 passed、3 skipped、1 deselected、2 subtests passed，用时 167.98 秒。

第十个维护切片把 `literature_procedure_fragments.py` 从 147 行降至 52 行：专利 `(1) product` 与期刊 `product (Entry 1)` 的逐行标题识别、产品名过滤和过程信号判断迁入 107 行的 `literature_procedure_line_fragments.py`。原 facade 继续优先调用 line parser，并在无逐行片段时回退到压平文本的显式 `Compound N`/命名化合物标题解析；`source_procedure_fragments` 及 HTML parser 兼容导出不变。Ruff、compileall、`git diff --check` 均通过，procedure/literature/架构聚焦回归 37 passed，超预算模块由 8 个降至 7 个；完整离线套件为 2699 passed、3 skipped、1 deselected、2 subtests passed，用时 174.44 秒。

第十一个维护切片把 `literature_materialization.py` 从 322 行降至 241 行：授权出版商结构化文本与补充 PDF 的联合物化、视觉页投影、来源 receipt 和语义标记迁入 97 行的 `literature_authorized_pdf_assets.py`。facade 继续拥有 structured-fulltext/PDF 获取顺序、缓存、代理和最终 PDF fallback；新 helper 通过显式 `pdf_materializer` 参数接收当前实现，保留测试 monkeypatch 与运行时注入。Ruff、compileall、`git diff --check` 均通过，literature/PDF/source-route/架构聚焦回归 55 passed，超预算模块由 7 个降至 6 个；完整离线套件为 2699 passed、3 skipped、1 deselected、2 subtests passed，用时 174.19 秒。

第十二个维护切片把 `route_workbench_proof_vectors.py` 从 289 行降至 106 行：edge exact/procedure/observation 收集、来源条件候选归一化、条件完整性和独立证明轴编译迁入 199 行的 `route_workbench_edge_proof_vector.py`。原模块继续拥有 route 级 weakest-edge 聚合，并兼容导出 `edge_proof_vector` 与 `PROOF_VECTOR_SCHEMA`，inspectors、fact rows 和 route rows 的入口不变。Ruff、compileall、`git diff --check` 均通过，Workbench/condition/架构聚焦回归 40 passed，超预算模块由 6 个降至 5 个；完整离线套件为 2699 passed、3 skipped、1 deselected、2 subtests passed，用时 173.98 秒。

第十三个维护切片把 `unified_campaign_spec.py` 从 425 行降至 382 行：JSON 值冻结/还原、字符串规范化、SHA-256 校验、规范摘要和 digest-bound row 编译迁入 66 行的 `campaign_contract_json.py`。四个不可变契约数据类、stock oracle builder 和所有公开导出保持原位；原模块通过私有别名继续使用相同 helper 语义。Ruff、compileall、`git diff --check` 均通过，契约/RunKernel/Gateway/Web/架构聚焦回归 72 passed，超预算模块由 5 个降至 4 个；完整离线套件为 2699 passed、3 skipped、1 deselected、2 subtests passed，用时 174.21 秒。

第十四个维护切片把仅超限 1 行的 `v4_planned_route_branches.py` 从 401 行压缩到 399 行：只将单符号 `source_conditions` 导入改为等价单行写法，没有迁移职责、增加抽象或改变 planned branch 投影。Ruff、compileall、`git diff --check` 均通过，Workbench/架构聚焦回归 36 passed，超预算模块由 4 个降至 3 个；完整离线套件为 2699 passed、3 skipped、1 deselected、2 subtests passed，用时 179.41 秒。

第十五个维护切片把 `campaign_operations.py` 从 181 行降至 135 行：model-free status/oracle/workbench benchmark、CPU/wall time 与 peak-memory 采样迁入 64 行的 `campaign_benchmark.py`。原操作模块继续拥有 Workbench/review bundle 导出、artifact GC 计划与兼容 `benchmark_campaign` 导出；Gateway 调用入口不变。Ruff、compileall、`git diff --check` 均通过，Gateway/操作/架构聚焦回归 21 passed，超预算模块由 3 个降至 2 个；完整离线套件为 2699 passed、3 skipped、1 deselected、2 subtests passed，用时 177.14 秒。

第十六个维护切片把 `campaign_gateway.py` 从 427 行降至 396 行：Gateway 结果封套与计划 payload 摘要迁入 46 行的 `campaign_gateway_projection.py`，默认 provider-set stock oracle 编译迁入 28 行的 `campaign_gateway_stock_oracle.py`。Gateway 继续拥有 run lifecycle、路径/index/provider registry、CLI/Web 委派和 `_open`/`_normalize_run_id`；target solver 依赖的 `_default_stock_oracle_reference` 私有兼容包装保留并委派到新 helper。Ruff、compileall、`git diff --check` 均通过，Gateway/Web/架构聚焦回归 42 passed，validation-fork 3 passed，超预算模块由 2 个降至 1 个；完整离线套件为 2699 passed、3 skipped、1 deselected、2 subtests passed，用时 177.06 秒。

第十七个维护切片把 `v4_route_workbench.py` 从 823 行降至 773 行：RDKit 2D depiction、结构 SVG 消毒和 Workbench molecule node 构造迁入 59 行的 `v4_route_nodes.py`。主编译器继续通过 `_node`/`_depiction` 兼容别名构造 canonical route、hypothesis 与 mechanism overlay 节点；route forest 和 HTML 公共入口不变。Ruff、compileall、`git diff --check` 均通过，Workbench/HTML/Web/架构聚焦回归 158 passed，超预算模块由 1 个降至 0 个，行数预算门首次全绿；不再 deselect 任何测试的完整离线套件为 2700 passed、3 skipped、2 subtests passed，用时 169.22 秒。

实验子任务排序切片新增 `experimental_work_scheduling.v1`：每个三域 work item 都摘要绑定 canonical deficit、exact-boundary dirty hint、domain plan priority、必检项广度、边界状态覆盖、experience uncertainty 和 executor-neutral 资源成本，输出可解释的 `information_gain_score`、`estimated_cost_units`、`value_per_cost`、原因、稳定 `rank_key` 与 canonical Action score。统一 target solver 只把该投影映射回既有 `PROGRAM_VALIDATE` signal；adaptive scheduler 实际按动态收益/成本选择，未创建第二 frontier 或队列。target/dataset 标签不参与评分，experience 只改变优先级，重摘要篡改仍由重投影失败关闭。聚焦回归 93 passed；不做 deselect 的完整离线套件为 2706 passed、3 skipped、11 warnings、2 subtests passed，用时 169.00 秒。

跨相似边界 applicability 切片新增 `program_applicability_model.v1`：只从 replay-validated Program memory 读取精确观察，在相同 capability、execution domain 或 mechanism strategy 内按输入/输出结构相似度降权，分别保留 exact/analog、positive/negative/inconclusive、冲突、置信度、不确定性和风险。模型摘要及独立重投影 oracle 接入 `program_experience_projection.v1`，候选 priority 与后续实验排序读取其有界结果；exact-boundary calibration、Claim authority、catalog、proof、completion 与 acceptance 均不改变。whole-cell/hybrid 现使用不同经验身份，target/dataset 标签不参与模型输入。三域/Gateway/CLI/Web 聚焦回归 116 passed；不做 deselect 的完整离线套件为 2712 passed、3 skipped、11 warnings、2 subtests passed，用时 177.69 秒。

实验外部作业闭环切片新增通用 `task_checkpoint` 事件与三份严格操作契约：operator identity 只保存 principal 类型和 authentication-context SHA-256；external job receipt 绑定 dispatch/task/request/provider、外部 job ID、单调 provider sequence、前驱 receipt、取消请求和记录者；cancellation request 精确绑定当前 receipt，且请求本身不结算。所有对象先进入 CAS，再由既有 RunKernel experiment task 的哈希链持有；pointer 可删除重建，未创建第二队列。相同并发 payload 幂等，不同 payload、错误前驱、类型混淆、fresh-digest 跨 dispatch 篡改、current frontier/provider 漂移、终态重开均失败关闭；取消请求后的 completed 竞态仍可审计并继续独立 result/domain gate，只有 cancelled acknowledgement 结算 cancelled。Gateway、CLI、Web、RunKernel 与架构聚焦回归 84 passed；Ruff、compileall、`git diff --check` 和全部架构门通过；不做 deselect 的完整离线套件为 2719 passed、3 skipped、11 warnings、2 subtests passed，用时 171.35 秒。

实验 HTTP transport 切片新增宿主配置的 `HttpExperimentExecutorProvider`、`experiment_job_operation_request.v1` 与 `experiment_job_transport_result.v1`：dispatch 只生成无副作用 handoff，显式 submit/poll/cancel 才调用宿主固定 endpoint；客户端不能提供 URL 或凭据。远端只允许 HTTPS，显式 loopback 测试除外；redirect 禁止，job ID URL-escape，响应大小和 timeout 有宿主硬上限，Bearer 值只从命名环境变量即时读取并从状态文本脱敏，CAS/事件/指针/封套只保存认证上下文与响应体摘要。每次 transport attempt 先进入原 experiment task checkpoint，成功后才追加 job receipt；timeout/HTTP/认证/响应错误保留审计且可用新 attempt 重试，checkpoint 后崩溃可消费缓存成功结果而不重发网络副作用。真实 loopback HTTP、Gateway submit→poll→cancel、取消 acknowledgement、配置漂移、CLI/Web 路由和凭据不落盘均已验证；聚焦回归 113 passed，Ruff、compileall、全部架构/行数预算门和 `git diff --check` 通过；不做 deselect 的完整离线套件为 2733 passed、3 skipped、11 warnings、2 subtests passed，用时 174.99 秒。当前没有生产 endpoint、设备或凭据，不能把 fixture 写成真实实验结果。

Action-class 公平服务切片新增 `campaign_action_class_service.v1` 与可重放 RunKernel reservation history：四类持续 eligible Action 使用固定 12-Action 最低服务窗口，deadline 压力出现前仍按原 value/cost 排序；blocked/absent class 的服务槽立即借出且不创建新预算、reservation 或队列。模型洪泛、四类最低服务、handler/resource blocked 借用、round-robin 独立性、reopen/resume ledger、B4 后科学轨迹和总任务硬上限均已验证。目标求解/Gateway/CLI 集成 73 passed；Ruff、compileall、全部架构/行数预算门与 `git diff --check` 通过；不做 deselect 的完整离线套件为 2738 passed、3 skipped、11 warnings、2 subtests passed，用时 225.44 秒。

跨 slice 收敛切片新增 `campaign_action_convergence_ledger.v1`：Action reservation 现在摘要绑定 opportunity/set 和 cohort 语义，runtime 从经 pointer/outcome digest 校验的 durable 历史恢复同 revision attempted、精确 no-gain binding 与 trailing streak。连续两次 slice/reopen 会继续计数且不 cache 重放已尝试 Action；达到历史上限的第三次 resume 零 dispatch 收敛；外部 graph progress/revision discontinuity 会清零 streak。目标求解/Gateway/CLI 集成 73 passed，调度/runtime/架构聚焦 51 passed；Ruff、compileall、全部架构/行数预算门与 `git diff --check` 通过；不做 deselect 的完整离线套件为 2740 passed、3 skipped、11 warnings、2 subtests passed，用时 183.68 秒。

精确边界重复价值切片在 `experimental_work_scheduling.v1` 中新增独立、可解释的 unchanged-exact-repeat penalty：supported/contraindicated/inconclusive/conflicting memory 只有在 strongest transfer scope 已是 exact boundary 且没有新 dirty hint 时降权；structural analog 和 dirty recompute 不受该惩罚。惩罚只进入 information gain/value-per-cost/Action score，最低值仍大于零，不修改 capability catalog、validation、Claim、proof、completion 或 acceptance。experience priority 纯策略已拆入 106 行模块，主 scheduling 保持 291 行，未提高 320 行预算。三域/架构聚焦 54 passed，相关入口集成 117 passed；完整离线套件为 2742 passed、3 skipped、11 warnings、2 subtests passed，用时 179.10 秒。

Codex replan pressure 切片新增 160 行的 target-blind `campaign_replan_pressure.v1`：只有经摘要验证的 `campaign_action_convergence_ledger.v1` 在同一 durable revision 连续 3 次无增益，并与 B1 路线多样性缺口同时成立时，才派生可信 `portfolio_stagnation` 状态事件；单独文本事件继续失败关闭。关键边拒绝、新路线族、共享瓶颈变化和来源冲突分别形成可审计 pressure 分量，并动态提高既有 `codex_global_replan` Action score；signal/budget gate、单一 deficit frontier、单一 action loop、RunKernel 硬预算和 canonical additive authority 均未改变。target/dataset/objective 标签不参与投影，tamper、replay/resume、B1 对照和模型预算耗尽均有回归。replan/runtime/架构聚焦 134 passed；Ruff、compileall、全部架构/行数预算门和 `git diff --check` 通过；不做 deselect 的完整离线套件为 2748 passed、3 skipped、11 warnings、2 subtests passed，用时 174.50 秒。

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

- [x] scheduler、RunKernel、action scorer 和 provider admission 由静态/契约门禁止读取 dataset ID、target index、target name、`RetroStar`、`benchmark_search` 等控制标签。
- [x] target name 只用于显示、日志和目录索引；Director prompt/run identity 与 ChemEnzy request 均使用结构派生 opaque identity，结构身份查询延后到 evidence Action，只有 exact-InChIKey 解析名可作为检索提示。
- [x] benchmark stock、采购 stock、in-house stock 统一绑定为不可变 Stock Oracle；核心只读取逐分子事实和 oracle digest，不读取用途标签。
- [x] 为 scheduler 核心目录增加静态禁词测试，阻止以后重新引入 dataset/objective 特判。
- [x] 同一 canonical 状态、预算向量和 provider 结果在不同显示名、旧 `objective_mode` 与并发完成顺序下产生相同 Action 轨迹/排序。

### 1.3 单一算法与 anytime 语义

- [x] 所有 proof axes 始终计算和报告，不再按模式禁用 reaction validation、evidence、conditions、self-evolution 或 Program review。
- [x] B0–B5 是同一运行轨迹上的成熟度快照，不是互斥模式。
- [x] 长期 loop 将固定 action cutoff/资源预算耗尽、无可执行动作、连续低边际收益、显式用户取消和不可恢复 Kernel 错误投影为互斥终止类型；paused 只暂停派发且不伪装科学终态。
- [x] 达到 B4 或 B5 只记录 milestone，不在核心内触发模式专用 return；运行时回归已证明 B4/B5 同时为真时仍执行后续 Action。
- [ ] 产品侧“拿到第一条库存闭合路线后停止/通知”实现为外部订阅或取消策略；论文评测使用固定预算 cutoff。

## 2. 完成定义

本轮只有同时满足以下条件才算完成：

- [ ] 主线不存在 benchmark 专用求解分支、专用 finalize、专用 replan 禁用或专用 action 开关。
- [ ] 每个任意 SMILES 都通过同一个 action loop 调用 ChemEnzy、Codex、文献、验证、库存和 Program 能力。
- [ ] ChemEnzy 在相同请求、随机种子和环境下的原始 proposal 集合与独立运行一致。
- [ ] 嵌入 V4 后，ChemEnzy 已发现且通过统一 host gate 的路线不会因缺证据、缺条件或 Codex 未选择而消失。
- [x] Codex 初始全局规划不阻塞目标级 ChemEnzy 搜索，后续 replan 由状态事件触发而非目标类别触发。
- [x] 同一运行可连续导出 time-to-first-route、B1、B2、B3、B4、B5 和 Program milestone。
- [ ] RetroStar-190 对 190 个目标使用同一 commit、同一配置、同一 scheduler、同一预算规则和冻结 stock hash。
- [ ] 完成 ChemEnzy-only、Codex-only、固定调度和统一自适应调度的全目标组件消融。
- [x] focused tests、完整离线测试、Ruff、全部架构门（含聚焦模块行数预算）和 `git diff --check` 全部通过。
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
- [x] 删除 `benchmark_search_can_finish_at_B4` 等会改变控制流的语义字段；B4 改为通用 milestone。
- [x] 删除 `objective_mode != "benchmark_search"` 对 self-evolution、evidence、conditions 和 replan 的开关。
- [x] 删除 `benchmark_search_completed` 作为独立终态；兼容字段只表达 B4 milestone，不再控制 kernel 或 disposition。
- [x] CLI/API 暂时接受旧 objective 值，但只能转换为兼容字段，并记录 `FutureWarning` 或 HTTP `Deprecation`/299 收据；Canonical Web 不再发送该字段。
- [x] 已从 V4 新建运行表单移除“求解算法模式”；表单只保留目标、执行档位、运行预算、库存来源和能力约束，结果成熟度在运行后查看。

### 4.2 新的通用输入

- [x] 定义 `UnifiedCampaignSpec`，只包含 target、不可变 stock-oracle reference、约束和多维资源预算；新 `RunSpec v2` 嵌入该契约，v1 摘要验证后兼容读取。
- [x] target 约束允许表达禁用试剂、最大步数、允许执行域、安全限制和库存来源，并在契约层拒绝 dataset/objective/benchmark 控制标签。
- [x] `RetrosynthesisAcceptanceSpec` 只作为质量审计/里程碑编译输入，不再选择 Action 集、scheduler 或 replan 分支。
- [x] 所有 runs 始终产生完整 `CampaignQualityState`：topology、reaction validation、exact evidence、stock、conditions、procurement、Program validation、diversity；无数据轴为 `not_assessed`，不伪造结论。
- [x] acceptance 可以在任意 snapshot 上被判定为 true，但不自动终止 action loop。

### 4.3 多维资源账本

- [x] 把预算明确拆成 native search、model、evidence、stock、validation、visual、Program/experiment、total task 和 total wall-time 维度；`campaign_task_budget.v1` 记录 settled/reserved/remaining，CLI/API/Web 与完整报告使用同一投影，模型预算扩展与 resume 不重置 Program/experiment 上限。
- [x] 将当前单一 `native_search` resource class 拆为 `native_search_target` 与 `native_search_frontier`；二者共享硬总上限，但拥有不同 reservation 规则。
- [x] 在 `RetrosynthesisRunBudget` 增加 target-native 最低服务、guided 上限/可借额度和借用策略字段；旧配置缺省时由 `max_attempt_runs` 派生兼容值。
- [x] target reserve 只有通过显式 release 事件才改变保护量；guided borrow 在 task reservation 中记录额度、原因和绑定的账本摘要。
- [x] Codex、evidence、validation、visual 和 Program 不写入 native-search 账本，也不会隐式减少 native expansion reserve。
- [x] 每种新 `campaign_action.v2` 在 reservation 前声明类级预计资源；handler 子任务自动继承 execution ID，settlement 从 RunKernel 事件链记录实际 task/native/model/墙钟用量与 variance；并发 Action 隔离归因，v1 receipt/in-flight reservation 仅在旧 SHA 精确匹配时兼容恢复。
- [x] 不允许文献/模型 Token 消耗隐式减少 ChemEnzy expansion 限额。
- [x] ChemEnzy timeout 只结算自己的 Action，不取消同 revision cohort 的 Codex；最终 revision 的未执行 validation/其他候选写入 `campaign_unexecuted_action_set.v1`，逐项保留 scheduler blocker、caller exclusion、已尝试/no-gain 或 action-limit/low-gain/budget 终止原因。
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
- [x] 新增 `campaign_action_estimate.v1`：reservation 前冻结成功概率区间、预期 route/proof/diversity 增益、依赖解除量、成本和不确定性；缺少校准依据时显式保留 `[0,1]`，不伪造点估计。
- [x] 新增 `campaign_action_result.v1`：不可变工件引用、实际资源/预计差异、状态、material events、候选/事实增量和失败类型；v1 outcome 仅在旧 Action SHA 与内容摘要精确匹配时兼容读取。
- [x] materialization、reaction validation、stock、conditions、evidence acquire/bind、target/guided ChemEnzy、Codex architecture/replan 与 Program discover/review/admit 已通过 RunKernel Action wrapper。
- [x] Program 专项 validation、实验 feedback 与后续新增能力也必须通过 RunKernel reserve → execute → settle，不得绕过账本直接修改 canonical graph。
- [x] 只有 host canonical ingestion 能把 proposal 提升为图事实；`ActionResult.fact_delta.changed` 只认 RunKernel 的 canonical graph revision，producer 自报 `changed` 仅作无权威诊断字段。
- [x] stale revision、cache replay、handler failure、CAS 摘要篡改和完整 canonical outcome replay 已由聚焦单测锁定。
- [x] 部分失败、timeout、取消和 handler failure 均写入标准失败类型并结算该 Action 遗留子任务；严格绑定的 in-flight Action 已通过重建 RunKernel 的跨进程恢复测试。

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
- [x] `cascade_planner/interfaces/target_solver_compat.py` 已承接旧 objective 展示、checkpoint cursor、外部反馈信号和 resume/trajectory 投影；生产 solver 仅导入兼容接口。

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

- [x] scheduler 只读取 canonical deficit/action 投影、milestones、资源可用量、handler/in-flight 状态和冻结策略；不拥有第二状态源。
- [x] scheduler/action core 由静态门保证不读取 objective、dataset、target label、benchmark manifest path 或 UI view；旧 view metadata 变化不改变排序。
- [x] canonical stock deficit 只携带 molecule/observation identity；Stock Oracle 名称不进入 scheduler，运行绑定只保留不可变 oracle digest。
- [x] scheduler 是只读确定性投影，只输出 Action 排序与解释，不执行动作或修改输入/图。

### 6.2 通用优先级

- [ ] 昂贵动作前先执行 identity、元素守恒、循环、重复和明显非法结构检查。
- [x] 没有库存闭合路线时，route discovery/stock closure 通过统一 action-class 服务窗口获得最低服务；规则不读取目标、数据集或 objective。
- [x] 已存在可物化候选时，确定性 materialization/validation 属于独立 closure class，持续 eligible 时不会被新的模型猜测长期饿死。
- [x] 搜索停滞、路线族单一、共享瓶颈或关键边反复失败时，提高 Codex 全局重构和替代路线动作价值；停滞只从摘要验证的 durable convergence streak 派生，单独文本事件不触发模型调用。
- [ ] 已有完整路线但 proof/evidence/conditions 开放时，逐步提高验证和证据动作价值；这些动作不能反向删除路线拓扑。
- [ ] 常规路线存在高代价连续区间、特定选择性瓶颈或已知能力匹配时，提高 Program discovery/review 价值。
- [x] 负结果和不确定结果会降低“同一 exact boundary 且无新 dirty signal”的重复实验价值；structural analog、新 dirty recompute 与 capability 本身仍保持可调度，不做全局禁用。

### 6.3 排序模型

- [x] 使用可解释的确定性初版，而不是立即训练黑箱 scheduler。
- [x] 初版分数显式记录 route closure、proof、diversity、dependency unblock、novelty、校准成功概率下界、cost 和 risk；未评估 `[0,1]` 不产生伪成功率奖励。
- [x] 所有权重在运行 RetroStar-190 全集前冻结并写入 manifest。
- [x] 对相同分数使用稳定 action ID 做 deterministic tie-break，并验证输入顺序变化不改变排序。
- [x] 每次选择记录全部候选、各分量、被选原因和未选原因。
- [x] 连续低收益达到阈值后记录 `converged_low_marginal_gain` 和未执行原因，不伪造 acceptance。

### 6.4 公平调度而非样本分组

- [x] 对四个固定 action classes 使用所有目标一致的 12-Action minimum service window；blocked/absent class 当轮自动出借服务槽，借用不增加 RunKernel 硬预算。
- [ ] ChemEnzy、Codex、evidence 和 validation 可并发，但共享同一事件循环、in-flight registry 和 canonical state。
- [x] 初始 ChemEnzy/Codex cohort 结果按 revision、幂等键和稳定 action 顺序合并，cache replay 不重复 reservation/settlement。
- [x] 任何 action class 都不能因目标来自某个数据集而被开启、关闭或获得额外预算；静态禁词门与 metadata/insertion-order 回归覆盖该约束。
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
- [x] 保留 conventional edge route 作为 enzyme/mechanism/whole-cell superstep 的显式 fallback。

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

- [x] replan 触发器只依赖物质事件和状态变化：关键边拒绝、新路线族、新 exact evidence、库存变化、共享瓶颈、搜索停滞和路线多样性不足；其中停滞要求固定 3 次 digest-verified durable no-gain 且 B1 未满足。
- [x] 去除 benchmark 模式对 depth、validation、evidence replan reason 的屏蔽。
- [x] replan 输出只能追加候选、调整优先级或提出替换 Program；不能删除既有 canonical 路线。
- [x] 每次 replan 报告前后 route family、edge、stock closure、proof 和资源增量。
- [x] `no_gain`、失败和超时作为 scheduler 学习/审计信号保留，不倒推污染已有事实。

### 9.3 文献与创新

- [ ] Codex 负责 source-consistent 文献路线假设和检索策略；exact authority 仍由 host 文献 connector 和绑定门授予。
- [ ] Codex 负责识别可被酶、whole-cell/hybrid 或机理一跳替换的长区间；Program 编译和专项验证继续由 host 严格执行。
- [ ] 文献外假设永远标为 hypothesis，不因 Codex 自信表达升级 proof。
- [x] Program 候选必须精确绑定输入/输出状态、replaced-edge span、capability/precedent、验证计划和 conventional fallback。

## 10. Anytime 轨迹、里程碑与输出

- [x] `campaign_trajectory.v2` 已在每个 Action settlement 后追加内容寻址 snapshot，并记录 milestone、资源、action decision、binding epochs 与资源曲线；v1 仅兼容读取。
- [x] 每个 snapshot 记录 graph/event revision、累计 RunKernel wall time、各资源维度、action counts、route counts、Pareto archive 和 B0–B5。
- [x] 记录 time-to-first-route/B1、time-to-first-host-valid-route/B2、time-to-B3/B4/B5 和 Program milestones。
- [x] 每个 snapshot 绑定控制面源码 bundle、配置、stock oracle、provider/model 和 UnifiedCampaignSpec/目标结构摘要。
- [x] resume 后继续同一 trajectory；event sequence 与累计 wall time 连续性显式审计，terminal no-op resume 保留原 trajectory digest。
- [x] W5 允许“已完成 checkpoint 收到新的 canonical action signal/实验反馈”时通过带工作指纹的 `run_reopened` 事件重新进入同一 action loop；不能直接刷新旧终态报告而跳过反馈。
- [x] benchmark harness 只读取固定 cutoff 的 trajectory projection，不向 solver 传 benchmark mode；旧 panel `--objective-mode` 仅警告并忽略，W8 四臂命令不再携带该参数，评分只读 `campaign_trajectory_cutoff_projection.v1`，最终状态单列为诊断。
- [x] Workbench 同时展示当前最优路线和历史上曾达到的 milestone；`workbench_trajectory_history.v1` 区分当前成立、历史达到但当前失效、从未达到，并保留首达 snapshot 和资源曲线，历史状态不恢复撤销 proof。
- [x] Gateway export 从摘要验证的 target report 导出独立内容寻址的 action trace、失败 trace、provider/canonical route lineage 和 trajectory 资源曲线；报告/轨迹损坏失败关闭，开放科学门不伪装成运行失败。

## 11. CLI、API 与 Web 迁移

- [x] CLI 新建运行参数使用 target、stock oracle、constraints 和多维 budget；通知/展示从统一 trajectory 派生。
- [x] 旧 `--objective-mode` 保留至 2026-10-01 兼容窗口，显式使用发出警告，只进入 claim projection，不进入 `solve_target()` 调度核心。
- [x] API 将 `objective_mode` 标为 deprecated：同步/异步入口返回 `Deprecation` 与 299，async response 另保留 `request_warnings[]`；结果继续返回统一 gates/milestones 与 Workbench link。
- [x] Web 已删除“Benchmark 检索闭合/科学证明/采购交付”新建任务选择；同一 trajectory 在结果页按库存、证据和科学成熟度轴查看。
- [x] Web 运行中心通过 `campaign_action_timeline.v1` 实时合并 checkpoint 已结算 Action 与 RunKernel 当前 wrapper reservation，在同一时间线显示 ChemEnzy、Codex、evidence、validation、conditions、stock、Program/experiment；child task 不重复，投影不是第二队列。
- [x] CLI/API/Web 均经同一 `CampaignGateway.solve_target()` 和 `unified_campaign_runtime`；入口只装配参数，不复制调度逻辑。`UnifiedCampaignSpec` 的显式值对象仍在 4.2 单独跟踪。
- [x] saved-run 恢复兼容旧 objective 字段：`saved_run_objective_compatibility.v1` 仅保留历史来源；同一保存状态保留/删除旧字段的双克隆恢复得到相同 Action binding 后缀，且 stage 连续编号不覆盖旧 execution。

## 12. 测试设计

### 12.1 架构与静态门

- [x] scheduler、RunKernel、Action compiler/runtime 均由静态禁词门阻止 dataset/objective/target-index 特判字符串。
- [x] `target_solver.py` 不再包含 benchmark 专用 finalize 或 B4 early return。
- [x] V4 dependency gate 禁止主线重新导入 legacy frontier、Blackboard、旧 route portfolio 或旧 controller。
- [x] application Action modules 禁止反向依赖 orchestration；统一 Action runtime 不导入 canonical graph/service，不能自行写图，只能由注册的 host handler 经 canonical ingestion 提交。
- [x] 继续执行现有 V4 dependency gate；既有超行数预算债务已清零，完整测试不再需要 deselect 该门。

### 12.2 单元测试

- [x] 相同 state/budget、不同旧 objective 值产生相同 action candidates 和排序的 focused test。
- [x] 相同 state 在 replay、resume 和并发完成顺序变化下产生相同 canonical digest；回归覆盖 replay digest、保存态双克隆 resume，以及交换 ChemEnzy/Codex 完成先后后的 exact Action binding 与 scientific digest。
- [ ] Pareto archive 保留 topology-valid/evidence-open 路线。
- [x] 缺证据、条件或 Program validation 不删除 B1/B4 路线。
- [ ] Codex 未选择的 ChemEnzy route 仍可继续 materialize/validate/stock audit。
- [ ] guided ChemEnzy 不替换 target-level route pool。
- [ ] stock oracle digest 改变只更新 stock facts 和后续 deficits，不改变代码路径。
- [x] budget/timeout/cancellation 已在长期 loop 闭合：budget 进入 Kernel 终态，Action timeout 标准结算，显式 cancel 不再派发并写入 backlog；native borrowing、reservation、settlement 均可重放。
- [x] milestone 达成不触发核心 early return。
- [x] `CampaignAction` binding 对同 revision/decision 稳定，对 revision 变化生成新 execution identity。
- [x] `CampaignActionRuntime` reserve/settle、cache replay、stale revision、handler unavailable 和 digest tamper 行为稳定。
- [x] action wrapper 资源账本不与 child worker/provider 成本双计。

### 12.3 集成测试

- [x] 用旧 benchmark/scientific/procurement 标签启动同一输入，在相同预算前缀内 action trace 完全一致；三种 fresh run 同时交换 provider 完成顺序，仍得到相同 Action decision/binding、route digest、gates、成本与 canonical scientific digest。
- [ ] ChemEnzy 与 Codex 初始动作互不阻塞；任一失败时另一方结果仍可进入 graph。
- [x] 文献获取、reaction validation、stock audit 和 condition enrichment 都由 open deficits 触发。
- [x] conventional route 与 enzyme/mechanism Program 在同一 campaign 中共存且 fallback 不丢失。
- [ ] B4 后继续运行可自然获得 B3/B5，不创建第二个 run。
- [ ] resume、checkpoint、API、Web、CLI 和 artifact export 读取同一 trajectory。

### 12.4 回归集

- [ ] 当前成功的 3 个 RetroStar smoke 目标继续成功。
- [x] standalone 成功而 embedded 失败的 Nirmatrelvir 代表目标已完成定因和通用修复。
- [x] Nirmatrelvir current-V4 zero-model replay 保持 raw/normalized parity，B4=true。
- [x] 至少一个文献驱动真实目标保留 exact-source 路径；官方 EPO 三案例 `v14` 在线迁移后已再次通过严格离线 replay。
- [x] 至少一个 enzyme superstep 阳性候选和一个无适用酶负对照保持严格语义；阳性仅表示结构适用候选，不伪装成 exact-substrate 实验阳性。
- [x] Program shadow store、实验 Claim 和 canonical graph digest 不受统一 scheduler 重构污染。

## 13. RetroStar-190 评测

### 13.0 当前范围：前 20/190 四臂 pilot

- [x] 使用 manifest 固定顺序前 20 个目标，四臂 case IDs 完全一致，未按结果挑样。
- [x] ChemEnzy-only、Codex-only、Unified round-robin、Unified adaptive 均为 20 completed、0 failed/incomplete。
- [x] 生成 run manifest、per-target metrics、paired comparison、failure taxonomy 和 panel summaries。
- [x] 形成停止判定：adaptive B4=15/20，与 round-robin 持平且低于 ChemEnzy-only 16/20；继续扩展到 190 的当前工程信息增益不足。
- [x] 明确 pilot 不是随机样本，不外推 190，不宣称 benchmark-wide improvement。
- [x] 本轮以 20/190 四臂结果和停止判定完成收口；190×4 仅保留为未来独立科学门，不是当前待执行任务。

### 13.1 未来 publication-scale 冻结协议（非当前施工范围）

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
- [x] 如实报告目前完成的是 manifest-prefix 20/190 四臂 pilot；全 190 完成前不得宣称 benchmark 整体提升。

交付物：

- [x] `docs/evaluation/UNIFIED_ANYTIME_RETROSTAR190_PROTOCOL.md`
- [x] `docs/evaluation/UNIFIED_ANYTIME_ABLATION_PLAN.md`
- [x] `docs/evaluation/REVIEWER_DEFENSE_CHECKLIST.md`
- [x] Pilot 机器可读 run manifest、per-target metrics、paired comparison 和 failure taxonomy 已生成并重新校验。
- [x] Pilot 结论与产物哈希：`docs/evaluation/RETROSTAR190_W8_PILOT20_RESULTS_20260808.md`。
- [x] 当前交付范围不包含 Full-190 结果；若未来需要 benchmark-wide 主张，则另行重启 publication-scale reviewer package。

## 15. 文档与清理收束

- [x] 更新 `docs/MAINLINE.md`：从固定 Codex-first 阶段图改为同一事件循环中的全局 Director + native search + unified Action frontier。
- [x] 更新 `CURRENT_ARCHITECTURE_STATUS.md`，明确当前实现与统一架构完成度。
- [x] 更新 CLI/API/Web 文档，删除“benchmark 到 B4、scientific 到 B5 是两套运行模式”的表述。
- [x] 更新架构演化时间线，解释为什么从阶段式 objective branching 转向 target-blind anytime scheduler。
- [x] 归档旧 objective compatibility 说明并冻结 2026-10-01 新请求字段移除期限。
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
- [x] 实现 action-class 最低服务保障与 blocked-class 服务槽借用；stable tie-break、RunKernel 事件重放和 round-robin 独立语义已完成。
- [x] 将连续低收益计数从单次 `run_anytime()` 扩展为跨 slice/resume 的 `campaign_action_convergence_ledger.v1`；旧绑定只作为链边界，外部 graph revision/revision discontinuity 会重置连续 streak，缓存 replay 不增加 attempt。
- [x] 门：scheduler target-blind 动态/静态测试与生产单 action-loop 集成回归已通过。

### Checkpoint E：非阻塞 Codex + ChemEnzy

- [x] 初始全局规划和 target-level native search 通过同 revision cohort 并发启动。
- [x] Codex proposal additive、ChemEnzy frontier monotonic、canonical merge 使用稳定 action identity/order 与唯一 graph union。
- [x] 门：ChemEnzy handler 与 canonical ingestion 不等待 Codex 完成；Codex 失败不会取消已完成的 ChemEnzy action。

### Checkpoint F：科学层与 Program 回归

- [x] B4 后继续同一 run 完成 evidence、conditions、replan 和 Program 工作；集成门观测到 B4 后的 replan、condition 与 Program Action，evidence material 保持在同一 trajectory。
- [x] conventional fallback、proof authority 和 Program shadow 边界不变。
- [x] 门：真实文献案卷与 enzyme positive/negative controls 通过；范围与限制见 `docs/architecture/SCIENTIFIC_LAYER_REGRESSION_20260809.md`。

### Checkpoint G：全量 190 与论文包

- [x] 完成固定 manifest-prefix 20/190 四臂 pilot、paired metrics、失败分类与明确停止判定。
- [ ] 冻结 commit/config 后运行四个全目标组件消融。
- [ ] 汇总 anytime、资源、召回损失和失败类型。
- [ ] 门：结果可复现、无目标特判、无挑样报告、审稿防御材料完整。

## 17. 明确不做

- [ ] 不为 RetroStar-190 编写目标 ID、SMILES、名称或 reference-route 特判。
- [ ] 不按“简单分子/复杂分子”“benchmark/scientific”人工分配不同 planner。
- [x] 不用 B4 早停掩盖后续科学证明缺口。
- [x] 不让 evidence/condition 缺失删除结构有效路线。
- [ ] 不让 Codex 计划覆盖 ChemEnzy native frontier。
- [x] 不把 enzyme/mechanism superstep 伪装成已验证普通 reaction edge。
- [ ] 不通过恢复 legacy scheduler 或 Blackboard 快速绕过统一设计。
- [ ] 不在全 190 test targets 上边看结果边逐目标调参。
- [x] 不用一次大重构清空已延期的行数预算债务；主线收口后按职责边界完成十七个独立维护切片，17 个超限模块已全部清零。

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
- [x] 第十刀（W6–W8-P）：W6 真实 embedded failure 已定位并修复；W7 冻结清单、完整离线门、190/190 preflight 与最终零模型回放均已完成；W8 的预算终态问题已逐层修复并保留审计；前 20/190 四臂 pilot 与机器可读汇总已完成。全量 W8-F 按 2026-08-09 范围决策延期。
- [x] 第十一刀（实验调度）：三域 `experimental_work_frontier.v1` 已加入 target-blind、摘要绑定的信息增益/成本排序，并把动态 priority/score 接回同一 canonical Action scheduler；同时为下一刀预留 applicability uncertainty/risk 输入。
- [x] 第十二刀（applicability 学习）：Program experience 已升级为 exact/structural-analog 分层、相似度加权且 execution-domain 隔离的 `program_applicability_model.v1`；模型只影响 proposal/validation priority。
- [x] 第十三刀（实验外部作业闭环）：`RunKernel` 新增 CAS 绑定、前驱有序且不改变预算/图/科学状态的通用 task checkpoint；严格的 `experiment_operator_identity.v1`、`experiment_external_job_receipt.v1` 与 `experiment_cancellation_request.v1` 已接入同一 experiment task。取消请求不会提前结算，只有绑定请求的外部 `cancelled` acknowledgement 才结算 cancelled；`completed`/`failed` receipt 仍需独立 result/domain gate。Gateway、CLI、HTTP、并发幂等、不同 payload 冲突、fresh-digest 绑定篡改、current frontier/provider 漂移、取消竞态和旧人工回填兼容均已回归，并作为第十四刀受控 HTTP transport 的操作权威基础；不得创建第二任务队列或虚构设备凭据。
- [x] 第十四刀（受控 HTTP transport）：默认 registry 可由宿主环境显式注册 HTTPS experiment bridge；dispatch 无网络副作用，submit/poll/cancel 通过稳定 operation ID、Bearer 环境引用、endpoint-config digest、service operator identity 和有界响应契约运行。所有 attempt/receipt/cancellation 继续复用同一 RunKernel task；timeout 可重试、成功 checkpoint 可恢复、取消仍以 acknowledgement 为结算门。当前只证明桥接能力与真实 HTTP 协议，不声称已连接生产设备；下一步需要部署方提供真实 bridge endpoint/凭据并取得可发布校准数据，或为不兼容桥接契约的厂商 API 编写受控 adapter。
- [x] 第十五刀（action-class 公平服务）：新增 target-blind `campaign_action_class_service.v1`，把全部 Action 冻结映射为 route discovery、deterministic closure、scientific proof、Program/experiment 四类。Adaptive 只在 12-Action 最低服务窗口即将违约时覆盖价值排序；无 eligible handler/resource 的 class 自动出借服务槽且不积累第二队列或新预算。服务历史从 RunKernel durable reservation 事件重建，resume/input-order/round-robin、模型洪泛不饿死 closure、blocked borrowing 与总任务硬上限均有回归。
- [x] 第十六刀（跨 slice 收敛）：新增 target-blind `campaign_action_convergence_ledger.v1`，从 RunKernel reservation、Action outcome pointer 和不可变 outcome digest 重建 attempted/no-gain/consecutive 状态。`run_anytime()` 在新 slice/resume 前恢复同 revision attempted 排除，达到历史 no-gain 上限时不再重复 dispatch；新 graph revision 或输入/输出 revision 断点清零连续 streak，精确 opportunity digest 改变才解除 no-gain binding。没有新增队列、预算或 canonical authority。
- [x] 第十七刀（精确边界重复降权）：实验 work scheduling 对已有 supported/negative/inconclusive/conflicting exact-boundary memory 增加 unchanged-repeat penalty；只有缺少新 dirty signal 时生效。structural analog、dirty recompute 和其他 capability 不被禁用，结果仍只改变 priority/score。
- [x] 第十八刀（Codex 动态重规划压力）：新增 `campaign_replan_pressure.v1`，从 gates、物质事件和摘要验证的 convergence ledger 形成 target-blind score。固定 3 次 durable no-gain 与 B1 路线多样性缺口共同派生停滞事件；关键边、新路线族、共享瓶颈和来源冲突独立加权。纯文本停滞、标签变化和篡改 ledger 均不能触发，模型预算耗尽仍由原 budget gate 拒绝；没有增加调用预算、队列或科学权威。
