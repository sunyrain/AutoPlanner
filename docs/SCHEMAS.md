# V4 Schema 索引

| Schema | 权威范围 | 主要生产者 |
| --- | --- | --- |
| `campaign_action_preflight.v1` | 从同一个 Action opportunity set 只读投影当前可处理的 canonical materialization cheap gate；复用 identity/admission/materialization 权威，只收窄下游 eligibility，不创建队列、预算或科学事实 | action scheduler |
| `autoplanner_run_spec.v2` | 嵌入统一 Campaign 输入的运行身份与兼容验收面；新运行唯一写入版本 | `RunKernel` |
| `autoplanner_run_spec.v1` | 旧目标、验收、硬预算；仅保摘要验证后的迁移读取 | `RunKernel` compatibility reader |
| `unified_campaign_spec.v1` | canonical target、不可变 stock-oracle reference、约束和多维资源预算；不含名称、dataset、objective 或 acceptance | API/CLI/Web adapters |
| `stock_oracle_reference.v1` | 冻结库存摘要或活解析器合同的内容寻址绑定 | stock adapter / provider registry |
| `target_constraints.v1` | 禁用试剂、最大步数、允许执行域、安全限制和库存来源 | API/CLI/Web adapters |
| `campaign_resource_budget.v1` | model/visual、native search、evidence、stock、validation、Program、experiment、total task 与 campaign wall-time 的运行预算向量 | API/CLI/Web adapters |
| `campaign_task_budget.v1` | evidence、stock、validation、Program、experiment 与 total task 的 settled/reserved/remaining 可重放投影 | `RunKernel` / status / reports |
| `campaign_action_convergence_ledger.v1` | 从 durable Action reservation/outcome 重建 attempted、exact no-gain binding、跨 slice consecutive streak 与 revision discontinuity；不创建队列或预算 | action runtime |
| `campaign_action_class_service.v1` | 四个 target-blind Action class 的 12-Action 最低服务 deadline、eligible/blocked 状态、服务槽借用、选中 class 与历史摘要；不创建队列或预算 | action scheduler / runtime |
| `campaign_action.v2` | revision-bound Action，包含 reservation 前固定的类级资源估计；v1 只允许按旧摘要严格兼容读取 | action scheduler / runtime |
| `campaign_action_estimate.v1` | reservation 前冻结的成功概率区间、route/proof/diversity 与依赖解除预期、成本和显式不确定性；未评估概率保留 `[0,1]`，不做伪校准 | action binder |
| `campaign_action_resource_estimate.v1` | target-blind resource-class 估计：wrapper/child task、native、model 与未知 token 维度 | action binder |
| `campaign_action_resource_usage.v1` | 按 action execution ID 从 RunKernel 事件链归因的实际 task/native/model/墙钟消耗 | `RunKernel` |
| `campaign_action_resource_accounting.v1` | 同一 Action 的预计、实际与逐维 variance；不授予科学权威 | action runtime |
| `campaign_action_result.v1` | revision-bound Action 收据：不可变工件引用、实际资源、状态、material events、候选/事实增量与失败类型；事实增量只认 RunKernel canonical graph revision，失败/timeout/cancel/partial 会释放该 Action 遗留子任务 | action runtime |
| `campaign_action_concurrent_cohort.v1` | 同一 input revision 的跨 resource Action 选择、wrapper reservation、bounded worker 上限、稳定 observation/commit 顺序、resource collision 与容量回退收据；不创建第二队列或科学权威 | action runtime |
| `campaign_anytime_action_loop.v1` | 唯一生产 Action runtime 的 start cohort、后续 concurrent cohorts、稳定 execution、收敛与未执行候选投影；历史 phase stage 只能只读消费 execution backlog | action runtime |
| `prepared_evidence_acquisition.v1` | frozen graph/request、connector acquisition、目标身份观察与摘要绑定；只允许后续稳定顺序 canonical ingestion，本身不授予 evidence authority | target solver evidence Action |
| `prepared_materialized_edge_validation.v1` | frozen graph edge/mapping、validation commands 与不可变 WorkerResult 摘要绑定；worker 执行不发布 canonical revision，稳定 commit 后才产生 reaction-proof ingestion | target solver validation Action |
| `active_campaign_action.v1` | 从 RunKernel in-flight wrapper reservation 投影的当前 Action 身份、类别、producer 与资源类；child task 不重复展示，不是第二队列 | campaign service status |
| `campaign_action_timeline.v1` | 将 checkpoint 中已结算 Action 与 RunKernel 当前 reservation 合并为 ChemEnzy/Codex/evidence/validation/condition/stock/Program/experiment 的统一只读时间线 | Web job progress |
| `saved_run_objective_compatibility.v1` | saved-run 恢复时记录 checkpoint/report 中旧 objective 字段及当前兼容视图；历史值只作 provenance，不能启停、排序或绑定 Action | target solver compatibility reader |
| `campaign_unexecuted_action_set.v1` | anytime 终态仍未执行的最终 revision 候选及 scheduler/loop 原因 | action runtime / target report |
| `campaign_quality_state.v1` | topology、reaction validation、exact evidence、stock、conditions、procurement、Program validation、diversity 八个独立质量轴 | campaign status / target report / Workbench |
| `campaign_trajectory_bindings.v1` | 每个 anytime snapshot 的控制面源码 bundle、配置、统一输入、stock oracle 与 provider/model 身份；缺失项显式为未观察，绑定变化形成新 epoch | target solver / trajectory compiler |
| `campaign_anytime_snapshot.v2` | event/graph revision、累计 RunKernel wall time、完整多维资源、Action/路线计数、紧凑 Pareto archive、B0–B5/Program milestones 与运行绑定；内容寻址工件不得经过有损 stage 裁剪 | target solver |
| `campaign_trajectory.v2` | 从 v1/v2 snapshot 重建首达时间、binding epochs、resume 连续性和资源曲线；v1 只兼容读取且不伪造缺失墙钟或绑定 | trajectory compiler / target report |
| `campaign_trajectory_cutoff_projection.v1` | 对内容寻址 trajectory 按冻结的累计 wall/task/model/token/search 等上限选择最后一个合法 v2 snapshot；截断后里程碑、路线、成本均不读取未来状态，资源回退时拒绝投影 | evaluation harness / reporting |
| `workbench_trajectory_history.v1` | Workbench 的只读轨迹摘要：当前/历史失效/未达到三态里程碑、首达时间、binding epochs 与完整资源曲线；历史达成不恢复已撤销 proof | route workbench / HTML / API |
| `campaign_review_bundle.v1` | 从摘要验证通过的 target solve report 生成四个独立内容寻址的审稿投影；报告损坏或缺失时失败关闭，不读取未验证 stage | campaign export / Gateway |
| `campaign_action_trace.v1` | 按报告 stage 顺序去重的 Action execution、reservation 前 estimate、scheduler decision 与不可变 outcome；不授予科学权威 | review bundle |
| `campaign_failure_trace.v1` | 显式 Action/stage failure、timeout、cancel、partial 与终态原因；开放的科学门不会被改写成运行失败 | review bundle |
| `campaign_route_lineage_export.v1` | provider raw/normalized/admitted/materialized lineage、最终 canonical Pareto lineage 与摘要验证候选生命周期的分层只读导出 | review bundle |
| `campaign_resource_curve_export.v1` | 经 trajectory 内容摘要验证的累计资源曲线、首达时间、resume continuity 与 binding epochs；不从零散 stage timing 重算 | review bundle |
| `canonical_candidate_lifecycle.v1` | 绑定 canonical graph revision/scientific SHA 与 portfolio digest 的五态候选审计投影；合并 canonical topology、proof、evidence、conditions、stock/route selection 和摘要有效的 ingestion rejection，不创建科学权威 | target report / review bundle |
| `canonical_candidate_lifecycle_record.v1` | 单个候选的 `rejected_invalid`、`quarantined_reviewable`、`admitted_unproved`、`validated` 或 `accepted` disposition，以及 canonical IDs、origin、独立 proof/evidence/stock/conditions/portfolio 轴和逐记录摘要 | candidate lifecycle |
| `autoplanner_run_event.v1` | 可重放运行事件 | `RunKernel` |
| `global_campaign_plan.v1` | 全局路线族和多步骨架 proposal | Global Director / replay |
| `canonical_retrosynthesis_hypergraph.v1` | 分子、超边、来源、库存和拓扑 | canonical ingestion |
| `deficit_frontier.v1` | 唯一待办优先队列投影 | graph compiler |
| `retrosynthesis_worker_result.v1` | 确定性 worker 结果 | worker runtime |
| `exact_source_reaction_record.v1` | 精确结构/反应关系观察；不授予 procedure 权威 | exact-source worker |
| `source_reaction_procedure_record.v1` | 来源位置、过程片段摘要、条件及完整度 | exact-source worker |
| `reaction_condition_completeness.v1` | 操作性条件字段及具名缺失组 | condition compiler |
| `route_innovation.v1` | 酶催化超级步骤或文献锚点后一跳机理外推；仅 proposal 权威 | global planner / canonical ingestion |
| `route_innovation_summary.v1` | 路线实际步数、化学等效步数、净节省及创新边集合 | route stitcher |
| `biocatalysis_capability.v1` | 带先例、底物域和净转化约束的酶能力搜索先验；不是酶验证 | capability catalog / precedent retrieval |
| `program_execution_capability.v1` / catalog | whole-cell 或 hybrid 的结构适用域、执行者、顺序 operation、辅因子/载体与专项验证要求；只是搜索先验 | execution capability catalog |
| `route_innovation_discovery.v1` | canonical 路线窗口匹配、酶/whole-cell/hybrid Program draft id 与机理一跳 admission | application discovery |
| `biocatalytic_program_proposal.v1` | 直连连续区间边界的非权威酶 Program、被替代 Program 与 fallback | biocatalytic Program compiler |
| `biocatalytic_program_route_candidate.v1` | baseline Program 路线中以一个酶 Program 替换连续区间的候选路线 | biocatalytic Program compiler |
| `biocatalytic_program_bundle.v1` | Program proposals、候选路线、拒绝项和物理/化学步数核算 | biocatalytic Program compiler |
| `biocatalytic_program_bundle_oracle.v1` | 投影新鲜度、边界引用、fallback 与非等价步数的失败关闭复算 | biocatalytic Program compiler |
| `mechanism_program_proposal.v1` | 一跳机理假设的已物化路线边界、被跨越 Program、锚点、可证伪检查与非权威 operation blueprint | mechanism Program compiler |
| `mechanism_restitched_program_route.v1` | 仅在一跳前体/产物精确接回同一路线状态时形成的完整候选；保留逐边 fallback | mechanism Program compiler |
| `mechanism_program_bundle.v1` | 可完整重拼的机理 Program、候选路线和不能接回路线的显式拒绝项 | mechanism Program compiler |
| `mechanism_program_bundle_oracle.v1` | graph/route/projection/discovery、新边界、连续区间、fallback 与完整 bundle 的失败关闭复算 | mechanism Program compiler |
| `mechanism_program_validation.v1` | 精确绑定机理 Program、创新、边界状态、机理签名、必检项与实验记录的结果；净转化成立不自动证明所提基元机理 | validation producer / host contract |
| `mechanism_validation_plan.v1` / frontier | 未通过机理专项门的精确边界、机理签名、竞争路径检查和严格输出契约；只生成实验任务 | mechanism validation frontier |
| `mechanism_experiment_feedback.v1` / projection / oracle | 机理实验的正向、负向、不确定 exact-boundary 反馈及确定性复算；不创建 reaction proof 或 store 写入 | mechanism feedback projector |
| `experimental_observation_claim.v1` | 统一表示生物催化、execution、mechanism 的精确边界实验观察；保留 positive / negative / inconclusive，不授予 canonical reaction proof | experimental Claim projector |
| `experimental_observation_claim_set.v1` / oracle | 三个验证域的不可变 Claim 集、拒绝项和七个来源摘要；可从领域 bundle/feedback/validation 完整复算 | experimental Claim projector |
| `exact_boundary_applicability.v1` / `capability_applicability_calibration.v1` / oracle | 仅对完全相同 domain、subject、boundary、signature 汇总正负与冲突，并输出 created/changed/removed dirty-domain 提示；不修改能力目录 | capability calibration projector |
| `experimental_claim_validation_pack.v1` | 与输入顺序无关、保留原领域绑定的验证记录 CAS 包 | experimental Claim store |
| `experimental_claim_admission_event.v1` / result | 显式开关后的 append-only、内容寻址 Claim 准入；绑定 graph、route、projection、discovery、validation pack、Claim set 六类 CAS | experimental Claim store |
| `experimental_claim_store_replay.v1` / status / oracle | 从六类 CAS 重新编译三个领域来源、Claim set 和 oracle；不产生 proof、completion、acceptance、Program admission 或 catalog mutation | experimental Claim store |
| `experimental_work_item.v1` / `experimental_work_frontier.v1` / oracle | 把三域 validation plan 和 exact-boundary dirty hint 投影为绑定当前 `deficit_frontier.v1` 摘要的只读实验子任务；不是第二个队列，不能发布到 `RunKernel` | experimental work projector |
| `experiment_execution_request.v1` | 执行器中立的精确边界请求；绑定 run/route、当前 frontier、源 plan、必检项、输出 schema 与资源提示 | experiment executor adapter |
| `experiment_execution_result.v1` / audit | 绑定 request、executor、原始工件 SHA-256、成功/失败/不确定/中止状态和一个领域验证候选；audit 只允许把候选交给既有领域 gate，不直接授予验证或 Claim | experiment executor adapter |
| `experiment_executor_policy.v1` / `experiment_executor_selection.v1` | 配置驱动的 provider allowlist、domain/network/cost 上限与宿主信任选择结果；调用方策略不能注册 provider 或提升 trust/kind/capability | host provider registry |
| `experiment_dispatch_handoff.v1` | 绑定 dispatch/request/executor 的人工或外部执行交接；`awaiting_external_result` 不表示实验完成，也不授予验证 | experiment executor provider |
| `experiment_dispatch_task.v1` | 嵌入 RunKernel experiment reservation metadata 的 request/provider/selection/CAS 绑定；不是第二队列 | RunKernel operational ledger |
| `experiment_dispatch_receipt.v1` | 内容寻址的 handoff 或 settlement 收据；pointer 只作可重建恢复索引，settlement 仍只释放领域 gate 候选 | experiment dispatch runtime |
| `autoplanner_task_lifecycle.v1` | 从 RunKernel 事件链投影单个任务的 absent/in-flight/settled 生命周期；无科学权威 | RunKernel |
| `execution_program_proposal.v1` | whole-cell/hybrid 连续区间的非权威 Program，显式记录 actors、operation、辅因子/载体与验证计划 | execution Program compiler |
| `execution_program_route_candidate.v1` | 用 execution Program 替换一个连续 baseline 区间的完整候选路线；逐边 fallback 不丢失 | execution Program compiler |
| `execution_program_bundle.v1` | whole-cell/hybrid proposals、重拼路线、拒绝项和物理操作/化学等效步数核算 | execution Program compiler |
| `execution_program_bundle_oracle.v1` | graph/route/projection/discovery、能力、边界、operation 与完整 bundle 的失败关闭复算 | execution Program compiler |
| `biocatalysis_program_validation.v1` | 与精确 Program/创新/输入输出摘要绑定的专项酶验证；不能授予路线闭合 | validation producer / host contract |
| `biocatalysis_validation_plan.v1` | 未验证酶 Program 的精确边界、酶/辅因子筛选矩阵、选择性目标和必做检测；只生成实验任务 | validation frontier compiler |
| `biocatalysis_validation_frontier.v1` | 当前 route review 中所有待专项验证计划的摘要绑定只读集合；不授予验证或准入 | validation frontier compiler |
| `biocatalysis_program_validation_pack.v1` | 影子准入事件引用的专项验证记录 CAS 包；保留 claim/condition 绑定 | biocatalytic Program store |
| `biocatalytic_program_admission_event.v1` | 已专项验证 superstep 的 append-only、内容寻址影子准入；保留 baseline fallback | biocatalytic Program store |
| `biocatalytic_program_admission_result.v1` | 显式准入、幂等发布和当前影子 store 状态 | biocatalytic Program store |
| `biocatalytic_program_store_replay.v1` | 所有酶 Program admission event 与六类 CAS 输入的重放摘要 | biocatalytic Program store |
| `biocatalytic_program_store_status.v1` / oracle | 当前 graph/route/discovery/bundle/validation 是否已有完全匹配的持久事件 | biocatalytic Program store |
| `route_program_innovation_review.v1` | discovery 与 Program bundle 的只读运行时审查；不执行 store admission | program innovation runtime |
| `program_route_candidate.v1` | 统一的只读 Program 路线候选；固定来源类型、Program/fallback、执行域、证据、资格和多轴指标，不授予路线权威 | Program candidate compiler |
| `program_route_candidate_set.v1` | baseline fallback 与创新替代方案的摘要绑定候选集合；低证据项保留在 exploration 空间 | Program candidate compiler |
| `reported_program_route_pack.v1` | 摘要绑定的 Candidate Route observation、可重算 Candidate Program projection 与选定 route ids；只用于审查 | reported Program route adapter |
| `program_route_portfolio.v1` | exploration、shadow optimizer、experimental-ready、process-ready 四种资格域上的确定性 Pareto 分层；不输出加权“最佳路线” | Program route optimizer |
| `program_route_portfolio_oracle.v1` | 重新编译并逐字复核候选摘要、目标定义、资格集合、Pareto front/layers 与 authority semantics | Program route optimizer |
| `chemical_state.v1` | canonical molecule 的 Phase-1 只读状态投影；当前不拥有生产 identity | transformation program projector |
| `operation_node.v1` | 单 reaction edge 的只读 operation 节点；当前不是可执行 DAG | transformation program projector |
| `transformation_program.v1` | 一个 edge 对应一个 program 的迁移投影；`authoritative=false` | transformation program projector |
| `transformation_program_projection.v1` | states、operations、programs、routes 的完整只读快照 | transformation program projector |
| `transformation_program_projection_oracle.v1` | V4 edge 与 program 投影的逐实体双读一致性 | transformation program projector |
| `route_program_dual_read.v1` | 当前 Workbench `route:*` 的 edge→Program 只读覆盖层；保留步数、proof、条件与 acceptance 快照 | route/program dual-read projector |
| `route_program_dual_read_oracle.v1` | Workbench revision、Program projection、逐路线映射及物理步数的失败关闭等价检查 | route/program dual-read projector |
| `transformation_program_projection_validation.v1` | Program 实体、引用、multiplicity、digest 与 authority=false 的 host contract 检查 | program admission |
| `transformation_program_admission_event.v1` | 历史 canonical graph 与 Program 投影的 append-only CAS 绑定；仅影子准入 | transformation program store |
| `transformation_program_admission_result.v1` | 显式准入、幂等发布与当前 store oracle 结果 | transformation program store |
| `transformation_program_store_replay.v1` | 所有 admission event 及其 graph/projection CAS 对象的重放摘要 | transformation program store |
| `transformation_program_store_status.v1` | 当前 canonical revision 与 durable Program 投影的双读状态 | transformation program store |
| `transformation_program_store_oracle.v1` | 当前投影是否已有完全匹配的 durable admission event | transformation program store |
| `transformation_program_migration_audit.v1` | 跨索引运行的只读迁移分类与 edge/program 计数校验；不执行准入 | program migration auditor |
| `candidate_route_observation.v1` | 从哈希绑定 Workbench 提取的完整候选路线、条件观察和警示；无科学权威 | candidate route projector |
| `candidate_chemical_state.v1` | 候选路线中的结构状态；不继承 canonical molecule authority | candidate Program projector |
| `candidate_operation_node.v1` | 候选转化操作及来源条件观察；条件不授予反应验证 | candidate Program projector |
| `candidate_transformation_program.v1` | 可包含 `inventory_gap` 的非权威 Program；不得参与闭合或 acceptance | candidate Program projector |
| `candidate_program_projection.v1` / oracle | 完整候选路线到 Program 的只读投影与确定性复算 | candidate Program projector |
| `candidate_program_migration_audit.v1` | 多 Workbench 内容去重、投影/空图/无效分类与来源诊断；不准入 | candidate migration auditor |
| `candidate_route_innovation_screen.v1` | Candidate Route 上的数据驱动酶窗口/机理候选筛选；零匹配是有效结果 | candidate innovation screen |
| `canonical_fact_lifecycle_event.v1` | 摘要绑定的撤销、过期与显式恢复事件 | host lifecycle control |
| `canonical_fact_lifecycle_state.v1` | 从 append-only 事件重放的当前事实状态 | lifecycle projector |
| `retrosynthesis_proof_policy.v1` | L0–L4、来源与库存规则 | proof policy |
| `proof_stitched_route_portfolio.v1` | 多样路线组合与 completion | proof stitcher |
| `retrosynthesis_route_workbench.v1` | 有界只读 UI snapshot | workbench projection |
| `retrosynthesis_route_workbench_delta.v1` | revision 间实体增删改 | workbench projection |
| `retrosynthesis_case_dossier.v1` | 审阅后的全局路线、精确来源与库存输入 | operator / upstream planner |
| `retrosynthesis_case_compile_result.v1` | 案卷到重放包的确定性编译摘要 | case compiler |
| `retrosynthesis_case_run_result.v1` | 编译、重放、闭合、导出与分阶段耗时 | one-command case runner |
| `retrosynthesis_replay_pack.v1` | 可移植的目标、事实、库存与期望 | reviewed golden case |
| `retrosynthesis_replay_result.v1` | 重放阶段、指标和 digest 校验 | replay runner |
| `autoplanner_campaign_gateway_result.v1` | CLI/API 操作包装 | campaign gateway |
| `primary_patent_html_materialization.v1` | 官方专利 HTML、段落窗口与零模型审计 | patent HTML adapter |
| `primary_patent_xml_materialization.v1` | EPO ST.36 XML、元素窗口、全文摘要与 publication 身份绑定 | patent XML adapter |
| `real_patent_procedure_gate.v1` | 单个真实专利目标边的来源、procedure、条件完整度与离线复现门禁 | real-case replay script |
| `real_patent_procedure_gate_suite.v1` | 三个独立官方专利与反应类型的结构化来源发布门禁、离线一致性和 fallback 计数 | real-case gate suite |
| `patent_reaction_template_library.v1` | 仓库外、跨 campaign、摘要绑定的专利模板记忆 | patent self-evolution |
| `patent_reaction_template_record.v1` | 原例重放、来源支持和复用成败统计 | deterministic template learner |
| `patent_template_retrieval.v1` | 交给全局 Director/统一准入链的 L0 模板候选 | deterministic template retriever |
| `local_source_ocr_materialization.v1` | 本地 OCR 执行、覆盖率与零模型审计 | local OCR adapter |
| `source_text_companion_binding.v1` | source/HTML 段落或 PDF/page image/OCR text 哈希重放绑定 | deterministic literature parser |
| `structured_fulltext_html.v1` | PMC 全文 HTML、操作性实验段落、段落文本哈希与官方来源 URL 的可重放 companion | literature evidence connector |
| `visual_source_candidate_request.v1` | 至多一次、页图哈希绑定的视觉请求 | visual evidence adapter |
| `visual_source_candidate_observation.v1` | 仅供全局 replan 的 L0 视觉候选 | host-normalized visual adapter |
| `autoplanner_repository_audit.v1` | 当前跟踪树只读审计 | repository audit |

所有持久化 schema 使用 canonical JSON SHA-256 绑定。摘要只证明字节与结构一致，不能
赋予化学真实性。反应、来源、库存和完成各有独立 authority；UI、RunIndex、Blackboard
和缓存都是可重建投影。

实验 Claim 是一个独立的精确边界观察层，不是第二个 canonical fact owner。只读 review 会为三个验证域
生成同一 Claim set；只有调用方显式设置 `enable_experimental_claim_admission=true` 时，非空 Claim set
才会进入追加式存储。重放必须从 graph、route、Program projection、discovery 和 validation pack 重新
生成领域 bundle/feedback/oracle，再逐字节比对 Claim set。成功、失败与不确定结果具有同等持久化资格，
但任何 Claim 都不能创建 reaction proof、Program admission、route completion/acceptance 或能力目录变更。

实验工作投影同样不拥有队列或科学权威。`deficit_frontier.v1` 仍是唯一 canonical 下一工作投影；
`experimental_work_frontier.v1` 只在当前路线创新 review 内展开专项验证子任务，并把相同 exact boundary、
subject 与 domain 的 dirty hint 映射回这些任务。执行结果必须先通过 current-frontier 重投影审计，再作为
候选进入原有生物催化、execution 或 mechanism validation gate；结果封套本身不能生成 Claim。

实验派发不新增调度事实所有者。宿主 registry 只允许预注册并带 host trust record 的
`experiment_executor`；公共 policy 只能在这些 provider 中进一步收窄，不能提升第三方声明。每个精确
request 产生一个稳定 dispatch/task identity，作为既有 RunKernel `validation` reservation 持久化；请求、
选择、handoff、原始结果、审计和 settlement 都进入 CAS，pointer 仅用于恢复。并发调用复用同一 reservation，
指针丢失可由声明幂等的同一 provider 重新物化。结算前必须验证当前 request、provider 版本、真实 CAS
artifact 和结果 schema；RunKernel settlement 的 accepted expansion 始终为空。

Program Phase-1 schema 默认只通过显式查询生成；只有 CLI/API/Gateway 明确启用
`enable_program_admission=true` 时，才把历史 canonical graph 与对应投影写入不可变 CAS，并追加
内容寻址 admission event。事件和 CAS ref 会在每次读取时重新执行 contract validation 与 oracle，
同时以 shadow authority scope 固定到 RunIndex，避免被 GC 误删。该 store 不写 canonical graph，
也不参与 proof、ranking、Workbench 或 acceptance。`edge_ids[]` 仍是生产路线权威；只有多类真实
回放连续通过双读验收后，`program_ids[]` 才能进入非权威影子路线字段。跨运行盘点可用
`transformation_program_migration_audit.v1` 将运行分为 `projection_ready`、`empty_graph`、
`canonical_replay_required` 和 `error`；其中旧 schema 不会被静默兼容，而是明确要求通过案卷或
golden replay pack 重建。

候选 Program schema 解决另一类迁移问题：来源确实展示了路线，但结构转录或反应物清单仍有缺口。
Canonical V4 的 edge admission 继续失败关闭；候选层只把结构有效且无自环/断图的转换标记为
`canonical_admissible`、`inventory_gap` 或 `blocked_candidate`。`source_exploration_closed=true` 只表示
展示链连通，投影强制 `production_closed=false`、`accepted=false`、`authoritative=false`。
Workbench 中可解析但尚未规范化的 SMILES 会在观察边界转为 RDKit canonical isomeric SMILES；真正
不可解析的结构仍以 `candidate_workbench_molecule_smiles_invalid` 失败关闭。

Candidate migration audit 以完整快照内容摘要去重，目标名只作标签；来源 Workbench 中的历史
`portfolio.accepted`、proof level 和条件计数仅作为 `source_diagnostics`，不升级为当前 authority。
Candidate innovation screen 先复算 Candidate Program oracle，再把分子/转换临时投影成筛选图；匹配
结果始终为 proposal，`no_applicable_enzyme_capability` 只有在存在可用能力和可枚举窗口时才可作为
负对照。正向或零匹配都不写 canonical graph、Program store、completion 或 acceptance。

生命周期事件必须绑定当前事实的 canonical ID 与原 `content_sha256`（reaction proof 绑定
`proof_digest`），并使用与事实类型匹配的 host authority scope。`revoke`/`expire` 不删除原事实；
`restore` 必须指向上一条失效事件。来源、结构记录、procedure、reaction proof 或库存观察失效后，
proof stitcher、DeficitFrontier、replay report 与 Workbench 只读取当前有效事实。

官方 XML、HTML 和 OCR 文字都只是确定性 parser 的输入。XML 路径必须重放 publication、
完整 artifact、元素范围和 procedure 文字哈希；HTML 路径必须重放 publication、完整 artifact、
段落范围和文字哈希；OCR 路径必须重放 PDF、页图、文字和引擎身份。
随后还须由 OPSIN/PubChem 独立重建产物与反应物，记录才可能进入 L3。视觉 observation
永远不在该 authority 范围内，即使其 SMILES 可被 RDKit 解析也仍只属于 L0 proposal。

精确结构观察与来源 procedure 是两个实体。Procedure 至少绑定 canonical edge、exact record、
source binding、来源位置和过程文本 SHA-256；缺少片段摘要时，即使输入含有条件字段也不能
获得 `source_exact_reaction_procedure` 权威。字段缺失必须显式记录，不能由模型或 UI 补齐。

`route_innovation.v1` 与来源证据正交。`biocatalytic_superstep` 必须记录酶类别/候选、选择性目标、
被替代的化学等效步数、辅因子和底物范围依据；EC 标签或普通原子映射不能满足专项
`biocatalysis_validation`。`mechanism_extrapolation` 必须绑定论文/专利锚点、严格限制为一跳，
记录机理说明和至少一项可证伪检查；锚点只证明中间体或上游边，不证明外推反应已被文献报道。

`biocatalysis_capability.v1` 禁止用目标名称作为匹配条件。匹配只读取窗口边界结构的净 motif/元素
变化、骨架相似度、SMARTS 适用域、碳/环规模和窗口长度。目录中的论文或数据库先例仅允许把
窗口送入筛选队列；只有专项 `biocatalysis_validation` 才能验证精确底物。外部模型提出的
`mechanism_extrapolation` 必须先经过 `route_innovation_discovery.v1` 的 host admission：前体需为
所绑定 route edge 的真实产物，产物必须是不同且可物化的结构，锚点必须位于同一路线，深度固定
为一跳。只有机理一跳候选进入 `CanonicalIngestionBatch.hypotheses`；酶窗口只输出 Program draft
candidate id。两者都不能旁路写图或写 Program store。

`program_execution_capability.v1` 与单酶能力共用结构边界匹配，但执行语义独立。whole-cell 必须同时
声明 organism/preparation 和细胞制备、细胞转化 operation；hybrid 必须同时含化学与生物转化，且
相应 actor 不得缺失。制备、后处理和分离都按物理操作计数，因此净节省允许为负；这种候选仍保留在
exploration，而不是因“文献有先例但当前方案操作较多”被全盘删除。未完成专项执行验证时，所有
whole-cell/hybrid 候选固定不能进入 shadow optimizer、route completion、canonical graph 或 store。

`execution_program_validation.v1` 必须精确绑定 Program、能力 ID 与能力摘要、执行域、输入/输出
ChemicalState、完整 operation sequence 摘要、必检项布尔结果、actor identity、claim、condition record
及辅因子/载体台账。`success`、`failure`、`inconclusive` 都是可保留结果；只有记录本身有效、结果为
`success`、全部必检项通过且所需台账闭合时，才能使候选进入只读 shadow optimizer。失败或不确定
结果写入 `capability_applicability_feedback.v1` 时只证明该精确边界上的观察，不得删除、禁用或修改
能力目录。`execution_validation_frontier.v1`、`capability_feedback_projection.v1` 及其 oracle 都是摘要
绑定的只读投影，不能写 Program store、canonical graph，也不能授予 proof、completion 或 acceptance。

`mechanism_program_validation.v1` 与 execution 验证使用相同的严格 JSON、摘要、Program/边界状态绑定
骨架，但保留独立领域语义。`success` 必须同时给出 `net_transform_observed`、
`mechanism_consistent` 或 `mechanism_discriminated` 之一；只有全部必检项通过的有效成功结果可使完整重拼
候选进入只读 shadow。`net_transform_observed` 只证明精确输入/输出边界上的净转化，不证明提议的基元
步骤；该差异由 `validation_vector.mechanism` 和候选警示显式保留。有效 `failure` / `inconclusive` 仍进入
`mechanism_feedback_projection.v1`，不得借锚点文献补证，也不得创建 canonical reaction proof、Program
store、completion 或 acceptance。

证明升级链为 L0 hypothesis → L1 materialized → L2 reaction validated → L3 exact source
→ L4 selected-route stock closed。任何阶段不得跳级，也不能由 aggregate count 推导 proof。
