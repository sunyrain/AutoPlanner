# Unified Campaign 输入与质量状态契约

更新：2026-08-10

## 1. 算法输入只有四类

`unified_campaign_spec.v1` 是所有新运行的不可变算法输入，且只包含：

1. canonical target SMILES；
2. `stock_oracle_reference.v1`；
3. `target_constraints.v1`；
4. `campaign_resource_budget.v1`。

目标名称、dataset ID、target index、benchmark manifest、旧 `objective_mode` 和 acceptance
均不属于该结构。名称仍可用于 UI、日志和目录；旧 objective 字段仍可在适配器中读取并发出
弃用收据，但不能改变 Action 注册、排序、预算或停止行为。

saved-run 恢复会从旧 checkpoint/report 生成摘要绑定的
`saved_run_objective_compatibility.v1`。收据逐位置保留已观察到的历史值，即使它不是现有枚举；
缺少字段同样合法。该收据不进入 deficit frontier、scheduler 或 capability registration。
恢复后的 `campaign_action_unified_core_NN` 与对应 snapshot 从既有最大 `NN + 1` 继续，保证旧 Action
execution 前缀不会被报告 stage 去重覆盖。对同一保存状态的克隆回归已证明：保留旧字段与完全删除
旧字段，在同一当前配置下产生相同的 Action binding 后缀。

运行身份与科学身份必须分离。`GlobalCampaignDirector` 的 task ID、context digest、plan ID、run ID 和
graph revision 是可重放的操作收据，但不能成为 canonical hypothesis 的来源身份。Director proposal
统一写为 `director_plan:<digest>`；该摘要覆盖计划的结构、transformation 和科学声明，同时在计算前
剔除 `content_sha256`、`plan_id`、`run_id`、`context_sha256` 与 `graph_revision`。因此 replay、resume
或 ChemEnzy/Codex 完成顺序变化不能仅凭运行收据重命名同一科学 proposal；实际计划化学变化仍会
改变摘要并产生不同来源身份。历史 artifact 不原地重写，旧 `director_task:*` 只按原始收据读取。

约束可表达禁用试剂、最大路线步数、允许执行域、安全上限和允许的库存来源 ID。安全限制的
key 会拒绝 dataset、objective、benchmark、RetroStar、PaRoutes 等控制标签，避免把测试集身份
伪装成化学约束。

## 2. Stock Oracle 绑定

库存输入必须能被内容寻址并区分两种合法形式：

- 冻结索引/清单：绑定文件或 provider-set 的精确 SHA-256；
- 活解析器：绑定 provider/解析器身份、实现合同和 source SHA-256；每次返回的库存 observation
  仍单独内容寻址。

运行时在实际库存查询前重新计算 builder binding；配置后的 builder 被替换、冻结库存文件变化、
provider set 变化或 boundary 不一致时均失败关闭。旧 `autoplanner_run_spec.v1` 没有这些信息，读取时
会迁移为 `compatibility-unbound` reference，`positive_authority=false`；它可用于历史回放，但不能被
表述为已证明库存输入完全可复现。

## 3. RunSpec 迁移

新写入使用 `autoplanner_run_spec.v2`，其中嵌入完整 `campaign_spec`。为保持事件链与既有摘要可重放，
reader 仍接受 v1，并先验证原始 v1 digest，再投影为内存中的统一结构。v1 重新序列化仍保持原字节
语义；新 run 不再写 v1。

`RunSpec` 中暂时保留的 target、acceptance 和 limits 字段属于运行内核兼容面：target 和 limits 必须
与嵌入的 unified spec 一致；acceptance 只供质量编译/历史完成状态使用，不进入 scheduler 输入。

## 4. 多维资源账本

`campaign_resource_budget.v1` 把资源上限固定为互不串账的维度：model（调用、输入/输出
Token、模型墙钟）、visual、native target/frontier search、evidence、stock、reaction validation、
Program、experiment、total task 和整个 campaign wall-time。Program 与 experiment 各有独立硬上限；
耗尽其中一类只会令该类 Action 不可调度，不会挪用 validation 或 native-search 配额。

`RunKernel.task_budget()` 输出可重放的 `campaign_task_budget.v1`。每个任务类同时报告 settled、
in-flight reserved、remaining 和 available；total 维度也把未结算 reservation 计入已承诺容量。
模型预算的显式扩展只替换 model 子向量，不会重置 Program/experiment 上限，进程恢复后继续执行
同一预算。完整 target report、validation fork、CLI/API compact result 和运行 status 都投影同一账本。

`target_solve_resource_envelope.v1` 另外保留 model/visual/native-search 观测，以及模型墙钟和整个
campaign 墙钟。两种墙钟不得互换：前者只累计模型调用，后者累计所有已结算任务。

新 `campaign_action.v2` 在 reservation 前保存 `campaign_action_resource_estimate.v1`。handler 执行时，
RunKernel 通过 execution ID 上下文把同线程产生的子任务绑定到该 Action；结算前再从事件链编译
`campaign_action_resource_usage.v1`，并把预计、实际和 variance 一起写入 Action outcome 与 settlement。
同 revision 的并发 Action 使用不同上下文，资源不会按线程完成顺序串账。历史 v1 Action 只在旧摘要
精确复算匹配时允许读取；历史收据不会被伪造为拥有新资源明细。

Anytime loop 结束时还会重新编译最终 revision，输出 `campaign_unexecuted_action_set.v1`。每个未执行
Action 必须带 scheduler blocker、caller exclusion、同 revision 已尝试/no-gain，或 action-limit、
low-gain、budget 等 loop 终止原因。ChemEnzy timeout 只结算自己的 Action；同 cohort Codex 会独立完成，
尚未启动的 validation 仍保留在 backlog，而不是被当作不存在。

## 5. 完整而互不排斥的质量状态

每个 campaign status、Workbench、target report 和 validation fork 都输出
`campaign_quality_state.v1`，固定含八个轴：

- topology
- reaction validation
- exact evidence
- stock
- conditions
- procurement
- Program validation
- diversity

每轴独立为 `satisfied`、`open` 或 `not_assessed`，同时给出 observed/required count、判定依据和元数据。
某轴缺少测量时不会伪造成功或失败；B5 为真也不会把 evidence、conditions 或 Program validation
联动改成成功。

`configured_acceptance` 是某一 snapshot 上的审计投影。它可以随 proof policy、来源有效性或库存
snapshot 改变，不能选择算法分支，也不能让 `CampaignActionRuntime.run_anytime()` 提前 return。
Action loop 只因动作上限/资源预算、无可执行动作、连续低收益、显式取消或不可恢复错误结束；产品
需要“拿到第一条路线就停”时，应在外部观察 milestone 后发送取消，而不是在核心中加入 B4/B5 特判。

同 revision 的 ChemEnzy/Codex provider 工作允许并发，但 canonical admission 使用稳定顺序；因此外部
延迟只改变墙钟，不改变 graph union、frontier 或后续 Action 排序。

## 6. Anytime trajectory

新生产运行写入 `campaign_anytime_snapshot.v2`。每个 snapshot 固定记录 RunKernel event sequence、
canonical graph revision、累计 campaign task wall time、model/native/task 等全部资源维度、唯一 Action
execution 计数、route/selected/Pareto/B1–B4 计数、紧凑 Pareto route archive，以及 B0–B5 和 Program
milestones。snapshot 同时绑定以下五类运行身份：

1. 核心控制面源码 bundle；
2. target solver 与 Director 配置；
3. `UnifiedCampaignSpec`/目标结构摘要；
4. stock-oracle reference 与底层 binding；
5. Codex model、ChemEnzy/evidence/condition adapter，以及运行后可观察到的 provider/model 工件指纹。

`campaign_trajectory.v2` 按持久 event sequence 重建轨迹，计算 first-route、B1、
first-host-valid-route、B2、B3、B4、B5 和 Program 首达时间，并输出 binding epochs 与资源曲线。
配置、代码或 provider 在 resume 间变化不会被隐藏，而会生成新的 binding epoch；累计 RunKernel 时间
下降会把 `resume_baseline_preserved` 置为 false。v1 snapshot 仍可读，但不会伪造其缺失的时间和绑定。
snapshot 本身是内容寻址权威记录，因此不得通过 UI/stage 限长器有损改写；紧凑展示应另做只读 projection。

评测使用 `campaign_trajectory_cutoff_projection.v1`，并且只允许声明运行前冻结的累计资源上限。投影
逐维检查 wall time、task、attempt、accepted expansion、model/token/model wall time 与 native search，
选择同时不超过所有声明上限的最后一个 v2 snapshot；time-to-first 只从该 snapshot 之前重建。最终
solver claim、最终 gates 和 cutoff 之后出现的路线只能进入 diagnostic `final_state`，不能参与评分。
旧 `objective_mode` 即使由兼容调用者传到 panel 也会被警告并忽略，不能进入 solver command。

Workbench 从同一 trajectory 生成 `workbench_trajectory_history.v1`，不另建历史状态库。当前路线继续由
当前 canonical graph/proof portfolio 决定；历史区只展示首达时间、当时 snapshot 引用与资源坐标。
若某 gate 后续因来源撤销或验证失效变为 false，界面必须显示“历史达到但当前失效”，不得用历史 true
覆盖当前 false。API、HTML 和 PDF 读取同一个已发布 Workbench 工件。

### 6.1 Reviewer export

`CampaignGateway.export()` 除 Workbench JSON/delta/HTML 外，还生成 `campaign_review_bundle.v1` 及四个可
独立复核的组件文件：`campaign_action_trace.v1`、`campaign_failure_trace.v1`、
`campaign_route_lineage_export.v1` 和 `campaign_resource_curve_export.v1`。Bundle 只接受
`content_sha256` 与内容一致的 target solve report；resource curve 还必须通过 trajectory 自身摘要验证。
任一容器损坏都显式返回 unavailable，不从最终 gates 或零散 stage timing 猜测缺失轨迹。失败 trace 只收录
显式运行失败和终态原因，不把 B3/B5 等开放科学门重命名为 runtime failure。

实时 Web 进度使用 `campaign_action_timeline.v1`。已结算记录从同一 target checkpoint 的 Action
execution receipt 投影；当前执行记录从 RunKernel `in_flight_tasks` 中只选择带完整 Action wrapper
身份的 reservation。Handler child task 不重复显示，且该时间线没有 reserve、settle、排序或 canonical
write 权限。轮询只能改变观察到的状态，不能形成第二套 scheduler 或 queue。

## 7. 对外投影

CLI/API/Web 的 compact 结果均保留 campaign contract digest、stock-oracle binding digest、约束和完整
quality state 和 `campaign_task_budget.v1`。完整报告仍保存各轴依据、canonical graph/portfolio 引用
与资源账本，以便 resume、validation fork 和离线审稿复算同一快照。
