# V4 操作手册

## 1. 本地配置

默认运行根目录为 `results/.autoplanner`。可使用以下环境变量或等价的 CLI 全局参数：

- `AUTOPLANNER_RUNTIME_ROOT`
- `AUTOPLANNER_RUNS_ROOT`
- `AUTOPLANNER_ARTIFACT_STORE_ROOT`
- `AUTOPLANNER_RUN_INDEX_PATH`
- `AUTOPLANNER_CACHE_ROOT`
- `AUTOPLANNER_SOURCE_ROOT`
- `AUTOPLANNER_EXTERNAL_DATA_ROOT`
- `AUTOPLANNER_MODEL_ROOT`
- `AUTOPLANNER_VENDOR_ROOT`

凭据只通过环境或仓库外显式路径提供。全局 CLI 参数必须写在子命令之前。

## 2. 创建和运行

```bash
python -m cascade_planner run \
  --run-id target-001 \
  --target-name TARGET \
  --target-smiles SMILES
```

这只创建权威 run，模型/视觉调用为 0。注入审阅后的 `global_campaign_plan.v1`：

```bash
python -m cascade_planner run \
  --run-id target-001 \
  --target-name TARGET \
  --target-smiles SMILES \
  --plan global_plan.json --materialize --closeout
```

同一 run id 和同一计划可安全重试。

### 2.1 任意陌生 SMILES 的有界求解

```bash
python -m cascade_planner solve-target \
  --target-name TARGET \
  --target-smiles SMILES \
  --run-id target-blind-001
```

新运行直接配置库存、验收和预算，不再选择“benchmark/scientific/procurement 求解模式”。
`--objective-mode` 只为旧脚本保留到 2026-10-01；显式使用会在 stderr 发出
`FutureWarning`，且不会改变 Action 集合、scheduler 或停止规则。迁移时使用：

- `--stock-boundary` 或冻结 stock/inventory 输入表达叶节点可用性；
- `--forbidden-reagent`、`--max-route-steps`、`--allowed-execution-domain`、
  `--safety-limit KEY=JSON_VALUE` 与 `--stock-source-id` 表达 target 约束；
- `--minimum-complete-routes`、`--minimum-edge-proof-level` 和
  `--minimum-source-groups` 表达验收；
- model、native search、evidence、validation、visual、Program、experiment 和 wall-time 参数表达资源；
- `--max-program-tasks` 与 `--max-experiment-tasks` 分别设置独立任务上限；两者不会借用
  validation 或 native-search 配额；
- 从最终 report 的 `milestones`、trajectory 与 Workbench 选择结果视图。

新 run 写入 `autoplanner_run_spec.v2` 和 `unified_campaign_spec.v1`。CLI 摘要、API job result、
运行 status 与完整报告都包含 campaign contract/Stock Oracle digest 及八轴
`campaign_quality_state.v1`。`resource_envelope.task_budget` 同时报告各任务类的 settled、reserved、
remaining 和 available；`model_wall_time_s` 与 `run_wall_time_s` 分别表示模型调用和整个 campaign
任务耗时。B5 是快照审计结果，不会提前截断 Action loop；需要首次闭合即停的
产品流程应监听 milestone 后显式取消。契约与旧 v1 迁移规则见
[`UNIFIED_CAMPAIGN_CONTRACT.md`](architecture/UNIFIED_CAMPAIGN_CONTRACT.md)。

恢复旧 `solve-target` 运行时，历史 checkpoint/report 中的 `objective_mode` 不需要人工迁移或删除：
求解器会在报告 stage 中写入 `saved_run_objective_compatibility.v1`，仅记录字段位置和值。旧值、未知值
或字段缺失都不参与 Action 注册、排序和停止。恢复产生的新 `campaign_action_unified_core_NN` 会接在
原最大序号之后；若看到序号从 01 重新开始，应视为不兼容实现或损坏的历史投影，停止使用该报告。

完整兼容期限见
[`OBJECTIVE_MODE_COMPATIBILITY.md`](architecture/OBJECTIVE_MODE_COMPATIBILITY.md)。

正常成本是一轮 `gpt-5.5` low-reasoning 全局路线组合；本地物化、映射、验证、库存审计
以及来源解析都不调用模型。专利证据按固定阶梯处理：官方 Google Patents 完整 HTML
→ 仅未闭合边的 PDF 原生文本 → 仅低文本页的 Tesseract OCR → 可选稀疏视觉候选。
HTML 字节、段落范围和规范化文字分别绑定哈希；搜索摘要不能成为 exact evidence。
Tesseract 是可选本地二进制：若不在 `PATH` 中，运行会记录
`local_ocr_engine_unavailable:tesseract`，不会暗中改用视觉模型。

仅当来源页确实无法由确定性解析闭合时，可显式允许一次稀疏视觉调用：

```bash
python -m cascade_planner solve-target \
  --target-name TARGET \
  --target-smiles SMILES \
  --run-id target-blind-visual-001 \
  --max-model-invocations 3 \
  --max-visual-invocations 1 \
  --max-visual-pages 2
```

三次模型额度分别覆盖初始全局规划、至多一次页面视觉候选、以及有新证据事件时的
一次全局 replan；若只给两次，系统会如实跳过 replan，而不是突破预算。视觉候选经过
主机 SMILES 规范化，但仍是 L0：它不能授予反应验证、exact-source 或库存权威。
同一 campaign 即使 provider 少报用量，也只会持久化准入一次视觉任务。

### 2.2 不重复全局规划的验证 fork

```bash
python -m cascade_planner fork-validation target-blind-001 \
  --run-id target-blind-validation-001
```

该命令重放原始全局案卷，在当前 host 上重新运行结构物化、反应验证、来源获取、库存审计
和 self-evo 学习。默认模型与视觉预算都是 0；需要核对图片型来源时才显式准入：

```bash
python -m cascade_planner fork-validation target-blind-001 \
  --run-id target-blind-vision-001 \
  --max-visual-invocations 1 \
  --max-visual-pages 2 \
  --visual-reasoning-effort low
```

此时派生预算只覆盖一次视觉调用，不会再次调用 Global Director。视觉 observation 必须经过
目标 root、结构连续性、物化和 host admission；被接受也仍不会自动获得 exact-source 权威。

### 2.3 论文 HTML 与受控 PDF 获取

论文来源依次尝试 Europe PMC XML、公开 PMC HTML、PDF 和页面恢复。PMC 对普通 HTTP
客户端返回 reCAPTCHA shell 时，系统只对 `https://pmc.ncbi.nlm.nih.gov` 启动一次隔离、
无凭据、无用户 profile 的 Playwright context；最终跳转 host、HTTP 状态、字节上限和正文
DOI 都必须再次通过校验。挑战页和最终 HTML 分别绑定 SHA-256，之后可从内容缓存重放。
对包含操作性实验段落的 PMC HTML，系统会过滤 Abstract/Figure/Discussion，解析来源内缩写、
产物和投料名称，先生成 target-connected source route，再用冻结 HTML、段落文本哈希和
OPSIN/PubChem registry 独立重建结构。只有 registry 与 host validation 都通过时才产生
exact row；论文标题、摘要和检索片段本身仍不授予证据等级。

需要校园网或人工已登录浏览器才能获取的 PDF 仍进入显式本机队列，不会把 cookie 或凭据
传给服务端。可只处理当前 case/source，避免混入旧请求：

```bash
python scripts/browser_pdf_fetch.py \
  --output-dir PATH_TO_PROXY_QUEUE \
  --case-id CASE_ID \
  --source-ref DOI_OR_SOURCE_REF \
  --max-items 1
```

也可用 `--title-contains` 进一步过滤。每个请求使用新的 tab；下载结果仍须经过 PDF identity、
hash、页面选择和来源绑定，成功下载本身不授予反应证明。

## 3. 从精确来源案卷一键重放

```bash
python -m cascade_planner replay-dossier \
  --dossier config/examples/artemisinin_v4_case_dossier.json \
  --run-id artemisinin-showcase \
  --output-dir local-showcase/artemisinin
```

该命令依次编译案卷、进入 canonical hypergraph、执行来源/反应/库存 worker、拼接
proof portfolio，并导出离线 workbench。输出带分阶段墙钟时间；默认模型和视觉调用均为 0。

需要单独审阅生成的可移植重放包时：

```bash
python -m cascade_planner compile-case \
  --dossier config/examples/artemisinin_v4_case_dossier.json \
  --output local-showcase/artemisinin-pack.json
```

案卷必须包含：少量全局路线族、原子映射反应、覆盖每条边的精确来源记录，以及覆盖
每个深层叶节点的带时间戳库存 offer。缺一项即失败关闭。`--map-missing` 只会调用本机
已安装的 RXNMapper，不会访问 hosted model 或网络。案卷负责输入事实的人工/上游审阅；
编译器不会把摘要哈希误当成化学真实性，也不会把目录可订购性表述为实时本地库存。

## 4. 科学案例重放

```bash
python -m cascade_planner replay-case \
  --pack config/examples/nirmatrelvir_v4_replay_pack.json \
  --run-id nirmatrelvir-golden
```

测试中断恢复：

```bash
python -m cascade_planner replay-case \
  --pack config/examples/nirmatrelvir_v4_replay_pack.json \
  --run-id nirmatrelvir-recovery \
  --stop-after evidence

python -m cascade_planner replay-case \
  --pack config/examples/nirmatrelvir_v4_replay_pack.json \
  --run-id nirmatrelvir-recovery
```

可暂停阶段为 `plan`、`materialization`、`evidence`、`validation`、`stock`。
重放包必须通过内容哈希、来源 artifact、反应身份/验证和库存 schema 校验。

## 5. 运行管理与校验

```bash
python -m cascade_planner list
python -m cascade_planner status target-001
python -m cascade_planner resume target-001 --materialize
python -m cascade_planner resume target-001 --closeout
python -m cascade_planner validate target-001
python -m cascade_planner replay target-001
python -m cascade_planner benchmark target-001 --iterations 3
```

`validate` 比较事件重放、snapshot、规范图 full-recompute oracle 和 workbench 绑定。

### 5.1 Program 迁移影子存储

只读查看当前 canonical graph 的 Program 投影不会创建任何文件：

```bash
python -m cascade_planner programs target-001
python -m cascade_planner program-routes target-001
python -m cascade_planner program-store target-001
python -m cascade_planner audit-programs --limit 100
python -m cascade_planner audit-programs --run-id target-001
```

只有以下显式命令会追加 admission event：

```bash
python -m cascade_planner admit-programs target-001 \
  --enable-program-admission
```

`audit-programs` 只读取 RunIndex 与各运行的 canonical graph/store，输出
`projection_ready`、`empty_graph`、`canonical_replay_required` 或 `error`；它不创建 Program
目录、CAS 对象或 admission event。

`program-routes` 从同一 graph revision 重新生成 Workbench 和 Program projection，把每条实际展示
`route:*` 的 `edge_ids[]` 一一映射为次级 `program_ids[]`，并复核 route identity、物理步数、proof、
条件和 acceptance 快照。它不会修改 Workbench、graph 或 Program store。真实 Fluvastatin current
run 的 5 条展示路线和 2 条 replacement route 已通过该 oracle：17 个 edge 引用映射到 11 个不同
Program，物理步数差异为 0；这仍不表示该 run 通过路线 acceptance。

对历史 Workbench 中“路线连通但部分转换低可信”的案例，可生成独立 Candidate Program 投影：

```bash
python scripts/project_candidate_programs.py path/to/route_workbench.json \
  --observation-output candidate-route-observation.json \
  --projection-output candidate-program-projection.json
```

该命令验证 Workbench 摘要、路线拓扑和逐转换 canonical admission，保留来源条件观察；输出中的
`inventory_gap` 会继续可见，但命令不写 canonical graph、Program store 或 acceptance。
输入结构在观察边界统一规范化；合法但非 canonical 的 SMILES 不会因文本形式不同被拒绝，不可解析
结构、自环、环路、断图或摘要篡改仍会失败关闭。

当前冻结的真实回归包括：Bufotalin 20 步（15 个 `canonical_admissible`、5 个 `inventory_gap`）和
Atorvastatin 11 步 L0 候选（11 个 `canonical_admissible`、条件观察为 0）。两者都强制
`production_closed=false`，不得据此声称反应或生产闭合。

批量盘点结果目录中的 Workbench，并按内容摘要去重：

```bash
python scripts/audit_candidate_workbenches.py results \
  --output candidate-workbench-migration-audit.json
```

输出只分类 `projection_ready`、`empty_graph`、`invalid_snapshot` 和未知 `error`；来源的历史 acceptance
只进入诊断字段。扫描器排除浏览器 profile，并对命名 Workbench 的 UTF-8、JSON 和 schema 错误失败
关闭。

对任一冻结 Candidate Route 执行数据驱动酶机会扫描：

```bash
python scripts/screen_candidate_innovations.py candidate-route-observation.json \
  --capabilities config/route_innovation_capabilities.v1.json \
  --output candidate-route-innovation-screen.json
```

Bufotalin 回归产生 5 个 Program draft proposal，其中一个 HSDH 窗口以 1 个酶步骤候选替代
6 个化学步骤；酶窗口不再进入 canonical reaction ingestion batch。
Ibrutinib 的 3 条可枚举路线对同一能力目录均为零匹配，是冻结的无适用酶负对照。两者都不授予
精确底物适用性、反应证明或路线闭合。

对 current canonical route 使用同一审查入口；返回值同时包含 `program_bundle`、
`validation_frontier`、`program_route_candidates`、`program_optimizer` 与
`program_optimizer_oracle`：

```bash
python -m cascade_planner program-innovations target-001 \
  --route-id route:example \
  --capabilities-json config/route_innovation_capabilities.v1.json
```

可把已经完成 Candidate Program 投影的完整文献/展示路线同时送入同一只读比较入口：

```bash
python -m cascade_planner program-innovations target-001 \
  --route-id route:example \
  --capabilities-json config/route_innovation_capabilities.v1.json \
  --reported-candidates-json reported-program-route-packs.json
```

输入文件是 `reported_program_route_pack.v1` 数组；每项必须同时包含 digest-bound
`candidate_route_observation.v1`、可由前者精确重算的 `candidate_program_projection.v1` 和
`route_ids`。adapter 按 canonical target SMILES 对齐，不读取目标名称。有逐步或 route-level 来源
绑定时标为 `literature`；没有来源时标为 `chemical` 并加入 `SOURCE_PROVENANCE_MISSING`，不会把旧
展示资产伪装成论文证据。两类 reported candidate 都只进入 exploration，固定
`shadow_optimizer=false`、`route_completion=false`，且该参数在 admission 入口会被显式拒绝。

`validation_frontier` 只为尚未通过专项验证的 Program 生成精确输入/输出状态、酶候选、辅因子/再生
矩阵、选择性目标和必做检测。它明确输出 `grants_validation=false`，文献类比或 SI 获取失败不会被
伪装成精确底物验证。当前 Bufotalin 六边区间已通过 fresh V4 run 重建为 current canonical 6→1
HSDH 阳性 proposal，但仍缺精确底物专项验证，因此只生成实验计划，不能持久准入。

`program_optimizer` 不选择一个隐含权重的“最佳路线”，而是在四个资格域分别给出完整 Pareto
分层：`exploration` 保留 baseline、reported Candidate Program 路线与所有低证据候选，`shadow_optimizer` 只接收已通过专项门的
替代方案，随后是更严格的 `experimental_ready` 和 `process_ready`。来源类别不参与评分；当前接入
的生成器已有 baseline、酶 Program 替代、digest-bound reported Candidate Program adapter、
结构驱动的机理一跳重拼器，以及数据驱动的 whole-cell/hybrid execution Program adapter。机理 proposal 只有在前体和产物分别精确
匹配同一路线的上游/下游状态、被跨越边构成无外部分支的连续区间时，才生成完整
`mechanism_restitched_program_route.v1`。无法接回路线的 proposal 继续显示在 discovery，并在
mechanism bundle 中给出 `mechanism_full_route_restitch_missing`，不会伪装成 route candidate。即使已
重拼，未验证机理候选仍为 exploration-only、最低 proof 0、反应/条件/来源缺口至少各 1，且锚点来源
不被解释为报道了外推反应。系统可消费严格的 `mechanism_program_validation.v1`：有效成功且全部必检项
通过时，只把完整重拼候选开放给只读 shadow；失败和不确定结果仍作为 exact-boundary feedback 保留。
`net_transform_observed` 只表示净转化成立，不等于提议的基元机理被证明；所有结果仍固定不创建 reaction
proof、Program store admission 或 route completion。预期成功率、纯化次数、原料/辅因子成本和 PMI 没有
可靠数据时会列在 `unmodeled_objectives`，不会由零值或目标名称猜测。即使某个低证据候选位于后续
Pareto layer，它仍保留在 exploration 输出中。

同一个 `--capabilities-json` 可加入 `execution_domain=whole_cell|hybrid` 的
`program_execution_capability.v1` 记录。记录必须声明结构适用域、文献先例、actors、顺序
`operation_blueprints`、辅因子/载体账本和选择性目标。whole-cell 必须有 organism/preparation；
hybrid 必须同时有 chemical 与 enzymatic/whole-cell transform。匹配后，系统使用与酶/机理相同的
连续区间重拼器生成完整 Program 路线并保留逐边 fallback。培养、后处理和分离均计入物理操作，
所以 `net_step_savings` 可以为负；候选仍在 exploration 可见。系统可消费严格的
`execution_program_validation.v1`：成功、失败和不确定结果都必须绑定精确边界、完整 operation sequence、
actor、claim、condition 与辅因子/载体台账。只有全部检查通过的成功记录能使候选进入只读 shadow；
失败与不确定结果保留为 exact-boundary capability feedback，不会删除能力或候选。当前已有执行器中立的
request/result 封套、原始工件摘要绑定、current-frontier 只读审计，以及宿主信任的人工交接 provider 与
RunKernel 单账本派发/恢复；仍没有真实实验设备或网络实验 provider，也没有 execution Program store
admission 路径。审计释放的验证候选必须再次作为 `validations` 输入通过原领域
gate，之后结果才可以进入独立实验 Claim store；所有 execution
候选仍固定 `route_completion=false`。Gateway/HTTP
始终返回 `execution_program_bundle` 与 `execution_oracle`；合法但不适用的能力返回 oracle 通过的空
bundle，不视为系统错误；同时返回 `execution_validation_frontier`、
`execution_capability_feedback` 与 `execution_feedback_oracle`。机理路径对应返回
`mechanism_validation_frontier`、`mechanism_experiment_feedback` 与 `mechanism_feedback_oracle`；未验证候选
产生计划，成功/失败/不确定结果都可被精确复算。

只有具有有效 `biocatalysis_program_validation.v1` 的候选才可显式写入独立影子 store：

```bash
python -m cascade_planner admit-program-innovation target-001 \
  --route-id route:example \
  --capabilities-json config/route_innovation_capabilities.v1.json \
  --validations-json exact-substrate-validations.json \
  --enable-biocatalytic-program-admission

python -m cascade_planner program-innovation-store target-001
```

该 store 与一边一 Program 的迁移 store 分离，追加内容寻址事件并固定 graph、route、baseline
projection、discovery、bundle、validation pack 六个 CAS 对象。事件可从空 RunIndex 重建 GC pin；
准入前后 canonical graph revision、proof、completion、acceptance 和 `edge_ids[]` 权威均不变。

三个验证域还会统一生成 `experimental_observation_claim_set.v1`。只读 review 已返回 Claim set、精确边界
applicability calibration、dirty-domain 提示及各自 oracle；只有显式命令才把非空 Claim set 写入独立
append-only store：

```bash
python -m cascade_planner admit-experimental-claims target-001 \
  --route-id route:example \
  --capabilities-json config/route_innovation_capabilities.v1.json \
  --mechanism-proposals-json mechanism-proposals.json \
  --validations-json experimental-validations.json \
  --enable-experimental-claim-admission

python -m cascade_planner experimental-claim-store target-001
```

该入口允许只含失败或不确定结果的非空 Claim set，不要求“成功”才保存；空观察集拒绝写入。事件固定
graph、route、Program projection、discovery、排序后的 validation pack 和 Claim set 六类 CAS。每次读取都
重新编译生物催化、execution、mechanism 的 bundle/feedback/oracle 和统一 Claim set；事件或 CAS 篡改
失败关闭。它不写 canonical graph、reaction proof、Program store、route completion/acceptance 或能力目录。

每次只读创新 review 还返回 `experimental_work_frontier` 与 oracle。工作项绑定唯一 canonical
`deficit_frontier`、被替代 edge span、源 validation plan 和稳定 `experiment_execution_request.v1`。
外部实验室、机器人或服务应把结果封装为 `experiment_execution_result.v1`，然后执行：

```bash
python -m cascade_planner audit-experiment-result target-001 \
  --route-id route:example \
  --capabilities-json config/route_innovation_capabilities.v1.json \
  --mechanism-proposals-json mechanism-proposals.json \
  --validations-json existing-validations.json \
  --result-json executor-result.json
```

命令会重建当前工作投影；旧 request、自造 request、摘要/边界不匹配或缺原始工件哈希的结果失败关闭。
通过只表示 `domain_validation_candidate` 可提交给既有 gate，不会自动追加 Claim 或修改 graph。

需要真正形成可恢复交接时，先准备严格的 provider policy。公共 policy 只能从宿主已经注册并信任的
executor 中选择，不能把第三方 provider 提升为实验执行者：

```json
{
  "schema_version": "experiment_executor_policy.v1",
  "enabled": true,
  "allowed_provider_ids": ["autoplanner.manual_experiment_executor"],
  "preferred_provider_ids": ["autoplanner.manual_experiment_executor"],
  "allowed_domains": ["biocatalytic"],
  "allow_network_access": false,
  "max_estimated_cost_units": 0
}
```

随后将原始 JSON 记录放入 CAS、派发并结算：

```bash
python -m cascade_planner stage-experiment-artifact target-001 \
  --artifact-json raw-assay.json \
  --logical-name raw-assay.json \
  --enable-experiment-artifact-staging

python -m cascade_planner dispatch-experiment target-001 \
  --route-id route:example \
  --capabilities-json config/route_innovation_capabilities.v1.json \
  --request-id experiment-request:... \
  --provider-policy-json experiment-provider-policy.json \
  --enable-experiment-dispatch

python -m cascade_planner settle-experiment-dispatch target-001 \
  --route-id route:example \
  --capabilities-json config/route_innovation_capabilities.v1.json \
  --dispatch-id experiment-dispatch:... \
  --result-json executor-result.json \
  --enable-experiment-settlement
```

RunKernel 的 `validation` reservation 是唯一操作账本；系统不会另建 experiment queue。请求、provider
selection、handoff、result、review 与 settlement 都是 CAS 对象。pointer 丢失时，幂等 provider 可用
`recover-experiment-dispatch ... --enable-experiment-dispatch-recovery` 重新物化同一 handoff。并发派发与
并发结算复用同一 task；无效 executor、缺失 CAS artifact、旧 frontier 或篡改收据不会结算任务。

HTTP 使用 `GET /api/v4/program-migration`、`GET /api/v4/runs/<run_id>/programs`、
`GET /api/v4/runs/<run_id>/programs/routes`、
`GET /api/v4/runs/<run_id>/programs/store` 和
`POST /api/v4/runs/<run_id>/programs/innovations`、
`GET /api/v4/runs/<run_id>/programs/innovations/store`、
`POST /api/v4/runs/<run_id>/programs/innovations/admit`、
`POST /api/v4/runs/<run_id>/programs/innovations/experiments/audit`、
`POST /api/v4/runs/<run_id>/programs/innovations/experiments/artifacts/json`、
`POST /api/v4/runs/<run_id>/programs/innovations/experiments/dispatch`、
`POST /api/v4/runs/<run_id>/programs/innovations/experiments/recover`、
`POST /api/v4/runs/<run_id>/programs/innovations/experiments/settle`、
`GET /api/v4/runs/<run_id>/programs/innovations/claims/store`、
`POST /api/v4/runs/<run_id>/programs/innovations/claims/admit`、
`POST /api/v4/runs/<run_id>/programs/admit`。创新审查 POST JSON 包含 `route_id`、`capabilities`，
并可选传 `mechanism_proposals` 与 `validations`；输出绑定区间边界、逐边 fallback，以及物理步数、
化学等效步数和净节省。即使专项验证有效，该入口仍为只读，不执行 Program store admission，
也不改变 route completion。`validations` 按显式 schema 分流：生物催化验证仍走已有影子准入审查，
execution 与 mechanism 验证都只影响各自的只读 shadow 与反馈投影。准入 POST JSON 必须包含
`{"enable_program_admission": true}`；酶 Program 准入还必须包含有效 `validations` 与
`{"enable_biocatalytic_program_admission": true}`；实验 Claim 准入必须包含非空有效观察与
`{"enable_experimental_claim_admission": true}`。三个写入口缺省都返回 409，不产生 store 事件。

Store 同时保存历史 canonical graph 与对应 Program projection 的不可变 CAS ref，并追加内容寻址
事件。重复准入同一 revision 是幂等的；canonical graph 前进后，`validate`/`replay` 会报告
`program_store_current_projection_equal=false`，直到新 revision 被显式准入。此门禁只检查迁移
一致性，不改变 `edge_ids[]`、proof、route completion 或 acceptance。
`benchmark` 不访问模型或网络。

## 6. 导出、Web 与存储维护

```bash
python -m cascade_planner export target-001 --output-dir local-export
python -m cascade_planner serve
python -m cascade_planner gc --dry-run --minimum-age-hours 24
python -m cascade_planner audit
```

`export` 会同时生成 `route_workbench.json`、delta、离线 HTML，以及
`campaign_review_bundle.json`、`campaign_action_trace.json`、
`campaign_failure_trace.json`、`campaign_route_lineage.json` 和
`campaign_resource_curve.json`。审稿包及四个组件均带独立 `content_sha256`；若运行没有
target solve report，或 report/trajectory 摘要无效，文件仍会写出但明确标记 `available=false`，
不会用最终状态猜测缺失历史。

默认启动隔离的 Canonical V4 surface，不装载旧 Blackboard compiler 或旧 campaign。
唯一首页和唯一 Web 逆合成启动入口为 `/`。它运行 Strategy Generator + 三分支 Route Builder，
并通过 SSE 逐次展示模型输出和 host replay。统一结果工作区为 `/v4`，集中展示兼容入口为
`/v4/showcase`；它们共用 `/api/v4/workspace`、`/api/v4/showcase`、`/api/v4/runs` 和同一
Workbench read model。旧 `/synthesis` 与 `/v4/console` 仅重定向到 `/`，不再拥有独立启动表单。
Program 迁移盘点为
`/api/v4/program-migration`，单运行影子投影为 `/api/v4/runs/<run_id>/programs`。默认仅绑定
`127.0.0.1`。CLI 不提供隐式删除模式；
GC 只生成 dry-run 计划。

默认 `--server auto`：优先使用 Waitress；未安装时自动回退到 Flask。需要显式指定开发服务器时使用：

```bash
python -m cascade_planner serve --server flask --host 127.0.0.1 --port 8878
```

浏览器打开 `http://127.0.0.1:8878/`，输入目标 SMILES 后由唯一 Strategy Builder 表单
`POST /api/v4/jobs`；首页通过 SSE 投影每次模型输出，同时每 3 秒同步本地任务。
`/v4` 结果工作区每 2.5 秒轮询 `/api/v4/jobs` 与单任务进度；首次路线、底层 provider 调用、来源数、exact rows、视觉
调用和 token 分开显示。运行详情的“统一 Action 时间线”每 2.5 秒从同一 checkpoint/RunKernel 状态更新，
把 ChemEnzy、Codex、证据、验证、条件、库存及 Program/实验显示为已完成、执行中、部分完成或失败；
它不是第二个任务队列。历史卡片是停止执行的不可变快照，内核原始状态只作审计，不能被
理解为仍有后台线程或已经达到 B3/L4。

网站任务队列只覆盖当前 Web gateway 注册的运行。若直接使用 CLI 并通过 `--run-dir`/独立输出目录创建另一份
`run_index.sqlite3`，该运行不会因位于仓库 `results/**` 下而自动进入网站；这是两个运行注册域，不是 SSE 丢失。
需要网页实时监控的 smoke 必须通过同一服务的 `POST /api/v4/jobs` 启动，随后使用返回的 `job_id` 连接
`/api/v4/live/<job_id>/events`。不要增加扫描任意结果目录的隐式导入器来制造第二个任务状态权威。
已知目标的 blind/benchmark HTTP 重现实验不能依赖 interactive 自动库存解析，必须在 JSON 请求中同时提供
`benchmark_stock_index`、`benchmark_stock_index_sha256` 和 `benchmark_stock_name`；缺少路径或哈希时会在任何
付费模型调用之前 fail closed。

Canonical Web 不发送 `objective_mode`。旧 API 客户端仍可暂时传入该字段，但
`POST /api/v4/solve-target` 和 `POST /api/v4/jobs` 会返回 `Deprecation: true`、HTTP
`Warning: 299`，async job 还会在 `request_warnings[]` 中保留收据。调用方必须在
2026-10-01 前迁移到显式 stock、acceptance 和 budget 参数。

仅在需要旧 Agent/Statin/RouteForest 综合界面时显式启动兼容 surface：

```bash
python scripts/legacy/serve_combined_web.py
```

该 surface 不是 Canonical V4 的科学权威，不能承载新产品逻辑。

## 7. 本地发布门

```bash
python -m pytest -q
python -m ruff check cascade_planner tests scripts
python -m cascade_planner audit
git diff --check
git status --short
```

仓库不使用 CI/Action。Nirmatrelvir golden 与 Artemisinin dossier 必须闭环；Paclitaxel 必须明确显示未验证的
多路线；无本地 fixture 的目标必须具名失败，不能假成功。实际 Codex A/B 只有在显式
非零预算下运行，并记录调用、token、时间和 portfolio gain。
