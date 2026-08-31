# Canonical V4 主干冗余审计与渐进迁移记录

审计基线：2026-07-16
原则：先保持主干运行、保存格式、proof 与 acceptance 不变，再删除可证明冗余；新能力只能先以
只读投影和双读 oracle 进入。

## 1. 回归基线

- 完整离线测试：`1770 passed, 3 skipped, 2 subtests passed`；
- 主干入口：CLI / `CampaignGateway` / `RetrosynthesisCampaignService` / Canonical Hypergraph；
- 主干禁止依赖旧 Blackboard controller、旧 Codex campaign、旧 frontier/portfolio 和
  RouteForest compiler；
- 仓库只读审计：无 tracked credential、generated artifact、duplicate asset 或 parse error。

该结果是本轮清理前的回退点。之后每个切片必须先通过 focused tests，再重跑完整离线集。

## 2. 冗余与风险盘点

| 发现 | 风险 | 审计判断 | 处理 |
| --- | --- | --- | --- |
| `web.app` 同时注册 V4 与旧 Agent/Statin/Blackboard API | 启动 V4 时装载旧 controller、RouteForest 和旧 campaign | 属于兼容面，不应继续作为默认主干 | 新增独立 `web.v4_app`；2026-07-26 已删除主 CLI combined surface，旧界面仅由 legacy 脚本启动 |
| Web 安全门嵌在组合 App | 新增独立 App 时容易复制或漏掉写请求保护 | 可证明的公共横切逻辑 | 抽为 `web.security.install_web_security`，两个 surface 共用 |
| V4 workbench authority 从 `route_forest_layout` 导入通用 SHA-256 | 名称和依赖方向暗示 V4 依赖旧显示布局 | 通用工具依赖泄漏 | 摘要函数移到 `runtime.canonical_json`；调用方已迁移，旧 alias 已删除 |
| Repository audit 把 `TYPE_CHECKING` 导入报成 dead import | 清理工具会诱导删除正确类型依赖 | 审计误报 | 增加专用 AST helper；当前 dead-import candidate 为 0 |
| Service/Gateway façade 已到行数预算上限 | 继续直接堆功能会形成新巨石 | 真实增长风险 | workbench publication、run identity 分拆为小模块后再接新能力 |
| 旧 Blackboard/RouteForest/campaign 约 1.7 万行以上 | 维护面大、概念重复 | 当前树没有 compatibility telemetry，但外部 saved run 未审计 | 不删除；登记 shim、禁止新产品逻辑、只允许兼容/正确性/删除修改 |
| `route_innovation` 把超级步骤附着在 reaction edge | 可展示创新，但不能改变真实路线搜索空间 | 有价值的过渡实现，不是 GRIA 核心 | 冻结为 regression bridge；不再扩大其生产语义 |
| 大量历史 scripts 与 launchers | 入口分散 | 多数用于回放/迁移/研究，不是主干写入权威 | 本轮不移动；后续按调用遥测和 golden migration 分批归档 |

## 3. 本轮实际清理

没有删除历史文件，也没有改变 canonical graph、proof portfolio 或 artifact schema。本轮完成：

1. 主干 Web 与组合兼容 Web 分离；
2. 共享 Web 安全门；
3. 通用 canonical JSON 摘要与旧布局模块解耦；
4. Repository audit 类型导入误报修复；
5. workbench publication 与 run identity 从满载 façade 抽离；
6. `legacy.combined_web_surface` 纳入 compatibility inventory。

隔离后的 `web.v4_app` 导入不会加载 `agentic_blackboard_controller`、RouteForest compiler 或
`codex_retrosynthesis` legacy campaign。此处记录的旧综合展示随后已于 2026-08-29 退役；
当前只通过 Canonical Web 与 Workbench 查看历史运行，不再保留可执行的 combined Web launcher。

## 4. 第一块新功能：Program 只读基座

新增 `application.transformation_programs`，从同一 canonical graph 确定性产生：

- `chemical_state.v1`；
- `operation_node.v1`；
- `transformation_program.v1`；
- `transformation_program_projection.v1`；
- `transformation_program_projection_oracle.v1`。

当前严格边界：

- 一个 canonical reaction edge 只读投影为一个 single-reaction program；
- 保留前体 multiplicity、edge/route 对应、来源/procedure/proof 引用；
- 只读查询不写 canonical graph，也不隐式持久化 program store；
- `edge_ids[]` 仍是生产路线权威；
- program validation vector 明确 `authoritative=false`；
- oracle 逐项比较 state、operation、program、route、counts 和 content digest；
- 新查询为 `GET /api/v4/runs/<run_id>/programs`，不进入默认 workbench 或 acceptance。

这是 Phase 1 的第一切片“schema + edge→program 双读投影”；第二切片的显式
program admission/store 见第 7 节。Route 仍未切换到 `program_ids[]`。

## 5. 后续删除门

任何 legacy 模块只有同时满足以下条件才可删除：

1. compatibility telemetry 在约定窗口内为 0，或所有调用方已列明并迁移；
2. saved-run/golden replay 已迁移到 canonical V4；
3. replacement 通过同等科学与展示验收；
4. import graph、CLI/Web 入口和文档不再引用；
5. 完整离线测试、Nirmatrelvir zero-model replay 和至少一个 fresh target smoke 同时通过。

原定下一切片的 append-only program admission/store 已按默认关闭方式完成；在多类真实回放中
store 与 projection 连续通过 oracle 以前，仍不得让 `program_ids[]` 参与 proof、ranking 或 completion。

## 6. 清理后门禁结果

- 完整离线集：`1776 passed, 3 skipped, 2 subtests passed`；
- Ruff：`cascade_planner`、`tests`、`scripts` 全部通过；
- `git diff --check`：通过（仅保留已有 JSON 行尾提示）；
- repository audit：`status=clean`，dead import、parse error、duplicate asset、credential candidate 均为 0；
- 最新架构 PDF：12 页，包含 Program 只读状态与 `edge_ids[]` 权威边界，并完成关键页视觉抽查。

## 7. 第二块新功能：append-only Program shadow store

新增 `application.transformation_program_store` 与独立 contract validator，严格保持以下顺序：

1. 显式 gate；缺少 `enable_program_admission=true` 时在任何目录/CAS 写入前拒绝；
2. 从当前 canonical graph 重新生成投影并执行逐实体 contract validation 与原 oracle；
3. 将历史 canonical graph 和 Program projection 写入不可变 ArtifactStore；
4. 追加以内容摘要寻址的 admission event，重复相同 revision 不产生第二个事件；
5. 以 `shadow_program_admission_*` scope 登记 RunIndex；GC 还会直接重放事件恢复 pin，因此重建空索引也不会误删引用对象；
6. 每次读取重放 event、graph CAS、projection CAS、contract validation 和 admission oracle；
7. `validate`/`replay` 仅在 store 已初始化时要求当前 canonical revision 有对应 admission。

写入入口只有：

- CLI：`admit-programs RUN_ID --enable-program-admission`；
- Gateway：`admit_programs(..., enable_program_admission=True)`；
- HTTP：`POST /api/v4/runs/<run_id>/programs/admit`，JSON 明确传入 true。

只读 `programs`、`program-store` 和两个 GET endpoint 不创建 store。事件/CAS 篡改、来源 graph
不匹配、authority 被伪造、entity reference 漂移或 oracle 不等都会失败关闭。16 路并发相同准入
只发布一个事件；Windows 长 run 路径也已通过真实回放修正。

Nirmatrelvir golden pack 已把 18 个 ChemicalState、12 个 single-reaction OperationNode、12 个
TransformationProgram 和 2 条路线无损写入/重放；准入前后 RunKernel state digest 与 canonical
scientific digest 不变，原 2 条闭合路线的生产判定不变。

## 8. 第二切片最终门禁

- 完整离线集：`1787 passed, 3 skipped, 2 subtests passed`；
- 新增覆盖：默认关闭、显式 CLI/API/Gateway 开关、幂等与 16 路并发、event/CAS 篡改、
  stale revision 检出与修复、Windows 长路径、RunIndex 重建后 GC pin 恢复；
- Nirmatrelvir golden replay：12 edges、2 routes、0 model/visual invocation，Program store 重放一致；
- Ruff、架构依赖/行数预算和 `git diff --check`：通过；
- 生产不变量：`edge_ids[]`、RunKernel state digest、canonical scientific digest、proof 与 completion
  在 Program admission 前后不变。

## 9. 第三块新功能：跨运行迁移盘点

新增 `interfaces.program_migration`，经 CLI `audit-programs`、Gateway `audit_programs` 和
`GET /api/v4/program-migration` 暴露同一只读审计。审计只从 canonical graph 生成 Program 投影并
读取 shadow store，不执行 admission；目标名仅作为标签，不参与规则。投影同时携带从 canonical
graph 本身导出的 `source_counts`，避免把可重建 RunIndex manifest 误当成科学计数权威。

2026-07-16 对当前 RunIndex 的现场盘点结果：

| 项目 | 结果 |
| --- | ---: |
| 索引运行 / 目标标签 | 38 / 10 |
| 非空 `projection_ready` | 2 |
| `empty_graph` | 3 |
| `canonical_replay_required` | 33 |
| 未分类 `error` | 0 |

33 个历史运行使用早期 canonical graph 形态，当前严格 loader 会以
`canonical_graph_validation_failed` 失败关闭。它们不是 Program 投影器丢边，而是需要从 reviewed
dossier 或 replay pack 显式重建；审计器不会用隐式兼容掩盖数据迁移债务。

新增 Artemisinin golden replay 作为第二个跨类别样本：5 个 ChemicalState、2 个
OperationNode/TransformationProgram、2 条路线；与 Nirmatrelvir 一样，影子准入前后 RunKernel 与
canonical scientific digest 不变。Bufotalin 与 Atorvastatin 现在已有 Candidate Program 回归，
但尚未成为 canonical replay/store 证据；总体仍未覆盖至少五类目标和无适用酶负对照，因此
`program_ids[]` 路线影子字段门禁**未满足**。

## 10. 第三切片门禁结果

- 完整离线集：`1791 passed, 3 skipped, 2 subtests passed`；
- Program migration、CLI、API、Gateway、Nirmatrelvir、Artemisinin 与架构预算聚焦集：
  `28 passed`；
- 当前 38-run 审计摘要 SHA-256：
  `7e1f3673c2bf6f865fdd899ca75faf150e0cef9a09c9caad7ac8b4543cfc4616`；
- Ruff 聚焦检查通过；未知迁移错误为 0；
- 生产不变量仍为 `edge_ids[]` 权威，审计和只读 projection 均不创建 Program store。

## 11. 第四块新功能：完整低可信路线的 Candidate Program 投影

对 Bufotalin 20 步展示案卷执行现行 canonical admission 时，15 步可进入搜索门，5 步因
`element_inventory_not_conserved` 被正确拒绝。直接放宽 edge admission 会污染科学图；全盘丢弃又会
违背“已报道路线应保留、缺口应警示”的产品语义。因此新增两个通用纯模块：

- `candidate_route_observations` 从任意摘要有效的 `retrosynthesis_route_workbench.v1` 提取路线相关
  状态、转换、来源、条件观察和警示；
- `candidate_programs` 对每个转换重跑 canonical admission，并分类为
  `canonical_admissible`、`inventory_gap` 或 `blocked_candidate`；自环、无效结构和断图仍失败关闭。

真实 Bufotalin 输出为 21 个 CandidateChemicalState、20 个 CandidateOperationNode、20 个
CandidateTransformationProgram、1 条完整候选路线；其中 15/5/0 分别属于上述三类，15 个操作保留
来源条件观察。来源案卷的 `exploration_closed` 得以保留，但投影强制
`production_closed=false`、`accepted=false`、`authoritative=false`，且不写 canonical graph 或
Program store。

冻结资产：

- `benchmarks/bufotalin_candidate_route_observation.v1.json`，canonical content SHA-256
  `0230f065ce67f48b2815a49f351a1877a8da0bbb56d5b495b8242711975a96be`；
- `benchmarks/bufotalin_candidate_program_projection.v1.json`，canonical content SHA-256
  `db3399b9e088c355f0d11ba4658e8cf3fdb5fd33c6f3990c0881178cc645f8e2`。

## 12. 第四切片门禁结果

- 完整离线集：`1794 passed, 3 skipped, 2 subtests passed`；
- Candidate Program 与架构预算聚焦集：`11 passed`；
- 真实 Bufotalin oracle：21 states、20 operations/programs、1 route、15 canonical-admissible、
  5 inventory-gap、0 blocked；15 个操作保留条件观察；
- 篡改、无效结构、自环、二节点环和断图均失败关闭；
- 主干不变量：原 5 个不守恒转换仍不能成为 canonical edge，候选路线仍不能改变 proof、completion
  或 acceptance。

## 13. 第五块迁移证据：Atorvastatin 与 12 个他汀资产盘点

旧 Atorvastatin 工作台的 11 步路线使用合法但未规范化的 SMILES。候选观察层现在会把可解析结构
统一成 canonical isomeric SMILES；不可解析结构仍失败关闭。真实投影得到 22 个 ChemicalState、
11 个 OperationNode/TransformationProgram 和 1 条路线，11 步均可通过元素库存准入，但全部仍是
无条件、无反应证明的 L0 候选，强制 `production_closed=false`、`accepted=false`。

冻结资产：

- `benchmarks/atorvastatin_candidate_route_observation.v1.json`，文件 SHA-256
  `78982b6c76a5931526d6b1fdd0339347bdf4653cff0565f2d65658025a6a2e68`；
- `benchmarks/atorvastatin_candidate_program_projection.v1.json`，文件 SHA-256
  `dd474bcdc386a92653903c4bdb50bfe86c2973c1f2dd287dcedfa46e0d7d8bb3`；
- `benchmarks/statin_candidate_migration_audit.v1.json`，记录 12 个独立母体实体的资产级分类，
  canonical content SHA-256 为
  `af4fd1d04fcf2572a9eceb1e8396ffda785969d78a1e5de100401da0ffcfd06e`。

盘点结果不是合成成功率：12 个实体中仅 Atorvastatin 为 `candidate_projection_ready`；Cerivastatin、
Mevastatin、Pitavastatin 的新 Workbench 为零路线；Fluvastatin、Rosuvastatin 只有旧
`partial_anchor`；Lovastatin、Pravastatin、Simvastatin 无可迁移路线资产；Crilvastatin、
Dalvastatin、Glenvastatin 只有 blind manifest。目录、HTML 或历史 `accepted` 字段均不能替代当前
路线重放、证据准入和 acceptance。

## 14. 第五切片门禁结果

- 完整离线集：`1796 passed, 3 skipped, 2 subtests passed`；
- Candidate Program、他汀盘点与架构预算聚焦集：`13 passed`；
- Atorvastatin oracle：22 states、11 operations/programs、1 route、11 canonical-admissible，条件
  观察为 0，生产闭合与 acceptance 均为 false；
- 12 个他汀实体恰好一次分类，分类计数与审计摘要一致；
- 合法非 canonical SMILES 规范化回归通过；摘要篡改、无效结构与非法拓扑仍失败关闭。

## 15. 第六块新功能：跨 Workbench Candidate Migration Auditor

新增 `interfaces.candidate_migration` 与 `scripts/audit_candidate_workbenches.py`。审计器读取任意
Workbench 集合，以完整快照摘要去重，再确定性复算 Candidate Program，分类为
`projection_ready`、`empty_graph`、`invalid_snapshot` 或未知 `error`。历史 Workbench 的
`portfolio.accepted`、proof level、来源和条件计数只保留为 `source_diagnostics`；目标名不参与规则。

对当前 `results` 的现场只读扫描结果：

| 项目 | 结果 |
| --- | ---: |
| 解析到的 Workbench 副本 | 303 |
| 内容去重后的唯一快照 / 重复副本 | 264 / 39 |
| 目标标签 | 34 |
| projection-ready / empty / invalid / unknown error | 231 / 18 / 15 / 0 |

审计 canonical SHA-256 为
`78bd574d2a33ccbb2f090b5011b9c160ba72f486005505d059a6ab6e1e0a12fe`。231 是快照数，不是独立
成功合成数；同目标的多个 revision 和测试夹具仍保持可见。Ibrutinib 与 Enzalutamide 各冻结一份
真实观察/投影回归；随后 Fluvastatin current replay 加入，与已有两份 golden replay 和
Bufotalin/Atorvastatin 一起组成 7 类覆盖。
`benchmarks/cross_category_program_regression_set.v1.json` 的 canonical content SHA-256 为
`20e1f0d5d5109ac0df8ba34458d2c5fc03b78e762888c14d96ac4b8a9f42ddb0`；其中 3 类属于
canonical replay/store，另 4 类仍是 Candidate Projection。

## 16. 第七块新功能：Candidate Route 酶机会扫描与零匹配负对照

新增 `application.candidate_innovation_screen` 与 `scripts/screen_candidate_innovations.py`。它先复算
Candidate Program oracle，再构造临时 screening graph 调用通用 `route_innovation_discovery`；不把
Candidate 状态提升为 canonical molecule/edge。只有能力目录非空且存在可枚举窗口时，零匹配才标记
为 `no_applicable_enzyme_capability` 并具备负对照资格。

真实回归的两端结果：

- Bufotalin：1 条 20 步路线、3 条可用能力、5 个酶 proposal；HSDH 候选覆盖 6 条化学边，形成
  1 个酶步骤候选、节省 5 步，边界最低 L1；仍带 `EXACT_SUBSTRATE_UNVALIDATED`；
- Ibrutinib：3 条路线均可枚举窗口、同一 3 条能力全部零匹配，酶候选为 0；三条路线均为有效
  no-applicable-enzyme 负对照；
- 两者都声明 `canonical_graph_not_modified=true`、`program_store_admission_performed=false`，匹配只为
  proposal，不改变 proof、completion 或 acceptance。

冻结 screen 文件的 canonical content SHA-256 分别为
`ddcf6947423e39172f978da8b9bbab31d10d8e46e6fbd5d2e916bdbd85aced37` 和
`4ccb15d22d94134e9de1c7b49375d31a0565241e0e0391e93050caa8802b4cf2`。

## 17. 第六/七切片当前门禁

- Candidate migration、跨类别回归、正向酶超步、零匹配负对照、原 route innovation 与架构预算
  聚焦集：`28 passed`；
- 完整离线集：`1806 passed, 3 skipped, 2 subtests passed`；
- 6 类正向回归与 1 个无适用酶负对照门已达到，但 4 类仍是非权威 Candidate Projection；
- `program_ids[]` 路线字段仍不启用：尚需将候选案卷升级为 current canonical replay，并完成 proof、
  条件、步数和 UI 的双读等价 oracle。

## 18. 第八块新功能：Fluvastatin current replay 与 Route/Program UI 双读

Fluvastatin 不再直接复用旧展示图，而是在独立 runtime、RunIndex、CAS、external-data 与 audit root
中从目标结构重跑。首次以仓库为 blind audit root 时因旧目标资料存在而正确失败关闭；改用隔离审计根
后，旧路线不作为 solver 输入。真实运行使用 proof profile、2 次模型调用、1 次视觉调用和两个 guided
ChemEnzy frontier，得到 revision 14 的 current canonical graph：44 molecules、30 edges、5 个
route-family。Workbench 选择 5 条路线，其中只有 1 条 exploration-complete，最低 proof 为 L1，条件
不完整，configured benchmark stock boundary 虽闭合，configured acceptance 仍为 false。

随后显式执行 shadow Program admission：44 ChemicalState、30 OperationNode、30 Program 与 5 条
route-family 投影全部通过 contract、projection oracle、append-only store replay、RunKernel replay 和
graph/workbench binding；canonical scientific SHA-256 保持
`b4dffd856884b9d6126423ca447b4ff1084b86fdf8c2ffb444a8da4bb02ea65b`，Program projection SHA-256 为
`24fdb289ca3c333b0e482dbd348a842acfcb2300794e7b3f7a61ea4adbc6f1f0`。

新增 `application.route_program_dual_read`，并通过 CLI `program-routes`、Gateway 与
`GET /api/v4/runs/<run_id>/programs/routes` 暴露。它把实际 Workbench `route:*` 与 replacement route
逐边映射为次级 Program id，同时摘要绑定完整 route row 和 proof/条件/acceptance 字段；graph revision、
scientific digest、target state、route identity 或物理步数任一不等即失败关闭。Fluvastatin 实际结果为
5 条展示路线、2 条 replacement route、17 个 edge 引用、11 个不同 Program、0 个物理步数差异，
overlay SHA-256 为
`8c04f2472a42e0f1668b2ac0dd946e458d686eb488335e99639f6a9761adf675`。

冻结回归包括 canonical graph、Workbench、Program admission event 和重放 receipt；目标改名不改变
映射结果，伪造 proof snapshot、graph revision 漂移和模拟 superstep 步数不等均失败关闭。当前聚焦
门禁 `36 passed`，但只证明一个真实 current run 的 UI 双读等价；在多类别与真实 superstep 非等价
问题解决前，`program_ids[]` 仍不得成为生产主语义。

纳入该切片后的当前完整离线门禁为
`1812 passed, 3 skipped, 11 warnings, 2 subtests passed`。第 17 节的 `1806 passed` 保留为上一稳定检查点，
不再是当前权威计数。

## 19. 第九块新功能：跨类别 current replay 的路线双读

Nirmatrelvir 与 Artemisinin 不复制冻结 UI 结果，而是由各自的 model-free replay pack 在独立
runtime、RunIndex 和 CAS 内重建 canonical graph 与 Workbench，再从 `CampaignGateway` 调用同一
Route/Program 双读 oracle。两个案例均有 2 条展示路线、0 个物理步数差异，oracle 全检查通过。

至此双读已覆盖肽样抗病毒药、倍半萷2类天然产物和他汀三个 current replay 类别，
不再存在“未跨类别”门禁；程序主语义仍不切换，剩余阻断是 4 类 Candidate Projection 尚未升级，
以及真实酶 superstep 可能使化学步数与 Program 操作数合理不等。
该切片的聚焦门禁为 `38 passed`，当前完整离线门禁为
`1814 passed, 3 skipped, 11 warnings, 2 subtests passed`。

## 20. 第十块新功能：真实酶 superstep 的 Program draft 与非等价契约

主干审计确认旧 `route_innovation` 路径把酶窗口继续送往 canonical reaction hypothesis，会把“多步
替代程序”错误压扁为一条特殊 reaction edge。该方向已停止：`route_innovation_discovery` 现在只把
文献外一跳机理候选交给 `CanonicalIngestionBatch.hypotheses`，酶窗口只输出 Program draft id。

新增 `application.biocatalytic_programs` 与独立 contracts。编译器从同一 canonical graph、baseline
Program projection 和 route discovery 创建直连连续区间边界的 `TransformationProgram` 候选；被
替代的逐边 Programs 始终作为 fallback 保留。候选路线分别记录物理 Program 数、化学等效步数和
净节省，不再要求 superstep 与 baseline 逐边映射伪等价。内部状态若被区间外路线节点消费，替代会
失败关闭；目标名称不参与候选生成。

专项 `biocatalysis_program_validation.v1` 必须绑定精确 Program、innovation、输入/输出状态和内容
摘要，并携带被接受的实验层级、选择性、结果及必要辅因子闭合。有效验证只使候选达到影子准入/优化
就绪；无论是否验证，都不能把 baseline route 标为完成。Gateway 与
`POST /api/v4/runs/<run_id>/programs/innovations` 只提供确定性审查，不修改 graph revision 或 Program
store。

真实回归仍保持诚实边界：Bufotalin 的 Candidate Projection 是当前 6→1 阳性样本，但没有精确底物
验证；以现有能力目录只读扫描 Fluvastatin current Workbench 的 5 条路线得到零 Program proposal，
因此没有强行制造 current canonical 阳性结论。下一门禁是 durable superstep shadow admission 和一例
经专项验证的 current canonical 真实酶阳性案卷，不是目标名特判。

## 21. 第十切片门禁结果

- 新编译器、旧创新路径隔离、Bufotalin/Ibrutinib 冻结资产、Gateway/HTTP 与架构预算聚焦集：
  `49 passed`；
- 完整离线集：`1822 passed, 3 skipped, 11 warnings, 2 subtests passed`；
- 新增回归覆盖 6→1 区间替代、逐边 fallback、专项验证摘要防伪、内部状态外部消费拒绝、目标改名
  不变、非字典序 edge id 的拓扑保持，以及 API 只读不修改 graph revision；
- 生产不变量继续成立：`edge_ids[]` 仍为权威，baseline route 的 proof、completion、acceptance 与
  Program store 均未被候选审查改变。

## 22. 第十一块新功能：独立酶 Program 影子准入、current canonical 阳性与验证前沿

主干审计确认 baseline `transformation_program_store` 的不可变契约是一条 canonical edge 对应一个
Chemical Program；复用它持久化 6→1 superstep 会混淆投影等价与替代语义。因此新增独立
`biocatalytic_program_store`，没有扩展原 store 的条件分支。只有同时满足以下条件的候选可准入：

- `status=admission_ready` 且 `eligible_for_shadow_admission=true`；
- 专项验证精确绑定 Program、innovation、输入/输出状态、claim 与 condition；
- bundle oracle 全检查通过，候选仍为 `eligible_for_route_completion=false`；
- 调用者显式设置 `enable_biocatalytic_program_admission=true`。

一次事件固定 source graph、source route、baseline projection、discovery、bundle 和 validation pack
六个 CAS 对象，并绑定策略、事件 identity、计数和 authority semantics。事件 append-only、内容寻址、
并发幂等；任何 event、CAS、摘要、route、projection、validation pack 或 oracle 漂移均失败关闭。
GC 在 RunIndex 丢失后直接扫描事件恢复六类 pin。为避免 Windows 长路径失败，内部事件目录缩短为
`.autoplanner/bio_programs/e`，公开 schema 与 API 不变。

这轮同时消除了四处新冗余：graph/route/projection/discovery/bundle 的构建收敛到
`program_innovation_materials`；Program Gateway 方法迁入专用 mixin；恢复和 CLI 分别拆到聚焦模块；
baseline 与 biocatalytic GC pin 在唯一 campaign operation 中合并。持久化顺序调整为先原子发布事件、
再写 RunIndex，避免发布失败留下无事件永久 pin；畸形 CAS payload 统一封装为 store corruption。

Bufotalin 的冻结 HSDH 六边区间已用 fresh current V4 `GlobalCampaignPlan` 重建，不再只在 Candidate
Projection 上观察。同一通用 capability 在 current canonical graph 上发现 6→1 Program proposal，
净节省 5 步。ACS 主文支持甾体 3-keto HSDH 的广泛选择性先例，但当前可获取材料不能证明该精确
输入/输出对；系统因此拒绝持久准入。新增 `biocatalysis_validation_frontier.v1` 自动给出精确状态与
SMILES、Ct3alpha/Ss3beta-HSDH 候选、辅因子/再生矩阵、选择性目标和六类必做检测，同时固定
`grants_validation=false`、`eligible_for_shadow_admission=false` 和
`eligible_for_route_completion=false`。文献获取失败不会被推断替代。

## 23. 第十一切片门禁结果

- Program innovation、current Bufotalin、Gateway、HTTP、CLI、GC、架构预算综合聚焦集：
  `63 passed`；
- 完整离线集：`1829 passed, 3 skipped, 11 warnings, 2 subtests passed`；
- Ruff 全仓检查通过；新增聚焦模块均低于架构行数预算；
- 12 个并发相同准入只发布一个事件；默认关闭、无验证拒绝、有效验证幂等、事件/CAS 篡改拒绝、
  空 RunIndex GC 恢复和 baseline store 隔离均有回归；
- 当前完成门改为：取得 current canonical 精确底物专项验证、把 Program 候选纳入公平 optimizer，
  再执行 Phase 2 路线主语义切换。生产不变量继续是 `edge_ids[]` 权威，proof、completion、acceptance
  与 canonical graph revision 均未被影子准入改变。

## 24. 第十二块新功能：通用 Program 候选契约与只读 Pareto 影子层

主干已有 `portfolio_selection` 的 edge-route Pareto 逻辑，但它同时承担 closure、来源多样性和展示
组合，直接扩展为 Program optimizer 会把旧 route authority 带入新层。此次只抽出 80 行以内、无
领域语义的确定性 dominance/Pareto-layer primitive；edge portfolio 继续保留原闭合与多样性职责，
Program 候选则使用独立 fail-closed 契约。没有复制 Gateway 或新增第二个 HTTP 入口。

`program_route_candidate.v1` 统一记录 source kind、Program/fallback/substitution、执行域、证据、警示、
资格和多轴指标。候选集合始终包含 baseline；未验证创新仍在 exploration 可见，但不能进入 shadow
optimizer。当前编译 adapter 只有 baseline 与 biocatalytic bundle，契约预留 literature、chemical、
whole-cell、hybrid、mechanism，但这些类别尚无生成适配器，因此不宣称已经公平比较所有来源。

`program_route_portfolio.v1` 在 exploration、shadow optimizer、experimental-ready 和 process-ready
四个资格域分别输出全部 Pareto layers。目标函数不包含目标名称或 source kind，也没有加权“最佳
路线”。成功率、纯化次数、原料/辅因子成本和 PMI 缺少可靠数据时明确列为 unmodeled，避免把未知
误当作零。oracle 从候选集合重新计算全部目标、资格、front 和 layers；摘要被重新计算也无法绕过
字段白名单、Program 列表唯一性、整数/有限指标、证据绑定和资格单向关系。

该只读层复用现有 `program-innovations` Gateway/CLI/HTTP 响应。它不写 canonical graph 或任何 store，
不授予 proof/completion/acceptance，`edge_ids[]` 继续是唯一生产路线权威。

## 25. 第十二切片聚焦门禁

- Program compiler/contracts/optimizer、current Bufotalin、Gateway、HTTP、CLI、旧 route portfolio 与
  架构预算聚焦集通过；
- 未验证 3→1 与 current Bufotalin 6→1 候选保持 exploration 可见、shadow 仅含 baseline；有效专项
  验证可使 3→1 替代在多轴上支配较长 fallback，但仍固定 `route_completion=false`；
- 相同指标的 mechanism 对照和 baseline 位于完全相同的 Pareto layers，来源类别不产生隐藏偏置；
- front 篡改、额外隐藏字段、重复 Program id 和越级 process-ready 即使重算摘要也会被 oracle/契约
  拒绝；CLI 端到端覆盖从 materialized route 到同一只读 portfolio 响应；
- 完整离线集：`1834 passed, 3 skipped, 11 warnings, 2 subtests passed`；Ruff、repository audit 与
  `git diff --check` 门禁通过；
- 生产不变量继续成立：没有 `program_ids[]` 主路线、没有 proof/completion/acceptance 变化，也没有
  新增目标专属规则。

## 26. 第十三块新功能：reported Candidate Program 完整路线 adapter

审计发现两条不能混用的路径：`mechanism_one_hop` 是锚点后的待物化新边，并不构成目标到起始物的
完整路线；Candidate Program Projection 则已经保存完整展示路线、逐步 Program、显式 inventory gap
和来源观察。前者若直接进入 route optimizer 会制造假闭合，因此继续保留在 discovery/ingestion
hypothesis；本轮只把后者接入统一 Program route candidate contract。

新增 `reported_program_route_pack.v1` 输入契约，同时固定 Candidate Route observation、可由它精确
重算的 Candidate Program projection 和 route id 选择。adapter 必须依次通过 projection oracle、
canonical target SMILES 对齐、route/program/edge membership 与双摘要校验。目标名称完全不参与。
候选证据现在显式保存 source artifact SHA-256；重算外层摘要不能掩盖内部 projection、字段或目标漂移。

来源分类不再一刀切：有 route-level 或逐步 source ref 的完整候选标为 `literature`；没有任何绑定来源
的旧规划路线标为 `chemical`，并加入 `SOURCE_PROVENANCE_MISSING`。两类都保留在 exploration；由于
仍是历史 Candidate Projection，固定不能进入 shadow optimizer、route completion 或生产 authority。
这既避免“论文全部不可信”，也避免把无来源展示路线涂成论文颜色。

adapter 复用现有 `program-innovations` Gateway/CLI/HTTP review，不新增 endpoint。CLI 使用
`--reported-candidates-json`；HTTP admission 显式拒绝 `reported_candidate_packs`，Gateway 写入方法也
没有该参数，因此 review-only 数据不可能被顺带写入 biocatalytic store。

## 27. 第十三切片门禁结果

- reported adapter、Candidate Projection、创新 discovery、Program optimizer、Gateway、HTTP、CLI、
  原 edge portfolio 与架构预算聚焦集：`93 passed`；
- Bufotalin 20 步真实冻结 Candidate Program route 以 DOI 绑定 `literature` 身份进入 exploration，保留
  20 个 Program、原 warning 和两个源 artifact 摘要，但不能进入 shadow；
- Atorvastatin 11 步冻结投影因没有 route/step source ref 被诚实标为 `chemical` 并显示来源缺失，未被
  冒充为文献路线；target name 改名不影响结果，异目标 pack、重摘要篡改和写入型 admission 均拒绝；
- 完整离线集：`1837 passed, 3 skipped, 11 warnings, 2 subtests passed`；
- 生产不变量继续成立：reported route 和 mechanism one-hop 均不改变 canonical graph、Program store、
  `edge_ids[]`、proof、completion 或 acceptance。

## 28. 第十四块新功能：机理一跳完整路线重拼

上一切片只允许 `mechanism_one_hop` 留在 discovery，因为单独一条新边不能证明目标到起始物仍连通。
本轮没有放宽该门，而是补齐缺少的确定性重拼层。候选前体必须保持为来源锚点边的真实产物；候选
产物必须与同一路线下游某一状态 canonical SMILES 精确一致；两者之间被跨越的边必须构成连续、
单输出的可替换区间，且任何内部状态都不能被区间外路线边消费。零匹配和多重匹配都不生成路线。

满足条件时，`mechanism_program_proposal.v1` 记录已物化输入/输出状态、等效参考区间、operation
blueprint、锚点、机理说明和可证伪检查；`mechanism_restitched_program_route.v1` 用一个 mechanism
Program 替换该区间，同时完整保留逐边 fallback。bundle 和 oracle 绑定 graph revision、scientific
digest、route、Program projection 与 discovery。目标名称不参与匹配，重算外层摘要不能掩盖内部
route candidate 篡改或旧 projection。

重拼仅证明拓扑完整，不验证化学。统一候选中的 source kind 为 `mechanism`，最低 proof 固定为 0，
反应、条件、来源和风险数据缺口至少各 1；`shadow_optimizer=false`、`route_completion=false`。锚点
来源仍可追踪，但明确不表示论文报道了外推反应。不能接回路线的 proposal 继续保留在 discovery，
并在 mechanism bundle 记录 `mechanism_full_route_restitch_missing`。

审计同时清除了未来会扩散的重复：graph/route/projection/discovery 新鲜度校验和连续 Program 区间边界/
替换算法已从酶专用模块抽为共享组件。biocatalytic 路径复用同一组件且原行为回归不变；whole-cell、
hybrid 后续不得再复制第三套闭合算法。

## 29. 第十四切片门禁结果

- 机理一跳正向重拼、不能接回路线、内部/外层摘要篡改、旧 projection、目标名称不变量、统一候选与
  Pareto oracle 均通过；Gateway/HTTP 显式返回 mechanism bundle/oracle；
- Program、reported route、biocatalytic store、CLI/HTTP、Workbench 与架构预算聚焦集：`53 passed`；
- 完整离线集：`1841 passed, 3 skipped, 11 warnings, 2 subtests passed`；
- 生产不变量继续成立：没有 mechanism Program store，没有 canonical graph 写入，没有生产
  `program_ids[]`、proof、completion 或 acceptance 变化。

## 30. 第十五块新功能：whole-cell / hybrid 通用执行 Program

审计先区分了两个同名但不同义的概念：旧代码中的 `hybrid` 主要表示搜索策略组合，不是化学—生物
执行程序；whole-cell 也只有散落的文献提示，没有权威 Program 契约。本轮没有复用或扩展这些旧标签，
而是新增版本化 `program_execution_capability.v1`。能力记录以结构净变化匹配路线窗口，并显式声明
execution domain、organism/enzyme/catalyst actors、顺序 operation blueprints、选择性、辅因子/再生、
载体、先例与专项验证要求。错误 schema、非布尔操作标志、缺 actor、缺 whole-cell preparation 或
缺少化学/生物双域 transform 的伪 hybrid 均在匹配前失败关闭。

共享结构匹配从 `route_innovation_capabilities` 抽到执行域中立的 `route_structure_matching`，whole-cell、
hybrid 与单酶路径只复用结构算术，不共享验证权威。命中的能力通过既有 graph/route/projection/
discovery 新鲜度门和连续 Program 区间替换器，生成显式输入/输出、operation、fallback 与验证计划的
完整 route candidate。hybrid 内部多个分子转换若尚无物化中间状态，会明确显示
`HYBRID_INTERNAL_STATES_UNMATERIALIZED`，不会伪造内部图。

物理操作数包含细胞制备、反应、后处理与分离，因此净操作节省允许为负。候选不会因指标较差被删除，
仍在 exploration 可见；同时固定 specialized validation deficit、proof 0、shadow=false、
route_completion=false。当前没有 execution validation producer、admission event 或 store 路径，
`program_innovation_runtime` 只把 bundle/oracle 送入只读候选编译器。合法但不适用能力返回 oracle 通过的
空 bundle；目标名称不参与匹配。

## 31. 第十五切片结构与入口门禁

- `route_innovation_discovery` 从 433 行降回 380 行预算内，whole-cell/hybrid 发现拆入聚焦模块；
- 共享 graph/route/span 算法没有复制第三套；执行模块、共享匹配层和 adapter 均进入 V4 模块注册与行数门；
- whole-cell 与 hybrid 同一边界可同时保留，负净节省仍进 exploration；非法能力和 bundle 篡改失败关闭；
- Gateway 与 Web API 固定返回 `execution_program_bundle` / `execution_oracle`，空结果也可复算；
- 聚焦主干集 `63 passed`；完整离线集 `1846 passed, 3 skipped, 11 warnings, 2 subtests passed`；
- canonical graph、baseline/biocatalytic store、`edge_ids[]`、proof、completion 和 acceptance 均未改变。

## 32. 第十六块新功能：execution 专项验证前沿与能力反馈

主干此前只生成 whole-cell/hybrid 验证计划文字，无法接收结果；单酶验证又包含 innovation、选择性和
酶辅因子专属语义，直接复用会把两个执行域混为一种。因此先抽出执行域中立的严格 JSON、内容摘要、
Program、输入/输出状态、claim 与 condition 绑定检查，再分别保留生物催化和 execution 领域门。

新增 `execution_program_validation.v1` 精确绑定 capability ID/摘要、execution domain、完整 operation
sequence、required checks、actor identity、claim、condition、台账和 outcome metrics。成功、失败与
不确定记录均可成为有效实验观察；只有成功、全部必检项为真且所需台账闭合时才开放只读 shadow。
失败不会导致候选或能力被删除，不确定也不会被静默丢弃。

`execution_validation_frontier.v1` 为每个尚未通过的 Program 生成精确边界结构、actor/operation matrix、
辅因子/载体台账、必检项与严格输出契约。`capability_feedback_projection.v1` 把有效结果分为 positive、
negative、inconclusive，适用范围固定为 `exact_boundary_only`；反馈行显式声明不修改或禁用能力目录，
oracle 可从 discovery、bundle 与原始 validations 完整复算。Gateway/Web 沿用现有 innovation review
入口并固定返回 frontier、feedback 和 oracle，没有新增写入口。

## 33. 第十六切片清理与门禁结果

- 共享 Program validation binding 和 validation-frontier freshness/state checks 已抽出，生物催化原行为
  回归不变；两种领域仍保留独立接受语义；
- validation ID 以整个输入批次为唯一域，跨 Program 重复、缺检查项、错误操作摘要、未绑定记录及投影
  篡改均失败关闭；合法失败仍在 exploration 可见；
- `execution_programs.py` 从 460 行降到 380 行，候选审计/操作边界绑定拆入 73 行纯函数模块；
  `program_innovation_materials.py` 从 128 行降到 96 行，机制/execution/candidate/optimizer 组合进入两个
  聚焦只读 facade；所有新模块均加入 V4 依赖和行数门禁；
- 完整离线集：`1850 passed, 3 skipped, 11 warnings, 2 subtests passed`；Ruff 与架构门禁通过；
- execution 仍没有 admission event/store；canonical graph、baseline/biocatalytic store、`edge_ids[]`、
  proof、completion 和 acceptance 均未改变。

## 34. 第十七块新功能：mechanism 专项验证与反馈

机理一跳此前只能完成结构边界重拼，无法区分“新反应尚未验证”“净转化已观察”和“所提机理受到
实验支持”。本切片新增 `mechanism_program_validation.v1`，精确绑定 Program、innovation、输入/输出
状态、机理签名、由 proposal 可证伪项派生的必检项、claim、condition、analytical record 和 outcome
metrics。成功、失败、不确定都是可保留观察；只有合法成功且全部必检项通过时才开放只读 shadow。

`mechanism_validation_frontier.v1` 为每个未通过候选生成严格实验输出契约；
`mechanism_feedback_projection.v1` 将结果投影为 positive、negative、inconclusive，并明确锚点来源没有
报道外推反应。`net_transform_observed`、`mechanism_consistent` 和 `mechanism_discriminated` 分级保留，
避免把净转化成立偷换为基元机理已证明。任何结果都不创建 canonical reaction proof、Program store、
completion 或 acceptance。

## 35. 第十七切片清理与门禁结果

- validation schema 由显式中立路由分成 biocatalytic、execution、mechanism 三组，未知旧 schema 仍沿
  原兼容路径处理；
- Program/边界绑定、frontier 新鲜度和反馈三态收集均使用公共严格契约，领域接受语义保持独立；
- 机制候选合法性和 SMILES 物化检查从 `mechanism_programs.py` 拆入 67 行聚焦模块，主编译器从 421 行
  降到 370 行；公共候选不复制权威快照，只以摘要和可展示警示保留机理支持状态；
- success / failure / inconclusive 同批回放、缺检查项、错误机理签名、解释不兼容、缺分析记录、未绑定
  Program、重复 validation id 与反馈篡改均已覆盖失败关闭测试；
- 完整离线集：`1853 passed, 3 skipped, 11 warnings, 2 subtests passed`；Ruff 与架构门禁通过；
- canonical graph、baseline/biocatalytic store、`edge_ids[]`、proof、completion 和 acceptance 均未改变。

## 36. 第十八块新功能：统一实验 Claim、精确边界校准与独立持久化

三种验证域此前各自返回结果，但没有一个中立层能同时保留成功、失败和不确定观察；直接把这些结果写入
`fact_lifecycle` 会混淆“既有 canonical 事实的失效控制”和“新实验观察”，扩展 biocatalytic Program
store 又会把候选准入与一般实验结果混为一谈。本轮因此新增独立的
`experimental_observation_claim_set.v1`：它只接收已经通过各领域结构契约的观察，精确绑定 domain、
Program、subject、输入/输出状态、原验证摘要、条件和 outcome，不产生 canonical reaction proof，也不
继承机理锚点文献的外推权威。execution 与 mechanism 的 positive、negative、inconclusive 全部保留；
biocatalysis v1 只能诚实表达已接受的正向精确底物观察，不伪造负向语义。

`capability_applicability_calibration.v1` 仅在 domain、subject、boundary、signature 完全一致时聚合，输出
positive、negative、conflicting 或 inconclusive-only，并比较前后投影产生 created/changed/removed
dirty-domain 提示。该提示只建议精确域重算，不修改、禁用或全局降权能力目录；机理假设也不会因此变成
通用 capability。

只读 review 不自动持久化。新增的 `experimental_claim_store` 只有在显式
`enable_experimental_claim_admission=true` 且 Claim set 非空时才追加内容寻址事件。事件固定 graph、route、
Program projection、discovery、按稳定顺序封装的 validation pack 与 Claim set 六类 CAS；每次读取都从前五项
重新编译三个领域的 bundle/feedback/oracle 和 Claim set，再逐字节比较存储对象。只含失败或不确定结果的
非空集合与成功结果具有同等持久化资格，但都不能授予 Program admission、proof、completion、acceptance
或 catalog mutation。

## 37. 第十八切片冗余清理与门禁

- baseline 与 biocatalytic store 重复的事件枚举、重复身份检测、原子发布及发布后重放已抽为
  `runtime.immutable_event_store`；两套原存储行为回归不变；
- baseline、biocatalytic 与 Claim store 的 RunIndex 丢失后 GC 扫描共用 `interfaces.replay_store_gc`，不再
  复制第三套目录发现、run spec 读取、事件重放和 ref 收集逻辑；
- Claim 持久化使用独立 Gateway/CLI/HTTP 写入口；普通创新 review 保持只读，不会顺带产生事件；
- 默认禁写、空集合、run 身份不一致、验证输入顺序、重复与并发准入、事件/CAS 篡改、完整源重投影、
  RunIndex 丢失后的 GC pin 恢复均已覆盖；
- 完整离线集：`1859 passed, 3 skipped, 11 warnings, 2 subtests passed`；Ruff、repository audit、V4
  架构依赖/行数预算和 `git diff --check` 均通过；
- canonical graph、baseline/biocatalytic Program store、`edge_ids[]`、reaction proof、completion、acceptance
  与能力目录均未改变。

## 38. 第十九块新功能：单一前沿绑定的实验工作与执行器中立 I/O

本轮先审计了 `deficit_frontier.py`、`frontier_runtime.py`、`worker_runtime.py` 和三类专项验证前沿。
结论是 `deficit_frontier.v1` 必须继续作为唯一 canonical 下一工作投影，冻结的
`frontier_scheduler.py` 不能复用；通用 `WorkerResult` 虽可复用内容寻址执行回执思想，但实验结果不能
直接进入 `CanonicalIngestionBatch`，否则会把观测误升格为反应图事实。

因此新增 `experimental_work_frontier.v1`：它绑定当前 canonical frontier 摘要，把生物催化、execution、
mechanism 三域 validation plan 展开为只读 subtask，并用各 plan 的 replaced-edge span 关联已有 validation
deficit。没有可关联 deficit 时明确标为 route-scoped shadow work，不伪造父任务。calibration dirty hint 仅在
domain、subject 和 exact boundary 同时相等时映射，动作固定为重算 route Program review；投影禁止发布到
`RunKernel`，也没有自己的 reservation、attempt、budget 或 completion counter。

`experiment_execution_request.v1` 固定 run/route、work item、当前 frontier、源 plan、精确状态、必检项、
输出 schema 和资源提示。`experiment_execution_result.v1` 绑定 request 摘要、executor 版本、按稳定顺序排列的
原始工件 SHA-256、四态 outcome 和一个领域验证候选。Gateway/CLI/HTTP 审计会先重建当前工作投影；旧 request、
自造 request、边界/摘要/工件不匹配均失败关闭。通过只表示候选可再次提交给既有领域 gate，不自动创建
validation、Claim、reaction proof、Program admission、completion、acceptance 或 catalog mutation。

回归覆盖生物催化成功、中止、request 篡改、execution 负向观察、dirty subject 精确映射，以及 Gateway、
CLI、HTTP 三入口。完整离线集为 `1861 passed, 3 skipped, 11 warnings, 2 subtests passed`；Ruff 与 V4
架构门禁通过。截至该切片仍没有真实实验 provider/设备 dispatch，也没有信息增益排序或跨相似边界泛化。

为避免 provider 接入继续膨胀单文件，request 构建/校验与 result 构建/审计/释放已分别收敛到
`experiment_execution_contracts.py`（226 行）和 `experiment_execution_results.py`（258 行）；二者各有独立
架构行数预算，orchestration 只负责 current-frontier 重投影，不复制契约判断。

## 39. 第二十块新功能：配置驱动 executor 与 RunKernel 单账本派发/恢复

接入 executor 前先审计了 provider SPI、`RunKernel` 事件链、CAS 与三类实验工作投影。结论是现有
`RunKernel` 已经拥有 reservation、预算、幂等、结算和恢复所需的唯一操作账本；为实验另建 queue、job store
或 completion counter 只会制造第二事实所有者。因此本轮没有增加实验队列，而是把每个当前
`experiment_execution_request.v1` 映射为一个稳定的 `validation` task，accepted expansion 固定为空。

新增宿主信任的 `ExperimentExecutorProvider` SPI、严格的 `experiment_executor_policy.v1` 与确定性选择结果。
provider 只能由宿主 registry 预注册并授予 trust record；请求 policy 只能通过 allowlist、domain、网络与成本
上限收窄选择，不能从 payload 注册 provider、修改 provider kind 或提升信任。当前内置的
`autoplanner.manual_experiment_executor` 无网络、零成本、确定性且声明幂等，只生成
`awaiting_external_result` 人工交接，不代表真实设备已经完成实验，也不授予 validation、Claim、route 或
catalog 权威。

派发会先重建唯一 current frontier 与 current request，再把 request、精确 provider selection、handoff、原始
artifact、result、domain review 和 settlement 全部内容寻址。reservation metadata 固定 selection 摘要，因而
并发调用和显式恢复能复用完全相同的不可变收据；pointer 只是可重建索引，删除后可从 RunKernel 生命周期与
CAS 恢复。结算前强制检查 reservation 中的 provider/version、CAS artifact 存在性、当前 frontier 与领域结果
审计；成功也只释放既有 domain gate 的验证候选，不直接写 validation、Claim、canonical graph、proof、
completion、acceptance、Program store 或 capability catalog。

清理同时修复了两处被并发回归暴露的 Windows 稳定性问题：ArtifactStore 对 `\\?\` 扩展路径与普通路径先做
等价 realpath 归一化，再执行共同根逃逸检查；`RunKernel` 的 snapshot/event 读取进入同一 writer lock，锁目录
创建/删除期间的瞬态 `Access denied` 进行有界重试，而真实 ACL 权限失败仍原样抛出。连续 10 轮并发
reservation、dispatch、settlement 回归均通过，完整离线集为
`1875 passed, 3 skipped, 11 warnings, 2 subtests passed`；Ruff、V4 架构门禁与 repository audit 通过。

下一步仍是在同一 SPI 上接入受控的真实设备或网络 provider，并补外部 job receipt、超时、取消、操作者身份、
信息增益/成本排序和跨相似边界 applicability 学习；这些能力不得绕开唯一 DeficitFrontier、RunKernel 单账本
或现有领域验证门。

## 40. 第二十一块主干修复：`unresolved` 分层、可恢复终态与共享后缀长路线

10 个剩余 statin 的隔离 blind rerun 不是只做展示汇总，而是暴露了三个主干级失败源。第一，Codex JSONL
可能先记录若干超时/重连错误，随后仍返回合法结构化结果并以 `turn.completed` 结束；旧 worker 只看到中间
错误或非零 CLI 退出码就丢弃方案。解析器现在记录 terminal event 顺序，只有未被后续完成事件恢复的错误才是
fatal，完成后的错误仍失败关闭。Crilvastatin 在相同输入上由 0 路线恢复为 4 条 target-rooted 路线、3 条
host reaction validated，证明该类 `unresolved` 原本是状态归并缺陷而非化学空解。

第二，旧 `unresolved` 同时表示无路线、已有假设但 host validation 未过、已有验证路线但 exact proof 未闭合。
当前 disposition 与展示将其拆为真实无路线、低可信路线待验证、路线已验证但 proof open、配置边界已闭合；
来源类型只决定颜色，低可信警示不删除文献报告或模型候选，也不提升其 scientific authority。统一 statin 面板
收录 10/10 完成报告和 50 条 target-rooted 路线，B1=10、B2=3、B3=0、B4=2、B5=2；所有 Workbench
和源报告均存在，反应节点可打开检查器显示条件、缺失项与来源哈希。

第三，proof profile 原有 `max_steps_per_skeleton=8`，使 20+ 步路线在结构上不可能进入主机。上限现为 24，
Director 明确要求在化学需要时展开 20+ 个单反应步骤，不得把多步化学伪装成一步；真正 one-pot、whole-cell
或 biocatalytic program 仍可作为显式程序候选。首次长深度试跑已经生成一个 12 步 de novo Simvastatin
skeleton，但又发现 host 用 target-forming precursor set 给路线族去重，因它与短路线共享最终反应而拒绝整族。
去重现按上游 canonical reaction program 判定：共享 target edge 或下游后缀合法，只有纯改名、截短且无上游
分歧、或上游程序完全相同的家族被拒绝。该规则不依赖目标名称，并保持 hypergraph 共享骨架语义。

定向 Director 回归为 `23 passed`；完整离线集为
`1889 passed, 3 skipped, 11 warnings, 2 subtests passed`，Ruff 通过。修复后 fresh blind run 仍须以最终
materialized Workbench 检查长路线是否被保留；即使保留，也不能越过逐边 reaction/evidence/condition/stock
门直接宣布文献闭合或工艺就绪。

## 41. 第二十二块主干修复：长规划骨架的 L0 保留与 RDKit 执行边界

后续 Simvastatin proof fresh run 给出了更精确的边界：主机接收了一个 12 步 target-rooted Director
skeleton，第二轮 replan 也再次生成并通过了 12 步拓扑审计；最终 canonical graph 含 22 条边，run 达到
`reaction_validated` 并按配置策略验收。但最终 Workbench 的已选路线只有 1/2/2/5/1 步。追踪发现不是
portfolio 简单偏爱短路线，而是同一 12 步骨架有 3 步省略了水、卤化源或其他参与物：Director 的
身份/连通门允许它们作为 proposal，canonical admission 的元素库存门正确拒绝了它们，旧投影却把这些
拒绝步骤连同完整 skeleton 关系一起丢失，长路线因此在 UI 中被截短。

修复没有放松 canonical edge 准入。结构身份仍有效但 admission 未通过的步骤现在以
`admission_rejected` 状态保留为 L0 planning fact，记录原始 `skeleton_id`、`proposal_id`、转化假设和
拒绝原因；它们不生成 materialization work，不进入 edge proof，也不能提高 B2/B3、route completion 或
acceptance。Workbench 只读投影按 skeleton 重新组合已物化步骤与 L0 步骤，整条规划路线继续可见，拒绝
步骤使用红色警示；canonical proof portfolio 路线计数保持不变。这一语义同时覆盖“文献路线缺一条可校准
边”和“模型长路线省略参与物”，而不依赖目标名称或特定反应规则。

RDKit 的边界也已明确：主机主干直接导入并使用 RDKit 做 canonical SMILES、立体化学、原子映射、反应中心
与准入审计，从未被禁用。被禁止的是 blind Director 子进程中的任意 shell / 本地 Python，因为它可能读取
工作区内的旧路线、PDF、benchmark 或缓存，破坏 target-name-and-SMILES-only 的盲测。若子进程尝试的
shell 在 Windows sandbox 创建进程前即被拒绝，随后仍返回合法结构化结果并以 `turn.completed` 结束，worker
现在只把这类可证明“从未执行”的尝试视为非致命；任何已经启动、部分输出或无法证明未执行的禁用工具仍
失败关闭。化学验证始终由宿主 RDKit 完成，不需要也不允许 Director 自行用 shell 复核。

## 42. 第二十三块主干修复：Director 契约恢复与长路线展示闭环

下一次 fresh run 暴露了另一类与化学无关的 `unresolved`：Director 声明了四个 route family，却只为其中
三个给出 skeleton。旧契约验证会因孤立 family metadata 拒绝整份计划，而事件重规划又只在首份计划已通过
时运行，导致一个可局部修复的输出终止整个 run。host 现在会确定性删除没有任何 skeleton 的孤立 family
元数据并记录 `route_family_without_skeleton_removed`；若计划仍因 `GlobalCampaignPlanValidationError` 失败，
则生成 `director_contract_rejected` 物质事件，允许在相同硬预算内执行一次有界 replan。该恢复只删除无化学
内容的元数据，不补造 reaction、molecule 或 route，也不放宽 canonical admission。Director 契约同时明确
要求每个声明的 family 至少包含一个 skeleton，尚未展开的 family 必须省略。

修复后的 Simvastatin V6 隔离 fresh run 完成到 `reaction_validated`，19/19 次扩展被 host 接收并触发一次
replan；配置边界 B0/B1/B2/B4/B5 通过，B3 exact-source 仍为 false。初始 canonical Workbench 中真实存在一条
12-edge、12-physical-step、target-rooted 的 L1 路线；最终 proof portfolio 选择了 5 条更短且 stock-closed 的
L2 路线，而 `planned_routes` 仍保留两条 12 步、全部已 materialized 的完整 skeleton。这证明 12 步路线既非
截图拼接，也未被短路线选择器从展示事实中抹除；同时它没有 exact procedure、逐步文献或完整条件，因此只能
作为规划路线展示，不能宣称文献闭合、条件完整或 process-ready。

独立展示包现同时给出“主路线 / 规划路线”和“最长展示路线”口径；Simvastatin 卡片显示 5/6 与 12 步，进入
Workbench 后可实际选择 12-step planner route，并保留来源颜色、L0 红色拒绝标识和只读警示。展示统计不再
用 proof portfolio 的最短入选路线覆盖完整规划深度。

收束时没有提高集成外观行数上限。`route_workbench.py` 的规划路线与 molecule/edge row 分别拆入两个聚焦
只读模块，主文件收敛到 725 行；`v4_route_workbench.py` 的 planner branch 与共享 graph row helper 拆出后
收敛到 792 行，继续低于既有 900/800 门禁。新模块各自加入 V4 import 审计和 100–400 行预算。完整离线集为
`1894 passed, 3 skipped, 11 warnings, 2 subtests passed`；Ruff 与 153 项 Workbench/route-forest/架构聚焦回归
均通过。

## 43. 第二十四块主干校正：闭合优先而非步数优先

Bufotalin 的 fresh target-only 运行证明，“能够承载 20+ 步”不能实现成“每条目标都必须至少 20 步”。V2
在深度硬阈值下返回 12 步主骨架和 8 步 extension，并在摘要中称为 20 步；host 拒绝 extension，因为它没有
以 campaign target 为根且内部存在祖先环。主干没有放松这个检查，也没有为了演示删除循环边。通用安全拼接
只允许同一路线族、extension 唯一根精确命中主骨架未展开叶、合并后不超深度且完整 DAG 再验证通过的纯结构
分段；V2 不满足这些条件，所以继续保持红色拒绝。

用户进一步确认完成标准是“闭合即可，不以更长为优”。因此 Bufotalin manifest 的最小深度恢复为默认 0；
24 步仍只是 proof profile 容量，Director 只在化学真实需要时展开长路线。为避免“结构闭合”和“证明/采购
闭合”继续混用，新只读投影 `declared_route_program_closure.v1` 按不可变 Director origin + family + skeleton
分组，只有每个声明 proposal 都对应 canonical edge 才标记 `declared_route_graph_closed`。缺步仍完整列出
step id、状态与 admission reason，路线长度显式声明不参与优化。

Bufotalin V3 最终读数为 4 个声明程序中 3 个结构闭合，分别 12、6、4 步；2 步生物转化程序有 1 个元素库存
守恒缺口。canonical portfolio 仍独立显示 4 条路线、18 条边、4 条 L2 反应边、5/8 benchmark 叶命中；精确
文献、条件和采购均未闭合，所以科学 claim 仍为 `reaction_validated` 且策略 acceptance 为 false。Showcase
不再把它归为“无目标根路线”，而显示“3 条结构闭合 · 证据/库存开放”。聚焦回归为 128 项闭合/Workbench/
Route Forest/架构用例和 127 项 Showcase/Workbench/Delivery 用例全部通过；浏览器实际点击 12 步路线与 R1
检查器，无 JavaScript error，反应卡未被遮罩且右侧条件/来源/Proof vector 可完整滚动。
