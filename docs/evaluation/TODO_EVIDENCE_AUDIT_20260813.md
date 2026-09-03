# TODO 剩余项证据审计

日期：2026-08-13

状态：持续维护。本文只按当前工作树、冻结回执、机器结果和直接测试分类，不用“未发现反例”替代完成证明。

## 1. 审计原则

历史文件 `docs/history/AUTOPLANNER_V4_TODO_20260815.md` 同时包含当时的施工项、普遍不变量和未来 publication-scale 门。每个未完成项按以下类别处理：

- `current-gap`：当前实现或直接覆盖仍缺失，应继续施工；
- `running-evaluation`：实现已冻结，等待不可变评测结果；
- `bounded-evidence`：已有有界证据，但不足以证明“任何请求”或“全部 190”之类普遍量词；
- `future-publication-gate`：明确要求 190×4，超出用户已确定的 20/190 停止范围；
- `not-reconstructable`：要求历史时点证据，但当时未冻结，不能事后伪造。

## 2. 当前最高优先级

### 2.1 SynthAtlas clean-20 四臂

分类：`running-evaluation`。

- 冻结 protocol 的四臂是 external snapshot、Codex-only、ChemEnzy-only、unified-adaptive；只有后三臂需要 live 执行。
- external snapshot 已完成 20/20；route-level C0 为 58/59、C1 为 40/59，C2–C6 不从外部声明继承。
- v3 因并发任务 elapsed 被错误相加成 wall time 而失效，原始结果只读保留，不进入正式分母。
- v4 receipt：`cb65eb18c8968c12158abd65988084569a7457a4c6d089c6a6f789e751dcf0b2`。
- v4 source bundle：817 files，SHA-256 `9dd24cfa41d101842505b9a3bb8424c9275fa9a9fcb1391f16468c2f531fc254`。
- 独立 unified target 001 烟测：fixed-cutoff projection 可用；wall=962.303094 s，compute=2436.119782 s；B1/B4 达成，B2/B3/B5 未达。
- 三条 live arm 正式运行已启动；严格汇总前必须验证 source bundle、receipt、case 顺序、20/20 completion、失败分母与每个 projection digest。

完成门：`scripts/summarize_strategy_closure_pilot.py` 生成 digest-valid paired summary，20 个目标在四臂中全部可审计，失败/timeout/partial 不被剔除，并回填资源表与 failure taxonomy。

### 2.2 已完成但旧审计尚未同步的能力

以下事项已有直接实现和测试，主 TODO 已同步勾选：

- 产品 milestone 订阅：实现 durable outbox、`notify-only` / `notify-and-cancel`、幂等 acknowledgement、失败重试与显式 RunKernel cancel；B4 仍是外部产品策略，不是 solver early return。
- RunKernel 时间账本：`task_wall_time_s` 使用已结算任务区间并集，`task_compute_time_s` 保留逐任务 elapsed 总和；并发 4/5/5 秒回归为 wall=5、compute=14，追加串行 3 秒后为 8/17。
- Pareto 路线保留：八轴分别覆盖 topology、stock、reaction、evidence、conditions、diversity、cost/length、Program；scalar utility 只负责显示；同叶 strict edge-superset 不再被永久删除。
- Program 泛化：Director 扫描连续多步区间，考虑 biocatalytic step/superstep、whole-cell、hybrid 和一跳 mechanism extrapolation，并强制绑定 replaced steps、区间边界、conventional fallback 与 specialized host validation。
- 完整离线门：2892 passed、3 skipped、2 subtests passed；全量 Ruff 与 `git diff --check` 通过。

## 3. 仍需通用能力施工的项目

### 3.1 ChemEnzy 普遍 raw parity（TODO 208/382）

分类：`bounded-evidence`。

固定请求已有严格 parity：相同路线 fingerprint、search trace 和 raw proposal digest；manifest 前 3 个目标中 2 个非空严格通过，1 个两臂均确定性空结果。该证据不能外推到任意 target/seed/model/stock 请求。

下一门：冻结分层请求 panel，覆盖不同目标复杂度、非空/空结果、至少两个 seed 与冷/热启动；只比较内容绑定后的 proposal digest，不把运行时回执噪声混入科学身份。

### 3.2 standalone B4 到 unified 的单调保留（TODO 209/403/405）

分类：`bounded-evidence`。

现有证据包括 Nirmatrelvir 39/39 lineage、三条 synthetic native route 的 materialize/validate/stock audit，以及 seed/guided pool 累加测试。尚不足以证明任何 standalone-host-audited B4 路线在 unified 中都可追踪，也不足以证明所有同预算 unified native-search expansion 均不低于 baseline。

下一门：使用真实分层 panel 成对运行 standalone-host-audited 与 unified；逐路线输出 raw → normalized → admitted → canonical edge → B2 → B4 的首损边界，并区分 provider failure 与 embedding loss。

### 3.3 单一 canonical ingestion 边界（TODO 366，已关闭）

分类：`completed-with-direct-evidence`。

- 全包 AST 门证明只有 `RetrosynthesisCampaignService` 构造 `CanonicalHypergraphStore`；只有受审计的 execution/replay 模块调用低层 `graph_store.apply()`。
- 新门精确冻结全部 `CanonicalIngestionBatch` 构造点、`apply_batch` 与 `apply_global_plan` 调用点；新增生产入口未显式审计即失败。
- canonical origin allowlist 覆盖 Codex、ChemEnzy、template/self-evo template、literature/source/visual、external strategy、manual、biocatalysis 和 mechanism hypothesis。
- 行为回归覆盖同边 ChemEnzy/template/manual 去重并保留来源、外部 strategy、文献 source route、专利模板及 Program mechanism candidate；6/6 通过，架构门 3/3 通过。

该完成项证明所有当前生产候选共享一个 canonical 写入边界；它不替代 TODO 209/403 的跨 standalone/unified 路线单调保留普遍量词。

### 3.4 单一生产控制流与 legacy 边界（TODO 575，已关闭）

分类：`completed-with-direct-evidence`。

- 主入口 `python -m cascade_planner` 只注册 canonical CLI；CLI/API/Web 均经 CampaignGateway 与 unified runtime。
- 全活动包只存在 `application.action_scheduler.schedule_next_action()` 与一个 `CampaignActionRuntime` 构造点；stage projector 不允许重新 dispatch。
- 活动包不反向导入 `cascade_planner.legacy` / `.research` state owners，V4 Workbench 不执行旧 RouteForest。
- 根脚本中唯一直接导入 legacy acceptance/portfolio 的 `compile_source_route_portfolio.py` 已物理迁入 `scripts/legacy/`；唯一 V3 golden 调用同步更新，归档工具 `--help` 和 golden tests 均通过。
- 新 AST 门禁止 `scripts/*.py` 根入口导入 legacy 控制流；ChemEnzy 子进程仅允许导入 `legacy.guard` 做显式能力隔离。
- 完整 `tests/test_v4_architecture.py` 为 14/14 passed。保留的 Program shadow admission store 和 one-step sidecar 不拥有 canonical write authority，是安全隔离层而非第二控制流。

## 4. Retro*-190 范围

用户已明确将范围限定为 manifest 顺序前 20/190 × 四臂；21–190 不自动启动。

现有冻结 pilot 已完成 80/80：ChemEnzy-only B4=16/20、Codex-only=1/20、unified round-robin=15/20、unified adaptive=15/20。该结果支持有界工程结论，不支持 190-target population claim。

因此：

- TODO 212–213、406、519–523、553、623–625 中明确写“全部 190”的项目保持 `future-publication-gate`；
- TODO 514–515、524、528–542、547–552 可在 20-target pilot 上生成有界版本，但不能把复选框原文的全量量词改成已完成；
- 当前 clean-20 完成后，先比较两套 20-target 证据并决定通用修复，不因“继续 TODO”直接重跑 21–190。

## 5. 不可事后伪造的历史项

- TODO 219 要求“本轮开始时”的 branch、HEAD、tracked/untracked 完整快照；如果开始时没有生成内容寻址回执，当前 `git status` 不能替代历史事实，分类为 `not-reconstructable`。
- TODO 220 的变更边界可由当前 diff 与 source bundle 部分证明，但不能反向证明整个长期施工过程中从未覆盖用户旧改动。
- 后续每次正式冻结都必须在运行前自动记录 commit、tree、dirty summary、tracked modifications/deletions、untracked manifest 和关键文件摘要，避免再次出现历史证据空洞。

## 6. 当前执行顺序

1. 不改冻结源码，完成 clean-20 v4 三条 live arm；
2. 生成 paired summary、资源账本、failure taxonomy 和 SynthEx 竞争结论；
3. 依据全 20 首损边界选择通用能力修复，不对测试目标逐个调参；
4. 扩展真实 ChemEnzy parity/monotonic-retention 分层 panel；
5. 用直接证据继续回填主 TODO；全 190 publication gate 保持显式延期。
