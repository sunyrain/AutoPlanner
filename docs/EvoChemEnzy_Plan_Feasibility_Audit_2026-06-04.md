# EvoChemEnzy Plan Feasibility Audit

日期：2026-06-04

审查对象：

- `docs/README.md`
- `docs/EvoChemEnzy_Agentic_CASP_Plan_2026-06-03.md`
- `docs/EvoChemEnzy_Code_Delivery_Checklist_2026-06-03.md`
- `docs/SMILES_FIRST_LITERATURE_STRATEGIC_WORKFLOW_2026-06-03.md`

整合范围：

```text
本审计同时吸收 docs/ 顶层 2026-05/2026-06 临时报告的结论。
这些临时报告已经归档；后续执行以 active docs 为准。
```

当前实施优先级：

```text
P0 = SMILES-first literature strategic workflow。
核心任务 = Codex 文献检索 + 高级中间体/战略断键生成 + route package validation。
完整 Case Blackboard / Controller / RouteStatus / guided rerun / compiled judge
进入 P1+，除最小 guardrail 外不作为当前开工阻塞项。
```

## 总体判断

计划方向是可实现的，且 active docs 的核心原则一致：

- ChemEnzy 保持 step-level inner engine；
- Codex/LLM 只做 episode-level 诊断、文献调研、策略草案和审计；
- 文献、断键、条件、策略必须先变成 typed artifact；
- validator / compiled judge / route auditor 决定真实性；
- target run 不能直接污染 production KB。

但目前计划还不是一个完整可直接执行的工程实施方案。它更接近：

```text
architecture intent + comprehensive checklist + Bufotalin workflow runbook
```

下一步需要补齐：

```text
minimal MVP scope + concrete schemas + compiler interfaces + benchmark packs + acceptance thresholds
```

## 可实现性评级

| 层级 | 当前可实现性 | 判断 |
| --- | --- | --- |
| P0 SMILES-first literature package | 中高 | Bufotalin 手工流程已证明可行，需抽象成 CLI、证据卡、候选生成和 validation |
| P0 route-package guardrail | 高 | 现有 proposal gate、product audit、route rendering、Bufotalin 经验可复用 |
| P1 trace / RouteStatus / guided rerun | 中 | 需要 case trace、RouteStatus、policy compiler 和 bounded rerun 接口 |
| P2 condition feasibility | 中 | 现有 condition prediction 可支撑原型，但 exact/analog/model source status 需统一 |
| P3 evidence-gated evolution | 中低 | 需要 candidate/shadow/staging/production KB 和 benchmark gate，当前只是设计 |
| fully automated literature-to-policy loop | 中低 | 文献访问、SI 提取、结构实例化和证据校验仍需大量 guardrail；不应作为 P0 |

## 已整合的阶段性证据

旧临时报告支撑了三个正向判断和三个限制。

正向判断：

- 单步 proposal coverage 确实可以被补强：`benchmark_v2_100` top16 union 从
  native 的 `17/100` 提升到 native + AiZ + RetroKNN 的 `33/100`。
- Bridge weak-label 数据可以训练出有效 gate：bridge verifier v0 test
  PR-AUC `0.9984`、precision `0.9809`、recall `0.9908`；precision-biased
  deployment threshold 为 `0.8410`。
- 酶步三元组 gate 已有可运行原型：enzyme SP-v1 test PR-AUC `0.9940`、
  precision `0.9443`、recall `0.9858`，能在受控 benchmark 中拒绝不可信酶步。

限制：

- Naive ensemble 会污染多步搜索：5-target smoke 中全节点打开 AiZ/RetroKNN 后
  平均搜索时间从 `0.0278s` 膨胀到 `12.97s` 或 `27.57s`，route-pool exact hit
  没有提升。
- P5 bridge evidence 是 diagnostic/partial evidence：14 条 evidence cards、
  0 条 stock-closed cards、0 条 route-solved cards，不能作为生产级路线声明。
- ChemEnzy 的 post-hoc EC assignment / condition prediction / active-site annotation
  不能等同于 search-time enzyme validation 或可执行工艺条件。

## 关键缺口

### 1. P0 必须聚焦 SMILES-first 文献战略断键

最新优先级已经调整：P0 的核心不是先完成 Case Blackboard、Controller、Worker、
guided adapter、Compiled Judge、Route Auditor、RouteStatus 和 Trace Store，而是先把
`SMILES-first literature strategic workflow` 变成可运行 pipeline。

风险：

- 如果继续把 P0 做成完整 agent 框架，第一轮实现会过大，难以验收；
- 如果先做 audit-only，会推迟项目最核心的“Codex 文献检索 + 高级中间体拆解生成”能力；
- 如果没有最小 guardrail，文献 surrogate 和 route anchor 又容易被误报为 solved route。

建议拆成：

```text
P0a: target SMILES profile + baseline/frontier extraction
P0b: Codex literature retrieval + evidence cards
P0c: strategic intermediate/disconnection candidates + hybrid route package
P0 guardrail: route package audit, no solved claim without stock/audit proof
P1a: RouteStatus + case trace
P1b: StrategicOperator compiler + bounded guided rerun
```

### 2. Artifact schema 没有完全落地

Checklist 已列出 artifact 类型和统一字段要求，但还没有具体字段、枚举、状态转换和
JSON schema。SMILES-first workflow 也定义了候选 JSON，但字段和 checklist 不完全对齐。

明显不一致：

- workflow 使用 `direction: retro | forward | route_anchor`；
- integration/checklist 倾向使用 `candidate_kind: exact_fragment_retro | forward_surrogate | route_anchor`；
- workflow 的候选记录没有强制 `artifact_id`、`case_id`、`validation_status`；
- `forward_surrogate` 还没有强制 `not_lab_procedure` 和 `surrogate_reason`。

建议先落地 P0 所需的最小 schema：

```text
TargetProfile
FrontierReport
LiteratureSearchTask
EvidenceCard
LiteratureCandidate
HybridRoutePackage
RoutePackageValidation
```

P1 再补：

```text
StructureProfile
FailureEvent
StrategicDisconnectionCard
StrategicOperator
RouteStatus
```

### 3. 文献 surrogate 和 “No raw reaction injection” 边界需要更硬

主计划明确禁止 raw LLM reaction injection，也写明 StrategicDisconnectionCard 不能直接
变成 reaction candidate。SMILES-first workflow 又允许输出可解析的
`forward_surrogate rxn_smiles`。

这不矛盾，但必须通过接口声明解决：

```text
forward_surrogate 只能是 validated artifact；
不能直接写入 ChemEnzy route tree；
只能经 StrategicOperator / ChemEnzySearchPolicy compiler 转为 subgoal、source hint、
guarded plugin candidate 或 rerun target；
所有 surrogate 必须标注 not_lab_procedure=true。
```

否则后续实现很容易把 Bufotalin 这类 surrogate 当成真实 reaction record 注入。

### 4. Evidence layer 缺少可执行工具契约

计划要求 literature tools、EvidenceCard、Evidence validation，但还没定义：

- 文献搜索工具清单；
- DOI / PubMed / publisher / patent / internal docs 的优先级；
- paywalled SI 或无法下载 SI 时的降级规则；
- 引用、短摘录、结构证据、条件证据的最低标准；
- exact / analog / precursor / same scaffold 的结构相似度或人工判定规则。

Bufotalin 试跑中已经暴露这个问题：可以定位文献策略，但不一定能取到精确 SI 底物。
这类情况必须输出 `family precedent` 或 `reaction precedent`，不能冒充 `exact evidence`。

### 5. ChemEnzySearchPolicy compiler 缺实际接口

计划说 StrategicOperator、EvidenceCard、FailureDiagnosis、JudgePolicy 会编译到
ChemEnzySearchPolicy，再进入 adapter、plugin、proposal gate 和 judge。

目前仍缺：

- `ChemEnzySearchPolicy` 具体字段；
- `RouteSearchConfig.search_flags` 的兼容映射；
- native chemical/enzyme plugin 如何消费 source allow/deny、subgoal、terminal blacklist；
- policy id / evidence refs 如何写入 run metadata；
- baseline policy 不改变默认行为的测试。

建议先做 compatibility shim：

```text
Validated artifacts -> simple dict search_flags -> existing adapter/plugin/gate
```

不要第一步就实现完整 policy runtime。

### 6. RouteStatus 状态机不够具体

三份文档都强调最终状态：

```text
solved
semisynthesis_closed
partial_anchor
fake_closed_rejected
unresolved
```

但还需要定义严格判定顺序：

- stock audit 通过才能 `solved`；
- anchor evidence 通过但上游未全闭合是 `semisynthesis_closed` 或 `partial_anchor`；
- product-like stock closure 被拒绝是 `fake_closed_rejected`；
- condition gap 是否会降低 solved confidence；
- 文献 surrogate 是否最多只能支持 `partial_anchor`，除非 exact route + stock 都通过。

### 7. Benchmark 尚未具体化

Checklist 列出了 benchmark set 和 metrics，但没有 frozen cases、样本数、阈值和命令。

建议先定义 P0 benchmark pack：

```text
simple_stock_closable: 5-10 cases
fake_closure_controls: 5 cases
NP_like_frontier: Bufotalin + 2-3 additional cases
semisynthesis_anchor: 3 cases
no_route_controls: 3 cases
```

P0 hard gates：

- baseline simple target latency 不显著回退；
- fake closure 不得报告 solved；
- RouteStatus 必须存在；
- no LLM call in inner loop 可由 trace/assertion 证明。

### 8. SMILES-first workflow 是 P0 pipeline 核心

SMILES-first 文档完整描述了人工/agent 工作流，Bufotalin 示例包也证明可执行。
因此它不应排到 P1 之后；它就是当前 P0 的主实现对象。但它还缺：

- generic CLI；
- target/frontier 自动选择；
- Codex 文献检索任务契约；
- evidence card JSONL；
- strategic intermediate / disconnection candidate generator；
- literature candidate JSONL validator；
- route package auditor / validation JSON；
- route figure generator 的通用包装；
- 与后续 CaseBlackboard artifact 的字段对齐。

建议 P0 直接落地：

```text
scripts/run_smiles_first_literature_workflow.py
```

输入 target SMILES、可选 frontier SMILES、family hint、objective、output root；
输出 `target_profile.json`、`frontier_report.json`、`evidence_cards.jsonl`、
`*_literature_rxn_candidates.jsonl`、`*_hybrid_retrosynthesis_route.json`、
`validation.json`、`summary.md` 和 figures。

## 完整性判断

当前计划在概念层面完整，覆盖了：

- episode controller；
- worker；
- evidence；
- strategic disconnection；
- policy compiler；
- compiled judge；
- route auditor；
- condition；
- evolution；
- CLI/API/Web；
- benchmark。

但在工程完整性上仍缺四类内容：

1. **接口完整性**：schema、枚举、字段、状态转换、compiler input/output。
2. **执行完整性**：每阶段的最小 CLI、命令、产物路径、trace 格式。
3. **验收完整性**：frozen benchmark、阈值、pass/fail 命令。
4. **降级完整性**：文献缺失、SI 不可访问、analog-only、条件缺失、预算耗尽时的输出规则。

## 推荐实施顺序

### Sprint 1：P0a，SMILES profile + frontier extraction

- `scripts/run_smiles_first_literature_workflow.py` CLI；
- `TargetProfile`；
- baseline ChemEnzy/template run 或 `--frontier-smiles` fallback；
- `FrontierReport`；
- advanced same-scaffold / no-progress / ordinary-decoration-only frontier flags；
- Bufotalin-like regression case。

### Sprint 2：P0b，Codex literature retrieval + evidence cards

- `LiteratureSearchTask`；
- Codex bounded literature research wrapper or manual artifact path；
- `EvidenceCard` schema；
- exact / family / reaction / analogy relation labels；
- source URL/DOI/local trace validation；
- `literature_search_report.md`。

### Sprint 3：P0c，strategic candidate generation + hybrid route package

- `LiteratureCandidate` schema；
- `exact_fragment_retro` generation；
- `forward_surrogate` generation with `not_lab_procedure=true`；
- `route_anchor` generation；
- `HybridRoutePackage`；
- `validation.json`；
- route map / summary。

### Sprint 4：P1，RouteStatus / trace / bounded rerun

- minimal CaseBlackboard；
- `RouteStatus` schema；
- route package audit wrapper；
- fake closure FailureEvent；
- `StrategicOperator`；
- minimal `ChemEnzySearchPolicy`；
- terminal blacklist / anchor whitelist / preferred subgoal；
- guided rerun with policy id and budget；
- benchmark replay。

### Sprint 5：P2/P3

- condition status；
- enzyme bridge structured action；
- evidence-gated candidate/shadow KB；
- benchmark promotion gate；
- rollback。

## 结论

这套计划值得继续推进。它的方向比“让 LLM 直接给路线”更可靠，因为它把 LLM 限制在
episode-level 调研和策略草案，把 ChemEnzy、validator、compiled judge、auditor 放在
真实性决策位置。

当前最大风险不是不可实现，而是把 P0 做成过大的通用 agent 框架，或者把 P0 做成
audit-only 而推迟真正核心能力。应先把 P0 收窄为 SMILES-first 文献战略断键闭环：
target profile、frontier、Codex 文献证据、高级中间体/断键候选、hybrid route package
和 validation。RouteStatus、case trace、compiled judge 与 guided rerun 放到 P1 工程化。
