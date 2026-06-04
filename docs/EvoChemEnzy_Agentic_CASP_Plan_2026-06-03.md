# EvoChemEnzy Agentic CASP 主线计划

日期：2026-06-03

## 0. 文档整合状态

整合日期：2026-06-04

本文现在是 EvoChemEnzy / ChemEnzy-agentic CASP 主线架构的唯一顶层计划文档。
以下临时阶段文档的结论已经吸收到本文、交付 checklist、可行性审计和
SMILES-first workflow 中，原文件归档到 `docs/archive/2026-05/` 或
`docs/archive/2026-06/`：

- 2026-05-19 至 2026-05-21 的 AutoPlanner/Cascade 主线、verifier proof、
  proposal training、CCTS decision、cleanup/status 报告。
- 2026-05-27 至 2026-05-28 的 enzyme bridge roadmap、data audit、
  bridge pack、bridge verifier、P5 evidence、ChemEnzy module audit、
  enzyme coverage、SP-v1 verifier、native-vs-enhanced benchmark 报告。
- 2026-06-02 至 2026-06-03 的 statin material-gate iteration 和
  Product / Route Audit TODO。

整合后的统一判断：

- 项目不再把 `one-pot cascade condition compatibility` 作为主创新点；它保留为
  route audit / condition feasibility 模块。
- 主创新表述收束为 `enzyme-aware chemo-enzymatic bridge planning`：识别化学
  中间体与酶底物/产物/EC 空间之间的可连接性，并通过 verifier-gated search 控制
  酶步假阳性。
- ChemEnzy native multi-step search 仍是 step-level inner engine；旧 CCTS、
  route-pool ranker、block scorer、learned route value、expert CSV/LLM review
  fallback 均为历史/复现实验，不再是默认 runtime 主线。
- AiZ/RetroKNN/外部 proposal 能提升单步候选覆盖，但 naive ensemble 会污染多步
  搜索；所有补充 proposal 必须走 gated routing / source budget / verifier gate。
- Bridge pack v0、bridge verifier v0、enzyme SP-v1 verifier 已证明 weak-label
  + hard negative + gate 的方向可行，但 P5 evidence 仍是 diagnostic/partial
  evidence，不是 stock-closed 生产级完整路线。
- Product / Route audit 是路线声明的硬门槛：stock closure、EC annotation、
  post-hoc condition prediction、planner score 都不能单独证明路线 solved。

当前实施优先级：

```text
P0 不先实现完整 agent 框架。
P0 先实现 SMILES-first 文献战略断键闭环：
target profile -> frontier -> Codex literature retrieval -> evidence cards
-> advanced intermediate / strategic disconnection candidates
-> hybrid route package -> validation -> route map / summary.

Case Blackboard、Codex Controller、RouteStatus、guided rerun、compiled judge、
enzyme bridge runtime 和 evolution manager 属于 P1+ 工程化，除 P0 guardrail stub
外不得阻塞上述闭环。
```

---

## 1. 一句话目标

EvoChemEnzy 的目标是把现有 ChemEnzy 逆合成内核升级为一个事件驱动、证据约束、人类化学家式的 agentic CASP 系统：

```text
ChemEnzy 负责 step-level 高速逆合成搜索。
Codex 主 agent 负责 episode-level 目标定调、失败诊断、实时文献调研、策略干预、条件建议、路线审计和自进化候选沉淀。
```

主线原则：

```text
No LLM in ChemEnzy inner loop.
No online LLM rerank.
No online LLM proposal judge.
No raw LLM reaction injection.
```

Codex / LLM 不直接做逆合成搜索，也不直接宣布路线成立；Codex 只通过受控工具调用产生结构化 artifact，再由 validator、adapter、ChemEnzy 和 auditor 消费。

---

## 2. 我们目前已有的基础

### 2.1 ChemEnzy 主内核

已有：

- `ChemEnzyBackendAdapter`：接入 vendor `ChemEnzyRetroPlanner`，输出标准化 route result。
- ChemEnzy native search：保留 vendor 内部 one-step proposal、多步搜索、route ranking 和 stock closure。
- ChemEnzy one-step sources：默认使用 GraphFP / BioNav ONMT / native BioNav one-step 等模型。
- `RouteSearchConfig.search_flags`：支持运行期 search context 注入，可以在不重建 planner 的情况下影响目标搜索。
- dry-run / preflight / structured failure：已有后端可用性检查和结构化失败返回。
- `ChemEnzyOneStepProposalProvider`：能把 ChemEnzy one-step 输出转换成 AutoPlanner route-tree `CandidateAction` 风格候选。
- semisynthesis rescue、proposal gate、GraphFP / dual-tower 化学候选补强：已有候选质量控制和保守补充能力。

当前定位：

```text
ChemEnzy 是唯一 step-level retrosynthesis inner engine。
ChemEnzy Function-Call Consumption Layer 负责把 agent 产出的策略、证据、失败诊断和约束编译成 ChemEnzy 可执行配置。
```

### 2.2 ChemEnzy native plugin 基础

已有：

- `NativeChemicalOneStepWrapper`：在 ChemEnzy vendor one-step expansion 点追加 conservative chemical tail candidates。
- `NativeEnzymeOneStepWrapper`：在 ChemEnzy vendor one-step expansion 点追加 bridge / SP-v1 / precedent gated enzyme candidates。
- chemical plugin 已记录 calls、added、duplicate、invalid、proposal gate rejected 等统计。
- enzyme plugin 已记录 bridge hits、SP-v1 accepted/rejected、material rejected、quality scored 等统计。
- enzyme step audit 已明确：EC annotation 不等于 enzyme step validation。

当前定位：

```text
native plugin 是 ChemEnzy expansion 点的受控补充。
enzyme plugin 下一步升级为结构化 enzyme action、enzyme router、selection cost 和 route-level audit。
```

### 2.3 Condition / Audit 基础

已有：

- condition prediction、EC assignment、active-site annotation 等能力已存在于 ChemEnzy / AutoPlanner 周边模块。
- 现有审计文档已指出 post-search annotation 不能直接等同于 search-time validation。
- route plausibility、proposal gate、enzyme step audit 等模块已经具备初步真实性检查能力。

当前定位：

```text
condition、cofactor、enzyme evidence、route audit 是主线一等模块。
进入 ChemEnzy inner loop 的形式是 compiled policy、deterministic feature 或 lightweight model。
```

### 2.4 Codex 主 agent 基础定位

当前定位：

```text
Codex 是 EvoChemEnzy 的主 episode-level agent。
Codex 负责读 repo、读 trace、读文档、调研文献、抽取文献报道的战略断键、诊断失败和生成结构化策略草案。
ChemEnzy 负责搜索，validator / auditor 负责真实性约束。
```

Codex 的优势用于：

- 多跳文件检索和项目上下文理解。
- 根据 ChemEnzy trace 定位卡点。
- 根据失败事件选择文献检索方向。
- 检索目标或类似天然产物最新报道路线中的战略断键。
- 抽取 total synthesis / semisynthesis / biosynthesis-inspired route 中真正解决核心骨架的断点。
- 汇总文献证据、路线审计和策略干预建议。
- 在需要时 spawn bounded Codex Research Worker 处理局部研究任务。

Codex 的输出必须落到 typed artifact：

```text
TargetTriage
ResearchReport
EvidenceCard draft
StrategicDisconnectionCard draft
FailureDiagnosis
StrategicOperator draft
ConditionCandidate draft
AuditReport draft
EvolutionCandidate draft
```

---

## 3. 最终目标架构

正式命名建议：

```text
EvoChemEnzy: Event-Driven Human-Chemist-Style Agentic CASP over ChemEnzy
```

总体架构：

```text
User Target
  |
  v
Target Resolver / Structure Profiler
  |
  v
Case Blackboard
  |
  v
Codex Chief Chemist Controller
  |
  |-- Codex Research Worker
  |-- Literature Search Tools
  |-- Evidence Extraction Tools
  |-- Evidence Validation Tools
  |-- Strategic Disconnection Mining Tools
  |-- Strategy Compiler
  |-- Condition Design Tools
  |-- Route Auditor
  |-- Evolution Manager
  |
  v
ChemEnzy Function-Call Consumption Layer
  |
  |-- Target/Profile -> search intent
  |-- Evidence -> anchor whitelist / terminal blacklist
  |-- StrategicDisconnectionCard -> subgoal / disconnection guidance
  |-- StrategicOperator -> ChemEnzySearchPolicy
  |-- FailureDiagnosis -> bounded rerun config
  |-- JudgePolicy -> compiled deterministic gate
  |-- Conditions -> route feasibility annotation
  |
  v
ChemEnzy Inner Engine
  |
  |-- baseline search
  |-- guided search
  |-- stuck-node rerun
  |-- native chemical plugin
  |-- native enzyme plugin
  |-- stock closure
  |-- compiled deterministic search-time judge
  |-- route ranking
  |
  v
Route Auditor
  |
  v
Final RouteStatus
```

最终输出强制给出明确路线状态：

```text
solved
semisynthesis_closed
partial_anchor
fake_closed_rejected
unresolved
```

---

## 4. 主运行模型

系统主表达采用事件驱动 episode loop：

```text
observe -> decide -> act -> validate -> update blackboard -> stop or continue
```

每一轮：

- `observe`：读取目标、结构画像、ChemEnzy trace、路线候选、失败事件、证据、预算。
- `decide`：Codex Chief Chemist Controller 选择下一步受控 action。
- `act`：调用 ChemEnzy、Codex Research Worker、文献工具、证据工具、审计工具、条件工具，或结束。
- `validate`：所有 LLM artifact 必须通过 deterministic validator。
- `update`：写入 Case Blackboard。
- `stop`：输出明确 RouteStatus，或在预算内继续下一轮。

状态机仍可保留，但只做底层 safety guard：

```text
action allowed / forbidden
budget limit
retry limit
artifact validation
trace recovery
stop condition
```

---

## 5. 受控 Action Space

Controller 只能选择有限动作：

```text
RESOLVE_TARGET
PROFILE_STRUCTURE
RUN_BASELINE_CHEMENZY
AUDIT_ROUTE
DIAGNOSE_FAILURE
RESEARCH_TARGET
RESEARCH_STUCK_NODE
EXTRACT_EVIDENCE
VALIDATE_EVIDENCE
COMPILE_STRATEGIC_OPERATOR
RUN_GUIDED_CHEMENZY
DESIGN_CONDITIONS
SUBMIT_EVOLUTION_CANDIDATE
FINAL_SOLVED
FINAL_SEMISYNTHESIS_CLOSED
FINAL_PARTIAL_ANCHOR
FINAL_FAKE_CLOSED_REJECTED
FINAL_UNRESOLVED
```

明确禁止：

```text
在线 LLM rank_actions
在线 LLM rank_open_leaves
在线 LLM judge_proposal
LLM 裸生成 reaction candidate 进入主线
多 agent 自由聊天决定路线
agent 直接修改 production KB
```

---

## 6. Codex 主 agent / worker 编排

Codex Chief Chemist Controller 是主 agent，负责 episode-level 控制。它可以直接调用工具，也可以 spawn bounded Codex Research Worker 执行局部研究任务。

### 6.1 何时 spawn Codex Research Worker

允许触发：

```text
RESEARCH_TARGET
RESEARCH_STUCK_NODE
MINE_STRATEGIC_DISCONNECTIONS
DIAGNOSE_FAILURE
AUDIT_ROUTE
EXTRACT_EVIDENCE
DESIGN_CONDITIONS
SUBMIT_EVOLUTION_CANDIDATE
```

典型任务：

- 读取 ChemEnzy trace，定位失败卡点。
- 检索 repo 内已有文档、路线、审计报告和实验记录。
- 实时检索目标、stuck node、类似物、前体、条件和失败先例。
- 专门挖掘目标或类似天然产物文献中的战略断键。
- 识别文献路线中真正降低复杂度、构建核心骨架或定义半合成锚点的断点。
- 汇总文献证据和不确定性。
- 生成可验证的策略草案。

### 6.2 Worker 输入输出

Worker 输入必须包含：

```text
case_id
task_type
target / stuck node / route fragment
current evidence
current failure event
allowed tools
budget
required artifact type
```

Worker 输出必须是：

```text
ResearchReport
EvidenceCard draft
StrategicDisconnectionCard draft
FailureDiagnosis
StrategicOperator draft
ConditionCandidate draft
AuditReport draft
EvolutionCandidate draft
```

Worker 输出不能直接进入 ChemEnzy。所有输出先进入 validator，再由 ChemEnzy Function-Call Consumption Layer 编译。

### 6.3 Worker 边界

Codex Worker 可以：

```text
读文件
查文献
读 trace
总结证据
提出失败假设
提出策略草案
```

Codex Worker 禁止：

```text
改 ChemEnzy search tree
在线 rerank ChemEnzy candidate
在线 judge ChemEnzy proposal
裸生成 reaction candidate 进入主线
直接宣布 route solved
直接写 production KB
```

### 6.4 生产化路径

原型阶段可以 spawn Codex 进程获得强检索和强上下文理解能力。

生产阶段应收敛为：

```text
Codex worker wrapper
allow-listed tools
typed artifact output
validator
trace store
budget limiter
replayable run record
```

---

## 7. LLM Function Call 如何被 ChemEnzy 消费

### 7.1 目标定调

Codex / tools 产出：

```text
TargetResolution
StructureProfile
TargetTriage
```

ChemEnzy 消费方式：

```text
baseline / guided / literature-assisted 决策
strict stock closure hint
source policy hint
plugin enable hint
```

简单目标直接 ChemEnzy baseline；复杂、高天然产物倾向、高假闭合风险目标进入 literature-assisted 或 guided 模式。

### 7.2 实时文献调研

Codex / tools 产出：

```text
LiteratureHit
EvidenceCard
```

ChemEnzy 消费方式：

```text
不直接消费原始文献。
只消费 validated evidence 编译出的 anchor whitelist、terminal blacklist、route mode prior、condition prior。
```

### 7.3 文献战略断键挖掘

Codex / tools 产出：

```text
StrategicDisconnectionCard
```

用途：

```text
从目标或类似天然产物文献路线中抽取关键断键。
弥补 ChemEnzy 模板、单步模型和 stock closure 对高级天然产物核心骨架的不足。
```

断键来源：

```text
exact target total synthesis
exact target semisynthesis
close analog total synthesis
close analog semisynthesis
biosynthesis-inspired disconnection
known precursor-to-target transformation
failed route / negative evidence
```

ChemEnzy 消费方式：

```text
validated StrategicDisconnectionCard
  -> strategic subgoal
  -> preferred disconnection pattern
  -> protected core / forbidden fake terminal
  -> semisynthesis anchor candidate
  -> stuck-node rerun target
  -> source promotion / demotion
  -> ChemEnzySearchPolicy
```

StrategicDisconnectionCard 不能直接变成 reaction candidate。它只提供断键指导、subgoal、anchor 和 policy 约束。

### 7.4 策略编译

Codex / tools 产出：

```text
StrategicOperator
```

Adapter 编译为：

```text
ChemEnzySearchPolicy
AnchorWhitelist
TerminalBlacklist
SourceWeights
BudgetConfig
RerunScope
JudgePolicy
```

这是 agent 影响 ChemEnzy 搜索的主入口。

### 7.5 失败诊断

输入：

```text
ChemEnzy trace
open leaves
rejected terminals
route candidates
route audit
```

Codex / tools 产出：

```text
FailureDiagnosis
StuckNodeIntervention
```

ChemEnzy 消费方式：

```text
stuck-node rerun target
forbidden terminal
required anchor
stricter stock policy
source promotion / demotion
bounded rerun budget
```

Agent 只诊断和提出干预，不直接改 search tree。

### 7.6 搜索期 judge

LLM 只允许在 episode 层生成：

```text
JudgePolicy
```

ChemEnzy inner loop 执行：

```text
Compiled Search-Time Judge
Deterministic Search-Time Gate
```

命名上避免使用 `LLM Search-Time Judge`，防止误解为每步调用 LLM。

### 7.7 条件设计

Codex / tools 产出：

```text
ConditionCandidate
ConditionGap
```

ChemEnzy / Auditor 消费方式：

```text
route annotation
condition confidence
route feasibility score
condition gap audit
```

条件不单独决定路线闭合，但影响最终可行性和 RouteStatus 解释。

### 7.8 自进化

Codex / tools 产出：

```text
EvolutionCandidate
ReactionRecordCandidate
TemplateCandidate
ConditionCandidate
```

系统消费方式：

```text
candidate KB
shadow KB
benchmark gate
staging KB
production KB
rollback
```

禁止在 target run 中直接写 production KB。

---

## 8. 需要做的主要改动

### 8.1 新增 Case Blackboard

目的：

```text
让目标、结构画像、证据、ChemEnzy run、失败事件、策略、条件、审计和自进化候选都有统一载体。
```

要求：

- 所有 agent 和工具只读写 blackboard artifact。
- 关键事实不靠自然语言上下文传递。
- 每个 artifact 有 trace id、来源、验证状态和预算消耗。

### 8.2 新增 Codex Episode Controller

目的：

```text
以 Codex 作为人类化学家式 episode-level controller。
```

要求：

- 每轮从 blackboard 观察当前状态。
- 只选择受控 action。
- 控制 LLM、文献、ChemEnzy rerun 和 wall-time 预算。
- 决定何时 final solved / unresolved。

### 8.3 新增 Codex Worker Wrapper

目的：

```text
把 spawn Codex 进程封装为可预算、可追踪、可校验的 research worker。
```

要求：

- 每个 worker 有明确 task type、输入 artifact、允许工具和预算。
- worker 输出必须是 typed artifact。
- worker 不能直接调用 ChemEnzy inner loop。
- worker 输出必须经过 validator。

### 8.4 新增 ChemEnzy Function-Call Consumption Layer

目的：

```text
把 LLM function call artifact 编译成 ChemEnzy 可执行策略。
```

改动：

- StrategicOperator -> ChemEnzySearchPolicy。
- EvidenceCard -> anchor whitelist / terminal blacklist。
- FailureDiagnosis -> bounded rerun config。
- JudgePolicy -> deterministic gate。
- ConditionCandidate -> route feasibility annotation。

### 8.5 强化 Compiled Search-Time Judge

目的：

```text
在 ChemEnzy 搜索期高速拦截假闭合和无效 terminal，但不调用 LLM。
```

重点拦截：

- 高级同骨架前体假 stock closure。
- no-complexity-drop terminal。
- product-like terminal。
- same-scaffold loop。
- unsupported enzyme core disconnection。
- suspicious stock / metabolite / precursor terminal。

### 8.6 强化 Route Auditor

目的：

```text
auditor 负责最终路线真实性判定，ChemEnzy planner score 作为输入信号。
```

Auditor 必须输出：

- stock audit。
- route mode audit。
- evidence audit。
- condition audit。
- fake closure audit。
- enzyme plausibility audit。
- unresolved core / stuck node。
- final RouteStatus。

### 8.7 新增 Live Literature Research + Evidence Validation

目的：

```text
让 agent 像化学家一样实时查文献，但文献必须先变成 validated evidence。
```

调研触发条件：

- 高复杂 / 高天然产物倾向目标的 literature-assisted mode。
- baseline no closure。
- fake closure。
- evidence gap。
- stuck node。
- condition gap。

禁止：

```text
无假设的大范围综述。
未经验证的文献直接进入 ChemEnzy。
```

### 8.8 强化 enzyme router 与结构化 enzyme action

目的：

```text
把 enzyme 增强升级为 node-level enzyme intent、结构化 enzyme action、selection cost 和 route-level audit。
```

改动方向：

- node-level enzyme/chemical expansion intent。
- candidate-level EC / cofactor / precedent / reaction-center / material sanity。
- enzyme selection cost 进入 ChemEnzy candidate cost。
- route-level cofactor / condition / material ledger。

### 8.9 新增 Condition Design Agent

目的：

```text
路线不仅要闭合，还要给出条件候选、证据等级和可行性。
```

优先级：

```text
exact literature condition
analog literature condition
template / cluster condition
condition model prediction
condition_unknown
```

### 8.10 新增 Evidence-Gated Evolution

目的：

```text
让系统能积累新模板、条件、anchor 和 controller trace，但不污染生产系统。
```

流程：

```text
EvidenceCard
  -> Candidate
  -> Shadow KB
  -> Benchmark Gate
  -> Staging KB
  -> Production KB
```

---

## 9. 典型运行策略

### 9.1 简单目标

```text
PROFILE_STRUCTURE
  -> RUN_BASELINE_CHEMENZY
  -> AUDIT_ROUTE
  -> FINAL_SOLVED or FINAL_UNRESOLVED
```

不先查文献。

### 9.2 中等复杂目标

```text
RUN_BASELINE_CHEMENZY
  -> AUDIT_ROUTE
  -> DIAGNOSE_FAILURE if needed
  -> RESEARCH_STUCK_NODE
  -> VALIDATE_EVIDENCE
  -> COMPILE_STRATEGIC_OPERATOR
  -> RUN_GUIDED_CHEMENZY
  -> AUDIT_ROUTE
```

调研只针对失败点。

### 9.3 高复杂 / 高天然产物倾向目标

```text
PROFILE_STRUCTURE
  -> RESEARCH_TARGET or literature-assisted guidance
  -> VALIDATE_EVIDENCE
  -> COMPILE_STRATEGIC_OPERATOR
  -> RUN_GUIDED_CHEMENZY
  -> AUDIT_ROUTE
  -> RESEARCH_STUCK_NODE only if needed
```

重点防止高级同骨架前体假闭合。

---

## 10. 阶段路线图

### P0：SMILES-first 文献战略断键闭环

目标：

```text
用户给一个新 SMILES 后，系统能识别普通规划器卡住的高级 frontier，
用 Codex 检索文献，抽取关键战略断键/高级中间体/route anchor，
并输出可校验 hybrid route package。
```

交付：

- SMILES-first CLI。
- Target structure profile。
- ChemEnzy/template baseline 或手动 frontier 输入。
- Advanced frontier detection。
- Codex literature retrieval task。
- EvidenceCard JSONL。
- `exact_fragment_retro` / `forward_surrogate` / `route_anchor` candidate generation。
- Hybrid retrosynthesis route package。
- SMILES/rxn validation report。
- Route map / summary。
- Minimal route-package audit guard。
- No LLM inner loop guard。

### P1：RouteStatus / trace / guided rerun 工程化

目标：

```text
把 P0 产出的文献战略 package 纳入统一 RouteStatus、case trace 和 bounded guided rerun。
```

交付：

- FailureEvent。
- Minimal Case Blackboard。
- RouteStatus。
- RouteAuditReport。
- StrategicOperator compiler。
- ChemEnzySearchPolicy。
- bounded guided rerun。
- Compiled terminal judge。

### P2：条件与可行性

目标：

```text
路线不仅闭合，还要说明条件证据和可行性等级。
```

交付：

- Condition Agent。
- Condition KB。
- condition confidence。
- condition gap audit。
- forward plausibility check。
- route feasibility annotation。

### P3：证据门控自进化

目标：

```text
系统能积累知识，但不会污染生产库。
```

交付：

- Reaction / Template / Condition candidates。
- Shadow KB。
- Benchmark Gate。
- Staging / Production KB。
- Rollback。
- Controller trace learning。

---

## 11. 验收标准

### P0 验收

- 给定合法 target SMILES 能生成 `target_profile.json`。
- 普通规划结果或手动 frontier 能生成 `frontier_report.json`。
- Codex 文献检索能输出可追溯 `evidence_cards.jsonl`。
- 能生成 `exact_fragment_retro`、`forward_surrogate`、`route_anchor` 三类候选。
- `forward_surrogate` 必须标注 `not_lab_procedure=true`。
- `route_anchor` 不能被当作 stock closure。
- 输出 `hybrid_retrosynthesis_route.json`、`validation.json`、`summary.md` 和路线图。
- P0 package 只能声明 planning material / partial anchor / unresolved，不能在无 stock/audit 证明时声明 solved。
- ChemEnzy inner loop 中无在线 LLM rerank / judge / proposal。

### P1 验收

- P0 package 能进入 case trace。
- 最终报告输出 RouteStatus。
- baseline 失败或 fake closure 后能生成 FailureEvent。
- 文献必须先变成 validated evidence，才能影响 ChemEnzy rerun。
- guided rerun 必须有明确 rerun reason 和预算。
- 达不到证据或预算耗尽时输出 unresolved。

### P2 验收

- top route 每一步都有 condition status。
- exact / analog / model-predicted / unknown 条件来源可区分。
- condition gap 会影响 route audit。
- final report 能说明路线闭合但条件不充分的情况。

### P3 验收

- target run 只能写 candidate / shadow knowledge。
- production KB 晋升必须经过 benchmark gate。
- 新知识不能提升 fake closure rate。
- 所有晋升可回滚。

---

## 12. 明确不做的方向

不采用：

```text
固定线性状态机作为主工作流。
自由 ReAct agent 做逆合成。
多 agent 自由讨论后决定路线。
LLM 每步 rerank ChemEnzy candidate。
LLM 每步 judge proposal。
LLM raw reaction injection 进入主线。
文献未经验证直接进入搜索。
target run 直接修改 production KB。
```

---

## 13. 当前计划总结

本计划围绕 ChemEnzy 构建一个更可信、更会调研、更会承认失败的 agentic CASP 系统。

最终主线是：

```text
Codex Chief Chemist Controller
  + Case Blackboard
  + Codex Research Worker
  + Allow-listed Tools
  + ChemEnzy Function-Call Consumption Layer
  + ChemEnzy Inner Engine
  + Compiled Search-Time Judge
  + Route Auditor
  + Evidence-Gated Evolution
```

最重要的一句话：

```text
Codex 在 episode 层观察、诊断、查证据和改策略；
ChemEnzy 在 step 层保持高速、可复现的逆合成搜索。
```
