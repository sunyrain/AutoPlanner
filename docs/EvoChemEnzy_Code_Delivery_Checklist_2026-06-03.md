# EvoChemEnzy 代码改造交付 Checklist

日期：2026-06-03

用途：这是后续代码改造的最终交付清单。后续 Codex / agent / 工程师必须按此逐项实现、测试、审计和打勾。每个模块完成后需要留下代码、测试、运行记录和验收结论。

整合状态：

```text
2026-06-04 已吸收 2026-05/2026-06 顶层临时报告。
临时报告保留在 docs/archive/，但不再作为后续实现的 source of truth。
实现、测试和验收以本 checklist、主线计划、可行性审计、SMILES-first workflow 为准。
```

总体目标：

```text
Codex Chief Chemist Controller 负责 episode-level 控制。
ChemEnzy Inner Engine 负责 step-level 搜索。
Codex Worker 负责受控检索、trace 诊断和策略草案。
Validator / Auditor 负责真实性约束。
ChemEnzy Function-Call Consumption Layer 负责把 artifact 编译成 ChemEnzy policy。
```

硬边界：

```text
No LLM in ChemEnzy inner loop.
No online LLM rerank.
No online LLM proposal judge.
No raw LLM reaction injection.
No direct production KB write from target run.
```

---

## 当前 P0 开发板（直接开工版）

更新日期：2026-06-04

当前全力开发目标：

```text
把 SMILES-first literature strategic workflow 做成可运行闭环。
核心能力是 Codex 文献检索 + 高级中间体/战略断键生成。
其它完整 agent 框架能力全部让位于这个 P0 闭环。
```

开工入口：

- 主执行文档：
  `docs/SMILES_FIRST_LITERATURE_STRATEGIC_WORKFLOW_2026-06-03.md`
- 工程拆解：
  本文件 `开发启动交付清单` 的 P0a / P0b / P0c / P0 Guardrail。
- 架构约束：
  `docs/EvoChemEnzy_Agentic_CASP_Plan_2026-06-03.md`
- 风险与可行性：
  `docs/EvoChemEnzy_Plan_Feasibility_Audit_2026-06-04.md`

P0 只接受以下交付物作为完成标准：

- [x] `scripts/run_smiles_first_literature_workflow.py`
- [x] `target_profile.json`
- [x] `baseline_routes.json` 或 baseline unavailable record
- [x] `frontier_report.json`
- [x] `literature_search_report.md`
- [x] `evidence_cards.jsonl`
- [x] `*_literature_rxn_candidates.jsonl`
- [x] `*_hybrid_retrosynthesis_route.json`
- [x] `validation.json`
- [x] `summary.md`
- [x] route map SVG / Mermaid / Graphviz source

P0 candidate 必须只分三类：

```text
exact_fragment_retro
forward_surrogate
route_anchor
```

P0 最小代码范围：

- [x] `cascade_planner/agent/smiles_first.py` 或 `cascade_planner/agent/target_profile.py`
- [x] `cascade_planner/agent/evidence_cards.py`
- [x] `cascade_planner/agent/literature_research.py`
- [x] `cascade_planner/agent/strategic_candidate_generation.py`
- [x] `cascade_planner/agent/route_package.py`
- [x] `tests/test_smiles_first_workflow.py`
- [x] `tests/test_literature_evidence_cards.py`
- [x] `tests/test_strategic_candidate_generation.py`

P0 验收命令：

```bash
pytest -q tests/test_smiles_first_workflow.py tests/test_literature_evidence_cards.py tests/test_strategic_candidate_generation.py tests/test_route_plausibility.py
python -m py_compile scripts/run_smiles_first_literature_workflow.py
python -m py_compile cascade_planner/agent/evidence_cards.py cascade_planner/agent/literature_research.py cascade_planner/agent/strategic_candidate_generation.py cascade_planner/agent/route_package.py
```

P0 禁止提前做成主任务的内容：

- 完整 Case Blackboard。
- 完整 Codex Chief Chemist Controller。
- 完整 RouteStatus 状态机。
- StrategicOperator compiler + guided ChemEnzy rerun。
- compiled search-time judge。
- enzyme bridge runtime integration。
- evolution manager / production KB promotion。

这些内容可以作为 guardrail stub 或 schema compatibility 预留，但不得阻塞
Codex 文献检索和高级中间体拆解生成闭环。

下一阶段入口：

- `docs/LITERATURE_TO_EXECUTABLE_TEMPLATE_CHECKLIST_2026-06-04.md`

该 checklist 是 P0/P1b 之后的主线，目标是把文献调研结果从
`advisory_strategy_template` 推进为 ChemEnzy 可消费的 executable one-step
template / proposal source。后续如果讨论“文献是否真正进入 ChemEnzy 搜索”，
以该文档的 L0-L7 阶段为执行清单。

---

## 开发启动交付清单

更新日期：2026-06-04

本节是实际开工顺序。最新优先级已经调整：P0 的核心不是先做完整
RouteStatus/blackboard/controller，而是先把
`SMILES-first literature strategic workflow` 做成可运行闭环。

P0 要先解决的问题是：

```text
用户给一个新 SMILES 后，系统能识别普通规划器卡住的高级 frontier，
用 Codex 进行受控文献检索，
抽取文献关键战略断键 / 高级中间体 / route anchor，
生成可校验的 exact_fragment_retro、forward_surrogate、route_anchor，
并输出 route package、validation 和路线图。
```

RouteStatus、fake-closure audit、blackboard、compiled judge 仍然重要，但在本轮中
它们是 P0 护栏和 P1 工程化任务，不再压过文献战略断键闭环。

### P0a：SMILES Profile + Baseline Frontier Extraction

目标：

```text
从用户 target SMILES 出发，自动得到结构画像、普通规划器结果和高级 unresolved frontier。
```

交付物：

- [x] 新增 SMILES-first CLI：输入 `target_name`、`target_smiles`、可选
  `family_hint`、`objective`、`output_dir`。
- [x] 生成 `target_profile.json`：canonical SMILES、InChIKey、formula、heavy atoms、
  rings、stereocenters、ring systems、linker bonds、side chains、family hints。
- [x] 调用普通 ChemEnzy/template baseline，或在 baseline 不可用时允许
  `--frontier-smiles` 手动输入高级 frontier。
- [x] 生成 `baseline_routes.json` 和 `frontier_report.json`。
- [x] frontier 判断必须标记：advanced same-scaffold、no complexity drop、
  high target similarity、ordinary decoration only、unresolved core。
- [x] 不允许因为普通步骤闭合了末端修饰就宣布全路线 solved。

建议代码范围：

- 新增 `scripts/run_smiles_first_literature_workflow.py`
- 新增 `cascade_planner/agent/smiles_first.py` 或 `cascade_planner/agent/target_profile.py`
- `cascade_planner/baselines/chem_enzy_adapter.py`
- `cascade_planner/baselines/route_plausibility.py`
- `tests/test_smiles_first_workflow.py`

验收命令：

```bash
pytest -q tests/test_smiles_first_workflow.py tests/test_route_plausibility.py
python -m py_compile scripts/run_smiles_first_literature_workflow.py
```

必须通过的验收：

- [x] 一个合法 target SMILES 能生成 `target_profile.json`。
- [x] 一个普通 baseline route 能生成 `baseline_routes.json`。
- [x] Bufotalin-like O-acetylation / late decoration 不会被视为完整骨架路线。
- [x] advanced frontier 能被写入 `frontier_report.json`。
- [x] baseline ChemEnzy default 不因该 CLI 改变。

停止条件：

- 如果 ChemEnzy baseline 接入成本过高，P0a 先支持 `--frontier-smiles` 手动输入，
  但必须保留 target profile 和 frontier report。
- 如果 RDKit 无法解析 target，停止并输出 target ambiguity，不进入文献阶段。

### P0b：Codex Literature Retrieval + Evidence Cards

目标：

```text
对 target/frontier 做 Codex 受控文献检索，找到 family、key bond-forming step、
高级中间体、route anchor 和可实例化战略断键证据。
```

交付物：

- [x] 定义最小 `LiteratureSearchTask`：target profile、frontier SMILES、family hints、
  query budget、allowed source types、required output schema。
- [x] 实现 Codex 文献检索入口：可以是 bounded worker、CLI wrapper，或先由
  Codex 手工执行但必须落盘为 schema artifact。
- [x] 输出 `literature_search_report.md`：检索 query、命中文献、证据等级、
  exact/family/reaction/analogy 区分。
- [x] 输出 `evidence_cards.jsonl`：每条证据包含 source metadata、URL/DOI/local ref、
  target relation、claim type、route role、confidence、limitations。
- [x] 证据必须优先回答：scaffold family、key construction、advanced intermediate、
  chiral-pool/biosynthetic/semisynthesis anchor、是否能实例化到 frontier。
- [x] 没有足够文献时输出 `unresolved_literature_gap`，不能编造 route。

建议代码范围：

- `scripts/run_smiles_first_literature_workflow.py`
- 新增 `cascade_planner/agent/literature_research.py`
- 新增 `cascade_planner/agent/evidence_cards.py`
- `tests/test_smiles_first_workflow.py`
- 新增 `tests/test_literature_evidence_cards.py`

验收命令：

```bash
pytest -q tests/test_smiles_first_workflow.py tests/test_literature_evidence_cards.py
python -m py_compile cascade_planner/agent/literature_research.py cascade_planner/agent/evidence_cards.py
```

必须通过的验收：

- [x] exact target evidence、family precedent、reaction precedent、analogy only 可区分。
- [x] 每个 evidence card 有可追溯 URL/DOI/local file ref。
- [x] evidence card 不能只是一段自然语言摘要；必须有结构化 fields。
- [x] untraceable evidence 被拒绝或标记为 draft-only。
- [x] 文献没有支持时输出 gap，不进入候选生成。

停止条件：

- 如果在线检索不可用，先支持人工/Codex 填写 `evidence_cards.jsonl`，但必须通过 validator。
- 如果只有 analogy-only evidence，P0 可以生成 critique / weak prior，但不能生成 solved route。

### P0c：Strategic Intermediate + Disconnection Candidate Generation

目标：

```text
把文献证据实例化为高级中间体拆解材料：exact_fragment_retro、forward_surrogate、
route_anchor，并生成可校验 route package 和路线图。
```

交付物：

- [x] 定义 `LiteratureCandidate` schema：
  `candidate_kind = exact_fragment_retro | forward_surrogate | route_anchor`。
- [x] 生成 `*_literature_rxn_candidates.jsonl`。
- [x] `exact_fragment_retro` 支持 dummy atom / labeled cut-site 表达高级 frontier 断键。
- [x] `forward_surrogate` 必须包含 `not_lab_procedure=true`、`surrogate_reason`、
  `literature_basis`，并通过 rxn SMILES 基础校验。
- [x] `route_anchor` 必须说明是 multi-step anchor / chiral-pool anchor /
  biosynthetic or semisynthesis anchor，`rxn_smiles` 可为空。
- [x] 生成 `*_hybrid_retrosynthesis_route.json`：Target -> ordinary steps ->
  frontier -> strategic disconnection -> fragments/anchor。
- [x] 生成 `validation.json`：SMILES/rxn parse、candidate kind、evidence refs、
  route_anchor not stock、surrogate not lab procedure。
- [x] 生成至少一个路线图 SVG 或 Mermaid/Graphviz source，显示文献接管点和上游 anchor。
- [x] 输出 `summary.md`：关键断键、文献依据、候选类型、限制、下一步。

建议代码范围：

- `scripts/run_smiles_first_literature_workflow.py`
- 新增 `cascade_planner/agent/strategic_candidate_generation.py`
- 新增 `cascade_planner/agent/route_package.py`
- `scripts/render_route_figures.py` 或轻量 route map renderer
- `tests/test_smiles_first_workflow.py`
- 新增 `tests/test_strategic_candidate_generation.py`

验收命令：

```bash
pytest -q tests/test_smiles_first_workflow.py tests/test_strategic_candidate_generation.py tests/test_route_plausibility.py
python -m py_compile scripts/run_smiles_first_literature_workflow.py
```

必须通过的验收：

- [x] Bufotalin-like case 能生成普通步骤 + 文献 strategic disconnection + anchor 的 package。
- [x] `forward_surrogate` 通过 parse 但不会被标记为 exact literature reaction。
- [x] `route_anchor` 不会被误计为 stock closure。
- [x] candidate 没有 evidence refs 时 validator 拒绝。
- [x] 输出中明确 `route_status = partial_anchor | unresolved`，除非 stock/audit 另行证明 solved。

### P0 Guardrail：Minimal Route Package Audit

目标：

```text
P0 核心是文献战略断键生成，但交付包必须防止误报 solved。
```

交付物：

- [x] 轻量 `route_package_audit`：检查 target match、candidate kind、SMILES/rxn、
  evidence refs、route_anchor not stock、surrogate not lab procedure。
- [x] 输出最小 `route_status`：`partial_anchor`、`literature_gap`、
  `invalid_package`、`ready_for_guided_rerun`。
- [x] 明确声明：P0 route package 是 planning material，不是生产级完整合成路线。

验收：

- [x] 没有 stock audit 的 route package 不能输出 `solved`。
- [x] analogy-only evidence 不能输出 `ready_for_guided_rerun`，只能输出 review/critique。
- [x] route anchor 被误写为单步 rxn 时 audit 拒绝。

### P1a：RouteStatus + Case Trace 工程化

目标：

```text
把 P0 route package 的判断纳入统一 RouteStatus 和 case trace。
```

交付物：

- [x] 定义最小 `RouteStatus` enum：`solved`、`semisynthesis_closed`、
  `partial_anchor`、`fake_closed_rejected`、`unresolved`。
- [x] 定义最小 artifact 基类字段和 append-only case trace。
- [x] 将 P0 package、evidence cards、candidate cards、validation、summary 写入 case bundle。
- [x] fake closure / unresolved core 生成 `FailureEvent`。

验收命令：

```bash
pytest -q tests/test_case_blackboard.py tests/test_product_route_feasibility_audit.py tests/test_smiles_first_workflow.py
```

必须通过的验收：

- [x] P0 package 能导出/导入 case bundle。
- [x] route package audit result 能转成 RouteStatus。
- [x] fake closure 和 literature gap 可追踪。

### P1b：StrategicOperator + Bounded Guided Rerun

目标：

```text
validated evidence 可以编译成有限 policy，并触发一次可追踪 guided ChemEnzy rerun。
```

交付物：

- [x] `StrategicOperator` schema。
- [x] `ChemEnzySearchPolicy` 最小字段：terminal blacklist、anchor whitelist、
  preferred subgoal、source budget、rerun reason、budget。
- [x] policy compiler。
- [x] guided run 记录 policy id、evidence refs、rerun budget。
- [x] bounded rerun 失败时输出 `unresolved`，不能强行 solved。

验收命令：

```bash
pytest -q tests/test_chem_enzy_baseline.py tests/test_route_tree_planner.py tests/test_product_route_feasibility_audit.py
```

必须通过的验收：

- [x] baseline run 不带 policy 时完全保持默认。
- [x] guided run 有 policy trace。
- [x] policy 不能包含 raw reaction injection。
- [x] rerun 没有改善时不会继续无限 rerun。

### P1c：Compiled Terminal Judge

目标：

```text
把 P0/P1 文献 anchor、terminal blacklist 和 fake-close guard 编译为搜索期轻量规则。
```

交付物：

- [x] 定义最小 `JudgePolicy`：terminal blacklist、anchor whitelist、stock tier rule、
  same-scaffold risk threshold、material sanity mode。
- [x] 实现 compiled judge：`accept`、`reject`、`defer`。
- [x] 接入 proposal gate 或 terminal closure 前的轻量路径。
- [x] rejected terminal 写入 route audit trace。
- [x] 保证 policy 为空时 baseline 行为不变。

验收：

- [x] 空 policy 不改变现有 route-tree / ChemEnzy default。
- [x] terminal blacklist 能阻止 fake stock closure。
- [x] validated anchor whitelist 能让 semisynthesis anchor 以 anchor 模式通过。
- [x] 无 LLM、无 online LLM judge。

### P2：Enzyme-aware Bridge Integration

目标：

```text
把已有 bridge verifier v0、enzyme precedent retrieval、SP-v1 gate 收束成结构化 enzyme action 和 route-level audit。
```

交付物：

- [x] 结构化 enzyme action：substrate、product、EC、reaction center、precedent、
  verifier score、source evidence、cofactor/common-metabolite flags。
- [x] enzyme router/source budget：只在 bridge/EC/evidence 支持时增加酶源预算。
- [x] SP-v1 gate 结果进入 action metadata 和 route audit。
- [x] route-level enzyme status：validated / generic_ec_only / rejected / unknown。
- [x] P5 partial evidence routes 只能作为 partial evidence，不显示为 production solved。

验收命令：

```bash
pytest -q tests/test_enzyme_sp_verifier_gate.py tests/test_enzyme_candidate_quality_audit.py tests/test_native_vs_enhanced_route_benchmark.py
```

必须通过的验收：

- [x] 无 bridge/EC trigger 的负例不打开 enzyme precedent floor。
- [x] SP-v1 reject 的 enzyme action 不进入 selected route。
- [x] generic EC-only route 不能是 validated enzyme step。
- [x] route audit 区分 enzyme proposal source 和 post-hoc EC annotation。

### P3：Controlled Worker + Evidence-gated Evolution

目标：

```text
最后再接 Codex Worker 和 evolution manager。Worker 只产出 artifact draft，不直接影响 ChemEnzy。
```

交付物：

- [x] `WorkerTask` schema 和 mock worker。
- [x] Codex Worker wrapper：budget、timeout、allowed tools、artifact output。
- [x] Worker output validator。
- [x] candidate / shadow / staging / production KB 分层。
- [x] benchmark gate 和 rollback。

必须通过的验收：

- [x] Worker 不能直接写 production KB。
- [x] Worker 不能直接标记 solved。
- [x] Worker 不能把 raw reaction 写入 route tree。
- [x] 只有 validated artifact 可以进入 policy compiler。
- [x] benchmark 失败阻止 promotion。

---

## A. 已吸收的现有实现与边界

后续实现不能从零假设。以下阶段性产物已经被吸收到本 checklist 的验收语境中：

- ChemEnzy native search 是默认 inner engine；旧 CCTS/ranker/block scorer/value
  line 只作复现或 diagnostic。
- Bridge pack v0 已有 P0 级训练包：67,209 positives、500,000 hard negatives、
  456,579 train rows、56,839 valid rows、53,791 test rows。
- Bridge verifier v0 已训练并通过 stress/leakage audit；route-search gate 推荐
  precision-biased threshold `0.8410`，自动接受强过滤 threshold `0.9331`。
- Enzyme SP-v1 verifier 已有 substrate-product-EC 三元组数据和 LightGBM gate；
  默认 gate threshold 来自训练产物 `0.36331207712759417`。
- Enzyme precedent retrieval 已能从 109,857 条 enzyme reaction precedents 生成
  proposal，但必须由 bridge/EC trigger、SP-v1、artifact gate 共同约束。
- Statin depth-20 material-gate iteration 证明 first-step material gate 能避免
  明显 scaffold/material artifact，同时保留 condition/protecting-group transfer
  warning；所有路线仍需 condition audit，不能当作可执行工艺。
- Product / Route Audit TODO 已整合为 `route_auditor` 和 `RouteStatus` 的验收输入：
  target match、step validity、terminal/stock、route mode、enzyme step、evidence、
  condition、FailureEvent、StrategicIntervention、Final RouteStatus 都是必填维度。

仍然不能宣称的内容：

- 不能宣称 P5 已产生 stock-closed production-grade chemo-enzymatic routes。
- 不能把 EC annotation 当作 enzyme validation。
- 不能把 weak-label verifier 指标当作专家真值。
- 不能把 high-similarity bridge 或 `forward_surrogate` 当作真实文献实验步骤。
- 不能把 AiZ/RetroKNN/enzyme precedent 全节点常开；必须由 gate/router 控制。

---

## 0. 交付物总览

本节及后续 `Case Blackboard`、`Codex Controller`、`ChemEnzy policy compiler`、
`Compiled Judge`、`Route Auditor`、`Condition Agent`、`Evolution Manager` 等章节是
P1+ 长期 backlog 和最终架构清单。当前 P0 开发不得跳过前面的
`当前 P0 开发板（直接开工版）` 和 `开发启动交付清单`，也不得让这些长期模块
阻塞 SMILES-first 文献检索 / 高级中间体拆解闭环。

### 0.1 必须新增或整理的主模块

- [x] `case_blackboard`：Case Blackboard 与 artifact store。
- [x] `codex_controller`：Codex Chief Chemist Controller。
- [x] `codex_worker`：Codex Research Worker wrapper。
- [x] `artifact_schemas`：typed artifact schemas。
- [x] `artifact_validators`：artifact validators。
- [x] `chem_enzy_consumption`：artifact -> ChemEnzy policy compiler。
- [x] `compiled_judge`：Compiled Search-Time Judge / deterministic gates。
- [x] `route_auditor`：最终 Route Auditor。
- [x] `literature_tools`：实时文献检索工具入口。
- [x] `evidence_layer`：Evidence extraction / validation。
- [x] `strategic_disconnection_miner`：文献战略断键挖掘。
- [x] `condition_agent`：Condition design / condition audit。
- [x] `evolution_manager`：candidate / shadow / benchmark gate。
- [x] `trace_store`：run trace、worker trace、audit trace。
- [x] `benchmarks`：P0/P1/P2/P3 regression benchmark。

### 0.2 必须改造的现有模块

- [x] `chem_enzy_adapter`：接收 compiled ChemEnzySearchPolicy。
- [x] `chem_enzy_native_chemical_plugin`：接收 compiled source / gate policy。
- [x] `chem_enzy_native_enzyme_plugin`：升级为结构化 enzyme action 支持。
- [x] `proposal_gate`：接入 compiled terminal / step gate。
- [x] `route_plausibility`：对齐新的 RouteStatus / AuditReport。
- [x] `enzyme_step_audit`：对齐 enzyme step status。
- [x] `product_route_feasibility_audit`：对齐最终 Product / Route Audit TODO。
- [x] web / CLI 输出：显示 RouteStatus、audit、FailureEvent、rerun history。

---

## 1. Case Blackboard

目标：为每个目标产物建立统一 case state。所有 Codex、工具、ChemEnzy、validator 和 auditor 只读写 blackboard artifact。

### 1.1 数据结构

- [x] 定义 `CaseBlackboard`。
- [x] 定义 `CaseId`、`RunId`、`ArtifactId`、`TraceId`。
- [x] 支持 target、structure profile、evidence、ChemEnzy runs、route candidates、failure events、strategic operators、conditions、audit reports、evolution candidates。
- [x] 每个 artifact 记录 source、created_at、validator status、parent refs。
- [x] 支持 append-only 写入。
- [x] 支持 artifact rejection record。

### 1.2 行为

- [x] 支持创建 case。
- [x] 支持读取当前 case summary。
- [x] 支持按 artifact type 查询。
- [x] 支持按 route id / step id / molecule id 查询。
- [x] 支持导出 JSON。
- [x] 支持从 JSON 恢复。

### 1.3 测试

- [x] 单测：创建 case。
- [x] 单测：追加 artifact。
- [x] 单测：validator reject 后不能作为 accepted artifact 查询。
- [x] 单测：JSON round-trip。
- [x] 单测：parent refs 保留。

### 1.4 验收

- [x] 同一个 case 的所有事件都有 trace id。
- [x] 关键事实不依赖自然语言上下文保存。
- [x] failed / rejected artifact 可追踪。

---

## 2. Typed Artifact Schemas

目标：所有 Codex / tool 输出都落到结构化 artifact。自然语言只能作为 explanation 字段，不能作为决策载体。

### 2.1 必须支持的 artifact

- [x] `TargetResolution`。
- [x] `StructureProfile`。
- [x] `TargetTriage`。
- [x] `ResearchReport`。
- [x] `EvidenceCard`。
- [x] `StrategicDisconnectionCard`。
- [x] `FailureEvent`。
- [x] `FailureDiagnosis`。
- [x] `StrategicOperator`。
- [x] `ChemEnzySearchPolicy`。
- [x] `JudgePolicy`。
- [x] `ConditionCandidate`。
- [x] `RouteAuditReport`。
- [x] `RouteStatus`。
- [x] `EvolutionCandidate`。
- [x] `WorkerRunRecord`。

### 2.2 Schema 规则

- [x] 每个 artifact 有 `schema_version`。
- [x] 每个 artifact 有 `artifact_id`。
- [x] 每个 artifact 有 `case_id`。
- [x] 每个 artifact 有 `source`。
- [x] 每个 artifact 有 `evidence_refs` 或 `input_refs`。
- [x] 每个 artifact 有 `validation_status`。
- [x] 每个 artifact 可序列化为 JSON。

### 2.3 测试

- [x] 单测：所有 artifact 可构造。
- [x] 单测：必填字段缺失时报错。
- [x] 单测：JSON round-trip。
- [x] 单测：schema version 固定。

### 2.4 验收

- [x] Codex Worker 输出不得绕过 schema。
- [x] ChemEnzy Consumption Layer 只接受 validated artifact。

---

## 3. Artifact Validators

目标：Codex 产出的草案必须被 deterministic validator 校验后才能影响 ChemEnzy。

### 3.1 Target / Structure validators

- [x] 检查 SMILES 可解析。
- [x] 检查 InChIKey / canonical SMILES 一致性。
- [x] 检查 stereochemistry status。
- [x] 标记 target ambiguity。

### 3.2 Evidence validators

- [x] 检查 source type。
- [x] 检查 target relation。
- [x] 检查 exact / analog / precursor / same scaffold 区分。
- [x] 检查 reaction role assignment。
- [x] 检查 atom mapping 或 reaction center 可检查性。
- [x] 检查 condition 信息是否可归一化。
- [x] 只有 validated EvidenceCard 可进入 ChemEnzy policy compiler。

### 3.3 StrategicOperator validators

- [x] 检查引用的 StrategicDisconnectionCard 是否已验证。
- [x] 检查 evidence refs 是否存在且已验证。
- [x] 检查 anchor whitelist 是否有证据。
- [x] 检查 terminal blacklist 是否有明确 reason。
- [x] 检查 rerun budget 是否有限。
- [x] 检查是否包含非法 raw reaction injection。

### 3.4 Condition validators

- [x] 检查 condition source type。
- [x] 区分 exact、analog、template、model-only、unknown。
- [x] 检查危险或不兼容标记。
- [x] 条件缺失时生成 condition gap。

### 3.5 Audit validators

- [x] 检查 RouteStatus 必填。
- [x] 检查 solved 必须有 stock audit。
- [x] 检查 semisynthesis_closed 必须有 anchor evidence。
- [x] 检查 fake_closed_rejected 必须有 rejected terminal / step。
- [x] 检查 unresolved 必须有 reason。

### 3.6 测试

- [x] 单测：invalid evidence 被拒绝。
- [x] 单测：无 evidence anchor 被拒绝。
- [x] 单测：超预算 rerun policy 被拒绝。
- [x] 单测：solved 无 stock audit 被拒绝。
- [x] 单测：semisynthesis_closed 无 evidence 被拒绝。

---

## 4. Codex Chief Chemist Controller

目标：实现 episode-level 控制器。它观察 blackboard，选择受控 action，调度 ChemEnzy、Codex Worker、工具和 auditor。

### 4.1 Action space

- [x] 实现 `RESOLVE_TARGET`。
- [x] 实现 `PROFILE_STRUCTURE`。
- [x] 实现 `RUN_BASELINE_CHEMENZY`。
- [x] 实现 `AUDIT_ROUTE`。
- [x] 实现 `DIAGNOSE_FAILURE`。
- [x] 实现 `RESEARCH_TARGET`。
- [x] 实现 `RESEARCH_STUCK_NODE`。
- [x] 实现 `EXTRACT_EVIDENCE`。
- [x] 实现 `VALIDATE_EVIDENCE`。
- [x] 实现 `COMPILE_STRATEGIC_OPERATOR`。
- [x] 实现 `RUN_GUIDED_CHEMENZY`。
- [x] 实现 `DESIGN_CONDITIONS`。
- [x] 实现 `SUBMIT_EVOLUTION_CANDIDATE`。
- [x] 实现 final actions。

### 4.2 Controller loop

- [x] 实现 `observe(case)`。
- [x] 实现 `decide_next_action(observation)`。
- [x] 实现 `execute_action(action)`。
- [x] 实现 `validate_artifact(artifact)`。
- [x] 实现 `update_blackboard(result)`。
- [x] 实现 `stop_or_continue(case)`。

### 4.3 Budget

- [x] max ChemEnzy runs。
- [x] max Codex Worker runs。
- [x] max literature rounds。
- [x] max stuck nodes。
- [x] max wall time。
- [x] max tool calls。
- [x] max reruns without improvement。

### 4.4 Guardrails

- [x] Controller 不能调用 ChemEnzy inner loop step-level hooks。
- [x] Controller 不能让 LLM rerank candidates。
- [x] Controller 不能让 LLM judge proposals online。
- [x] Controller 不能把 raw LLM reaction 写入 route tree。
- [x] Controller 不能直接写 production KB。

### 4.5 测试

- [x] 单测：简单目标走 baseline。
- [x] 单测：fake closure 后触发 audit / diagnosis。
- [x] 单测：无新 evidence 时不 rerun。
- [x] 单测：预算耗尽输出 unresolved。
- [x] 单测：非法 action 被拒绝。

---

## 5. Codex Research Worker Wrapper

目标：把 spawn Codex 进程封装成可预算、可追踪、可校验的 research worker。

### 5.1 Worker request

- [x] 定义 `WorkerTask`。
- [x] 包含 case id、task type、input refs、allowed tools、budget、required artifact type。
- [x] 支持 target research。
- [x] 支持 stuck-node research。
- [x] 支持 strategic disconnection mining。
- [x] 支持 route audit research。
- [x] 支持 condition research。
- [x] 支持 evolution candidate research。

### 5.2 Worker execution

- [x] 实现 Codex process spawn。
- [x] 捕获 stdout / stderr / exit code。
- [x] 捕获 tool calls。
- [x] 设置 timeout。
- [x] 设置 max output size。
- [x] 设置 allowed working directory。
- [x] 支持 dry-run / mock worker。

### 5.3 Worker output

- [x] 输出必须解析为 typed artifact。
- [x] 支持 ResearchReport。
- [x] 支持 EvidenceCard draft。
- [x] 支持 StrategicDisconnectionCard draft。
- [x] 支持 FailureDiagnosis。
- [x] 支持 StrategicOperator draft。
- [x] 支持 ConditionCandidate draft。
- [x] 支持 AuditReport draft。
- [x] 支持 EvolutionCandidate draft。

### 5.4 Worker safety

- [x] Worker 不能直接修改 ChemEnzy route tree。
- [x] Worker 不能直接写 production KB。
- [x] Worker 不能直接标记 final solved。
- [x] Worker 输出必须进入 validator。
- [x] Worker run 必须写 trace。

### 5.5 测试

- [x] 单测：mock worker 成功返回 artifact。
- [x] 单测：invalid JSON 输出被拒绝。
- [x] 单测：timeout 被记录。
- [x] 单测：超预算被拒绝。
- [x] 单测：worker 输出不经 validator 不可被 consumption layer 使用。

---

## 6. ChemEnzy Function-Call Consumption Layer

目标：把 validated artifact 编译成 ChemEnzy 可执行配置。

### 6.1 Policy compiler

- [x] `TargetTriage -> search intent`。
- [x] `StructureProfile -> stock risk / source hints`。
- [x] `EvidenceCard -> anchor whitelist / terminal blacklist`。
- [x] `StrategicDisconnectionCard -> subgoal / disconnection guidance`。
- [x] `StrategicOperator -> ChemEnzySearchPolicy`。
- [x] `FailureDiagnosis -> bounded rerun config`。
- [x] `JudgePolicy -> compiled deterministic gate config`。
- [x] `ConditionCandidate -> route annotation policy`。

### 6.2 ChemEnzySearchPolicy

- [x] mode: baseline / guided / literature-assisted / stuck-node rerun。
- [x] route mode prior。
- [x] strategic subgoals。
- [x] preferred disconnection hints。
- [x] protected core patterns。
- [x] source weights。
- [x] source allow / deny。
- [x] anchor whitelist。
- [x] terminal blacklist。
- [x] strict stock closure。
- [x] plugin config。
- [x] budget config。
- [x] rerun reason。

### 6.3 Adapter integration

- [x] `chem_enzy_adapter` 接收 policy。
- [x] policy 编译到 `RouteSearchConfig.search_flags`。
- [x] policy 进入 runtime search flags。
- [x] policy 进入 native chemical plugin config。
- [x] policy 进入 native enzyme plugin config。
- [x] policy 进入 proposal gate / compiled judge。
- [x] run metadata 记录 policy id 和 evidence refs。

### 6.4 测试

- [x] 单测：baseline policy 不改变默认 ChemEnzy 行为。
- [x] 单测：terminal blacklist 被传入 judge。
- [x] 单测：anchor whitelist 被传入 judge。
- [x] 单测：plugin config 被正确传递。
- [x] 单测：invalid artifact 不能编译 policy。

---

## 7. Compiled Search-Time Judge

目标：在 ChemEnzy 搜索期高速执行 deterministic gate，拦截假闭合和无效 terminal。

### 7.1 Terminal gates

- [x] simple commercial stock accept。
- [x] validated anchor accept with route mode。
- [x] same-scaffold high similarity reject。
- [x] no-complexity-drop terminal reject。
- [x] product-like terminal reject。
- [x] suspicious stock tier defer / reject。

### 7.2 Step gates

- [x] invalid SMILES reject。
- [x] material sanity reject。
- [x] impossible heavy atom gain reject。
- [x] same-scaffold loop reject。
- [x] unsupported enzyme core disconnection reject。

### 7.3 Judge output

- [x] decision: accept / reject / defer。
- [x] reason code。
- [x] severity。
- [x] affected molecule / step。
- [x] required action。
- [x] evidence refs。

### 7.4 Integration

- [x] 接入 proposal gate。
- [x] 接入 terminal closure 前。
- [x] 接入 route audit trace。
- [x] 记录 rejected terminals。

### 7.5 测试

- [x] same-scaffold fake stock 被拒绝。
- [x] simple stock 被接受。
- [x] validated semisynthesis anchor 被接受为 anchor。
- [x] no-complexity-drop terminal 被拒绝。
- [x] unsupported enzyme core break 被拒绝。

---

## 8. Route Auditor

目标：对 ChemEnzy route result 给出最终真实性判定。

### 8.1 Audit dimensions

- [x] target match audit。
- [x] step structural audit。
- [x] stock audit。
- [x] route mode audit。
- [x] enzyme step audit。
- [x] evidence audit。
- [x] condition audit。
- [x] fake closure audit。
- [x] unresolved core audit。

### 8.2 RouteStatus

- [x] solved。
- [x] semisynthesis_closed。
- [x] partial_anchor。
- [x] fake_closed_rejected。
- [x] unresolved。

### 8.3 Report

- [x] top route summary。
- [x] rejected route summary。
- [x] rejected terminal list。
- [x] FailureEvent list。
- [x] route mode explanation。
- [x] next action。

### 8.4 测试

- [x] solved 需要 stock audit 通过。
- [x] semisynthesis_closed 需要 anchor evidence。
- [x] fake closure 输出 fake_closed_rejected。
- [x] no route 输出 unresolved。
- [x] condition gap 不误报 solved confidence high。

---

## 9. Evidence Layer

目标：文献和检索结果必须先变成 validated evidence。

### 9.1 Literature tools

- [x] target exact search。
- [x] semisynthesis search。
- [x] isolation / fermentation search。
- [x] analog / precursor search。
- [x] strategic disconnection search。
- [x] named total synthesis route search。
- [x] latest synthesis / semisynthesis route search。
- [x] stuck-node transformation search。
- [x] condition precedent search。
- [x] internal docs / repo search。

### 9.2 Evidence extraction

- [x] source metadata。
- [x] claim type。
- [x] target relation。
- [x] reaction extraction。
- [x] condition extraction。
- [x] route mode classification。
- [x] strategic disconnection extraction。
- [x] confidence。

### 9.3 Evidence validation

- [x] exact / analog / precursor 区分。
- [x] structure checked。
- [x] role assignment checked。
- [x] reaction center / atom mapping checked。
- [x] condition normalized。
- [x] usable_for_search flag。

### 9.4 测试

- [x] analog evidence 不被当成 exact target evidence。
- [x] isolation evidence 不被当成 synthesis route solved。
- [x] unvalidated literature hit 不能进入 policy。
- [x] validated anchor 可以进入 whitelist。

---

## 10. Strategic Disconnection Miner

目标：让 Codex 从天然产物及类似物文献路线中抽取“战略断键”，弥补模板和单步逆合成对高级天然产物核心骨架断开的不足。

### 10.1 输入

- [x] target structure。
- [x] target scaffold / core。
- [x] close analogs。
- [x] ChemEnzy stuck node。
- [x] failed disconnection。
- [x] validated EvidenceCard。
- [x] literature hits。

### 10.2 抽取内容

- [x] 文献路线的关键 retrosynthetic disconnection。
- [x] 对应 forward key step。
- [x] 断开的 bond / substructure。
- [x] 产生的 strategic subgoal。
- [x] semisynthesis anchor。
- [x] protected core / preserved scaffold。
- [x] disconnection rationale。
- [x] exact / analog applicability。
- [x] evidence refs。

### 10.3 StrategicDisconnectionCard

- [x] target relation: exact / analog / precursor / same scaffold。
- [x] route claim: total synthesis / semisynthesis / biosynthesis-inspired / failed。
- [x] disconnection type。
- [x] product-side pattern。
- [x] precursor-side pattern。
- [x] strategic subgoal。
- [x] anchor candidate。
- [x] forbidden fake terminal implication。
- [x] confidence。
- [x] usable_for_search。

### 10.4 Validation

- [x] Evidence refs validated。
- [x] 断键与文献 route scheme 一致。
- [x] 断键能映射到目标或 close analog。
- [x] exact 和 analog 不能混淆。
- [x] 断键必须带来复杂度下降或明确 anchor。
- [x] 不能把 isolation-only claim 当成 synthetic disconnection。
- [x] failed route 只能作为 negative guidance。

### 10.5 编译到 ChemEnzy

- [x] 生成 strategic subgoal hint。
- [x] 生成 preferred disconnection hint。
- [x] 生成 anchor whitelist candidate。
- [x] 生成 terminal blacklist / protected core rule。
- [x] 生成 stuck-node rerun target。
- [x] 生成 source promotion / demotion。
- [x] 写入 ChemEnzySearchPolicy。

### 10.6 测试

- [x] exact target 文献断键可生成 usable StrategicDisconnectionCard。
- [x] analog 文献断键不会被标成 exact。
- [x] isolation 文献不会生成 synthesis disconnection。
- [x] failed route 生成 negative guidance。
- [x] validated disconnection 能编译进 ChemEnzySearchPolicy。
- [x] invalid disconnection 不能影响 ChemEnzy。

---

## 11. Condition Agent / Condition Audit

目标：路线闭合后给出条件候选、条件证据和 feasibility status。

### 10.1 Condition sources

- [x] exact literature condition。
- [x] analog literature condition。
- [x] template / cluster condition。
- [x] model-predicted condition。
- [x] condition_unknown。

### 10.2 ConditionCandidate

- [x] step id。
- [x] source type。
- [x] reagent / catalyst / enzyme。
- [x] solvent。
- [x] temperature。
- [x] pH / buffer。
- [x] atmosphere。
- [x] evidence refs。
- [x] risk flags。
- [x] confidence。

### 10.3 Route feasibility

- [x] step-level condition status。
- [x] route-level condition gap。
- [x] enzyme / chemical transition risk。
- [x] cofactor gap。
- [x] safety / incompatibility flags。

### 10.4 测试

- [x] exact condition 优先于 model prediction。
- [x] unknown condition 生成 condition_gap。
- [x] risky condition 影响 route audit。
- [x] condition status 出现在 final report。

---

## 12. Evolution Manager

目标：新知识只能进入 candidate / shadow，经 benchmark gate 后再晋升。

### 11.1 Candidate types

- [x] ReactionRecordCandidate。
- [x] TemplateCandidate。
- [x] ConditionCandidate。
- [x] AnchorCandidate。
- [x] ControllerPolicyTrace。

### 11.2 Promotion flow

- [x] candidate KB。
- [x] shadow KB。
- [x] benchmark gate。
- [x] staging KB。
- [x] production KB。
- [x] rollback。

### 11.3 Gates

- [x] evidence source credible。
- [x] structure validated。
- [x] atom mapping / role assignment checked。
- [x] template replay passes。
- [x] overgeneralization check。
- [x] benchmark true solved rate non-decreasing。
- [x] fake closure rate non-increasing。
- [x] condition quality non-decreasing。

### 11.4 测试

- [x] target run 不能写 production。
- [x] failed benchmark blocks promotion。
- [x] rollback restores previous KB。
- [x] fake closure regression blocks promotion。

---

## 13. CLI / API / Web 输出

目标：让用户和后续模型能看到 case、trace、audit、RouteStatus 和 rerun history。

### 12.1 CLI

- [x] 新增 run case command。
- [x] 新增 audit route command。
- [x] 新增 inspect blackboard command。
- [x] 新增 worker trace command。
- [x] 新增 rerun with policy command。

### 12.2 API

- [x] create case。
- [x] run baseline ChemEnzy。
- [x] run guided ChemEnzy。
- [x] get blackboard。
- [x] get route audit。
- [x] get worker trace。
- [x] get final report。

### 12.3 Web

- [x] 显示 RouteStatus。
- [x] 显示 stock audit。
- [x] 显示 fake closure rejected terminals。
- [x] 显示 evidence refs。
- [x] 显示 condition status。
- [x] 显示 rerun history。

### 12.4 测试

- [x] CLI smoke。
- [x] API smoke。
- [x] Web payload schema test。
- [x] final report contains RouteStatus。

---

## 14. Benchmark / Regression

目标：所有改动必须能证明没有破坏 ChemEnzy baseline，并降低 fake solved 风险。

### 13.1 Benchmark sets

- [x] simple stock-closable targets。
- [x] medchem-like targets。
- [x] NP-like high fake-close-risk targets。
- [x] semisynthesis anchor targets。
- [x] literature-known strategic disconnection targets。
- [x] analog-disconnection transfer controls。
- [x] enzyme-assisted targets。
- [x] condition-known reactions。
- [x] unresolved / no-route controls。

### 13.2 Metrics

- [x] baseline true solved rate。
- [x] guided true solved rate。
- [x] fake closure rejection precision。
- [x] unresolved correctness。
- [x] route mode classification accuracy。
- [x] strategic disconnection extraction precision。
- [x] strategic disconnection policy usefulness。
- [x] condition coverage。
- [x] Codex Worker calls per case。
- [x] ChemEnzy runtime overhead。
- [x] benchmark regression count。

### 13.3 Hard gates

- [x] Simple target baseline latency does not regress materially。
- [x] No LLM call occurs inside ChemEnzy inner loop。
- [x] Fake closure rate decreases or stays flat。
- [x] Solved claims require stock audit。
- [x] Semisynthesis claims require anchor evidence。

---

## 15. Final Integration Acceptance

最终交付前逐项确认：

- [x] simple target 可以直接 baseline，无 Codex Worker。
- [x] complex target 可以 literature-assisted。
- [x] baseline fake closure 能生成 FailureEvent。
- [x] Codex Worker 能针对 stuck node 产出 ResearchReport。
- [x] Codex Worker 能从文献产出 StrategicDisconnectionCard。
- [x] EvidenceCard 必须 validated 后进入 policy。
- [x] StrategicDisconnectionCard 必须 validated 后进入 policy。
- [x] StrategicOperator 能编译为 ChemEnzySearchPolicy。
- [x] guided ChemEnzy run 记录 policy id。
- [x] compiled judge 拦截假 terminal。
- [x] Route Auditor 输出 RouteStatus。
- [x] Condition Agent 输出 condition status。
- [x] Evolution Candidate 只进入 candidate / shadow。
- [x] benchmark gate 阻止坏模板晋升。
- [x] final report 包含 audit、evidence、condition、rerun history。
- [x] 所有核心路径有测试。
- [x] 所有 run 可追踪、可回放、可导出。

---

## 16. 后续模型执行规则

后续模型开始代码改造时必须遵守：

```text
每次只改一个模块或一组紧密相关模块。
每次改动后更新本 checklist 对应 checkbox。
每次新增 artifact 必须有 schema、validator、round-trip test。
每次接入 ChemEnzy 必须证明 baseline default 不变。
每次引入 Codex Worker 必须证明不会进入 ChemEnzy inner loop。
每次声明 solved 必须有 RouteStatus 和 stock audit。
```

每轮交付必须报告：

```text
implemented:
  - checked items
tests:
  - commands
  - pass/fail
risks:
  - remaining gaps
next:
  - next checklist items
```
