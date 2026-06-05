# AutoPlanner Project Master Table

日期：2026-06-04

用途：本文件是当前项目总表。它把
`docs/EvoChemEnzy_Code_Delivery_Checklist_2026-06-03.md`、
`docs/LITERATURE_TO_EXECUTABLE_TEMPLATE_CHECKLIST_2026-06-04.md` 和
`docs/SMILES_FIRST_LITERATURE_STRATEGIC_WORKFLOW_2026-06-03.md`
合并成一个执行视图。后续不能只看 checkbox 宣称完成；每个能力必须同时满足：

1. 有真实执行入口。
2. 有 trace / artifact / backend 证据。
3. 有 deterministic validator 或 audit gate。
4. 有测试或可复现验收命令。

## 硬边界

```text
No LLM in ChemEnzy inner loop.
No online LLM rerank.
No online LLM proposal judge.
No raw LLM reaction injection.
No direct production KB write from target run.
No "Codex/API worker is used" claim without WorkerRunRecord backend plus command/provider evidence.
```

## 当前总体结论

| Area | Status | 真实后端/实现 | 主要证据 | 剩余缺口 |
|---|---:|---|---|---|
| P0 SMILES-first literature route package | Complete | `api_json` worker default + explicit `local/manual` deterministic modes | `run_smiles_first_literature_workflow.py`, P0 tests | 真实 provider 运行需要非空 `key.txt` |
| Literature-to-executable template plugin | Complete for MVP | RDKit retron/applicability + literature one-step plugin | L0-L7 checklist, template/plugin tests | multi-step route segment 由 `literature_segments.py` 覆盖 |
| Case trace / blackboard / RouteStatus | Complete | append-only artifacts + worker output import | `case_trace.py`, `case_blackboard.py`, controller tests | none for master scope |
| Codex Worker wrapper | Complete | Codex CLI subprocess + direct `api_json` provider calls | `WorkerRunRecord.backend`, `command`, provider metadata, tests | 外部 provider 可用性需运行时确认 |
| Codex Controller | Complete | bounded action loop with research/validate/compile/rerun/audit handlers | `codex_controller.py`, controller loop tests | none for master scope |
| Evidence validation | Complete | deterministic typed artifact/evidence/segment validators | `evidence_cards.py`, `artifact_validators.py`, segment tests | none for master scope |
| Strategic disconnection mining | Complete | evidence cards -> disconnection cards -> policy | `strategic_disconnection_miner.py`, `chem_enzy_policy.py` | none for master scope |
| Guided ChemEnzy policy | Complete | compiled `ChemEnzySearchPolicy` in search flags + rerun history | CLI/Web guided policy tests | none for master scope |
| Compiled judge / route auditor | Complete | deterministic terminal/route/condition gates | `terminal_judge.py`, `route_auditor.py` | none for master scope |
| Condition agent | Complete | condition candidate schema, extraction validation, audit downgrade | `condition_agent.py`, route auditor tests | none for master scope |
| Enzyme bridge | Mostly complete | verifier/retrieval/gates | P2 tests and modules | production-quality enzyme validation remains guarded |
| Evolution manager | Complete for guarded scope | route segment/condition candidate promotion gates + rollback tests | `evolution_manager.py`, `test_evolution_hardening.py` | production promotion remains offline-gated |
| Web/CLI traceability | Complete | cases, audit, worker trace, guided policy, final report, artifact filters | Web/CLI smoke tests | none for master scope |
| Benchmarks/regression | Complete for master scope | P0/P1/P2/P3 + segment/condition/evolution tests | master acceptance command | none for master scope |

## Codex CLI Reality Gate

从 2026-06-04 起，凡是声称“调用 Codex”的路径必须满足以下任一条件：

| Gate | Required Evidence |
|---|---|
| CLI worker | `worker-trace` 输出 `backend: codex_cli` 且 `command` 以 `codex ... exec` 开头 |
| API JSON worker | `worker-trace` 输出 `backend: api_json`、provider/base URL/model metadata、status 和 output validation |
| Web worker | `/api/worker-trace` 的 `worker_trace.backend in {codex_cli, api_json}`，并显示 command/provider |
| Controller worker | action result 里 `worker_trace.backend in {codex_cli, api_json}`，并保留 validation status |
| Mock/dry-run | 必须显示 `backend: dry_run_mock`、`mock_output` 或 `default_mock`，不能写成真实 Codex run |
| Failure | 外部 provider timeout/auth/network failure 必须返回 `worker_error` / `timeout`，不能静默降级为 mock |

实现位置：

- `cascade_planner/agent/codex_worker.py`
- `cascade_planner/agent/codex_controller.py`
- `cascade_planner/agent/cli.py`
- `cascade_planner/web/app.py`

验收命令：

```bash
pytest -q tests/test_codex_worker_controller_evolution.py tests/test_agent_deepseek_key_guards.py tests/test_web_app.py
python -m py_compile cascade_planner/agent/codex_worker.py cascade_planner/agent/codex_controller.py cascade_planner/agent/cli.py cascade_planner/web/app.py
```

真实 Codex smoke 命令模板：

```bash
python -m cascade_planner.agent.cli worker-trace --task-json /path/to/non_dry_run_worker_task.json
```

合格输出必须含有：

```json
{
  "backend": "codex_cli",
  "command": ["codex", "...", "exec", "..."],
  "status": "accepted_draft"
}
```

如果 status 是 `worker_error` 或 `timeout`，说明真实 Codex 后端被调用但外部运行失败；
这不是 mock 成功，不能计入功能完成。

2026-06-04 Wellau/Codex CLI reality note:

- 此前本机会话环境测试过 `codex-cli 0.135.0`、`https://api.wellau.com/v1`、`gpt-5.5`；
  程序 `api_json` worker key 以 `key.txt` 为准。
- `/models`、`/chat/completions`、`/responses` 可达，基础 API 未断开。
- `codex exec` 普通短 prompt 能返回，但会多次 `Reconnecting...`。
- `codex exec --output-schema --output-last-message` worker smoke 在 180s 内发生
  `stream disconnected before completion: Upstream request failed`，带 search 和
  no-search 都复现。
- 结论：Codex CLI 保留为强 harness 后端，但 Wellau 下不能作为唯一稳定默认后端。
  复杂文献检索闭环已新增 `api_json` worker 后端，并继续经过同一 schema validator。

## Retrosynthesis Worker Key Policy

2026-06-04 closure decision:

- Chat/session credentials remain separate from program worker credentials.
- Literature/retrosynthesis `api_json` workers read the API key directly from
  `/root/autodl-tmp/AutoPlanner/key.txt` (`DEFAULT_RETROSYNTHESIS_KEY_FILE`).
- `api_json` worker key resolution does not fall back to
  `AUTOPLANNER_*_API_KEY`, `OPENAI_API_KEY`, or `DEEPSEEK_API_KEY`.
- Missing or empty `key.txt` returns `worker_error: missing API key for api_json worker`;
  it does not silently switch to mock, local curated data, or the chat key.
- Default worker provider is `https://api.wellau.com/v1` with `chat/completions`;
  Base URL/model/provider may still be configured by environment variables, but only the worker key is isolated.
- `/key.txt` is gitignored. At this closure pass the file exists but is empty, so a real external run will fail closed until it is populated.

## 阶段总表

| Phase | Goal | Status | Done | Missing | Next Acceptance |
|---|---|---:|---|---|---|
| P0a | SMILES profile + baseline frontier | Complete | target profile, frontier report, baseline/manual fallback | none for master scope | P0 pytest + py_compile |
| P0b | Worker/literature retrieval + evidence cards | Complete | `LiteratureSearchTask`, default `api_json`, trace persistence, local/manual explicit modes | runtime provider key must be supplied in `key.txt` | Non-mock worker creates traceable EvidenceCard artifact |
| P0c | Strategic candidates + hybrid route package | Complete | exact fragment retro, surrogate, anchor, validation, route map | route package still planning material only | Bufotalin-like package remains partial_anchor |
| P0 Guardrail | Prevent fake solved | Complete | route package audit, no stock audit no solved | none for P0 scope | solved claim requires stock audit |
| P1a | Case trace / RouteStatus | Complete | case bundle export/import, FailureEvent, worker append | none for master scope | Worker output -> typed artifact -> bundle append |
| P1b | StrategicOperator + guided rerun | Complete | policy compiler, guided config, rerun trace, controller rerun loop | none for master scope | Controller RESEARCH -> VALIDATE -> COMPILE -> RERUN smoke |
| P1c | Compiled terminal judge | Complete | deterministic accept/reject/defer gates, segment/condition audit hooks | none for master scope | fake terminal rejected in runtime trace |
| P1d | Codex condition adaptation | Complete | `ConditionCandidate`, condition worker mapping, audit downgrade tests | none for master scope | Codex/API condition worker draft + route audit downgrade |
| P1e | Literature route segments | Complete | `literature_segments.py`, cards, recursive unroll, mismatch downgrade tests | none for master scope | 3-step segment recursively expands and stops on mismatch |
| P2 | Enzyme-aware bridge | Mostly complete | bridge verifier, enzyme precedent, SP-v1 gates | expert-grade enzyme validation remains guarded | generic EC-only cannot be validated enzyme step |
| P3 | Controlled worker + evolution | Complete | worker schema, real Codex CLI/API JSON backends, validators, layered KB, artifact ingest | none for master scope | Worker artifact validated before policy consumption |
| P3b | Self-evo hardening | Complete | segment/condition candidate promotion and rollback benchmarks | production remains offline-gated | no target-run production promotion |
| L0-L7 | Literature executable template plugin | Complete for MVP | retron, applicability, instantiation, plugin, A/B benchmark | multi-step SI route extraction not included | plugin step visible in ChemEnzy route with audit trace |
| Web/CLI | User/API surfaces | Complete | cases, audit, worker trace, guided policy, final report, filters | none for master scope | `/api/worker-trace` shows non-mock backend evidence |
| Benchmark | Regression gates | Complete | P0/P1/P2/P3 plus segment, condition, self-evo tests | none for master scope | master acceptance command below |

## Ideal Retrosynthesizer Closure Checklist

目标：普通 case 由 ChemEnzy 稳定处理；只有复杂天然产物、高级同骨架 frontier、
native failed / fake closure / user-requested literature case 才进入文献检索。
理想状态不是让 LLM 替代 ChemEnzy，而是让 episode-level worker 产出受控 artifact，
再由 deterministic gates 编译成 ChemEnzy 可消费的策略或 external one-step source。

### R0：复杂性和失败分流

- [x] 定义 `LiteratureEscalationPolicy`。
- [x] 输入 native ChemEnzy result、route audit、frontier report、user objective。
- [x] native solved + audit passed 时不触发文献检索。
- [x] `native_failed`、`unclosed_route`、`fake_closure_risk`、
  `advanced_frontier_detected`、`route_audit_failed`、`user_requested_literature`
  时触发文献模式。
- [x] 记录 `escalation_reason`、`source_evidence`、`token_budget_class`。
- [x] Acceptance：phenolic glycoside/native-solved negative control 不进入 literature mode；
  Bufotalin-like same-scaffold frontier 进入 literature mode。

### R1：稳定 worker 后端

- [x] 新增 `api_json` worker 后端，直接调用 configured OpenAI-compatible
  `/responses` 或 `/chat/completions`。
- [x] `api_json` 输出必须转换成 `WorkerRunRecord`，字段包括 backend、provider、
  base URL fingerprint、model、status、elapsed、usage、output_validation。
- [x] Codex CLI backend 继续保留，但只作为强 harness；provider 断流不能静默降级 mock。
- [x] 默认复杂文献检索预算：普通复杂 case `80k tokens/target`，高复杂天然产物
  `150k tokens/target`，全文/SI 多轮检索 `400k+` 上限。
- [x] Acceptance：同一个 `WorkerTask` 能通过 `api_json` 产出 accepted draft；
  Codex CLI failure 记录为 rejected/worker_error 而不是 mock success。

### R2：真实文献检索到 EvidenceCard

- [x] `run-case --literature-backend codex|api_json|local|manual|pubmed|local_pubmed`。
- [x] 为 target/frontier 构造 `LiteratureSearchTask`，限制 query budget、
  allowed source types、required artifact type。
- [x] 持久化 `WorkerRunRecord` 到 case output。
- [x] 把 accepted draft 转换为 `EvidenceCard` / `ResearchReport` typed artifact。
- [x] 执行 `validate_typed_artifact()` 和 `validate_evidence_card()`。
- [x] accepted/rejected artifacts 都 append 到 case bundle。
- [x] Acceptance：non-mock worker 能生成 traceable EvidenceCard；
  unresolved literature gap 不生成战略候选。

### R3：Controller 闭环

- [x] 完成默认 action handlers：
  `EXTRACT_EVIDENCE`、`VALIDATE_EVIDENCE`、`DIAGNOSE_FAILURE`、
  `DESIGN_CONDITIONS`、`COMPILE_STRATEGIC_OPERATOR`、`RUN_GUIDED_CHEMENZY`。
- [x] `update_blackboard()` append worker outputs、validation records、rerun history。
- [x] Controller 流程固定为：
  `observe -> research -> validate -> mine disconnection -> compile policy -> guided rerun -> audit`。
- [x] Controller 不允许 raw reactions、online rerank、production KB write。
- [x] Acceptance：unresolved case 触发 worker，验证证据，编译 policy，发起 guided rerun，
  并用 RouteStatus 记录最终 unresolved/partial/solved。

### R4：LiteratureRouteSegment

- [x] 新增 `literature_segments.py`。
- [x] 定义 `LiteratureRouteSegmentCard` 和 `SegmentStepCandidate`。
- [x] 支持从文献/SI 抽取 2-5 步 route segment，逐步 unroll。
- [x] 每条 edge 必须通过 applicability、product reconstruction、atom accounting、
  condition/source gate。
- [x] mismatch、high-risk condition、budget exhausted、audit failed 时停止。
- [x] Acceptance：3-step exact segment 可递归展开；analog/mismatch segment 被降级；
  native-solved negative control 不因 segment 插件制造假提升。

### R5：Condition worker 和条件门控

- [x] 增加 condition research task mapping：route step -> `ConditionCandidate`。
- [x] exact condition 必须有 evidence refs；analog condition 必须标注 scope gap。
- [x] model-only condition 只能作为 feasibility hint，不能当作 executable procedure。
- [x] route audit 在 unknown/high-risk condition 上 downgrade confidence。
- [x] Acceptance：condition-rich exact/analog/model-only benchmark 输出 gap rate、
  risky flag precision、audit downgrade count。

### R6：证据到 ChemEnzy 执行桥

- [x] validated evidence/disconnection/condition 编译成 `ChemEnzySearchPolicy`。
- [x] validated executable literature template 可进入 `LiteratureOneStepPlugin`。
- [x] external one-step proposal 必须带 source、template id、evidence refs、
  `not_lab_procedure`、`requires_audit`。
- [x] ChemEnzy inner loop 只消费 deterministic validated proposal，不调用 LLM。
- [x] Acceptance：文献 template step 在 ChemEnzy route trace 中可见；
  reconstruction/audit 不通过时 proposal 被拒绝。

### R7：Evolution hardening

- [x] `EvolutionCandidate` 覆盖 route segments、segment steps、condition candidates。
- [x] target run 只能写 candidate/shadow，不能写 production。
- [x] offline replay gate 覆盖 segment promotion、condition quality、fake closure delta。
- [x] rollback tests 覆盖 bad segment、bad condition、bad evidence。
- [x] Acceptance：无 target-run production promotion；bad artifact promotion 可回滚且有 trace。

### R8：Master benchmark and UX

- [x] benchmark pack 包含 native solved、native failed、fake closure、segment exact、
  segment analog、segment mismatch、condition exact/analog/model-only、self-evo rollback。
- [x] Web/CLI 显示 worker backend、command/provider、status、validation reasons。
- [x] 对 mock/dry-run/default_mock 显示非真实后端警告。
- [x] Artifact browser 支持 worker traces、accepted/rejected artifacts、RouteStatus history。
- [x] Acceptance：master command 包含 P1d/P1e/P3b tests 后才允许声明 ideal-loop complete。

## Optimized Execution Order

1. **R0 + R1 first**：先做分流和稳定 `api_json` worker。没有稳定 worker，后续文献闭环都会被
   Codex CLI 长流不稳定拖住；没有分流，token 会浪费在 ChemEnzy 已能解决的 case 上。
2. **R2 next**：把真实文献检索落成 EvidenceCard / ResearchReport typed artifacts。
3. **R3**：关闭 controller loop，让 evidence 真正进入 policy/rerun/audit。
4. **R4 + R5**：补多步文献 segment 和 condition gates，这是复杂天然产物路线可信度的核心。
5. **R6**：让 validated literature artifact 进入 ChemEnzy one-step / policy consumption layer。
6. **R7 + R8**：最后做 evolution hardening、rollback、benchmark pack 和 UX。

## Closure Evidence

| Row | Implemented | Primary modules | Primary tests |
|---|---|---|---|
| A. Worker literature retrieval default | `run-case`/Web/script default to `api_json`; explicit `local/manual` remain deterministic; worker traces and typed evidence are persisted into case outputs | `smiles_first.py`, `codex_worker.py`, `cli.py`, `web/app.py`, `run_smiles_first_literature_workflow.py` | `test_smiles_first_workflow.py`, `test_agent_deepseek_key_guards.py`, `test_web_app.py` |
| B. Controller loop | bounded observe/research/validate/diagnose/condition/compile/rerun/audit loop; blackboard receives worker records, validations, rerun history | `codex_controller.py`, `case_blackboard.py`, `case_trace.py` | `test_codex_worker_controller_evolution.py`, `test_case_blackboard.py` |
| C. Worker condition extraction | condition candidates are typed, evidence-scoped, and audited with confidence downgrade on gaps/high risk | `condition_agent.py`, `route_auditor.py`, `artifact_validators.py` | `test_agent_route_auditor_condition.py`, `test_codex_worker_controller_evolution.py` |
| D. Literature route segments | segment cards, step candidates, recursive unroll, mismatch/high-risk/budget stop gates | `literature_segments.py`, `literature_one_step_plugin.py`, `route_auditor.py` | `test_literature_segments.py`, `test_literature_one_step_plugin.py` |
| E. Evolution hardening | route segment/condition/evidence candidates remain candidate/shadow unless offline gates pass; rollback tests cover bad artifacts | `evolution_manager.py` | `test_evolution_hardening.py`, `test_codex_worker_controller_evolution.py` |
| F. Web/CLI UX hardening | backend, command/provider, status, validation reasons, non-real warning, worker/rejected artifact filters | `cli.py`, `web/app.py` | `test_web_app.py`, `test_agent_deepseek_key_guards.py` |
| G. Master benchmark pack | master command includes SMILES-first, literature evidence/templates, route plausibility, blackboard, worker/controller, condition, segment, evolution, Web/CLI guards | `tests/` master set | master acceptance command below |

Remaining runtime caveat: external provider success is intentionally not claimed when
`/root/autodl-tmp/AutoPlanner/key.txt` is empty. The implemented behavior is fail-closed
`worker_error`, not fallback.

## Master Acceptance Command

Current targeted command:

```bash
pytest -q \
  tests/test_smiles_first_workflow.py \
  tests/test_literature_evidence_cards.py \
  tests/test_strategic_candidate_generation.py \
  tests/test_route_plausibility.py \
  tests/test_literature_template_cards.py \
  tests/test_template_applicability.py \
  tests/test_executable_template_validation.py \
  tests/test_literature_one_step_plugin.py \
  tests/test_literature_template_plugin_benchmark.py \
  tests/test_case_blackboard.py \
  tests/test_agent_schemas.py \
  tests/test_agent_artifact_contracts.py \
  tests/test_codex_worker_controller_evolution.py \
  tests/test_agent_route_auditor_condition.py \
  tests/test_literature_segments.py \
  tests/test_evolution_hardening.py \
  tests/test_agent_deepseek_key_guards.py \
  tests/test_web_app.py
```

P1d/P1e/P3b regression tests are now part of this command.

2026-06-04 closure run:

```text
pytest master command: 127 passed, 7 warnings
py_compile worker/controller/CLI/Web/segment/condition/evolution modules: passed
```

## Reporting Rule

Every future delivery must report:

```text
implemented:
  - exact code modules
  - exact checklist rows
worker_backend:
  - backend field from WorkerRunRecord
  - command prefix or provider metadata
  - status
tests:
  - commands
  - pass/fail
remaining:
  - missing rows from this master table
```
