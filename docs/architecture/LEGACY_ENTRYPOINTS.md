# 旧入口能力映射

更新：2026-07-26

本轮隔离与验证收尾见
[Legacy Cleanup Closeout](LEGACY_CLEANUP_CLOSEOUT_20260726.md)。

新 operator 只使用 `python -m cascade_planner`。旧脚本分为三类：

## 已移除包装器

| 旧入口 | 替代 |
| --- | --- |
| `run_autoplanner_web_waitress.py` | `python -m cascade_planner serve --server waitress` |
| `start_agent_workbench.ps1` | `python -m cascade_planner serve` 后访问 `/v4` |
| `scripts/archive/*` | Git 历史；不在当前树执行 |

## 冻结的 V3 campaign 兼容入口

这些脚本仅用于 P10 之前读取或重放已保存的 V3 运行，不再接收新架构功能：

实现与命令名统一归档在 `scripts/legacy/`；`scripts/` 根目录不再保留 V3 包装器。

主 CLI 已不再提供 combined Web surface。历史 combined UI 的唯一入口是
`python scripts/legacy/serve_combined_web.py`。

V3 recursive campaign 及其 scheduler/ledger/portfolio/acceptance 实现已从主线包目录迁入
`cascade_planner/legacy/orchestration_runtime/` 和
`cascade_planner/legacy/application_runtime/`。仓库内兼容调用必须使用这些显式路径。

旧 blackboard controller、action planners、blackboard events、tool dispatcher、runner 和 RouteForest
compiler 已迁入 `cascade_planner/legacy/harness_runtime/`。`cascade_planner/harness/` 仅保留 V4 仍复用的
无状态科学/渲染组件。

V3 proposal bus、Codex edge verification、parent-route proof、route objectives 与
target-side strategy 也已迁入 `cascade_planner/legacy/harness_runtime/`；旧
`cascade_planner.harness.*` 路径已删除，closeout 重放必须显式经过 legacy 命名空间。

旧 workflow plan/preflight/progress、analogical/process/recursive helpers、failure critic、
hypothesis closeout reports 和 legacy-to-V4 controller adapter 同样位于该 runtime。
这些模块只服务冻结 blackboard 调度，不属于 canonical V4 orchestration。
旧 selfEVO replay/memory 与 declarative tool registry/execution policy 也已并入该 runtime；
主线 evolution manager 不依赖这些 saved-run helper。
旧 selected-route parent proof 位于 `legacy/application_runtime/`，edge signature 位于
`legacy/routes_runtime/`，不可变 closeout revision 位于 `legacy/runtime/`。主线 `runtime`
包根不再导出历史 closeout API。
Codex-entry typed schemas 已迁入 `legacy/harness_runtime/schemas.py`，旧 advisory route-consensus
graph assembler 已迁入 `legacy/routes_runtime/graph.py`。主线 `cascade_planner.routes` 不再导出
graph assembly/frontier API。
旧 visual structure-chain validator 也已迁入 `legacy/harness_runtime/`；当前
`interfaces.visual_evidence` 不通过该验证器获取或提升证据。

blackboard route adapter、外部边 receipt 和 admitted-hyperedge journal 位于
`cascade_planner/legacy/routes_runtime/` 与 `cascade_planner/legacy/orchestration_runtime/`；canonical
`cascade_planner.routes` 不再导出 blackboard rebuild。

- `legacy/run_codex_entry_agentic_blackboard.py`
- `legacy/run_codex_entry_controller.py`
- `legacy/resume_agentic_blackboard.py`
- `legacy/refresh_agentic_closeout_artifacts.py`
- `legacy/evaluate_agentic_run.py`
- `legacy/render_route_forest.py`
- `legacy/render_blackboard_timeline.py`
- `legacy/validate_example_runs.py`
- `legacy/validate_legacy_example_runs.py`
- `legacy/smoke_route_forest_history.py`
- `legacy/migrate_codex_campaign_v2.py`
- `legacy/audit_architecture_v2.py`
- `legacy/compile_source_route_portfolio.py`（冻结的 V3 source-route portfolio/acceptance 重放）
- `legacy/serve_combined_web.py`（显式启动冻结 combined UI；不属于主 CLI）

迁移完成条件是 Nirmatrelvir/Paclitaxel golden 和历史 saved run 都能从 V4 canonical artifacts 重放。之后才能删除对应大模块，不能提前用“没有引用”判断科学兼容性。

## 保留的专用工具

下列程序不是 campaign owner，因此不会塞入主 CLI：

- 数据/运行时：`download_brenda.py`、`diagnose_chem_enzy_runtime.py`、`setup_chem_enzy_*.sh`
- 合法来源获取：`browser_pdf_fetch.py`、`local_pdf_proxy.py`、`sync_local_pdf_proxy.py`、`tsinghua_pdf_gateway.py`
- 数据审计/查询：`audit_strategic_disconnections.py`、`query_strategic_disconnections.py`、`reaudit_route_pool.py`
- 外部 baseline/实验：`run_chem_enzy_smoke.py`、`evaluate_chem_enzy_onmt_checkpoint_exact.py`、`run_chem_enzy_onmt_adapter_experiment.py`
- P10 golden/案例：`legacy/benchmark_nirmatrelvir_v3.py`、`legacy/run_nirmatrelvir_v3_golden.py`、`run_bufotalin_fullflow_wellau.py`、`run_statin_panel_literature_self_evo.py`
- 受控研究 worker：`run_open_structure_template_agent.py`（实现位于 `cascade_planner.research`）、`run_smiles_first_literature_workflow.py`、`legacy/run_codex_entry_pdf_visual_followup.py`

这些工具的输出必须通过 V4 worker/canonical ingestion 才能影响路线 proof。独立 CLI 文件的存在不代表它拥有第二套图、frontier 或完成判定。

## 质量边界

`pyproject.toml` 仅为上述冻结研究目录和 legacy script 记录已有 Ruff
错误类别；V4 application、interfaces、orchestration、runtime、Web 和全部测试
不在豁免范围。架构测试会阻止豁免模式扩展到主线。迁移某个旧模块时，应同时
移除它的豁免，而不是把新逻辑继续写进旧树。
