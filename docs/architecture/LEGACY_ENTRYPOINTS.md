# 旧入口能力映射

更新：2026-07-13

新 operator 只使用 `python -m cascade_planner`。旧脚本分为三类：

## 已移除包装器

| 旧入口 | 替代 |
| --- | --- |
| `run_autoplanner_web_waitress.py` | `python -m cascade_planner serve --server waitress` |
| `start_agent_workbench.ps1` | `python -m cascade_planner serve` 后访问 `/v4` |
| `scripts/archive/*` | Git 历史；不在当前树执行 |

## 冻结的 V3 campaign 兼容入口

这些脚本仅用于 P10 之前读取或重放已保存的 V3 运行，不再接收新架构功能：

- `run_codex_entry_agentic_blackboard.py`
- `run_codex_entry_controller.py`
- `resume_agentic_blackboard.py`
- `refresh_agentic_closeout_artifacts.py`
- `evaluate_agentic_run.py`
- `render_route_forest.py`
- `render_blackboard_timeline.py`
- `validate_example_runs.py`
- `validate_legacy_example_runs.py`
- `smoke_route_forest_history.py`
- `migrate_codex_campaign_v2.py`

迁移完成条件是 Nirmatrelvir/Paclitaxel golden 和历史 saved run 都能从 V4 canonical artifacts 重放。之后才能删除对应大模块，不能提前用“没有引用”判断科学兼容性。

## 保留的专用工具

下列程序不是 campaign owner，因此不会塞入主 CLI：

- 数据/运行时：`download_brenda.py`、`diagnose_chem_enzy_runtime.py`、`setup_chem_enzy_*.sh`
- 合法来源获取：`browser_pdf_fetch.py`、`local_pdf_proxy.py`、`sync_local_pdf_proxy.py`、`tsinghua_pdf_gateway.py`
- 数据审计/查询：`audit_strategic_disconnections.py`、`query_strategic_disconnections.py`、`reaudit_route_pool.py`
- 外部 baseline/实验：`run_chem_enzy_smoke.py`、`evaluate_chem_enzy_onmt_checkpoint_exact.py`、`run_chem_enzy_onmt_adapter_experiment.py`
- P10 golden/案例：`benchmark_nirmatrelvir_v3.py`、`run_nirmatrelvir_v3_golden.py`、`run_bufotalin_fullflow_wellau.py`、`run_statin_panel_literature_self_evo.py`
- 受控研究 worker：`run_open_structure_template_agent.py`、`run_smiles_first_literature_workflow.py`、`run_codex_entry_pdf_visual_followup.py`

这些工具的输出必须通过 V4 worker/canonical ingestion 才能影响路线 proof。独立 CLI 文件的存在不代表它拥有第二套图、frontier 或完成判定。
