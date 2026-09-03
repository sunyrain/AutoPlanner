# Legacy Cleanup Closeout

更新时间：2026-07-26

## 结论

Canonical V4 已完成主线隔离收口。V4 默认启动路径不直接导入
`cascade_planner.legacy` 或 `cascade_planner.research`；历史 V3、Blackboard、
RouteForest compiler、旧评估/训练链和研究实现只能通过显式 legacy/research
入口加载。

本轮不拆分超出行数预算的 12 个模块。除该结构性待办外，兼容出口、视觉预算
回归、RouteForest 展示回归、边界测试和静态检查均已完成。

## 减负结果

当前 `cascade_planner` Python 规模如下：

| 区域 | Python 文件 | Python 行数 |
| --- | ---: | ---: |
| V4 主线 | 452 | 165,667 |
| `cascade_planner.legacy` | 270 | 146,469 |
| `cascade_planner.research` | 18 | 9,563 |

历史实现已从主线命名空间物理迁出，legacy/research 合计占当前包体 Python
代码约 48.5%。旧测试和脚本分别位于 `tests/legacy/` 与 `scripts/legacy/`。

本轮最终工作区的原始 Git 统计为 505 个 tracked 文件变更、+10,166/-233,869
行。该统计包含文件迁移以及工作区中已有的用户修改，不能作为纯删除量；上表的
命名空间规模才是架构减负的主要度量。

## 已完成边界

- V3 application/orchestration/harness/routes/web 旧实现迁入
  `cascade_planner.legacy.*_runtime`。
- 旧 eval、训练、审计、replay、CCTS、Cascade Search checkpoint 和
  CascadeBoard 实验链迁入 legacy runtime，并受显式 guard 约束。
- 根级 `AUTOPLANNRELLM` 迁入 `cascade_planner.research.autoplannrellm`。
- 主 CLI 不再注册 `solve-case` 或 combined Web；历史入口位于
  `scripts/legacy/`。
- `write_run_manifest_compatibility` 已从 active runtime 迁入
  `cascade_planner.legacy.runtime.run_manifest_compatibility`。
- 主线视觉预算现在区分重复调用与预算耗尽；RouteForest 保留稳定的
  persisted/default branch 选择和完整路线数量语义。
- `tests/test_legacy_namespace.py` 继续守护 active/legacy 导出边界。

## 验证结果

以下结果不包含超行数预算测试：

- 主线：`1086 passed, 1 skipped, 1 deselected, 2 subtests passed`
- Legacy：`1009 passed, 2 skipped`
- Namespace：`494 passed`
- 聚焦回归：`687 passed`
- Ruff：通过
- `compileall`：通过
- `git diff --check`：通过

当前唯一已知失败是明确暂缓的
`test_new_focused_modules_stay_within_practical_line_budgets`，仍有 12 个模块
超过既定行数预算。本轮没有修改预算，也没有通过拆分模块掩盖该待办。

## 后续删除门

`compatibility_inventory` 仍登记 19 个 shim。彻底删除 legacy 实现前必须完成：

1. compatibility telemetry 在约定窗口内为零，或调用方已完成迁移；
2. Nirmatrelvir/Paclitaxel golden 和历史 saved run 可由 V4 canonical artifacts
   重放；
3. replacement 通过同等科学、展示和恢复验收；
4. CLI、Web、import graph 和文档不再引用旧实现；
5. 完整离线测试、zero-model replay 和 fresh target smoke 同时通过。

本记录对应的是“主线隔离完成、历史实现保留为冻结兼容”的收尾状态，不等同于
立即删除全部 legacy 代码。
