# 专用与兼容工具

创建、恢复、检查、重放、基准、导出、GC 和 Web 服务统一使用：

```bash
python -m cascade_planner --help
```

本目录只保留不能合理并入 campaign CLI 的外部数据工具、合法文献获取工具和当前 baseline 程序。P10 前读取历史 V3 saved runs 的冻结实现与命令全部集中在 `scripts/legacy/`；根目录不保留旧命令包装。完整分类、替代入口和删除里程碑见 [旧入口能力映射](../docs/architecture/LEGACY_ENTRYPOINTS.md)。

combined V3/V4 Web 不再是主 CLI surface；需要历史界面时显式运行 `python scripts/legacy/serve_combined_web.py`。

专用工具不得直接写 V4 canonical graph、proof 或 completion。它们的结果必须形成可重放 worker result，再经过统一 ingestion。

当前 open-structure 研究 worker 的实现集中在 `cascade_planner/research/`，边界与模块职责见
[Research Runtime](../docs/architecture/RESEARCH_RUNTIME.md)。根脚本只负责显式启动，不拥有第二套 campaign 状态。
deterministic literature registry 与 patent XML gate replay 脚本也使用该 research 命名空间。
