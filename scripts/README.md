# 专用与兼容工具

创建、恢复、检查、重放、基准、导出、GC 和 Web 服务统一使用：

```bash
python -m cascade_planner --help
```

本目录只保留不能合理并入 campaign CLI 的外部数据工具、合法文献获取工具、baseline/golden 程序，以及 P10 前读取历史 V3 saved runs 的冻结兼容脚本。完整分类、替代入口和删除里程碑见 [旧入口能力映射](../docs/architecture/LEGACY_ENTRYPOINTS.md)。

专用工具不得直接写 V4 canonical graph、proof 或 completion。它们的结果必须形成可重放 worker result，再经过统一 ingestion。
