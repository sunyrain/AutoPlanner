# AutoPlanner V4 文档

当前文档只保留实现、操作和验收所需内容。历史长报告、渲染图片、生成运行和外部
语料不再复制到当前树；需要时可从 Git 历史或仓库外数据目录读取。

## 主要入口

- [MAINLINE.md](MAINLINE.md)：系统边界、全局 Codex 和完成定义
- [RUNBOOK.md](RUNBOOK.md)：CLI、恢复、重放、Web、导出和本地发布
- [SCHEMAS.md](SCHEMAS.md)：权威 schema 与证明升级链
- [TESTING.md](TESTING.md)：离线测试与本地质量门
- [SHOWCASE_CASES.md](SHOWCASE_CASES.md)：P10 科学与展示案例

## 架构细节

- [V4_MODULE_AND_COMPATIBILITY_MAP.md](architecture/V4_MODULE_AND_COMPATIBILITY_MAP.md)
- [V4_ROUTE_WORKBENCH.md](architecture/V4_ROUTE_WORKBENCH.md)
- [DATA_AND_STORAGE_POLICY.md](architecture/DATA_AND_STORAGE_POLICY.md)
- [LEGACY_ENTRYPOINTS.md](architecture/LEGACY_ENTRYPOINTS.md)
- [RETROSYNTHESIS_V4_IMPLEMENTATION_TODO.md](architecture/RETROSYNTHESIS_V4_IMPLEMENTATION_TODO.md)

文档中的“已实现”指仓库能力；单个分子是否完成，只能由该 run 的 proof portfolio
和 acceptance contract 判定。
