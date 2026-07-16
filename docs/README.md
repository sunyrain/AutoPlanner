# AutoPlanner 文档

当前文档只保留实现、操作和验收所需内容。历史长报告、渲染图片、生成运行和外部
语料不再复制到当前树；需要时可从 Git 历史或仓库外数据目录读取。

## 主要入口

- [当前架构状态](architecture/CURRENT_ARCHITECTURE_STATUS.md)：已实现、过渡实现、目标设计和迁移门的唯一状态页
- [主干冗余审计](architecture/MAINLINE_REDUNDANCY_AUDIT_20260716.md)：稳定基线、兼容面清理、删除门和 Program 渐进迁移
- [当前架构路线 PDF](deliverables/AutoPlanner_当前架构路线与GRIA迁移图_2026-07-16.pdf)：当前真实运行路线、创新兼容层和下一代迁移图
- [MAINLINE.md](MAINLINE.md)：系统边界、全局 Codex 和完成定义
- [ARCHITECTURE_EVOLUTION_TIMELINE.md](ARCHITECTURE_EVOLUTION_TIMELINE.md)：按时间梳理架构变迁、权威迁移和理念进步
- [架构演进 PDF](deliverables/AutoPlanner_逆合成架构演进与理念进步_2026-07-15.pdf)：专业排版的完整审阅报告
- [架构演进 PPT](deliverables/AutoPlanner_逆合成架构演进与理念进步_2026-07-15.pptx)：15 页可编辑演示稿
- [RUNBOOK.md](RUNBOOK.md)：CLI、恢复、重放、Web、导出和本地发布
- [SCHEMAS.md](SCHEMAS.md)：权威 schema 与证明升级链
- [TESTING.md](TESTING.md)：离线测试与本地质量门
- [SHOWCASE_CASES.md](SHOWCASE_CASES.md)：P10 科学与展示案例

## 架构细节

- [架构文档导航](architecture/README.md)：当前事实、目标设计、迁移记录与历史文档的阅读顺序
- [GENERAL_RETROSYNTHESIS_INNOVATION_ARCHITECTURE.md](architecture/GENERAL_RETROSYNTHESIS_INNOVATION_ARCHITECTURE.md)：面向化学、酶催化、级联和机理外推的一等 Transformation Program 目标架构（设计，尚未完整实现）
- [V4_MODULE_AND_COMPATIBILITY_MAP.md](architecture/V4_MODULE_AND_COMPATIBILITY_MAP.md)
- [V4_ROUTE_WORKBENCH.md](architecture/V4_ROUTE_WORKBENCH.md)
- [DATA_AND_STORAGE_POLICY.md](architecture/DATA_AND_STORAGE_POLICY.md)
- [LEGACY_ENTRYPOINTS.md](architecture/LEGACY_ENTRYPOINTS.md)
- [RETROSYNTHESIS_V4_IMPLEMENTATION_TODO.md](architecture/RETROSYNTHESIS_V4_IMPLEMENTATION_TODO.md)

文档中的“已实现”指仓库能力；单个分子是否完成，只能由该 run 的 proof portfolio
和 acceptance contract 判定。
