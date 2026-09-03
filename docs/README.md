# AutoPlanner 文档入口

本目录只保留一条当前事实链。判断系统现在如何运行、某条路线是否完成或论文可以主张什么时，按下面顺序阅读；带日期的审计和旧版本文档只作为历史证据，不再定义当前架构。

## 当前权威

1. [AUTOPLANNER_ARCHITECTURE.md](architecture/AUTOPLANNER_ARCHITECTURE.md)：当前唯一架构总图，定义 Canonical Host、统一 Action loop、在线纠偏和多轴闭合。
2. [SYNTHEX_COMPONENT_CONTRACT.md](architecture/SYNTHEX_COMPONENT_CONTRACT.md)：Strategy、Builder、Critic、Editor 与 Host 的冻结职责和输入输出边界。
3. [MAINLINE.md](MAINLINE.md)：运行主干、完成定义和产品入口。
4. [RUNBOOK.md](RUNBOOK.md)：启动、暂停、断点续跑、重放与 Web 操作。
5. [SCHEMAS.md](SCHEMAS.md)：权威 schema、canonical replay 和证明升级链。

`self_correcting_sequential` 是同一 Host/runtime 上的研究 profile，不是另一套 V8、V9 或兼容架构。历史标签仅用于定位旧 artifact。

## 论文与证据

- [PUBLICATION_READINESS.md](publication/PUBLICATION_READINESS.md)：唯一投稿就绪清单，区分已具备证据、pilot 结果和发表阻塞项。
- [arXiv draft](../paper/arxiv/README.md)：论文源文件、证据快照生成方法和编译入口。
- [Recent total-synthesis benchmark](../benchmarks/recent_total_synthesis/README.md)：论文发现、筛选、结构/路线候选与人工接纳边界。
- [Literature source storage](../benchmarks/recent_total_synthesis/SOURCE_STORAGE.md)：本地正文/SI 缓存、Git receipt、哈希校验与离线备份方式。
- [REVIEWER_DEFENSE_CHECKLIST.md](evaluation/REVIEWER_DEFENSE_CHECKLIST.md)：publication-scale benchmark 尚未完成的检查项。
- [UNIFIED_ANYTIME_ABLATION_PLAN.md](evaluation/UNIFIED_ANYTIME_ABLATION_PLAN.md)：统一管线消融定义。
- [SynthEx evidence bundle](evaluation/evidence/synthex_2608_07454/source_bundle.json)：与 SynthEx 对照的本地页级证据。

## 历史与设计目标

- [architecture/README.md](architecture/README.md)：当前事实、迁移记录和历史文档的完整导航。
- [archive/](archive/)：冻结基线和旧实现说明；不授予当前行为权威。
- [GENERAL_RETROSYNTHESIS_INNOVATION_ARCHITECTURE.md](architecture/GENERAL_RETROSYNTHESIS_INNOVATION_ARCHITECTURE.md)：长期设计目标；未被当前状态页确认的能力均不得写成已实现。

路线页面、截图、文件名、模型自报、库存闭合或 `paper_equivalent_solved` 都不能单独升级科学结论。单次运行的真实层级必须由该 run 的 canonical graph、proof portfolio、独立验证轴和 acceptance contract 共同判定。
