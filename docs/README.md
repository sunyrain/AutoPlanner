# AutoPlanner 文档

当前合成设计主架构见
[`architecture/STRATEGY_FIRST_SYNTHESIS.md`](architecture/STRATEGY_FIRST_SYNTHESIS.md)：
StrategyCard 持久化、结构正交性、独立 Codex Critic、graph-edit replay、战略/证据双轴评分，
以及酶与 mechanism 路线的原生竞争顺序。

当前文档只保留实现、操作和验收所需内容。历史长报告、渲染图片、生成运行和外部
语料不再复制到当前树；需要时可从 Git 历史或仓库外数据目录读取。

## 主要入口

- [当前架构状态](architecture/CURRENT_ARCHITECTURE_STATUS.md)：已实现、过渡实现、目标设计和迁移门的唯一状态页
- [主干冗余审计](architecture/MAINLINE_REDUNDANCY_AUDIT_20260716.md)：稳定基线、兼容面清理、删除门和 Program 渐进迁移
- [当前架构路线 PDF](deliverables/AutoPlanner_当前架构路线与GRIA迁移图_2026-07-16.pdf)：当前真实运行路线、创新兼容层和下一代迁移图
- [MAINLINE.md](MAINLINE.md)：系统边界、全局 Codex 和完成定义
- [MILESTONE_SUBSCRIPTION_POLICY.md](architecture/MILESTONE_SUBSCRIPTION_POLICY.md)：首次 B4 的产品级幂等通知/显式取消契约，与固定预算论文评测隔离
- [ARCHITECTURE_EVOLUTION_TIMELINE.md](ARCHITECTURE_EVOLUTION_TIMELINE.md)：按时间梳理架构变迁、权威迁移和理念进步
- [架构演进 PDF](deliverables/AutoPlanner_逆合成架构演进与理念进步_2026-07-15.pdf)：专业排版的完整审阅报告
- [架构演进 PPT](deliverables/AutoPlanner_逆合成架构演进与理念进步_2026-07-15.pptx)：15 页可编辑演示稿
- [RUNBOOK.md](RUNBOOK.md)：CLI、恢复、重放、Web、导出和本地发布
- [SCHEMAS.md](SCHEMAS.md)：权威 schema 与证明升级链
- [SYNTHEX_SYNTHATLAS_PAPER_CARD_20260812.md](evaluation/SYNTHEX_SYNTHATLAS_PAPER_CARD_20260812.md)：SynthEx/SynthAtlas 全文证据卡、强项、边界与可证伪研究设想
- [SYNTHEX_COMPETITIVE_RESPONSE_20260813.md](evaluation/SYNTHEX_COMPETITIVE_RESPONSE_20260813.md)：竞争能力差异、真实 EPO independent-critic 消融与可发表主张边界
- [STRATEGY_TO_EXPERIMENT_CLOSURE_PROTOCOL.md](evaluation/STRATEGY_TO_EXPERIMENT_CLOSURE_PROTOCOL.md)：外部战略到独立证据/实验闭环的预注册对照协议
- [SYNTHATLAS_CLEAN20_EXTERNAL_BASELINE_20260812.md](evaluation/SYNTHATLAS_CLEAN20_EXTERNAL_BASELINE_20260812.md)：污染过滤 clean-20 的冻结方法、20/20 盲测 preflight 与公开快照 C0/C1 基线
- [SYNTHATLAS_CLEAN20_MATCHED_RESULTS_20260813.md](evaluation/SYNTHATLAS_CLEAN20_MATCHED_RESULTS_20260813.md)：同宿主四臂正式运行身份、预注册结果表、资源/失败分母和解释边界
- [SYNTHATLAS_CLEAN20_V3_INVALIDATION_20260813.md](evaluation/SYNTHATLAS_CLEAN20_V3_INVALIDATION_20260813.md)：v3 并发墙钟重复计费的双样本复现、主动终止依据与新批次重跑门
- [TODO_EVIDENCE_AUDIT_20260813.md](evaluation/TODO_EVIDENCE_AUDIT_20260813.md)：未完成项的当前缺口、证据回填、条件性验收与未来 publication-scale 门分类
- [RESULT_FIRST_OPTIMIZATION_20260813.md](evaluation/RESULT_FIRST_OPTIMIZATION_20260813.md)：结果优先调度、ChemEnzy 误杀修复、资源配置与真实 canary 验证
- [RESULT_FIRST20_RUN_MANIFEST_20260813.md](evaluation/RESULT_FIRST20_RUN_MANIFEST_20260813.md)：20-target 正式运行的冻结参数、结果优先指标与恢复命令
- [RESULT_FIRST20_V1_INVALIDATION_20260814.md](evaluation/RESULT_FIRST20_V1_INVALIDATION_20260814.md)：旧批次的 canonical 父路线绑定丢失证据、作废边界与通用修复
- [RESULT_FIRST20_V2_RUN_MANIFEST_20260814.md](evaluation/RESULT_FIRST20_V2_RUN_MANIFEST_20260814.md)：修复后 fresh 20-target 批次的冻结身份、结果口径与复现命令
- [RESULT_FIRST20_V2_INVALIDATION_20260814.md](evaluation/RESULT_FIRST20_V2_INVALIDATION_20260814.md)：首目标再次触发 300 秒硬超时后的主动停止依据，以及 V3 的软截止门
- [RESULT_FIRST20_V3_RUN_MANIFEST_20260814.md](evaluation/RESULT_FIRST20_V3_RUN_MANIFEST_20260814.md)：软截止、父谱系复联和 B4 交付边界修复后的 fresh 20-target 冻结批次
- [RESULT_FIRST20_V3_INVALIDATION_20260814.md](evaluation/RESULT_FIRST20_V3_INVALIDATION_20260814.md)：V3 搜索库存与评分库存错配的双样本证据、作废边界和通用修复
- [RESULT_FIRST20_V4_RUN_MANIFEST_20260814.md](evaluation/RESULT_FIRST20_V4_RUN_MANIFEST_20260814.md)：同库存 canary 验收、V4 20-target 冻结身份与结果优先运行口径
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
