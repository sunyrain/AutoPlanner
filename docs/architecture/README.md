# 架构文档导航

更新：2026-08-27

本目录按“当前事实、目标设计、迁移记录、历史边界”分工。判断某项能力是否已经完成时，
必须先读当前状态页，不能从类名、fixture、展示截图或设计文档反推实现状态。

## 权威阅读顺序

1. [AUTOPLANNER_ARCHITECTURE.md](AUTOPLANNER_ARCHITECTURE.md)：当前系统总图，先解释 Canonical V4、
   paper-aligned V8、`v9_smoke` 和 GRIA 的嵌套关系、组件分层及事实权威。
2. [CURRENT_ARCHITECTURE_STATUS.md](CURRENT_ARCHITECTURE_STATUS.md)：当前 V4 宿主主线的实现状态与迁移门。
3. [SYNTHEX_COMPONENT_CONTRACT.md](SYNTHEX_COMPONENT_CONTRACT.md)：冻结的 paper-aligned V8 组件合同，
   也是判断 V9 与 SynthEx 论文差异的复现基线。
4. [archive/paper-aligned-v8-20260826/BASELINE.md](archive/paper-aligned-v8-20260826/BASELINE.md)：
   2026-08-26 论文对齐基线、正式运行证据和已知局限；对应 Git 引用
   paper-aligned-v8-freeze-20260826。
5. [V9_CAUSAL_TRANSACTIONAL_RETROSYNTHESIS.md](V9_CAUSAL_TRANSACTIONAL_RETROSYNTHESIS.md)：
   当前 `v9_smoke` 的实现状态、三分子 25-step 结果、与 SynthEx 的差异，以及尚未落地的
   Strategy-Guided Self-Correcting Search 目标态。Strategy review、关键事件 audit 和 final
   Critic/Editor 已运行；搜索异常 audit、ReactionWitness 与事务式 Path Repair 尚未实现。
6. [MAINLINE_REDUNDANCY_AUDIT_20260716.md](MAINLINE_REDUNDANCY_AUDIT_20260716.md)：
   当前主干冗余证据、已执行清理和渐进迁移保护门。
7. [../MAINLINE.md](../MAINLINE.md)：当前 Canonical V4 运行主干及完成定义。
8. [GENERAL_RETROSYNTHESIS_INNOVATION_ARCHITECTURE.md](GENERAL_RETROSYNTHESIS_INNOVATION_ARCHITECTURE.md)：
   下一代 GRIA 目标架构；除状态页明确标记外均为设计目标。
9. [V4_MODULE_AND_COMPATIBILITY_MAP.md](V4_MODULE_AND_COMPATIBILITY_MAP.md)：当前模块和兼容边界。

## 实施与历史记录

- [IDEAL_RETROSYNTHESIS_ARCHITECTURE_AND_TODO.md](IDEAL_RETROSYNTHESIS_ARCHITECTURE_AND_TODO.md)：
  2026-07-14～15 的 V4 收敛清单与实施日志，现为历史迁移记录，不再是下一代设计权威。
- [RETROSYNTHESIS_V4_IMPLEMENTATION_TODO.md](RETROSYNTHESIS_V4_IMPLEMENTATION_TODO.md)：V4 交付历史。
- [BLACKBOARD_CAPABILITY_MIGRATION.md](BLACKBOARD_CAPABILITY_MIGRATION.md)：Blackboard 能力迁移和删除门。
- [LEGACY_ENTRYPOINTS.md](LEGACY_ENTRYPOINTS.md)：旧入口及禁止重新授权的边界。
- [DATA_AND_STORAGE_POLICY.md](DATA_AND_STORAGE_POLICY.md)：数据与 artifact 存储规则。
- [V4_ROUTE_WORKBENCH.md](V4_ROUTE_WORKBENCH.md)：当前展示投影合同。

## 状态词约定

| 状态 | 含义 |
| --- | --- |
| 已实现 | 生产路径存在，并有相称的自动测试或可重放 artifact |
| 过渡实现 | 能力可运行，但仍建立在 V4 reaction-edge 语义上，不能代表 GRIA 抽象已完成 |
| 设计完成 | schema、边界和迁移顺序已定义，但生产运行时尚未采用 |
| 未实现 | 不存在生产级一等实体、写入路径或端到端验收 |

“设计完成”不等于“系统完成”；单个目标是否闭合仍只能由该 run 的 acceptance contract 判定。
