# V4 Schema 索引

| Schema | 权威范围 | 主要生产者 |
| --- | --- | --- |
| `autoplanner_run_spec.v1` | 目标、验收、硬预算 | `RunKernel` |
| `autoplanner_run_event.v1` | 可重放运行事件 | `RunKernel` |
| `global_campaign_plan.v1` | 全局路线族和多步骨架 proposal | Global Director / replay |
| `canonical_retrosynthesis_hypergraph.v1` | 分子、超边、来源、库存和拓扑 | canonical ingestion |
| `deficit_frontier.v1` | 唯一待办优先队列投影 | graph compiler |
| `retrosynthesis_worker_result.v1` | 确定性 worker 结果 | worker runtime |
| `retrosynthesis_proof_policy.v1` | L0–L4、来源与库存规则 | proof policy |
| `proof_stitched_route_portfolio.v1` | 多样路线组合与 completion | proof stitcher |
| `retrosynthesis_route_workbench.v1` | 有界只读 UI snapshot | workbench projection |
| `retrosynthesis_route_workbench_delta.v1` | revision 间实体增删改 | workbench projection |
| `retrosynthesis_replay_pack.v1` | 可移植的目标、事实、库存与期望 | reviewed golden case |
| `retrosynthesis_replay_result.v1` | 重放阶段、指标和 digest 校验 | replay runner |
| `autoplanner_campaign_gateway_result.v1` | CLI/API 操作包装 | campaign gateway |
| `autoplanner_repository_audit.v1` | 当前跟踪树只读审计 | repository audit |

所有持久化 schema 使用 canonical JSON SHA-256 绑定。摘要只证明字节与结构一致，不能
赋予化学真实性。反应、来源、库存和完成各有独立 authority；UI、RunIndex、Blackboard
和缓存都是可重建投影。

证明升级链为 L0 hypothesis → L1 materialized → L2 reaction validated → L3 exact source
→ L4 selected-route stock closed。任何阶段不得跳级，也不能由 aggregate count 推导 proof。
