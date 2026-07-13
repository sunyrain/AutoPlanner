# V4 schema 索引

| Schema | 权威范围 | 主要生产者 |
| --- | --- | --- |
| `autoplanner_run_spec.v1` | 目标、接受条件、硬预算 | `RunKernel` |
| `autoplanner_run_event.v1` | 可重放运行事件 | `RunKernel` |
| `global_campaign_plan.v1` | 全局路线族和多步骨架提议 | global director / replay input |
| `canonical_retrosynthesis_hypergraph.v1` | 分子、反应边、来源/库存绑定和拓扑 | canonical ingestion |
| `deficit_frontier.v1` | 唯一待办优先队列投影 | canonical graph compiler |
| `retrosynthesis_worker_result.v1` | 可重放 worker 结果 | deterministic workers |
| `retrosynthesis_proof_policy.v1` | L0–L4 和库存/信源规则 | proof policy |
| `proof_stitched_route_portfolio.v1` | 小型多样路线组合和 completion | proof stitcher |
| `retrosynthesis_route_workbench.v1` | 有界只读 UI snapshot | route workbench projection |
| `retrosynthesis_route_workbench_delta.v1` | revision 间 entity upsert/remove | route workbench projection |
| `autoplanner_campaign_gateway_result.v1` | CLI/API 操作结果包装 | campaign gateway |
| `autoplanner_repository_audit.v1` | 当前跟踪树只读审计 | repository audit |

所有持久化 schema 使用 canonical JSON SHA-256 绑定。摘要只能证明字节/结构一致，不能赋予化学真实性。反应、来源、库存和完成各有独立 authority；UI、RunIndex、Blackboard 和缓存都是可重建投影。

升级链为 L0 hypothesis → L1 materialized → L2 reaction validated → L3 exact source → L4 selected-route stock closed。任何阶段不得跳级，也不得从 aggregate count 推导 proof。
