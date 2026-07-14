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
| `retrosynthesis_case_dossier.v1` | 审阅后的全局路线、精确来源与库存输入 | operator / upstream planner |
| `retrosynthesis_case_compile_result.v1` | 案卷到重放包的确定性编译摘要 | case compiler |
| `retrosynthesis_case_run_result.v1` | 编译、重放、闭合、导出与分阶段耗时 | one-command case runner |
| `retrosynthesis_replay_pack.v1` | 可移植的目标、事实、库存与期望 | reviewed golden case |
| `retrosynthesis_replay_result.v1` | 重放阶段、指标和 digest 校验 | replay runner |
| `autoplanner_campaign_gateway_result.v1` | CLI/API 操作包装 | campaign gateway |
| `primary_patent_html_materialization.v1` | 官方专利 HTML、段落窗口与零模型审计 | patent HTML adapter |
| `local_source_ocr_materialization.v1` | 本地 OCR 执行、覆盖率与零模型审计 | local OCR adapter |
| `source_text_companion_binding.v1` | source/HTML 段落或 PDF/page image/OCR text 哈希重放绑定 | deterministic literature parser |
| `visual_source_candidate_request.v1` | 至多一次、页图哈希绑定的视觉请求 | visual evidence adapter |
| `visual_source_candidate_observation.v1` | 仅供全局 replan 的 L0 视觉候选 | host-normalized visual adapter |
| `autoplanner_repository_audit.v1` | 当前跟踪树只读审计 | repository audit |

所有持久化 schema 使用 canonical JSON SHA-256 绑定。摘要只证明字节与结构一致，不能
赋予化学真实性。反应、来源、库存和完成各有独立 authority；UI、RunIndex、Blackboard
和缓存都是可重建投影。

官方 HTML 和 OCR 文字都只是确定性 parser 的输入。HTML 路径必须重放 publication、
完整 artifact、段落范围和文字哈希；OCR 路径必须重放 PDF、页图、文字和引擎身份。
随后还须由 OPSIN/PubChem 独立重建产物与反应物，记录才可能进入 L3。视觉 observation
永远不在该 authority 范围内，即使其 SMILES 可被 RDKit 解析也仍只属于 L0 proposal。

证明升级链为 L0 hypothesis → L1 materialized → L2 reaction validated → L3 exact source
→ L4 selected-route stock closed。任何阶段不得跳级，也不能由 aggregate count 推导 proof。
