# `objective_mode` 兼容与移除计划

状态：deprecated compatibility metadata

生效日期：2026-08-09

计划从新请求契约移除：2026-10-01

## 当前语义

`benchmark_search`、`scientific_proof` 和 `procurement_delivery` 不再选择求解算法、Action 集合、调度器、replan、证据、条件或停止规则。它们只保留在旧 CLI/API 请求和历史报告中，作为输出兼容标签。

所有新运行都进入同一个 `RunKernel`、canonical hypergraph、deficit frontier 和 `CampaignActionRuntime.run_anytime()`。B1–B5 是同一 trajectory 的并行成熟度轴；B4 不是 benchmark 专用终态，B5 也不是另一套 scientific solver。

## 新调用方式

调用方应直接表达真实输入：

- 库存边界：`stock_boundary`、冻结 stock index 或 inventory snapshot；
- 科学验收：minimum routes、edge proof level、independent source groups；
- 资源：model/native search/evidence/validation/visual/total wall-time budgets；
- 展示与提醒：运行结束后读取 `milestones`、trajectory 和 Workbench，不传回 solver。

Canonical Web 新建任务表单已不再发送 `objective_mode`。CLI 未显式传参时也不会构造旧标签请求；内部报告暂用 `scientific_proof` 作为兼容展示默认值。

## 兼容收据

- CLI 显式使用 `--objective-mode` 时发出可见 `FutureWarning`；
- `POST /api/v4/solve-target` 和 `POST /api/v4/jobs` 收到该字段时返回 `Deprecation: true` 与 HTTP `Warning: 299`；
- async job 首次响应把同一说明保存在 `request_warnings[]`；
- 字段值仍会进入旧 claim projection，但 scheduler 与 capability registration 不读取它。
- saved-run 恢复另外写入 `saved_run_objective_compatibility.v1`，逐位置记录旧 checkpoint/report
  观察值；未知值和字段缺失均可读取，且都不能启停或重排 Action；
- 恢复 Action stage 从历史最大序号继续。同一保存状态的双克隆门已验证：一份保留旧字段、一份
  删除全部旧字段，在相同当前配置下保留相同旧 execution 前缀并产生相同 Action binding 后缀。

## 移除门

2026-10-01 起从新 CLI/API 请求 schema 删除该字段。删除前必须满足：

1. Canonical Web 已连续 30 天不发送该字段；
2. 冻结 benchmark harness 已改为显式 stock/acceptance/budget 参数；
3. saved-run replay 对旧字段只读，缺少字段也能产生相同 action decisions（已通过双克隆恢复门）；
4. 静态门继续证明 scheduler、RunKernel 和 action modules 不读取 objective 标签。

历史 artifact 中的 `objective_mode` 不删除、不重写；它只作为 provenance 被读取。旧调用在移除日之后应收到具名 `objective_mode_removed`，不得静默映射为另一套算法。
