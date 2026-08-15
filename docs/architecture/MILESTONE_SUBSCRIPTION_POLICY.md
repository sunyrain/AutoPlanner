# 外部 Milestone 订阅与取消策略

状态：首版已实现（2026-08-13）

## 目的

产品可以在首次获得库存闭合路线时立即通知用户，或在通知后显式停止继续计算；统一 solver 本身仍把 B4
视为同一 anytime trajectory 上的里程碑，不能恢复 B4 early return。论文评测不启用本策略，始终使用冻结的
资源 cutoff。

## 权威边界

- 触发事实只来自摘要有效的 `campaign_anytime_snapshot.v2`，且
  `milestones.B4_stock_boundary=true`；provider 的 `solved`、UI 路线数、历史摘要和未校验 JSON 均无触发权威。
- 订阅器只读 target-solver checkpoint/trajectory，不写 canonical graph，不选择 Action，也不修改 proof axis。
- 停止只能通过现有 `RunKernel.cancel()` 追加显式事件完成；订阅器不能终止线程、删除任务或直接改状态文件。
- 已在执行的 Action 可以安全结算；cancel 后 runtime 在下一派发边界停止并记录未执行 Action。通知不得声称
  所有在途工作已经被抢占。

## 策略

第一版支持两种显式产品策略：

- `notify-only`：首次 B4 写入幂等通知回执，但继续运行；
- `notify-and-cancel`：先持久化通知回执，再请求 Kernel cancel。

默认不启用。CLI/API/Web 必须由调用者显式选择，benchmark runner 不暴露或不注册该产品参数。

当前入口为 Gateway `observe_milestone(...)` 和 HTTP
`POST /api/v4/runs/<run_id>/milestone-subscriptions/observe`。它们只在产品调用方显式提交
`notify-only` 或 `notify-and-cancel` 时工作；target solver、blind panel 和策略闭环 runner 均未接入该参数。

## 持久化回执

每个 run 的 `.autoplanner/milestone-subscriptions/` 保存内容寻址回执，至少绑定：

- run id、策略、milestone 名；
- 首次命中 snapshot SHA-256、event sequence、graph revision 和 observed time；
- 通知状态、尝试次数与最终 channel 回执；
- cancel event id/SHA-256（若启用）；
- schema、语义和自身内容摘要。

同一 run、策略、milestone 和首次 snapshot 生成稳定 idempotency key。重复轮询、进程恢复或 Web 刷新只能
读取同一回执，不能重复通知或追加不同 cancel 事件。损坏、未知 schema、无效 snapshot digest 或 run id
不一致必须失败关闭。

## 通知传输

核心先提供本地 durable outbox，不直接耦合邮件、Webhook 或 UI。传输适配器只能消费 outbox 并追加
acknowledgement；发送失败保留为可重试状态，不撤销已经观察到的 B4，也不自动降级为静默成功。

## 取消与通知顺序

`notify-and-cancel` 使用如下顺序：

1. 原子写入 `milestone-observed` outbox 回执；
2. 通过 Gateway 打开同一 run 并调用 `RunKernel.cancel()`；
3. 把 cancel event 的摘要绑定回订阅回执；
4. channel 适配器发送通知并追加 ack。

先持久化观察可避免“已取消但通知证据丢失”。通知传输失败不恢复已取消 run；UI 必须明确显示
`milestone observed / cancellation requested / notification pending` 三个独立状态。

## 恢复与并发

- RunKernel 的追加事件和跨进程 writer lock 仍是取消权威；订阅器不另建状态机权威。
- 多个观察器并发命中同一 snapshot 时，只有一个 outbox identity；其他观察器获得同一既有回执。
- `cancelled` run 不得 resume/reopen。用户如需继续探索，应显式 fork 新 run，而不是清除取消事件。
- `notify-only` 的正常 resume 不重复首达通知；若订阅的是另一个 milestone，则使用独立 identity。

## 验收测试

1. 无 B4 snapshot 时无回执、无取消；
2. 无效摘要、旧 v1 snapshot 或 provider 自报 solved 不触发；
3. 首次合法 B4 只生成一份 outbox 回执；
4. 重复观察和并发观察幂等；
5. `notify-only` 不改变 Kernel 状态和 Action 序列；
6. `notify-and-cancel` 写入一个 cancel 事件，runtime 后续不派发新 Action；
7. 在途 Action 结算语义明确、无预算泄漏；
8. checkpoint/restart 后不重复通知；
9. 传输失败保留 pending，可重试且不重复 cancel；
10. blind panel/fixed-cutoff runner 未注册产品策略，轨迹与未启用时一致。
