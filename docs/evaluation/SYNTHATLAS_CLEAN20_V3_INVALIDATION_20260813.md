# SynthAtlas clean-20 v3 运行失效记录

日期：2026-08-13

结论：`synthatlas_strategy_closure_clean20_live_contract_v3` 不能用于四臂性能比较。原始目录、execution receipt、事件链和两个统一臂失败样本全部保留，但不进入正式结果分母。

## 冻结身份

- 结果根目录：`results/shared/synthatlas_strategy_closure_clean20_live_contract_v3`
- execution receipt SHA-256：`37d3761f68f46e8484459f22277b5a04a78d09c8b5452432355d301c8dcbfbb0`
- source bundle SHA-256：`737266b1cd9118f5b81f000c5d3440e42c81f4ba2d36c99d2b7d85dbf5dfbcd7`
- 停止前状态：Codex-only 20/20、ChemEnzy-only 20/20、unified-adaptive 2 个已结算但均为 `projection_unavailable`，第 3 个刚启动；unified-round-robin 尚未启动。

停止是方法学失效后的主动终止，不是性能 early stopping。后续目标没有执行，其缺失不得解释为失败。

## 可复现故障

RunKernel v1 把每个 `task_settled.elapsed_s` 直接相加并作为 run wall time。统一臂在同一 revision 并发运行 ChemEnzy、Codex 和 evidence action；每个外层 action 的 elapsed 都包含同一个 barrier 等待区间，Codex action 还包含已单独结算的模型子任务。因此同一段物理时间被重复收费。

| Target | 旧账本 wall time | 从原事件链按区间并集重放 | 任务秒总量 | 结果影响 |
| --- | ---: | ---: | ---: | --- |
| opaque target 001 | 2026.169140 s | 606.331776 s | 2026.169140 s | 错误触发 1800 s run wall-time 上限；固定 cutoff 无可用 snapshot |
| opaque target 002 | 2045.542666 s | 606.447042 s | 2045.542666 s | 同一故障独立复现；再次 `projection_unavailable` |

两个样本的真实事件跨度约 607 秒，均明显低于 1800 秒。旧账本数值适合作为 task-compute seconds 审计，但不适合作为 wall-clock budget。

## 修复与重跑门

通用修复将两个量拆开：

- `task_wall_time_s`：所有已结算任务在事件时间轴上的执行区间并集；并发和嵌套区间只收费一次，串行区间累加。
- `task_compute_time_s`：每个任务 `elapsed_s` 的总和；继续用于资源成本和并发放大审计。

回归测试固定了“3 个重叠任务 elapsed 为 4/5/5 秒时 wall=5、compute=14；再追加 3 秒串行任务后 wall=8、compute=17”的语义。新正式运行必须生成新的 source bundle 和 execution receipt，从四个 arm 的 target 001 重新开始；不得复用 v3 已完成 arm 形成跨代码版本比较。
