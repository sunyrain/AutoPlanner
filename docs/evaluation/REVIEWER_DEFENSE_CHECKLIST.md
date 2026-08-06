# RetroStar-190 Reviewer Defense Checklist

日期：2026-08-06  
状态：运行前检查表；结果项待 W8 完成后填写。

## 算法一致性

- [x] 只有一个 RunKernel、canonical graph、deficit frontier 和 Action runtime。
- [x] 四臂通过 handler registration 或 scheduler policy 消融，不建立目标专用 solver。
- [x] Scheduler、RunKernel 和 provider admission 不读取 target index、dataset ID 或 reference route。
- [x] Round-robin 与 adaptive 共用相同 eligibility、dependency 和 host gates。
- [x] 不按简单/复杂、成功/失败或 benchmark/scientific 对目标分组。

## 输入与泄漏

- [x] 190 个目标以 opaque identity 输入。
- [x] Reference routes 对 planner 不可见。
- [x] Frozen stock 只作为 membership oracle，不加载 RetroStar reaction/value model。
- [x] Fresh preflight 扫描 target SMILES、InChIKey 和仓库残留。
- [ ] 正式四臂 snapshot 的 manifest、stock、environment hashes 完全一致。

## 公平性

- [x] 四臂使用相同 case budget、host validator、stock audit 和报告代码。
- [x] 未使用的 capability 预算不会转化为另一 capability 的额外额度。
- [x] Batching/并发仅用于运行管理，不改变算法。
- [ ] 四臂全部覆盖 190 个目标。
- [ ] timeout、失败、空结果与部分结果全部进入分母。

## 指标与论断

- [x] B4 是 RetroStar 可比主指标。
- [x] B2、B3、B5 分开报告，不伪装成 B4。
- [x] Codex 只有 proposal authority；host gate 授予 reaction/evidence authority。
- [x] 未验证 enzyme/mechanism Program 保持 proposal-only，并保留 conventional fallback。
- [ ] 发布逐目标 paired differences、wins/losses/ties 和 95% CI。
- [ ] 发布 raw → normalized → admitted/materialized → B4 损失分解。
- [ ] 在 190×4 完成前不宣称 benchmark-wide improvement。

## 可复现性

- [x] 代码、配置、stock、环境和本地 executable 使用 SHA-256 冻结。
- [x] Scheduler 系数、kind order 与 tie-break 已公开。
- [x] 如实声明远端模型权重和 sampling 非位级冻结。
- [ ] 发布四臂 run manifest、panel status digest 与原始 action/resource traces。
- [ ] 所有论文表格能够从机器可读汇总重新生成。
