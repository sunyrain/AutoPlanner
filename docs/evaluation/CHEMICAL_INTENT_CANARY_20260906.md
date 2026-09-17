# 化学意图与图干预：本轮实现检查

本文保留前一轮 A/B canary 的历史结果。后续结构语义修复、失败恢复、配对与独立模型评审见 [2026-09-06 控制与对照报告](CHEMICAL_INTENT_CONTROL_AND_ABLATION_20260906.md)。

2026-09-06。对应 [方法设计](../architecture/CHEMICAL_INTENT_PLANNING_20260906.md)。结论是：新 Editor 的语义动作已进入真实生产调用路径，跨步依赖能被模型明确输出；**本轮没有获得成功修复的 BCH 路线，也没有证明质量提高**。

## 1. 做了哪些通用改动

- 初始和连续 Strategy 加入从最难承诺倒推、正向检验、retron/潜在对称性、极性、暂时官能团和立体接力等指引，不引入命名反应硬规则。
- 最终 Critic 输出重要 `chemical_dependencies`，允许局部 pass 的制备步骤成为需要协调修改的前提来源。Host 对辅助引用错误单独诊断，保留有效逐步判断。
- Path Repair Editor 输出 `change_step_ids` 和化学目标，Host 在实际 reaction occurrence tree 上连接最小修改区域。历史两端点指令继续可重放。贯通检查修复了旧 candidate 入口仍要求端点的问题。
- 复审从原判断派生原问题、前提与修复目标；换 step ID 不再造成问题上下文缺失。该信息要求独立重评，不继承旧 verdict。
- 从巨型编排文件拆出化学推理和修复范围模块；补充修复被预算阻止时的真实原因日志。

上述生产代码没有目标名称、BCH atom maps 或具体反应的特例分支。

## 2. 两项真实检查，分别回答不同问题

共同输入是保存的 BCH 第三条七步路线。模型为 Astra medium，原注册表、原始路线和历史判断均未改写。本轮没有启动完整五案例批次。

| 检查 | 输入 | 实际调用 | 结果与能支持的结论 |
|---|---|---|---|
| A：新增依赖表达 | 原路线；新 Critic 提示与 schema | 1 次最终 Critic | 输出 4 条依赖，正确关联醇选择、活化和关环；整体 uncertain，没有明确拒绝几何问题，未触发 Editor |
| B：修复执行 | 同一原路线；此前保存的 reject **原样使用** | 1 次 Editor、5 次 Builder、0 次 re-Critic | 两步被选中重建；两次醇边界立体不匹配；最终未到 exact cut frontier，未提交 |

A 的新判断没有被并入 B 的固定判断，B 不包含 A 新增的结构化依赖。因此 B 验证的是新 Editor 动作与 Host/Builder 执行链，**不是新增依赖字段改善修复率的实验**。此前判断来自同一模型，也不当作专家 gold label。A 没有收到此前的静态构象计算；只完成了普通生产评审。

B 启动前曾设定 32k output budget，低于现有 Editor 与最终 Critic 合计 36k reserve，零模型调用即停止。保留该失败目录，修正为 64k 后执行；没有因化学结果不佳反复采样。实际执行上限为 8 次模型调用、500k input、64k output、1500 秒；Builder 上限为 6，最终只调用了 5 次，停止原因为 `route_builder_input_token_allocation_exhausted`。

## 3. 具体暴露的能力缺口

**判断稳定性。** 新 Critic 知道下游关环要求背面可达几何，并正确指出上游状态的依赖，但没有再次确定当前底物的相对几何矛盾。它把“可从结构检查的问题”留在了 uncertain。不能把新字段正确输出解释成检测能力已经提高。

**意图与结构操作失配。** Editor 提出保留原醇并改变活化方式，选择关环和活化两个步骤。Builder 的首次关环编辑生成了不适合按保留构型活化接回原醇的中间体；下一步得到的醇与精确边界异构。随后尝试另一离去基接力，仍得到相同的醇异构体失配，并继续延长到新的受保护中间体。

两个边界拒绝都另做了去 atom map 的 RDKit 身份对照：连接关系相同、含立体的分子身份不同。这不是仅凭 map 编号或 `@` 字符差异作出的结论；也不构成反应速率或活化能证明。见 [结构化检查](evidence/chemical_intent_canary_20260906/results.json)。

**回退粒度。** 已有机制能拒绝错误的重接边界，但真实搜索随后仍主要从当前中间体向上游寻找补救，没有在本次有限预算中回到造成失配的最早结构选择。新 Editor 意图能够描述跨步改变，却还没有自动成为每个 Builder 图动作的化学约束。这是比再加一层一般性 Critic 更应优先解决的问题。

**原路线保护。** 返回 `repair_unresolved`，事务为 `retained_uncommitted_prefix`，`completion_boundary_reached=false`、`final_frontier_restored=false`。候选没有进入整路复审，更没有覆盖原路线。保存的旧七步反应身份经独立比较保持一致。`schema_accepted` 的 Worker 输出数不等于新反应进入权威路线的数量。

## 4. 成本与日志

| 角色 | Worker 调用 | 输入 token | 其中缓存输入 | 输出 token |
|---|---:|---:|---:|---:|
| 新最终 Critic | 1 | 173,684 | 111,488 | 2,827 |
| 固定输入 Editor | 1 | 56,325 | 4,096 | 483 |
| 修复 Builder | 5 | 400,525 | 123,008 | 3,302 |
| 合计 | 7 | 630,534 | 238,592 | 6,612 |

推理输出共 3,731 token，是输出的子集；缓存输入是输入的子集，不重复加总。B 的执行墙钟约 370 秒。工具记录合计 4 次结构检查、3 次网页搜索；本轮没有大量网页搜索。

无工具的 Editor 仍记录 56,325 输入；两个有结构检查的 Builder 各有两次 provider response，累计输入约 112k。这说明短业务 prompt 不等于小 provider 输入，工具回合还会重复携带上下文。日志把本地 prompt/schema 字节与实际 provider token 分别记录；provider system instructions、工具 schema 和 server-added context 的内容当前不可见，不能把差值全部归因于本轮的化学提示，也不能断言 fallback model metadata 就是成本根因。

## 5. 软件验证与下一步

本轮聚焦测试 276 项通过，14 个 subtests 通过；边界、关键事件、统一 runtime 和 Web worker 下游检查另有 94 项通过。最小连接区域由一个六节点分支树全部 63 个非空选择集的穷举 oracle 检查；新增贯通测试覆盖新 wire 到实际 Host 图切分、错误辅助依赖保留判断、step ID 替换后复审仍携带原问题。

软件测试通过不代表化学能力已提高。真实 B 没有到达复审，所以本轮“复审保留原问题”的新路径目前由集成测试覆盖，尚无本次成功修复后的真实调用证据。

下一阶段优先做：让关键几何和立体关系成为可定位的结构观察；把错误重接反推到造成错误状态的 Builder 动作，测试从该处回退；将前提要求绑定到实际干预，校核结构操作是否兑现意图。再以同输入、同预算的消融测试比较检测、范围选择与修复，避免仅凭一次成功或自评分宣布进步。

原始证据：[A 结果](../../results/discussion/chemical-intent-canary-20260906/bch3/critic-result.json)、[B 结果](../../results/discussion/chemical-intent-canary-20260906/bch3-fixed-critic-budget64k/result.json)、[B 模型 I/O](../../results/discussion/chemical-intent-canary-20260906/bch3-fixed-critic-budget64k/repair-worker/model-io.jsonl)、[零调用预算失败](../../results/discussion/chemical-intent-canary-20260906/bch3-fixed-critic/result.json)。[调用脚本](evidence/chemical_intent_canary_20260906/run.py)默认只准备输入，只有 `--execute` 才调用模型；[汇总脚本](evidence/chemical_intent_canary_20260906/summarize.py)只读已有结果。
