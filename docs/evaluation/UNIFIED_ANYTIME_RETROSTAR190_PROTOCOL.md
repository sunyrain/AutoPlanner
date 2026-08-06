# Unified Anytime RetroStar-190 Protocol

日期：2026-08-06  
状态：W8 冻结协议；正式结果尚未生成。

## 1. 研究问题

评测只回答一个统一问题：在相同 target、库存、预算、host gates 和报告代码下，ChemEnzy、Codex 与 target-blind scheduler 的组件贡献分别是什么。

RetroStar 可比主指标是固定预算内至少存在一条 target-rooted、host-admitted、终端叶全部属于冻结库存的路线，即 B4。B2 reaction validation、B3 exact evidence 与 B5 configured scientific acceptance 始终单独报告，不重定义 B4。

## 2. 冻结输入

- Targets：`benchmarks/retrostar190_v4.json`，190 个 opaque cases；
- Protocol：`benchmarks/retrostar190_v4.protocol.json`；
- Stock：`data_external/retrostar190/retrostar_emolecules_stock.sqlite3`；
- Stock SHA-256：`30c828d6780e534d8368f4eb74f844c889683453080d44053ba298a7bebdd79c`；
- Model identity：`gpt-5.5`，reasoning effort `low`；
- Execution profile：`standard`；
- 每个 panel worker：1；visual：关闭；
- 每个 case 使用 manifest 中完全相同的预算与 acceptance projection。

远端模型权重和 sampling 不是位级冻结；因此必须保存 raw provider payload、模型身份、时间、token、action trace 与 digest。Host scheduler、代码、配置、库存和输入必须 content-addressed。

## 3. 四个全目标算法臂

| Arm | ChemEnzy Actions | Codex Actions | Scheduler | Host gates |
| --- | --- | --- | --- | --- |
| `chemenzy-only` | target + guided | 不注册 | adaptive host ordering | 相同 |
| `codex-only` | 不注册 | architecture + replan | adaptive host ordering | 相同 |
| `unified-round-robin` | 注册 | 注册 | 固定 kind cursor | 相同 |
| `unified-adaptive` | 注册 | 注册 | deficit/value driven | 相同 |

每个 arm 必须覆盖全部 190 个目标。不得按分子难度、target index、成功概率或人工判断选择算法；不得逐目标调参。

## 4. 公平运行

- 四臂读取同一个 manifest 顺序和 stock oracle；
- 每个 case 使用独立 external/runtime/run/artifact root；
- batching 或并发只用于机器调度，不改变算法、预算或目标集合；
- timeout、失败、空结果与部分结果全部进入 190 分母；
- resume 只能继续同一 snapshot、同一 arm 和同一 trajectory；
- 正式运行开始后不得根据测试目标结果修改 scheduler 系数或 arm 配置。

运行入口：

```text
python scripts/run_retrostar190_w8.py \
  --output-root results/.autoplanner/retrostar190-w8-20260806 \
  --parallel-arms 1
```

可用 `--arm` 选择尚未完成的整臂，但不能选择该臂内的目标子组作为论文结果。

## 5. 指标

逐目标至少报告：

- B1/B2/B3/B4/B5；
- structural、materialized、host-validated、stock-closed、evidence-closed route counts；
- time-to-first milestones；
- ChemEnzy native expansions；
- Codex calls/input tokens/output tokens/model wall time；
- action kind counts、失败、resume 和总 wall time；
- raw → normalized → selected → materialized → B4 的首损边界。

聚合报告使用完整 190 分母，并给出 `unified-adaptive` 相对另外三臂的逐目标 paired difference、adaptive wins/losses/ties 和固定种子 paired bootstrap 95% CI。

## 6. 失败分类

分类必须来自保存的 report，而不是人工印象：

- provider 无候选；
- normalization/host admission 损失；
- canonical merge/materialization 损失；
- stock miss；
- budget/timeout；
- runtime failure；
- portfolio/reporting omission；
- 证据不足时保留 unclassified，不强行归因。

Host chemistry rejection 作为独立科学诊断报告，但不能把 B2=false 自动解释为 RetroStar B4 failure。

## 7. 完成门

W8 只有在四臂均覆盖 190 个目标、所有失败保留、paired metrics 与 failure taxonomy 生成、配置/输入/环境哈希一致、且审稿检查表通过后才完成。在此之前不得宣称 benchmark-wide improvement。
