# Unified Anytime RetroStar-190 Protocol

日期：2026-08-06  
状态：W8 四臂冻结协议与 760/760 fresh preflight 已完成；run-wide guided budget 已修复为 target 1 + guided 5 的硬上限，`-b` 根目录的 `unified-adaptive` 正式全 190 正在运行。

机器可读冻结清单：`benchmarks/retrostar190_w8_freeze_20260806.json`。

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

运行前证据：`results/.autoplanner/retrostar190-w8-preflight-20260806-a` 中四个 arm 各通过 190/190，合计 760 passed、0 failed、0 provider calls；四臂的 manifest、stock 与 base environment hashes 一致。

首次运行根目录 `results/.autoplanner/retrostar190-w8-formal-20260806-a` 只保留为预算失败审计，不进入任何结果。该运行在第一个目标观察到 14 次 frontier settlement，而配置值为 5，因此被立即停止。

修复后的正式根目录为 `results/.autoplanner/retrostar190-w8-formal-20260806-b`。RunKernel 现在把广义 attempt budget 重新绑定为 target native 1、guided frontier 5、native total 6；首个正式 case 的不可变 run spec 已确认这些值。Adaptive 完成后，其余三臂由隐藏接续器顺序启动。

首个正式 adaptive case 已完成：499.391 s，B1=true、B2=true、B4=true，5 条 target-rooted/materialized skeleton、1 条 reaction-validated skeleton、1 条 stock-closed skeleton；native-search 精确结算为 target 1 + frontier 5，未越界。B3/B5 仍为 false，未放宽科学门。

第二个正式 adaptive case 已完成：508.562 s，B1=true、B4=true，5 条 target-rooted/materialized skeleton、3 条 stock-closed skeleton；native-search 同样精确结算为 target 1 + frontier 5，模型调用为 1。B2/B3/B5 均为 false，结果按冻结门原样保留；报告 SHA-256 为 `6174538f19de97f0f422ac8d2eea34d992d7fa87868c7a74f707a22f57565fb3`。当前 target 003 已自动接棒。

前四个正式 adaptive case 已完成，其中前三个 B4=true，第四个是首个 B4=false。Case 004 并非 provider 无候选或 embedded host 丢失：ChemEnzy seed 产出并被 host 接纳 1 条 route，最终保留 4 条 target-rooted/materialized skeleton，且 1 条达到 reaction-validated，但 0 条在冻结 eMolecules stock 中闭合。因此首损分类为 `stock_miss_after_host_validated_route`。其五次 guided frontier settlement 中 2 次 completed、3 次 unresolved，其中 2 次接近 90 s timeout；这些结果原样保留，不据此逐目标调参。报告 SHA-256 为 `901eff96fdeb31fe88c976c8ea3759732a50959b763d22924ff5839208951763`。当前 target 005 已自动接棒。
