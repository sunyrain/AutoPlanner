# Unified Anytime RetroStar-190 Protocol

日期：2026-08-07
状态：W8 四臂冻结协议已完成；`-g` 的终态覆盖异常已修复；`-h` 前 10 例均 completed，但 Case 001 暴露“无后续 worker dispatch 时全局上限未在最终 disposition 前终态化”的收据不一致并已停止排除；统一终态化 API 已冻结，下一次正式运行从全新根 `-i` 启动。

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

Case 005 随后以独立的运行预算失败结束，不能作为 B4 失败计分：`RunKernelBudgetError: run_total_task_budget_exhausted`。它严格完成 native target 1 + frontier 5，但在 closeout 前达到 `max_total_tasks=256`；settled task 分解为 model 1、other 138、proposal 64、stock 6、validation 47，accepted expansion/attempt 均为 64。该 case 保留在完整 panel 分母和 failure taxonomy 中，结果不进入 B4 分数。当前 target 006 已自动接棒。

首个按 manifest 顺序的 10-target 检查点已形成：9 个 completed、1 个 failed；6 个 B4=true、3 个 B4=false，另 1 个总任务预算失败不计作 B4=false。完成目标共得到 12 条 stock-closed skeleton；10 个处理目标共使用 12 次模型调用，总 elapsed 5591.327 s。三个 B4=false 均属于 `stock_miss_after_host_validated_route`，唯一运行失败为 `run_total_task_budget_exhausted`。所有 10 个目标的 native search 均为 target 1 + frontier 5，预算违规数为 0。该早期检查点只用于运行健康与失败模式审计，不构成 benchmark-wide improvement 声明；当前 target 011 已自动接棒。

Case 018 再次复现 `run_total_task_budget_exhausted`：elapsed 489.375 s，native search 仍严格为 target 1 + frontier 5，但 settled tasks 达到 256/256；任务分解为 model 1、other 137、proposal 57、stock 7、validation 54。至前 18 个处理目标，16 个 completed、2 个 failed，B4=true 12 个、B4=false 4 个；同类总任务预算失败为 2/18。两次失败的任务混合不同，说明问题是复杂目标在统一 action loop 中共同挤占通用总任务上限，而非某个固定 action 的单一死循环。正式冻结配置不变，失败均保留且不重跑；当前 target 019 已自动接棒。

Case 019 第三次复现同一异常：elapsed 929.593 s，settled tasks 256/256，任务分解为 model 1、other 137、proposal 66、stock 3、validation 49；至 19 个处理目标共有 3 个相同运行失败。该频率证明问题不是科学失败，而是全局预算耗尽未被 anytime 终止器吸收。`-b` 因而停止并永久排除出结果，所有目录保留作审计；target 020 的在途运行被中止，不恢复、不计分。修复不提高 `max_total_tasks=256`，不改变 native 1+5、scheduler、模型或 B2/B3/B5 门，只把 `run_total_task_budget_exhausted` 与 `run_wall_time_budget_exhausted` 转换为正常 `budget_exhausted` 终态，使 closeout 和 B0–B5 投影仍可生成。聚焦回归 1 passed；新正式根固定为 `results/.autoplanner/retrostar190-w8-formal-20260806-c`，必须 fresh preflight 后从头运行。

预算终止修复已冻结在实现提交 `68cd156`；核心 runtime SHA-256 为 `46625bf43cb126158362e32d63189efc85ba464c046f68b46791e7c9d49c7038`。新根 `-c` 不从 `-b` resume，下一步只执行四臂 fresh preflight。

`-c` 四臂 fresh preflight 每臂均为 184 passed、6 failed，共 736/760；六个唯一失败 case 在四臂完全一致，且 planner/provider/model 均未启动。失败原因不是实现回归，而是 tracked freeze manifest 为审计方便写入了六条含目标名的旧运行目录路径，触发 `target_material_already_present_in_repository`。修复删除这些路径，仅保留 case ID、报告 SHA 与结构化指标；blind scan 白名单不变。`-c` 永久保留为失败预检审计，新根改为 `results/.autoplanner/retrostar190-w8-formal-20260806-d`。

`-d` fresh preflight 已在 2026-08-06 完成：四臂各 190 passed、0 failed，共 760/760；所有 760 份 case receipt 均绑定到各自 arm 的 panel snapshot。四臂 manifest SHA-256 均为 `2d31de46f20cac4dec3c89f822d9059c4fa6ee68f43261929ba5c6b06a4f7623`，stock SHA-256 均为 `30c828d6780e534d8368f4eb74f844c889683453080d44053ba298a7bebdd79c`，base environment SHA-256 均为 `0d5204f3178b5d292c19d51b0114e117774795ab454c25ae1f492abb38e6622f`。该目录为 preflight-only，completed count 为 0，且不存在 solve report、provider payload 或 RunKernel spec，因此 provider/model 调用为 0。

`-d` 不直接转为正式运行：preflight-only 已为每个 case 建立预检目录，复用会破坏 fresh-run 证明。正式四臂根固定为 `results/.autoplanner/retrostar190-w8-formal-20260806-e`，不从 `-d` resume；按 `unified-adaptive`、`chemenzy-only`、`codex-only`、`unified-round-robin` 顺序串行执行，`parallel-arms=1`。正式根自身仍会在任何 provider 工作前重新执行同一 blind preflight。

`-e` 在首个 adaptive case 运行 493.109 s 后停止并永久排除：settled tasks 达到 256/256，任务分解为 model 1、other 140、proposal 62、stock 4、validation 49；native search 仍严格为 target 1 + frontier 5。问题不是 ChemEnzy、模型或科学门，而是上一版修复只让 `run_anytime()` 返回正常 `budget_exhausted`，没有在返回前把 RunKernel 持久化为终态；随后遗留的 loop 外 `discover_director_source_hints()` 尝试预留 evidence task，再次抛出 `run_total_task_budget_exhausted`，因此没有生成 solve report。第二个在途 case 随 orchestrator 一并停止，不恢复、不计分，全部目录保留审计。

提交 `13ec14e` 封住该边界：全局总任务或 wall-time 预算耗尽时，anytime runtime 在返回前持久化 RunKernel `budget_exhausted`；service 在该终态下不执行、不预留任何新 worker command，也不消耗 command idempotency key；已有 canonical state、B0–B5 投影和 `budget_exhausted` closeout 仍可生成。非全局预算错误继续抛出，`max_total_tasks=256`、native 1+5、scheduler、模型和 B2/B3/B5 门均未改变。真实 source-hints 复现栈与 closeout 聚焦回归共 2 passed。新正式根为 `results/.autoplanner/retrostar190-w8-formal-20260806-f`，不从 `-e` resume，仍按四臂固定顺序串行执行。

`-f` 前 5 个 adaptive case 均正常 completed、0 CLI failed，证明 post-loop 崩溃已消除；Cases 004 与 005 都精确达到 256/256 tasks、kernel/stop decision 均为 `budget_exhausted`、native search 均为 target 1 + frontier 5，并都生成完整 B0–B5 报告。Case 004 为 B1/B2/B4=true，Case 005 为 B1/B4=true、B2=false；未放宽任何科学门。

但 Case 004 同时暴露更窄的报告一致性问题：它在 anytime 返回后的投影阶段才达到 256 tasks，最终 stop decision 为 `budget_exhausted`，而 canonical closeout 仍写成 `unresolved`；Case 005 在 anytime 内达到上限，closeout 正确写成 `budget_exhausted`。该差异会污染 failure taxonomy，故 `-f` 在 5 completed、0 failed 后停止并永久排除；第六个在途 case 不恢复、不计分。提交 `896ab1b` 让 service 在 post-loop 派发 worker 前根据当前总任务/wall-time 计数终态化，并让 provisional/final closeout 都读取调用时的真实全局计数。聚焦回归仍为 2 passed；新正式根固定为 `results/.autoplanner/retrostar190-w8-formal-20260806-g`。

`-g` 前 18 个 adaptive case 均正常 completed；前 10 例为 B4 7/10、0 failed、10 次模型调用，所有 native search 均未超过 target 1 + frontier 5，已出现的 256-task 报告中 kernel、stop 与 canonical closeout 一致。Case 018 本轮以 213 tasks 正常 paused closeout，完整报告保留。

Case 019 随后在 256/256 tasks 时先正确进入 RunKernel `budget_exhausted`，但最终 director outcome limit 分支仍尝试执行 `budget_exhausted -> unresolved`，触发 `run_status_transition_invalid`，未生成报告。该问题不涉及 provider、模型、scheduler、科学门或预算值，而是最终 disposition 的终态优先级错误。`-g` 因而在 18 completed、1 failed 后停止并永久排除；Case 020 在途停止，不恢复、不计分。提交 `9671fb1` 保留 director outcome/automatic continuation 诊断 stage，但已存在任意 RunKernel 终态时不再覆盖状态；聚焦回归 3 passed。新正式根为 `results/.autoplanner/retrostar190-w8-formal-20260807-h`。

`-h` 前 10 个 adaptive case 均正常 completed、0 failed，B4=8/10，native search 无预算违规。Cases 005 等在 256 tasks 时已得到一致的 kernel/stop/closeout `budget_exhausted`，证明终态覆盖崩溃已消除。

但 Case 001 在恰好完成最后一个任务后达到 256 tasks，后续没有新的 worker dispatch，因此 service dispatch guard 没有机会持久化终态；canonical closeout 已正确为 `budget_exhausted`，随后 director outcome limit 在仍为 running 的 kernel 上合法写入 `unresolved`，最终形成 closeout 与 kernel/stop 不一致。该问题不造成 CLI failed，但会污染 failure taxonomy，因此 `-h` 在 10 completed、0 failed 后停止并永久排除；Case 011 在途停止，不恢复、不计分。提交 `a330a88` 将全局 task/wall-time 上限终态化收束为单一 service API，并在 worker dispatch 前与最终 disposition 前共同调用。聚焦回归 3 passed；新正式根为 `results/.autoplanner/retrostar190-w8-formal-20260807-i`。
