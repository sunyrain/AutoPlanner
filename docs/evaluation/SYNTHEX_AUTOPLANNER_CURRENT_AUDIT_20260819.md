# SynthEx 对照下的 AutoPlanner 当前实现审计（2026-08-19）

> Source coverage: Full paper + official repository + local code and run artifacts  
> Extraction confidence: High for local implementation and arXiv v1; limited for unreleased SynthEx source  
> Locator mode: structure-grounded  
> Primary analytical lens: Methods  
> Secondary analytical lens: Resource  
> Context verification: Targeted external check  
> Card completeness: Complete relative to the inspected sources

来源边界：`[Paper]` 只复述 arXiv:2608.07454v1；`[External]` 指论文以外的官方资源；`[Run]` 指本地冻结运行；`[Code]` 指当前工作树；`[Analysis]` 和 `[Hypothesis]` 是本次审计判断。SynthEx 官方仓库尚未发布实现代码，因此这里只比较论文协议，不声称复现其隐藏实现。

## 00 2026-08-20 修复后状态（取代下文的历史“未修好”判定）

下文 02–16 节保留 2026-08-19 当时的故障审计，不应再被当作当前代码结论。修复后权威状态如下：

- `[Code/Test · 2026-08-20 latest]` 论文主臂上游已从未接线的 ChemEnzy best-first facade 切换为真实 `AiZynthFinder.MctsSearchTree`：Codex 逐节点请求仍由主进程执行，ReactionJSON 仍由宿主回放，AiZ 隔离进程负责 UCB 选择、兄弟动作保留、循环剪枝和回传。`paper_synthex` 强制 `strategy_tree_engine=aizynthfinder_mcts`，ChemEnzy 仅保留为普通配置的显式兼容引擎；若 AiZ 运行时或冻结库存缺失，会在付费调用前失败。sidecar 回溯 canary、完整 Director 接线 canary 及 Director/target/protocol 回归均已通过，尚未据此启动新付费实验。
- `[Code/Test]` ReactionJSON 使用宿主映射命名空间重放；RouteJSON 使用开放叶集合编译 DAG，非根步骤必须绑定宿主产生的前体，循环和歧义 fail-closed。JSON/短尾/规范化/接受/酶契约集成回归为 205/205；库存、协议、汇总和目标入口追加回归为 106/106。
- `[Run]` 冻结 ZINC 来源为 AiZynthFinder 官方 `zinc_stock_17_04_20.hdf5`（17,422,831 unique，MD5 `00de71724ec1d4a8463c1b0d8a5d0941`）；冻结 eMolecules 来源为 `SchwallerGroup/synthelite@f8d40df87601b69762cc7ef4831af3abda16a9dc` 的 `stocks/eMolecule.csv`。
- `[Analysis]` 论文声明的 39,684,411 是输入条目口径，不是 full-InChIKey 集合基数。eMolecules 23,081,629 行中有 23,081,535 个合法键和 22,876,045 个唯一键；与 ZINC 重叠 820,049 个，因此实际唯一联合库存为 `17,422,831 + 22,876,045 - 820,049 = 39,478,827`。差值 205,584 正好是 205,490 个重复合法行加 94 个无效行。
- `[Run]` 最终 full-InChIKey SQLite membership oracle 含 39,478,827 个唯一成员，SHA-256 `4d2f601ddd5af10b1c179ec583062d3ba3136553e285944d125e7b5ce19b5a65`；论文声明条目数另存，不再误作主键表计数。
- `[Run]` 论文主臂和酶伴随臂真实入口 preflight 均为 `ready_for_paid_experiment=true, issues=[]`。三目标名称已改为无论文提示的 opaque case 名；两份历史 target-only manifest 以内容哈希允许，路线与论文材料仍不进入 planner。
- `[Run]` `synthexfig1-paper-3-v39` 已按目标级串行、目标内三条独立 Codex 分支启动；模型 `gpt-5.6-terra`、reasoning `medium`，逐节点最多 25 次/分支，逐叶短尾 500 iterations / 1200 s / depth 6，首次 stock-closed 分支即停止。运行结果尚未完成，因此目前不声明三目标 solved rate 或酶增益。

## 01 基本信息

- 论文：*Strategy-first synthesis planning for complex natural products*，arXiv:2608.07454v1，2026-08-07。[Paper: title, abstract, Methods]
- 系统：SynthEx；资源：SynthAtlas；主要模型：`gemini-3.1-pro-preview`。[Paper: Methods]
- 论文：https://arxiv.org/html/2608.07454v1
- 官方仓库：https://github.com/schwallergroup/SynthEx
- `[External]` 截至 2026-08-19，仓库 README 仍表示代码尚未发布，无法源码级核查其调度、提示词和错误恢复。
- AutoPlanner 审计对象：Git HEAD `18eabeb168047077c663f6fda5594c6e37452742`（2026-08-18，`Honor complete RouteJSON in paper mode`）及其未提交工作树。
- `[Code]` 当前有 20 个已跟踪文件被修改，约 `+1844/-199`，另有未跟踪 `codex_worker_events/`；结论属于当前工作树审计，不是稳定发布版结论。

## 02 一句话总结

`[Analysis]` AutoPlanner 目前尚未修好：论文等价 solved 指标和短尾参数虽已实现，paper profile 却会把“只有 StrategyCard、没有路线步骤”的分支当作成功；同时它要求一次模型调用交付整条 RouteJSON，而非宿主逐节点回放、保留有效前缀并继续搜索。最近一次 1800 秒 canary 又在模型调用前零 token 失败。因此，历史 3/4 B4 不能代表当前代码，也不能与 SynthEx 的 63.9% 直接比较。

## 03 研究问题

本次审计回答：当前流程是否按“三条独立战略、每条最多 25 次逐节点展开”运行；未通过是否只是门槛更严格；短尾为什么没有运行；当前结果能否与论文对比。结论依次为：否；否；上游没有产生可物化开放叶节点；不能。

## 04 研究背景与发展路径

`[Paper]` SynthEx 将高层战略和常规叶节点搜索分工：LLM 生成多种战略并用 ReactionJSON 表达图编辑，Critic/Editor 局部修复，短程 AiZynthFinder 补齐未购买叶节点。论文 solved 由目标根连通路线的全部叶节点命中同一库存定义，不要求精确文献证据或条件完整性。

`[Analysis]` AutoPlanner 先前把证据、条件、来源和工程审计置于主路径，增强了可审计性，却掩盖了 reach 本身尚未形成。近期已在 paper profile 中分离这些指标；当前瓶颈变为“战略如何可靠落地为宿主可回放的逐步反应图”。

## 05 论文识别的核心痛点

| 痛点 | SynthEx 的处理 | AutoPlanner 当前状态 |
|---|---|---|
| 模板空间难提出稀有断键 | 三个独立 LLM 战略 | 已有三分支 StrategyCard |
| 自由文本/SMILES 难稳定落地 | ReactionJSON 图编辑、宿主执行 | 已有宿主编译器 |
| 长提示生成整路脆弱 | LLM 作为 MCTS policy 逐次扩展 | paper profile 反而要求一次完整 RouteJSON |
| 局部错误不应丢整路 | Critic/Editor 最多六轮局部编辑 | 整路草案失败时合法前缀不能入图 |
| 常规叶节点浪费 LLM | 每个未购叶节点短程搜索 | 参数存在，但无 materialized frontier 可触发 |
| 结果口径混乱 | reach、stock solved、步骤质量分开 | 指标已分层，执行成功状态仍误报 |

## 06 核心思想

`[Paper]` 真正增益来自搜索结构：先提出战略，再把每次图编辑放入可持续扩展的搜索树，最后用模板搜索完成普通叶节点。Appendix D.1 报告每个目标最多 75 次 LLM policy calls，即 3 条战略 × 每条最多 25 次扩展。

`[Analysis]` 对 AutoPlanner 最重要的原则是：Codex 每次只负责一个可验证的局部决策；宿主编译并回放；合法前缀立即成为下一次调用的精确上下文。不能要求模型在同一次调用中引用尚未由宿主产生的后续 atom maps。

## 07 方法概览

### SynthEx 论文协议

1. 默认生成三条独立战略。
2. LLM policy 在 MCTS 中提出 ReactionJSON；单次最多重试 3 次，超时 600 秒。
3. Route Builder 形成路线，Critic/Editor 最多六轮局部修复，Analyst 汇总。
4. 对未在库叶节点执行短程 AiZynthFinder：depth 6、500 iterations、1200 秒。
5. 一条目标根连通路线的全部叶节点命中同一 ZINC + eMolecules 库存即 solved。

### AutoPlanner 当前 paper profile

1. 默认 `gpt-5.6-terra` medium，三分支，名义每分支 25 次展开。
2. `require_complete_route_json=True` 把 Route Builder 提示改成“一次返回完整线性 RouteJSON”。
3. 整路回放通过后步骤才进入 `multi_step_skeletons`；否则只留下 StrategyCard。
4. 总模型墙钟仍处于单目标 1800 秒包络，并预留 30% 给 Critic/Editor。
5. paper-equivalent solved 已与 B2、条件、证据分开，并配置短尾 6/500/1200。

## 08 核心模块拆解

| 模块 | 预期职责 | 当前实现 | 审计结论 |
|---|---|---|---|
| Strategy Generator | 三条独立战略 | `_BRANCH_MANDATES`、StrategyCard | 第三臂硬性指定酶/全细胞/混合，不是论文等价三自由战略 |
| Route Builder | 连续逐节点图编辑 | `_expand_one_branch_node()` | paper profile 实际要求完整 RouteJSON |
| Host compiler | 回放 ReactionJSON、固定 maps | `routejson_compiler.py` | 基础设施正确，应成为主循环核心 |
| Admission | 只有真实路线才成功 | `usable` 筛选 | 严重错误：StrategyCard 单独即可 `SUCCEEDED` |
| Plan output | 输出 target-rooted skeleton | `_compile_plan()` | 无步骤仍输出 family，但 skeleton 为空 |
| Critic/Editor | 最多六轮局部修复 | reserve + repair loop | 每分支仅预留 2 次，与最多 13 次不匹配 |
| Short tail | 对开放叶节点搜索 | 6/500/1200 | 上游零 skeleton 时无 frontier，调用为 0 |
| Stock oracle | 论文等价闭合 | metric v2 | 本地库存及身份键与论文不同 |

## 09 关键公式与判定

- `[Paper]` `paper reach = 存在至少一条目标根连通路线`。
- `[Paper]` `paper solved = 存在至少一条目标根连通路线，且全部叶节点命中同一库存`。
- `[Analysis]` B2、条件和精确证据应独立报告，不参与这两个值；当前指标层已如此实现。
- 本地 stock：`eMolecules-23081629-canonical-smiles`，23,081,629 条，canonical SMILES 身份键。
- 论文 stock：ZINC + eMolecules，39,684,411 条，full InChIKey 精确命中。
- 当前结果只能标为 `metric_valid_for_bound_stock_but_not_paper_stock_comparable`。

## 10 实验设计与证据链

### 论文结果

`[Paper]` 1,098 个天然产物目标上：模板基线 151/1,098 = 13.8%；仅战略层 275/1,098 = 25.0%；战略与短尾拼接 702/1,098 = 63.9%。论文的 1800 秒是完整 AiZynthFinder baseline 的目标上限；每个短尾另有 1200 秒。论文未把整个 LLM 多代理流程约束为 1800 秒。

### AutoPlanner 历史四目标快照

`[Run]` 2026-08-17 旧代码得到 Traversiadiene、Dibohemamine A、Monascuspirolide A B4 true，Cyclopiamine B B4 false，即 3/4；共 14 frontiers、6 条 stock-closed routes、0 次 short-tail model calls，且库存不是论文库存。

`[Analysis]` 这个 75% 是极小、非随机、旧代码、非论文库存样本，不能估计总体 solved rate；当前工作树此后改动约 2,043 行，也不能用它证明当前代码已修好。

### case004 最近三次关键 canary

| 运行 | 固定上限 | 模型用量 | 结果 | 诊断 |
|---|---:|---:|---|---|
| v29 | 1200 s | 4 calls；55,847 in；8,675 out；839.266 s | B1/B4 false | Route Builder 实际只获得约 840 s |
| v30 | 1200 s | 3 calls；39,249 in；5,216 out；839.250 s | 2 families；0 skeletons；B1 false | StrategyCard-only 被标为 director accepted |
| v32 | 1800 s | 9 invocations；0 in；0 out；0.015 s | director failed | CLI 在接触模型前失败 |

证据目录：`D:/Autoplanner/canary_runs/synthexfive-004-final-v29`、`v30`、`v32`。

### 定向测试

- 相关小集合：51 passed。
- 扩大选择集：77 passed；54 个 setup errors 来自 pytest 临时目录 PermissionError，无 assertion failure。
- 最小复现：StrategyCard 成功、所有 Route Builder 返回 worker error 时，director 仍返回 `state=succeeded`、1 route family、0 skeleton。
- `[Analysis]` 缺少“paper profile 在零 skeleton 时必须失败”的测试、真实延迟预算测试和新进程 Codex CLI smoke test。

## 11 对结论的正确解释

1. 当前失败不只是“科学门槛比论文严格”。v30 在 B2、证据、条件之前就没有 skeleton，首先是 reach/control-flow 失败。
2. 短尾未运行不是短尾配置错误，而是不存在 materialized open leaf；正确顺序应为 `materialize -> topology admission -> stock -> short tail`。
3. v32 的 1800 秒修正只解决旧 1200 秒硬截断，随后暴露零 token CLI 故障。当前工作树添加了 `--skip-git-repo-check`，尚无 live canary 验证。
4. 即便 CLI 修好，1800 秒总包络仍不论文等价：论文允许最多 75 次 policy calls、单次 600 秒；AutoPlanner 把三条战略、Builder、Critic/Editor 全挤入 1800 秒。
5. 源码默认虽为 `gpt-5.6-terra` medium，但 v29/v30/v32 实际记录仍是 `gpt-5.6-sol` medium；没有成功运行证明新默认生效。

## 12 作者明确承认的局限

`[Paper]` SynthEx 评估 route reach 和 per-step paper quality，不是湿实验可行性；不验证立体化学结果；Critic/Editor 共享同类模型，可能有共同盲点；LLM 成本高；条件整合和实验闭环仍是未来方向。条件可出现在步骤中，但不是 solved 必要条件。

`[External]` 官方 README 报告 strategic/stitched 为 20.8%/67.2%，与 arXiv v1 的 25.0%/63.9% 不一致。代码、数据 manifest 未发布时，比较必须固定 arXiv v1。

## 13 批判性分析

### P0：错误成功状态

`[Code]` `sequential_strategy_director.py:238-256` 把 `require_complete_route_json` 下仅含 `strategy_card` 的分支视为 usable，并在 `:285-298` 返回 `AgentState.SUCCEEDED`。随后 `_compile_plan()` 生成 `hypothesis_only_materialization_pending` family，却没有 skeleton。下游会看到“全局规划已接受”，但没有路线、开放叶节点或短尾任务。

### P0：paper profile 不是逐节点搜索

`[Code]` `:1140-1163` 将 paper profile 切换成完整 RouteJSON 一次性交付。测试 `test_three_independent_branches_expand_one_node_per_call` 使用 `require_complete_route_json=False`；paper-mode 测试反而断言不用单节点提示。绿灯测试覆盖的是不同兼容路径，不是约定的 paper workflow。

### P0：完整路线提示的 atom-map 合同不可满足

`[Code]` 提示要求后续步骤使用前一宿主回放产生的精确 atom maps，但宿主只能在模型整次返回后回放。模型无法预知中间 maps；后段任一错误会拒绝整条草案，合法前缀也不入图。逐节点 host replay 可消除该矛盾。

### P1：有效一步路线会被拒绝

`[Code]` 根节点 complete-route 最小深度为 2，`_route_json_diagnostic()` 返回 `single_root_disconnection_is_not_a_complete_route`，排除了叶节点已全部在库的一步战略解。

### P1：预算合同不一致

`[Code]` 单目标总模型墙钟 1800 秒并预留 30% 给 Critic/Editor。v29/v30 旧 1200 秒 cutoff 下，Builder 恰在约 839.25 秒停止，符合 70% 切片。Critic/Editor 只按每分支 2 次调用预留，而六轮理论上需初始 critic + 6 组 editor/critic，最多 13 次/分支。

### P1：强制酶臂破坏论文匹配

`[Code]` 第三分支硬编码为酶/全细胞/混合。酶是潜在优势，但强占 paper-matched 三臂会混合流程差异和策略域差异，应作为第四臂或独立 ablation。

### P1：实现尚未冻结

`[Code]` 最大改动在 `sequential_strategy_director.py`（约 931 行 diff）。没有 commit、运行 manifest、当前代码成功 canary 三者绑定时，继续累计 benchmark 会产生不可归因数据。

## 14 学到的知识

### Agent-derived knowledge candidates

- “更严格”只能解释 B2/证据/条件下降，不能解释 B1 skeleton 为零。
- 最可靠的代理交互单位是“单个开放叶节点 + 一次图编辑 + 宿主回放”。
- 状态必须区分 `strategy_proposed`、`route_materialized`、`target_rooted`、`stock_closed`；不能被通用 `succeeded` 覆盖。
- 短尾是消费者能力；无合法 frontier 时调大 iterations、depth 或 timeout 没有帮助。
- 酶价值应由新增独特闭合、化学步缩短和失败类型证明，而非默默替换论文匹配臂。

## 15 与现有知识和本项目的连接

AutoPlanner 已有 canonical hypergraph、RouteJSON 编译、反应验证、库存审计、条件和证据分层。这些资产无需删除，但应移出第一阶段 reach 的阻塞路径。合适结构是：

`Codex 战略种子 -> 单节点 ReactionJSON -> 宿主编译/回放 -> 保留合法前缀 -> stock/short-tail -> 独立 B2/条件/证据 -> 可选酶路线 ablation`。

当前代码接近这个结构，却在 paper profile 选择了相反的“单调用完整路线”开关，并允许战略元数据冒充成功路线。主要问题是控制流合同，而不是 Codex 化学能力不足。

## 16 研究设想

### A. Paper-matched reach arm

- 创新状态（Innovation status）：`partially checked`；指标层和宿主编译器已存在，paper profile 控制流尚未符合合同。
- 三条自由、独立、不强制域的 StrategyCard。
- 每条最多 25 次逐节点 ReactionJSON；每次只处理一个开放叶节点。
- 宿主立即编译并持久化合法前缀；失败只封禁当前 edit。
- 所有叶命中统一库存即 paper-equivalent solved；B2、条件、证据另报。
- Validation: 先以一个 live canary 观察至少一次宿主物化和 short-tail，再冻结 5-target matched canary。
- 可能失败：Codex 单节点 edit 仍可能低接受率；库存不匹配仍阻止论文 solved-rate 对比。

### B. Enzyme advantage arm

- 创新状态（Innovation status）：`unverified`；当前只有强制分支机制，没有独立匹配消融结果。
- 增加第四条酶/混合战略，或在化学主链出现高难度红氧/区域选择性瓶颈时触发。
- 报告新增 solved targets、相对最佳纯化学路线的步骤缩短、独特闭合率和失败类型。
- Validation: 同一目标和预算比较三化学臂与三化学臂加酶臂，冻结库存和宿主验证器。
- 可能失败：酶底物范围与条件信息不足，第四臂可能增加成本而不产生独特闭合。

### C. 最小修复与验证顺序

1. StrategyCard-only 不得 director succeeded；零 skeleton 必须 incomplete/failed，并产生明确续行动作。
2. paper profile 恢复单节点 host-compiled 循环；允许一步 stock closure；保留合法前缀。
3. 重定义阶段预算，至少保证 3 个战略种子和每臂一次 Builder，并让六轮 repair 的调用预算与配置一致；不要把 1800 秒总包络称为 paper-equivalent。
4. 先做独立 Codex CLI 新进程 smoke test，再做一个 `gpt-5.6-terra` medium 端到端 canary；验收链必须实际出现 `materialize -> stock -> short tail`。
5. 通过后再跑 5-target matched canary；论文库存未补齐前只报 bound-stock solved，不对比 63.9%。
6. 酶臂单独 ablation，避免污染三臂论文匹配结果。

最终判定：`[Analysis]` **not fixed / implementation-contract failure before scientific comparison**。当前应先修复两个 P0 控制流问题，并用真实 live canary 证明当前代码、模型和配置共同工作。
