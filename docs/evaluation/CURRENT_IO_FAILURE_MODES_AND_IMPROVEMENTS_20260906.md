# 当前逆合成 IO：问题定位与改进方向

2026-09-06 后续真实模型补评已完成，见 [主论文对照与补评报告](SYNTHEX_MAINPAPER_ROUTE_COMPARISON_20260906.md)：TCF 三个路线族均 uncertain，BCH 两个 uncertain、一个 reject。误删条件已恢复到实际评审输入，BCH 三个上下文均成功送审；新的 reject 指向下游桥环闭合几何，Appel 局部反转本身通过。以下“本轮没有调用付费模型”等描述特指最初的代码修复与确定性验证阶段。新模型结果独立保存，未改写原始运行。

2026-09-06 修复更新：下列三个 P0 已落实到生产代码，并用保存 IO 确定性重放。原始 `audit.json` 和所有运行文件保持修复前历史；本文后续编号章节保留当时的诊断与计数，旧代码行号用于说明当时位置。

| 问题 | 本轮通用修复 | 保存 IO 验证 |
| --- | --- | --- |
| Host 丢失条件 | 由共享条件规范化函数仅过滤明确的独立占位；保留混合句和可能命名试剂的缩写。Critic 压缩也不再切掉条件后半句或后续条件项 | 原始 Builder 输出经实际 ReactionJSON 编译、Host 步骤、Critic 输入和导出字段重放，7 条误删条件全部保留；仍是 advisory |
| BCH 立体等价误拦截 | compiler/replay/review 使用共享 `canonical_stereo_smiles`，去映射后重新清理立体状态，保留 mapped namespace 的独立合同 | 7 个运行共 21 个路线族的评审上下文由 18 个可编译恢复到 21 个，新增 3 个均为 BCH；不产生新的化学 verdict |
| 失败 Critic 覆盖判断 | 首次和 follow-up 使用同一结果解释函数；有效判断与失败尝试分离，历史归约保留适用旧判断并标注复审未覆盖的证据 | 8 次失败调用均标为待审；其中两次保留原有 uncertain，其余无旧判断的不产生 pass；旧格式失败历史无需改写 |

清理：移除 Director 的子串条件过滤；将 5 个纯评审/状态函数移到 [key_event_review.py](../../cascade_planner/orchestration/key_event_review.py)，合并两处首次/随访结果解释；三个去映射身份实现收敛到 [stereochemistry.py](../../cascade_planner/application/stereochemistry.py)。修复提示也只读取有效旧评审。自动调用次数仍受原预算约束，待审投影不增加自动无限重试。没有新增并行状态账本、审批或整套 campaign 重跑。

确定性验证入口：[verify_p0_fixes.py](evidence/current_io_review_20260905/verify_p0_fixes.py)，结果为 [p0_fix_replay.json](evidence/current_io_review_20260905/p0_fix_replay.json)。运行 `python docs/evaluation/evidence/current_io_review_20260905/verify_p0_fixes.py` 仅写该验证结果，不修改历史 `audit.json`。新增持久化测试实际经过 Builder expansion→Host→graph store→Route Critic context→原生导出数据读取；负例覆盖真实 R/S、E/Z、同位素、电荷、连接关系、指定/未指定构型，及剪枝、真实 reject、有效不匹配、首次失败和不同证据范围。

验证结果：核心 7 个受影响测试模块 **357 passed**；`test_target_solver.py` 与 `test_v4_showcase_export.py` **100 passed**，合计 **457 项通过**。本轮生产代码、新增重放脚本和相关测试的 Ruff 检查通过，`git diff --check` 无空白错误。测试保留现有 dependency/runtime warnings；未以清空 warning 或改写历史 verdict 冒充修复。

本轮没有调用付费模型，也没有重启现有服务。新进程/后续运行会使用修复；历史文件中已经丢失的条件和基于缺失输入产生的 verdict 不会自动改变。下一步最小必要动作是对受影响路线补建完整输入，再补做对应评审，而不是把本次工程恢复计作新的 viable 路线或实验成功率提升。

原始诊断日期：2026-09-06。范围为 9 月 5 日五案例更新，以及 R/S 第二轮 Astra 运行。检查 7 个运行、515 条 Worker 记录；重点阅读战略与化学评审输出、Editor 指令，并将相关 Builder 原始输出、后续评审输入和 Host 代码逐项对齐。未逐字人工通读全部完整提示。原始诊断阶段没有调用模型、修改生产代码、修改原运行或升级任何路线状态。

**主要判断：持续战略规划、关键事件批评和局部修订确实在运行，但目前路线质量同时受三类问题限制：Host 丢失或误解化学信息；有效批评没有充分转成可执行动作；方案本身仍缺少具体底物上的选择性与兼容性依据。首先修复信息传递和状态语义，再评价模型能力，才能避免把工程缺陷误判成化学能力不足。**

本轮定位了三个可重复的 P0：条件文本被误删、立体等价结构被误拦截、无效 Critic 输出覆盖有效判断。证据来自保存 IO 与生产纯函数的本地重放，不是另一个模型的推测。

机器统计与复现入口：[audit.json](evidence/current_io_review_20260905/audit.json)、[audit_io.py](evidence/current_io_review_20260905/audit_io.py)。其中 `group_metrics` 为分组汇总，`condition_deletion_probes`、`final_context_diagnostics` 和 `failed_review_supersession_probes` 为具体诊断。脚本只读取保存的运行与 SQLite，立体等价性替换仅在该 Python 进程内生效。

## 1. 样本和计数口径

| 本轮五案例 | Worker 尝试 | Builder 尝试 | Key Critic 尝试 | 有效 pass / reject / uncertain | 最终结构候选 / 库存闭合 |
| --- | ---: | ---: | ---: | --- | --- |
| Homocubane 72 | 89 | 47 | 21 | 1 / 0 / 17 | 3 / 0 |
| Empagliflozin | 88 | 52 | 16 | 3 / 2 / 10 | 3 / 2 |
| (R)-5-Phenylazepan-2-one | 46 | 31 | 3 | 2 / 0 / 0 | 4 / 2 |
| TCF-azetidine | 64 | 39 | 9 | 3 / 0 / 6 | 4 / 2 |
| 1,3-BCH boronate 2a | 77 | 33 | 19 | 4 / 1 / 14 | 3 / 1 |
| 合计 | 364 | 202 | 68 | 13 / 3 / 47 | 17 / 7 |

此前自动表按输出 artifact/metadata 分类，把 3 次 timeout 留在 `unknown`。本次从原始输入的任务类型补齐：Homocubane 两次属于 Key Critic，Empagliflozin 一次属于 Builder。因此物理尝试数由 66/201 校正为 **68/202**；调用总数、有效 verdict 和原 Worker 已知用量不变。68 次 Key Critic 中，3 次非法 JSON、2 次超时没有有效化学评审。

五例最终仍为 **0 条严格整路线反应验证**。17 条结构候选包括规范图变体，不等于 17 条独立战略或实验可行路线。Homocubane 缺少预算内可用快照；四个有固定预算观测的目标合计 14 条结构候选、7 条库存闭合，不能删掉第五个目标的分母。Beckmann、TCF 完成了当前版本最终评审，另三例未完成；保存分支评审不能自动替代当前版本评审。

R/S 作为另一组上下文：151 次尝试，Builder 93、Key Critic 25、Strategy Generator 10、Strategy Critic 10、Route Critic 12、Editor 1。以下复审和 token 指标按两组分别统计，不与早期 R/S 候选总量混用。

## 2. P0：Host 把已经给出的化学条件删掉了

这是本次配对阅读 IO 后最需要更正归因的问题。单看 Critic，会以为 Builder 没写条件；原始输出证明部分条件是在进入 Host 后丢失的。

### TCF 的完整因果链

1. Builder `director:4dd2907e28b246a9f3f5b8cc:branch:2:node:5` 的原始输出写了：`Hypothesized screening conditions: UV irradiation with catalytic xanthone photosensitizer.` 另有溶剂、气氛等第二句。
2. `_clean_condition_text` 对整句做子串匹配，只要命中 `screen`、`screening`、`tbd` 等任一标记，就返回空字符串。
3. 首次 Key Critic 输入只剩第二句，确实看不到 UV 或 xanthone。Critic 因此报告“没有 irradiation/excitation”。
4. 补出 Wittig 上游后再次评审，仍收到缺失该句的条件，继续报告同一问题；分支终点评审与规范图最终评审也保留这个缺口。

直接证据：[TCF 原始 Builder 输出，第 36 行](<D:/Autoplanner/AutoPlanner/results/.autoplanner/manuscript-current-astra-medium25-20260905/runs/opaque target 004/.autoplanner/director-workspace/model-io.jsonl:36>)；[首次 Critic 输入，第 37 行](<D:/Autoplanner/AutoPlanner/results/.autoplanner/manuscript-current-astra-medium25-20260905/runs/opaque target 004/.autoplanner/director-workspace/model-io.jsonl:37>)；同文件复审输入第 55 行、输出第 64 行，最终规范路线评审输入第 123 行。并发 IO 不保证相邻两行属于同一任务，必须用 task ID 对齐。

代码入口：[条件标记表](../../cascade_planner/orchestration/sequential_strategy_director.py:227)、[写入路线前清洗](../../cascade_planner/orchestration/sequential_strategy_director.py:15317)、[子串删除函数](../../cascade_planner/orchestration/sequential_strategy_director.py:15406)。这发生在持久化路线条件的过程中，影响后续评审、导出和复审的一致性。

### 不是单一 case

用同一生产函数重放全部 Builder 输出，五例有 6 条条件句被删除，S 有 1 条。包括：

| 原始条件文本中的内容 | 删除原因 | 影响 |
| --- | --- | --- |
| Homocubane 的 `TBDMSCl, imidazole, dry DMF` | `tbd` 子串 | 具体硅基保护试剂被当作待定占位符 |
| Homocubane 的 `TBDPSCl`；Empagliflozin 的 `TBDMSCl` | 同上 | 同一缺陷跨目标、跨试剂出现 |
| TCF 的 UV / xanthone 条件 | `screening` | 光激发实现被删，制造缺条件诊断 |
| S 的选择性脱硅条件句，含 `TBDMS` | `tbd` 子串 | 整句条件被删，而非仅删除不确定措辞 |
| Beckmann 的 preparative chiral HPLC；BCH 的异构体收集说明 | `screened` / `screening` | 筛选尚未完成的事实与已有分离意图一并丢失 |

其中最后两类仍可能缺少足够的实验实现细节，不能恢复文本后就视为可执行。TCF 指定了光照及一个光敏剂假设，也不代表该底物对的反应性和区域选择性已经成立。**应保留假设供 Critic 审查，同时保持其“未验证”身份。**

通用改进：只将真正独立的空值/占位输入归为空缺；保留含具体化学内容的完整条件句，将待筛选或未验证作为已有条件记录的属性。不要把“不是证据”实现为“删除化学内容”，也不要仅给 TBDMS 或当前例子加白名单。

验收：试剂缩写、光化学、电化学、催化剂手性、选择性分离等条件，在 Builder 输出→规范路线→Key/Route Critic→原生导出之间保真；明确的独立占位（如 `to be determined`）不获得条件完整性或证据通过。`TBD` 也可能命名有机碱，纯文本规范化不能擅自判定其无化学含义；保留该歧义供评审，不赋予证据权威。先重建受影响输入，再评估相应缺条件 verdict 是否仍成立。

## 3. P0：BCH 的立体规范化造成最终评审误拦截

此前根据错误名称判断为“mapped boundary 缺失”，现在已定位：**三条规范路线根产物的映射存在，但 map-independent 分子身份比较不一致。**

[route_review_context.py 的 `_canonical_smiles`](../../cascade_planner/application/route_review_context.py:120) 先解析 mapped SMILES，再移除 atom maps，随后直接输出 canonical isomeric SMILES，没有在去映射后重新清理立体状态。桥环的对称路径和冗余立体标注使同一结构留下不同的规范字符串，触发 [第 381 行](../../cascade_planner/application/route_review_context.py:381) 的 `final_route_critic_mapped_boundary_incomplete`。

两个输入的完整 InChIKey 都是 `RDGIWRAUTQMKJZ-SIIHOXLZSA-N`。复用仓库已有 [stereo_molecule](../../cascade_planner/application/stereochemistry.py:15)，去映射后重新指定立体状态，规范结构一致。

在所选 7 个运行的 21 个规范路线族上，本地评审上下文编译由 **18/21 可用变为 21/21 可用**；新增的 3 个恰好是 BCH，原来可用的 18 个仍可用。翻转 BCH 真实 map 6 手性中心后仍被区分；独立的 R/S、E/Z、同位素、指定/未指定构型对照也仍不同。

这只证明阻断原因和修复方向；没有部署替换、没有重跑 Critic，更没有把 BCH 变成 viable。后续应统一复用一个保留真实立体与同位素差异的分子身份函数，减少 Director、repair、canonical review 各自维护规范化规则的偏差。不能忽略所有 `@`，也不能绕过边界检查。

## 4. P0：超时和坏 JSON 被误记成“不是检查点”

两个真实例子：

| 运行 / task | 原判断 | 新调用 | 保存的新记录 |
| --- | --- | --- | --- |
| Homocubane `critic:5b99381548375a3a42b0` | checkpoint=true，uncertain | timeout | not_checkpoint / unavailable / assessment={} |
| Empagliflozin `critic:f468f99c49434a58c42f` | checkpoint=true，uncertain | 非法 JSON | not_checkpoint / unavailable / assessment={} |

对应 IO：[Homocubane 输入第 103 行](<D:/Autoplanner/AutoPlanner/results/.autoplanner/manuscript-current-astra-medium25-20260905-startup-recovery/runs/opaque target 001/.autoplanner/director-workspace/model-io.jsonl:103>)、[Empagliflozin 失败输出第 48 行](<D:/Autoplanner/AutoPlanner/results/.autoplanner/manuscript-current-astra-medium25-20260905/runs/opaque target 002/.autoplanner/director-workspace/model-io.jsonl:48>)。

原因有两层：[follow-up 处理](../../cascade_planner/orchestration/sequential_strategy_director.py:4277) 没得到有效 assessment 时，将 checkpoint_match 推成 false，写成 not_checkpoint；[检查点状态推导](../../cascade_planner/orchestration/sequential_strategy_director.py:15102) 又按 obligation 让最新记录覆盖旧记录，没有先区分有效化学判断和运行失败。

用两组保存的真实历史做最小重放，保持同一 focus/evidence 步骤集合：包含失败行时，`checkpoint_executed=false`、confidence 为空；只使用先前有效评审时，恢复 `checkpoint_executed=true`、confidence=uncertain。这里的 executed 仅表示关键事件发生且已有非拒绝评审，不是化学通过。额外负例确认：真实的新 reject 仍须优先，不能因为其上游证据边被剪掉就恢复旧 uncertain。

Homocubane 另一首次评审超时、Beckmann 一次首次评审坏 JSON，也被记成 not_checkpoint；它们没有可恢复的旧判断，应保持待评审。不能给初次失败补造 pass。

通用改进：运行是否完成、输出是否有效、是否匹配检查点、化学 verdict 各自表达；只有有效的新化学 assessment 才能取代旧判断。新证据尚未完成审查时，应保留旧结论的适用范围并标记复审待完成，不把旧结论冒充已覆盖新证据。优先在已有历史归约函数中修正，不再增加第二套状态账本。

## 5. 最大的决策缺口：uncertain 有记录，缺少有效的后续动作

按提示中的直接上游证据 follow-up 标记统计：

| 分组 | uncertain 后的直接上游复审 | 有效返回 | verdict 变化 | 已知输入 token |
| --- | ---: | ---: | --- | ---: |
| 五案例 | 25 | 23 | 23 次均仍为 uncertain；另 1 timeout、1 坏 JSON | 2,015,565 |
| R/S 第二轮 | 11 | 11 | 11 次均仍为 uncertain | 825,977 |

不能据此说 36 次复审全无价值：它们有时排除了具体矛盾、补充了竞争路径。但“新增一条上游步骤”不自动提供解决旧选择性问题的信息。TCF 的重复缺条件诊断首先受第 2 节信息丢失影响；Homocubane 的竞争光反应、BCH 的成环选择性、R/S 的 allene 形成与保留问题，则还有真实未解决的化学假设。

动作边界不是模型自由决定的：[Key Critic 提示](../../cascade_planner/orchestration/sequential_strategy_director.py:10501) 明确要求 pass/uncertain 使用 `repair_scope=none`，非 reject 的 `required_change_kind=none`。所以“Editor 少，说明模型主动认为无须修改”并不准确。协议允许 uncertain 提出一句建议，却没有同等明确地让这类建议落到条件补充、证据查询或小范围实现变更。

TCF 后续 [Builder 输入第 51 行](<D:/Autoplanner/AutoPlanner/results/.autoplanner/manuscript-current-astra-medium25-20260905/runs/opaque target 004/.autoplanner/director-workspace/model-io.jsonl:51>) 也说明了信息与权限的限制：继承的是“下游区域选择性仍有风险”的战略摘要；`connected_path_reactions` 主要是 reaction family 与图编辑摘要；Builder 要求只展开当前叶，无法在这次动作中修改已执行的下游光化学条件。不能单纯要求当前 Builder“更加注意 Critic”。

建议利用现有 obligation 和依赖关系，让不确定性区分为四种可处理情形：

| 不确定性的来源 | 有效动作 | 重新评审的触发 |
| --- | --- | --- |
| 具体条件/催化剂缺少实现信息 | 补充或修订该步骤的条件假设 | 条件实质变化 |
| 底物范围、选择性、兼容性缺少先例 | 针对该问题查文献并绑定证据 | 有相关正面或反面证据 |
| 化学上合理但尚无实验的新颖假设 | 保留研究候选并明确风险 | 有新的化学信息；不反复求同一判断 |
| 已有具体矛盾或竞争反应主导 | 单步、片段或战略修订 | 对应错误已被实际改变 |

只在相关前体、反应条件、直接消费者或证据改变后触发模型复审。是否有变化应由既有结构与任务上下文判断，避免为判断是否调用 Critic 再调用一个同等成本的大模型。也不要把全部 uncertain 改成 reject，那会压制合理的新颖方案。

## 6. 化学能力仍有哪些缺口

**战略预审有实际作用，但大部分仍止于保留风险。** 五案例 35 次 Strategy Critic 共审阅 45 张卡：42 keep、3 revise、0 replace。3 次 revise 包括把 Homocubane 最早薄弱检查点前移、修正所需双键数，以及将不合适的热闭环表述改成光化学。keep 比例高不等于预审无用，也不应追求固定拒绝率。更值得加强的是：在战略提出时评估关键前体是否更易获得、上游是否不断引入另一个同样困难的成环假设，以及有多少相互独立的未证实选择性押注。

**连续战略能保留下游约束，但尚未充分评价全路线风险积累。** 本轮 30 次上游战略生成及对应预审证明连续规划在运行。Empagliflozin 和 TCF 的新战略会保留下游所需位点，并声明旧风险未解决；这有意义。不过局部每一步“没有明确矛盾”，并不等于连续多步的整体可靠性高。不能把主观 pass/uncertain 当作独立概率相乘，也不能只按库存闭合或总步数排名。应让排序关注决定成败的少数未解决步骤及其相互依赖。

**产物画出构型与给出构型来源仍有差距。** Empagliflozin 的 Rh(I)/chiral-diene 假设没有指定足够的配体身份和手性匹配；S 的一种不对称还原仍用“S-directing”催化剂措辞，缺少可核对的选择性依据。R/S 还有 HWE 的 E 选择性、allene 的直接/转位连接竞争、脱保护与硫酯/共轭体系兼容等问题。目标结构、CIP 正确和 Host 图编辑通过都不能替代这些论证。S 某脱硅步骤的“无条件”还叠加了第 2 节 Host 删除，必须先恢复真实提案再评价。

**修复能改变结构，但修后 uncertain 仍是研究候选。** BCH 两笔事务分别加入显式异构体分离，以及调整前体相对构型使声称的后续反转与图表示相容；都有 boundary rebuilt、suffix stitched，复评 reject→uncertain。这证明控制不是空转。加入拆分后，还要评价异构体比例、可分性、获取量和整个方案代价，不能把“可以拆分”当成普适解法。Empagliflozin 的撤回事务则说明局部修订仍可能无法形成可接受接回。

**目标获取、战略兑现和合成洞察要分别计量。** Beckmann 分支 2 原战略是 oxime→扩环→拆分，Builder 却在只做 racemic lactam→单一对映体的 preparatory 拆分提案后命中库存，实际未执行 Beckmann 构建。Route Critic 已标 strategy_adherence=false、uncertain。这是可能有价值的目标获取候选，不是完成了 Beckmann 战略；其分离条件又受 Host 清洗影响。当前提示明确允许库存闭合的机会式短路线，不应强行增加骨架步骤去满足标签。评价和展示中分别报告即可。

## 7. 输出与成本：不能只用搜索次数解释

五案例已知 Worker 用量为输入 **24,131,640**，其中缓存输入 **16,099,712**；输出 **203,185**，包含已计入的推理输出。361 条有用量、3 条没有 Worker 汇总用量。

| 角色 | 已知输入 token |
| --- | ---: |
| Builder | 11,888,766 |
| Key Critic | 5,971,682 |
| Strategy Critic | 1,940,914 |
| Strategy Generator | 1,860,962 |
| Route Critic | 2,294,748 |
| Editor | 174,568 |

在有 Worker 用量的调用里，**首个响应输入合计 20,519,769；后续响应输入合计 3,611,871，约占 15.0%**。初始五次无工具 Strategy 调用的本地提示只有 3,758–4,093 字符，首响应仍达 52,269–52,399 输入 token。说明当前调用本身有很重的输入成本，不能只归因于模型多搜索或多想。日志没有观测 provider 系统指令和工具 schema，不能用本地字符数精确反算其组成。

五例记录 208 次网页搜索/打开事件、56 次结构检查和 4 次 shell 工具事件；这不是 208 篇独立论文。后续响应也包含结构工具等交互，不能全部算成网页成本。部分输出已引用相关专利或反应家族文献，但仍未形成严格整路线证据闭合。下一步应按“该来源是否改变了一个具体底物/选择性判断”评价检索，不以事件总数评价收益。

还有一处日志口径缺口：Homocubane 两个 timeout 的 Worker `usage={}`，各自绑定的请求遥测却记录了一个带用量的完成响应，输入分别 **55,621、59,405**，共 **115,026**，未进入上述 24,131,640。已确认两个日志路径、task ID 与 conversation 分别对应，带用量完成事件均在各日志第 7 行。它们是可恢复的部分观测，不是完整失败调用成本，也不是本次擅自增加的预算结算。应区分已结算用量、失败调用可观测部分和仍未知部分，避免空值变零、重复相加或把 API 重试次数直接当作付费调用次数。

输出完整性也值得优先处理：五例有 4 次非法 JSON，均在原始 stdout 中出现未闭合字符串，发生在 Homocubane/Empagliflozin/Beckmann 的 Key Critic 及 BCH 的 Builder。有效 Key Critic 的 124 条 reason 中，**55 条正好 260 字符**，并有明显断句或异字符。schema 的 [260 字符限制](../../cascade_planner/agent/codex_worker.py:2584) 和 Host 多处切片值得检查，但尚无 provider 对照，不能断言它就是非法 JSON 的确定根因。应先保留结束原因、原始响应完整性和校验错误，再减少脆弱的字符硬截，要求短而完整的说明；必要时只重试该次失败评审，不重新生成整条路线。

预算还影响可评价性：Homocubane 第一个持久化轨迹快照就已累计 6,005,976 输入，高于 6M 截止线。Director 内部做了大量工作才向外层提交快照，无法从保存状态复原预算内当时可用的最佳路线。应在已有 Worker/路线持久化边界同步结算与提交可恢复进展，并给最终评审预留与实际调用相符的资源；不能事后扩预算、删失败目标或把超预算路线替代预算内结果。

## 8. 建议实施顺序与验收

| 顺序 | 改进 | 首要验收证据 |
| --- | --- | --- |
| P0-1 | 条件内容保真，修掉整句子串过滤 | 本次真实条件原样进入规范路线与 Critic；试剂缩写不再误删；假设仍不授予证据通过 |
| P0-2 | 统一去映射后的立体分子身份 | BCH 三条上下文可编译；真实 R/S、E/Z、同位素和未指定构型仍正确区分 |
| P0-3 | 失败调用与化学判断分离 | 两个真实历史重放不再被空 assessment 覆盖；首次失败保持待审；后来的真实 reject 不丢失 |
| P1-1 | uncertain 的定向处理与相关信息触发复审 | 同一未改变风险不重复消耗模型；条件/证据任务实际消除或细化问题；记录代价与遗漏 |
| P1-2 | 输出完整性、失败用量与预算内进度 | 非法 JSON 不转成化学状态；失败用量缺口显式；终点评审有资源且能恢复预算内观测 |
| P1-3 | 战略可实现性与关键步骤证据 | 比较前体难度、关键风险数量及依赖；按具体底物绑定来源；区分目标获取和战略完成 |
| P2 | 编排模块逐步拆分 | 从这次已暴露的纯函数边界开始：条件投影、Critic 结果归约、复审调度、评审输入；继续由现有图/Kernel 持有事实 |

这些修复预期首先降低误诊、误拦截和重复成本，让既有化学提案得到完整评审；它们不会自动解决真实底物上的选择性，也无法据此估计总体成功率提升百分比。更有价值的质量改进要来自定向证据、具体催化/立体实现、前体可获取性和路线风险决策。

验证宜先用保存 IO 做确定性重放，确认信息与状态语义；之后只对输入被实质改变或仍缺评审的当前路线补做必要评审。若要比较搜索质量，再用同模型、库存、工具权限和预算的独立目标对照。不要为修复日志语义而反复启动完整昂贵 campaign。

对 arXiv 故事的含义：持续提出战略和事件批评已有运行证据，但“稀疏且有效”仍需证明批评改变了什么、是否消除了关键错误、付出多少成本。正在建设的文献反应/路线库应支持具体底物边界、选择性来源、失败/不适用范围与路线级依赖评价；将目标获取、战略完成、关键步骤专家接受率和预算内有价值候选数分别报告。本次七个开发案例可做回归，不再当作未见 benchmark，也不能据此量化超过 SynthEx 的幅度。

## 9. 2026-09-07：先修现有执行可靠性，模型 Host 单列实验

本轮为可靠性修复。默认入口仍使用现有顺序战略编排；[模型 Host](../architecture/LLM_HOST_EXPERIMENT_20260907.md)目前只有独立设计，不改变默认调度。历史 IO、规范图和化学评审未改写，本轮没有新的付费模型调用或化学质量分数。

| 控制/问题 | 分类与影响 | 本轮处理 | 验证边界 |
| --- | --- | --- | --- |
| 固定预留低于实际调用成本；缺用量按零释放 | K1，资源核算 | 抽出 `model_call_budget.py`，统一共享账本、外层额度检查和恢复时的用量解释；按角色/模型已观察成本校准预留 | 已知值、真实零、字段别名、单轴缺失、并发在途与最终 Critic 预留 |
| 关键 Critic 超时后一直没有有效判断 | K1，局部运行损失 | 沿现有历史为仍选中且未被后续有效评审覆盖的范围提供一次补审；补审失败继续待审 | 初审失败、后续证据评审失败、删除证据、成功/不确定/拒绝、再次失败上限 |
| 到终点才补早期 checkpoint，完整路线令提示超限 | K1，恢复入口误阻塞 | 局部补审保留目标侧连接上下文、焦点和明确指定的证据；完整路线评审保持原范围 | 两个真实保存路线的补审提示可派发 |
| 新原子编号与整条路线已保留身份碰撞 | K0 身份控制保留；K1 反馈修复 | 明确推荐无后续引用的新原子不编号；诊断返回冲突编号、可用新编号起点和统一更新新原子引用的方法 | 经真实编译器报错→Builder 反馈投影→无编号片段编译，原有身份不变 |
| 摘要把 `add_bond` 单键写成 `order None` | 信息投影缺陷 | 按原语真实语义显示单键；不改变图操作 | 原语/编译器与编排回归 |
| 命令关闭搜索却仍记录搜索事件 | 配置与执行记录差异，根因未定 | 日志和用量报告显式输出请求模式、观察事件数和差异状态 | 保存 IO 可重算；不删工具事件，不把配置差异直接改写成化学拒绝 |

预算原始 `input_tokens` / `output_tokens` 继续报告已知用量；`budget_exposure` 把实际已知量与 `unknown_*_tokens_held` 分开记录。共享调用的预留随 Worker 记录持久化；恢复后的账本和外层额度检查读取同一含义。Executor 抛出异常也写成无有效输出的 Worker 记录，避免只留下错误文本而丢掉用量不确定性。认证失败且没有用量报告时不虚构消耗；若失败记录确实报告了 token，资源核算仍承认这部分支出，语义调用成功数另计。

当前预留取已观察到的同角色、同模型调用上界与基础预留的较大值；冷启动角色暂参考已有同模型成本。最终必须做的初次路线 Critic 预留随观测更新，不预扣所有假想 Editor 轮次。此估算可能使小预算更早停止构建；它不是 provider 的硬 token 上限，不能承诺从未见过的高成本调用绝不超额。

补审沿用 `key_event_critic_history`、原焦点 occurrence、原 Strategy 和评审范围。预算不足或提示暂不可用尚不算一次已派发尝试；相同连续延期日志不重复追加。成功返回仍按原语义区分 pass、uncertain、reject 与非 checkpoint；有效 reject 进入原回退/修复入口，失败不会覆盖旧判断。恢复得到 uncertain 后，后来出现的新直接前体证据仍可触发原有定向复审。

保存 IO 验证使用 `traversiadiene-granularity-astra-medium25-20260907` 的 96 个 Worker 记录，以及 `results/discussion/traversiadiene-granularity-comparison-20260907/new-run.json` 所指向的规范图：

- 已知输入仍为 **6,020,846**；两次超时均无 Worker 汇总用量。按该历史中已观察的 Critic 成本估算，共保留 **259,830** 输入 token 的未知占用，预算核算为 **6,280,676**。后两者是估算额度，不能作为实际账单或补造的原始用量。
- S2 的焦点 `codex:branch:2:node:2:candidate:1` 仍在 18 条反应的选中路线中；S3 的 `codex:branch:3:node:6:candidate:1` 仍在 23 条反应的选中路线中。在有可用预算的隔离内存状态中调用实际补审接口，提示分别为 **11,834 / 16,238 字节**，均低于默认 24,000 字节上限。模拟执行器再次超时后各只派发一次，第二次检查不重试；原图与评审没有被写回。这证明恢复路径可执行，不是化学已通过，也不是原耗尽预算的运行被继续执行。
- 保存命令中 96 次都请求 `web_search="disabled"`；其中 **48 次调用**各记录一个网页搜索/打开事件。新版用量报告能直接指出该差异。[官方配置文档](https://developers.openai.com/codex/config-reference/)规定 `disabled` 应移除搜索工具，当前证据没有建立为什么执行结果不同。本轮不增加猜测性的重复开关；需在后续获授权的最小 provider 对照中核实实际工具暴露。

控制清理：预算不变量原来有共享账本与外层手工累加两套解释，现在合并到一个模块；基于现有 Critic 历史增加有界恢复，没有新建评审状态副本。结构身份、图重放、有效评审版本绑定、预算和持久化这些 K0/K1 控制保留；没有新增运行前证书、清单或审批门槛。

剩余的能力问题仍包括可行路线的主动效率改写、跨步依赖驱动的更广泛复审，以及关键反应的底物级外部证据。这些不能靠本轮可靠性修复宣称已解决；后续化学质量比较以修复后的默认流程作为基线。

验证结果：相关预算、关键评审、编排、图编译、Worker、用量报告、目标求解、修复恢复和反应输入测试 **439 项通过，另有 14 个子测试通过**；ReactionJSON 原语重放 **42 项通过**。补审截止时间检查加入后，受影响的编排测试 **217 项再次通过**。本轮涉及文件的 Ruff 与已跟踪修改的 `git diff --check` 通过。现状可进入后续小规模 provider 对照；模型 Host 和化学收益结论仍待单独验证。

## 10. 2026-09-07：smoke25 后补齐外层收尾

随后 Traversiadiene 的真实 smoke25 暴露出跨层缺口：Director 已做三条整路评审，结束时用了 5,977,734 输入 token；外层 Builder 仍追加 60,407，使最终评审因预算耗尽全部不可用。这一历史结果与路线风险保留在 [smoke 报告](../../results/discussion/traversiadiene-reliability-smoke25-20260907/RESULTS.md)，不算可靠性验收通过。

运行后已把外层 Builder 和最终 Critic 接入 RunKernel 的逐次费用预留。准入在事件写锁内计入已知消耗、未知占用、在途调用、本次估算与实际待审路线所需额度；可选扩展预算不足时正常延期并进入收尾。使用已有同角色/模型估算算法，不再由外层只检查累计费用。失败调用报告的 token 仍结算，未知部分独立保留，事件恢复后不丢失。

最终评审可从同次运行的 Worker 日志接回输入完全一致的完整结果。绑定依据是每条结果已有的 `portable_model_input_sha256`，并重新校验模型、推理档位和输出契约；相同 task ID 的旧输入不能给不同调用结果提供绑定。评审槽位接到当前步骤与路线版本，uncertain / reject 保持原判断。后来的失败记录不覆盖同契约的有效结果，JSONL 按 LF 读取以保留字符串内部 Unicode 分隔符。当前只支持有完整槽位的 paper-matched 评审，旧式自由文本结果不跨步骤 ID 复用。

保存 IO 的隔离重放经过真实预算准入及 canonical ingestion：S1、S3 接回完整 uncertain 评审，新增步骤的 S2 正确保留待审；无 provider 调用，反应边、库存与费用计数不变。另有求解器集成回归证明：预算够三次评审但不够再扩展时，实际跳过 Builder 并完成三次评审。相关六文件 440 项及追加预算场景 1 项通过，Ruff 与差异检查通过。

这一修复减少重复评审和收尾丢失，尚不提供新的化学质量证据。高水位预留无法硬性限制 provider 的实际消耗，完整运行效果仍需后续小规模实跑；本次未新增付费运行，也未重启现有网页服务。关键步骤外部证据、uncertain 后的动作价值、搜索配置差异等能力与运行问题仍按前文保留。
