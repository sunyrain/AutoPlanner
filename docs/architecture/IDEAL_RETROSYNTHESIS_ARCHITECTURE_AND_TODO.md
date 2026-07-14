# 理想逆合成主线架构与实施 TODO

状态：前瞻性主线设计；TODO 只有在真实盲测验收后才能关闭
创建：2026-07-14
适用入口：任意陌生目标 SMILES，不预置目标案卷、路线或前体答案

本文是当前实现向理想主线收敛的唯一前瞻性清单。已有
`RETROSYNTHESIS_V4_IMPLEMENTATION_TODO.md` 记录的是历史能力交付，不能再用“类、字段、
fixture 或单元测试存在”代表端到端能力已经可用。

## 1. 北极星

用户只提供一个陌生 SMILES。系统在硬预算内输出一个小型、多信源、可替换且可审计的
路线组合，并诚实区分：

1. Codex 提出的全局路线骨架；
2. ChemEnzy、模板和文献生成的候选反应；
3. 已物化和已通过主机验证的反应；
4. 有精确结构来源的反应；
5. 有可复现实验条件的反应；
6. 叶节点经过真实采购或厂内库存审计的路线；
7. 尚未闭合的缺口与下一项最有价值的工作。

Codex 的核心价值是一次总揽整个 campaign：同时设计和重排若干路线族、共享中间体、
来源策略、替代模块与止损条件。它不是逐边调用的单步模板。ChemEnzy 是受全局规划和
统一 frontier 调度的候选生成器，不是第二套搜索状态，也没有路线完成权。

系统不承诺任意分子必然成功。系统承诺在预算内给出最好的可验证结果，并且不会把
模型自信、基准库存命中、结构共现或预算耗尽包装成完整路线。

## 2. 产品级输出档位

“完整”必须带档位，禁止无修饰地显示“完整路线”。

| 档位 | 必要条件 | 允许用途 |
| --- | --- | --- |
| `exploration_closed` | 连通结构骨架到基准搜索叶；允许缺条件和真实采购证明 | 比较断键和搜索覆盖 |
| `reaction_validated` | 所有选中边已物化并通过当前主机反应验证 | 化学家审阅的路线建议 |
| `literature_grounded` | 关键边有精确反应来源；来源中的结构与当前边严格绑定 | 文献路线比较 |
| `condition_complete` | 每条选中边都有来源条件，或明确标记的预测/模板条件及风险 | 实验设计草案 |
| `procurement_closed` | 所有叶均由新鲜、版本化、可重放的真实供应记录闭合 | 可采购路线组合 |
| `process_ready` | 所有边达到配置的反应、条件、来源和冲突门槛，所有叶真实闭合 | 可交付的工艺案卷候选 |

默认盲测不能再以 `benchmark_search` 作为“完整合成”的同义词。高级片段只有在真实供应
记录或厂内库存存在时才能终止递归，否则必须继续向上游展开。

## 3. 唯一权威架构

```text
SMILES + acceptance profile + hard budgets
                    |
                    v
         Target intake and capability snapshot
         - identity / salt / stereo audit
         - Codex, ChemEnzy, mapper, evidence, stock capabilities
                    |
                    v
                  RunKernel
        events / recovery / all resource ledgers
                    |
          +---------+----------+
          |                    |
          v                    v
 cheap seed providers     CampaignContextCompiler
 template memory          whole graph + deficits + provider summaries
 bounded ChemEnzy probe            |
 source hints                     v
          |              Codex GlobalCampaignDirector
          |              multi-route architecture only
          +---------+----------+
                    | proposal artifacts
                    v
             Canonical admission gate
 identity / balance / atom jump / cycle / duplicate / ancestry
                    |
                    v
        Canonical retrosynthesis AND/OR hypergraph
 molecules / reactions / origins / sources / conditions / stock
                    |
                    v
              One DeficitFrontier
 materialize / validate / find source / extract procedure / predict condition
 guided ChemEnzy expansion / stock audit / conflict / diversify / recurse
                    |
                    v
             Idempotent worker runtime
                    |
                    +---- every result returns through admission
                    |
                    v
        Proof vector stitcher + route portfolio optimizer
                    |
                    v
        profile-specific acceptance or explicit unresolved
                    |
                    v
    bounded workbench + process dossier + replay/audit bundle
```

只有以下对象拥有权威状态：

- `RunKernel`：事件、恢复、任务和全部预算；
- canonical hypergraph：分子、反应、来源、条件、库存和冲突事实；
- `DeficitFrontier`：从当前图派生的唯一待办视图；
- proof/acceptance compiler：路线组合和各完成档位；
- artifact store：输入、输出、来源片段、模型结果和运行快照。

Blackboard 只保留为可重建的协作/审计投影。Codex、ChemEnzy、模板、文献导入和人工输入
都不得维护私有路线图，也不得绕过 canonical admission。

## 4. 组件职责

### 4.1 Target intake 与 Provider Registry

启动时生成不可变的 `CapabilitySnapshot`，分别记录每个 provider 的：

- 配置来源：CLI、环境变量、用户配置、自动发现或默认值；
- 解释器、vendor、模型、库存与版本摘要；
- `discovered`、`importable`、`model_loadable`、`smoke_tested`、
  `campaign_ready` 五级状态；
- 最后成功时间、预检耗时、失败原因和建议修复；
- 允许的调用次数、CPU/GPU 秒、迭代数和超时预算。

ChemEnzy 解析优先级固定为：显式 CLI/RunSpec > `CHEMENZY_ENV_PREFIX` > 用户级配置 >
已注册 Conda 环境发现 > 仓库默认。不得在源码中写入某台机器的绝对路径；不得静默跳过。
仅成功 import 不能称为 `production_ready`，模型加载和最小搜索必须单独呈现。

### 4.2 Codex GlobalCampaignDirector

初始调用看到目标、能力快照、少量 ChemEnzy/模板种子、来源提示和完整预算。输出至少包括：

- 3–5 个战略不同的路线族；
- 每族的多步骨架和明确的合成/逆合成方向；
- 共享中间体和可替换反应模块；
- 每步反应类别、结构假设、来源检索计划和关键风险；
- 哪些 frontier 值得调用 ChemEnzy，及其目标、约束和停止条件；
- 路线族淘汰、转向和最终组合准则。

Codex 只产生 proposal。它不能授予结构、反应、条件、来源、库存或完成证明。默认一个
campaign 为 1 次初始全局调用，最多 1 次由物质事件触发的重规划；最终文字总结不应再
调用昂贵模型，除非 RunSpec 明确预留预算。

### 4.3 ChemEnzy provider

ChemEnzy 有两种受限模式：

1. `seed_probe`：Director 前的小型多步种子，给出候选路线摘要；
2. `guided_frontier_expansion`：Director 或 frontier 针对某个开放叶、retron 或共享中间体
   发出的局部任务。

原始 `solved` 仅表示 provider 内部找到路径。每个步骤必须转成统一
`CandidateReaction`，经过廉价门、物化、mapping、验证和库存审计。默认总 ChemEnzy
预算应小于 120 秒；连续低有效率、重复边或验证拒绝会提前停止。

### 4.4 Evidence 与 condition pipeline

发现来源和抽取事实必须分开，采用不可逆降级链：

```text
official HTML/XML -> native PDF text -> local OCR -> opt-in page vision
```

上一层已经闭合的边不进入下一层。每个实验条件记录必须绑定：来源、段落/页码、文本或
图像摘要、精确反应边和抽取器版本。

条件 schema 至少包含：

- 反应物、试剂、催化剂和当量；
- 溶剂及浓度；
- 温度程序、时间、气氛与压力；
- 加料顺序、预活化和特殊设备；
- 后处理、纯化、产率、规模和物态；
- 缺失字段、冲突字段与原文位置；
- `source_exact`、`template_suggested` 或 `model_predicted` 权威类型。

“精确结构共现”与“精确实验过程”必须是不同记录。条件为空时 UI 必须显示缺口，不能
留白。预测条件只帮助排序和实验设计，永远不能冒充来源条件。

### 4.5 单一 proof vector

单一 L0–L4 颜色不足以表达真实状态。每条边至少保留以下正交轴：

```text
identity:   proposed | materialized | source_exact
reaction:   untested | mapped | host_validated | source_reaction_exact
conditions: missing | predicted | template_supported | source_exact
sources:    0 | 1 group | 2+ independent groups | conflicted
stock:      unknown | benchmark_hit | catalog_seen | offer_verified | in_house
process:    unknown | plausible | reviewed | executable_candidate
```

路线档位由所有选中边和叶节点的最弱轴决定。颜色只是该档位的投影；producer 用独立 badge
或纹理表示，不能把“由谁提出”和“有多可信”混成一种颜色。

### 4.6 DeficitFrontier 与调度

Frontier 只包含可执行的 typed deficits：

- 结构未物化；
- 反应未验证或验证冲突；
- 已发现来源尚未抽取；
- 条件缺失或条件冲突；
- 独立来源不足；
- 叶节点库存未知或过期；
- 高级叶需要继续逆推；
- 路线缺少战略多样性；
- ChemEnzy/模板可能提升某个开放 frontier；
- 关键拒绝需要一次全局重规划。

排序目标是“单位成本的 portfolio 验收增益”，而不是分支数量。已发现可用来源时，抽取
优先于继续发现；离完成最近的路线缺口优先于产生更多 L0；同一规范边只计一次 accepted
expansion，失败只消耗 attempt。

### 4.7 Self-evolution

学习单元不是裸模板，而是版本化 `ReactionKnowledgeRecord`：

- 精确来源反应和可重放位置；
- 当前版本已接受的 mapping/反应验证；
- reaction-center 模板和适用域；
- 来源条件 envelope；
- 成功/失败复用边、独立来源数和隔离状态。

状态为 `quarantined -> source_observed -> replay_validated -> reuse_validated`。下一次 campaign
中模板和条件模式都只能作为 proposal memory，从 L0 重新进入正常 admission。模板库与
blind benchmark 快照隔离，禁止答案泄漏。

### 4.8 Route compiler 与 UI

Workbench 默认只显示 3–5 条 Pareto 路线，支持两种不混用的方向：

- 逆合成视图：目标 `R1 -> R2...` 指向前体；
- 正合成视图：起始物 `S1 -> S2...` 指向目标。

会聚路线显示偏序 stage，不强行伪造线性 step。每个中间体显示结构式、名称/分子式、
库存状态和来源；每个反应卡显示 producer、反应类别、proof vector、条件摘要、来源和
缺口。辅助试剂默认折叠，不参与主骨架布局。

独立视图必须包括：全局骨架、已展开候选、反应验证路线、文献/条件覆盖、采购闭合和
process-ready。未展开骨架可以显示，但绝不能混入已验证路径计数。

## 5. 成本与停止合同

建议提供三个显式运行档位：

| 档位 | Codex | ChemEnzy | 目标 |
| --- | --- | --- | --- |
| `fast_explore` | 1 次全局调用 | seed/guided 合计 <= 60 s | 多路线骨架和诚实缺口 |
| `validated_plan` | 1 + 最多 1 次重规划 | 合计 <= 120 s | 尽可能得到多条 L2 路线和真实叶审计 |
| `process_dossier` | 同上，允许异步证据工作 | 只补关键 frontier | 条件、来源、采购和冲突闭合 |

所有档位分别记录 model calls/tokens、ChemEnzy CPU/GPU 秒和迭代、网络请求、OCR/vision、
attempt、accepted expansion、缓存命中与总墙钟时间。

停止条件只有：

1. 配置的 acceptance profile 已满足；
2. 任一硬预算耗尽；
3. frontier 没有可执行工作；
4. 最近窗口的验收增益为零且没有物质事件可触发重规划；
5. 用户取消。

停止后必须输出 unresolved deficits；不得通过降低门槛宣告成功。

## 6. 分阶段实施 TODO

依赖顺序：P0 -> P1 -> P2；P3/P4 可在 P2 后并行；P5 依赖 P3/P4；P6/P7 依赖 P5；
P8 依赖 P2/P3；P9 验收全部；P10 最后删除兼容代码。

### P0 — 建立诚实基线与完成定义

- [ ] 将当前 Vismodegib 结果冻结为“结构来源存在、条件为空、基准叶过早闭合”的回归样例。
- [ ] 生成当前四个 blind case 的 proof-vector/条件/库存/producer 覆盖矩阵。
- [ ] 把所有无修饰的 `complete` 映射为明确的 acceptance profile。
- [x] 在报告和 UI 中将 `benchmark_search` 重命名为“搜索边界”，禁止显示为采购/工艺完成。
- [ ] 为旧 L0–L4 到 proof vector 编写只读迁移器和兼容测试。
- [ ] 建立端到端能力矩阵；只有真实 provider 调用才可标成 integrated。

验收：Vismodegib 不再显示“完整工艺路线”；五条现有路径的条件缺口和高级叶状态逐项可见，
科学图 digest 不被展示迁移改变。

### P1 — Provider control plane 与真实 ChemEnzy 接线

- [x] 将现有版本化 `ProviderRegistry` 接入 target-only V4，并补充不可变
  `CapabilitySnapshot`；不得再创建第三套 provider registry。
- [x] 给 `solve-target`、API 和 Web 增加 ChemEnzy enable/env/budget 参数。
- [ ] 按显式配置、环境、用户配置、Conda 发现、默认值的顺序解析解释器。
- [x] 将 `importable`、`model_loadable`、`smoke_tested`、`campaign_ready` 分开，不再用一次
  import probe 宣称 production ready。
- [ ] 增加最小真实搜索 smoke；结果和 stderr/stdout 进入 artifact store。
- [x] 每个 run 显示 ChemEnzy `ran/skipped/unavailable/timeout/failed` 及原因。
- [x] 将 seed probe 的每个步骤以 `origin_kind=chemenzy` 进入统一 admission，禁止 raw solved
  直接进入 route portfolio。
- [x] 增加 guided frontier 请求 schema，支持目标叶、retron、约束和停止条件。

验收：不在源码硬编码 `D:\\conda\\envs\\py312`，但本机通过 CLI/自动发现可选中它；一个
全新 SMILES run 同时出现 Codex 与 ChemEnzy origin，ChemEnzy 失败时 run 仍诚实继续且 UI
不静默。

### P2 — Proof vector 与严格 acceptance

- [ ] 建立 identity/reaction/conditions/sources/stock/process 六轴 schema。
- [ ] 把 exact structure observation 与 exact reaction procedure 拆开。
- [ ] 建立六个产品输出档位及逐档 acceptance compiler。
- [ ] Route stitcher 以所有边/叶的最弱轴判定，不再由最高单边等级抬升整条路线。
- [ ] 将真实 offer、catalog seen 和 benchmark hit 分开。
- [ ] 增加冲突、撤销、过期来源和过期库存后的增量降级。
- [ ] 保留旧 L0–L4 为 UI 派生色，取消其科学权威。

验收：`conditions={}` 的 exact-structure 边不能通过 `condition_complete`；benchmark hit 不能
通过 `procurement_closed`；撤销任一关键事实后路线立即降级且可重放。

### P3 — 反应条件与实验过程抽取

- [ ] 定义完整 condition/procedure schema 和来源片段绑定。
- [ ] 优先从官方专利 HTML/XML 的实施例抽取条件、后处理、纯化、产率和规模。
- [ ] 仅对未闭合边回退 PDF native text、OCR 和显式准入 vision。
- [ ] 将结构表、合成段落、通法与具体实施例关联到同一反应边。
- [ ] 增加单位标准化、同义试剂、当量换算和温度/时间程序解析。
- [ ] 表示字段缺失和多来源冲突，不自动选择看似更好的条件。
- [ ] 增加可插拔 condition predictor，但输出固定为 `model_predicted`。
- [ ] 将 Vismodegib EP3381900A1 的目标边补成真实 procedure 回归样例；若原文确实无条件，
  必须记录“来源只支持结构”而不是伪造。

验收：至少三个真实专利案例可从 HTML 端到端重放条件；网络关闭时从 CAS 重放得到相同
digest；预测条件永不提升 exact-source 或 process-ready 权威。

### P4 — Codex 全局规划与 guided providers

- [x] Director 初始上下文同时包含目标、能力快照、ChemEnzy seed 摘要和 self-evo 候选。
- [ ] 扩展 GlobalCampaignPlan，使每个骨架步骤携带方向、反应类、来源策略、替代模块和
  provider 请求建议。
- [ ] Codex 可在一次响应中合并共享中间体、淘汰路线族和重排多条路线。
- [ ] 编译 Codex precursor/retron 为真实 frontier candidates。
- [ ] 把 guided ChemEnzy、模板 replay、文献 exact route 都实现为相同的 provider task。
- [ ] 只有关键拒绝、新 exact procedure、库存变化、共享瓶颈或真实停滞可触发一次重规划。
- [ ] 记录每次 Director 调用对 portfolio gate 的实际增益；无增益调用进入回归指标。

验收：一个复杂陌生目标使用至多两次 Codex 调用完成跨至少三个路线族的全局决策；局部
ChemEnzy 扩展不会触发逐边 Codex；相同上下文不会重复付费。

### P5 — Evidence-first 闭环 frontier

- [ ] 将条件、provider expansion 和高级叶递归加入 typed deficit。
- [ ] 调度分数改为预期 acceptance/portfolio 增益除以成本和失败风险。
- [ ] 已发现来源自动优先抽取；exact row/procedure 到达后自动恢复同一 campaign。
- [ ] ChemEnzy 连续重复、低验证率或无 portfolio 增益时提前停止。
- [ ] attempt、accepted expansion、provider compute 和模型 token 分账。
- [ ] 增加 dirty-subgraph 增量重算与 full-recompute oracle。
- [ ] 当没有可执行 frontier 时生成具体 unresolved dossier，而不是空成功。

验收：日志中只有一个 frontier 和一个 expansion 计数权威；单 child 不会双计；证据到达
能够推进原 run，不创建第二个 campaign。

### P6 — 真实库存与上游递归

- [ ] 接入至少一个版本化、许可明确的真实供应目录/快照。
- [ ] 供应记录绑定供应商、目录号、规格、地区、抓取时间和结构身份。
- [x] 对所有选中叶执行 stock audit；高级片段无 offer 时自动形成上游展开 deficit。
- [ ] 支持 `procurement` 与 `in_house` 两种不同权威边界。
- [ ] 处理盐型、保护态、溶剂化物和立体化学不一致，禁止模糊命中闭合。
- [ ] 库存过期或撤销后自动重算受影响路线。

验收：Vismodegib 的高级前体只有真实 offer 才能作为叶；否则继续展示和调度其上游结构，
直至真实闭合或明确 unresolved。

### P7 — Self-evolution 2.0

- [ ] 将模板库升级为 `ReactionKnowledgeRecord`，包含来源、验证和 condition envelope。
- [ ] 只有 exact patent reaction + 当前 host accepted proof 可进入学习队列。
- [ ] 新模板先在原例 replay，再进入 quarantine；跨底物复用成功后升级成熟度。
- [ ] 模板/条件模式应用后全部重新物化、mapping 和验证。
- [ ] 三次不同边失败且零成功的记录自动隔离。
- [ ] 冻结 benchmark 开始时的知识库 digest，避免运行中答案泄漏。
- [ ] 提供知识库增长、命中率、验证率、失败率和覆盖化学空间报告。

验收：连续两轮不同 blind targets 可观察到知识库增长和经验证复用，同时模板本身不能让
任何边越级获得反应、条件、来源或库存证明。

### P8 — 可解释且美观的路线工作台

- [ ] 修复正合成/逆合成方向与 step 编号，支持会聚偏序 stage。
- [ ] 默认路线展示所有真实中间体结构；辅助试剂折叠到反应卡。
- [ ] 每条边分开显示 producer、proof vector、条件类型、来源和缺口。
- [ ] 条件为空显示明确的“未抽取/来源未提供”，禁止空白卡片。
- [ ] 增加骨架、展开、反应验证、条件覆盖、采购闭合、process-ready 六视图。
- [ ] 颜色只投影可信档位；producer 使用 badge/纹理；提供色盲可读方案。
- [ ] 主图只显示小型 portfolio；替代模块按需展开，共享中间体只渲染一次。
- [ ] 保持单一 world transform、帧合并、结构缓存、culling 和拖拽性能门槛。
- [ ] Inspector 可直接定位来源段落、条件原文、验证详情、库存 offer 和拒绝原因。

验收：70 步压力图拖拽无明显闪烁；Vismodegib 默认图不会把一步晚期组装伪装成全工艺；
任一反应的提出者、可信度、条件和中间结构在两次点击内可见。

### P9 — Fresh blind benchmark 与发布门

- [ ] 建立至少 8 个仓库历史中没有路线案卷的结构多样复杂目标。
- [ ] preflight 扫描目标、同义名、SMILES、InChIKey、关键中间体和答案泄漏。
- [ ] 每个 case 从新运行目录、冻结 provider/模板/库存 snapshot 和陌生 SMILES 开始。
- [ ] 同时报出 `fast_explore`、`validated_plan` 与可选 `process_dossier` 结果。
- [ ] 记录每轴覆盖率、虚假闭合率、路线多样性、条件覆盖、真实采购覆盖和全部成本。
- [ ] 失败案例必须保留并分类，不允许通过换目标只展示成功。
- [ ] 进行关闭 ChemEnzy、关闭 self-evo、关闭 replan 的消融实验。

发布门：

- 100% case 在硬预算内返回可审计成功或明确 unresolved；
- 0 条 benchmark-only 叶被宣称采购闭合；
- 0 条预测条件被宣称来源精确；
- 至少 6/8 case 得到两条战略不同的 `reaction_validated` 路线；
- 至少 4/8 case 得到真实 `procurement_closed` 路线；
- 至少 3/8 case 得到逐边条件可见的 `condition_complete` 路线；
- 中位 Codex 调用 <= 2、总输入 <= 60k、总输出 <= 15k；
- 中位 ChemEnzy 总时间 <= 120 s，且报告其对最终 portfolio 的独立增益。

这些是首版工程门槛，不是宣称任意复杂分子都能自动生成工艺。达到后再依据失败分布提高。

### P10 — 清理兼容架构与仓库

- [ ] 统计一个发布周期内 legacy blackboard、旧 frontier、旧 RouteForest 写路径的实际调用。
- [ ] 将零调用且已有 V4 替代的入口先标 deprecated，再删除。
- [ ] 删除重复 expansion/proof/stock 状态和只为旧 UI 存在的转换层。
- [ ] 将历史大报告、运行结果、模型、库存、PDF 和缓存迁出 Git 工作树；只保留小型 fixture。
- [ ] 合并相互重复或结论过期的架构文档，修复编码损坏文档。
- [ ] 保留一个 CLI、一个 API、一个 Web service 和一个 canonical ingestion path。
- [ ] 加入静态依赖、文件体积、死入口、兼容使用量和生成物误提交审计。

验收：主线端到端 blind benchmark 不导入 blackboard controller、旧 queue 或旧 RouteForest；
兼容清单中所有剩余项都有使用证据、负责人和删除日期；仓库只保留运行所需源码与小型验收资产。

## 7. 执行顺序与当前第一批工作

第一批只做 P0 + P1，不立即重跑昂贵复杂分子：

1. 把 Vismodegib 现状固化成诚实回归；
2. 建立 proof vector 的最小 schema 和 completion label 迁移；
3. 建立 ProviderRegistry/CapabilitySnapshot；
4. 把 ChemEnzy 环境、预算和状态贯通 CLI/API/Web；
5. 用一个小型 smoke 和一个中等陌生分子验证真实 ChemEnzy ingestion；
6. 只有这些门通过后，才进入条件抽取和新 blind benchmark。

每个阶段交付时必须同时提供：代码、schema、迁移、真实端到端 artifact、失败样例、成本报告、
focused tests、全量本地测试和本文 checkbox 的证据链接。没有真实运行证据的项目保持未完成。

## 8. 2026-07-14 受控实现与 fresh blind 记录

本轮新增一轮、非递归的 guided provider 闭环：负库存叶由 canonical frontier 生成
`expansion` deficit，只调度优先级最高的一个叶；ChemEnzy 默认最多 1 条路线、4 轮、60 秒。
候选仍经过同一 materialization、reaction validation 和二次 stock audit。集成测试证明 seed 与
guided 请求共享一个 hypergraph，且 guided edge 保留 `origin_kind=chemenzy`。

Fresh blind 选择仓库此前不存在的 Asciminib SMILES。结果如下：

- ChemEnzy seed：2 轮、top-k 5，返回 0 条候选；失败被保留，没有伪造贡献。
- Codex：一次 `initial_architecture` 全局调用，无 replan、无视觉调用；生成 4 个战略路线族、
  10 条去重反应边。
- Host：10/10 边物化并得到 L2 reaction validation；4 个目标根骨架通过 B2。
- 原运行成本：输入 18,368、输出 5,375、模型墙钟 212.687 秒。配置的 run 输出预算为
  5,000，因 Director 曾错误保留 7,000 单次上限而超出 375；该 bug 已修复为 Director
  output/wall 上限不得超过 run 总预算。
- 零模型 validation fork：0 token、0 模型调用，重放同一计划后 4 个骨架保持 reaction
  validated，5 个 canonical 路线变体完成 `benchmark_search` 闭合，B5 通过。
- 未完成轴：exact literature/source、conditions、procurement 和 process 均保持 false。

本记录只证明 `fast_explore/exploration_closed` 已能在复杂 fresh target 上受控工作，不计作
P9 的 8-case 发布门，也不把 benchmark catalog 命中解释为真实采购 offer。

## 9. 2026-07-14 论文检索、视觉链与 ChemEnzy 统一接线

本轮将论文能力从 legacy blackboard 迁入 target-only V4 主线：

- 新增有界论文 provider：使用 Director `source_plan` 与目标名检索 Crossref/DOI，冻结 PDF
  字节；也支持显式 DOI/本地 PDF seed 作为可复现回归入口。
- PDF 先扫描 native text 选择焦点页，再最多渲染 4 页；禁止为了找图先渲染整篇长 PDF。
- patent 与 paper provider 可组合，单个来源失败不会抹掉另一来源的结果；来源仍独立归因。
- 视觉候选不再只进入 replan prompt。目标连接、逐步连续性审计通过后，它经过统一
  materialization 和 reaction validation 写入 canonical hypergraph；随后叶审计与 guided
  ChemEnzy 使用同一个 deficit frontier。
- 视觉条件固定为 `model_extracted_source_condition_candidate`，不能提升 exact-source、
  reaction-proof 或 process-ready 权威。
- 增加 root anchoring：首步必须与输入目标精确一致，或仅缺失立体标记但去立体连接性严格
  一致；分子式/连接性不一致时整条视觉链只保留为 observation，不得生成断开路线。

真实 Bufotalin 回归使用 DOI `10.1016/j.tet.2025.134610` 的 6 页本地 PDF。文本聚焦只渲染
第 2--5 页。一次 `gpt-5.5`、low、无 repair 的视觉调用耗时 183.859 s，输入 20,458、输出
3,701 tokens，提取 5 个晚期步骤及 HF-pyridine、Ac2O、NaBH4、TMSOTf、m-CPBA 条件。
模型同时正确声明低置信度和立体结构缺口。其近似目标结构错误多出一个氧原子，因此新的
root audit 将 5/5 步全部拒绝入图。该结果证明视觉通道真实可运行，也证明“看见路线”不等于
“得到精确结构路线”；下一验收项是结构化来源/OCSR 或受控高分辨率 crop 解析得到正确 root，
然后才能触发文献终点的 ChemEnzy 递归。

## 10. 2026-07-15 陌生分子冷启动性能复盘

他汀 blind panel 的旧基线暴露了一个明确的架构问题：首条可见路线中位耗时为
192.889 秒，而 Global Director 单次调用中位耗时为 186.844 秒，约占 97%。这不是
ChemEnzy、反应验证或 HTML 获取本身慢，而是旧流程把“首条路线可见”错误地绑定到了
一次同时承担全局架构、联网检索、长篇解释和多路线穷举的大模型调用。

本轮已将冷启动改成渐进式流水线：

- `fast` 首轮只要求 2 条战略不同的完整骨架、每条最多 5 步、输出最多 3,800 tokens；
  初始 Director 默认不携带联网工具。用户仍可显式启用初轮联网。
- HTML/XML/专利正文预取与 Director 并行，但延迟隐藏阶段执行
  `html_first_no_pdf`；只有目标边仍缺证据时才进入 PDF、页面渲染和视觉提取。
- 初始骨架物化、确定性验证后立即发布中间 workbench；ChemEnzy 只扩展被全局 Director
  选中的高价值 frontier，不阻塞首屏。
- exact row、procedure、新来源路线或关键拒绝到达后，最多触发一次
  `event_replan`。重规划读取来源自己的路线观察和条件，不把来源标题强行贴到不兼容旧边。
- evidence observations 从旧样例的 21,250 bytes 压缩到 9,130 bytes，减少 57.0%；只保留
  最相关的 4 个来源、每源有限的 procedure 和 source-route proposal。
- `attempt_budget`、accepted expansion、provider compute 与模型调用继续分账；缓存键绑定
  目标结构、能力快照、知识库摘要和配置，使相同上下文可以重放而不重复付费。

真实 EP2032715B1 的 Simvastatin 来源回归还修复了另一个吞吐瓶颈：文献正文已经包含
monacolin J、DMB-S-MMP、LovD 和转化描述，但旧解析器把叙述性标题当成化合物名，导致
0 条来源路线。加入来源内别名、叙述性产物标题和浓度型反应物解析后，确定性解析得到
1 条目标连接的来源候选；本地 RXNMapper 用时 1.407 秒并通过原子盘点。它当前仍是待主机
验证的 L0/source-route proposal，不是 exact procedure 或 B3，因此展示不得越级宣称。

这轮改造已经去掉已知的串行等待和重复上下文，但尚未用新的模型冷启动再次测得端到端
墙钟时间；旧基线与新设计必须在展示中明确分栏。下一性能验收用 1 个真正陌生的复杂目标，
要求首个已验证骨架目标不高于 90 秒、Codex 最多 2 次、视觉默认 0 次，并分别报告首屏、
证据补全和最终 dossier 的耗时。未达到时先分析阶段时序，不通过提高 token 预算掩盖问题。

同日完成了运行时模块边界清理：proof replacement、workbench inspector、路线证据投影、
文献候选规范化、Europe PMC XML、PDF fallback、视觉 observation normalization 和 Web
后台任务从六个大门面中拆出。原门面文件行数分别降为 264、832、741、300、552 和 245；
新模块全部加入依赖方向与行数预算测试。V4 API 中两段 return 后不可达的旧实现已删除。

本地全量验收为 1688 passed、3 skipped、0 failed，另有 2 个 subtests passed；Ruff 全仓通过。
实时控制台已在 `http://127.0.0.1:8878/v4` 以同一 statin runtime 重载，返回 9 个历史 job。
汇总展示保留 9 个 blind 基线，并链接独立 Lovastatin 修复复跑；基线秒数与新运行时优化明确
分栏，尚未执行的新冷启动 SLO 不被显示为已完成。
