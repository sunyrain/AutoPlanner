# Strategy-to-Experiment Closure Protocol

状态：评价设计草案，2026-09-02 更新；尚未冻结或启动正式实验。本文只定义未来问题、输入、预算和指标，不包含性能声明。

## 研究问题

在同一组复杂靶标、同一库存 oracle、同一宿主验证器和同一总资源预算下，不同规划器能否提出并在逐步执行与纠错中保留**值得合成化学家投入实验资源的路线级洞察**？

主终点不是 solve rate 或库存闭合，而是盲评路线价值：route-defining insight、关键步可信度、立体/选择性逻辑、收敛与效率，以及专家是否愿意推进实验。结构物化、库存、证据和实验闭环用于说明这条洞察被执行到什么程度。

核心比较单位是 provider-neutral strategic route，不是网页、模型品牌或目标名称。外部系统的 `solved`、`productive`、`feasible`、自然语言条件和相似文献命中数均不计作宿主证明。与文献战略相似是一个有用的 reference endpoint，但文献路线不是唯一真值；不相似而被专家认为可信且值得实验的方案属于成功的 novel alternative。

## 冻结输入

- Pilot：在运行任何闭环结果前冻结 20–30 个靶标；扩大规模需要另行冻结 manifest 和停止规则。
- Arms：至少包括单一 LLM 战略、ChemEnzy、模板/搜索基线、外部战略输入，以及同预算的 round-robin/adaptive 组合。
- Route interchange：`external_strategy_route_bundle.v1`，支持正向 reaction SMILES 或显式 `product_smiles + precursor_smiles`。
- Stock：所有 arms 使用相同内容摘要的 stock oracle；不可把不同供应库产生的 solve 差异归因给规划器。
- Host pipeline：所有路线经过相同 canonical identity、admission、reaction validation、exact evidence、condition resolution、stock/procurement 和 experimental program gate。
- 目标特判：禁止使用 target ID、名称、SMILES、公开参考路线或 dataset label 分配 provider、预算或评分。

截至 2026-08-12 的公开 SynthAtlas 数据快照只作为候选外部战略源：

- `dataVer=20260809-00e8823-5a1cf6`；manifest 报告 1,098 compounds、3,243 routes、33,145 reactions、36,434 molecules。
- `manifest.json` SHA-256：`15ebf813335d5f95b216b63b9d9728ab3bc62a4332039728a2857838cdfe7731`。
- `index.json` SHA-256：`2d23854cf76cdf6bea2e14f2ed5ab3e98cb07582b98cb666bc24d4025d0f76d9`。
- 实验实现不得在核心代码硬编码该版本；运行 manifest 必须记录 URL、版本、抓取时间、文件摘要和所选 route ids。
- SynthEx 代码状态必须独立冻结：截至 2026-08-12，官方仓库 commit `5f41a6b21e3906fde93e84c88bb91f9dc4d37e6f` 尚未发布实现或正式 ReactionJSON/RouteJSON specification；README 的 20.8%/67.2% 与 arXiv v1 Table 2 的 25.0%/63.9% 不一致。对照只能使用明确命名的公开 route snapshot 或未来指定 commit，不能混用两个结果口径。

## 路线价值主终点

每个 target 的候选路线先按冻结规则选入随机化盲评包。至少三名独立合成化学家在不知道系统、论文和库存闭合标签的情况下完成一个短表：

- route-defining insight、关键转化可信度、立体/选择性逻辑、收敛/路线经济性、风险与备选各 1--5 分；
- `pursue / redesign / stop` 实验决策；
- 一句决定性理由和一个最严重 blocker。

主要统计量为 target-level paired preference、`pursue` rate、各维度分布与评分者异质性。先评价路线的内在价值，再在 evaluator-only 阶段判断是否复现论文战略。不能用同一 backbone 的 Critic 代替独立专家，也不能把多条路线当成独立 target 扩大样本量。

## 六级闭环次级终点

每条路线分别计算，不能用总数或别的路线替代最弱轴：

| Level | 名称 | 完成条件 | 明确不等价 |
|---|---|---|---|
| C0 | Strategic structure | 全部步骤可解析，目标一致、路线连通、无环、前体 multiplicity 与立体身份保留 | 提供方声称 solved |
| C1 | Canonical materialization | 每步通过宿主 admission 并进入 canonical hypergraph | LLM Critic pass |
| C2 | Host reaction validation | 每个物化 edge 都有宿主验证的 reaction proof | 有相似反应或命名反应 |
| C3 | Exact source/procedure | 每步绑定精确反应记录和可定位 procedure；独立来源按验收策略计算 | Reaxys close/related count、综述描述 |
| C4 | Exact complete conditions | 每步精确 procedure 的 agents、solvent、temperature、time 四组完整 | 自然语言建议或模型预测条件 |
| C5 | Stock/procurement closure | 每个叶节点在冻结 oracle 边界内有有效观察；procurement 另报 | “常见 building block” |
| C6 | Experimental closure | 预先定义的 Program/实验结果绑定精确边界并通过领域 gate | 纸面可行性、专家评分或内部自评 |

闭环终点报告固定预算内达到的最高 level、C3/C4/C6 route rate、首次到达各 level 的 wall time/资源，以及最弱轴分布。它们解释路线价值能否转化为可执行证据，但不替代专家主终点。`all leaves in stock` 只对应 C5 的一部分，不能单独命名为路线成功或实验成功。

## 预算公平性

至少同时报告：

- wall time；
- LLM input/output tokens 与价格；
- native/template expansion 数、CPU/GPU 时间；
- evidence/stock/validation/experiment task 数；
- 每个闭环 level 的 value-per-cost Pareto frontier。

不得把一个大型 LLM inference 与一个模板网络 expansion 当作相同“call”。允许报告 reach-under-generous-budget，但必须与 compute-matched 结果分开。

## 统计分析

- Target-level paired comparison；同一 target 的多路线先按预注册 selector 聚合，不把路线当独立 target。
- 二元终点报告配对差和 bootstrap CI；时间/成本报告 ECDF、median/IQR 和受限均值。
- 多专家评分时，以 rater/item 的依赖结构做 cluster bootstrap，不把每条评分当独立样本。
- 必报 unsolved、invalid、disconnected、unvalidated、source-missing、condition-incomplete、stock-open、experiment-negative 八类失败。
- 独立报告 provider unique contribution、arm complementarity、adaptive selection regret 和 confidence calibration。

## 停止与升级规则

- Pilot 可在预定样本完成后停止；不得看到结果后删除困难 target 或新增目标专用规则。
- 若任何 arm 的 C0 失败率超过 20%，先修复通用 interchange/admission，再重启全 pilot；旧结果保留为失败记录。
- 若所有 arms 在 C3 之前被相同外部来源访问瓶颈阻断，结论只能是“证据层未能区分”，不能宣称规划器等价。
- 扩大到 190 或更大样本前，冻结 commit、配置、模型/stock 内容摘要、数据 manifest、随机种子、预算和分析脚本。

## 实验升级规则

- 任何路线都不能仅凭 stock-closed 或 Critic pass 进入湿实验。
- 首轮只选择通过 Host 图回放、无关键 identity/stereo 缺陷、关键反应完成独立审查、专家达到预注册价值门且完成安全审查的候选。
- 实验单位首先是一到两个 route-defining transformations，而不是立即承诺完整全合成；成功后才升级到短路线片段，再决定是否推进全路线。
- 移交包必须冻结 route/structure 版本、前向反应、规模、材料、conditions/precedent、分析 readout、风险、停止条件和 fallback。
- 失败实验属于 C6 的负结果并保留在分母；只能根据预先定义的实验设计复测，不能因结果不理想删除候选。

## 最小发表主张

只有实际结果满足后才允许写入论文：

1. provider-neutral 路线输入可在不继承外部自报权威的情况下确定性重放；
2. 同一系统能把战略 reach 与反应、证据、条件、库存和实验闭环分开量化；
3. 在至少一个预注册 matched benchmark 上，系统提高独立专家的路线价值或实验推进率，而非仅提高 C0/C5；
4. 对高价值候选，C3/C4 或 C6 的提升在 compute-matched 或完整成本曲线下仍成立，并通过目标盲、内容摘要绑定和失败全量报告审计。
