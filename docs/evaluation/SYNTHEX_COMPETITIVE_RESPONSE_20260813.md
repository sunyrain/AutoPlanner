# SynthEx / SynthAtlas 竞争响应与 AutoPlanner 推进结果

日期：2026-08-13

## 结论

SynthEx 最亮眼的不是单步预测，而是把策略生成、显式 ReactionJSON 编辑和迭代 route critic 组合成面向复杂天然产物的完整叙事。AutoPlanner 不应复制“同一模型反复修改路线”的表面形式，而应把论文明确留空的边界做成主要贡献：提供方无关的 strategy-to-experiment closure、独立 host 权限、精确来源/条件/库存/实验 Claim，以及可重放的资源与失败账本。

目前已经完成三项实质推进：

1. SynthAtlas 路线可通过 provider-neutral 导入和 canonical host gate 重放，外部的 `solved`、`feasible`、条件文本或 critic verdict 均不继承证明权威；
2. clean-20 外部快照已有 20/20 target C0/C1，route-level C0=58/59、C1=40/59，C2–C6 保持开放；三条 live arm 已完成盲测 preflight，正式运行必须独立完成后再比较；
3. 真实官方 EPO 三案例的 independent critic 消融已完成：同 backbone 自评能撤回无依据条件，但不能恢复精确事实；只有新增、摘要绑定、结构一致的 host evidence 能触发 exact-source repair。

## 对方工作的亮点

- Strategy-first：先提出竞争性策略，再展开完整路线，适合评估会聚性成键和复杂骨架构建；
- 可编辑中间表示：ReactionJSON / RouteJSON 把反应编辑定位到局部步骤，便于迭代而非整条重生成；
- 明确的多角色流水线：Strategy、Expansion、Critic、Editor、Analyst 形成连贯演示；
- 数据规模与天然产物定位强：1,098 个 target、3,243 条路线和 33,145 个 atom-mapped steps 对社区有潜在训练与评测价值；
- 进行了 10 位化学家的盲评，不只依赖自动指标。

## 关键不足

- Critic、Editor 和 Analyst 使用同一 backbone。blocking rate 下降证明的是对同一内部准则的收敛，不是独立化学事实或实验可行性；
- 专家比较只覆盖已共享策略框架的 47 个 target，主要评估所提出步骤，不证明系统寻找可行策略的端到端可靠性；
- 没有实验验证，论文也明确承认未验证立体化学结果；
- 未闭合精确 procedure、条件完整性、实际采购/库存和实验结果；
- 比较是 reach comparison，不是 compute-matched comparison；单次随机运行、seed 和方差报告不足；
- 截至本次审计，官方仓库仍未发布实现代码和完整正式字段规范；README 与论文 v1 的部分成功率数字不一致，不能把 README 数字当作稳定论文基线。

## 已完成的真实 independent critic 消融

数据集是三份不同反应类型的官方 EPO ST.36 XML procedure：Vismodegib 酰氯酰胺化、DMB-S-MMP 硫酯化、Nirmatrelvir C4 酸性酯水解。三份 source、元素范围、结构解析、procedure text 和完整条件均有 SHA-256 / binding ID，并可零网络、零模型离线重放。

模型只看到规范化 product/reactants 和空条件；专利号、target 名、来源和参考条件均隐藏。每例先生成初稿，再由同一 `gpt-5.5 low` backbone 自评一次。host 随后才加载此前冻结的官方 evidence，并要求反应结构身份、source artifact、source binding 和 procedure text digest 同时匹配才触发 repair。

| Arm | Assessed | 条件字段存在率 | 冻结 oracle recall | source-text exact recall | 非匹配字段率 | exact-source condition closure | 模型成本 |
|---|---:|---:|---:|---:|---:|---:|---:|
| blind initial model draft | 3/3 | 77.8% | 8.3% | 0.0% | 100.0% | 0/3 | 3 calls；50,160 in / 2,514 out |
| same-backbone self-critique | 3/3 | 4.2% | 0.0% | 0.0% | 33.3% | 0/3 | 3 calls；53,017 in / 2,617 out |
| evidence-triggered host repair | 3/3 | 100.0% | 100.0% | 100.0% | 0.0% | 3/3 | 0 calls / 0 tokens（离线 replay） |

主指标使用在 extraction 之外预先冻结的 oracle criteria：`equals` 要求 time/yield 等字段相等，`contains` 要求 solvent/base 包含指定值。source-text exact recall 另按完整规范化字段严格相等判定，两者都不使用语义相似度。初稿提出的 DCM/THF、LiOH、室温或 1–4 h 等常规条件虽然合理，但与真实 procedure 不同，不能算精确命中。自评的有效作用是删除无依据细节，不是生成被隐藏的 17 h、2 h、3 days、实际溶剂、收率和后处理。

该结果只支持一个有限结论：在真实 exact-source procedure closure 上，新增独立证据优于同 backbone 自评。它不证明三项实验在本项目中重新执行成功，也不外推完整路线、库存闭合或天然产物总体表现。

## 可发表方向

论文主张应从“我们也能生成很多路线”改为：

> AutoPlanner is a provider-neutral, evidence-triggered synthesis campaign system that separates strategic proposal from independently auditable reaction, procedure, stock, and experimental authority.

主实验应包括：

- matched clean-20：SynthAtlas snapshot、Codex-only、ChemEnzy-only、unified-adaptive，按 target 配对并报告 C0–C6；
- critic 消融：same-backbone self-critique 对 evidence-triggered repair；
- exact procedure / conditions：真实 source 子集的字段级 recall、错误新增率和结构回归；
- 资源：模型 calls/tokens、ChemEnzy expansions、host tasks、wall time 和恢复次数；
- failure taxonomy：unsolved、invalid、unvalidated、source-missing、condition-incomplete、stock-open 和 experiment-negative 分开。

不能提前主张 unified 或 AutoPlanner 优于 SynthEx；必须等 clean-20 live arms 完整结束。当前三案例 critic 结果是机制证据，不是大样本 superiority claim。

## 机器可读证据

- `results/shared/independent_critic_ablation_20260813/execution-receipt.json`
- `results/shared/independent_critic_ablation_20260813/summary.json`
- `results/shared/independent_critic_ablation_20260813/summary.md`
- `results/shared/patent_procedure_gate_20260717/patent-procedure-gate/summary.json`
- `results/shared/synthatlas_strategy_closure_clean20_external_snapshot/summary.json`
