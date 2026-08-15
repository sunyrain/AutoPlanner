# SynthAtlas clean-20 external snapshot baseline（2026-08-12）

## 结论

本轮完成的不是 live SynthEx 复现，而是一个公开路线快照经过 AutoPlanner 同一宿主闭环的基线。clean-20 中，20/20 靶标至少有一条公开路线通过 C0 与 C1；59 条公开路线中 58 条通过 C0（98.3%），40 条完整通过 C1（67.8%）。C2–C6 均未由导入动作关闭。

这组结果说明 SynthAtlas 的战略结构覆盖很强，但公开快照距离可验证、可溯源、条件完整和实验闭环仍有明确缺口。它也验证了 AutoPlanner 的竞争方向不应只是“再生成更多路线”，而应把 provider-neutral 战略输入可靠推进到独立闭环。

## 冻结对象

- 公开数据版本：`20260809-00e8823-5a1cf6`。
- 公开 manifest SHA-256：`15ebf813335d5f95b216b63b9d9728ab3bc62a4332039728a2857838cdfe7731`。
- 公开 index SHA-256：`2d23854cf76cdf6bea2e14f2ed5ab3e98cb07582b98cb666bc24d4025d0f76d9`。
- SynthEx 仓库观测 commit：`5f41a6b21e3906fde93e84c88bb91f9dc4d37e6f`；该 commit 尚未发布 live 实现与正式 ReactionJSON/RouteJSON 规范。
- 靶标选择：按公开 index 首次出现顺序扫描唯一 canonical target；在任何 live arm 运行前，排除其目标身份、公开同义词或 ≥8 重原子路线中间体已存在于 tracked 非库存知识制品的候选，直到收满 20 个。
- 候选审计：扫描 29 个候选，预运行排除 9 个，未按模型结果排除；所有排除原因保留于 evaluator-only pack。
- 路线变体：20 个靶标的全部 59 条公开变体，无结果后删样。
- 盲测 preflight：20/20 通过。
- 冻结库存：Retro*-190 eMolecules index，23,081,629 个成员，SHA-256 `30c828d6780e534d8368f4eb74f844c889683453080d44053ba298a7bebdd79c`。

Planner-facing manifest 只包含 opaque target、通用 acceptance 和预算。公开名称、路线、同义词及关键中间体只保存在 `data_external` evaluator pack；live planner 不得读取该 pack。

## 外部快照臂结果

| 终点 | Route-level | Target 至少一条 | 含义 |
|---|---:|---:|---|
| C0 战略结构 | 58/59（98.3%） | 20/20 | 目标一致、结构可解析、路线连通且无环 |
| C1 canonical materialization | 40/59（67.8%） | 20/20 | 全部步骤进入宿主 canonical hypergraph |
| C2 host reaction validation | 0/59 | 0/20 | 导入不授予反应证明 |
| C3 exact source/procedure | 0/59 | 0/20 | 公开 close/related count 与描述不等于精确 procedure |
| C4 complete exact conditions | 0/59 | 0/20 | 自然语言条件不等于来源绑定的完整条件 |
| C5 frozen stock closure | 0/59 | 0/20 | 本臂仅做结构导入/物化，尚未运行叶节点 stock audit |
| C6 experimental closure | 未评估 | 未评估 | 需要绑定真实 Program/实验结果 |

资源账本：0 次路线生成模型调用、0 input/output token；495 个确定性宿主 materialization task，495 个 accepted expansion。每条变体使用隔离 RunKernel，避免一条变体的 ancestor state 影响另一条变体；所有 RunKernel 绑定同一个冻结 stock-oracle binding。

## 失败分类

- 17 条路线：至少一步 `element_inventory_not_conserved`。
- 1 条路线：`element_inventory_not_conserved` 且存在 `surplus_advanced_precursor_fragment`。
- 1 条路线：公开路线含重复 product / 无变化步骤，C0 失败。

元素缺口主要见于保护、卤化或官能团引入步骤：公开 `rxn_smiles` 只列主要结构前体，把贡献 Si、Br、Boc 等原子的试剂留在自然语言条件中。AutoPlanner 当前不会把条件文本升级为结构权威，因此将这些步骤保留为可解释的 L0/C0 route hypothesis，而不伪装为 C1 canonical edge。未来正确修复是 provider 给出结构化 reagent contribution 或正式 ReactionJSON edit，不是针对 SynthAtlas 放松元素守恒。

## 可发表含义与边界

亮点是：外部系统擅长 C0 reach，AutoPlanner 可作为 provider-neutral closure substrate，明确量化从“看起来有路线”到“可物化、可验证、可溯源、条件完整、库存闭合和实验闭合”的逐级损失。

本结果不能支持以下声明：

- 不能称为 live SynthEx 成本或 solve-rate 复现；公开快照生成成本不可观测。
- 不能声称 AutoPlanner 已在 C3/C4/C6 超越 SynthEx；clean-20 的 Codex-only、ChemEnzy-only 和 unified-adaptive 三个 live arms 尚未运行。
- 不能把 C0 或外部 `solved` 当实验可行性。
- 不能把公开自然语言条件计作 exact procedure evidence。

## 下一门

在不改变 clean-20 manifest、预算、库存、泄漏包和分析脚本的前提下，依次运行 Codex-only、ChemEnzy-only 与 unified-adaptive 三臂。三臂完成前只报告冻结、preflight 和 external baseline；完成后再做 target-level paired C0–C6、provider unique contribution、互补性、失败分类和完整成本曲线。
