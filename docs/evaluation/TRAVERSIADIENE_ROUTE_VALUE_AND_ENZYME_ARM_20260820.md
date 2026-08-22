# Traversiadiene 路线价值与酶法增益臂收束审查

日期：2026-08-20

## 当前化学主臂结论

- 运行：`synthexfig1-001-aizmcts-paid-v2`
- 论文等价结果：`paper_equivalent_solved=true`，`B4=true`
- 路线：7 步，4 个终端叶全部命中同一冻结 ZINC + eMolecules 库存。
- 搜索价值：证明 `Codex ReactionJSON policy + AiZynthFinder MCTS/UCB + exact stock` 热路径已经可产生库存闭合路线。
- 化学结论：不能作为实验执行路线。独立 Critic 的总体判定为 `reject`，且 Editor 写回失败。

### 主要硬伤

1. 第 2 步所谓片段偶联的产物与第一个前体相同，第二个烃片段没有进入产物；两个前体也都没有互补反应手柄。
2. 第 5 步声称 Sonogashira 偶联，但烯基一侧是未官能团化的烃，不是烯基卤化物或等效亲电体。
3. 两次炔烃半氢化没有锁定所需 E/Z 几何；后续多烯级联对几何高度敏感。
4. 末端烯烃氧化脱氢在多烯底物上的区域选择性和异构化风险未解决。
5. 路线定义的阳离子多烯环化没有明确离去/起始手柄，拓扑、终止和非对映选择性均未证明。

因此，这条路线的价值定位是：**高搜索价值、中等战略启发、低实验可执行性**。它适合作为 B4 canary 和“B4 不等于 B2”的正例，不适合作为合成交付路线。

## 为什么酶法臂值得独立运行

Traversiadiene 并非只有泛化的“可能可由萜烯酶形成”推断。现有主文、补充资料和专利公开了 Traversiadiene cyclase / AaTS（AaT09930）相关体系，并报告异源体系生成、分离及 NMR 结构确认的 Traversiadiene。该生物转化与当前化学路线最薄弱的三环骨架和多立体中心构建直接对应。

酶臂应检验的不是“是否能给任何一步贴 enzyme 标签”，而是：

- 能否提出 GGPP/等效 C20 焦磷酸底物到 Traversiadiene 的精确 substrate-product 边界；
- 能否给出 diterpene cyclase / AaTS 类催化剂假设、金属离子/焦磷酸离去相关辅因子账本和异源表达方案；
- 能否保留化学备用路线，并把未经验证的省步数保持为 `null`；
- 能否在 exact-substrate 生物催化验证之前保持 hypothesis-only。

## 已接入的 companion arm

仓库已有独立 `enzyme_advantage` 模式，且与论文匹配主臂隔离：

- 同一目标、同一冻结库存、同一模型和 3 × 25 节点上限；
- 每一步仍由 host-replayed ReactionJSON 绑定精确分子边界；
- 生物步骤要求 enzyme/EC/candidate、选择性目标、底物范围依据、辅因子评估、化学 fallback 和可证伪验证计划；
- exact-substrate biocatalysis proof 未通过前，不授予反应证明或净省步结论。

本轮独立付费运行：`synthexfig1-001-enzyme-paid-v1`。

## 酶法 companion arm 实际结果

运行已完成，终态为 `scientific_status=unresolved`，并非运行失败：

- B0/B1 通过，B2/B3/B4/B5 未通过；`paper_equivalent_solved=false`。
- 形成 1 条目标根、RouteJSON 图闭合的 4 步路线；3 个叶中 2 个命中冻结库存，`stock_closure_rate=0.666667`。
- 16 次模型调用，247,472 input token、37,571 output token，模型累计时间 685.875 s；另执行 1 次 AiZynthFinder frontier short-tail。
- 路线由 4 条物化边组成，只有“C20 多烯醛还原为烯丙醇”达到 L2 host validation；其余 3 条仍为 L1。
- 独立 Critic 对酶主链的判定为 `uncertain`：机理类别合理，但没有 exact enzyme identity、exact substrate acceptance 和产物保真证明。

路线骨架为：

1. C20 多烯二磷酸经 class-I diterpene cyclase 假设性级联环化得到目标；
2. C20 多烯醇与焦磷酸片段形成二磷酸；
3. C20 多烯醛还原为烯丙醇；
4. 磷酸组分脱水形成焦磷酸。

### 真正的亮点

目标在 planner 中仍是 opaque case，且禁用了路线文献输入；Codex 仅从结构出发便提出：

- 线性 C20 烯丙基二磷酸作为边界底物；
- class-I diterpene cyclase（EC 4.2.3.-）；
- 一次构建 3 条成环键和相关立体中心；
- Mg2+ 依赖、无酶/热失活/去金属对照、产物谱、NMR 与手性验证。

这与公开的 AaTS/AaT09930 Traversiadiene cyclase 体系在战略层面高度吻合。因此酶臂证明了一个真实优势：系统能在不知道目标名称和已知路线的情况下，把化学主臂最薄弱的多烯环化与立体控制压缩为正确类别的生物催化超级步骤。

### 仍然不能收束的硬伤

1. **精确底物不对。** 本轮生成的 C20 二磷酸 InChIKey 为 `JXTSLGKOONGGCS-UHFFFAOYSA-K`；权威 GGPP 连接关系的 InChIKey 为 `OINNEUNVOZHBOX-UHFFFAOYSA-K`。二者分子式相同但不是同一结构。
2. **几何未锁定。** 已知 all-trans-GGPP 需要明确的烯烃几何；本轮结构没有编码相应 E/Z 边界。
3. **上游供料低价值。** 把焦磷酸继续拆成两个磷酸组分没有解决核心碳骨架库存问题，还引入了混合缩合磷酸盐风险。
4. **主要碳叶未闭合。** 未命中库存的是 C20 多烯醛；AiZynthFinder short-tail 已执行但未找到库存闭合尾段。
5. **酶步骤仍未验证。** 路线正确保留 `unvalidated_biocatalytic_edge_ids`，因此 credited biocatalytic step、superstep 和净省步数均为 0。
6. **三分支没有形成三条可比较路线。** 只有第一条酶战略保留为完整物化路线；第二条未进入最终闭合图，第三条因正交性重试仍未形成可用分支。

因此，该酶路线的价值定位是：**高战略价值、中等方法学价值、低当前交付价值**。它比现有化学路线更接近正确的路线定义事件，但还不是 exact-boundary 酶路线。

## 本轮运行时故障与修复

首次执行在完成 16 次付费调用后被 `campaign_action_pointer_binding_invalid` 中断。根因不是化学或缓存内容损坏，而是同一 semantic execution 在重新调度时，`round_robin_cursor`、scheduler label 等诊断字段改变了完整 Action 哈希；缓存 receipt 因而被错误拒绝。

已完成通用修复：

- execution id 继续严格绑定 action opportunity、opportunity set 和图 revision；
- outcome 内容哈希、execution id 和内部 pointer/outcome 一致性仍 fail-closed；
- 仅允许同一 execution id 跨调度诊断字段漂移复用 receipt；
- 真实恢复未重复模型调用，2.8 s 内从 revision 15 生成终态报告。

Critic → Editor 的另一个通用故障也已修复：surgical Editor 的非法 ReactionJSON 现在会接收 host replay 诊断并在有界窗口内重试，而不是一次失败就把六轮修复预算全部丢弃。该补丁在本次付费进程启动后才加载，因此本次路线结果不应被视为 Editor 修复后的重跑结果。

## 收束判据

### 最小可发表 canary

- 化学主臂至少 1 个盲目标达到 B4；已完成。
- 酶法 companion arm 在同一目标上产生结构闭合路线，并明确命中或错过已知 diterpene-cyclase 战略；已完成，战略命中、exact substrate 未命中。
- 修复 Critic → Editor 写回，使明显被拒绝的路线不能以“未修复 B4 路线”混入推荐结果；代码与聚焦测试已完成，尚需在下一次真实 Editor 触发中验证。
- 报告同时展示 paper solved、reaction validation、condition/evidence 和 enzyme exact-boundary 四条独立轴；基础字段已具备，汇总仍需统一。

单目标 canary 还剩两个决定性动作：

1. 对酶根节点做 exact-GGPP + all-trans 几何约束的局部修复，并拒绝同分异构底物；
2. 对修复后的开放碳叶执行标准 short-tail，随后绑定 AaTS/AaT09930 exact evidence 与条件，而不是继续拆无机磷酸叶。

### 可形成方法论文的小样本结论

- 完成冻结的 3-target Figure 1 化学主臂与酶法 companion 配对结果；当前仅完成 1/3 的配对运行，且该酶臂未 B4 闭合。
- 至少再加入一组不以萜烯环化为主的酶优势目标，避免酶臂只对 Traversiadiene 特例成立。
- 冻结路线质量评分：原子来源、反应手柄、立体/几何、关键步骤验证、库存闭合。
- 将“发现已知 exact enzyme precedent”与“LLM 自主提出 enzyme-class hypothesis”分开计分。

所以可以收束的是：**ReactionJSON/AiZ MCTS 热路径、B4 独立指标、盲酶战略发现能力**。尚不能收束的是：**实验可执行路线、exact enzyme/substrate 证明、酶法净省步优势和多目标统计结论**。下一轮应做一次根节点局部修复 canary；它通过后再扩到剩余 2 个 Figure 1 目标，而不应立即再跑完整 3 × 25。

## 来源

- [Yuan et al., *Efficient exploration of terpenoid biosynthetic gene clusters in filamentous fungi*, Nature Catalysis (2022)](https://doi.org/10.1038/s41929-022-00762-x).
- [中国专利 CN108239631B](https://patents.google.com/patent/CN108239631B/zh)，AaTS/GGPPS-Aa 异源体系与 Traversiadiene NMR 鉴定。
- [Natural Product Reports supplementary table](https://www.rsc.org/suppdata/d3/np/d3np00052d/d3np00052d1.pdf)，列出 AaT09930 与 Traversiadiene。
- [IUPHAR/BPS all-trans-geranylgeranyl diphosphate](https://www.guidetopharmacology.org/GRAC/LigandDisplayForward?ligandId=3052)，用于 exact GGPP 连接关系和 InChIKey 对照。
