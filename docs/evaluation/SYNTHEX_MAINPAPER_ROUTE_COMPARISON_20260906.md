> ????????[4 ?????? 14 ???????](../../results/discussion/synthex-mainpaper-comparison-20260906/RESULTS.md)??????????????????

# SynthEx 主论文路线对照与 P0 修复后补评

日期：2026-09-06。

**可以找到支撑故事的主论文案例，但目前证据支持“识别并处理未落实的化学前提”，还不支持“已经修好 SynthEx 的同一条天然产物路线”。** 最有价值的是 Fig. 1d 的 Cyclopiamine B：关键串联关环的上游立体前提仍未落实。对它进行一次隐藏来源标签和已有评语的模型复核后，我们也识别到了这项依赖。与此同时，BCH 补评暴露出我们自己的同类缺口：局部 Appel 反转修对，并不保证下游桥环闭合的几何正确。

这不是主论文路线的全面错误率统计。本次核对 Fig. 1d 的三个展示分子和 Fig. 5 的修复实例，并阅读 Fig. 2 三个案例的正文来限定比较口径；没有将五个本地测试分子冒充主论文的同靶点比较。

## 1. 来源与比较边界

- [SynthEx 主论文 v1](https://arxiv.org/abs/2608.07454v1)，本地 PDF 位于 `evidence/synthex_paper_card_20260820/`。已检查 [Fig. 1，PDF 第 5 页](evidence/synthex_mainpaper_review_20260906/fig1-mainpaper.png) 和 [Fig. 5，PDF 第 19 页](evidence/synthex_mainpaper_review_20260906/fig5-mainpaper.png)。
- 2026-09-06 从 SynthAtlas 公开数据域重新获取六份路线 JSON，版本为 `20260809-00e8823-5a1cf6`。[来源与获取时间](evidence/synthex_mainpaper_review_20260906/sources.json)。Cyclopiamine B S3、Traversiadiene S2 与 9 月 5 日缓存内容相同。主站配置请求返回 403，本次使用此前已取得的公开配置所指向的固定版本；不声称已确认主站当天切换情况。
- 下文 `idx` 是公开 JSON 的零起始、目标向前体排列的步骤索引。它不等于实验正向步骤号，汇聚路线也不能简单用总步数倒排。
- 网站 `Good` 是综合评估等级，不能直接等同于我们的 `viable`。公开数据中这些路线的 `solved=false`；不能说作者宣称已实验完成。风险字段属于公开分析注释；`critic_verdict=null` 不能证明原始 Critic 没有发现问题。

## 2. 主论文到底有哪些可讨论的问题

| 主论文案例 | 可核实的事实 | 合理结论 | 不能据此声称什么 |
|---|---|---|---|
| **Cyclopiamine B，Fig. 1d，对应 S3 串联骨架** | `idx=6` 的片段偶联生成指定的新立体中心；公开 `stereo_comment` 称缺少明确控制策略是 “a major flaw”，并说明该中心影响后续串联反应。`idx=1` 正是图示 aza-Michael/Michael 串联。路线仍为 Good、overall 8、feasibility 7 | 上游立体前提未落实，下游关键反应及整路线不能仅凭图结构认定已可执行 | 不是“完全无手性来源”：前体已有手性吡咯烷中心。也不是已证明串联反应不可能 |
| **Traversiadiene，Fig. 1d，对应 S2 的 ketyl/Grob 序列** | 展示 4 步；叶 Mol E 被公开注释称为非商品、需专门多步合成的复杂三环中间体，`productive=false` | 关键骨架重组策略有价值，但困难的上游合成未包含在这 4 步中；应比较同一起始边界 | 不能把它当成 4 步全合成，也不能说作者隐瞒了未闭合状态；我们历史 7 步库存闭合候选不能直接拿来比步数胜负 |
| **Dibohemamine A，Fig. 1d，对应 S3 的 [3+2] 步骤** | `idx=3/5` 的现有条件写了 AgOAc/手性 BOX；图示也标出 BOX。但 `stereo_comment`、`main_risk` 仍说缺少手性催化剂，`asymmetric=[]` | 条件与分析注释存在不同步；具体 BOX 构型、底物适用性仍待证明 | **排除“没有任何手性催化剂”这条指控**。两个偶联单体在 SMILES 中相同，目标拆回后的两半也相同；“enantiomeric monomers”措辞不准，不能升级成目标结构接错 |
| **Monascuspirolide A，Fig. 5d** | 正文明示延后螺缩酮化、提前 HWE、插入保护/脱保护，修复酸敏感性与 Claisen 竞争反应 | 对方已经有实质性的路线级 Critic–Editor 和顺序修复，是比较时必须承认的能力 | 不能把“有 Critic”“能重排/插保护”“有回退修订”作为我们独有贡献。公开 S1/S3 有相关序列，但没有完整迭代对应记录，不指定它们就是图中的 iteration 6 |

直接来源：[Cyclopiamine B S3](https://synthatlas.epfl.ch/#/route/npa006547_s3)、[Traversiadiene S2](https://synthatlas.epfl.ch/#/route/npa000656_s2)、[Dibohemamine A S3](https://synthatlas.epfl.ch/#/route/npa030041_s3)。完整快照在上述来源目录。

Fig. 2 的 Okaramine M、Melonine、Chanoclavine→Lysergol 是另外三个案例。论文分别讨论文献战略重现、不同连接链构象的替代方案、尚无实验确认的建议；这些限制本身不能充当路线错误证据。本轮没有确认它们的具体化学错误。此前的 Terreulactone C 无明确手性来源案例目前只作为网站补充案例，未确认属于主论文展示。

## 3. 对同一条 Cyclopiamine B 路线做了什么

新增 **1 次 Astra medium 评审，10 个步骤全部评审、实际工具调用 0 次**。输入保留全部反应 SMILES、两种条件字段、原提案的反应描述及战略 query，隐藏分子名称、作者、网站等级、`main_risk`、`stereo_comment` 和其他已有评语。沿用生产 Route Critic 的化学判定口径；公开 JSON 没有原子映射和操作程序，因此改用明确标注的非映射输入，不补造操作或因缺失映射判错。

结果为 **uncertain：3 步 pass、7 步 uncertain、0 步 reject**。

- `idx=6`：识别到远端已有手性中心不足以确定新中心；还指出另一个受体位点和后续自由基关环的竞争风险。
- `idx=1`：认可串联成键在结构上合理，保留新季碳中心的非对映选择性问题。
- 整路线明确指出：“Unresolved stereoselectivity in fragment coupling propagates into both downstream ring-forming events.”

这增加了**同靶点问题识别**的过程证据。它仍是一位模型评审者的一次判断，不是独立专家验证，也没有提交、接回或复评一条修订后的 Cyclopiamine B 路线。不能用其 uncertain 与网站 Good 的不同直接计算质量提升。

证据：[完整输入](../../results/discussion/p0-route-reassessment-20260906/external-route-diagnostic/prompt.txt)、[结果与完整 Worker 记录](../../results/discussion/p0-route-reassessment-20260906/external-route-diagnostic/result.json)、[适配脚本](evidence/synthex_mainpaper_review_20260906/review_external.py)。

## 4. BCH／TCF 的实际补评结果

读取原始规范图和 Builder IO，通过当前生产函数恢复两处被误删的条件，仅在内存副本和独立目录中构造输入。沿用原模型 `gpt-6-astra`、medium、600 秒超时、96,000 字节输入上限，每个选中的路线族评审一次；没有再次进行战略搜索，也没有修改原运行状态。这些是修复后输入的新评审，不能替换 9 月 5 日实验结果或算作同一次实验的分数升级。

| 路线族 | 步骤数 | 新整路线判断 | 最关键的发现 |
|---|---:|---|---|
| TCF 1 | 8 | uncertain | 拥挤 Mannich、醇活化及四元环闭合仍待验证 |
| TCF 2 | 6 | uncertain | **现在明确读到 UV/xanthone**，不再误称缺少光激发；仍无法证明该亚胺/烯烃组合的光反应活性和区域选择性 |
| TCF 3 | 4 | uncertain | 酸性条件下四元环 Mannich 闭合仍有竞争途径风险 |
| BCH 1 | 8 | uncertain | 硼化选择性会影响后续光环化；两项均未被产物立体标注证明 |
| BCH 2 | 7 | uncertain | **拆分步骤及目标桥头构型要求已被读到**；但拆分、Wolff 缩环、保持笼架的脱羧及后续硼化仍需验证 |
| BCH 3 | 7 | **reject** | Appel 步骤的反转本身得到 pass；**它给出的溴代前体仍不满足下游桥环闭合的背面进攻几何** |

六次有效评审覆盖 40 个步骤：18 pass、21 uncertain、1 reject。BCH 原先因立体等价性误拦截而缺失的三份最终评审均已完成，保存在独立补评目录；原注册表未改写。

**BCH 3 是这轮最重要的新发现。** 旧修订解决了“Appel 宣称反转却画成保留”，但固定的下游溴代前体本身还存在几何问题。新 Critic 要求协调“醇异构体选择→Appel→桥环闭合”三个步骤，不能再次只改醇端。

为核对“同面”判断，对保存的溴代前体做了 RDKit ETKDG/UFF 静态构象检查：8/8 个构象的侧链和 Br 在环平面同侧；仅反转 C16 的对照为 0/8 同侧。这支持该相对几何诊断，**不提供活化能、反应速率或收率证据**，对照也不是已接回的修复路线。[计算结果及方法](evidence/p0_route_reassessment_20260906/bch_ring_face_check.json)。

复评数据：[汇总与用量](evidence/p0_route_reassessment_20260906/results.json)、[恢复和调用脚本](evidence/p0_route_reassessment_20260906/reevaluate.py)、[TCF 2](../../results/discussion/p0-route-reassessment-20260906/tcf/branch-2/result.json)、[BCH 3](../../results/discussion/p0-route-reassessment-20260906/bch/branch-3/result.json)。

本轮六次内部复评消耗 input 646,700、其中 cached input 300,416、output 10,297；另一次外部路线诊断为 input 55,301、output 2,270。缓存数是输入的子集，reasoning 也是输出的子集，均不重复加总。内部复评继承生产工具设置，实际有 9 次网页搜索/打开、5 次结构检查；不能把内部六次调用当成零检索对照。

## 5. 论文故事与接下来的最小实验

建议把主张写成：

> 天然产物多步逆合成的核心困难之一，是把一个吸引人的战略逐步落实为满足相互依赖的成键、反应性和立体前提的路线。AutoPlanner 持续提出和更新战略，在关键事件及新增前体改变既有前提时进行选择性评审，并按依赖范围修订路线；结构接回、模型判断和实验可信度分别记录。

目前的证据强度应分清：

| 主张 | 当前证据 |
|---|---|
| SynthEx 主论文展示路线存在未落实的上游立体前提 | **有**：Cyclopiamine B 公开结构、公开分析注释及本轮同靶点评审一致 |
| 我们能识别同靶点的这类问题 | **有初步证据**：一次隐藏来源标签的诊断复现关键依赖；不是准确率结论 |
| 我们已通用修复条件丢失、立体等价性误拦截、失败评审覆盖 | 已有确定性回放；本轮新增真实模型证据支持前两项的输入/调用恢复。第三项沿用原先回放证据 |
| 我们已彻底修好该类跨步立体问题 | **未达到**：BCH 3 说明局部修正仍会留下下游几何矛盾 |
| 我们已修好 SynthEx 的 Cyclopiamine B 路线 | **尚无证据**：未进行同目标完整修订、接回与复评 |
| 我们比 SynthEx 提高了多少 | **不能量化**：尚缺同输入、同预算、独立评价的对照 |

下一步应先把 BCH 3 的三个相关步骤作为一个真实修复片段，保持目标及其余路线边界，验证修订是否消除具体几何矛盾、是否产生新问题。随后在固定的同一组主论文输入上比较“只在末尾 Critic”与“持续战略＋关键节点 Critic”，统一预算和工具条件；由隐藏系统来源的专家判定关键问题是否被发现、是否真实修正、以及合理创新是否被误拒。

这比单纯展示更多搜索次数、更高内部评分或一条缩短的路线更能检验核心方法。文献全合成数据库应提供独立参照步骤、立体前提与原始证据，并隔离生成与评价输入；它能支撑该实验，但目前仍在策展，不能当成已完成的正式 benchmark。
