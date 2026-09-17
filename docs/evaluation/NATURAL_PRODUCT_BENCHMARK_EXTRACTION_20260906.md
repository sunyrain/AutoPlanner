# 天然产物路线 benchmark：2026-09-06 提取与完善

本轮新增 9 条完整有序路线候选，补全化合物编号、结构、汇聚前体、逐步转化、收率边界和原文定位。累计 12 条，来自 4 篇论文。所有记录仍是来源复核后的自动候选，正式专家接纳目标仍为 0。

## 数据增量

| 内容 | 本轮前 | 本轮后 |
|---|---:|---:|
| 完整有序目标路线候选 | 3 | 12 |
| 有完整候选的独立论文 | 3 | 4 |
| 按目标展开的操作记录 | 14 | 65 |
| 按论文、前体/产物编号及条件去重的来源步骤 | 14 | 26 |
| 全范围 RDKit-valid 目标结构候选 | 145 | 147 |
| 来源图式一致的结构候选 | 73 | 75 |
| P0 可进入结构/路线复核的目标 | 32 | 41 |

新增 Xanchryone A、B、I、J、L、M、N 七条各六步路线，复用已核对的 K 路线前五步，但逐一绑定各自氨基酸、最终产物、收率和 SI 定位。八个目标属于同一集体合成论文，不能算八项独立策略。优化 Figure 3 路线与早期 Scheme 2 探索路径分开；31A/31B 保留为不可分离互变异构体混合物。M 的侧链手性依据来源与 RDKit CIP 核对，不能把反应中消失的氨基酸 α-手性中心机械复制到产物。

新增 Inonophenol A/B 两条汇聚路线。恢复了缓存 HTML 中链接的原始 Figure 1 与 Scheme 2，并核对实验节、SI 表征名称、分子式和 S 构型。A 为 5 个总操作、最长线性 4 步；B 为 4 个总操作、最长线性 3 步。独立的手性膦酸酯制备支路纳入操作总数。83% 是 HWE 与去保护两步合并收率；含副产物的约 92% 中间体不作纯品单步收率。图式与实验节的时间差异同时保留，不擅自合并为唯一条件。

## 通用流程修复

1. 同一论文的所有目标共享一次来源解析，避免集体合成重复读取正文和 SI；字节相同的附件只解析一次。
2. 所有来源先检查，再从正文和 SI 交替选取段落。正文达到上限不再提前终止 SI 检查；未找到目标名的段落仍不冒充完整路线。
3. 对缺来源目标保留显式缺口记录；支持按论文增量重提取并合并原结果。
4. 按提取文本识别“不同 PDF 哈希、相同正文”，排除冒充 SI 的重复正文；识别以 `.zip` 后缀保存的 Word OOXML，保留真实段落定位。
5. 路线验证检查目标祖先 DAG，拒绝夹入无关支路；分别计算总操作数与最长线性序列，验证合并收率的步骤范围。
6. 已有完整图式路线可以直接进入复核，不再被“段落未命中目标名”挡住。页面展示混合物、合并收率、来源差异和步数口径。
7. 重新生成复核包保留已有 `submission.json`，把新机器建议放入 `submission-template.json`。47 份原有提交逐文件对比均未改变。
8. 建立覆盖所有 253 个目标的 P0/P1 清单，自动生成 README、覆盖统计及结构摘要；数据盘点不再停留在旧的“1 条路线”。P1 当前进入完整覆盖清单，专家 HTML 复核包仍以 P0 为范围。

## 本轮读原文发现的后续重点

**Catunaregin / Epicatunaregin，10.1021/acscatal.6c03096。** 标记为 SI 的一个 PDF 实际重复正文；真正 SI 在另一个 ZIP 中，是 Word OOXML。已读取其中 7c/8c 表征。小试图式的 7c 74% 与放大实验的 7c 65%、8c 5% 必须分别保留，不能将 74% 分给两个目标。已有两个视觉候选使用相同的无立体 SMILES，且 Catunaregin 存在连接关系冲突；两个竞争图都能得到 C16H20O5，说明分子式和 RDKit 合法性不足以裁决结构。这两条尚未计入完整路线，下一步应优先核对笼状结构连接关系及两个异构体的对应。

**Glabridin，10.1016/j.bioorg.2026.110182。** 本文明确合成 (±)-Glabridin；名称检索得到的单一天然对映体不能替代实验产物。缓存正文有路线图说明，但缺原图，尚未填写有序结构路线。应先取得原图，再明确消旋产物的任务定义。

上述判断和来源定位已记录在 `benchmarks/recent_total_synthesis/curation_candidates/extraction_findings_20260906.jsonl`，也进入逐目标覆盖清单。

## 当前边界和继续顺序

候选范围为 133 篇论文 / 253 个目标，131 篇有来源包，242 个目标有段落线索。当前完整路线为 12/253，尚有 241 个目标未形成完整有序路线，178 个目标结构或立体化学未完全解决，2 篇候选论文仍缺来源包。本轮对 5 篇论文、14 个目标实际重提取了正文/SI；其余旧段落记录保留，并标明来源覆盖尚未按新流程检查。

下一批应增加独立论文与不同骨架的路线，尤其是需要汇聚、复杂立体控制与策略回退的多步合成，避免新增数量主要来自同一家族的末端变体。采用“来源图式与实验节 → 化合物身份 → 完整 DAG → 战略事件 → 独立专家复核”的顺序。相同论文族和共享路线不得跨训练/测试集；跨论文的路线复用仍需进一步检查。当前完整候选主要是较短路线，尚不能代表整个复杂天然产物合成任务分布。

## 结果与复现

- [完整路线与覆盖表](../../benchmarks/recent_total_synthesis/ROUTE_EXTRACTION_STATUS.md)
- [专家复核入口](../../output/recent_total_synthesis_review_packets/index.html)
- [Inonophenol A：含汇聚支路的完整候选](../../output/recent_total_synthesis_review_packets/target_truth/A_exact_source/target-slot-20f639a07b678e53/index.html)
- [Xanchryone M：手性侧链候选](../../output/recent_total_synthesis_review_packets/target_truth/A_exact_source/target-slot-f14c284311416a0f/index.html)

```powershell
python scripts/extract_recent_total_synthesis_route_candidates.py --paper-id <paper-id> --merge-output
python scripts/build_recent_total_synthesis_benchmark.py --offline
python scripts/build_recent_total_synthesis_review_packets.py
```

完整候选已通过来源哈希、结构可解析性、分子式、目标结构一致性及反应 DAG 检查。87 项相关测试通过；数据集不变量检查通过；58 个 HTML 页面的本地链接均存在；本地浏览器抽查 Inonophenol A 和 Xanchryone M 页面，操作数量正确且无横向溢出。47 份既有专家提交内容保持不变。这些检查验证提取合同与展示，不替代专家的化学判断。
