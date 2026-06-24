# SMILES-First Literature Strategic Retrosynthesis Workflow

本说明把 Bufotalin 试跑中形成的经验整理成可复用 agent 流程。目标是在用户只给
新 SMILES、没有本地专用规则或历史结果时，仍能先用通用规划器处理普通步骤，在
高级天然产物/复杂骨架拆解失败处调用文献战略断键，并输出可机读路线材料和路线图。

整合状态：

```text
2026-06-04 已吸收 Product / Route Audit TODO、statin material-gate iteration、
P5 bridge evidence、enzyme SP-v1 gate、ChemEnzy module audit 等临时文档中的
路线真实性边界。旧文档归档后，本文件是新 SMILES 文献战略断键流程的执行入口。
```

## P0 开发定位

本流程是当前开发 P0 核心任务。

优先级声明：

```text
当前开工以本文件为 P0 执行入口。
交付 checklist 负责把本文件拆成工程任务。
Agentic CASP 主线计划和 feasibility audit 负责长期架构与风险边界。
如果 P0 范围出现冲突，以本文件的 SMILES-first 文献战略断键闭环为准。
```

P0 目标不是先完成完整 controller、blackboard、RouteStatus 状态机或搜索期 judge；
P0 目标是先把下面这条链路做成可运行、可校验、可复现的最小产品：

```text
target SMILES
  -> structure profile
  -> ordinary ChemEnzy/template baseline or manual frontier input
  -> advanced frontier detection
  -> Codex literature retrieval
  -> evidence cards
  -> strategic intermediate / disconnection candidates
  -> hybrid route package
  -> SMILES/rxn validation
  -> route map + summary
```

P0 成功标准：

- 给一个新 SMILES，能输出 `target_profile.json` 和 `frontier_report.json`。
- 能对 frontier 触发 Codex 文献检索，并输出可追溯 `evidence_cards.jsonl`。
- 能生成三类候选：`exact_fragment_retro`、`forward_surrogate`、`route_anchor`。
- 能输出 `*_hybrid_retrosynthesis_route.json`、`validation.json`、`summary.md`
  和路线图。
- 能明确标注 package 是 planning material；没有 stock/audit 证明时不能声称 solved。

### P0 交付清单

第一轮开发只交付下面这些内容；其它 controller / blackboard / guided rerun /
enzyme bridge / evolution 都不得抢占 P0 主线。

**P0a：SMILES profile + frontier extraction**

- `scripts/run_smiles_first_literature_workflow.py` CLI。
- `target_profile.json`：canonical SMILES、InChIKey、formula、heavy atoms、rings、
  stereocenters、ring systems、linker bonds、side chains、family hints。
- `baseline_routes.json`：普通 ChemEnzy/template baseline，或记录 baseline unavailable。
- `frontier_report.json`：高级 unresolved frontier、same-scaffold risk、
  ordinary-decoration-only、no-complexity-drop、unresolved-core flags。
- 支持 `--frontier-smiles` 手动输入，作为 ChemEnzy baseline 尚未接通时的 P0 fallback。

**P0b：Codex literature retrieval + evidence cards**

- `LiteratureSearchTask`：target profile、frontier SMILES、family hints、query budget、
  allowed source types、required output schema。
- `literature_search_report.md`：检索 query、命中文献、证据等级、限制。
- `evidence_cards.jsonl`：source metadata、URL/DOI/local ref、target relation、
  claim type、route role、confidence、limitations。
- 文献证据必须区分 `exact_target_or_intermediate`、`family_precedent`、
  `reaction_precedent`、`analogy_only`。
- 无可追溯证据时输出 `unresolved_literature_gap`，停止候选生成。

**P0c：advanced intermediate / strategic disconnection generation**

- `*_literature_rxn_candidates.jsonl`。
- 候选类型只允许：
  `exact_fragment_retro`、`forward_surrogate`、`route_anchor`。
- `exact_fragment_retro` 表达 target/frontier 上的拓扑精确断键，优先用 dummy atom
  或 labeled cut-site。
- `forward_surrogate` 必须包含 `not_lab_procedure=true`、`surrogate_reason`、
  `literature_basis`，并且只能作为 validated planning artifact。
- `route_anchor` 必须说明 multi-step / chiral-pool / semisynthesis / biosynthetic
  anchor，不允许伪装成单步 `rxn_smiles` 或 stock closure。
- `*_hybrid_retrosynthesis_route.json`：Target -> ordinary steps -> frontier ->
  strategic disconnection -> fragments / anchor。
- `validation.json`：SMILES/rxn parse、candidate kind、evidence refs、
  surrogate/anchor guardrail、route package status。
- `summary.md` 和路线图：必须显示普通步骤、失败 frontier、文献接管点和上游 anchor。

**P0 guardrail：minimal route package audit**

- 没有 stock/audit 证明时，P0 输出只能是 `planning_material`、
  `partial_anchor`、`literature_gap` 或 `invalid_package`。
- `route_anchor` 不能被计为 stock solved。
- `forward_surrogate` 不能被计为 exact literature reaction 或实验方案。
- analogy-only evidence 只能支持 critique / weak prior，不能支持 solved route。

### P0 暂不开发

以下能力属于 P1 或更后，不作为当前核心交付：

- 完整 Case Blackboard。
- 完整 Codex Chief Chemist Controller。
- 完整 RouteStatus 状态机。
- ChemEnzy guided rerun / StrategicOperator compiler。
- compiled search-time judge。
- enzyme-aware bridge runtime integration。
- evidence-gated evolution / production KB promotion。

## 适用场景

适合使用本流程的输入：

- 用户给出一个新 target SMILES，并要求逆合成、关键步骤或路线图。
- 普通 ChemEnzy/template 搜索能处理末端官能团修饰，但在复杂天然产物核心、高级
  中间体、手性骨架或大环/多环体系上失败。
- 文献中存在明确的战略构建逻辑，例如 C-C 偶联、糖苷键/苷元断键、steroid
  chiral-pool anchor、macrolactonization、Pictet-Spengler、Corey-lactone 等。

不适合直接自动执行的情况：

- 用户要求实验可执行工艺条件、放大工艺或安全评估，但没有文献实验细节。
- 目标处于受控、危险、双用途或强监管类别，需要先走合规/安全审查。
- 文献只能支持家族层面类比，不能支持具体结构实例化；这时只能输出 critique 或
  hypothesis，不能输出已解决路线。

## 核心原则

1. **SMILES 先行**：先从输入 SMILES 识别 scaffold、环系、非环连接键、官能团和
   明显的可断键，不假设本地有专用模板。
2. **普通步骤先交给 ChemEnzy**：末端酰化、保护/脱保护、氧化还原、小片段替换等
   可以由普通规划器或通用 reaction record 表达。
3. **失败点才调用文献战略断键**：当普通搜索到达高级天然产物核心却无法继续合理
   简化时，把该 frontier 交给文献检索和战略断键模块。
4. **三类输出必须分开**：
   - `exact_fragment_retro`：从 target/frontier SMILES 直接断出来的拓扑精确
     dummy-fragment retrosynthesis。
   - `forward_surrogate`：根据文献反应类型构造的可解析 forward `rxn_smiles`，
     用于模型/规划器，但不冒充 SI 中的精确底物。
   - `route_anchor`：文献高级中间体或 chiral-pool anchor；不能当作单步反应，也
     不能自动视作 stock。
5. **不要把高级类似物当 stock**：高级天然产物类似物只说明半合成可行，不说明
   骨架构建已被解决。
6. **RouteStatus 后置判定**：路线图、文献候选、surrogate 和 anchor 都只是
   planning material。最终是否 solved 必须由 product / route audit 输出
   `RouteStatus`，不能由 agent、planner score、EC annotation 或 stock hit 直接宣布。

## 已整合的真实性门槛

从临时审计文档吸收的硬规则：

- `advanced_same_scaffold_fake_close`：高级同骨架 terminal 不能默认当 stock。
- `route_anchor`：可以作为文献锚点或上游假设，不能作为单步反应闭合。
- `forward_surrogate`：必须标注 `not_lab_procedure=true` 和 `surrogate_reason`；
  只能进入 validated artifact / policy compiler，不能直接注入 ChemEnzy route tree。
- `enzyme step`：EC annotation、post-hoc enzyme classifier、generic EC、active-site
  annotation 都不是单独的 enzyme validation；必须记录 substrate/product/EC、
  precedent、verifier score 和 artifact gate。
- `condition`：condition prediction 是 feasibility hint，不是可执行工艺；所有
  condition-known / analog / model-predicted / unknown 来源必须分开。
- `material gate`：明显 no-complexity-drop、product-like terminal、无解释大骨架增长
  必须生成 FailureEvent 或 unresolved；transfer reagent / protecting group / carrier
  reagent 只能在有 atom-contribution 解释时降为 warning。

## Agent 流程

### 1. 输入规范化

输入最少包含：

```json
{
  "target_name": "optional",
  "target_smiles": "...",
  "objective": "route | key_step | critique | route_figure"
}
```

agent 应先用 RDKit 或等价工具校验：

- SMILES 是否可解析；
- canonical/isomeric SMILES；
- 分子式、重原子数、环数、手性中心；
- 非环连接两个环系统的键；
- 明显侧链、糖基、内酯/内酰胺、大环、芳杂环、steroid/terpene/alkaloid 等家族信号。

### 2. 普通规划器阶段

调用 ChemEnzy 或通用模板时，记录：

- 搜索配置和 stock policy；
- top routes；
- 每步 `rxn_smiles`、条件预测、stock 状态；
- 未解决 frontier；
- gate/failure reasons。

如果普通规划只得到末端修饰路线，例如 Bufotalin 的 O-acetylation，则不能立即宣称
完成全流程。应检查剩余高级核心是否仍然 product-like。

### 3. 失败 frontier 判断

进入文献战略断键的触发条件：

- frontier 与 target 共享大部分多环/天然产物核心；
- 普通搜索继续拆解时出现 large unexplained heavy-atom/carbon gain；
- route 只在保护基/酰基/氧化态上循环，没有减少核心复杂度；
- terminal 是高级天然产物类似物、供应商类似物或同 scaffold analog；
- 用户明确要求“关键步骤”“文献战略断键”“高级中间体拆解”。

Bufotalin 试跑中的典型 frontier 是 Deacetylbufotalin：普通步骤能解释 O-acetylation，
但无法解释 bufadienolide steroid core 的战略构建。

### 4. 文献提取阶段

文献检索应优先回答四个问题：

- 该目标属于哪个可识别天然产物/药物 scaffold family？
- 文献中的 key bond-forming step 是什么？
- 文献使用哪些高级中间体或 chiral-pool anchors？
- 该策略能否实例化成 target/frontier SMILES 上的具体断键？

提取时必须区分：

- `exact evidence`：文献直接给出目标或同一中间体；
- `family precedent`：同 scaffold 或同天然产物家族；
- `reaction precedent`：同反应类型，但底物不同；
- `analogy only`：只能作为 ranking prior，不能作为 route solution。

### 5. 结构实例化

对文献策略生成三层材料：

```json
{
  "candidate_id": "...",
  "direction": "retro | forward | route_anchor",
  "target_smiles": "...",
  "product_smiles": "...",
  "precursor_smiles": ["..."],
  "rxn_smiles": "...",
  "reaction_class": "...",
  "strategic_bond": "...",
  "literature_basis": "...",
  "use_case": "...",
  "confidence": "...",
  "references": ["..."]
}
```

要求：

- `exact_fragment_retro` 优先用 dummy atom 标记断键位点；
- `forward_surrogate` 必须能被 RDKit 解析，但要标记为 surrogate；
- `route_anchor` 的 `rxn_smiles` 可以为空，必须说明它是多步 anchor；
- 所有记录都要有 `confidence` 和 `use_case`；
- 不允许把文献多步 anchor 压缩成一个虚假的单步 `rxn_smiles`。

### 6. 路线拼接

推荐路线图层级：

```text
Target
<= ordinary ChemEnzy-compatible steps
Frontier advanced intermediate
<= literature strategic disconnection
Strategic fragments / activated partner
<= route anchor
Reviewed chiral-pool or biosynthetic starting material
```

Bufotalin 对应：

```text
Bufotalin
<= O-acetylation
Deacetylbufotalin
<= C17-2-pyrone strategic disconnection
C17-functional steroid + 2-pyrone partner
<= steroid chiral-pool anchor
androstenedione-like scaffold
```

### 7. 校验门槛

交付前至少检查：

- JSON/JSONL 可解析；
- 所有非空 SMILES 可解析；
- 所有非空 `rxn_smiles` 只有一个 `>>`，左右各片段可解析；
- dummy-fragment retro 与 target/frontier 的断键一致；
- forward surrogate 标注清楚，不写成“实验已验证”；
- anchor 不进入 ordinary stock solve；
- 路线图能显示普通步骤、失败 frontier、文献接管点和上游 anchor。

## Bufotalin 经验

本次模拟运行暴露了几个可复用问题：

- **低战略路线会误导结论**：只做 Deacetylbufotalin 到 Bufotalin 的 O-acetylation
  是合理普通步骤，但不是 Bufotalin 的关键骨架逆合成。
- **文献关键步骤应接管高级 frontier**：对 bufadienolide，关键不是拆 steroid
  tetracycle，而是识别 C17-2-pyrone installation。
- **surrogate 必须降级标注**：C17-bromosteroid + stannyl/Bpin pyrone 的
  `rxn_smiles` 可以用于规划，但不是逐字 SI 底物。
- **anchor 不能伪装成单步**：androstenedione 到 oxygenated C17-functional steroid
  是文献多步上游，不应写成一个单步反应。
- **图和 JSON 要同步**：用户追问“step 几”时，必须能从路线图对应到 route JSON 的
  `index`、`reaction_type` 和 `step_role`。

## 推荐交付物

每次新 SMILES 跑完整流程，建议输出：

- `*_literature_rxn_candidates.jsonl`：候选关键断键/反应记录；
- `*_hybrid_retrosynthesis_route.json`：拼接后的路线；
- `summary.md`：人读结论、关键 `rxn_smiles`、限制；
- `validation.json`：SMILES/rxn 校验结果；
- `figures/*_retrosynthesis_map.svg`：流程图，显示失败 frontier 和文献接管；
- `figures/scheme_route_01.svg`：线性路线 scheme。

Bufotalin 示例包：

```text
results/shared/bufotalin_hybrid_literature_20260603/
```

## 下一步优化方向

- 把文献候选生成器抽象成通用 CLI：输入 target SMILES、frontier SMILES、family hint，
  输出 `rxn_candidates.jsonl`。
- 为 `exact_fragment_retro` 增加 atom-map 和断键坐标，便于 UI 高亮。
- 为 `forward_surrogate` 增加 `surrogate_reason` 和 `not_lab_procedure` 强制字段。
- 在 material gate 中加入 “advanced analogue is not stock” 的硬规则。
- 建立 route package audit：检查 route JSON 中每个 `route_anchor` 是否被误计为
  solved stock。
