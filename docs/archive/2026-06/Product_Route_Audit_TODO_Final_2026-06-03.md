# Product / Route Audit TODO 最终版

日期：2026-06-03

用途：给后续 Codex / agent / 人工审计者直接照着执行，对 ChemEnzy 输出路线做产物与路线真实性审计。每一步都必须打勾、记录证据、写明结论。未完成的步骤不能跳过。

核心原则：

```text
宁可 unresolved，也不能 fake solved。
planner score 不是路线真实性。
stock closure 不是路线成立。
EC annotation 不是 enzyme validation。
文献摘要不是 validated evidence。
```

---

## 0. 审计输入准备

- [ ] 确认目标产物输入形式。
  - 记录：target name / SMILES / InChIKey / source。
  - 检查：是否有盐型、互变异构、立体化学、省略手性或名称歧义。
  - 产出：`TargetResolution`。

- [ ] 标准化目标结构。
  - 记录：canonical SMILES、InChIKey、stereo status。
  - 检查：RDKit 是否可解析，目标是否与用户意图一致。
  - 失败处理：目标不明确时停止路线审计，标记 `target_ambiguous`。

- [ ] 收集 ChemEnzy 输出。
  - 记录：run id、config、search flags、stock source、one-step sources、plugin config。
  - 记录：route candidates、route steps、terminal molecules、scores、trace。
  - 检查：是否包含 raw route、normalized route、step metadata、proposal source。

- [ ] 建立审计记录。
  - 记录：audit id、auditor、date、input run id、target id。
  - 产出：`ProductRouteAuditCase`。

---

## 1. 目标与路线结构一致性

- [ ] 检查 route product 是否等于目标产物。
  - 对每条 route 的 root product 做 canonical comparison。
  - 检查 stereochemistry 是否一致。
  - 结论：`target_match = exact | stereo_mismatch | tautomer_or_salt_variant | mismatch`。

- [ ] 检查每一步 reaction product 是否对应父节点。
  - 每个 step 必须满足 `reactants >> expected_product`。
  - 记录 product mismatch、missing product、multi-product ambiguity。

- [ ] 检查反应方向。
  - 确认 route step 是 retrosynthetic disconnection，不是 forward direction 被误读。
  - 标记 direction suspicious 的 step。

- [ ] 检查重复节点和循环。
  - 检查 same molecule 是否在同一路径中重复出现。
  - 检查 same-scaffold loop。
  - 发现循环时标记 `same_scaffold_loop` 或 `exact_cycle`。

---

## 2. Step-level 结构合法性

对每一个 route step 执行：

- [ ] 检查 SMILES 可解析。
  - product、main reactant、aux reactants 都必须 RDKit 可解析。
  - 不可解析时 step 标记 `invalid_smiles`。

- [ ] 检查反应式完整性。
  - 有 product。
  - 有至少一个 reactant。
  - reactant/product 不为空。
  - aux reactants 不应包含目标产物本身。

- [ ] 检查物料变化。
  - 记录 heavy atom delta、carbon delta、hetero atom delta、ring delta。
  - 标记异常大增益、大丢失、明显物料不守恒。

- [ ] 检查复杂度下降。
  - 逆合成 step 的 reactants 应相对 product 有合理复杂度下降或明确 strategic anchor 解释。
  - 无下降时标记 `no_complexity_drop`。

- [ ] 检查 product-like reactant。
  - reactant 与 target / parent product 过高相似时标记。
  - 需要继续判断是否是 validated semisynthesis anchor。

- [ ] 检查明显伪反应。
  - 单原子或无意义片段解释复杂产物。
  - 大骨架凭空出现。
  - 关键环系无来源。
  - stereocenter 无解释。

---

## 3. Terminal / Stock 审计

对每一个 terminal molecule 执行：

- [ ] 检查是否在 stock 中。
  - 记录 stock source、vendor/source tier、lookup key。
  - 区分 `commercial_stock`、`internal_stock`、`database_seen`、`unknown`。

- [ ] 检查 stock 可信度。
  - commercial building block：可作为 total synthesis terminal。
  - known metabolite / natural product / advanced analog：不能默认作为 total synthesis terminal。
  - database-only hit：不能默认等于可购买。

- [ ] 检查高级同骨架假闭合。
  - terminal 与 target 相似度高。
  - terminal 保留目标核心骨架。
  - terminal 复杂度接近目标。
  - terminal 缺少直接 anchor evidence。
  - 满足时标记 `advanced_same_scaffold_fake_close`。

- [ ] 检查 no-progress terminal。
  - terminal 与 parent product 基本一致。
  - terminal 只是脱保护、盐型、互变异构或轻微官能团差异。
  - 标记 `no_progress_terminal`。

- [ ] 判断 terminal route role。
  - `simple_commercial_building_block`
  - `validated_semisynthesis_anchor`
  - `isolation_or_fermentation_anchor`
  - `advanced_np_like_unvalidated_terminal`
  - `unknown_or_unavailable`

- [ ] 生成 stock audit 结论。
  - 允许 total synthesis closure。
  - 允许 semisynthesis closure。
  - 允许 partial anchor。
  - 拒绝 fake closure。
  - 未知则 unresolved。

---

## 4. Route Mode 审计

- [ ] 判断路线类型。
  - total synthesis
  - semisynthesis
  - isolation / fermentation anchor
  - enzyme-assisted synthesis
  - mixed route
  - partial anchor
  - unresolved

- [ ] 检查路线声明是否过度。
  - 含高级 anchor 的路线不能报告为普通 total synthesis solved。
  - 只闭合到 analog / precursor 的路线不能报告为 exact solved。
  - 只有 isolation evidence 的路线不能报告为 synthetic route solved。

- [ ] 检查关键断点是否属于正确 route mode。
  - 简单 building block disconnection：chemical synthesis。
  - tailoring / late-stage modification：可能 enzyme-assisted。
  - 核心天然产物骨架：通常需要文献 anchor 或明确 total synthesis evidence。

- [ ] 给出 route mode 结论。
  - 记录：`route_mode`、理由、关键 terminal、关键 evidence。

---

## 5. Enzyme Step 审计

对每个 enzyme-like 或 EC-annotated step 执行：

- [ ] 区分 proposal source 和 post-hoc annotation。
  - 记录：step 是否来自 enzyme proposal source。
  - 记录：step 是否只是 post-hoc classified enzymatic。
  - EC annotation 不自动通过。

- [ ] 检查 enzyme reaction plausibility。
  - 反应中心是否合理。
  - 反应类型是否与 EC class 匹配。
  - 底物/产物变化是否符合已知 enzyme chemistry。
  - 是否存在不合理 core disconnection。

- [ ] 检查 cofactor / cosubstrate。
  - 记录 NAD(P)H / NAD(P)+ / PLP / SAM / FAD / O2 / H2O2 等需求。
  - 检查 route-level cofactor debt。
  - 缺失时标记 `cofactor_unknown`。

- [ ] 检查 enzyme precedent。
  - exact precedent。
  - close substrate analog precedent。
  - EC-only weak support。
  - no precedent。

- [ ] 检查 enzyme material sanity。
  - 大骨架构建是否凭空发生。
  - 碳数/杂原子数变化是否合理。
  - 环系变化是否符合 enzyme class。

- [ ] 给出 enzyme step status。
  - `validated_enzyme_step`
  - `plausible_enzyme_hypothesis`
  - `weak_ec_annotation_only`
  - `unsupported_enzyme_step`
  - `reject_enzyme_artifact`

---

## 6. 文献 / Evidence 审计

仅在复杂目标、假闭合、stuck node、semisynthesis anchor、enzyme step 或 condition gap 时执行。

- [ ] 明确检索问题。
  - target exact synthesis。
  - target semisynthesis。
  - target isolation / fermentation。
  - stuck node transformation。
  - terminal anchor evidence。
  - enzyme precedent。
  - condition precedent。

- [ ] 收集证据来源。
  - paper。
  - patent。
  - database。
  - web source。
  - internal route document。

- [ ] 区分证据关系。
  - exact target。
  - analog。
  - precursor。
  - same scaffold。
  - unrelated。

- [ ] 区分证据类型。
  - total synthesis。
  - semisynthesis。
  - isolation。
  - fermentation。
  - analog route。
  - single-step condition。
  - failed attempt。

- [ ] 验证 evidence 可用性。
  - 结构一致。
  - reaction role assignment 合理。
  - atom mapping / reaction center 可检查。
  - 条件信息可归一化。
  - 证据可链接到 route step、terminal 或 stuck node。

- [ ] 生成 EvidenceCard。
  - 未验证证据只能作为 research note。
  - 通过验证的 evidence 才能影响 anchor whitelist、terminal blacklist、StrategicOperator 或 route audit。

---

## 7. Condition / Feasibility 审计

对每个 route step 执行：

- [ ] 检查是否有条件来源。
  - exact literature condition。
  - analog literature condition。
  - template / cluster condition。
  - condition model prediction。
  - condition_unknown。

- [ ] 记录条件要素。
  - reagent。
  - catalyst / enzyme。
  - solvent。
  - temperature。
  - atmosphere。
  - pH / buffer。
  - time。
  - yield or reported outcome。

- [ ] 检查官能团兼容性。
  - 酸/碱敏感。
  - 氧化/还原敏感。
  - 水/有机溶剂兼容性。
  - 保护基需求。
  - stereochemistry risk。

- [ ] 检查 route-level feasibility。
  - 相邻步骤条件冲突。
  - enzyme / chemical step 切换风险。
  - isolation / purification 是否必要。
  - cofactor regeneration 是否缺失。

- [ ] 给出 condition status。
  - `condition_supported`
  - `condition_analog_supported`
  - `condition_model_only`
  - `condition_gap`
  - `condition_risky`

---

## 8. Failure Event 生成

如果路线不能直接通过，必须生成 FailureEvent。

- [ ] 判断失败类型。
  - `target_mismatch`
  - `invalid_step`
  - `no_closure`
  - `fake_closure`
  - `advanced_same_scaffold_fake_close`
  - `no_complexity_drop`
  - `stock_gap`
  - `evidence_gap`
  - `condition_gap`
  - `unsupported_enzyme_break`
  - `same_scaffold_loop`

- [ ] 定位失败节点。
  - route id。
  - step id。
  - node id。
  - molecule SMILES。
  - terminal molecule。
  - failed disconnection。

- [ ] 写明失败原因。
  - 不允许只写 “bad route”。
  - 必须说明是哪条规则、哪个结构、哪个 terminal 或哪个 evidence gap 导致失败。

- [ ] 给出建议下一步。
  - `RESEARCH_STUCK_NODE`
  - `RESEARCH_TARGET`
  - `COMPILE_STRATEGIC_OPERATOR`
  - `RUN_GUIDED_CHEMENZY`
  - `DESIGN_CONDITIONS`
  - `FINAL_UNRESOLVED`

---

## 9. Strategic Intervention 审计

只有在有明确 FailureEvent 或 validated evidence 时执行。

- [ ] 判断是否需要 rerun。
  - 有新 evidence。
  - 有明确 terminal blacklist。
  - 有 validated anchor。
  - 有明确 source policy 修改。
  - 有明确 stuck node。

- [ ] 生成或更新 StrategicOperator。
  - anchor whitelist。
  - terminal blacklist。
  - route mode prior。
  - source promotion / demotion。
  - strict stock closure。
  - bounded rerun budget。

- [ ] 检查 rerun reason。
  - 每次 rerun 必须有明确原因。
  - 不能为了“再试试”无依据重跑。

- [ ] 检查预算。
  - max reruns。
  - max literature rounds。
  - max stuck nodes。
  - max wall time。
  - max tool calls。

- [ ] 编译 ChemEnzySearchPolicy。
  - LLM / Codex 输出不能直接作为 ChemEnzy 参数。
  - 必须通过 deterministic adapter。

---

## 10. Final RouteStatus 判定

每条 route 必须给出最终状态。

- [ ] `solved`
  - target exact match。
  - 所有 steps 结构合法。
  - terminals 是可信 simple stock。
  - 无 fake closure。
  - 关键步骤有足够条件 / 可行性支持。

- [ ] `semisynthesis_closed`
  - target exact match。
  - 终点闭合到 validated semisynthesis anchor。
  - anchor 有直接证据。
  - 报告中明确不是普通 total synthesis closure。

- [ ] `partial_anchor`
  - 找到合理 anchor 或 stuck node。
  - 未完整闭合到 simple stock。
  - 证据不足以声明 solved。

- [ ] `fake_closed_rejected`
  - planner 表面闭合。
  - terminal / stock / no-complexity / same-scaffold audit 失败。
  - 明确记录被拒绝的 terminal 或 step。

- [ ] `unresolved`
  - 无可信闭合路线。
  - 证据不足。
  - 预算耗尽。
  - 无明确下一步干预假设。

---

## 11. Final Audit Report 必填项

- [ ] target summary。
- [ ] route candidate summary。
- [ ] top route RouteStatus。
- [ ] stock audit table。
- [ ] step-level structural audit table。
- [ ] route mode audit。
- [ ] enzyme step audit。
- [ ] evidence audit。
- [ ] condition audit。
- [ ] fake closure / rejected terminal list。
- [ ] FailureEvent list。
- [ ] StrategicOperator / rerun history。
- [ ] unresolved core or stuck node。
- [ ] next recommended action。

---

## 12. 审计停止条件

- [ ] 目标结构不明确：停止，输出 `target_ambiguous`。
- [ ] 所有 top routes 均 fake closure：停止或触发 bounded rerun。
- [ ] 无新 evidence 且 rerun 无改善：输出 `unresolved`。
- [ ] 预算耗尽：输出 `unresolved`，附失败事件和已查证据。
- [ ] route 达到 solved / semisynthesis_closed 标准：输出 final report。

---

## 13. 后续模型执行提示

执行时按顺序完成：

```text
0. 审计输入准备
1. 目标与路线结构一致性
2. Step-level 结构合法性
3. Terminal / Stock 审计
4. Route Mode 审计
5. Enzyme Step 审计
6. 文献 / Evidence 审计
7. Condition / Feasibility 审计
8. Failure Event 生成
9. Strategic Intervention 审计
10. Final RouteStatus 判定
11. Final Audit Report
12. 审计停止条件
```

每一步必须记录：

```text
checked: yes / no
evidence: file / URL / route id / step id / molecule id
decision: accept / reject / defer / unresolved
reason: short explicit reason
next_action: action enum or none
```

最终要求：

```text
没有 RouteStatus 的路线报告不合格。
没有 stock audit 的 solved 不合格。
没有 fake closure 检查的 closure 不合格。
没有 evidence audit 的 semisynthesis anchor 不合格。
没有 condition status 的 route feasibility 不完整。
```
