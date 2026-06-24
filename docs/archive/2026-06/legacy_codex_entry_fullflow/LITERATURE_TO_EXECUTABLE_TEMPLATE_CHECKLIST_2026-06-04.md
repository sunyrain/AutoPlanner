# Literature-to-Executable Template Checklist

日期：2026-06-04

用途：本 checklist 定义下一阶段开发主线：把文献调研结果从 `advisory_strategy_template`
推进到 ChemEnzy 可消费的 one-step executable template / proposal source。当前 P0/P1b
已经证明能生成 evidence cards、strategic cards、policy 和 guided rerun trace；但这还不够。
下一阶段必须证明文献内容能进入 ChemEnzy 的单步扩展层，而不是只作为报告、审计或
source-budget hint。

## 总体判断

当前能力边界：

```text
已完成：文献 -> evidence card -> strategic disconnection card -> advisory strategy template -> ChemEnzySearchPolicy
不足：文献 -> 可执行单步模板 -> ChemEnzy one_step proposal -> MCTS route tree
```

下一阶段目标：

```text
当 ChemEnzy native route fail、未闭合、或审计失败时，
系统能把文献战略断键实例化为当前 target/frontier 上的可执行 one-step proposal，
并作为受控 external one-step source 进入 ChemEnzy 搜索。
```

硬边界：

```text
No raw unvalidated LLM reaction injection.
No solved claim from literature template alone.
No direct production KB write from a target run.
No template promotion without applicability + reconstruction + audit.
Literature route may enter ChemEnzy only through deterministic validated artifacts.
```

---

## Phase L0：触发条件收紧

目标：文献检索和模板实例化只在原生 ChemEnzy 不够可信时触发。

- [x] 定义 `LiteratureTriggerReason` 枚举。
  - `native_failed`
  - `unclosed_route`
  - `fake_closure_risk`
  - `advanced_frontier_detected`
  - `route_audit_failed`
  - `user_requested_literature`

- [x] 在 native ChemEnzy run 后执行 route audit。
  - 检查 `route_count == 0`。
  - 检查 leaf stock closure。
  - 检查 same-scaffold fake closure。
  - 检查 no-complexity-drop terminal。
  - 检查 advanced natural-product frontier。

- [x] 只有存在明确 trigger reason 时生成 `LiteratureSearchTask`。

- [x] 对 phenolic glycoside 这类 native already solved 案例，默认不触发文献模板 rerun。
  - 可以保留为 smoke test。
  - 不能作为 literature enhancement gain 证据。

验收：

- [x] native solved + audit passed 时不进入 literature mode。
- [x] native failed 时进入 literature mode。
- [x] native partial / unclosed 时进入 literature mode。
- [x] native fake closure 时进入 literature mode。

---

## Phase L1：文献模板 schema 升级

目标：区分“建议模板”和“可执行模板”，避免把 family-level 文献直接当反应。

- [x] 新增 `LiteratureTemplateCard` schema。
  - `template_id`
  - `evidence_refs`
  - `reaction_class`
  - `template_level`
  - `product_retron`
  - `break_bonds`
  - `precursor_roles`
  - `applicability`
  - `scope_limits`
  - `safety_flags`
  - `promotion_status`

- [x] `template_level` 只允许：
  - `advisory_strategy`
  - `retron_pattern`
  - `executable_template_candidate`
  - `validated_executable_template`
  - `route_anchor_only`

- [x] 新增 `TemplateApplicabilityReport` schema。
  - `target_smiles`
  - `frontier_smiles`
  - `matched_retron_atoms`
  - `matched_bonds`
  - `match_confidence`
  - `mismatch_reasons`
  - `allowed_use`

- [x] 新增 `ExecutableTemplateCandidate` schema。
  - `product_smiles`
  - `reactant_smiles`
  - `rxn_smiles`
  - `atom_mapping_status`
  - `template_smarts`
  - `source_template_id`
  - `not_lab_procedure`
  - `proposal_source`

- [x] 保留当前 `advisory_strategy_template.v1`，但明确它不能直接进入 one-step proposal。

验收：

- [x] advisory template 不能被 ChemEnzy direct consume。
- [x] executable template candidate 必须有 product-specific applicability report。
- [x] route anchor 不能伪装成 executable template。

---

## Phase L2：retron / applicability gate

目标：只有当前 target/frontier 结构真的匹配文献断键时，才允许实例化模板。

- [x] 为每个文献策略生成 product-side retron pattern。
  - glycoside：anomeric C-O / C-N / C-S linkage。
  - C-glycoside：anomeric C-C aryl linkage。
  - macrolactone：macrocyclic ester C-O bond。
  - taxane：C13 side-chain ester/carbamate boundary。
  - bufadienolide：steroid C17 to 2-pyrone C-C bond。
  - Corey lactone：side-chain installation boundary。

- [x] 实现 RDKit retron matcher。
  - 输入：frontier/product SMILES + retron pattern。
  - 输出：候选断键 atom indices。
  - 支持多 match，必须排序和记录 ambiguity。

- [x] 实现 mismatch 降级规则。
  - exact retron match：允许进入 executable candidate。
  - same family but wrong linkage：只能 advisory/rerank。
  - analogy only：只能 critique 或继续检索。
  - no retron match：禁止实例化。

- [x] 实现 product-specific cut。
  - 在当前 frontier 上切断匹配 bond。
  - 生成 dummy-labeled fragments。
  - 记录 atom indices 和 bond type。

验收：

- [x] phenolic O-glycoside 匹配 O-glycosidic retron。
- [x] O-glycoside 不误匹配 C-glycoside executable template。
- [x] paclitaxel/taxane 只匹配 taxane side-chain boundary，不误匹配 macrocycle。
- [x] bufadienolide C17-pyrone match 能定位 steroid-pyrone C-C boundary。

---

## Phase L3：模板实例化与反向重构

目标：把文献断键从“文字策略”变成当前 product 的可执行 one-step candidate。

- [x] 实现 `instantiate_literature_template(product_smiles, template_card)`。
  - 使用 applicability report 的 matched bond。
  - 生成 reactant fragments。
  - 标注 precursor roles。
  - 输出 `ExecutableTemplateCandidate`。

- [x] 实现 fragment role assignment。
  - aglycone acceptor。
  - sugar donor / sugar precursor。
  - steroid core。
  - pyrone coupling partner。
  - seco acid。
  - side-chain fragment。

- [x] 实现 forward reconstruction audit。
  - fragments recombination 后必须能回到 product connectivity。
  - heavy atom accounting 必须守恒或有明确 leaving-group / reagent explanation。
  - dummy atom attachment 必须可解释。

- [x] 实现 basic chemical sanity gate。
  - SMILES 可解析。
  - valence 合法。
  - reactant 不等于 product。
  - complexity 有合理下降。
  - 不允许大骨架无解释增长。

- [x] 给每个 candidate 生成 `TemplateValidationReport`。
  - `accepted`
  - `reasons`
  - `confidence`
  - `allowed_for_one_step_source`

验收：

- [x] glycoside 可生成 aglycone + sugar-side fragment candidate。
- [x] bufadienolide 可生成 steroid fragment + pyrone fragment candidate。
- [x] taxane 可生成 taxane core + side-chain fragment candidate。
- [x] reconstruction 不通过时 candidate 被拒绝。

---

## Phase L4：ChemEnzy direct consumption bridge

目标：让文献模板作为 ChemEnzy 的 external one-step proposal source，而不是只进入 policy。

- [x] 新增 `LiteratureOneStepPlugin`。
  - 接口兼容 ChemEnzy `one_step.run(product_smiles)`。
  - 输入当前 product。
  - 查询已验证 executable templates。
  - 返回 product-specific reactant proposals。

- [x] 输出格式对齐 ChemEnzy multi one-step wrapper。
  - `reactants`
  - `scores`
  - `templates`
  - `costs`
  - `source_policy_decision`
  - `literature_template_trace`

- [x] plugin proposal 必须带来源标记。
  - `source_model = literature_template_plugin`
  - `evidence_refs`
  - `template_id`
  - `not_lab_procedure`
  - `requires_audit = true`

- [x] 接入 `NativeChemicalOneStepWrapper` 或新增 external source wrapper。
  - 不破坏 ChemEnzy native model。
  - 可通过 search flag 开关启用。
  - 可以与 native one-step sources 并行。

- [x] cascade source policy 支持 `literature_template_plugin` domain。
  - domain 可标记为 `literature_chemical` / `literature_biocatalytic`。
  - native failed 时提高 topk。
  - audit passed native solved 时不启用。

验收：

- [x] 文献 plugin 能在单步扩展时返回 candidate。
- [x] ChemEnzy MCTS route tree 中能看到 `literature_template_plugin` step。
- [x] route output 明确区分 native proposal 和 literature plugin proposal。
- [x] 禁用 plugin 后结果回到 native baseline。

---

## Phase L5：路线级拼接与递归 anchor expansion

目标：文献模板不仅能修一个 stuck node，还能递归处理高级 anchor。

- [x] 定义 `RouteAnchorExpansionTask`。
  - anchor SMILES / name。
  - source evidence。
  - required closure type。
  - parent route reference。

- [x] 对 route anchor 生成 child target。
  - baccatin / 10-DAB。
  - Corey lactone。
  - androstenedione-like steroid core。
  - seco acid / macrolactone precursor。

- [x] child target 先跑 native ChemEnzy。

- [x] child native fail 或 unclosed 时，再触发文献模板流程。

- [x] 拼接 parent route + child route。

- [x] RouteStatus 只允许在全部 leaf audit 后升级。
  - `solved`
  - `semisynthesis_closed`
  - `partial_anchor`
  - `unresolved`

验收：

- [x] taxane 不再只停在 parent semisynthesis anchor，而能生成 baccatin/10-DAB child task。
- [x] bufadienolide steroid anchor 能作为 child expansion task。
- [x] 未闭合 anchor 不允许 claim solved。

---

## Phase L6：A/B benchmark

目标：证明文献模板 direct consumption 真的提升 ChemEnzy，而不是只生成漂亮报告。

- [x] 建立 native vs literature-template-plugin benchmark。

- [x] 每个 case 运行三种配置。
  - native ChemEnzy。
  - policy-only guided ChemEnzy。
  - executable literature template plugin ChemEnzy。

- [x] 指标。
  - solved rate。
  - route_count。
  - stock closure rate。
  - fake closure rejection rate。
  - route audit pass rate。
  - literature plugin step precision。
  - reconstruction pass rate。

- [x] 必选案例。
  - bufotalin / bufadienolide C17-pyrone。
  - taxane semisynthesis。
  - artemisinin peroxide anchor。
  - macrolactonization。
  - Corey lactone prostaglandin。
  - phenolic glycoside negative-control / already-native-solved。

- [x] 明确 negative controls。
  - 文献模板不应伤害 native already solved case。
  - 不匹配 retron 不应生成 executable candidate。
  - analogy-only evidence 不应进入 plugin。

验收：

- [x] 至少 2 个 native failed / unclosed case 通过 literature template plugin 改善。
- [x] phenolic glycoside 不被误报为 literature gain。
- [x] 所有 plugin steps 有 evidence refs 和 template validation report。

---

## Phase L7：安全与审计边界

目标：模板可执行不等于路线可信，必须保留审计边界。

- [x] executable template 只能提升 proposal recall，不能直接决定 solved。

- [x] 每个 literature plugin step 必须进入 product/route audit。

- [x] 条件预测必须标注来源。
  - literature known。
  - literature analog。
  - model predicted。
  - unknown。

- [x] 对危险/受控/双用途类别保留合规 gate。

- [x] 对 forward surrogate 保留降级标签。
  - 如果只是 surrogate，不得 promoted 为 validated executable template。

- [x] 对生产 KB promotion 单独设 gate。
  - 多案例复现。
  - negative-control 通过。
  - source evidence stable。
  - no target-run direct write。

验收：

- [x] plugin step 不绕过 route audit。
- [x] template validation report 可追溯。
- [x] fake closure 仍能被 terminal judge 拒绝。

---

## 新增代码范围

建议新增模块：

- [x] `cascade_planner/agent/literature_templates.py`
- [x] `cascade_planner/agent/template_applicability.py`
- [x] `cascade_planner/agent/executable_template_validation.py`
- [x] `cascade_planner/baselines/literature_one_step_plugin.py`
- [x] Historical benchmark runner archived locally under
  `archive/harness_prep_20260605/scripts/run_literature_template_plugin_benchmark.py`.

建议修改模块：

- [x] `cascade_planner/agent/strategic_candidate_generation.py`
- [x] `cascade_planner/agent/chem_enzy_policy.py`
- [x] `cascade_planner/baselines/chem_enzy_adapter.py`
- [x] `vendor/ChemEnzyRetroPlanner/retro_planner/common/prepare_utils.py` 或本地 wrapper 接入层
- [x] `vendor/ChemEnzyRetroPlanner/retro_planner/search_frame/mcts_star/cascade_source_policy.py`

建议新增测试：

- [x] `tests/test_literature_template_cards.py`
- [x] `tests/test_template_applicability.py`
- [x] `tests/test_executable_template_validation.py`
- [x] `tests/test_literature_one_step_plugin.py`
- [x] Historical benchmark test archived locally under
  `archive/harness_prep_20260605/tests/test_literature_template_plugin_benchmark.py`.

---

## 最小可交付版本

MVP 只做三类模板：

- [x] O-glycoside glycosidic-bond split。
- [x] bufadienolide C17-pyrone split。
- [x] taxane C13 side-chain split。

MVP 必须证明：

- [x] 文献模板能被实例化为当前 product 的 one-step candidate。
- [x] candidate 能作为 plugin proposal 被 ChemEnzy MCTS 看见。
- [x] candidate 不是 raw LLM injection，而是 deterministic validated artifact。
- [x] native / policy-only / plugin 三者 A/B 结果可比较。

MVP 不要求：

- [x] 覆盖全部天然产物家族。
- [x] 自动读 SI 全实验路线。
- [x] 自动宣布 total synthesis solved。
- [x] 把所有文献写入生产 KB。

---

## 完成证据（2026-06-04）

实现证据：

- L0 trigger / L1 schema / L5 anchor / L7 safety gates：
  `cascade_planner/agent/literature_templates.py`
- L2 retron matcher / applicability / product-specific cut：
  `cascade_planner/agent/template_applicability.py`
- L3 instantiation / reconstruction / chemical sanity / validation report：
  `cascade_planner/agent/executable_template_validation.py`
- L4 ChemEnzy one-step source：
  `cascade_planner/baselines/literature_one_step_plugin.py`
- L4 local wrapper 接入 ChemEnzy planner：
  `cascade_planner/baselines/chem_enzy_adapter.py`
- L4 one-step provider normalization：
  `cascade_planner/baselines/chem_enzy_onestep.py`
- L0 triggered literature task gate：
  `cascade_planner/agent/literature_research.py` 和
  `cascade_planner/agent/smiles_first.py`
- advisory-only guard：
  `cascade_planner/agent/strategic_candidate_generation.py`
- cascade source policy / route-tree source group 支持：
  `cascade_planner/route_tree/source_gate.py`
- L6 A/B benchmark was historical evidence and is now archived locally under
  `archive/harness_prep_20260605/scripts/run_literature_template_plugin_benchmark.py`.

测试证据：

- `tests/test_literature_template_cards.py`
- `tests/test_template_applicability.py`
- `tests/test_executable_template_validation.py`
- `tests/test_literature_one_step_plugin.py`
- Historical A/B benchmark test archived locally:
  `archive/harness_prep_20260605/tests/test_literature_template_plugin_benchmark.py`

已执行验证：

```bash
pytest -q tests/test_literature_template_cards.py tests/test_template_applicability.py tests/test_executable_template_validation.py tests/test_literature_one_step_plugin.py
pytest -q tests/test_agent_artifact_contracts.py tests/test_literature_evidence_cards.py tests/test_smiles_first_workflow.py
python -m py_compile cascade_planner/agent/literature_templates.py cascade_planner/agent/template_applicability.py cascade_planner/agent/executable_template_validation.py cascade_planner/baselines/literature_one_step_plugin.py cascade_planner/baselines/chem_enzy_adapter.py cascade_planner/baselines/chem_enzy_onestep.py cascade_planner/agent/chem_enzy_policy.py cascade_planner/agent/smiles_first.py
```

边界说明：

- 未直接修改 vendored ChemEnzy 文件；`vendor/...prepare_utils.py` 和
  `vendor/...cascade_source_policy.py` 对应任务通过本地 wrapper 接入层
  `LiteratureTemplateOneStepWrapper`、`ChemEnzyBackendAdapter._configure_native_autoplanner_plugins`
  和 `route_tree/source_gate.py` 完成。
- “MVP 不要求”下的勾选表示这些能力已明确保持在 MVP 外，并由
  `no_solved_claim`、`production_kb_promotion_gate`、
  `template_compliance_gate`、`not_raw_reaction_injection` 等 guard 防止误用；
  不是表示已经实现全自动 SI 解析或生产 KB 写入。
