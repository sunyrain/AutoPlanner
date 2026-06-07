# Atorvastatin Template Extraction 全流程说明

日期：2026-06-06
示例：`atorvastatin_template_extraction_demo_20260606`
目标：用 atorvastatin 这个真实 demo 说明 AutoPlanner 从失败的 native route、开放文献证据、source-detail exact step、下游编译、ChemEnzy replay、route expansion 到 self-evo gate 的完整链路。

本文只描述仓库中已经落盘的可审计产物，不把文献片段当成已解决路线，也不把结构模板直接写入生产 KB。核心原则是：Codex/开放研究可以发现、整理和提出候选，但 solved 判定、route closure、template 可执行性和 KB 晋级必须由 harness/validator 做确定性判断。

## 0. 关键结论

本 demo 的最终状态不是 solved，而是 `partial_source_detail_segment_handoff_not_solved`。

关键数字来自 `results/shared/atorvastatin_template_extraction_demo_20260606/demo_summary.json`：

| 字段 | 值 |
|---|---:|
| `case_id` | `atorvastatin` |
| `target_name` | `atorvastatin` |
| `compiled_accepted` | `true` |
| `executable_template_maturity` | `executable_ready` |
| `one_step_row_count` | `2` |
| `template_card_count` | `4` |
| `route_expansion_child_target_count` | `2` |
| `agent_followup_action_count` | `5` |
| `plugin_replay_return_counts` | `[1, 1]` |
| `free_acid_target_replay_return_count` | `0` |
| `self_evo_replay_accepted` | `true` |
| `self_evo_staging_candidate_count` | `1` |
| `self_evo_production_write_blocked` | `true` |

安全状态：

| 安全字段 | 值 | 含义 |
|---|---:|---|
| `no_solved_claim` | `true` | 不能声称 atorvastatin 已解决 |
| `production_write_blocked` | `true` | 不能写入生产 KB |
| `raw_reaction_injection` | `false` | 没有把 LLM 反应直接注入 ChemEnzy |

为什么 `executable_ready` 但仍不是 solved：两个 source-detail exact steps 已经能被 deterministic compiler 转成 one-step rows 并供 literature template plugin replay 使用；但这两个步骤只覆盖一个高级保护中间体到 atorvastatin hemi-calcium salt 的 endgame segment，不包含从 stock starting materials 到高级中间体的闭合，也没有完成 calcium salt/free acid target normalization 的 solved 审计。

## 1. 产物目录

本说明基于以下目录：

```text
results/shared/atorvastatin_template_extraction_demo_20260606/
```

主要文件：

| 文件 | 作用 |
|---|---|
| `demo_summary.json` | demo 汇总，记录数量、状态、安全门和 source run |
| `structure_template_report.md` | 面向人的结构/模板研究报告 |
| `open_agent_audit.json` | open structure agent 的最终审计 |
| `evidence/literature_sources.json` | 文献、PubChem、专利 metadata 的分级和排除记录 |
| `evidence/pubchem_validated_compounds.json` | RDKit/PubChem 结构锚点验证 |
| `evidence/source_detail_curator_records.json` | 人工/curator 结构化的 source-detail route-step 记录 |
| `evidence/source_detail_resolution_pack.json` | harness 统一后的 source-detail resolution pack |
| `downstream_consumables.json` | open research 产出的下游消费草案 |
| `compiled_downstream_consumables.json` | deterministic compiler 汇总后的可消费产物 |
| `compiled_literature_template_plugin.json` | template cards、one-step rows、validation reports 和 plugin flags |
| `compiled_guided_chemenzy_requests.json` | ChemEnzy guided rerun policies/operators |
| `compiled_route_expansion_tasks.json` | route expansion tasks 和 source-detail child targets |
| `compiled_executable_template_maturity.json` | executable template 成熟度与缺口说明 |
| `template_plugin_replay.json` | literature template plugin replay 结果 |
| `agent_followup_handoff.json` | 下一轮 agent/tool handoff 动作 |
| `self_evo_staging_kb.json` | self-evo staging 编译结果 |
| `self_evo_replay_gate.json` | self-evo replay gate 结果 |

原始 source run 目录记录在 `demo_summary.json`：

```text
results/shared/atorvastatin_latest_small_stock_depth20_real_20260606/open_structure_research_rerun_after_seed_fix
```

## 2. 全流程概览

端到端数据流如下：

```text
target input
-> deterministic preflight / native ChemEnzy run
-> route verifier rejects fake closure
-> open structure/template research
-> literature/PubChem/patent source triage
-> source-detail curator records
-> source_detail_resolution_pack
-> downstream_consumables draft
-> deterministic downstream compiler
-> literature template plugin one-step rows
-> guided ChemEnzy rerun payloads
-> route expansion child targets
-> self-evo staging and replay gate
-> agent_followup_handoff
-> final audit: partial, not solved, production blocked
```

这里最重要的边界是：open research 阶段只能交付 evidence-backed drafts；compiler 和 validators 决定哪些 draft 可以进入 one-step/plugin/replay；route verifier 决定是否 solved；self-evo gate 决定是否能晋级，但本 run 明确禁止生产写入。

## 3. 输入目标与 native ChemEnzy 审计

目标是 atorvastatin free acid。`structure_template_candidates.json` 中的 canonical target：

```text
CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O
```

结构锚点：

| 角色 | 名称 | Formula | InChIKey | 来源状态 |
|---|---|---|---|---|
| target | atorvastatin free acid | `C33H35FN2O5` | `XUKUURHRXDUEBC-KAYWLYCHSA-N` | target input + PubChem CID/formula metadata |
| intermediate | atorvastatin tert-butyl ester | `C37H43FN2O5` | `GCPKKGVOCBYRML-LOYHVIPDSA-N` | local ChemEnzy immediate precursor + PubChem name/formula/CID metadata |
| intermediate | atorvastatin acetonide tert-butyl ester | `C40H47FN2O5` | `NPPZOMYSGNZDKY-ROJLCIKYSA-N` | PubChem web metadata SMILES，RDKit-valid |
| source product | atorvastatin hemi-calcium salt | `C66H68CaF2N4O10` | `FQCKMBLVYCEXJB-MNSAWQCASA-L` | source method product representation，free-acid normalization pending |

native ChemEnzy 在 depth 20 下报告 solved，并返回 29 条路线；但 harness route verifier 接受了 0 条。`structure_template_report.md` 和 `open_agent_audit.json` 记录的降级原因包括：

- `fake_closed_rejected`
- hidden nonstock reactants
- large atom jump concern
- 缺少 condition/evidence support
- native route verifier accepted zero routes

因此 frontier 保持为 atorvastatin free-acid target，不能从 native ChemEnzy 的 solved flag 直接进入 solved 状态。

## 4. 开放文献和结构证据分流

`evidence/literature_sources.json` 按关系类型管理来源：

```text
exact_target
exact_intermediate
close_analog
family_only
method_reference
unrelated
unusable
```

核心规则是：

```text
Only exact_target/exact_intermediate sources with source-grounded structures
may support executable templates or source_detail_route_steps.
```

被保留的关键来源：

| `source_id` | 关系 | 证据级别 | 用途 |
|---|---|---|---|
| `web_pubchem_atorvastatin_cid60823_turn1search4` | `exact_target` | PubChem identity/formula/CID metadata | free-acid target identity anchor |
| `web_pubchem_tbutyl_ester_cid11238911_turn1search0` | `exact_intermediate` | PubChem name/formula/CID metadata | tert-butyl ester identity anchor |
| `web_pubchem_acetonide_tbutyl_cid10168503_turn1search1` | `exact_intermediate` | PubChem descriptor SMILES/formula/CID metadata | acetonide tert-butyl ester structure anchor |
| `web_pmc_s13065_015_0082_7_turn0search6` | `exact_target` | open-access methods + metadata | two source-detail route steps and route segment |
| `web_joc_jo402829b_turn0search1` | `exact_target` | primary literature metadata | extraction task |
| `web_rsc_b919115c_turn0search2` | `exact_intermediate` | primary literature metadata | extraction task/self-evo seed |
| `web_patent_wo2005097742_turn0search3` | `exact_intermediate` | patent metadata, not fetched | extraction task only |
| `web_ep1922315b1_turn1search19` | `exact_intermediate` | patent metadata/PDF not fetched | extraction task only |
| `web_tandf_2015_1111382_turn0search8` | `exact_intermediate` | primary literature metadata | pyrrole nucleus extraction task |
| `web_bcsj_20210178_turn0search9` | `exact_target` | primary literature metadata | extraction task |
| `harness_crossref_lipitor_chapter_10_1002_9780470909775_ch9` | `exact_target` | process chapter metadata | extraction task |

被排除或降级的来源：

- `excluded_sources` 共 20 条。
- 人工 case name 中的 `latest`、`small`、`stock`、`real` 触发了无关 CrossRef 命中，这些结果保留在 excluded/gap 中，不能支持路线。
- atorvastatin lactone 找到 PubChem CID/formula metadata，但本 run 没有 source-grounded SMILES，所以被拒绝为结构候选。
- patent metadata 只作为 extraction task evidence；本 run 没有抓取 patent full text/PDF。

## 5. Source-Detail 结构化

`evidence/source_detail_curator_records.json` 给出两个 source-detail route steps，均来自 open-access PMC source：

```text
source_ref: web_pmc_s13065_015_0082_7_turn0search6
doi: 10.1186/s13065-015-0082-7
pmc: PMC4333361
segment_id: litseg_pmc_kg_atorvastatin_4_to_5_to_calcium_v1
```

### 5.1 Step 1：advanced ketal ester 4 -> diol tert-butyl ester 5

`step_id`：

```text
sd_step_pmc_kg_atorvastatin_4_to_5_deketalization_v1
```

Product：

```text
tert-Butyl-(3R,5R)-7-[2-(4-fluorophenyl)-5-isopropyl-3-phenyl-4-(phenylcarbamoyl)pyrrol-1-yl]-3,5-dihydroxyheptanoate (compound 5)
```

Product SMILES：

```text
CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)OC(C)(C)C
```

Reactant：

```text
advanced ketal ester atorvastatin precursor 4 / atorvastatin acetonide tert-butyl ester
```

Reactant SMILES：

```text
CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H]1C[C@H](CC(=O)OC(C)(C)C)OC(C)(C)O1
```

Condition candidate 字段是 evidence-backed audit metadata，不是生产 KB 模板：

| 字段 | 值 |
|---|---|
| `reaction_class` | acidic ketal deprotection / diol tert-butyl ester crystallization |
| `solvent_candidates` | isopropyl alcohol, water |
| `reagent_candidates` | aqueous hydrochloric acid |
| `temperature_C` | `60` |
| `duration_h` | `1` |
| `reported_yield` | `96%` |
| `isolation` | crystalline compound 5 isolated after cooling/centrifugation/drying |
| `condition_status` | `evidence_backed` |
| `source_type` | `exact` |

Applicability：

```text
exact advanced atorvastatin intermediate deketalization;
route segment evidence for target-family synthesis, not stock closure
```

### 5.2 Step 2：diol tert-butyl ester 5 -> atorvastatin hemi-calcium salt

`step_id`：

```text
sd_step_pmc_kg_atorvastatin_5_to_hemi_calcium_v1
```

Product：

```text
atorvastatin hemi-calcium salt (compound 1)
```

Product SMILES：

```text
CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)[O-].CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)[O-].[Ca+2]
```

Reactant：

```text
compound 5 tert-butyl diol ester, two organic equivalents represented for Ca(atorvastatin)2 stoichiometry
```

Reactant SMILES：

```text
CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)OC(C)(C)C
```

Condition candidate 字段：

| 字段 | 值 |
|---|---|
| `reaction_class` | tert-butyl ester hydrolysis and calcium salt formation |
| `solvent_candidates` | methanol, water, ethyl acetate, ethanol workup |
| `reagent_candidates` | sodium hydroxide, calcium acetate monohydrate |
| `temperature_C` | `40` |
| `hydrolysis_duration_min` | `30` |
| `reported_yield` | `78.7%` |
| `reported_purity` | `99.9%` |
| `isolation` | hemi-calcium salt isolated after biphasic extraction and ethanol crystallization |
| `condition_status` | `evidence_backed` |
| `source_type` | `exact` |

Applicability：

```text
exact target salt-form endgame;
downstream must normalize/audit against free-acid target frontier before claiming solved
```

## 6. Source-Detail Resolution Pack

`evidence/source_detail_resolution_pack.json` 是 harness 对 source-detail extraction 的统一产物。它的 source policy：

| 字段 | 值 |
|---|---:|
| `do_not_fabricate_smiles` | `true` |
| `explicit_smiles_markers_required` | `true` |
| `harness_owned_source_detail_resolution` | `true` |
| `structured_curator_records_allowed` | `true` |
| `pmc_xml_signal_scan_only` | `true` |
| `full_text_content_stored` | `false` |
| `procedure_text_stored` | `false` |
| `no_solved_claim` | `true` |
| `production_write_blocked` | `true` |

Summary：

| 字段 | 值 |
|---|---:|
| `access_probe_count` | `5` |
| `curator_record_count` | `2` |
| `curator_step_count` | `2` |
| `gap_count` | `10` |
| `resolved_queue_count` | `0` |
| `signal_audit_count` | `0` |
| `source_detail_route_step_count` | `2` |

该 pack 对两个 step 都补齐了 compiler 需要的标准化字段：

- `schema_version: source_detail_route_step.v1`
- `relation_type: exact`
- `product_smiles`
- `reactant_smiles`
- `source_ref`
- `evidence_refs`
- `condition_candidate`
- `applicability.status: passed`
- `applicability.product_reconstruction_passed: true`
- `not_raw_reaction_injection: true`
- `no_solved_claim: true`
- `production_write_blocked: true`

仍保留 10 个 extraction gaps，主要原因是 metadata-only source 或 DOI 不能直接解析为 source-grounded route structures。这些 gap 被转成 extraction tasks，而不是被编造成模板。

## 7. Downstream Consumables 草案

`downstream_consumables.json` 是 open research 给 deterministic compiler 的草案输入。它包含：

| 字段 | 数量 | 说明 |
|---|---:|---|
| `source_detail_route_steps` | `2` | PMC exact endgame steps |
| `literature_route_segments` | `1` | compound 4 -> 5 -> hemi-calcium segment |
| `literature_template_cards` | `4` | advisory strategy cards |
| `guided_rerun_requests` | `4` | guided ChemEnzy rerun requests |
| `route_expansion_tasks` | `4` | route expansion tasks |
| `executable_template_extraction_tasks` | `11` | 后续 source-detail 提取任务 |
| `evolution_candidates` | `4` | self-evo candidate/shadow seeds |
| `executable_template_candidates` | `0` | 没有直接生产级 executable template candidate |
| `rejected_consumables` | `4` | 原生假闭合、lactone 缺 SMILES、false-positive metadata 等 |

Planner handoff 明确写明：

```text
solved: false
production_kb_promotion: false
next_action:
validate PMC source-detail route segment,
normalize calcium salt/free acid target representation,
then run guided ChemEnzy from protected intermediate anchors
```

## 8. Deterministic Downstream Compiler

编译入口在 `cascade_planner/harness/downstream_compiler.py`：

```text
compile_downstream_consumables(...)
write_compiled_downstream_artifacts(...)
```

编译器做五类事情：

1. `_compile_guided_requests`：把 `guided_rerun_requests` 转成 `StrategicOperator` 和 `chem_enzy_search_policy`。
2. `_compile_route_expansion_tasks`：把 route expansion draft 转成 bounded rerun policies。
3. `_compile_template_plugin`：把 advisory template cards 和 source-detail exact steps 编译成 plugin 输入。
4. `_compile_executable_template_maturity`：输出成熟度和缺口，不放大 solved 声明。
5. `_compile_self_evo`：把候选放入 candidate/shadow/staging flow，但保持 production blocked。

最终输出 `compiled_downstream_consumables.json`：

| 字段 | 值 |
|---|---:|
| `accepted` | `true` |
| `case_id` | `atorvastatin` |
| `guided_chemenzy.policy_payloads` | `8` |
| `guided_chemenzy.operators` | `8` |
| `route_expansion.tasks` | `4` |
| `route_expansion.child_targets` | `2` |
| `literature_template_plugin.enabled` | `true` |
| `literature_template_plugin.template_cards` | `4` |
| `literature_template_plugin.one_step_rows` | `2` |
| `self_evo.staging_candidate_count` | `1` |

`compiled_reasons` 中仍记录：

```text
invalid_candidate_type
invalid_target_smiles
missing_case_id
segment_step_count_out_of_range
```

这些不是编译失败，而是 rejected/gap 项的审计原因；因为仍有 template rows、policy payloads、extraction tasks 和 self-evo staging candidate，整体 `accepted=true`。

## 9. Source-Detail Exact Step 到 One-Step Row

核心函数：

```text
_source_detail_exact_step_one_step_row(step)
```

它只接受满足以下条件的 source-detail step：

- `step_id` 非空
- `relation_type == "exact"`
- product SMILES 是 RDKit-valid
- reactant SMILES 存在且全部 RDKit-valid
- `source_ref` 非空
- `evidence_refs` 非空
- `applicability.status` 是 `passed` 或 `exact`
- `applicability.product_reconstruction_passed == true`
- condition audit 不是 `gap` 或 `high`

如果通过，compiler 生成 one-step row：

```text
reactants: "." joined reactant_smiles
scores: 0.62
model_full_name: autoplanner.literature_template_plugin
reaction_domains: literature_chemical
source_policy_decision: enabled_literature_template_plugin
no_solved_claim: true
requires_audit: true
```

每个 row 的 `template_validation_report` 都包含：

- `accepted: true`
- `allowed_for_one_step_source: true`
- `confidence: high`
- `audit_required: true`
- `no_solved_claim: true`
- forward reconstruction audit passed
- basic chemical sanity passed

需要注意的 atom accounting policy：

```text
source_detail_exact_step_allows_reagent_or_byproduct_atoms_outside_precursor_list
```

这意味着 one-step row 用于 source-grounded exact literature step replay；reagents/byproducts 可以在 condition fields 中表示，不要求都在 precursor list 中。

## 10. Literature Template Plugin

`compiled_literature_template_plugin.json` 中：

```text
enabled: true
one_step_rows: 2
template_cards: 4
validation_reports: 8
```

Plugin flags：

| 字段 | 值 |
|---|---:|
| `enabled` | `true` |
| `top_k` | `4` |
| `max_added` | `2` |
| `requires_audit` | `true` |
| `not_raw_reaction_injection` | `true` |

四张 template cards 都是 advisory strategy，不允许 direct consumption：

| `template_id` | 级别 | 用途 |
|---|---|---|
| `ltc_atorvastatin_tbutyl_ester_to_free_acid_advisory_v1` | `advisory_strategy` | tert-butyl ester deprotection to free acid seed |
| `ltc_acetonide_tbutyl_deprotection_hydrolysis_advisory_v1` | `advisory_strategy` | acetonide tert-butyl ester deprotection/hydrolysis anchor |
| `ltc_biocatalytic_keto_to_hydroxy_intermediate_extraction_seed_v1` | `advisory_strategy` | biocatalytic intermediate extraction seed |
| `ltc_paal_knorr_pyrrole_nucleus_extraction_seed_v1` | `advisory_strategy` | pyrrole nucleus construction extraction seed |

两个 executable one-step rows 的 `template_id`：

```text
source_detail_exact_step:sd_step_pmc_kg_atorvastatin_4_to_5_deketalization_v1
source_detail_exact_step:sd_step_pmc_kg_atorvastatin_5_to_hemi_calcium_v1
```

这里的 executable 是 plugin/harness 意义上的可 replay row，不是 solved route，也不是 production KB template。

## 11. Template Plugin Replay

`template_plugin_replay.json` 用编译出的 plugin rows 做了 replay。

结果：

| 字段 | 值 |
|---|---:|
| `compiled_one_step_row_count` | `2` |
| `compiled_template_card_count` | `4` |
| `plugin_state.added_candidates` | `2` |
| `plugin_state.calls` | `3` |
| `plugin_state.validation_passed` | `2` |
| `plugin_state.validation_rejected` | `0` |
| `plugin_state.error_count` | `0` |
| `production_write_blocked` | `true` |

Replay row 1：

```text
product: compound 5 tert-butyl diol ester
reactants_returned:
  advanced ketal ester / acetonide tert-butyl ester
validation_allowed: true
```

Replay row 2：

```text
product: atorvastatin hemi-calcium salt
reactants_returned:
  compound 5 tert-butyl diol ester . compound 5 tert-butyl diol ester
validation_allowed: true
```

Free-acid target replay 返回空：

```text
free_acid_target_replay.reactants_returned: []
```

原因也记录在 JSON 中：

```text
Expected empty unless a compiled row is grounded on the free-acid target;
current source-detail rows are advanced intermediate and calcium-salt endgame steps.
```

这正是不能把 demo 判定为 solved 的关键。

## 12. Guided ChemEnzy Requests

`compiled_guided_chemenzy_requests.json` 中有 8 个 operators/policy payloads，其中 4 个来自 guided requests，4 个来自 route expansion tasks 的 policy 编译。

统一预算形态：

```text
max_reruns: 1
max_iterations: 50
max_depth: 15
expansion_topk: 100
mode: guided
not_raw_reaction_injection: true
```

核心 guided request：

| `operator_id` | 意图 |
|---|---|
| `grr_frontier_with_verified_identity_anchors_v1` | 用 free acid target、tert-butyl ester、acetonide tert-butyl ester 结构锚点重新引导 frontier expansion |
| `grr_test_tbutyl_ester_first_disconnection_v1` | 测试 tert-butyl ester -> free acid 的第一断键假设 |
| `grr_expand_acetonide_tbutyl_intermediate_after_extraction_v1` | 在 source-detail extraction 后扩展 acetonide tert-butyl intermediate |
| `grr_replay_pmc_kg_segment_with_salt_normalization_v1` | replay PMC kilogram-scale segment，并要求 salt/free-acid normalization audit |

必须保留的 audit gates：

- no hidden nonstock reactants
- no large atom jump
- source-detail required before executable template
- conditions required for route acceptance
- RDKit parse
- salt/free-acid equivalence audit
- stock closure for compound 4 precursor

## 13. Route Expansion Child Targets

`compiled_route_expansion_tasks.json` 中有 2 个 child targets，均由 source-detail exact one-step row 的 reactants 派生。

派生函数：

```text
_source_detail_child_targets_from_one_step_rows(rows, case_id=...)
```

### Child target 1

`child_target_id`：

```text
atorvastatin_source_detail_exact_step_sd_step_pmc_kg_atorvastatin_4_to_5_deketalization_v1_reactant_1
```

SMILES：

```text
CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H]1C[C@H](CC(=O)OC(C)(C)C)OC(C)(C)O1
```

含义：探索 PMC Step 1 的 upstream reactant，即高级 ketal/acetonide tert-butyl ester 中间体如何继续向 stock closure 展开。

### Child target 2

`child_target_id`：

```text
atorvastatin_source_detail_exact_step_sd_step_pmc_kg_atorvastatin_5_to_hemi_calcium_v1_reactant_1
```

SMILES：

```text
CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)OC(C)(C)C
```

含义：探索 PMC Step 2 的 upstream reactant，即 compound 5 tert-butyl diol ester 如何进一步闭合或连接到上游路线。

每个 child target 都携带：

```text
source: source_detail_one_step_reactant
mode: guided
rerun_reason: explore upstream reactant from source-detail literature step
preferred_reaction_classes: [source_detail_upstream_expansion]
max_depth: 15
max_iterations: 50
expansion_topk: 100
not_raw_reaction_injection: true
no_solved_claim: true
production_write_blocked: true
```

实际工具入口：

```text
run_route_expansion_subgoal_search
```

该工具会读取 compiled downstream 中的 child targets，构造子目标 ChemEnzy 请求，并对每个子目标 raw route 再跑 route verifier。只有 verifier accepted 才能把 subgoal 标为 solved。

## 14. Agent Follow-Up Handoff

`agent_followup_handoff.json` 生成 5 个后续动作：

| `action_id` | `tool_name` | 原因 |
|---|---|---|
| `atorvastatin_replay_literature_template_plugin` | `run_guided_chemenzy_rerun` | `source_detail_one_step_rows_available` |
| `atorvastatin_explore_source_detail_upstream_reactants` | `run_route_expansion_subgoal_search` | `source_detail_step_reactants_are_upstream_child_targets` |
| `atorvastatin_guided_chemenzy_policy_rerun` | `run_guided_chemenzy_rerun` | `compiled_guided_chemenzy_policy_available` |
| `atorvastatin_route_expansion_subgoal_search` | `run_route_expansion_subgoal_search` | `compiled_route_expansion_tasks_available` |
| `atorvastatin_self_evo_replay_gate` | `run_self_evo_replay_gate` | `self_evo_staging_candidates_available` |

所有 action 都带：

```text
no_solved_claim: true
production_write_blocked: true
```

这保证下一轮 agent 可以继续推进，但不能绕过 solved/production gate。

## 15. Self-Evo Staging 和 Replay Gate

`self_evo_staging_kb.json`：

```text
accepted: true
staging_candidate_count: 1
production_write_blocked: true
```

进入 staging 的候选：

```text
evo_candidate_atorvastatin_tbutyl_deprotection_shadow_v1
```

候选类型：

```text
TemplateCandidate
```

payload ref：

```text
ltc_atorvastatin_tbutyl_ester_to_free_acid_advisory_v1
```

promotion blockers：

```text
source_detail_product_reactant_step_missing
condition_candidate_missing
PubChem typed SMILES not extracted for tert-butyl ester
```

`self_evo_replay_gate.json`：

| 字段 | 值 |
|---|---:|
| `accepted` | `true` |
| `allow_production` | `false` |
| `production_write_blocked` | `true` |
| `production_promoted_count` | `0` |
| `staging_candidate_count` | `1` |
| `benchmark_gate.accepted` | `true` |
| `true_solved_rate_delta` | `0.0` |
| `fake_closure_rate_delta` | `0.0` |
| `condition_quality_delta` | `0.0` |

原因：

```text
production_not_requested
target_run_production_blocked
```

self-evo 的含义是：候选可以作为 target-run staging memory 被 replay/audit，但不能进入 production layer。

## 16. Open Agent Audit

`open_agent_audit.json` 的最终状态：

```text
final_status: partial_source_detail_segment_handoff_not_solved
solved: false
production_kb_promotion: false
```

通过的检查：

- manifest read first
- native ChemEnzy audit inspected
- harness prefetch preserved and enriched
- RDKit validation
- executable template guard
- production KB guard
- source-detail pack processed before broad search
- source-detail route steps emitted

主要限制：

- patent 和可选 PDF/full-text fetching 被 deferred。
- native ChemEnzy immediate tert-butyl ester 是 RDKit-valid，并有 PubChem name/formula support，但本 run 没有 typed PubChem connector SMILES。
- atorvastatin lactone 有 CID/formula metadata，但缺 source-grounded SMILES。
- 人工 case name 带来的 false positives 不能支持 route evidence。
- PMC segment 从高级 protected intermediate 4 开始，终止于 calcium salt representation，不是 free-acid target 的完整 stock-closed route。

下一步动作：

- deterministic validators 先验证 PMC source-detail segment 和 salt/free-acid normalization。
- typed PubChem/curator 提取 CID 11238911 和 CID 6483036 的 source-grounded SMILES。
- 从 PMC article 和 exact-intermediate patent metadata 提取更多 exact product/reactant rows。
- 用 target、tert-butyl ester、acetonide tert-butyl ester anchors 做 guided ChemEnzy，并强制 no hidden nonstock closure。
- 在 deterministic validators 接受 source-detail route steps 和 conditions 前，不晋级 production。

## 17. 为什么这是完整但保守的链路

这个 demo 已经完成了从 evidence 到 executable plugin row 的关键闭环：

```text
source-grounded exact literature step
-> RDKit-valid product/reactant
-> condition audit ok
-> one-step row
-> plugin replay returns expected reactants
-> upstream child target generated
-> guided ChemEnzy / route expansion / self-evo follow-up action generated
```

但它仍刻意不做三件事：

1. 不把 native ChemEnzy 的 solved flag 当成 solved，因为 route verifier 拒绝了所有 native routes。
2. 不把 advisory template cards 当成 executable templates，因为 direct consumption 被禁用。
3. 不把 self-evo staging candidate 写入 production KB，因为 target-run production gate 被关闭。

因此这个例子展示的是 AutoPlanner 当前主线想要的行为：开放 agent 可以把文献中的结构化证据变成可 replay 的下游资产，但所有高风险决策都被 deterministic gates 接住。

## 18. 用模板后的效果实验

为了补上“两个 source-detail 模板用上之后，完整路线是否变好”的实测结果，新增了一组对比实验：

```text
results/shared/atorvastatin_template_plugin_effect_20260606/
```

实验设置：

- baseline：原始 atorvastatin free-acid native ChemEnzy run。
- free-acid with templates：同一 free-acid target，显式启用 `compiled_literature_template_plugin.json` 中的 2 个 one-step rows 和 4 个 advisory cards。
- child target 1：模板 A 派生的 acetonide tert-butyl ester upstream child target。
- child target 2：模板 B 派生的 compound 5 tert-butyl diol ester upstream child target。
- 每个返回路线的实验都再跑同一个 deterministic `verify_chemenzy_raw_routes`。

汇总文件：

```text
results/shared/atorvastatin_template_plugin_effect_20260606/effect_comparison_summary.json
```

结果表：

| 实验 | ChemEnzy 原始状态 | 原始返回路线 | Verifier 状态 | 结论 |
|---|---|---:|---|---|
| baseline free acid | `solved` | 29 | `fake_closed_rejected`, accepted 0/29 | 原生 ChemEnzy 假闭合 |
| free acid + templates | `failed` | 0 | `no_routes_returned` | 完整 free-acid 路线没有变好 |
| child target 1 + templates | `solved` | 757 | `fake_closed_rejected`, accepted 0/757 | 上游子目标仍是假闭合 |
| child target 2 + templates | `failed` | 0 | `no_routes_returned` | compound 5 子目标未闭合 |

free-acid target 的带模板结果：

```text
raw_result_path: results/shared/atorvastatin_template_plugin_effect_20260606/free_acid_with_templates_raw_result.json
verifier_path:   results/shared/atorvastatin_template_plugin_effect_20260606/free_acid_with_templates_verifier.json
search_status:   failed
backend_failure: IndexError: list index out of range
n_results:       0
verifier:        no_routes_returned
```

这里的 template plugin 不是没有启用。raw metadata 里记录：

```text
enabled: true
one_step_row_count: 2
template_card_count: 4
validation_passed: 1
added_candidates: 1
duplicate_candidates: 1
```

也就是说，模板确实进入 one-step candidate flow，并加入过候选；但它没有带来完整路线，反而触发了 ChemEnzy 后端搜索失败。因此完整 free-acid 路线层面没有改善。

child target 1 是模板 A 的上游 reactant：

```text
CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H]1C[C@H](CC(=O)OC(C)(C)C)OC(C)(C)O1
```

ChemEnzy 对它返回了大量候选：

```text
n_results: 757
native_raw_n_routes: 987
search_status: solved
```

但 verifier 结果是：

```text
accepted: false
route_status: fake_closed_rejected
accepted_route_count: 0
rejected_route_count: 757
reasons:
  hidden_nonstock_reactants
  large_atom_jump
  no_verifier_accepted_stock_closed_route
```

这说明上游 child target 搜索看起来“有路线”，但本质仍是同一类假闭合，不能用于拼接完整 atorvastatin route。

child target 2 是 compound 5 tert-butyl diol ester：

```text
CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)OC(C)(C)C
```

结果：

```text
search_status: failed
backend_failure: IndexError: list index out of range
n_results: 0
verifier: no_routes_returned
```

最终判断：

```text
complete_free_acid_route_improved: false
partial_piece_improved: false
```

这组实验把当前状态收紧了：两个 source-detail 模板本身质量仍然有效，可以 replay 文献 endgame；但把它们接入 ChemEnzy 后，还没有得到更好的完整 atorvastatin free-acid route。真正瓶颈不是 endgame 模板，而是：

- free-acid target + plugin rerun 的后端稳定性；
- salt/free-acid normalization；
- compound 5 / acetonide tert-butyl ester 上游 stock closure；
- source-detail exact steps 与 child-target search result 的自动 stitching；
- verifier 仍然识别到 hidden nonstock 和 large atom jump。

因此这两个模板目前的实测用途仍限定为：

```text
文献 endgame replay
-> 生成上游 child targets
-> 指导后续 source-detail extraction / route expansion
```

它们尚未证明可以改善完整路线搜索，也不能作为 solved route 或 production KB 条目。

## 19. 复现和检查入口

常用检查命令：

```bash
python -m json.tool results/shared/atorvastatin_template_extraction_demo_20260606/demo_summary.json
python -m json.tool results/shared/atorvastatin_template_extraction_demo_20260606/compiled_downstream_consumables.json
python -m json.tool results/shared/atorvastatin_template_extraction_demo_20260606/compiled_literature_template_plugin.json
python -m json.tool results/shared/atorvastatin_template_extraction_demo_20260606/template_plugin_replay.json
python -m json.tool results/shared/atorvastatin_template_extraction_demo_20260606/self_evo_replay_gate.json
python -m json.tool results/shared/atorvastatin_template_plugin_effect_20260606/effect_comparison_summary.json
```

需要定位代码时，看这些入口：

| 文件 | 关注点 |
|---|---|
| `scripts/run_open_structure_template_agent.py` | open structure/template agent launcher、manifest、retrieval prefetch、source-detail resolution |
| `cascade_planner/harness/downstream_compiler.py` | downstream consumables 编译、one-step rows、follow-up actions、self-evo staging |
| `cascade_planner/harness/source_detail_resolution.py` | source-detail extraction pack resolution |
| `cascade_planner/harness/tools.py` | `run_guided_chemenzy_rerun`、`run_route_expansion_subgoal_search`、`run_self_evo_replay_gate` |
| `cascade_planner/harness/self_evo_replay.py` | self-evo replay gate |

最终一句话：这个 atorvastatin demo 不是“路线已解决”的例子，而是“从文献 source-detail 到可审计下游 template/plugin/expansion/self-evo handoff”的完整样例。
