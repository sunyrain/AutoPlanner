# Bufotalin 全流程逆合成工作流

生成时间: 2026-06-07T02:58:00.240227+00:00

## 当前结论

- 文献 source-detail 链: `15` 步，完整接到 compound `11`。
- downstream compiler: `compiled_accepted=True`, `executable_status=executable_ready`。
- literature one-step plugin: `one_step_row_count=15`, `plugin_max_added=15`。
- chain probe: `accepted=True`, `terminal_reached=True`。
- ChemEnzy smoke: `raw_solved=True`, 但 verifier accepted=`False`，原因: `advanced_same_scaffold_terminal, large_atom_jump, no_verifier_accepted_stock_closed_route`。

## 新增结构图

- PNG: [bufotalin_retrosynthesis_structure_route_20260607.png](bufotalin_retrosynthesis_structure_route_20260607.png)
- PDF: [bufotalin_retrosynthesis_structure_route_20260607.pdf](bufotalin_retrosynthesis_structure_route_20260607.pdf)

## Bufotalin 全流程

1. **目标输入与 preflight**
   - 输入 target name/smiles/family hint。
   - RDKit 校验 SMILES、heavy atom count、复杂度和初始风险标签。
   - 复杂天然产物默认不能只信 ChemEnzy raw solved，后续必须经过 verifier 和文献/模板证据闭环。

2. **复用或跳过前段 ChemEnzy**
   - bufotalin 这个阶段不需要重复跑最初 ChemEnzy baseline；可复用已有 raw result、route audit 和 verifier feedback。
   - 关键判定: raw_solved=true 只代表 native search 返回了 stock-closed routes，不等于 verified solved。

3. **开放文献/结构研究队列**
   - 从 literature_sources、retrieval prefetch、source_material_locator 和 open_structure_research artifacts 里找 exact-target / exact-intermediate source。
   - 本 case 的关键 DOI 是 `10.1016/j.tet.2025.134610`；早期 advisory DOI/anchor 只作为方向性先验。

4. **PDF/图像证据提取**
   - 渲染本地全文 PDF 页面和 scheme crops。
   - source-detail worker 只抽结构化字段: compound label、product_smiles、reactant_smiles、condition_candidate、source_locator、source_excerpt。
   - 不保存全文实验步骤，不写 production KB。

5. **连续中间体 SMILES 识别与校验**
   - Codex/vision 按论文 scheme 连续识别 11、24、25、23、26、27、28、19、20、14、22、30、31、32、33、bufotalin。
   - RDKit 校验所有候选 SMILES、formula、exact mass/heavy atoms，并检查链连续性。
   - 论文正向链 `11 -> ... -> bufotalin` 在 harness 内倒成逆合成链 `bufotalin -> ... -> 11`。

6. **生成 source_detail_curator_records.v1**
   - 通过校验的结构链进入 curator records。
   - provenance=`codex_source_text_translation`，structure_derivation 记录 source_locator、confidence 和 tool_checks。
   - 默认 `main_reactant_only=true`，例如 14 -> 22 的 2-pyrone 偶联不会把辅底物当作后续 steroid child target。

7. **source-detail resolution**
   - resolver 消费 curator records，产出 source_detail_route_steps。
   - 当前 bufotalin 结果: `source_detail_route_step_count=15`, `resolution_gap_count=0`。

8. **downstream compiler**
   - compiler 把 exact source_detail_route_steps 晋级为 executable literature one-step rows。
   - 这些 rows 带 source_ref/evidence_refs/condition_candidate/applicability，不是 advisory template。
   - 当前状态: `one_step_row_count=15`, `executable_status=executable_ready`。

9. **literature_template_plugin 接入 ChemEnzy**
   - guided rerun 可以把 compiled `literature_template_plugin` 作为 one-step source。
   - 插件负责在匹配 product_smiles 时返回文献 exact reactant row。
   - 子目标也可由 compiled child_targets 进入 route expansion。

10. **ChemEnzy rerun / route expansion / verifier**
   - ChemEnzy 保留探索能力，但必须经过 deterministic verifier。
   - 当前 15-row smoke run 仍被 verifier 拒绝，说明 native route fake-closed 或 large atom jump，不能当作 solved。

11. **hybrid route set 与报告**
   - 文献链作为 high-weight baseline；ChemEnzy 探索路线作为 alternative/exploratory candidates。
   - 报告输出 PDF、Markdown、report_data JSON、completion audit 和结构路线图。

## 已验证逆合成链

| # | step_id | retrosynthetic edge | key condition | yield |
|---:|---|---|---|---|
| 1 | `tet2025_33_to_bufotalin_deprotection` | 33 -> bufotalin: global silyl deprotection | HF-pyridine; THF; room temperature; 160 h | 93% |
| 2 | `tet2025_32_to_33_acetylation` | 32 -> 33: C3 acetylation | Ac2O; pyridine; room temperature; 20 h | 90% |
| 3 | `tet2025_31_to_32_nabh4_reduction` | 31 -> 32: ketone reduction | NaBH4; MeOH; THF; 0 C; 2 h | 90% |
| 4 | `tet2025_30_to_31_tmsotf_rearrangement` | 30 -> 31: Lewis-acid rearrangement | TMSOTf; 2,6-lutidine; DCM; -78 C to room temperature; 1 h | 68% |
| 5 | `tet2025_22_to_30_mcpba_epoxidation` | 22 -> 30: epoxidation | m-CPBA; Na2CO3; DCM; -78 C to -20 C; 10 h | 70% |
| 6 | `tet2025_14_to_22_pyrone_coupling` | 14 -> 22: C17 2-pyrone coupling | compound 21; Pd(PPh3)4; CuI; LiCl; DMSO; THF; 60 C; 12 h | 62% |
| 7 | `tet2025_20_to_14_vinyl_iodide_formation` | 20 -> 14: vinyl iodide formation | Pd/C, H2; hydrazine monohydrate; triethylamine; I2; THF/water; EtOH; THF; room temperature; 50 C; room temperature; 12 h; 6 h; 1 h | 73% over two steps |
| 8 | `tet2025_19_to_20_seo2_allylic_oxidation` | 19 -> 20: C14 allylic oxidation | SeO2; formic acid; dioxane/water 4:1; 125 C; 16 h | 52% |
| 9 | `tet2025_28_to_19_tbs_protection` | 28 -> 19: C3 TBS protection | TBSCl; imidazole; DMF; 70 C; 16 h | 74% |
| 10 | `tet2025_27_to_28_deketalization` | 27 -> 28: C17 deketalization | p-TsOH monohydrate; acetone; 60 C; 1.5 h | 56% |
| 11 | `tet2025_26_to_27_elimination` | 26 -> 27: elimination | t-BuOK; DMSO; 70 C; 16 h | 73% |
| 12 | `tet2025_23_to_26_c16_bromination` | 23 -> 26: C16 bromination | pyridinium perbromide; THF; 0 C to room temperature; 2 h | 87% |
| 13 | `tet2025_25_to_23_kselectride_reduction` | 25 -> 23: stereoselective reduction | K-selectride; THF; -20 C to -5 C; 10 min then warm | 72% |
| 14 | `tet2025_24_to_25_hydrogenation` | 24 -> 25: stereoselective hydrogenation | Pd/C; H2; 4-methylpyridine; room temperature; 23 h | 92% |
| 15 | `tet2025_11_to_24_c17_ketalization` | 11 -> 24: C17 ketalization | p-TsOH monohydrate; ethylene glycol; ethylene glycol; room temperature; 2 h | 93% |

## 关键 artifacts

- `chain_tool_probe`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_harness_chain_tool_probe_20260607_terminal_fixed/source_detail_route_chain_audit.json`
- `chemenzy_smoke_summary`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_tet2025_fullroute_15row_smoke_rerun_20260607/summary.json`
- `chemenzy_verifier`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_tet2025_fullroute_15row_smoke_rerun_20260607/verifier.json`
- `compiled_downstream`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_tet2025_pdf_fullroute_to_androstenedione_source_detail_20260607/compiled_downstream_consumables.json`
- `source_detail_chain_audit`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_tet2025_pdf_fullroute_to_androstenedione_source_detail_20260607/source_detail_route_chain_audit.json`
- `source_detail_curator_records`: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_tet2025_pdf_fullroute_to_androstenedione_source_detail_20260607/source_detail_curator_records.json`

## 工程接口

- `extract_pdf_literature_structures`: 生成 PDF/page/crop 证据索引。
- `validate_literature_intermediate_chain`: 校验视觉/来源结构候选链并倒成逆合成链。
- `build_source_detail_curator_records`: 生成 curator records 并触发 source-detail resolution / downstream compile。
- `compile_source_detail_chain_route`: 从 one-step rows unroll 文献链并审计 terminal。
- `compile_hybrid_route_set`: 汇总文献 baseline 与 ChemEnzy exploratory routes。
