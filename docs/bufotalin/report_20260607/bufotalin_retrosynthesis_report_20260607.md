# Bufotalin 逆合成审计报告

- 生成时间: 2026-06-06T17:50:16.346852+00:00
- 主文献: doi:10.1016/j.tet.2025.134610 (Construction of advanced intermediate sharing C14-beta-OH for the synthesis of bufotalin)
- Target: bufotalin, C26H36O6, exact MW 444.2512

## 结论

文献 source-detail 链已经从 bufotalin 逆向展开到 compound 11，共 15 步；compiler 产生 15 条 executable one-step rows，状态为 executable_ready。
ChemEnzy smoke run 的 raw_solved=true 不能视为完成，因为 route verifier 拒绝了全部路线: advanced_same_scaffold_terminal, large_atom_jump, no_verifier_accepted_stock_closed_route。

## 逆合成步骤

| # | disconnection | conditions | yield |
|---:|---|---|---|
| 1 | 33 -> bufotalin: global silyl deprotection | HF-pyridine; THF; room temperature; 160 h | 93% |
| 2 | 32 -> 33: C3 acetylation | Ac2O; pyridine; room temperature; 20 h | 90% |
| 3 | 31 -> 32: ketone reduction | NaBH4; MeOH; THF; 0 C; 2 h | 90% |
| 4 | 30 -> 31: Lewis-acid rearrangement | TMSOTf; 2,6-lutidine; DCM; -78 C to room temperature; 1 h | 68% |
| 5 | 22 -> 30: epoxidation | m-CPBA; Na2CO3; DCM; -78 C to -20 C; 10 h | 70% |
| 6 | 14 -> 22: C17 2-pyrone coupling | compound 21; Pd(PPh3)4; CuI; LiCl; DMSO; THF; 60 C; 12 h | 62% |
| 7 | 20 -> 14: vinyl iodide formation | Pd/C, H2; hydrazine monohydrate; triethylamine; I2; THF/water; EtOH; THF; room temperature; 50 C; room temperature; 12 h; 6 h; 1 h | 73% over two steps |
| 8 | 19 -> 20: C14 allylic oxidation | SeO2; formic acid; dioxane/water 4:1; 125 C; 16 h | 52% |
| 9 | 28 -> 19: C3 TBS protection | TBSCl; imidazole; DMF; 70 C; 16 h | 74% |
| 10 | 27 -> 28: C17 deketalization | p-TsOH monohydrate; acetone; 60 C; 1.5 h | 56% |
| 11 | 26 -> 27: elimination | t-BuOK; DMSO; 70 C; 16 h | 73% |
| 12 | 23 -> 26: C16 bromination | pyridinium perbromide; THF; 0 C to room temperature; 2 h | 87% |
| 13 | 25 -> 23: stereoselective reduction | K-selectride; THF; -20 C to -5 C; 10 min then warm | 72% |
| 14 | 24 -> 25: stereoselective hydrogenation | Pd/C; H2; 4-methylpyridine; room temperature; 23 h | 92% |
| 15 | 11 -> 24: C17 ketalization | p-TsOH monohydrate; ethylene glycol; ethylene glycol; room temperature; 2 h | 93% |

## Harness 证据

- curator_step_count: 15
- source_detail_route_step_count: 15
- one_step_row_count: 15
- chain_probe_accepted: True

## Artifact refs

- source_detail_chain_audit: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_tet2025_pdf_fullroute_to_androstenedione_source_detail_20260607/source_detail_route_chain_audit.json`
- source_detail_curator_records: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_tet2025_pdf_fullroute_to_androstenedione_source_detail_20260607/source_detail_curator_records.json`
- compiled_downstream: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_tet2025_pdf_fullroute_to_androstenedione_source_detail_20260607/compiled_downstream_consumables.json`
- chemenzy_smoke_summary: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_tet2025_fullroute_15row_smoke_rerun_20260607/summary.json`
- chemenzy_verifier: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_tet2025_fullroute_15row_smoke_rerun_20260607/verifier.json`
- chain_tool_probe: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_harness_chain_tool_probe_20260607_terminal_fixed/source_detail_route_chain_audit.json`
