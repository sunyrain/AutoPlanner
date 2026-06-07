# Bufotalin 当前全流程运行报告

- 生成时间: 2026-06-07T16:31:08.928336+00:00
- run_dir: `/root/autodl-tmp/AutoPlanner/results/shared/bufotalin_canonical_real_tools_20260607_154411`
- final verdict: `fake_closed_rejected`
- solved: `False`

## 一个 SMILES 进入 AutoPlanner 后经历什么

1. 结构预检：解析 SMILES，生成 canonical/isomeric SMILES、InChIKey、formula、rings、stereocenters 和风险标记。
2. 工作流计划：controller/Codex 只规划工具顺序；计划必须通过 schema、strategy、run_semantics 和 raw-reaction guard。
3. Native ChemEnzy：真实调用 ChemEnzy 作为 route generator；raw_solved 不等于 solved。
4. Route verifier/audit：独立检查 target identity、stock closure、hidden non-stock、large atom jump、advanced same-scaffold terminal。
5. Frontier 与 literature gate：只有 route unresolved/fake/advanced 或合法 literature-first 理由才进入文献/source-detail。
6. Open research：先读 manifest、prefetch、source-detail pack，记录 agent_access_status/content_scope，再排本地 PDF fallback。
7. Downstream compiler：把可靠证据编译成 guided rerun、template card、route expansion task、self-evo staging candidate。
8. Guided rerun / route expansion：再次真实调用 ChemEnzy 或子目标搜索，但仍必须过 verifier。
9. Final verdict：只有 deterministic artifact bundle 和 verifier 能给 solved/fake_closed/partial/unresolved 结论。

## Bufotalin 本次真实运行结论

- Native ChemEnzy: elapsed `288.13` s, raw route count `179`, verifier accepted `0`.
- Native verifier reasons: `advanced_same_scaffold_terminal, large_atom_jump, no_verifier_accepted_stock_closed_route`.
- Open research: `partial_handoff_ready`, sources `30`, PDF requests `3`.
- Guided ChemEnzy: elapsed `284.494` s, reasons `advanced_same_scaffold_terminal, large_atom_jump, no_verifier_accepted_stock_closed_route`.
- Route expansion: status `solved`, accepted subgoals `1` / `2`.
- Final: `fake_closed_rejected`, reasons `advanced_same_scaffold_terminal, downstream_consumables_missing_evidence_refs:route_expansion_tasks:1, fake_closure_evidence_present, large_atom_jump, no_verifier_accepted_stock_closed_route, open_agent_boundary_violation:context_boundary:large_raw_artifact_dump, open_structure_research_nonzero_exit, raw_reaction_injection, route_verifier_rejected_raw_routes`.

## PDF / Source Access

- 已下载本地 PDF: `/root/autodl-tmp/AutoPlanner/1-s2.0-S0040402025001668-main.pdf`, exists=`True`.
- queued PDF fallback: `10.1016/j.tet.2025.134610` scope=`article` status=`queued`
- queued PDF fallback: `10.1016/j.steroids.2024.109555` scope=`article` status=`queued`
- queued PDF fallback: `10.1021/jo00934a013` scope=`article` status=`queued`
