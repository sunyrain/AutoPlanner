# AutoPlanner Mainline

Last update: 2026-06-04.

本文件是仓库当前唯一主线入口。其它计划、审计、报告和 checklist 只在本文件引用时
作为支撑材料使用；旧的阶段报告、临时 TODO 和 benchmark 讨论均归入
`docs/archive/` 的 provenance。

## Main Conclusion

AutoPlanner-Cascade 当前不是旧的 v4 learned-value / CCTS / route-ranker
训练主线。当前主线是：

```text
ChemEnzy native multi-step planning
-> conservative material and route audit
-> SMILES-first literature mode when native search is weak
-> evidence cards and strategic disconnection candidates
-> deterministic retron/applicability/reconstruction gates
-> validated literature one-step proposal source
-> ChemEnzy rerun / route package / audit output
```

硬边界：

- No LLM in the ChemEnzy inner loop.
- No online LLM rerank.
- No online LLM proposal judge.
- No raw LLM reaction injection.
- No solved-route claim from stock hit, EC annotation, planner score,
  condition prediction, or literature summary alone.

## Authority Order

1. P0 execution:
   `docs/SMILES_FIRST_LITERATURE_STRATEGIC_WORKFLOW_2026-06-03.md`
2. P0 engineering checklist:
   `docs/EvoChemEnzy_Code_Delivery_Checklist_2026-06-03.md`
3. Literature-to-ChemEnzy bridge:
   `docs/LITERATURE_TO_EXECUTABLE_TEMPLATE_CHECKLIST_2026-06-04.md`
4. Long-term architecture:
   `docs/EvoChemEnzy_Agentic_CASP_Plan_2026-06-03.md`
5. Risk and feasibility constraints:
   `docs/EvoChemEnzy_Plan_Feasibility_Audit_2026-06-04.md`
6. Repository boundary:
   `docs/CODEBASE_HYGIENE_AUDIT_2026-06-04.md`
7. Historical provenance:
   `docs/archive/`

If these documents conflict, use the first matching item in the authority order.

## Active Code Surface

Primary implementation anchors:

- `cascade_planner/agent/smiles_first.py`
- `cascade_planner/agent/evidence_cards.py`
- `cascade_planner/agent/literature_research.py`
- `cascade_planner/agent/strategic_candidate_generation.py`
- `cascade_planner/agent/route_package.py`
- `cascade_planner/agent/literature_templates.py`
- `cascade_planner/agent/template_applicability.py`
- `cascade_planner/agent/executable_template_validation.py`
- `cascade_planner/agent/route_auditor.py`
- `cascade_planner/agent/chem_enzy_policy.py`
- `cascade_planner/baselines/chem_enzy_adapter.py`
- `cascade_planner/baselines/chem_enzy_onestep.py`
- `cascade_planner/baselines/literature_one_step_plugin.py`
- `cascade_planner/baselines/chem_enzy_native_chemical_plugin.py`
- `cascade_planner/baselines/chem_enzy_native_enzyme_plugin.py`
- `scripts/run_smiles_first_literature_workflow.py`
- `scripts/run_literature_template_plugin_benchmark.py`
- `scripts/run_chem_enzy_plan_for_web.py`

Strategic-disconnection curated sources are under:

```text
data/strategic_disconnections/
```

They are small, active, and used by the SMILES-first literature path. Keep them
versioned.

## Legacy Surface

These paths remain for compatibility, old reports, or research reproduction,
but they are not the default next implementation path unless this file or an
active checklist promotes them:

- older CCTS and v4 learned-value modules;
- old route-pool ranker / LambdaRank / reservoir lineage;
- strict-review and route-block value training scripts;
- most standalone benchmark and replay scripts under `cascade_planner/eval/`;
- large local bridge/verifier packs under `data/bridge_pack_v0*` and
  `data/enzyme_sp_verifier_v1/`.

Do not delete historical code only because it is not the current mainline. Move
or remove it only with a separate migration manifest and import/test checks.

## Repository Boundary

Commit:

- source code under `cascade_planner/`, `scripts/`, and `tests/`;
- active docs under `docs/`;
- small curated data under `data/`;
- static showcase assets under `docs/statins/` and `docs/bufotalin/`;
- example env files such as `.env.local.example`.

Keep local-only:

- `.env`, `.env.*` except documented examples;
- `.claude/`;
- `config/`;
- `AI_OS_AutoResearch/`;
- `vendor/ChemEnzyRetroPlanner/`;
- `data_external/`;
- `data/bridge_pack_v0*/`;
- `data/enzyme_sp_verifier_v1/`;
- `results/shared/`;
- generated `results/v2/` reports and route render outputs;
- root-level `.mar`, `.pdf`, `.pptx`, checkpoints, model weights, and run logs.

Root-level reference materials belong in:

```text
docs/archive/2026-06/reference_materials/
```

## Minimum Verification

Before pushing mainline edits, run at least:

```bash
pytest -q tests/test_smiles_first_workflow.py \
  tests/test_literature_evidence_cards.py \
  tests/test_strategic_candidate_generation.py \
  tests/test_route_plausibility.py

pytest -q tests/test_literature_template_cards.py \
  tests/test_template_applicability.py \
  tests/test_executable_template_validation.py \
  tests/test_literature_one_step_plugin.py \
  tests/test_literature_template_plugin_benchmark.py

python -m py_compile scripts/run_smiles_first_literature_workflow.py \
  scripts/run_literature_template_plugin_benchmark.py \
  cascade_planner/agent/smiles_first.py \
  cascade_planner/agent/evidence_cards.py \
  cascade_planner/agent/literature_research.py \
  cascade_planner/agent/strategic_candidate_generation.py \
  cascade_planner/agent/route_package.py \
  cascade_planner/agent/literature_templates.py \
  cascade_planner/agent/template_applicability.py \
  cascade_planner/agent/executable_template_validation.py \
  cascade_planner/baselines/literature_one_step_plugin.py
```

Full test discovery is still useful, but some historical tests exercise legacy
expectations and should be interpreted against the authority order above.
