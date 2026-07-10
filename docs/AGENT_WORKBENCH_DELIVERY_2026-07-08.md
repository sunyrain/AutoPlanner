# AutoPlanner Agent Workbench Delivery

## 入口

- 启动：`powershell -ExecutionPolicy Bypass -File scripts/start_agent_workbench.ps1`
- 页面：`http://127.0.0.1:7860/agent`
- 旧版 ChemEnzy UI：`http://127.0.0.1:7860/`

## 软件流程

1. 输入目标名称、family hint 和目标 SMILES。
2. 设置黑板轮数、ChemEnzy、文献检索、PDF/视觉和子目标展开预算。
3. 启动 Agent fullflow。
4. 左侧保持输入，中间实时显示黑板步骤、动作、日志和 artifact。
5. 右侧嵌入最终路线图：主画布只展示一条核心路线；点击步骤后可在检查器中切换备选，并沿备选分支重排后续片段。

## 当前验收样例

- `bufotalin_full_exact_stitch_rerun_20260622_073847`
  - 路线图：`results/shared/bufotalin_full_exact_stitch_rerun_20260622_073847/route_forest.html`
  - 结果：solved mixed route，4 条探索分支，20 步最终整合路线，41 个总步骤，包含 exact rows、视觉链和 ChemEnzy 子目标闭合。
  - 已验证：点击主线第 1 步备选后，主画布从 20 步切换为 5 步 ChemEnzy 子目标闭合分支，后续步骤随备选重排。

- `ui_agent_complex_atorvastatin_web_direct_chemenzy_20260707_174922_0ad499`
  - 路线图：`results/shared/ui_agent_runs/ui_agent_complex_atorvastatin_web_direct_chemenzy_20260707_174922_0ad499/route_forest.html`
  - 结果：30 个总步骤、12 条探索分支；包含 recommended Paal-Knorr 主线、ChemEnzy direct verified branch 和模型提案分支。

- `ui_agent_delivery_smoke_artemisinin_delivery_smoke_20260707_182250_6e2d65`
  - 路线图：`results/shared/ui_agent_runs/ui_agent_delivery_smoke_artemisinin_delivery_smoke_20260707_182250_6e2d65/route_forest.html`
  - 结果：从 Web Agent 入口重新启动的轻量端到端 smoke；任务完成，黑板步骤 5 条，生成 route forest。该结果用于证明软件链路，不作为复杂路线质量样例。

- `ui_agent_delivery_smoke_paclitaxel_delivery_smoke_20260707_182713_9f9839`
  - 路线图：`results/shared/ui_agent_runs/ui_agent_delivery_smoke_paclitaxel_delivery_smoke_20260707_182713_9f9839/route_forest.html`
  - 结果：从 Web Agent 入口重新启动的复杂代表分子 smoke；任务完成，黑板步骤 5 条，生成 14 个总步骤、11 条探索分支，核心展示为 10-DAB / baccatin III 半合成 paclitaxel 推荐路线。

## 已知输入注意

- 当前工作台要求目标 SMILES 可被 RDKit 解析。paclitaxel 样例已替换为 PubChem 返回且 RDKit 可解析的 isomeric SMILES；后续可继续增加名称到结构解析。

## 验证

- `python -m py_compile cascade_planner/web/app.py cascade_planner/harness/route_forest.py`
- `python -m pytest tests/test_route_forest.py tests/test_parent_route_proof.py`
- `python -m pytest tests/test_web_app.py`
- `python scripts/validate_agent_workbench_delivery.py`

截图保存在：`results/shared/ui_delivery_screenshots_20260708/`
