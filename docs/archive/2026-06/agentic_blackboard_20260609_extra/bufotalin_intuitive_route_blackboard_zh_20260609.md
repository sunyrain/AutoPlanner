# Bufotalin 自主探索复盘：路线图与黑板变化

结论：v7 完整跑满预算，最终 `unresolved`，没有通过父路线证明。
PDF：`/root/autodl-tmp/AutoPlanner/1-s2.0-S0040402025001668-main.pdf`

## 15 步文献路线图
- 33 -> bufotalin：HF-pyridine；pyridine, THF；room temperature；93%
- 32 -> 33：Ac2O；pyridine；room temperature；90%
- 31 -> 32：NaBH4；MeOH, THF；0 °C；90%
- 30 -> 31：TMSOTf, 2,6-lutidine；DCM；-78 °C to room temperature；68%
- 22 -> 30：m-CPBA, Na2CO3；DCM；-78 °C to room temperature；70%
- 14 -> 22：Bu3Sn-substituted 2-pyrone 21, CuI, LiCl；DMSO, THF；60 °C；62%
- 20 -> 14：1) H2; 2) N2H4·H2O, Et3N; 3) I2, Et3N；THF, H2O, EtOH；room temperature then 50 °C then room temperature；73% over two steps
- 19 -> 20：SeO2, formic acid；dioxane/water；125 °C；52%
- 28 -> 19：TBSCl, imidazole；DMF；70 °C；74%
- 27 -> 28：p-TsOH；acetone；60 °C；56%
- 26 -> 27：t-BuOK；DMSO；70 °C；73%
- 23 -> 26：pyridinium perbromide；THF；0 °C to room temperature；87%
- 25 -> 23：K-selectride；THF；-5 °C；72%
- 24 -> 25：H2；4-methylpyridine；room temperature；92%
- 11 -> 24：ethylene glycol, p-TsOH；ethylene glycol；room temperature；93%

## 为什么失败
- v7 真实自主读图只稳定拿到 31 -> 32 -> 33 -> bufotalin 三步；30 到 11 被读图代理保守标为缺口。
- 路线搜索仍出现大重原子跳跃；后端 solved 标志被确定性审查拒绝。
- 子目标没有和父路线、完整文献段、库存闭合同时连通，所以不能宣称 solved。

## 和旧成功版对比
- 旧成功版：15 步文献链 accepted，终端 11 reached=True，拼接 solved=True。
- 本轮修复后复核：v6 读图结果可整理出 15 步，但这是离线复核，不是 v7 当轮最终证明。

验证：python -m pytest tests/test_agentic_blackboard_controller.py tests/test_failure_critic.py tests/test_parent_route_proof.py tests/test_open_research_experience.py tests/test_codex_entry_harness_contract.py -q -> 153 passed, 2 skipped
