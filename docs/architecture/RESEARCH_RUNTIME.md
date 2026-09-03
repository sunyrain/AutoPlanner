# Research Runtime

更新：2026-07-26

`cascade_planner.research` 是当前、非权威的外部研究 worker 支撑层。它不是 legacy，
也不是 canonical V4 campaign runtime。

## 唯一入口

当前 open-structure 研究流程由下列脚本显式启动：

```bash
python scripts/run_open_structure_template_agent.py --help
```

普通 `python -m cascade_planner`、target solver、V4 Web 和后台 campaign 不会自动加载
`cascade_planner.research`。

## 模块所有权

| 模块 | 职责 |
| --- | --- |
| `open_research_contract` | 外部研究 JSON/artifact 合同与 fail-closed 校验 |
| `open_research_experience` | 有界经验摘要、manifest 与 fallback 审计 |
| `open_research_retrieval` | 非权威检索预取、checkpoint 与消费审计 |
| `open_research_seed_consumables` | 为下一轮研究生成本地下游 seed |
| `source_detail_resolution` | DOI/PMC/curator source-detail resolution pack |
| `source_material_locator` | 元数据级 publisher/SI/material locator |
| `downstream_compiler` | 将研究候选编译为非权威 downstream consumables |
| `source_detail_chain_builder` | source-detail route steps、curator records 与 hybrid audit |
| `route_failure_feedback` | 为后续研究生成结构化失败反馈，不改变 proof |
| `real_patent_procedure_gate` | 官方 patent XML procedure 的离线 replay/release gate |
| `autoplannrellm` | 显式开关控制的 DeepSeek route-tree 研究实验，不属于 V4 runtime |

## 权威边界

- research 输出只能形成候选、提取任务、worker result 或 replayable artifact；
- research 模块不能直接写 canonical graph、proof、completion、stock closure 或 production KB；
- source locator 和 retrieval metadata 不授予 exact-source authority；
- 只有 V4 worker ingestion 和 host validation 可以把结果投影到当前运行；
- legacy blackboard 可以读取 research helper，但 research 不依赖 legacy runtime。
- `autoplannrellm` 的旧根级包路径已删除，实验入口为
  `python -m cascade_planner.research.autoplannrellm.runner`；环境变量名称保持兼容，
  但该实验仍必须显式启用。
- Active prior and CLI code use `cascade_planner.agent.deepseek_credentials`
  for dependency-free key normalization. They do not import the research
  DeepSeek client; research-only controllers remain explicitly gated.

## 删除与演进

旧 `cascade_planner.harness.open_research_*`、`source_detail_*`、
`source_material_locator`、`downstream_compiler`、`route_failure_feedback` 和
`real_patent_procedure_gate` 路径已删除，不保留 shim。后续当前研究能力应继续进入
`cascade_planner.research` 或正式 `interfaces`，不得重新写回泛化 harness。

验证入口：

```bash
python -m pytest -q tests/test_research_namespace.py tests/test_open_research_experience.py
```
