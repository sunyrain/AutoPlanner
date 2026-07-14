# V4 操作手册

## 1. 本地配置

默认运行根目录为 `results/.autoplanner`。可使用以下环境变量或等价的 CLI 全局参数：

- `AUTOPLANNER_RUNTIME_ROOT`
- `AUTOPLANNER_RUNS_ROOT`
- `AUTOPLANNER_ARTIFACT_STORE_ROOT`
- `AUTOPLANNER_RUN_INDEX_PATH`
- `AUTOPLANNER_CACHE_ROOT`
- `AUTOPLANNER_SOURCE_ROOT`
- `AUTOPLANNER_EXTERNAL_DATA_ROOT`
- `AUTOPLANNER_MODEL_ROOT`
- `AUTOPLANNER_VENDOR_ROOT`

凭据只通过环境或仓库外显式路径提供。全局 CLI 参数必须写在子命令之前。

## 2. 创建和运行

```bash
python -m cascade_planner run \
  --run-id target-001 \
  --target-name TARGET \
  --target-smiles SMILES
```

这只创建权威 run，模型/视觉调用为 0。注入审阅后的 `global_campaign_plan.v1`：

```bash
python -m cascade_planner run \
  --run-id target-001 \
  --target-name TARGET \
  --target-smiles SMILES \
  --plan global_plan.json --materialize --closeout
```

同一 run id 和同一计划可安全重试。

### 2.1 任意陌生 SMILES 的有界求解

```bash
python -m cascade_planner solve-target \
  --target-name TARGET \
  --target-smiles SMILES \
  --run-id target-blind-001
```

正常成本是一轮 `gpt-5.5` low-reasoning 全局路线组合；本地物化、映射、验证、库存审计
以及 PDF 原生文本/Tesseract OCR 都不调用模型。Tesseract 是可选本地二进制：若不在
`PATH` 中，运行会记录 `local_ocr_engine_unavailable:tesseract`，不会暗中改用视觉模型。

仅当来源页确实无法由确定性解析闭合时，可显式允许一次稀疏视觉调用：

```bash
python -m cascade_planner solve-target \
  --target-name TARGET \
  --target-smiles SMILES \
  --run-id target-blind-visual-001 \
  --max-model-invocations 3 \
  --max-visual-invocations 1 \
  --max-visual-pages 2
```

三次模型额度分别覆盖初始全局规划、至多一次页面视觉候选、以及有新证据事件时的
一次全局 replan；若只给两次，系统会如实跳过 replan，而不是突破预算。视觉候选经过
主机 SMILES 规范化，但仍是 L0：它不能授予反应验证、exact-source 或库存权威。
同一 campaign 即使 provider 少报用量，也只会持久化准入一次视觉任务。

## 3. 从精确来源案卷一键求解

```bash
python -m cascade_planner solve-case \
  --dossier config/examples/artemisinin_v4_case_dossier.json \
  --run-id artemisinin-showcase \
  --output-dir local-showcase/artemisinin
```

该命令依次编译案卷、进入 canonical hypergraph、执行来源/反应/库存 worker、拼接
proof portfolio，并导出离线 workbench。输出带分阶段墙钟时间；默认模型和视觉调用均为 0。

需要单独审阅生成的可移植重放包时：

```bash
python -m cascade_planner compile-case \
  --dossier config/examples/artemisinin_v4_case_dossier.json \
  --output local-showcase/artemisinin-pack.json
```

案卷必须包含：少量全局路线族、原子映射反应、覆盖每条边的精确来源记录，以及覆盖
每个深层叶节点的带时间戳库存 offer。缺一项即失败关闭。`--map-missing` 只会调用本机
已安装的 RXNMapper，不会访问 hosted model 或网络。案卷负责输入事实的人工/上游审阅；
编译器不会把摘要哈希误当成化学真实性，也不会把目录可订购性表述为实时本地库存。

## 4. 科学案例重放

```bash
python -m cascade_planner replay-case \
  --pack config/examples/nirmatrelvir_v4_replay_pack.json \
  --run-id nirmatrelvir-golden
```

测试中断恢复：

```bash
python -m cascade_planner replay-case \
  --pack config/examples/nirmatrelvir_v4_replay_pack.json \
  --run-id nirmatrelvir-recovery \
  --stop-after evidence

python -m cascade_planner replay-case \
  --pack config/examples/nirmatrelvir_v4_replay_pack.json \
  --run-id nirmatrelvir-recovery
```

可暂停阶段为 `plan`、`materialization`、`evidence`、`validation`、`stock`。
重放包必须通过内容哈希、来源 artifact、反应身份/验证和库存 schema 校验。

## 5. 运行管理与校验

```bash
python -m cascade_planner list
python -m cascade_planner status target-001
python -m cascade_planner resume target-001 --materialize
python -m cascade_planner resume target-001 --closeout
python -m cascade_planner validate target-001
python -m cascade_planner replay target-001
python -m cascade_planner benchmark target-001 --iterations 3
```

`validate` 比较事件重放、snapshot、规范图 full-recompute oracle 和 workbench 绑定。
`benchmark` 不访问模型或网络。

## 6. 导出、Web 与存储维护

```bash
python -m cascade_planner export target-001 --output-dir local-export
python -m cascade_planner serve
python -m cascade_planner gc --dry-run --minimum-age-hours 24
python -m cascade_planner audit
```

Web 为 `/v4`，JSON API 为 `/api/v4/runs`。默认仅绑定 `127.0.0.1`。CLI 不提供隐式
删除模式；GC 只生成 dry-run 计划。

## 7. 本地发布门

```bash
python -m pytest -q
python -m ruff check cascade_planner tests scripts
python -m cascade_planner audit
git diff --check
git status --short
```

仓库不使用 CI/Action。Nirmatrelvir golden 与 Artemisinin dossier 必须闭环；Paclitaxel 必须明确显示未验证的
多路线；无本地 fixture 的目标必须具名失败，不能假成功。实际 Codex A/B 只有在显式
非零预算下运行，并记录调用、token、时间和 portfolio gain。
