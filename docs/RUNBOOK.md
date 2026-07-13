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

## 3. 科学案例重放

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

## 4. 运行管理与校验

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

## 5. 导出、Web 与存储维护

```bash
python -m cascade_planner export target-001 --output-dir local-export
python -m cascade_planner serve
python -m cascade_planner gc --dry-run --minimum-age-hours 24
python -m cascade_planner audit
```

Web 为 `/v4`，JSON API 为 `/api/v4/runs`。默认仅绑定 `127.0.0.1`。CLI 不提供隐式
删除模式；GC 只生成 dry-run 计划。

## 6. 本地发布门

```bash
python -m pytest -q
python -m ruff check cascade_planner tests scripts
python -m cascade_planner audit
git diff --check
git status --short
```

仓库不使用 CI/Action。Nirmatrelvir golden 必须闭环；Paclitaxel 必须明确显示未验证的
多路线；无本地 fixture 的目标必须具名失败，不能假成功。实际 Codex A/B 只有在显式
非零预算下运行，并记录调用、token、时间和 portfolio gain。
