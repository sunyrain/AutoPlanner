# V4 操作手册

## 1. 本地配置

默认运行根目录为 `results/.autoplanner`。可用环境变量或 CLI 全局参数覆盖：

- `AUTOPLANNER_RUNTIME_ROOT`
- `AUTOPLANNER_RUNS_ROOT`
- `AUTOPLANNER_ARTIFACT_STORE_ROOT`
- `AUTOPLANNER_RUN_INDEX_PATH`
- `AUTOPLANNER_CACHE_ROOT`
- `AUTOPLANNER_SOURCE_ROOT`
- `AUTOPLANNER_EXTERNAL_DATA_ROOT`
- `AUTOPLANNER_MODEL_ROOT`
- `AUTOPLANNER_VENDOR_ROOT`

凭据只允许通过环境或仓库外的显式路径提供。CLI 输出不会打印密钥。

## 2. 创建和注入全局路线

```bash
python -m cascade_planner run \
  --run-id target-001 \
  --target-name TARGET \
  --target-smiles SMILES
```

这只创建权威 run，模型调用为 0。全局路线可由人工、离线 golden 或后续有界 director 生成，然后注入：

```bash
python -m cascade_planner run \
  --run-id target-001 \
  --target-name TARGET \
  --target-smiles SMILES \
  --plan global_plan.json --materialize
```

同一个 run id 和同一计划可安全重试；idempotency key 由计划摘要和图 revision 绑定。

## 3. 运行管理

```bash
python -m cascade_planner list
python -m cascade_planner status target-001
python -m cascade_planner resume target-001 --materialize
python -m cascade_planner resume target-001 --closeout
```

`status` 分别显示 hypothesis、frontier、attempt、accepted expansion、model totals、proof portfolio 和 stop decision。

## 4. 校验与重放

```bash
python -m cascade_planner validate target-001
python -m cascade_planner replay target-001
python -m cascade_planner benchmark target-001 --iterations 3
```

`validate` 对比 event replay、snapshot、canonical graph full-recompute oracle 和 workbench digest。`replay` 必须在恢复前后产生相同状态摘要。`benchmark` 不访问模型或网络。

## 5. 导出与 Web

```bash
python -m cascade_planner export target-001 --output-dir local-export
python -m cascade_planner serve
```

Web 入口是 `/v4`，JSON API 是 `/api/v4/runs`。默认 Waitress 只绑定 `127.0.0.1`；对外暴露时必须放在认证反向代理后。开发时可使用 `--server flask`。

## 6. 存储维护

```bash
python -m cascade_planner gc --dry-run --minimum-age-hours 24
python -m cascade_planner audit
```

CLI 不提供隐式删除模式。GC 自动 pin 所有 pointer 和 RunIndex 已知 artifact，只输出候选计划。真正删除必须通过单独审阅的应用层调用并显式确认。

## 7. 本地发布门

```bash
python -m pytest -q
python -m ruff check cascade_planner tests scripts
python -m cascade_planner audit
git diff --check
git status --short
```

本仓库不使用 CI/Action。P10 还需运行 Nirmatrelvir、Paclitaxel、多个复杂目标和至少一个无本地 fixture 的模型免费 baseline；可选 Codex 增益必须另行记录调用、token、时间和 portfolio gain。
