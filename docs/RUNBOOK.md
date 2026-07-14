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
以及来源解析都不调用模型。专利证据按固定阶梯处理：官方 Google Patents 完整 HTML
→ 仅未闭合边的 PDF 原生文本 → 仅低文本页的 Tesseract OCR → 可选稀疏视觉候选。
HTML 字节、段落范围和规范化文字分别绑定哈希；搜索摘要不能成为 exact evidence。
Tesseract 是可选本地二进制：若不在 `PATH` 中，运行会记录
`local_ocr_engine_unavailable:tesseract`，不会暗中改用视觉模型。

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

### 2.2 不重复全局规划的验证 fork

```bash
python -m cascade_planner fork-validation target-blind-001 \
  --run-id target-blind-validation-001
```

该命令重放原始全局案卷，在当前 host 上重新运行结构物化、反应验证、来源获取、库存审计
和 self-evo 学习。默认模型与视觉预算都是 0；需要核对图片型来源时才显式准入：

```bash
python -m cascade_planner fork-validation target-blind-001 \
  --run-id target-blind-vision-001 \
  --max-visual-invocations 1 \
  --max-visual-pages 2 \
  --visual-reasoning-effort low
```

此时派生预算只覆盖一次视觉调用，不会再次调用 Global Director。视觉 observation 必须经过
目标 root、结构连续性、物化和 host admission；被接受也仍不会自动获得 exact-source 权威。

### 2.3 论文 HTML 与受控 PDF 获取

论文来源依次尝试 Europe PMC XML、公开 PMC HTML、PDF 和页面恢复。PMC 对普通 HTTP
客户端返回 reCAPTCHA shell 时，系统只对 `https://pmc.ncbi.nlm.nih.gov` 启动一次隔离、
无凭据、无用户 profile 的 Playwright context；最终跳转 host、HTTP 状态、字节上限和正文
DOI 都必须再次通过校验。挑战页和最终 HTML 分别绑定 SHA-256，之后可从内容缓存重放。

需要校园网或人工已登录浏览器才能获取的 PDF 仍进入显式本机队列，不会把 cookie 或凭据
传给服务端。可只处理当前 case/source，避免混入旧请求：

```bash
python scripts/browser_pdf_fetch.py \
  --output-dir PATH_TO_PROXY_QUEUE \
  --case-id CASE_ID \
  --source-ref DOI_OR_SOURCE_REF \
  --max-items 1
```

也可用 `--title-contains` 进一步过滤。每个请求使用新的 tab；下载结果仍须经过 PDF identity、
hash、页面选择和来源绑定，成功下载本身不授予反应证明。

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

本地开发环境没有安装 Waitress 时使用：

```bash
python -m cascade_planner serve --server flask --host 127.0.0.1 --port 8878
```

浏览器打开 `http://127.0.0.1:8878/v4`。表单会 `POST /api/v4/jobs`，页面每 2 秒轮询
`/api/v4/jobs` 与单任务进度；首次路线、ChemEnzy provider 调用、来源数、exact rows、视觉
调用和 token 分开显示。历史卡片是停止执行的不可变快照，内核原始状态只作审计，不能被
理解为仍有后台线程或已经达到 B3/L4。

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
