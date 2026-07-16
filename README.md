# AutoPlanner V4

AutoPlanner V4 是一个“全局规划、确定性执行、最弱环节验收”的逆合成系统。
Codex 的职责是纵览整个 campaign：同时设计少量不同路线族、多步骨架、共享中间体、
证据策略和失败后的全局转向。它不是被逐边循环调用的单步反应预测器。

化学事实由主机侧逐级建立：

```text
Codex 全局路线组合（proposal only）
                 ↓
结构物化 → 反应验证 → 精确来源绑定 → 全部叶节点库存审计
                 ↓
      小型多样路线组合 + 硬验收合同
```

系统只有四个状态权威：

1. `RunKernel`：事件、恢复、任务和全运行预算；
2. canonical hypergraph：分子、反应超边、来源、库存和路线拓扑；
3. deficit frontier：唯一待办队列；
4. proof portfolio：路线选择、最弱边/叶证明和完成判定。

Blackboard 只作为旧运行的兼容投影，不能再写入 V4 化学状态。Codex、ChemEnzy、
模板、文献和人工候选都必须进入同一规范化入口。

## 快速开始

查看唯一命令入口：

```bash
python -m cascade_planner --help
```

创建零模型调用的运行：

```bash
python -m cascade_planner run \
  --run-id aspirin-demo \
  --target-name aspirin \
  --target-smiles 'CC(=O)Oc1ccccc1C(=O)O'
```

从任意陌生 SMILES 启动一次有硬预算的全局 Codex campaign：

```bash
python -m cascade_planner solve-target \
  --target-name TARGET \
  --target-smiles SMILES \
  --run-id target-blind-001
```

默认只进行全局文本规划和确定性证据处理，视觉调用为 0。内置 patent connector
先冻结官方 Google Patents 完整 HTML，再从哈希绑定的段落范围确定性重建产物和反应物；
HTML 已闭合的边不会下载或渲染 PDF。只有未闭合边才依次回退到 PDF 原生文本、本机
Tesseract OCR，最后才可用
`--max-visual-invocations 1 --max-model-invocations 3` 显式准入一次稀疏页面视觉调用。
Europe PMC/PMC 论文通道同样可把哈希冻结的操作性 HTML 段落编译为来源路线，并经独立
名称结构解析、原子盘点和 host validation 生成 exact row；这一路径默认模型/视觉调用均为 0。

通过专利 exact row 且当前 host 反应验证的反应会自动进入外部
`data_external/self-evo/patent-reaction-template-library.json`。模板先在原始反应上重放，
之后只作为下一批陌生 SMILES 的全局路线选项和 L0 前沿候选；所有复用边仍须重新物化、
映射和验证。该链路不增加模型调用，可用 `--no-patent-self-evo` 禁用，或用
`--self-evo-library PATH` 冻结 benchmark 的模板库快照。
视觉结果只回到下一次全局 replan 作为 L0 候选，不能直接获得 L2/L3 或库存证明。

已结束的 blind campaign 可在不重复全局规划的情况下重放最新来源、验证和库存策略：

```bash
python -m cascade_planner fork-validation target-blind-001 \
  --run-id target-blind-validation-001
```

validation fork 默认仍为零模型；只有显式加入
`--max-visual-invocations 1 --max-visual-pages 2` 才允许一次稀疏页面视觉调用。公开 PMC
正文若对普通 HTTP 客户端返回 reCAPTCHA，系统会在无凭据、无用户 profile 的隔离浏览器中
重取该 PMC 页面，并再次校验最终 host、DOI、大小和内容哈希；它不会借用校园登录态。

注入已经审阅的全局路线计划并物化：

```bash
python -m cascade_planner run \
  --run-id target-001 \
  --target-name TARGET \
  --target-smiles SMILES \
  --plan global_plan.json --materialize --closeout
```

从空运行目录重放 Nirmatrelvir 科学验收案例：

```bash
python -m cascade_planner replay-case \
  --pack config/examples/nirmatrelvir_v4_replay_pack.json \
  --run-id nirmatrelvir-golden
```

该案例重建 2 条完整路线、12 条规范超边、15 条精确来源记录和 7 个库存叶，
全程 0 次模型/视觉调用。可用 `--stop-after evidence` 验证暂停恢复。

从小型精确来源案卷一键完成 Artemisinin 编译、验收和离线展示：

```bash
python -m cascade_planner solve-case \
  --dossier config/examples/artemisinin_v4_case_dossier.json \
  --run-id artemisinin-showcase \
  --output-dir local-showcase/artemisinin
```

该案例保留“从青蒿酸开始”和“直接采购二氢青蒿酸开始”两种采购边界，闭合 2 条路线、
2 条验证超边、3 条精确来源记录和 4 个库存叶，并输出分阶段耗时。案卷入口适合接收
Codex 的全局多步路线组合；它不是逐边调用 Codex，也不会在没有来源事实或 proposal
provider 时伪装成任意未见分子的自动发现器。

启动 Web：

```bash
python -m cascade_planner serve
```

默认 `--server auto` 会优先使用 Waitress；当前环境未安装 Waitress 时自动回退到 Flask，不会因可选
服务器缺失而使页面无法启动。

打开 `http://127.0.0.1:7860/v4` 进入唯一主页面：从任意 SMILES 发起逆合成 campaign、
查看实时和历史运行、打开路线 Workbench，以及切换展示案例和 benchmark 审计，均在同一页面
完成。旧的 `/v4/console`、`/v4/showcase`、`/agent`、`/statins` 和 `/showcase` 仅作兼容
重定向。CLI、V4 API 和 Web workbench 共用 `CampaignGateway` 与
`RetrosynthesisCampaignService`。页面每 2 秒读取 canonical checkpoint，并分别显示路线
结构闭合、证据待补、模型/视觉调用和历史快照；历史内核记录为 `running` 不代表进程仍在执行。

## 可信度与完成

- L0：断键或路线假设；
- L1：规范结构超边已物化；
- L2：当前主机确定性反应验证通过；
- L3：反应验证与精确可信来源均已绑定；
- L4：所选路线的全部边和全部叶节点达到配置的证明/库存边界。

分支很多、Agent 返回成功、预算耗尽、队列暂时为空都不等于完成。颜色只展示已有
proof，不能赋予 proof。

## 仓库边界

- 源码：`cascade_planner/`
- 本地测试：`tests/`
- 小型版本化夹具：`config/examples/`、`data/`
- 运行、CAS、缓存与导出：默认 `results/.autoplanner/`，Git 忽略
- 外部语料、模型、vendor 数据、PDF：仓库外配置，Git 忽略

本仓库不使用 GitHub Actions 或 CI。提交前在本地运行完整测试、Ruff、仓库审计和
`git diff --check`。

进一步阅读：[主线架构](docs/MAINLINE.md)、[操作手册](docs/RUNBOOK.md)、
[Schema 索引](docs/SCHEMAS.md)、[展示案例](docs/SHOWCASE_CASES.md) 和
[V4 实施清单](docs/architecture/RETROSYNTHESIS_V4_IMPLEMENTATION_TODO.md)。
