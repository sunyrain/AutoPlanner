# AutoPlanner

AutoPlanner V4 是一个由 Codex 做全局路线规划、由确定性 worker 做事实升级的逆合成系统。它不把 Codex 限制成单步反应预测器：全局 director 一次查看目标、候选路线族、共享中间体、证据缺口、库存边界、冲突和预算，再决定整个 campaign 的下一组工作。

系统只有四个状态权威：

1. `RunKernel`：事件、恢复和全运行预算；
2. canonical reaction hypergraph：分子、反应超边和路线拓扑；
3. deficit frontier：所有待办工作，确定性缺口优先于模型提议；
4. proof portfolio：反应证据、库存闭合、路线多样性和最终完成判定。

Blackboard 只允许作为兼容投影，不再保存第二套 expansion、proof 或 stock 状态。Codex 提议不能直接宣布成功；候选必须依次经过结构物化、反应验证、精确来源绑定和叶节点库存审计。

## 快速开始

安装依赖后，用唯一主入口查看所有命令：

```bash
python -m cascade_planner --help
```

创建一个不调用模型的新运行：

```bash
python -m cascade_planner run \
  --run-id aspirin-demo \
  --target-name aspirin \
  --target-smiles 'CC(=O)Oc1ccccc1C(=O)O'
```

如果已有经过审阅的 `global_campaign_plan.v1`，可在同一次运行中注入并物化：

```bash
python -m cascade_planner run \
  --run-id aspirin-demo \
  --target-name aspirin \
  --target-smiles 'CC(=O)Oc1ccccc1C(=O)O' \
  --plan plan.json --materialize
```

检查、确定性重放和导出：

```bash
python -m cascade_planner status aspirin-demo
python -m cascade_planner validate aspirin-demo
python -m cascade_planner replay aspirin-demo
python -m cascade_planner benchmark aspirin-demo --iterations 3
python -m cascade_planner export aspirin-demo --output-dir local-export
python -m cascade_planner gc --dry-run
```

`run` 的模型和视觉调用预算固定为 0。P10 的可选 Codex campaign 必须显式配置 runner 和硬预算；任何 CLI 基线、校验、重放、导出或 GC 都不会偷偷访问网络或模型。

启动 WebUI：

```bash
python -m cascade_planner serve
```

打开 `http://127.0.0.1:7860/v4`。CLI、V4 API 和 Web workbench 都调用同一个 `CampaignGateway` 和 `RetrosynthesisCampaignService`，因此不会再出现“命令行已扩展、网页仍读旧黑板”的分叉状态。

## 可信度与完成定义

界面严格区分：

- L0：Codex/ChemEnzy/模板的断键假设；
- L1：结构和元素守恒通过、反应超边已物化；
- L2：当前主机确定性反应验证通过；
- L3：精确反应步骤绑定可信来源；
- L4：所选路线的反应证据和全部叶节点库存边界均闭合。

颜色只显示已经存在的 proof，不会赋予 proof。路线分支数、Agent 返回成功、预算耗尽和“没有更多任务”都不等于逆合成完成。

## 仓库边界

- 源码：`cascade_planner/`
- 本地测试：`tests/`
- 小型版本化夹具：`data/`、`config/examples/`
- 外部语料/模型/vendor：Git 忽略，由环境变量配置
- 运行、CAS、缓存和导出：默认位于 `results/.autoplanner/`，Git 忽略
- 历史报告：不再复制到当前树，可从 Git 历史读取

仓库没有 GitHub Actions。提交前在本地运行完整测试、Ruff、仓库审计和 `git diff --check`。

更多说明见 [文档入口](docs/README.md)、[主线架构](docs/MAINLINE.md)、[操作手册](docs/RUNBOOK.md) 和 [V4 实现清单](docs/architecture/RETROSYNTHESIS_V4_IMPLEMENTATION_TODO.md)。
