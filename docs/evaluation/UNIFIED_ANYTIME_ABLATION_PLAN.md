# Unified Anytime W8 Ablation Plan

日期：2026-08-06

## 1. 精确 arm 定义

| Arm | Harness flags | 唯一组件差异 |
| --- | --- | --- |
| `chemenzy-only` | `--ablation chemenzy-only` → `--no-codex` | 不注册 Codex architecture/replan handlers |
| `codex-only` | `--ablation codex-only` → `--no-chemenzy --no-guided-chemenzy` | 不注册 ChemEnzy target/guided handlers |
| `unified-round-robin` | `--ablation unified-round-robin` → `--action-scheduler round_robin` | 用固定 kind cursor 替代 adaptive score ordering |
| `unified-adaptive` | `--ablation unified-adaptive` → `--action-scheduler adaptive` | 完整统一系统 |

上述 flags 不改变 target manifest、stock、acceptance、host validator、evidence authority、route portfolio 或报告器。

## 2. Round-robin 语义

Round-robin 仍使用同一个 canonical deficit frontier 和同一组 handlers：

- eligibility、dependency、resource availability 与 materialization-before-validation 约束保持有效；
- action kind 按冻结 `_KIND_ORDER` 循环；
- 同 kind 内使用 stable action ID；
- adaptive value score 仍记录但不参与 round-robin 排序；
- 不启动 adaptive 专属的 initial concurrent cohort。

因此该臂比较的是 scheduler ordering，而不是第二套 solver。

## 3. 能力臂语义

ChemEnzy-only 仍执行统一 host materialization、reaction validation、stock audit、conditions、evidence 和 Program host gates；仅 Codex proposal actions 不注册。

Codex-only 仍执行相同 host gates；仅 ChemEnzy target/guided native Actions 不注册。未使用的 native target reserve 必须显式释放，不得转化为额外模型预算。

## 4. 运行与汇总

完整四臂：

```text
python scripts/run_retrostar190_w8.py \
  --output-root results/.autoplanner/retrostar190-w8-20260806
```

断点续跑：

```text
python scripts/run_retrostar190_w8.py \
  --output-root results/.autoplanner/retrostar190-w8-20260806 \
  --resume
```

汇总：

```text
python scripts/summarize_retrostar190_w8.py \
  --w8-root results/.autoplanner/retrostar190-w8-20260806
```

汇总器生成：

- `w8-run-manifest.json`；
- `w8-panel-summaries.json`；
- `w8-per-target-metrics.json`；
- `w8-paired-comparison.json`；
- `w8-failure-taxonomy.json`。

Paired bootstrap 使用 evaluator-side 固定 seed `20260806` 和 5,000 replicates；该 seed 只影响置信区间计算，不影响 planner。
