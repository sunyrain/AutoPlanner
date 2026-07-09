# Retro Enzyme 化学约束模型评估报告

生成时间：2026-07-09

## 评估对象

当前推荐模型为 `Retro Enzyme（化学约束 rerank 版本）`。

- AutoPlanner 接入模块：`cascade_planner/baselines/enzretro_onestep.py`
- 模型 checkpoint：`/root/autodl-tmp/auto planner-1/enzretro_model_code_package/outputs/enzretro_ft_filtered_strict_singleprod_v1/best_model.pt`
- 推理设置：beam search `beam=5`，返回 `top5`
- 启用设置：`--require-execute --lipid-filter --chemistry-constraints --dedupe-substrate`
- 说明：Retro Enzyme 是本项目中对 EnzRetro/SSREdits 酶逆合成生成模型的命名与 AutoPlanner 封装。

## 指标定义

### ECREACT / RetroEnzyme Test

ECREACT 测试集使用 EnzRetro 论文/官方代码复现口径的指标；这一部分评估完整底物集合、SSREdits 和 EC 号，不使用 AutoPlanner benchmark 中的“主要底物”口径：

- `Substrate exact`：执行预测 SSREdits 后得到的完整底物集合与 gold 底物集合完全一致。
- `SSREdits exact`：预测编辑序列与 gold 编辑序列完全一致。
- `EC exact`：完整 EC 号完全一致。
- `Execute rate`：预测 SSREdits 可以成功执行为底物的比例。

### AutoPlanner Enzyme Benchmark

AutoPlanner benchmark 只汇报以下指标：

- `主要底物 exact`：预测候选包含 benchmark 标注的核心/主要底物。
- `EC1 exact`：EC 第一位完全一致。
- `EC2 exact`：EC 前两位完全一致。
- `主要底物 + EC 正确`：同一个候选中主要底物正确，且完整 EC 号正确。

这里的 `主要底物 + EC 正确` 不要求完整反应物集合完全一致，因此不因 NAD/NADP/ATP/H2O/H+/NH3 等辅因子是否出现而扣分。

ChemEnzy/OpenNMT 原有模型不输出 EC 号，因此 EC1、EC2 和 `主要底物 + EC 正确` 记为 N/A。

## ECREACT / RetroEnzyme Test

test size：`2559`

| 模型 | Substrate exact | SSREdits exact | EC exact | Execute rate |
|---|---:|---:|---:|---:|
| EnzRetro 官方 checkpoint | 58.23% | 66.55% | 73.43% | 95.35% |
| Retro Enzyme 当前模型 | 56.19% | 63.15% | 72.45% | 94.69% |

## AutoPlanner Enzyme Benchmark

benchmark 来源：`AutoPlanner/data/benchmark_v2_100.json` 中抽取的酶催化步骤。

有效酶步骤数：`141`

### Top-1

| 模型 | 主要底物 exact | EC1 exact | EC2 exact | 主要底物 + EC 正确 |
|---|---:|---:|---:|---:|
| AutoPlanner 原有 ChemEnzy/OpenNMT | 3.55% | N/A | N/A | N/A |
| Retro Enzyme + 化学约束 | 15.60% | 49.65% | 27.66% | 1.42% |

### Top-5

| 模型 | 主要底物 exact | EC1 exact | EC2 exact | 主要底物 + EC 正确 |
|---|---:|---:|---:|---:|
| AutoPlanner 原有 ChemEnzy/OpenNMT | 12.06% | N/A | N/A | N/A |
| Retro Enzyme + 化学约束 | 17.73% | 52.48% | 29.79% | 1.42% |

## 结论

1. 在 ECREACT / RetroEnzyme 同分布测试集上，Retro Enzyme 当前模型接近官方 EnzRetro checkpoint：Substrate exact 为 `56.19%`，EC exact 为 `72.45%`。

2. 在 AutoPlanner enzyme benchmark 上，Retro Enzyme 的主要优势是核心底物识别和 EC 号输出：
   - 主要底物 top1 exact 从 ChemEnzy/OpenNMT 的 `3.55%` 提高到 `15.60%`。
   - 主要底物 top5 exact 从 ChemEnzy/OpenNMT 的 `12.06%` 提高到 `17.73%`。
   - Retro Enzyme 能输出 EC1/EC2/完整 EC，ChemEnzy/OpenNMT 当前 one-step baseline 不输出 EC。

3. 化学约束主要作用在候选 rerank 阶段：先要求 SSREdits 能执行得到合法底物，再综合辅因子识别、骨架相似度、元素/重原子变化、脂质链异常片段和 EC 大类-编辑类型一致性，对 beam search 候选重新排序。具体包括：
   - 过滤或惩罚不可执行的 SSREdits 候选；
   - 对长脂质链、异常大分子片段等 artifact 加惩罚；
   - 识别 NAD/NADP/ATP/磷酸等常见辅因子或辅助物，避免它们主导主要底物选择；
   - 依据产物与候选主底物的骨架相似度、重原子变化和元素守恒进行加权；
   - 检查 EC 大类与 SSREdits 编辑类型是否一致，例如氧化还原、转移、水解、异构化等。

4. 本报告中的 AutoPlanner `主要底物 + EC 正确` 不考虑辅因子是否与 benchmark 完全一致，更符合路线规划中“找到核心前体并给出酶类别”的目标。

## 模型论文引用

Retro Enzyme 当前实现基于 EnzRetro 方法。APA 格式引用如下：

Cao, Y., Chen, H., Zhang, T., Zhao, X., Li, B., & Zheng, X. (2026). EnzRetro: Enzymatic retrosynthetic planning with site-specific reaction edits based on sequence generative architecture. *Exploration, 6*, 70129. https://doi.org/10.1002/exp2.70129

