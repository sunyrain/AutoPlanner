# Program 创新与自进化完成切片（2026-07-16）

## 结论

本切片把此前的两项“建议能力”接成了可重放的代码路径，但没有伪造实验事实：

- 酶替代多步化学仍以边界明确的 `TransformationProgram` 表示，连续区间可压缩为一个物理 Program，原逐步 Program 永久保留为 fallback；只有精确绑定当前底物、产物、Program、innovation、条件和结果的专项验证，才允许进入既有生物催化影子 store。
- 文献外机理创新仍严格限制为一跳。候选必须从路线内文献锚点产物出发，并精确重接同一路线的下游状态；接不回完整路线的候选不会进入 Program optimizer。成功验证必须绑定 Program、innovation、输入/输出边界、机理签名、分析记录、竞争路径和全部可证伪检查。
- 通过上述门的一跳机理 Program 现在可显式写入独立的 append-only、CAS 绑定影子 store。写入、读取和恢复都会重新编译 bundle 与 oracle；锚点文献不被解释为报道或支持外推反应；生产 graph、reaction proof、completion、acceptance 和 `edge_ids[]` 均不改变。
- 已进入 append-only Experimental Claim store 的正例、负例、不确定和冲突结果，现在可以显式学习到外部、摘要绑定的 `program_experience_library.v1`。重复学习幂等，篡改失败关闭。
- 经验记忆只影响候选排序和验证优先级：精确边界正例最多加 0.12，精确负例最多减 0.18；结构类似转移还要求能力/机理策略一致、净 motif 与元素变化相同且分子相似度过门。正负冲突不加分并显示警示。任何经验匹配都不能代替当前候选的精确验证。

## 运行路径

```text
route + capability / one-hop mechanism proposal
                    │
                    ▼
       boundary discovery + full route restitch
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
 enzyme Program          mechanism Program
          │                   │
 exact domain validation + falsifiable checks
          │                   │
          ▼                   ▼
 independent append-only shadow stores
                    │
                    ▼
 exact-boundary Experimental Claim store
                    │ explicit learning gate
                    ▼
 external Program Experience Library
                    │ ranking/validation priority only
                    └──────────────► next route review
```

## 新的稳定入口

- `POST /api/v4/runs/<run_id>/programs/innovations`：统一只读审查，自动读取有效的 Program experience library。
- `POST /api/v4/runs/<run_id>/programs/innovations/mechanisms/admit`：显式准入已验证且完整重接的一跳机理 Program。
- `GET /api/v4/runs/<run_id>/programs/innovations/mechanisms/store`：重放机理影子 store。
- `POST /api/v4/runs/<run_id>/programs/innovations/claims/admit`：显式持久化精确边界实验 Claim。
- `POST /api/v4/runs/<run_id>/programs/innovations/experience/learn`：显式把当前 run 的可重放 Claim 学入跨任务经验库。
- `GET /api/v4/runs/<run_id>/programs/innovations/experience`：读取经验库及 generation/content digest。

恢复、`validate`、`replay` 和 artifact GC 已覆盖 baseline、biocatalytic、mechanism、experimental Claim 四类 store。HTTP 路由已从主 `v4_api.py` 拆到聚焦的 Program innovation registrar；Gateway 也拆为 Program migration 与 Program innovation 两个 mixin，避免继续膨胀主入口。

## 尚未完成、不能宣称的部分

- 当前 Bufotalin 6→1 酶候选仍缺真实精确底物验证，所以只能作为带 `EXACT_SUBSTRATE_UNVALIDATED` 的 proposal，不能接管生产路线。
- 代码可以生成和派发可证伪任务，但当前默认 provider 仍是人工交接，不代表真实设备或网络实验已经执行。
- Program experience 是受控先验，不是反应证明或自动 capability mutation；尚未实现以真实实验吞吐校准的概率模型和信息增益调度器。
- `program_ids[]` 尚未成为生产路线主语义；只有在多类别真实案卷与故障注入通过后，才能讨论从 `edge_ids[]` 切换。

## 本切片回归门

- 新增经验学习、冲突保留、摘要篡改失败关闭、机理影子准入幂等和 fallback 保留测试。
- V4 architecture import/line-budget 门覆盖新增 application、orchestration、interface 和 HTTP 模块。
- 最终全量测试数字以本轮提交前的实际门禁结果为准，不在文档中预填。
