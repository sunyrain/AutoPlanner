# SynthEx 复现流程：组件职责与数据合同

本文档描述 paper-matched SynthEx 流程的当前真实实现。它是维护时的边界清单：每个组件只做自己的工作；模型声明不是结构、库存或 solved 事实；任何新增状态都必须先在本文中找到唯一归属。

## 1. 当前阶段顺序

```text
Target
  -> Strategy Generator（一次生成 3 个 Strategy）
  -> 每个 Strategy 独立进入 AiZ MCTS
       -> AiZ 选择一个当前节点
       -> Route Builder 给出一个 ReactionJSON expansion
       -> Host 编译并回放 ReactionJSON
       -> AiZ 接收真实前体、做树搜索与精确库存终止
  -> 对可回放战略路线运行 Critic -> Editor -> Critic 循环
  -> Host 将可用路线投影到 canonical graph
  -> Host 对目标可达开放叶做精确库存审计
  -> 合格的库存未闭叶进入 AiZ short-tail
  -> Host 校验、拼接、物化并计算最终库存闭合/solved
  -> Analyst（论文有；当前尚未实现）
```

当前 Critic/Editor 位于战略 Route Builder 之后、全局 canonical ingestion 和 short-tail 之前。Critic/Editor 的判断是模型内部一致性审查，不是实验验证，也不拥有 solved 权限。

## 2. 全局不可违反的不变量

1. 分子结构由 Host 对 ReactionJSON 回放得到。Strategy、Builder、Critic、Editor 都不能直接把自写 SMILES 变成结构事实。
2. `precursor_smiles` 和 `mapped_precursor_smiles` 的唯一结构权威是 Host compiler/replay。
3. 库存由绑定的精确 full-InChIKey oracle 判定。任何模型或 provider 的库存文字都只是 advisory。
4. solved 仅由 Host 在目标根连通、全部步骤可物化、全部终端叶库存闭合后计算。
5. Builder 不拥有 handoff、失败、停止或终止 Strategy 的权限；战略层仅由 Host/MCTS 的库存状态、可扩展状态和预算结束。
6. short-tail 只能接收目标根可达、真实回放产生、已做负库存观察的开放叶。
7. Builder 不自报具有判定权的 `strategy_relation`。Builder 的 `move_role=key|enabling|supporting` 仅作为同一路径下一次调用的紧凑记忆，不参与 admission，也不证明 Strategy 已执行；Critic 必须根据完整回放路线独立核查。
8. Critic 可以标记 blocking，Editor 可以提交 RouteJSON 的依赖闭合替换范围，但两者都不能降低 Host 的结构 admission 标准。
9. 工具预算只统计实际开始执行的调用。被 sandbox 在执行前拦截的尝试保留在 worker 审计记录中，但不得吞掉已经完整生成且通过 schema 的结构化 artifact；实际执行的越权或超预算调用仍然拒绝整轮 worker 输出。

### 2.1 与原论文的 Editor 对齐边界

依据论文第 2.5 节、Methods 4.3 和 Figure 5，Editor 必须接收完整 RouteJSON 与 Critic annotations，允许重排、插入、删除步骤以及修改条件/官能团，并在保留关键 disconnection 和整体 Strategy 的前提下做 surgical editing；编辑后的完整路线再交回 Critic。论文把该过程类比为 source-file 的 surgical / exact replacement，但没有规定模型每轮必须重新生成所有未修改行。

当前实现保持上述外部语义：Editor 看完整路线，Host 最终得到并全量回放完整 edited RouteJSON。wire output 改为 `replace_span`，只是把“未修改内容的复制”从模型移交给 Host；它不是只修一行的局部补丁，也不是降低论文允许的编辑能力。若化学依赖要求，替换范围可以从一行扩展为多行，直至整条路线。

## 3. 组件合同

### 3.1 Sequential Strategy Director（Host 编排器）

职责：建立三个独立 Strategy 分支，分配调用/时间/token 预算，驱动 Strategy、Builder、AiZ、Critic、Editor，并输出 `GlobalCampaignPlan`。

输入：canonical target、执行 profile、分支数、每分支 Builder 调用上限、Critic/Editor 轮数、模型配置、AiZ runtime 和 stock binding。

输出：每个分支的 Strategy、Host 回放步骤、开放叶、搜索诊断、Critic/Editor 记录，以及最终 plan。

拥有的权限：选择阶段顺序和预算；接纳或拒绝模型 artifact；调用 Host compiler；按 MCTS/库存/预算结束战略层；记录并处理 Host/runtime 的客观失败；把剩余真实开放叶统一交给 short-tail。

禁止拥有的权限：不能把模型文字当库存或 solved；不能绕过 ReactionJSON replay；不能把不连通 partial route 伪装成完整路线。

失败去向：Strategy 未生成、sidecar/runtime 异常、预算耗尽、连续输出或回放被拒绝、AiZ 没有可扩展节点，或最终没有可用 Host-replayed 路线时，Director 保留客观诊断，不输出伪路线。这些状态由 Host/runtime 产生，不能由 Builder 自报。

### 3.2 Strategy Generator（LLM）

职责：先于反应搜索提出三个实质不同的高层合成假设。每个假设指出关键正向事件、使其成立的 reactive-handle motif，以及主要立体或官能团控制。

输入：paper-matched 模式只接收 canonical `campaign_target` 和 `strategy_count=3`。三个 Strategy 在一次模型调用中共同生成。

输出：恰好三个 compact StrategyCard；每张只有：

- `strategy_query`：一句高层方向；
- `strategy_signature`：短身份签名。

拥有的权限：在内部比较骨架/拓扑、关键反应、官能团/保护和立体控制维度；提出路线方向。

禁止拥有的权限：不画前体，不写 ReactionJSON，不给完整路线、条件、库存、证据或 solved；不要求 Builder 一次生成全路线。

失败去向：输出数量、字段或多样性合同不成立时，该 Strategy portfolio 不进入 Builder。

### 3.3 Route Builder（LLM，单节点策略）

职责：对 AiZ 当前选中的一个节点，先在内部推演从当前叶经过 Strategy 命名关键构建并通向可得前体的完整化学路径，再在该路线语境中比较断键，只返回当前一个最好的真实反应动作。“只输出一个 ReactionJSON”是输出边界，不是思考边界。复杂 concerted/cascade 反应可以在同一个反应中包含多个相互关联的 graph edit，但不能用虚构中间步拆开。

输入：

- `target_smiles`：campaign target，仅用于保持目标语境；
- `strategy.strategy_query`、`strategy.strategy_signature`：方向假设和命名关键构建；Builder 用它约束完整路线推演，但本次只物化当前一个反应；
- `selected_leaf_mapped`：本次唯一可编辑的 Host mapped product；
- `connected_path_reactions`：目标到当前叶已经 Host 回放的 `step_id`、观察性 `claimed_move_role`、`reaction_family`、`edit_summary`；`claimed_move_role` 只是早先 Builder 的自报标签，Builder 必须依据真实 edit summary/结构重新判断关键构建是否已经发生，不能从 `claimed_move_role=key` 推断完成；
- `ancestor_smiles`：防止回到路径祖先的结构负记忆；
- `last_rejection_for_this_leaf`：该叶最近一次 Host replay/cycle 失败，只携带原始 typed compiler error、`operation_index`、`failed_operation` 和必要的局部端点事实；不回灌整组 operations，也不得压成笼统的 replay failed。

输出只有一个 expansion object：

- `reaction_intent`：合并 reaction family 与 rationale 的一句简短反应意图；
- `move_role`：仅用于后续同路径记忆的 `key`、`enabling` 或 `supporting`；只有当前 operations 所定义的可回放 precursor-to-product 转换真实执行命名关键构建时才能写 `key`，否则按实际作用写 `enabling` 或 `supporting`；它不是 Strategy 执行证明，也不参与 admission；
- `reaction_operations`：一组有序 ReactionJSON operations，不设论文未规定的 12-operation 上限；
- 模型端不输出 `order`：`add_bond` 固定新增单键；新增双键或三键时随后使用 `change_bond_order`；`add_group` 的连接键级直接写在 `fragment_smiles` 的 `[*]` 键中。Host 继续兼容并严格校验历史输入中的显式 `order`。
- `conditions`：简短条件假设，催化剂写在此处。

拥有的权限：设计当前 mapped product 的一个局部 ReactionJSON；在内部推演完整路线因果以选择这一个当前动作。

结构真实性规则：Strategy 点名的关键反应不能只出现在 `reaction_intent`、催化剂或条件文字中。若命名构建消耗或产生特定 reactive handle，相关 mapped atoms/bonds 必须参与定义该构建的 operations；若命名构建依赖立体控制，相关立体/几何信息必须在可回放结构或 operations 中被表达或有意变换。当前结构缺少必要手柄或立体设置时，应诚实输出对应的 `enabling` 步骤，而不能把不相干 graph edit 标成 `key`。

禁止拥有的权限：不输出 precursor SMILES，不改 atom-map namespace，不声明结构已回放、库存、路线完整、Critic 通过或 solved；不能输出 `handoff`、`fail`、`abort`、`give_up`、`stop` 或任何其他终止 Strategy 的动作。

继续规则：Builder 始终返回当前最佳可用 expansion。缺少关键手柄时，可以逐次执行必要的 enabling reaction，不要求一个调用完成整条路线；当前叶已经具备命名关键构建所需拓扑时，应优先执行该关键构建，而不是继续累积无关 enabling/supporting 转换。反应复杂、不确定、constraint conflict 或没有找到理想断键都不能授权停止；是否继续调用由 Host/MCTS/预算决定。

失败去向：ReactionJSON 不可回放时，Host 拒绝该 candidate，并只把底层 typed compiler error、失败 operation 及其索引和必要端点事实交给下一次 Builder 调用；完整失败候选留在 worker record，不复制进 prompt。Host 在剩余预算内重试。预算耗尽、连续回放失败、AiZ 无可扩展节点或 sidecar 异常由 Host/runtime 记录，Builder 本身不能主动终止 Strategy。

### 3.4 Worker structured-output wrapper（Host）

职责：给每种 LLM 角色提供严格、精简的 JSON schema；把模型 wire output 包入 durable artifact；验证动作、字段和无 solved 声明。

输入：角色任务、prompt、模型输出、Host-owned target/selected product context。

输出：`accepted_draft` 或带稳定 reason code 的拒绝记录。

拥有的权限：要求恰好一个 expansion；拒绝 Builder 控制字段和非法结构；添加 artifact id、case id、source 等 Host-owned envelope。

禁止拥有的权限：schema 通过不等于化学正确、结构可回放、库存闭合或 handoff 合格。

运行时工具合同：Strategy、Builder、Critic、Editor 不设置 tool-call 次数上限，也不因模型调用过工具而拒绝已经生成的结构化 artifact。原始 tool-call 记录仍保留在 `WorkerRunRecord.tool_calls`，仅用于观测；Prompt 不讨论工具是否允许。工具调用不绕过后续 schema、RouteJSON compiler、Critic 或库存审查。

失败去向：非法结构化输出以及实际执行的越权/超预算工具调用停留在 worker record，不进入 compiler/AiZ。执行前被 sandbox 拦截的工具尝试不构成结构化输出失败，Host 继续按正常边界编译该 artifact。

### 3.5 ReactionJSON Compiler / Replay（Host 结构权威）

职责：将 Builder/Editor 的有序 atom-level graph edits 应用于当前 mapped product，确定性地产生 canonical 与 mapped precursors；维护 atom provenance、依赖关系和 map namespace。

输入：Host mapped product、ReactionJSON operations、可选条件元数据。

输出：Host-derived `product_smiles`、`mapped_product_smiles`、`precursor_smiles`、`mapped_precursor_smiles`、replay audit 和可物化 route step。

拥有的权限：拒绝不存在的 map、非法 primitive、元素 transmutation、无效 fragment attachment、原子来源断裂、拓扑不连通、RouteJSON 依赖错误和循环。

禁止拥有的权限：不评价反应在实验上是否可行，不因为命名反应看似合理而放行错误 graph edit，不判断库存或 solved。

失败去向：返回局部、可操作的 replay diagnostic；不得通过降低 admission 标准来迁就 Editor。

### 3.6 AiZ MCTS strategic search（每个 Strategy 的树搜索）

职责：在一个独立 Strategy 分支中拥有 MCTS/UCB 节点选择、OR candidate 实例化、循环剪枝、库存终止和回传。Builder 是它的节点 expansion policy，Host-replayed candidate 是唯一可加入树的动作。

输入：target、冻结 Strategy、Host-replayed ReactionJSON candidate、精确 stock query、调用/深度/迭代上限。

输出：最佳目标根连通路径的 route steps、开放叶状态、是否全叶库存闭合、policy-call 与 MCTS 诊断。

拥有的权限：选择下一次展开哪个节点；在多个真实树状态中选择投影路径；按 stock 状态停止搜索。

禁止拥有的权限：不能重新绘制 Host precursor，不能把 Host 拒绝的 Builder 输出变成 deferred leaf，不能把 target 本身的库存命中当零步解。

失败去向：非法 Builder 输出或不可回放 candidate 被 Host 拒绝并在预算内重试；预算耗尽、无可扩展节点和 runtime 异常由 AiZ/Host 记录。战略搜索结束后，Host 保留真实开放叶，统一做库存审计和 short-tail。

### 3.7 Exact stock oracle（Host / AiZ 绑定库存）

职责：使用冻结的 ZINC + eMolecules full-InChIKey 集合做精确 membership 判断。paper-matched 目标根使用 leave-target-out 规则，避免“目标本身可购买”产生零步解。

输入：canonical molecule identity/full InChIKey、绑定的 content-addressed stock index。

输出：正/负库存观察；只有 Host 接纳的正观察才能关闭叶。

拥有的权限：关闭真实开放叶；为战略层和最终路线提供库存事实。

禁止拥有的权限：不评价反应、不证明路径连通；provider 或 LLM 的库存字段不能覆盖它。

### 3.8 Critic（LLM）

职责：按正向合成依赖顺序模拟完整 Host-replayed RouteJSON，逐步识别具体 chemical contradiction，并独立核查路线是否真正执行 Strategy 的关键构建。

输入：campaign target；Strategy query/signature；每一步的 Host-derived mapped product/precursors、ReactionJSON operations、reaction family、conditions 和完整路线依赖。Builder 的 key/anchor 自我声明不作为证据。

输出：逐步 assessment（pass/uncertain/reject、blocking、blocking type、简短原因、条件评价、最小修复建议）、`strategy_adherence` 和 overall assessment。

拥有的权限：把具体机理、原子来源、官能团、化学选择性、立体或顺序矛盾标记为 blocking；要求 Editor 修复。

禁止拥有的权限：不改 RouteJSON，不把缺文献/缺库存/条件略欠具体单独当 blocker，不声明实验可行、库存或 solved，也不能靠删路线后缀降低 blocking rate。

失败去向：有 blocker 时进入 Editor；没有 blocker 时结束循环；Critic unavailable/reject 会被记录，但不直接擦除真实路线或授予/撤销 stock closure。

### 3.9 Editor（LLM）

职责：接收完整 RouteJSON 和 Critic annotations，对阻断问题做协调、可回放的文档级修复，同时保留核心 Strategy；输出最小但化学上充分的依赖闭合替换范围。

输入：campaign target、Strategy query/signature、完整当前 RouteJSON（含所有 step IDs、Host mapped product/precursors、ReactionJSON、条件和依赖链接）、Critic annotations，以及最多两条 `rejected_net_edit_signatures`。重试时只额外接收最新 Host materialization failure、已回放 prefix、真实 mapped open precursor 和 mapped frontier；完整失败 `replace_span` 留在 worker record，不复制进下一轮 prompt。

输出：一个短 `repair_summary` 和一个 `replace_span`：

```json
{
  "repair_summary": "...",
  "replace_span": {
    "remove_step_ids": ["old_step_2", "old_step_3"],
    "revised_steps": [
      {
        "step_id": "old_step_2",
        "product_smiles": "...",
        "reaction_family": "...",
        "conditions": [],
        "catalyst": "",
        "reaction_operations": [
          {"op": "break_bond", "map_a": 12, "map_b": 19}
        ]
      }
    ]
  }
}
```

`remove_step_ids` 选择旧路线中被替换的行；`revised_steps` 只包含完整替换化学。模型不输出 `entry_boundary_step_id`、`resume_at_step_id`、`mapped_product_smiles` 或 precursor lists，因为这些边界由 Host 从当前 RouteJSON 推导，不能形成第二套可漂移声明。真实 schema 要求每个 revised step 至少一个有效 primitive。

拥有的权限：通过替换一行、多行或全部行实现重排、插入、删除或替换步骤；改变条件、官能团、保护基、反应 handle、路线长度和终端前体；必要时扩大范围以协调修改多步依赖。这里“最小”指完成化学依赖闭合所需的最小范围，不是机械追求最少行。

边界规则：Editor 先决定保留哪些 target-side 步骤，并把每个保留步骤实际消费的 Host-derived precursor 作为保留边界；修订后的上游化学必须直接产生它需要重接的每个边界。若不存在化学上连贯的直接连接，就把不兼容的保留步骤一并纳入 `remove_step_ids` 并扩大替换范围。不得仅为连接两个独立设计的端点而虚构中间转化，也不得靠改名或更换条件掩盖相同的已拒绝 graph edit。

禁止拥有的权限：不直接写 precursor 事实，不重编号或猜测 mapped frontier，不破坏 target root 或无理由放弃 Strategy key disconnection，不宣称库存/solved，不把 Critic 已拒绝的同一 normalized net edit 换名字后复用。Editor schema 没有 `fail` / `unrepairable` 捷径。

Host 合并与失败去向：Host 按 step ID 保留所有未列入 `remove_step_ids` 的行，在第一个被移除行的位置插入 `revised_steps`，然后从 campaign target 全量编译合并后的 RouteJSON。旧 step ID 必须唯一；remove IDs 必须存在且不重复；revised IDs 不得与保留行冲突。任何入口错误、后缀未重接、map/provenance 错误、循环或孤立分支仍由现有 DAG compiler 拒绝。失败时原路线保持权威；下一轮只接收真实 replay boundary 和失败 operation 的最小因果焦点，达到轮数上限后保留原路线和诊断，不降低 admission。

### 3.10 Critic–Editor loop（Host 状态机）

职责：`Critic -> 若 blocking 则 Editor replace_span -> Host 合并并全量重放 -> Critic`，直到没有 blocker 或达到迭代上限。

输入：一份目标根连通、Host-replayable RouteJSON，以及配置的最大修复轮数。

输出：最新可回放路线、每轮 Critic/Editor history、最终 critique 与 materialization diagnostics。

拥有的权限：控制循环、保留上一份可回放路线、机械合并替换范围、拒绝不连通 Editor draft。

禁止拥有的权限：不能因为新 draft 文字更好就覆盖旧路线；不能用拓扑分数或库存偏好代替“是否修复 blocking reaction”的判断。

### 3.11 Canonical graph ingestion / final materialization（Host）

职责：把 Director 输出的目标根 RouteJSON 转成 canonical hypergraph，保存路线 family/step/分子 lineage；把战略段与合格 short-tail 段拼成同一目标根路线；最终计算开放叶、库存闭合和 solved。

输入：Host-replayed strategic skeleton、route family、开放叶、Critic/Editor diagnostics、short-tail hypotheses 和 stock observations。

输出：canonical molecules/hypotheses/routes、proof portfolio、route admission 和最终科学状态。

拥有的权限：结构 identity 去重、目标根 topology 校验、依赖闭合、路线拼接、最终 stock-closed/solved 计算。

禁止拥有的权限：不能把 disconnected provider island 接到目标；不能把 partial route、Critic pass 或 provider_solved 单独升级为完整解。

### 3.12 Deficit frontier / action scheduler（Host）

职责：从 canonical graph 的当前事实派生下一项缺口。对目标根可达开放叶，先做精确库存审计；库存明确为负且仍为开放叶时，才生成一次 short-tail expansion action。

输入：canonical graph、目标根可达 route boundaries、stock observations、已尝试 provider actions。

输出：`STOCK` 或 `EXPANSION` deficit，以及带 route-family binding 的调度 action。

拥有的权限：抑制 target root、disconnected leaf、已库存闭合叶、重复 provider 尝试和未显式 eligible 的 short-tail 调用。

禁止拥有的权限：不发明结构、不重复消费同一 frontier 的 provider 预算、不把 action 完成等同于路线完成。

### 3.13 AiZ short-tail runtime

职责：对一个已绑定的目标根开放叶运行短模板搜索；paper-matched 配置为最多 6 transforms、500 iterations、1200 s，并只选一条完整、全叶 provider-stock-closed 的 coherent tail。

输入：`frontier_smiles`、父 route-family ids、显式 AiZ runtime/config binding、paper short-tail eligibility、负库存观察。

输出：一条选中的 provider route（若存在）、route lineage、runtime binding 和 provider diagnostics；随后进入 Host canonical ingestion。

拥有的权限：在模板空间内搜索该 leaf 的上游路线并报告 provider stock 状态。

禁止拥有的权限：不能搜索完整 campaign target 来替代战略层；paper profile 不接纳 partial tail；`provider_solved=true` 不能绕过目标根拼接和 Host stock/admission。

失败去向：无完整 tail 时保持该父路线未闭合，并记录一次已结算的 provider attempt；不递归制造新的 paper short-tail 叶。

### 3.14 Model I/O journal（观测，不是科学状态）

职责：把每次模型实际输入、wire output、durable artifact/status 追加到 `model-io.jsonl`，供运行时监视与复盘。

拥有的权限：记录和恢复相同 task contract 的 worker call。

禁止拥有的权限：日志内容不能作为结构、库存、Critic 或 solved 权威；prompt/model contract digest 改变后不得复用旧记录。

### 3.15 Web job registry / live projection（Host 观测面）

职责：`POST /api/v4/jobs` 在当前 Web gateway 中创建唯一后台任务记录；`GET /api/v4/jobs` 合并该进程的任务记录与同一 gateway `list_runs()` 可见的历史运行。实时页面只从该任务绑定 run 的 `model-io.jsonl` 派生 Strategy、Builder、Critic/Editor 进度，不创建第二套科学状态。

输入：Web gateway job/run registry、run directory、append-only `model-io.jsonl` 和 canonical run status。

输出：任务队列、SSE live projection、历史运行入口和 workbench 链接。

拥有的权限：展示同一 gateway 已注册运行；把 durable model I/O 和 Host 状态投影成人类可读进度。

禁止拥有的权限：不扫描任意 `results/**` 来猜测外部运行，不导入另一个隔离 `run_index.sqlite3` 作为第二权威，不从浏览器状态推导结构、库存或 solved。

同步合同：需要在网站实时查看的 smoke 必须通过 `/api/v4/jobs` 启动，或显式使用同一 gateway 注册。直接 CLI 启动并把运行索引写入独立输出目录的实验不会自动出现在网站队列；其报告仍有效，但属于另一个运行注册域。`run_scope=blind` 的 benchmark smoke 还必须显式提交 `benchmark_stock_index`、`benchmark_stock_index_sha256` 和 `benchmark_stock_name`；只有 interactive paper-profile 请求才允许从已经绑定的 AiZ config 自动解析冻结库存。

生命周期合同：Web gateway 的 mutable job row 与后台 worker thread 属于当前进程；gateway 重启后，持久运行只恢复为 `historical_snapshot`，已有 `model-io.jsonl` 仍可回放，但不会伪装成已恢复执行。需要跨进程续跑时必须由独立持久 executor/checkpoint 合同负责，不能靠扫描 `results/**` 或复制一份 running 状态来推断。

### 3.16 Analyst（论文存在，当前未实现）

论文职责：对最终路线给出 infeasible/poor/acceptable/good/excellent 五级可行性评分，并指出关键步骤和主要风险。

当前状态：代码中没有与论文等价的独立 Analyst LLM 阶段。现有 route scoring、Critic verdict 或 UI 摘要不能冒充 Analyst。若未来实现，输入必须是最终 Host-materialized route；输出只能是面向人的评价，仍不得授予结构、库存或 solved。

## 4. 状态与去向

Builder 没有动作/reason 词表和终止通道。worker schema 只接纳一个 ReactionJSON expansion；Host 将其回放为 candidate 后交给 AiZ。任何遗留的 `builder_action`、`builder_reason`、`stop_signal` 或 `stop_reason` 都由合同校验拒绝。

| 状态 | 谁产生 | 含义 | 下一步 | 绝不代表 |
|---|---|---|---|---|
| Builder expansion | Builder | 提议当前节点的一个 ReactionJSON | Host compile/replay；成功后交 AiZ | 前体真实、库存、solved |
| `stock_closed` | Host/AiZ exact stock | 某个真实叶或整条树满足精确库存条件 | 关闭叶或结束战略搜索 | 化学可行、Critic 通过 |
| `budget_exhausted` | Host/AiZ runtime | 达到调用/时间/迭代上限 | 保留真实 partial diagnostics | Builder 动作或 solved |
| `calls_exhausted` | AiZ policy runtime | 达到该 Strategy 的 Builder 调用上限 | 停止继续调用并保留已回放路径 | Builder 动作或 solved |
| `paper_strategy_sidecar_failed` / `aizynthfinder_strategy_sidecar_failed` | Director | sidecar 客观执行异常 | 保留分支诊断，不伪造路线 | Builder 可主动选择的动作或 solved |
| replay/contract rejection reason | Worker/compiler/Director | Builder 输出非法或 ReactionJSON 无法回放 | 在剩余预算内重试；耗尽后保留诊断 | Strategy 失败、库存或 solved |
| `critic_blocked` | Critic | 存在具体化学/Strategy 合同矛盾 | Editor 修复 | 结构无效或库存失败 |
| `editor_repaired` | Host（合并并重放成功后） | Editor span 已合并并完整编译为新的真实 RouteJSON | 再 Critic | 实验可行或 solved |
| `editor_failed` | Host | Editor draft 无法完整回放或未解除 blocker | 重试或保留旧路线/诊断 | 可以降低 admission 标准 |
| `provider_solved` | AiZ provider | provider 对其局部搜索树报告全叶库存 | Host 拼接、重放和库存复核 | campaign target 已 solved |
| `solved` | Host final materialization | 目标根连通完整路线的全部终端叶精确库存闭合 | 输出最终路线 | LLM/Provider 可自行声明 |

## 5. 权限速查

| 组件 | 设计 Strategy | 写 ReactionJSON | 派生 precursor | 判库存 | 判 blocking | 改 RouteJSON | 判 solved |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strategy Generator | 是 | 否 | 否 | 否 | 否 | 否 | 否 |
| Route Builder | 受 Strategy 引导 | 是，单节点 | 否 | 否 | 否 | 否 | 否 |
| Host compiler | 否 | 否 | 是 | 否 | 仅结构错误 | 否 | 否 |
| AiZ strategic MCTS | 否 | 否 | 否 | 使用绑定 oracle | 否 | 否 | 否 |
| Critic | 审查 adherence | 否 | 否 | 否 | 是 | 否 | 否 |
| Editor | 必须保留核心 Strategy | 是，依赖闭合替换范围 | 否 | 否 | 修复 blocker | 是 | 否 |
| Exact stock oracle | 否 | 否 | 否 | 是 | 否 | 否 | 否 |
| AiZ short-tail | 否 | 模板动作 | provider 内部 | advisory + Host 复核 | 否 | 否 | 否 |
| Final Host | 否 | 否 | 接纳已回放结构 | 是 | 记录 Critic | 接纳/拒绝 | 是 |
| Analyst（未实现） | 否 | 否 | 否 | 否 | 总结风险 | 否 | 否 |

## 6. 修改检查清单

修改 Strategy/Builder/Critic/Editor prompt、schema 或状态机时，必须逐项检查：

1. 模型 schema 是否只允许该角色拥有的动作？
2. prompt 是否与 schema 使用同一动作名和同一语义？
3. Host 是否再次验证上下文条件，而不是无条件信任合法 JSON？
4. sidecar/wire 是否只传递 Builder 的一个 ReactionJSON expansion，并把 Host/runtime 终止与失败放在独立诊断通道？
5. Host/runtime 的失败状态是否可能被投影成开放叶、partial route、库存或 solved？
6. Builder 是否看到了当前 mapped leaf、完整连接路径的反应角色和最近因果失败？
7. Editor 是否拿到完整 RouteJSON、Critic annotations、Host 真实 mapped open precursor，以及至多两条结构化负记忆？
8. provider route 是否绑定到真实 parent route family 和 target-reachable leaf？
9. 每个不变量是否只有一个 Host 权威，而不是多个重复 flag/guard？
10. 至少运行一个正例和一个负例；不得仅用 schema 单测代替真实状态机测试。
11. 模型 schema 中是否意外重新引入 `fail`、`abort`、`give_up` 或 `stop` shortcut？
12. Host 客观失败是否与模型动作完全分离？
13. 被 sandbox 在执行前拦截的工具尝试是否仍被错误计入预算或吞掉结构化 artifact？实际执行的越权调用是否仍会被拒绝？
14. 需要网站实时展示的运行是否由同一 Web gateway `/api/v4/jobs` 注册，而不是写入孤立运行索引？
15. Editor 是否只输出 `replace_span`，且 Host 而非模型拥有 entry/resume/mapped boundary 推导？
16. 未列入 `remove_step_ids` 的 prefix、suffix 和 sibling rows 是否原样保留，并在合并后经过完整 DAG replay？
17. 聚焦测试是否覆盖单行替换、多行/整路线替换、成功重接 suffix，以及边界不匹配时拒绝？
