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
7. Builder 不自报具有判定权的 `strategy_relation`。Builder 的 `checkpoint_relation=preparatory|executes_checkpoint` 仅作为稀疏 Critic 的调度 metadata，不参与 admission，也不证明 Strategy 已执行；Host 必须先回放，Critic 再根据真实图编辑确认 `checkpoint_match`。
8. Critic 可以标记 blocking，Editor 可以提交 RouteJSON 的依赖闭合替换范围，但两者都不能降低 Host 的结构 admission 标准。
9. 工具预算只统计实际开始执行的调用。被 sandbox 在执行前拦截的尝试保留在 worker 审计记录中，但不得吞掉已经完整生成且通过 schema 的结构化 artifact；实际执行的越权或超预算调用仍然拒绝整轮 worker 输出。

### 2.1 与原论文的 Editor 对齐边界

依据论文第 2.5 节、Methods 4.3 和 Figure 5，Editor 必须接收完整 RouteJSON 与 Critic annotations，允许重排、插入、删除步骤以及修改条件/官能团，并在保留关键 disconnection 和整体 Strategy 的前提下做 surgical editing；编辑后的完整路线再交回 Critic。论文把该过程类比为 source-file 的 surgical / exact replacement，但没有规定模型每轮必须重新生成所有未修改行。

当前实现保持上述外部语义：Editor 看完整路线，Host 最终得到并全量回放完整 edited RouteJSON。wire output 改为 `replace_span`，只是把“未修改内容的复制”从模型移交给 Host；它不是只修一行的局部补丁，也不是降低论文允许的编辑能力。若化学依赖要求，替换范围可以从一行扩展为多行，直至整条路线。

## 3. 组件合同

### 3.1 Sequential Strategy Director（Host 编排器）

职责：建立三个独立 Strategy 分支，分配调用/时间/token 预算，驱动 Strategy、Builder、AiZ、Critic、Editor，并输出 `GlobalCampaignPlan`。

输入：canonical target、执行 profile、分支数、每分支 Builder 调用上限、最终 Critic/Editor 修复轮数、模型配置、AiZ runtime 和 stock binding。在线 key-event Critic 没有独立的“每个 Strategy 固定 N 次”配额。

输出：每个分支的 Strategy、Host 回放步骤、开放叶、搜索诊断、Critic/Editor 记录，以及最终 plan。

拥有的权限：选择阶段顺序和预算；接纳或拒绝模型 artifact；调用 Host compiler；按 MCTS/库存/预算结束战略层；记录并处理 Host/runtime 的客观失败；把剩余真实开放叶统一交给 short-tail。

禁止拥有的权限：不能把模型文字当库存或 solved；不能绕过 ReactionJSON replay；不能把不连通 partial route 伪装成完整路线。

失败去向：Strategy 未生成、sidecar/runtime 异常、预算耗尽、连续输出或回放被拒绝、AiZ 没有可扩展节点，或最终没有可用 Host-replayed 路线时，Director 保留客观诊断，不输出伪路线。这些状态由 Host/runtime 产生，不能由 Builder 自报。

### 3.2 Strategy Generator（LLM）

职责：先于反应搜索提出三个实质不同的高层合成假设。每个假设指出关键正向事件、使其成立的 reactive-handle motif，以及主要立体或官能团控制。

输入：paper-matched 模式只接收 canonical `campaign_target` 和 `strategy_count=3`。三个 Strategy 在一次模型调用中共同生成。

输出：恰好三个 compact StrategyCard；每张只有：

- `strategy_query`：一句高层方向；
- `critical_assumption`：一句最重要的化学假设；
- `critic_checkpoint`：一句应触发稀疏审查的实际图变换；它不能是准备步骤、固定步数或 required-map checklist。

拥有的权限：在内部比较骨架/拓扑、关键反应、官能团/保护和立体控制维度；提出路线方向。

禁止拥有的权限：不画前体，不写 ReactionJSON，不给完整路线、条件、库存、证据或 solved；不要求 Builder 一次生成全路线。

失败去向：输出数量、字段或多样性合同不成立时，该 Strategy portfolio 不进入 Builder。

Strategy Critic 复用同一个三字段输出合同，不建立第二套 verdict。初始 portfolio Critic 读取 target topology 与三张卡；后续 Strategy Generator 和 Strategy Critic 读取同一个 Host 派生的 `strategy_horizon_context`：当前真实 mapped leaf、该分子 occurrence 的 `connected_path_reactions`、已完成 milestones，以及当前 split 的 sibling co-precursor 状态。完整 RouteJSON 仍由 Host、路线级 Critic 和 Editor 持有，不能再把“全分支最后若干行”复制成 Strategy 的 prefix；那会把 sibling 的上游反应伪装成当前叶历史。Critic 据此检查新方向能否从当前叶合成既有下游反应 spine。`active sibling horizon` 不等于 `completed milestone`：切换到没有适用 horizon 的 sibling 可以生成新 Strategy，但只有当前 mapped target-to-leaf spine 上已经由 Key Critic `pass` 的 checkpoint 才能进入 `completed_milestones`；仅附着在 preparatory step 上或属于其他 sibling 的 Strategy 继续保存在 branch state，返回该 lineage 时恢复，不能伪装成 Host 已完成事实。Strategy horizon 不等于 Builder 的下一步反应：Builder 可以先逐步完成必要的保护、氧化还原、解掩蔽或 reactive-handle 安装，Critic 只能审查这些准备步骤的相容性与顺序，不能用其中一个准备步骤替换路线定义级 horizon 或 checkpoint。只要当前叶仍有复杂主骨架，Critic 的修订或替换就必须保持骨架构建、重排、立体接力或收敛简化的 route-defining 粒度；只有主骨架已经简单且不存在待解决的路线定义级问题时，外围官能团调整才可能成为新 Strategy。Critic 不得把新叶当孤立分子审查，也不得为显得具体而自行发明命名反应。若 `critical_assumption` 依赖同一个关键事件产生的立体或选择性结果，`critic_checkpoint` 必须保留该结果，不能削弱为“只形成一根键”。

### 3.3 Route Builder（LLM，单节点策略）

职责：对 AiZ 当前选中的一个节点，先在内部推演从当前叶经过 Strategy 命名关键构建并通向可得前体的完整化学路径，再在该路线语境中比较断键，只返回当前一个最好的真实反应动作。“只输出一个 ReactionJSON”是输出边界，不是思考边界。复杂 concerted 或真正不可分的 cascade 可以在同一个反应中包含多个相互关联的 graph edit，但不能用虚构中间步拆开；独立的保护/脱保护、活化、氧化还原、workup 共价变化或第二套试剂阶段必须是相邻的独立 route edge，即使实验上可以不分离中间体，也不能为了通过 Critic 而合并计步。

输入：

- `target_smiles`：campaign target，仅用于保持目标语境；
- `strategy.strategy_query`、`strategy.critical_assumption`、`strategy.critic_checkpoint`：方向假设、关键化学风险和唯一审查 checkpoint；Builder 用它们约束完整路线推演，但本次只物化当前一个反应；
- `selected_leaf_mapped`：本次唯一可编辑的 Host mapped product；
- `connected_path_reactions`：目标到当前叶已经 Host 回放的 `step_id`、`checkpoint_relation`、`reaction_family`、`edit_summary`；`checkpoint_relation` 只是早先 Builder 的调度 metadata，不能从中推断关键事件成立；
- `current_split_context`：仅包含产生当前叶的父反应，以及同一次 split 的 Host-mapped sibling co-precursors；它用于判断偶联手柄和官能团兼容性，不扩展成整棵搜索树；Host 优先按 mapped boundary 追踪 lineage，不能因 AiZ 的未映射立体投影不同而丢失父步骤或 sibling；
- `ancestor_smiles`：防止回到路径祖先的结构负记忆；
- `last_rejection_for_this_leaf`：该叶最近一次 Host replay/cycle 失败，只携带原始 typed compiler error、`operation_index`、`failed_operation` 和必要的局部端点事实；不回灌整组 operations，也不得压成笼统的 replay failed。
- `pending_checkpoint_feedback.active_constraints`：从 append-only `key_event_critic_history` 派生、只属于当前 Strategy 与当前 mapped leaf lineage、且能够由同一父 leaf 的新 candidate 或一个以该 leaf 为 product 的新准备步骤修正的 blocking obligation；不同 reject 不得互相覆盖，preparatory move 不得擦除，sibling leaf 与新 Strategy 不得串用。若 Key Critic 判定修复必须修改或重排 focus 之前任何已接纳行，Host 不把该 finding 再交给普通 Builder，而是立即停止当前 selected branch expansion 并调度既有 transactional Path Repair。`uncertain` 是 Critic 自己的 evidence debt，不进入后续 Builder prompt，也不能要求后续叶修改已经物化的旧边；AiZ 选中该关键事件直接 mapped precursor 的新上游步骤后，Critic 最多做一次局部证据复审。只有复审 `pass` 且 focus/evidence steps 都已进入当前路径，或同一 horizon 的新 checkpoint `pass` 被选中后才退休。它不复制完整 Critic 输出，也不建立第二个可写状态权威。

输出只有一个 expansion object：

- `reaction_intent`：合并 reaction family 与 rationale 的一句简短反应意图；
- `checkpoint_relation`：仅为 `preparatory` 或 `executes_checkpoint`；只有当前 operations 本身实现 `strategy.critic_checkpoint` 时才选择后者。它只请求一次 Host 回放后的 Critic 调度，不证明 Strategy 已执行，也不参与 admission；
- `reaction_operations`：一组有序 ReactionJSON operations，不设论文未规定的 12-operation 上限；
- 模型端不输出 `order`：`add_bond` 固定新增单键；新增双键或三键时随后使用 `change_bond_order`；`add_group` 的连接键级直接写在 `fragment_smiles` 的 `[*]` 键中。Host 继续兼容并严格校验历史输入中的显式 `order`。
- 模型端不计算 RDKit `stereo_atom_maps`：`set_bond_stereo` 只表达键端点和 E/Z/CIS/TRANS/NONE/ANY 意图；Host 在完成图编辑与价态补全后按当前结构派生 reference neighbours。历史 artifact 中错误的 reference maps 不得覆盖 Host 派生结果。
- 论文公开的十个 primitive 继续固定为 `reactionjson_public_profile.2026-08-17.v1`。当前管线若需给新生成或原本未指定的四面体中心赋绝对构型，只能使用显式版本化扩展 `set_tetrahedral_stereo(map_idx, configuration=R|S)`；Host 尝试 RDKit CW/CCW 并验证实际 CIP。扩展使用必须进入 replay audit，不能伪装成论文公开 profile，也不能把 `invert_stereocenter` 复用为“创建手性”。
- `conditions`：简短条件假设，催化剂写在此处。

拥有的权限：设计当前 mapped product 的一个局部 ReactionJSON；在内部推演完整路线因果以选择这一个当前动作。

结构真实性规则：Strategy 点名的关键反应不能只出现在 `reaction_intent`、催化剂或条件文字中。若命名构建消耗或产生特定 reactive handle，相关 mapped atoms/bonds 必须参与定义该构建的 operations；若命名构建依赖立体控制，相关立体/几何信息必须在可回放结构或 operations 中被表达或有意变换。当前结构缺少必要手柄或立体设置时，应诚实输出 `checkpoint_relation=preparatory`，不能把不相干 graph edit 标成 `executes_checkpoint`。一个 key edge 若还需要独立脱保护、氧化还原或 workup 才得到其 product，必须拆成相邻步骤；名称、多个 conditions 字符串或“一锅完成”不能把两个独立化学事件变成一个 route edge。

禁止拥有的权限：不输出 precursor SMILES，不改 atom-map namespace，不声明结构已回放、库存、路线完整、Critic 通过或 solved；不能输出 `handoff`、`fail`、`abort`、`give_up`、`stop` 或任何其他终止 Strategy 的动作。

继续规则：Builder 始终返回当前最佳可用 expansion。缺少关键手柄时，可以逐次执行必要的 enabling reaction，不要求一个调用完成整条路线；当前叶已经具备命名关键构建所需拓扑时，应优先执行该关键构建，而不是继续累积无关 enabling/supporting 转换。反应复杂、不确定、constraint conflict 或没有找到理想断键都不能授权停止；是否继续调用由 Host/MCTS/预算决定。

失败去向：ReactionJSON 不可回放时，Host 拒绝该 candidate，并只把底层 typed compiler error、失败 operation 及其索引和必要端点事实交给下一次 Builder 调用；完整失败候选留在 worker record，不复制进 prompt。Host 在剩余预算内重试。预算耗尽、连续回放失败、AiZ 无可扩展节点或 sidecar 异常由 Host/runtime 记录，Builder 本身不能主动终止 Strategy。

### 3.4 Worker structured-output wrapper（Host）

职责：给每种 LLM 角色提供严格、精简的 JSON schema；把模型 wire output 包入 durable artifact；验证动作、字段和无 solved 声明。

输入：角色任务、prompt、模型输出、Host-owned target/selected product context。

输出：`accepted_draft` 或带稳定 reason code 的拒绝记录。

拥有的权限：要求恰好一个 expansion；拒绝 Builder 控制字段和非法结构；添加 artifact id、case id、source 等 Host-owned envelope。

禁止拥有的权限：schema 通过不等于化学正确、结构可回放、库存闭合或 handoff 合格。

运行时工具合同：Strategy、Builder、Critic、Editor 不设置 tool-call 次数上限，也不因模型调用过工具而拒绝已经生成的结构化 artifact。原始 tool-call 记录仍保留在 `WorkerRunRecord.tool_calls`，仅用于观测；Prompt 不讨论工具是否允许。隔离 worker 暴露的只读 `inspect_mapped_smiles` 直接返回局部原子邻接、键级、环路径和立体事实，避免模型为结构检查另写 RDKit shell。工具调用不绕过后续 schema、RouteJSON compiler、Critic 或库存审查。

失败去向：非法结构化输出以及实际执行的越权工具调用停留在 worker record，不进入 compiler/AiZ。工具失败和执行前被 sandbox 拦截的尝试都保留在 worker record 作为观测信息，但不覆盖合法结构化 artifact 的正常 schema、compiler、Critic 和库存判定。

### 3.5 ReactionJSON Compiler / Replay（Host 结构权威）

职责：将 Builder/Editor 的有序 atom-level graph edits 应用于当前 mapped product，确定性地产生 canonical 与 mapped precursors；维护 atom provenance、依赖关系和整条 RouteJSON 的保留 map namespace。

输入：Host mapped product、ReactionJSON operations、可选条件元数据。

输出：Host-derived `product_smiles`、`mapped_product_smiles`、`precursor_smiles`、`mapped_precursor_smiles`、replay audit 和可物化 route step。

拥有的权限：为 `add_group` 分配全路线唯一且单调递增的新 map，保留已删除 map 的 tombstone；在最终图上派生 stereo reference neighbours；拒绝不存在的 map、非法 primitive、元素 transmutation、无效 fragment attachment、原子来源断裂、拓扑不连通、RouteJSON 依赖错误和循环。若 provider 未给 mapped reaction，admission 可直接读取可信 Host replay audit 的 mapped product/precursors 判断各前体是否真实贡献 mapped atoms。

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

在线 key-event 调度：每个通过 Host replay、且 Builder 标为 `executes_checkpoint` 的新候选都应进入一次独立 Critic；不得按每个 Strategy 固定两次或其他魔法数字封顶。该 horizon 仅在 Critic pass 且候选被 AiZ 选入当前路径后退休，或者因 Builder/全局真实模型预算、wall time、MCTS/分支终止而自然结束。共享预算仍为每条已建立路线保护一次最终 Route Critic；在线 Critic 只在实际发起时结算，不能预占假想调用，也不能消费受保护的最终 Critic 槽位。当前 3×25 论文级 profile 的共享 operational envelope 为 240 次模型调用、6M input tokens、2M output tokens，以覆盖最坏情况下每个 Builder candidate 一次在线 Critic及后续 Improvement；它不是论文指标，也不是 Critic 私有配额。

在线反馈记忆：`key_event_critic_history` 是唯一事实流。Host 从中按 Strategy digest、mapped leaf lineage 和稳定 obligation id 派生 active constraints；不同 checkpoint attempt 的原因累积，同一 obligation 的新证据复审替换旧判断而不覆盖其他 obligation。下一次 Builder 与下一次 key-event Critic 都看到当前未解决约束。未选中的 pass、sibling 或新 Strategy 均不影响它。

输入：campaign target；Strategy query/signature；每一步的 Host-derived mapped product/precursors、ReactionJSON operations、reaction family、conditions 和完整路线依赖。Builder 的 key/anchor 自我声明不作为证据。

输出：逐步化学 assessment（pass/uncertain/reject、blocking、blocking type、简短原因、条件评价、最小修复建议）。在线 Key Critic 还输出一个紧凑的 `repair_scope`：`focus_edge` 表示保持既定 mapped product 不变，用一个替换反应修正当前未接纳 edge；`route_span` 表示单个 edge 不足，必须插入、重排或重建一个局部多步 span，或修改当前 mapped product／更早步骤，由 Editor 与 Host 事务性重建；`strategy_horizon` 表示现有 checkpoint 或 critical assumption 已无法由 edge 或局部 span 修复，Host 在同一 mapped leaf occurrence 上生成不同的下一阶段 Strategy；`none` 只用于 pass/uncertain。修复作用域与化学 `blocking_type` 相互独立，Host 不再把多步或产物侧缺陷误派给同父 Builder，也不让已被证伪的 Strategy 靠反复换名或换实现无限续命。最终 Route Critic 不输出该字段。另用一个短的 route-level `coupled_blocker_groups` 声明必须联合修复的 reject groups，不要求每个 pass/uncertain 行重复填空，也不靠关键词推断化学耦合。另输出非阻断观察 metadata `strategy_adherence` 和只汇总化学有效性的 overall assessment。库存闭合短路线即使未执行初始 Strategy 也保留为 opportunistic stock route；不得仅为迎合 Strategy 触发 Editor 或继续拆库存叶。

拥有的权限：把具体机理、原子来源、官能团、化学选择性、立体或步骤间顺序矛盾标记为 blocking；要求 Editor 修复。Strategy 本身没有 admission 权限，`strategy_adherence=false` 不能单独产生 blocker。

禁止拥有的权限：不改 RouteJSON，不把缺文献/缺库存/条件略欠具体单独当 blocker，不声明实验可行、库存或 solved，也不能靠删路线后缀降低 blocking rate。

失败去向：有 blocker 时进入 Editor；没有 blocker 时结束循环；Critic unavailable/reject 会被记录，但不直接擦除真实路线或授予/撤销 stock closure。

### 3.9 Editor（LLM，冻结 SynthEx/paper profile）

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

### 3.9b Path Repair Editor（LLM 扩展）

职责：仍读取完整当前 RouteJSON 和 Critic annotations，但只决定“从哪一个真实步骤开始重建”以及局部重建必须达到的化学目标；不再兼任 ReactionJSON 编写、atom-map 传播或 DAG 迁移。在线 Key Critic 触发时，repair context 是完整已接纳前缀加上一个 Host-replayed、但未接纳的 rejected focus；该 provisional 行只暴露跨步依赖，绝不成为权威路线。

输出严格只有：

```json
{
  "rollback_start_step_id": "first_target_side_step_to_change",
  "rebuild_through_step_id": "last_upstream_step_to_regenerate",
  "repair_goal": "一句可执行的局部化学修复目标",
  "active_constraints": ["至多五条无法由当前结构恢复的路线级约束"]
}
```

两个边界都必须是当前 repair context 中的真实 step ID。RouteJSON 按 target-rooted 顺序排列：`rollback_start_step_id` 是最靠 target-side、必须改变的第一行，`rebuild_through_step_id` 是最上游、仍须重建的最后一行。在线 accepted-prefix repair 中，`rollback_start_step_id` 必须属于已接纳前缀；provisional rejected focus 只能帮助界定上游重建终点和 blocker，不能被单独替换后冒充前缀修复。Host 用真实祖先关系与 Critic `coupled_blocker_groups` 合并 blocker component；Editor 仍看到完整路线，并可用 `additional_coupled_blocker_step_ids` 补充 Critic 漏掉、但确实共享不可分割官能团状态或步骤时序的 deferred blocker。Host 只接受当前 deferred blocker id，不接受任意扩权。Editor 同时给出 `preserved_suffix_compatible`；若它声明 repair goal 与待保留 exact suffix 不兼容，Host 在零 Builder 调用时拒绝该边界，要求扩大 `rebuild_through_step_id`。Editor 不输出保留行、mapped boundary、precursor、ReactionJSON 或库存字段。

Host 事务：旧路线先保持权威；Host 编译 durable DAG 得到真实 mapped rollback frontier，并给 Builder 唯一的紧凑 `path_repair` contract：repair goal、active constraints、完整已回放反应摘要和 suffix reconnect boundary。普通 `pending_checkpoint_feedback` 不再同时送入 repair Builder，避免同一 Critic finding 以两套合同重复出现。Final Critic 全文只供 Editor 使用，不再在每轮 Builder 中重复。命名空间只保留 target、durable rows、live siblings，以及 suffix 中不属于 reconnect boundary 的内部 maps；被替换路径独有的 maps 不再作为 tombstone 阻止合法复用。Host 为 unmapped `add_group` 分配的新 maps 必须写回 compiled `fragment_smiles`，使下一步和整路 replay 使用同一 provenance identity。`reserved_atom_maps` 只约束新 Builder edit 的 admission；已经 Host-materialized 的 rebuild prefix 重放时不得再次套用同一 reservation，否则会把其合法的显式 maps 误判为自碰撞。

AiZ 以零模型调用重放 durable steps，普通 Builder 从 frontier 每次只写一个 ReactionJSON；Builder 没有 handoff/fail/stop/solved。到达旧 suffix 的分子边界时 Host 在下一次付费调用前停止局部搜索。suffix 只在 exact mapped boundary，或 stereo-aware whole-molecule isomorphism 下重接；新旧边界共有的 atom maps 是 durable provenance，匹配时必须保持原号，只有单侧独有 maps 才可翻译。单个分子 occurrence 内由芳环、叔丁基、硅基等化学等价原子产生的多个 automorphism 采用稳定的确定性 translation，随后仍须通过完整 RouteJSON replay；多个分子 occurrence、真实 map 冲突或立体不一致不得拼接。Builder 新动作若产生与 suffix 同连接关系但错误立体的前体，Host 在该动作入树前返回 mismatch maps，让 AiZ 在同一父 leaf 请求修正，不沿该错误前体继续上游扩展。map 冲突、分子 occurrence 非唯一、孤立后缀或完整 replay 失败均恢复旧路线。有 suffix 时要求局部重建与 suffix 重接完成；无 suffix 时，至少一个替换步骤经 Host 回放且完整 provisional route replay 成功后立即做路线级 re-Critic。`path_repair.repair_goal` 只指导替换化学；Builder 的 `checkpoint_relation` 保持原有 Strategy Critic 调度语义，不能表示修复完成，也不是 proof、admission、handoff、stop 或 stock claim。仅当完整 replay 成功但分子边界确实尚未出现时，事务保存为非权威诊断 `retained_uncommitted_prefix`；`boundary_mapping_ambiguous`、`boundary_stereo_mismatch`、prefix/suffix replay error 与 map conflict 必须保留各自因果原因并记为 `rolled_back_uncommitted`，不能统一改写为“未到边界”。旧路线始终是唯一权威路线。进入 `rebuilt_pending_recritic` 的候选若只剩预先 deferred 的 sibling blockers，则提交当前 component 并继续下一轮；重建 component 仍被拒绝或出现新 blocker 时恢复旧路线。

Builder 只有一个配置化分支扩展上限 `max_node_expansions_per_branch`：初始构建阶段和 transactional repair 阶段都读取这一权威值；repair 调用在所有 Editor 事务间累计，计数仅用于成本归因，不再拥有独立的 6-call admission 权限。所有调用继续由同一全局模型调用/token/wall-time ledger 结算，到达 suffix 时 Host 提前停止。`max_route_local_repair_rounds` 只限制 Critic/Editor 事务轮数。

### 3.10 Critic–Editor loop（Host 状态机）

职责：`Critic -> 若 blocking 则 Editor replace_span -> Host 合并并全量重放 -> Critic`，直到没有 blocker 或达到迭代上限。

现行事务路径为：`Critic -> 若 blocking 且需跨步协调，则 Path Repair Editor directive -> Host dependency/chemical-coupling rollback -> boundary preflight -> Builder 增量重建 -> suffix stitch -> 原子提交/恢复 -> Critic`。冻结 paper control profile 仍执行上一段 `replace_span` 流程；两者由显式配置分开，不能根据 loose paper flag 推断。

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

职责：从 canonical graph 的当前事实派生下一项缺口。对目标根可达开放叶，先做精确库存审计；库存明确为负且仍为开放叶时，才生成一次 short-tail expansion action。该 native lane 结算后，增强型 profile 可为同一 route-family 生成 `CODEX_FRONTIER_EXPAND`；它复用原分支 policy-call 余额，不创建第三套重试预算。冻结论文基线不注册这项增强。

输入：canonical graph、目标根可达 route boundaries、stock observations、已尝试 provider actions。

输出：`STOCK` 或 `EXPANSION` deficit，以及带 route-family binding 的 native short-tail 或 route-bound Builder action。

拥有的权限：抑制 target root、disconnected leaf、已库存闭合叶、重复 provider 尝试和未显式 eligible 的 short-tail 调用；在原分支 Builder 预算耗尽时停止生成 Builder action，但继续保留未闭 leaf 事实。

禁止拥有的权限：不发明结构、不重复消费同一 frontier 的 provider 预算、不把 action 完成等同于路线完成。

### 3.13 AiZ short-tail runtime

职责：对一个已绑定的目标根开放叶运行短模板搜索；paper-matched 配置为最多 6 transforms、500 iterations、1200 s，并只选一条完整、全叶 provider-stock-closed 的 coherent tail。

输入：`frontier_smiles`、父 route-family ids、显式 AiZ runtime/config binding、paper short-tail eligibility、负库存观察。

输出：一条选中的 provider route（若存在）、route lineage、runtime binding 和 provider diagnostics；随后进入 Host canonical ingestion。

拥有的权限：在模板空间内搜索该 leaf 的上游路线并报告 provider stock 状态。

禁止拥有的权限：不能搜索完整 campaign target 来替代战略层；paper profile 不接纳 partial tail；`provider_solved=true` 不能绕过目标根拼接和 Host stock/admission。

失败去向：无完整 tail 时保持该父路线未闭合，并记录一次已结算的 provider attempt；不递归制造新的 paper short-tail 叶。冻结论文基线到此结束；增强型 profile 将未闭 leaf 交回同一 Action loop，由 route-bound Builder 在原分支剩余预算内继续。

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
| Editor（冻结 paper） | 必须保留核心 Strategy | 是，依赖闭合替换范围 | 否 | 否 | 修复 blocker | 是 | 否 |
| Path Repair Editor | 必须保留核心 Strategy | 否，只给 rollback intent | 否 | 否 | 定位 blocker 修复范围 | 否 | 否 |
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
7. Editor 是否拿到完整 RouteJSON 和 Critic annotations？full-route Editor 重试时是否只额外拿到 Host 真实 mapped boundary；Path Repair Editor 是否没有收到或输出第二套 precursor/map 事实？
8. provider route 是否绑定到真实 parent route family 和 target-reachable leaf？
9. 每个不变量是否只有一个 Host 权威，而不是多个重复 flag/guard？
10. 至少运行一个正例和一个负例；不得仅用 schema 单测代替真实状态机测试。
11. 模型 schema 中是否意外重新引入 `fail`、`abort`、`give_up` 或 `stop` shortcut？
12. Host 客观失败是否与模型动作完全分离？
13. 被 sandbox 在执行前拦截的工具尝试是否仍被错误计入预算或吞掉结构化 artifact？实际执行的越权调用是否仍会被拒绝？
14. 需要网站实时展示的运行是否由同一 Web gateway `/api/v4/jobs` 注册，而不是写入孤立运行索引？
15. 冻结 paper Editor 是否只输出 `replace_span`；Path Repair Editor 是否只输出两个 step boundary、短 repair goal/constraints、可选 coupled blocker ids 与 suffix compatibility？entry/resume/mapped boundary 是否仍只由 Host 推导？
16. full-route repair 中未列入 `remove_step_ids` 的行，或局部依赖路径以外的 target-side/sibling/suffix 行，是否保留并经过完整 DAG replay？
17. transactional repair 是否拒绝 unrelated rollback、保留旧权威路线、只保留 live map namespace，并证明 deletion-only 不能提交？
18. 聚焦测试是否覆盖单行替换、多行/整路线替换、fresh-map 跨步引用、对称 atom-map 的确定性 suffix 重接、多个分子 occurrence 的拒绝，以及连接或立体不匹配边界拒绝？
