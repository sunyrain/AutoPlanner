# Strategy-first synthesis architecture

AutoPlanner 的默认 `synthex_matched` 路径以战略生成而不是文献检索为起点：

```text
opaque target structure
  -> three blind StrategyCards
  -> structurally orthogonal route construction
  -> independent Codex forward critic
  -> deterministic structural materialization and graph-edit replay
  -> chemical / enzymatic / whole-cell / hybrid / mechanism competition
  -> literature, conditions and experimental closure
```

## Durable strategy contract

`strategy_card.v1` 是 canonical 设计事实，不是 prompt 临时文本。它包含：

- route-defining key transformation and key-bond signature;
- topology, convergence, stereochemical and protection plans;
- execution domain (`chemical`, `enzymatic`, `whole_cell`, `hybrid`, or `mechanistic`);
- optional ReactionJSON edit signature;
- evidence-independent `strategy_digest` and `strategy_id`.

该卡片从 Codex proposal 贯穿 `GlobalCampaignPlan`、materialization command、canonical
route family/hypothesis/edge 和 proof portfolio。后续步骤必须绑定冻结的
`strategy_digest`；冲突替换以 `strategy_replacement_conflict` 拒绝，而不是悄悄退化为
FGI 路线。

摘要只证明身份一致性，不授予 reaction proof、source authority 或 route completion。

## Independent critic

Critic 使用与设计调用分离的 Codex worker：

- 不联网，不接收目标名称、运行 ID、文献、来源或参考路线；
- 只接收目标结构、冻结的 StrategyCard、步骤结构、graph edits 和弱条件预测；
- 前向检查原子来源、机理、官能团兼容性、化学/位点选择性、立体化学、步骤顺序、
  竞争路径以及酶身份/能力；
- 输出 `viable`, `uncertain`, or `reject`；缺少论文只能导致 `uncertain`，不能单独构成
  化学拒绝；
- 不授予 reaction proof、evidence、stock 或 solved 权威。

在 Codex Critic 之前仍有一个确定性 K0 preflight，仅检查结构可解析、元素/原子来源、
ancestor cycle 和 ReactionJSON replay。它不是模型 Critic 的替代品。

## Orthogonality

三条战略的正交性优先比较：

1. mapped graph-edit digest and changed atom-map pairs;
2. key-bond signature;
3. topology-change signature;
4. convergence and stereochemical construction signatures;
5. execution domain.

模型撰写的 `strategy_signature` 只作为旧卡片的兼容诊断，不能通过改名规避重复检测。

## Scoring and lifecycle

路线同时携带两个相互独立的向量：

- `strategic_value`: key-bond leverage, topology transformation, complexity drop,
  stereochemical leverage, convergence and protection efficiency;
- `evidence_maturity`: host reaction validation, exact source binding, condition
  completeness and source independence.

Portfolio 探索允许保留“高战略价值、低证据成熟度”路线。只有最终 delivery/acceptance
仍由现有 host proof、stock 和 source policy 决定。

Program discovery 对 StrategyCard 中的 enzymatic/whole-cell/hybrid/mechanistic 路线可在
B4 前参与竞争；Program admission、validation、experimental claims 和 evidence closure
仍由后段显式开关和 host gate 管理。常规化学路线始终保留为 fallback。

## Control disposition

| Control | Class | Decision |
| --- | --- | --- |
| SMILES identity, atom/element provenance, ancestor-cycle rejection | K0 | keep at structural admission |
| ReactionJSON ordered replay and expected-precursor conformance | K0/K1 | keep and propagate end to end |
| Frozen StrategyCard digest and route-step binding | K1 | keep at route-family boundary |
| Independent Codex chemical critique | diagnostic/K0 escalation | run before evidence; only concrete contradictions reject |
| Candidate `source_channel` / `evidence_refs` during blind strategy generation | K3 | optional; no strategic weight |
| Exact evidence, condition completeness and source independence | K2 | move to credibility/delivery boundary |
| Registry recognition of unfamiliar C-C construction | K4 as exploration gate | advisory until host validation; do not delete the route |
