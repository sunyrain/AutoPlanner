# SynthEx / AutoPlanner contract reconciliation (2026-08-21)

This note records the implementation audit after comparing the published
SynthEx protocol with the local AutoPlanner execution path. It is intentionally
about the *reach* endpoint; reaction validation, evidence, conditions and
enzyme claims remain independent sidecars.

## What the paper actually specifies

| Published component | Required behavior | AutoPlanner contract |
|---|---|---|
| Strategy Generator | Three independent strategy hypotheses; four-point analysis (scaffold motif, key bond-forming event, functional-group/protection conflicts, stereochemistry) | Three `StrategyCardReport` branches with the neutral `paper_independent` prompt; no convergent/topology/enzyme mandate |
| Route Builder | LLM policy is called inside AiZynthFinder MCTS; one graph edit (`ReactionJSON`) at a selected node | AiZynthFinder sidecar owns UCB/selection/back-propagation; Codex returns one host-replayed ReactionJSON edit per request |
| Search budget | At most 25 policy expansions per strategy, 75 per target | Per-branch ceiling is 25; actual calls are now recorded and a non-stock-closed early stop is a hard failure |
| Critic / Editor | Operates on the accumulated route after Route Builder; bounded local repair loop | Full RouteJSON Editor is run only after host replay; editor cannot silently discard a valid suffix |
| Short tail | Only after an open, target-reachable leaf exists; AiZynthFinder depth 6 / 500 iterations / 1200 s | Strict `accept_partial_routes=false`; materialize → validation/admission → stock → short tail |
| Solved | One target-rooted connected route with every terminal leaf in the same ZINC + eMolecules membership oracle | `paper_reach` and `paper_equivalent_solved` are independent of B2/B3/conditions/evidence |

Primary source: [SynthEx arXiv v1](https://arxiv.org/abs/2608.07454) and the
[official repository](https://github.com/schwallergroup/SynthEx). The repository
README still states that the implementation is not released, so the comparison
is to the paper protocol rather than an unavailable byte-identical source tree.

## Why earlier runs contained so many problems

The previous runs were not one bug. They mixed four execution contracts:

1. `paper_synthex` advertised AiZynthFinder but still allowed a legacy
   `require_complete_route_json=false` path. In other places that same flag was
   interpreted as “ask the model for a complete route in one response”. This
   was the opposite of the paper's node-wise policy loop.
2. The model invocation envelope was shared by StrategyCard, Route Builder,
   Critic/Editor and provider work. A nominal 3×25 ceiling therefore did not
   imply 75 policy calls. The old report could show `paper_algorithm_equivalent`
   from configuration while the observed branch policy calls were only 5–8.
3. Partial AiZ routes could enter the host graph in non-paper profiles, and
   the short-tail stage was then skipped when no materialized target-reachable
   frontier existed. This made a correct short-tail parameter look as if the
   search engine had failed.
4. A StrategyCard-only outcome could be serialized as a successful global
   director result. That hid the fact that no RouteJSON, open leaf, stock
   audit, or short-tail task had been created.

These are control-flow and accounting mismatches, not evidence that Codex is
chemically incapable. They also explain why simply “using a stronger LLM” did
not repair the result: the stronger model was often never asked the required
node-local question, or its valid prefix was rejected by a later mixed-mode
gate.

## Changes now in the code

- `require_complete_route_json` is now a final RouteJSON admission contract.
  AiZ MCTS remains compiler-first and the node prompt is always one ReactionJSON
  edit; the flag no longer switches the paper arm to a one-shot route prompt.
- `paper_synthex` forces `strategy_tree_engine=aizynthfinder_mcts`, three
  independent branches, top-1 ReactionJSON, six local repair rounds, and the
  AiZ short-tail parameters above. ChemEnzy, web, evidence, conditions,
  Program and enzyme sidecars are disabled for the reach arm.
- Each paper branch records `required_calls`, `actual_calls`, `selected_depth`,
  `selected_open_leaves`, `selected_solved` and `calls_exhausted`. A branch
  that stops below 25 without stock closure is rejected with
  `paper_policy_call_budget_not_exhausted`; it cannot enter short-tail or be
  reported as solved.
- The protocol now explicitly distinguishes node-wise ReactionJSON from the
  final complete RouteJSON admission.

## Live Traversiadiene smoke test after the fix

Run: `canary_runs/synthexfig1-001-paper-3x25-v2e`.

- `paper_reach=false`; no solved claim was emitted.
- The run terminated honestly at the new hard gate after 25 total model
  invocations and about 1,073 s. The three StrategyCards were generated, then
  node-wise Route Builder calls were observed on all three isolated AiZ trees;
  Critic/Editor also ran on the materialized branches.
- The branches stopped before the 25-call ceiling without stock closure. The
  old behavior would have retained those partial routes and continued to a
  misleading “unresolved but route-like” report. The new behavior reports an
  execution-contract failure, so no short-tail call is counted for this run.

This is a useful negative smoke result: it proves the new accounting and gate
are live. It does **not** yet measure the scientific SynthEx reach rate. The
next improvement is therefore Route Builder policy quality/continuation (why
the Codex ReactionJSON stream exhausts its local tree early), followed by a
fresh 3×25 run; it is not another audit or evidence pass.

## Tests

The targeted profile and AiZ integration tests pass (`126 passed` in
`tests/test_target_solver.py` and `tests/test_sequential_strategy_director.py`).

