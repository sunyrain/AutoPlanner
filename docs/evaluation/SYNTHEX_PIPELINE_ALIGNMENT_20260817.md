# SynthEx pipeline alignment — 2026-08-17

## Outcome

AutoPlanner's `synthex_matched` path now separates strategic selection from
route construction. The former R4 path asked one model response to choose a
strategy, draw precursors, and write graph edits; any disagreement erased the
whole hypothesis. The new path persists the StrategyCard first and retries only
the failed ReactionJSON materialization.

## Paper-to-host mapping

| SynthEx paper component | AutoPlanner implementation after this change | Status |
| --- | --- | --- |
| Strategy Generator, three strategies | three blind `StrategyCardReport` calls; each compares 3–5 high-level alternatives | aligned |
| Route Builder, continuous expansion | three isolated branches, up to 25 one-node Route Builder calls per branch | aligned |
| ReactionJSON is the structural representation | model returns ordered edits; host derives canonical precursor structures and completes ordinary edited valence | aligned, provisional public profile |
| Critic and Editor preserve the key strategy | independent forward critic plus up to six host-validated route-local repair events; the StrategyCard survives failed materialization | semantically aligned; loop scheduling differs |
| Analyst route ranking | strategic score now uses canonical root structures and edit identity instead of model-declared complexity labels | stronger host-side measurement |
| Short-tail completion | every distinct open leaf receives depth 6 / 500 iterations / 1,200 s under the matched profile | aligned |
| Solved endpoint | all leaves must hit one frozen stock | aligned in logic, not yet stock-equivalent |

Primary paper: <https://arxiv.org/html/2608.07454v1>

## Cached R4 counterfactual replay

Source: `results/synthex_figure1_traversiadiene_canary_20260816_r4`.

- Cached Route Builder calls: 9.
- Calls with a deterministically replayable edit program under the new host
  authority: 7.
- Calls where the model-drawn precursor exactly matched the replayed structure:
  0.
- True edit failures: 1 invalid bond delta and 1 missing edit program.

Therefore the previous run's dominant failure was not lack of a strategic idea:
seven of nine calls contained a replayable topology edit, but the old two-authority
contract discarded all seven because the separately drawn precursor disagreed.
This replay is a counterfactual admission test, not a new solved result or a live
performance claim.

## Live matched canary result

Target: opaque SynthEx Figure 1 target 001 (traversiadiene structure). The R6
Codex phase used `gpt-5.6-terra` at medium effort and reached the fixed 1,800 s
model cutoff after 12 calls (210,252 input tokens; 22,525 output tokens). It
produced three independent StrategyCards and five replayable ReactionJSON steps.

Two host integration defects discovered by the canary were fixed before scoring:
per-step ReactionJSON digests had contaminated StrategyCard identity, and frontier
materialization had dropped the authoritative strategy/edit payload. A zero-model
replay after those fixes produced B1=true, three target-rooted materialized
skeletons and five edges, but left four stock-open leaves.

The corrected guided-tail continuation then ran exactly one native ChemEnzy search
for each of those four leaves using the same frozen stock, depth 6, 500 iterations,
1,200 s timeout, seed 0 and a one-route host portfolio. All four searches returned
host-admitted stock-closed proposals and stopped early; total native-search wall
time was 56.172 s. The six returned tail steps materialized without ingestion
rejection. Final topology metrics were:

- B1 global multi-route: true.
- B4 stock boundary: true.
- canonical stock-closed routes: 3.
- target-rooted distinct skeletons: 6.
- model calls in the continuation: 0; native-search units: 4.
- B2 host-validated routes: false; only 1/6 new tail edges passed the current
  generic reaction verifier.
- B3 exact evidence and B5 configured scientific acceptance: false.

Artifact: `results/synthex_figure1_r6_guided_tail_v4/target-validation-fork-report.json`.
This is a positive topology/stock-closure canary under the bound eMolecules index,
not an experimentally supported synthesis claim.

The matched host budget is now frozen at 120 model invocations, 1,200,000 input
tokens, 200,000 output tokens and 1,800 seconds. The extra invocation headroom is
explicitly reserved for the Codex Critic/Editor phase and route-local repairs;
the benchmark manifest carries the same values. This is a declared host-policy
extension, not evidence that the paper's compute accounting has been reproduced.

## Remaining comparability blockers

1. The paper uses exact full-InChIKey membership in a 39,684,411-member combined
   ZINC + eMolecules stock. The current head-to-head protocol is bound to a
   23,081,629-member canonical-SMILES eMolecules index, so its solve rate is not
   paper-equivalent.
2. The paper's same-backbone Critic/Editor loop and AutoPlanner's independent
   critic plus host-validated event repair do not have identical compute accounting;
   both must be reported separately.
3. The current generic reaction verifier rejects five chemically ambitious tail
   edges even though atom mapping succeeds. Topology solve rate and reaction-valid
   solve rate must remain separate until this gap is evaluated on a larger panel.
4. One positive live canary is insufficient for a performance claim; the matched
   panel must be expanded after the exact stock is available.

## Control disposition

| Control | Class | Decision |
| --- | --- | --- |
| StrategyCard generation | K1 scientific policy | separate and persist before route construction |
| Model-drawn precursor equality | K3 duplicate authority | remove from route admission |
| ReactionJSON replay | K0 structural invariant | keep as sole structure writer |
| Edited-atom valence completion | K0 deterministic normalization | apply generically to touched atoms; explicit H operation overrides |
| Reaction proof, evidence, conditions | K2 delivery credibility | report separately; do not block topology exploration |
| Exact paper stock | benchmark comparability | required before a paper-equivalent solve-rate claim |

Launch decision: `matched_topology_pilot_ready`. Scientific-proof and exact-paper
comparability claims remain blocked by reaction validation and stock mismatch.
