# Bufotalin chemoenzymatic fusion formal run — v13

Date: 2026-08-20

## Outcome

The paid, paper-matched formal run completed normally and terminated as
`unresolved`; it was not paused and did not exhaust any configured resource
limit. It produced two target-rooted, materialized route skeletons, but neither
route reached the frozen ZINC + eMolecules stock. The paper-equivalent result is
therefore **unsolved**.

Milestones: B0=true, B1=true, B2=true, B3=false, B4=false, B5=false. Here B2
means host structural reaction validation. It is not exact substrate/enzyme
validation: the P450/whole-cell step remains an explicitly unproven execution
hypothesis.

## Frozen protocol

- Run ID: `bufotalin-chemoenzymatic-fusion-formal-v13`
- Model: `gpt-5.6-terra`, reasoning effort `medium`
- Search profile: `paper_synthex` / `synthex_matched`
- Strategy: three independent Codex branches; up to 25 sequential nodes per branch
- Local repair: up to six Editor rounds
- Native search: AiZynthFinder MCTS short tail, 500 iterations, depth 6,
  1,200 s timeout for every distinct stock-rejected open leaf
- Stock: frozen ZINC + eMolecules, full InChIKey identity, 39,478,827 unique members
- Stock SHA-256: `4d2f601ddd5af10b1c179ec583062d3ba3136553e285944d125e7b5ce19b5a65`

## Resource use

| Measure | Observed |
|---|---:|
| Total wall time | 1,486.042 s (24 min 46 s) |
| Model calls | 29 |
| Input tokens | 475,538 |
| Output tokens | 68,524 |
| Model wall time | 1,378.953 s |
| Accepted canonical expansions | 2 |
| Native short-tail searches | 2 |

The Director workload comprised six StrategyCard reports, sixteen
RetrosynthesisProposal reports, and seven ChemicalStrategyCritique reports.
Five proposal calls were local Editor revisions. Critic outcomes were five
`reject` and two `uncertain`. One final proposal call was an unsuccessful local
replan that consumed 15,131 input and 2,152 output tokens in about 45.7 s.

## Route assessment

### Route A — late P450/whole-cell hydroxylation

Retrosynthetic step: bufotalin to the corresponding advanced deoxy
steroid–pyrone precursor, with molecular oxygen explicitly bound as the source
of the new oxygen atom.

- The ReactionJSON edit replayed and passed host structural validation.
- The execution proposal includes a steroid-active P450/whole-cell panel,
  aerobic conditions, glucose-supported reducing-equivalent regeneration, and
  selectivity assays.
- No exact enzyme/host has been validated on this advanced substrate; site and
  facial selectivity, acetate stability, and competing oxidation remain open.
- The advanced deoxy precursor was absent from the frozen stock; its bounded
  AiZynthFinder short tail returned no stock-closed continuation.

This is a useful biological tailoring hypothesis, but not a practical synthesis
route by itself.

### Route B — P450 tailoring plus C9–C20 reductive annulation

The shared P450/whole-cell hydroxylation is preceded retrosynthetically by an
intramolecular reductive coupling of a tethered C9/C20 dibromide.

- Both ReactionJSON steps materialized into one connected target-rooted graph.
- Critic classified the annulation as `uncertain`, not accepted precedent: the
  constrained secondary-centre coupling may undergo dehalogenation,
  elimination, heteroaromatic reduction, or fail to close.
- Host reaction validation rejected the chemical edge because mapping alone did
  not establish a trusted transform or exact precedent.
- The doubly brominated advanced steroid precursor was absent from stock; the
  bounded short tail returned no continuation.

This route has more strategic content than Route A, but currently remains a
high-risk hypothesis whose precursor is harder, not easier, than the starting
point required for a useful stock-closed route.

### Rejected edited route

A P450 plus secondary-alkyl-boronate/2-bromo-alpha-pyrone Suzuki-type strategy
survived Critic as `uncertain`, but the host admission layer rejected its
materialization for `surplus_advanced_precursor_fragment`. This is evidence that
Editor/Critic were active and that a chemically worded proposal was not allowed
to become a disconnected or atom-inconsistent route.

## Final metrics

| Metric | Result |
|---|---:|
| Target-rooted routes | 2 |
| Materialized routes | 2 |
| Host reaction-validated routes | 1 |
| Exact biocatalytic validations | 0 |
| Condition-complete routes | 0 |
| Exact-evidence routes | 0 |
| Stock-closed routes | 0 |
| Paper reach | true |
| Paper-equivalent solved | false |

## Defects exposed and repaired

1. Shared reaction edges now retain every independent StrategyCard instead of
   losing one route-family binding during materialization.
2. Reaction validation now accepts a newly installed atom only when the
   ReactionJSON edit replays, an explicit donor/cosubstrate is present, and the
   mapper's new-atom inventory exactly matches the deficit. This fixed the O2
   hydroxylation case without weakening missing-donor rejection.
3. Every pending paper-matched leaf short tail now precedes any global/local
   replan. In v13 the second short tail eventually ran, but one replan had
   incorrectly jumped ahead of it; future runs no longer spend that call.
4. Biocatalytic-step contracts now count as physical biocatalytic steps in route
   summaries even when no legacy route-innovation annotation was emitted. This
   is display/accounting only and grants no enzyme proof.
5. A failed replan now settles the live stage marker instead of leaving a
   terminal report marked `running`.

Regression results: 131 scheduler/runtime/solver tests passed after the
short-tail priority repair; 35 focused scheduler, route-innovation, and failed
replan tests passed after final projection fixes.

## Conclusion

The fusion machinery is functioning: Codex proposed chemical and biological
steps, Editor/Critic revised and rejected candidates, ReactionJSON produced
connected canonical graphs, host validation distinguished structural replay
from exact enzyme proof, stock was checked against the paper-comparable oracle,
and both open leaves received bounded AiZynthFinder tails.

The negative result is now primarily chemical/search quality rather than a
missing execution stage. Both surviving ideas disconnect only to very advanced,
non-stock steroid intermediates. The next experiment should force each branch
to demonstrate monotonic precursor simplification and an early purchasable or
short-tail-reachable anchor before spending further node calls, while keeping
the P450 arm as one optional local transformation rather than the entire route.
