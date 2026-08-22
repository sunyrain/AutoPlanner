# SynthEx matched case 004 — paper-budget result (v35)

Date: 2026-08-19  
Target: `opaque SynthEx five-target case 004`  
Target SMILES: `CCOC(=O)C1(O)c2cc3c(cc2C(=O)C1(C)O)C1(CC(=O)C[C@@H](C)O1)O[C@H](C)C3`

## Executive result

The frozen v35 run is **paper-equivalent solved under the bound local stock oracle**.
It produced one target-rooted, six-edge route whose six leaves all hit the same bound
eMolecules stock. The corrected panel summary therefore records `B4=true` and
`paper_equivalent_solved=true`.

This is not yet a SynthEx paper-stock-comparable result. The run used the local
23,081,629-member eMolecules canonical-SMILES index, whereas the paper metric uses
the authors' 39,684,411-member combined ZINC + eMolecules stock keyed by full
InChIKey. The proper interpretation is therefore:

> solved for this target and this bound stock; not evidence for a paper-comparable
> solved rate.

The stricter AutoPlanner scientific status remains `unresolved`, with disposition
`stock_closed_proof_open`: exact sources, complete conditions, and strict reaction
proof are not closed. This does not negate the paper-equivalent topological result;
it is an independently reported axis.

## Frozen matched protocol

The local paper extraction records the following SynthEx search contract:

- 3 independent strategies;
- at most 25 LLM policy expansions per strategy (75 policy calls);
- Critic/Editor repair for at most 6 rounds;
- up to 3 retries and a 600 s timeout for an individual model call;
- for every non-stock open leaf, a short AiZynthFinder tail with depth 6,
  500 iterations, and 1,200 s timeout;
- paper solved means that at least one connected target-rooted route has every leaf
  in the same combined ZINC + eMolecules stock.

AutoPlanner's matched operational envelope was frozen at 120 model invocations,
1,200,000 input tokens, 200,000 output tokens, 70,200 s aggregate model time, and
86,400 s emergency run time. The aggregate time ceilings are our conservative
orchestration bounds derived from the per-call protocol; they are not values reported
as aggregate wall time by the paper. The panel used one worker and did not apply the
obsolete ten-task cutoff.

## Outcome table

| Axis | Result | Interpretation |
|---|---:|---|
| B0 blind input | pass | Target-only blind case contract retained |
| B1 target-rooted route | pass | 2 target-rooted materialized routes observed |
| B2 host validation | pass | 1 route passed the standard host credibility milestone |
| B3 exact multi-source evidence | fail | 0 exact-evidence and 0 exact-procedure routes |
| B4 one-stock closure | pass | 1 strict stock-closed route; all 6 selected leaves hit |
| B5 configured proof acceptance | fail | Exact proof/conditions remain open |
| Paper-equivalent solved | **yes** | 1 existential target-rooted stock-closed route |
| Paper-stock comparable | **no** | Local stock identity differs from the paper stock |

Route accounting at the fixed cutoff:

- 1 candidate route;
- 2 canonical materialized / target-rooted routes;
- 1 selected route and 1 stock-closed route;
- 1 standard host-validated route, but 0 strict-host-validated complete routes;
- 0 condition-complete, exact-evidence, exact-procedure, or configured-complete routes.

`B2=true` must not be read as wet-lab validation. Several short-tail reactions still
lack L2 proof, which is why `strict_host_validated_route_count=0` and B3/B5 remain
false.

## Actual route construction

Codex supplied the root strategic edit as late-stage ester installation:

`target ethyl ester <- CCBr + corresponding complex carboxylic acid`

The forward hypothesis was carboxylate O-alkylation. It carried a model-predicted
condition hypothesis (`cesium carbonate, bromoethane, DMF, 20–50 °C`) and passed
deterministic ReactionJSON replay, but it has no exact source and remains
hypothesis-only.

The complex non-stock acid leaf then triggered the matched short-tail search. One
ChemEnzy/AiZynthFinder provider call returned a five-step tail in 168.438 s after 35
iterations; host materialization admitted all five proposed edges. Stitching that
tail to the Codex root produced the final six-edge route.

The six terminal stock hits were:

1. `CCBr`
2. `C[Si](C)(C)C=[N+]=[N-]`
3. `COC(=O)C(C)=O`
4. `COC(=O)c1cc(O)ccc1C`
5. `Cc1ccc(S(=O)(=O)O)cc1`
6. `O=C([O-])[O-]`

The five short-tail steps are template/model proposals, not literature-backed
procedures. Their low aggregate route score (`2.5149e-05`) and missing conditions
must not be hidden by the B4 success.

## Cost and latency

| Resource | Observed | Frozen limit |
|---|---:|---:|
| Model invocations | 64 | 120 |
| Input tokens | 1,184,086 | 1,200,000 |
| Output tokens | 150,005 | 200,000 |
| Model wall time | 10,142.734 s | 70,200 s |
| Total run wall time | 10,331.661 s (2 h 52 min 12 s) | 86,400 s |
| Accepted expansions | 6 | 96 |
| Short-tail calls | 1 | bounded per open leaf |

Time to first structural route was 10,142.989 s; first standard host-valid route was
10,144.031 s; B4 arrived at 10,331.661 s. The run stopped naturally near the input
token cap rather than because of the old task-count or 30-minute cutoff.

Efficiency is still poor: 64 Codex calls and almost the full 1.2 M input budget
yielded only one accepted Codex root expansion before the short-tail supplied five
more edges. This single-case solve verifies pipeline continuity, not efficient search.

## Conditions, evidence, and enzyme interpretation

The paper-matched arm intentionally disables condition enrichment, condition
prediction sidecars, enzyme assignment, and the enzyme sidecar. Consequently:

- condition-complete routes: 0;
- exact evidence/source records: 0;
- enzyme assignments: 0.

The root-step condition text is a Codex hypothesis only. The five short-tail steps
have empty catalyst, solvent, temperature, and pH fields. No conclusion about the
value of AutoPlanner's enzyme capability can be drawn from this arm; enzyme support
requires a separate matched ablation after the chemical benchmark result is frozen.

## Faults found and removed

Three control defects materially affected this experiment series:

1. **Total-task/model-wall starvation.** The runner confused the standalone 1,800 s
   AiZynthFinder baseline and the 1,200 s per-leaf tail with the complete LLM workflow.
   The paper profile now separates per-tail, aggregate model, and emergency run walls.
2. **Sequential-plan byte-budget rejection.** A 240,000-byte single-child response
   guard was incorrectly applied to the host-compiled output of many bounded sequential
   calls. That discarded the initial three-branch plan in v35. The guard now remains
   active for single-call plans but does not reject a host-assembled sequential plan;
   individual child outputs and total tokens remain bounded.
3. **Operational/scientific status conflation.** The old panel summary discarded B4
   and paper-equivalent metrics when strict B5 was false, producing a misleading
   `incomplete`/zero-score summary. A valid frozen projection now marks the benchmark
   observation operationally `completed`, while `scientific_status=unresolved` and
   `accepted_under_configured_policy=false` remain explicit.

The v35 chemistry was not rerun after defects 2 and 3: the target report and frozen
trajectory remain unchanged. Only the derived summary was rebuilt from those frozen
artifacts after the reporting fix. Future runs receive the corrected initial-plan
handling.

## Bottom line and next valid experiment

This run answers the immediate workflow question positively: the intended sequence
`Codex strategic edit -> materialize/validate/stock -> short-tail search -> stitch ->
one-stock closure` works end-to-end within the matched resource envelope.

It does not establish that AutoPlanner outperforms SynthEx. A defensible head-to-head
still requires (1) the exact combined ZINC + eMolecules full-InChIKey stock oracle and
(2) a frozen multi-target cohort. The highest-value next experiment is a small
multi-target rerun with the corrected sequential-plan handling, followed by the full
paper-stock cohort; evidence, condition, and enzyme enrichment should be scored as
separate post-B4 axes.

## Authoritative artifacts

- Frozen target report: `D:/Autoplanner/canary_runs/synthexfive-004-paper-budget-v35/runs/opaque SynthEx five-target case 004/target-only-solve-report.json`
- Corrected derived panel summary: `D:/Autoplanner/canary_runs/synthexfive-004-paper-budget-v35/panel-summary.json`
- Human-readable panel summary: `D:/Autoplanner/canary_runs/synthexfive-004-paper-budget-v35/panel-summary.md`
- Short-tail provider result: `D:/Autoplanner/canary_runs/synthexfive-004-paper-budget-v35/runs/opaque SynthEx five-target case 004/chemenzy-v4-guided-42197a8509f7-result.json`
- Local SynthEx source extraction: `docs/evaluation/evidence/synthex_2608_07454/source_bundle.json`

