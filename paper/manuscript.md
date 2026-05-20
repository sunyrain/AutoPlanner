# AutoPlanner: Condition- and Compatibility-Aware Chemoenzymatic Cascade Retrosynthesis with Structured Route Boards and LLM Priors

## Abstract

Computer-aided synthesis planning has made substantial progress on single-step
retrosynthesis, tree search, stock-aware termination, and interactive route
generation. However, chemoenzymatic cascades require a different notion of route
feasibility: a route must not only be retrosynthetically reachable, but also
plausible under cross-step temperature, pH, enzyme, solvent, cofactor, and
operation-mode constraints. We introduce AutoPlanner, a condition- and
compatibility-aware cascade planning framework built around a structured
CascadeBoard route representation. AutoPlanner combines an order-agnostic
skeleton inpainter, chemical and enzymatic candidate generators, route-level
condition diagnostics, enzyme-evidence accounting, learned route scoring, and
optional LLM-derived strategic priors. A newly added cascade-constrained
AO*-style search prototype treats fixed steps as anchors and uses stock,
condition, compatibility, and enzyme evidence as search-control terms. Unlike
LLM-agent retrosynthesis systems
that may use language models as route generators, AutoPlanner treats language
models only as prior and critique layers; all factual claims are grounded in
route JSON exports and deterministic validators. On a frozen 100-target cascade
benchmark, the current type-aligned beam planner achieves a 100% plan rate, 69%
filled type-sequence GT@5, 27% exact-reaction route-pool recovery, 51%
ground-truth-reactant route-pool recovery, 67% strict stock solve, and 94%
condition/compatibility screen success. A stock-prioritized mode raises strict
stock solve to 96%, at the cost of lower exact-reaction recovery. The main
scientific contribution is a route-level feasibility interface for
chemoenzymatic cascade planning, not a generic claim of superior solve rate on
standard retrosynthesis benchmarks.

## 1. Introduction

Retrosynthesis planning systems typically optimize synthetic reachability: given
a target molecule, they search for disconnections whose terminal reactants are
available in stock. This formulation is appropriate for many organic synthesis
benchmarks, but it is incomplete for chemoenzymatic cascade design. A cascade
route can be chemically reachable yet experimentally implausible if adjacent
steps require incompatible pH windows, if a metal catalyst or organic solvent
inactivates an enzyme, if cofactors conflict, or if a one-pot operation mode is
assumed where sequential addition or intermediate isolation is required.

Recent systems have expanded retrosynthesis planning in several directions.
AiZynthFinder remains a strong open-source search-centered planner. ASKCOS
provides a broad synthesis planning suite with one-step models, feasibility and
condition predictors, and interactive planning tools. Syntheseus emphasizes
reproducible benchmarking of retrosynthesis search. Hybrid synthetic-enzymatic
planning has also been studied, including systems that combine enzymatic and
organic transformations. Meanwhile, LLM-based systems such as tool-using
chemistry agents and LLM-guided retrosynthesis methods have shown that language
models can help with strategy, constraints, tool orchestration, and route
critique.

AutoPlanner targets a narrower gap. We focus on route-level cascade feasibility
for chemoenzymatic routes, using curated cascade catalysis data and a structured
route-board intermediate representation. This choice changes the planning
object. The planner must represent not only molecules and reactions, but also
reaction class, EC evidence, temperature, pH, candidate provenance,
compatibility risk, and operation mode.

Our contributions are:

1. A CascadeBoard route representation that stores slot-level molecules,
   reaction candidates, enzyme annotations, conditions, fixed fields, evidence,
   and diagnostics.
2. A live skeleton-to-fill planning path combining an OA-ARM skeleton inpainter,
   RetroChimera, Enzyformer, v3 enzymatic retrieval, and EnzExpand.
3. A route JSON export contract that supports benchmark reporting and future LLM
   critique without relying on ungrounded free text.
4. Deterministic route-level metrics for filled-route status, stock solve,
   condition-window success, enzyme-evidence coverage, and cascade
   compatibility.
5. A conservative LLM-prior interface that can propose strategy priors and route
   critiques while being forbidden from inventing reaction facts.

## 2. Related Work

### 2.1 Search-Centered Retrosynthesis

AiZynthFinder is an open-source retrosynthesis planner built around one-step
models, search, stock checks, and route scoring. Its later versions expanded
model support, ONNX inference, scoring, clustering, and search options. ASKCOS
offers a broad platform for synthesis planning, including multiple one-step
models, condition recommendation, reaction feasibility, outcome prediction, and
interactive workflows. Syntheseus provides a common framework for benchmarking
retrosynthesis algorithms and highlights that evaluation choices can change
apparent rankings.

These systems provide essential baselines, but their core metric is usually
reachability or route recovery, not cascade experimental compatibility.

### 2.2 Hybrid Organic-Enzymatic Planning

Prior work has combined organic and enzymatic transformations in multistep
planning. This is close to AutoPlanner's domain, but hybrid transformation
coverage alone is not sufficient for cascade design. The key distinction is
whether the system treats cross-step condition compatibility and operation mode
as first-class route objects.

### 2.3 LLM and Agentic Chemistry Systems

Tool-using chemistry agents such as ChemCrow and autonomous experimental
systems such as Coscientist show that LLMs can orchestrate tools and propose
scientific workflows. Recent LLM retrosynthesis systems explore route
generation, strategic search priors, and agent-as-judge evaluation. These
systems motivate LLM integration, but also expose a safety issue: language
models may produce unsupported chemical claims.

AutoPlanner therefore uses LLMs only as priors and critics. Reaction validity,
stock status, enzyme evidence, and condition compatibility remain owned by
structured tools and validators.

## 3. Data

AutoPlanner uses curated cascade catalysis records. The trainable view currently
contains 3028 steps, 1563 adjacent step pairs, and 1748 cascades. The loader
reports 2550 steps with temperature, 1673 with pH, and 2864 with solvent fields.
Common transformation classes include oxidation, reduction, acylation,
racemization, amination, C-C coupling, and hydrolysis. EC1 coverage is dominated
by oxidoreductases, hydrolases, and transferases.

This dataset differs from ordinary single-step reaction corpora because it
contains route-level context: operation mode, pairwise mode, compatibility
labels, issue types, mitigation strategies, and cascade conditions. Coverage is
still incomplete, especially for yield, enzyme sequence evidence, and failed
experiments. These gaps directly shape the next data requirements.

## 4. Method

### 4.1 CascadeBoard Representation

A CascadeBoard is a linear route board composed of slots. Each slot may contain:

- product, main reactant, auxiliary reactants, and reaction SMILES
- reaction type
- EC number and enzyme identifier
- temperature, pH, solvent, and catalyst fields
- candidate source and scores
- fixed fields for user constraints

Route-level objects store quality, risk, compatibility diagnostics, edit
history, and global constraints. This representation can express fixed starting
materials, fixed reaction classes, preferred enzymatic steps, condition windows,
and advanced anchor constraints.

### 4.2 Skeleton Generation

AutoPlanner first predicts a route skeleton using an order-agnostic autoregressive
masking model. The skeleton contains reaction types, EC1 classes, temperature,
pH, and compatibility/operation-mode diagnostics. Skeleton generation is
separated from molecular fill so that route-level priors and constraints can be
represented before candidate enumeration.

### 4.3 Molecular Fill

Given a skeleton, AutoPlanner fills concrete reactions using:

- RetroChimera for chemical retrosynthesis
- Enzyformer for enzymatic retrosynthesis
- v3 retrieval over curated enzymatic precedents
- EnzExpand as a template fallback

Each candidate retains provenance. This is essential for later evidence scoring
and for preventing LLM-generated text from becoming unverified chemistry.
The current fill stage uses bounded slot-level beam expansion. This replaced an
earlier skeleton sampling plus linear greedy fill path. The beam can be run in a
type-aligned mode, which keeps candidate choices close to the predicted
skeleton reaction class, or in a stock-aware mode, which prioritizes buyable
terminal reactants.

We also implement a cascade-constrained AO*-style prototype (`cc_aostar`). It
keeps skeletons as route priors, but expands molecule/reaction alternatives in
a best-first AND-OR graph. Fixed user steps are treated as anchor constraints:
the search may explore alternative prefixes or suffixes, but route candidates
must satisfy anchored reaction, reactant, type, or EC fields when those fields
are specified. Candidate priority combines one-step model score, skeleton type
alignment, terminal stock, adjacent T/pH compatibility, and structured enzyme
evidence. This mode is implemented and smoke-tested, but the full 100-target
comparison shows that it is currently a stock- and speed-biased prototype, not
the best exact-recovery mode. The headline exact-recovery result in this
manuscript remains the type-aligned beam run.

### 4.4 Route Export and Metrics

The route export contract serializes each route into JSON with slot-level
fields, candidate provenance, condition diagnostics, enzyme evidence, stock
status when enabled, and explanation fields. This contract is the only allowed
input to route critique.

Current deterministic metrics include:

- `filled_route`
- `candidate_source_counts`
- `strict_stock_solve`
- `condition_window_success`
- `enzyme_evidence_coverage`
- `cascade_compatibility_success`
- exact-reaction route-pool recovery
- ground-truth reactant recovery
- candidate-pool recovery and candidate rank
- route edit distance

Cascade compatibility is reported as structured dimension checks over available
fields: solvent risk, metal/enzyme conflict, oxidant/reductant conflict,
cofactor cross-talk, oxygen/water sensitivity, and suggested operation mode.
These are deterministic screens, not experimental success labels.

### 4.5 LLM Prior and Critique Layer

The LLM layer is deliberately outside the chemistry source-of-truth path. It can
provide:

- route-mode priors
- reaction-type priors
- enzyme-class priors
- condition-risk priors
- route critiques grounded in exported JSON

It cannot:

- invent reaction SMILES as accepted facts
- mark stock as available
- invent enzyme availability, yield, pH, or temperature
- override RDKit, candidate provenance, stock checks, or evidence validators

When no LLM API is available, AutoPlanner uses a deterministic motif-based prior
fallback.

In the current implementation, priors are used as skeleton ordering and
budget-control signals. `rerank` keeps the prior as a soft skeleton-ordering
term. `stock_aware` adds terminal stock preference during beam filling and final
route sorting. `critic_control` expands prior-ranked skeletons iteratively and
stops only when deterministic route checks find acceptable filled, stock,
condition-compatible, and cascade-compatible routes or the budget is exhausted.
The prior can change search order and budget use, but it cannot create a
reaction candidate, set stock availability, or change any route validator.
The same boundary applies to `cc_aostar`: LLM priors may reorder skeletons and
increase search budgets, but accepted chemistry still comes only from the
candidate generators and deterministic validators.

## 5. Experiments

### 5.1 Smoke Verification

We verified the current system through:

- dependency consistency with `python -m pip check`
- package compilation
- v2/v3 loader execution
- skeleton prediction
- mock CLI JSON output
- live CLI JSON output
- one-target live benchmark execution
- two-target `cc_aostar` smoke benchmark execution
- full100 `cc_aostar` equal-benchmark merge against the type-aligned beam run
- generator-bottleneck diagnosis over the frozen 100-target artifact
- DeepSeek provider check with deterministic fallback disabled

The one-target smoke run is not used as a performance claim. It verifies that
the live route export and benchmark plumbing work. The provider check verifies
that the LLM path can resolve to a real provider without storing API keys or
using provider output as chemistry truth.
The `cc_aostar` smoke run verifies integration, while the full100 merge
measures its current tradeoff against the type-aligned beam baseline.

### 5.2 Full Live Benchmark Protocol

The live benchmark is run on the frozen 100-target set with top-5 route outputs.
The benchmark is sharded across two GPUs and merged after execution. It reports
skeleton and filled-route metrics separately:

- `skeleton_type_GT@1`
- `skeleton_type_GT@5`
- `filled_route_any`
- `filled_type_GT@1`
- `filled_type_GT@5`
- `terminal_GT_reactant_in_top5`
- `strict_stock_solve_any`
- `condition_window_success_any`
- `cascade_compatibility_success_any`
- exact reaction recovery and route edit distance
- candidate-pool recovery and candidate rank
- candidate source counts

For LLM-prior comparison, we also run equal-budget prior modes:

- `none`: no-agent baseline
- `deterministic`: RDKit motif prior
- `deepseek`: DeepSeek structured prior

All three settings use the same sampled skeleton budget, filled-route budget,
candidate generators, stock checker, and route validators.

The current full live run is executed by:

```bash
CUDA_VISIBLE_DEVICES=0 python -m cascade_planner.cascadeboard.live_benchmark \
  --bench data/benchmark_v2_100.json --output results/v2/live_benchmark_shard0.json \
  --shard-index 0 --num-shards 2 --n-results 5 --device cuda

CUDA_VISIBLE_DEVICES=1 python -m cascade_planner.cascadeboard.live_benchmark \
  --bench data/benchmark_v2_100.json --output results/v2/live_benchmark_shard1.json \
  --shard-index 1 --num-shards 2 --n-results 5 --device cuda
```

After both shards finish:

```bash
python -m cascade_planner.cascadeboard.live_benchmark \
  --merge results/v2/live_benchmark_shard0.json results/v2/live_benchmark_shard1.json \
  --output results/v2/live_benchmark_full.json --check-stock
```

### 5.3 Current Results

The dual-GPU live benchmark completed on the frozen 100-target set. The merged
headline artifact is `results/v2/live_benchmark_beam_type_aligned_full.json`.
The benchmark used top-5 routes per target and post-processed terminal
reactants with the ZINC stock checker.

| Metric | Value |
|---|---:|
| Plan rate | 100% |
| Skeleton type GT@1 | 57% |
| Skeleton type GT@5 | 69% |
| Filled type GT@1 | 57% |
| Filled type GT@5 | 69% |
| Terminal GT reactant in top-5 | 25% |
| Exact reaction in route pool | 27% |
| Candidate exact reaction in pool | 37% |
| Exact full-route reaction match | 1% |
| GT reactant in route pool | 51% |
| Candidate GT reactant in pool | 72% |
| Strict stock solve, any top-5 route | 67% |
| Condition-window success, any top-5 route | 94% |
| Cascade compatibility screen, any top-5 route | 94% |
| Average time per target | 9.266 s |

The search and scoring progression is:

| Run | Filled type GT@5 | Exact reaction | Candidate exact reaction | GT reactant | Candidate GT reactant | Stock solve | s/target |
|---|---:|---:|---:|---:|---:|---:|---:|
| Greedy exact-metric baseline | 73% | 6% | n/a | 29% | n/a | 46% | 3.528 |
| Candidate-pool instrumentation | 73% | 6% | 19% | 29% | 53% | 46% | 3.596 |
| Beam rerank | 53% | 22% | 34% | 44% | 70% | 61% | 9.311 |
| Beam stock-aware | 53% | 18% | 36% | 43% | 70% | 96% | 9.509 |
| Beam type-aligned | 69% | 27% | 37% | 51% | 72% | 67% | 9.266 |
| CC-AO* prototype | 62% | 26% | 31% | 54% | 64% | 91% | 2.011 |

The CC-AO* row is a full100 merge from two CUDA shards with `n_results=5`,
`skeleton_samples=5`, and `search_budget=20`. It improves strict stock solve
over the type-aligned beam run (91% versus 67%), improves route-pool
GT-reactant recovery (54% versus 51%), and is faster in this configuration,
but it lowers filled type GT@5, candidate-pool coverage, and
condition/compatibility screen success. Exact-reaction route-pool recovery is
near parity but still slightly lower (26% versus 27%). This means the current
AO* prototype should be treated as a search-control baseline to optimize, not
as a replacement for the type-aligned beam.

Candidate provenance across all exported route slots:

| Source | Count |
|---|---:|
| RetroChimera | 324 |
| Enzyformer | 175 |
| v3 retrieval | 398 |
| EnzExpand | 16 |

Per-domain filled type GT@5:

| Domain | n | Filled type GT@5 | Strict stock solve | Condition screen | Compatibility screen |
|---|---:|---:|---:|---:|---:|
| all_chemical | 25 | 92.0% | 56.0% | 100.0% | 100.0% |
| all_enzymatic | 42 | 54.8% | 64.3% | 95.2% | 95.2% |
| chemoenzymatic | 28 | 67.9% | 75.0% | 85.7% | 85.7% |
| hybrid_mimetic | 1 | 100.0% | 100.0% | 100.0% | 100.0% |
| whole_cell_biocatalytic | 4 | 75.0% | 100.0% | 100.0% | 100.0% |

These results should be interpreted carefully. `filled_type_GT@5` is stricter
than skeleton-only recovery because it requires a filled route, but it is still
a type-sequence metric. The exact-recovery metrics are much harsher and remain
low. The compatibility metric is a deterministic route-export screen based on
available T/pH, solvent, catalyst, cofactor, oxygen/water, filled-slot, and
enzyme-evidence fields; it is not an experimental success label.

### 5.4 Route-Level Case Studies

A successful all-enzymatic example is target
`N[C@@H](CCCP(=O)(O)O)C(=O)O`. The top route matches the ground-truth
type sequence, passes the condition and cascade-compatibility screens, and has
a strict stock-terminating route under the ZINC stock checker. Both filled
steps come from Enzyformer and carry EC1 oxidoreductase intent. The exported
route terminates at `CCC[C@H](N)C(=O)O`, with predicted step temperatures near
28-31 C and pH near 7.8-8.1. This example illustrates the intended use of
CascadeBoard: the route is not only filled, but also exposes candidate
provenance, enzyme class, stock status, and condition compatibility in one
route artifact.

A contrasting all-enzymatic example is target
`CCCCNC1=C(C)C(=O)C=C(C)C1=O`. The planner recovers the route type sequence
and passes the deterministic condition and compatibility screens, but the
strict stock metric fails because the terminal reactants include `C#N` and
`C1=CCCCCCC1`. The route critic therefore rejects the route with a
`stock_dead_end` finding and suggests an alternative route-mode search. This is
the desired behavior: the route is not reported as solved merely because the
skeleton and step types look plausible.

### 5.5 LLM Prior and Critic-Control Ablation

We ran a small equal-budget prior and critic-control ablation to verify the LLM
integration path. This is not the main 100-target benchmark; it is a deployment
and claim-boundary test. The GPU run used 10 frozen targets, top-5 filled
routes, 10 sampled skeletons per target, and the same critic-control budget.
DeepSeek was actually called for all 10 targets and did not fall back to the
deterministic prior.

| Prior mode | Prior source count | Filled GT@5 | Exact rxn | Candidate exact rxn | GT reactant | Stock | s/target |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0 | 80% | 20% | 40% | 50% | 100% | 15.018 |
| deterministic | 10 | 80% | 20% | 50% | 60% | 100% | 17.684 |
| DeepSeek | 10 | 80% | 20% | 50% | 60% | 100% | 20.989 |

The result is mixed and still not publication-grade evidence of LLM lift. Priors
increased candidate exact-reaction coverage and route-pool GT-reactant recovery
on this small set, but did not improve filled type GT@5, exact selected-route
recovery, or stock solve, and DeepSeek was slower. Therefore, the present claim
is safety and workflow readiness plus weak evidence that priors can alter
candidate coverage, not route-level performance improvement.

### 5.6 External Step Baselines

We also collected local single-step expansion baselines on 3028 annotated
cascade steps. These baselines are not full route-planning results and do not
evaluate cascade compatibility fields.

| Baseline | top-1 | top-5 | top-10 | top-50 |
|---|---:|---:|---:|---:|
| AiZynthFinder policy expansion | 8.3% | 12.9% | 15.2% | 16.7% |
| Syntheseus MEGAN | 10.0% | 14.6% | 17.0% | 23.4% |
| Syntheseus RootAligned | 7.8% | 14.4% | 16.0% | 19.9% |

The purpose of this comparison is limited: generic single-step expanders recover
only a minority of annotated cascade step reactants, and they do not cover the
cascade-condition fields that AutoPlanner exports. It does not establish
generic retrosynthesis superiority.

## 6. Discussion

AutoPlanner's main design choice is to make cascade feasibility explicit. This
has several consequences.

First, standard retrosynthesis solve rate is not the only relevant metric. A
chemoenzymatic route should also report condition-window success, operation
mode, enzyme evidence, and cross-step compatibility.

Second, the benchmark reveals different bottlenecks by domain. All-chemical
targets are relatively strong under type-sequence recovery, while all-enzymatic
targets show lower stock solve despite high condition-screen success. This
suggests that enzymatic candidate generation and stock-terminating search remain
the main route-completion bottlenecks.

The exact-recovery metrics sharpen this diagnosis. Type-sequence recovery is
not enough: the route can have the right transformation labels while still
missing the actual ground-truth reactions and starting materials. The
type-aligned beam raises exact-reaction route-pool recovery from 6% to 27% and
GT-reactant route-pool recovery from 29% to 51%, but exact full-route recovery
is still only 1%.

A generator-bottleneck diagnosis on the same full100 artifact shows why search
alone cannot solve the problem. In 63% of targets, no exact ground-truth
reaction appears in the exported candidate pools. In 35% of targets, a
ground-truth reactant appears while the exact reaction is still absent,
indicating salts, cofactors, stereochemistry, missing auxiliary reactants, or
reaction-normalization/generation gaps. In 10% of targets, an exact candidate
is present but not selected into the route. Thus the next route-recovery gains
must combine better candidate generation with better search.

The CC-AO* diagnosis is even more direct: exact candidates are absent from the
candidate pools for 81% of targets, while exact-candidate selector misses drop
to 1%. In its current configuration, AO* is not mainly failing because it
chooses the wrong exact candidate; it is searching a lower-coverage pool and
penalizing condition/compatibility more often. The next algorithmic work should
therefore add generator expansion and compatibility-aware repair rather than
only changing the route selector.

Third, LLMs are useful but must be constrained and evaluated honestly. The
current DeepSeek integration successfully produces structured priors without
becoming a chemistry source of truth, but the small equal-budget ablation shows
only weak candidate-coverage lift and no route-level lift. This supports the
design boundary: LLM priors are promising for search control, but they should
not be claimed as a performance driver until they improve exact route recovery,
stock solve, or compatibility outcomes at equal budget.

Fourth, data quality is central. The strongest next improvements are not larger
language models but better cascade annotations: more 3+ step cascades, failed
examples, alternative routes for the same target, UniProt/cofactor evidence,
condition variants, and drug-intermediate cascades.

## 7. Limitations

The current system has several limitations:

- The benchmark's filled GT metric is type-sequence based, not exact reaction
  recovery.
- Exact full-route recovery is still weak: 1% on the current 100-target
  type-aligned beam run.
- Candidate generation remains the primary bottleneck.
- The first full100 CC-AO* run is useful but not stronger overall: it improves
  stock solve, GT-reactant recovery, and speed, but regresses filled type GT,
  candidate-pool coverage, and compatibility screens relative to type-aligned
  beam.
- Enzyme evidence is now exported as structured fields when present, including
  UniProt accession/status, organism, sequence length, cofactor, substrate
  similarity, condition match, and literature precedent. Coverage remains
  incomplete until the full snapshot is regenerated with the UniProt enrichment
  path that can now populate sequence, tax ID, protein-existence, Rhea, and
  cofactor fields.
- Condition and cascade compatibility are currently deterministic screens and
  should be calibrated against more empirical cascade outcomes.
- LLM priors have not yet improved equal-budget search in the current small
  deployment ablation beyond weak candidate-coverage changes.

## 8. Conclusion

AutoPlanner reframes chemoenzymatic retrosynthesis as condition- and
compatibility-aware cascade planning. Its core contribution is a structured
route-board representation and factual route export layer that make experimental
feasibility visible to planners, benchmarks, and future LLM critics. This
direction avoids competing as another generic LLM retrosynthesis agent and
instead targets the route-level constraints that make cascade catalysis
scientifically distinct.

## References

- AiZynthFinder 4.0: https://link.springer.com/article/10.1186/s13321-024-00860-x
- ASKCOS 2025: https://arxiv.org/abs/2501.01835
- Syntheseus: https://pubs.rsc.org/en/content/articlelanding/2024/fd/d4fd00093e
- Hybrid synthetic/enzymatic planning: https://www.nature.com/articles/s41467-022-35422-y
- ChemEnzyRetroPlanner: https://www.nature.com/articles/s41467-025-65898-3
- ChemCrow: https://www.nature.com/articles/s42256-024-00832-8
- Coscientist: https://www.nature.com/articles/s41586-023-06792-0
- Synthegy: https://arxiv.org/abs/2503.08537
- LARC: https://arxiv.org/abs/2508.11860
- AOT*: https://arxiv.org/abs/2509.20988
