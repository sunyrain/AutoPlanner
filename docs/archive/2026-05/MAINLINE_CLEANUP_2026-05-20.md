# Mainline Cleanup - 2026-05-20

## Decision

Old ranker/CCTS/fallback work is now guarded archive material. It is retained
only for reproducibility and report audit. It should not be used as the default
runtime path, model claim, or next-stage training target.

## Current Mainline

- ChemEnzy native multi-step search.
- Product-route feasibility audit / material sanity filtering.
- Cascade verifier-first data and preference generation.
- Rule cascade verifier as a conservative route metric / hard-gate candidate.
  Unpartitioned ChemEnzy routes are treated as sequential stepwise syntheses by
  default, not as one-pot cascades.
- Future cascade-aware proposal/verifier work that does not require expert
  labels.

## Guarded/Frozen

- CCTS v0/v1/v2/v3 transition/runtimes.
- Route-pool LambdaRank / old route-pool ranker.
- Adjacent-step cascade pair scorer.
- Block-coherence and block-hard pack/scorer experiments.
- V4 product-value / action-source / provider-retrieval lineage.
- CBA v0 / reservoir / controller-v2 lineage.
- Expert CSV / LLM review fallback workflows.
  This includes strict-model review worklists/readiness/merge scripts.

Direct execution of these scripts now requires:

```bash
AUTOPLANNER_ALLOW_LEGACY_RESEARCH=1
```

The guarded archive manifest is:

```text
archive/code/frozen_research_2026-05-20/README.md
```

## Runtime Change

`scripts/run_chem_enzy_plan_for_web.py` no longer enables legacy cascade
action/source hooks by default. The Web runner now reports them as disabled
unless a request explicitly asks for legacy hooks and
`AUTOPLANNER_ALLOW_LEGACY_RESEARCH=1` is set.

This prevents old action-value/source-policy artifacts from silently affecting
new ChemEnzy native searches.

`cascade_planner/eval/run_v4_heldout_chem_enzy_pool.py` may still be used for a
plain native ChemEnzy pool, but its trace/action-value options are also guarded
by the same opt-in flag.

Verifier-first runtime tools:

- `scripts/rerank_routes_with_cascade_verifier.py` reranks existing route pools
  with rule verifier scores; learned verifier output defaults to annotation
  only.
- `scripts/compare_cascade_verifier_rerank_batch.py` compares original route
  ordering against rule reranking plus optional learned-verifier annotations.

Current promotion status: rule verifier is acceptable as a conservative gate
and Web metric. The Web request flag is `enable_rule_verifier_gate` (alias:
`cascade_verifier_gate`); it is off by default and filters only after every
route has received a verifier report.

Learned-verifier runtime support now lives in the main package:

```text
cascade_planner/cascade_verifier/features.py
cascade_planner/cascade_verifier/learned.py
```

The Web request flag `enable_learned_verifier_annotation` attaches learned
feasibility probabilities and failure-reason evidence to each route under
`metrics.learned_cascade_verifier`. It uses annotation-only policy and does not
affect route order, hard-gate behavior, or displayed route count. Missing or
unloadable model files are reported in `route_set_metrics.learned_verifier_annotation`
without failing the ChemEnzy search.

The learned verifier is calibrated but not promoted as a default reranker.
The default CLI policy is now `--learned-verifier-policy annotation_only`,
which writes learned feasibility probabilities and failure-reason evidence
without changing route order. Learned reranking remains an explicit ablation via
`calibrated_conservative` or `raw_score`.
The current calibrated model is:

```text
results/shared/cascade_verifier_mainline_20260521/learned_verifier_v4_30k_calibrated.joblib
```

Its recommended feasible threshold is `0.98`; at that threshold the 30K
perturbation test precision is `0.9828` and recall is `0.2511`. On the seven
statin presentation/report packages, calibrated learned reranking caused
`0` extra top1 changes beyond the rule verifier. Keep it as a high-confidence
annotation/gate candidate, not a ranking default.

Condition extraction has been consolidated for the rule and learned verifier
feature path. The verifier now reads conditions from direct step fields,
`condition_predictions`, nested `condition` / `step_conditions` payloads, and
`raw_metadata` condition/reagent containers before checking temperature, pH,
solvent, and enzyme-toxicity rules. The retrained artifact:

```text
results/shared/cascade_verifier_mainline_20260521/learned_verifier_v4_30k_condition_extraction.joblib
```

has the same 30K metrics as the calibrated model above (`0.98` recommended
threshold, test precision `0.9828`, recall `0.2511` at that threshold), so this
is a robustness/data-ingestion improvement rather than a model-lift claim.
Rechecking the static statin showcase gives `26/26` rule-feasible routes and
`0` learned top1 changes:

```text
results/shared/cascade_verifier_mainline_20260521/statin_static_showcase3_condition_extraction_matrix/matrix.md
```

The learned-verifier feature path was then made stage-aware so temperature and
pH conflict features follow the rule verifier's same-stage contract instead of
using only whole-route spans. Artifact:

```text
results/shared/cascade_verifier_mainline_20260521/learned_verifier_v4_30k_stage_aware.joblib
```

On the 30K perturbation pack this is a real offline improvement:

- test accuracy: `0.9094 -> 0.9651`
- reason micro F1: `0.9689 -> 0.9852`
- reason macro F1: `0.9653 -> 0.9821`
- at threshold `0.98`: test precision `0.9609`, recall `0.7577`

However, on the static statin showcase it causes one learned top1 change
(`atorvastatin`, `route-0013 -> route-0001`) without audit-rank improvement.
Decision: keep stage-aware learned verifier as a stronger offline annotation /
failure-reason model, but do not promote it as the default live reranker. The
default gate remains the rule verifier; learned verifier use remains
annotation-only by default, with conservative/experimental reranking available
only when explicitly requested.

Rechecking the static statin showcase with the stage-aware artifact in
annotation-only mode gives `26/26` rule-feasible routes and `0` learned extra
top1 changes:

```text
results/shared/cascade_verifier_mainline_20260521/statin_static_showcase3_stage_aware_annotation_only_matrix/matrix.md
```

ChemEnzy proposal-side fine-tuning status:

- DPO/LoRA is not locally ready; the vendor tree lacks a DPO loss and LoRA
  adapter entry.
- Supervised OpenNMT continue-training is locally runnable. A chosen-only
  supervised seed pack has now been extracted from verifier preferences; it
  uses only `chosen_cascade` routes and deliberately excludes every
  `rejected_cascade` from positives.
- Current chosen-only supervised inputs:

```text
results/shared/cascade_verifier_mainline_20260521/verifier_preference_chosen_seed_pack_v4_1477.json
results/shared/cascade_verifier_mainline_20260521/verifier_preference_chosen_seed_pack_v4_1477.md
results/shared/cascade_verifier_mainline_20260521/chem_enzy_onmt_corpus_chosen_v4_1477_smiles_token/manifest.md
results/shared/cascade_verifier_mainline_20260521/chem_enzy_supervised_adapter_readiness_chosen_v4_1477.md
```

The readiness manifest now reports
`ready_for_supervised_adapter_training_not_direct_dpo`: `29,079` verifier
preference pairs are available, `1,477` unique chosen seed routes were emitted,
the SMILES-token ONMT corpus contains `2,421` plain and `2,848` context step
examples, and direct DPO/LoRA remains blocked by missing vendor loss/adapter
code.
- First executed chosen-only plain ONMT smoke:

```text
results/shared/chem_enzy_adapter_mainline_20260521/chosen_v4_1477_token_plain_smoke_exec/experiment_manifest.md
```

It completed preprocess/train/eval and produced
`plain_chosen_adapter_low_lr_step_2.pt`, but exact-recall A/B shows no lift
over the native checkpoint on the 20-example smoke (`adapter_ab_decision:
hold_no_exact_recall_lift`). This confirms the training path works; it is not a
proposal-model promotion.
- Chosen-only context-mode checks:

```text
results/shared/chem_enzy_adapter_mainline_20260521/chosen_v4_1477_context_vocab_audit_against_old_extended.md
results/shared/chem_enzy_adapter_mainline_20260521/chosen_v4_1477_context_smoke_exec/experiment_manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/chosen_v4_1477_context_steps30_lr0001/experiment_manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/chosen_v4_1477_context_tiny20_overfit_steps200/experiment_manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/chosen_v4_1477_context_lr001_steps300_full_eval/experiment_manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/chosen_v4_1477_context_adapter_training_summary.md
```

The context corpus is compatible with the experimental extended-vocab
checkpoint (`context_src_oov_rate: 0.0`). Both executed context runs completed,
including the 30-step low-learning-rate run, but exact-recall A/B still shows
no lift. On the 30-step 50-example valid/test checks, native and adapter are
both `top1_exact=0` and `top5_exact=0`; the manifest decision is
`hold_no_exact_recall_lift`. Validation accuracy/perplexity improved locally
inside the trainer (`74.2553`, `7.21888`), but exact product-to-reactant recall
did not improve, so this checkpoint remains unpromoted.

A tiny-overfit diagnostic then confirmed that the context adapter path is
actually connected: training on 20 duplicated context rows for 200 steps moved
valid/test exact recall from native `0/0` to adapter top1/top5 `4/5` over the
same 20 rows. This rules out a broken pretokenized evaluation path.

The first useful full chosen-only context run is:

```text
results/shared/chem_enzy_adapter_mainline_20260521/chosen_v4_1477_context_lr001_steps300_full_eval/experiment_manifest.md
```

With `learning_rate=0.001`, `train_steps=300`, and full valid/test evaluation,
the adapter improves over the extended-vocab baseline:

- valid: native top1/top5 `0/0`, adapter `5/20` over `433` examples.
- test: native top1/top5 `1/1`, adapter `8/19` over `435` examples.

Decision: this is a real proposal-side learning signal from the chosen-only
context objective, but it is still not a live proposal-model promotion. Absolute
exact recall remains low and promotion still requires route-level search A/B
with no regression against native ChemEnzy.

Route-level A/B with the 300-step chosen-only context adapter:

```text
results/shared/chem_enzy_adapter_mainline_20260521/chosen_context_lr001_steps300_route_ab_limit5/route_ab_summary.md
```

On five `benchmark_v2_100` targets, a gated first-step context ONMT sidecar
using the new checkpoint produced proposals but did not improve route-pool
metrics over native ChemEnzy. Candidate/result GT reaction and reactant coverage
were unchanged, while average cascade search time increased from `0.0034s` to
`0.1506s`. Decision: hold route-level promotion. The next useful work is
proposal-quality improvement and conservative sidecar filtering, not blindly
promoting this checkpoint. A direct top-level proposal audit confirms the
sidecar returned proposals for every target but had `0` GT reaction/reactant
hits; after adding self-reaction filtering, the audit still reports `0` GT hits
with `avg_returned=2.2`.

Self-reaction noise audit and clean corpus:

```text
results/shared/cascade_verifier_mainline_20260521/chem_enzy_onmt_corpus_chosen_v4_1477_smiles_token_clean_noself/manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/chosen_v4_1477_context_clean_noself_summary.md
```

The original chosen-only corpus contained substantial identity/self-reaction
noise: `482` seed steps had the product also listed as a reactant, with `438`
pure self-reaction steps in the underlying seed pack. The corpus builder now
filters those steps by default. The rebuilt clean corpus has product-in-reactants
`0` across plain/context train/valid/test and remains compatible with the
extended context vocabulary (`context_src_oov_rate: 0.0`).

Training the same `lr=0.001`, `steps=300` context adapter on the clean corpus
keeps an offline top5 learning signal but does not solve top-level proposal
quality:

- clean valid: native top1/top5 `0/0`, adapter `0/10` over `362` examples.
- clean test: native top1/top5 `1/1`, adapter `5/15` over `363` examples.
- clean top-level proposal audit on the same 5 benchmark targets: `0` exact GT
  reaction hits, `0` target-step GT reaction hits, `0` GT reactant hits.

Decision: self-reaction filtering is a necessary data-quality fix and should
stay in the corpus builder, but proposal quality still needs a harder objective
or route-targeted training set before any route-level promotion benchmark.

Top-level-only clean corpus and adapter:

```text
results/shared/cascade_verifier_mainline_20260521/chem_enzy_onmt_corpus_chosen_v4_1477_smiles_token_clean_noself_toplevel/manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/chosen_v4_1477_context_clean_noself_toplevel_summary.md
```

This corpus keeps only step `0`, matching the current live sidecar policy that
uses context ONMT only at the first retrosynthetic step. It emits `1,235`
context examples and skips `1,688` non-top-level steps plus `92` self-reaction
steps. The adapter trained on this corpus (`lr=0.001`, `steps=300`) again shows
offline top5 exact-recall lift:

- top-level valid: native top1/top5 `0/0`, adapter `0/5` over `189` examples.
- top-level test: native top1/top5 `0/0`, adapter `4/9` over `194` examples.

However, the same 5-target top-level proposal audit still reports `0` exact GT
reaction hits, `0` target-step GT reaction hits, and `0` GT reactant hits.
Decision: top-level-only SFT aligns the training scope with the sidecar use
case, but chosen-only next-step likelihood is still insufficient for proposal
quality. The next training objective should use hard negatives or route-targeted
positive construction instead of just another SFT run.

No-expert hard-negative proposal preference packs have now been generated from
the clean top-level corpus:

```text
results/shared/chem_enzy_adapter_mainline_20260521/context_onmt_proposal_preferences_toplevel_clean_train.md
results/shared/chem_enzy_adapter_mainline_20260521/context_onmt_proposal_preferences_toplevel_clean_valid.md
results/shared/chem_enzy_adapter_mainline_20260521/context_onmt_proposal_preferences_toplevel_clean_test.md
```

They contain `2005` train, `444` valid, and `453` test pairwise preferences.
Rejected sides are rule-generated hard negatives (`self`, `drop_aux`,
`cross_swap`), not expert labels and not supervised positives. A consistency
audit confirms zero canonical chosen/rejected ties and zero malformed `self`
negatives across all three files. This is now the proposal-quality training
substrate for the next mainline experiment.

A first lightweight no-expert proposal preference scorer has been trained from
those packs:

```text
results/shared/chem_enzy_adapter_mainline_20260521/context_onmt_proposal_preference_scorer_toplevel_clean/context_onmt_proposal_preference_scorer_report.md
results/shared/chem_enzy_adapter_mainline_20260521/context_preference_scorer_toplevel_clean_audit_limit5.md
```

Offline preference metrics are strong for this hard-negative task: valid
pairwise accuracy `0.939189`, test pairwise accuracy `0.927152`, with valid/test
AUC around `0.90`. The scorer is now available as an opt-in
`ChemEnzyContextONMTProposalProvider` score/filter/rerank layer. However,
reranking the same 5-target top-level context-ONMT audit still yields `0` exact
GT reaction hits and `0` GT reactant hits. Decision: keep it as a
proposal-quality discrimination artifact, not a generator promotion. The
remaining mainline bottleneck is still candidate generation, not only candidate
ranking.

Route-level 5-target A/B with this scorer-reranked sidecar:

```text
results/shared/chem_enzy_adapter_mainline_20260521/context_preference_scorer_route_ab_limit5_summary.md
```

It leaves GT coverage unchanged from the native baseline
(`top_result_exact_reaction_in_pool=0.2`,
`top_result_gt_reactant_in_pool=0.8`) and changes `0` top routes. Average
cascade-search time rises from `0.0034s` to `0.168s`. This closes the current
hard-negative scorer experiment as an offline discriminator plus opt-in
diagnostic layer, not a promoted runtime scorer.

Training coverage audit:

```text
results/shared/chem_enzy_adapter_mainline_20260521/context_onmt_toplevel_training_coverage_benchmark_v2_100_train.md
results/shared/chem_enzy_adapter_mainline_20260521/context_onmt_toplevel_training_coverage_benchmark_v2_100_all.md
```

The clean top-level context corpus has `0` exact product matches and `0` exact
top-level GT reaction matches against `benchmark_v2_100`; this remains true
when train/valid/test are combined. All-split nearest-pair support reaches only
`14/100` targets at the `0.70` threshold, and `66/100` targets are classified as
`out_of_distribution`. This is stronger evidence for the current bottleneck:
chosen-only SFT is training on a top-level distribution that largely does not
cover the benchmark top-level proposal task. The next mainline training step
should therefore build route-targeted / benchmark-style positives or a larger
synthetic top-level proposal pack, not another reranker on the same candidate
pool.

Benchmark-style top-level positive diagnostics:

```text
results/shared/chem_enzy_adapter_mainline_20260521/benchmark_toplevel_proposal_generation_diagnostics.md
results/shared/chem_enzy_adapter_mainline_20260521/benchmark_v2_100_toplevel_onmt_smiles_token/manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/benchmark_v2_100_toplevel_context_lr001_steps200_eval/experiment_manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/benchmark_v2_100_toplevel_context_alltrain_overfit_steps300/experiment_manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/benchmark_v2_100_toplevel_plain_alltrain_overfit_steps300/experiment_manifest.md
```

The benchmark-derived top-level corpus contains only `81` usable positives.
It is useful as a diagnostic but too small for a promotion claim. A 70/15/15
context SFT gets valid top5 `0/12` and test top5 `1/13`. An all-train context
overfit still reaches only top1 `3/80`, top5 `5/80`; all-train plain fine-tuning
regresses from native top5 `10/80` to adapter top5 `9/80`. This means the
current ONMT continue-training path can improve token-level loss but does not
reliably generate complete benchmark-style reactant sets at this data scale.
The next mainline work should construct a substantially larger top-level
proposal-generation pack, likely synthetic/route-targeted, before repeating
adapter training.

External top-level proposal corpus diagnostic:

```text
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_onmt_smiles_token_150k/manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_150k_training_coverage_benchmark_v2_100_all.md
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_context_vocab_extended/write_report.md
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_150k_context_vocab_audit_after_extension.md
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_150k_context_lr0003_steps300_eval_pretok/experiment_manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_150k_context_lr0003_steps300_len200/experiment_manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_150k_context_lr0003_steps300_len200/top_level_proposal_audit_limit5.md
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_150k_context_lr0003_steps300_len200/top_level_gap_diagnostic_limit5.md
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_150k_context_lr0003_steps300_len200/prediction_error_valid200.md
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_150k_context_lr0003_steps300_len200/prediction_error_test200.md
```

The new external builder (`scripts/build_external_toplevel_onmt_corpus.py`)
combines USPTO50K, enzymatic_retro, and ecreact top-level positives. A
50K/source run emits `109692` context examples after dedupe. Coverage against
`benchmark_v2_100` improves materially over the clean chosen-only corpus:
`targets_with_exact_reaction=11/100`, `targets_with_exact_product=27/100`, and
`targets_with_near_pair_ge_0_70=29/100`; `46/100` targets remain
`out_of_distribution`. This supports external positives as a proposal-expansion
substrate, but still argues for route-targeted/synthetic cascade positives.

The external corpus exposed two necessary training gates:

- Vocab gate: source and target OOV must both be audited. Extending the shared
  ONMT vocab from `276` to `319` tokens clears source/target OOV to `0`.
- Length gate: vendored ONMT preprocess defaults to `src/tgt_seq_length=50`.
  The first 300-step smoke therefore trained only `8385/98600` train rows. A
  length audit shows `src/tgt<=200` would retain about `84101/98600` rows.

The short-sequence smoke is not a promotion result. Its 200-example exact-recall
check is valid top5 `0/200` for both native and adapter, and test top5
native `0/200` vs adapter `1/200`.

The len200 follow-up run fixes the length gate and uses GPU training:
`84101/98600` train rows, validation accuracy `87.3655`, and saved checkpoint
`external_toplevel_150k_len200_adapter_lr0003_step_300.pt`. Exact-recall is
still flat on the 200-example checks: valid top5 `0/200` for both native and
adapter, test top5 `0/200` for both. Decision: external proposal data is now a
credible coverage substrate and the training pipeline is technically sound, but
the current context SFT objective does not yet produce exact reactant-set
proposal gains. Do not promote this adapter. The next mainline step should
change the proposal objective/data form, e.g. route-targeted top-level positives,
multi-candidate generation training, or synthetic cascade packs, before route
level promotion testing.

Direct top-level proposal audit on the first 5 benchmark targets shows a small
candidate-pool gain but no promotion signal: all 5 targets return proposals,
exact GT reaction `0/5`, target-step GT reaction `0/5`, GT-reactant hit `1/5`.
The one hit is rank 4 for `O=c1[nH]c2ccccc2o1`, proposing the GT precursor
`Nc1ccccc1O` but missing the exact reaction details. This supports continuing
proposal-side work, but route-level promotion should wait until exact or
reactant hit rates are materially higher.

The gap diagnostic joins benchmark coverage and proposal audit results. For the
first 5 targets it reports `proposal_partial_reactant_hit=1`,
`generator_missed_known_reactant_side=2`, and `target_domain_gap=2`; no exact GT
reaction or exact target product exists in the external corpus for these five.
This means the next mainline data should be route-targeted or synthetic
cascade-derived top-level positives, with a reactant-set/reaction-completion
objective. More epochs on the same external context SFT objective are unlikely
to fix the current miss mode.

Prediction-error diagnosis on the len200 exact-recall outputs shows the
seq2seq generator is often far from the target reactant set, not just one
candidate away. On valid/test 200-row samples, top-k predictions have no
GT-reactant overlap for `161/200` and `159/200` rows, and contain at least one
invalid molecule for `109/200` and `117/200` rows. The next proposal objective
therefore needs legality/constraint handling and reactant-set completion, not
just longer training on the same context SFT target.

Canonical external top-level corpus and adapter check:

```text
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_onmt_smiles_token_canonical_150k/manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_onmt_smiles_token_canonical_150k/training_coverage_benchmark_v2_100.md
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_onmt_smiles_token_canonical_150k/context_vocab_audit_extended.md
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_canonical_150k_context_lr0003_steps300_len200/experiment_manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_canonical_150k_context_lr0003_steps300_len200/prediction_error_valid200.md
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_canonical_150k_context_lr0003_steps300_len200/prediction_error_test200.md
```

The corpus builders now canonicalize training product/reactant SMILES by
default and retain raw strings in metadata. The canonical 50K/source external
run emits `109679` context examples and clears the extended-vocab source/target
OOV gate (`0.0/0.0`). Benchmark coverage is essentially unchanged from the
non-canonical corpus: exact top-level reaction coverage is `10/100`, exact
product coverage is `25/100`, and near-pair coverage remains `29/100`.

The matched 300-step len200 adapter run is technically complete but not a
proposal-model promotion. Valid/test 200-row exact-recall A/B remains
`0/200` top5 for both native and adapter. Error diagnosis is also not improved:
valid/test rows with no target-reactant overlap are `164/200` and `158/200`,
and invalid top-k rows are `112/200` and `110/200`. Decision: canonicalization
is a necessary data hygiene fix for reproducibility, but it does not solve
reactant-set generation.

Reactant-set completion objective:

```text
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_canonical_150k_reactant_completion/manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_canonical_150k_reactant_completion_vocab_extended/write_report.md
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_canonical_150k_reactant_completion/context_vocab_audit_after_extension.md
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_canonical_150k_completion_context_lr0003_steps300_len250/experiment_manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_canonical_150k_completion_context_lr0003_steps300_len250/completion_diagnostic_valid200.md
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_canonical_150k_completion_context_lr0003_steps300_len250/completion_diagnostic_test200.md
```

`scripts/build_context_onmt_reactant_completion_corpus.py` converts the clean
external top-level positives into a candidate-conditioned completion task:
source = context tokens plus `<candidate>` and a rule-perturbed reactant side;
target = the complete canonical reactant side. Corruptions include `drop_one`,
`self`, `cross_swap`, and `empty`. This creates `415539` examples
(`373388/21331/20820` train/valid/test). The only new vocab token is
`<candidate>`; extending the checkpoint from `319` to `320` tokens clears
source/target OOV to `0.0/0.0`.

The first 300-step completion adapter also does not show exact top-k lift:
valid/test 200-row exact-recall remains `0/200` for both native and adapter.
The completion-specific diagnostic shows the model is not merely copying the
candidate side (`1/200` copy rows on both valid/test); the dominant failure is
still invalid or off-target generation. Valid/test rows with invalid top-k are
`136/200` and `129/200`, and rows with no target overlap are `169/200` and
`164/200`.

Decision: the completion data interface is useful and should be kept, but the
current OpenNMT seq2seq continuation is not enough. The next mainline proposal
work should add constrained/validity-aware decoding, a token-level legality
filter, or a different reactant-set model before spending more compute on the
same unconstrained ONMT generation objective.

Conservative validity filtering for generated proposals:

```text
cascade_planner/cascade_search/proposal_validity.py
scripts/filter_onmt_exact_recall_predictions.py
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_canonical_150k_context_lr0003_steps300_len200/exact_recall_valid200_filtered.md
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_canonical_150k_context_lr0003_steps300_len200/exact_recall_test200_filtered.md
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_canonical_150k_completion_context_lr0003_steps300_len250/exact_recall_valid200_filtered.md
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_canonical_150k_completion_context_lr0003_steps300_len250/exact_recall_test200_filtered.md
```

A shared proposal-validity layer now filters only conservative failure modes by
default: unparsable molecules, empty reactant sides, self-reactions, and
duplicate canonical reactant sides. Atom-count ratio filtering is available as
an explicit opt-in and is not enabled by default, because large terminal
fragments can be legitimate group-transfer or protecting-group contexts.

`ChemEnzyContextONMTProposalProvider` now applies this validity filter before
turning context-ONMT outputs into cascade actions. The exact-recall evaluator
and the post-hoc filter script both parse `<product>` from context-mode sources
so self-reaction filtering uses the actual product SMILES, not the full context
string.

Post-hoc filtering confirms the layer cleans the candidate pool but does not
create a proposal-model lift. On canonical top-level valid/test 200-row
samples, the adapter retains nonempty filtered proposals for `177/200` and
`174/200` rows while rejecting `342` and `351` raw predictions; filtered top-k
exact remains `0/200`. On completion valid/test, the adapter retains
`172/200` and `176/200` rows while rejecting `410` and `402` raw predictions;
filtered top-k exact also remains `0/200`. Decision: keep this as a runtime
candidate-pool hygiene layer, not as evidence of generator improvement.

Validity-aware over-sampling ablation:

```text
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_canonical_150k_context_lr0003_steps300_len200/exact_recall_valid200_beam15_filtered_eval_summary.json
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_canonical_150k_completion_context_lr0003_steps300_len250/exact_recall_valid200_beam15_filtered_eval_summary.json
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_canonical_150k_context_lr0003_steps300_len200/top_level_proposal_audit_limit5_rawx3.md
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_canonical_150k_context_lr0003_steps300_len200/top_level_proposal_audit_limit5_unfiltered.md
results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_canonical_150k_context_lr0003_steps300_len200/top_level_proposal_audit_limit5_rawx3_unfiltered.md
```

`ChemEnzyContextONMTProposalProvider` now supports raw over-sampling before
validity filtering through `raw_topk_multiplier` (default `3`). The provider
raises internal `topk/beam_size/n_best` to `requested_top_k *
raw_topk_multiplier`, filters invalid/self/duplicate candidates, and still
returns only the requested top-k. This is a decoding hygiene layer, not model
training.

On canonical top-level valid200, beam/topk `15` plus filtering increases
filtered nonempty rows to `198/200` but filtered exact remains `0/200`; on the
completion valid200 sample it increases filtered nonempty to `191/200` with
filtered exact still `0/200`. Raw beam15 also increases target-reactant overlap
in the unfiltered diagnostic (`55/200` rows with any GT-reactant overlap), but
at the cost of many invalid outputs (`169/200` rows with invalid top-k).

The front-five benchmark top-level audit does not improve. With the canonical
adapter, unfiltered topk5, rawx3 unfiltered, and rawx3 filtered all report
`0/5` exact GT reaction hits and `0/5` GT-reactant hits. The earlier `1/5`
partial reactant hit came from the non-canonical checkpoint and was not
recovered by this canonical run. Decision: keep raw over-sampling plus
filtering as a sidecar hygiene/default, but do not treat it as a benchmark
proposal-quality lift.

Known-legal corpus proposal-pool coverage baseline:

```text
cascade_planner/cascade_search/proposals.py::LegalCorpusProposalProvider
scripts/audit_legal_corpus_top_level_proposals.py
results/shared/chem_enzy_adapter_mainline_20260521/legal_corpus_constrained_provider_top100/audit.json
results/shared/chem_enzy_adapter_mainline_20260521/legal_corpus_constrained_provider_top100/audit.md
results/shared/chem_enzy_adapter_mainline_20260521/legal_corpus_constrained_provider_top100/legal_corpus_index.pkl
```

`LegalCorpusProposalProvider` is a constrained proposer, not a route retriever
or learned ranker. It indexes canonical external top-level corpus metadata and
returns only reactant sides that appeared in known corpus reactions, ranked by
exact product match and product-neighborhood Morgan/Tanimoto similarity. It
uses the same conservative proposal-validity filter as context-ONMT outputs;
atom-count ratio filtering remains disabled by default to avoid false positives
on group-transfer/protecting-group cases.

On full `benchmark_v2_100` with `topk=100` and the canonical external
context train/valid/test metadata, the legal-corpus provider returns `100`
legal candidates for every target. It recovers exact target-step GT reactions
for `18/100` targets and at least one GT reactant for `62/100` targets. Exact
hit ranks are `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 9, 13, 34, 45, 81, 86]`;
`11` exact hits come from exact-product corpus entries and `7` from nearest
product entries. This is a meaningful candidate-pool coverage lower bound and
is much stronger than unconstrained ONMT exact generation, but it is not a
route-quality, condition-compatibility, or promotion claim. Next mainline step:
use this legal candidate-pool signal to build constrained proposal generation
or retrieval-augmented expansion, then let cascade-aware verifier/search decide
conditions and route composition.

The audit script now accepts `--index-cache`; a cache smoke confirms subsequent
runs load the 109K-reaction RDKit fingerprint index from
`legal_corpus_index.pkl` (`loaded_from_cache=True`) instead of rebuilding it.
The cache includes a source-file signature; if a caller passes only a subset of
the corpus, the provider reports `cache_status=signature_mismatch` and builds an
in-memory index without overwriting the full cache. A full-corpus CLI benchmark
dry-run confirms the route-level runner can use the sidecar with
`cache_status=hit` and `indexed_reactions=109679`.

Route-level opt-in wiring:

```bash
PYTHONPATH=. python cascade_planner/eval/run_cascade_search_benchmark.py \
  --benchmark data/benchmark_v2_100.json \
  --output results/shared/chem_enzy_adapter_mainline_20260521/legal_corpus_constrained_provider_top100/runner_cli_dryrun_fullcache_limit1.json \
  --limit 1 \
  --dry-run \
  --cascade-max-depth 1 \
  --cascade-expansion-budget 4 \
  --cascade-result-limit 1 \
  --use-legal-corpus-proposals \
  --legal-corpus results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_onmt_smiles_token_canonical_150k/context.train.meta.jsonl \
  --legal-corpus results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_onmt_smiles_token_canonical_150k/context.valid.meta.jsonl \
  --legal-corpus results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_onmt_smiles_token_canonical_150k/context.test.meta.jsonl \
  --legal-corpus-topk 5 \
  --legal-corpus-candidate-pool-size 32 \
  --legal-corpus-index-cache results/shared/chem_enzy_adapter_mainline_20260521/legal_corpus_constrained_provider_top100/legal_corpus_index.pkl
```

Legal-only route-level sanity:

```text
results/shared/chem_enzy_adapter_mainline_20260521/legal_corpus_constrained_provider_top100/runner_legal_only_dryrun_full100_top100_depth1.json
```

This dry-run disables real ChemEnzy planning, enables only the legal-corpus
sidecar at `topk=100`, and limits cascade search to depth 1. It is a search-layer
sanity check, not a promoted planner. The full `benchmark_v2_100` run reports
`exact_reaction_in_route_pool=0.11`, `gt_reactant_in_route_pool=0.26`,
`result_exact_reaction_in_pool=0.13`, and `result_gt_reactant_in_pool=0.43`;
all 100 targets use the cached 109679-reaction index with `cache_status=hit`.
This is lower than the direct top-level audit's `18/100` exact target-step
candidate-pool hit because route search and result limits keep only part of the
top100 sidecar pool. Decision: legal-corpus proposals should remain an opt-in
candidate-pool sidecar. The next mainline work is score calibration/search
fusion with verifier state, not a route-level promotion claim.
- `scripts/run_chem_enzy_onmt_adapter_experiment.py` now normalizes evaluation
  split arguments. If no `--eval-split` is provided it evaluates `valid`; if
  callers pass explicit splits, duplicates are removed so future manifests do
  not report repeated `valid` rows.
- Reproducible experiment runner:
  `scripts/run_chem_enzy_onmt_adapter_experiment.py`
- Current dry-run manifest:
  `results/shared/chem_enzy_adapter_mainline_20260521/plain_runner_low_lr/experiment_manifest.md`
- Smoke artifact:
  `results/shared/chem_enzy_adapter_mainline_20260521/README.md`
- Plain supervised adapter grid:
  `results/shared/chem_enzy_adapter_mainline_20260521/plain_supervised_adapter_grid_summary.md`
- Tokenized plain adapter grid:
  `results/shared/chem_enzy_adapter_mainline_20260521/token_plain_adapter_grid_summary.md`
- Context vocab audit:
  `results/shared/chem_enzy_adapter_mainline_20260521/context_vocab_audit_smiles_token.md`
- Context vocab extension:
  `results/shared/chem_enzy_adapter_mainline_20260521/context_vocab_extended/write_report.md`
- Context adapter grid:
  `results/shared/chem_enzy_adapter_mainline_20260521/context_adapter_grid_summary.md`
- Important correction: ChemEnzy ONMT checkpoint behavior matches SMILES-token
  source formatting better than the earlier single-character corpus. Token-mode
  native exact recall is stronger on the same split, but plain
  product-to-reactant continue-training still shows no stable gain over the
  native checkpoint.
- Direct context-mode continue-training was initially blocked because context
  tokens were out-of-vocabulary for the native checkpoint and the vendored
  OpenNMT trainer loads `checkpoint['vocab']` when `-train_from` is used.
  The experimental extension tool now expands the shared ONMT vocab from 200 to
  276 tokens and makes context source OOV rate 0.
- A small context-prefix adapter grid now shows a real learning signal:
  best test top5 exact recall is 9/435 versus 1/435 for the untrained
  vocab-extended checkpoint. This is still not a live proposal-model promotion,
  because absolute recall is low and the live ChemEnzy wrapper currently
  supplies only product SMILES, not cascade context source strings.
- Runtime now has an experimental opt-in tokenizer flag:
  `chem_enzy_onmt_tokenizer` / `AUTOPLANNER_CHEMENZY_ONMT_TOKENIZER`. Default
  remains `char` until route-level A/B confirms the `token` path improves live
  ChemEnzy search.
- Route-level tokenizer A/B helper:
  `scripts/compare_chem_enzy_onmt_tokenizer_ab.py`
- Current smoke:
  `results/shared/chem_enzy_adapter_mainline_20260521/tokenizer_route_ab_smoke/tokenizer_ab_summary.md`.
  On `CCO` both `char` and `token` solved with 7 displayed routes; this proves
  the opt-in path works but does not promote token mode.
- Current benchmark3 A/B:
  `results/shared/chem_enzy_adapter_mainline_20260521/tokenizer_route_ab_benchmark3/tokenizer_ab_quality_summary.md`.
  On three compact benchmark targets, both modes solved all targets; token had
  one higher route-count case and one lower route-count case. Quality counts
  are also mixed: one target gains one multistep/ge3 route, another loses one
  multistep/feasible route. It changes the route pool but is not a route-level
  promotion signal yet.
- Experimental live context-ONMT sidecar provider:
  `ChemEnzyContextONMTProposalProvider`.
  Runtime smoke:
  `results/shared/chem_enzy_adapter_mainline_20260521/context_onmt_provider_smoke/smoke_result.md`.
  Route-level smoke:
  `results/shared/chem_enzy_adapter_mainline_20260521/context_onmt_route_ab_smoke/summary.md`.
  The provider loads the context-prefix checkpoint and returns probability-scale
  proposal scores, but route-level metrics were unchanged on the current
  two-target smoke. Recursive use inflated search expansion, so benchmark
  runner defaults gate it to early-step augmentation when explicitly enabled.
  It remains an opt-in sidecar, not a promoted proposal model.
- Any adapter checkpoint remains sidecar-only until full valid/test exact
  recall and route-level proposal recall beat the native checkpoint without
  regression.

## Allowed Reproduction Mode

Historical reports may still be reproduced by setting the legacy opt-in
environment variable and running the old script directly. New experiments should
not depend on that mode unless a current decision document re-promotes the line.
