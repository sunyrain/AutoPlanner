# Scripts

Active repository utilities only.

| Script | Purpose |
|---|---|
| `audit_strategic_disconnections.py` | Audit curated strategic-disconnection source files for ID uniqueness, family coverage, evidence traceability, source domains, and compliance-gated entries. |
| `query_strategic_disconnections.py` | Query the merged strategic-disconnection source layer by free-text or `family_id`. |
| `download_brenda.py` | Download or refresh BRENDA condition data inputs. |
| `monitor_autoplanner_web.py` | Poll the local WebUI service, CUDA status, queued jobs, and latest output/rejected artifacts. |
| `run_autoplanner_web_waitress.py` | Start the Waitress-backed local WebUI service used for collaborator testing. |
| `run_chem_enzy_plan_for_web.py` | Run one ChemEnzy native route search from a WebUI JSON request and emit Web-compatible JSON. |
| `reaudit_route_pool.py` | Refresh product/condition audit metadata for an exported route-pool JSON after audit-rule changes. |
| `render_linear_route_schemes.py` | Render top routes as forward, paper-style synthesis schemes with condition-audit markers. |
| `render_route_trees.py` | Render route topology as continuous Graphviz/RDKit trees. |
| `render_route_figures.py` | Render appendix-style per-step route figures. |
| `summarize_statin_depth20_routes.py` | Regenerate the formal statin depth-20 product-audit and strict enzyme-route summary from final benchmark rows. |
| `run_chem_enzy_smoke.py` | Run or dry-run the external ChemEnzyRetroPlanner core-search baseline and write normalized JSON. |
| `run_route_tree_gold_smoke.py` | Run the current AutoPlanner `route_tree` baseline on the same v3 gold smoke targets used by ChemEnzy. |
| `setup_chem_enzy_runtime.sh` | Download/unpack the ChemEnzy runtime under `/root/autodl-tmp` without filling the root conda envs directory. |
| `setup_chem_enzy_vendor.sh` | Clone/update the ignored ChemEnzyRetroPlanner vendor checkout without running heavy model setup. |
| `setup_enzyformer.sh` | Prepare the external Enzyformer checkout/checkpoints expected by wrappers. |
| `build_cascade_perturbation_pack.py` | Build rule-generated cascade perturbation examples for verifier training. |
| `train_cascade_verifier_from_pack.py` | Train/evaluate the learned cascade verifier from perturbation packs. |
| `build_cascade_verifier_preference_pack.py` | Build verifier-derived route preference pairs without expert labels. |
| `build_supervised_seed_pack_from_verifier_preferences.py` | Extract chosen-only supervised seed routes from verifier preferences. |
| `build_chem_enzy_cascade_onmt_corpus.py` | Convert supervised seed routes into ChemEnzy OpenNMT plain/context corpora. |
| `build_context_onmt_proposal_preference_pack.py` | Build no-expert top-level proposal preference pairs with rule-generated hard negatives. |
| `build_context_onmt_reactant_completion_corpus.py` | Build no-expert reactant-set completion corpora by adding a corrupted candidate side to context-ONMT sources. |
| `build_benchmark_toplevel_onmt_corpus.py` | Build benchmark-style top-level GT ONMT corpora for proposal-generation diagnostics. |
| `build_external_toplevel_onmt_corpus.py` | Build larger external top-level ONMT proposal corpora from USPTO50K, enzymatic_retro, and ecreact sources. |
| `check_chem_enzy_dpo_readiness.py` | Audit whether ChemEnzy supports direct DPO/LoRA or only supervised adapter training. |
| `audit_chem_enzy_onmt_context_vocab.py` | Check context-corpus OOV compatibility with an ONMT checkpoint. |
| `run_chem_enzy_onmt_adapter_experiment.py` | Dry-run or execute a reproducible ChemEnzy OpenNMT supervised adapter experiment with preprocess/train/exact-recall manifests. |
| `audit_context_onmt_top_level_proposals.py` | Directly audit context-ONMT top-level proposals against benchmark GT reactions/reactants. |
| `audit_legal_corpus_top_level_proposals.py` | Audit a constrained known-legal reaction corpus provider as a proposal-pool coverage baseline. |
| `audit_condition_prediction_attribution.py` | Attribute condition warnings by target, route, source, domain, and step issue for benchmark route rows. |
| `audit_context_onmt_training_coverage.py` | Audit whether clean context-ONMT training rows cover benchmark top-level GT steps. |
| `diagnose_top_level_proposal_gap.py` | Join coverage and proposal audits to classify whether misses are data coverage gaps, generator misses, or partial reactant hits. |
| `diagnose_onmt_prediction_errors.py` | Classify ONMT exact-recall prediction failures by invalid molecules, GT-reactant overlap, molecule-count mismatch, and similarity. |
| `diagnose_reactant_completion_predictions.py` | Diagnose reactant-completion predictions by corruption type, invalid molecules, candidate copying, and target overlap. |
| `filter_onmt_exact_recall_predictions.py` | Recompute exact-recall JSONs after conservative proposal validity filtering without rerunning ONMT translation. |
| `rerank_routes_with_cascade_verifier.py` | Apply rule verifier scores and optional learned annotation to an existing route pool. |
| `compare_cascade_verifier_rerank_batch.py` | Compare original route order against rule-verifier reranking/annotation across route packages. |
| `compare_chem_enzy_onmt_tokenizer_ab.py` | Route-level A/B smoke for ChemEnzy ONMT `char` vs `token` tokenizer modes through the Web runner. |

Verifier-first route gating/reranking:

```bash
# Web-compatible ChemEnzy run with conservative rule-verifier filtering enabled.
PYTHONPATH=. python scripts/run_chem_enzy_plan_for_web.py \
  --input request.json \
  --output plan.json
```

Set `"enable_rule_verifier_gate": true` in `request.json` to hide routes that
the rule cascade verifier marks infeasible. The default is false, so existing
route pools are not changed unless the request opts in. Unpartitioned ChemEnzy
routes are treated as sequential stepwise syntheses.

Set `"enable_learned_verifier_annotation": true` to attach learned verifier
probabilities and reason evidence to each Web route under
`metrics.learned_cascade_verifier`. This uses annotation-only policy: it does
not change ranking, route count, or the rule gate. The shared runtime utilities
are in `cascade_planner/cascade_verifier/features.py` and
`cascade_planner/cascade_verifier/learned.py`.

Current calibrated learned verifier:

```text
results/shared/cascade_verifier_mainline_20260521/
  learned_verifier_v4_30k_calibrated.joblib
  learned_verifier_v4_30k_calibrated_report.md
  package_matrix_calibrated/matrix.md
```

Use `--learned-verifier-policy annotation_only` for the mainline path. This
writes learned feasibility probabilities and failure-reason evidence without
changing route order. `calibrated_conservative` and `raw_score` are ablation
policies only; learned probability sorting is not a promoted default.

ChemEnzy supervised adapter smoke:

```text
results/shared/chem_enzy_adapter_mainline_20260521/README.md
results/shared/chem_enzy_adapter_mainline_20260521/plain_runner_low_lr/experiment_manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/token_plain_adapter_grid_summary.md
results/shared/chem_enzy_adapter_mainline_20260521/context_vocab_audit_smiles_token.md
results/shared/chem_enzy_adapter_mainline_20260521/context_vocab_extended/write_report.md
results/shared/chem_enzy_adapter_mainline_20260521/context_adapter_grid_summary.md
```

This confirms the local OpenNMT continue-training path runs from the native
ChemEnzy checkpoint. It does not show route/proposal improvement yet; the
low-learning-rate smoke preserves nonempty generation, while exact recall on
the 20-example validation smoke remains 0.

Reproducible dry-run manifest for the next adapter check:

```bash
PYTHONPATH=. python scripts/run_chem_enzy_onmt_adapter_experiment.py \
  --output-dir results/shared/chem_enzy_adapter_mainline_20260521/plain_runner_low_lr \
  --eval-split test \
  --eval-limit 20
```

Use `--translate-tokenizer token` when evaluating SMILES-token corpora built
with `build_chem_enzy_cascade_onmt_corpus.py --tokenizer smiles_token`.
Add `--execute` only when intentionally running preprocess/train/evaluation in
the ChemEnzy runtime. The manifest is not a promotion artifact; promotion still
requires full valid/test comparison and route-level proposal recall improvement
over the native checkpoint.

Current result: token-mode native evaluation is stronger than the earlier
single-character corpus, but plain product-to-reactant continue-training is not
promoted. Context-mode training is now technically unblocked by an experimental
shared-vocab checkpoint extension; the small context-prefix grid shows a
learning signal. A live `ChemEnzyContextONMTProposalProvider` can now construct
context-conditioned ONMT source strings during cascade search, but route-level
smoke results do not show a promotion signal yet:

```text
results/shared/chem_enzy_adapter_mainline_20260521/context_onmt_provider_smoke/smoke_result.md
results/shared/chem_enzy_adapter_mainline_20260521/context_onmt_route_ab_smoke/summary.md
```

Keep this provider opt-in and gated to early-step sidecar proposal augmentation
until a larger route-level benchmark beats the native checkpoint without
regression.

Chosen-only supervised adapter input from verifier preferences:

```text
results/shared/cascade_verifier_mainline_20260521/verifier_preference_chosen_seed_pack_v4_1477.md
results/shared/cascade_verifier_mainline_20260521/chem_enzy_onmt_corpus_chosen_v4_1477_smiles_token/manifest.md
results/shared/cascade_verifier_mainline_20260521/chem_enzy_supervised_adapter_readiness_chosen_v4_1477.md
```

This is the current mainline training input for ChemEnzy continue-training:
`chosen_cascade` only, `rejected_cascade` excluded from positives, SMILES-token
ONMT corpus ready. It is not DPO; the readiness manifest still reports direct
DPO/LoRA blocked by missing vendor loss/adapter code.

Executed chosen-only plain ONMT smoke:

```text
results/shared/chem_enzy_adapter_mainline_20260521/chosen_v4_1477_token_plain_smoke_exec/experiment_manifest.md
```

This verifies preprocess/train/eval works on the chosen-only corpus. The A/B
summary is `hold_no_exact_recall_lift`, so the adapter remains unpromoted.

Executed chosen-only context ONMT checks:

```text
results/shared/chem_enzy_adapter_mainline_20260521/chosen_v4_1477_context_vocab_audit_against_old_extended.md
results/shared/chem_enzy_adapter_mainline_20260521/chosen_v4_1477_context_smoke_exec/experiment_manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/chosen_v4_1477_context_steps30_lr0001/experiment_manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/chosen_v4_1477_context_tiny20_overfit_steps200/experiment_manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/chosen_v4_1477_context_lr001_steps300_full_eval/experiment_manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/chosen_v4_1477_context_adapter_training_summary.md
```

The chosen-only context corpus has zero source OOV against the experimental
extended-vocab checkpoint and the 2-step/30-step training runs complete.
However, the 30-step valid/test exact-recall A/B remains flat
(`top1_exact=0`, `top5_exact=0` for native and adapter on both 50-example
checks).

A tiny-overfit diagnostic shows the context path is connected: 20 duplicated
rows trained for 200 steps move from native `0/0` to adapter top1/top5 `4/5`.
The stronger full chosen-only context run (`lr=0.001`, `steps=300`) improves
the complete splits:

- valid: native top1/top5 `0/0`, adapter `5/20` over 433 examples.
- test: native top1/top5 `1/1`, adapter `8/19` over 435 examples.

This is a useful training signal, not a live proposer promotion. Route-level
search A/B is still required before using the adapter outside sidecar
experiments.

Clean no-self chosen-only corpus:

```text
results/shared/cascade_verifier_mainline_20260521/chem_enzy_onmt_corpus_chosen_v4_1477_smiles_token_clean_noself/manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/chosen_v4_1477_context_clean_noself_summary.md
results/shared/cascade_verifier_mainline_20260521/chem_enzy_onmt_corpus_chosen_v4_1477_smiles_token_clean_noself_toplevel/manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/chosen_v4_1477_context_clean_noself_toplevel_summary.md
```

The corpus builder now filters self-reaction steps where the product appears in
the reactants. Rebuilding the chosen-only corpus removes `482` noisy steps and
leaves product-in-reactants at `0` across emitted plain/context splits. The
clean context adapter keeps a top5 offline signal but still has `0` top-level GT
reaction/reactant hits on the 5-target proposal audit, so this is a data-quality
fix rather than a promotion.

`--step-scope top_level` builds a first-step-only corpus for the current
context sidecar policy. The top-level clean adapter also improves offline top5
exact recall, but still has `0` GT reaction/reactant hits on the same top-level
proposal audit. Treat chosen-only SFT as a baseline; the next proposal-quality
work needs hard negatives or route-targeted positives.

No-expert proposal preference packs from the clean top-level corpus:

```text
results/shared/chem_enzy_adapter_mainline_20260521/context_onmt_proposal_preferences_toplevel_clean_train.md
results/shared/chem_enzy_adapter_mainline_20260521/context_onmt_proposal_preferences_toplevel_clean_valid.md
results/shared/chem_enzy_adapter_mainline_20260521/context_onmt_proposal_preferences_toplevel_clean_test.md
```

These packs contain rule-generated hard negatives for proposal-quality
training/evaluation, not expert labels and not supervised positives:

- train: `852` source examples, `2005` preference pairs.
- valid: `189` source examples, `444` preference pairs.
- test: `194` source examples, `453` preference pairs.

The builder skips canonical chosen/rejected ties, and the generated files pass
a consistency audit with zero canonical chosen/rejected equality and zero bad
`self` negatives. This is the current mainline substrate for the next
proposal-quality objective after the chosen-only SFT baseline.

First no-expert proposal preference scorer:

```text
results/shared/chem_enzy_adapter_mainline_20260521/context_onmt_proposal_preference_scorer_toplevel_clean/context_onmt_proposal_preference_scorer_report.md
results/shared/chem_enzy_adapter_mainline_20260521/context_preference_scorer_toplevel_clean_audit_limit5.md
```

The scorer learns the rule-generated preference task offline:

- valid pairwise accuracy: `0.939189`, AUC `0.904386`.
- test pairwise accuracy: `0.927152`, AUC `0.906198`.

It is wired into `ChemEnzyContextONMTProposalProvider` as an opt-in
score/filter/rerank layer. In the first 5-target top-level proposal audit,
preference reranking still gives `0` exact GT reaction hits and `0` GT reactant
hits. Decision: useful hard-negative discrimination signal, but not a generator
promotion. The remaining blocker is proposal generation: the correct reactants
are still absent from the returned context-ONMT candidate pool.

Route-level 5-target A/B with the scorer enabled:

```text
results/shared/chem_enzy_adapter_mainline_20260521/context_preference_scorer_route_ab_limit5_summary.md
```

The scorer-reranked sidecar leaves GT coverage unchanged relative to native
baseline (`top_result_exact_reaction_in_pool=0.2`,
`top_result_gt_reactant_in_pool=0.8`) and changes `0` top routes, while average
cascade-search time rises from `0.0034s` to `0.168s`. This confirms the scorer
is not a route-level promotion artifact.

Training coverage audit for the clean top-level corpus:

```text
results/shared/chem_enzy_adapter_mainline_20260521/context_onmt_toplevel_training_coverage_benchmark_v2_100_train.md
results/shared/chem_enzy_adapter_mainline_20260521/context_onmt_toplevel_training_coverage_benchmark_v2_100_all.md
```

Against `benchmark_v2_100`, the clean top-level corpus has `0` exact product
matches and `0` exact top-level GT reaction matches, even when train/valid/test
are combined. All-split nearest-pair support covers only `14/100` targets at
the `0.70` threshold; `66/100` targets are classified as
`out_of_distribution`. This explains why SFT and preference reranking did not
recover benchmark reactants: the training substrate is not a benchmark-style
top-level proposal corpus.

Benchmark-style top-level proposal diagnostics:

```text
results/shared/chem_enzy_adapter_mainline_20260521/benchmark_toplevel_proposal_generation_diagnostics.md
results/shared/chem_enzy_adapter_mainline_20260521/benchmark_v2_100_toplevel_onmt_smiles_token/manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/benchmark_v2_100_toplevel_context_lr001_steps200_eval/experiment_manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/benchmark_v2_100_toplevel_context_alltrain_overfit_steps300/experiment_manifest.md
results/shared/chem_enzy_adapter_mainline_20260521/benchmark_v2_100_toplevel_plain_alltrain_overfit_steps300/experiment_manifest.md
```

The benchmark-style corpus has only `81` usable top-level positives. A
70/15/15 context run gives valid top5 `0/12` and test top5 `1/13`. Even an
all-train context overfit reaches only top1 `3/80`, top5 `5/80`; plain all-train
fine-tuning regresses from native top5 `10/80` to adapter top5 `9/80`. Decision:
current ONMT SFT can reduce token loss but does not reliably generate complete
top-level reactant sets at this scale. The next proposal-generation step needs
larger synthetic/route-targeted positives, not another reranker on the same
candidate pool.

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

The 150K-source build emits `109692` context positives after dedupe. It improves
benchmark-v2 top-level coverage versus the clean chosen-only corpus:
`targets_with_exact_reaction=11/100`, `targets_with_exact_product=27/100`, and
`targets_with_near_pair_ge_0_70=29/100`, with `46/100` still OOD. This is
evidence that external positives help proposal coverage, but not enough to
claim target-domain coverage.

The external corpus required shared source/target vocab expansion from `276` to
`319` tokens. The post-extension vocab audit reports `0` source and target OOV.
The first 300-step context adapter smoke trained only the default ONMT
`src/tgt_seq_length<=50` subset: train rows dropped from `98600` to `8385`.
Its 200-example exact-recall check is weak: valid top5 `0/200` for both native
and adapter, test top5 native `0/200` vs adapter `1/200`. Treat this as a
pipeline smoke, not a model improvement.

The follow-up len200 run explicitly uses
`--src-seq-length 200 --tgt-seq-length 200`, retains `84101/98600` train rows,
uses GPU training via `-world_size 1 -gpu_ranks 0`, and reaches validation
accuracy `87.3655`. Exact-recall still does not improve: valid top5 `0/200` for
both native and adapter, test top5 `0/200` for both. Decision: external
top-level positives improve coverage and the training pipeline is now
technically sound, but the current context SFT objective is not sufficient for
complete reactant-set generation. Do not promote this adapter; next work should
change the proposal objective/data form, not simply add more steps on the same
objective.

Direct 5-target top-level proposal audit gives a small but insufficient signal:
`targets_with_proposals=5/5`, exact GT reaction `0/5`, target-step GT reaction
`0/5`, and GT-reactant hit `1/5`. The hit is rank 4 for
`O=c1[nH]c2ccccc2o1`, proposing `Nc1ccccc1O>>O=c1[nH]c2ccccc2o1`. This means
external training can move a plausible precursor into the candidate pool, but
not enough to justify route-level promotion.

Gap diagnosis on those 5 targets classifies failures as:
`proposal_partial_reactant_hit=1`, `generator_missed_known_reactant_side=2`,
and `target_domain_gap=2`; there are no exact GT reactions or exact products in
the external corpus for those five targets. The next mainline training data
should therefore be route-targeted/synthetic and include a reactant-set or
reaction-completion objective. Continuing the same external context SFT for
more steps is not the right next experiment.

Prediction-error diagnosis on len200 exact-recall outputs confirms the failure
is not a near-miss problem. On valid/test 200-row samples, top-k predictions
have no GT-reactant overlap for `161/200` and `159/200` rows, and contain at
least one invalid molecule for `109/200` and `117/200` rows. The next proposal
work must add legality/constraint handling and reactant-set completion; simply
training the same seq2seq objective longer is unlikely to recover exact
reactant sets.

Current route-level A/B:

```text
results/shared/chem_enzy_adapter_mainline_20260521/chosen_context_lr001_steps300_route_ab_limit5/route_ab_summary.md
```

The new checkpoint's gated first-step context ONMT sidecar produced proposals
but did not improve route-pool metrics on five benchmark targets. Candidate and
result GT coverage were unchanged, while average cascade search time increased.
Keep it sidecar-only. Direct top-level proposal audit shows `0` GT
reaction/reactant hits on the same five targets, even after conservative
self-reaction filtering, so next work should improve proposal quality before
larger route-level promotion tests.

Direct top-level proposal audit command:

```bash
PYTHONPATH=. /root/autodl-tmp/chem_enzy_runtime/envs/retro_planner_env/bin/python \
  scripts/audit_context_onmt_top_level_proposals.py \
  --benchmark data/benchmark_v2_100.json \
  --model results/shared/chem_enzy_adapter_mainline_20260521/chosen_v4_1477_context_lr001_steps300_full_eval/context_chosen_adapter_lr001_step_300.pt \
  --output results/shared/chem_enzy_adapter_mainline_20260521/chosen_context_lr001_steps300_route_ab_limit5/top_level_proposal_audit_filtered.json \
  --markdown-output results/shared/chem_enzy_adapter_mainline_20260521/chosen_context_lr001_steps300_route_ab_limit5/top_level_proposal_audit_filtered.md \
  --limit 5 \
  --topk 4 \
  --beam-size 8 \
  --min-score 0.05
```

Known-legal corpus candidate-pool baseline:

```bash
PYTHONPATH=. python scripts/audit_legal_corpus_top_level_proposals.py \
  --benchmark data/benchmark_v2_100.json \
  --corpus results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_onmt_smiles_token_canonical_150k/context.train.meta.jsonl \
  --corpus results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_onmt_smiles_token_canonical_150k/context.valid.meta.jsonl \
  --corpus results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_onmt_smiles_token_canonical_150k/context.test.meta.jsonl \
  --output results/shared/chem_enzy_adapter_mainline_20260521/legal_corpus_constrained_provider_top100/audit.json \
  --markdown-output results/shared/chem_enzy_adapter_mainline_20260521/legal_corpus_constrained_provider_top100/audit.md \
  --topk 100 \
  --candidate-pool-size 1024 \
  --index-cache results/shared/chem_enzy_adapter_mainline_20260521/legal_corpus_constrained_provider_top100/legal_corpus_index.pkl
```

Current result on `benchmark_v2_100`: `100/100` targets return legal
candidates, exact target-step GT reaction hit is `18/100`, and GT-reactant hit
is `62/100`. This is proposal-pool coverage evidence only; it is not a
route-quality or condition-compatibility promotion signal. The optional
`--index-cache` stores RDKit fingerprints so later audits do not rebuild the
109K-reaction index. Cache files are source-signature guarded: mismatched corpus
inputs build an in-memory index and do not overwrite the existing cache.

The same provider is wired into the route-level benchmark runner as an opt-in
sidecar:

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

Legal-only depth-1 route sanity:

```text
results/shared/chem_enzy_adapter_mainline_20260521/legal_corpus_constrained_provider_top100/runner_legal_only_dryrun_full100_top100_depth1.json
```

This run uses `--dry-run`, disables real ChemEnzy routes, enables only the
legal-corpus sidecar at `topk=100`, and caps cascade depth at 1. It reports
exact reaction in result programs for `13/100` targets and GT-reactant evidence
in result programs for `43/100`. Treat this as search-layer sanity only; the
next task is score calibration/search fusion, not promotion.

Verifier condition extraction:

```text
cascade_planner/cascade_verifier/condition_extraction.py
results/shared/cascade_verifier_mainline_20260521/learned_verifier_v4_30k_condition_extraction_report.md
results/shared/cascade_verifier_mainline_20260521/statin_static_showcase3_condition_extraction_matrix/matrix.md
```

This consolidates condition/reagent parsing for rule checks and learned-verifier
features. It improves robustness to nested route export formats; the 30K
calibration metrics are unchanged, so it is not a learned-model promotion.

Stage-aware learned-verifier features:

```text
results/shared/cascade_verifier_mainline_20260521/learned_verifier_v4_30k_stage_aware_report.md
results/shared/cascade_verifier_mainline_20260521/statin_static_showcase3_stage_aware_matrix/matrix.md
results/shared/cascade_verifier_mainline_20260521/statin_static_showcase3_stage_aware_annotation_only_matrix/matrix.md
```

This improves offline verifier metrics on the 30K perturbation pack, but it
changes one static statin showcase top1 without audit-rank improvement. Keep it
as offline annotation / reason-classification evidence; do not promote it as a
default reranker. In annotation-only mode on the same static showcase it causes
`0` learned extra top1 changes.

Route-level tokenizer A/B smoke:

```bash
PYTHONPATH=. /root/autodl-tmp/chem_enzy_runtime/envs/retro_planner_env/bin/python \
  scripts/compare_chem_enzy_onmt_tokenizer_ab.py \
  --target CCO \
  --output-dir results/shared/chem_enzy_adapter_mainline_20260521/tokenizer_route_ab_smoke \
  --max-steps 4 \
  --iterations 2 \
  --expansion-topk 5
```

Common Web commands:

```bash
PYTHONPATH=. AUTOPLANNER_WEB_HOST=0.0.0.0 AUTOPLANNER_WEB_PORT=7991 \
  CHEMENZY_ENV_PREFIX=/root/autodl-tmp/chem_enzy_runtime/envs/retro_planner_env \
  python scripts/run_autoplanner_web_waitress.py

PYTHONPATH=. python scripts/monitor_autoplanner_web.py --url http://127.0.0.1:7991 --once
```

The main path does not require expert labels, a filled expert CSV, or
`DEEPSEEK_API_KEY`.

Archived/fallback scripts from the route/block value, strict review,
expert/LLM review, CCTS, route-pool ranker, and old v4 value lines require
`AUTOPLANNER_ALLOW_LEGACY_RESEARCH=1` and are documented in:

```text
archive/code/frozen_research_2026-05-20/README.md
docs/archive/2026-05/MAINLINE_CLEANUP_2026-05-20.md
```

They are retained for reproducing historical reports only. Do not use them as
mainline training targets or runtime dependencies.

Historical migration, cluster smoke, and packaging helpers are archived under
`archive/cleanup_2026-05-05/scripts/` when present. Generated release bundles
belong under ignored local artifact folders such as `releases/`.
