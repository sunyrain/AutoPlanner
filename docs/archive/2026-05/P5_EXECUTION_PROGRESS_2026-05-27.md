# P5 Execution Progress

Date: 2026-05-27

## Current Objective

Execute the enzyme-bridge plan through P5:

1. Build auditable bridge data.
2. Train verifier v0 without expert labels.
3. Gate enzyme bridge candidates before they pollute multi-step search.
4. Run native / ungated / gated / gated+verifier comparisons.
5. Package evidence-supported chemo-enzymatic routes and limitation analysis for reporting.

## Todo Status

| Stage | Task | Status | Evidence |
|---|---|---:|---|
| P0 | Disk/data audit and data/resource plan | Done | `docs/DATA_AUDIT_AND_RESOURCE_PLAN_2026-05-27.md` |
| P0 | Compare current data against planned data | Done | `docs/BRIDGE_PACK_PROGRESS_VS_PLAN_2026-05-27.md` |
| P0 | Build bridge pack v0 | Done | `data/bridge_pack_v0/manifest.json` |
| P1 | Exact bridge and similarity bridge construction | Done | `exact_bridge_strict.parquet`, `similarity_bridge_filtered.parquet` |
| P1 | Hard negative pool construction | Done | `hard_negative_pool.parquet` with 500,000 rows |
| P1 | Train EnzymeFeasibilityVerifier v0 | Done | `results/shared/bridge_verifier_v0_20260527/bridge_verifier_v0_report.json` |
| P1 | Verifier calibration and scoring interface | Done | `scripts/score_bridge_verifier_v0.py` |
| P1 | BridgeRetriever / BridgeScorer runtime wrapper | Done | `cascade_planner/cascade_search/bridge_retriever_v0.py` |
| P1 | Verifier bridge-hit runtime cache | Done | `data/bridge_pack_v0/bridge_candidates_scored.parquet` |
| P1 | Retriever unit smoke | Done | `tests/test_bridge_retriever_v0.py` |
| P2 | Candidate-level ungated vs gated ablation | Done | `results/shared/bridge_gate_ablation_v0_20260527/bridge_gate_ablation_report.md` |
| P2 | Molecule-level source-gate ablation | Done | `results/shared/bridge_gate_ablation_v0_20260527/bridge_source_gate_ablation_report.md` |
| P2 | Controlled route-level native / ungated / gated / gated+verifier ablation | Done | `results/shared/bridge_gate_ablation_v0_20260527/bridge_route_gate_ablation_report.md` |
| P2 | Live-provider route-level smoke | Done | `results/shared/bridge_gate_ablation_v0_20260527/bridge_live_provider_smoke_report.md` |
| P2 | Larger live-provider route-level benchmark | Partial | 4-target all-policy smoke done; 12-target bridge-positive live route evidence done; full native/ungated/gated comparison still needed |
| P2 | GatedSidecarRouter in route search | Partial | `BridgeAwareSourceGate` added; fallback bypass fixed |
| P2 | EnzymeBridgeProposalProvider in route search | Pending | Retriever exists; route-action semantics still pending |
| P3 | Live enzyme target probe | Done | `results/shared/bridge_gate_ablation_v0_20260527/live_enzyme_bridge_target_probe_report.md` |
| P3 | EC-conditioned enzyme proposer | Not started | Current live evidence shows provider selection/calibration bottleneck |
| P3 | Verifier v1 with richer substrate-product-enzyme labels | Not started | Requires bridge pack expansion |
| P3 | Sequence-aware enzyme features / frozen ESM embeddings | Not started | Data/compute allocation pending |
| P4 | 3D shape/pharmacophore validator | Not started | Top bridge validation layer |
| P4 | Active-site evidence examples | Not started | Needs selected route/bridge cases |
| P4 | Bridge-supported enzyme selection bonus | Partial | Runtime switch added and tested; bonus=2.0 improves normal live enzyme selection from 3/12 to 4/12 targets |
| P4 | Fair live policy benchmark | Partial | Depth-1 4p/4n benchmark shows bonus=2.0 improves recall without observed FPR; deeper benchmark still small |
| P5 | Live route quality audit | Done | `results/shared/bridge_route_quality_audit_v0_20260528/bridge_live_route_quality_audit_report.md` |
| P5 | 10-20 evidence-supported bridge routes | Done for partial-route evidence | 14 production-policy partial cards in `results/shared/p5_bridge_evidence_package_v0_20260528/p5_bridge_evidence_package.md` |
| P5 | Evidence cards for selected routes | Done for partial-route evidence | controlled draft cards, live diagnostic cards, P5 production-policy partial package |
| P5 | Final comparison and failure analysis | Done for diagnostic P5 | This document plus `docs/P5_BRIDGE_EVIDENCE_REPORT_2026-05-28.md`; stock-closed production route remains unsolved |

## Completed Results

### Bridge Pack v0

| Data artifact | Rows |
|---|---:|
| chemical_product_pool.parquet | 89,387 |
| enzyme_reaction_pool.parquet | 109,857 |
| enzyme_substrate_product_pool.parquet | 82,454 |
| enzyme_sequence_pool.parquet | 190,750 |
| exact_bridge_strict.parquet | 37,209 |
| similarity_bridge_filtered.parquet | 30,000 |
| hard_negative_pool.parquet | 500,000 |
| verifier train/valid/test total | 567,209 |

### Verifier v0

Held-out test metrics:

| Metric | Value |
|---|---:|
| PR-AUC | 0.9984 |
| precision@0.5 | 0.9809 |
| recall@0.5 | 0.9908 |
| recommended high-precision gate threshold | 0.84099 |

### Runtime Bridge-Hit Cache

Built pre-scored runtime cache:

| Artifact | Value |
|---|---:|
| `bridge_candidates_scored.parquet` rows | 67,209 |
| verifier-pass rows | 65,467 |
| cache size | 4.3 MB |
| cache build time | 75.147 s |
| retriever load time after optimization | ~1.8 s |
| single retrieve time after optimization | ~0.01 s |

This fixes the main runtime issue found during route-gate testing: verifier scoring should be precomputed rather than called repeatedly inside the search loop.

### Candidate-Level Gate Ablation

Test split: 53,791 rows, 6,383 positives, 47,408 negatives.

| Policy | Accepted | Precision | Recall | FPR | Cost / TP |
|---|---:|---:|---:|---:|---:|
| native_no_bridge | 0 | 0.0000 | 0.0000 | 0.0000 | 0.00 |
| ungated_bridge | 53,791 | 0.1187 | 1.0000 | 1.0000 | 8.43 |
| tanimoto_ge_0_50 | 17,808 | 0.3584 | 1.0000 | 0.2410 | 2.79 |
| tanimoto_ge_0_80 | 11,421 | 0.5589 | 1.0000 | 0.1063 | 1.79 |
| verifier_ge_0_50 | 6,447 | 0.9809 | 0.9908 | 0.0026 | 1.02 |
| verifier_precision_gate | 6,170 | 0.9893 | 0.9563 | 0.0014 | 1.01 |

Conclusion: the verifier gate turns the enzyme bridge sidecar from high-recall/noisy to high-precision/low-FPR. This supports gating before any route-search integration.

### Molecule-Level Source-Gate Ablation

Source-allocation smoke: 400 molecules, balanced as 200 bridge-positive and 200 chemical products without bridge hits.

| Policy | Triggered enzyme budget | Precision | Recall | FPR | Mean enzyme budget |
|---|---:|---:|---:|---:|---:|
| native_no_enzyme | 0 | 0.0000 | 0.0000 | 0.0000 | 0.00 |
| ungated_default_source_gate | 400 | 0.5000 | 1.0000 | 1.0000 | 3.50 |
| bridge_aware_source_gate | 200 | 1.0000 | 1.0000 | 0.0000 | 2.00 |

Conclusion: bridge-aware source allocation prevents enzyme-side proposal calls on molecules with no bridge evidence in this smoke setting.

### Controlled Route-Level Gate Ablation

Route-tree benchmark: 20 targets, balanced as 10 bridge-positive and 10 bridge-negative frontier molecules. Proposal engines are controlled stubs, so this isolates route-level gate behavior rather than chemistry solved-rate.

| Policy | Enzyme routes | True enzyme | False enzyme | Precision | Recall | False enzyme rate | Mean enzyme calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| native_no_enzyme | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.00 |
| ungated_default_source_gate | 20 | 10 | 10 | 0.5000 | 1.0000 | 1.0000 | 2.00 |
| bridge_gate_no_verifier | 10 | 10 | 0 | 1.0000 | 1.0000 | 0.0000 | 1.00 |
| bridge_gate_verifier | 10 | 10 | 0 | 1.0000 | 1.0000 | 0.0000 | 1.00 |

Conclusion: route-level gating suppresses false enzyme route selection in the controlled search loop. This still does not prove live ChemEnzy solved-rate improvement; it proves the source-gate wiring behaves correctly.

### Live-Provider Smoke

Small route-tree smoke using actual live proposal providers. Targets: 4, with available sources `chemtemplates`, `enzexpand`, `enzyformer`, `retrochimera`.

| Policy | Enzyme routes | False enzyme | Mean enzyme source calls | Mean elapsed s |
|---|---:|---:|---:|---:|
| native_no_enzyme | 0 | 0 | 0.00 | 12.38 |
| ungated_default_source_gate | 1 | 1 | 9.50 | 2.89 |
| bridge_gate_no_verifier | 0 | 0 | 1.00 | 0.26 |
| bridge_gate_verifier | 0 | 0 | 1.00 | 0.26 |

Runtime is cache-order biased because providers are shared across policies; source-call counts are the primary signal. The live smoke shows gating reduces enzyme-side source calls, but it did not produce true bridge-positive enzyme routes. Therefore it is not yet sufficient for final P5 route evidence.

### Live Enzyme Target Probe

Probe script:

- `scripts/probe_live_enzyme_bridge_targets_v0.py`
- output: `results/shared/bridge_gate_ablation_v0_20260527/live_enzyme_bridge_target_probe_report.md`

The probe scans verifier-pass bridge-positive molecules and calls live enzyme providers directly. It now reports both raw candidates and usable candidates after filtering action-contract failures such as self-loops, missing reactants, product mismatch, and tiny largest-reactant artifacts.

| Targets | Raw covered | Usable covered | Raw candidates | Usable candidates |
|---:|---:|---:|---:|---:|
| 30 | 30 | 30 | 199 | 165 |

Per-source usable count:

| Source | Returned | Usable |
|---|---:|---:|
| `enzexpand` | 112 | 94 |
| `enzyformer` | 87 | 71 |
| `retrorules` | 0 | 0 |

Observed invalid/noisy patterns:

- `self_loop`: 18 candidates;
- `tiny_largest_reactant`: 16 candidates, including cases where a small reagent alone was proposed as the precursor for a large product.

Conclusion: live enzyme providers are not completely missing bridge-positive targets. They can produce enzyme-like proposals, but a substantial fraction still requires route-level filtering and better confidence calibration.

### Live Bridge Route Evidence

Route evidence script:

- `scripts/run_bridge_live_route_evidence_v0.py`
- rows: `results/shared/bridge_gate_ablation_v0_20260527/bridge_live_route_evidence_rows.jsonl`
- cards: `results/shared/bridge_gate_ablation_v0_20260527/bridge_live_route_evidence_cards.md`
- report: `results/shared/bridge_gate_ablation_v0_20260527/bridge_live_route_evidence_report.md`

Run configuration:

| Setting | Value |
|---|---:|
| targets | 12 bridge-positive probe hits |
| max depth | 2 |
| branch factor | 6 |
| expansion budget | 12 |
| returned routes per target | 3 |
| live sources | `chemtemplates`, `enzexpand`, `enzyformer`, `retrochimera` |

Results:

| Policy | Targets | Routes returned | Targets with selected enzyme route | Selected enzyme routes | Mean elapsed s |
|---|---:|---:|---:|---:|---:|
| `normal_bridge_gated` | 12 | 36 | 3 | 5 | 18.331 |
| `enzyme_only_bridge_gated` | 12 | 36 | 12 | 36 | 3.848 |

Evidence cards generated:

| Card type | Count | Meaning |
|---|---:|---|
| integrated live search (`normal_bridge_gated`) | 3 | Production-like route-tree search naturally selected at least one enzyme step |
| enzyme-provider capability diagnostic (`enzyme_only_bridge_gated`) | 12 | Enzyme providers can create route fragments when chemical competition is removed |
| total live evidence cards | 15 | Useful diagnostic evidence, not yet final production-quality P5 route set |

Important limitation: all returned live evidence routes are partial route-tree results, not stock-closed routes. Their value is to identify where enzyme proposals are available and where route selection chooses or ignores them. They should not yet be presented as solved synthetic plans.

### Bridge-Supported Enzyme Selection Bonus

Implemented a runtime route-selection bonus:

- code: `cascade_planner/route_tree/search.py`;
- test: `tests/test_route_tree_planner.py::test_bridge_supported_enzyme_bonus_enters_selection_cost`;
- env switch: `AUTOPLANNER_ROUTE_TREE_BRIDGE_ENZYME_BONUS`;
- default: `0.0`, so existing behavior is unchanged unless explicitly enabled.

The bonus applies only when:

1. the candidate action is enzymatic;
2. its `source_gate` metadata records verifier-pass bridge hits;
3. the source-gate reason is `bridge_gate_hits`.

This keeps the bridge evidence as a selection prior rather than a pseudo-reaction.

Bonus=2.0 live route evidence run:

- output dir: `results/shared/bridge_gate_ablation_v0_20260527_bonus2`;
- report: `results/shared/bridge_gate_ablation_v0_20260527_bonus2/bridge_live_route_evidence_report.md`.

| Policy | Targets | Routes returned | Targets with selected enzyme route | Selected enzyme routes | Mean elapsed s |
|---|---:|---:|---:|---:|---:|
| `normal_bridge_gated` baseline bonus=0 | 12 | 36 | 3 | 5 | 18.331 |
| `normal_bridge_gated` bonus=2.0 | 12 | 36 | 4 | 8 | 18.608 |
| `enzyme_only_bridge_gated` bonus=2.0 | 12 | 36 | 12 | 36 | 3.861 |

Interpretation: adding bridge-supported selection evidence improves integrated normal search, but only modestly. The remaining gap is not simple source allocation; enzyme candidates still need better proposal quality, EC/reaction-center evidence, and negative-target FPR testing before this can be treated as a production setting.

### Fair Live Policy Benchmark

Benchmark script:

- `scripts/run_bridge_live_policy_benchmark_v0.py`

Design:

- positives: live-enzyme probe positives with usable enzyme candidates;
- negatives: chemical product pool molecules with no bridge hit;
- fresh live-engine wrapper per policy, so provider caches do not make later policies artificially faster;
- policies: `native_no_enzyme`, `ungated_default_source_gate`, `bridge_gate_verifier`, `bridge_gate_verifier_bonus2`.

Depth-2 framework smoke:

- output dir: `results/shared/bridge_live_policy_benchmark_v0_20260528_smoke`;
- targets: 2 positive / 2 negative;
- max depth: 2;
- branch factor: 5;
- expansion budget: 8.

| Policy | Selected targets | True | False | Precision | Recall | FPR | Mean enzyme calls | Mean elapsed s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `native_no_enzyme` | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 18.52 |
| `ungated_default_source_gate` | 1 | 1 | 0 | 1.0000 | 0.5000 | 0.0000 | 14.50 | 18.59 |
| `bridge_gate_verifier` | 1 | 1 | 0 | 1.0000 | 0.5000 | 0.0000 | 1.00 | 16.08 |
| `bridge_gate_verifier_bonus2` | 1 | 1 | 0 | 1.0000 | 0.5000 | 0.0000 | 1.00 | 15.96 |

Depth-1 4p/4n root-selection benchmark:

- output dir: `results/shared/bridge_live_policy_benchmark_v0_20260528_depth1_4p4n`;
- targets: 4 positive / 4 negative;
- max depth: 1;
- branch factor: 6;
- expansion budget: 8.

| Policy | Selected targets | True | False | Precision | Recall | FPR | Mean enzyme calls | Mean elapsed s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `native_no_enzyme` | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 3.76 |
| `ungated_default_source_gate` | 1 | 1 | 0 | 1.0000 | 0.2500 | 0.0000 | 2.00 | 2.79 |
| `bridge_gate_verifier` | 1 | 1 | 0 | 1.0000 | 0.2500 | 0.0000 | 1.00 | 2.52 |
| `bridge_gate_verifier_bonus2` | 4 | 4 | 0 | 1.0000 | 1.0000 | 0.0000 | 1.00 | 2.57 |

Interpretation: the depth-1 benchmark is currently the strongest evidence that bridge verifier support should be used in action selection, not only source gating. It improved root enzyme-route recall from 1/4 to 4/4 without observed negative-target FPR in this small set. This still needs a larger depth-2/3 run and route-quality audit before promotion to final P5 production evidence.

### Live Route Quality Audit

Audit script:

- `scripts/audit_bridge_live_routes_v0.py`
- report: `results/shared/bridge_route_quality_audit_v0_20260528/bridge_live_route_quality_audit_report.md`
- rows: `results/shared/bridge_route_quality_audit_v0_20260528/bridge_live_route_quality_audit_rows.jsonl`

Inputs audited:

- baseline live route evidence;
- bonus=2.0 live route evidence;
- depth-2 fair benchmark smoke;
- depth-1 4p/4n fair benchmark.

Summary:

| Metric | Value |
|---|---:|
| selected enzyme routes audited | 94 |
| diagnostic-only routes | 72 |
| production-policy partial candidates | 22 |
| production-policy stock-closed candidates | 0 |
| production positive routes | 22 |
| production negative routes | 0 |
| unique targets among production partial candidates | 7 |
| hard artifact flags | 0 |

Risk flags:

| Risk | Count |
|---|---:|
| generic EC | 92 |
| missing EC | 8 |
| larger reactant than product | 4 |

Interpretation: the current live routes are not failing because of obvious self-loop/product-mismatch/tiny-reagent artifacts. They are failing as final synthesis routes because they are partial and because the enzyme evidence is mostly generic EC/template-level rather than enzyme/substrate-specific.

### P5 Evidence Package

Package script:

- `scripts/build_p5_bridge_evidence_package_v0.py`

Package outputs:

- `results/shared/p5_bridge_evidence_package_v0_20260528/p5_bridge_evidence_package.json`
- `results/shared/p5_bridge_evidence_package_v0_20260528/p5_bridge_evidence_package.md`

Selection:

- include only production-policy live routes:
  - `normal_bridge_gated`;
  - `bridge_gate_verifier`;
  - `bridge_gate_verifier_bonus2`;
- exclude enzyme-only diagnostic routes;
- exclude ungated control routes;
- require positive label and production-candidate quality audit status.

Package summary:

| Metric | Value |
|---|---:|
| evidence cards | 14 |
| unique targets | 7 |
| stock-closed cards | 0 |
| route-solved cards | 0 |
| hard-flag cards | 0 |
| `bridge_gate_verifier_bonus2` cards | 4 |
| `normal_bridge_gated` cards | 8 |
| `bridge_gate_verifier` cards | 2 |

Boundary: this is a P5 diagnostic evidence package, not a set of final solved synthesis plans.

## Current Gap After P5 Package

The project has not yet shown live-provider route-level solved-rate improvement to stock-closed routes. The current evidence proves that verifier gating can suppress false enzyme bridge candidates on the weak-label bridge benchmark, in a controlled route-tree loop, and in live-provider benchmarks by reducing enzyme-side source calls.

The new live probe changes the diagnosis: live enzyme providers can produce usable proposals for bridge-positive molecules, but integrated route-tree search selects enzyme steps for only 3/12 bridge-positive targets under the normal bridge-gated policy. A bridge-supported selection bonus improves this to 4/12 targets and 8 selected enzyme routes in the shallow route-evidence run, and improves root-selection recall from 1/4 to 4/4 in the depth-1 fair benchmark without observed negative FPR. Therefore the next bottleneck is not simple provider absence; it is route-level calibration/selection and the quality of enzyme proposal scoring.

The P5 evidence package contains 14 production-policy partial route cards. That satisfies the diagnostic evidence-packaging objective, but it does not satisfy a stricter production objective of stock-closed retrosynthetic planning.

## Route-Gate Wiring Update

`BridgeAwareSourceGate` has been added to route-tree source allocation. It uses verifier-pass bridge evidence as a trigger:

- no verifier-pass bridge and no explicit EC context: suppress enzyme-side proposal budgets;
- verifier-pass bridge exists: allocate a controlled fraction of budget to enzyme-side sources;
- explicit EC context: bypass bridge gate, because the route skeleton already requests an enzyme step.

The route-tree fallback path was also guarded so that a no-hit bridge gate cannot be bypassed by fallback source retries.

## Draft P5 Evidence Cards

Generated 10 draft bridge route cards from the verifier-gated controlled route-tree run:

- `results/shared/bridge_gate_ablation_v0_20260527/bridge_route_evidence_cards.json`
- `results/shared/bridge_gate_ablation_v0_20260527/bridge_route_evidence_cards.md`

These cards combine controlled route selection with real bridge verifier evidence. They are useful as case-selection artifacts, but they are not final live-provider route evidence.

Generated 15 live-provider diagnostic evidence cards:

- `results/shared/bridge_gate_ablation_v0_20260527/bridge_live_route_evidence_cards.json`
- `results/shared/bridge_gate_ablation_v0_20260527/bridge_live_route_evidence_cards.md`

Only 3 of these are integrated normal-gated route cards. The remaining 12 are enzyme-only provider-capability diagnostics. They are useful for debugging and target selection, but they are not production-policy route evidence.

Generated 14 production-policy partial P5 evidence cards:

- `results/shared/p5_bridge_evidence_package_v0_20260528/p5_bridge_evidence_package.json`
- `results/shared/p5_bridge_evidence_package_v0_20260528/p5_bridge_evidence_package.md`

These are the current P5 deliverable cards. They are evidence-supported partial route fragments, not solved stock-closed routes.

## Next Work

1. Add enzyme proposal quality calibration before route selection:
   - penalize self-loop/tiny-reagent artifacts before candidate ranking;
   - downweight enzyme proposals without EC/reaction-center support;
   - promote verifier-supported enzyme proposals only when they improve frontier progress.
2. Run a fair larger live-provider route-search sidecar benchmark, including bonus=0 and bonus=2.0:
   - native baseline;
   - ungated enzyme sidecar;
   - bridge-gated sidecar;
   - verifier-gated sidecar.
3. Add negative targets to the live benchmark so FPR and candidate pollution are measured under the same provider/cache conditions.
4. Record search cost per useful bridge, enzyme FPR, candidate pollution, route plausibility, route depth, and runtime.
5. Promote to P5 evidence routes only after normal live-provider gated mode beats ungated mode or the remaining gap is explicitly reported as a route-selection/proposer-quality limitation.
