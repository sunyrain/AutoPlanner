# ChemEnzy Module Audit, 2026-05-28

## Scope

This audit reads the local ChemEnzyRetroPlanner vendor implementation under
`vendor/ChemEnzyRetroPlanner` and the AutoPlanner adapter layer.  It focuses on
where enzyme decisions are made, whether they affect search, and which modules
are weak targets for AutoPlanner enzyme proposal/selection work.

## Bottom Line

ChemEnzy is a strong route generator, but its enzyme capability is not a
validated enzyme-step selector.

The current pipeline has three different signals that are easy to conflate:

1. Enzymatic proposal sources, such as `onmt_models.bionav_one_step`,
   `template_relevance.bkms_metabolic`, and `template_relevance.reaxys_biocatalysis`.
2. Post-search reaction classification and EC assignment.
3. Optional active-site annotation through EasIFA.

Only the first category affects route search.  The second and third categories
mostly annotate routes after they have already been generated.  Therefore a high
count of EC-annotated routes does not prove good enzyme-step selection.

## Call Chain

The route generation path is:

1. `ChemEnzyBackendAdapter._vendor_config` sets search flags and enables or
   disables condition/enzyme assignment.
2. `RSPlanner.prepare_plan` builds the stock set, one-step model ensemble,
   optional value function, optional cascade cost/source hooks, and the MolStar
   planner.
3. `prepare_multi_single_step` calls each selected one-step model and merges
   their candidates into one expansion result.
4. `prepare_molstar_planner` passes the merged expansion result into
   `mol_planner`.
5. `MolTree.prepare_expansion` converts scores to scalar costs and optionally
   applies the cascade cost model.
6. `RSPlanner.plan` ranks complete successful routes.
7. `predict_rxn_attributes` optionally annotates generated routes with
   condition, organic/enzyme classification, and EC assignment.

Key code locations:

- `cascade_planner/baselines/chem_enzy_adapter.py`
- `vendor/ChemEnzyRetroPlanner/retro_planner/api.py`
- `vendor/ChemEnzyRetroPlanner/retro_planner/common/prepare_utils.py`
- `vendor/ChemEnzyRetroPlanner/retro_planner/search_frame/mcts_star/`

## Module Audit

| Module | Search-time role | Current behavior | Weak point | Required strengthening |
|---|---:|---|---|---|
| Stock / starting molecules | Yes | Loads large stock files and applies simple atom-count filters. | Stock closure can make implausible biosynthetic precursors look solved if they exist in stock; no biosynthetic availability class. | Add source-aware stock tiers, enzyme/metabolite stock labels, and penalties for suspicious terminal cofactors or long-prenyl intermediates. |
| GraphFP USPTO model | Yes | Template model applies predicted retrosynthetic templates with RDChiral. | Strong chemistry source, not enzyme-aware; template scores are not calibrated against enzyme sources. | Keep as chemical baseline; calibrate against enzyme candidates in shared cost space. |
| ONMT `bionav_one_step` | Yes | Seq2seq product-to-reactants model; output has no structured EC/cofactor/reaction-center evidence. | This is the main native enzyme-ish generator, but it is not an explicit enzyme proposal model at search time. | Wrap/replace with EC-conditioned, cofactor-aware proposal metadata and reaction-center checks. |
| Template relevance: `bkms_metabolic` / `reaxys_biocatalysis` | Yes | Template retrieval/application source treated like other template sources. | Enzyme provenance is mostly source-level, not candidate-level validation; template match can be mechanistically wrong. | Preserve EC/source evidence per candidate and score reaction-center/EC/cofactor compatibility before MCTS backup. |
| Multi one-step merger | Yes | Concatenates model outputs and converts each score to `-log(score * weight)`. | Scores from graph/template/ONMT are not calibrated; equal weights make weak enzyme candidates compete unfairly or vanish arbitrarily. | Add calibrated per-source priors and candidate-level enzyme selection cost. |
| Reaction filter | Optional search-time | Disabled in current statin benchmark; if enabled, it is a generic feasibility classifier. | Not enzyme-specific; does not validate EC, enzyme entity, cofactors, or active site. | Keep as generic plausibility screen only; do not use as enzyme validation. |
| MolStar / MCTS search | Yes | Selects open molecule node by scalar value; expansion candidates are backed up by scalar reaction cost. | The search state does not natively know route enzyme intent, cofactor debt, consecutive enzyme steps, or enzyme/chemical stage. | Add enzyme-router state and enzyme-aware cost terms before candidate backup. |
| Depth/value function | Optional search-time | Molecule value heuristic; often disabled in AutoPlanner comparison runs. | Molecule-only value cannot decide whether an enzyme step is appropriate. | Train/source a node-level enzyme-router value model, not just depth value. |
| Cascade source policy | Optional search-time | Rebalances or filters model sources based on coarse domain/model names. | Source-level only; cannot decide which specific enzyme reaction is good. | Upgrade to enzyme router: per-node probability of enzyme expansion, source budget, and continuous enzyme/chemical alternation support. |
| Cascade cost model | Optional search-time | Adds rule/learned scalar adjustments for domain preference, condition, cofactor, weak evidence if metadata exists. | Most native candidates lack rich enzyme metadata, so enzyme-specific terms are often inactive or coarse. | Feed candidate-level EC, cofactors, reaction center, material audit, precedent evidence, and enzyme support into this cost. |
| Condition prediction | Post-search in current flow | Predicts conditions after route generation when enabled. | Does not guide whether an enzyme step is condition-compatible during search. | Convert condition envelope into search state and penalize incompatible enzyme/chemical transitions online. |
| Organic/enzyme reaction classifier | Post-search annotation | Classifies each generated reaction as Organic or Enzymatic from RXNFP. | This is not a search decision and not proof of enzymatic feasibility; can over-mark or misclassify route steps. | Use as weak signal only; require independent candidate provenance and validation. |
| EC assignment | Post-search annotation | Predicts Top-K EC numbers from reaction fingerprint. | EC label alone is not enzyme validation; no specific enzyme, substrate scope, cofactor, or active-site check. | Treat EC as a hypothesis; validate against reaction-center and precedent/enzyme evidence before ranking. |
| EasIFA active-site annotation | Interactive / post-hoc | Queries one enzyme by EC and predicts active site labels for a structure. | Not connected to route search or ranking; one EC query does not validate the proposed reaction path. | Use only after candidate has a specific enzyme entity; add as late-stage confidence, not as route generator. |
| Pathway ranker | Post-search | Default depth ranker or optional tree-LSTM; cascade cost ranker if enabled. | Route ranking is not enzyme-plausibility aware unless cascade cost already encoded it. | Rank complete routes by enzyme-step validation, cofactor ledger, condition continuity, and material sanity. |
| AutoPlanner native enzyme plugin | Search-time proposal injection | Adds bridge/SP-v1-gated precedent candidates to ChemEnzy one-step expansion. | Current version lacks full enzyme router and route-level selection cost; statin injected representative routes failed route-level audit. | Rebuild as structured enzyme provider plus selection cost, not a sidecar or simple appended candidate source. |
| Adapter/export layer | Reporting | Normalizes ChemEnzy routes and extracts EC annotations. | Easy to confuse generated enzyme source with post-hoc EC annotation. | Export explicit fields: proposal source, post-hoc classifier label, EC hypothesis, validated enzyme evidence, and audit status. |

## Enzyme-Specific Weak Points

### 1. No Explicit Enzyme Router

ChemEnzy can query enzyme-like sources, but there is no strong node-level
decision model for "this molecule should now be disconnected by an enzyme step".
The source policy hook can rebalance source top-k, but it operates mostly on
source names and broad route context.  It does not yet use reaction-center,
cofactor, EC, or enzyme precedent features.

This is the highest-impact gap.

### 2. Enzyme Proposal Is Weakly Structured

Native enzyme-ish proposals are just reactant SMILES plus scores.  They usually
do not carry:

- EC class actually used to generate the candidate,
- required cofactors,
- regenerated cofactors,
- enzyme/entity evidence,
- reaction-center match,
- precedent reaction IDs,
- substrate/product similarity to precedent,
- confidence calibrated against chemical candidates.

Without this metadata, MolStar can only choose candidates by generic scalar
cost.

### 3. Score Calibration Is Not Reliable Across Sources

The multi-source wrapper concatenates outputs and computes cost using each
source's own score.  Scores from graph templates, template relevance, and ONMT
are not naturally comparable.  Equal source weights are especially risky for
enzyme selection because a low-quality enzyme proposal can be promoted or buried
for score-scale reasons rather than chemical reason.

### 4. EC Annotation Is Not Validation

The EC assignment module runs after routes are generated.  It is useful as a
weak annotation but should not be counted as a validated enzyme step.  A route
can have many EC labels while none of the steps has been validated against a
specific enzyme, cofactor ledger, or active site.

This explains why the statin rerun produced many EC-annotated native routes
without proving that native enzyme-step selection is chemically reliable.

### 5. Active-Site Support Is Disconnected

EasIFA is present, but the normal route search does not use it to choose enzyme
steps.  It is a late interactive annotation tool.  It cannot rescue weak
proposal selection unless the system first selects a specific enzyme entity and
passes a plausible reaction to it.

### 6. Route-Level Material Sanity Is Not Enforced During Search

The statin plugin comparison showed that local enzyme precedent/SP-v1 acceptance
can still lead to route-level material failures.  This is not just a plugin
problem; native ChemEnzy can also produce large unexplained construction steps
that later receive EC annotations.  Material and reaction-center sanity must
enter candidate cost before route selection, not only after.

## Strengthening Priorities

### P0: Build an Enzyme-Step Audit Dataset From ChemEnzy Native Runs

For every step in every route, export:

- product and reactants,
- source model,
- template/source,
- whether generated by enzyme-like source,
- whether post-hoc classified as enzymatic,
- EC assignment and confidence,
- condition prediction if available,
- material audit result,
- route position and adjacent step domains.

This gives a real error map instead of debating from route counts.

P0 implementation status:

- Implemented `cascade_planner/baselines/enzyme_step_audit.py`.
- Implemented `scripts/export_chem_enzy_enzyme_step_audit.py`.
- Updated the adapter to preserve one-step source metadata from
  `cascade_cost.source_model` when the zero-adjustment cascade metadata hook is
  enabled.
- Smoke test: `pytest -q tests/test_enzyme_step_audit.py tests/test_route_plausibility.py`
  passes.

P0 statin native run:

- Output directory:
  `results/shared/chem_enzy_enzyme_step_audit_20260528_statins_native_iter10`.
- Run settings: 9 statins, ChemEnzy native adapter-default one-step sources
  (`graphfp_models.USPTO-full_remapped`, `onmt_models.bionav_one_step`),
  `iterations=10`, `max_depth=4`, `expansion_topk=50`, enzyme assignment
  enabled.
- Routes: 401.
- Steps: 1287.
- Source split: 508 graphfp chemical-like steps, 779 BioNav enzyme-like steps.
- Post-hoc enzymatic steps: 789.
- Material audit failed steps: 245, all from BioNav enzyme-like source in this
  run.
- Chemical-source steps post-hoc marked enzymatic: 182.
- Low Top-1 EC confidence steps: 1237.

This confirms that the weak point is not absence of enzyme labels.  The weak
point is that enzyme-like proposal, post-hoc enzyme marking, EC confidence, and
material sanity are poorly aligned.

### P1: Add Search-Time Enzyme Router

At every molecule node, predict or score:

- chemical-only expansion,
- enzyme-only expansion,
- mixed expansion with budget split.

The router must allow enzyme after enzyme and chemical after enzyme.  It should
not assume that enzyme steps are rare side branches.

### P2: Replace "Enzyme Candidate = Reactants + Score" With Structured Actions

Every enzyme proposal should carry:

- EC hypothesis,
- reaction class/mechanism tag,
- cofactors required/regenerated,
- reaction-center signature,
- precedent evidence,
- substrate/product similarity,
- source reliability,
- validation flags.

The existing native enzyme plugin should be refactored into this structured
provider, not kept as a simple appended candidate list.

### P3: Enzyme Selection Cost Inside MolStar

Before a candidate is inserted into the AND/OR tree, compute a calibrated enzyme
selection cost:

- material sanity,
- reaction-center compatibility,
- EC mechanism compatibility,
- cofactor ledger,
- condition compatibility,
- precedent support,
- source calibration,
- route-context fit.

This cost should alter MCTS backup and route extraction.

### P4: Route-Level Rerank Is Still Needed, But Not Sufficient

After search, rerank routes by complete enzyme evidence and route-level sanity.
This is a safety layer, not the main contribution.

## What Not To Claim

- Do not claim "ChemEnzy native has no enzyme routes".
- Do not claim "EC-annotated route count" equals enzyme-step quality.
- Do not claim the current AutoPlanner enzyme plugin improves ChemEnzy on the
  statin panel.
- Do not present post-hoc validation as the main contribution.

## Correct Project Reframe

The useful thesis is:

AutoPlanner strengthens ChemEnzy by taking over enzyme-step proposal and
selection, using reaction-center, EC, cofactor, precedent, condition, and
route-context evidence inside search.  Post-hoc validation remains a gate, but
the main improvement must happen before MCTS commits to expansion candidates.
