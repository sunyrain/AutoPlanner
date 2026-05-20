# AutoPlanner Cascade Progress Report

Date: 2026-05-19

This report is written for an expert-facing PPT discussion. It summarizes what we have built and debugged, why the current task is not just ordinary retrosynthesis, how the international CASP frontier usually works, why those systems struggle with chemoenzymatic cascades, and what remains to strengthen before claiming product-ready route planning.

## 1. Executive Summary

We started from a practical failure mode: a complex statin-like target was previously reported as `no route`, while the user expected actual training, benchmark, and route-quality evidence. Running and auditing the 200-iteration ChemEnzy search showed that the system did generate many routes, but the original reporting and filtering logic was too coarse.

For target:

```text
CC(C)C1=NC(=NC(=C1/C=C/[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)N(C)S(=O)(=O)C
```

The route pool contained:

| Metric | Count |
| --- | ---: |
| Raw routes | 906 |
| Raw unique route signatures | 393 |
| Original kept routes | 640 |
| Original kept unique route signatures | 243 |
| Original rejected routes | 266 |
| Original rejected unique route signatures | 150 |

After re-auditing the 640 kept routes with the corrected current logic:

| Current audit class | Count |
| --- | ---: |
| `triage_fragment` | 360 |
| `needs_chemist_review` | 232 |
| `reject_artifact` | 48 |

The important correction was conceptual:

1. Large terminal molecules are not automatically advanced product-like intermediates. Some large molecules are carrier reagents, for example Wittig/HWE reagents, where most atoms are sacrificed and only a small fragment is introduced.
2. Elements missing from the explicit reaction reactants can sometimes be explained by condition reagents, for example chlorination from `POCl3`. Reaction SMILES often omit stoichiometric reagents and therefore cannot be treated as complete atom provenance by default.

This does not mean route validation should be loosened. It means the audit needs a more chemically faithful model:

```text
explicit reactants + condition reagent evidence + reaction role + atom contribution
```

instead of:

```text
terminal heavy atom count alone
```

## 2. Current Deliverables

Key artifacts generated in this round:

| Artifact | Path | Purpose |
| --- | --- | --- |
| Re-audited route JSON | `results/v2/ui_chem_enzy_plan_20260519_032819_3764f7_reaudited_current.json` | Same 640 routes, but with corrected current product audit fields. |
| Top10 route figures | `results/v2/route_figures_3764f7_top10_current/index.html` | HTML index showing the top10 rendered synthesis-route SVGs. |
| Individual route SVGs | `results/v2/route_figures_3764f7_top10_current/route_01.svg` to `route_10.svg` | Vector route figures suitable for reports/PPT drafts. |
| Route figure renderer | `scripts/render_route_figures.py` | Standalone script to render route JSON into SVG/HTML. |
| Usability TSV | `results/v2/ui_chem_enzy_plan_20260519_032819_3764f7_usable_assessment.tsv` | Route-level sortable usability buckets from the earlier audit pass. |
| Shortlist report | `results/v2/ui_chem_enzy_plan_20260519_032819_3764f7_route_shortlist.md` | Human-readable shortlist from the previous pass. |

Validation run:

```bash
python -m pytest tests/test_render_route_figures.py tests/test_route_plausibility.py tests/test_product_route_feasibility_audit.py
```

Result:

```text
13 passed
```

## 3. Work Stages So Far

### Stage 1: Clarifying the Real Failure

Initial issue: the system/UI suggested `no route` for a target where a deep ChemEnzy search had actually generated routes. We inspected the result files under `results/v2/` and found that there were 906 raw routes and 640 post-filter kept routes.

Key correction:

- The relevant run was `iterations=200`, `max_depth=20`, `expansion_topk=100`.
- It was not truly `max_depth=200`.
- `no route` was a reporting/filtering interpretation problem for this target, not absence of generated route candidates.

### Stage 2: Route Pool Triage

We separated route existence from route usability. The first audit grouped routes into:

- strict autonomous candidate: no route reached this level
- triage fragment: route contains a useful disconnection or fragment assembly idea
- needs chemist review: route may be useful as a reference but is not reliable enough
- reject/artifact: atom-balance or obvious material-source artifacts

Initial result before the carrier/condition correction:

- 120 unique `triage_fragment` candidates
- 123 unique `needs_chemist_review` routes
- 150 unique rejected/artifact routes

This was useful but too harsh on large carrier reagents and too naive about condition reagents.

### Stage 3: Route-Level Diagnosis

The user challenged route 5 from the 640 kept routes. The route initially appeared problematic because it contained:

```text
CCOC(=O)C=P(c1ccccc1)(c1ccccc1)c1ccccc1
```

This is a large 25-heavy-atom molecule. The old terminal audit marked it as `advanced_or_product_like_terminal`. On chemical inspection, this is not appropriate: it is ethyl 2-(triphenylphosphoranylidene)acetate, a common stabilized Wittig reagent. Its phenyl groups are not product-like inherited structure; they are a carrier/leaving scaffold.

The same route also had a step where `Cl` appeared in the product but not in the explicit reactants. The condition predictor listed `POCl3`, so this is not necessarily an impossible step. It should be marked as condition-reagent-supported rather than raw unsupported element gain.

After re-audit, 640-route route 5 became:

```text
route_class: triage_fragment
issues: []
tags: acylating_piece_present, aryl_coupling_hint, carrier_reagent_terminal
product_like_terminal: false
large_polycyclic_terminal: false
effective_max_terminal_heavy_atoms: 10
route_plausibility_passed: true
```

This example is now a strong demonstration slide: it shows why a naive audit falsely rejects chemically reasonable route components, and why condition-aware atom provenance is necessary.

### Stage 4: Product Audit Correction

Implemented first-pass corrections:

1. `product_route_feasibility_audit.py`
   - Records raw terminal size and effective terminal size separately.
   - Detects carrier reagent terminals and labels them as `carrier_reagent_terminal`.
   - Uses effective terminal metrics for product-like terminal classification.

2. `route_plausibility.py`
   - Adds condition reagent element counts.
   - Separates raw element gains from condition-supported and unexplained gains.
   - Adds `unexplained_new_element_source` for key elements with no explicit or condition-supported source.

This is not the final generic solution. It is an important first step, but the correct next layer is atom contribution/role inference rather than enumerating special cases.

### Stage 5: Top10 Route Figure Rendering

Added `scripts/render_route_figures.py`. It renders top-k routes as static SVG panels using RDKit molecule drawings. Each route figure includes:

- target structure
- step-by-step retrosynthetic disconnections
- reactant/product structures
- route class, score, tags
- reaction type, EC annotation, and temperature when available
- terminal materials

Generated current top10:

```text
results/v2/route_figures_3764f7_top10_current/index.html
```

The SVGs are vector assets and can be inserted into PPT directly. For very long routes, the figures are suitable for supplementary/report slides. For main-deck slides, select route 1 or route 5 and crop/condense to show only the key disconnection block plus terminal materials.

## 4. Methods We Selected and Why

### 4.1 Route Generation

Current route generation uses ChemEnzyRetroPlanner through the CascadePlanner wrapper. The route pool is then interpreted by product-aware audit logic.

Why this route-pool strategy is appropriate:

- Long cascade routes are not well represented by a single top-1 prediction.
- Search generates many partially reasonable alternatives; useful evidence comes from comparing them.
- We do not have expert labels and should not assume we will get them later.

### 4.2 No-Expert-Label Training Strategy

The user clarified that there are no expert labels now and there will not be expert labels later. Therefore, the training direction should avoid supervised expert-label dependence.

Recommended signal sources:

| Signal type | How to generate it | Why it is usable without experts |
| --- | --- | --- |
| Material sanity | atom/element provenance checks, stock closure, invalid SMILES, self loops | fully automatic |
| Condition support | condition reagent can explain missing elements | generated from model output and chemistry rules |
| Route consistency | product of one step must match upstream precursor of previous step | structural check |
| Step confidence | backend retro score, condition score, enzyme confidence | model-provided |
| Route diversity | unique reaction-signature clustering | automatic |
| Hard negatives | corrupt routes, unsupported element additions, trivial stock closures | synthetic labels |
| Pairwise preferences | route with fewer severe issues beats route with issues, shorter route beats longer route when risk equal | weak supervision |

This is closer to weak supervision, self-supervision, and rule-guided ranking than expert-labeled classification.

### 4.3 Product Audit Instead of Stock Closure Alone

Strict stock closure is not enough. A route can close to stock because:

- a large advanced intermediate was treated as stock
- a carrier reagent was misinterpreted as a product-like terminal
- a reagent or condition was omitted from reaction SMILES
- a model-generated step silently introduced an element or functional group

Therefore we now separate:

```text
stock_closed
route_solved
product_audit_class
route_plausibility
terminal_profile
condition-supported atom provenance
```

### 4.4 Generic Role-Based Direction

The current implementation has a first-pass carrier reagent detector. The stronger general solution should be:

1. For every terminal or reagent, estimate its atom contribution to the final product.
2. Classify material role:
   - structural building block
   - carrier reagent
   - leaving group source
   - protecting/deprotecting reagent
   - oxidant/reductant
   - acid/base/salt
   - catalyst/ligand/solvent
3. Evaluate route quality from role and atom provenance, not from heavy atom count alone.

This aligns better with green chemistry concepts such as atom economy, PMI, and E-factor, but adapts them to incomplete AI-generated reaction records.

## 5. International Frontier

### 5.1 Neural-Symbolic CASP and MCTS

Segler, Preuss, and Waller combined Monte Carlo tree search with neural policies and symbolic reaction templates for synthesis planning in Nature 2018. This is the template/MCTS paradigm that influenced systems such as AiZynthFinder. The key idea is recursive retrosynthesis guided by a learned policy, with route search over purchasable precursors.

Source: Segler et al., Nature 2018, `Planning chemical syntheses with deep neural networks and symbolic AI`, DOI `10.1038/nature25978`.

### 5.2 AiZynthFinder

AiZynthFinder is an open-source retrosynthesis planner based on MCTS and neural template policies. It recursively breaks molecules down to purchasable precursors and has become a widely used reference implementation.

Source: Genheden et al., Journal of Cheminformatics 2020, DOI `10.1186/s13321-020-00472-1`.

### 5.3 ASKCOS and AI-Robotic Synthesis

ASKCOS-related work connects retrosynthetic planning, condition recommendation, feasibility assessment, and robotic flow execution. The 2019 Science paper demonstrated an AI-informed platform for flow synthesis of organic compounds.

Source: Coley et al., Science 2019, DOI `10.1126/science.aax1566`.

### 5.4 Template Extraction and Stereo Handling

RDChiral addresses a central technical issue in template-based retrosynthesis: consistent stereochemistry handling in retrosynthetic SMARTS extraction/application.

Source: Coley, Green, Jensen, JCIM 2019, DOI `10.1021/acs.jcim.9b00286`.

### 5.5 Biocatalytic CASP

RetroBioCat targets biocatalytic reactions and cascades. It is closer to our chemoenzymatic direction than standard small-molecule CASP tools because it explicitly considers enzyme transformations and cascade planning.

Source: Finnigan et al., Nature Catalysis 2020, `RetroBioCat as a computer-aided synthesis planning tool for biocatalytic reactions and cascades`, DOI `10.1038/s41929-020-00556-z`.

### 5.6 Synthetic Accessibility and Fast Feasibility Scores

RAscore learns a fast classifier for whether AiZynthFinder can find a synthetic route, intended as a rapid retrosynthetic accessibility proxy. SCScore similarly estimates synthetic complexity from reaction corpora. These are useful for screening, but they do not by themselves validate a detailed cascade route.

Source: Thakkar et al., Chemical Science 2021, DOI `10.1039/D0SC05401A`.

### 5.7 Green Chemistry Metrics

Internationally recognized metrics are useful for route assessment:

- Atom economy: how much input material ends up in the desired product.
- PMI: total mass of all materials used per mass of product. ACS GCI Pharmaceutical Roundtable uses it as a common pharmaceutical process metric.
- E-factor: mass of waste per mass of product, introduced to focus on waste generation.

These metrics motivate our role-based audit. However, AI route JSON usually lacks stoichiometry, yields, workup, and complete reagent accounting, so we can only compute proxies unless the planner emits richer records.

## 6. Why International CASP Still Struggles with Cascades

Standard CASP is strong at recursive disconnection but weak at cascade-level feasibility. Main reasons:

1. Step independence
   - Most planners score one retrosynthetic step at a time.
   - Cascades require compatibility across steps: solvent, pH, temperature, enzyme class, cofactor, intermediate stability, and isolation/no-isolation assumptions.

2. Incomplete reaction records
   - Reaction SMILES often list only reactants/products and omit reagents, catalysts, salts, water, cofactors, bases, acids, oxidants, and chlorinating agents.
   - This creates false atom-balance alarms or hides true material-source errors.

3. Route-level objective mismatch
   - Search usually optimizes stock closure and template probability.
   - Product-ready cascade planning needs route-level constraints: compatible operations, enzyme evidence, condition windows, isolation strategy, and cumulative risk.

4. Weak enzyme grounding
   - EC class is a broad family label, not an enzyme sequence, activity assay, substrate scope, or operational condition.
   - A route with low EC confidence is a hypothesis, not a demonstrated biocatalytic plan.

5. Domain split
   - Organic CASP and biocatalytic pathway planning are often developed separately.
   - Chemoenzymatic cascades require both chemical and enzymatic transformations in one consistent route state.

6. Long-route error compounding
   - In a 17-20 step route, even small per-step hallucinations become route-level failure.
   - Top-1 route scores are not enough; auditing and reranking are required.

## 7. How We Currently Handle Cascades

Our current handling is a layered approach:

```text
route generation
  -> route pool
  -> product audit
  -> plausibility audit
  -> condition-supported atom source checks
  -> route-class/risk ranking
  -> shortlist/report/figures
```

Specific cascade-oriented elements now present:

- Multi-step route pool rather than single route.
- `route_plausibility` audit for every step.
- Product-aware terminal profile.
- Condition reagent support for missing elements.
- Carrier reagent recognition for non-product-like large reagents.
- Route class taxonomy: `triage_fragment`, `needs_chemist_review`, `reject_artifact`.
- Top10 SVG rendering for expert review.

This is still not full cascade validation. It is a route triage and audit framework that catches many obvious issues and prevents some false rejections.

## 8. Representative High-Value Long Routes

### Route 1

Figure:

```text
results/v2/route_figures_3764f7_top10_current/route_01.svg
```

Audit summary:

```text
route_class: triage_fragment
score: 0.0112
n_steps: 17
tags: acylating_piece_present, aryl_coupling_hint, carrier_reagent_terminal
```

Why it is representative:

- Long route with aryl coupling and side-chain construction logic.
- Uses carrier reagent terminal but no longer treated as product-like terminal.
- Suitable as a slide showing a high-ranked route and the need for route-level audit.

### Route 5

Figure:

```text
results/v2/route_figures_3764f7_top10_current/route_05.svg
```

Audit summary after current correction:

```text
route_class: triage_fragment
score: 0.00131
n_steps: 18
issues: []
effective_max_terminal_heavy_atoms: 10
condition-supported Cl source: yes
```

Why it is important:

- It demonstrates why the old audit was chemically overzealous.
- The Wittig phosphorane is a valid carrier reagent rather than an advanced product-like terminal.
- The `Cl` in a later step is supported by condition reagent `POCl3`.
- This route should be discussed as a high-value triage route, not as a strict autonomous process.

## 9. What Needs Strengthening

### 9.1 Generic Atom Contribution Instead of Reagent White Lists

Current carrier detection is useful but not enough. We need a generic `atom_contribution_profile`:

```text
terminal_smiles
raw_heavy_atoms
atoms_retained_in_product
retained_fraction
target_coverage
role_guess
role_confidence
evidence
```

Implementation options:

- exact atom mapping when mapped reactions are available
- RXNMapper or equivalent for unmapped reaction SMILES
- MCS fallback when atom mapping fails
- rule-based role priors for catalysts, solvents, salts, oxidants, reductants, protecting agents, halogenating agents, olefination reagents

The key is not a single percentage threshold. The model should ask: which atoms of this molecule persist into the product, and what chemical role explains the rest?

### 9.2 Condition Reagent Role Assignment

Condition support should be graded:

| Evidence level | Meaning | Action |
| --- | --- | --- |
| explicit reactant supported | element appears in reaction reactants | pass |
| condition reagent supported | element appears in small predicted reagent and reaction type is plausible | pass with note |
| weak condition supported | element appears only in catalyst/solvent or very large reagent | warn |
| unsupported | element appears with no source | severe issue |

We currently implemented a conservative version using only `condition_predictions[].Reagent`. The next version should infer reagent role and reaction type.

### 9.3 Enzyme Evidence Calibration

Many enzymatic steps have EC confidence around 0.07-0.20. This should not be treated as strong enzyme evidence.

Needed metrics:

- EC top-1 confidence
- EC family specificity
- substrate similarity to known enzyme reactions
- condition compatibility
- cofactor requirement
- enzyme availability or sequence evidence

### 9.4 Cascade State Model

A cascade route should carry a state:

```text
solvent / pH / temperature / redox state / cofactor / isolation flag / intermediate stability
```

Each step should update or constrain this state. Current route generation does not fully model these transitions.

### 9.5 Long-Route Compression for Expert Review

Top10 full figures are useful, but 17-20 step SVGs are too dense for a main PPT slide. We need a route abstraction layer:

- collapse known reagent-preparation subtrees
- highlight key bond-forming steps
- show only strategic disconnections in main slide
- keep full route in appendix

### 9.6 Benchmarking

Need to separate:

- route generation metrics: route count, unique signatures, depth distribution
- audit metrics: reject count, issue count, condition-supported corrections
- reranking metrics: how often corrected audit promotes chemically sensible routes
- expert-free validation metrics: consistency, atom provenance, stock realism, route diversity

If expert labels remain unavailable, benchmark should use locked synthetic test sets and rule-derived labels, plus spot-check case studies for external discussion.

## 10. Suggested PPT Structure

1. Problem statement
   - We are not only asking “does a route exist”; we ask whether a long chemoenzymatic cascade route is chemically meaningful.

2. Target and initial confusion
   - Show target SMILES/structure.
   - Show raw routes: 906 generated, not true `no route`.

3. Why stock closure is misleading
   - Examples: advanced terminal, carrier reagent, missing condition reagent.

4. International CASP landscape
   - Segler/Waller, AiZynthFinder, ASKCOS, RetroBioCat.

5. Why cascade is harder
   - Step compatibility, enzymes, conditions, cofactors, atom provenance.

6. Our CascadePlanner route-pool approach
   - Generation, audit, rerank, shortlist.

7. Route audit correction
   - Large terminal is not always advanced intermediate.
   - Condition reagent can supply missing elements.

8. Route 5 case study
   - Show `route_05.svg`.
   - Explain Wittig reagent and `POCl3` correction.

9. Top10 route figures
   - Show index or 2-3 representative SVGs.

10. Current results
   - Re-audited 640 routes: 360 triage, 232 review, 48 reject.

11. No-expert-label training plan
   - Weak supervision, rule-derived preferences, hard negatives, route-pool reranking.

12. Next work
   - Generic atom contribution, condition role assignment, enzyme calibration, cascade state model.

## 11. Recommended Claims for Expert Presentation

Safe claims:

- The system can generate long chemoenzymatic route candidates for the target.
- Existing route generation is not enough; product-level audit is required.
- Naive terminal-heavy-atom filtering is chemically wrong for carrier reagents.
- Condition reagent awareness is necessary because reaction SMILES omit important stoichiometric reagents.
- We now have route-level SVG rendering for expert review and PPT preparation.

Claims to avoid for now:

- Do not claim autonomous executable synthesis.
- Do not claim enzyme assignments are validated.
- Do not claim all 360 `triage_fragment` routes are experimentally usable.
- Do not claim the current carrier-reagent logic is fully general.

## 12. References

1. Segler, M. H. S.; Preuss, M.; Waller, M. P. Planning chemical syntheses with deep neural networks and symbolic AI. Nature 555, 604-610 (2018). https://www.nature.com/articles/nature25978
2. Genheden, S.; Thakkar, A.; Chadimova, V. et al. AiZynthFinder: a fast, robust and flexible open-source software for retrosynthetic planning. Journal of Cheminformatics 12, 70 (2020). https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00472-1
3. Coley, C. W. et al. A robotic platform for flow synthesis of organic compounds informed by AI planning. Science 365, eaax1566 (2019). https://doi.org/10.1126/science.aax1566
4. Coley, C. W.; Green, W. H.; Jensen, K. F. RDChiral: An RDKit Wrapper for Handling Stereochemistry in Retrosynthetic Template Extraction and Application. JCIM 59, 2529-2537 (2019). https://pubs.acs.org/doi/10.1021/acs.jcim.9b00286
5. Finnigan, W.; Hepworth, L. J.; Flitsch, S. L.; Turner, N. J. RetroBioCat as a computer-aided synthesis planning tool for biocatalytic reactions and cascades. Nature Catalysis 4, 98-104 (2021). https://www.nature.com/articles/s41929-020-00556-z
6. Thakkar, A.; Chadimova, V.; Bjerrum, E. J.; Engkvist, O.; Reymond, J.-L. Retrosynthetic accessibility score: rapid machine learned synthesizability classification from AI driven retrosynthetic planning. Chemical Science 12, 3339-3349 (2021). https://pubs.rsc.org/en/content/articlehtml/2021/sc/d0sc05401a
7. ACS Green Chemistry Institute. Process Mass Intensity Calculation Tool. https://www.acs.org/green-chemistry-sustainability/green-chemistry-nexus/articles/process-mass-intensity-calculation-tool.html
8. ACS GCI Pharmaceutical Roundtable. Process Mass Intensity Metric. https://learning.acsgcipr.org/guides-and-metrics/metrics/process-mass-intensity-metric/
9. Sheldon, R. A. The E factor at 30: a passion for pollution prevention. Green Chemistry 25, 1704 (2023). https://pubs.rsc.org/fa/content/articlelanding/2023/gc/d2gc04747k

