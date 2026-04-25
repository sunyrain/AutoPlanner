# AutoPlanner — SOTA Upgrade Proposal (2026 H2)

**Project**: ChemEnzyRetroPlanner-Next  
**Authors (model side)**: AutoPlanner team  
**Counterparty (data side)**: cascade dataset curation team  
**Status**: Draft v1.0 — for cluster migration & joint review  
**Date**: 2026-04-23  

---

## 0. TL;DR

We propose a **6-month, mutual-commitment upgrade** of the ChemEnzyRetroPlanner stack to **published 2024–2026 SOTA** in six modules (single-step chem, single-step enz, multi-step search, condition prediction, enzyme recommender, route scoring), benchmarked on **public third-party suites** (Syntheseus, ReactZyme, CARE, Catechol). In return, the dataset team commits to deliver the prioritised data improvements (P0 + P1) on the same timeline. **All metrics in §7 are contractual.**

---

## 1. Current state (v1, April 2026)

### 1.1 Pipeline

```
raw cascade JSON ──► normalize ──► Uniprot enrich ──► atom-map (RXNMapper)
                                                               │
                          ┌────────────────────────────────────┘
                          ▼
       ┌──────────────────────────────┐
       │ EnzExpand template MLP       │   (DRFP-2048 → 325 templates, ONNX)
       │ Condition heads (5)          │   (T, pH, solvent, EC1, transformation)
       │ Enzyme recommender           │   (Tanimoto + frequency + Uniprot)
       └──────────────┬───────────────┘
                      │
                      ▼
        AiZynth MCTS (USPTO + EnzExpand multi-policy)
                      │
                      ▼
         Annotated multi-step routes
```

### 1.2 Canonical dataset (v1)
- 1754 records / 3115 cascades / 6306 step-rows / 4243 enzyme components
- Native Uniprot ID coverage 38.4% → **enriched to 77.8%**
- 1499 unique (EC, organism) pairs · 3295 unique transformations
- 325 single-LHS retro templates (ONNX-deployable)

### 1.3 Headline metrics (v1, all DOI-grouped CV)

| Module | Metric | v1 |
|---|---|---|
| EnzExpand template | EC1-avg top-1 / top-10 | **25 % / 50 %** |
| Condition (T) | MAE / R² | **13.2 °C / negative** |
| Condition (pH) | MAE | **0.74** |
| Condition (EC1 logreg) | acc / top-3 | **72.4 % / 95.0 %** |
| Condition (catalyst class) | acc / top-3 | **60.2 % / 86.1 %** |
| Condition (transformation) | acc / top-3 | **58.3 % / 82.3 %** |
| Condition (solvent top-12) | acc / top-3 | **58.7 % / 78.9 %** |
| Multi-step (depth ≤6) | est. solve rate | **~55 %** |
| Multi-step | GT@5 recall (when GT path known) | **~30 %** |
| Enzyme recommender | Uniprot top-1 hit | **~15 %** |

---

## 2. Verified 2024–2026 SOTA references

> Every paper below has been independently verified on arxiv.org / NeurIPS proceedings. Sources marked **★** are the strongest candidates for direct integration; **△** are alternatives.

### 2.1 Single-step retrosynthesis (chemical)

| Method | Year | Reference | Headline |
|---|---|---|---|
| **★ RetroChimera** | Dec 2024 | Maziarz et al., arXiv:2412.05269 | Ensemble of diverse inductive biases; pharma-chemist preference |
| △ Atom-anchored LLMs | Oct 2025 | Hassen et al., arXiv:2510.16590 | ≥74 % reactant prediction; LLM with atom anchors |
| △ DiffER | May 2025 | Current et al., arXiv:2505.23721 | Categorical diffusion for retrosynthesis |
| △ SynBridge | Jul 2025 | Lin et al., arXiv:2507.08475 | Discrete flow, bidirectional |

### 2.2 Single-step retrosynthesis (enzymatic / biocatalysis)

| Method | Year | Reference | Headline |
|---|---|---|---|
| **★ ReactZyme** | Aug 2024 | Hua et al., NeurIPS 2024 | 81K SwissProt × Rhea pairs; canonical enzyme-reaction benchmark |
| **★ EnzymeFlow** | Oct 2024 | Hua et al., arXiv:2410.00327 | Flow matching for reaction-conditioned pocket generation |
| △ GENzyme | Nov 2024 | Hua et al., arXiv:2411.16694 | Reaction-conditioned de novo enzyme design |

### 2.3 Multi-step search

| Method | Year | Reference | Headline |
|---|---|---|---|
| **★ DESP** | Jul 2024 | Yu et al., NeurIPS 2024 (Spotlight) | Bidirectional, goal-constrained; SOTA solve-rate + expansion-efficiency |
| **★ Syntheseus (framework)** | Oct 2023 / 2024 | Maziarz et al., arXiv:2310.19796; Faraday Discussions 2024 | Reference benchmarking framework over 21 algorithms |
| △ DirectMultiStep | May 2024 | Shee et al., JCIM 2024 | Direct end-to-end multi-step generation; MoE transformer |
| △ Retro-fallback | Oct 2023 | Tripp et al., ICLR 2024 | Uncertainty-aware ensemble |

### 2.4 Reaction condition prediction

| Method | Year | Reference | Headline |
|---|---|---|---|
| **★ Chemma** | Apr 2025 | Zhang et al., arXiv:2504.18340 | LLM-based; SOTA yield prediction; validated on real Suzuki–Miyaura experiments |
| △ Modular Multi-Task | Feb 2026 | Pang et al., arXiv:2602.10404 | LLM adapters for T, pH, solvent, reagent jointly |
| **★ Catechol Benchmark** | Jun 2025 | Boyne et al., NeurIPS 2025 (Benchmarks) | Few-shot solvent selection; first standard benchmark |

### 2.5 Enzyme function & recommendation

| Method | Year | Reference | Headline |
|---|---|---|---|
| **★ CARE Benchmark** | Jun 2024 | Yang et al., arXiv:2406.15669 | Enzyme classification + retrieval benchmark |
| **★ ReactZyme** (also in §2.2) | Aug 2024 | Hua et al., NeurIPS 2024 | Reaction → enzyme retrieval task |
| △ RXNRECer | Mar 2026 | Shi et al., arXiv:2603.12694 | Active-learning + PLM for fine-grained EC annotation |
| △ UniZyme | Feb 2025 | Li et al., arXiv:2502.06914 | Active-site knowledge for enzyme prediction |

### 2.6 Route scoring & feasibility

| Method | Year | Reference | Headline |
|---|---|---|---|
| **★ RetroCast / SynthArena** | Dec 2025 | Morgunov et al., arXiv:2512.07079 | Unified evaluation framework, chemical validity emphasis |
| **★ Leap** | Mar 2024 | Calvi et al., arXiv:2403.13005 | Synthesizability scoring with intermediate awareness; > SCScore/RAScore |
| △ Trustworthy Retro Ensemble | Oct 2025 | Sadowski et al., arXiv:2510.10645 | Diverse-ensemble hallucination reduction |

### 2.7 Foundation models & dependencies

| Asset | Year | Source |
|---|---|---|
| **ESM-3 (open small)** | 2024 | EvolutionaryScale, HuggingFace `EvolutionaryScale/esm3-sm-open-v1` |
| **Boltz-1** / **Chai-1** (protein–ligand co-folding) | 2024–2025 | Boltz / Chai labs (optional for binding pre-screen) |
| **MolFormer-XL / ChemBERTa-3** | 2024–2025 | HuggingFace |

---

## 3. Module-by-module upgrade plan

> **Convention**: each module gives (a) what we replace, (b) the chosen SOTA reference, (c) implementation steps, (d) acceptance metric & benchmark.

### M1 — Single-step chemical retrosynthesis: AiZynth-default MLP → RetroChimera-style ensemble

- **Current**: AiZynth's bundled USPTO MLP, top-1 ≈ 44 % (USPTO-50K).
- **Target**: ensemble of three diverse inductive biases per RetroChimera (template + graph + transformer).
- **Steps**:
  1. Integrate **Syntheseus** as the model bus (it already wraps LocalRetro, MEGAN, GLN, transformer-retro).
  2. Add 1 transformer member (LocalRetro or DirectMultiStep encoder for diversity).
  3. Ensemble logits (RetroChimera-style score combination).
- **Acceptance metric**: USPTO-50K top-1 ≥ **52 %** (verified by Syntheseus harness, public split).

### M2 — Single-step enzymatic retrosynthesis: DRFP→MLP → ReactZyme + ESM-3 dual encoder

- **Current**: DRFP-2048 → MLP(1024) → 325 templates; EC1-avg top-1 25 %.
- **Target**: dual-tower (a) reaction graph encoder, (b) ESM-3 enzyme embedding; trained on ReactZyme + our cascade dataset.
- **Steps**:
  1. Adopt **ReactZyme** train/test splits as benchmark; publish our model on the official task.
  2. Reaction encoder: graph2smiles or LocalRetro-enz variant (atom-/bond-centred local templates).
  3. Enzyme tower: pre-trained **ESM-3-sm-open** frozen, with a 2-layer projection.
  4. InfoNCE contrastive training (CLEAN-style).
  5. Re-export as ONNX for plug-in to AiZynth bridge.
- **Acceptance metric**:
  - EC1-avg single-step top-1 ≥ **45 %** (DOI-grouped CV on our v2 dataset).
  - Top-1 on EC2 transferases ≥ **15 %** (currently 0 %).
  - ReactZyme retrieval Recall@10 ≥ baseline + 3 pp (third-party benchmark).

### M3 — Multi-step search: AiZynth UCT-MCTS → Syntheseus + DESP

- **Current**: AiZynth MCTS UCT, est. solve-rate ~55 %, ~60–120 s/target.
- **Target**: drop-in **Syntheseus** runner with **DESP** (bidirectional, goal-constrained) as primary search.
- **Steps**:
  1. Wrap our EnzExpand ONNX as a Syntheseus `BackwardReactionModel`.
  2. Add **DESP** as the main planner; keep AiZynth MCTS as fallback / ablation.
  3. (Optional) train a **value head** on cascade dataset's `is_demonstrated_success` + yield as DESP guidance.
- **Acceptance metric**:
  - Solve rate on a 100-target unseen-DOI benchmark: ≥ **75 %** (depth ≤ 6).
  - GT@5 recall when full GT route known: ≥ **60 %**.
  - Average wall-clock ≤ **45 s/target** (single A100 + 16-core CPU).

### M4 — Reaction condition prediction: DRFP+linear → Chemma-style multi-task LLM

- **Current**: DRFP-2048 → logreg/ridge separately per head; T MAE 13.2 °C, R² < 0.
- **Target**: shared LLM/transformer backbone with 5 heads (T, pH, solvent, EC1, transformation), optionally with **ESM-3 cross-attention** for enzyme-specific conditions.
- **Steps**:
  1. Backbone choice: a small LLM adapter approach (Chemma-style) or a from-scratch RXN-Transformer if compute-bound.
  2. Enzyme cross-attention path: when catalyst is enzyme, fuse ESM-3 sequence embedding.
  3. Joint loss: regression (T/pH) + multi-class (solvent/EC1/transform) + auxiliary yield head.
  4. Evaluate on: (a) our DOI CV; (b) **Catechol Benchmark** for solvent.
- **Acceptance metric**:
  - T MAE ≤ **8 °C**, R² ≥ **0.30**.
  - pH MAE ≤ **0.50**.
  - Solvent top-3 ≥ **85 %**; on Catechol few-shot ≥ baseline.

### M5 — Enzyme recommender: Tanimoto + frequency → ESM-3 × DRFP contrastive (CLEAN-2-style)

- **Current**: Morgan2 Tanimoto + frequency on 4243 components; Uniprot top-1 hit ~15 %.
- **Target**: dual-tower contrastive model on (reaction, enzyme sequence). Aligns with **ReactZyme** and **CARE**.
- **Steps**:
  1. Train ESM-3 (frozen) protein tower + DRFP/transformer reaction tower with InfoNCE on Rhea/SwissProt (~81K pairs from ReactZyme).
  2. Fine-tune on cascade dataset enzyme components (4243 + 3300 enriched).
  3. Inference: embed query reaction → ANN over Uniprot enzyme bank → top-K with EC + organism.
- **Acceptance metric**:
  - Uniprot top-1 hit ≥ **35 %** on DOI-LOO from our dataset.
  - EC4 top-1 ≥ **50 %**.
  - **CARE retrieval Recall@10** ≥ baseline (third-party).

### M6 — Route scoring: in-stock + prior → Leap + LLM-judge

- **Current**: AiZynth's `in_stock_frac` + policy prior; no learned route scorer.
- **Target**: composite reward = (a) **Leap** synthesizability + (b) DESP value head + (c) **LLM-as-judge** rubric (Claude/GPT-4o); plus our cofactor regrader for enz steps.
- **Steps**:
  1. Integrate Leap as a step-level scorer.
  2. Train DESP value head on cascade dataset success labels.
  3. Stand up an LLM-judge prompt with chemist-defined rubric (cost / safety / cofactor friendliness).
  4. Compose reward; validate against expert rankings on 50 routes.
- **Acceptance metric**:
  - GT-route in top-5 recall ≥ **60 %** on 100-target benchmark.
  - Spearman ρ with expert ranking on 50 routes ≥ **0.65**.

---

## 4. Phasing

| Phase | Months | Model deliverables | Data deliverables (counterparty) |
|---|---|---|---|
| **P1 — Foundations** | M0 → M1 | Migrate to cluster · stand up Syntheseus + DESP wrapper · ESM-3 download · M5 baseline (CLEAN-style) | P0 batch: EC2 + EC6 reactions reach n≥400/200; rxn_smiles 100 % parsable |
| **P2 — Core** | M1 → M3 | M1 (RetroChimera-style ensemble) + M4 (Chemma-style condition LLM) integrated; mid-term benchmark pass | P1 batch: condition fields strongly typed; solvent vocabulary; yield ≥ 50 % coverage; stereochemistry consistency |
| **P3 — Scoring & evaluation** | M3 → M5 | M3 (DESP value head trained) + M6 (Leap + LLM-judge); full 7-metric KPI report on Syntheseus + ReactZyme + CARE + Catechol | P2 batch: intermediates SMILES 100 %; cofactor controlled vocab; substrate-scope examples; ≥200 negative samples |
| **P4 — Final benchmark & paper** | M5 → M6 | Fixed-seed third-party benchmarks; reproducibility tarball; v3 model card | Final 100-target held-out benchmark frozen jointly |

---

## 5. KPI table (contract core)

| # | KPI | Module | Current (v1) | Target (v2, M+6) | Verification |
|---|---|---|---|---|---|
| K1 | EnzExpand top-1 (EC1-avg) | M2 | 25 % | **45 %** | DOI-CV on v2 dataset |
| K2 | Single-step chem top-1 (USPTO-50K) | M1 | ~44 % | **≥ 52 %** | Syntheseus public split |
| K3 | Multi-step solve rate (depth ≤ 6) | M3 | ~55 % | **≥ 75 %** | 100-target unseen-DOI |
| K4 | Multi-step GT@5 recall | M3+M6 | ~30 % | **≥ 60 %** | 100 GT-routes |
| K5 | Condition T MAE | M4 | 13.2 °C | **≤ 8 °C** | DOI-CV; R² ≥ 0.30 |
| K6 | Condition pH MAE | M4 | 0.74 | **≤ 0.50** | DOI-CV |
| K7 | Uniprot recommender top-1 | M5 | ~15 % | **≥ 35 %** | DOI-LOO; ReactZyme/CARE Recall@10 ≥ baseline |

> Failure clause: any KPI < 80 % of its target value triggers a joint review — additional engineering effort and/or scope adjustment, not a data-side penalty.

---

## 6. Counterparty commitments (data team)

For our KPIs to be reachable, the dataset team commits to:

1. **P0 delivery within M+1**:
   - EC2 transferase reactions ≥ 400 (currently 69); EC6 ligase ≥ 200 (currently 7).
   - 100 % `rxn_smiles` field coverage; all RDKit-parsable.
   - Native `uniprot_id` coverage ≥ 80 % (currently 38 %); add NCBI/PDB IDs where available.

2. **P1 delivery within M+3**:
   - All `temperature_c`, `ph` numeric (no strings); solvent controlled vocabulary (≤ 20 entries); all units explicit.
   - `step_yield_percent` ≥ 50 % coverage; `step_ee_percent` 100 % on chiral.
   - Stereochemistry consistency: flat SMILES for racemic; full stereo SMILES for selective + `selectivity` flag.

3. **P2 delivery within M+5**:
   - `intermediates SMILES` 100 % coverage (incl. transient marked).
   - `cofactor_required` controlled vocabulary; regeneration system; stoichiometry.
   - Substrate-scope examples per enzyme (≥ 3 where reported); ≥ 200 negative samples.

4. **Schema lock**: v2 JSON Schema co-signed; CI gate on every commit.

5. **Held-out test set**: 100 records co-frozen; not visible to model team during development.

6. **Quality control**: ≥ 5 % expert sampling for P1 fields.

---

## 7. Cluster compute budget (estimate)

| Workload | GPU-hours (A100 40GB) |
|---|---|
| ESM-3-sm fine-tune (M5) | ~80 |
| LocalRetro / RetroChimera training (M1, M2) | ~120 |
| Chemma-style condition LLM (M4) | ~60 |
| DESP value-head training (M3, M6) | ~40 |
| Benchmark sweeps (Syntheseus + ReactZyme + CARE + Catechol) | ~60 |
| Buffer (debugging, ablations) | ~80 |
| **Total** | **≈ 440 GPU-hours** |

---

## 8. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| ESM-3 7B doesn't fit on 40GB A100 | M | Use ESM-3 small open (1.4B); fallback to ESM-2 650M |
| Chemformer / Box-auth checkpoints unavailable | M | Use HuggingFace open MolFormer-XL or ChemBERTa-3 |
| Counterparty P0 data slips | M | We continue M3 + M6 work in parallel which doesn't depend on EC2/6 expansion |
| Syntheseus ↔ ONNX integration friction | L | Already validated; minor wrapper work |
| Real GPU cluster CUDA mismatch | L | Pin torch 2.5.1+cu121 + lock file in tarball |
| LLM-as-judge cost / reproducibility | M | Cache all judge calls; use open Llama-3-70B as fallback |

---

## 9. Repository structure target on cluster

```
~/AutoPlanner/
├── cascade_planner/                  # current package (preserved)
├── autoplanner_next/                 # NEW: SOTA modules
│   ├── single_step/                  # M1 ensemble + M2 dual-tower
│   ├── multi_step/                   # M3 desp wrappers
│   ├── conditions/                   # M4 LLM condition heads
│   ├── recommender/                  # M5 contrastive
│   └── scorer/                       # M6 Leap + LLM-judge
├── benchmarks/                       # Syntheseus / ReactZyme / CARE harnesses
├── checkpoints/                      # ESM-3 / Chemformer / our trained
├── third_party/                      # syntheseus, desp, leap clones
├── aizdata/, results/, data/, ...    # current data + caches
└── PROPOSAL.md (this file)
```

---

## 10. Sign-off

| Party | Name / Role | Signature | Date |
|---|---|---|---|
| Model team lead |  |  |  |
| Data team lead |  |  |  |
| PI |  |  |  |

---

## Appendix A — verified arxiv links (one-line each)

- RetroChimera: https://arxiv.org/abs/2412.05269
- Atom-anchored LLMs: https://arxiv.org/abs/2510.16590
- DiffER: https://arxiv.org/abs/2505.23721
- SynBridge: https://arxiv.org/abs/2507.08475
- ReactZyme (NeurIPS 2024): https://arxiv.org/abs/2408.13659
- EnzymeFlow: https://arxiv.org/abs/2410.00327
- GENzyme: https://arxiv.org/abs/2411.16694
- DESP (NeurIPS 2024 Spotlight): https://arxiv.org/abs/2407.06334
- Syntheseus framework: https://arxiv.org/abs/2310.19796 · github.com/microsoft/syntheseus
- Retro-fallback (ICLR 2024): https://arxiv.org/abs/2310.09270
- DirectMultiStep (JCIM 2024): https://arxiv.org/abs/2405.13983
- Chemma: https://arxiv.org/abs/2504.18340
- Catechol Benchmark (NeurIPS 2025 B&D): https://arxiv.org/abs/2506.07619
- CARE Benchmark: https://arxiv.org/abs/2406.15669
- RXNRECer: https://arxiv.org/abs/2603.12694
- UniZyme: https://arxiv.org/abs/2502.06914
- RetroCast / SynthArena: https://arxiv.org/abs/2512.07079
- Leap: https://arxiv.org/abs/2403.13005
- Trustworthy Retro Ensemble: https://arxiv.org/abs/2510.10645

> ⚠️ Five names from our earlier internal brainstorm are **not** real published works and are explicitly retracted: "R-SMILES 2", "RxnFormer", "BioCatSet", "AlphaSynthesis", "CHIMERA-LLM". The correct, verified analogues are above.

---

## Appendix B — current asset list (April 2026)

See `MIGRATION_MANIFEST.md` (separate file in repository root). Active migration tarball ≈ 190 MB without ZINC stock; ≈ 820 MB with ZINC.
