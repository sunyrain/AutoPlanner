# External weights & datasets to plug into the cascade planner

This project's snapshot is enzyme-heavy (1490 enz / 537 chem steps over 901 DOIs)
— too small to train a generic chemistry policy, and still thin for some EC
classes. We integrate or recommend the following external sources.

---

## 1. Chemistry retrosynthesis (USPTO) — INTEGRATED

- Files (already present in `aizdata/`):
  - `uspto_model.onnx`              — single-step expansion policy (~1.7M rxns)
  - `uspto_templates.csv.gz`        — 50k+ template library
  - `uspto_filter_model.onnx`       — in-scope reaction filter
  - `uspto_ringbreaker_model.onnx`  — ring-opening expander
  - `zinc_stock.hdf5`               — building-block stock
- Provider: AiZynthFinder, MolecularAI/Astra Zeneca, MIT-licensed.
  ```
  pip install aizynthfinder            # we use a separate .venv_aizynth
  download_public_data <out_dir>       # one-shot, ~700MB
  ```
- Wired via `cascade_planner/demo/aizynth_bridge.py` (subprocess, JSON IO),
  used by `cascade_planner/demo/hybrid_retrosynthesis.py`.
- Head-to-head numbers on our cascade steps:
  - chemical steps (n=537):  AiZynth top-50 ≈ 23–35% (depending on slice).
  - enzymatic steps (n=1490): AiZynth top-50 ≈ 8% (out-of-domain, expected).

## 2. Enzymatic retrosynthesis — IN-HOUSE + CANDIDATES TO ADD

In-house: EnzExpand-A — rdchiral atom-mapping + 1024-MLP over Morgan-2.
- 161 templates (mf≥2), 633 trainable steps; CV top-50 ≈ 53%.
- Trained model NOT yet pickled; the demo retrains it each run (~30 s).

External weight options (none integrated yet, all open):

| Source | Coverage | License | Format | URL |
|---|---|---|---|---|
| **EnzymeMap** (Heid et al., 2023) | ~30k atom-mapped enzymatic rxns + extracted templates | CC-BY-4.0 | JSON / SMILES | https://zenodo.org/records/8338363 |
| **RetroBioCat** rule set | ~600 expert-curated biocatalysis SMARTS rules | MIT | JSON | https://github.com/willfinnigan/RetroBioCat (data/biocatalysis_rules.json) |
| **RetroRules** biocatalysis subset | 412k Rhea/MetaCyc-derived rules at 4 radii | Open data | TSV | https://retrorules.org/ |
| **ECREACT** | ~62k EC-tagged reaction SMILES (USPTO + Rhea) | CC-BY-4.0 | TSV | https://github.com/rxn4chemistry/biocatalysis-model |
| **RXN4Chem biocatalysis-T5** | T5 forward/retro fine-tuned on ECREACT | Apache-2.0 | HuggingFace | https://huggingface.co/rxn4chemistry/biocatalysis-T5 |

Recommended next addition: **EnzymeMap templates** (drop-in replacement for our
`extract_templates`, gives us ~10x more reactions for the MLP). Plug-in path:
```
cascade_planner/expand/enzymemap_loader.py    # to create
  load EnzymeMap json -> [(product_smi, template_smarts), ...]
  union with our extracted pairs -> larger TemplateMLP
```

## 3. Reaction conditions — IN-HOUSE + STRONGER OPTIONS

In-house: 5 logreg / DRFP-2048 heads + EC1-stratified T mean.
- EC1 logreg macroF1=0.591 / acc=73.5%; transformation 0.517; T MAE=13.5 °C.

Stronger external option (pre-trained, plug-and-play):

| Source | What it predicts | License | URL |
|---|---|---|---|
| **Parrot** (Wang et al., 2023) | catalyst+solvent+reagent+T as a sequence | MIT | repo: `retro_planner/packages/parrot/` (already vendored, has `Dockerfile_gpu` + `serve_parrot_in_docker.sh`) — needs USPTO checkpoint download |
| **RXN-Reagents (IBM RXN)** | reagents/conditions for an arbitrary rxn SMILES | requires API key | https://rxn.app.accenture.com/ |
| **ChemEnzyRetroPlanner condition_predictor** | wrapper around Parrot | inherited | `retro_planner/packages/condition_predictor/` |

Status: Parrot serving was set up for this repo (`config_inference_use_uspto.yaml`,
`build_parrot_in_docker.sh`); not yet wired into the cascade demo.

## 4. Enzyme function / active site — VENDORED, NOT WIRED

- **EasIFA** (Active-site identification): `retro_planner/packages/easifa/`.
- **organic_enzyme_rxn_classifier** (binary chem/enz + EC predictor):
  `retro_planner/packages/organic_enzyme_rxn_classifier/`. Needs `rxnfp` +
  pre-trained checkpoint (not in repo).

Use case in cascade planner: confirm a proposed enzyme step is feasible by
predicting the EC + active-site residues of the candidate transformation.

---

## Concrete short-term TODOs

1. **Cache trained EnzExpand-A** in `results/enzexpand_model.pt` + `tpls.json`,
   so demo no longer retrains every call.
2. **Pull EnzymeMap** (~30 MB) into `data_external/enzymemap/` and union into
   `extract_templates`. Expected: more EC2 / EC4 templates.
3. **Run Parrot once** via the existing `serve_parrot_in_docker.sh`, then
   call its REST endpoint from `hybrid_retrosynthesis.py` to replace our
   logreg solvent/catalyst heads with Parrot's full condition sequence.
4. **Wire RetroBioCat rules** as a 3rd, rule-only enzyme expander
   (no training needed, very high precision on green-chem reactions).
5. **Publish a hybrid head-to-head** on the cascade snapshot:
   USPTO + EnzExpand vs USPTO + EnzymeMap + RetroBioCat, by EC and by step.
