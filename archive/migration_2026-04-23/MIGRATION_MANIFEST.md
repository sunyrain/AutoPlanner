# GPU Cluster Migration Manifest

**Source**: `D:\Research\AutoPlanner` (Windows workstation)
**Target**: GPU cluster (Linux + CUDA, ≥1×A100 40GB recommended)
**Date prepared**: 2026-04-23

---

## 1. Asset classification

### A. **MUST migrate** (active code + canonical data + trained artefacts) — total ≈ **820 MB**

| Category | Path | Size | Notes |
|---|---|---|---|
| Source code | `cascade_planner/` | 0.5 MB | 31 active modules |
| Canonical dataset (raw) | `cascade_dataset.json` | 30 MB | 1754 records |
| Canonical dataset (enriched) | `cascade_dataset.normalized.uniprot.json` | 24 MB | training input |
| Uniprot REST cache | `data/uniprot_cache.json` | 0.15 MB | 990 entries; saves ~3 h of API hits |
| EnzExpand ONNX (ours) | `aizdata/enzexpand_model.onnx` | 13.3 MB | 325-template MLP |
| EnzExpand template tsv | `aizdata/enzexpand_templates.csv.gz` | 20 KB | retro_template + hash |
| AiZynth USPTO model | `aizdata/uspto_model.onnx` | 87 MB | re-downloadable but slow |
| AiZynth USPTO templates | `aizdata/uspto_templates.csv.gz` | 3.2 MB |  |
| AiZynth USPTO filter | `aizdata/uspto_filter_model.onnx` | 16 MB |  |
| AiZynth ringbreaker | `aizdata/uspto_ringbreaker_model*.{onnx,csv.gz}` | 15 MB |  |
| ZINC stock | `aizdata/zinc_stock.hdf5` | **632 MB** | dominant size; can re-download via AiZynth |
| Configs | `aizdata/config*.yml` | <1 KB |  |
| Atom-mapping cache | `results/enzexpand_atommap_cache.json` | 0.4 MB | saves rxnmapper rerun |
| EnzymeMap templates | `results/enzymemap_templates.json.gz` | 0.4 MB | preprocessed |
| Demo outputs | `results/route_*.json`, `demo_*.json` | 0.13 MB | for paper figures |
| Eval results | `results/*.csv` | 0.1 MB | metric tables |

> **Tip**: ZINC stock (632 MB) can be skipped during transfer and re-downloaded on the cluster with `download_public_data` (saves migration bandwidth).

### B. **OPTIONAL** (archive — keep cold backup, do not load into active workspace) — total ≈ **143 MB**

| Path | Size | Reason |
|---|---|---|
| `archive/datasets/` | 142 MB | older snapshots (0915, 2009 etc.) — superseded by current canonical dataset |
| `archive/code/` | 0.1 MB | retired scripts |
| `archive/docs/` | 0.1 MB | early planning docs |
| `archive/logs/` | 0.2 MB | historical run logs |
| `archive/results/` | 0.8 MB | old eval results |

> Recommend: tar+gzip the whole `archive/` once, keep on cold storage / NAS, **do not deploy on cluster**.

### C. **DO NOT migrate** (rebuild on cluster) — saves ≈ **1.1 GB** + ≈ **30 MB**

| Path | Size | Why skip |
|---|---|---|
| `.venv_aizynth/` | **1.09 GB** | Python virtual env — Linux/Windows incompatible; rebuild via `pip install` |
| `ChemEnzyRetroPlanner/` | 29 MB | Web UI / Singularity scaffolding — separate concern, deploy later |
| Misc root logs (`route_*.log`, `demo_pipeline.log`) | <100 KB | reproducible from runs |

### D. **External datasets to fetch on cluster** (large, not in repo)

| Resource | Approx size | Source |
|---|---|---|
| EnzymeMap v2 (Brenda 2023) | 100 MB | already on disk at `data_external/enzymemap/enzymemap_v2_brenda2023.csv.gz` ✓ |
| ESM-3 1.4B / 7B weights | 5–30 GB | EvolutionaryScale `huggingface.co/EvolutionaryScale/esm3-sm-open-v1` |
| ChemBERTa / MolFormer-XL | 1–4 GB | HuggingFace |
| LocalRetro / Chemformer USPTO ckpts | 1–4 GB | KAIST / MolecularAI Box (auth) |
| DESP repo + value net | 0.5 GB | github.com/coleygroup/desp |
| Syntheseus framework | small + deps | github.com/microsoft/syntheseus |
| Pistachio / USPTO-50K splits | 50–500 MB | Coley Group / Reaxys |

---

## 2. Migration tarball plan

```
autoplanner_migration_2026-04-23.tar.gz       (~190 MB without ZINC, ~820 MB with)
├── cascade_planner/                          # source code
├── data/
│   ├── cascade_dataset.json
│   ├── cascade_dataset.normalized.uniprot.json
│   └── uniprot_cache.json
├── aizdata/
│   ├── enzexpand_model.onnx
│   ├── enzexpand_templates.csv.gz
│   ├── uspto_model.onnx
│   ├── uspto_templates.csv.gz
│   ├── uspto_filter_model.onnx
│   ├── uspto_ringbreaker_model.onnx
│   ├── uspto_ringbreaker_templates.csv.gz
│   ├── (zinc_stock.hdf5)                     # OPTIONAL
│   ├── config.yml
│   ├── config_eval.yml
│   └── config_hybrid.yml
├── results/
│   ├── enzexpand_atommap_cache.json          # speed-up cache
│   ├── enzymemap_templates.json.gz
│   ├── route_*.json                          # demo outputs
│   ├── demo_pipeline_report.json
│   ├── demo_picks.json
│   ├── conditions_metrics_newdataset.csv
│   ├── honest_l1_loo_newdataset.csv
│   ├── enzexpand_summary.csv
│   └── aizynthfinder_*.csv
├── data_external/enzymemap/enzymemap_v2_brenda2023.csv.gz
├── pyproject.toml | requirements.txt         # see §3
├── PROPOSAL.md                               # final SOTA proposal (this drop)
├── MIGRATION_MANIFEST.md                     # this file
└── README_CLUSTER.md                         # short setup instructions
```

Separate cold tarball: `autoplanner_archive_2026-04-23.tar.gz` (143 MB) — keep on NAS only.

---

## 3. Cluster environment spec

### Hardware
- **GPU**: 1× NVIDIA A100 40GB minimum (recommended 80GB for ESM-3 7B)
- **CPU**: ≥16 cores (atom-mapping, MCTS rollouts are CPU-bound)
- **RAM**: ≥64 GB
- **Disk**: ≥200 GB local SSD (datasets + checkpoints)

### Software
- Linux (Ubuntu 22.04 / RHEL 9)
- CUDA 12.1+
- Python 3.11
- conda or uv

### Two parallel envs (mirror current Windows setup)
1. **`autoplanner-main`** (training & condition models)
   - torch 2.5.1+cu121, transformers 4.43+, rxnmapper 0.4.3, drfp, RDKit, sklearn, onnxruntime-gpu, pandas, numpy
2. **`autoplanner-aizynth`** (multi-step search)
   - aizynthfinder 4.4.1, rdchiral 1.1.0, onnxruntime-gpu

Add for the SOTA work:
- `syntheseus`, `desp` (from source)
- `esm` (EvolutionaryScale) for ESM-3
- `transformers` upgrade for ReactionT5 / Chemma-style LLMs

---

## 4. Transfer protocol

Use the helper script `scripts/build_migration_tarball.ps1` (committed alongside this manifest).
Default behaviour: builds **active tarball only**, no ZINC stock, no `.venv_aizynth/`, no `archive/`.

Verification on receiving end:
```bash
sha256sum autoplanner_migration_2026-04-23.tar.gz
tar -tzf autoplanner_migration_*.tar.gz | head -30
```

After unpack on cluster, run:
```bash
bash README_CLUSTER.md   # see for env build + smoke test
python -m cascade_planner.demo.demo_pipeline   # sanity reruns 6 demos
```

---

## 5. Single-line summary

> ~190 MB active tarball (without ZINC) covers everything needed to **resume training on the cluster within 1 hour**. ZINC + archive add 775 MB if you want full reproducibility from cold.
