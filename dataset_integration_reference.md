# Dataset & Tool Integration Reference

## 1. ReactZyme (NeurIPS 2024, Hua et al.)

- **GitHub**: https://github.com/WillHua127/ReactZyme
- **Paper**: https://arxiv.org/abs/2408.13659
- **OpenReview**: https://openreview.net/forum?id=xepxnDQoGq
- **Zenodo (raw data)**: https://zenodo.org/records/13635807
- **Data source**: Derived from SwissProt and Rhea databases (entries up to January 8, 2024)
- **Task framing**: Enzyme-reaction prediction as a retrieval problem — rank enzymes by catalytic ability for specific reactions
- **Dataset size**: The largest enzyme-reaction dataset to date. The underlying SwissProt-Rhea data yields ~328,192 enzyme-reaction pairs (confirmed by the EnzymeFlow paper, same data source/author).
- **Raw data files** (from Zenodo):
  - `cleaned_uniprot_rhea.tsv` — main enzyme-reaction pairs
  - `uniprot_molecules.tsv` — UniProt molecule data
  - `uniprot_rhea.tsv` — UniProt-Rhea mappings
  - `rhea_molecules.tsv` — Rhea molecule SMILES
  - `saprot_seq.pt` — pre-computed structure-aware protein sequences (FoldSeek/SaProt)
- **Data format**: TSV files. Protein sequences (amino acid) + reaction SMILES from Rhea (derived from ChEBI identifiers)
- **3 split strategies**:
  1. **Time-based** (`new_time/`) — temporal split for realistic evaluation
  2. **Sequence-SMILES based** (`new_seq_smi/`) — split by sequence/SMILES similarity
  3. **Molecule-SMILES based** (`new_mol_smi/`) — split by molecular similarity
- **Only positive pairs provided** — negative sample generation is left as an open design choice (example script: `prepare_negative.py`)
- **SMILES format**: Reaction SMILES (reactants>>products) from Rhea database, based on ChEBI chemical representations
- **Protein representations supported**: ESM-2 embeddings, SaProt (structure-aware), UniMol (for reactions), MAT (Molecular Attention Transformer)
- **Note**: REXzyme (AI4PD/REXzyme on HuggingFace) is a different model — not the benchmark itself

---

## 2. EnzymeMap (Heid et al., 2023)

- **Zenodo**: https://zenodo.org/doi/10.5281/zenodo.7841780
- **Paper**: https://pubs.rsc.org/en/content/articlehtml/2023/sc/d3sc02048g (Chemical Science)
- **GitHub (code)**: https://github.com/hesther/reactiondatabase
- **Zenodo record ID**: 8254726 (latest v2)
- **Number of reactions**: 349,458 rows (349,459 including header) in the main CSV
- **Files on Zenodo**:
  - `enzymemap_v2_brenda2023.csv.gz` — 12.5 MB compressed (main dataset)
  - `raw_unmapped_v2_brenda2023.csv.gz` — 8.5 MB compressed (raw unmapped reactions)
  - `compound_to_smiles.json` — 8.8 MB (compound ID to SMILES mapping)
- **Description**: Large dataset of atom-mapped, balanced enzymatic reactions sorted by EC number, extracted from BRENDA 2023
- **Data format**: CSV with atom-mapped reaction SMILES
- **Exact columns** (14 total): `rxn_idx`, `mapped`, `unmapped`, `orig_rxn_text`, `rule`, `rule_id`, `source`, `steps`, `quality`, `natural`, `organism`, `protein_refs`, `protein_db`, `ec_num`
  - `mapped`: Atom-mapped reaction SMILES (reactants>>products with atom-map numbers, e.g. `[CH3:1]...>>[CH3:1]...`)
  - `unmapped`: Same reaction without atom mapping
  - `orig_rxn_text`: Human-readable reaction text (e.g. "acetaldehyde + NADH + H+ = ethanol + NAD+")
  - `rule`: SMARTS-based reaction template
  - `rule_id`: Template identifier
  - `source`: Direction info ("direct", "direct reversed")
  - `steps`: "single" or multi-step
  - `quality`: Confidence score (0-1 float)
  - `natural`: Boolean — whether the reaction is natural
  - `organism`: Source organism
  - `ec_num`: EC number (e.g. "1.1.1.1")
- **Template extraction**: Reaction templates extracted using RDKit-based atom-mapping; the paper validates atom-mapping quality and provides corrected/curated mappings
- **Use case**: ML models for predicting enzymatic reactions or bioretrosynthesis
- **License**: Open (see Zenodo page for specifics)

---

## 3. USPTO-50K and USPTO-MIT

### USPTO-50K
- **Canonical source**: https://github.com/connorcoley/retrosim (Coley et al.) and various retrosynthesis repos
- **Also available via**: https://github.com/otori-bird/retrosynthesis, https://github.com/bigchem/synthesis
- **TDC (Therapeutics Data Commons)**: `from tdc.generation import RetroSyn; data = RetroSyn(name='USPTO-50K')`
- **Origin**: 50,036 reactions extracted from US patent literature, classified into 10 reaction classes by Schneider et al.
- **Canonical split**: Random split. Train ~40,029 / Val ~5,004 / Test ~5,004 (exact numbers vary slightly by preprocessing version; total confirmed 50,036 via TDC)
- **Format**: CSV/TSV with columns: `id`, `class`, `reactants>reagents>product` (reaction SMILES)
- **SMILES**: Canonical RDKit SMILES, reaction format `reactants>reagents>product` or `reactants>>products`

### USPTO-MIT
- **Origin**: ~480K reactions from USPTO, processed by Jin et al. (2017)
- **Canonical source**: https://github.com/wengong-jin/nips17-rexgen
- **Also on HuggingFace**: https://huggingface.co/datasets/yuyuc/chem-uspto (~1.71M rows for the full version)
- **Full USPTO via TDC**: 1,939,253 reactions (`from tdc.generation import RetroSyn; data = RetroSyn(name='USPTO')`)
- **Split**: Train ~400K / Val ~40K / Test ~40K (80/10/10 split)
- **Format**: Reaction SMILES (reactants>>products)
- **T5Chem reference**: https://yzhang.hpc.nyu.edu/T5Chem (provides standardized USPTO_MIT splits)

---

## 4. CARE Benchmark (Yang et al., 2024 — NeurIPS 2024)

- **GitHub**: https://github.com/jsunn-y/CARE
- **Paper**: https://arxiv.org/abs/2406.15669
- **Pretrained models**: https://huggingface.co/jsunn-y/CARE_pretrained
- **Processed datasets (Zenodo)**: https://zenodo.org/records/14004425 (CARE_datasets.zip)
- **Raw data (Zenodo)**: https://zenodo.org/records/12207966 (CARE_raw_data.zip)
- **Tasks**:
  - Task 1: EC number classification — classify a protein sequence by its EC number
  - Task 2: EC number retrieval — retrieve an EC number given a chemical reaction
- **Data format**: CSV files. Each row = unique protein-EC pair (Task 1) or reaction-EC pair (Task 2)
- **Task 1 splits** (protein classification):
  - Train: `protein_train.csv`
  - Test splits: `30_protein_test.csv` (<30% identity), `30-50_protein_test.csv` (30-50% identity), `price_protein_test.csv` (misclassified/Price), `promiscuous_protein_test.csv`
  - Optional auxiliary: `reaction2EC.csv`, `text2EC.csv`
- **Task 2 splits** (reaction retrieval):
  - Easy: `easy_reaction_train.csv` / `easy_reaction_test.csv`
  - Medium: `medium_reaction_train.csv` / `medium_reaction_test.csv`
  - Hard: `hard_reaction_train.csv` / `hard_reaction_test.csv`
  - Optional auxiliary: `protein2EC.csv`, `text2EC.csv`
- **Evaluation metric**: k=1 classification/retrieval accuracy at 4 EC hierarchy levels (X.X.X.X, X.X.X.-, X.X.-.-, X.-.-.-)
- **Baseline models included**:
  - Task 1: CLEAN, Pika
  - Task 2: CREEP (Contrastive Reaction-EnzymE Pretraining), CLIPZyme
- **CREEP model**: Aligns representations from reaction, protein, and text modalities. Fine-tunes pretrained LMs. Available on HuggingFace.
- **License**: See GitHub repo

---

## 5. Rhea Database (Current)

- **Homepage**: https://www.rhea-db.org/
- **Statistics**: https://www.rhea-db.org/statistics
- **Download**: https://www.rhea-db.org/help/download
- **FTP**: https://ftp.expasy.org/databases/rhea/
- **REST API**: https://www.rhea-db.org/help/rest-api (RESTful URLs for programmatic access)
- **Formats available**:
  - RDF/OWL (primary semantic format)
  - TSV flat files
  - Complete release tarballs (from release 100 onward)
- **Number of reactions**: 18,343 unique reaction quartets (current, confirmed from rhea-db.org/statistics)
- **Unique reaction participants**: 15,125
- **Content**: Expert-curated knowledgebase of biochemical and transport reactions of biological interest
- **Chemical representation**: Uses ChEBI (Chemical Entities of Biological Interest) ontology for reaction participants
- **Reaction structure**: Each reaction stored as a "quartet" (undirected, left-to-right, right-to-left, bidirectional)
- **Cross-references**: Linked to UniProtKB (standard for enzyme/transporter annotation), EC numbers, Gene Ontology
- **SPARQL endpoint**: https://www.expasy.org/resources/sparql-rhea-db-org
- **License**: Freely available, no restrictions
- **Citation**: "Rhea, the reaction knowledgebase in 2022" (PMID: 34755880)

---

## 6. BRENDA Enzyme Database

- **Homepage**: https://www.brenda-enzymes.org/
- **Download page**: https://brenda-enzymes.org/download.php
- **Registration**: https://brenda-enzymes.org/register.php (free registration required)
- **License**: CC BY 4.0 for the online version
- **Commercial use**: Available as in-house database for commercial users via geneXplain distributor
- **Download formats**:
  - BRENDA JSON file (structured, machine-readable)
  - Flat file format
  - JSON schema documented at: https://www.brenda-enzymes.org/schemas/docs/1.1.0/brenda.schema.html
- **Temperature/pH data availability**: Yes — BRENDA contains:
  - Temperature Optimum (°C)
  - Temperature Range (°C)
  - Temperature Stability (°C)
  - pH Optimum
  - pH Range
  - pH Stability
  - Turnover Number (1/s)
  - Km values, Ki values, IC50
  - Specific Activity
- **Data fields reference**: https://www.brenda-enzymes.org/datafields.php
- **SOAP API**: https://support.brenda-enzymes.org/soap.php
- **Python parser**: https://github.com/Robaina/BRENDApyrser
- **ML-relevant**: Over 3 million enzyme entries with organism growth temperature labels (used in DeepET for thermal adaptation). Extensive kinetic parameter data suitable for ML.

---

## 7. Catechol Benchmark (NeurIPS 2025 D&B Track)

- **Paper**: https://arxiv.org/abs/2512.19530 ("Learning Continuous Solvent Effects from Transient Flow Data")
- **Also**: https://arxiv.org/abs/2506.07619 ("Time-series Solvent Selection Data for Few-shot Machine Learning")
- **OpenReview**: https://openreview.net/forum?id=6l8q74TabE
- **Hackathon page**: http://www.imperial.ac.uk/events/203225/catechol-benchmark-hackathon-neurips-2025-dnb/
- **Origin**: Imperial College London + SOLVE Chemistry
- **Dataset size**: 1,227 experimental yield measurements
- **Reaction**: Catechol rearrangement of allyl-substituted catechol
- **Conditions**: 24 pure solvents, varying temperatures and residence times
- **Data type**: High-throughput transient flow chemistry data (time-series)
- **Task**: Yield prediction, solvent selection via few-shot ML
- **Format**: Tabular (process conditions + yield measurements)
- **Note**: This is a reaction optimization/process chemistry benchmark, NOT an enzyme dataset. It focuses on solvent effects in organic chemistry.
- **Status**: NeurIPS 2025 D&B track submission. No public download link found as of April 2025. Data will likely be released via Zenodo upon acceptance. Monitor the arxiv papers (2512.19530, 2506.07619) and the Imperial hackathon page for data availability announcements.

---

## 8. Syntheseus Framework (Microsoft Research)

- **GitHub**: https://github.com/microsoft/syntheseus
- **PyPI**: https://pypi.org/project/syntheseus/
- **Documentation**: https://microsoft.github.io/syntheseus/stable/
- **Paper**: Faraday Discussions 2024 ("Re-evaluating retrosynthesis algorithms with syntheseus"); arXiv: https://arxiv.org/abs/2310.19796
- **License**: MIT

### Installation
```bash
conda env create -f environment_full.yml
conda activate syntheseus-full
pip install "syntheseus[all]"
```
Or selective install with subset of models:
```bash
pip install "syntheseus[chemformer,local-retro,megan,mhn-react,retro-knn,root-aligned]"
```
Note: `torch 2.2.2` pinned in full env. For `torch 1.x`, use `syntheseus 0.6.0`.

### Supported pre-trained models (wrappers included):
- Chemformer
- LocalRetro
- MEGAN
- MHNreact
- RetroKNN
- RootAligned

### Adding custom models
- Tutorial: https://microsoft.github.io/syntheseus/stable/tutorials/custom_model/
- Wrap your model into the shared `BackwardReactionModel` or `ForwardReactionModel` interface
- Implement `_get_reactions()` method that takes a `Molecule` and returns predicted `Reaction` objects
- Example from docs:
```python
from syntheseus.reaction_prediction.inference import BackwardReactionModel
class MyModel(BackwardReactionModel):
    def _get_reactions(self, inputs, num_results, **kwargs):
        # Your model inference here
        ...
```

### Search algorithms included:
- Retro* (A* search)
- MCTS
- Breadth-first
- PDVN
- And others

---

## 9. DESP — Double-Ended Synthesis Planning (NeurIPS 2024 Spotlight)

- **GitHub**: https://github.com/coleygroup/desp
- **Paper**: https://arxiv.org/abs/2407.06334
- **NeurIPS**: https://proceedings.neurips.cc/paper_files/paper/2024/hash/cd091a4d8e97157d32940428f902c7b0-Abstract-Conference.html
- **Authors**: Kevin Yu, Jihye Roh, Ziang Li, Wenhao Gao, Runzhong Wang, Connor W. Coley (MIT)
- **Description**: Bidirectional graph search that interleaves expansions from both the target molecule and goal starting materials
- **Key innovation**: Goal-constrained bidirectional search — combines top-down retrosynthesis with bottom-up forward synthesis

### Pre-trained models
- **Download**: https://figshare.com/articles/preprint/25956076 (`desp_data.zip`)
- **Model checkpoints included**:
  1. `model_retro.pt` — One-step retrosynthesis model (input: 2048-dim, output: 270,794 templates)
  2. `model_fwd.pt` — Forward template model (input: 4096-dim, output: 196,339 templates)
  3. `model_bb.pt` — Building block model (input: 6144-dim, output: 256-dim)
  4. `retro_value.pt` — Retro* value function (input: 2048-dim, output: 1)
  5. `syn_dist.pt` — Synthetic distance model (input: 4096-dim, output: 1)
- **Building blocks**: eMolecules catalog, stored as 256-bit Morgan fingerprints (radius 2)
- **Template files**: `idx2template_retro.json` (270,794 retro templates), `idx2template_fwd.json` (196,339 forward templates)

### Integration requirements
- GPU required (building block index ~3 GB VRAM)
- Conda environment: `conda env create -f environment.yml`
- Install data: unzip `desp_data.zip` into `/desp/data/`
- Two strategies: `f2e` (forward-to-exact) and `f2f` (forward-to-forward)

### Python API
```python
from DESP import DESP
desp = DESP(strategy='f2e')
result, route = desp.search(
    'TARGET_SMILES',
    ['STARTING_MATERIAL_SMILES']
)
```
- Key params: `iteration_limit=500`, `top_n=50` (retro templates), `top_m=25` (fwd templates), `max_depth_top=21`, `max_depth_bot=11`

### Benchmark sets included
- Pistachio Reachable, Pistachio Hard, USPTO-190 (target + starting material pairs)

---

## 10. ESM-2 / ESM-3 Protein Language Models

### ESM-2 (Meta/Facebook AI Research)

**GitHub**: https://github.com/facebookresearch/esm

**HuggingFace Model IDs and sizes**:

| Model ID | Parameters | Layers | Embedding Dim | HuggingFace URL |
|---|---|---|---|---|
| `facebook/esm2_t6_8M_UR50D` | 8M | 6 | 320 | https://huggingface.co/facebook/esm2_t6_8M_UR50D |
| `facebook/esm2_t12_35M_UR50D` | 35M | 12 | 480 | https://huggingface.co/facebook/esm2_t12_35M_UR50D |
| `facebook/esm2_t30_150M_UR50D` | 150M | 30 | 640 | https://huggingface.co/facebook/esm2_t30_150M_UR50D |
| `facebook/esm2_t33_650M_UR50D` | 650M | 33 | 1280 | https://huggingface.co/facebook/esm2_t33_650M_UR50D |
| `facebook/esm2_t36_3B_UR50D` | 3B | 36 | 2560 | https://huggingface.co/facebook/esm2_t36_3B_UR50D |
| `facebook/esm2_t48_15B_UR50D` | 15B | 48 | 5120 | https://huggingface.co/facebook/esm2_t48_15B_UR50D |

**Max input length**: 1,022 amino acids (longer sequences need truncation/chunking)

**Extracting embeddings (HuggingFace Transformers)**:
```python
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
inputs = tokenizer(sequence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Per-residue embeddings: outputs.last_hidden_state  shape: (1, seq_len, embed_dim)
# Mean pooling for sequence-level embedding:
embedding = outputs.last_hidden_state.mean(dim=1)  # shape: (1, embed_dim)
```

**Also available via the original ESM library**:
```python
import esm
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
```

### ESM-3 (EvolutionaryScale)

- **GitHub**: https://github.com/evolutionaryscale/esm
- **PyPI**: `pip install esm`
- **Open model**: `EvolutionaryScale/esm3-sm-open-v1` on HuggingFace
  - URL: https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1
  - Trained on 2.78B natural proteins (3.15B with synthetic augmentation)
  - 236M protein structures, 539M proteins with function annotations
  - Total: 771 billion tokens
- **Model family collection**: https://hf.co/collections/EvolutionaryScale/esm3-model-family
- **Architecture**: Multimodal masked generative language model — jointly processes sequence, structure, and function as discrete token tracks
- **Larger models**: Available via EvolutionaryScale API (not open-weight)
- **ESM C** (companion model): Best for pure representation learning / embeddings. Available via the `esm` PyPI package.
- **Key difference from ESM-2**: ESM-3 is multimodal (sequence + structure + function), while ESM-2 is sequence-only. For pure sequence embeddings, ESM-2 is simpler to use. For structure-aware tasks, ESM-3 is more powerful.
