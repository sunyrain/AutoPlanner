# 数据审计与资源调配计划 2026-05-27

## 结论摘要

当前项目已经具备一批有价值的原始数据和中间实验结果，但还没有完成下一阶段最关键的训练资产建设。

核心判断：

1. 本地已有 EnzymeMap、ECREACT、Rhea、ReactZyme、酶反应 JSON、v4 cascade、30K verifier preference、外部化学反应训练包和 AiZ/RetroKNN proposal 资产。
2. 这些数据大多仍处于“原始反应库/旧实验包/展示包”状态，尚未统一转成 `chemical_product_pool`、`enzyme_substrate_product_pool`、`bridge tiers`、`hard negative pool`、`enzyme feasibility verifier` 训练集。
3. 目前最应优先建设的不是端到端大模型，而是：数据标准化、chemo-enzymatic bridge weak labels、hard negatives、retriever/verifier、gated proposal routing。
4. 当前机器有 2 张 RTX 4080 SUPER 32GB、128 CPU、503GB RAM，适合 P0/P1 小到中等规模训练；但本机 `/root/autodl-tmp` 只剩约 84GB，可用于审计和原型，不足以长期承载全量数据工程和大规模缓存。
5. 如果要全局优先本项目，最需要调配的是：更大 NVMe 存储、FAISS/向量检索环境、完整 BRENDA/RetroBioCat/MetaCyc/KEGG 或授权商业反应数据、以及 4 卡级别训练节点。

## 当前数据资产

### 化学反应与 proposal 资产

| 资产 | 本地状态 | 审计结果 | 作用 |
|---|---:|---:|---|
| ChemEnzyRetroPlanner vendor | 已有 | `vendor/ChemEnzyRetroPlanner` 约 7.9GB | 原生 planner、stock、模型依赖 |
| ChemEnzy stock | 已有 | `origin_dict.csv` 约 1.29GB；ZINC stock 约 524MB | 购买/库存闭合判断 |
| AiZynthFinder ONNX 模型 | 已有 | `workspace/aizdata` 约 767MB | sidecar chemical proposer |
| RetroKNN 外部 150K 索引 | 已有 | `retroknn_external_150k_onestep.pkl` 约 64MB | 快速检索式 proposer |
| 外部 top-level 训练包 | 已有 | train 98,147；valid 5,576；test 5,460 | adapter / proposal 训练实验 |
| Pistachio ringbreaker MAR | 已放入根目录 | 130.9MB | 需要进一步接入/解析 |
| BKMS metabolic MAR | 已放入根目录 | 152.9MB | 需要进一步接入/解析 |

已验证的 proposal 结果：

| 指标 | ChemEnzy native | AiZ USPTO | RetroKNN 150K | 三者 union |
|---|---:|---:|---:|---:|
| benchmark_v2_100 top1 命中 | 5/100 | 8/100 | 16/100 | 21/100 |
| top5 命中 | 12/100 | 16/100 | 17/100 | 29/100 |
| top16 命中 | 17/100 | 17/100 | 18/100 | 33/100 |
| 平均单步延迟 | 1.49s | 0.276s | 0.110s | 未单独计 |

解释：新增 proposal 能显著增加单步候选覆盖，但直接塞进多步搜索会污染路线池。5 个目标的 limit5 smoke 中，native 平均 0.028s，ensemble top4 平均 12.97s，full ensemble 平均 27.57s；exact reaction in route pool 从 native 的 0.2 降到 0.0。这说明重点必须转到 gated routing 和 verifier，而不是无差别 ensemble。

### 酶反应数据

| 数据源 | 本地状态 | 审计结果 | 说明 |
|---|---:|---:|---|
| EnzymeMap v2 BRENDA 2023 | 已有 | 349,458 行；unique unmapped 47,640；unique mapped 47,974；EC 4,552 | 最适合构建 enzyme substrate/product pool 和反应中心规则 |
| ECREACT 1.0 | 已有 | 62,222 条；EC 6,289；source 4 类 | EC-conditioned enzymatic proposer / verifier 数据源 |
| enzymatic_retro_data | 已有 | train 318,900；val 35,433；train EC 4,693 | 反应级 enzyme retrosynthesis 数据 |
| Rhea release 140 | 压缩包已有 | 436MB；archive 内 125 个成员，含 SDF、RDF、txt、Biopax | 高质量 balanced reaction 和 ChEBI/EC 链接，尚需展开标准化 |
| ReactZyme | 压缩包已有 | 517MB；含 reaction split、enzyme split、UniProt/Rhea 表、SaProt seq | 可用于 enzyme-substrate/reaction 数据扩展，尚需展开标准化 |
| BRENDA lookup | 只有派生小表 | `brenda_lookup_full.json` 5,622 条条件记录 | 条件辅助可用，但不是完整 BRENDA 反应库 |
| 本地 enzyme sequence TSV | 很小 | 1,116 条 UniProt；305 个 EC | 不足以训练 sequence-aware verifier，只能做 smoke / cache |
| ESM cache | 很小 | 约 4.7MB | 不是系统性 sequence embedding 资产 |

EnzymeMap 详细审计：

| 项 | 数值 |
|---|---:|
| 总行数 | 349,458 |
| unique EC | 4,552 |
| unique mapped reaction | 47,974 |
| unique unmapped reaction | 47,640 |
| natural=True | 77,938 |
| natural=False | 271,520 |
| rows with protein refs | 110,308 |
| unique organisms | 7,467 |

### Cascade 与旧 verifier 资产

| 资产 | 本地状态 | 审计结果 | 说明 |
|---|---:|---:|---|
| cascade v4 all quality index | 已有 | 3,810 条 | 全部 v4 cascade 索引 |
| cascade v4 high quality | 已有 | 3,744 条 | 高质量 cascade 主集 |
| cascade v4 steps | 已有 | 8,609 行 | step 条件和反应信息 |
| cascade v4 species | 已有 | 21,225 行 | species/input/output 信息 |
| cascade v4 catalysts | 已有 | 9,444 行 | enzyme/catalyst 元数据 |
| cascade v4 gold reactions | 已有 | 2,885 条 | gold 子集 |
| cascade v4 substrate scope | 已有 | 3,458 行 | scope 信息 |
| verifier DPO pairs | 已有 | 29,079 对 | 规则扰动偏好包 |
| verifier chosen seed pack | 已有 | 1,477 seeds | 旧 cascade verifier 主线 |
| pseudo_cascades_enzymemap | 已有 | 10,000 条 2-step skeleton | 只有 target + step 类型/EC1，不是可训练 molecule-chain bridge |

关键解释：旧 verifier 主要解决的是 route/cascade 合理性、条件兼容、规则扰动偏好；它不是新的 substrate-product-enzyme feasibility verifier。二者不能混为一谈。

## 当前完成度判断

这是针对“下一阶段 enzyme bridge / verifier / gated search 主线”的完成度估计。

| 模块 | 完成度 | 判断 |
|---|---:|---|
| 原始公开数据落盘 | 约 60-70% | EnzymeMap、ECREACT、Rhea、ReactZyme、酶反应 JSON 已有；但 BRENDA full、RetroBioCat、MetaCyc/KEGG/PathBank/MetaNetX、商业反应库尚不完整 |
| 化学 product pool | 约 20% | 有外部 top-level 109K 和 stock，但尚未统一 canonical/InChIKey/fingerprint/parquet |
| enzyme substrate/product pool | 约 20% | 反应库足够，但尚未标准化成 substrate/product/EC/cofactor/protein/evidence schema |
| exact bridge pack | 0-5% | 没看到当前主线成品 `exact_bridge.parquet/jsonl`；旧 template bridge audit 不能替代 |
| similarity bridge pack | 0-5% | 没看到成品 similarity bridge 分层数据 |
| cofactor/common metabolite blacklist | 0-10% | 规则里有零散处理，但没有独立可复用数据资产 |
| enzyme hard negatives | 0-10% | 有旧 route hard-negative/CCTS 实验，但不是 enzyme-substrate/product hard negatives |
| substrate-product-enzyme verifier | 0-10% | 尚未训练；旧 cascade verifier 不覆盖这个核心任务 |
| bridge retriever | 10-20% | RetroKNN/AiZ sidecar 证明检索/补 proposal 有价值；enzyme bridge retriever 尚未成型 |
| gated proposal routing | 10-20% | 已证明 naive ensemble 会变差；router/gate 还未主线化 |
| 3D/active-site validator | 0-5% | 只有极少 UniProt/ESM cache，没有批量结构验证资产 |

总体结论：P0 数据审计已经具备基础；P0 数据产品尚未完成。下一步应进入“数据产品化”而不是继续直接跑多步搜索。

## 当前阻塞点

### 1. 数据还没有统一 schema

目前数据分散在 CSV、JSON、JSONL、压缩包和旧结果目录中。需要统一为：

```text
reaction_id
source
rxn_smiles
mapped_rxn_smiles
substrates
products
ec_number
enzyme_id / uniprot_id
organism
cofactors
reaction_center
stereochemistry
evidence_source
quality_flags
canonical_smiles
inchikey
fingerprint
```

没有这个 schema，后续 verifier、retriever、bridge scorer 都只能继续做临时脚本。

### 2. Bridge weak label 尚未成品化

我们需要的不是“有很多酶反应”，而是：

```text
chemical product exact/similar to enzymatic substrate
chemical product exact/similar to enzymatic product
reaction center compatible
stereochemistry compatible
EC/reaction type plausible
not cofactor/common metabolite artifact
```

当前没有成品 `bridge_confidence_tiers`，因此还无法训练真正的 enzyme bridge scorer。

### 3. Hard negative 还没对准核心问题

旧 hard-negative/CCTS 实验说明在 ChemEnzy 候选池内重排有小幅收益，但没有解决“错误酶步太多”的根因。下一阶段负样本应围绕：

| 负样本类型 | 目的 |
|---|---|
| 同 EC 不同底物空间 | 防止只看 EC |
| 相似底物但反应中心不同 | 防止 fingerprint 假阳性 |
| 同骨架不同手性 | 控制立体选择性 |
| common metabolite/cofactor artifact | 防止 ATP/NADH/H2O 等产生虚假 bridge |
| 化学上可反应但无酶 precedent | 防止把有机反应误迁移成酶反应 |
| 同产物不同 enzyme family | 防止 family leakage |

### 4. Search 需要 gate，不需要全开 ensemble

单步候选覆盖增加是真实的，但 naive ensemble 会让多步搜索变慢且变差。当前应建设：

```text
cheap molecule features
native confidence
enzyme substrate-space retriever hit
bridge scorer
route depth / stagnation state
budget-aware proposer routing
```

只有 gate 通过时才调用更重的 sidecar proposal 或 enzyme proposal。

## 需要的数据清单

### A. 立即可用但需要标准化的数据

| 数据 | 位置 | 处理任务 |
|---|---|---|
| EnzymeMap v2 | `data_external/enzymemap/enzymemap_v2_brenda2023.csv.gz` | 解析 mapped/unmapped、EC、protein_refs、organism、quality；拆 substrate/product pool |
| ECREACT | `data_external/ecreact/ecreact-1.0.csv` | 解析 rxn_smiles、EC、source；补充 enzyme reaction pool |
| enzymatic_retro_data | `data_external/enzymatic_retro_data/*.json` | 解析 product/reactants/EC；构建 enzyme proposer/verifier 基础 |
| Rhea 140 | `data_external/rhea/140.tar.bz2` | 展开 txt/SDF/RDF，抽 ChEBI/Rhea/EC/compound links |
| ReactZyme | `data_external/reactzyme/13635807.zip` | 展开 reaction/enzyme/time split、UniProt/Rhea、sequence tensors |
| v4 cascade | `dataset_v4_release/*` | 抽取真实 cascade seed、step/catalyst/condition/evidence |
| 外部 chemical 109K | `results/shared/chem_enzy_adapter_mainline_20260521/external_toplevel_onmt_smiles_token_150k` | 构建 chemical product pool |
| AiZ/RetroKNN assets | `workspace/aizdata`、`proposal_ablation_20260527` | 保留为 proposal sidecar 和 gate 对照 |

### B. 需要补齐或申请的数据

| 优先级 | 数据 | 用途 | 备注 |
|---|---|---|---|
| P0 | cofactor/common metabolite blacklist | 去除虚假 bridge | 可从 Rhea/ChEBI/KEGG + 手写列表生成 |
| P0 | purchasable building block catalog | route close / semisynthesis 判断 | 现有 stock 可用，但需要标准化和版本锁定 |
| P0 | RetroBioCat reaction/substrate scope | 更贴近合成生物催化 | 若能拿到，优先级很高 |
| P0/P1 | full BRENDA reaction + organism/enzyme table | 条件、EC、酶-底物证据 | 本地只有小 lookup；需要完整授权导出或可用子集 |
| P1 | KEGG/MetaCyc/PathBank/MetaNetX | metabolic pathway / bridge pretraining | 用于真实生物通路和 common metabolite 过滤 |
| P1 | UniProt reviewed enzyme sequences by EC | sequence-aware verifier | 先要 Swiss-Prot reviewed，不要直接拉全 TrEMBL |
| P1 | PDB/AlphaFold selected structures | top bridge 3D validator | 只对高置信 enzyme family 做，不进全节点搜索 |
| P1/P2 | USPTO full / Pistachio / Reaxys authorized reactions | chemical product pool 和 high-quality chemical proposer | 若有授权，优先接入；无授权不阻塞 P0 |

### C. 应生成的数据产品

第一阶段必须落盘这些文件：

```text
data/bridge_pack_v0/chemical_product_pool.parquet
data/bridge_pack_v0/enzyme_substrate_product_pool.parquet
data/bridge_pack_v0/cofactor_common_metabolite_blacklist.parquet
data/bridge_pack_v0/exact_bridge.parquet
data/bridge_pack_v0/similarity_bridge_raw.parquet
data/bridge_pack_v0/similarity_bridge_filtered.parquet
data/bridge_pack_v0/bridge_confidence_tiers.parquet
data/bridge_pack_v0/hard_negative_pool.parquet
data/bridge_pack_v0/verifier_train.jsonl
data/bridge_pack_v0/verifier_valid.jsonl
data/bridge_pack_v0/verifier_test.jsonl
data/bridge_pack_v0/manifest.json
data/bridge_pack_v0/report.md
```

目标规模：

| 数据产品 | P0 目标 | P1 目标 |
|---|---:|---:|
| chemical reactions/products | 300K-500K reactions 或当前 109K 起步 | 0.8M-1.5M |
| enzyme reactions | 20K-50K unique reactions | 50K-100K |
| enzyme substrate/product molecules | 10K-30K unique molecules | 50K+ |
| exact bridge links | 1K-10K | 10K-50K |
| high-confidence similarity bridge | 10K-30K | 50K-200K |
| verifier positives | 50K-200K | 200K-1M |
| hard negatives | 0.5M-2M raw pool | 5M-20M raw pool |
| route/search states | 50K-200K | 0.5M-5M |

## 需要的计算资源

### 当前机器状态

| 项 | 当前值 |
|---|---:|
| GPU | 2 x NVIDIA RTX 4080 SUPER |
| 单卡显存 | 32GB |
| GPU 当前占用 | 0MB，空闲 |
| CPU | 128 logical cores |
| RAM | 503GB，总可用约 439GB |
| `/root/autodl-tmp` | 150GB，总剩余约 84GB |
| `/` | 30GB，总剩余约 4.3GB |
| RDKit | 已安装，2023.09.6 |
| PyArrow | 已安装，24.0.0 |
| Torch | 已安装，2.3.0+cu121 |
| Transformers | 已安装，4.49.0 |
| FAISS | 未安装 |

判断：当前机器足够做 P0 数据审计、RDKit 标准化、小模型 verifier、少量 ESM embedding；不适合承载全量 P1 数据产品、长期多版本缓存和大规模向量索引。

### P0 最小配置

| 资源 | 建议 |
|---|---|
| GPU | 1-2 x 24GB/32GB GPU |
| CPU | 32-64 cores |
| RAM | 128-256GB |
| NVMe | 1-4TB |
| 软件 | RDKit、PyArrow、DuckDB、FAISS、scikit-learn、PyTorch |
| 用途 | 数据标准化、exact/similarity bridge、hard negative v0、MLP/GBDT verifier |

当前机器满足 GPU/CPU/RAM，但不满足长期存储。建议至少扩到 1TB 可用 NVMe。

### P1 推荐配置

| 资源 | 建议 |
|---|---|
| GPU | 4 x A100 80GB / L40S / RTX 4090/4080 32GB |
| CPU | 64-128 cores |
| RAM | 256-512GB |
| NVMe | 10TB |
| 数据层 | Parquet + DuckDB/PostgreSQL + FAISS |
| 用途 | ESM embedding precompute、dual encoder retriever、cross-attention verifier、large hard-negative sampling、gated search logs |

### P2 论文级扩展配置

| 资源 | 建议 |
|---|---|
| GPU | 8-16 x A100/H100 |
| RAM | 1TB |
| 存储 | 20-50TB |
| 调度 | Slurm/Ray/Kubernetes |
| 用途 | sequence-aware verifier、3D reranking、large route search farm、多模型 ablation |

P2 暂时不是当前瓶颈。当前最缺的是 P0/P1 的数据产品化和检索/验证闭环。

## 推荐执行顺序

### 第 1 周：数据标准化

交付：

```text
chemical_product_pool.parquet
enzyme_substrate_product_pool.parquet
cofactor_common_metabolite_blacklist.parquet
source_coverage_report.md
```

重点：

1. 统一 canonical SMILES/InChIKey。
2. 统一 EC、source、reaction_id。
3. 从 EnzymeMap/ECREACT/enzymatic_retro_data/Rhea/ReactZyme 抽 substrate/product。
4. 先不用大模型，先保证数据可查、可复现、可去重。

### 第 2 周：Bridge weak labels

交付：

```text
exact_bridge.parquet
similarity_bridge_raw.parquet
similarity_bridge_filtered.parquet
bridge_confidence_tiers.parquet
bridge_report.md
```

重点：

1. Exact InChIKey bridge。
2. Fingerprint similarity bridge。
3. 去除 common metabolite/cofactor artifact。
4. reaction center / stereochemistry 基础过滤。
5. 输出 Tier 1-5，不把低置信 similarity 当正样本。

### 第 3-4 周：Hard negatives 与 verifier v0

交付：

```text
hard_negative_pool.parquet
verifier_train.jsonl
verifier_valid.jsonl
verifier_test.jsonl
enzyme_feasibility_verifier_v0.joblib/pt
verifier_eval_report.md
```

重点：

1. 每个 positive 配 EC hard negative、reaction-center hard negative、stereo hard negative、cofactor artifact negative。
2. 先训练 fingerprint + EC embedding + MLP/GBDT。
3. 指标优先看 hard negative rejection、bridge precision@k，而不是 route solved rate。

### 第 5-8 周：Gated search

交付：

```text
bridge_retriever.faiss
bridge_scorer_v0.pt/joblib
gated_router_config.yaml
search_ablation_report.md
```

对比：

```text
native baseline
native + ungated AiZ/RetroKNN
native + gated chemical sidecar
native + gated enzyme bridge retriever
native + gated bridge + verifier
```

必须记录：

```text
single-step proposal coverage
false enzyme proposal rate
search time per target
candidate branching factor
route plausibility pass rate
bridge-supported route count
```

## 资源调配建议

如果全局优先本项目，我建议按下面顺序调配：

1. 立即给本项目扩 1-4TB NVMe，优先解决数据产品和索引落盘。
2. 安装 FAISS 或分配已有 FAISS 环境，用于 similarity bridge 和 retriever。
3. 补齐 full BRENDA / RetroBioCat / KEGG-MetaCyc-PathBank-MetaNetX / UniProt reviewed enzyme sequence 数据权限或下载任务。
4. 保留当前 2 x 4080 SUPER 做 P0/P1 原型；如进入 verifier v1 和 ESM embedding，调配 4 卡节点。
5. 暂缓 8-16 卡大模型训练，等 bridge tier 和 hard negative 指标证明有效后再扩。

## 下一步可执行任务

最小闭环任务：

```text
build_bridge_pack_v0.py
  input:
    EnzymeMap
    ECREACT
    enzymatic_retro_data
    Rhea
    ReactZyme
    external chemical product pool
  output:
    product/substrate pools
    exact bridge
    similarity bridge
    hard negatives
    verifier splits
    audit report
```

这一步完成后，我们才真正有资格说：

```text
我们不是在泛泛做 chemo-enzymatic retrosynthesis，
而是在没有专家标签的情况下，系统构造 enzyme bridge weak labels，
并训练模型学会何时接受或拒绝酶步。
```
