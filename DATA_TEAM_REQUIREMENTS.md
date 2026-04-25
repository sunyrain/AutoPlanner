# 数据团队需求文档 — AutoPlanner SOTA 升级配套

> 日期：2026-04-24
> 背景：模型侧已完成第一轮 SOTA 升级（K1/K5/K7 达标），以下数据改进是进一步突破 K2-K4/K6 的关键依赖。

---

## P0：阻塞性需求（直接影响 KPI 达标）

### P0-1：BRENDA 账号注册 + flat file 下载

**影响 KPI：K6 (pH MAE 0.589 → 目标 0.50)**

当前 pH 预测的瓶颈是缺少酶特异性 pH_opt 数据。BRENDA 数据库有 ~30-50K 条 (EC×organism → pH_opt) 记录，是唯一能突破 0.50 的数据源。

**操作：**
1. 注册 https://www.brenda-enzymes.org/register.php （免费，CC BY 4.0）
2. 设置环境变量后运行：
   ```bash
   export BRENDA_EMAIL="your@email.com"
   export BRENDA_PASSWORD="your_password"
   python scripts/download_brenda.py
   ```
3. 脚本会自动通过 SOAP API 下载 202 个 EC 的 T_opt/pH_opt 数据

**预期收益：** pH MAE 0.589 → 0.45-0.50（基于 BRENDA 的酶特异性 pH_opt 比 EC 中位数精确得多）

### P0-2：UniProt ID 标注覆盖扩展

**影响 KPI：K7 (enzyme recommender)**

当前数据中 2,258 个酶催化 trainable 步骤关联了 683 个唯一 UniProt ID（72.1% 覆盖率）。但 8,748 个总步骤中有 3,243 个有 UniProt 标注（37%）。

**需求：**
- 对 `rxn_smiles_status != "ok"` 的步骤，如果有 EC+organism 信息，补充 UniProt ID（通过 UniProt REST API 自动化）
- 目标：UniProt 覆盖率从 37% → 60%+
- 已提供工具：`cascade_planner/data/enrich_uniprot.py`

**预期收益：** 更大的 enzyme bank → dual-tower 模型泛化更好 → K7 从 38.8% → 45%+

---

## P1：高优先级（显著提升模型效果）

### P1-1：rxn_smiles 质量提升

**影响 KPI：K1 (EnzExpand), K3/K4 (multi-step)**

当前 8,748 步中只有 5,030 步（57.5%）的 rxn_smiles_status 为 "ok"。43% 的步骤因 SMILES 缺失/错误而不可训练。

**需求：**
- 优先修复 `missing_rhs`（387 步有 UniProt 但缺产物 SMILES）
- 优先修复 `missing_lhs`（247 步有 UniProt 但缺反应物 SMILES）
- 对于 `missing_both`（2,410 步），至少补充主产物 SMILES

**预期收益：** trainable 步骤从 3,028 → 4,500+，模板覆盖率和 MLP 泛化显著提升

### P1-2：EC 4-level 补全

**影响 KPI：K1, K5, K6**

v2-strict 过滤中 1,931 步（22%）因 EC 非 4-level（如 `1.1.1.-`）被丢弃。

**需求：**
- 对 EC 3-level 的步骤，通过文献回查或 UniProt 注释补全第 4 位
- 对确实无法确定的，标记为 `ec_level=3` 而非丢弃

**预期收益：** 可训练酶催化步骤增加 ~30%

### P1-3：organism 字段标准化

**影响 KPI：K5, K6, K7**

当前 organism 字段格式不统一（如 "E. coli" vs "Escherichia coli" vs "E.coli K12"），影响 BRENDA 查表和 UniProt 匹配。

**需求：**
- 统一为 NCBI Taxonomy 标准名称
- 提供 taxonomy_id 字段

---

## P2：中优先级（长期竞争力）

### P2-1：atom-mapping 质量审计

当前使用 rxnmapper 自动映射，但 37% 的模板应用失败（fires_fail），主要原因是立体化学漂移。

**需求：**
- 对高频模板（top-50）的反应进行人工 atom-mapping 审核
- 标记 stereo-sensitive 步骤

### P2-2：级联兼容性标注扩展

AutoPlanner 的独特优势是级联级别的兼容性标注（`compatibility_label`, `issue_types`, `mitigation_strategies`）。

**需求：**
- 扩展标注覆盖率（当前 ~40% 的级联有兼容性标注）
- 增加定量兼容性评分（1-5 分制）

### P2-3：yield/ee/conversion 数据补全

当前 yield 覆盖率仅 17.5%，ee 19.4%，conversion 20.5%。

**需求：**
- 优先补全 yield 数据（对路线评分至关重要）
- 目标：yield 覆盖率 → 50%+

---

## 数据格式要求

新增/修改的数据请保持与 `cascade_dataset_v2.json` 相同的 schema（2.0.0）。关键字段：

```json
{
  "steps": [{
    "rxn_smiles": "reactants>>products",        // P1-1: 确保完整
    "rxn_smiles_status": "ok",                   // P1-1: 修复后更新
    "catalyst_components": [{
      "ec_number": "1.1.1.1",                    // P1-2: 补全 4-level
      "uniprot_id": "P07246",                    // P0-2: 扩展覆盖
      "organism": "Saccharomyces cerevisiae"     // P1-3: 标准化
    }],
    "temperature_c": 30.0,
    "ph": 7.5,
    "yield_percent": 85.0                        // P2-3: 补全
  }]
}
```

---

## 优先级排序

| 优先级 | 需求 | 预计工作量 | 影响 KPI | 预期提升 |
|--------|------|-----------|----------|---------|
| P0-1 | BRENDA 注册下载 | 10 分钟 | K6 | pH MAE -0.09 |
| P0-2 | UniProt 覆盖扩展 | 1-2 天 | K7 | +6pp |
| P1-1 | rxn_smiles 修复 | 1-2 周 | K1/K3/K4 | trainable +50% |
| P1-2 | EC 4-level 补全 | 3-5 天 | K1/K5/K6 | trainable +30% |
| P1-3 | organism 标准化 | 2-3 天 | K5/K6/K7 | 查表命中率 +20% |
| P2-1 | atom-mapping 审计 | 1 周 | K1 | fires_fail -15pp |
| P2-2 | 兼容性标注扩展 | 2-3 周 | K3/K4 | 路线评分质量 |
| P2-3 | yield 数据补全 | 1-2 周 | 路线评分 | yield 覆盖 +30pp |
