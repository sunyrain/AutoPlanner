"""End-to-end per-step demo for the presentation.

For each of the 6 hand-picked demo cases (in results/shared/demo_picks.json), runs:
  Step 1  Condition predictor  (EC1-mean for T/pH, mode for solvent)   vs GT
  Step 2  Enzyme recommender   (Tanimoto + frequency, by_product)     vs GT
  Step 3  EnzExpand template   (ONNX top-K + rdchiral)                vs GT precursors

All training-side computation excludes the demo's own DOI to avoid leakage.

Output: results/shared/demo_pipeline_report.json + a printed table.
"""
from __future__ import annotations

import collections
import gzip
import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from cascade_planner.conditions.enzyme_recommend import collect_records, by_product
from cascade_planner.paths import aizdata_dir, shared_dir

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = shared_dir()
DATA_PATH = ROOT / "cascade_dataset_v2.normalized.json"
AIZDATA = aizdata_dir()
ONNX_PATH = AIZDATA / "enzexpand_model.onnx"
TPL_PATH = AIZDATA / "enzexpand_templates.csv.gz"

TOPK_TPL = 25  # how many templates we try with rdchiral
TOPK_REC = 3
SOLV_TOP = 12


# -------------------------------------------------- utils

def canon(smi: str) -> str:
    m = Chem.MolFromSmiles(smi) if smi else None
    return Chem.MolToSmiles(m) if m else ""


def canon_set(dot: str) -> frozenset[str]:
    out = set()
    for s in dot.split("."):
        s = s.strip()
        if not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            return frozenset()
        out.add(Chem.MolToSmiles(m))
    return frozenset(out)


def morgan2(smi: str, n_bits: int = 2048) -> np.ndarray:
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return np.zeros(n_bits, dtype=np.float32)
    bv = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def main_product_smi(rxn: str) -> str:
    rhs = rxn.split(">>", 1)[1]
    frags = [(m.GetNumHeavyAtoms(), m) for m in (Chem.MolFromSmiles(s) for s in rhs.split(".")) if m]
    return Chem.MolToSmiles(max(frags, key=lambda x: x[0])[1])


def lhs_set(rxn: str) -> frozenset[str]:
    return canon_set(rxn.split(">>", 1)[0])


# -------------------------------------------------- load template table

def load_templates(path: Path) -> list[str]:
    """Returns list of retro_template strings, ordered by template_code (= ONNX index)."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        idx_code = header.index("template_code")
        idx_tpl = header.index("retro_template")
        rows = []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            rows.append((int(parts[idx_code]), parts[idx_tpl]))
    rows.sort(key=lambda x: x[0])
    return [t for _, t in rows]


# -------------------------------------------------- per-step models

def fit_condition_model(rows_train):
    """Build dict-based predictors:
      - per-EC1 mean for T, pH (fall back to global mean)
      - per-EC1 mode for solvent (top-12 + 'other')
      - global mode for transformation_super
    """
    by_ec = collections.defaultdict(list)
    sv_by_ec = collections.defaultdict(collections.Counter)
    sv_global = collections.Counter()
    tr_global = collections.Counter()
    cat_global = collections.Counter()
    def _f(v):
        try:
            return float(v)
        except Exception:
            if isinstance(v, str):
                s = v.strip().lower()
                if "room" in s or "ambient" in s or s == "rt":
                    return 25.0
                if "ice" in s:
                    return 4.0
            return None
    for r in rows_train:
        ec1 = (r.get("ec") or "").split(".")[0] if r.get("ec") else ""
        T = _f(r.get("T")); pH = _f(r.get("pH"))
        if T is not None and pH is not None:
            by_ec[ec1].append((T, pH))
        sv = (r.get("solvent") or "").strip().lower() if r.get("solvent") else ""
        if sv:
            sv_by_ec[ec1][sv] += 1
            sv_global[sv] += 1
        ts = r.get("transform_super")
        if ts:
            tr_global[ts] += 1
        # catalyst class always 'enzyme' here, skip
    keep_sv = {s for s, _ in sv_global.most_common(SOLV_TOP)}
    return {
        "by_ec": by_ec,
        "sv_by_ec": sv_by_ec,
        "sv_global": sv_global,
        "tr_global": tr_global,
        "keep_sv": keep_sv,
    }


def predict_conditions(model, ec1: str):
    """Return dict {T, pH, solvent, transform_super}."""
    pool = model["by_ec"].get(ec1) or [v for vs in model["by_ec"].values() for v in vs]
    if pool:
        T_pred = float(np.mean([t for t, _ in pool]))
        pH_pred = float(np.mean([p for _, p in pool]))
    else:
        T_pred, pH_pred = 30.0, 7.0
    sv_counter = model["sv_by_ec"].get(ec1) or model["sv_global"]
    # restrict to keep_sv top-12
    filtered = collections.Counter({k: v for k, v in sv_counter.items() if k in model["keep_sv"]})
    sv_pred = (filtered.most_common(1) or sv_counter.most_common(1) or [("water", 0)])[0][0]
    tr_pred = (model["tr_global"].most_common(1) or [("hydrolysis", 0)])[0][0]
    return {"T": T_pred, "pH": pH_pred, "solvent": sv_pred, "transform_super": tr_pred}


# -------------------------------------------------- enzexpand template inference

def template_topk(session, product_smi: str, k: int):
    fp = morgan2(product_smi).reshape(1, -1)
    out = session.run(None, {session.get_inputs()[0].name: fp})[0][0]  # (n_tpl,)
    idx = np.argsort(-out)[:k]
    return idx, out[idx]


def apply_template(tpl: str, product_smi: str, max_outcomes: int = 5):
    from rdchiral.main import rdchiralRunText
    try:
        return rdchiralRunText(tpl, product_smi)[:max_outcomes]
    except Exception:
        return []


# -------------------------------------------------- main

def main():
    print("[load] dataset + ONNX + templates ...")
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    rows_all = collect_records(data)
    print(f"  enzyme records: {len(rows_all)}")
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    templates = load_templates(TPL_PATH)
    print(f"  enzexpand templates: {len(templates)}")

    picks = json.loads((RESULTS / "demo_picks.json").read_text(encoding="utf-8"))
    print(f"  demo picks: {len(picks)}\n")

    report = []
    for i, c in enumerate(picks, 1):
        print("=" * 80)
        print(f"DEMO {i}  {c['tname']}  (EC {c['ec']})")
        print(f"  doi: {c['doi']}")
        print(f"  rxn: {c['rxn']}")
        print()

        # --- Hold out same DOI ---
        rows_train = [r for r in rows_all if r["doi"] != c["doi"]]
        prod_smi = main_product_smi(c["rxn"])
        gt_lhs = lhs_set(c["rxn"])
        ec1_gt = (c["ec"] or "").split(".")[0]

        # ===== STEP 1 — Conditions =====
        model = fit_condition_model(rows_train)
        pred = predict_conditions(model, ec1_gt)
        gt_sv = (c.get("solvent") or "").strip().lower()
        # coerce GT T/pH to float for arithmetic (dataset may store strings)
        def _gtf(v, default):
            try:
                return float(v)
            except Exception:
                if isinstance(v, str):
                    s = v.strip().lower()
                    if "room" in s or "ambient" in s or s == "rt":
                        return 25.0
                    if "ice" in s:
                        return 4.0
                return default
        T_gt = _gtf(c["T"], 25.0); pH_gt = _gtf(c["pH"], 7.0)
        cond_row = {
            "T_gt": T_gt, "T_pred": round(pred["T"], 1), "T_err": round(pred["T"] - T_gt, 1),
            "pH_gt": pH_gt, "pH_pred": round(pred["pH"], 2), "pH_err": round(pred["pH"] - pH_gt, 2),
            "solvent_gt": gt_sv, "solvent_pred": pred["solvent"],
            "solvent_match": gt_sv == pred["solvent"],
            "tsuper_gt": c["tsuper"], "tsuper_pred": pred["transform_super"],
            "tsuper_match": c["tsuper"] == pred["transform_super"],
        }
        print("  [Step 1 — Conditions]")
        print(f"    T:        gt={T_gt}°C   pred={cond_row['T_pred']}°C   |Δ|={abs(cond_row['T_err'])}")
        print(f"    pH:       gt={pH_gt}      pred={cond_row['pH_pred']}      |Δ|={abs(cond_row['pH_err'])}")
        print(f"    solvent:  gt={gt_sv:<20s}  pred={pred['solvent']:<20s}  match={cond_row['solvent_match']}")
        print(f"    tsuper:   gt={c['tsuper']:<20s}  pred={pred['transform_super']:<20s}  match={cond_row['tsuper_match']}")

        # ===== STEP 2 — Enzyme Recommender =====
        recs = by_product(rows_train, prod_smi, TOPK_REC)
        rec_rows = []
        gt_up = c.get("uniprot")
        gt_ec = c.get("ec")
        hit_rank = None
        for rk, it in enumerate(recs, 1):
            row = {
                "rank": rk, "ec": it["ec"], "uniprot": it["uniprot_id"],
                "protein": it.get("protein_name"),
                "tanimoto": it.get("best_tanimoto"),
                "n_records": it.get("n_supporting_records"),
            }
            rec_rows.append(row)
            same_ec = (it["ec"] or "").split(".")[:3] == (gt_ec or "").split(".")[:3]
            if hit_rank is None and (it["uniprot_id"] == gt_up or same_ec):
                hit_rank = rk
        print("  [Step 2 — Enzyme recommender]   (gt EC={}, gt Uniprot={})".format(gt_ec, gt_up))
        for r in rec_rows:
            print(f"    #{r['rank']}  EC {r['ec']:<10s}  Uniprot={r['uniprot']}  sim={r['tanimoto']}  n={r['n_records']}  | {r['protein']}")
        print(f"    -> GT-rank: {hit_rank if hit_rank else 'miss'}")

        # ===== STEP 3 — EnzExpand template =====
        tpl_idx, tpl_score = template_topk(sess, prod_smi, TOPK_TPL)
        outcomes_per_tpl = []
        gt_match_rank = None
        any_outcome_rank = None
        first_outcome = None
        for rk, (ti, sc) in enumerate(zip(tpl_idx, tpl_score), 1):
            outs = apply_template(templates[int(ti)], prod_smi, max_outcomes=5)
            if outs and any_outcome_rank is None:
                any_outcome_rank = rk
                first_outcome = outs[0]
            sets = [canon_set(o) for o in outs]
            if gt_match_rank is None and any(s == gt_lhs for s in sets):
                gt_match_rank = rk
            outcomes_per_tpl.append({
                "rank": rk, "tpl_idx": int(ti), "score": float(sc),
                "n_outcomes": len(outs),
                "outcomes": outs[:2],
            })
        print(f"  [Step 3 — EnzExpand template]   (top-{TOPK_TPL})")
        # show top 5
        for o in outcomes_per_tpl[:5]:
            outs_short = " | ".join(o["outcomes"]) if o["outcomes"] else "(no fire)"
            print(f"    #{o['rank']}  tpl={o['tpl_idx']:<4d}  score={o['score']:.3f}  outs={o['n_outcomes']}  {outs_short[:100]}")
        gt_lhs_str = ".".join(sorted(gt_lhs))
        print(f"    GT precursor:    {gt_lhs_str}")
        print(f"    First fire   :   #{any_outcome_rank}  -> {first_outcome}")
        print(f"    GT exact match: {'rank #' + str(gt_match_rank) if gt_match_rank else 'MISS'}")
        print()

        report.append({
            "demo": i,
            "case": {
                "doi": c["doi"], "step_id": c["step_id"], "rxn": c["rxn"],
                "ec_gt": c["ec"], "uniprot_gt": gt_up, "enz_name_gt": c["enz_name"],
                "T_gt": c["T"], "pH_gt": c["pH"], "solvent_gt": gt_sv,
                "tsuper_gt": c["tsuper"], "tname_gt": c["tname"],
                "product": prod_smi,
                "gt_precursor": gt_lhs_str,
            },
            "step1_conditions": cond_row,
            "step2_recommender": {
                "topk": rec_rows,
                "gt_rank": hit_rank,
            },
            "step3_template": {
                "first_fire_rank": any_outcome_rank,
                "first_fire_outcome": first_outcome,
                "gt_match_rank": gt_match_rank,
                "top5": outcomes_per_tpl[:5],
            },
        })

    out_path = RESULTS / "demo_pipeline_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] {out_path}")

    # ----- summary table -----
    print("\n" + "=" * 80)
    print("SUMMARY  (delta vs GT, lower is better; OK = exact match)")
    print("=" * 80)
    hdr = f"{'#':>2}  {'EC':<8}  {'|dT|':>5}  {'|dpH|':>5}  {'solv':<5}  {'tsup':<5}  {'recK':<5}  {'tplK':<6}"
    print(hdr)
    for r in report:
        c1 = r["step1_conditions"]; c2 = r["step2_recommender"]; c3 = r["step3_template"]
        print(f"{r['demo']:>2}  {r['case']['ec_gt']:<8}  {abs(c1['T_err']):>5.1f}  {abs(c1['pH_err']):>5.2f}  "
              f"{'OK' if c1['solvent_match'] else '--':<5}  {'OK' if c1['tsuper_match'] else '--':<5}  "
              f"{('#'+str(c2['gt_rank'])) if c2['gt_rank'] else 'miss':<5}  "
              f"{('#'+str(c3['gt_match_rank'])) if c3['gt_match_rank'] else 'miss':<6}")


if __name__ == "__main__":
    main()
