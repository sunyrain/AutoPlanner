"""Convert cascade_dataset_v2.json (schema 2.0.0, flat list) to the loader_v2-compatible
shape ({"records_kept": [...]} with cascade.purpose_assessment.recommended_for_supervised_training=True).

Also emits a JSON quality report.

Usage:
    python -m cascade_planner.data.normalize_v2 \
        --in cascade_dataset_v2.json \
        --out cascade_dataset_v2.normalized.json \
        --report cascade_dataset_v2.quality.json
"""
from __future__ import annotations

import argparse, json, collections, statistics
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def _rxn_parses(rxn: str) -> bool:
    if not rxn or ">>" not in rxn:
        return False
    lhs, rhs = rxn.split(">>", 1)
    if not lhs.strip() or not rhs.strip():
        return False
    for side in (lhs, rhs):
        for s in side.split("."):
            if Chem.MolFromSmiles(s) is None:
                return False
    return True


def normalize(records: list[dict]) -> dict:
    out_records = []
    for rec in records:
        new_rec = {
            "doi": rec.get("doi") or rec.get("title", ""),
            "title": rec.get("title", ""),
            "cascades": [],
            "schema_version": rec.get("schema_version"),
            "record_uuid": rec.get("record_uuid"),
        }
        for c in rec.get("cascades", []):
            cnew = dict(c)
            cnew.setdefault("purpose_assessment", {})
            cnew["purpose_assessment"].setdefault(
                "recommended_for_supervised_training", True
            )
            cnew.setdefault("compatibility_annotation", {
                "compatibility_label": None, "issue_types": [], "mitigation_strategies": [],
                "evidence_strength": None,
            })
            new_rec["cascades"].append(cnew)
        out_records.append(new_rec)
    return {
        "metadata": {
            "source_schema_version": "2.0.0",
            "normalizer": "cascade_planner.data.normalize_v2",
            "n_records": len(out_records),
        },
        "records_kept": out_records,
        "records_excluded": [],
    }


def quality_report(records: list[dict]) -> dict:
    n_rec = len(records); n_cas = 0; n_steps = 0
    n_rxn_ok = 0; n_rxn_parse = 0; n_rxn_mapped = 0
    cascade_lens = []
    domains = collections.Counter(); modes = collections.Counter()
    transformations = collections.Counter()
    n_with_ec = 0; ec_lvls = collections.Counter(); ec_classes = collections.Counter()
    multi_ec_steps = 0; weird_ec = 0
    n_with_uniprot = 0
    n_with_temp = 0; n_with_ph = 0; n_with_solvent = 0
    n_with_yield = 0; n_with_ee = 0; n_with_conv = 0
    n_with_cofactor_req = 0; n_with_pdb = 0
    enzymatic_steps = 0
    chemical_steps = 0

    for rec in records:
        for c in rec.get("cascades", []):
            n_cas += 1
            domains[c.get("route_domain", "?")] += 1
            modes[c.get("operation_mode", "?")] += 1
            steps = c.get("steps", []) or []
            cascade_lens.append(len(steps))
            for st in steps:
                n_steps += 1
                if st.get("transformation_superclass"):
                    transformations[st["transformation_superclass"]] += 1
                rxn = (st.get("rxn_smiles") or "").strip()
                if st.get("rxn_smiles_status") == "ok":
                    n_rxn_ok += 1
                if _rxn_parses(rxn):
                    n_rxn_parse += 1
                if rxn and any(f":{i}" in rxn for i in range(1, 200)):
                    n_rxn_mapped += 1
                cats = st.get("catalyst_components") or []
                ec_set = set(); has_up = False; has_pdb = False; cof_req = False; is_enz = False
                for cc in cats:
                    if not isinstance(cc, dict): continue
                    if cc.get("catalyst_class") == "enzyme": is_enz = True
                    ec = cc.get("ec_number") or ""
                    if isinstance(ec, str) and ec:
                        if ";" in ec or "," in ec:
                            weird_ec += 1
                        else:
                            ec_set.add(ec)
                            ec_lvls[ec.count(".") + 1] += 1
                            ec_classes[ec.split(".")[0]] += 1
                    if cc.get("uniprot_id"): has_up = True
                    ext = cc.get("enzyme_external_ids") or {}
                    if isinstance(ext, dict) and ext.get("pdb"): has_pdb = True
                    if cc.get("cofactor_required") is not None: cof_req = True
                if is_enz:
                    enzymatic_steps += 1
                else:
                    chemical_steps += 1
                if ec_set:
                    n_with_ec += 1
                    if len(ec_set) > 1:
                        multi_ec_steps += 1
                if has_up: n_with_uniprot += 1
                if has_pdb: n_with_pdb += 1
                if cof_req: n_with_cofactor_req += 1
                cond = st.get("step_conditions") or {}
                if isinstance(cond, dict):
                    if cond.get("temperature_c") is not None: n_with_temp += 1
                    if cond.get("ph") is not None: n_with_ph += 1
                    if cond.get("solvent"): n_with_solvent += 1
                outc = st.get("step_outcome") or {}
                if isinstance(outc, dict):
                    if outc.get("step_yield_percent") is not None: n_with_yield += 1
                    if outc.get("step_ee_percent") is not None: n_with_ee += 1
                    if outc.get("step_conversion_percent") is not None: n_with_conv += 1

    def pct(n): return round(100 * n / max(1, n_steps), 2)

    return {
        "n_records": n_rec, "n_cascades": n_cas, "n_steps": n_steps,
        "cascade_len_mean": round(sum(cascade_lens) / max(1, len(cascade_lens)), 3),
        "cascade_len_median": int(statistics.median(cascade_lens)) if cascade_lens else 0,
        "cascade_len_max": max(cascade_lens) if cascade_lens else 0,
        "route_domains": dict(domains.most_common()),
        "operation_modes": dict(modes.most_common()),
        "transformations_top10": dict(transformations.most_common(10)),
        "rxn_status_ok": n_rxn_ok, "rxn_status_ok_pct": pct(n_rxn_ok),
        "rxn_rdkit_parsable": n_rxn_parse, "rxn_rdkit_parsable_pct": pct(n_rxn_parse),
        "rxn_atom_mapped": n_rxn_mapped, "rxn_atom_mapped_pct": pct(n_rxn_mapped),
        "with_ec": n_with_ec, "with_ec_pct": pct(n_with_ec),
        "ec_levels": dict(ec_lvls), "ec_classes": dict(ec_classes.most_common()),
        "ec_multi_per_step": multi_ec_steps,
        "weird_ec_strings": weird_ec,
        "with_uniprot": n_with_uniprot, "with_uniprot_pct": pct(n_with_uniprot),
        "with_pdb": n_with_pdb, "with_pdb_pct": pct(n_with_pdb),
        "with_cofactor_required_field": n_with_cofactor_req,
        "with_temperature": n_with_temp, "with_temperature_pct": pct(n_with_temp),
        "with_ph": n_with_ph, "with_ph_pct": pct(n_with_ph),
        "with_solvent": n_with_solvent, "with_solvent_pct": pct(n_with_solvent),
        "with_yield": n_with_yield, "with_yield_pct": pct(n_with_yield),
        "with_ee": n_with_ee, "with_ee_pct": pct(n_with_ee),
        "with_conversion": n_with_conv, "with_conversion_pct": pct(n_with_conv),
        "enzymatic_steps": enzymatic_steps, "chemical_steps": chemical_steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--report", dest="report", default=None)
    args = ap.parse_args()

    src = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    if isinstance(src, dict) and "records_kept" in src:
        records = src["records_kept"]
    elif isinstance(src, list):
        records = src
    else:
        raise SystemExit("Unrecognized input shape")

    norm = normalize(records)
    Path(args.out).write_text(json.dumps(norm, ensure_ascii=False), encoding="utf-8")
    print(f"WROTE {args.out}  records={len(norm['records_kept'])}")

    rep = quality_report(records)
    if args.report:
        Path(args.report).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"WROTE {args.report}")
    else:
        print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
