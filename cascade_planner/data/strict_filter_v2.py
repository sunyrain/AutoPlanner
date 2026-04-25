"""Build a STRICT v2 subset by filtering out problematic steps.

Rules (applied per step inside cascade_dataset_v2.normalized.json):
  R1. drop steps with rxn_smiles_status != "ok"  (already enforced by _rxn_valid in loader, but we double-check via the field)
  R2. drop multi-EC steps  (ec_number contains ';' or ',')
  R3. drop EC strings that are not 4-level (e.g. "1.x.x.x", partial)
  R4. drop identity reactions (canonical(LHS) == canonical(RHS), e.g. racemizations)
  R5. drop steps where step_role indicates non-supervised intent
       (step_role in {"deracemization", "resolution", "purification"})

We DO NOT drop steps lacking atom-mapping (the cache covers everything we
need), and DO NOT drop steps lacking enzyme — chemo steps are still useful
for chem engines.

Output:
  cascade_dataset_v2.strict.json — same shape as v2.normalized.json,
  but cascades with all steps dropped are removed entirely.
  cascade_dataset_v2.strict.report.json — counts of each filter outcome.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from collections import Counter
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

EC_PAT = re.compile(r"^\d+\.\d+\.\d+\.\d+$")


def canon(smi: str) -> str | None:
    if not smi:
        return None
    parts = sorted(p for p in smi.split(".") if p)
    out = []
    for p in parts:
        m = Chem.MolFromSmiles(p)
        if m is None:
            return None
        out.append(Chem.MolToSmiles(m, canonical=True))
    return ".".join(sorted(out))


def is_identity(rxn: str) -> bool:
    if not rxn or ">>" not in rxn:
        return False
    lhs, rhs = rxn.split(">>", 1)
    L = canon(lhs); R = canon(rhs)
    return L is not None and L == R


def filter_step(s: dict) -> tuple[bool, str]:
    if (s.get("rxn_smiles_status") or "ok") != "ok":
        return False, "rxn_status_not_ok"
    rxn = (s.get("rxn_smiles") or "").strip()
    if not rxn or ">>" not in rxn:
        return False, "no_rxn"
    cats = s.get("catalyst_components") or []
    ecs = []
    for c in cats:
        e = c.get("ec_number")
        if e:
            for tok in re.split(r"[;,]", e):
                tok = tok.strip()
                if tok:
                    ecs.append(tok)
    if len(ecs) > 1:
        return False, "multi_ec"
    if ecs and not EC_PAT.match(ecs[0]):
        return False, "ec_not_4level"
    role = (s.get("step_role") or "").lower().strip()
    if role in {"deracemization", "resolution", "purification"}:
        return False, f"role_{role}"
    if is_identity(rxn):
        return False, "identity_rxn"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="cascade_dataset_v2.normalized.json")
    ap.add_argument("--out", default="cascade_dataset_v2.strict.json")
    ap.add_argument("--report", default="cascade_dataset_v2.strict.report.json")
    args = ap.parse_args()

    data = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    counts = Counter()
    n_records = 0; n_cascades_in = 0; n_cascades_out = 0
    n_steps_in = 0; n_steps_out = 0

    out_records = []
    for art in data.get("records_kept", []):
        n_records += 1
        kept_cascades = []
        for c in art.get("cascades", []):
            n_cascades_in += 1
            kept_steps = []
            for s in c.get("steps", []) or []:
                n_steps_in += 1
                ok, reason = filter_step(s)
                counts[reason] += 1
                if ok:
                    kept_steps.append(s); n_steps_out += 1
            if kept_steps:
                cc = dict(c)
                cc["steps"] = kept_steps
                cc["total_steps"] = len(kept_steps)
                kept_cascades.append(cc)
                n_cascades_out += 1
        if kept_cascades:
            ar = dict(art)
            ar["cascades"] = kept_cascades
            out_records.append(ar)

    out_obj = {"records_kept": out_records}
    Path(args.out).write_text(json.dumps(out_obj, ensure_ascii=False), encoding="utf-8")

    rep = {
        "input": args.inp,
        "output": args.out,
        "n_records_in": len(data.get("records_kept", [])),
        "n_records_out": len(out_records),
        "n_cascades_in": n_cascades_in,
        "n_cascades_out": n_cascades_out,
        "n_steps_in": n_steps_in,
        "n_steps_out": n_steps_out,
        "n_steps_dropped": n_steps_in - n_steps_out,
        "drop_reasons": dict(counts),
        "step_retention_pct": round(100 * n_steps_out / max(1, n_steps_in), 2),
    }
    Path(args.report).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
