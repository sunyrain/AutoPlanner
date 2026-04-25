"""Filter EnzExpand template pool to single-fragment LHS retro templates only,
then retrain MLP and re-export ONNX + TSV.
"""
from __future__ import annotations
import collections, hashlib, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

from cascade_planner.data.loader_v2 import load_v2
from cascade_planner.demo.final_integrated_eval import build_template_pool
from cascade_planner.expand.enz_template import train as train_mlp
from cascade_planner.expand.enzymemap_loader import (
    extract_templates_from_enzymemap, load_filtered)
from cascade_planner.expand.export_enzexpand_onnx import SoftmaxWrapper, template_hash

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent.parent
AIZDATA = ROOT / "aizdata"

def lhs_frag_count(t: str) -> int:
    return t.split(">>")[0].count(".") + 1

def main():
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cascade_dataset_v2.normalized.json")
    args = ap.parse_args()
    print(f"[load] {args.data}")
    steps, _, _ = load_v2(args.data)
    enz_train = [s for s in steps if s.ec_number]
    print(f"  enz steps: {len(enz_train)}")

    print("[em] loading EnzymeMap")
    em_df = load_filtered(min_quality=0.95, only_single=True, ec1_balance=4000)
    em_rows = extract_templates_from_enzymemap(em_df)
    print(f"  rows: {len(em_rows)}")

    X, y, tpls, _ = build_template_pool(enz_train, em_rows)
    n0 = len(tpls)
    print(f"[pool] before filter: n_templates={n0}, n_samples={X.shape[0]}")

    # build map old_idx -> new_idx for templates with single-fragment LHS
    keep = [i for i, t in enumerate(tpls) if lhs_frag_count(t) == 1]
    keep_set = set(keep)
    remap = {old: new for new, old in enumerate(keep)}
    print(f"[filter] kept {len(keep)} / {n0} single-LHS templates")

    mask = np.array([int(yi) in keep_set for yi in y])
    X2 = X[mask]
    y2 = np.array([remap[int(yi)] for yi in y[mask]])
    tpls2 = [tpls[i] for i in keep]
    n_tpl = len(tpls2)
    print(f"  after filter: n_samples={X2.shape[0]}, n_templates={n_tpl}")

    print(f"[train] MLP epochs=25 hidden=1024")
    model = train_mlp(X2, y2, n_tpl=n_tpl, epochs=25, hidden=1024,
                      dropout=0.4, batch=512, seed=0, verbose=True).eval()

    # Export ONNX
    onnx_path = AIZDATA / "enzexpand_model.onnx"
    print(f"[onnx] {onnx_path}")
    wrapped = SoftmaxWrapper(model).cpu().eval()
    dummy = torch.zeros(1, X2.shape[1], dtype=torch.float32)
    torch.onnx.export(wrapped, dummy, str(onnx_path),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17)
    print(f"  {onnx_path.stat().st_size/1024:.0f} KB")

    # Write filtered TSV
    occ = collections.Counter(int(v) for v in y2.tolist())
    rows = [dict(template_code=c, retro_template=t,
                 template_hash=template_hash(t),
                 classification="enz.unrecognized",
                 library_occurence=int(occ.get(c, 0)))
            for c, t in enumerate(tpls2)]
    df = pd.DataFrame(rows)
    tpl_path = AIZDATA / "enzexpand_templates.csv.gz"
    df.to_csv(tpl_path, sep="\t", index=False, compression="gzip")
    print(f"[tpl] wrote {len(df)} templates to {tpl_path}")

    print("[done] EnzExpand re-exported with single-LHS retro templates only.")

if __name__ == "__main__":
    main()
