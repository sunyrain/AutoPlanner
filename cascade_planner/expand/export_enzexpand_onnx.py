"""Export EnzExpand template MLP to ONNX so AiZynthFinder can load it natively
as a second expansion policy alongside USPTO.

Pipeline
--------
1. Build template pool (local enz + EnzymeMap) using the same logic as
   ``final_integrated_eval.build_template_pool``.
2. Train a TemplateMLP (Morgan2-2048 -> n_tpl) on the full data.
3. Wrap with softmax + export to ONNX (input shape (1, 2048), output (1, n_tpl)).
4. Write templates TSV.gz in AiZynth's exact schema:
       template_code\tretro_template\ttemplate_hash\tclassification\tlibrary_occurence
5. Emit a hybrid YAML config registering BOTH USPTO and EnzExpand policies.

Run:
    python -m cascade_planner.expand.export_enzexpand_onnx \\
        --data cascade_dataset_v2.normalized.json \\
        --em-cap 4000 --epochs 30 --hidden 1024
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cascade_planner.data.loader_v2 import load_v2
from cascade_planner.demo.final_integrated_eval import build_template_pool
from cascade_planner.expand.enz_template import TemplateMLP, train as train_mlp
from cascade_planner.expand.enzymemap_loader import (extract_templates_from_enzymemap,
                                                      load_filtered)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
AIZDATA = ROOT / "aizdata"


class SoftmaxWrapper(nn.Module):
    """Wraps an MLP that returns logits and applies softmax for ONNX export.

    AiZynth's ``_cutoff_predictions`` does ``cumsum`` over the predictions,
    so we must emit normalised probabilities, not logits.
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.base(x), dim=-1)


def template_hash(tpl: str) -> str:
    return hashlib.sha256(tpl.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cascade_dataset_v2.normalized.json")
    ap.add_argument("--em-cap", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--out-prefix", default="enzexpand")
    args = ap.parse_args()

    AIZDATA.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.data}")
    steps, _, _ = load_v2(args.data)
    enz_train = [s for s in steps if s.ec_number]
    print(f"  enzymatic steps: {len(enz_train)}")

    print(f"[em] loading EnzymeMap (q>=0.95 single-step, cap {args.em_cap}/EC1)")
    em_df = load_filtered(min_quality=0.95, only_single=True, ec1_balance=args.em_cap)
    em_rows = extract_templates_from_enzymemap(em_df)
    print(f"  EnzymeMap usable rows: {len(em_rows)}")

    print(f"[pool] building template pool ...")
    X, y, tpls, _ = build_template_pool(enz_train, em_rows)
    n_tpl = len(tpls)
    n_samples = X.shape[0]
    print(f"  n_samples={n_samples}  n_templates={n_tpl}  feature_dim={X.shape[1]}")

    print(f"[train] EnzExpand MLP ({args.epochs} ep, hidden={args.hidden}) ...")
    model = train_mlp(X, y, n_tpl=n_tpl, epochs=args.epochs, hidden=args.hidden,
                      dropout=args.dropout, batch=512, seed=0, verbose=True)
    model.eval()

    # ---------------- export ONNX ---------------- #
    onnx_path = AIZDATA / f"{args.out_prefix}_model.onnx"
    print(f"[onnx] exporting to {onnx_path}")
    wrapped = SoftmaxWrapper(model).cpu().eval()
    dummy = torch.zeros(1, X.shape[1], dtype=torch.float32)
    torch.onnx.export(
        wrapped, dummy, str(onnx_path),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"  exported  ({onnx_path.stat().st_size/1024:.0f} KB)")

    # ---------------- write templates TSV.gz ---------------- #
    tpl_path = AIZDATA / f"{args.out_prefix}_templates.csv.gz"
    print(f"[tpl] writing {tpl_path}")
    # AiZynth: index_col=0, sep="\t"; columns: template_code, retro_template,
    # template_hash, classification, library_occurence (popped from metadata)
    occ = collections.Counter(int(v) for v in y.tolist())
    rows = []
    for code, tpl in enumerate(tpls):
        rows.append(dict(
            template_code=code,
            retro_template=tpl,
            template_hash=template_hash(tpl),
            classification="enz.unrecognized",
            library_occurence=int(occ.get(code, 0)),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(tpl_path, sep="\t", index=False, compression="gzip")
    print(f"  wrote {len(df)} templates  ({tpl_path.stat().st_size/1024:.0f} KB)")

    # ---------------- hybrid YAML config ---------------- #
    cfg_path = AIZDATA / "config_hybrid.yml"
    print(f"[cfg] writing {cfg_path}")
    yaml_str = f"""expansion:
  uspto:
    - {AIZDATA / 'uspto_model.onnx'}
    - {AIZDATA / 'uspto_templates.csv.gz'}
  enzexpand:
    - {onnx_path}
    - {tpl_path}
filter:
  uspto: {AIZDATA / 'uspto_filter_model.onnx'}
stock:
  zinc: {AIZDATA / 'zinc_stock.hdf5'}
"""
    cfg_path.write_text(yaml_str.replace("\\", "/"))
    print(yaml_str)
    print("\n[done] EnzExpand registered as a native AiZynth expansion policy.")
    print("Run multi-step retro with both USPTO + EnzExpand:")
    print(f"  python -m cascade_planner.multistep.plan_route \\")
    print(f"      --product '<SMILES>' --policies uspto enzexpand \\")
    print(f"      --config aizdata/config_hybrid.yml")


if __name__ == "__main__":
    main()
