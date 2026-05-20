"""Train a neural USPTO product-to-template preselector.

This is supervised by the USPTO-50K row ids embedded in RetroRules/USPTO
template rows (`REACTIONS=USPTOB_<row_index>`). It intentionally trains a model
rather than adding a reaction-memory retrieval layer.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cascade_planner.cascadeboard.chemical_template_preselector import ChemicalTemplatePreselectorModel
from cascade_planner.vnext.features import morgan_fp


def train_chemical_template_preselector(
    *,
    uspto_tab: Path,
    template_csv: Path,
    output_dir: Path,
    max_rows: int = 50000,
    max_templates: int = 5000,
    n_bits: int = 512,
    hidden: int = 512,
    epochs: int = 8,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 20260507,
    device: str = "auto",
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    templates = _load_uspto_templates(template_csv)
    reaction_to_templates: dict[int, list[str]] = defaultdict(list)
    template_support: dict[str, int] = {}
    for row in templates:
        tid = row["template_id"]
        support = int(row.get("reactions_count") or 0)
        template_support[tid] = support
        for rid in row.get("reaction_ids") or []:
            reaction_to_templates[int(rid)].append(tid)
    vocab = [
        tid
        for tid, _ in Counter({
            tid: template_support.get(tid, 0)
            for ids in reaction_to_templates.values()
            for tid in ids
        }).most_common(max_templates)
    ]
    vocab_set = set(vocab)
    template_to_index = {tid: idx for idx, tid in enumerate(vocab)}

    products: list[str] = []
    labels: list[int] = []
    skipped = Counter()
    with uspto_tab.open("r", encoding="utf-8", newline="") as fh:
        for idx, row in enumerate(csv.DictReader(fh, delimiter="\t")):
            if len(products) >= max_rows:
                break
            product = row.get("product") or ""
            ids = [tid for tid in reaction_to_templates.get(idx, []) if tid in vocab_set]
            if not product:
                skipped["missing_product"] += 1
                continue
            if not ids:
                skipped["no_template_label"] += 1
                continue
            best = max(ids, key=lambda tid: template_support.get(tid, 0))
            products.append(product)
            labels.append(template_to_index[best])

    if not products:
        raise ValueError("no trainable USPTO product/template rows were built")

    x = np.asarray([morgan_fp(smi, n_bits=n_bits) for smi in products], dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    order = np.arange(len(y))
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    split = max(1, int(len(order) * 0.9))
    train_idx = order[:split]
    val_idx = order[split:]

    torch_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    model = ChemicalTemplatePreselectorModel(n_bits, len(vocab), hidden=hidden, dropout=0.15).to(torch_device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx])),
        batch_size=batch_size,
        shuffle=True,
    )
    val_x = torch.from_numpy(x[val_idx]).to(torch_device)
    val_y = torch.from_numpy(y[val_idx]).to(torch_device)

    history = []
    best = {"val_loss": float("inf"), "state": None, "epoch": 0}
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for bx, by in train_loader:
            bx = bx.to(torch_device)
            by = by.to(torch_device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(bx), by)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        metrics = _eval(model, val_x, val_y, loss_fn)
        metrics.update({"epoch": epoch, "train_loss": round(sum(losses) / max(len(losses), 1), 6)})
        history.append(metrics)
        if metrics["val_loss"] < best["val_loss"]:
            best = {
                "val_loss": metrics["val_loss"],
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), output_dir / "chemical_template_preselector.pt")
    manifest = {
        "schema_version": "chemical_template_preselector.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "uspto_tab": str(uspto_tab),
        "template_csv": str(template_csv),
        "n_bits": n_bits,
        "hidden": hidden,
        "max_rows": max_rows,
        "rows": len(products),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "n_templates": len(vocab),
        "template_ids": vocab,
        "best_epoch": best["epoch"],
        "best_val_loss": round(float(best["val_loss"]), 6),
        "history": history,
        "skipped": dict(skipped),
    }
    (output_dir / "chemical_template_preselector.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "chemical_template_preselector.md").write_text(_report(manifest), encoding="utf-8")
    return manifest


def _load_uspto_templates(path: Path) -> list[dict[str, Any]]:
    out = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("VALID") or "").lower() != "true":
                continue
            tid = row.get("TEMPLATE_ID") or ""
            template = row.get("TEMPLATE") or ""
            if not tid or ">>" not in template:
                continue
            reaction_ids = []
            for token in (row.get("REACTIONS") or "").split(";"):
                token = token.strip()
                if token.startswith("USPTOB_"):
                    try:
                        reaction_ids.append(int(token.split("_", 1)[1]))
                    except ValueError:
                        pass
            if not reaction_ids:
                continue
            out.append({
                "template_id": tid,
                "template": template,
                "reactions_count": int(float(row.get("REACTIONS_COUNT") or 0)),
                "reaction_ids": reaction_ids,
            })
    return out


def _eval(model, val_x, val_y, loss_fn) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        logits = model(val_x)
        loss = float(loss_fn(logits, val_y).detach().cpu())
        max_k = min(50, logits.shape[-1])
        _, pred = torch.topk(logits, k=max_k, dim=-1)
        y = val_y.unsqueeze(1)
        return {
            "val_loss": round(loss, 6),
            "top1": round(float((pred[:, :1] == y).any(dim=1).float().mean().detach().cpu()), 6),
            "top10": round(float((pred[:, : min(10, max_k)] == y).any(dim=1).float().mean().detach().cpu()), 6),
            "top50": round(float((pred[:, :max_k] == y).any(dim=1).float().mean().detach().cpu()), 6),
        }


def _report(manifest: dict[str, Any]) -> str:
    last = manifest.get("history", [{}])[-1]
    return "\n".join([
        "# Chemical Template Preselector",
        "",
        f"Rows: `{manifest.get('rows')}`",
        f"Templates: `{manifest.get('n_templates')}`",
        f"Best epoch: `{manifest.get('best_epoch')}`",
        f"Best val loss: `{manifest.get('best_val_loss')}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| val top1 | {last.get('top1')} |",
        f"| val top10 | {last.get('top10')} |",
        f"| val top50 | {last.get('top50')} |",
        "",
    ]) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Train USPTO chemical template preselector")
    ap.add_argument("--uspto-tab", default="data/uspto50k.tab")
    ap.add_argument("--template-csv", default="data_external/retrorules/templates_uspto.csv.gz")
    ap.add_argument("--output-dir", default="results/shared/chemical_template_preselector/uspto_product_mlp_20260507")
    ap.add_argument("--max-rows", type=int, default=50000)
    ap.add_argument("--max-templates", type=int, default=5000)
    ap.add_argument("--n-bits", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    manifest = train_chemical_template_preselector(
        uspto_tab=Path(args.uspto_tab),
        template_csv=Path(args.template_csv),
        output_dir=Path(args.output_dir),
        max_rows=args.max_rows,
        max_templates=args.max_templates,
        n_bits=args.n_bits,
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    print(json.dumps({k: manifest.get(k) for k in ["rows", "n_templates", "best_epoch", "best_val_loss", "history"]}, indent=2))


if __name__ == "__main__":
    main()
