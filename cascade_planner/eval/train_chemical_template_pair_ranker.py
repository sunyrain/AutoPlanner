"""Train a pairwise USPTO product/template ranker."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs
from torch.utils.data import DataLoader, TensorDataset

from cascade_planner.cascadeboard.chemical_template_pair_ranker import ChemicalTemplatePairRankerModel
from cascade_planner.expand.enz_template import apply_template_to_product, canon_set
from cascade_planner.vnext.features import morgan_fp


def train_pair_ranker(
    *,
    uspto_tab: Path,
    template_csv: Path,
    output_dir: Path,
    max_rows: int = 20000,
    max_templates: int = 5000,
    negatives_per_positive: int = 12,
    hard_negative_attempts: int = 80,
    generated_negative_attempts: int = 0,
    n_bits: int = 256,
    hidden: int = 512,
    epochs: int = 6,
    batch_size: int = 1024,
    lr: float = 1e-3,
    seed: int = 20260507,
    device: str = "auto",
) -> dict[str, Any]:
    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    templates = _load_templates(template_csv)
    templates = sorted(templates, key=lambda row: row["reactions_count"], reverse=True)[:max_templates]
    by_id = {row["template_id"]: row for row in templates}
    reaction_to_templates: dict[int, list[str]] = defaultdict(list)
    for row in templates:
        for rid in row["reaction_ids"]:
            reaction_to_templates[int(rid)].append(row["template_id"])
    template_ids = list(by_id)

    rows_x: list[np.ndarray] = []
    rows_y: list[float] = []
    product_cache: dict[str, np.ndarray] = {}
    template_cache: dict[str, np.ndarray] = {}
    products_seen = 0
    negative_stats: dict[str, int] = {"generated": 0, "product_smarts": 0, "random": 0}
    with uspto_tab.open("r", encoding="utf-8", newline="") as fh:
        for ridx, row in enumerate(csv.DictReader(fh, delimiter="\t")):
            if products_seen >= max_rows:
                break
            product = row.get("product") or ""
            positive_ids = [tid for tid in reaction_to_templates.get(ridx, []) if tid in by_id]
            if not product or not positive_ids:
                continue
            product_mol = Chem.MolFromSmiles(product)
            if product_mol is None:
                continue
            true_reactants = canon_set(row.get("reactant") or "")
            if not true_reactants:
                continue
            positives = positive_ids[:2]
            negative_ids, stats = _sample_negatives(
                templates,
                set(positive_ids),
                negatives_per_positive * len(positives),
                rng,
                product=product,
                product_mol=product_mol,
                true_reactants=true_reactants,
                hard_negative_attempts=hard_negative_attempts,
                generated_negative_attempts=generated_negative_attempts,
            )
            for key, value in stats.items():
                negative_stats[key] = negative_stats.get(key, 0) + value
            for tid in positives:
                rows_x.append(_feature(product, product_mol, by_id[tid], n_bits, product_cache, template_cache))
                rows_y.append(1.0)
            for tid in negative_ids:
                rows_x.append(_feature(product, product_mol, by_id[tid], n_bits, product_cache, template_cache))
                rows_y.append(0.0)
            products_seen += 1

    x = np.asarray(rows_x, dtype=np.float32)
    y = np.asarray(rows_y, dtype=np.float32)
    order = np.arange(len(y))
    np.random.default_rng(seed).shuffle(order)
    split = max(1, int(len(order) * 0.9))
    train_idx, val_idx = order[:split], order[split:]
    torch_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    model = ChemicalTemplatePairRankerModel(n_bits, hidden=hidden, dropout=0.15).to(torch_device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx])),
        batch_size=batch_size,
        shuffle=True,
    )
    val_x = torch.from_numpy(x[val_idx]).to(torch_device)
    val_y = torch.from_numpy(y[val_idx]).to(torch_device)
    best = {"val_loss": float("inf"), "epoch": 0, "state": None}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for bx, by in loader:
            bx = bx.to(torch_device)
            by = by.to(torch_device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(bx), by)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        metrics = _eval(model, val_x, val_y, loss_fn)
        metrics["epoch"] = epoch
        metrics["train_loss"] = round(sum(losses) / max(len(losses), 1), 6)
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
    torch.save(model.cpu().state_dict(), output_dir / "chemical_template_pair_ranker.pt")
    manifest = {
        "schema_version": "chemical_template_pair_ranker.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "uspto_tab": str(uspto_tab),
        "template_csv": str(template_csv),
        "products_seen": products_seen,
        "pairs": int(len(y)),
        "positives": int(y.sum()),
        "negatives": int(len(y) - y.sum()),
        "n_bits": n_bits,
        "hidden": hidden,
        "max_templates": max_templates,
        "negatives_per_positive": negatives_per_positive,
        "hard_negative_attempts": hard_negative_attempts,
        "generated_negative_attempts": generated_negative_attempts,
        "negative_sampling": negative_stats,
        "best_epoch": best["epoch"],
        "best_val_loss": round(float(best["val_loss"]), 6),
        "history": history,
    }
    (output_dir / "chemical_template_pair_ranker.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "chemical_template_pair_ranker.md").write_text(_report(manifest), encoding="utf-8")
    return manifest


def _feature(product: str, product_mol, template: dict[str, Any], n_bits: int, product_cache, template_cache) -> np.ndarray:
    pfp = product_cache.get(product)
    if pfp is None:
        pfp = morgan_fp(product, n_bits=n_bits)
        product_cache[product] = pfp
    tid = template["template_id"]
    tfp = template_cache.get(tid)
    query = template["query"]
    if tfp is None:
        arr = np.zeros(n_bits, dtype=np.float32)
        if query is not None:
            fp = Chem.PatternFingerprint(query, fpSize=n_bits)
            DataStructs.ConvertToNumpyArray(fp, arr)
        tfp = arr
        template_cache[tid] = tfp
    match = float(query is not None and product_mol.HasSubstructMatch(query))
    atoms = float(query.GetNumAtoms() if query is not None else 0.0) / 64.0
    support = min(float(template.get("reactions_count") or 0.0), 100.0) / 100.0
    return np.concatenate([pfp, tfp, np.asarray([match, atoms, support], dtype=np.float32)])


def _sample_negatives(
    templates: list[dict[str, Any]],
    positives: set[str],
    n: int,
    rng: random.Random,
    *,
    product: str,
    product_mol,
    true_reactants: frozenset[str],
    hard_negative_attempts: int = 80,
    generated_negative_attempts: int = 0,
) -> tuple[list[str], dict[str, int]]:
    out = []
    seen = set(positives)
    template_ids = [row["template_id"] for row in templates]
    stats = {"generated": 0, "product_smarts": 0, "random": 0}
    attempts = 0
    while len(out) < n and attempts < max(0, generated_negative_attempts):
        attempts += 1
        row = templates[rng.randrange(len(templates))]
        tid = row["template_id"]
        if tid in seen:
            continue
        query = row.get("query")
        if query is None or not product_mol.HasSubstructMatch(query):
            continue
        outcomes = apply_template_to_product(row["template"], product, max_outcomes=2)
        if not outcomes:
            continue
        if any(outcome == true_reactants for outcome in outcomes):
            continue
        seen.add(tid)
        out.append(tid)
        stats["generated"] += 1
    attempts = 0
    while len(out) < n and attempts < max(0, hard_negative_attempts):
        attempts += 1
        row = templates[rng.randrange(len(templates))]
        tid = row["template_id"]
        if tid in seen:
            continue
        query = row.get("query")
        if query is None or not product_mol.HasSubstructMatch(query):
            continue
        seen.add(tid)
        out.append(tid)
        stats["product_smarts"] += 1
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        tid = template_ids[rng.randrange(len(template_ids))]
        if tid in seen:
            continue
        seen.add(tid)
        out.append(tid)
        stats["random"] += 1
    return out, stats


def _load_templates(path: Path) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("VALID") or "").lower() != "true":
                continue
            tid = row.get("TEMPLATE_ID") or ""
            template = row.get("TEMPLATE") or ""
            if not tid or ">>" not in template:
                continue
            product_smarts = template.split(">>", 1)[0]
            query = Chem.MolFromSmarts(product_smarts)
            if query is None:
                continue
            reaction_ids = []
            for token in (row.get("REACTIONS") or "").split(";"):
                if token.startswith("USPTOB_"):
                    try:
                        reaction_ids.append(int(token.split("_", 1)[1]))
                    except ValueError:
                        pass
            if not reaction_ids:
                continue
            rows.append({
                "template_id": tid,
                "template": template,
                "product_smarts": product_smarts,
                "query": query,
                "reactions_count": int(float(row.get("REACTIONS_COUNT") or 0)),
                "reaction_ids": reaction_ids,
            })
    return rows


def _eval(model, val_x, val_y, loss_fn) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        logits = model(val_x)
        loss = float(loss_fn(logits, val_y).detach().cpu())
        pred = (torch.sigmoid(logits) >= 0.5).float()
        acc = float((pred == val_y).float().mean().detach().cpu())
        pos_mask = val_y >= 0.5
        neg_mask = ~pos_mask
        pos_mean = float(torch.sigmoid(logits[pos_mask]).mean().detach().cpu()) if pos_mask.any() else 0.0
        neg_mean = float(torch.sigmoid(logits[neg_mask]).mean().detach().cpu()) if neg_mask.any() else 0.0
    return {
        "val_loss": round(loss, 6),
        "accuracy": round(acc, 6),
        "positive_score_mean": round(pos_mean, 6),
        "negative_score_mean": round(neg_mean, 6),
    }


def _report(manifest: dict[str, Any]) -> str:
    last = manifest.get("history", [{}])[-1]
    return "\n".join([
        "# Chemical Template Pair Ranker",
        "",
        f"Products: `{manifest.get('products_seen')}`",
        f"Pairs: `{manifest.get('pairs')}`",
        f"Positives: `{manifest.get('positives')}`",
        f"Negatives: `{manifest.get('negatives')}`",
        f"Best epoch: `{manifest.get('best_epoch')}`",
        f"Best val loss: `{manifest.get('best_val_loss')}`",
        f"Generated-negative attempts: `{manifest.get('generated_negative_attempts')}`",
        f"Negative sampling: `{manifest.get('negative_sampling')}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| accuracy | {last.get('accuracy')} |",
        f"| positive score mean | {last.get('positive_score_mean')} |",
        f"| negative score mean | {last.get('negative_score_mean')} |",
        "",
    ]) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Train pairwise USPTO chemical template ranker")
    ap.add_argument("--uspto-tab", default="data/uspto50k.tab")
    ap.add_argument("--template-csv", default="data_external/retrorules/templates_uspto.csv.gz")
    ap.add_argument("--output-dir", default="results/shared/chemical_template_preselector/uspto_pair_mlp_20260507")
    ap.add_argument("--max-rows", type=int, default=20000)
    ap.add_argument("--max-templates", type=int, default=5000)
    ap.add_argument("--negatives-per-positive", type=int, default=12)
    ap.add_argument("--hard-negative-attempts", type=int, default=80)
    ap.add_argument("--generated-negative-attempts", type=int, default=0)
    ap.add_argument("--n-bits", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    manifest = train_pair_ranker(
        uspto_tab=Path(args.uspto_tab),
        template_csv=Path(args.template_csv),
        output_dir=Path(args.output_dir),
        max_rows=args.max_rows,
        max_templates=args.max_templates,
        negatives_per_positive=args.negatives_per_positive,
        hard_negative_attempts=args.hard_negative_attempts,
        generated_negative_attempts=args.generated_negative_attempts,
        n_bits=args.n_bits,
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    print(json.dumps({k: manifest.get(k) for k in ["products_seen", "pairs", "best_epoch", "best_val_loss", "history"]}, indent=2))


if __name__ == "__main__":
    main()
