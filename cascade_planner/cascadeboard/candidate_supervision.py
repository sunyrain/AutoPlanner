"""Candidate-pool supervision and recall diagnostics for CascadeBoard."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader, TensorDataset

from cascade_planner.cascadeboard.candidate_cache import (
    canon_set, canon_smiles, merge_candidate_caches,
)

RDLogger.DisableLog("rdApp.*")


def _fp(smiles: str | None, n_bits: int = 256) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return arr
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def _tanimoto(a: str | None, b: str | None) -> float:
    ma = Chem.MolFromSmiles(a or "")
    mb = Chem.MolFromSmiles(b or "")
    if ma is None or mb is None:
        return 0.0
    fa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, nBits=2048)
    fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fa, fb))


def _main_product(rxn: str) -> str | None:
    if ">>" not in (rxn or ""):
        return None
    rhs = rxn.split(">>", 1)[1]
    parts = [canon_smiles(x.strip()) for x in rhs.split(".")]
    parts = [x for x in parts if x]
    if not parts:
        return None
    return sorted(parts, key=lambda s: (-Chem.MolFromSmiles(s).GetNumHeavyAtoms(), s))[0]


def _reactant_set_from_row(row: dict[str, Any]) -> frozenset[str]:
    parts = [row.get("main_reactant") or ""]
    parts.extend(row.get("aux_reactants") or [])
    return canon_set(".".join(parts))


def _candidate_features(
    product: str,
    row: dict[str, Any],
    *,
    gt_ec: str | None,
    gt_reaction_type: str | None,
) -> np.ndarray:
    main = row.get("main_reactant") or ""
    aux = ".".join(row.get("aux_reactants") or [])
    reactants = ".".join([x for x in [main, aux] if x])
    source = row.get("source") or ""
    ec = row.get("ec") or ""
    rt = row.get("reaction_type") or ""
    score = float(row.get("score") or 0.0)
    rank = float(row.get("rank") or 0.0)
    dual_score = float(row.get("dual_tower_score") or 0.0)
    e_enzyme = float(row.get("e_enzyme") or 0.0)
    source_vec = np.array([
        1.0 if source == "enzexpand" else 0.0,
        1.0 if source == "retrochimera" else 0.0,
        1.0 if row.get("enzyme_source") == "dual_tower" else 0.0,
        1.0 if ec and gt_ec and ec == gt_ec else 0.0,
        1.0 if ec and gt_ec and ec.split(".")[0] == gt_ec.split(".")[0] else 0.0,
        1.0 if rt and gt_reaction_type and rt == gt_reaction_type else 0.0,
        score,
        1.0 / (1.0 + rank),
        dual_score,
        e_enzyme,
    ], dtype=np.float32)
    return np.concatenate([_fp(product), _fp(reactants), source_vec], dtype=np.float32)


class CandidateReranker(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 192),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_step_rows(data_path: str) -> list[dict[str, Any]]:
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    rows = []
    for art in data.get("records_kept", []):
        doi = art.get("doi", "")
        for cascade in art.get("cascades", []) or []:
            domain = cascade.get("route_domain", "")
            for step in cascade.get("steps", []) or []:
                rxn = step.get("rxn_smiles") or ""
                if ">>" not in rxn:
                    continue
                product = _main_product(rxn)
                gt_set = canon_set(rxn.split(">>", 1)[0])
                if not product or not gt_set:
                    continue
                ec = None
                for cat in step.get("catalyst_components") or []:
                    if cat and cat.get("ec_number"):
                        ec = cat.get("ec_number")
                        break
                rows.append({
                    "doi": doi,
                    "domain": domain,
                    "step_id": step.get("step_id"),
                    "product": product,
                    "gt_reactants": sorted(gt_set),
                    "ec": ec,
                    "ec1": ec.split(".")[0] if ec and "." in ec else "",
                    "reaction_type": step.get("transformation_superclass") or "",
                })
    return rows


def build_candidate_examples(
    *,
    data_path: str,
    cache_paths: list[str],
    output_dataset: str,
    max_candidates_per_step: int = 40,
    nearest_threshold: float = 0.35,
    max_negatives_per_positive: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    rng = random.Random(seed)
    caches = [json.loads(Path(p).read_text(encoding="utf-8")) for p in cache_paths]
    cache = merge_candidate_caches(*caches)
    steps = load_step_rows(data_path)

    examples = []
    step_reports = []
    by_domain = defaultdict(list)
    by_ec1 = defaultdict(list)
    source_hit = Counter()

    for step in steps:
        product = step["product"]
        rows = cache.get(product) or []
        rows = rows[:max_candidates_per_step]
        if not rows:
            rep = {**step, "n_candidates": 0, "exact_rank": None, "nearest_rank": None, "nearest_similarity": 0.0}
            step_reports.append(rep)
            by_domain[step["domain"]].append(rep)
            by_ec1[step["ec1"]].append(rep)
            continue

        gt_set = frozenset(step["gt_reactants"])
        exact = []
        nearest_idx = None
        nearest_sim = -1.0
        for i, row in enumerate(rows):
            cand_set = _reactant_set_from_row(row)
            exact_hit = cand_set == gt_set
            if exact_hit:
                exact.append(i)
                source_hit[row.get("source", "unknown")] += 1
            # Similarity by best main-reactant-to-GT fragment match.
            main = row.get("main_reactant")
            sim = max((_tanimoto(main, gt) for gt in gt_set), default=0.0)
            if sim > nearest_sim:
                nearest_sim = sim
                nearest_idx = i

        positive_idx = set(exact)
        weak_positive_idx = set()
        if not positive_idx and nearest_idx is not None and nearest_sim >= nearest_threshold:
            weak_positive_idx.add(nearest_idx)

        negatives = [i for i in range(len(rows)) if i not in positive_idx and i not in weak_positive_idx]
        rng.shuffle(negatives)
        pos_count = max(len(positive_idx) + len(weak_positive_idx), 1)
        keep_neg = set(negatives[:max_negatives_per_positive * pos_count])
        keep_idx = sorted(positive_idx | weak_positive_idx | keep_neg)

        for i in keep_idx:
            row = rows[i]
            label = 1.0 if i in positive_idx or i in weak_positive_idx else 0.0
            examples.append({
                "doi": step["doi"],
                "domain": step["domain"],
                "ec1": step["ec1"],
                "product": product,
                "candidate": row,
                "label": label,
                "label_type": "exact" if i in positive_idx else ("nearest" if i in weak_positive_idx else "negative"),
                "features": _candidate_features(
                    product,
                    row,
                    gt_ec=step["ec"],
                    gt_reaction_type=step["reaction_type"],
                ).round(6).tolist(),
            })

        rep = {
            **step,
            "n_candidates": len(rows),
            "exact_rank": (min(exact) + 1) if exact else None,
            "nearest_rank": (nearest_idx + 1) if nearest_idx is not None else None,
            "nearest_similarity": round(max(nearest_sim, 0.0), 4),
        }
        step_reports.append(rep)
        by_domain[step["domain"]].append(rep)
        by_ec1[step["ec1"]].append(rep)

    def summarize(sub: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(sub)
        with_pool = [r for r in sub if r["n_candidates"] > 0]
        exact = [r for r in sub if r["exact_rank"] is not None]
        return {
            "n_steps": n,
            "candidate_pool_coverage": len(with_pool) / max(n, 1),
            "exact_gt_in_pool": len(exact) / max(n, 1),
            "exact_gt_in_pool_given_pool": len(exact) / max(len(with_pool), 1),
            "exact_gt_at_1": sum(r["exact_rank"] == 1 for r in sub) / max(n, 1),
            "exact_gt_at_5": sum((r["exact_rank"] or 10**9) <= 5 for r in sub) / max(n, 1),
            "mean_nearest_similarity": sum(r["nearest_similarity"] for r in sub) / max(n, 1),
        }

    dataset = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d"),
            "data_path": data_path,
            "cache_paths": cache_paths,
            "n_steps": len(steps),
            "n_examples": len(examples),
            "n_exact_positive": sum(e["label_type"] == "exact" for e in examples),
            "n_nearest_positive": sum(e["label_type"] == "nearest" for e in examples),
            "source_exact_hits": dict(source_hit),
            "feature_dim": len(examples[0]["features"]) if examples else 0,
            "nearest_threshold": nearest_threshold,
        },
        "overall": summarize(step_reports),
        "by_domain": {k: summarize(v) for k, v in sorted(by_domain.items())},
        "by_ec1": {k: summarize(v) for k, v in sorted(by_ec1.items())},
        "step_reports": step_reports,
        "examples": examples,
    }
    out = Path(output_dataset)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    return dataset


def train_candidate_reranker(
    *,
    dataset_path: str,
    output_model: str,
    output_report: str,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    examples = dataset.get("examples", [])
    if not examples:
        raise SystemExit("No candidate examples available")

    dois = sorted({e["doi"] for e in examples})
    val_dois = set(dois[:max(1, len(dois) // 5)])
    train_ex = [e for e in examples if e["doi"] not in val_dois]
    val_ex = [e for e in examples if e["doi"] in val_dois]
    if not train_ex or not val_ex:
        raise SystemExit("Candidate examples could not be split")

    def tensors(rows):
        x = torch.tensor([r["features"] for r in rows], dtype=torch.float32)
        y = torch.tensor([r["label"] for r in rows], dtype=torch.float32)
        return x, y

    x_train, y_train = tensors(train_ex)
    x_val, y_val = tensors(val_ex)
    model = CandidateReranker(x_train.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos_weight = (len(y_train) - float(y_train.sum())) / max(float(y_train.sum()), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    dl = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    history = []
    best_val = float("inf")
    best_state = None
    for ep in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in dl:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = float(loss_fn(val_logits, y_val).item())
            pred = (torch.sigmoid(val_logits) >= 0.5).float()
            val_acc = float((pred == y_val).float().mean().item())
        row = {
            "epoch": ep + 1,
            "train_loss": round(total / max(len(train_ex), 1), 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        }
        history.append(row)
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()

    out_model = Path(output_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state or model.state_dict(),
        "feature_dim": x_train.shape[1],
        "model_class": "CandidateReranker",
        "dataset_path": dataset_path,
    }, out_model)

    report = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d"),
            "dataset_path": dataset_path,
            "output_model": output_model,
            "n_train": len(train_ex),
            "n_val": len(val_ex),
            "n_pos_train": int(y_train.sum().item()),
            "n_pos_val": int(y_val.sum().item()),
            "pos_weight": round(float(pos_weight), 4),
        },
        "best_val_loss": round(best_val, 4),
        "history": history,
    }
    Path(output_report).parent.mkdir(parents=True, exist_ok=True)
    Path(output_report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cascade_dataset_v2.normalized.json")
    ap.add_argument("--cache", action="append", required=True)
    ap.add_argument("--dataset-output", default="results/shared/cascadeboard_candidate_supervision_v1.json")
    ap.add_argument("--model-output", default="results/shared/cascadeboard_candidate_reranker_v1.pt")
    ap.add_argument("--report-output", default="results/v2/cascadeboard_candidate_supervision_report.json")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--no-train", action="store_true")
    args = ap.parse_args()

    dataset = build_candidate_examples(
        data_path=args.data,
        cache_paths=args.cache,
        output_dataset=args.dataset_output,
    )
    print(json.dumps({
        "overall": dataset["overall"],
        "by_domain": dataset["by_domain"],
        "n_examples": dataset["metadata"]["n_examples"],
    }, indent=2, ensure_ascii=False))
    if not args.no_train:
        report = train_candidate_reranker(
            dataset_path=args.dataset_output,
            output_model=args.model_output,
            output_report=args.report_output,
            epochs=args.epochs,
        )
        print(json.dumps({
            "best_val_loss": report["best_val_loss"],
            "output_model": args.model_output,
            "output_report": args.report_output,
        }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
