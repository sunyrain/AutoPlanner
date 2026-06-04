#!/usr/bin/env python3
"""Train a dual-tower template retriever for product-only retrosynthesis."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, MACCSkeys, rdMolDescriptors
from torch.utils.data import DataLoader, Dataset


RDLogger.DisableLog("rdApp.*")

SCHEMA_VERSION = "dual_tower_template_retriever.v2"
FEATURE_SETS = ("baseline", "enhanced")
ARCHITECTURES = ("baseline", "residual")
PRODUCT_DESCRIPTOR_DIM = 18
TEMPLATE_DESCRIPTOR_DIM = 24


@dataclass
class PairRow:
    product: str
    template: str
    template_id: int


class TemplatePairDataset(Dataset):
    def __init__(self, rows: list[PairRow], *, n_bits: int, feature_set: str):
        self.rows = rows
        self.n_bits = n_bits
        self.feature_set = feature_set

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, int]:
        row = self.rows[idx]
        return (
            product_features(row.product, self.n_bits, self.feature_set),
            template_features(row.template, self.n_bits, self.feature_set),
            int(row.template_id),
        )


class FingerprintTower(nn.Module):
    def __init__(self, n_bits: int, hidden: int, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bits, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(x), p=2, dim=-1)


class ResidualTower(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dim: int, dropout: float):
        super().__init__()
        self.input = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.block1 = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        h = h + self.block1(h)
        h = h + self.block2(h)
        return nn.functional.normalize(self.output(h), p=2, dim=-1)


class DualTowerRetriever(nn.Module):
    def __init__(
        self,
        *,
        n_bits: int,
        hidden: int,
        dim: int,
        dropout: float,
        product_dim: int | None = None,
        template_dim: int | None = None,
        architecture: str = "baseline",
    ):
        super().__init__()
        product_dim = int(product_dim or n_bits)
        template_dim = int(template_dim or n_bits)
        if architecture == "baseline":
            self.product_tower = FingerprintTower(product_dim, hidden, dim, dropout)
            self.template_tower = FingerprintTower(template_dim, hidden, dim, dropout)
        elif architecture == "residual":
            self.product_tower = ResidualTower(product_dim, hidden, dim, dropout)
            self.template_tower = ResidualTower(template_dim, hidden, dim, dropout)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        self.logit_scale = nn.Parameter(torch.tensor(math.log(20.0)))

    def forward(self, product_fp: torch.Tensor, template_fp: torch.Tensor) -> torch.Tensor:
        product_vec = self.product_tower(product_fp)
        template_vec = self.template_tower(template_fp)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * product_vec @ template_vec.T


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = load_pairs(args.pairs_jsonl, limit=args.limit)
    if args.shuffle:
        random.shuffle(rows)
    if args.valid_pairs_jsonl:
        train_rows = rows
        valid_rows = load_pairs(args.valid_pairs_jsonl, limit=args.valid_limit)
    else:
        train_rows, valid_rows = split_rows(rows, args.valid_fraction)
    product_dim, template_dim = feature_dims(args.n_bits, args.feature_set)
    train_ds = TemplatePairDataset(train_rows, n_bits=args.n_bits, feature_set=args.feature_set)
    valid_ds = TemplatePairDataset(valid_rows, n_bits=args.n_bits, feature_set=args.feature_set)
    dl_kwargs: dict[str, Any] = {"num_workers": args.num_workers}
    if int(args.num_workers) > 0:
        dl_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **dl_kwargs)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, **dl_kwargs)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DualTowerRetriever(
        n_bits=args.n_bits,
        hidden=args.hidden,
        dim=args.dim,
        dropout=args.dropout,
        product_dim=product_dim,
        template_dim=template_dim,
        architecture=args.architecture,
    ).to(device)
    if args.init_model:
        payload = torch.load(args.init_model, map_location="cpu")
        model.load_state_dict(payload["state_dict"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    started = time.monotonic()
    best_state = None
    best_recall = -1.0
    for epoch in range(max(1, args.epochs)):
        train_loss = train_epoch(model, train_dl, opt, device)
        valid_metrics = evaluate(model, valid_dl, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            **valid_metrics,
        }
        history.append(row)
        if valid_metrics["recall@10"] > best_recall:
            best_recall = valid_metrics["recall@10"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "settings": {
            "n_bits": args.n_bits,
            "hidden": args.hidden,
            "dim": args.dim,
            "dropout": args.dropout,
            "feature_set": args.feature_set,
            "product_dim": product_dim,
            "template_dim": template_dim,
            "architecture": args.architecture,
        },
    }
    torch.save(artifact, args.output_dir / "dual_tower_fp_retriever.pt")
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "data": {
            "rows": len(rows),
            "train_rows": len(train_rows),
            "valid_rows": len(valid_rows),
            "unique_templates": len({row.template_id for row in rows}),
            "product_dim": product_dim,
            "template_dim": template_dim,
        },
        "history": history,
        "elapsed_s": round(time.monotonic() - started, 3),
        "contract": _contract(args.feature_set, args.architecture),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")


def train_epoch(model: DualTowerRetriever, dl: DataLoader, opt: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total = 0.0
    n_seen = 0
    for product_fp, template_fp, template_ids in dl:
        product_fp = product_fp.float().to(device)
        template_fp = template_fp.float().to(device)
        template_ids = template_ids.to(device)
        logits = model(product_fp, template_fp)
        loss = multi_positive_contrastive_loss(logits, template_ids)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += float(loss.item()) * len(product_fp)
        n_seen += len(product_fp)
    return total / max(n_seen, 1)


def evaluate(model: DualTowerRetriever, dl: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total = 0
    hits = {1: 0, 5: 0, 10: 0}
    losses = []
    with torch.no_grad():
        for product_fp, template_fp, template_ids in dl:
            product_fp = product_fp.float().to(device)
            template_fp = template_fp.float().to(device)
            template_ids = template_ids.to(device)
            logits = model(product_fp, template_fp)
            losses.append(float(multi_positive_contrastive_loss(logits, template_ids).item()))
            order = torch.argsort(logits, dim=1, descending=True)
            ordered_template_ids = template_ids[order]
            for k in hits:
                hits[k] += int(
                    (ordered_template_ids[:, : min(k, order.shape[1])] == template_ids[:, None]).any(dim=1).sum().item()
                )
            total += int(logits.shape[0])
    return {
        "valid_loss": round(float(np.mean(losses)) if losses else 0.0, 6),
        "recall@1": round(hits[1] / max(total, 1), 6),
        "recall@5": round(hits[5] / max(total, 1), 6),
        "recall@10": round(hits[10] / max(total, 1), 6),
    }


def load_pairs(path: Path, *, limit: int | None) -> list[PairRow]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(PairRow(
                product=str(item.get("product") or ""),
                template=str(item.get("template") or ""),
                template_id=int(item.get("template_id") or 0),
            ))
            if limit is not None and len(rows) >= int(limit):
                break
    return [row for row in rows if row.product and row.template]


def multi_positive_contrastive_loss(logits: torch.Tensor, template_ids: torch.Tensor) -> torch.Tensor:
    positive = template_ids[:, None].eq(template_ids[None, :]).float()
    row_loss = _multi_positive_axis_loss(logits, positive, dim=1)
    col_loss = _multi_positive_axis_loss(logits.T, positive.T, dim=1)
    return 0.5 * (row_loss + col_loss)


def _multi_positive_axis_loss(logits: torch.Tensor, positive: torch.Tensor, *, dim: int) -> torch.Tensor:
    log_prob = logits - torch.logsumexp(logits, dim=dim, keepdim=True)
    denom = positive.sum(dim=dim).clamp_min(1.0)
    return -((positive * log_prob).sum(dim=dim) / denom).mean()


def split_rows(rows: list[PairRow], valid_fraction: float) -> tuple[list[PairRow], list[PairRow]]:
    pivot = max(1, int(round(len(rows) * (1.0 - valid_fraction))))
    return rows[:pivot], rows[pivot:] or rows[-min(len(rows), 1):]


def feature_dims(n_bits: int, feature_set: str) -> tuple[int, int]:
    if feature_set == "baseline":
        return int(n_bits), int(n_bits)
    if feature_set == "enhanced":
        product_dim = (4 * int(n_bits)) + 167 + PRODUCT_DESCRIPTOR_DIM
        template_dim = (6 * int(n_bits)) + TEMPLATE_DESCRIPTOR_DIM
        return product_dim, template_dim
    raise ValueError(f"Unsupported feature_set: {feature_set}")


def product_features(smiles: str, n_bits: int, feature_set: str) -> np.ndarray:
    if feature_set == "baseline":
        return morgan_fp(smiles, n_bits)
    if feature_set == "enhanced":
        return np.concatenate(
            [
                morgan_fp(smiles, n_bits, radius=2, use_features=False),
                morgan_fp(smiles, n_bits, radius=3, use_features=False),
                morgan_fp(smiles, n_bits, radius=2, use_features=True),
                pattern_fp(smiles, n_bits),
                maccs_fp(smiles),
                product_descriptor_features(smiles),
            ]
        ).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported feature_set: {feature_set}")


def template_features(template: str, n_bits: int, feature_set: str) -> np.ndarray:
    if feature_set == "baseline":
        return template_fp(template, n_bits)
    if feature_set == "enhanced":
        return _template_features_cached(str(template or ""), int(n_bits)).copy()
    raise ValueError(f"Unsupported feature_set: {feature_set}")


def morgan_fp(smiles: str, n_bits: int, *, radius: int = 2, use_features: bool = False) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return arr
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        int(radius),
        nBits=n_bits,
        useChirality=True,
        useFeatures=bool(use_features),
    )
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def pattern_fp(smiles: str, n_bits: int) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return arr
    fp = Chem.PatternFingerprint(mol, fpSize=n_bits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def maccs_fp(smiles: str) -> np.ndarray:
    arr = np.zeros(167, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return arr
    fp = MACCSkeys.GenMACCSKeys(mol)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def product_descriptor_features(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return np.zeros(PRODUCT_DESCRIPTOR_DIM, dtype=np.float32)
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    hetero_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))
    values = np.asarray(
        [
            _scaled(mol.GetNumHeavyAtoms(), 80.0),
            _scaled(Descriptors.MolWt(mol), 800.0),
            _scaled(Crippen.MolLogP(mol) + 10.0, 20.0),
            _scaled(rdMolDescriptors.CalcTPSA(mol), 250.0),
            _scaled(Lipinski.NumHDonors(mol), 10.0),
            _scaled(Lipinski.NumHAcceptors(mol), 20.0),
            _scaled(Lipinski.NumRotatableBonds(mol), 30.0),
            _scaled(rdMolDescriptors.CalcNumRings(mol), 12.0),
            _scaled(rdMolDescriptors.CalcNumAromaticRings(mol), 10.0),
            _scaled(rdMolDescriptors.CalcNumAliphaticRings(mol), 10.0),
            float(rdMolDescriptors.CalcFractionCSP3(mol)),
            _scaled(abs(Chem.GetFormalCharge(mol)), 5.0),
            _scaled(aromatic_atoms, 60.0),
            _scaled(hetero_atoms, 40.0),
            _scaled(rdMolDescriptors.CalcNumAmideBonds(mol), 10.0),
            _scaled(rdMolDescriptors.CalcNumBridgeheadAtoms(mol), 10.0),
            _scaled(rdMolDescriptors.CalcNumSpiroAtoms(mol), 10.0),
            _scaled(sum(1 for atom in mol.GetAtoms() if atom.HasProp("_CIPCode")), 20.0),
        ],
        dtype=np.float32,
    )
    return values


def template_fp(template: str, n_bits: int) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.float32)
    text = str(template or "")
    if not text:
        return arr
    normalized = "".join(ch for ch in text if not ch.isspace())
    for n in (2, 3, 4, 5):
        if len(normalized) < n:
            continue
        for idx in range(len(normalized) - n + 1):
            token = normalized[idx : idx + n]
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bit = int.from_bytes(digest, "little") % n_bits
            arr[bit] = 1.0
    return arr


@lru_cache(maxsize=500000)
def _template_features_cached(template: str, n_bits: int) -> np.ndarray:
    lhs, rhs = _split_template(template)
    return np.concatenate(
        [
            template_fp(template, n_bits),
            template_fp(lhs, n_bits),
            template_fp(rhs, n_bits),
            smarts_pattern_fp(lhs, n_bits),
            smarts_pattern_fp(rhs, n_bits),
            smarts_token_fp(template, n_bits),
            template_descriptor_features(template, lhs, rhs),
        ]
    ).astype(np.float32, copy=False)


def _split_template(template: str) -> tuple[str, str]:
    if ">>" not in template:
        return str(template or ""), ""
    lhs, rhs = template.split(">>", 1)
    return lhs, rhs


def smarts_pattern_fp(smarts: str, n_bits: int) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.float32)
    if not smarts:
        return arr
    bitvect = None
    for part in [item for item in str(smarts).split(".") if item]:
        try:
            mol = Chem.MolFromSmarts(part)
            if mol is None:
                continue
            fp = Chem.PatternFingerprint(mol, fpSize=n_bits)
            bitvect = fp if bitvect is None else bitvect | fp
        except Exception:
            continue
    if bitvect is not None:
        DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def smarts_token_fp(template: str, n_bits: int) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.float32)
    text = str(template or "")
    tokens = re.findall(r"\[[^\]]+\]|Cl|Br|Si|Se|Na|Li|Mg|Al|Ca|[A-Z][a-z]?|[cnops]|\d+|[#:=@+\-()/.\\]", text)
    for idx, token in enumerate(tokens):
        for prefix in ("tok", f"pos{idx % 17}"):
            digest = hashlib.blake2b(f"{prefix}:{token}".encode("utf-8"), digest_size=8).digest()
            bit = int.from_bytes(digest, "little") % n_bits
            arr[bit] = 1.0
    return arr


def template_descriptor_features(template: str, lhs: str, rhs: str) -> np.ndarray:
    text = str(template or "")
    bracket_tokens = re.findall(r"\[[^\]]+\]", text)
    mapped_atoms = re.findall(r":\d+", text)
    values = [
        _scaled(len(text), 2000.0),
        _scaled(len(lhs), 1200.0),
        _scaled(len(rhs), 1200.0),
        _scaled(text.count("."), 20.0),
        _scaled(text.count("("), 60.0),
        _scaled(text.count("="), 80.0),
        _scaled(text.count("#"), 30.0),
        _scaled(text.count("@"), 20.0),
        _scaled(text.count("+"), 20.0),
        _scaled(text.count("-"), 40.0),
        _scaled(len(bracket_tokens), 120.0),
        _scaled(len(set(mapped_atoms)), 80.0),
        _scaled(len(mapped_atoms), 120.0),
        _scaled(sum(1 for ch in text if ch.isdigit()), 120.0),
        _scaled(sum(1 for ch in text if ch in "cnops"), 120.0),
        _scaled(text.count("Cl"), 20.0),
        _scaled(text.count("Br"), 20.0),
        _scaled(text.count("F"), 40.0),
        _scaled(text.count("I"), 20.0),
        _scaled(text.count("N") + text.count("n"), 80.0),
        _scaled(text.count("O") + text.count("o"), 80.0),
        _scaled(text.count("S") + text.count("s"), 40.0),
        _scaled(text.count("P") + text.count("p"), 30.0),
        float(">>" in text),
    ]
    return np.asarray(values, dtype=np.float32)


def _scaled(value: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return float(np.clip(float(value) / float(denom), -1.0, 1.0))


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Dual Tower FP Retriever",
        "",
        str(report["contract"]),
        "",
        "| epoch | loss | recall@1 | recall@5 | recall@10 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("history") or []:
        lines.append(
            f"| {row['epoch']} | {row['valid_loss']} | {row['recall@1']} | {row['recall@5']} | {row['recall@10']} |"
        )
    return "\n".join(lines)


def _contract(feature_set: str, architecture: str) -> str:
    if feature_set == "enhanced" or architecture == "residual":
        return (
            "Enhanced dual tower: multi-view product fingerprints, structured SMARTS/template "
            "features, and residual towers. This is intended as a stronger retrieval baseline, "
            "but exact one-step quality still requires rdchiral evaluation."
        )
    return (
        "Baseline fingerprint dual tower retained for checkpoint compatibility and ablation."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--valid-pairs-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-model", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--valid-limit", type=int)
    parser.add_argument("--n-bits", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="baseline")
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="baseline")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
