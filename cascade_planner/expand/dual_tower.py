"""Dual-tower contrastive model for enzymatic retrosynthesis.

Replaces the template-MLP approach (enz_template.py, lift=0.79) with a
contrastive architecture that learns to match DRFP reaction fingerprints
to ESM-2 enzyme embeddings in a shared 256-dim space.

Architecture
------------
Reaction tower:  DRFP-2048 -> Linear(2048,512) -> LN -> ReLU -> Linear(512,256) -> L2-norm
Enzyme tower:    ESM-2 (1280) -> Linear(1280,512) -> LN -> ReLU -> Linear(512,256) -> L2-norm
Loss:            Symmetric InfoNCE (temperature-scaled cross-entropy)
Inference:       Embed query reaction -> cosine sim against enzyme bank -> top-K
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

from cascade_planner.data.loader_v2 import load_v2
from cascade_planner.paths import results_dir, shared_dir
from cascade_planner.training.featurize_v2 import drfp_one

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ towers


class ReactionTower(nn.Module):
    """DRFP-2048 -> 256-dim L2-normalised embedding."""

    def __init__(self, in_dim: int = 2048, hidden: int = 512, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class EnzymeTower(nn.Module):
    """ESM-2 mean-pool (1280-dim) -> 256-dim L2-normalised embedding."""

    def __init__(self, in_dim: int = 1280, hidden: int = 512, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ------------------------------------------------------------------ model


class DualTowerModel(nn.Module):
    """Contrastive dual-tower: reaction <-> enzyme matching."""

    def __init__(
        self,
        rxn_dim: int = 2048,
        enz_dim: int = 1280,
        hidden: int = 512,
        embed_dim: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.rxn_tower = ReactionTower(rxn_dim, hidden, embed_dim)
        self.enz_tower = EnzymeTower(enz_dim, hidden, embed_dim)
        # learnable log-temperature (initialised to match the CLI default)
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(temperature), dtype=torch.float32)
        )

    def forward(
        self, rxn_fp: torch.Tensor, enz_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rxn_tower(rxn_fp), self.enz_tower(enz_emb)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=1e-4, max=1.0)

    def info_nce_loss(
        self, rxn_embed: torch.Tensor, enz_embed: torch.Tensor
    ) -> torch.Tensor:
        """Symmetric InfoNCE: average of rxn->enz and enz->rxn directions."""
        # (B, B) cosine similarity matrix (embeddings are already L2-normed)
        logits = rxn_embed @ enz_embed.T / self.temperature
        B = logits.size(0)
        labels = torch.arange(B, device=logits.device)
        loss_r2e = F.cross_entropy(logits, labels)
        loss_e2r = F.cross_entropy(logits.T, labels)
        return (loss_r2e + loss_e2r) / 2.0

    def score(self, rxn_fp: torch.Tensor, enz_emb: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between a single reaction and enzyme."""
        r = self.rxn_tower(rxn_fp)
        e = self.enz_tower(enz_emb)
        return (r * e).sum(dim=-1)


# ------------------------------------------------------------------ dataset


class DualTowerDataset(Dataset):
    """Yields (drfp_vector, esm_embedding) pairs.

    Parameters
    ----------
    rxn_smiles : list[str]
        Reaction SMILES for each sample.
    enz_embeddings : np.ndarray
        (N, 1280) pre-computed ESM-2 embeddings.
    n_bits : int
        DRFP fingerprint length.
    """

    def __init__(
        self,
        rxn_smiles: list[str],
        enz_embeddings: np.ndarray,
        n_bits: int = 2048,
    ):
        assert len(rxn_smiles) == len(enz_embeddings)
        self.rxn_smiles = rxn_smiles
        self.enz_embeddings = enz_embeddings
        self.n_bits = n_bits
        # cache DRFP computation (shared with featurize_v2 module-level cache)
        self._fp_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.rxn_smiles)

    def _get_drfp(self, rxn: str) -> np.ndarray:
        if rxn not in self._fp_cache:
            self._fp_cache[rxn] = drfp_one(rxn, n_bits=self.n_bits)
        return self._fp_cache[rxn]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fp = self._get_drfp(self.rxn_smiles[idx])
        emb = self.enz_embeddings[idx]
        return (
            torch.tensor(fp, dtype=torch.float32),
            torch.tensor(emb, dtype=torch.float32),
        )


# ------------------------------------------------------------------ enzyme bank


def build_enzyme_bank(
    esm_cache_path: str | Path,
    metadata_path: str | Path | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Load all cached ESM-2 embeddings into a matrix for ANN search.

    Returns
    -------
    embeddings : np.ndarray, shape (N, 1280)
    metadata   : list[dict] with keys 'seq_hash', 'uniprot_id', 'dim'
    """
    cache_dir = Path(esm_cache_path)
    index_path = cache_dir / "cache_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"No cache_index.json in {cache_dir}")

    with open(index_path) as f:
        index = json.load(f)

    embeddings = []
    metadata = []
    for seq_hash, entry in index.items():
        npy_path = cache_dir / entry["path"]
        if not npy_path.exists():
            continue
        emb = np.load(npy_path).astype(np.float32)
        embeddings.append(emb)
        metadata.append({
            "seq_hash": seq_hash,
            "uniprot_id": entry.get("uniprot_id"),
            "dim": int(emb.shape[0]),
        })

    if not embeddings:
        raise ValueError(f"No embeddings found in {cache_dir}")

    return np.stack(embeddings), metadata


# ------------------------------------------------------------------ evaluation


def evaluate(
    model: DualTowerModel,
    val_loader: DataLoader,
    enzyme_bank: np.ndarray,
    enzyme_meta: list[dict],
    *,
    ks: tuple[int, ...] = (1, 5, 10, 50),
    device: str = DEVICE,
) -> dict:
    """Compute Recall@K for each reaction in the validation set.

    For each reaction we embed it, compute cosine similarity against the
    full enzyme bank, and check whether the correct enzyme appears in the
    top-K results.

    Returns dict with recall_at_1, recall_at_5, etc. plus lift vs random.
    """
    model.eval()
    bank_size = len(enzyme_bank)

    # Pre-embed the full enzyme bank through the enzyme tower
    bank_t = torch.tensor(enzyme_bank, dtype=torch.float32, device=device)
    with torch.no_grad():
        bank_embed = model.enz_tower(bank_t)  # (N_bank, embed_dim)

    hits = {k: 0 for k in ks}
    total = 0

    with torch.no_grad():
        for rxn_fp, enz_emb in val_loader:
            rxn_fp = rxn_fp.to(device)
            enz_emb = enz_emb.to(device)
            rxn_embed = model.rxn_tower(rxn_fp)  # (B, embed_dim)

            # Ground-truth enzyme embedding through the tower
            gt_embed = model.enz_tower(enz_emb)  # (B, embed_dim)

            # Cosine similarity against the bank
            sims = rxn_embed @ bank_embed.T  # (B, N_bank)

            # For each sample, find the bank entry closest to the ground-truth
            # enzyme (by raw ESM-2 cosine sim) — this is the "correct" index
            for i in range(rxn_fp.size(0)):
                gt_sim = gt_embed[i] @ bank_embed.T  # (N_bank,)
                correct_idx = gt_sim.argmax().item()

                # Rank of the correct enzyme in the reaction's similarity list
                rank = (sims[i] > sims[i, correct_idx]).sum().item() + 1
                for k in ks:
                    if rank <= k:
                        hits[k] += 1
                total += 1

    results = {}
    for k in ks:
        recall = hits[k] / max(total, 1)
        random_baseline = min(k, bank_size) / max(bank_size, 1)
        lift = recall / random_baseline if random_baseline > 0 else float("inf")
        results[f"recall_at_{k}"] = recall
        results[f"random_at_{k}"] = random_baseline
        results[f"lift_at_{k}"] = lift
    results["n_eval"] = total
    results["bank_size"] = bank_size
    return results


# ------------------------------------------------------------------ training


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_dual_tower(
    rxn_smiles_train: list[str],
    enz_emb_train: np.ndarray,
    groups_train: np.ndarray,
    *,
    rxn_smiles_pretrain: list[str] | None = None,
    enz_emb_pretrain: np.ndarray | None = None,
    enzyme_bank: np.ndarray | None = None,
    enzyme_meta: list[dict] | None = None,
    rxn_dim: int = 2048,
    enz_dim: int = 1280,
    hidden: int = 512,
    embed_dim: int = 256,
    temperature: float = 0.07,
    epochs: int = 50,
    pretrain_epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    n_folds: int = 5,
    seed: int = 42,
    tag: str = "v1",
    eval_only: bool = False,
    checkpoint_path: str | None = None,
) -> dict:
    """Full training pipeline with DOI-grouped K-fold CV.

    Returns summary dict with per-fold and aggregate metrics.
    """
    _set_seed(seed)
    out_dir = results_dir() / "dual_tower"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_groups = len(set(groups_train))
    n_folds = min(n_folds, n_groups)
    print(f"[dual_tower] samples={len(rxn_smiles_train)}  "
          f"DOI groups={n_groups}  folds={n_folds}")

    all_fold_results = []

    for fold, (tr_idx, te_idx) in enumerate(
        GroupKFold(n_splits=n_folds).split(
            np.arange(len(rxn_smiles_train)), groups=groups_train
        )
    ):
        print(f"\n{'='*60}")
        print(f"[fold {fold}]  train={len(tr_idx)}  val={len(te_idx)}")
        _set_seed(seed + fold)

        model = DualTowerModel(
            rxn_dim=rxn_dim, enz_dim=enz_dim, hidden=hidden,
            embed_dim=embed_dim, temperature=temperature,
        ).to(DEVICE)

        if eval_only and checkpoint_path:
            ckpt = Path(checkpoint_path)
            if ckpt.exists():
                model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
                print(f"  loaded checkpoint: {ckpt}")
        else:
            # --- optional pre-training on ReactZyme pairs ---
            if rxn_smiles_pretrain is not None and enz_emb_pretrain is not None:
                print(f"  [pretrain] ReactZyme pairs: {len(rxn_smiles_pretrain)}")
                _train_loop(
                    model,
                    rxn_smiles_pretrain, enz_emb_pretrain,
                    epochs=pretrain_epochs, batch_size=batch_size, lr=lr,
                    phase="pretrain",
                )

            # --- fine-tune on AutoPlanner enzymatic steps ---
            rxn_tr = [rxn_smiles_train[i] for i in tr_idx]
            emb_tr = enz_emb_train[tr_idx]
            print(f"  [finetune] AutoPlanner train: {len(rxn_tr)}")
            _train_loop(
                model, rxn_tr, emb_tr,
                epochs=epochs, batch_size=batch_size, lr=lr,
                phase="finetune",
            )

            # save checkpoint
            ckpt_path = out_dir / f"dual_tower_{tag}_fold{fold}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  saved checkpoint -> {ckpt_path}")

        # --- evaluate ---
        rxn_te = [rxn_smiles_train[i] for i in te_idx]
        emb_te = enz_emb_train[te_idx]
        val_ds = DualTowerDataset(rxn_te, emb_te)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        if enzyme_bank is not None and enzyme_meta is not None:
            metrics = evaluate(
                model, val_loader, enzyme_bank, enzyme_meta, device=DEVICE,
            )
        else:
            # fallback: use training enzymes as the bank
            bank = enz_emb_train[tr_idx]
            bank_meta = [{"idx": int(i)} for i in tr_idx]
            metrics = evaluate(
                model, val_loader, bank, bank_meta, device=DEVICE,
            )

        metrics["fold"] = fold
        metrics["n_train"] = len(tr_idx)
        metrics["n_val"] = len(te_idx)
        all_fold_results.append(metrics)

        # print fold results
        for k in (1, 5, 10, 50):
            rk = metrics.get(f"recall_at_{k}", 0)
            bk = metrics.get(f"random_at_{k}", 0)
            lk = metrics.get(f"lift_at_{k}", 0)
            print(f"  Recall@{k:<2d}  model={rk:.3f}  baseline={bk:.4f}  lift={lk:.1f}")

    # --- aggregate ---
    print(f"\n{'='*60}")
    print("[dual_tower] aggregate results")
    summary = {"tag": tag, "n_folds": n_folds, "seed": seed}
    for k in (1, 5, 10, 50):
        vals = [r[f"recall_at_{k}"] for r in all_fold_results]
        baselines = [r[f"random_at_{k}"] for r in all_fold_results]
        lifts = [r[f"lift_at_{k}"] for r in all_fold_results]
        mean_r = np.mean(vals)
        mean_b = np.mean(baselines)
        mean_l = np.mean(lifts)
        summary[f"recall_at_{k}"] = float(mean_r)
        summary[f"random_at_{k}"] = float(mean_b)
        summary[f"lift_at_{k}"] = float(mean_l)
        print(f"  Recall@{k:<2d}  model={mean_r:.3f}  baseline={mean_b:.4f}  lift={mean_l:.1f}")

    # save summary
    import pandas as pd
    pd.DataFrame(all_fold_results).to_csv(
        out_dir / f"dual_tower_{tag}_folds.csv", index=False,
    )
    pd.DataFrame([summary]).to_csv(
        out_dir / f"dual_tower_{tag}_summary.csv", index=False,
    )
    print(f"\n[save] {out_dir / f'dual_tower_{tag}_summary.csv'}")
    return summary


def _train_loop(
    model: DualTowerModel,
    rxn_smiles: list[str],
    enz_embeddings: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    phase: str = "train",
) -> None:
    """Inner training loop (shared by pre-train and fine-tune)."""
    ds = DualTowerDataset(rxn_smiles, enz_embeddings)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for rxn_fp, enz_emb in loader:
            rxn_fp = rxn_fp.to(DEVICE)
            enz_emb = enz_emb.to(DEVICE)

            rxn_embed, enz_embed = model(rxn_fp, enz_emb)
            loss = model.info_nce_loss(rxn_embed, enz_embed)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        if (ep + 1) % 5 == 0 or ep == 0:
            cur_lr = scheduler.get_last_lr()[0]
            print(f"    [{phase}] epoch {ep+1:3d}/{epochs}  "
                  f"loss={avg_loss:.4f}  lr={cur_lr:.2e}")


# ------------------------------------------------------------------ data prep


def _load_esm_cache(cache_dir: str | Path) -> dict[str, np.ndarray]:
    """Load ESM-2 cache index and return {uniprot_id: embedding}."""
    cache_dir = Path(cache_dir)
    index_path = cache_dir / "cache_index.json"
    if not index_path.exists():
        return {}
    with open(index_path) as f:
        index = json.load(f)

    mapping: dict[str, np.ndarray] = {}
    for seq_hash, entry in index.items():
        uid = entry.get("uniprot_id")
        if not uid:
            continue
        npy_path = cache_dir / entry["path"]
        if npy_path.exists():
            mapping[uid] = np.load(npy_path).astype(np.float32)
    return mapping


def _load_reactzyme_pairs(
    pairs_path: str | Path,
    esm_lookup: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray]:
    """Load ReactZyme (rxn_smiles, uniprot_id) pairs and resolve ESM-2 embeddings.

    Returns (rxn_smiles_list, enz_embeddings) for pairs where the embedding exists.
    """
    import pandas as pd

    df = pd.read_csv(pairs_path, sep="\t")
    rxns = []
    embs = []
    for _, row in df.iterrows():
        uid = str(row.get("uniprot_id", "")).strip()
        rxn = str(row.get("reaction_smiles", row.get("rxn_smiles", ""))).strip()
        if not rxn or ">>" not in rxn or uid not in esm_lookup:
            continue
        rxns.append(rxn)
        embs.append(esm_lookup[uid])

    if not embs:
        return [], np.empty((0, 1280), dtype=np.float32)
    return rxns, np.stack(embs)


def _prepare_autoplanner_data(
    data_path: str | Path,
    esm_lookup: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Extract (rxn_smiles, enz_embedding, doi_group) from AutoPlanner enzymatic steps.

    Resolves EC number -> UniProt ID -> ESM-2 embedding.  Steps without a
    resolvable embedding are skipped.

    Returns (rxn_smiles, enz_embeddings, groups) arrays.
    """
    steps, _, _ = load_v2(data_path)
    enz_steps = [s for s in steps if s.ec_number and s.rxn_smiles]
    print(f"  enzymatic steps with EC: {len(enz_steps)}")

    # Try to match by EC number or by any available UniProt ID in the cache.
    # The ESM cache is keyed by uniprot_id; we also try ec_number as a
    # fallback key (some datasets store EC as the lookup key).
    rxns = []
    embs = []
    dois = []
    n_matched = 0
    for s in enz_steps:
        emb = None
        # Try direct uniprot_id lookup if the step carries one
        # (StepRowV2 doesn't have uniprot_id, but the EC number may be
        # used as a proxy key in some cache setups)
        ec = s.ec_number
        # Try EC as key (some caches index by EC)
        if ec in esm_lookup:
            emb = esm_lookup[ec]
        else:
            # Try partial EC matches — find any cached enzyme with this EC
            # This is a heuristic; in production we'd have a proper EC->UniProt map
            for uid, e in esm_lookup.items():
                if uid.startswith(ec):
                    emb = e
                    break

        if emb is None:
            # Last resort: pick any embedding (for development/testing only)
            continue

        rxns.append(s.rxn_smiles)
        embs.append(emb)
        dois.append(s.doi)
        n_matched += 1

    print(f"  matched to ESM-2 embeddings: {n_matched}/{len(enz_steps)}")
    if not embs:
        return [], np.empty((0, 1280), dtype=np.float32), np.array([])
    return rxns, np.stack(embs), np.array(dois)


# ------------------------------------------------------------------ CLI


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train dual-tower contrastive model for enzyme-reaction matching"
    )
    ap.add_argument("--data", default="cascade_dataset_v2.normalized.json",
                    help="AutoPlanner dataset JSON")
    ap.add_argument("--esm-cache", default="results/shared/esm_cache/",
                    help="Directory with pre-computed ESM-2 embeddings")
    ap.add_argument("--reactzyme-pairs", default=None,
                    help="Optional ReactZyme pairs TSV for pre-training")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--pretrain-epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--eval", action="store_true", dest="eval_only",
                    help="Evaluation-only mode (load checkpoint)")
    ap.add_argument("--checkpoint", default=None,
                    help="Path to model checkpoint (for --eval mode)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    )

    t0 = time.time()

    # 1) Load ESM-2 cache
    print(f"[load] ESM-2 cache: {args.esm_cache}")
    esm_lookup = _load_esm_cache(args.esm_cache)
    print(f"  cached enzymes: {len(esm_lookup)}")

    # 2) Prepare AutoPlanner data
    print(f"[load] AutoPlanner data: {args.data}")
    rxn_smiles, enz_emb, groups = _prepare_autoplanner_data(args.data, esm_lookup)
    if len(rxn_smiles) == 0:
        print("[abort] no matched reaction-enzyme pairs; "
              "run esm_embedder.py first to populate the cache")
        return

    # 3) Optional ReactZyme pre-training data
    rxn_pre, emb_pre = None, None
    if args.reactzyme_pairs and Path(args.reactzyme_pairs).exists():
        print(f"[load] ReactZyme pairs: {args.reactzyme_pairs}")
        rxn_pre, emb_pre = _load_reactzyme_pairs(args.reactzyme_pairs, esm_lookup)
        print(f"  matched ReactZyme pairs: {len(rxn_pre)}")
        if len(rxn_pre) == 0:
            rxn_pre, emb_pre = None, None

    # 4) Build enzyme bank for evaluation
    print("[load] building enzyme bank ...")
    try:
        enzyme_bank, enzyme_meta = build_enzyme_bank(args.esm_cache)
        print(f"  enzyme bank size: {len(enzyme_meta)}")
    except (FileNotFoundError, ValueError) as e:
        print(f"  enzyme bank unavailable ({e}); using train split as bank")
        enzyme_bank, enzyme_meta = None, None

    # Infer ESM-2 embedding dimension from data
    enz_dim = enz_emb.shape[1] if len(enz_emb) > 0 else 1280

    # 5) Train
    summary = train_dual_tower(
        rxn_smiles_train=rxn_smiles,
        enz_emb_train=enz_emb,
        groups_train=groups,
        rxn_smiles_pretrain=rxn_pre,
        enz_emb_pretrain=emb_pre,
        enzyme_bank=enzyme_bank,
        enzyme_meta=enzyme_meta,
        rxn_dim=2048,
        enz_dim=enz_dim,
        hidden=args.hidden,
        embed_dim=args.embed_dim,
        temperature=args.temperature,
        epochs=args.epochs,
        pretrain_epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_folds=args.folds,
        seed=args.seed,
        tag=args.tag,
        eval_only=args.eval_only,
        checkpoint_path=args.checkpoint,
    )

    elapsed = time.time() - t0
    print(f"\n[done] {elapsed:.0f}s total")


if __name__ == "__main__":
    main()
