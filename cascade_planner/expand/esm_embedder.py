"""Extract protein sequence embeddings from ESM-2 and cache to disk.

Provides batched inference with OOM recovery, MD5-keyed disk caching,
and a CLI for bulk embedding from TSV files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = "/root/autodl-tmp/AutoPlanner/results/shared/esm_cache/"
ESM2_MAX_TOKENS = 1022  # 1024 minus <cls> and <eos>


def _seq_hash(sequence: str) -> str:
    return hashlib.md5(sequence.encode()).hexdigest()


class ESMEmbedder:
    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: str = "cuda:0",
        cache_dir: str | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / "cache_index.json"
        self._index: dict[str, dict] = self._load_index()

        logger.info("Loading ESM-2 model %s on %s", model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_index(self) -> dict[str, dict]:
        if self.index_path.exists():
            with open(self.index_path) as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        with open(self.index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def _get_cached(self, seq_hash: str) -> np.ndarray | None:
        entry = self._index.get(seq_hash)
        if entry is None:
            return None
        p = self.cache_dir / entry["path"]
        if p.exists():
            return np.load(p).astype(np.float32)
        return None

    def _put_cache(
        self, seq_hash: str, embedding: np.ndarray, uniprot_id: str | None = None
    ) -> None:
        fname = f"{seq_hash}.npy"
        np.save(self.cache_dir / fname, embedding.astype(np.float16))
        self._index[seq_hash] = {
            "uniprot_id": uniprot_id,
            "dim": int(embedding.shape[0]),
            "path": fname,
        }

    def load_cache(self) -> dict[str, np.ndarray]:
        """Bulk-load all cached embeddings. Returns {seq_hash: embedding}."""
        out: dict[str, np.ndarray] = {}
        for h, entry in self._index.items():
            p = self.cache_dir / entry["path"]
            if p.exists():
                out[h] = np.load(p).astype(np.float32)
        return out

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed_sequence(self, sequence: str) -> np.ndarray:
        """Embed a single sequence. Returns (D,) float32 array."""
        h = _seq_hash(sequence)
        cached = self._get_cached(h)
        if cached is not None:
            return cached

        tokens = self.tokenizer(
            sequence[:ESM2_MAX_TOKENS],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=ESM2_MAX_TOKENS + 2,
        ).to(self.device)

        out = self.model(**tokens)
        # Exclude <cls> (0) and <eos> (-1)
        hidden = out.last_hidden_state[0, 1:-1, :]
        emb = hidden.mean(dim=0).cpu().numpy().astype(np.float32)

        self._put_cache(h, emb)
        self._save_index()
        return emb

    @torch.no_grad()
    def embed_batch(
        self, sequences: list[str], batch_size: int = 8
    ) -> np.ndarray:
        """Embed a list of sequences. Returns (N, D) float32 array."""
        results: list[np.ndarray] = [None] * len(sequences)  # type: ignore[list-item]
        to_compute: list[tuple[int, str]] = []

        for i, seq in enumerate(sequences):
            h = _seq_hash(seq)
            cached = self._get_cached(h)
            if cached is not None:
                results[i] = cached
            else:
                to_compute.append((i, seq[:ESM2_MAX_TOKENS]))

        if to_compute:
            self._run_batches(to_compute, results, batch_size)
            self._save_index()

        return np.stack(results)

    def _run_batches(
        self,
        items: list[tuple[int, str]],
        results: list[np.ndarray],
        batch_size: int,
    ) -> None:
        idx = 0
        pbar = tqdm(total=len(items), desc="ESM-2 embedding")
        while idx < len(items):
            batch = items[idx : idx + batch_size]
            try:
                self._forward_batch(batch, results)
                pbar.update(len(batch))
                idx += batch_size
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch_size <= 1:
                    logger.error(
                        "OOM even with batch_size=1 at sequence index %d, skipping",
                        batch[0][0],
                    )
                    # Store zeros so we don't crash downstream
                    for orig_idx, seq in batch:
                        dim = self.model.config.hidden_size
                        results[orig_idx] = np.zeros(dim, dtype=np.float32)
                    pbar.update(len(batch))
                    idx += 1
                else:
                    batch_size = max(1, batch_size // 2)
                    logger.warning("OOM — reducing batch size to %d", batch_size)
        pbar.close()

    @torch.no_grad()
    def _forward_batch(
        self,
        batch: list[tuple[int, str]],
        results: list[np.ndarray],
    ) -> None:
        seqs = [s for _, s in batch]
        tokens = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=ESM2_MAX_TOKENS + 2,
        ).to(self.device)

        out = self.model(**tokens)
        hidden = out.last_hidden_state  # (B, L, D)
        mask = tokens["attention_mask"]  # (B, L)

        # Zero out special tokens (<cls>=pos0, <eos>=last non-pad)
        mask = mask.clone()
        mask[:, 0] = 0
        for i in range(mask.size(0)):
            last = mask[i].sum().item()  # includes cls which we just zeroed
            # <eos> sits right after the real tokens
            eos_pos = int(last)
            if eos_pos < mask.size(1):
                mask[i, eos_pos] = 0

        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / lengths  # (B, D)
        pooled_np = pooled.cpu().numpy().astype(np.float32)

        for (orig_idx, seq), emb in zip(batch, pooled_np):
            results[orig_idx] = emb
            self._put_cache(_seq_hash(seq), emb)

    # ------------------------------------------------------------------
    # File-based embedding
    # ------------------------------------------------------------------

    def embed_from_file(
        self,
        tsv_path: str,
        id_col: str = "uniprot_id",
        seq_col: str = "sequence",
    ) -> dict[str, np.ndarray]:
        """Read TSV, embed all sequences, return {id: embedding}."""
        import pandas as pd

        df = pd.read_csv(tsv_path, sep="\t")
        ids = df[id_col].tolist()
        seqs = df[seq_col].tolist()

        embeddings = self.embed_batch(seqs)

        mapping: dict[str, np.ndarray] = {}
        for uid, seq, emb in zip(ids, seqs, embeddings):
            mapping[uid] = emb
            h = _seq_hash(seq[:ESM2_MAX_TOKENS])
            if h in self._index:
                self._index[h]["uniprot_id"] = uid
        self._save_index()
        return mapping


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="ESM-2 sequence embedder")
    parser.add_argument("--input", required=True, help="Input TSV file")
    parser.add_argument(
        "--model",
        default="facebook/esm2_t33_650M_UR50D",
        help="HuggingFace model name",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="embeddings.npz", help="Output .npz path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    embedder = ESMEmbedder(
        model_name=args.model, device=args.device
    )
    mapping = embedder.embed_from_file(args.input)

    np.savez_compressed(args.output, **{k: v for k, v in mapping.items()})
    logger.info("Saved %d embeddings to %s", len(mapping), args.output)


if __name__ == "__main__":
    main()
