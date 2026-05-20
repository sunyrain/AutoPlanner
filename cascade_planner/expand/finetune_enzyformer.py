"""Fine-tune Enzyformer (Chemformer BART) for enzymatic retrosynthesis.

Pure PyTorch — no PL dependency. Loads the USPTO-pretrained checkpoint
and fine-tunes on enzymatic reactions (product → reactants, EC-conditioned).

Usage:
  python -m cascade_planner.expand.finetune_enzyformer \
    --pretrained data_external/enzyformer/Retrosynthesis/ckpt/last_USPTO_FULL.ckpt \
    --output results/shared/enzyformer_retro_finetuned.pt \
    --epochs 50 --lr 3e-4 --batch 32
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).resolve().parent.parent.parent

# ── Tokenizer ──────────────────────────────────────────────────────────

VOCAB_PATH = ROOT / "data_external" / "enzyformer" / "Chemformer" / "bart_vocab_downstream.json"
SMILES_RE = re.compile(
    r"(\[[^\]]+\]|Br|Cl|Si|Se|se|@@|@|>>|[A-Z][a-z]?|[a-z]|[0-9]|[^A-Za-z0-9\[\]])"
)

def _load_vocab():
    with open(VOCAB_PATH) as f:
        data = json.load(f)
    vlist = data.get("vocabulary", data)
    if isinstance(vlist, dict):
        vlist = vlist["vocabulary"]
    t2i = {t: i for i, t in enumerate(vlist)}
    i2t = {i: t for t, i in t2i.items()}
    return t2i, i2t

TOKEN2ID, ID2TOKEN = _load_vocab()
PAD, UNK, BOS, EOS = TOKEN2ID["<PAD>"], TOKEN2ID["?"], TOKEN2ID["^"], TOKEN2ID["&"]
VOCAB_SIZE = len(TOKEN2ID)


def tokenize(smi: str) -> list[int]:
    tokens = ["^"] + SMILES_RE.findall(smi) + ["&"]
    return [TOKEN2ID.get(t, UNK) for t in tokens]


def detokenize(ids: list[int]) -> str:
    return "".join(ID2TOKEN.get(i, "?") for i in ids if i not in (PAD, BOS, EOS))


# ── SMILES augmentation (R-SMILES style) ──────────────────────────────

def augment_smiles(smi: str) -> str:
    """Random SMILES augmentation via RDKit."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    try:
        return Chem.MolToSmiles(mol, doRandom=True)
    except Exception:
        return smi


# ── Dataset ────────────────────────────────────────────────────────────

class EnzReactionDataset(Dataset):
    """Enzymatic reactions for backward prediction (product → reactants)."""

    def __init__(self, reactions: list[dict], max_len: int = 256, augment: bool = True):
        self.reactions = reactions
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.reactions)

    def __getitem__(self, idx):
        rxn = self.reactions[idx]
        product = rxn["product"]
        reactants = rxn["reactants"]
        ec = rxn.get("ec", "")

        if self.augment:
            product = augment_smiles(product)
            # Augment each reactant separately then rejoin
            parts = reactants.split(".")
            parts = [augment_smiles(p) for p in parts]
            reactants = ".".join(parts)

        # EC conditioning: prepend [EC:X.Y.Z.W] token to product
        if ec and ec != "None":
            ec_token = f"[{ec}]"
            src_smi = ec_token + product
        else:
            src_smi = product

        src_ids = tokenize(src_smi)[: self.max_len]
        tgt_ids = tokenize(reactants)[: self.max_len]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch):
    srcs, tgts = zip(*batch)
    src_max = max(s.shape[0] for s in srcs)
    tgt_max = max(t.shape[0] for t in tgts)
    src_padded = torch.full((len(srcs), src_max), PAD, dtype=torch.long)
    tgt_padded = torch.full((len(tgts), tgt_max), PAD, dtype=torch.long)
    src_mask = torch.ones(len(srcs), src_max, dtype=torch.bool)
    tgt_mask = torch.ones(len(tgts), tgt_max, dtype=torch.bool)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_padded[i, : s.shape[0]] = s
        src_mask[i, : s.shape[0]] = False
        tgt_padded[i, : t.shape[0]] = t
        tgt_mask[i, : t.shape[0]] = False
    # Transpose to (seq_len, batch) for nn.Transformer
    return src_padded.T, src_mask, tgt_padded.T, tgt_mask


# ── Model ──────────────────────────────────────────────────────────────

class EnzyformerBART(nn.Module):
    def __init__(self, vocab_size=523, d_model=512, n_heads=8, n_layers=6,
                 d_ff=2048, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, d_model)
        self.register_buffer("pos_emb", self._sinusoidal_pos(max_seq_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, activation="gelu", batch_first=False
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers, norm=nn.LayerNorm(d_model))
        dec_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_ff, dropout, activation="gelu", batch_first=False
        )
        self.decoder = nn.TransformerDecoder(dec_layer, n_layers, norm=nn.LayerNorm(d_model))
        self.token_fc = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _sinusoidal_pos(max_len, d_model):
        encs = torch.tensor([dim / d_model for dim in range(0, d_model, 2)])
        encs = 10000 ** encs
        rows = []
        for pos in range(max_len):
            s = torch.sin(torch.tensor(float(pos)) / encs)
            c = torch.cos(torch.tensor(float(pos)) / encs)
            rows.append(torch.stack([s, c], dim=1).flatten()[:d_model])
        return torch.stack(rows)

    def _embed(self, ids):
        seq_len = ids.shape[0]
        return self.emb(ids) * math.sqrt(self.d_model) + self.pos_emb[:seq_len].unsqueeze(1)

    def forward(self, src_ids, tgt_ids, src_pad_mask=None, tgt_pad_mask=None):
        """Teacher-forced forward pass. Returns logits (tgt_len, batch, vocab)."""
        memory = self.encoder(self._embed(src_ids), src_key_padding_mask=src_pad_mask)
        tgt_len = tgt_ids.shape[0]
        causal = torch.triu(torch.ones(tgt_len, tgt_len, device=src_ids.device) * float("-inf"), diagonal=1)
        out = self.decoder(
            self._embed(tgt_ids), memory,
            tgt_mask=causal, memory_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask,
        )
        return self.token_fc(out)

    @torch.no_grad()
    def beam_search(self, src_ids, num_beams=10, max_len=200):
        """Beam search decoding. src_ids: (seq_len, 1)."""
        memory = self.encoder(self._embed(src_ids))
        memory = memory.expand(-1, num_beams, -1).contiguous()
        dec = torch.full((1, num_beams), BOS, dtype=torch.long, device=src_ids.device)
        scores = torch.zeros(num_beams, device=src_ids.device)
        scores[1:] = -1e9
        finished = []
        for _ in range(max_len):
            tgt_len = dec.shape[0]
            causal = torch.triu(torch.ones(tgt_len, tgt_len, device=dec.device) * float("-inf"), diagonal=1)
            out = self.decoder(self._embed(dec), memory, tgt_mask=causal)
            logits = self.token_fc(out[-1])
            lp = F.log_softmax(logits, dim=-1)
            cand = scores.unsqueeze(1) + lp
            flat = cand.view(-1)
            topk_s, topk_i = flat.topk(num_beams * 2)
            new_dec, new_sc = [], []
            for rank in range(topk_i.shape[0]):
                bi = (topk_i[rank] // self.vocab_size).item()
                ti = (topk_i[rank] % self.vocab_size).item()
                sc = topk_s[rank].item()
                seq = dec[:, bi].tolist() + [ti]
                if ti == EOS:
                    finished.append((sc / len(seq), seq[1:]))
                else:
                    new_dec.append(seq)
                    new_sc.append(sc)
                if len(new_dec) >= num_beams:
                    break
            if len(finished) >= num_beams or not new_dec:
                break
            ml = max(len(s) for s in new_dec)
            padded = [s + [PAD] * (ml - len(s)) for s in new_dec[:num_beams]]
            dec = torch.tensor(padded, dtype=torch.long, device=src_ids.device).T
            scores = torch.tensor(new_sc[:num_beams], device=src_ids.device)
            memory = memory[:, :dec.shape[1], :]
        finished.sort(key=lambda x: -x[0])
        return finished[:num_beams]


# ── Data loading ───────────────────────────────────────────────────────

def load_enzymatic_reactions() -> list[dict]:
    """Load enzymatic reactions from cascade_dataset_v3.json."""
    data_path = ROOT / "cascade_dataset_v3.json"
    data = json.loads(data_path.read_text())
    records = data if isinstance(data, list) else data.get("records_kept", data.get("records", []))
    rxns = []
    for rec in records:
        cascades = rec.get("cascades", [rec])
        for cas in cascades:
            for step in cas.get("steps", []):
                rxn_smi = step.get("rxn_smiles")
                cats = step.get("catalyst_components", [])
                ec = cats[0].get("ec_number", "") if cats else ""
                if rxn_smi and ec and ec != "None" and ">>" in rxn_smi:
                    parts = rxn_smi.split(">>")
                    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                        rxns.append({"reactants": parts[0].strip(), "product": parts[1].strip(), "ec": ec})
    return rxns


def load_pretrained_weights(model: EnzyformerBART, ckpt_path: str):
    """Load pretrained Chemformer weights into our standalone model."""
    import sys
    sys.path.insert(0, str(ROOT / "data_external" / "enzyformer" / "Chemformer"))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded pretrained: missing={len(missing)}, unexpected={len(unexpected)}")
    return model


# ── Training ───────────────────────────────────────────────────────────

def train(args):
    print("Loading enzymatic reactions...")
    rxns = load_enzymatic_reactions()
    random.shuffle(rxns)
    n_val = max(50, len(rxns) // 10)
    train_rxns, val_rxns = rxns[n_val:], rxns[:n_val]
    print(f"Train: {len(train_rxns)}, Val: {len(val_rxns)}")

    train_ds = EnzReactionDataset(train_rxns, augment=True)
    val_ds = EnzReactionDataset(val_rxns, augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, num_workers=0)

    print("Building model...")
    model = EnzyformerBART(
        vocab_size=VOCAB_SIZE, d_model=512, n_heads=8, n_layers=6,
        d_ff=2048, max_seq_len=512, dropout=args.dropout,
    ).to(DEVICE)

    if args.pretrained:
        load_pretrained_weights(model, args.pretrained)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {params:.1f}M params, device={DEVICE}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_tokens = 0.0, 0
        for src, src_mask, tgt, tgt_mask in train_dl:
            src, src_mask = src.to(DEVICE), src_mask.to(DEVICE)
            tgt, tgt_mask = tgt.to(DEVICE), tgt_mask.to(DEVICE)
            # Teacher forcing: input = tgt[:-1], target = tgt[1:]
            logits = model(src, tgt[:-1], src_pad_mask=src_mask, tgt_pad_mask=tgt_mask[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), tgt[1:].reshape(-1),
                ignore_index=PAD, label_smoothing=0.1,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * tgt[1:].ne(PAD).sum().item()
            n_tokens += tgt[1:].ne(PAD).sum().item()
        scheduler.step()
        train_loss = total_loss / max(n_tokens, 1)

        # Validation
        model.eval()
        val_loss_sum, val_tokens = 0.0, 0
        with torch.no_grad():
            for src, src_mask, tgt, tgt_mask in val_dl:
                src, src_mask = src.to(DEVICE), src_mask.to(DEVICE)
                tgt, tgt_mask = tgt.to(DEVICE), tgt_mask.to(DEVICE)
                logits = model(src, tgt[:-1], src_pad_mask=src_mask, tgt_pad_mask=tgt_mask[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE), tgt[1:].reshape(-1),
                    ignore_index=PAD,
                )
                val_loss_sum += loss.item() * tgt[1:].ne(PAD).sum().item()
                val_tokens += tgt[1:].ne(PAD).sum().item()
        val_loss = val_loss_sum / max(val_tokens, 1)

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output)
            improved = " ★"

        if epoch % 5 == 0 or epoch == 1 or improved:
            print(f"  Epoch {epoch:3d}: train={train_loss:.4f} val={val_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}{improved}")

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Saved to: {args.output}")

    # Quick eval: beam search on 5 val examples
    model.load_state_dict(torch.load(args.output, map_location=DEVICE, weights_only=True))
    model.eval()
    print("\nSample predictions:")
    for rxn in val_rxns[:5]:
        ec_tok = f"[{rxn['ec']}]" if rxn["ec"] else ""
        src = torch.tensor([tokenize(ec_tok + rxn["product"])], dtype=torch.long).T.to(DEVICE)
        results = model.beam_search(src, num_beams=5, max_len=150)
        pred = detokenize(results[0][1]) if results else "EMPTY"
        gt = rxn["reactants"]
        match = "✓" if pred == gt else "✗"
        print(f"  {match} EC={rxn['ec'][:7]:7s} GT={gt[:40]:40s} Pred={pred[:40]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", default=str(ROOT / "data_external/enzyformer/Retrosynthesis/ckpt/last_USPTO_FULL.ckpt"))
    parser.add_argument("--output", default=str(ROOT / "results/shared/enzyformer_retro_finetuned.pt"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()
    train(args)
