"""Learned Route Scorer for CascadeBoard++.

4-layer Transformer, d=128, ~2M params.
Multi-task: route_score + compat + opmode + issues + yield + ee + conversion.
Trained on v3 3,810 routes with pairwise ranking loss.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# Reuse vocabularies from skeleton_inpainter
from cascade_planner.cascadeboard.skeleton_inpainter import (
    RTYPE_TO_ID, NUM_RTYPES, NUM_EC1, EC2_TO_ID, NUM_EC2,
    STEP_ROLE_TO_ID, STEP_TYPE_TO_ID, STEP_MODE_TO_ID,
    ATMO_TO_ID, CATCLASS_TO_ID, ENG_TO_ID, BIOFMT_TO_ID,
    COFMODE_TO_ID, COMPAT_TO_ID, COMPAT_VOCAB, OPMODE_TO_ID, OPMODE_VOCAB,
    ISSUE_TYPE_VOCAB, NUM_ISSUES, DOMAIN_TO_ID,
    EVIDENCE_WEIGHT, morgan_fp, _normalize_atmosphere, _normalize_cofactor_mode,
    _extract_slot_features, SlotFeatures,
)

MAX_SLOTS = 8


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScorerSample:
    """A single training sample for the route scorer."""
    target_fp: np.ndarray           # (2048,)
    domain_id: int
    n_steps: int
    slots: list[SlotFeatures]
    # Per-step outcome labels
    yields: list[float]             # per-step yield (0-1), -1 = missing
    ees: list[float]                # per-step ee (0-1), -1 = missing
    # Global labels
    compat_label: int = -1
    opmode_label: int = -1
    issue_labels: list[int] = field(default_factory=list)
    evidence_weight: float = 1.0
    # Route quality score (derived from annotations)
    route_score: float = -1.0       # -1 = missing


def _compat_to_score(label: str) -> float:
    """Convert compatibility label to a 0-1 score."""
    mapping = {
        "empirically_compatible": 1.0,
        "compatible_with_mitigation": 0.7,
        "compatible_with_compromise": 0.4,
        "unclear": 0.2,
        "sequential_preferred": 0.3,
        "empirically_incompatible": 0.0,
    }
    return mapping.get(label, 0.5)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class StepEncoder(nn.Module):
    """Encode a single step into d_model."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.rtype_emb = nn.Embedding(NUM_RTYPES, 16)
        self.ec1_emb = nn.Embedding(NUM_EC1, 8)
        self.ec2_emb = nn.Embedding(NUM_EC2, 8)
        self.step_role_emb = nn.Embedding(len(STEP_ROLE_TO_ID), 8)
        self.step_type_emb = nn.Embedding(len(STEP_TYPE_TO_ID), 4)
        self.catclass_emb = nn.Embedding(len(CATCLASS_TO_ID), 4)
        self.engineering_emb = nn.Embedding(len(ENG_TO_ID), 4)
        self.biofmt_emb = nn.Embedding(len(BIOFMT_TO_ID), 4)
        self.cofmode_emb = nn.Embedding(len(COFMODE_TO_ID), 4)
        self.position_emb = nn.Embedding(MAX_SLOTS, 8)
        # Continuous: T, pH, substrate_conc, reaction_time
        self.cond_proj = nn.Linear(4, 8)
        # Total: 16+8+8+8+4+4+4+4+4+8+8 = 76
        self.proj = nn.Linear(76, d_model)

    def forward(
        self,
        rtype_ids: torch.Tensor,        # (B, S)
        ec1_ids: torch.Tensor,
        ec2_ids: torch.Tensor,
        step_role_ids: torch.Tensor,
        step_type_ids: torch.Tensor,
        catclass_ids: torch.Tensor,
        engineering_ids: torch.Tensor,
        biofmt_ids: torch.Tensor,
        cofmode_ids: torch.Tensor,
        positions: torch.Tensor,
        cond_feats: torch.Tensor,       # (B, S, 4)
    ) -> torch.Tensor:
        parts = [
            self.rtype_emb(rtype_ids),
            self.ec1_emb(ec1_ids),
            self.ec2_emb(ec2_ids),
            self.step_role_emb(step_role_ids),
            self.step_type_emb(step_type_ids),
            self.catclass_emb(catclass_ids),
            self.engineering_emb(engineering_ids),
            self.biofmt_emb(biofmt_ids),
            self.cofmode_emb(cofmode_ids),
            self.position_emb(positions),
            self.cond_proj(cond_feats),
        ]
        x = torch.cat(parts, dim=-1)  # (B, S, 76)
        return self.proj(x)


class PairEncoder(nn.Module):
    """Encode pairwise features between adjacent steps."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        # Pairwise features: |ΔT|, |ΔpH|, same_EC1, type_transition (one-hot would be huge, use embedding)
        self.proj = nn.Linear(4, d_model)

    def forward(self, step_embs: torch.Tensor, cond_feats: torch.Tensor, ec1_ids: torch.Tensor) -> torch.Tensor:
        """Compute pairwise features for adjacent steps.

        Returns (B, S-1, d_model) pairwise embeddings.
        """
        B, S, _ = cond_feats.shape
        if S < 2:
            return torch.zeros(B, 0, step_embs.shape[-1], device=step_embs.device)

        # ΔT, ΔpH between adjacent steps
        delta_T = torch.abs(cond_feats[:, 1:, 0] - cond_feats[:, :-1, 0])  # (B, S-1)
        delta_pH = torch.abs(cond_feats[:, 1:, 1] - cond_feats[:, :-1, 1])
        same_ec1 = (ec1_ids[:, 1:] == ec1_ids[:, :-1]).float()
        # Step distance (always 1 for adjacent, but could be useful)
        step_dist = torch.ones(B, S - 1, device=step_embs.device)

        pair_feats = torch.stack([delta_T, delta_pH, same_ec1, step_dist], dim=-1)  # (B, S-1, 4)
        return self.proj(pair_feats)


class LearnedRouteScorer(nn.Module):
    """4-layer Transformer scorer for complete routes.

    Input: step embeddings + pairwise embeddings + global context
    Output: route_score, compat, opmode, issues, per-step yield/ee
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_slots: int = MAX_SLOTS,
        dropout: float = 0.2,
        fp_dim: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_slots = max_slots

        # Encoders
        self.step_encoder = StepEncoder(d_model)
        self.pair_encoder = PairEncoder(d_model)

        # Global context
        self.target_fp_proj = nn.Linear(fp_dim, 32)
        self.domain_emb = nn.Embedding(len(DOMAIN_TO_ID), 16)
        self.nsteps_emb = nn.Embedding(max_slots + 1, 16)
        self.global_proj = nn.Linear(32 + 16 + 16, d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding
        self.pos_emb = nn.Embedding(max_slots * 2 + 2, d_model)  # steps + pairs + cls + global

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.emb_drop = nn.Dropout(dropout)

        # Output heads
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.compat_head = nn.Linear(d_model, len(COMPAT_VOCAB))
        self.opmode_head = nn.Linear(d_model, len(OPMODE_VOCAB))
        self.issue_head = nn.Linear(d_model, NUM_ISSUES)

        # Per-step heads (from step embeddings)
        self.yield_head = nn.Linear(d_model, 1)
        self.ee_head = nn.Linear(d_model, 1)

    def forward(
        self,
        step_inputs: dict[str, torch.Tensor],
        target_fp: torch.Tensor,        # (B, 2048)
        domain_ids: torch.Tensor,       # (B,)
        n_steps_ids: torch.Tensor,      # (B,)
    ) -> dict[str, torch.Tensor]:
        B = target_fp.shape[0]
        S = step_inputs["rtype_ids"].shape[1]

        # Encode steps
        step_embs = self.step_encoder(
            step_inputs["rtype_ids"],
            step_inputs["ec1_ids"],
            step_inputs["ec2_ids"],
            step_inputs["step_role_ids"],
            step_inputs["step_type_ids"],
            step_inputs["catclass_ids"],
            step_inputs["engineering_ids"],
            step_inputs["biofmt_ids"],
            step_inputs["cofmode_ids"],
            step_inputs["positions"],
            step_inputs["cond_feats"],
        )  # (B, S, d)

        # Encode pairwise
        pair_embs = self.pair_encoder(
            step_embs, step_inputs["cond_feats"], step_inputs["ec1_ids"]
        )  # (B, S-1, d)

        # Global context
        tfp = self.target_fp_proj(target_fp)
        dom = self.domain_emb(domain_ids)
        ns = self.nsteps_emb(n_steps_ids)
        global_emb = self.global_proj(torch.cat([tfp, dom, ns], dim=-1)).unsqueeze(1)  # (B, 1, d)

        # CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d)

        # Concatenate: [CLS, GLOBAL, STEP_0, ..., STEP_n, PAIR_0, ..., PAIR_n-1]
        tokens = torch.cat([cls, global_emb, step_embs, pair_embs], dim=1)  # (B, 2+S+(S-1), d)
        n_tokens = tokens.shape[1]

        # Positional encoding
        positions = torch.arange(n_tokens, device=tokens.device).unsqueeze(0).expand(B, -1)
        positions = positions.clamp(max=self.pos_emb.num_embeddings - 1)
        tokens = self.emb_drop(tokens + self.pos_emb(positions))

        # Transformer
        out = self.transformer(tokens)  # (B, n_tokens, d)

        # Extract outputs
        cls_out = out[:, 0, :]          # (B, d) — global representation
        step_out = out[:, 2:2 + S, :]   # (B, S, d) — step representations

        # Global heads
        route_score = torch.sigmoid(self.score_head(cls_out).squeeze(-1))  # (B,) in [0,1]
        compat_logits = self.compat_head(cls_out)       # (B, 4)
        opmode_logits = self.opmode_head(cls_out)       # (B, 7)
        issue_logits = self.issue_head(cls_out)         # (B, NUM_ISSUES)

        # Per-step heads
        yield_preds = torch.sigmoid(self.yield_head(step_out).squeeze(-1))  # (B, S) in [0,1]
        ee_preds = torch.sigmoid(self.ee_head(step_out).squeeze(-1))        # (B, S) in [0,1]

        return {
            "route_score": route_score,
            "compat_logits": compat_logits,
            "opmode_logits": opmode_logits,
            "issue_logits": issue_logits,
            "yield_preds": yield_preds,
            "ee_preds": ee_preds,
        }




# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_scorer_dataset(data_path: str, max_steps: int = MAX_SLOTS) -> list[ScorerSample]:
    """Build scorer dataset from v3 JSON."""
    data = json.loads(Path(data_path).read_text())
    if isinstance(data, dict):
        records = data.get("records_kept", data.get("records", []))
    else:
        records = data

    samples = []
    for rec in records:
        for cas in rec.get("cascades", []):
            steps = cas.get("steps", [])
            if len(steps) < 2 or len(steps) > max_steps:
                continue

            tp = cas.get("target_products") or [{}]
            t_smi = ""
            if tp and isinstance(tp[0], dict):
                t_smi = tp[0].get("smiles", "")
            if not t_smi:
                continue

            tfp = morgan_fp(t_smi)
            if tfp.sum() == 0:
                continue

            dom = cas.get("route_domain", "")
            domain_id = DOMAIN_TO_ID.get(dom, DOMAIN_TO_ID["other"])

            slots = [_extract_slot_features(s, cas) for s in steps]
            # Mark all as observed (scorer sees complete routes)
            for slot in slots:
                slot.is_observed = 1

            # Per-step yields and ee
            yields = []
            ees = []
            for s in steps:
                outcome = s.get("step_outcome") or {}
                y = outcome.get("step_yield_percent")
                yields.append(y / 100.0 if y is not None else -1.0)
                e = outcome.get("step_ee_percent")
                ees.append(e / 100.0 if e is not None else -1.0)

            # Global labels
            ca = cas.get("compatibility_annotation") or {}
            compat_str = ca.get("compatibility_label", "")
            compat_label = COMPAT_TO_ID.get(compat_str, -1)
            opmode_label = OPMODE_TO_ID.get(cas.get("operation_mode", ""), -1)

            issue_labels = []
            for it in (ca.get("issue_types") or []):
                idx = next((i for i, v in enumerate(ISSUE_TYPE_VOCAB) if v == it), -1)
                if idx >= 0:
                    issue_labels.append(idx)

            ev_str = ca.get("evidence_strength", "workflow_only")
            ev_weight = EVIDENCE_WEIGHT.get(ev_str, 0.3)

            route_score = _compat_to_score(compat_str)

            samples.append(ScorerSample(
                target_fp=tfp,
                domain_id=domain_id,
                n_steps=len(steps),
                slots=slots,
                yields=yields,
                ees=ees,
                compat_label=compat_label,
                opmode_label=opmode_label,
                issue_labels=issue_labels,
                evidence_weight=ev_weight,
                route_score=route_score,
            ))

    return samples


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_scorer_batch(
    batch: list[ScorerSample],
    device: str = "cpu",
) -> dict[str, Any]:
    """Collate scorer samples into batched tensors."""
    B = len(batch)
    S = max(s.n_steps for s in batch)

    rtype_ids = torch.zeros(B, S, dtype=torch.long)
    ec1_ids = torch.zeros(B, S, dtype=torch.long)
    ec2_ids = torch.zeros(B, S, dtype=torch.long)
    step_role_ids = torch.zeros(B, S, dtype=torch.long)
    step_type_ids = torch.zeros(B, S, dtype=torch.long)
    catclass_ids = torch.zeros(B, S, dtype=torch.long)
    engineering_ids = torch.zeros(B, S, dtype=torch.long)
    biofmt_ids = torch.zeros(B, S, dtype=torch.long)
    cofmode_ids = torch.zeros(B, S, dtype=torch.long)
    positions = torch.zeros(B, S, dtype=torch.long)
    cond_feats = torch.zeros(B, S, 4)

    target_fps = torch.zeros(B, 2048)
    domain_ids = torch.zeros(B, dtype=torch.long)
    n_steps_ids = torch.zeros(B, dtype=torch.long)

    # Labels
    route_scores = torch.zeros(B)
    compat_labels = torch.full((B,), -100, dtype=torch.long)
    opmode_labels = torch.full((B,), -100, dtype=torch.long)
    issue_labels = torch.zeros(B, NUM_ISSUES)
    yield_labels = torch.full((B, S), -1.0)
    ee_labels = torch.full((B, S), -1.0)
    evidence_weights = torch.ones(B)

    for b, sample in enumerate(batch):
        n = sample.n_steps
        target_fps[b] = torch.from_numpy(sample.target_fp)
        domain_ids[b] = sample.domain_id
        n_steps_ids[b] = n

        for i, slot in enumerate(sample.slots):
            rtype_ids[b, i] = slot.rtype_id
            ec1_ids[b, i] = slot.ec1_id
            ec2_ids[b, i] = slot.ec2_id
            step_role_ids[b, i] = slot.step_role_id
            step_type_ids[b, i] = slot.step_type_id
            catclass_ids[b, i] = slot.catalyst_class_id
            engineering_ids[b, i] = slot.engineering_id
            biofmt_ids[b, i] = slot.biocat_format_id
            cofmode_ids[b, i] = slot.cofactor_mode_id
            positions[b, i] = i
            cond_feats[b, i] = torch.tensor([
                slot.T_norm, slot.pH_norm,
                slot.substrate_conc_norm, slot.reaction_time_norm,
            ])

        # Labels
        route_scores[b] = sample.route_score if sample.route_score >= 0 else 0.5
        if sample.compat_label >= 0:
            compat_labels[b] = sample.compat_label
        if sample.opmode_label >= 0:
            opmode_labels[b] = sample.opmode_label
        for idx in sample.issue_labels:
            issue_labels[b, idx] = 1.0
        evidence_weights[b] = sample.evidence_weight

        for i in range(n):
            if sample.yields[i] >= 0:
                yield_labels[b, i] = sample.yields[i]
            if sample.ees[i] >= 0:
                ee_labels[b, i] = sample.ees[i]

    return {
        "step_inputs": {
            "rtype_ids": rtype_ids.to(device),
            "ec1_ids": ec1_ids.to(device),
            "ec2_ids": ec2_ids.to(device),
            "step_role_ids": step_role_ids.to(device),
            "step_type_ids": step_type_ids.to(device),
            "catclass_ids": catclass_ids.to(device),
            "engineering_ids": engineering_ids.to(device),
            "biofmt_ids": biofmt_ids.to(device),
            "cofmode_ids": cofmode_ids.to(device),
            "positions": positions.to(device),
            "cond_feats": cond_feats.to(device),
        },
        "target_fp": target_fps.to(device),
        "domain_ids": domain_ids.to(device),
        "n_steps_ids": n_steps_ids.to(device),
        "route_scores": route_scores.to(device),
        "compat_labels": compat_labels.to(device),
        "opmode_labels": opmode_labels.to(device),
        "issue_labels": issue_labels.to(device),
        "yield_labels": yield_labels.to(device),
        "ee_labels": ee_labels.to(device),
        "evidence_weights": evidence_weights.to(device),
    }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_scorer_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Multi-task loss for the route scorer."""
    device = outputs["route_score"].device
    B = outputs["route_score"].shape[0]
    ew = batch["evidence_weights"]

    # Route score regression (MSE)
    L_score = F.mse_loss(outputs["route_score"], batch["route_scores"], reduction="none")
    L_score = (L_score * ew).mean()

    # Compat classification
    L_compat = torch.tensor(0.0, device=device)
    valid_compat = batch["compat_labels"] >= 0
    if valid_compat.any():
        ce = F.cross_entropy(
            outputs["compat_logits"][valid_compat],
            batch["compat_labels"][valid_compat],
            reduction="none",
        )
        L_compat = (ce * ew[valid_compat]).mean()

    # Opmode classification
    L_opmode = torch.tensor(0.0, device=device)
    valid_opmode = batch["opmode_labels"] >= 0
    if valid_opmode.any():
        ce = F.cross_entropy(
            outputs["opmode_logits"][valid_opmode],
            batch["opmode_labels"][valid_opmode],
            reduction="none",
        )
        L_opmode = (ce * ew[valid_opmode]).mean()

    # Issue detection (multi-label BCE)
    L_issue = F.binary_cross_entropy_with_logits(
        outputs["issue_logits"],
        batch["issue_labels"],
        reduction="none",
    ).mean(dim=-1)
    L_issue = (L_issue * ew).mean()

    # Per-step yield (masked)
    L_yield = torch.tensor(0.0, device=device)
    valid_yield = batch["yield_labels"] >= 0
    if valid_yield.any():
        L_yield = F.mse_loss(
            outputs["yield_preds"][valid_yield],
            batch["yield_labels"][valid_yield],
            reduction="mean",
        )

    # Per-step ee (masked)
    L_ee = torch.tensor(0.0, device=device)
    valid_ee = batch["ee_labels"] >= 0
    if valid_ee.any():
        L_ee = F.mse_loss(
            outputs["ee_preds"][valid_ee],
            batch["ee_labels"][valid_ee],
            reduction="mean",
        )

    # Pairwise ranking loss: routes with higher compat should score higher
    L_rank = torch.tensor(0.0, device=device)
    if B >= 2:
        scores = outputs["route_score"]
        targets = batch["route_scores"]
        n_pairs = 0
        rank_loss = torch.tensor(0.0, device=device)
        for i in range(min(B, 16)):
            for j in range(i + 1, min(B, 16)):
                if targets[i] != targets[j]:
                    sign = 1.0 if targets[i] > targets[j] else -1.0
                    rank_loss += F.relu(0.1 - sign * (scores[i] - scores[j]))
                    n_pairs += 1
        if n_pairs > 0:
            L_rank = rank_loss / n_pairs

    total = (
        2.0 * L_score + 1.0 * L_compat + 0.5 * L_opmode
        + 0.3 * L_issue + 0.5 * L_yield + 0.3 * L_ee + 1.0 * L_rank
    )

    metrics = {
        "loss": total.item(),
        "L_score": L_score.item(),
        "L_compat": L_compat.item(),
        "L_opmode": L_opmode.item(),
        "L_issue": L_issue.item(),
        "L_yield": L_yield.item(),
        "L_ee": L_ee.item(),
        "L_rank": L_rank.item(),
    }
    return total, metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_scorer(
    data_path: str = "cascade_dataset_v3.json",
    epochs: int = 150,
    batch_size: int = 64,
    lr: float = 3e-4,
    warmup_steps: int = 300,
    save_dir: str = "results/shared/learned_scorer",
    seed: int = 42,
):
    """Train the Learned Route Scorer."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading data from {data_path}...")
    samples = build_scorer_dataset(data_path)
    print(f"  Samples: {len(samples)}")

    # Train/val split (80/20)
    random.shuffle(samples)
    split = int(len(samples) * 0.8)
    train_samples = samples[:split]
    val_samples = samples[split:]
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

    model = LearnedRouteScorer().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,} ({n_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * (len(train_samples) // batch_size + 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    history = []

    for epoch in range(epochs):
        model.train()
        random.shuffle(train_samples)
        train_loss_sum = 0.0
        n_batches = 0

        for i in range(0, len(train_samples), batch_size):
            batch_samples = train_samples[i:i + batch_size]
            batch = collate_scorer_batch(batch_samples, device=device)

            outputs = model(
                step_inputs=batch["step_inputs"],
                target_fp=batch["target_fp"],
                domain_ids=batch["domain_ids"],
                n_steps_ids=batch["n_steps_ids"],
            )

            loss, metrics = compute_scorer_loss(outputs, batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss_sum += metrics["loss"]
            n_batches += 1

        avg_train_loss = train_loss_sum / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_compat_correct = val_compat_total = 0
        n_val_batches = 0

        with torch.no_grad():
            for i in range(0, len(val_samples), batch_size):
                batch_samples = val_samples[i:i + batch_size]
                batch = collate_scorer_batch(batch_samples, device=device)

                outputs = model(
                    step_inputs=batch["step_inputs"],
                    target_fp=batch["target_fp"],
                    domain_ids=batch["domain_ids"],
                    n_steps_ids=batch["n_steps_ids"],
                )

                loss, _ = compute_scorer_loss(outputs, batch)
                val_loss_sum += loss.item()
                n_val_batches += 1

                valid_compat = batch["compat_labels"] >= 0
                if valid_compat.any():
                    preds = outputs["compat_logits"][valid_compat].argmax(dim=-1)
                    val_compat_correct += (preds == batch["compat_labels"][valid_compat]).sum().item()
                    val_compat_total += valid_compat.sum().item()

        avg_val_loss = val_loss_sum / max(n_val_batches, 1)
        val_compat_acc = val_compat_correct / max(val_compat_total, 1)

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_compat_acc": val_compat_acc,
        })

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  ep{epoch+1:3d}: train={avg_train_loss:.4f} val={avg_val_loss:.4f} "
                f"compat_acc={val_compat_acc:.3f} lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path / "best.pt")

    torch.save(model.state_dict(), save_path / "final.pt")
    (save_path / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Saved to {save_path}/")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_scorer(
    checkpoint_path: str = "results/shared/learned_scorer/best.pt",
    device: str = "cpu",
) -> LearnedRouteScorer:
    model = LearnedRouteScorer()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@dataclass
class ScoreResult:
    route_score: float
    compat_pred: str
    opmode_pred: str
    issues_pred: list[str]
    yield_preds: list[float]
    ee_preds: list[float]


@torch.no_grad()
def score_route(
    model: LearnedRouteScorer,
    target_smi: str,
    slots: list[SlotFeatures],
    domain: str = "chemoenzymatic",
    device: str = "cpu",
) -> ScoreResult:
    """Score a single complete route."""
    model.eval()
    n = len(slots)

    sample = ScorerSample(
        target_fp=morgan_fp(target_smi),
        domain_id=DOMAIN_TO_ID.get(domain, DOMAIN_TO_ID["other"]),
        n_steps=n,
        slots=slots,
        yields=[-1.0] * n,
        ees=[-1.0] * n,
    )

    batch = collate_scorer_batch([sample], device=device)
    outputs = model(
        step_inputs=batch["step_inputs"],
        target_fp=batch["target_fp"],
        domain_ids=batch["domain_ids"],
        n_steps_ids=batch["n_steps_ids"],
    )

    compat_idx = outputs["compat_logits"][0].argmax().item()
    opmode_idx = outputs["opmode_logits"][0].argmax().item()
    issue_probs = torch.sigmoid(outputs["issue_logits"][0])
    issues = [ISSUE_TYPE_VOCAB[i] for i in range(NUM_ISSUES) if issue_probs[i] > 0.5]

    return ScoreResult(
        route_score=outputs["route_score"][0].item(),
        compat_pred=COMPAT_VOCAB[compat_idx],
        opmode_pred=OPMODE_VOCAB[opmode_idx],
        issues_pred=issues,
        yield_preds=[outputs["yield_preds"][0, i].item() * 100 for i in range(n)],
        ee_preds=[outputs["ee_preds"][0, i].item() * 100 for i in range(n)],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Learned Route Scorer")
    ap.add_argument("--data", default="cascade_dataset_v3.json")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save-dir", default="results/shared/learned_scorer")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_scorer(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        seed=args.seed,
    )
