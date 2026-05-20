"""Train CascadeBoard++ Transformer model.

Masked Route Modeling + Edit Action prediction.
"""
from __future__ import annotations

import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from cascade_planner.cascadeboard.route_encoder import (
    CascadeBoardTransformer, REACTION_TYPE_TO_ID, count_params,
    D_MODEL, NUM_REACTION_TYPES, NUM_EC1_CLASSES, NUM_EC2_CLASSES, NUM_EDIT_TYPES,
    OBJECTIVE_TO_ID, smiles_to_morgan_fp, inject_constraint_features,
    constraint_features_from_slot_dicts, EC2_TO_ID,
    COMPAT_TO_ID, OPMODE_TO_ID, ISSUE_TYPE_TO_IDX, PAIRWISE_TO_ID,
    DOMAIN_TO_ID,
)
from cascade_planner.cascadeboard import EditType
from cascade_planner.cascadeboard.training_data import (
    load_cascade_routes, build_training_data, TrainingSample,
    DIFFICULTY_EASY, DIFFICULTY_MEDIUM, DIFFICULTY_HARD, DIFFICULTY_HARDEST,
    route_to_board, mask_single_field, mask_full_step, mask_half, mask_all,
    corrupt_replace_type, corrupt_shift_T, corrupt_shift_pH,
    corrupt_replace_ec, corrupt_replace_enzyme, corrupt_swap_order,
    corrupt_delete_step, corrupt_insert_extra, route_ok,
    _attach_route_labels,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_MASK_FNS = [mask_single_field, mask_full_step, mask_half, mask_all]
_CORRUPT_FNS = [
    corrupt_replace_type, corrupt_shift_T, corrupt_shift_pH,
    corrupt_replace_ec, corrupt_replace_enzyme, corrupt_swap_order,
    corrupt_delete_step, corrupt_insert_extra,
]


# ---------------------------------------------------------------------------
# Online augmentation dataset — one sample per route per __getitem__
# ---------------------------------------------------------------------------

class OnlineRouteDataset(Dataset):
    """Each item is a route; mask/corruption is sampled online per access.

    Also builds real preference pairs from v3 compatibility annotations
    for route optimization training (shorter>longer, compatible>incompatible).
    """

    COMPAT_RANK = {'empirically_compatible':0, 'compatible_with_mitigation':1,
                   'compatible_with_compromise':2, 'unclear':3, 'empirically_incompatible':4, '':5}

    def __init__(self, routes: list[dict], max_slots: int = 8, candidate_cache: dict | None = None):
        self.routes = routes
        self.max_slots = max_slots
        self.candidate_cache = candidate_cache
        self._boards = []
        for r in routes:
            b = route_to_board(r)
            if b.n_steps >= 2:
                self._boards.append((r, b))
        # Build preference pair index: group routes by DOI for cross-route comparison
        self._doi_groups = {}
        for idx, (r, b) in enumerate(self._boards):
            doi = r.get("doi", "")
            if doi:
                self._doi_groups.setdefault(doi, []).append(idx)
        self._multi_dois = [doi for doi, idxs in self._doi_groups.items() if len(idxs) > 1]

    def __len__(self):
        return len(self._boards)

    def _encode_candidates(self, sample, max_slots: int = 8, K: int = 10):
        """Encode candidate pool features for each slot. Returns (S, K, 29) + (S, K) mask."""
        import torch
        feats = torch.zeros(max_slots, K, 29)
        mask = torch.zeros(max_slots, K, dtype=torch.bool)
        if not self.candidate_cache:
            return feats, mask
        from cascade_planner.cascadeboard.candidate_cache import canon_smiles
        for i, slot in enumerate(sample.slots[:max_slots]):
            product = slot.get("product", "")
            if not product:
                continue
            key = canon_smiles(product) or product
            cands = self.candidate_cache.get(key, [])
            for j, c in enumerate(cands[:K]):
                score = float(c.get("score", 0))
                rt = c.get("reaction_type", "")
                type_id = REACTION_TYPE_TO_ID.get(rt, 0)
                ec = c.get("ec", "")
                ec1 = int(ec.split(".")[0]) if ec and ec[0].isdigit() else 0
                feats[i, j, 0] = score
                if 0 < type_id < 20:
                    feats[i, j, type_id] = 1.0  # one-hot type (indices 1-19)
                if 0 < ec1 < 8:
                    feats[i, j, 20 + ec1] = 1.0  # one-hot ec1 (indices 21-27)
                feats[i, j, 28] = 1.0  # has_candidate flag
                mask[i, j] = True
        return feats, mask

    def __getitem__(self, idx):
        route, board = self._boards[idx]
        rng = random.Random()
        roll = rng.random()
        if roll < 0.40:
            # 40% skeleton generation: mask ALL slots (learn from-scratch planning)
            sample = mask_all(board, rng)
        elif roll < 0.60:
            # 20% partial mask (learn completion)
            fn = rng.choice([mask_single_field, mask_full_step, mask_half])
            sample = fn(board, rng)
        elif roll < 0.85:
            # 25% corruption (learn repair)
            fn = rng.choice(_CORRUPT_FNS)
            sample = fn(board, rng)
        else:
            # 15% route_ok (learn energy calibration)
            sample = route_ok(board, rng)
        _attach_route_labels(sample, route)
        encoded = CascadeBoardDataset._encode_sample(sample, self.max_slots)
        # Add candidate features if cache available
        if self.candidate_cache:
            cand_feats, cand_mask = self._encode_candidates(sample, self.max_slots)
            encoded["candidate_features"] = cand_feats
            encoded["candidate_mask"] = cand_mask

        # Real preference signal: encode compatibility rank as a scalar for ranking loss
        # 0=best (empirically_compatible) → 4=worst (incompatible)
        compat_rank = self.COMPAT_RANK.get(route.get("compatibility_label", ""), 3)
        encoded["compat_rank"] = torch.tensor(float(compat_rank), dtype=torch.float32)

        return encoded

    @staticmethod
    def from_routes(routes, max_slots=8):
        return OnlineRouteDataset(routes, max_slots)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CascadeBoardDataset(Dataset):
    def __init__(self, samples: list[TrainingSample], max_slots: int = 8):
        self.samples = samples
        self._max_slots = max_slots

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._encode_sample(self.samples[idx], self._max_slots)

    @staticmethod
    def _encode_sample(s: TrainingSample, max_slots: int = 8) -> dict:
        n_slots = min(len(s.slots), max_slots)

        # Encode input slots (masked)
        type_ids = []
        ec1_ids = []
        conditions = []
        scores = []
        is_fixed = []

        for i in range(n_slots):
            slot = s.slots[i]
            rt = slot.get("reaction_type") or ""
            type_ids.append(REACTION_TYPE_TO_ID.get(rt, 0))

            ec = slot.get("ec") or ""
            ec1 = int(ec.split(".")[0]) if ec and ec[0].isdigit() else 0
            ec1_ids.append(ec1)

            T = slot.get("T")
            pH = slot.get("pH")
            T_norm = (T - 37) / 30 if T is not None else 0.0
            pH_norm = (pH - 7) / 3 if pH is not None else 0.0
            conditions.append([T_norm, pH_norm])

            e_r = slot.get("e_retro") or 0.0
            e_e = slot.get("e_enzyme") or 0.0
            e_c = slot.get("e_condition") or 0.0
            scores.append([e_r, e_e, e_c])

            fixed = 1 if slot.get("fixed") else 0
            is_fixed.append(fixed)

        # Pad to max_slots
        while len(type_ids) < max_slots:
            type_ids.append(0)
            ec1_ids.append(0)
            conditions.append([0.0, 0.0])
            scores.append([0.0, 0.0, 0.0])
            is_fixed.append(0)

        # Encode labels (true slots) — handle length mismatch from insert/delete corruptions
        true_type_ids = []
        true_ec1_ids = []
        true_ec2_ids = []
        true_conditions = []
        true_scores = []
        true_is_fixed = []
        true_yield = []
        true_ee = []
        yield_mask = []
        ee_mask = []

        n_true = min(n_slots, len(s.true_slots))
        for i in range(n_true):
            ts = s.true_slots[i]
            rt = ts.get("reaction_type") or ""
            true_type_ids.append(REACTION_TYPE_TO_ID.get(rt, 0))

            ec = ts.get("ec") or ""
            ec1 = int(ec.split(".")[0]) if ec and ec[0].isdigit() else 0
            true_ec1_ids.append(ec1)

            # EC2 label
            ec2_key = "NONE"
            if ec and "." in ec:
                parts = ec.split(".")
                if len(parts) >= 2:
                    ec2_key = f"{parts[0]}.{parts[1]}"
            true_ec2_ids.append(EC2_TO_ID.get(ec2_key, 0))

            T = ts.get("T")
            pH = ts.get("pH")
            T_norm = (T - 37) / 30 if T is not None else 0.0
            pH_norm = (pH - 7) / 3 if pH is not None else 0.0
            true_conditions.append([T_norm, pH_norm])
            true_scores.append([
                ts.get("e_retro") or 0.0,
                ts.get("e_enzyme") or 0.0,
                ts.get("e_condition") or 0.0,
            ])
            true_is_fixed.append(1 if ts.get("fixed") else 0)

            # Yield / EE labels (masked where not available)
            y = ts.get("yield_pct")
            true_yield.append(y if y is not None else 0.0)
            yield_mask.append(1.0 if y is not None else 0.0)
            e = ts.get("ee_pct")
            true_ee.append(e if e is not None else 0.0)
            ee_mask.append(1.0 if e is not None else 0.0)

        while len(true_type_ids) < max_slots:
            true_type_ids.append(0)
            true_ec1_ids.append(0)
            true_ec2_ids.append(0)
            true_conditions.append([0.0, 0.0])
            true_scores.append([0.0, 0.0, 0.0])
            true_is_fixed.append(0)
            true_yield.append(0.0)
            true_ee.append(0.0)
            yield_mask.append(0.0)
            ee_mask.append(0.0)

        # Mask: which slots need prediction
        mask = [0] * max_slots
        for mi in s.mask_indices:
            if mi < max_slots:
                mask[mi] = 1

        issue_mask = [0] * max_slots
        issue_indices = s.issue_indices or ([s.edit_slot] if s.edit_slot is not None else s.mask_indices)
        for ii in issue_indices:
            if ii is not None and 0 <= ii < max_slots:
                issue_mask[ii] = 1

        # Target FP: real Morgan fingerprint of the route target.
        target_fp = inject_constraint_features(
            smiles_to_morgan_fp(s.target_smiles),
            constraint_features_from_slot_dicts(s.slots),
        )

        # Edit action label. Model classes are zero-indexed in EditType order:
        # FILL_FIELD=0, REPLACE_STEP=1, ..., RELAX_CONSTRAINT=7.
        edit_action_id = EditType.FILL_FIELD.value - 1
        if s.edit_action:
            try:
                edit_action_id = EditType[s.edit_action].value - 1
            except KeyError:
                edit_action_id = EditType.FILL_FIELD.value - 1

        edit_slot = s.edit_slot
        if edit_slot is None and s.mask_indices:
            edit_slot = s.mask_indices[0]
        if edit_slot is None:
            edit_slot = 0
        edit_slot = max(0, min(int(edit_slot), max_slots - 1))
        edit_action_valid = 0.0 if s.edit_action_valid is False else 1.0
        severity = s.severity
        if severity is None:
            severity = max(sum(issue_mask), 1) / max(n_slots, 1)

        return {
            "target_fp": torch.tensor(target_fp),
            "objective_id": torch.tensor(
                OBJECTIVE_TO_ID.get(s.objective, OBJECTIVE_TO_ID["balanced"]),
                dtype=torch.long,
            ),
            "type_ids": torch.tensor(type_ids, dtype=torch.long),
            "ec1_ids": torch.tensor(ec1_ids, dtype=torch.long),
            "conditions": torch.tensor(conditions, dtype=torch.float32),
            "scores": torch.tensor(scores, dtype=torch.float32),
            "is_fixed": torch.tensor(is_fixed, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "true_type_ids": torch.tensor(true_type_ids, dtype=torch.long),
            "true_ec1_ids": torch.tensor(true_ec1_ids, dtype=torch.long),
            "true_ec2_ids": torch.tensor(true_ec2_ids, dtype=torch.long),
            "true_conditions": torch.tensor(true_conditions, dtype=torch.float32),
            "true_scores": torch.tensor(true_scores, dtype=torch.float32),
            "true_is_fixed": torch.tensor(true_is_fixed, dtype=torch.long),
            "true_yield": torch.tensor(true_yield, dtype=torch.float32),
            "true_ee": torch.tensor(true_ee, dtype=torch.float32),
            "yield_mask": torch.tensor(yield_mask, dtype=torch.float32),
            "ee_mask": torch.tensor(ee_mask, dtype=torch.float32),
            "edit_action_id": torch.tensor(edit_action_id, dtype=torch.long),
            "edit_slot": torch.tensor(edit_slot, dtype=torch.long),
            "edit_action_valid": torch.tensor(edit_action_valid, dtype=torch.float32),
            "issue_mask": torch.tensor(issue_mask, dtype=torch.float32),
            "severity": torch.tensor(severity, dtype=torch.float32),
            "n_slots": torch.tensor(n_slots, dtype=torch.long),
            # Candidate pool features
            "candidate_features": torch.zeros(max_slots, 10, 29),
            "candidate_mask": torch.zeros(max_slots, 10, dtype=torch.bool),
            # v3 real labels
            "compat_label": torch.tensor(
                COMPAT_TO_ID.get(s.compatibility_label, 0), dtype=torch.long,
            ),
            "compat_valid": torch.tensor(
                1.0 if s.compatibility_label in COMPAT_TO_ID else 0.0, dtype=torch.float32,
            ),
            "opmode_label": torch.tensor(
                OPMODE_TO_ID.get(s.operation_mode, len(OPMODE_TO_ID) - 1), dtype=torch.long,
            ),
            "opmode_valid": torch.tensor(
                1.0 if s.operation_mode in OPMODE_TO_ID else 0.0, dtype=torch.float32,
            ),
            "issue_type_labels": torch.tensor(
                [1.0 if it in (s.issue_types or []) else 0.0 for it in ISSUE_TYPE_TO_IDX],
                dtype=torch.float32,
            ),
            "issue_type_valid": torch.tensor(
                1.0 if s.compatibility_label else 0.0, dtype=torch.float32,
            ),
            "pairwise_labels": _encode_pairwise(s.pairwise_modes, max_slots),
            "pairwise_valid": _encode_pairwise_valid(s.pairwise_modes, max_slots),
            # Context: domain + starting material
            "domain_id": torch.tensor(
                DOMAIN_TO_ID.get(getattr(s, 'route_domain', ''), DOMAIN_TO_ID.get('unknown', 5)),
                dtype=torch.long,
            ),
            "sm_fp": torch.tensor(
                smiles_to_morgan_fp(getattr(s, 'starting_material', '') or ''),
                dtype=torch.float32,
            ),
        }


# ---------------------------------------------------------------------------
# Helper: encode pairwise labels
# ---------------------------------------------------------------------------

def _encode_pairwise(pairwise_modes: list[str], max_slots: int) -> torch.Tensor:
    """Encode pairwise_mode labels for adjacent step pairs. Returns (max_slots-1,) long tensor."""
    labels = []
    for i in range(max_slots - 1):
        if i < len(pairwise_modes) and pairwise_modes[i] in PAIRWISE_TO_ID:
            labels.append(PAIRWISE_TO_ID[pairwise_modes[i]])
        else:
            labels.append(0)
    return torch.tensor(labels, dtype=torch.long)


def _encode_pairwise_valid(pairwise_modes: list[str], max_slots: int) -> torch.Tensor:
    """Validity mask for pairwise labels. Returns (max_slots-1,) float tensor."""
    valid = []
    for i in range(max_slots - 1):
        if i < len(pairwise_modes) and pairwise_modes[i] in PAIRWISE_TO_ID:
            valid.append(1.0)
        else:
            valid.append(0.0)
    return torch.tensor(valid, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Focal loss for class-imbalanced edit classification
# ---------------------------------------------------------------------------

def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, reduction: str = "none") -> torch.Tensor:
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma) * ce


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_cascadeboard(
    train_samples: list[TrainingSample],
    val_samples: list[TrainingSample],
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 5e-4,
    device: str = DEVICE,
    save_dir: str = "results/shared/cascadeboard_model",
    log_every: int = 1,
    curriculum: bool = False,
    dropout: float = 0.2,
    label_smoothing: float = 0.0,
    grad_clip: float = 0.0,
    weight_decay: float = 0.01,
):
    """Train CascadeBoard++ Transformer.

    When curriculum=True, training is phased:
      Phase 1 (0-25%):  easy only (single field mask, route_ok)
      Phase 2 (25-50%): + medium (full step mask, single corruptions)
      Phase 3 (50-75%): + hard (half mask, swap, delete, insert)
      Phase 4 (75-100%): all including hardest (mask_all)
    """

    print(f"Training CascadeBoard++ on {device}")
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}, Curriculum: {curriculum}")

    if curriculum:
        phase_bins = {
            1: [s for s in train_samples if s.difficulty <= DIFFICULTY_EASY],
            2: [s for s in train_samples if s.difficulty <= DIFFICULTY_MEDIUM],
            3: [s for s in train_samples if s.difficulty <= DIFFICULTY_HARD],
            4: train_samples,
        }
        for p, ss in phase_bins.items():
            print(f"  Phase {p}: {len(ss)} samples")

    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"  Dropout: {dropout}, LabelSmooth: {label_smoothing}, GradClip: {grad_clip}, WD: {weight_decay}")

    # Datasets
    val_ds = CascadeBoardDataset(val_samples)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    # Model — configurable dropout
    model = CascadeBoardTransformer().to(device)
    for module in model.modules():
        if isinstance(module, nn.TransformerEncoderLayer):
            module.dropout = nn.Dropout(dropout)
    print(f"  Model: {count_params(model)/1e6:.2f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history = []
    prev_phase = 0

    for epoch in range(epochs):
        # Select curriculum phase based on training progress
        if curriculum:
            progress = epoch / max(epochs - 1, 1)
            if progress < 0.25:
                phase = 1
            elif progress < 0.50:
                phase = 2
            elif progress < 0.75:
                phase = 3
            else:
                phase = 4
            cur_train = phase_bins[phase]
            if phase != prev_phase:
                print(f"  >>> Curriculum phase {phase}: {len(cur_train)} samples (epoch {epoch+1})")
                prev_phase = phase
        else:
            cur_train = train_samples

        train_ds = CascadeBoardDataset(cur_train)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        # ----- Train -----
        model.train()
        train_loss_sum = 0
        n_batches = 0

        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(
                target_fp=batch["target_fp"],
                objective_ids=batch["objective_id"],
                slot_type_ids=batch["type_ids"],
                slot_ec1_ids=batch["ec1_ids"],
                slot_conditions=batch["conditions"],
                slot_scores=batch["scores"],
                slot_is_fixed=batch["is_fixed"],
            )
            clean_out = model(
                target_fp=batch["target_fp"],
                objective_ids=batch["objective_id"],
                slot_type_ids=batch["true_type_ids"],
                slot_ec1_ids=batch["true_ec1_ids"],
                slot_conditions=batch["true_conditions"],
                slot_scores=batch["true_scores"],
                slot_is_fixed=batch["true_is_fixed"],
            )

            # Losses (only on masked positions)
            mask = batch["mask"]  # (B, S)
            B, S = mask.shape

            # L_inpaint: type classification
            type_logits = out["type_logits"][:, :S]  # (B, S, C)
            true_types = batch["true_type_ids"][:, :S]
            l_type = F.cross_entropy(
                type_logits.reshape(-1, NUM_REACTION_TYPES),
                true_types.reshape(-1),
                reduction="none",
                label_smoothing=label_smoothing,
            ).reshape(B, S)
            l_type = (l_type * mask).sum() / mask.sum().clamp(min=1)

            # L_inpaint: EC1 classification
            ec1_logits = out["ec1_logits"][:, :S]
            true_ec1 = batch["true_ec1_ids"][:, :S]
            l_ec1 = F.cross_entropy(
                ec1_logits.reshape(-1, NUM_EC1_CLASSES),
                true_ec1.reshape(-1),
                reduction="none",
                label_smoothing=label_smoothing,
            ).reshape(B, S)
            l_ec1 = (l_ec1 * mask).sum() / mask.sum().clamp(min=1)

            # L_inpaint: condition regression — Huber loss per expert advice
            cond_preds = out["cond_preds"][:, :S]  # (B, S, 2)
            true_conds = batch["true_conditions"][:, :S]
            l_cond = F.smooth_l1_loss(cond_preds, true_conds, reduction="none").mean(dim=-1)  # (B, S)
            l_cond = (l_cond * mask).sum() / mask.sum().clamp(min=1)

            # L_edit: edit action classification (focal loss for class balance)
            edit_logits = out["edit_type_logits"]  # (B, NUM_EDIT)
            true_edit = batch["edit_action_id"]
            l_edit_raw = focal_loss(edit_logits, true_edit, gamma=2.0)
            edit_valid = batch["edit_action_valid"]
            l_edit = (l_edit_raw * edit_valid).sum() / edit_valid.sum().clamp(min=1)

            # L_edit_target: predict which slot should be edited.
            edit_target_logits = out["edit_target_logits"]  # (B, S)
            true_edit_slot = batch["edit_slot"].clamp(max=S - 1)
            slot_valid = (
                torch.arange(S, device=device).unsqueeze(0)
                < batch["n_slots"].unsqueeze(1)
            )
            edit_target_logits = edit_target_logits.masked_fill(~slot_valid, -1e9)
            l_edit_target_raw = F.cross_entropy(edit_target_logits, true_edit_slot, reduction="none")
            l_edit_target = (l_edit_target_raw * edit_valid).sum() / edit_valid.sum().clamp(min=1)

            # L_energy: contrastive — corrupted samples should have higher energy
            energy_pred = out["energy"]
            clean_energy = clean_out["energy"]
            corruption_severity = batch["severity"]
            l_energy = 0.5 * F.mse_loss(energy_pred, corruption_severity)
            l_energy = l_energy + 0.5 * F.mse_loss(clean_energy, torch.zeros_like(clean_energy))

            # L_route_rank / L_pref: literature route should score better than corrupted input.
            # Energy is trained lower-is-better, so corrupted energy should exceed clean energy.
            margin = 0.10
            l_route_rank = F.softplus(clean_energy + margin - energy_pred).mean()
            l_pref = -F.logsigmoid(energy_pred - clean_energy).mean()

            # L_feasibility
            feasibility = out["feasibility"]
            target_feas = 1.0 - corruption_severity
            l_feas = F.mse_loss(feasibility, target_feas)

            # L_constraint: predict which slots have issues
            issue_scores = out["issue_scores"][:, :S]
            issue_mask = batch["issue_mask"][:, :S]
            l_constraint = F.binary_cross_entropy(issue_scores, issue_mask)

            # Total loss with all components
            loss = (
                1.0 * l_type
                + 1.0 * l_ec1
                + 0.5 * l_cond
                + 0.3 * l_edit
                + 0.3 * l_edit_target
                + 0.2 * l_feas
                + 0.2 * l_energy
                + 0.2 * l_route_rank
                + 0.1 * l_pref
                + 0.1 * l_constraint
            )

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1

            # Per-loss tracking for diagnosis
            if not hasattr(train_cascadeboard, '_loss_accum'):
                train_cascadeboard._loss_accum = {}
            for lname, lval in [
                ("type", l_type), ("ec1", l_ec1), ("cond", l_cond),
                ("edit", l_edit), ("edit_target", l_edit_target),
                ("energy", l_energy), ("rank", l_route_rank),
                ("pref", l_pref), ("feas", l_feas), ("constraint", l_constraint),
            ]:
                train_cascadeboard._loss_accum.setdefault(lname, 0.0)
                train_cascadeboard._loss_accum[lname] += lval.item()

        scheduler.step()
        avg_train = train_loss_sum / max(n_batches, 1)

        # ----- Validate -----
        model.eval()
        val_loss_sum = 0
        val_type_correct = 0
        val_type_total = 0
        val_edit_correct = 0
        val_edit_total = 0
        val_target_correct = 0
        val_target_total = 0
        val_rank_correct = 0
        val_rank_total = 0
        n_val = 0

        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(
                    target_fp=batch["target_fp"],
                    objective_ids=batch["objective_id"],
                    slot_type_ids=batch["type_ids"],
                    slot_ec1_ids=batch["ec1_ids"],
                    slot_conditions=batch["conditions"],
                    slot_scores=batch["scores"],
                    slot_is_fixed=batch["is_fixed"],
                )
                clean_out = model(
                    target_fp=batch["target_fp"],
                    objective_ids=batch["objective_id"],
                    slot_type_ids=batch["true_type_ids"],
                    slot_ec1_ids=batch["true_ec1_ids"],
                    slot_conditions=batch["true_conditions"],
                    slot_scores=batch["true_scores"],
                    slot_is_fixed=batch["true_is_fixed"],
                )

                mask = batch["mask"]
                B, S = mask.shape

                type_logits = out["type_logits"][:, :S]
                true_types = batch["true_type_ids"][:, :S]
                l_type = F.cross_entropy(
                    type_logits.reshape(-1, NUM_REACTION_TYPES),
                    true_types.reshape(-1),
                    reduction="none",
                ).reshape(B, S)
                l_type = (l_type * mask).sum() / mask.sum().clamp(min=1)

                val_loss_sum += l_type.item()
                n_val += 1

                # Accuracy on masked positions
                pred_types = type_logits.argmax(dim=-1)  # (B, S)
                correct = ((pred_types == true_types) * mask).sum().item()
                total = mask.sum().item()
                val_type_correct += correct
                val_type_total += total

                pred_edits = out["edit_type_logits"].argmax(dim=-1)
                edit_valid = batch["edit_action_valid"] > 0.5
                val_edit_correct += ((pred_edits == batch["edit_action_id"]) & edit_valid).sum().item()
                val_edit_total += edit_valid.sum().item()

                edit_target_logits = out["edit_target_logits"][:, :S]
                slot_valid = (
                    torch.arange(S, device=device).unsqueeze(0)
                    < batch["n_slots"].unsqueeze(1)
                )
                edit_target_logits = edit_target_logits.masked_fill(~slot_valid, -1e9)
                pred_targets = edit_target_logits.argmax(dim=-1)
                true_targets = batch["edit_slot"].clamp(max=S - 1)
                val_target_correct += ((pred_targets == true_targets) & edit_valid).sum().item()
                val_target_total += edit_valid.sum().item()
                val_rank_correct += (out["energy"] > clean_out["energy"]).sum().item()
                val_rank_total += out["energy"].numel()

        avg_val = val_loss_sum / max(n_val, 1)
        val_acc = val_type_correct / max(val_type_total, 1)
        val_edit_acc = val_edit_correct / max(val_edit_total, 1)
        val_target_acc = val_target_correct / max(val_target_total, 1)
        val_route_rank_acc = val_rank_correct / max(val_rank_total, 1)

        # Per-loss diagnosis logging
        per_loss = {}
        if hasattr(train_cascadeboard, '_loss_accum') and n_batches > 0:
            for lname, lsum in train_cascadeboard._loss_accum.items():
                per_loss[f"train_{lname}"] = round(lsum / n_batches, 4)
            train_cascadeboard._loss_accum = {}

        history.append({
            "epoch": epoch, "train_loss": round(avg_train, 4),
            "val_loss": round(avg_val, 4), "val_type_acc": round(val_acc, 4),
            "val_edit_acc": round(val_edit_acc, 4),
            "val_edit_target_acc": round(val_target_acc, 4),
            "val_route_rank_acc": round(val_route_rank_acc, 4),
            **per_loss,
        })

        if (epoch + 1) % max(log_every, 1) == 0 or epoch == 0:
            print(
                f"  epoch {epoch+1:3d}: train={avg_train:.4f} val={avg_val:.4f} "
                f"type_acc={val_acc:.3f} edit_acc={val_edit_acc:.3f} "
                f"target_acc={val_target_acc:.3f} rank_acc={val_route_rank_acc:.3f}"
            )

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path / "best.pt")

    # Save final
    torch.save(model.state_dict(), save_path / "final.pt")
    (save_path / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to {save_path}")

    return model, history


# ---------------------------------------------------------------------------
# Online augmentation training — expert P0 recipe
# ---------------------------------------------------------------------------

def train_cascadeboard_online(
    train_ds: OnlineRouteDataset,
    val_ds: CascadeBoardDataset,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: str = DEVICE,
    save_dir: str = "results/shared/cascadeboard_model",
    log_every: int = 1,
    dropout: float = 0.3,
    label_smoothing: float = 0.1,
    grad_clip: float = 1.0,
    weight_decay: float = 0.05,
    target_fp_dropout: float = 0.0,
    embedding_dropout: float = 0.0,
    route_latent_dim: int = 0,
    d_model: int = D_MODEL,
    n_layers: int = 4,
):
    """Train with online augmentation: each route sampled fresh every epoch."""
    print(f"Training CascadeBoard++ (online) on {device}")
    print(f"  Train routes: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"  Dropout: {dropout}, LabelSmooth: {label_smoothing}, GradClip: {grad_clip}, WD: {weight_decay}")
    print(f"  FP_drop: {target_fp_dropout}, Emb_drop: {embedding_dropout}, Bottleneck: {route_latent_dim}, d={d_model}, L={n_layers}")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    model = CascadeBoardTransformer(
        d_model=d_model, n_layers=n_layers,
        target_fp_dropout=target_fp_dropout,
        embedding_dropout=embedding_dropout,
        route_latent_dim=route_latent_dim,
    ).to(device)
    for module in model.modules():
        if isinstance(module, nn.TransformerEncoderLayer):
            module.dropout = nn.Dropout(dropout)
    print(f"  Model: {count_params(model)/1e6:.2f}M params")
    print(f"  Steps/epoch: ~{len(train_ds) // batch_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Linear warmup + cosine decay
    warmup_steps = min(200, len(train_ds) // batch_size)
    total_steps = epochs * (len(train_ds) // batch_size + 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    history = []
    global_step = 0

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0
        n_batches = 0
        loss_accum = {}

        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            B = batch["target_fp"].shape[0]
            S = batch["mask"].shape[1]

            out = model(
                target_fp=batch["target_fp"],
                objective_ids=batch["objective_id"],
                slot_type_ids=batch["type_ids"],
                slot_ec1_ids=batch["ec1_ids"],
                slot_conditions=batch["conditions"],
                slot_scores=batch["scores"],
                slot_is_fixed=batch["is_fixed"],
                candidate_features=batch.get("candidate_features"),
                candidate_mask=batch.get("candidate_mask"),
                domain_ids=batch.get("domain_id"),
                # sm_fp disabled: 21% empty + 2048-dim noise broke convergence in v22
            )
            clean_out = model(
                target_fp=batch["target_fp"],
                objective_ids=batch["objective_id"],
                slot_type_ids=batch["true_type_ids"],
                slot_ec1_ids=batch["true_ec1_ids"],
                slot_conditions=batch["true_conditions"],
                slot_scores=batch["true_scores"],
                slot_is_fixed=batch["true_is_fixed"],
            )

            mask = batch["mask"]

            # L_type
            type_logits = out["type_logits"][:, :S]
            true_types = batch["true_type_ids"][:, :S]
            l_type = F.cross_entropy(
                type_logits.reshape(-1, NUM_REACTION_TYPES),
                true_types.reshape(-1), reduction="none",
                label_smoothing=label_smoothing,
            ).reshape(B, S)
            l_type = (l_type * mask).sum() / mask.sum().clamp(min=1)

            # L_ec1
            ec1_logits = out["ec1_logits"][:, :S]
            true_ec1 = batch["true_ec1_ids"][:, :S]
            l_ec1 = F.cross_entropy(
                ec1_logits.reshape(-1, NUM_EC1_CLASSES),
                true_ec1.reshape(-1), reduction="none",
                label_smoothing=label_smoothing,
            ).reshape(B, S)
            l_ec1 = (l_ec1 * mask).sum() / mask.sum().clamp(min=1)

            # L_ec2: hierarchical EC2 classification
            ec2_logits = out["ec2_logits"][:, :S]
            true_ec2 = batch["true_ec2_ids"][:, :S]
            l_ec2 = F.cross_entropy(
                ec2_logits.reshape(-1, NUM_EC2_CLASSES),
                true_ec2.reshape(-1), reduction="none",
                label_smoothing=label_smoothing,
            ).reshape(B, S)
            l_ec2 = (l_ec2 * mask).sum() / mask.sum().clamp(min=1)

            # L_cond (Huber)
            cond_preds = out["cond_preds"][:, :S]
            true_conds = batch["true_conditions"][:, :S]
            l_cond = F.smooth_l1_loss(cond_preds, true_conds, reduction="none").mean(dim=-1)
            l_cond = (l_cond * mask).sum() / mask.sum().clamp(min=1)

            # L_edit (focal)
            edit_logits = out["edit_type_logits"]
            true_edit = batch["edit_action_id"]
            l_edit_raw = focal_loss(edit_logits, true_edit, gamma=2.0)
            edit_valid = batch["edit_action_valid"]
            l_edit = (l_edit_raw * edit_valid).sum() / edit_valid.sum().clamp(min=1)

            # L_edit_target
            edit_target_logits = out["edit_target_logits"]
            true_edit_slot = batch["edit_slot"].clamp(max=S - 1)
            slot_valid = torch.arange(S, device=device).unsqueeze(0) < batch["n_slots"].unsqueeze(1)
            edit_target_logits = edit_target_logits.masked_fill(~slot_valid, -1e9)
            l_edit_target = F.cross_entropy(edit_target_logits, true_edit_slot, reduction="none")
            l_edit_target = (l_edit_target * edit_valid).sum() / edit_valid.sum().clamp(min=1)

            # L_energy (weight reduced per expert)
            energy_pred = out["energy"]
            clean_energy = clean_out["energy"]
            severity = batch["severity"]
            l_energy = 0.5 * F.smooth_l1_loss(energy_pred, severity) + 0.5 * F.smooth_l1_loss(clean_energy, torch.zeros_like(clean_energy))

            # L_rank + L_pref (now uses real compat_rank as additional signal)
            margin = 0.10
            l_rank = F.softplus(clean_energy + margin - energy_pred).mean()
            l_pref = -F.logsigmoid(energy_pred - clean_energy).mean()

            # L_rank_real: energy should correlate with real compatibility rank
            # compat_rank: 0=best → 4=worst, so higher rank = higher energy
            compat_rank = batch["compat_rank"]  # (B,)
            compat_rank_norm = compat_rank / 4.0  # normalize to 0-1
            l_rank_real = F.smooth_l1_loss(torch.sigmoid(clean_energy), compat_rank_norm)

            # L_feas
            feasibility = out["feasibility"]
            l_feas = F.smooth_l1_loss(feasibility, 1.0 - severity)

            # L_constraint
            issue_scores = out["issue_scores"][:, :S]
            issue_mask_t = batch["issue_mask"][:, :S]
            l_constraint = F.binary_cross_entropy(issue_scores, issue_mask_t)

            # ===== v3 REAL-LABEL LOSSES (replacing synthetic proxies) =====

            # L_compat: 4-level compatibility classification (replaces binary L_rank)
            compat_logits = out["compat_logits"]  # (B, 4)
            compat_label = batch["compat_label"]  # (B,)
            compat_valid = batch["compat_valid"]  # (B,)
            l_compat_raw = F.cross_entropy(compat_logits, compat_label, reduction="none", label_smoothing=label_smoothing)
            l_compat = (l_compat_raw * compat_valid).sum() / compat_valid.sum().clamp(min=1)

            # L_opmode: operation mode classification
            opmode_logits = out["opmode_logits"]  # (B, 7)
            opmode_label = batch["opmode_label"]  # (B,)
            opmode_valid = batch["opmode_valid"]  # (B,)
            l_opmode_raw = F.cross_entropy(opmode_logits, opmode_label, reduction="none", label_smoothing=label_smoothing)
            l_opmode = (l_opmode_raw * opmode_valid).sum() / opmode_valid.sum().clamp(min=1)

            # L_issue_real: multi-label issue type detection (replaces synthetic L_constraint)
            issue_type_logits = out["issue_type_logits"]  # (B, 15)
            issue_type_labels = batch["issue_type_labels"]  # (B, 15)
            issue_type_valid = batch["issue_type_valid"]  # (B,)
            l_issue_real_raw = F.binary_cross_entropy_with_logits(issue_type_logits, issue_type_labels, reduction="none").mean(dim=-1)
            l_issue_real = (l_issue_real_raw * issue_type_valid).sum() / issue_type_valid.sum().clamp(min=1)

            # L_pairwise: pairwise mode classification per adjacent step pair
            pairwise_logits = out.get("pairwise_logits")  # (B, S-1, 5) or None
            pairwise_labels = batch["pairwise_labels"][:, :S-1] if S > 1 else None  # (B, S-1)
            pairwise_valid = batch["pairwise_valid"][:, :S-1] if S > 1 else None  # (B, S-1)
            l_pairwise = torch.tensor(0.0, device=device)
            if pairwise_logits is not None and pairwise_labels is not None and S > 1:
                n_pairs = min(pairwise_logits.shape[1], pairwise_labels.shape[1])
                if n_pairs > 0:
                    pw_logits = pairwise_logits[:, :n_pairs].reshape(-1, 5)
                    pw_labels = pairwise_labels[:, :n_pairs].reshape(-1)
                    pw_valid = pairwise_valid[:, :n_pairs].reshape(-1)
                    l_pw_raw = F.cross_entropy(pw_logits, pw_labels, reduction="none")
                    l_pairwise = (l_pw_raw * pw_valid).sum() / pw_valid.sum().clamp(min=1)

            # Loss weights: real labels as auxiliary signal, edit policy remains primary
            loss = (
                1.0 * l_type
                + 0.7 * l_ec1
                + 0.5 * l_ec2
                + 0.2 * l_cond
                + 0.5 * l_edit
                + 0.5 * l_edit_target
                + 0.3 * l_rank
                + 0.3 * l_pref
                + 0.3 * l_rank_real     # NEW: energy calibrated to real compatibility
                + 0.1 * l_feas
                + 0.2 * l_constraint
                + 0.3 * l_compat        # real: reduced from 0.8
                + 0.2 * l_opmode        # real: reduced from 0.5
                + 0.2 * l_issue_real    # real: reduced from 0.5
                + 0.2 * l_pairwise      # real: reduced from 0.5
            )

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            train_loss_sum += loss.item()
            n_batches += 1
            for lname, lval in [
                ("type", l_type), ("ec1", l_ec1), ("cond", l_cond),
                ("edit", l_edit), ("edit_tgt", l_edit_target),
                ("energy", l_energy), ("rank", l_rank), ("pref", l_pref),
                ("rank_real", l_rank_real),
                ("feas", l_feas), ("constr", l_constraint),
                ("compat", l_compat), ("opmode", l_opmode),
                ("issue_real", l_issue_real), ("pairwise", l_pairwise),
            ]:
                loss_accum[lname] = loss_accum.get(lname, 0.0) + lval.item()

        avg_train = train_loss_sum / max(n_batches, 1)

        # Validate
        model.eval()
        val_loss_sum = 0
        val_type_correct = val_type_total = 0
        val_edit_correct = val_edit_total = 0
        val_target_correct = val_target_total = 0
        val_rank_correct = val_rank_total = 0
        n_val = 0

        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(
                    target_fp=batch["target_fp"],
                    objective_ids=batch["objective_id"],
                    slot_type_ids=batch["type_ids"],
                    slot_ec1_ids=batch["ec1_ids"],
                    slot_conditions=batch["conditions"],
                    slot_scores=batch["scores"],
                    slot_is_fixed=batch["is_fixed"],
                )
                clean_out = model(
                    target_fp=batch["target_fp"],
                    objective_ids=batch["objective_id"],
                    slot_type_ids=batch["true_type_ids"],
                    slot_ec1_ids=batch["true_ec1_ids"],
                    slot_conditions=batch["true_conditions"],
                    slot_scores=batch["true_scores"],
                    slot_is_fixed=batch["true_is_fixed"],
                )
                mask = batch["mask"]
                B, S = mask.shape

                # Val type acc
                type_logits = out["type_logits"][:, :S]
                true_types = batch["true_type_ids"][:, :S]
                l_type = F.cross_entropy(
                    type_logits.reshape(-1, NUM_REACTION_TYPES),
                    true_types.reshape(-1), reduction="none",
                ).reshape(B, S)
                l_type = (l_type * mask).sum() / mask.sum().clamp(min=1)
                val_loss_sum += l_type.item()

                pred_types = type_logits.argmax(dim=-1)
                correct = ((pred_types == true_types) & (mask > 0)).sum().item()
                total = mask.sum().item()
                val_type_correct += correct
                val_type_total += total

                # Val edit acc
                edit_logits = out["edit_type_logits"]
                pred_edit = edit_logits.argmax(dim=-1)
                true_edit = batch["edit_action_id"]
                ev = batch["edit_action_valid"]
                val_edit_correct += ((pred_edit == true_edit) * ev).sum().item()
                val_edit_total += ev.sum().item()

                # Val target acc
                et_logits = out["edit_target_logits"]
                pred_tgt = et_logits.argmax(dim=-1)
                true_tgt = batch["edit_slot"].clamp(max=S - 1)
                val_target_correct += ((pred_tgt == true_tgt) * ev).sum().item()
                val_target_total += ev.sum().item()

                # Val rank acc
                e_pred = out["energy"]
                e_clean = clean_out["energy"]
                val_rank_correct += (e_pred > e_clean).float().sum().item()
                val_rank_total += B

                n_val += 1

        avg_val = val_loss_sum / max(n_val, 1)
        val_acc = val_type_correct / max(val_type_total, 1)
        val_edit_acc = val_edit_correct / max(val_edit_total, 1)
        val_target_acc = val_target_correct / max(val_target_total, 1)
        val_rank_acc = val_rank_correct / max(val_rank_total, 1)

        per_loss = {f"train_{k}": round(v / n_batches, 4) for k, v in loss_accum.items()} if n_batches > 0 else {}

        history.append({
            "epoch": epoch, "train_loss": round(avg_train, 4),
            "val_loss": round(avg_val, 4), "val_type_acc": round(val_acc, 4),
            "val_edit_acc": round(val_edit_acc, 4),
            "val_edit_target_acc": round(val_target_acc, 4),
            "val_route_rank_acc": round(val_rank_acc, 4),
            "global_step": global_step,
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            **per_loss,
        })

        if (epoch + 1) % max(log_every, 1) == 0 or epoch == 0:
            print(
                f"  epoch {epoch+1:3d} (step {global_step:5d}): "
                f"train={avg_train:.4f} val={avg_val:.4f} "
                f"type={val_acc:.3f} edit={val_edit_acc:.3f} "
                f"tgt={val_target_acc:.3f} rank={val_rank_acc:.3f} "
                f"lr={optimizer.param_groups[0]['lr']:.1e}"
            )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path / "best.pt")

    torch.save(model.state_dict(), save_path / "final.pt")
    (save_path / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to {save_path}")
    return model, history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    from collections import Counter
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cascade_dataset_v2.normalized.json")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--n-mask", type=int, default=20)
    ap.add_argument("--n-corrupt", type=int, default=10)
    ap.add_argument("--save-dir", default="results/shared/cascadeboard_model")
    ap.add_argument("--summary", default="results/shared/training_summary.json")
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--curriculum", action="store_true", help="Enable curriculum learning (easy→hard phasing)")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--stratified-split", action="store_true", help="Stratified DOI split by route_domain")
    ap.add_argument("--online", action="store_true", help="Online augmentation: 1 mask + 1 corrupt per route per epoch")
    ap.add_argument("--fp-drop", type=float, default=0.0, help="Target FP dropout")
    ap.add_argument("--emb-drop", type=float, default=0.0, help="Embedding dropout")
    ap.add_argument("--bottleneck", type=int, default=0, help="Route latent bottleneck dim (0=off)")
    ap.add_argument("--d-model", type=int, default=256, help="Model dimension")
    ap.add_argument("--n-layers", type=int, default=4, help="Transformer layers")
    args = ap.parse_args()

    routes = load_cascade_routes(args.data)

    if args.online:
        # Online mode: no pre-materialized samples, use OnlineRouteDataset
        print(f"Online augmentation mode: {len(routes)} routes, ~2 samples/route/epoch")
        # Still need DOI split at route level
        if args.stratified_split:
            from collections import defaultdict
            doi_domain = defaultdict(lambda: Counter())
            for r in routes:
                doi_domain[r.get("doi", "")][r.get("route_domain", "")] += 1
            doi_primary = {doi: counts.most_common(1)[0][0] for doi, counts in doi_domain.items() if counts}
            domain_dois = defaultdict(list)
            for doi, domain in sorted(doi_primary.items()):
                domain_dois[domain].append(doi)
            val_dois = set()
            for domain, d_list in domain_dois.items():
                n_val = max(1, len(d_list) // 5)
                val_dois.update(d_list[:n_val])
        else:
            all_dois = sorted(set(r.get("doi", "") for r in routes))
            val_dois = set(all_dois[:len(all_dois) // 5])

        train_routes = [r for r in routes if r.get("doi", "") not in val_dois]
        val_routes = [r for r in routes if r.get("doi", "") in val_dois]
        print(f"  Train routes: {len(train_routes)}, Val routes: {len(val_routes)}")

        # Build small val set (materialized, for stable metrics)
        val_samples = build_training_data(val_routes, n_mask_per_route=3, n_corrupt_per_route=2)

        summary = {
            "mode": "online",
            "n_routes": len(routes),
            "n_train_routes": len(train_routes),
            "n_val_routes": len(val_routes),
            "n_val_samples": len(val_samples),
        }
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))

        # Load candidate cache for candidate attention
        from cascade_planner.cascadeboard.planner import _load_real_cache
        cand_cache = _load_real_cache()
        print(f"  Candidate cache: {len(cand_cache)} products")

        train_ds = OnlineRouteDataset(train_routes, candidate_cache=cand_cache)
        val_ds_obj = CascadeBoardDataset(val_samples)

        train_cascadeboard_online(
            train_ds=train_ds,
            val_ds=val_ds_obj,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            save_dir=args.save_dir,
            log_every=args.log_every,
            dropout=args.dropout,
            label_smoothing=args.label_smoothing,
            grad_clip=args.grad_clip,
            weight_decay=args.weight_decay,
            target_fp_dropout=args.fp_drop,
            embedding_dropout=args.emb_drop,
            route_latent_dim=args.bottleneck,
            d_model=args.d_model,
            n_layers=args.n_layers,
        )
        return

    samples = build_training_data(
        routes,
        n_mask_per_route=args.n_mask,
        n_corrupt_per_route=args.n_corrupt,
    )

    summary = {
        "n_routes": len(routes),
        "n_samples": len(samples),
        "n_mask_per_route": args.n_mask,
        "n_corrupt_per_route": args.n_corrupt,
        "curriculum": args.curriculum,
        "types": dict(Counter(s.corruption_type for s in samples)),
        "difficulty_distribution": dict(Counter(s.difficulty for s in samples)),
        "edit_actions": dict(Counter(s.edit_action or "FILL_FIELD" for s in samples)),
        "training_objectives": [
            "masked_inpainting",
            "edit_action_classification",
            "edit_target_classification",
            "issue_slot_detection",
            "clean_vs_corrupted_route_ranking",
            "synthetic_bradley_terry_preference",
            "route_ok_energy_feasibility_calibration",
        ],
        "notes": [
            "route_ok samples mask edit/action loss because EditType has no NO_OP class",
            "delete_step corruption is oversampled to balance INSERT_STEP repair labels",
            "replace_ec and replace_enzyme are separate corruption labels but share REPLACE_ENZYME action class",
        ],
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    # Split by DOI — optionally stratified by route_domain
    if args.stratified_split:
        # Group DOIs by their dominant route_domain
        from collections import defaultdict
        doi_domain = defaultdict(lambda: Counter())
        for r in routes:
            doi_domain[r.get("doi", "")][r.get("route_domain", "")] += 1
        doi_primary = {doi: counts.most_common(1)[0][0] for doi, counts in doi_domain.items() if counts}

        domain_dois = defaultdict(list)
        for doi, domain in sorted(doi_primary.items()):
            domain_dois[domain].append(doi)

        val_dois = set()
        for domain, d_list in domain_dois.items():
            n_val = max(1, len(d_list) // 5)
            val_dois.update(d_list[:n_val])
        print(f"  Stratified split: {len(val_dois)} val DOIs across {len(domain_dois)} domains")
    else:
        dois = sorted(set(s.source_doi for s in samples))
        val_dois = set(dois[:len(dois) // 5])

    train = [s for s in samples if s.source_doi not in val_dois]
    val = [s for s in samples if s.source_doi in val_dois]

    train_cascadeboard(
        train,
        val,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        save_dir=args.save_dir,
        log_every=args.log_every,
        curriculum=args.curriculum,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
    )


if __name__ == "__main__":
    main()
