"""Route Encoder + Edit Policy for CascadeBoard.

Layer 3+4: Slot Chain Transformer with ROUTE_LATENT token,
edit action heads, and inpainting heads.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit import DataStructs

from cascade_planner.cascadeboard import CascadeBoard, Slot, EditType

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_REACTION_TYPES = 20  # 19 types + NONE
NUM_EC1_CLASSES = 8      # 1-7 + NONE
NUM_EC2_CLASSES = 38     # top EC2 classes with >=10 occurrences + NONE
NUM_EDIT_TYPES = len(EditType)

EC2_VOCAB = [
    "NONE", "1.1", "3.1", "2.6", "2.4", "2.7", "1.14", "4.1", "1.4", "1.11",
    "3.2", "1.2", "4.2", "3.5", "1.6", "1.10", "2.1", "1.5", "3.4", "5.3",
    "5.1", "1.3", "4.3", "2.3", "1.13", "3.7", "2.2", "6.2", "5.4", "3.8",
    "6.1", "1.7", "2.5", "3.3", "6.3", "4.4", "3.6", "5.2",
]
EC2_TO_ID = {v: i for i, v in enumerate(EC2_VOCAB)}

# v3 real-label vocabularies
COMPAT_VOCAB = ["empirically_compatible", "compatible_with_mitigation", "compatible_with_compromise", "unclear"]
COMPAT_TO_ID = {v: i for i, v in enumerate(COMPAT_VOCAB)}

OPMODE_VOCAB = [
    "one_pot_simultaneous", "one_pot_sequential_addition", "sequential_isolated",
    "continuous_flow", "telescoped_no_isolation", "compartmentalized", "other",
]
OPMODE_TO_ID = {v: i for i, v in enumerate(OPMODE_VOCAB)}

ISSUE_TYPE_VOCAB = [
    "product_inhibition", "ph_window_mismatch", "metal_enzyme_conflict",
    "stability_decay", "substrate_inhibition", "temperature_window_mismatch",
    "solvent_incompatibility", "side_reaction_due_to_support", "cofactor_cross_talk",
    "oxygen_requirement_conflict", "overoxidation", "water_sensitivity",
    "expression_burden", "reversibility_issue", "unknown",
]
ISSUE_TYPE_TO_IDX = {v: i for i, v in enumerate(ISSUE_TYPE_VOCAB)}

PAIRWISE_VOCAB = ["simultaneous", "sequential_addition", "telescoped", "isolated_transfer", "compartmentalized"]
PAIRWISE_TO_ID = {v: i for i, v in enumerate(PAIRWISE_VOCAB)}

DOMAIN_VOCAB = ["all_enzymatic", "chemoenzymatic", "all_chemical", "whole_cell_biocatalytic", "hybrid_mimetic", "unknown"]
DOMAIN_TO_ID = {v: i for i, v in enumerate(DOMAIN_VOCAB)}
MAX_SLOTS = 8
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 4
D_SLOT_INPUT = 128       # per-slot raw feature dim before projection


# ---------------------------------------------------------------------------
# Slot Encoder: multi-modal embedding per slot
# ---------------------------------------------------------------------------

class SlotEncoder(nn.Module):
    """Encode a single slot into a d_model vector."""

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.type_emb = nn.Embedding(NUM_REACTION_TYPES, 32)
        self.ec1_emb = nn.Embedding(NUM_EC1_CLASSES, 32)
        self.cond_proj = nn.Linear(2, 16)   # [T_norm, pH_norm]
        self.score_proj = nn.Linear(3, 16)  # [e_retro, e_enzyme, e_condition]
        self.fixed_emb = nn.Embedding(2, 8) # is_fixed indicator
        self.proj = nn.Linear(32 + 32 + 16 + 16 + 8, d_model)

    def forward(
        self,
        type_ids: torch.Tensor,     # (B, S) int
        ec1_ids: torch.Tensor,      # (B, S) int
        conditions: torch.Tensor,   # (B, S, 2) float [T_norm, pH_norm]
        scores: torch.Tensor,       # (B, S, 3) float [e_retro, e_enzyme, e_cond]
        is_fixed: torch.Tensor,     # (B, S) int {0,1}
    ) -> torch.Tensor:
        """Returns (B, S, d_model)."""
        t = self.type_emb(type_ids)
        e = self.ec1_emb(ec1_ids)
        c = self.cond_proj(conditions)
        s = self.score_proj(scores)
        f = self.fixed_emb(is_fixed)
        return self.proj(torch.cat([t, e, c, s, f], dim=-1))


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

class SpecialTokens(IntEnum):
    ROUTE_LATENT = 0
    TARGET = 1
    OBJECTIVE = 2
    # Slots start at index 3


# ---------------------------------------------------------------------------
# CascadeBoardTransformer
# ---------------------------------------------------------------------------

class CascadeBoardTransformer(nn.Module):
    """Slot Chain Transformer with ROUTE_LATENT, edit policy, and inpainting heads.

    Input tokens: [ROUTE_LATENT, TARGET, OBJECTIVE, SLOT_0, ..., SLOT_n]
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        max_slots: int = MAX_SLOTS,
        target_fp_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        route_latent_dim: int = 0,
    ):
        super().__init__()
        self.d_model = d_model

        # Special token embeddings
        self.route_latent_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.target_proj = nn.Linear(2048, d_model)  # Morgan FP → d_model
        self.target_fp_drop = nn.Dropout(target_fp_dropout) if target_fp_dropout > 0 else nn.Identity()
        self.objective_emb = nn.Embedding(4, d_model)  # balanced/industrial/green/novelty
        self.domain_emb = nn.Embedding(6, d_model)    # all_enzymatic/chemoenzymatic/all_chemical/whole_cell/hybrid/unknown
        self.sm_proj = nn.Linear(2048, d_model)        # starting material Morgan FP → d_model

        # Slot encoder
        self.slot_encoder = SlotEncoder(d_model)

        # Input embedding dropout (applied to all token embeddings before transformer)
        self.emb_drop = nn.Dropout(embedding_dropout) if embedding_dropout > 0 else nn.Identity()

        # Positional encoding for slots
        self.pos_emb = nn.Embedding(max_slots + 3, d_model)  # +3 for special tokens

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Route latent bottleneck (optional, per expert §11.3)
        self._use_bottleneck = route_latent_dim > 0
        if self._use_bottleneck:
            self.route_bottleneck = nn.Sequential(
                nn.Linear(d_model, route_latent_dim),
                nn.GELU(),
                nn.Linear(route_latent_dim, d_model),
            )

        # ----- Output heads -----

        # From ROUTE_LATENT: global heads
        self.energy_head = nn.Linear(d_model, 1)
        self.feasibility_head = nn.Linear(d_model, 1)
        self.edit_type_head = nn.Linear(d_model, NUM_EDIT_TYPES)
        self.edit_target_head = nn.Linear(d_model, max_slots)
        self.issue_head = nn.Linear(d_model, max_slots)  # per-slot issue score

        # v3 real-label heads (from ROUTE_LATENT)
        NUM_COMPAT_CLASSES = 4   # empirically_compatible / with_mitigation / with_compromise / unclear+incompatible
        NUM_OPMODE_CLASSES = 7   # one_pot_simultaneous / one_pot_sequential / sequential_isolated / continuous_flow / telescoped / compartmentalized / other
        NUM_ISSUE_TYPES = 15     # multi-label: ph_mismatch, T_mismatch, product_inhibition, etc.
        NUM_PAIRWISE_CLASSES = 5 # simultaneous / sequential_addition / telescoped / isolated_transfer / compartmentalized
        self.compat_head = nn.Linear(d_model, NUM_COMPAT_CLASSES)
        self.opmode_head = nn.Linear(d_model, NUM_OPMODE_CLASSES)
        self.issue_type_head = nn.Linear(d_model, NUM_ISSUE_TYPES)  # multi-label sigmoid
        self.pairwise_head = nn.Linear(d_model * 2, NUM_PAIRWISE_CLASSES)  # concat adjacent slot pairs

        # From each SLOT: inpainting heads
        self.type_head = nn.Linear(d_model, NUM_REACTION_TYPES)
        self.ec1_head = nn.Linear(d_model, NUM_EC1_CLASSES)
        self.ec2_head = nn.Linear(d_model, NUM_EC2_CLASSES)  # hierarchical EC2
        self.cond_head = nn.Linear(d_model, 2)  # T, pH
        self.yield_head = nn.Linear(d_model, 1)  # predicted yield (0-100%)
        self.ee_head = nn.Linear(d_model, 1)     # predicted ee (0-100%)
        self.candidate_query = nn.Linear(d_model, d_model)  # for attention over candidates

        # Candidate cross-attention: slot attends over its candidate pool
        # Candidate features: [score, type_id one-hot(20), ec1 one-hot(8)] = 29-dim
        self.cand_feat_dim = 29
        self.cand_key_proj = nn.Linear(self.cand_feat_dim, d_model)
        self.cand_attn_out = nn.Linear(d_model, d_model)

    def _candidate_cross_attention(
        self,
        slot_out: torch.Tensor,          # (B, S, d)
        candidate_features: torch.Tensor, # (B, S, K, cand_feat_dim)
        candidate_mask: torch.Tensor,     # (B, S, K) bool, True=valid candidate
    ) -> torch.Tensor:
        """Cross-attention from slot states to candidate features. Returns enriched slot states."""
        B, S, d = slot_out.shape
        K = candidate_features.shape[2]

        queries = self.candidate_query(slot_out)  # (B, S, d)
        keys = self.cand_key_proj(candidate_features)  # (B, S, K, d)

        # Scaled dot-product attention per slot
        attn_logits = torch.einsum("bsd,bskd->bsk", queries, keys) / (d ** 0.5)  # (B, S, K)
        attn_logits = attn_logits.masked_fill(~candidate_mask, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=-1)  # (B, S, K)

        # Weighted sum of candidate keys as values
        attn_out = torch.einsum("bsk,bskd->bsd", attn_weights, keys)  # (B, S, d)
        return slot_out + self.cand_attn_out(attn_out)  # residual connection

    def forward(
        self,
        target_fp: torch.Tensor,      # (B, 2048)
        objective_ids: torch.Tensor,   # (B,) int
        slot_type_ids: torch.Tensor,   # (B, S) int
        slot_ec1_ids: torch.Tensor,    # (B, S) int
        slot_conditions: torch.Tensor, # (B, S, 2)
        slot_scores: torch.Tensor,     # (B, S, 3)
        slot_is_fixed: torch.Tensor,   # (B, S) int
        slot_mask: torch.Tensor | None = None,
        candidate_features: torch.Tensor | None = None,  # (B, S, K, 29)
        candidate_mask: torch.Tensor | None = None,       # (B, S, K) bool
        domain_ids: torch.Tensor | None = None,           # (B,) int — route_domain
        sm_fp: torch.Tensor | None = None,                # (B, 2048) — starting material FP
    ) -> dict[str, torch.Tensor]:
        B, S = slot_type_ids.shape

        # Encode special tokens
        route_latent = self.route_latent_emb.expand(B, -1, -1)  # (B, 1, d)
        target_emb = self.target_proj(self.target_fp_drop(target_fp)).unsqueeze(1)  # (B, 1, d)
        obj_emb = self.objective_emb(objective_ids).unsqueeze(1)  # (B, 1, d)

        # New context tokens: domain + starting material
        context_tokens = []
        if domain_ids is not None:
            context_tokens.append(self.domain_emb(domain_ids).unsqueeze(1))
        if sm_fp is not None:
            context_tokens.append(self.sm_proj(sm_fp).unsqueeze(1))

        # Encode slots
        slot_embs = self.slot_encoder(
            slot_type_ids, slot_ec1_ids, slot_conditions, slot_scores, slot_is_fixed,
        )  # (B, S, d)

        # Concatenate: [ROUTE_LATENT, TARGET, OBJECTIVE, (DOMAIN?), (SM?), SLOT_0, ..., SLOT_n]
        parts = [route_latent, target_emb, obj_emb] + context_tokens + [slot_embs]
        tokens = torch.cat(parts, dim=1)
        n_prefix = 3 + len(context_tokens)

        # Embedding dropout
        tokens = self.emb_drop(tokens)

        # Positional encoding (dynamic size based on context tokens)
        n_tokens = n_prefix + S
        positions = torch.arange(n_tokens, device=tokens.device).unsqueeze(0).expand(B, -1)
        positions = positions.clamp(max=self.pos_emb.num_embeddings - 1)
        tokens = tokens + self.pos_emb(positions)

        # Transformer forward
        out = self.transformer(tokens)  # (B, 3+S, d)

        # Split outputs — slot offset depends on context tokens
        route_latent_out = out[:, 0, :]           # (B, d)
        slot_out = out[:, n_prefix:, :]           # (B, S, d)

        # Optional route latent bottleneck
        if self._use_bottleneck:
            route_latent_out = route_latent_out + self.route_bottleneck(route_latent_out)

        # Optional candidate cross-attention: enrich slot states with candidate pool info
        if candidate_features is not None and candidate_mask is not None:
            slot_out = self._candidate_cross_attention(slot_out, candidate_features, candidate_mask)

        # ----- Global heads (from ROUTE_LATENT) -----
        energy = self.energy_head(route_latent_out).squeeze(-1)          # (B,)
        feasibility = torch.sigmoid(self.feasibility_head(route_latent_out).squeeze(-1))
        edit_type_logits = self.edit_type_head(route_latent_out)          # (B, NUM_EDIT)
        edit_target_logits = self.edit_target_head(route_latent_out)[:, :S]  # (B, S)
        issue_scores = torch.sigmoid(self.issue_head(route_latent_out)[:, :S])  # (B, S)

        # v3 real-label heads (from ROUTE_LATENT)
        compat_logits = self.compat_head(route_latent_out)               # (B, 4)
        opmode_logits = self.opmode_head(route_latent_out)               # (B, 7)
        issue_type_logits = self.issue_type_head(route_latent_out)       # (B, 15)

        # Pairwise head: concat adjacent slot pairs → predict pairwise_mode
        pairwise_logits = None
        if S >= 2:
            left = slot_out[:, :-1, :]   # (B, S-1, d)
            right = slot_out[:, 1:, :]   # (B, S-1, d)
            pair_feats = torch.cat([left, right], dim=-1)  # (B, S-1, 2d)
            pairwise_logits = self.pairwise_head(pair_feats)  # (B, S-1, 5)

        # ----- Per-slot heads -----
        type_logits = self.type_head(slot_out)       # (B, S, NUM_TYPES)
        ec1_logits = self.ec1_head(slot_out)         # (B, S, NUM_EC1)
        ec2_logits = self.ec2_head(slot_out)         # (B, S, NUM_EC2)
        cond_preds = self.cond_head(slot_out)        # (B, S, 2) [T, pH]
        yield_preds = torch.sigmoid(self.yield_head(slot_out).squeeze(-1)) * 100  # (B, S) 0-100%
        ee_preds = torch.sigmoid(self.ee_head(slot_out).squeeze(-1)) * 100        # (B, S) 0-100%
        cand_queries = self.candidate_query(slot_out) # (B, S, d) for candidate attention

        return {
            "energy": energy,
            "feasibility": feasibility,
            "edit_type_logits": edit_type_logits,
            "edit_target_logits": edit_target_logits,
            "issue_scores": issue_scores,
            "compat_logits": compat_logits,
            "opmode_logits": opmode_logits,
            "issue_type_logits": issue_type_logits,
            "pairwise_logits": pairwise_logits,
            "type_logits": type_logits,
            "ec1_logits": ec1_logits,
            "ec2_logits": ec2_logits,
            "cond_preds": cond_preds,
            "yield_preds": yield_preds,
            "ee_preds": ee_preds,
            "cand_queries": cand_queries,
            "route_latent": route_latent_out,
            "slot_states": slot_out,
        }


# ---------------------------------------------------------------------------
# Board → Tensor conversion
# ---------------------------------------------------------------------------

REACTION_TYPE_TO_ID = {
    None: 0, "": 0,
    "oxidation": 1, "reduction": 2, "acylation": 3, "hydrolysis": 4,
    "amination": 5, "C_C_coupling": 6, "isomerization": 7,
    "phosphorylation": 8, "glycosylation": 9,
    "functional_group_interconversion": 10, "racemization": 11,
    "esterification": 12, "other": 13, "resolution": 14, "cofactor_regeneration": 15,
    "dehalogenation": 16, "amidation": 17, "epoxide_hydrolysis": 18,
}

OBJECTIVE_TO_ID = {"balanced": 0, "industrial": 1, "green": 2, "novelty": 3}
CONSTRAINT_FEATURE_NAMES = (
    "fixed_slot_frac",
    "fixed_ec_frac",
    "fixed_starting_material",
    "one_pot",
    "max_delta_T",
    "max_delta_pH",
    "prefer_enzymatic",
    "has_exclusions",
)


def smiles_to_morgan_fp(smiles: str | None, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """Convert SMILES to a Morgan fingerprint, falling back to zeros if invalid."""
    fp = np.zeros(n_bits, dtype=np.float32)
    if not smiles:
        return fp
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return fp
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(bitvect, fp)
    return fp


def _fixed_field_names(fixed: object) -> set[str]:
    if fixed is None:
        return set()
    if isinstance(fixed, set):
        return {str(x) for x in fixed}
    if isinstance(fixed, (list, tuple)):
        return {str(x) for x in fixed}
    return set()


def constraint_features_from_slot_dicts(
    slots: list[dict],
    global_constraints: dict | None = None,
) -> np.ndarray:
    """Encode fixed-field/global constraints into a compact feature vector.

    The vector is later written into reserved target fingerprint positions so the
    existing model architecture and old checkpoints remain shape-compatible.
    """
    n_slots = max(len(slots), 1)
    fixed_sets = [_fixed_field_names(s.get("fixed")) for s in slots]
    global_constraints = global_constraints or {}
    has_exclusions = any(
        k in global_constraints
        for k in ("exclude_catalyst", "exclude_solvent", "exclude_ec")
    )
    max_delta_t = float(global_constraints.get("max_delta_T") or 0.0)
    max_delta_ph = float(global_constraints.get("max_delta_pH") or 0.0)
    return np.array([
        sum(bool(fs) for fs in fixed_sets) / n_slots,
        sum("ec" in fs for fs in fixed_sets) / n_slots,
        1.0 if fixed_sets and "main_reactant" in fixed_sets[-1] else 0.0,
        1.0 if global_constraints.get("one_pot") else 0.0,
        min(max_delta_t / 100.0, 1.0),
        min(max_delta_ph / 5.0, 1.0),
        1.0 if global_constraints.get("prefer_enzymatic") else 0.0,
        1.0 if has_exclusions else 0.0,
    ], dtype=np.float32)


def constraint_features_from_board(board: CascadeBoard) -> np.ndarray:
    slots = [s.to_dict() for s in board.slots]
    return constraint_features_from_slot_dicts(slots, board.global_constraints)


def inject_constraint_features(
    target_fp: np.ndarray,
    constraint_features: np.ndarray | None,
) -> np.ndarray:
    """Write constraint features into reserved tail positions of a 2048-bit FP."""
    fp = np.array(target_fp, dtype=np.float32, copy=True)
    if constraint_features is None or len(constraint_features) == 0:
        return fp
    n = min(len(constraint_features), len(fp))
    fp[-n:] = constraint_features[:n]
    return fp


def board_to_tensors(
    board: CascadeBoard,
    target_fp: np.ndarray | None = None,
    objective: str = "balanced",
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Convert a CascadeBoard to model input tensors."""
    S = board.n_steps

    # Target fingerprint (placeholder if not provided)
    if target_fp is None:
        target_smiles = board.slots[0].product if board.slots else None
        target_fp = smiles_to_morgan_fp(target_smiles)
    target_fp = inject_constraint_features(target_fp, constraint_features_from_board(board))

    type_ids = []
    ec1_ids = []
    conditions = []
    scores = []
    is_fixed = []

    for slot in board.slots:
        type_ids.append(REACTION_TYPE_TO_ID.get(slot.reaction_type, 0))
        ec1 = int(slot.ec.split(".")[0]) if slot.ec and slot.ec[0].isdigit() else 0
        ec1_ids.append(ec1)
        T_norm = (slot.T - 37) / 30 if slot.T is not None else 0.0
        pH_norm = (slot.pH - 7) / 3 if slot.pH is not None else 0.0
        conditions.append([T_norm, pH_norm])
        scores.append([
            slot.e_retro or 0.0,
            slot.e_enzyme or 0.0,
            slot.e_condition or 0.0,
        ])
        is_fixed.append(1 if slot.fixed_fields else 0)

    return {
        "target_fp": torch.tensor(target_fp, dtype=torch.float32, device=device).unsqueeze(0),
        "objective_ids": torch.tensor([OBJECTIVE_TO_ID.get(objective, 0)], dtype=torch.long, device=device),
        "slot_type_ids": torch.tensor([type_ids], dtype=torch.long, device=device),
        "slot_ec1_ids": torch.tensor([ec1_ids], dtype=torch.long, device=device),
        "slot_conditions": torch.tensor([conditions], dtype=torch.float32, device=device),
        "slot_scores": torch.tensor([scores], dtype=torch.float32, device=device),
        "slot_is_fixed": torch.tensor([is_fixed], dtype=torch.long, device=device),
    }


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
