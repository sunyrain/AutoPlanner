"""Cascade Skeleton Inpainter — OA-ARM Transformer.

Order-Agnostic Autoregressive Model for cascade route skeleton generation.
Trained with random permutation order; at inference, fixed slots are "observed"
and remaining slots are generated one-by-one conditioned on all known context.

Architecture: 6-layer Transformer Decoder, d=256, 8 heads, ~8M params.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter
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

# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

REACTION_TYPE_VOCAB = [
    "NONE", "oxidation", "reduction", "acylation", "hydrolysis",
    "amination", "C_C_coupling", "isomerization", "phosphorylation",
    "glycosylation", "functional_group_interconversion", "racemization",
    "esterification", "other", "resolution", "cofactor_regeneration",
    "dehalogenation", "amidation", "epoxide_hydrolysis",
]
RTYPE_TO_ID = {v: i for i, v in enumerate(REACTION_TYPE_VOCAB)}
NUM_RTYPES = len(REACTION_TYPE_VOCAB)

NUM_EC1 = 8  # 0=no enzyme, 1-7

EC2_VOCAB = [
    "NONE", "1.1", "3.1", "2.6", "2.4", "2.7", "1.14", "4.1", "1.4", "1.11",
    "3.2", "1.2", "4.2", "3.5", "1.6", "1.10", "2.1", "1.5", "3.4", "5.3",
    "5.1", "1.3", "4.3", "2.3", "1.13", "3.7", "2.2", "6.2", "5.4", "3.8",
    "6.1", "1.7", "2.5", "3.3", "6.3", "4.4", "3.6", "5.2",
]
EC2_TO_ID = {v: i for i, v in enumerate(EC2_VOCAB)}
NUM_EC2 = len(EC2_VOCAB)

STEP_ROLE_VOCAB = [
    "NONE", "productive_transformation", "racemization", "kinetic_resolution",
    "cofactor_regeneration", "deracemization", "activation", "deprotection",
    "reporter_conversion", "equilibrium_shift", "signal_amplification", "protection",
]
STEP_ROLE_TO_ID = {v: i for i, v in enumerate(STEP_ROLE_VOCAB)}

STEP_TYPE_VOCAB = ["NONE", "productive", "racemization", "cofactor_regeneration", "workup"]
STEP_TYPE_TO_ID = {v: i for i, v in enumerate(STEP_TYPE_VOCAB)}

STEP_MODE_VOCAB = ["NONE", "charged_at_t0", "sequential_addition", "triggered_by_condition_shift", "unknown"]
STEP_MODE_TO_ID = {v: i for i, v in enumerate(STEP_MODE_VOCAB)}

ATMOSPHERE_VOCAB = ["NONE", "aerobic", "inert_N2", "inert_Ar", "H2", "O2_enriched", "anaerobic", "other"]
ATMO_TO_ID = {v: i for i, v in enumerate(ATMOSPHERE_VOCAB)}

CATALYST_CLASS_VOCAB = [
    "NONE", "enzyme", "metal_catalyst", "organocatalyst", "reagent_mediated",
    "acid_base_catalyst", "whole_cell", "photocatalyst", "other",
]
CATCLASS_TO_ID = {v: i for i, v in enumerate(CATALYST_CLASS_VOCAB)}

ENGINEERING_VOCAB = ["NONE", "wild_type", "engineered_mutant", "directed_evolution_variant",
                     "coexpressed_system", "fusion_construct", "unknown"]
ENG_TO_ID = {v: i for i, v in enumerate(ENGINEERING_VOCAB)}

BIOCAT_FORMAT_VOCAB = ["NONE", "purified_enzyme", "immobilized_enzyme", "whole_cell_resting",
                       "crude_lysate", "fusion_enzyme", "whole_cell_growing", "unknown"]
BIOFMT_TO_ID = {v: i for i, v in enumerate(BIOCAT_FORMAT_VOCAB)}

PROCESS_TAGS_VOCAB = [
    "dkr", "cofactor_regeneration_coupled", "multi_enzyme_module",
    "deracemization", "immobilized_cascade", "non_preparative_application",
    "kinetic_resolution", "whole_cell_regeneration",
]
PTAG_TO_ID = {v: i for i, v in enumerate(PROCESS_TAGS_VOCAB)}
NUM_PTAGS = len(PROCESS_TAGS_VOCAB)

COFACTOR_MODE_VOCAB = ["NONE", "GDH", "FDH", "substrate-coupled", "enzyme-coupled",
                       "electrochemical", "photochemical", "other"]
COFMODE_TO_ID = {v: i for i, v in enumerate(COFACTOR_MODE_VOCAB)}

DOMAIN_VOCAB = ["all_enzymatic", "chemoenzymatic", "all_chemical", "other"]
DOMAIN_TO_ID = {v: i for i, v in enumerate(DOMAIN_VOCAB)}

OBJECTIVE_VOCAB = ["balanced", "industrial", "green", "novelty"]
OBJ_TO_ID = {v: i for i, v in enumerate(OBJECTIVE_VOCAB)}

COMPAT_VOCAB = ["empirically_compatible", "compatible_with_mitigation",
                "compatible_with_compromise", "unclear"]
COMPAT_TO_ID = {v: i for i, v in enumerate(COMPAT_VOCAB)}

OPMODE_VOCAB = [
    "one_pot_simultaneous", "one_pot_sequential_addition", "sequential_isolated",
    "continuous_flow", "telescoped_no_isolation", "compartmentalized", "batch_other",
]
OPMODE_TO_ID = {v: i for i, v in enumerate(OPMODE_VOCAB)}

ISSUE_TYPE_VOCAB = [
    "product_inhibition", "ph_window_mismatch", "metal_enzyme_conflict",
    "stability_decay", "substrate_inhibition", "temperature_window_mismatch",
    "solvent_incompatibility", "side_reaction_due_to_support", "cofactor_cross_talk",
    "oxygen_requirement_conflict", "overoxidation", "water_sensitivity",
    "expression_burden", "reversibility_issue", "unknown",
]
NUM_ISSUES = len(ISSUE_TYPE_VOCAB)

MAX_SLOTS = 8

# Evidence strength → loss weight
EVIDENCE_WEIGHT = {
    "strong_process_evidence": 1.0,
    "weak_process_evidence": 0.5,
    "workflow_only": 0.3,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def morgan_fp(smi: str, nbits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
    return np.array(fp, dtype=np.float32)


def _normalize_atmosphere(raw: str | None) -> str:
    if not raw:
        return "NONE"
    r = raw.lower()
    if "aerobic" in r or "air" in r:
        return "aerobic"
    if "n2" in r or "nitrogen" in r or "inert_n2" in r:
        return "inert_N2"
    if "ar" in r or "argon" in r or "inert_ar" in r:
        return "inert_Ar"
    if "h2" in r and "co" not in r:
        return "H2"
    if "o2" in r:
        return "O2_enriched"
    if "anaerobic" in r:
        return "anaerobic"
    if raw == "not_specified" or raw == "":
        return "NONE"
    return "other"


def _normalize_cofactor_mode(raw: str | None) -> str:
    if not raw:
        return "NONE"
    if raw in COFMODE_TO_ID:
        return raw
    if "gdh" in raw.lower():
        return "GDH"
    if "fdh" in raw.lower():
        return "FDH"
    if "substrate" in raw.lower():
        return "substrate-coupled"
    if "enzyme" in raw.lower():
        return "enzyme-coupled"
    if "electro" in raw.lower():
        return "electrochemical"
    if "photo" in raw.lower():
        return "photochemical"
    if raw == "none":
        return "NONE"
    return "other"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SlotFeatures:
    """Features for a single slot in the skeleton."""
    rtype_id: int = 0
    ec1_id: int = 0
    ec2_id: int = 0
    T_norm: float = 0.0       # (T - 37) / 30
    pH_norm: float = 0.0      # (pH - 7) / 3
    step_role_id: int = 0
    step_type_id: int = 0
    step_mode_id: int = 0
    isolated: int = 0         # 0/1
    atmosphere_id: int = 0
    catalyst_class_id: int = 0
    engineering_id: int = 0
    biocat_format_id: int = 0
    process_tags: list[int] = field(default_factory=list)  # multi-hot indices
    cofactor_mode_id: int = 0
    substrate_conc_norm: float = 0.0  # log(mM)/5 clamped
    reaction_time_norm: float = 0.0   # log(h)/3 clamped
    is_observed: int = 0      # 0=to-predict, 1=observed/fixed


@dataclass
class SkeletonSample:
    """A single training sample for the OA-ARM inpainter."""
    target_fp: np.ndarray           # (2048,)
    domain_id: int
    objective_id: int
    n_steps: int
    slots: list[SlotFeatures]
    # Global labels
    compat_label: int = -1          # -1 = missing
    opmode_label: int = -1
    issue_labels: list[int] = field(default_factory=list)  # multi-hot indices
    evidence_weight: float = 1.0




# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SlotEmbedder(nn.Module):
    """Embed a single slot's categorical + continuous features into d_model."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.rtype_emb = nn.Embedding(NUM_RTYPES, 32)
        self.ec1_emb = nn.Embedding(NUM_EC1, 16)
        self.ec2_emb = nn.Embedding(NUM_EC2, 16)
        self.step_role_emb = nn.Embedding(len(STEP_ROLE_VOCAB), 8)
        self.step_type_emb = nn.Embedding(len(STEP_TYPE_VOCAB), 4)
        self.step_mode_emb = nn.Embedding(len(STEP_MODE_VOCAB), 4)
        self.atmosphere_emb = nn.Embedding(len(ATMOSPHERE_VOCAB), 4)
        self.catclass_emb = nn.Embedding(len(CATALYST_CLASS_VOCAB), 4)
        self.engineering_emb = nn.Embedding(len(ENGINEERING_VOCAB), 4)
        self.biofmt_emb = nn.Embedding(len(BIOCAT_FORMAT_VOCAB), 4)
        self.cofmode_emb = nn.Embedding(len(COFACTOR_MODE_VOCAB), 4)
        self.observed_emb = nn.Embedding(2, 4)
        self.isolated_emb = nn.Embedding(2, 2)
        self.position_emb = nn.Embedding(MAX_SLOTS, 16)

        # Continuous features: T_norm, pH_norm, substrate_conc, reaction_time
        self.cond_proj = nn.Linear(4, 16)
        # Process tags: multi-hot → linear
        self.ptag_proj = nn.Linear(NUM_PTAGS, 8)

        # Total raw dim: 32+16+16+8+4+4+4+4+4+4+4+4+2+16+16+8 = 146
        self.proj = nn.Linear(146, d_model)

    def forward(
        self,
        rtype_ids: torch.Tensor,        # (B, S)
        ec1_ids: torch.Tensor,          # (B, S)
        ec2_ids: torch.Tensor,          # (B, S)
        step_role_ids: torch.Tensor,    # (B, S)
        step_type_ids: torch.Tensor,    # (B, S)
        step_mode_ids: torch.Tensor,    # (B, S)
        atmosphere_ids: torch.Tensor,   # (B, S)
        catclass_ids: torch.Tensor,     # (B, S)
        engineering_ids: torch.Tensor,  # (B, S)
        biofmt_ids: torch.Tensor,       # (B, S)
        cofmode_ids: torch.Tensor,      # (B, S)
        isolated: torch.Tensor,         # (B, S)
        observed: torch.Tensor,         # (B, S)
        positions: torch.Tensor,        # (B, S)
        cond_feats: torch.Tensor,       # (B, S, 4) [T, pH, subst_conc, rxn_time]
        ptag_feats: torch.Tensor,       # (B, S, NUM_PTAGS) multi-hot
    ) -> torch.Tensor:
        parts = [
            self.rtype_emb(rtype_ids),
            self.ec1_emb(ec1_ids),
            self.ec2_emb(ec2_ids),
            self.step_role_emb(step_role_ids),
            self.step_type_emb(step_type_ids),
            self.step_mode_emb(step_mode_ids),
            self.atmosphere_emb(atmosphere_ids),
            self.catclass_emb(catclass_ids),
            self.engineering_emb(engineering_ids),
            self.biofmt_emb(biofmt_ids),
            self.cofmode_emb(cofmode_ids),
            self.isolated_emb(isolated),
            self.observed_emb(observed),
            self.position_emb(positions),
            self.cond_proj(cond_feats),
            self.ptag_proj(ptag_feats),
        ]
        x = torch.cat(parts, dim=-1)  # (B, S, 146)
        return self.proj(x)            # (B, S, d_model)


class SkeletonInpainter(nn.Module):
    """OA-ARM Transformer for cascade skeleton inpainting.

    6-layer decoder, d=256, 8 heads. Causal mask is dynamically built
    per sample based on the generation order (permutation).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        max_slots: int = MAX_SLOTS,
        dropout: float = 0.2,
        fp_dim: int = 2048,
        fp_proj_dim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_slots = max_slots

        # Slot embedder
        self.slot_embedder = SlotEmbedder(d_model)

        # Global condition tokens
        self.target_fp_proj = nn.Linear(fp_dim, fp_proj_dim)
        self.objective_emb = nn.Embedding(len(OBJECTIVE_VOCAB), 16)
        self.domain_emb = nn.Embedding(len(DOMAIN_VOCAB), 16)
        self.nsteps_emb = nn.Embedding(max_slots + 1, 16)
        self.global_proj = nn.Linear(fp_proj_dim + 16 + 16 + 16, d_model)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Dropout on embeddings
        self.emb_drop = nn.Dropout(dropout)

        # Per-slot output heads
        self.rtype_head = nn.Linear(d_model, NUM_RTYPES)
        self.ec1_head = nn.Linear(d_model, NUM_EC1)
        self.ec2_head = nn.Linear(d_model, NUM_EC2)
        self.T_head = nn.Linear(d_model, 1)
        self.pH_head = nn.Linear(d_model, 1)

        # Global output heads (from mean-pooled slot outputs)
        self.compat_head = nn.Linear(d_model, len(COMPAT_VOCAB))
        self.opmode_head = nn.Linear(d_model, len(OPMODE_VOCAB))
        self.issue_head = nn.Linear(d_model, NUM_ISSUES)

    def _build_causal_mask(self, S: int, perm: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        """Build attention mask for OA-ARM.

        For training: each position in the permutation can attend to
        all positions that come before it in the permutation order,
        plus all observed (fixed) positions.

        perm: (B, S) — generation order indices
        observed_mask: (B, S) — 1 if slot is observed/fixed

        Returns: (B*n_heads, S, S) or (B, S, S) float mask (0=attend, -inf=block)
        """
        B = perm.shape[0]
        device = perm.device

        # Build order matrix: order[b, i] = position of slot i in the permutation
        order = torch.zeros(B, S, dtype=torch.long, device=device)
        order.scatter_(1, perm, torch.arange(S, device=device).unsqueeze(0).expand(B, -1))

        # Slot j can attend to slot k if:
        #   - k is observed (fixed), OR
        #   - order[k] < order[j] (k was generated before j in this permutation)
        # Shape: (B, S, S) where [b, j, k] = can j attend to k?
        order_j = order.unsqueeze(2).expand(B, S, S)  # (B, S, S)
        order_k = order.unsqueeze(1).expand(B, S, S)  # (B, S, S)
        can_attend = (order_k < order_j)  # k generated before j

        # Also allow attending to observed slots
        obs_k = observed_mask.unsqueeze(1).expand(B, S, S).bool()  # (B, S, S)
        can_attend = can_attend | obs_k

        # Self-attention: each slot can attend to itself
        eye = torch.eye(S, device=device, dtype=torch.bool).unsqueeze(0).expand(B, S, S)
        can_attend = can_attend | eye

        # Convert to float mask: 0 where can attend, -inf where blocked
        mask = torch.zeros(B, S, S, device=device)
        mask.masked_fill_(~can_attend, float("-inf"))
        return mask

    def forward(
        self,
        slot_inputs: dict[str, torch.Tensor],
        target_fp: torch.Tensor,        # (B, 2048)
        objective_ids: torch.Tensor,    # (B,)
        domain_ids: torch.Tensor,       # (B,)
        n_steps_ids: torch.Tensor,      # (B,)
        perm: torch.Tensor | None = None,       # (B, S) generation order
        observed_mask: torch.Tensor | None = None,  # (B, S) 1=observed
    ) -> dict[str, torch.Tensor]:
        B = target_fp.shape[0]
        S = slot_inputs["rtype_ids"].shape[1]

        # Encode slots
        slot_embs = self.slot_embedder(**slot_inputs)  # (B, S, d)
        slot_embs = self.emb_drop(slot_embs)

        # Encode global condition as memory for cross-attention
        tfp = self.target_fp_proj(target_fp)           # (B, fp_proj_dim)
        obj = self.objective_emb(objective_ids)        # (B, 16)
        dom = self.domain_emb(domain_ids)              # (B, 16)
        ns = self.nsteps_emb(n_steps_ids)              # (B, 16)
        global_emb = self.global_proj(torch.cat([tfp, obj, dom, ns], dim=-1))  # (B, d)
        memory = global_emb.unsqueeze(1)               # (B, 1, d)

        # Build causal mask for OA-ARM
        if perm is not None and observed_mask is not None:
            tgt_mask = self._build_causal_mask(S, perm, observed_mask)
            # TransformerDecoder expects (S, S) or (B*nhead, S, S)
            # We'll use the (B, S, S) form — need to repeat for heads
            n_heads = 8
            tgt_mask = tgt_mask.unsqueeze(1).expand(B, n_heads, S, S)
            tgt_mask = tgt_mask.reshape(B * n_heads, S, S)
        else:
            tgt_mask = None

        # Decoder forward
        out = self.decoder(
            tgt=slot_embs,
            memory=memory,
            tgt_mask=tgt_mask,
        )  # (B, S, d)

        # Per-slot predictions
        rtype_logits = self.rtype_head(out)   # (B, S, NUM_RTYPES)
        ec1_logits = self.ec1_head(out)       # (B, S, NUM_EC1)
        ec2_logits = self.ec2_head(out)       # (B, S, NUM_EC2)
        T_pred = self.T_head(out).squeeze(-1) # (B, S)
        pH_pred = self.pH_head(out).squeeze(-1)  # (B, S)

        # Global predictions (mean pool over slots)
        global_repr = out.mean(dim=1)         # (B, d)
        compat_logits = self.compat_head(global_repr)   # (B, 4)
        opmode_logits = self.opmode_head(global_repr)   # (B, 7)
        issue_logits = self.issue_head(global_repr)     # (B, NUM_ISSUES)

        return {
            "rtype_logits": rtype_logits,
            "ec1_logits": ec1_logits,
            "ec2_logits": ec2_logits,
            "T_pred": T_pred,
            "pH_pred": pH_pred,
            "compat_logits": compat_logits,
            "opmode_logits": opmode_logits,
            "issue_logits": issue_logits,
        }




# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _extract_slot_features(step: dict, cas: dict) -> SlotFeatures:
    """Extract SlotFeatures from a v3 step dict."""
    conds = step.get("step_conditions") or {}
    cats = step.get("catalyst_components") or []
    cat0 = cats[0] if cats else {}

    # EC parsing
    ec_str = cat0.get("ec_number", "") if cat0 else ""
    ec1_id = 0
    ec2_id = 0
    if ec_str:
        parts = ec_str.split(".")
        try:
            ec1_id = min(int(parts[0]), 7)
        except ValueError:
            ec1_id = 0
        if len(parts) >= 2:
            ec2_key = f"{parts[0]}.{parts[1]}"
            ec2_id = EC2_TO_ID.get(ec2_key, 0)

    # Temperature / pH normalization
    T_raw = conds.get("temperature_c")
    pH_raw = conds.get("ph")
    T_norm = (T_raw - 37) / 30 if T_raw is not None else 0.0
    pH_norm = (pH_raw - 7) / 3 if pH_raw is not None else 0.0

    # Substrate concentration: log-normalize
    sub_conc = conds.get("substrate_concentration_mM")
    if sub_conc is not None and sub_conc > 0:
        substrate_conc_norm = min(math.log(sub_conc + 1) / 5.0, 1.0)
    else:
        substrate_conc_norm = 0.0

    # Reaction time: log-normalize
    rxn_time = conds.get("reaction_time_h")
    if rxn_time is not None and rxn_time > 0:
        reaction_time_norm = min(math.log(rxn_time + 1) / 3.0, 1.0)
    else:
        reaction_time_norm = 0.0

    # Process tags from cascade level
    ptags = []
    for tag in (cas.get("special_process_tags") or []):
        if tag in PTAG_TO_ID:
            ptags.append(PTAG_TO_ID[tag])

    # Atmosphere
    atmo = _normalize_atmosphere(conds.get("atmosphere"))

    # Catalyst class
    cat_class = cat0.get("catalyst_class", "") if cat0 else ""
    if cat_class not in CATCLASS_TO_ID:
        cat_class = "other" if cat_class else "NONE"

    # Engineering status
    eng = cat0.get("engineering_status", "") if cat0 else ""
    if eng not in ENG_TO_ID:
        eng = "unknown" if eng else "NONE"

    # Biocatalyst format
    biofmt = cat0.get("biocatalyst_format", "") if cat0 else ""
    if biofmt not in BIOFMT_TO_ID:
        biofmt = "unknown" if biofmt else "NONE"

    # Cofactor mode
    cofmode = _normalize_cofactor_mode(cat0.get("cofactor_regeneration_mode") if cat0 else None)

    return SlotFeatures(
        rtype_id=RTYPE_TO_ID.get(step.get("transformation_superclass", ""), 0),
        ec1_id=ec1_id,
        ec2_id=ec2_id,
        T_norm=T_norm,
        pH_norm=pH_norm,
        step_role_id=STEP_ROLE_TO_ID.get(step.get("step_role", ""), 0),
        step_type_id=STEP_TYPE_TO_ID.get(step.get("step_type", ""), 0),
        step_mode_id=STEP_MODE_TO_ID.get(step.get("step_mode", ""), 0),
        isolated=1 if step.get("intermediate_isolated") else 0,
        atmosphere_id=ATMO_TO_ID.get(atmo, 0),
        catalyst_class_id=CATCLASS_TO_ID.get(cat_class, 0),
        engineering_id=ENG_TO_ID.get(eng, 0),
        biocat_format_id=BIOFMT_TO_ID.get(biofmt, 0),
        process_tags=ptags,
        cofactor_mode_id=COFMODE_TO_ID.get(cofmode, 0),
        substrate_conc_norm=substrate_conc_norm,
        reaction_time_norm=reaction_time_norm,
        is_observed=0,
    )


def build_dataset(data_path: str, max_steps: int = MAX_SLOTS) -> list[SkeletonSample]:
    """Build skeleton inpainter dataset from v3 JSON."""
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

            # Target SMILES
            tp = cas.get("target_products") or [{}]
            t_smi = ""
            if tp and isinstance(tp[0], dict):
                t_smi = tp[0].get("smiles", "")
            if not t_smi:
                continue

            tfp = morgan_fp(t_smi)
            if tfp.sum() == 0:
                continue

            # Domain
            dom = cas.get("route_domain", "")
            domain_id = DOMAIN_TO_ID.get(dom, DOMAIN_TO_ID["other"])

            # Extract slot features
            slots = [_extract_slot_features(s, cas) for s in steps]

            # Global labels
            ca = cas.get("compatibility_annotation") or {}
            compat_label = COMPAT_TO_ID.get(ca.get("compatibility_label", ""), -1)
            opmode_label = OPMODE_TO_ID.get(cas.get("operation_mode", ""), -1)

            issue_labels = []
            for it in (ca.get("issue_types") or []):
                idx = next((i for i, v in enumerate(ISSUE_TYPE_VOCAB) if v == it), -1)
                if idx >= 0:
                    issue_labels.append(idx)

            ev_str = ca.get("evidence_strength", "workflow_only")
            ev_weight = EVIDENCE_WEIGHT.get(ev_str, 0.3)

            samples.append(SkeletonSample(
                target_fp=tfp,
                domain_id=domain_id,
                objective_id=0,  # balanced default
                n_steps=len(steps),
                slots=slots,
                compat_label=compat_label,
                opmode_label=opmode_label,
                issue_labels=issue_labels,
                evidence_weight=ev_weight,
            ))

    return samples


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

def augment_constraint_simulation(sample: SkeletonSample, rng: random.Random) -> SkeletonSample:
    """Randomly fix 1-3 slots to simulate user constraints."""
    import copy
    aug = copy.deepcopy(sample)
    n_fix = rng.randint(1, min(3, aug.n_steps - 1))
    fix_indices = rng.sample(range(aug.n_steps), n_fix)
    for idx in fix_indices:
        aug.slots[idx].is_observed = 1
    return aug


def augment_condition_perturbation(sample: SkeletonSample, rng: random.Random) -> SkeletonSample:
    """Add Gaussian noise to T/pH."""
    import copy
    aug = copy.deepcopy(sample)
    for slot in aug.slots:
        if slot.T_norm != 0.0:
            slot.T_norm += rng.gauss(0, 5.0 / 30.0)  # σ=5°C in normalized space
        if slot.pH_norm != 0.0:
            slot.pH_norm += rng.gauss(0, 0.5 / 3.0)   # σ=0.5 pH in normalized space
    return aug


def augment_step_deletion(sample: SkeletonSample, rng: random.Random) -> SkeletonSample | None:
    """Delete a random non-terminal step."""
    if sample.n_steps < 3:
        return None
    import copy
    aug = copy.deepcopy(sample)
    idx = rng.randint(1, aug.n_steps - 2)
    aug.slots.pop(idx)
    aug.n_steps -= 1
    return aug


def augment_dataset(
    samples: list[SkeletonSample],
    target_size: int = 50000,
    seed: int = 42,
) -> list[SkeletonSample]:
    """Augment dataset to target_size via constraint simulation + perturbation."""
    rng = random.Random(seed)
    augmented = list(samples)  # start with originals

    aug_fns = [augment_constraint_simulation, augment_condition_perturbation, augment_step_deletion]

    while len(augmented) < target_size:
        base = rng.choice(samples)
        fn = rng.choice(aug_fns)
        result = fn(base, rng)
        if result is not None:
            augmented.append(result)

    rng.shuffle(augmented)
    return augmented




# ---------------------------------------------------------------------------
# Collation: samples → batched tensors
# ---------------------------------------------------------------------------

def collate_batch(
    batch: list[SkeletonSample],
    device: str = "cpu",
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Collate a batch of SkeletonSamples into padded tensors.

    For training, generates a random permutation per sample (OA-ARM).
    """
    if rng is None:
        rng = random.Random()

    B = len(batch)
    S = max(s.n_steps for s in batch)

    # Slot feature tensors (B, S)
    rtype_ids = torch.zeros(B, S, dtype=torch.long)
    ec1_ids = torch.zeros(B, S, dtype=torch.long)
    ec2_ids = torch.zeros(B, S, dtype=torch.long)
    step_role_ids = torch.zeros(B, S, dtype=torch.long)
    step_type_ids = torch.zeros(B, S, dtype=torch.long)
    step_mode_ids = torch.zeros(B, S, dtype=torch.long)
    atmosphere_ids = torch.zeros(B, S, dtype=torch.long)
    catclass_ids = torch.zeros(B, S, dtype=torch.long)
    engineering_ids = torch.zeros(B, S, dtype=torch.long)
    biofmt_ids = torch.zeros(B, S, dtype=torch.long)
    cofmode_ids = torch.zeros(B, S, dtype=torch.long)
    isolated = torch.zeros(B, S, dtype=torch.long)
    observed = torch.zeros(B, S, dtype=torch.long)
    positions = torch.zeros(B, S, dtype=torch.long)
    cond_feats = torch.zeros(B, S, 4)
    ptag_feats = torch.zeros(B, S, NUM_PTAGS)

    # Global features
    target_fps = torch.zeros(B, 2048)
    objective_ids = torch.zeros(B, dtype=torch.long)
    domain_ids = torch.zeros(B, dtype=torch.long)
    n_steps_ids = torch.zeros(B, dtype=torch.long)

    # Labels
    rtype_labels = torch.full((B, S), -100, dtype=torch.long)  # -100 = ignore
    ec1_labels = torch.full((B, S), -100, dtype=torch.long)
    ec2_labels = torch.full((B, S), -100, dtype=torch.long)
    T_labels = torch.zeros(B, S)
    pH_labels = torch.zeros(B, S)
    T_mask = torch.zeros(B, S)  # 1 where T label is valid
    pH_mask = torch.zeros(B, S)
    compat_labels = torch.full((B,), -100, dtype=torch.long)
    opmode_labels = torch.full((B,), -100, dtype=torch.long)
    issue_labels = torch.zeros(B, NUM_ISSUES)
    evidence_weights = torch.ones(B)

    # Permutations for OA-ARM
    perms = torch.zeros(B, S, dtype=torch.long)
    slot_valid = torch.zeros(B, S)  # 1 for real slots, 0 for padding

    for b, sample in enumerate(batch):
        n = sample.n_steps
        target_fps[b] = torch.from_numpy(sample.target_fp)
        objective_ids[b] = sample.objective_id
        domain_ids[b] = sample.domain_id
        n_steps_ids[b] = n

        # Generate random permutation for non-observed slots
        observed_indices = [i for i in range(n) if sample.slots[i].is_observed]
        free_indices = [i for i in range(n) if not sample.slots[i].is_observed]
        rng.shuffle(free_indices)
        # Permutation: observed first (arbitrary order), then free in random order
        perm_list = observed_indices + free_indices
        # Pad to S
        perm_list += list(range(n, S))
        perms[b] = torch.tensor(perm_list[:S])

        for i, slot in enumerate(sample.slots):
            positions[b, i] = i
            observed[b, i] = slot.is_observed

            if slot.is_observed:
                # Observed slots: feed ground truth as input (model conditions on these)
                rtype_ids[b, i] = slot.rtype_id
                ec1_ids[b, i] = slot.ec1_id
                ec2_ids[b, i] = slot.ec2_id
                step_role_ids[b, i] = slot.step_role_id
                step_type_ids[b, i] = slot.step_type_id
                step_mode_ids[b, i] = slot.step_mode_id
                atmosphere_ids[b, i] = slot.atmosphere_id
                catclass_ids[b, i] = slot.catalyst_class_id
                engineering_ids[b, i] = slot.engineering_id
                biofmt_ids[b, i] = slot.biocat_format_id
                cofmode_ids[b, i] = slot.cofactor_mode_id
                isolated[b, i] = slot.isolated
                cond_feats[b, i] = torch.tensor([
                    slot.T_norm, slot.pH_norm,
                    slot.substrate_conc_norm, slot.reaction_time_norm,
                ])
                for tag_idx in slot.process_tags:
                    if tag_idx < NUM_PTAGS:
                        ptag_feats[b, i, tag_idx] = 1.0
            else:
                # Non-observed (free) slots: MASK input features (zeros)
                # Only position and is_observed=0 are set; all else stays zero.
                # Labels are set below for loss computation.
                pass

            # Labels: only for non-observed (free) slots
            if not slot.is_observed:
                rtype_labels[b, i] = slot.rtype_id
                ec1_labels[b, i] = slot.ec1_id
                ec2_labels[b, i] = slot.ec2_id
                T_labels[b, i] = slot.T_norm
                pH_labels[b, i] = slot.pH_norm
                T_mask[b, i] = 1.0 if slot.T_norm != 0.0 else 0.0
                pH_mask[b, i] = 1.0 if slot.pH_norm != 0.0 else 0.0

            slot_valid[b, i] = 1.0

        # Global labels
        if sample.compat_label >= 0:
            compat_labels[b] = sample.compat_label
        if sample.opmode_label >= 0:
            opmode_labels[b] = sample.opmode_label
        for idx in sample.issue_labels:
            issue_labels[b, idx] = 1.0
        evidence_weights[b] = sample.evidence_weight

    result = {
        "slot_inputs": {
            "rtype_ids": rtype_ids.to(device),
            "ec1_ids": ec1_ids.to(device),
            "ec2_ids": ec2_ids.to(device),
            "step_role_ids": step_role_ids.to(device),
            "step_type_ids": step_type_ids.to(device),
            "step_mode_ids": step_mode_ids.to(device),
            "atmosphere_ids": atmosphere_ids.to(device),
            "catclass_ids": catclass_ids.to(device),
            "engineering_ids": engineering_ids.to(device),
            "biofmt_ids": biofmt_ids.to(device),
            "cofmode_ids": cofmode_ids.to(device),
            "isolated": isolated.to(device),
            "observed": observed.to(device),
            "positions": positions.to(device),
            "cond_feats": cond_feats.to(device),
            "ptag_feats": ptag_feats.to(device),
        },
        "target_fp": target_fps.to(device),
        "objective_ids": objective_ids.to(device),
        "domain_ids": domain_ids.to(device),
        "n_steps_ids": n_steps_ids.to(device),
        "perm": perms.to(device),
        "observed_mask": observed.to(device),
        # Labels
        "rtype_labels": rtype_labels.to(device),
        "ec1_labels": ec1_labels.to(device),
        "ec2_labels": ec2_labels.to(device),
        "T_labels": T_labels.to(device),
        "pH_labels": pH_labels.to(device),
        "T_mask": T_mask.to(device),
        "pH_mask": pH_mask.to(device),
        "compat_labels": compat_labels.to(device),
        "opmode_labels": opmode_labels.to(device),
        "issue_labels": issue_labels.to(device),
        "evidence_weights": evidence_weights.to(device),
        "slot_valid": slot_valid.to(device),
    }
    return result




# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute OA-ARM training loss.

    Loss = L_rtype + 0.7*L_ec1 + 0.3*L_ec2 + 0.2*L_T + 0.2*L_pH
           + 0.5*L_compat + 0.3*L_opmode + 0.2*L_issue

    All per-slot losses are masked to only count non-observed (free) slots.
    Evidence weight scales global losses.
    """
    device = outputs["rtype_logits"].device
    B, S, _ = outputs["rtype_logits"].shape

    # Per-slot classification losses (ignore_index=-100 handles padding+observed)
    L_rtype = F.cross_entropy(
        outputs["rtype_logits"].reshape(B * S, -1),
        batch["rtype_labels"].reshape(B * S),
        ignore_index=-100, label_smoothing=0.1,
    )
    L_ec1 = F.cross_entropy(
        outputs["ec1_logits"].reshape(B * S, -1),
        batch["ec1_labels"].reshape(B * S),
        ignore_index=-100, label_smoothing=0.1,
    )
    L_ec2 = F.cross_entropy(
        outputs["ec2_logits"].reshape(B * S, -1),
        batch["ec2_labels"].reshape(B * S),
        ignore_index=-100, label_smoothing=0.1,
    )

    # Per-slot regression losses (Huber, masked)
    T_mask = batch["T_mask"]  # (B, S)
    pH_mask = batch["pH_mask"]
    T_diff = outputs["T_pred"] - batch["T_labels"]
    pH_diff = outputs["pH_pred"] - batch["pH_labels"]

    L_T = torch.tensor(0.0, device=device)
    if T_mask.sum() > 0:
        L_T = F.huber_loss(
            outputs["T_pred"][T_mask > 0],
            batch["T_labels"][T_mask > 0],
            reduction="mean", delta=0.5,
        )

    L_pH = torch.tensor(0.0, device=device)
    if pH_mask.sum() > 0:
        L_pH = F.huber_loss(
            outputs["pH_pred"][pH_mask > 0],
            batch["pH_labels"][pH_mask > 0],
            reduction="mean", delta=0.5,
        )

    # Global classification losses (weighted by evidence_strength)
    ew = batch["evidence_weights"]  # (B,)

    L_compat = torch.tensor(0.0, device=device)
    valid_compat = batch["compat_labels"] >= 0
    if valid_compat.any():
        ce = F.cross_entropy(
            outputs["compat_logits"][valid_compat],
            batch["compat_labels"][valid_compat],
            reduction="none",
        )
        L_compat = (ce * ew[valid_compat]).mean()

    L_opmode = torch.tensor(0.0, device=device)
    valid_opmode = batch["opmode_labels"] >= 0
    if valid_opmode.any():
        ce = F.cross_entropy(
            outputs["opmode_logits"][valid_opmode],
            batch["opmode_labels"][valid_opmode],
            reduction="none",
        )
        L_opmode = (ce * ew[valid_opmode]).mean()

    # Issue detection: multi-label BCE
    L_issue = F.binary_cross_entropy_with_logits(
        outputs["issue_logits"],
        batch["issue_labels"],
        reduction="none",
    ).mean(dim=-1)  # (B,)
    L_issue = (L_issue * ew).mean()

    # Total
    total = (
        1.0 * L_rtype + 0.7 * L_ec1 + 0.3 * L_ec2
        + 0.2 * L_T + 0.2 * L_pH
        + 0.5 * L_compat + 0.3 * L_opmode + 0.2 * L_issue
    )

    metrics = {
        "loss": total.item(),
        "L_rtype": L_rtype.item(),
        "L_ec1": L_ec1.item(),
        "L_ec2": L_ec2.item(),
        "L_T": L_T.item(),
        "L_pH": L_pH.item(),
        "L_compat": L_compat.item(),
        "L_opmode": L_opmode.item(),
        "L_issue": L_issue.item(),
    }
    return total, metrics


def train(
    data_path: str = "cascade_dataset_v3.json",
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 3e-4,
    warmup_steps: int = 500,
    augment_target: int = 50000,
    save_dir: str = "results/shared/skeleton_inpainter",
    seed: int = 42,
):
    """Train the OA-ARM Skeleton Inpainter."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Build dataset
    print(f"Loading data from {data_path}...")
    samples = build_dataset(data_path)
    print(f"  Raw samples: {len(samples)}")

    # Augment
    print(f"Augmenting to ~{augment_target} samples...")
    all_samples = augment_dataset(samples, target_size=augment_target, seed=seed)
    print(f"  Augmented: {len(all_samples)}")

    # Train/val split (80/20)
    rng = random.Random(seed)
    rng.shuffle(all_samples)
    split = int(len(all_samples) * 0.8)
    train_samples = all_samples[:split]
    val_samples = all_samples[split:]
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Model
    model = SkeletonInpainter().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,} ({n_params/1e6:.1f}M)")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * (len(train_samples) // batch_size + 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    history = []
    collate_rng = random.Random(seed)

    for epoch in range(epochs):
        # Train epoch
        model.train()
        rng.shuffle(train_samples)
        train_loss_sum = 0.0
        n_batches = 0

        for i in range(0, len(train_samples), batch_size):
            batch_samples = train_samples[i:i + batch_size]
            batch = collate_batch(batch_samples, device=device, rng=collate_rng)

            outputs = model(
                slot_inputs=batch["slot_inputs"],
                target_fp=batch["target_fp"],
                objective_ids=batch["objective_ids"],
                domain_ids=batch["domain_ids"],
                n_steps_ids=batch["n_steps_ids"],
                perm=batch["perm"],
                observed_mask=batch["observed_mask"],
            )

            loss, metrics = compute_loss(outputs, batch)

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
        val_rtype_correct = 0
        val_rtype_total = 0
        val_ec1_correct = 0
        val_ec1_total = 0
        n_val_batches = 0

        with torch.no_grad():
            for i in range(0, len(val_samples), batch_size):
                batch_samples = val_samples[i:i + batch_size]
                batch = collate_batch(batch_samples, device=device, rng=collate_rng)

                outputs = model(
                    slot_inputs=batch["slot_inputs"],
                    target_fp=batch["target_fp"],
                    objective_ids=batch["objective_ids"],
                    domain_ids=batch["domain_ids"],
                    n_steps_ids=batch["n_steps_ids"],
                    perm=batch["perm"],
                    observed_mask=batch["observed_mask"],
                )

                loss, _ = compute_loss(outputs, batch)
                val_loss_sum += loss.item()
                n_val_batches += 1

                # Accuracy on free slots
                rtype_preds = outputs["rtype_logits"].argmax(dim=-1)  # (B, S)
                ec1_preds = outputs["ec1_logits"].argmax(dim=-1)
                valid = batch["rtype_labels"] >= 0
                if valid.any():
                    val_rtype_correct += (rtype_preds[valid] == batch["rtype_labels"][valid]).sum().item()
                    val_rtype_total += valid.sum().item()
                valid_ec1 = batch["ec1_labels"] >= 0
                if valid_ec1.any():
                    val_ec1_correct += (ec1_preds[valid_ec1] == batch["ec1_labels"][valid_ec1]).sum().item()
                    val_ec1_total += valid_ec1.sum().item()

        avg_val_loss = val_loss_sum / max(n_val_batches, 1)
        val_rtype_acc = val_rtype_correct / max(val_rtype_total, 1)
        val_ec1_acc = val_ec1_correct / max(val_ec1_total, 1)

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_rtype_acc": val_rtype_acc,
            "val_ec1_acc": val_ec1_acc,
        })

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  ep{epoch+1:3d}: train={avg_train_loss:.4f} val={avg_val_loss:.4f} "
                f"rtype_acc={val_rtype_acc:.3f} ec1_acc={val_ec1_acc:.3f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path / "best.pt")

    # Save final
    torch.save(model.state_dict(), save_path / "final.pt")
    (save_path / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Saved to {save_path}/")
    return model




# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str = "results/shared/skeleton_inpainter/best.pt",
    device: str = "cpu",
) -> SkeletonInpainter:
    """Load a trained SkeletonInpainter from checkpoint."""
    model = SkeletonInpainter()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@dataclass
class SkeletonResult:
    """Result of skeleton generation."""
    types: list[str]
    ec1s: list[int]
    ec2s: list[str]
    Ts: list[float]       # denormalized °C
    pHs: list[float]      # denormalized
    compat_pred: str
    opmode_pred: str
    issues_pred: list[str]
    log_prob: float = 0.0


def _sort_by_constraint_proximity(
    free_indices: list[int],
    observed_indices: list[int],
    n_steps: int,
) -> list[int]:
    """Sort free indices by proximity to observed (fixed) slots.

    Slots adjacent to fixed slots are generated first (most constrained).
    """
    if not observed_indices:
        return free_indices

    obs_set = set(observed_indices)
    scored = []
    for idx in free_indices:
        min_dist = min(abs(idx - o) for o in obs_set)
        scored.append((min_dist, idx))
    scored.sort()
    return [idx for _, idx in scored]


@torch.no_grad()
def generate_skeleton(
    model: SkeletonInpainter,
    target_smi: str,
    n_steps: int,
    fixed_slots: dict[int, dict] | None = None,
    domain: str = "chemoenzymatic",
    objective: str = "balanced",
    temperature: float = 0.8,
    device: str = "cpu",
) -> SkeletonResult:
    """Generate a single skeleton using OA-ARM iterative decoding.

    fixed_slots: {step_idx: {"ec": "1.x", "T": 37, "reaction_type": "reduction", ...}}
    """
    if n_steps < 1 or n_steps > MAX_SLOTS:
        raise ValueError(f"n_steps must be between 1 and {MAX_SLOTS}; got {n_steps}")

    model.eval()
    fixed_slots = fixed_slots or {}

    # Initialize slot features
    slots = [SlotFeatures() for _ in range(n_steps)]

    # Apply fixed constraints
    observed_indices = []
    for idx, fields in fixed_slots.items():
        if idx >= n_steps:
            continue
        slot = slots[idx]
        slot.is_observed = 1
        observed_indices.append(idx)
        if "reaction_type" in fields:
            slot.rtype_id = RTYPE_TO_ID.get(fields["reaction_type"], 0)
        if "ec" in fields:
            ec_str = fields["ec"]
            parts = ec_str.split(".")
            try:
                slot.ec1_id = min(int(parts[0]), 7)
            except ValueError:
                pass
            if len(parts) >= 2:
                slot.ec2_id = EC2_TO_ID.get(f"{parts[0]}.{parts[1]}", 0)
        if "T" in fields:
            slot.T_norm = (fields["T"] - 37) / 30
        if "pH" in fields:
            slot.pH_norm = (fields["pH"] - 7) / 3

    # Determine generation order
    free_indices = [i for i in range(n_steps) if i not in set(observed_indices)]
    gen_order = _sort_by_constraint_proximity(free_indices, observed_indices, n_steps)

    # Prepare global inputs
    tfp = torch.from_numpy(morgan_fp(target_smi)).unsqueeze(0).to(device)
    obj_id = torch.tensor([OBJ_TO_ID.get(objective, 0)], dtype=torch.long, device=device)
    dom_id = torch.tensor([DOMAIN_TO_ID.get(domain, DOMAIN_TO_ID["other"])], dtype=torch.long, device=device)
    ns_id = torch.tensor([n_steps], dtype=torch.long, device=device)

    total_log_prob = 0.0

    # Iterative generation: one slot at a time
    for target_idx in gen_order:
        # Build current slot inputs (all slots, with current state)
        batch = _slots_to_batch(slots, n_steps, device)

        # Build permutation: observed first, then already-generated, then remaining
        already_generated = [i for i in gen_order[:gen_order.index(target_idx)]]
        current_observed = observed_indices + already_generated
        perm_list = current_observed + [target_idx] + [
            i for i in range(n_steps) if i not in current_observed and i != target_idx
        ]
        # Pad to max_slots
        perm_list = perm_list + list(range(n_steps, MAX_SLOTS))
        perm = torch.tensor([perm_list[:MAX_SLOTS]], dtype=torch.long, device=device)

        obs_mask = torch.zeros(1, MAX_SLOTS, dtype=torch.long, device=device)
        for i in current_observed:
            if i < MAX_SLOTS:
                obs_mask[0, i] = 1

        # Forward pass
        outputs = model(
            slot_inputs=batch,
            target_fp=tfp,
            objective_ids=obj_id,
            domain_ids=dom_id,
            n_steps_ids=ns_id,
            perm=perm[:, :batch["rtype_ids"].shape[1]],
            observed_mask=obs_mask[:, :batch["rtype_ids"].shape[1]],
        )

        # Sample from predictions for target_idx
        s_idx = target_idx  # position in the slot sequence

        # Reaction type
        rtype_logits = outputs["rtype_logits"][0, s_idx] / temperature
        rtype_probs = F.softmax(rtype_logits, dim=-1)
        rtype_id = torch.multinomial(rtype_probs, 1).item()
        total_log_prob += torch.log(rtype_probs[rtype_id]).item()

        # EC1
        ec1_logits = outputs["ec1_logits"][0, s_idx] / temperature
        ec1_probs = F.softmax(ec1_logits, dim=-1)
        ec1_id = torch.multinomial(ec1_probs, 1).item()
        total_log_prob += torch.log(ec1_probs[ec1_id]).item()

        # EC2
        ec2_logits = outputs["ec2_logits"][0, s_idx] / temperature
        ec2_probs = F.softmax(ec2_logits, dim=-1)
        ec2_id = torch.multinomial(ec2_probs, 1).item()

        # T and pH (deterministic from regression head)
        T_norm = outputs["T_pred"][0, s_idx].item()
        pH_norm = outputs["pH_pred"][0, s_idx].item()

        # Update slot
        slots[target_idx].rtype_id = rtype_id
        slots[target_idx].ec1_id = ec1_id
        slots[target_idx].ec2_id = ec2_id
        slots[target_idx].T_norm = T_norm
        slots[target_idx].pH_norm = pH_norm
        slots[target_idx].is_observed = 1  # now "known"

    # Final forward for global predictions
    batch = _slots_to_batch(slots, n_steps, device)
    obs_all = torch.ones(1, batch["rtype_ids"].shape[1], dtype=torch.long, device=device)
    perm_final = torch.arange(MAX_SLOTS, device=device).unsqueeze(0)[:, :batch["rtype_ids"].shape[1]]
    outputs = model(
        slot_inputs=batch,
        target_fp=tfp,
        objective_ids=obj_id,
        domain_ids=dom_id,
        n_steps_ids=ns_id,
        perm=perm_final,
        observed_mask=obs_all,
    )

    # Decode global predictions
    compat_idx = outputs["compat_logits"][0].argmax().item()
    opmode_idx = outputs["opmode_logits"][0].argmax().item()
    issue_probs = torch.sigmoid(outputs["issue_logits"][0])
    issues = [ISSUE_TYPE_VOCAB[i] for i in range(NUM_ISSUES) if issue_probs[i] > 0.5]

    # Build result
    return SkeletonResult(
        types=[REACTION_TYPE_VOCAB[s.rtype_id] for s in slots],
        ec1s=[s.ec1_id for s in slots],
        ec2s=[EC2_VOCAB[s.ec2_id] for s in slots],
        Ts=[37 + 30 * s.T_norm for s in slots],
        pHs=[7 + 3 * s.pH_norm for s in slots],
        compat_pred=COMPAT_VOCAB[compat_idx] if compat_idx < len(COMPAT_VOCAB) else "unclear",
        opmode_pred=OPMODE_VOCAB[opmode_idx] if opmode_idx < len(OPMODE_VOCAB) else "batch_other",
        issues_pred=issues,
        log_prob=total_log_prob,
    )


def _slots_to_batch(slots: list[SlotFeatures], n_steps: int, device: str) -> dict[str, torch.Tensor]:
    """Convert current slot state to a single-sample batch dict for the model."""
    S = n_steps
    rtype_ids = torch.zeros(1, S, dtype=torch.long, device=device)
    ec1_ids = torch.zeros(1, S, dtype=torch.long, device=device)
    ec2_ids = torch.zeros(1, S, dtype=torch.long, device=device)
    step_role_ids = torch.zeros(1, S, dtype=torch.long, device=device)
    step_type_ids = torch.zeros(1, S, dtype=torch.long, device=device)
    step_mode_ids = torch.zeros(1, S, dtype=torch.long, device=device)
    atmosphere_ids = torch.zeros(1, S, dtype=torch.long, device=device)
    catclass_ids = torch.zeros(1, S, dtype=torch.long, device=device)
    engineering_ids = torch.zeros(1, S, dtype=torch.long, device=device)
    biofmt_ids = torch.zeros(1, S, dtype=torch.long, device=device)
    cofmode_ids = torch.zeros(1, S, dtype=torch.long, device=device)
    isolated = torch.zeros(1, S, dtype=torch.long, device=device)
    observed = torch.zeros(1, S, dtype=torch.long, device=device)
    positions = torch.arange(S, device=device).unsqueeze(0)
    cond_feats = torch.zeros(1, S, 4, device=device)
    ptag_feats = torch.zeros(1, S, NUM_PTAGS, device=device)

    for i, slot in enumerate(slots):
        rtype_ids[0, i] = slot.rtype_id
        ec1_ids[0, i] = slot.ec1_id
        ec2_ids[0, i] = slot.ec2_id
        step_role_ids[0, i] = slot.step_role_id
        step_type_ids[0, i] = slot.step_type_id
        step_mode_ids[0, i] = slot.step_mode_id
        atmosphere_ids[0, i] = slot.atmosphere_id
        catclass_ids[0, i] = slot.catalyst_class_id
        engineering_ids[0, i] = slot.engineering_id
        biofmt_ids[0, i] = slot.biocat_format_id
        cofmode_ids[0, i] = slot.cofactor_mode_id
        isolated[0, i] = slot.isolated
        observed[0, i] = slot.is_observed
        cond_feats[0, i] = torch.tensor([
            slot.T_norm, slot.pH_norm,
            slot.substrate_conc_norm, slot.reaction_time_norm,
        ])
        for tag_idx in slot.process_tags:
            if tag_idx < NUM_PTAGS:
                ptag_feats[0, i, tag_idx] = 1.0

    return {
        "rtype_ids": rtype_ids,
        "ec1_ids": ec1_ids,
        "ec2_ids": ec2_ids,
        "step_role_ids": step_role_ids,
        "step_type_ids": step_type_ids,
        "step_mode_ids": step_mode_ids,
        "atmosphere_ids": atmosphere_ids,
        "catclass_ids": catclass_ids,
        "engineering_ids": engineering_ids,
        "biofmt_ids": biofmt_ids,
        "cofmode_ids": cofmode_ids,
        "isolated": isolated,
        "observed": observed,
        "positions": positions,
        "cond_feats": cond_feats,
        "ptag_feats": ptag_feats,
    }


@torch.no_grad()
def generate_multiple_skeletons(
    model: SkeletonInpainter,
    target_smi: str,
    n_steps: int,
    k: int = 5,
    fixed_slots: dict[int, dict] | None = None,
    domain: str = "chemoenzymatic",
    objective: str = "balanced",
    temperature: float = 0.8,
    device: str = "cpu",
) -> list[SkeletonResult]:
    """Generate k diverse skeletons by repeated sampling."""
    results = []
    seen_types = set()

    for _ in range(k * 3):  # oversample to get k unique
        if len(results) >= k:
            break
        skel = generate_skeleton(
            model, target_smi, n_steps,
            fixed_slots=fixed_slots,
            domain=domain, objective=objective,
            temperature=temperature, device=device,
        )
        key = tuple(skel.types)
        if key not in seen_types:
            seen_types.add(key)
            results.append(skel)

    # Sort by log_prob (higher = more confident)
    results.sort(key=lambda r: r.log_prob, reverse=True)
    return results[:k]


def _fixed_slots_from_constraints(constraints: dict[str, Any] | None) -> dict[int, dict]:
    """Extract fixed slot fields from compiled-style raw constraints."""
    fixed: dict[int, dict] = {}
    if not constraints:
        return fixed
    for item in constraints.get("fixed_steps", []) or []:
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        values = item.get("values") or {}
        if isinstance(values, dict):
            fixed[idx] = dict(values)
    return fixed


def _to_route_skeleton(skel: SkeletonResult):
    """Convert an OA-ARM SkeletonResult to skeleton_planner.RouteSkeleton."""
    from cascade_planner.cascadeboard.skeleton_planner import RouteSkeleton

    return RouteSkeleton(
        n_steps=len(skel.types),
        types=skel.types,
        ec1s=skel.ec1s,
        Ts=skel.Ts,
        pHs=skel.pHs,
        compatibility=skel.compat_pred,
        operation_mode=skel.opmode_pred,
        issues=skel.issues_pred,
        pairwise_modes=[],
        log_prob=float(getattr(skel, "log_prob", 0.0) or 0.0),
    )


def plan_with_skeleton_inpainter(
    target: str,
    n_steps: int,
    domain: str = "chemoenzymatic",
    objective: str = "balanced",
    constraints: dict[str, Any] | None = None,
    fixed_slots: dict[int, dict] | None = None,
    model_path: str = "results/shared/skeleton_inpainter/best.pt",
    retro_engine: dict | None = None,
    device: str = "cpu",
    n_results: int = 5,
    n_candidates_per_skeleton: int = 2,
    temperature: float = 0.8,
) -> list:
    """End-to-end live planning entry point used by the public CLI.

    This adapter keeps the OA-ARM skeleton model separate from the legacy v20
    CascadeBoard policy checkpoint, then delegates molecular filling and rule
    scoring to ``skeleton_planner.plan_with_skeleton``.
    """
    from cascade_planner.cascadeboard.skeleton_planner import plan_with_skeleton

    model = load_model(model_path, device=device)
    fixed = fixed_slots or _fixed_slots_from_constraints(constraints)
    skeletons = generate_multiple_skeletons(
        model,
        target,
        n_steps=n_steps,
        k=max(n_results, 1),
        fixed_slots=fixed,
        domain=domain,
        objective=objective,
        temperature=temperature,
        device=device,
    )

    results = []
    for skel in skeletons:
        results.extend(
            plan_with_skeleton(
                target=target,
                skeleton=_to_route_skeleton(skel),
                retro_engine=retro_engine,
                constraints=constraints,
                objective=objective,
                n_steps=n_steps,
                n_candidates=n_candidates_per_skeleton,
                device=device,
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:n_results]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="OA-ARM Skeleton Inpainter")
    sub = ap.add_subparsers(dest="cmd")

    # Train
    train_p = sub.add_parser("train")
    train_p.add_argument("--data", default="cascade_dataset_v3.json")
    train_p.add_argument("--epochs", type=int, default=200)
    train_p.add_argument("--batch-size", type=int, default=64)
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--augment-target", type=int, default=50000)
    train_p.add_argument("--save-dir", default="results/shared/skeleton_inpainter")
    train_p.add_argument("--seed", type=int, default=42)

    # Predict
    pred_p = sub.add_parser("predict")
    pred_p.add_argument("--target", required=True, help="Target SMILES")
    pred_p.add_argument("--n-steps", type=int, default=3)
    pred_p.add_argument("--k", type=int, default=5)
    pred_p.add_argument("--domain", default="chemoenzymatic")
    pred_p.add_argument("--objective", default="balanced")
    pred_p.add_argument("--checkpoint", default="results/shared/skeleton_inpainter/best.pt")
    pred_p.add_argument("--fix", type=str, default=None, help='JSON: {"1": {"ec": "1.x", "T": 37}}')

    args = ap.parse_args()

    if args.cmd == "train":
        train(
            data_path=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            augment_target=args.augment_target,
            save_dir=args.save_dir,
            seed=args.seed,
        )
    elif args.cmd == "predict":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(args.checkpoint, device=device)
        fixed = json.loads(args.fix) if args.fix else None
        # Convert string keys to int
        if fixed:
            fixed = {int(k): v for k, v in fixed.items()}

        results = generate_multiple_skeletons(
            model, args.target, args.n_steps, k=args.k,
            fixed_slots=fixed, domain=args.domain,
            objective=args.objective, device=device,
        )

        print(f"\nGenerated {len(results)} skeletons for: {args.target}")
        print(f"  n_steps={args.n_steps}, domain={args.domain}, objective={args.objective}")
        if fixed:
            print(f"  fixed_slots={fixed}")
        print()

        for i, skel in enumerate(results):
            print(f"Skeleton {i+1} (log_prob={skel.log_prob:.3f}):")
            for j in range(args.n_steps):
                ec_str = f"EC{skel.ec1s[j]}.{skel.ec2s[j].split('.')[-1] if skel.ec2s[j] != 'NONE' else 'x'}" if skel.ec1s[j] > 0 else "chem"
                print(f"  Step {j}: {skel.types[j]:30s} {ec_str:10s} T={skel.Ts[j]:.0f}°C pH={skel.pHs[j]:.1f}")
            print(f"  Compat: {skel.compat_pred}, OpMode: {skel.opmode_pred}")
            if skel.issues_pred:
                print(f"  Issues: {skel.issues_pred}")
            print()
    else:
        ap.print_help()
