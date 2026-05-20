"""Neural vNext models for candidate-pool and route-level planning."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from cascade_planner.vnext.schema import BOTTLENECK_LABELS, SOURCE_BUDGET_GROUPS


@dataclass
class VNextModelConfig:
    n_bits: int = 256
    candidate_feature_dim: int = 0
    route_feature_dim: int = 0
    step_token_dim: int = 48
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    max_steps: int = 8


class StepEncoder(nn.Module):
    """Dual encoder for product/reactant single-step supervision.

    This is the pretraining block that lets pure chemical and pure enzymatic
    single-step data share a representation before route-level fine-tuning.
    """

    def __init__(
        self,
        *,
        n_bits: int = 256,
        metadata_dim: int = 32,
        d_model: int = 128,
        n_reaction_type_buckets: int = 32,
        n_ec1_classes: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_bits = n_bits
        self.metadata_dim = metadata_dim
        self.d_model = d_model
        self.product_proj = _mlp(n_bits, d_model, d_model, dropout=dropout)
        self.reactant_proj = _mlp(n_bits, d_model, d_model, dropout=dropout)
        self.metadata_proj = _mlp(metadata_dim, d_model, d_model, dropout=dropout)
        self.fusion = _mlp(d_model * 3, d_model * 2, d_model, dropout=dropout)
        self.match_head = nn.Linear(d_model, 1)
        self.reaction_type_head = nn.Linear(d_model, n_reaction_type_buckets)
        self.ec1_head = nn.Linear(d_model, n_ec1_classes)
        self.condition_head = nn.Linear(d_model, 2)

    def forward(
        self,
        product_fp: torch.Tensor,
        reactant_fp: torch.Tensor,
        metadata: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if metadata is None:
            metadata = torch.zeros(product_fp.shape[0], self.metadata_dim, device=product_fp.device)
        if metadata.shape[-1] != self.metadata_dim:
            metadata = _resize_last_dim(metadata, self.metadata_dim)
        product_emb = self.product_proj(product_fp)
        reactant_emb = self.reactant_proj(reactant_fp)
        metadata_emb = self.metadata_proj(metadata)
        step_emb = self.fusion(torch.cat([product_emb, reactant_emb, metadata_emb], dim=-1))
        return {
            "product_embedding": product_emb,
            "reactant_embedding": reactant_emb,
            "step_embedding": step_emb,
            "match_logit": self.match_head(step_emb).squeeze(-1),
            "reaction_type_logits": self.reaction_type_head(step_emb),
            "ec1_logits": self.ec1_head(step_emb),
            "condition": self.condition_head(step_emb),
        }


class CandidatePoolCrossAttentionRanker(nn.Module):
    """Listwise ranker over all candidates for a single retrosynthesis step."""

    def __init__(
        self,
        *,
        candidate_feature_dim: int,
        route_feature_dim: int = 0,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.candidate_feature_dim = candidate_feature_dim
        self.route_feature_dim = route_feature_dim
        self.d_model = d_model
        self.candidate_proj = nn.Linear(candidate_feature_dim, d_model)
        self.route_proj = nn.Linear(max(route_feature_dim, 1), d_model)
        self.pool_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.score_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.pool_value_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(
        self,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
        route_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # candidate_features: (B, K, F), candidate_mask: True for valid.
        if candidate_mask is None:
            candidate_mask = torch.ones(candidate_features.shape[:2], dtype=torch.bool, device=candidate_features.device)
        else:
            candidate_mask = candidate_mask.bool()
        cand = self.candidate_proj(candidate_features)
        if route_features is None:
            route_features = torch.zeros(candidate_features.shape[0], max(self.route_feature_dim, 1), device=candidate_features.device)
        elif route_features.shape[-1] != max(self.route_feature_dim, 1):
            route_features = _resize_last_dim(route_features, max(self.route_feature_dim, 1))
        context = self.route_proj(route_features).unsqueeze(1) + self.pool_token
        tokens = torch.cat([context, cand], dim=1)
        padding_mask = torch.cat([
            torch.zeros(candidate_features.shape[0], 1, dtype=torch.bool, device=candidate_features.device),
            ~candidate_mask,
        ], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        candidate_states = encoded[:, 1:, :]
        logits = self.score_head(candidate_states).squeeze(-1)
        logits = logits.masked_fill(~candidate_mask, -1e4)
        return {
            "candidate_logits": logits,
            "candidate_states": candidate_states,
            "pool_embedding": encoded[:, 0, :],
            "pool_value_logit": self.pool_value_head(encoded[:, 0, :]).squeeze(-1),
        }


class RouteStateTransformer(nn.Module):
    """Global route value, compatibility, and bottleneck predictor."""

    def __init__(
        self,
        *,
        step_token_dim: int,
        route_feature_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_steps: int = 8,
        n_bottlenecks: int = len(BOTTLENECK_LABELS),
    ):
        super().__init__()
        self.step_token_dim = step_token_dim
        self.route_feature_dim = route_feature_dim
        self.max_steps = max_steps
        self.step_proj = nn.Linear(step_token_dim, d_model)
        self.route_proj = nn.Linear(route_feature_dim, d_model)
        self.route_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_emb = nn.Embedding(max_steps + 1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.value_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.solved_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.stock_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.progressive_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.compatibility_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.bottleneck_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_bottlenecks))

    def forward(
        self,
        step_tokens: torch.Tensor,
        step_mask: torch.Tensor | None,
        route_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if step_mask is None:
            step_mask = torch.ones(step_tokens.shape[:2], dtype=torch.bool, device=step_tokens.device)
        else:
            step_mask = step_mask.bool()
        step_tokens = step_tokens[:, : self.max_steps, :]
        step_mask = step_mask[:, : self.max_steps]
        route = self.route_token.expand(step_tokens.shape[0], 1, -1) + self.route_proj(route_features).unsqueeze(1)
        steps = self.step_proj(step_tokens)
        positions = self.pos_emb(torch.arange(steps.shape[1] + 1, device=steps.device)).unsqueeze(0)
        tokens = torch.cat([route, steps], dim=1) + positions
        padding_mask = torch.cat([
            torch.zeros(step_tokens.shape[0], 1, dtype=torch.bool, device=step_tokens.device),
            ~step_mask,
        ], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        route_state = encoded[:, 0, :]
        return {
            "route_state": route_state,
            "value_logit": self.value_head(route_state).squeeze(-1),
            "solved_logit": self.solved_head(route_state).squeeze(-1),
            "stock_logit": self.stock_head(route_state).squeeze(-1),
            "progressive_logit": self.progressive_head(route_state).squeeze(-1),
            "compatibility_logit": self.compatibility_head(route_state).squeeze(-1),
            "bottleneck_logits": self.bottleneck_head(route_state),
        }


class SearchPolicyNetwork(nn.Module):
    """Node/action policy plus value, bottleneck, and budget heads for search."""

    def __init__(
        self,
        *,
        route_model: RouteStateTransformer,
        action_feature_dim: int,
        node_feature_dim: int = 0,
        d_model: int = 128,
        n_source_budgets: int = len(SOURCE_BUDGET_GROUPS),
    ):
        super().__init__()
        self.route_model = route_model
        self.node_feature_dim = max(int(node_feature_dim or 0), 1)
        self.action_proj = nn.Linear(action_feature_dim, d_model)
        self.node_proj = nn.Linear(self.node_feature_dim, d_model)
        self.action_score = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.node_score = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.budget_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(int(n_source_budgets or 0), 1)),
        )

    def forward(
        self,
        step_tokens: torch.Tensor,
        step_mask: torch.Tensor | None,
        route_features: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        node_features: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        route_out = self.route_model(step_tokens, step_mask, route_features)
        route_vector = route_out["route_state"]
        route_state = route_vector.unsqueeze(1).expand(-1, action_features.shape[1], -1)
        action_state = self.action_proj(action_features)
        action_logits = self.action_score(torch.cat([route_state, action_state], dim=-1)).squeeze(-1)
        if action_mask is not None:
            action_logits = action_logits.masked_fill(~action_mask.bool(), -1e4)

        if node_features is None:
            node_features = torch.zeros(
                step_tokens.shape[0],
                1,
                self.node_feature_dim,
                dtype=step_tokens.dtype,
                device=step_tokens.device,
            )
        elif node_features.shape[-1] != self.node_feature_dim:
            node_features = _resize_last_dim(node_features, self.node_feature_dim)
        node_state = self.node_proj(node_features)
        node_route = route_vector.unsqueeze(1).expand(-1, node_features.shape[1], -1)
        node_logits = self.node_score(torch.cat([node_route, node_state], dim=-1)).squeeze(-1)
        if node_mask is not None:
            node_logits = node_logits.masked_fill(~node_mask.bool(), -1e4)

        return {
            **route_out,
            "action_logits": action_logits,
            "node_policy_logits": node_logits,
            "budget_logits": self.budget_head(route_vector),
        }


def _mlp(in_dim: int, hidden: int, out_dim: int, *, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
        nn.LayerNorm(out_dim),
    )


def _resize_last_dim(tensor: torch.Tensor, size: int) -> torch.Tensor:
    if tensor.shape[-1] == size:
        return tensor
    if tensor.shape[-1] > size:
        return tensor[..., :size]
    pad = torch.zeros(*tensor.shape[:-1], size - tensor.shape[-1], dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=-1)
