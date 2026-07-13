"""Bounded local atom mapping for materialized V4 reaction edges."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Iterable, Mapping


REACTION_MAPPING_REPORT_SCHEMA = "local_reaction_mapping_report.v1"
ReactionMapper = Callable[[list[str]], Iterable[str | Mapping[str, Any]]]


class ReactionMappingError(RuntimeError):
    """The configured local atom mapper is unavailable or malformed."""


@dataclass(frozen=True, slots=True)
class ReactionMappingConfig:
    batch_size: int = 16
    max_reactions: int = 48

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.max_reactions < 1:
            raise ValueError("reaction mapping limits must be positive")


def map_reactions_locally(
    reactions: Iterable[str],
    *,
    mapper: ReactionMapper | None = None,
    config: ReactionMappingConfig | None = None,
) -> dict[str, Any]:
    """Map unique reactions in bounded batches without any hosted model call."""

    active = config or ReactionMappingConfig()
    unique = list(dict.fromkeys(str(value).strip() for value in reactions if str(value).strip()))
    truncated = len(unique) > active.max_reactions
    selected = unique[: active.max_reactions]
    started = time.monotonic()
    selected_mapper = (
        mapper if mapper is not None else (_rxnmapper() if selected else None)
    )
    mapped: dict[str, str] = {}
    failures: list[dict[str, str]] = []
    for offset in range(0, len(selected), active.batch_size):
        assert selected_mapper is not None
        batch = selected[offset : offset + active.batch_size]
        try:
            raw_results = list(selected_mapper(batch))
        except Exception as exc:
            failures.extend(
                {
                    "reaction_smiles": reaction,
                    "reason": f"atom_mapper_error:{type(exc).__name__}:{exc}",
                }
                for reaction in batch
            )
            continue
        if len(raw_results) != len(batch):
            failures.extend(
                {
                    "reaction_smiles": reaction,
                    "reason": "atom_mapper_result_count_mismatch",
                }
                for reaction in batch
            )
            continue
        for reaction, raw in zip(batch, raw_results, strict=True):
            value = (
                str(raw.get("mapped_rxn") or raw.get("mapped_reaction_smiles") or "")
                if isinstance(raw, Mapping)
                else str(raw or "")
            )
            if ">>" not in value or ":" not in value:
                failures.append(
                    {
                        "reaction_smiles": reaction,
                        "reason": "atom_mapper_output_invalid",
                    }
                )
            else:
                mapped[reaction] = value
    return {
        "schema_version": REACTION_MAPPING_REPORT_SCHEMA,
        "backend": "injected" if mapper is not None else "rxnmapper",
        "requested_count": len(unique),
        "mapped_count": len(mapped),
        "failure_count": len(failures),
        "truncated": truncated,
        "mapped_reactions": mapped,
        "failures": failures,
        "elapsed_s": (
            round(max(0.0, time.monotonic() - started), 6) if selected else 0.0
        ),
        "semantics": {
            "local_only": True,
            "hosted_model_calls": 0,
            "mapping_is_not_reaction_proof": True,
        },
    }


def _rxnmapper() -> ReactionMapper:
    try:
        from rxnmapper import RXNMapper
    except (ImportError, OSError) as exc:
        raise ReactionMappingError(f"rxnmapper_unavailable:{type(exc).__name__}") from exc
    try:
        model = RXNMapper()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ReactionMappingError(f"rxnmapper_initialization_failed:{type(exc).__name__}") from exc

    def run(values: list[str]) -> Iterable[Mapping[str, Any]]:
        return model.get_attention_guided_atom_maps(values)

    return run


__all__ = [
    "REACTION_MAPPING_REPORT_SCHEMA",
    "ReactionMapper",
    "ReactionMappingConfig",
    "ReactionMappingError",
    "map_reactions_locally",
]
