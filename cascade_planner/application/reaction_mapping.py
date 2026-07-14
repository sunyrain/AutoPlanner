"""Bounded local atom mapping for materialized V4 reaction edges."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import RLock
import time
from typing import Any, Callable, Iterable, Mapping


REACTION_MAPPING_REPORT_SCHEMA = "local_reaction_mapping_report.v1"
ReactionMapper = Callable[[list[str]], Iterable[str | Mapping[str, Any]]]


_RXNMAPPER_INSTANCE: Any | None = None
_RXNMAPPER_INIT_LOCK = RLock()
_RXNMAPPER_RUN_LOCK = RLock()


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
        "backend": (
            "injected"
            if mapper is not None
            else str(
                getattr(selected_mapper, "_autoplanner_backend", "rxnmapper")
            )
        ),
        "mapper_python": (
            ""
            if mapper is not None
            else str(getattr(selected_mapper, "_autoplanner_python", sys.executable))
        ),
        "mapper_version": str(
            getattr(selected_mapper, "_autoplanner_version", "")
        ),
        "mapper_instance_reused": bool(
            getattr(selected_mapper, "_autoplanner_instance_reused", False)
        ),
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
    global _RXNMAPPER_INSTANCE

    try:
        from rxnmapper import RXNMapper
    except (ImportError, OSError) as exc:
        isolated = _isolated_rxnmapper()
        if isolated is not None:
            return isolated
        raise ReactionMappingError(f"rxnmapper_unavailable:{type(exc).__name__}") from exc
    with _RXNMAPPER_INIT_LOCK:
        reused = _RXNMAPPER_INSTANCE is not None
        if _RXNMAPPER_INSTANCE is None:
            try:
                _RXNMAPPER_INSTANCE = RXNMapper()
            except (OSError, RuntimeError, ValueError) as exc:
                raise ReactionMappingError(
                    f"rxnmapper_initialization_failed:{type(exc).__name__}"
                ) from exc
        model = _RXNMAPPER_INSTANCE

    def run(values: list[str]) -> Iterable[Mapping[str, Any]]:
        # The attention mapper holds mutable model state.  One process-level
        # instance removes repeated multi-second initialization while this
        # lock keeps concurrent web jobs deterministic and isolated.
        with _RXNMAPPER_RUN_LOCK:
            return model.get_attention_guided_atom_maps(values)

    run._autoplanner_backend = "rxnmapper_shared_process"  # type: ignore[attr-defined]
    run._autoplanner_python = sys.executable  # type: ignore[attr-defined]
    run._autoplanner_instance_reused = reused  # type: ignore[attr-defined]
    try:
        version = metadata.version("rxnmapper")
    except metadata.PackageNotFoundError:
        version = ""
    run._autoplanner_version = version  # type: ignore[attr-defined]
    return run


def _isolated_rxnmapper() -> ReactionMapper | None:
    executable = _discover_rxnmapper_python()
    if executable is None:
        return None

    def run(values: list[str]) -> Iterable[Mapping[str, Any]]:
        program = (
            "import json,sys; from rxnmapper import RXNMapper; "
            "values=json.loads(sys.stdin.read()); "
            "print(json.dumps(RXNMapper().get_attention_guided_atom_maps(values)))"
        )
        completed = subprocess.run(
            [str(executable), "-c", program],
            input=json.dumps(values),
            capture_output=True,
            text=True,
            timeout=180.0,
            check=False,
        )
        if completed.returncode != 0:
            raise ReactionMappingError(
                f"isolated_rxnmapper_exit_{completed.returncode}:"
                f"{completed.stderr[-500:]}"
            )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise ReactionMappingError("isolated_rxnmapper_output_missing")
        result = json.loads(lines[-1])
        if not isinstance(result, list):
            raise ReactionMappingError("isolated_rxnmapper_output_invalid")
        return result

    run._autoplanner_backend = "rxnmapper_isolated_subprocess"  # type: ignore[attr-defined]
    run._autoplanner_python = str(executable)  # type: ignore[attr-defined]
    return run


def _discover_rxnmapper_python() -> Path | None:
    configured = str(os.environ.get("AUTOPLANNER_RXNMAPPER_PYTHON") or "").strip()
    local_app_data = Path(str(os.environ.get("LOCALAPPDATA") or ""))
    candidates = [
        Path(configured) if configured else None,
        local_app_data / "Programs" / "Python" / "Python312" / "python.exe",
        Path(shutil.which("python") or ""),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if candidate is None or not str(candidate):
            continue
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        key = str(resolved).casefold()
        if key in seen or resolved == Path(sys.executable).resolve() or not resolved.is_file():
            continue
        seen.add(key)
        try:
            probe = subprocess.run(
                [
                    str(resolved),
                    "-c",
                    "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('rxnmapper') else 2)",
                ],
                capture_output=True,
                text=True,
                timeout=10.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if probe.returncode == 0:
            return resolved
    return None


__all__ = [
    "REACTION_MAPPING_REPORT_SCHEMA",
    "ReactionMapper",
    "ReactionMappingConfig",
    "ReactionMappingError",
    "map_reactions_locally",
]
