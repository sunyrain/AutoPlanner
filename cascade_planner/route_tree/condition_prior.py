"""BRENDA condition priors for enzymatic route-tree actions."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BRENDA_CONDITION_PRIOR_ENV = "AUTOPLANNER_ROUTE_TREE_BRENDA_CONDITION_PRIOR"
BRENDA_CONDITION_PRIOR_CACHE_ENV = "AUTOPLANNER_ROUTE_TREE_BRENDA_CONDITION_PRIOR_CACHE"
TRUE_VALUES = {"1", "true", "yes", "on"}
DEFAULT_CACHE_PATHS = (
    REPO_ROOT / "results/shared/brenda_lookup_full.json",
    REPO_ROOT / "results/shared/brenda_lookup.json",
)


def brenda_condition_prior_from_env(
    *,
    ec: str,
    metadata: dict[str, Any] | None = None,
    fill_T: bool = True,
    fill_pH: bool = True,
) -> dict[str, Any] | None:
    """Return a conservative BRENDA T/pH prior for an enzymatic EC query.

    The prior is opt-in through ``AUTOPLANNER_ROUTE_TREE_BRENDA_CONDITION_PRIOR``.
    It never falls back to hard-coded global values; if the local BRENDA cache
    has no EC or EC-prefix support, no prior is emitted.
    """
    if not _env_enabled():
        return None
    ec_query = _normalize_ec_query(ec)
    if not ec_query:
        return None
    cache_path = _cache_path_from_env()
    if cache_path is None:
        return None
    lookup = _lookup_from_cache(str(cache_path))
    if lookup is None:
        return None

    metadata = dict(metadata or {})
    organism = _organism_from_metadata(metadata)
    out: dict[str, Any] = {
        "schema_version": "route_tree_brenda_condition_prior.v1",
        "source": "brenda_condition_prior",
        "cache_path": str(cache_path),
        "ec": ec_query,
        "organism": organism or "",
    }
    if fill_T:
        value, source = _predict_value(lookup, "T_opt", ec_query, organism)
        if value is not None:
            out["temperature_c"] = round(float(value), 6)
            out["temperature_source"] = source
    if fill_pH:
        value, source = _predict_value(lookup, "pH_opt", ec_query, organism)
        if value is not None:
            out["ph"] = round(float(value), 6)
            out["ph_source"] = source

    if "temperature_c" not in out and "ph" not in out:
        return None
    return out


def condition_prediction_from_prior(prior: dict[str, Any]) -> dict[str, Any]:
    """Convert a BRENDA prior payload into the route-export condition row."""
    return {
        "source": "brenda_condition_prior",
        "condition_label": "BRENDA EC condition prior",
        "ec": prior.get("ec", ""),
        "organism": prior.get("organism", ""),
        "temperature_c": prior.get("temperature_c"),
        "temperature_source": prior.get("temperature_source"),
        "ph": prior.get("ph"),
        "ph_source": prior.get("ph_source"),
    }


def clear_brenda_condition_prior_cache() -> None:
    _lookup_from_cache.cache_clear()


def _env_enabled() -> bool:
    return str(os.environ.get(BRENDA_CONDITION_PRIOR_ENV) or "").strip().lower() in TRUE_VALUES


def _cache_path_from_env() -> Path | None:
    raw = str(os.environ.get(BRENDA_CONDITION_PRIOR_CACHE_ENV) or "").strip()
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        return path if path.exists() else None
    for path in DEFAULT_CACHE_PATHS:
        if path.exists():
            return path
    return None


@lru_cache(maxsize=8)
def _lookup_from_cache(path_text: str) -> Any | None:
    path = Path(path_text)
    if not path.exists():
        return None
    try:
        from cascade_planner.data.brenda_conditions import build_ec_lookup

        raw = json.loads(path.read_text(encoding="utf-8"))
        brenda_data: dict[tuple[str, str], dict[str, float]] = {}
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            ec, organism = _split_cache_key(str(key))
            if not ec:
                continue
            entry: dict[str, float] = {}
            for field in ("T_opt", "pH_opt"):
                try:
                    if field in value:
                        entry[field] = float(value[field])
                except (TypeError, ValueError):
                    continue
            if entry:
                brenda_data[(ec, organism.lower())] = entry
        if not brenda_data:
            return None
        return build_ec_lookup(brenda_data)
    except Exception:
        return None


def _split_cache_key(key: str) -> tuple[str, str]:
    if "||" in key:
        ec, organism = key.split("||", 1)
        return _normalize_ec_query(ec), organism.strip()
    return _normalize_ec_query(key), ""


def _predict_value(lookup: Any, field: str, ec: str, organism: str) -> tuple[float | None, str]:
    by_ec_org = getattr(lookup, "_by_ec_org", {}) or {}
    org = (organism or "").strip().lower()
    if org and ec in by_ec_org:
        entry = (by_ec_org.get(ec) or {}).get(org)
        if entry and field in entry:
            return float(entry[field]), "brenda_ec4_organism"

    by_prefix = getattr(lookup, "_by_prefix", {}) or {}
    for prefix in _ec_prefixes(ec):
        entry = by_prefix.get(prefix)
        if entry and field in entry:
            return float(entry[field]), f"brenda_ec{prefix.count('.') + 1}_median"
    return None, ""


def _organism_from_metadata(metadata: dict[str, Any]) -> str:
    evidence = metadata.get("evidence") if isinstance(metadata.get("evidence"), dict) else {}
    for payload in (metadata, evidence):
        for key in ("organism", "uniprot_lookup_organism", "species"):
            value = payload.get(key)
            if value not in (None, "", [], {}):
                return str(value).strip()
    return ""


def _normalize_ec_query(ec: str) -> str:
    parts = []
    for raw in str(ec or "").strip().split("."):
        token = raw.strip()
        if not token or token.lower() in {"x", "-", "*", "n/a", "na", "none"}:
            break
        if not token.isdigit():
            break
        parts.append(token)
    return ".".join(parts)


def _ec_prefixes(ec: str) -> list[str]:
    parts = [part for part in str(ec or "").split(".") if part]
    return [".".join(parts[:idx]) for idx in range(len(parts), 0, -1)]
