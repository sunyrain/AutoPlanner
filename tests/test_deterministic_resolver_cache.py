from __future__ import annotations

from pathlib import Path

from cascade_planner.harness.deterministic_resolver_cache import (
    DeterministicResolverCache,
)


def _cache(path: Path, *, authority: str = "parser.v1", ttl: float = 60.0):
    return DeterministicResolverCache(
        path,
        authority_id=authority,
        opsin_base_url="https://opsin.example",
        pubchem_base_url="https://pubchem.example",
        failure_ttl_s=ttl,
    )


def test_resolver_cache_persists_success_and_failure(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    cache.put("structure", "compound a", "CCO", success=True)
    cache.put("structure", "compound b", "", success=False)
    report = cache.flush()

    reloaded = _cache(tmp_path)

    assert report["entry_count"] == 2
    assert reloaded.get("structure", "compound a") == (True, "CCO")
    assert reloaded.get("structure", "compound b") == (True, "")
    ref = report["artifact_ref"]
    assert ref["semantics"]["grants_no_scientific_authority"] is True


def test_resolver_cache_namespace_change_invalidates_entries(
    tmp_path: Path,
) -> None:
    cache = _cache(tmp_path, authority="parser.v1")
    cache.put("structure", "compound", "CC", success=True)
    cache.flush()

    changed = _cache(tmp_path, authority="parser.v2")

    assert changed.get("structure", "compound") == (False, None)


def test_expired_failure_is_a_miss(tmp_path: Path) -> None:
    cache = _cache(tmp_path, ttl=0.0)
    cache.put("structure", "unresolved", "", success=False)
    cache.flush()

    reloaded = _cache(tmp_path, ttl=0.0)

    assert reloaded.get("structure", "unresolved") == (False, None)
