from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest

from cascade_planner.runtime.artifact_store import (
    ArtifactCorruptionError,
    ArtifactReferenceError,
    ArtifactStore,
    ArtifactStoreError,
)


def test_json_objects_are_canonical_deduplicated_and_verifiable(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "store")

    first = store.put_json(
        {"b": 2, "a": 1},
        logical_name="first.json",
        producer="test",
    )
    second = store.put_json(
        {"a": 1, "b": 2},
        logical_name="second.json",
        producer="test",
    )

    assert first.sha256 == second.sha256
    assert store.verify(first) is True
    assert store.read_json(second) == {"a": 1, "b": 2}
    assert len(list(store.objects_root.glob("*/*"))) == 1
    assert first.to_dict()["semantics"]["grants_no_scientific_authority"] is True


def test_materialized_compatibility_copy_cannot_mutate_object(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "store")
    ref = store.put_bytes(b"immutable")

    compatibility = store.materialize(ref, tmp_path / "run" / "artifact.bin")
    compatibility.write_bytes(b"mutable compatibility file")

    assert store.read_bytes(ref) == b"immutable"
    assert store.verify(ref) is True


def test_concurrent_identical_writers_create_one_valid_object(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "store")
    payload = b"same artifact" * 10_000

    with ThreadPoolExecutor(max_workers=8) as pool:
        refs = list(pool.map(lambda _: store.put_bytes(payload), range(32)))

    assert len({ref.sha256 for ref in refs}) == 1
    assert len(list(store.objects_root.glob("*/*"))) == 1
    assert store.read_bytes(refs[0]) == payload


def test_object_corruption_fails_closed(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "store")
    ref = store.put_bytes(b"original")
    store.object_path(ref.sha256).write_bytes(b"corrupt")

    with pytest.raises(ArtifactCorruptionError, match="size_mismatch"):
        store.verify(ref)


def test_pointer_pins_object_and_gc_requires_confirmation(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "store")
    pinned = store.put_bytes(b"pinned", logical_name="pinned.bin")
    unpinned = store.put_bytes(b"unpinned", logical_name="unused.bin")
    store.write_pointer("runs/example/latest", pinned, metadata={"revision": 1})

    loaded, pointer = store.load_pointer("runs/example/latest")
    plan = store.garbage_collection_plan(minimum_age_s=0)

    assert loaded.sha256 == pinned.sha256
    assert pointer["metadata"]["revision"] == 1
    assert [row["sha256"] for row in plan["candidates"]] == [unpinned.sha256]
    with pytest.raises(ArtifactStoreError, match="explicit_confirmation"):
        store.collect_garbage(plan)

    result = store.collect_garbage(plan, confirm=True)

    assert result["removed"] == [unpinned.sha256]
    assert store.contains(pinned)
    assert not store.contains(unpinned)


def test_pointer_rejects_path_escape_and_invalid_json(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "store")
    ref = store.put_bytes(b"data")
    with pytest.raises(ArtifactReferenceError, match="name_invalid"):
        store.write_pointer("../escape", ref)
    with pytest.raises(ArtifactStoreError, match="not_canonicalizable"):
        store.put_json({"bad": float("nan")})

    pointer_path = store.write_pointer("valid", ref)
    pointer_path.write_text(json.dumps({"bad": True}), encoding="utf-8")
    with pytest.raises(ArtifactReferenceError, match="contract_invalid"):
        store.load_pointer("valid")
