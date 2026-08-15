#!/usr/bin/env python3
"""Freeze a target-count-parameterized SynthAtlas strategy-closure pilot.

The checked-in outputs are a target-only manifest and a protocol.  Route
documents are written only to an evaluator root under ``data_external`` (or
outside the repository) so a live planner cannot read the reference answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.application.blind_benchmark_contract import (  # noqa: E402
    load_blind_manifest,
)
from cascade_planner.eval.strategy_closure_pilot import (  # noqa: E402
    compile_strategy_closure_leakage_pack,
    compile_strategy_closure_pilot,
)

DEFAULT_DATA_BASE_URL = "https://data.synthatlas.xyz"
DEFAULT_DATA_VERSION = "20260809-00e8823-5a1cf6"
DEFAULT_MANIFEST_SHA256 = (
    "15ebf813335d5f95b216b63b9d9728ab3bc62a4332039728a2857838cdfe7731"
)
DEFAULT_INDEX_SHA256 = (
    "2d23854cf76cdf6bea2e14f2ed5ab3e98cb07582b98cb666bc24d4025d0f76d9"
)
DEFAULT_UPSTREAM_COMMIT = "5f41a6b21e3906fde93e84c88bb91f9dc4d37e6f"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-base-url", default=DEFAULT_DATA_BASE_URL)
    parser.add_argument("--data-version", default=DEFAULT_DATA_VERSION)
    parser.add_argument("--source-manifest")
    parser.add_argument("--source-index")
    parser.add_argument("--expected-manifest-sha256", default=DEFAULT_MANIFEST_SHA256)
    parser.add_argument("--expected-index-sha256", default=DEFAULT_INDEX_SHA256)
    parser.add_argument("--target-count", type=int, default=20)
    parser.add_argument(
        "--target-manifest",
        default="benchmarks/synthatlas_strategy_closure20.v1.json",
    )
    parser.add_argument(
        "--protocol",
        default="benchmarks/synthatlas_strategy_closure20.protocol.json",
    )
    parser.add_argument(
        "--evaluator-root",
        default="data_external/synthatlas/strategy_closure20_20260812",
    )
    parser.add_argument(
        "--stock-protocol", default="benchmarks/retrostar190_v4.protocol.json"
    )
    parser.add_argument("--paper-version", default="arXiv:2608.07454v1")
    parser.add_argument("--official-repository-commit", default=DEFAULT_UPSTREAM_COMMIT)
    parser.add_argument(
        "--frozen-at",
        default=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    target_manifest_path = _repository_output(args.target_manifest)
    protocol_path = _repository_output(args.protocol)
    evaluator_root = Path(args.evaluator_root).expanduser().resolve()
    _validate_evaluator_root(evaluator_root)
    _require_fresh_outputs(target_manifest_path, protocol_path, evaluator_root)

    manifest_bytes = _read_or_fetch(
        args.source_manifest,
        f"{args.data_base_url.rstrip('/')}/manifest.json",
    )
    index_bytes = _read_or_fetch(
        args.source_index,
        f"{args.data_base_url.rstrip('/')}/{args.data_version}/index.json",
    )
    _require_digest(manifest_bytes, args.expected_manifest_sha256, "manifest")
    _require_digest(index_bytes, args.expected_index_sha256, "index")
    source_manifest = _json_value(manifest_bytes, "source manifest")
    index_rows = _json_value(index_bytes, "source index")
    if not isinstance(source_manifest, dict):
        raise SystemExit("SynthAtlas source manifest is not an object")
    if str(source_manifest.get("dataVer") or "") != args.data_version:
        raise SystemExit("SynthAtlas data version does not match manifest")
    if not isinstance(index_rows, list):
        raise SystemExit("SynthAtlas source index is not a list")

    selected_route_ids = _selected_route_ids(index_rows, target_count=args.target_count)
    route_documents = _download_routes(
        selected_route_ids,
        base_url=args.data_base_url,
        data_version=args.data_version,
        workers=args.workers,
    )
    source_snapshot = {
        "data_base_url": args.data_base_url,
        "data_version": args.data_version,
        "manifest_sha256": _digest_bytes(manifest_bytes),
        "index_sha256": _digest_bytes(index_bytes),
        "manifest_counts": dict(source_manifest.get("counts") or {}),
        "official_repository_commit": args.official_repository_commit,
        "paper_version": args.paper_version,
    }
    artifacts = compile_strategy_closure_pilot(
        index_rows=index_rows,
        route_documents=route_documents,
        source_snapshot=source_snapshot,
        target_count=args.target_count,
        case_id_prefix=f"synthatlas{args.target_count}",
        frozen_at=args.frozen_at,
        stock_binding=_stock_binding(args.stock_protocol),
    )

    evaluator_root.mkdir(parents=True)
    routes_root = evaluator_root / "routes"
    routes_root.mkdir()
    for route_id, document in sorted(route_documents.items()):
        _write_json(routes_root / f"{route_id}.json", document)
    _write_json(evaluator_root / "evaluator_pack.json", artifacts["evaluator_pack"])
    _write_bytes(evaluator_root / "source_manifest.json", manifest_bytes)
    _write_json(
        evaluator_root / "selected_index_rows.json",
        [row for row in index_rows if str(row.get("id") or "") in selected_route_ids],
    )
    _write_json(target_manifest_path, artifacts["target_manifest"])
    _write_json(protocol_path, artifacts["protocol"])
    leakage_pack = compile_strategy_closure_leakage_pack(
        manifest_file_sha256=_file_digest(target_manifest_path),
        evaluator_pack=artifacts["evaluator_pack"],
    )
    _write_json(evaluator_root / "blind_leakage_audit_pack.json", leakage_pack)
    cases = load_blind_manifest(target_manifest_path)
    receipt = {
        "schema_version": "strategy_closure_pilot_freeze_receipt.v1",
        "frozen_at": args.frozen_at,
        "target_count": len(cases),
        "route_count": len(selected_route_ids),
        "target_manifest_path": _portable_path(target_manifest_path),
        "target_manifest_file_sha256": _file_digest(target_manifest_path),
        "protocol_path": _portable_path(protocol_path),
        "protocol_file_sha256": _file_digest(protocol_path),
        "evaluator_pack_file_sha256": _file_digest(
            evaluator_root / "evaluator_pack.json"
        ),
        "leakage_audit_pack_file_sha256": _file_digest(
            evaluator_root / "blind_leakage_audit_pack.json"
        ),
        "evaluator_root": _portable_path(evaluator_root),
        "preflight": artifacts["protocol"]["preflight"],
    }
    _write_json(evaluator_root / "freeze_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


def _selected_route_ids(index_rows: list[Any], *, target_count: int) -> list[str]:
    if target_count < 1:
        raise SystemExit("target count must be positive")
    selected_targets: list[str] = []
    seen: set[str] = set()
    for row in index_rows:
        if not isinstance(row, dict):
            raise SystemExit("SynthAtlas index row is not an object")
        target = str(row.get("target_smiles") or "")
        if target not in seen:
            seen.add(target)
            selected_targets.append(target)
        if len(selected_targets) == target_count:
            break
    if len(selected_targets) != target_count:
        raise SystemExit("SynthAtlas index has insufficient unique targets")
    selected = set(selected_targets)
    route_ids = [
        str(row.get("id") or "")
        for row in index_rows
        if isinstance(row, dict) and str(row.get("target_smiles") or "") in selected
    ]
    if not route_ids or any(not value for value in route_ids):
        raise SystemExit("SynthAtlas selected route identity is invalid")
    return route_ids


def _download_routes(
    route_ids: list[str], *, base_url: str, data_version: str, workers: int
) -> dict[str, dict[str, Any]]:
    if workers < 1 or workers > 32:
        raise SystemExit("workers must be between 1 and 32")

    def fetch(route_id: str) -> tuple[str, dict[str, Any]]:
        url = f"{base_url.rstrip('/')}/{data_version}/routes/{route_id}.json"
        value = _json_value(_fetch(url), f"route {route_id}")
        if not isinstance(value, dict):
            raise RuntimeError(f"route {route_id} is not an object")
        return route_id, value

    with ThreadPoolExecutor(max_workers=min(workers, len(route_ids))) as pool:
        return dict(pool.map(fetch, route_ids))


def _read_or_fetch(path: str | None, url: str) -> bytes:
    if path:
        return Path(path).expanduser().resolve().read_bytes()
    return _fetch(url)


def _fetch(url: str) -> bytes:
    request = Request(
        url, headers={"User-Agent": "AutoPlanner/strategy-closure-freeze"}
    )
    with urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read()


def _stock_binding(path: str) -> dict[str, Any]:
    source = _repository_output(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    stock = dict((value.get("inputs") or {}).get("stock") or {})
    if not stock.get("index_sha256") or not stock.get("member_count"):
        raise SystemExit("stock protocol does not contain a frozen stock binding")
    stock["binding_protocol_path"] = _portable_path(source)
    stock["binding_protocol_sha256"] = _file_digest(source)
    return stock


def _repository_output(value: str) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise SystemExit("tracked freeze output must remain inside repository") from exc
    return resolved


def _validate_evaluator_root(path: Path) -> None:
    try:
        relative = path.relative_to(ROOT)
    except ValueError:
        return
    if not relative.parts or relative.parts[0] != "data_external":
        raise SystemExit("evaluator root inside repository must be under data_external")


def _require_fresh_outputs(*paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise SystemExit("refusing to overwrite frozen outputs: " + ", ".join(existing))


def _require_digest(payload: bytes, expected: str, label: str) -> None:
    if _digest_bytes(payload) != str(expected).strip().lower():
        raise SystemExit(f"SynthAtlas {label} SHA-256 mismatch")


def _json_value(payload: bytes, label: str) -> Any:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON for {label}") from exc


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_bytes(path, payload.encode("utf-8"))


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
