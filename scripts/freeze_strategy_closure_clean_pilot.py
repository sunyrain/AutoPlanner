#!/usr/bin/env python3
"""Freeze the first 20 SynthAtlas targets passing a pre-run knowledge audit.

Selection is deterministic and happens before any live arm runs.  A target is
excluded when its identity, public synonym, or a route-derived intermediate
with at least eight heavy atoms already appears in tracked non-inventory
repository knowledge.  The exclusion audit is evaluator-only.
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

from rdkit import Chem

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.application.blind_benchmark_contract import (  # noqa: E402
    BLIND_CASE_SCHEMA,
    BlindCase,
    audit_blind_preflight,
    canonical_smiles,
    load_blind_manifest,
)
from cascade_planner.eval.strategy_closure_pilot import (  # noqa: E402
    DEFAULT_ACCEPTANCE,
    DEFAULT_BUDGET,
    compile_strategy_closure_leakage_pack,
    compile_strategy_closure_pilot,
)

DATA_BASE_URL = "https://data.synthatlas.xyz"
DATA_VERSION = "20260809-00e8823-5a1cf6"
MANIFEST_SHA256 = "15ebf813335d5f95b216b63b9d9728ab3bc62a4332039728a2857838cdfe7731"
INDEX_SHA256 = "2d23854cf76cdf6bea2e14f2ed5ab3e98cb07582b98cb666bc24d4025d0f76d9"
UPSTREAM_COMMIT = "5f41a6b21e3906fde93e84c88bb91f9dc4d37e6f"
SELECTION_ALGORITHM = (
    "first_unique_canonical_target_passing_prerun_repository_absence_audit.v1"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        default=str(Path(sys.prefix) / "unused-synthatlas-manifest.json"),
    )
    parser.add_argument(
        "--source-index",
        default=str(Path(sys.prefix) / "unused-synthatlas-index.json"),
    )
    parser.add_argument("--target-count", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--target-manifest",
        default="benchmarks/synthatlas_strategy_closure_clean20.v1.json",
    )
    parser.add_argument(
        "--protocol",
        default="benchmarks/synthatlas_strategy_closure_clean20.protocol.json",
    )
    parser.add_argument(
        "--evaluator-root",
        default="data_external/synthatlas/strategy_closure_clean20_20260812",
    )
    parser.add_argument(
        "--allowed-prior-artifact",
        action="append",
        default=["benchmarks/synthatlas_strategy_closure20.v1.json"],
    )
    parser.add_argument(
        "--stock-protocol", default="benchmarks/retrostar190_v4.protocol.json"
    )
    parser.add_argument(
        "--frozen-at",
        default=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )
    args = parser.parse_args(argv)

    target_manifest_path = _repo_path(args.target_manifest)
    protocol_path = _repo_path(args.protocol)
    evaluator_root = _repo_path(args.evaluator_root)
    if evaluator_root.relative_to(ROOT).parts[0] != "data_external":
        raise SystemExit("evaluator root must be under data_external")
    for path in (target_manifest_path, protocol_path, evaluator_root):
        if path.exists():
            raise SystemExit(f"refusing to overwrite frozen output: {path}")

    manifest_bytes = _read_or_fetch(
        args.source_manifest,
        f"{DATA_BASE_URL}/manifest.json",
    )
    index_bytes = _read_or_fetch(
        args.source_index,
        f"{DATA_BASE_URL}/{DATA_VERSION}/index.json",
    )
    _require_digest(manifest_bytes, MANIFEST_SHA256, "manifest")
    _require_digest(index_bytes, INDEX_SHA256, "index")
    public_manifest = json.loads(manifest_bytes)
    index_rows = json.loads(index_bytes)
    if public_manifest.get("dataVer") != DATA_VERSION or not isinstance(
        index_rows, list
    ):
        raise SystemExit("SynthAtlas source snapshot shape mismatch")
    candidates = _candidate_rows(index_rows, max_candidates=args.max_candidates)
    allowed = [_repo_path(value) for value in args.allowed_prior_artifact]
    selected: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    all_downloaded: dict[str, dict[str, Any]] = {}
    for candidate_index, candidate in enumerate(candidates, start=1):
        documents = _download_routes(candidate["route_ids"], workers=args.workers)
        all_downloaded.update(documents)
        intermediates = _route_intermediates(documents.values())
        case = BlindCase(
            case_id=f"selection-candidate-{candidate_index:04d}",
            target_name="opaque target",
            target_smiles=candidate["target_smiles"],
            acceptance=DEFAULT_ACCEPTANCE,
            budget=DEFAULT_BUDGET,
            schema_version=BLIND_CASE_SCHEMA,
        )
        source_name = str(candidate["source_target_name"] or "").strip()
        synonym_values = (
            []
            if source_name.casefold() in {"", "not named", "unnamed", "unknown"}
            else [source_name]
        )
        report = audit_blind_preflight(
            case,
            repository_root=ROOT,
            run_dir=evaluator_root / "selection-runs" / case.case_id,
            additional_allowed_paths=allowed,
            additional_leakage_needles={
                "target_synonym": synonym_values,
                "key_intermediate_smiles": intermediates,
            },
            target_synonym_not_applicable_reason=(
                "public source provides no usable target name"
                if not synonym_values
                else ""
            ),
        )
        accepted = report.get("accepted") is True
        audit_rows.append(
            {
                "candidate_ordinal": candidate_index,
                "target_identity_sha256": hashlib.sha256(
                    candidate["target_smiles"].encode("utf-8")
                ).hexdigest(),
                "route_variant_count": len(candidate["route_ids"]),
                "accepted": accepted,
                "reasons": list(report.get("reasons") or []),
                "repository_matches": list(report.get("repository_matches") or []),
            }
        )
        if accepted:
            selected.append(candidate)
            print(
                json.dumps(
                    {
                        "candidate": candidate_index,
                        "accepted": True,
                        "selected_count": len(selected),
                    }
                ),
                flush=True,
            )
            if len(selected) == args.target_count:
                break
        else:
            print(
                json.dumps(
                    {
                        "candidate": candidate_index,
                        "accepted": False,
                        "reasons": report.get("reasons") or [],
                    }
                ),
                flush=True,
            )
    if len(selected) != args.target_count:
        raise SystemExit("insufficient clean targets within max candidates")
    selected_target_smiles = [row["target_smiles"] for row in selected]
    selected_route_ids = {route_id for row in selected for route_id in row["route_ids"]}
    selected_documents = {
        route_id: all_downloaded[route_id] for route_id in selected_route_ids
    }
    artifacts = compile_strategy_closure_pilot(
        index_rows=index_rows,
        route_documents=selected_documents,
        source_snapshot={
            "data_base_url": DATA_BASE_URL,
            "data_version": DATA_VERSION,
            "manifest_sha256": MANIFEST_SHA256,
            "index_sha256": INDEX_SHA256,
            "manifest_counts": dict(public_manifest.get("counts") or {}),
            "official_repository_commit": UPSTREAM_COMMIT,
            "paper_version": "arXiv:2608.07454v1",
        },
        target_count=args.target_count,
        frozen_at=args.frozen_at,
        stock_binding=_stock_binding(args.stock_protocol),
        selected_target_smiles=selected_target_smiles,
        selection_algorithm=SELECTION_ALGORITHM,
        selection_audit=audit_rows,
        case_id_prefix="synthatlas-clean20",
    )
    evaluator_root.mkdir(parents=True)
    routes_root = evaluator_root / "routes"
    routes_root.mkdir()
    for route_id, document in sorted(selected_documents.items()):
        _write_json(routes_root / f"{route_id}.json", document)
    _write_json(evaluator_root / "evaluator_pack.json", artifacts["evaluator_pack"])
    _write_json(evaluator_root / "selection_audit.json", {"rows": audit_rows})
    _write_bytes(evaluator_root / "source_manifest.json", manifest_bytes)
    _write_json(target_manifest_path, artifacts["target_manifest"])
    _write_json(protocol_path, artifacts["protocol"])
    leakage = compile_strategy_closure_leakage_pack(
        manifest_file_sha256=_file_digest(target_manifest_path),
        evaluator_pack=artifacts["evaluator_pack"],
    )
    _write_json(evaluator_root / "blind_leakage_audit_pack.json", leakage)
    load_blind_manifest(target_manifest_path)
    receipt = {
        "schema_version": "strategy_closure_clean_pilot_freeze_receipt.v1",
        "frozen_at": args.frozen_at,
        "target_count": len(selected),
        "candidate_count_audited": len(audit_rows),
        "pre_run_exclusion_count": sum(not row["accepted"] for row in audit_rows),
        "route_count": len(selected_documents),
        "target_manifest_sha256": _file_digest(target_manifest_path),
        "protocol_sha256": _file_digest(protocol_path),
        "evaluator_pack_sha256": _file_digest(evaluator_root / "evaluator_pack.json"),
        "leakage_pack_sha256": _file_digest(
            evaluator_root / "blind_leakage_audit_pack.json"
        ),
        "selection_algorithm": SELECTION_ALGORITHM,
    }
    _write_json(evaluator_root / "freeze_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


def _candidate_rows(
    index_rows: list[Any], *, max_candidates: int
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    ordered: list[str] = []
    for raw in index_rows:
        if not isinstance(raw, dict):
            raise SystemExit("SynthAtlas index row is not an object")
        target = canonical_smiles(raw.get("target_smiles"))
        if not target:
            raise SystemExit("SynthAtlas index target is invalid")
        if target not in grouped:
            if len(ordered) == max_candidates:
                continue
            ordered.append(target)
            grouped[target] = {
                "target_smiles": target,
                "source_target_name": str(raw.get("name") or "unnamed target"),
                "route_ids": [],
            }
        if target in grouped:
            grouped[target]["route_ids"].append(str(raw.get("id") or ""))
    return [grouped[target] for target in ordered]


def _route_intermediates(documents: Any) -> list[str]:
    values: set[str] = set()
    for document in documents:
        target = canonical_smiles(document.get("target_smiles"))
        for step in document.get("steps") or []:
            reaction = str(step.get("rxn_smiles") or "")
            parts = reaction.split(">")
            if len(parts) != 3:
                continue
            for raw in (parts[0] + "." + parts[2]).split("."):
                canonical = canonical_smiles(raw)
                molecule = Chem.MolFromSmiles(canonical) if canonical else None
                if molecule is not None and molecule.GetNumHeavyAtoms() >= 8:
                    values.add(canonical)
        values.discard(target)
    return sorted(values)


def _download_routes(
    route_ids: list[str], *, workers: int
) -> dict[str, dict[str, Any]]:
    def fetch(route_id: str) -> tuple[str, dict[str, Any]]:
        url = f"{DATA_BASE_URL}/{DATA_VERSION}/routes/{route_id}.json"
        value = json.loads(_fetch(url))
        if not isinstance(value, dict):
            raise RuntimeError(f"route is not an object: {route_id}")
        return route_id, value

    with ThreadPoolExecutor(max_workers=min(max(1, workers), len(route_ids))) as pool:
        return dict(pool.map(fetch, route_ids))


def _read_or_fetch(path: str, url: str) -> bytes:
    source = Path(path).expanduser().resolve()
    return source.read_bytes() if source.is_file() else _fetch(url)


def _fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "AutoPlanner/clean-pilot-freeze"})
    with urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read()


def _stock_binding(value: str) -> dict[str, Any]:
    path = _repo_path(value)
    protocol = json.loads(path.read_text(encoding="utf-8"))
    stock = dict(protocol["inputs"]["stock"])
    stock["binding_protocol_path"] = path.relative_to(ROOT).as_posix()
    stock["binding_protocol_sha256"] = _file_digest(path)
    return stock


def _repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    resolved.relative_to(ROOT)
    return resolved


def _require_digest(payload: bytes, expected: str, label: str) -> None:
    if hashlib.sha256(payload).hexdigest() != expected:
        raise SystemExit(f"SynthAtlas {label} SHA-256 mismatch")


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(
        path,
        (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode(),
    )


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
