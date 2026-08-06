#!/usr/bin/env python
"""Run AiZynthFinder's ONNX expansion policy behind a neutral JSON contract.

This file intentionally does not import ``cascade_planner`` or any ChemEnzy
adapter.  It is designed to run in an isolated Python environment containing
only AiZynthFinder and its own dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path
from typing import Any


REQUEST_SCHEMA = "autoplanner.one_step_sidecar_request.v1"
RESPONSE_SCHEMA = "autoplanner.one_step_sidecar_response.v1"


def _canonical_json_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_smiles(smiles: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _canonical_side(smiles_values: list[str]) -> tuple[str, ...]:
    fragments: list[str] = []
    for value in smiles_values:
        for fragment in str(value or "").split("."):
            canonical = _canonical_smiles(fragment.strip())
            if canonical:
                fragments.append(canonical)
    return tuple(sorted(fragments))


def _validate_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != REQUEST_SCHEMA:
        raise ValueError("unexpected request schema")
    semantics = payload.get("semantics")
    if not isinstance(semantics, dict):
        raise ValueError("request omitted semantic safety flags")
    if semantics.get("shadow_only") is not True:
        raise ValueError("sidecar only accepts shadow-only requests")
    if semantics.get("canonical_route_write_authority") is not False:
        raise ValueError("sidecar cannot accept canonical write authority")
    if semantics.get("candidate_is_not_evidence") is not True:
        raise ValueError("sidecar candidates cannot be treated as evidence")
    queries = payload.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("request queries must be a non-empty list")
    seen: set[str] = set()
    for query in queries:
        if not isinstance(query, dict):
            raise ValueError("each query must be an object")
        query_id = str(query.get("query_id") or "")
        product_smiles = str(query.get("product_smiles") or "")
        top_k = int(query.get("top_k") or 0)
        if not query_id or query_id in seen:
            raise ValueError("query_id must be non-empty and unique")
        if not product_smiles:
            raise ValueError(f"query {query_id} has no product_smiles")
        if top_k < 1 or top_k > 100:
            raise ValueError(f"query {query_id} top_k must be between 1 and 100")
        seen.add(query_id)
    return payload


def _load_policy(model_path: Path, templates_path: Path, policy_name: str, cutoff: int) -> Any:
    from aizynthfinder.aizynthfinder import AiZynthFinder

    config = {
        "expansion": {
            policy_name: {
                "type": "template-based",
                "model": str(model_path),
                "template": str(templates_path),
                "cutoff_cumulative": 0.999,
                "cutoff_number": cutoff,
            }
        }
    }
    finder = AiZynthFinder(configdict=config)
    finder.expansion_policy.select(policy_name)
    return finder.expansion_policy


def _predict(policy: Any, query: dict[str, Any]) -> dict[str, Any]:
    from aizynthfinder.chem import TreeMolecule

    started = time.perf_counter()
    product = _canonical_smiles(str(query["product_smiles"]))
    if not product:
        return {
            "query_id": query["query_id"],
            "status": "invalid_product",
            "product_smiles": str(query["product_smiles"]),
            "candidates": [],
            "diagnostics": {"elapsed_s": round(time.perf_counter() - started, 6)},
        }
    molecule = TreeMolecule(parent=None, smiles=product)
    actions, priors = policy.get_actions([molecule])
    actions = list(actions or [])
    priors = list(priors or [])
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    failures = 0
    for action_index, action in enumerate(actions):
        try:
            options = list(getattr(action, "reactants", []) or [])
        except Exception:
            failures += 1
            continue
        metadata = dict(getattr(action, "metadata", {}) or {})
        raw_score = metadata.get("policy_probability")
        if raw_score is None and action_index < len(priors):
            raw_score = priors[action_index]
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.0
        for option in options:
            option_items = list(option) if isinstance(option, (list, tuple)) else [option]
            side = _canonical_side(
                [str(getattr(item, "smiles", "") or item) for item in option_items]
            )
            if not side or side in seen:
                continue
            seen.add(side)
            candidates.append(
                {
                    "rank": len(candidates) + 1,
                    "reactant_smiles": list(side),
                    "reaction_smiles": ".".join(side) + f">>{product}",
                    "score": score,
                    "source": "aizynthfinder_onnx.uspto",
                    "proposal_type": "template_policy",
                    "template_code": str(metadata.get("template_code") or ""),
                    "template_rank": action_index + 1,
                }
            )
            if len(candidates) >= int(query["top_k"]):
                break
        if len(candidates) >= int(query["top_k"]):
            break
    return {
        "query_id": query["query_id"],
        "status": "ok",
        "product_smiles": product,
        "candidates": candidates,
        "diagnostics": {
            "elapsed_s": round(time.perf_counter() - started, 6),
            "raw_action_count": len(actions),
            "template_application_failures": failures,
        },
    }


def _forbidden_imports() -> list[str]:
    forbidden: list[str] = []
    for name, module in sys.modules.items():
        module_path = str(getattr(module, "__file__", "") or "").replace("\\", "/").lower()
        lowered_name = str(name).lower()
        if (
            "chem_enzy" in lowered_name
            or lowered_name == "retro_planner"
            or lowered_name.startswith("retro_planner.")
            or "/vendor/chemenzyretroplanner/" in module_path
        ):
            forbidden.append(str(name))
    return sorted(set(forbidden))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--templates", required=True, type=Path)
    parser.add_argument("--policy-name", default="uspto")
    parser.add_argument("--cutoff-number", type=int, default=100)
    args = parser.parse_args()

    started = time.perf_counter()
    try:
        request = _validate_request(json.load(sys.stdin))
        if not args.model.is_file():
            raise FileNotFoundError(f"model artifact not found: {args.model}")
        if not args.templates.is_file():
            raise FileNotFoundError(f"template artifact not found: {args.templates}")
        policy = _load_policy(
            args.model.resolve(),
            args.templates.resolve(),
            str(args.policy_name),
            max(1, min(1000, int(args.cutoff_number))),
        )
        results = [_predict(policy, query) for query in request["queries"]]
        forbidden_imports = _forbidden_imports()
        response = {
            "schema_version": RESPONSE_SCHEMA,
            "request_id": request["request_id"],
            "request_sha256": _canonical_json_sha256(request),
            "status": "ok" if not forbidden_imports else "isolation_violation",
            "provider": {
                "provider_id": "aizynthfinder_onnx.uspto",
                "software": "aizynthfinder",
                "software_version": importlib.metadata.version("aizynthfinder"),
                "software_license": "MIT",
                "model_artifact_license": "unknown_requires_provenance_review",
                "model_sha256": _file_sha256(args.model),
                "templates_sha256": _file_sha256(args.templates),
            },
            "results": results,
            "diagnostics": {
                "elapsed_s": round(time.perf_counter() - started, 6),
                "python": sys.version.split()[0],
                "forbidden_imports": forbidden_imports,
            },
            "semantics": {
                "shadow_only": True,
                "canonical_route_write_authority": False,
                "candidate_is_not_evidence": True,
            },
        }
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    json.dump(response, sys.stdout, ensure_ascii=False, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
