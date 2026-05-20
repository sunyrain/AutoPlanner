"""Add RCR/Parrot condition predictions to exported top-route documents."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate route-doc JSON files with predicted reaction conditions.")
    parser.add_argument("--route-doc", action="append", default=[], help="Route document JSON to annotate.")
    parser.add_argument("--route-doc-dir", help="Directory containing *_routes.json files.")
    parser.add_argument("--output-dir", help="Optional output directory. Defaults to in-place updates.")
    parser.add_argument("--vendor-root", default="vendor/ChemEnzyRetroPlanner")
    parser.add_argument("--model", choices=["rcr"], default="rcr")
    parser.add_argument("--condition-topk", type=int, default=3)
    parser.add_argument("--top-routes", type=int, default=5)
    args = parser.parse_args()

    paths = _route_doc_paths(args.route_doc, args.route_doc_dir)
    if not paths:
        raise ValueError("no route docs found")
    docs = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    rxns = _collect_reactions([doc for _path, doc in docs], top_routes=args.top_routes)
    predictor = _load_rcr_predictor(Path(args.vendor_root))
    predictions: dict[str, list[dict[str, Any]]] = {}
    failures: dict[str, str] = {}
    for idx, rxn in enumerate(rxns, start=1):
        try:
            predictions[rxn] = _predict_rcr(predictor, rxn, topk=args.condition_topk)
        except Exception as exc:  # pragma: no cover - depends on optional vendor env.
            failures[rxn] = f"{type(exc).__name__}: {exc}"
        if idx % 10 == 0 or idx == len(rxns):
            print(json.dumps({"predicted": idx, "total": len(rxns), "failures": len(failures)}, ensure_ascii=False))

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for path, doc in docs:
        annotated = _apply_predictions(doc, predictions=predictions, failures=failures, top_routes=args.top_routes)
        out_path = (output_dir / path.name) if output_dir is not None else path
        out_path.write_text(json.dumps(annotated, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(str(out_path))
    print(json.dumps({"route_docs": len(paths), "unique_reactions": len(rxns), "written": written}, indent=2, ensure_ascii=False))


def _route_doc_paths(route_docs: list[str], route_doc_dir: str | None) -> list[Path]:
    paths = [Path(path) for path in route_docs]
    if route_doc_dir:
        paths.extend(sorted(Path(route_doc_dir).glob("*_routes.json")))
    out = []
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _collect_reactions(docs: list[dict[str, Any]], *, top_routes: int) -> list[str]:
    rxns: list[str] = []
    seen = set()
    for doc in docs:
        routes = [route for route in doc.get("routes") or [] if isinstance(route, dict)]
        for route in routes[: max(0, top_routes)]:
            for step in route.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                if step.get("condition_predictions"):
                    continue
                rxn = str(step.get("reaction_smiles") or "")
                if rxn and rxn not in seen:
                    rxns.append(rxn)
                    seen.add(rxn)
    return rxns


def _load_rcr_predictor(vendor_root: Path) -> Any:
    import yaml

    from cascade_planner.baselines.chem_enzy_adapter import (
        _patch_dgl_graphbolt_optional_import,
        _patch_numpy_legacy_aliases,
        _patch_optional_easifa_import,
        _patch_torchdata_legacy_aliases,
        _vendor_pythonpath,
    )

    vendor_root = vendor_root.resolve()
    retro_root = vendor_root / "retro_planner"
    config = yaml.safe_load((retro_root / "config" / "config.yaml").read_text(encoding="utf-8"))
    rcr_config = dict((config.get("condition_config") or {}).get("rcr") or {})
    if not rcr_config:
        raise ValueError("rcr condition_config not found")
    with _vendor_pythonpath(vendor_root):
        _patch_numpy_legacy_aliases()
        _patch_torchdata_legacy_aliases()
        _patch_dgl_graphbolt_optional_import()
        _patch_optional_easifa_import(False)
        from retro_planner.common.prepare_utils import init_rcr

        return init_rcr(rcr_config, str(retro_root))


def _predict_rcr(predictor: Any, rxn: str, *, topk: int) -> list[dict[str, Any]]:
    combos, scores = predictor(rxn, topk, return_scores=True)
    rows = []
    for rank, combo in enumerate(combos, start=1):
        values = list(combo) + [None] * 6
        score = scores[rank - 1] if rank - 1 < len(scores) else None
        rows.append(
            {
                "Rank": f"Top-{rank}",
                "Temperature": values[0],
                "Solvent": values[1],
                "Reagent": values[2],
                "Catalyst": values[3],
                "Score": None if score is None else f"{float(score):.4f}",
            }
        )
    return rows


def _apply_predictions(
    doc: dict[str, Any],
    *,
    predictions: dict[str, list[dict[str, Any]]],
    failures: dict[str, str],
    top_routes: int,
) -> dict[str, Any]:
    doc = dict(doc)
    failure_rows = []
    routes = [route for route in doc.get("routes") or [] if isinstance(route, dict)]
    for route in routes[: max(0, top_routes)]:
        for step in route.get("steps") or []:
            if not isinstance(step, dict) or step.get("condition_predictions"):
                continue
            rxn = str(step.get("reaction_smiles") or "")
            if rxn in predictions:
                step["condition_predictions"] = predictions[rxn]
            elif rxn in failures:
                failure_rows.append({"reaction_smiles": rxn, "error": failures[rxn]})
        metrics = dict(route.get("metrics") or {})
        metrics["condition_coverage"] = _condition_coverage(route.get("steps") or [])
        route["metrics"] = metrics
    doc["condition_prediction_failures"] = failure_rows
    doc["condition_annotation"] = {
        "model": "rcr",
        "top_routes": top_routes,
        "unique_reaction_failures": len(failures),
    }
    return doc


def _condition_coverage(steps: list[dict[str, Any]]) -> float | None:
    if not steps:
        return None
    return sum(1 for step in steps if step.get("condition_predictions")) / len(steps)


if __name__ == "__main__":
    main()
