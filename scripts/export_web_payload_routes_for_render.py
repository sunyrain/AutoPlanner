"""Export a web_payload route bundle into renderer-ready route docs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-name", default="")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    payload_path = Path(args.payload)
    output_dir = Path(args.output_dir)
    route_doc_dir = output_dir / "route_docs"
    route_doc_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    target = str(payload.get("target") or payload.get("target_smiles") or "")
    name = str(args.target_name or payload.get("target_name") or "target")
    routes = [route for route in payload.get("routes") or [] if isinstance(route, dict)]
    selected = routes[: max(0, int(args.top_k or 0))]
    doc_routes = [_route_doc(route, rank=idx) for idx, route in enumerate(selected, start=1)]
    doc = {
        "schema_version": "web_payload_route_doc.v1",
        "source_payload": str(payload_path),
        "target": target,
        "target_smiles": target,
        "target_name": name,
        "cascade_id": name,
        "source_route_count": len(routes),
        "search_status": payload.get("search_status"),
        "proposal_gate": payload.get("proposal_gate"),
        "routes": doc_routes,
    }
    doc_path = route_doc_dir / f"{_slug(name)}_top{int(args.top_k)}_routes.json"
    doc_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "schema_version": "web_payload_route_export.v1",
        "payload": str(payload_path),
        "route_doc": str(doc_path),
        "target": target,
        "target_name": name,
        "source_route_count": len(routes),
        "selected_routes": len(doc_routes),
        "selected_steps": [route.get("n_steps") or len(route.get("steps") or []) for route in doc_routes],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(_summary_md(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "route_doc": str(doc_path)}, indent=2))


def _route_doc(route: dict[str, Any], *, rank: int) -> dict[str, Any]:
    steps = [_step_doc(step, index=idx) for idx, step in enumerate(route.get("steps") or [], start=1)]
    product_audit = {
        "route_class": "source_supported_semisynthesis"
        if (route.get("metrics") or {}).get("source_supported_semisynthesis")
        else "web_payload_route",
        "condition_audit": _condition_audit(steps),
        "issues": [],
        "tags": [],
        "risk_order": 0,
    }
    return {
        "rank": rank,
        "display_rank": rank,
        "original_rank": route.get("original_route_rank") or route.get("route_rank") or rank,
        "score": route.get("score"),
        "solved": bool((route.get("metrics") or {}).get("route_solved") or (route.get("metrics") or {}).get("strict_stock_solve")),
        "n_steps": len(steps),
        "metrics": {**dict(route.get("metrics") or {}), "condition_coverage": _condition_coverage(steps)},
        "product_audit": product_audit,
        "steps": steps,
    }


def _step_doc(step: dict[str, Any], *, index: int) -> dict[str, Any]:
    out = dict(step)
    out["index"] = index
    out.setdefault("reaction_smiles", str(step.get("reaction_smiles") or step.get("rxn_smiles") or ""))
    out.setdefault("product", _reaction_product(out.get("reaction_smiles")))
    if not out.get("main_reactant"):
        lhs = _rxn_lhs_parts(out.get("reaction_smiles"))
        if lhs:
            out["main_reactant"] = lhs[0]
            out.setdefault("aux_reactants", lhs[1:])
    out.setdefault("condition_predictions", [row for row in step.get("condition_predictions") or [] if isinstance(row, dict)])
    return out


def _condition_audit(steps: list[dict[str, Any]]) -> dict[str, Any]:
    step_rows = []
    for idx, step in enumerate(steps, start=1):
        condition = (step.get("condition_predictions") or [{}])[0]
        temp = _safe_float(condition.get("Temperature") or condition.get("temperature_c"))
        score = _safe_float(condition.get("Score") or condition.get("score"))
        issues = []
        if not condition:
            issues.append("missing_condition_prediction")
        risk = "warn" if issues else "ok"
        step_rows.append(
            {
                "step_index": idx,
                "risk": risk,
                "issues": issues,
                "has_condition_prediction": bool(condition),
                "temperature_c": temp,
                "condition_score": score,
                "solvent": condition.get("Solvent") or condition.get("solvent") or "",
                "reagent": condition.get("Reagent") or condition.get("reagent") or "",
                "catalyst": condition.get("Catalyst") or condition.get("catalyst") or "",
            }
        )
    temps = [row["temperature_c"] for row in step_rows if row.get("temperature_c") is not None]
    warn = [row for row in step_rows if row["risk"] == "warn"]
    return {
        "schema_version": "route_condition_audit.v1",
        "route_risk": "warn" if warn else "ok",
        "route_issues": ["warning_condition_steps"] if warn else [],
        "step_count": len(step_rows),
        "predicted_condition_count": sum(1 for row in step_rows if row["has_condition_prediction"]),
        "high_risk_step_count": 0,
        "warning_step_count": len(warn),
        "temperature_min_c": min(temps) if temps else None,
        "temperature_max_c": max(temps) if temps else None,
        "temperature_span_c": (max(temps) - min(temps)) if temps else None,
        "steps": step_rows,
    }


def _condition_coverage(steps: list[dict[str, Any]]) -> float:
    return sum(1 for step in steps if step.get("condition_predictions")) / len(steps) if steps else 0.0


def _reaction_product(rxn_smiles: Any) -> str:
    rxn = str(rxn_smiles or "")
    if ">>" not in rxn:
        return ""
    _lhs, rhs = rxn.split(">>", 1)
    return rhs.split(".")[0] if rhs else ""


def _rxn_lhs_parts(rxn_smiles: Any) -> list[str]:
    rxn = str(rxn_smiles or "")
    if ">>" not in rxn:
        return []
    lhs, _rhs = rxn.split(">>", 1)
    return [part for part in lhs.split(".") if part]


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summary_md(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Web Payload Route Export",
            "",
            f"- payload: `{summary.get('payload')}`",
            f"- route_doc: `{summary.get('route_doc')}`",
            f"- source routes: `{summary.get('source_route_count')}`",
            f"- selected routes: `{summary.get('selected_routes')}`",
            f"- selected steps: `{summary.get('selected_steps')}`",
            "",
        ]
    )


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip().lower()).strip("_")
    return slug or "target"


if __name__ == "__main__":
    main()
