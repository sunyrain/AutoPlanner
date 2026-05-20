#!/usr/bin/env python3
"""Build a verifier-first perturbation pack from route artifacts.

The pack is designed for rule-verifier smoke tests and later learned verifier
training. It creates hard negative variants by perturbing route material,
condition, enzyme, cofactor, and ordering fields while preserving the original
route payload for traceability.
"""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from cascade_planner.cascade_verifier import verify_cascade_route
from cascade_planner.cascade_verifier.schema import (
    CASCADE_PERTURBATION_SPECS,
    PERTURBATION_PACK_SCHEMA_VERSION,
)


def main() -> None:
    args = _parse_args()
    source_paths = [Path(path) for path in args.input]
    rows = []
    for path in source_paths:
        rows.extend(_route_rows(path))
    if args.skip_reject_artifacts:
        rows = [
            row
            for row in rows
            if str(((row["route"].get("product_audit") or {}).get("route_class") or "")) != "reject_artifact"
        ]
    rows = rows[: max(0, int(args.max_routes))]

    examples: list[dict[str, Any]] = []
    skipped_seed_verifier_fail = 0
    used_seed_routes = 0
    for route_idx, row in enumerate(rows):
        route = _with_default_stage_partition(row["route"], args.default_stage_mode)
        target = row["target_smiles"]
        seed_report = verify_cascade_route(route, target_smiles=target).to_dict()
        if args.require_seed_verifier_pass and not bool(seed_report.get("feasible")):
            skipped_seed_verifier_fail += 1
            continue
        used_seed_routes += 1
        if args.include_seeds:
            examples.append(
                {
                    "example_id": f"seed_{route_idx:04d}",
                    "label": 1,
                    "split_hint": "seed_positive",
                    "source_path": row["source_path"],
                    "source_target_index": row.get("target_index"),
                    "source_route_index": row["route_index"],
                    "target_smiles": target,
                    "perturbation_type": "seed",
                    "expected_failure_reasons": [],
                    "seed_verifier_feasible": bool(seed_report.get("feasible")),
                    "cascade": copy.deepcopy(route),
                }
            )

        made = 0
        for spec in CASCADE_PERTURBATION_SPECS:
            if made >= int(args.perturbations_per_route):
                break
            perturbed = _perturb_route(route, spec["type"])
            if perturbed is None:
                continue
            examples.append(
                {
                    "example_id": f"neg_{route_idx:04d}_{made:02d}_{spec['type']}",
                    "label": 0,
                    "split_hint": "rule_negative",
                    "source_path": row["source_path"],
                    "source_target_index": row.get("target_index"),
                    "source_route_index": row["route_index"],
                    "target_smiles": target,
                    "perturbation_type": spec["type"],
                    "expected_failure_reasons": list(spec["expected_failure_reasons"]),
                    "seed_verifier_feasible": bool(seed_report.get("feasible")),
                    "cascade": perturbed,
                }
            )
            made += 1

    pack = {
        "schema_version": PERTURBATION_PACK_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "metadata": {
            "source_paths": [str(path) for path in source_paths],
            "max_routes": int(args.max_routes),
            "routes_seen": len(rows),
            "routes_used": used_seed_routes,
            "skipped_seed_verifier_fail": skipped_seed_verifier_fail,
            "include_seeds": bool(args.include_seeds),
            "default_stage_mode": args.default_stage_mode,
            "perturbations_per_route": int(args.perturbations_per_route),
            "skip_reject_artifacts": bool(args.skip_reject_artifacts),
            "contract": (
                "Perturbation labels are rule-derived. They train or test a verifier, "
                "not an expert feasibility oracle."
            ),
        },
        "perturbation_specs": CASCADE_PERTURBATION_SPECS,
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pack, indent=2), encoding="utf-8")
    print(f"wrote {len(examples)} examples from {len(rows)} routes to {args.output}")


def _route_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    if isinstance(payload, list):
        for target_idx, target_row in enumerate(payload):
            if not isinstance(target_row, dict):
                continue
            route = _route_from_cascade_row(target_row)
            if route is None:
                continue
            rows.append(
                {
                    "source_path": str(path),
                    "target_index": target_idx,
                    "route_index": 0,
                    "target_smiles": str(target_row.get("target_smiles") or route.get("target") or ""),
                    "route": route,
                }
            )
        return rows
    if isinstance(payload.get("routes"), list):
        target = str(payload.get("target") or payload.get("target_smiles") or "")
        for idx, route in enumerate(payload.get("routes") or []):
            if isinstance(route, dict):
                rows.append(
                    {
                        "source_path": str(path),
                        "target_index": None,
                        "route_index": idx,
                        "target_smiles": target,
                        "route": route,
                    }
                )
    for target_idx, target_row in enumerate(payload.get("targets") or []):
        if not isinstance(target_row, dict):
            continue
        target = str(target_row.get("target_smiles") or "")
        route_list = (target_row.get("planner_output") or {}).get("routes") or target_row.get("routes") or []
        for route_idx, route in enumerate(route_list):
            if isinstance(route, dict):
                rows.append(
                    {
                        "source_path": str(path),
                        "target_index": target_idx,
                        "route_index": route_idx,
                        "target_smiles": target,
                        "route": route,
                    }
                )
    return rows


def _route_from_cascade_row(row: dict[str, Any]) -> dict[str, Any] | None:
    raw_steps = row.get("steps") or row.get("gt_route") or []
    if not isinstance(raw_steps, list) or not raw_steps:
        return None
    steps: list[dict[str, Any]] = []
    # v3/v4 cascade rows are stored in forward order; verifier route-order
    # checks use retrosynthetic expansion order, so reverse them here.
    for step in reversed([item for item in raw_steps if isinstance(item, dict)]):
        reactants = _step_reactants_from_structured_step(step)
        product = str(step.get("product_smiles") or _rhs_first_product(str(step.get("rxn_smiles") or "")) or "")
        condition = step.get("step_conditions") or step.get("condition") or {}
        step_row = {
            "product": product,
            "main_reactant": str(step.get("main_reactant") or (reactants[0] if reactants else "")),
            "aux_reactants": reactants[1:] if reactants else [],
            "reactants": reactants,
            "reactant_smiles": reactants,
            "reaction_smiles": str(step.get("rxn_smiles") or ""),
            "rxn_smiles": str(step.get("rxn_smiles") or ""),
            "source": "dataset_v4_release",
            "ec": str(step.get("ec_number") or ""),
            "reaction_type": str(step.get("transformation_superclass") or ""),
            "T": condition.get("temperature_c") if isinstance(condition, dict) else None,
            "pH": condition.get("ph") if isinstance(condition, dict) else None,
            "solvent": condition.get("solvent") if isinstance(condition, dict) else "",
            "raw_metadata": {
                "step_id": step.get("step_id"),
                "step_index": step.get("step_index"),
                "catalyst_classes": step.get("catalyst_classes") or [],
                "cofactors": step.get("cofactors") or [],
            },
        }
        if step_row["ec"]:
            step_row["enzyme_ec_annotations"] = [{"ec_number": step_row["ec"], "confidence": 1.0}]
        steps.append(step_row)
    return {
        "target": str(row.get("target_smiles") or ""),
        "steps": steps,
        "metrics": {"route_solved": True, "strict_stock_solve": False},
        "metadata": {
            "cascade_id": row.get("cascade_id"),
            "doi": row.get("doi"),
            "route_domain": row.get("route_domain"),
            "quality_tier": row.get("quality_tier"),
            "compatibility_label": row.get("compatibility_label"),
            "split": row.get("split"),
        },
    }


def _step_reactants_from_structured_step(step: dict[str, Any]) -> list[str]:
    values = step.get("reactants")
    if isinstance(values, list) and values:
        return [str(value) for value in values if value]
    rxn = str(step.get("rxn_smiles") or "")
    if ">>" not in rxn:
        return []
    return [part for part in rxn.split(">>", 1)[0].split(".") if part]


def _rhs_first_product(rxn_smiles: str) -> str:
    if ">>" not in rxn_smiles:
        return ""
    for part in rxn_smiles.split(">>", 1)[1].split("."):
        if part:
            return part
    return ""


def _perturb_route(route: dict[str, Any], perturbation_type: str) -> dict[str, Any] | None:
    out = copy.deepcopy(route)
    steps = [step for step in out.get("steps") or out.get("step_annotations") or [] if isinstance(step, dict)]
    if not steps:
        return None
    if perturbation_type in {"atom_balance_drop_reactant", "atom_balance_drop_reactant_main"}:
        product = _step_product(steps[0])
        if not product:
            return None
        steps[0]["main_reactant"] = ""
        steps[0]["aux_reactants"] = []
        steps[0]["reactants"] = []
        steps[0]["reactant_smiles"] = []
        steps[0]["reaction_smiles"] = f">>{product}"
        steps[0]["rxn_smiles"] = f">>{product}"
        return out
    if perturbation_type in {"atom_balance_drop_reactant_aux", "atom_balance_tiny_carbon_source"}:
        product = _step_product(steps[0])
        if not product:
            return None
        steps[0]["main_reactant"] = ""
        steps[0]["aux_reactants"] = []
        steps[0]["reactants"] = []
        steps[0]["reactant_smiles"] = []
        steps[0]["reaction_smiles"] = f">>{product}"
        steps[0]["rxn_smiles"] = f">>{product}"
        return out
    if perturbation_type in {"temperature_conflict", "temperature_conflict_wide"}:
        if len(steps) < 2:
            return None
        _set_condition(steps[0], temperature=25.0)
        _set_condition(steps[1], temperature=95.0)
        out["stage_partition"] = ["stage_1" for _ in steps]
        return out
    if perturbation_type == "temperature_conflict_mild":
        if len(steps) < 2:
            return None
        _set_condition(steps[0], temperature=10.0)
        _set_condition(steps[1], temperature=45.0)
        out["stage_partition"] = ["stage_1" for _ in steps]
        return out
    if perturbation_type == "temperature_conflict_freezing_heated":
        if len(steps) < 2:
            return None
        _set_condition(steps[0], temperature=-20.0)
        _set_condition(steps[1], temperature=60.0)
        out["stage_partition"] = ["stage_1" for _ in steps]
        return out
    if perturbation_type in {"ph_conflict", "ph_conflict_acidic_basic"}:
        if len(steps) < 2:
            return None
        _set_condition(steps[0], ph=2.0)
        _set_condition(steps[1], ph=10.5)
        out["stage_partition"] = ["stage_1" for _ in steps]
        return out
    if perturbation_type == "ph_conflict_neutral_to_basic":
        if len(steps) < 2:
            return None
        _set_condition(steps[0], ph=6.0)
        _set_condition(steps[1], ph=9.5)
        out["stage_partition"] = ["stage_1" for _ in steps]
        return out
    if perturbation_type == "ph_conflict_strong_acid_base":
        if len(steps) < 2:
            return None
        _set_condition(steps[0], ph=1.0)
        _set_condition(steps[1], ph=13.0)
        out["stage_partition"] = ["stage_1" for _ in steps]
        return out
    if perturbation_type == "solvent_conflict":
        if len(steps) < 2:
            return None
        _set_condition(steps[0], solvent="water")
        _set_condition(steps[1], solvent="dichloromethane")
        out["stage_partition"] = ["stage_1" for _ in steps]
        return out
    if perturbation_type == "solvent_conflict_reverse":
        if len(steps) < 2:
            return None
        _set_condition(steps[0], solvent="dichloromethane")
        _set_condition(steps[1], solvent="water")
        out["stage_partition"] = ["stage_1" for _ in steps]
        return out
    if perturbation_type in {"enzyme_toxicity", "enzyme_toxicity_solvent"}:
        step = _first_enzymatic_step(steps) or steps[0]
        step["ec"] = step.get("ec") or "1.1.1.1"
        step["source"] = step.get("source") or "CascadePlanner enzyme module"
        step["enzyme_ec_annotations"] = step.get("enzyme_ec_annotations") or [{"ec_number": "1.1.1.1", "confidence": 1.0}]
        _set_condition(step, solvent="dichloromethane", catalyst="LDA")
        return out
    if perturbation_type == "enzyme_toxicity_reagent":
        step = _first_enzymatic_step(steps) or steps[0]
        step["ec"] = step.get("ec") or "1.1.1.1"
        step["source"] = step.get("source") or "CascadePlanner enzyme module"
        step["enzyme_ec_annotations"] = step.get("enzyme_ec_annotations") or [{"ec_number": "1.1.1.1", "confidence": 1.0}]
        _set_condition(step, solvent="water", catalyst="DIBAL-H")
        return out
    if perturbation_type == "enzyme_toxicity_strong_base":
        step = _first_enzymatic_step(steps) or steps[0]
        step["ec"] = step.get("ec") or "1.1.1.1"
        step["source"] = step.get("source") or "CascadePlanner enzyme module"
        step["enzyme_ec_annotations"] = step.get("enzyme_ec_annotations") or [{"ec_number": "1.1.1.1", "confidence": 1.0}]
        _set_condition(step, solvent="water", catalyst="NaH")
        return out
    if perturbation_type in {"cofactor_ledger_gap", "cofactor_ledger_gap_single"}:
        req = dict(steps[0].get("cofactor_requirements") or {})
        req["NADPH"] = float(req.get("NADPH") or 0.0) + 1.0
        steps[0]["cofactor_requirements"] = req
        steps[0].pop("cofactor_regenerations", None)
        return out
    if perturbation_type == "cofactor_ledger_gap_multi":
        req = dict(steps[0].get("cofactor_requirements") or {})
        req["NADPH"] = float(req.get("NADPH") or 0.0) + 1.0
        req["ATP"] = float(req.get("ATP") or 0.0) + 1.0
        steps[0]["cofactor_requirements"] = req
        steps[0].pop("cofactor_regenerations", None)
        return out
    if perturbation_type == "cofactor_ledger_gap_atp":
        req = dict(steps[0].get("cofactor_requirements") or {})
        req["ATP"] = float(req.get("ATP") or 0.0) + 1.0
        steps[0]["cofactor_requirements"] = req
        steps[0].pop("cofactor_regenerations", None)
        return out
    if perturbation_type == "cofactor_ledger_gap_fad":
        req = dict(steps[0].get("cofactor_requirements") or {})
        req["FAD"] = float(req.get("FAD") or 0.0) + 1.0
        steps[0]["cofactor_requirements"] = req
        steps[0].pop("cofactor_regenerations", None)
        return out
    if perturbation_type in {"route_order_shuffle", "route_order_swap_pair"}:
        if len(steps) < 2:
            return None
        raw_steps = out.get("steps") if isinstance(out.get("steps"), list) else out.get("step_annotations")
        raw_steps[0], raw_steps[1] = raw_steps[1], raw_steps[0]
        return out
    if perturbation_type == "route_order_reverse":
        if len(steps) < 2:
            return None
        raw_steps = out.get("steps") if isinstance(out.get("steps"), list) else out.get("step_annotations")
        raw_steps.reverse()
        return out
    if perturbation_type == "route_order_rotate":
        if len(steps) < 3:
            return None
        raw_steps = out.get("steps") if isinstance(out.get("steps"), list) else out.get("step_annotations")
        raw_steps[:] = raw_steps[1:] + raw_steps[:1]
        return out
    return None


def _with_default_stage_partition(route: dict[str, Any], mode: str) -> dict[str, Any]:
    out = copy.deepcopy(route)
    steps = [step for step in out.get("steps") or out.get("step_annotations") or [] if isinstance(step, dict)]
    if not steps or isinstance(out.get("stage_partition"), list):
        return out
    if mode == "single":
        out["stage_partition"] = ["stage_1" for _ in steps]
    else:
        out["stage_partition"] = [f"stage_{idx + 1}" for idx in range(len(steps))]
    return out


def _set_condition(
    step: dict[str, Any],
    *,
    temperature: float | None = None,
    ph: float | None = None,
    solvent: str | None = None,
    catalyst: str | None = None,
) -> None:
    if temperature is not None:
        step["T"] = float(temperature)
    if ph is not None:
        step["pH"] = float(ph)
    if solvent is not None:
        step["solvent"] = solvent
    if catalyst is not None:
        step["catalyst"] = catalyst
    predictions = step.get("condition_predictions")
    if not isinstance(predictions, list) or not predictions:
        predictions = [{}]
        step["condition_predictions"] = predictions
    row = predictions[0]
    if isinstance(row, dict):
        if temperature is not None:
            row["Temperature"] = float(temperature)
        if ph is not None:
            row["pH"] = float(ph)
        if solvent is not None:
            row["Solvent"] = solvent
        if catalyst is not None:
            row["Reagent"] = catalyst


def _first_enzymatic_step(steps: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    for step in steps:
        text = " ".join(str(step.get(key) or "") for key in ("source", "source_model", "reaction_type", "model_name"))
        if step.get("ec") or step.get("enzyme_ec_annotations") or "enzyme" in text.lower():
            return step
    return None


def _step_product(step: dict[str, Any]) -> str:
    return str(step.get("product") or step.get("product_smiles") or "")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build verifier-first cascade perturbation pack")
    parser.add_argument("--input", nargs="+", required=True, help="Route artifact JSON path(s)")
    parser.add_argument("--output", type=Path, required=True, help="Output perturbation pack JSON")
    parser.add_argument("--max-routes", type=int, default=50)
    parser.add_argument("--perturbations-per-route", type=int, default=5)
    parser.add_argument("--include-seeds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-reject-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-seed-verifier-pass", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--default-stage-mode",
        choices=["stepwise", "single"],
        default="stepwise",
        help="Stage partition assigned to routes that do not already export one.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
