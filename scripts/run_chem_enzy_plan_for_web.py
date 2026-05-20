"""Run one ChemEnzyRetroPlanner native search and emit web-compatible JSON."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml
from rdkit import Chem, RDLogger

from cascade_planner.baselines.chem_enzy_adapter import (
    ChemEnzyBackendAdapter,
    DEFAULT_ONE_STEP_MODELS,
    DEFAULT_STOCKS,
)
from cascade_planner.baselines.route_contract import RouteCandidate, RouteSearchConfig, RouteStepCandidate


RDLogger.DisableLog("rdApp.*")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run ChemEnzy native core search for the AutoPlanner web UI")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--vendor-root", default="vendor/ChemEnzyRetroPlanner")
    ap.add_argument("--gpu", type=int, default=-1)
    args = ap.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    started = time.monotonic()
    config = _route_config_from_payload(payload, args.gpu)
    adapter = ChemEnzyBackendAdapter(
        vendor_root=Path(args.vendor_root),
        gpu=args.gpu,
        enable_condition_prediction=bool(payload.get("enable_condition_prediction", False)),
        enable_enzyme_assignment=bool(payload.get("enable_enzyme_assignment", False)),
        enable_easifa=bool(payload.get("enable_easifa", False)),
    )
    result = adapter.run_target(config)
    output = _web_payload_from_result(result, payload, config, time.monotonic() - started, vendor_root=Path(args.vendor_root))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


def _route_config_from_payload(payload: dict[str, Any], gpu: int) -> RouteSearchConfig:
    preset = str(payload.get("search_preset") or "quick").lower()
    max_depth = _as_int(payload.get("max_steps"), 6, lo=1, hi=20)
    if preset == "quick":
        iterations = _as_int(payload.get("chem_enzy_iterations"), 10, lo=1, hi=200)
        expansion_topk = _as_int(payload.get("chem_enzy_expansion_topk"), 50, lo=1, hi=200)
    elif preset == "thorough":
        iterations = _as_int(payload.get("chem_enzy_iterations"), 50, lo=1, hi=500)
        expansion_topk = _as_int(payload.get("chem_enzy_expansion_topk"), 100, lo=1, hi=500)
    else:
        iterations = _as_int(payload.get("chem_enzy_iterations"), 25, lo=1, hi=300)
        expansion_topk = _as_int(payload.get("chem_enzy_expansion_topk"), 75, lo=1, hi=300)
    search_flags = {
        "gpu": gpu,
        "condition_model": payload.get("condition_model", "rcr"),
        "keep_search": True,
        "use_filter": payload.get("use_filter", False),
        "use_depth_value_fn": payload.get("use_depth_value_fn", False),
        "include_cascade_expansion_trace": True,
        "cascade_search_context": {
            "enabled": True,
            "target_smiles": str(payload["target_smiles"]),
            "search_preset": preset,
            "domain": payload.get("domain", "chemoenzymatic"),
        },
        "use_cascade_cost_model": True,
        "cascade_cost_model": _default_cascade_cost_model(),
        "use_cascade_source_policy": True,
        "cascade_source_policy": _default_cascade_source_policy(),
    }
    return RouteSearchConfig(
        target_smiles=str(payload["target_smiles"]),
        stock_names=_stock_names_from_payload(payload),
        max_iterations=iterations,
        max_depth=max_depth,
        expansion_topk=expansion_topk,
        one_step_models=list(payload.get("one_step_models") or DEFAULT_ONE_STEP_MODELS),
        search_flags=search_flags,
    )


def _stock_names_from_payload(payload: dict[str, Any]) -> list[str]:
    explicit = payload.get("stock_names")
    if explicit:
        return list(explicit)
    mode = str(payload.get("stock_mode") or "commercial").strip().lower()
    if mode in {"commercial", "zinc", "zinc_fix", "zinc-fix"}:
        return ["Zinc_Fix-stock"]
    if mode in {"benchmark-n5", "paroutes-n5", "n5"}:
        return ["PaRotes_n5-stock"]
    if mode in {"building-block", "building_block", "strict", "paroutes-n1", "n1"}:
        return ["PaRotes_n1-stock"]
    return list(DEFAULT_STOCKS)


def _default_cascade_cost_model() -> dict[str, Any]:
    model_path = Path("results/shared/dataset_v4_release/v4_full_training_stage3_fineshard_20260511/models/cascade_state_action_value_e4.pt")
    config: dict[str, Any] = {
        "enabled": True,
        "weights": {
            "learned_action_value_score_reward": 0.35,
        },
    }
    if model_path.exists():
        config["action_value_model_path"] = str(model_path)
    return config


def _default_cascade_source_policy() -> dict[str, Any]:
    model_path = Path("results/shared/dataset_v4_release/v4_full_training_stage3_fineshard_20260511/models/cascade_source_value_baseline.pt")
    config: dict[str, Any] = {
        "enabled": True,
    }
    if model_path.exists():
        config["source_value_model_path"] = str(model_path)
    return config


def _web_payload_from_result(
    result: Any,
    request_payload: dict[str, Any],
    config: RouteSearchConfig,
    elapsed_s: float,
    vendor_root: Path | str | None = None,
) -> dict[str, Any]:
    routes = [_web_route(route, index) for index, route in enumerate(result.routes)]
    strict_solved = any(bool((route.get("metrics") or {}).get("route_solved")) for route in routes)
    status = "solved" if strict_solved else "partial" if routes else "failed"
    message = (
        "ChemEnzy native core search returned stock-closed routes"
        if strict_solved
        else "ChemEnzy native core search returned routes, but terminal reactants are not all in the selected stock"
        if routes
        else "ChemEnzy native core search returned no route"
    )
    output = {
        "ok": not bool(result.failures) and (bool(routes) or strict_solved),
        "target": result.target_smiles,
        "objective": "chem_enzy_native",
        "constraints": request_payload.get("constraints"),
        "n_results": len(routes),
        "time_s": round(elapsed_s, 3),
        "routes": routes,
        "route_set_metrics": {
            "diversity": {
                "n_routes": len(routes),
                "unique_full_signatures": len({_route_signature(route) for route in routes}),
            }
        },
        "ui_metadata": {
            "backend": "CascadePlanner",
            "engine": "ChemEnzyRetroPlanner",
            "planner_strategy": "CascadePlanner search with ChemEnzy RSPlanner core and AutoPlanner-Cascade hooks",
            "search_mode": "chem_enzy_native",
            "search_preset": request_payload.get("search_preset", "quick"),
            "stock_mode": request_payload.get("stock_mode", "commercial"),
            "max_depth": config.max_depth,
            "iterations": config.max_iterations,
            "expansion_topk": config.expansion_topk,
            "condition_prediction_enabled": bool(request_payload.get("enable_condition_prediction", False)),
            "enzyme_assignment_enabled": bool(request_payload.get("enable_enzyme_assignment", False)),
            "condition_model": request_payload.get("condition_model", "rcr"),
            "one_step_models": config.one_step_models,
            "stock_names": config.stock_names,
            "cascade_hooks": {
                "cost_model": bool(config.search_flags.get("use_cascade_cost_model")),
                "source_policy": bool(config.search_flags.get("use_cascade_source_policy")),
                "expansion_trace": bool(config.search_flags.get("include_cascade_expansion_trace")),
                "action_value_model_path": (config.search_flags.get("cascade_cost_model") or {}).get("action_value_model_path"),
                "source_value_model_path": (config.search_flags.get("cascade_source_policy") or {}).get("source_value_model_path"),
            },
            "saved_at": None,
        },
        "skeletons": [],
        "depth_attempts": [
            {
                "depth": config.max_depth,
                "elapsed_s": round(elapsed_s, 3),
                "n_skeletons": 0,
                "n_routes": len(routes),
                "planner": "CascadePlanner",
                "engine": "ChemEnzyRetroPlanner",
                "status": status,
                "best": _route_summary(routes[0]) if routes else None,
            }
        ],
        "search_status": {
            "status": status,
            "solved": strict_solved,
            "native_returned_routes": bool(routes),
            "best_depth": config.max_depth,
            "message": message,
        },
        "failure_diagnosis": [failure.category for failure in result.failures],
        "backend_failures": [failure.to_dict() for failure in result.failures],
        "raw_backend_metadata": result.raw_backend_metadata,
    }
    output["failure_analysis"] = _failure_analysis(result, request_payload, config, vendor_root=Path(vendor_root) if vendor_root else None)
    return output


def _web_route(route: RouteCandidate, index: int) -> dict[str, Any]:
    steps = [_web_step(step, idx) for idx, step in enumerate(route.steps)]
    metrics = _route_metrics(route, steps)
    return {
        "score": route.score,
        "confidence": 1.0 if route.solved else 0.0,
        "n_steps": len(steps),
        "quality_vector": {},
        "risk_vector": {},
        "constraint_report": {"search_mode": "CascadePlanner", "backend": "CascadePlanner", "engine": route.backend},
        "bottleneck_slot": None,
        "bottleneck_reason": "",
        "global_constraints": {},
        "steps": steps,
        "metrics": metrics,
        "explanation": {
            "why_selected": "Returned by CascadePlanner using ChemEnzyRetroPlanner as the multi-step search engine.",
            "uncertainty_table": {
                "expansions": None,
                "generated_reactions": None,
            },
        },
        "route_rank": index,
        "raw_backend_metadata": route.raw_backend_metadata,
    }


def _web_step(step: RouteStepCandidate, index: int) -> dict[str, Any]:
    reactants = list(step.reactant_smiles or [])
    main = reactants[0] if reactants else ""
    aux = reactants[1:]
    condition = _top_condition_prediction(step)
    enzyme = _top_enzyme_annotation(step)
    reaction_type = _reaction_type(step)
    ec = str(enzyme.get("ec_number") or "") if enzyme else ""
    catalyst = _condition_value(condition, "Catalyst", "catalyst") or _condition_value(condition, "Reagent", "reagent")
    if ec and not catalyst:
        catalyst = f"EC {ec}"
    temperature = _condition_value(condition, "Temperature", "temperature", "temperature_c")
    ph = _condition_value(condition, "pH", "ph")
    solvent = _condition_value(condition, "Solvent", "solvent")
    condition_score = _safe_float(_condition_value(condition, "Score", "score", "confidence"))
    enzyme_score = _safe_float(enzyme.get("confidence")) if enzyme else None
    condition_notes = _condition_notes(condition, enzyme)
    return {
        "index": index,
        "product": step.product_smiles,
        "main_reactant": main,
        "aux_reactants": aux,
        "reaction_smiles": step.rxn_smiles,
        "reaction_type": reaction_type,
        "ec": ec,
        "enzyme_uid": enzyme.get("uniprot_id") if enzyme else None,
        "catalyst": catalyst or "",
        "T": _safe_float(temperature),
        "pH": _safe_float(ph),
        "solvent": str(solvent or ""),
        "condition_predictions": list(step.condition_predictions or []),
        "enzyme_ec_annotations": list(step.enzyme_ec_annotations or []),
        "evidence": {
            "backend": "CascadePlanner",
            "engine": "ChemEnzyRetroPlanner",
            "condition_prediction_available": bool(step.condition_predictions),
            "enzyme_annotation_available": bool(step.enzyme_ec_annotations),
        },
        "source": _display_source(step),
        "scores": {
            "retro": step.score,
            "enzyme": enzyme_score,
            "condition": condition_score,
            "confidence": step.score,
        },
        "fixed_fields": [],
        "is_filled": True,
        "is_enzymatic": bool(step.enzyme_ec_annotations),
        "stock_status": dict(step.stock_status or {}),
        "reaction_interpretation": {
            "reaction_class": reaction_type,
            "forward_summary": _reaction_summary(reaction_type, step),
            "reaction_principle": _reaction_principle(reaction_type),
            "likely_added_or_removed": _reactant_change_notes(reactants),
            "catalysis_and_conditions": condition_notes,
            "atom_change": _atom_change_notes(step.rxn_smiles),
        },
        "candidate_pool": {"n_candidates": 0, "top_candidates": []},
    }


def _route_metrics(route: RouteCandidate, steps: list[dict[str, Any]]) -> dict[str, Any]:
    terminal_stock_status = _terminal_stock_status(steps)
    strict_stock = (
        all(bool(value) for value in terminal_stock_status.values())
        if terminal_stock_status
        else bool(route.solved)
    )
    native_returned_route = bool(route.solved)
    stock_closed = bool(native_returned_route and strict_stock)
    return {
        "professional_solved": stock_closed,
        "diagnostic_solved": bool(native_returned_route and not stock_closed),
        "route_solved": stock_closed,
        "strict_stock_solve": strict_stock,
        "native_returned_route": native_returned_route,
        "terminal_reactants": list(terminal_stock_status),
        "terminal_stock_status": terminal_stock_status,
        "progressive_route": native_returned_route,
        "filled_route": native_returned_route,
        "n_steps": len(steps),
        "retrosynthesis_progress": {
            "main_chain_reduction": 1.0 if native_returned_route else 0.0,
            "largest_leaf_reduction": 1.0 if native_returned_route else 0.0,
            "progressive_steps": len(steps),
            "progressive_step_fraction": 1.0 if native_returned_route else 0.0,
        },
        "cascade_compatibility": {
            "cascade_compatibility_success": None,
            "issues": [],
        },
        "condition": {"condition_window_success": None},
        "enzyme_evidence": {"enzyme_evidence_coverage": None},
        "operation_transitions": {"operation_score": None, "issues": []},
        "candidate_pool": {
            "steps_with_candidates": 0,
            "total_candidates": 0,
            "candidate_pool_coverage": 0.0,
        },
    }


def _terminal_stock_status(steps: list[dict[str, Any]]) -> dict[str, bool | None]:
    products = {str(step.get("product") or "") for step in steps if step.get("product")}
    terminal: dict[str, bool | None] = {}
    fallback: dict[str, bool | None] = {}
    for step in steps:
        for smi, ok in (step.get("stock_status") or {}).items():
            text = str(smi or "")
            if not text:
                continue
            fallback.setdefault(text, ok)
            if text not in products:
                terminal[text] = ok
    return terminal or fallback


def _route_summary(route: dict[str, Any]) -> dict[str, Any]:
    metrics = route.get("metrics") or {}
    progress = metrics.get("retrosynthesis_progress") or {}
    return {
        "n_steps": route.get("n_steps"),
        "score": route.get("score"),
        "route_solved": metrics.get("route_solved"),
        "professional_solved": metrics.get("professional_solved"),
        "strict_stock_solve": metrics.get("strict_stock_solve"),
        "main_chain_reduction": progress.get("main_chain_reduction"),
        "largest_leaf_reduction": progress.get("largest_leaf_reduction"),
    }


def _route_signature(route: dict[str, Any]) -> str:
    return "|".join(step.get("reaction_smiles") or "" for step in route.get("steps") or [])


def _reaction_type(step: RouteStepCandidate) -> str:
    if step.enzyme_ec_annotations:
        return "enzymatic"
    source = str(step.source_model or "")
    if source.startswith("[") or ">>" in source:
        return "template"
    return source or "reaction"


def _display_source(step: RouteStepCandidate) -> str:
    if step.enzyme_ec_annotations:
        return "CascadePlanner enzyme module"
    source = str(step.source_model or "")
    if source.startswith("[") or ">>" in source:
        return "Template proposal"
    if source in {"", "ChemEnzyRetroPlanner"}:
        return "CascadePlanner"
    return source


def _top_condition_prediction(step: RouteStepCandidate) -> dict[str, Any]:
    for row in step.condition_predictions or []:
        if isinstance(row, dict):
            return row
    return {}


def _top_enzyme_annotation(step: RouteStepCandidate) -> dict[str, Any]:
    for row in step.enzyme_ec_annotations or []:
        if isinstance(row, dict):
            return row
    return {}


def _condition_value(row: dict[str, Any], *keys: str) -> Any:
    if not isinstance(row, dict):
        return None
    lower = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
        low = key.lower()
        if low in lower and lower[low] not in (None, ""):
            return lower[low]
    return None


def _condition_notes(condition: dict[str, Any], enzyme: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    temp = _condition_value(condition, "Temperature", "temperature", "temperature_c")
    ph = _condition_value(condition, "pH", "ph")
    solvent = _condition_value(condition, "Solvent", "solvent")
    reagent = _condition_value(condition, "Reagent", "reagent")
    catalyst = _condition_value(condition, "Catalyst", "catalyst")
    score = _condition_value(condition, "Score", "score", "confidence")
    if temp not in (None, ""):
        notes.append(f"T={temp} C")
    if ph not in (None, ""):
        notes.append(f"pH={ph}")
    if solvent:
        notes.append(f"solvent={solvent}")
    if reagent:
        notes.append(f"reagent={reagent}")
    if catalyst:
        notes.append(f"catalyst={catalyst}")
    if score not in (None, ""):
        notes.append(f"condition_score={score}")
    if enzyme.get("ec_number"):
        notes.append(f"enzyme_ec={enzyme.get('ec_number')}")
    if enzyme.get("confidence") not in (None, ""):
        notes.append(f"enzyme_confidence={enzyme.get('confidence')}")
    return notes


def _reaction_summary(reaction_type: str, step: RouteStepCandidate) -> str:
    cls = str(reaction_type or "unknown reaction")
    reactants = " + ".join(step.reactant_smiles or [])
    if reactants:
        return f"{cls} proposal connects precursor(s) {reactants} to the displayed product."
    return f"{cls} proposal from ChemEnzyRetroPlanner; inspect reaction SMILES before mechanism assignment."


def _reaction_principle(reaction_type: str) -> str:
    key = str(reaction_type or "").lower().replace("_", " ")
    rules = [
        ("enzym", "Predicted enzymatic transformation; EC assignment, if present, is a catalyst-family hypothesis."),
        ("hydrolysis", "Hydrolysis cleaves a labile bond with water or aqueous conditions."),
        ("ester", "Esterification or transesterification forms or exchanges an ester linkage."),
        ("acyl", "Acyl transfer installs or exchanges an acyl group on a nucleophile."),
        ("coupling", "Coupling joins molecular fragments through a newly formed bond."),
        ("c-c", "C-C bond formation links two carbon fragments."),
        ("oxid", "Oxidation raises oxidation state or introduces oxygen-containing functionality."),
        ("reduct", "Reduction lowers oxidation state or adds hydride/hydrogen equivalents."),
        ("amin", "Amination forms or exchanges a C-N bond."),
        ("alkyl", "Alkylation installs an alkyl substituent through substitution or transfer chemistry."),
        ("deprotect", "Deprotection removes a protecting group to reveal a functional handle."),
        ("isomer", "Isomerization rearranges stereochemistry or connectivity without major atom-count change."),
        ("template", "Template-matched retrosynthetic disconnection; mechanism depends on the underlying reaction template."),
    ]
    for token, text in rules:
        if token in key:
            return text
    return "Mechanistic hypothesis only: use reaction SMILES, template/source, and predicted conditions for chemist review."


def _reactant_change_notes(reactants: list[str]) -> list[str]:
    if len(reactants) <= 1:
        return []
    return [f"multiple precursor/coupling partners: {' . '.join(reactants)}"]


def _atom_change_notes(rxn_smiles: str) -> dict[str, Any]:
    if ">>" not in str(rxn_smiles or ""):
        return {"notes": []}
    lhs, rhs = str(rxn_smiles).split(">>", 1)
    reactants = [part for part in lhs.split(".") if part]
    products = [part for part in rhs.split(".") if part]
    reactant_heavy = sum(_heavy_atoms(smi) for smi in reactants)
    product_heavy = sum(_heavy_atoms(smi) for smi in products)
    delta = product_heavy - reactant_heavy
    notes = []
    if delta > 0:
        notes.append(f"product has {delta} more heavy atom(s) than listed reactants")
    elif delta < 0:
        notes.append(f"product has {-delta} fewer heavy atom(s) than listed reactants")
    return {
        "reactant_heavy_atoms": reactant_heavy,
        "product_heavy_atoms": product_heavy,
        "heavy_atom_delta": delta,
        "notes": notes,
    }


def _heavy_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    return int(mol.GetNumHeavyAtoms()) if mol is not None else 0


def _failure_analysis(
    result: Any,
    request_payload: dict[str, Any],
    config: RouteSearchConfig,
    *,
    vendor_root: Path | None = None,
) -> dict[str, Any]:
    failures = [failure.to_dict() for failure in result.failures]
    if not failures:
        return {"available": False, "diagnosis": [], "retry_suggestions": []}
    target = str(result.target_smiles or request_payload.get("target_smiles") or "")
    target_heavy = _heavy_atoms(target)
    stock_membership = _target_stock_membership(target, config.stock_names, vendor_root=vendor_root)
    categories = [str(row.get("category") or "") for row in failures]
    diagnosis: list[str] = []
    suggestions: list[str] = []
    if "no_route_found" in categories:
        diagnosis.extend(
            [
                "ChemEnzy returned no stock-closed successful route; product-audit post-filter did not run.",
                "The backend does not expose the failed partial search tree in this Web path, so this is a search-level diagnosis rather than a step-level proof.",
            ]
        )
        if stock_membership.get("target_in_selected_stock"):
            diagnosis.append(
                "Target itself is present in the selected stock, but ChemEnzy excludes the target from stock to avoid a trivial zero-step solution."
            )
            suggestions.append("report this as target_in_stock_but_excluded; it is a purchasability signal, not a synthetic route")
            suggestions.append("use a route-review mode that shows target-in-stock separately from retrosynthesis success")
        if target_heavy >= 38:
            diagnosis.append(f"Target is large for stock-closed search ({target_heavy} heavy atoms); more iterations/depth or broader stock may be needed.")
        if config.max_iterations < 100:
            suggestions.append("increase chem_enzy_iterations to 100-200")
        if config.expansion_topk < 150:
            suggestions.append("increase chem_enzy_expansion_topk to 150-200")
        if config.max_depth < 16:
            suggestions.append("increase max_steps to 16-20")
        suggestions.append("try risk_guarded post-filter only after routes exist; filtering cannot recover no-route cases")
    if any(cat == "backend_initialization_failed" for cat in categories):
        suggestions.append("disable optional condition/enzyme annotation and retry")
    if any(cat == "backend_annotation_failed" for cat in categories):
        diagnosis.append("Routes may be valid, but optional condition/enzyme annotation failed.")
        suggestions.append("rerun without condition/enzyme annotation if route search is the priority")
    return {
        "available": True,
        "target_heavy_atoms": target_heavy or None,
        "failure_categories": categories,
        "diagnosis": diagnosis,
        "retry_suggestions": _dedupe(suggestions),
        "search_config": {
            "preset": request_payload.get("search_preset", "quick"),
            "max_depth": config.max_depth,
            "iterations": config.max_iterations,
            "expansion_topk": config.expansion_topk,
            "condition_prediction_enabled": bool(request_payload.get("enable_condition_prediction", False)),
            "enzyme_assignment_enabled": bool(request_payload.get("enable_enzyme_assignment", False)),
            "one_step_models": config.one_step_models,
            "stock_names": config.stock_names,
            "target_stock_membership": stock_membership,
        },
    }


def _target_stock_membership(target_smiles: str, stock_names: list[str], *, vendor_root: Path | None) -> dict[str, Any]:
    target_mol = Chem.MolFromSmiles(str(target_smiles or ""))
    if target_mol is None or vendor_root is None:
        return {"available": False, "target_in_selected_stock": False}
    canonical = Chem.MolToSmiles(target_mol, isomericSmiles=True)
    stock_paths = _stock_paths(vendor_root, stock_names)
    hits: list[str] = []
    checked: list[str] = []
    for stock_name, path in stock_paths.items():
        if not path.exists() or not path.is_file():
            continue
        checked.append(stock_name)
        try:
            if _smiles_in_stock_file(canonical, path):
                hits.append(stock_name)
        except OSError:
            continue
    return {
        "available": bool(checked),
        "canonical_target_smiles": canonical,
        "checked_stocks": checked,
        "hit_stocks": hits,
        "target_in_selected_stock": bool(hits),
        "note": "ChemEnzy uses exclude_target=true, so an exact stock hit is not returned as a zero-step route.",
    }


def _stock_paths(vendor_root: Path, stock_names: list[str]) -> dict[str, Path]:
    config_path = vendor_root / "retro_planner" / "config" / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    stock_cfg = cfg.get("stocks") or {}
    base = vendor_root / "retro_planner"
    selected = set(stock_names or [])
    out: dict[str, Path] = {}
    for name, rel in stock_cfg.items():
        if selected and name not in selected:
            continue
        path = Path(str(rel))
        out[str(name)] = path if path.is_absolute() else base / path
    return out


def _smiles_in_stock_file(canonical: str, path: Path) -> bool:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            first = line.strip().split(",", 1)[0].strip()
            if first == canonical:
                return True
    return False


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int, *, lo: int, hi: int) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return default
    return min(max(out, lo), hi)


if __name__ == "__main__":
    main()
