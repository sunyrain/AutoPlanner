"""Feature extraction shared by learned cascade verifier training and runtime."""
from __future__ import annotations

from collections import Counter
from typing import Any

from rdkit import Chem, RDLogger

from cascade_planner.baselines.route_contract import RouteStepCandidate
from cascade_planner.baselines.route_plausibility import audit_step_plausibility
from cascade_planner.cascade_verifier.condition_extraction import condition_predictions, condition_value


RDLogger.DisableLog("rdApp.*")


def cascade_verifier_features(example: dict[str, Any]) -> dict[str, float]:
    route = example.get("cascade") or {}
    steps = [step for step in route.get("steps") or [] if isinstance(step, dict)]
    meta = route.get("metadata") or {}
    feats: dict[str, float] = {
        "bias": 1.0,
        "n_steps": float(len(steps)),
        "target_heavy": float(_heavy_atoms(example.get("target_smiles"))),
        f"route_domain={meta.get('route_domain') or 'unknown'}": 1.0,
        f"quality={meta.get('quality_tier') or 'unknown'}": 1.0,
        f"compatibility={meta.get('compatibility_label') or 'unknown'}": 1.0,
    }
    condition_spans = _condition_span_features(steps, route.get("stage_partition") or [])
    feats.update(condition_spans)
    feats["enzyme_steps"] = float(sum(1 for step in steps if _is_enzymatic(step)))
    feats["toxic_enzyme_condition_hits"] = float(sum(1 for step in steps if _is_enzymatic(step) and _enzyme_toxic(step)))
    feats["cofactor_gap_count"] = float(len(_cofactor_gaps(steps)))
    feats["route_order_mismatch_count"] = float(_route_order_mismatch_count(steps, example.get("target_smiles") or ""))
    feats["solvent_conflict_pairs"] = float(_solvent_conflict_pairs(steps, route.get("stage_partition") or []))

    material_counts = []
    for step in steps:
        audit = audit_step_plausibility(
            RouteStepCandidate(
                product_smiles=str(step.get("product") or ""),
                reactant_smiles=_step_reactants(step),
                rxn_smiles=str(step.get("reaction_smiles") or ""),
                condition_predictions=condition_predictions(step),
            )
        )
        material_counts.append(float(len(audit.get("reasons") or [])))
        for key in ("heavy_atom_gain", "carbon_gain", "hetero_atom_gain"):
            feats[f"max_{key}"] = max(float(feats.get(f"max_{key}", 0.0)), float(audit.get(key) or 0.0))
    feats["material_issue_steps"] = float(sum(1 for value in material_counts if value > 0))

    for step in steps:
        reaction_type = str(step.get("reaction_type") or "unknown").lower()
        feats[f"rxn_type={reaction_type}"] = feats.get(f"rxn_type={reaction_type}", 0.0) + 1.0
        ec = str(step.get("ec") or "")
        if ec:
            feats[f"ec1={ec.split('.', 1)[0]}"] = feats.get(f"ec1={ec.split('.', 1)[0]}", 0.0) + 1.0
        solvent_class = _solvent_class(str(condition_value(step, "solvent") or ""))
        if solvent_class:
            feats[f"solvent_class={solvent_class}"] = feats.get(f"solvent_class={solvent_class}", 0.0) + 1.0
    return feats


def _step_reactants(step: dict[str, Any]) -> list[str]:
    out = []
    if step.get("main_reactant"):
        out.append(str(step.get("main_reactant")))
    out.extend(str(smi) for smi in step.get("aux_reactants") or [] if smi)
    if not out and isinstance(step.get("reactants"), list):
        out.extend(str(smi) for smi in step.get("reactants") if smi)
    return out


def _is_enzymatic(step: dict[str, Any]) -> bool:
    text = " ".join(str(step.get(key) or "") for key in ("source", "reaction_type", "model_name")).lower()
    return bool(step.get("ec") or step.get("enzyme_ec_annotations") or "enzyme" in text)


def _enzyme_toxic(step: dict[str, Any]) -> bool:
    solvent = str(condition_value(step, "solvent") or "").lower()
    catalyst = str(condition_value(step, "catalyst") or "").lower()
    return any(token in solvent for token in ("dichloromethane", "dcm", "chloroform", "dmf", "pyridine", "acetonitrile")) or any(
        token in catalyst for token in ("lda", "dibal", "pocl3", "socl2", "nah")
    )


def _cofactor_gaps(steps: list[dict[str, Any]]) -> dict[str, float]:
    required: Counter[str] = Counter()
    regenerated: Counter[str] = Counter()
    for step in steps:
        for key, amount in (step.get("cofactor_requirements") or {}).items():
            required[str(key)] += float(amount or 0.0)
        for key, amount in (step.get("cofactor_regenerations") or {}).items():
            regenerated[str(key)] += float(amount or 0.0)
    return {key: value - regenerated.get(key, 0.0) for key, value in required.items() if value > regenerated.get(key, 0.0)}


def _route_order_mismatch_count(steps: list[dict[str, Any]], target: str) -> int:
    needed = {_canonical(target) or _canonical(str(steps[0].get("product") or ""))} if steps else set()
    needed.discard("")
    count = 0
    for step in steps:
        product = _canonical(str(step.get("product") or ""))
        if product and needed and product not in needed:
            count += 1
        needed.discard(product)
        for reactant in _step_reactants(step):
            can = _canonical(reactant)
            if can:
                needed.add(can)
    return count


def _condition_span_features(steps: list[dict[str, Any]], partition: list[str]) -> dict[str, float]:
    if not partition or len(partition) != len(steps):
        partition = [f"stage_{idx + 1}" for idx in range(len(steps))]
    route_temps = [_safe_float(condition_value(step, "temperature")) for step in steps]
    route_phs = [_safe_float(condition_value(step, "ph")) for step in steps]
    route_temps = [value for value in route_temps if value is not None]
    route_phs = [value for value in route_phs if value is not None]
    stage_temp_spans: list[float] = []
    stage_ph_spans: list[float] = []
    for stage in dict.fromkeys(partition):
        stage_steps = [step for step, step_stage in zip(steps, partition) if step_stage == stage]
        temps = [_safe_float(condition_value(step, "temperature")) for step in stage_steps]
        temps = [value for value in temps if value is not None]
        phs = [_safe_float(condition_value(step, "ph")) for step in stage_steps]
        phs = [value for value in phs if value is not None]
        if len(temps) >= 2:
            stage_temp_spans.append(float(max(temps) - min(temps)))
        if len(phs) >= 2:
            stage_ph_spans.append(float(max(phs) - min(phs)))
    return {
        "route_temperature_span": float(max(route_temps) - min(route_temps)) if len(route_temps) >= 2 else 0.0,
        "route_ph_span": float(max(route_phs) - min(route_phs)) if len(route_phs) >= 2 else 0.0,
        "max_stage_temperature_span": max(stage_temp_spans) if stage_temp_spans else 0.0,
        "max_stage_ph_span": max(stage_ph_spans) if stage_ph_spans else 0.0,
        "temperature_conflict_stage_pairs": float(sum(1 for value in stage_temp_spans if value > 20.0)),
        "ph_conflict_stage_pairs": float(sum(1 for value in stage_ph_spans if value > 2.0)),
        # Compatibility aliases keep old checkpoints interpretable, but their
        # values now follow the verifier's same-stage conflict contract.
        "temperature_span": max(stage_temp_spans) if stage_temp_spans else 0.0,
        "ph_span": max(stage_ph_spans) if stage_ph_spans else 0.0,
    }


def _solvent_conflict_pairs(steps: list[dict[str, Any]], partition: list[str]) -> int:
    count = 0
    if not partition or len(partition) != len(steps):
        partition = [f"stage_{idx + 1}" for idx in range(len(steps))]
    for idx, (left, right) in enumerate(zip(steps, steps[1:])):
        if partition[idx] != partition[idx + 1]:
            continue
        classes = {_solvent_class(str(condition_value(left, "solvent") or "")), _solvent_class(str(condition_value(right, "solvent") or ""))}
        if classes == {"aqueous", "hydrophobic"}:
            count += 1
    return count


def _solvent_class(solvent: str) -> str:
    text = solvent.lower()
    if any(token in text for token in ("water", "h2o", "buffer", "pbs", "phosphate", "tris", "hepes")):
        return "aqueous"
    if any(token in text for token in ("dichloromethane", "dcm", "chloroform", "toluene", "hexane", "heptane")):
        return "hydrophobic"
    return "organic" if text else ""


def _canonical(smiles: str | None) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def _heavy_atoms(smiles: str | None) -> int:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    return int(mol.GetNumHeavyAtoms()) if mol is not None else 0


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
