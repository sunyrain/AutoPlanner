"""Rule-first cascade verifier used before learned verifier training.

This module is a high-precision guard and perturbation-label engine. It is not
an expert feasibility oracle; it only validates failure modes that can be
checked from exported route/cascade records.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from rdkit import Chem, RDLogger

from cascade_planner.baselines.route_contract import RouteStepCandidate
from cascade_planner.baselines.route_plausibility import audit_step_plausibility
from cascade_planner.cascade_verifier.schema import (
    CascadeVerifierFinding,
    CascadeVerifierResult,
    VerifierFailureReason,
)

RDLogger.DisableLog("rdApp.*")

MATERIAL_REASONS = {
    "large_unexplained_heavy_atom_gain",
    "large_unexplained_carbon_gain",
    "large_unexplained_hetero_atom_gain",
    "unexplained_new_element_source",
    "invalid_product_smiles",
    "invalid_or_missing_reactants",
}

ENZYME_TOXIC_SOLVENT_TOKENS = {
    "dichloromethane",
    "dcm",
    "chloroform",
    "dmf",
    "pyridine",
    "acetonitrile",
    "meCN".lower(),
}
ENZYME_TOXIC_REAGENT_TOKENS = {
    "lda",
    "dibal",
    "dibal-h",
    "pocl3",
    "socl2",
    "oxalyl chloride",
    "organolithium",
    "nbuli",
    "n-buli",
    "nah",
}
AQUEOUS_SOLVENT_TOKENS = {"water", "h2o", "buffer", "pbs", "phosphate buffer", "tris", "hepes"}
HYDROPHOBIC_SOLVENT_TOKENS = {"dichloromethane", "dcm", "chloroform", "toluene", "hexane", "heptane"}


def verify_cascade_route(
    route: dict[str, Any],
    *,
    target_smiles: str | None = None,
    assume_single_stage: bool = True,
    temperature_tolerance_c: float = 10.0,
    ph_tolerance: float = 1.0,
) -> CascadeVerifierResult:
    """Verify one route/cascade dict and return typed failure findings."""
    steps = _extract_steps(route)
    target = str(target_smiles or route.get("target") or route.get("target_smiles") or "")
    if not target and steps:
        target = str(steps[0].get("product") or "")

    findings: list[CascadeVerifierFinding] = []
    findings.extend(_material_findings(steps))
    findings.extend(_product_mismatch_findings(steps))
    findings.extend(_route_order_findings(steps, target))
    partition = _stage_partition(route, len(steps), assume_single_stage=assume_single_stage)
    findings.extend(
        _condition_findings(
            steps,
            partition,
            temperature_tolerance_c=float(temperature_tolerance_c),
            ph_tolerance=float(ph_tolerance),
        )
    )
    findings.extend(_cofactor_findings(steps))

    severity = sum(max(0.0, float(finding.severity)) for finding in findings)
    score = max(0.0, min(1.0, 1.0 - severity / max(1.0, len(steps) + 2.0)))
    metrics = {
        "n_steps": len(steps),
        "n_stages": len(set(partition)) if partition else 0,
        "stage_partition": partition,
        "reason_counts": dict(Counter(_reason_text(finding.reason) for finding in findings)),
        "contract": (
            "rule-checkable cascade verifier; high precision for perturbation labels, "
            "not an expert feasibility oracle"
        ),
    }
    return CascadeVerifierResult(feasible=not findings, score=round(score, 4), findings=tuple(findings), metrics=metrics)


def _extract_steps(route: dict[str, Any]) -> list[dict[str, Any]]:
    raw_steps = route.get("steps") or route.get("step_annotations") or []
    return [step for step in raw_steps if isinstance(step, dict)]


def _material_findings(steps: list[dict[str, Any]]) -> list[CascadeVerifierFinding]:
    findings: list[CascadeVerifierFinding] = []
    for idx, step in enumerate(steps):
        product = _step_product(step)
        reactants = _step_reactants(step)
        audit = audit_step_plausibility(
            RouteStepCandidate(
                product_smiles=product,
                reactant_smiles=reactants,
                rxn_smiles=str(step.get("reaction_smiles") or step.get("rxn_smiles") or ""),
                source_model=str(step.get("source") or step.get("source_model") or ""),
                condition_predictions=_condition_predictions(step),
                enzyme_ec_annotations=list(step.get("enzyme_ec_annotations") or []),
            )
        )
        reasons = [str(reason) for reason in audit.get("reasons") or [] if str(reason) in MATERIAL_REASONS]
        if reasons:
            findings.append(
                CascadeVerifierFinding(
                    reason=VerifierFailureReason.ATOM_BALANCE,
                    severity=1.0,
                    step_index=idx,
                    message="Step has unexplained material/element gain relative to listed sources.",
                    evidence={"plausibility_reasons": reasons, "audit": audit},
                )
            )
    return findings


def _product_mismatch_findings(steps: list[dict[str, Any]]) -> list[CascadeVerifierFinding]:
    findings: list[CascadeVerifierFinding] = []
    for idx, step in enumerate(steps):
        rxn = str(step.get("reaction_smiles") or step.get("rxn_smiles") or "")
        if ">>" not in rxn:
            continue
        rhs = [part for part in rxn.split(">>", 1)[1].split(".") if part]
        if len(rhs) != 1:
            continue
        product = _canonical(_step_product(step))
        rhs_product = _canonical(rhs[0])
        if not product or not rhs_product:
            findings.append(
                CascadeVerifierFinding(
                    reason=VerifierFailureReason.INVALID_SMILES,
                    severity=1.0,
                    step_index=idx,
                    message="Step product or reaction product SMILES is invalid.",
                    evidence={"step_product": _step_product(step), "reaction_product": rhs[0]},
                )
            )
        elif product != rhs_product:
            findings.append(
                CascadeVerifierFinding(
                    reason=VerifierFailureReason.PRODUCT_MISMATCH,
                    severity=1.0,
                    step_index=idx,
                    message="Reaction SMILES product does not match exported step product.",
                    evidence={"step_product": _step_product(step), "reaction_product": rhs[0]},
                )
            )
    return findings


def _route_order_findings(steps: list[dict[str, Any]], target_smiles: str) -> list[CascadeVerifierFinding]:
    if not steps:
        return []
    needed = {_canonical(target_smiles) or _canonical(_step_product(steps[0]))}
    needed.discard("")
    findings: list[CascadeVerifierFinding] = []
    for idx, step in enumerate(steps):
        product = _canonical(_step_product(step))
        if product and needed and product not in needed:
            findings.append(
                CascadeVerifierFinding(
                    reason=VerifierFailureReason.ROUTE_ORDER_MISMATCH,
                    severity=0.8,
                    step_index=idx,
                    message="Retrosynthetic step expands a product that is not currently opened by earlier steps.",
                    evidence={"product": _step_product(step), "open_products": sorted(needed)},
                )
            )
        if product in needed:
            needed.remove(product)
        for reactant in _step_reactants(step):
            can = _canonical(reactant)
            if can:
                needed.add(can)
    return findings


def _condition_findings(
    steps: list[dict[str, Any]],
    partition: list[str],
    *,
    temperature_tolerance_c: float,
    ph_tolerance: float,
) -> list[CascadeVerifierFinding]:
    findings: list[CascadeVerifierFinding] = []
    by_stage: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for idx, step in enumerate(steps):
        by_stage.setdefault(partition[idx] if idx < len(partition) else "stage_1", []).append((idx, step))
        if _is_enzymatic_step(step) and _enzyme_toxic_condition(step):
            findings.append(
                CascadeVerifierFinding(
                    reason=VerifierFailureReason.ENZYME_TOXICITY,
                    severity=0.8,
                    step_index=idx,
                    stage_id=partition[idx] if idx < len(partition) else "stage_1",
                    message="Enzymatic step is paired with a strongly enzyme-incompatible solvent or reagent.",
                    evidence=_condition_summary(step),
                )
            )

    for stage_id, rows in by_stage.items():
        for left, right in zip(rows, rows[1:]):
            left_idx, left_step = left
            right_idx, right_step = right
            left_temp = _safe_float(_condition_value(left_step, "temperature"))
            right_temp = _safe_float(_condition_value(right_step, "temperature"))
            if left_temp is not None and right_temp is not None:
                if abs(left_temp - right_temp) > 2.0 * temperature_tolerance_c:
                    findings.append(
                        CascadeVerifierFinding(
                            reason=VerifierFailureReason.TEMPERATURE_CONFLICT,
                            severity=0.7,
                            step_index=right_idx,
                            stage_id=stage_id,
                            message="Adjacent same-stage steps have non-overlapping temperature envelopes.",
                            evidence={"left_step_index": left_idx, "left_T": left_temp, "right_T": right_temp},
                        )
                    )
            left_ph = _safe_float(_condition_value(left_step, "ph"))
            right_ph = _safe_float(_condition_value(right_step, "ph"))
            if left_ph is not None and right_ph is not None:
                if abs(left_ph - right_ph) > 2.0 * ph_tolerance:
                    findings.append(
                        CascadeVerifierFinding(
                            reason=VerifierFailureReason.PH_CONFLICT,
                            severity=0.7,
                            step_index=right_idx,
                            stage_id=stage_id,
                            message="Adjacent same-stage steps have non-overlapping pH envelopes.",
                            evidence={"left_step_index": left_idx, "left_pH": left_ph, "right_pH": right_ph},
                        )
                    )
            left_class = _solvent_class(str(_condition_value(left_step, "solvent") or ""))
            right_class = _solvent_class(str(_condition_value(right_step, "solvent") or ""))
            if left_class and right_class and {left_class, right_class} == {"aqueous", "hydrophobic"}:
                findings.append(
                    CascadeVerifierFinding(
                        reason=VerifierFailureReason.SOLVENT_CONFLICT,
                        severity=0.5,
                        step_index=right_idx,
                        stage_id=stage_id,
                        message="Adjacent same-stage steps use incompatible aqueous/hydrophobic solvent classes.",
                        evidence={
                            "left_step_index": left_idx,
                            "left_solvent": _condition_value(left_step, "solvent"),
                            "right_solvent": _condition_value(right_step, "solvent"),
                        },
                    )
                )
    return findings


def _cofactor_findings(steps: list[dict[str, Any]]) -> list[CascadeVerifierFinding]:
    required: Counter[str] = Counter()
    regenerated: Counter[str] = Counter()
    for step in steps:
        required.update(_cofactor_map(step, "cofactor_requirements"))
        required.update(_cofactor_map(step, "cofactors_required"))
        regenerated.update(_cofactor_map(step, "cofactor_regenerations"))
        regenerated.update(_cofactor_map(step, "cofactors_regenerated"))
    gaps = {
        name: float(amount) - float(regenerated.get(name) or 0.0)
        for name, amount in required.items()
        if float(amount) - float(regenerated.get(name) or 0.0) > 1e-9
    }
    if not gaps:
        return []
    return [
        CascadeVerifierFinding(
            reason=VerifierFailureReason.COFACTOR_LEDGER_GAP,
            severity=0.6,
            message="Cofactor requirements are not balanced by regeneration records.",
            evidence={"unclosed_requirements": gaps, "regenerated": dict(regenerated)},
        )
    ]


def _stage_partition(route: dict[str, Any], n_steps: int, *, assume_single_stage: bool) -> list[str]:
    raw = route.get("stage_partition")
    if isinstance(raw, list) and len(raw) == n_steps:
        return [str(value or "stage_1") for value in raw]
    if not assume_single_stage:
        return [f"stage_{idx + 1}" for idx in range(n_steps)]
    return ["stage_1" for _ in range(n_steps)]


def _step_product(step: dict[str, Any]) -> str:
    return str(step.get("product") or step.get("product_smiles") or "")


def _step_reactants(step: dict[str, Any]) -> list[str]:
    reactants: list[str] = []
    main = step.get("main_reactant")
    if main:
        reactants.append(str(main))
    for smi in step.get("aux_reactants") or []:
        if smi:
            reactants.append(str(smi))
    for key in ("reactants", "reactant_smiles"):
        values = step.get(key)
        if isinstance(values, list):
            reactants.extend(str(value) for value in values if value)
    if not reactants:
        rxn = str(step.get("reaction_smiles") or step.get("rxn_smiles") or "")
        if ">>" in rxn:
            reactants.extend(part for part in rxn.split(">>", 1)[0].split(".") if part)
    return _dedupe(reactants)


def _condition_predictions(step: dict[str, Any]) -> list[dict[str, Any]]:
    values = step.get("condition_predictions") or []
    return [row for row in values if isinstance(row, dict)]


def _condition_value(step: dict[str, Any], field: str) -> Any:
    aliases = {
        "temperature": ("T", "Temperature", "temperature", "temperature_c"),
        "ph": ("pH", "ph", "PH"),
        "solvent": ("solvent", "Solvent"),
        "catalyst": ("catalyst", "Catalyst", "reagent", "Reagent"),
    }
    for key in aliases[field]:
        if key in step and step[key] not in (None, ""):
            return step[key]
    for prediction in _condition_predictions(step):
        for key in aliases[field]:
            if key in prediction and prediction[key] not in (None, ""):
                return prediction[key]
    return None


def _condition_summary(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "temperature": _condition_value(step, "temperature"),
        "pH": _condition_value(step, "ph"),
        "solvent": _condition_value(step, "solvent"),
        "catalyst_or_reagent": _condition_value(step, "catalyst"),
    }


def _is_enzymatic_step(step: dict[str, Any]) -> bool:
    if step.get("ec") or step.get("ec_number") or step.get("enzyme_ec_annotations"):
        return True
    text = " ".join(str(step.get(key) or "") for key in ("source", "source_model", "reaction_type", "model_name"))
    return any(token in text.lower() for token in ("enzyme", "enzymatic", "ec "))


def _enzyme_toxic_condition(step: dict[str, Any]) -> bool:
    solvent = str(_condition_value(step, "solvent") or "").lower()
    catalyst = str(_condition_value(step, "catalyst") or "").lower()
    return any(token in solvent for token in ENZYME_TOXIC_SOLVENT_TOKENS) or any(
        token in catalyst for token in ENZYME_TOXIC_REAGENT_TOKENS
    )


def _solvent_class(solvent: str) -> str:
    text = solvent.strip().lower()
    if not text:
        return ""
    if any(token in text for token in AQUEOUS_SOLVENT_TOKENS):
        return "aqueous"
    if any(token in text for token in HYDROPHOBIC_SOLVENT_TOKENS):
        return "hydrophobic"
    return "organic"


def _cofactor_map(step: dict[str, Any], key: str) -> dict[str, float]:
    value = step.get(key)
    if isinstance(value, dict):
        return {str(k): float(v or 0.0) for k, v in value.items() if k}
    if isinstance(value, list):
        return {str(item): 1.0 for item in value if item}
    return {}


def _canonical(smiles: str | None) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return ""
    # Route continuity should not fail only because one data source omits
    # stereochemical marks on the same constitutional intermediate.
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _reason_text(reason: VerifierFailureReason | str) -> str:
    return reason.value if isinstance(reason, VerifierFailureReason) else str(reason)


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
