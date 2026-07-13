"""Enzyme-step discovery, repair, and ranking for ChemEnzy route audits."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cascade_planner.baselines.chem_enzy_step_quality import evaluate_enzyme_step_quality
from cascade_planner.baselines.route_contract import RouteStepCandidate
from cascade_planner.baselines.route_plausibility import audit_step_plausibility
from cascade_planner.cascadeboard.enzyme_precedent_retrieval import retrieve_enzyme_precedents
from cascade_planner.route_tree.schema import CandidateAction


SCHEMA_VERSION = "chem_enzy_enzyme_step_enhancement.v1"

ENZYMATIC_SOURCE_TOKENS = (
    "enzyme",
    "enzymatic",
    "bionav",
    "bkms",
    "biocatalysis",
    "ecreact",
    "ec_",
)


@dataclass(frozen=True)
class EnzymeStepEnhancementConfig:
    """Runtime knobs for evidence-backed enzyme-step improvement."""

    pack_dir: Path = Path("data/bridge_pack_v0")
    retrieve_top_k: int = 24
    output_top_k: int = 5
    max_ec_contexts: int = 3
    min_candidate_quality: float = 0.70
    min_efficiency_score: float = 0.68
    min_upgrade_delta: float = 0.12
    min_similarity: float | None = None
    require_sp_v1_acceptance: bool = True

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None = None) -> "EnzymeStepEnhancementConfig":
        raw = dict(raw or {})
        return cls(
            pack_dir=Path(str(raw.get("pack_dir") or cls.pack_dir)),
            retrieve_top_k=_as_int(raw.get("retrieve_top_k"), cls.retrieve_top_k, lo=1),
            output_top_k=_as_int(raw.get("output_top_k"), cls.output_top_k, lo=1),
            max_ec_contexts=_as_int(raw.get("max_ec_contexts"), cls.max_ec_contexts, lo=0, hi=7),
            min_candidate_quality=_as_float(raw.get("min_candidate_quality"), cls.min_candidate_quality),
            min_efficiency_score=_as_float(raw.get("min_efficiency_score"), cls.min_efficiency_score),
            min_upgrade_delta=_as_float(raw.get("min_upgrade_delta"), cls.min_upgrade_delta),
            min_similarity=_as_float_or_none(raw.get("min_similarity")),
            require_sp_v1_acceptance=bool(raw.get("require_sp_v1_acceptance", cls.require_sp_v1_acceptance)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["pack_dir"] = str(self.pack_dir)
        return data


def evaluate_step_enhancement(
    step: RouteStepCandidate,
    *,
    scorer: Any | None = None,
    config: EnzymeStepEnhancementConfig | None = None,
) -> dict[str, Any]:
    """Return missing/repair/efficiency opportunities for one selected step."""
    cfg = config or EnzymeStepEnhancementConfig()
    current = current_step_quality(step)
    candidate_rows = retrieve_candidate_rows(step, config=cfg)
    candidates = [
        scored_candidate_from_row(step, row, rank=rank, scorer=scorer, config=cfg)
        for rank, row in enumerate(candidate_rows, start=1)
    ]
    candidates = [row for row in candidates if row]
    candidates.sort(key=lambda row: (float(row.get("efficiency_score") or 0.0), float(row.get("quality_score") or 0.0)), reverse=True)
    viable = [row for row in candidates if is_viable_candidate(row, config=cfg)]
    best = viable[0] if viable else {}
    recommendation = classify_recommendation(current=current, best=best, config=cfg)
    return {
        "schema_version": SCHEMA_VERSION,
        "available": bool(best),
        "recommended_kind": recommendation["kind"],
        "recommended": recommendation["kind"] != "no_change",
        "reasons": recommendation["reasons"],
        "current": current,
        "best_candidate": best,
        "candidate_count": len(candidates),
        "viable_candidate_count": len(viable),
        "top_candidates": viable[: cfg.output_top_k],
        "config": cfg.to_dict(),
    }


def current_step_quality(step: RouteStepCandidate) -> dict[str, Any]:
    """Score the currently selected ChemEnzy step with the same visible contract."""
    template = _template(step)
    quality = template.get("autoplanner_enzyme_quality_v1") if isinstance(template.get("autoplanner_enzyme_quality_v1"), dict) else {}
    ec_numbers = _step_ec_numbers(step)
    if not quality:
        quality = evaluate_enzyme_step_quality(
            product_smiles=step.product_smiles,
            reactants=step.reactant_smiles,
            source_model=step.source_model,
            template={
                "model_full_name": step.source_model,
                "source": step.source_model,
                "evidence": template.get("evidence") if isinstance(template.get("evidence"), dict) else {},
                "enzyme_sp_verifier_v1": template.get("enzyme_sp_verifier_v1") if isinstance(template.get("enzyme_sp_verifier_v1"), dict) else {},
            },
            ec_numbers=ec_numbers,
        )
    material = audit_step_plausibility(step)
    has_search_time_enzyme = has_search_time_enzyme_source(step)
    has_posthoc_ec = bool(ec_numbers)
    domain = "enzymatic" if has_search_time_enzyme else "chemical_or_unknown"
    efficiency = current_efficiency_proxy(step=step, quality=quality, material=material)
    return {
        "source_model": step.source_model,
        "proposal_domain": domain,
        "is_enzyme_like": has_search_time_enzyme,
        "has_search_time_enzyme_source": has_search_time_enzyme,
        "has_posthoc_ec_annotation": has_posthoc_ec,
        "has_ec": has_posthoc_ec,
        "ec_numbers": ec_numbers,
        "quality_decision": quality.get("decision") or "",
        "quality_score": quality.get("quality_score"),
        "quality_flags": list(quality.get("flags") or []),
        "material_passed": bool(material.get("passed")),
        "material_reasons": list(material.get("reasons") or []),
        "efficiency_score": efficiency,
    }


def retrieve_candidate_rows(
    step: RouteStepCandidate,
    *,
    config: EnzymeStepEnhancementConfig,
) -> list[dict[str, Any]]:
    ecs = _step_ec_numbers(step)
    ec_contexts: list[str] = []
    for ec in ecs:
        head = str(ec or "").split(".", 1)[0]
        if head in {"1", "2", "3", "4", "5", "6", "7"} and head not in ec_contexts:
            ec_contexts.append(head)
    ec_contexts = ec_contexts[: config.max_ec_contexts]
    if not ec_contexts:
        ec_contexts = [""]

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ec1 in ec_contexts:
        retrieved = retrieve_enzyme_precedents(
            step.product_smiles,
            ec_class=ec1,
            top_k=config.retrieve_top_k,
            min_similarity=config.min_similarity,
            pool_path=config.pack_dir / "enzyme_reaction_pool.parquet",
        )
        for row in retrieved:
            key = str(row.get("rxn_smiles") or row.get("main_reactant") or "")
            if key and key not in seen:
                seen.add(key)
                rows.append(row)
    return rows


def scored_candidate_from_row(
    step: RouteStepCandidate,
    row: dict[str, Any],
    *,
    rank: int,
    scorer: Any | None,
    config: EnzymeStepEnhancementConfig,
) -> dict[str, Any]:
    action = CandidateAction.from_candidate(step.product_smiles, row, rank=rank, source="enzyme_precedent")
    sp_payload = {}
    if scorer is not None:
        try:
            sp_payload = scorer.score_action(product=step.product_smiles, action=action).to_dict()
        except Exception as exc:
            sp_payload = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    quality = evaluate_enzyme_step_quality(
        product_smiles=step.product_smiles,
        reactants=action.reactants,
        source_model="enzyme_precedent",
        template={
            "model_full_name": "enzyme_precedent",
            "source": action.source,
            "ec": action.ec,
            "evidence": dict(action.metadata.get("evidence") or {}),
            "enzyme_sp_verifier_v1": sp_payload,
        },
        sp_payload=sp_payload,
        ec_numbers=[action.ec] if action.ec else None,
    )
    efficiency = enzyme_efficiency_proxy(row=row, quality=quality, sp_payload=sp_payload)
    evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
    return {
        "rank": rank,
        "main_reactant": action.main_reactant,
        "aux_reactants": list(action.aux_reactants),
        "rxn_smiles": action.rxn_smiles,
        "ec": action.ec,
        "source": action.source,
        "quality_decision": quality.get("decision"),
        "quality_score": quality.get("quality_score"),
        "quality_flags": list(quality.get("flags") or []),
        "efficiency_score": efficiency,
        "sp_v1_score": sp_payload.get("score"),
        "sp_v1_threshold": sp_payload.get("threshold"),
        "sp_v1_accepted": sp_payload.get("accepted"),
        "product_similarity": evidence.get("product_similarity") or row.get("precedent_product_similarity"),
        "transition_quality_score": (evidence.get("transition_signature") or {}).get("transition_quality_score"),
        "transition_flags": list((evidence.get("transition_signature") or {}).get("transition_flags") or []),
        "occurrences": evidence.get("occurrences"),
        "precedent_reaction_id": row.get("precedent_reaction_id") or evidence.get("reaction_id"),
        "rhea_ids": list(row.get("rhea_ids") or []),
        "example_ids": list(evidence.get("example_ids") or [])[:5],
    }


def is_viable_candidate(row: dict[str, Any], *, config: EnzymeStepEnhancementConfig) -> bool:
    if row.get("quality_decision") != "pass":
        return False
    if float(row.get("quality_score") or 0.0) < float(config.min_candidate_quality):
        return False
    if float(row.get("efficiency_score") or 0.0) < float(config.min_efficiency_score):
        return False
    if config.require_sp_v1_acceptance and row.get("sp_v1_accepted") is False:
        return False
    return True


def classify_recommendation(
    *,
    current: dict[str, Any],
    best: dict[str, Any],
    config: EnzymeStepEnhancementConfig,
) -> dict[str, Any]:
    if not best:
        return {"kind": "no_change", "reasons": ["no_viable_enzyme_precedent_candidate"]}
    current_eff = float(current.get("efficiency_score") or 0.0)
    best_eff = float(best.get("efficiency_score") or 0.0)
    reasons = [
        f"best_efficiency={best_eff:.3f}",
        f"current_efficiency={current_eff:.3f}",
        f"best_quality={float(best.get('quality_score') or 0.0):.3f}",
    ]
    if not current.get("has_search_time_enzyme_source"):
        reasons.append("selected_step_has_no_enzyme_source")
        if current.get("has_posthoc_ec_annotation"):
            reasons.append("selected_step_only_has_posthoc_ec_annotation")
        return {"kind": "missing_enzyme_step", "reasons": reasons}
    current_bad = (
        current.get("quality_decision") in {"", "warn", "reject"}
        or not current.get("material_passed")
        or "missing_sp_v1" in set(current.get("quality_flags") or [])
        or "missing_bridge_or_precedent_evidence" in set(current.get("quality_flags") or [])
    )
    if current_bad:
        reasons.append("selected_enzyme_step_lacks_required_evidence")
        return {"kind": "wrong_enzyme_step_replacement", "reasons": reasons}
    if best_eff >= current_eff + float(config.min_upgrade_delta):
        reasons.append("candidate_has_higher_efficiency_proxy")
        return {"kind": "efficient_enzyme_step_upgrade", "reasons": reasons}
    return {"kind": "no_change", "reasons": ["current_step_already_competitive", *reasons]}


def enzyme_efficiency_proxy(
    *,
    row: dict[str, Any],
    quality: dict[str, Any],
    sp_payload: dict[str, Any],
) -> float:
    """Evidence-based proxy for efficient enzyme use, not kinetic kcat/KM."""
    evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
    transition = evidence.get("transition_signature") if isinstance(evidence.get("transition_signature"), dict) else {}
    similarity = _float(evidence.get("product_similarity") or row.get("precedent_product_similarity"), 0.0)
    transition_score = _float(transition.get("transition_quality_score"), 0.0)
    occurrences = _float(evidence.get("occurrences"), 0.0)
    sp_score = _float(sp_payload.get("score"), 0.0)
    sp_threshold = _float(sp_payload.get("threshold"), 0.0)
    sp_margin = max(0.0, sp_score - sp_threshold) if sp_payload else 0.0
    quality_score = _float(quality.get("quality_score"), 0.0)
    ec = str(row.get("ec") or "")
    ec_specific = 1.0 if ec and "-" not in ec and ec.count(".") >= 3 else 0.4 if ec else 0.0
    aux_count = len(row.get("aux_reactants") or [])
    transition_flags = set(str(flag) for flag in transition.get("transition_flags") or [])
    penalty = 0.0
    penalty += min(0.12, 0.025 * aux_count)
    if "main_transition_self_loop" in transition_flags:
        penalty += 0.15
    if "weak_main_transition_similarity" in transition_flags:
        penalty += 0.10
    if "large_main_transition_delta_review" in transition_flags:
        penalty += 0.08
    score = (
        0.40 * quality_score
        + 0.18 * max(0.0, min(1.0, sp_score))
        + 0.16 * max(0.0, min(1.0, similarity))
        + 0.10 * max(0.0, min(1.0, transition_score))
        + 0.06 * max(0.0, min(1.0, sp_margin / 0.4))
        + 0.05 * max(0.0, min(1.0, math.log10(occurrences + 1.0) / 3.0))
        + 0.05 * ec_specific
        - penalty
    )
    return round(max(0.0, min(1.0, score)), 6)


def current_efficiency_proxy(
    *,
    step: RouteStepCandidate,
    quality: dict[str, Any],
    material: dict[str, Any],
) -> float:
    score = 0.45 * _float(quality.get("quality_score"), 0.0)
    if step.enzyme_ec_annotations:
        top_conf = _float(step.enzyme_ec_annotations[0].get("confidence"), 0.0)
        score += 0.12 * max(0.0, min(1.0, top_conf))
    if material.get("passed"):
        score += 0.18
    if has_search_time_enzyme_source(step):
        score += 0.05
    flags = set(str(flag) for flag in quality.get("flags") or [])
    if "missing_sp_v1" in flags:
        score -= 0.08
    if "missing_bridge_or_precedent_evidence" in flags:
        score -= 0.08
    return round(max(0.0, min(1.0, score)), 6)


def is_enzyme_like_step(step: RouteStepCandidate) -> bool:
    return has_search_time_enzyme_source(step)


def has_search_time_enzyme_source(step: RouteStepCandidate) -> bool:
    """True only when the selected proposal itself came from an enzyme source.

    Post-hoc EC assignment is useful evidence, but it does not mean ChemEnzy
    actually selected an enzyme precedent/template during search.
    """
    text = str(step.source_model or "").lower()
    metadata = step.raw_backend_metadata if isinstance(step.raw_backend_metadata, dict) else {}
    template = metadata.get("template") if isinstance(metadata.get("template"), dict) else {}
    cascade_cost = metadata.get("cascade_cost") if isinstance(metadata.get("cascade_cost"), dict) else {}
    text = " ".join(
        [
            text,
            str(template.get("model_full_name") or ""),
            str(template.get("model_name") or ""),
            str(template.get("source") or ""),
            str(template.get("reaction_type") or ""),
            str(cascade_cost.get("source_model") or ""),
            str(cascade_cost.get("reaction_domain") or ""),
        ]
    ).lower()
    return any(token in text for token in ENZYMATIC_SOURCE_TOKENS)


def make_default_sp_v1_scorer() -> Any | None:
    try:
        from cascade_planner.cascade_search.enzyme_sp_verifier_v1 import EnzymeSPVerifierV1Scorer

        return EnzymeSPVerifierV1Scorer()
    except Exception:
        return None


def _template(step: RouteStepCandidate) -> dict[str, Any]:
    metadata = step.raw_backend_metadata if isinstance(step.raw_backend_metadata, dict) else {}
    template = metadata.get("template")
    return dict(template) if isinstance(template, dict) else {}


def _step_ec_numbers(step: RouteStepCandidate) -> list[str]:
    values: list[str] = []
    for item in step.enzyme_ec_annotations or []:
        if isinstance(item, dict):
            ec = str(item.get("ec_number") or item.get("EC Number") or "").strip()
            if ec:
                values.append(ec)
    template = _template(step)
    ec = str(template.get("ec") or "").strip()
    if ec:
        values.append(ec)
    return list(dict.fromkeys(values))


def _as_int(value: Any, default: int, *, lo: int | None = None, hi: int | None = None) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = int(default)
    if lo is not None:
        out = max(lo, out)
    if hi is not None:
        out = min(hi, out)
    return out


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
