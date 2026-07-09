"""Search-time quality scoring for ChemEnzy enzyme-step candidates."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from cascade_planner.baselines.route_contract import RouteStepCandidate
from cascade_planner.baselines.route_plausibility import audit_step_plausibility


SCHEMA_VERSION = "chem_enzy_enzyme_step_quality.v1"

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
class EnzymeStepQualityConfig:
    """Thresholds for conservative enzyme-candidate acceptance."""

    max_heavy_gain: int = 3
    max_carbon_gain: int = 2
    max_hetero_gain: int = 3
    pass_score: float = 0.70
    reject_on_material_failure: bool = True
    reject_below_score: float | None = None

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None = None) -> "EnzymeStepQualityConfig":
        raw = dict(raw or {})
        reject_below = raw.get("reject_below_score")
        return cls(
            max_heavy_gain=_as_int(raw.get("max_heavy_gain"), cls.max_heavy_gain, lo=0),
            max_carbon_gain=_as_int(raw.get("max_carbon_gain"), cls.max_carbon_gain, lo=0),
            max_hetero_gain=_as_int(raw.get("max_hetero_gain"), cls.max_hetero_gain, lo=0),
            pass_score=_as_float(raw.get("pass_score"), cls.pass_score),
            reject_on_material_failure=bool(raw.get("reject_on_material_failure", cls.reject_on_material_failure)),
            reject_below_score=None if reject_below in (None, "") else _as_float(reject_below, 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_enzyme_step_quality(
    *,
    product_smiles: str,
    reactants: Iterable[str],
    source_model: str = "",
    template: dict[str, Any] | None = None,
    sp_payload: dict[str, Any] | None = None,
    ec_numbers: Iterable[str] | None = None,
    config: EnzymeStepQualityConfig | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe quality record for a proposed enzyme step.

    This is a conservative screen, not a chemistry proof.  It makes the
    search-visible distinction between "enzyme-labeled" and "enzyme-supported"
    candidates explicit.
    """
    cfg = config or EnzymeStepQualityConfig()
    reactant_list = [str(item) for item in reactants if str(item or "")]
    template = dict(template or {})
    sp = _mapping(sp_payload) or _mapping(template.get("enzyme_sp_verifier_v1"))
    evidence = _mapping(template.get("evidence"))
    ec_list = _ec_numbers(ec_numbers, template=template, sp_payload=sp, evidence=evidence)
    rxn_smiles = f"{'.'.join(reactant_list)}>>{product_smiles}"
    material = audit_step_plausibility(
        RouteStepCandidate(
            product_smiles=str(product_smiles or ""),
            reactant_smiles=reactant_list,
            rxn_smiles=rxn_smiles,
            source_model=str(source_model or ""),
        ),
        max_heavy_gain=cfg.max_heavy_gain,
        max_carbon_gain=cfg.max_carbon_gain,
        max_hetero_gain=cfg.max_hetero_gain,
    )

    sp_score = _as_float_or_none(sp.get("score"))
    sp_threshold = _as_float_or_none(sp.get("threshold"))
    sp_accepted = sp.get("accepted")
    sp_accepted_bool = bool(sp_accepted) if sp_accepted is not None else None
    bridge_or_precedent = _has_bridge_or_precedent_evidence(evidence)
    source_is_enzyme_like = _is_enzyme_like_source(source_model) or _is_enzyme_like_source(template.get("source"))

    components = {
        "material": 0.42 if material.get("passed") else 0.0,
        "sp_v1": _sp_component(sp_score=sp_score, sp_threshold=sp_threshold, accepted=sp_accepted_bool),
        "ec": 0.12 if ec_list else 0.0,
        "precedent": 0.12 if bridge_or_precedent else 0.0,
        "source": 0.06 if source_is_enzyme_like else 0.0,
    }
    score = max(0.0, min(1.0, sum(float(value) for value in components.values())))
    flags = _quality_flags(
        material=material,
        sp_payload=sp,
        sp_accepted=sp_accepted_bool,
        ec_numbers=ec_list,
        bridge_or_precedent=bridge_or_precedent,
    )
    if not material.get("passed"):
        score = min(score, 0.40)

    decision = "pass" if score >= float(cfg.pass_score) else "warn"
    if cfg.reject_on_material_failure and not material.get("passed"):
        decision = "reject"
    if cfg.reject_below_score is not None and score < float(cfg.reject_below_score):
        decision = "reject"

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "quality_score": round(float(score), 6),
        "pass_score": round(float(cfg.pass_score), 6),
        "components": {key: round(float(value), 6) for key, value in components.items()},
        "flags": flags,
        "ec_numbers": ec_list,
        "source_model": str(source_model or ""),
        "source_is_enzyme_like": source_is_enzyme_like,
        "bridge_or_precedent_evidence": bridge_or_precedent,
        "sp_v1": {
            "available": bool(sp),
            "accepted": sp_accepted_bool,
            "score": sp_score,
            "threshold": sp_threshold,
        },
        "material_sanity": {
            "passed": bool(material.get("passed")),
            "reasons": list(material.get("reasons") or []),
            "heavy_atom_gain": material.get("heavy_atom_gain"),
            "carbon_gain": material.get("carbon_gain"),
            "hetero_atom_gain": material.get("hetero_atom_gain"),
            "unexplained_element_gains": material.get("unexplained_element_gains") or {},
        },
        "config": cfg.to_dict(),
    }


def _quality_flags(
    *,
    material: dict[str, Any],
    sp_payload: dict[str, Any],
    sp_accepted: bool | None,
    ec_numbers: list[str],
    bridge_or_precedent: bool,
) -> list[str]:
    flags: list[str] = []
    if not material.get("passed"):
        flags.append("material_sanity_failed")
        flags.extend(f"material_{reason}" for reason in material.get("reasons") or [])
    if not sp_payload:
        flags.append("missing_sp_v1")
    elif sp_accepted is False:
        flags.append("sp_v1_rejected")
    if not ec_numbers:
        flags.append("missing_ec_evidence")
    if not bridge_or_precedent:
        flags.append("missing_bridge_or_precedent_evidence")
    return flags


def _sp_component(*, sp_score: float | None, sp_threshold: float | None, accepted: bool | None) -> float:
    if accepted is True:
        if sp_score is None:
            return 0.22
        return 0.20 + 0.08 * max(0.0, min(1.0, sp_score))
    if sp_score is None:
        return 0.0
    if sp_threshold is not None and sp_threshold > 0:
        ratio = max(0.0, min(1.0, sp_score / max(sp_threshold, 1e-6)))
        return 0.16 * ratio
    return 0.12 * max(0.0, min(1.0, sp_score))


def _ec_numbers(
    explicit: Iterable[str] | None,
    *,
    template: dict[str, Any],
    sp_payload: dict[str, Any],
    evidence: dict[str, Any],
) -> list[str]:
    values: list[str] = []
    for raw in (
        explicit,
        [template.get("ec")] if template.get("ec") else None,
        sp_payload.get("ec_numbers"),
        evidence.get("ec_numbers"),
        evidence.get("enzyme_ec_sample"),
    ):
        if isinstance(raw, str):
            values.append(raw)
        else:
            try:
                values.extend(str(item) for item in raw or [] if str(item or "").strip())
            except TypeError:
                pass
    return list(dict.fromkeys(item.strip() for item in values if item and item.strip()))


def _has_bridge_or_precedent_evidence(evidence: dict[str, Any]) -> bool:
    if not evidence:
        return False
    direct_keys = {
        "bridge_id",
        "bridge_direction",
        "match_type",
        "transition_signature",
        "reaction_id",
        "enzyme_reaction_id",
        "precedent_id",
        "source_db",
        "doi",
        "uniprot_id",
        "uniprot_accession",
        "substrate_product_similarity",
    }
    return any(key in evidence and evidence.get(key) not in (None, "", []) for key in direct_keys)


def _is_enzyme_like_source(value: Any) -> bool:
    text = str(value or "").lower()
    return any(token in text for token in ENZYMATIC_SOURCE_TOKENS)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_int(value: Any, default: int, *, lo: int | None = None) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = int(default)
    if lo is not None:
        out = max(lo, out)
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
