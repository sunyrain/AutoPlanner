"""Shared schema for verifier-first cascade training data.

The verifier schema is intentionally narrower than expert feasibility review:
it records rule-checkable failure modes that can be used for perturbation
packs, preference generation, and search-time guardrails.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


SCHEMA_VERSION = "cascade_verifier.v1"
PERTURBATION_PACK_SCHEMA_VERSION = "cascade_perturbation_pack.v1"


class VerifierFailureReason(str, Enum):
    ATOM_BALANCE = "atom_balance_violation"
    PRODUCT_MISMATCH = "product_mismatch"
    INVALID_SMILES = "invalid_smiles"
    TEMPERATURE_CONFLICT = "temperature_conflict"
    PH_CONFLICT = "ph_conflict"
    SOLVENT_CONFLICT = "solvent_conflict"
    ENZYME_TOXICITY = "enzyme_toxicity"
    COFACTOR_LEDGER_GAP = "cofactor_ledger_gap"
    ROUTE_ORDER_MISMATCH = "route_order_mismatch"


@dataclass(frozen=True)
class CascadeVerifierFinding:
    reason: VerifierFailureReason | str
    severity: float = 1.0
    step_index: int | None = None
    stage_id: str = ""
    message: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reason"] = self.reason.value if isinstance(self.reason, VerifierFailureReason) else str(self.reason)
        return data


@dataclass(frozen=True)
class CascadeVerifierResult:
    feasible: bool
    score: float
    findings: tuple[CascadeVerifierFinding, ...] = field(default_factory=tuple)
    metrics: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @property
    def reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            reason = finding.reason.value if isinstance(finding.reason, VerifierFailureReason) else str(finding.reason)
            counts[reason] = counts.get(reason, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feasible": bool(self.feasible),
            "score": float(self.score),
            "reason_counts": self.reason_counts,
            "findings": [finding.to_dict() for finding in self.findings],
            "metrics": self.metrics,
        }


CASCADE_PERTURBATION_SPECS: list[dict[str, Any]] = [
    {
        "type": "atom_balance_drop_reactant_main",
        "expected_failure_reasons": [VerifierFailureReason.ATOM_BALANCE.value],
        "description": "Replace a material precursor with a tiny molecule so the product gains unexplained atoms.",
    },
    {
        "type": "atom_balance_drop_reactant_aux",
        "expected_failure_reasons": [VerifierFailureReason.ATOM_BALANCE.value],
        "description": "Drop supporting reactant material so the step becomes materially unbalanced.",
    },
    {
        "type": "atom_balance_tiny_carbon_source",
        "expected_failure_reasons": [VerifierFailureReason.ATOM_BALANCE.value],
        "description": "Use a one-carbon precursor for a larger product to create an unexplained carbon gain.",
    },
    {
        "type": "temperature_conflict_wide",
        "expected_failure_reasons": [VerifierFailureReason.TEMPERATURE_CONFLICT.value],
        "description": "Force two same-stage steps into non-overlapping temperature envelopes.",
    },
    {
        "type": "temperature_conflict_mild",
        "expected_failure_reasons": [VerifierFailureReason.TEMPERATURE_CONFLICT.value],
        "description": "Use a smaller but still incompatible temperature split between same-stage steps.",
    },
    {
        "type": "temperature_conflict_freezing_heated",
        "expected_failure_reasons": [VerifierFailureReason.TEMPERATURE_CONFLICT.value],
        "description": "Pair a subzero step with a heated step inside one stage.",
    },
    {
        "type": "ph_conflict_acidic_basic",
        "expected_failure_reasons": [VerifierFailureReason.PH_CONFLICT.value],
        "description": "Force two same-stage enzymatic/process steps into non-overlapping pH envelopes.",
    },
    {
        "type": "ph_conflict_neutral_to_basic",
        "expected_failure_reasons": [VerifierFailureReason.PH_CONFLICT.value],
        "description": "Use a narrower but still incompatible pH split between same-stage steps.",
    },
    {
        "type": "ph_conflict_strong_acid_base",
        "expected_failure_reasons": [VerifierFailureReason.PH_CONFLICT.value],
        "description": "Pair strongly acidic and strongly basic conditions inside one stage.",
    },
    {
        "type": "solvent_conflict",
        "expected_failure_reasons": [VerifierFailureReason.SOLVENT_CONFLICT.value],
        "description": "Force adjacent steps into incompatible aqueous/hydrophobic solvent classes.",
    },
    {
        "type": "solvent_conflict_reverse",
        "expected_failure_reasons": [VerifierFailureReason.SOLVENT_CONFLICT.value],
        "description": "Reverse aqueous/hydrophobic solvent assignment across adjacent same-stage steps.",
    },
    {
        "type": "enzyme_toxicity_solvent",
        "expected_failure_reasons": [VerifierFailureReason.ENZYME_TOXICITY.value],
        "description": "Place an enzymatic step under strongly enzyme-incompatible solvent/reagent conditions.",
    },
    {
        "type": "enzyme_toxicity_reagent",
        "expected_failure_reasons": [VerifierFailureReason.ENZYME_TOXICITY.value],
        "description": "Place an enzymatic step under a strongly enzyme-incompatible reagent.",
    },
    {
        "type": "enzyme_toxicity_strong_base",
        "expected_failure_reasons": [VerifierFailureReason.ENZYME_TOXICITY.value],
        "description": "Place an enzymatic step under a strong-base reagent condition.",
    },
    {
        "type": "cofactor_ledger_gap_single",
        "expected_failure_reasons": [VerifierFailureReason.COFACTOR_LEDGER_GAP.value],
        "description": "Add a cofactor requirement without a matching regeneration or material source.",
    },
    {
        "type": "cofactor_ledger_gap_multi",
        "expected_failure_reasons": [VerifierFailureReason.COFACTOR_LEDGER_GAP.value],
        "description": "Add multiple cofactor requirements without matching regeneration records.",
    },
    {
        "type": "cofactor_ledger_gap_atp",
        "expected_failure_reasons": [VerifierFailureReason.COFACTOR_LEDGER_GAP.value],
        "description": "Add an ATP requirement without a matching regeneration record.",
    },
    {
        "type": "cofactor_ledger_gap_fad",
        "expected_failure_reasons": [VerifierFailureReason.COFACTOR_LEDGER_GAP.value],
        "description": "Add an FAD requirement without a matching regeneration record.",
    },
    {
        "type": "route_order_swap_pair",
        "expected_failure_reasons": [VerifierFailureReason.ROUTE_ORDER_MISMATCH.value],
        "description": "Shuffle retrosynthetic steps so a step expands an intermediate not yet opened.",
    },
    {
        "type": "route_order_reverse",
        "expected_failure_reasons": [VerifierFailureReason.ROUTE_ORDER_MISMATCH.value],
        "description": "Reverse route order so later intermediates appear before their predecessors.",
    },
    {
        "type": "route_order_rotate",
        "expected_failure_reasons": [VerifierFailureReason.ROUTE_ORDER_MISMATCH.value],
        "description": "Rotate a multi-step route so the first expansion is no longer the target.",
    },
]
