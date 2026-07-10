"""Typed, content-addressed domain records for the route hypergraph v2 overlay.

The planner still emits the established ``*.v1`` dictionaries.  These frozen
records provide a migration boundary with stable identities and hashes without
making the advisory graph a route-proof authority.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterable, Mapping

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")

ROUTE_HYPERGRAPH_OVERLAY_SCHEMA = "route_hypergraph_overlay.v2"
ROUTE_NEIGHBORHOOD_SCHEMA = "route_neighborhood.v2"


def canonicalize_smiles(value: Any) -> str:
    """Return canonical isomeric SMILES, or an empty string for invalid input."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    molecule = Chem.MolFromSmiles(raw)
    if molecule is None:
        return ""
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def stable_content_hash(schema_version: str, payload: Mapping[str, Any]) -> str:
    """Hash a schema-qualified JSON payload using deterministic serialization."""
    encoded = json.dumps(
        {"schema_version": schema_version, **dict(payload)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def stable_domain_id(prefix: str, *parts: Any) -> str:
    """Build a stable identifier from identity-defining values."""
    encoded = "\x1f".join(_identity_text(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()[:24]}"


@dataclass(frozen=True, slots=True)
class MoleculeIdentity:
    """Stereochemistry-preserving identity for one molecule node."""

    SCHEMA_VERSION: ClassVar[str] = "molecule_identity.v2"

    smiles: str
    names: tuple[str, ...] = ()
    external_identifiers: tuple[tuple[str, str], ...] = ()
    canonical_isomeric_smiles: str = field(init=False)
    molecule_id: str = field(init=False)
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        canonical = canonicalize_smiles(self.smiles)
        names = _text_tuple(self.names)
        identifiers = tuple(
            sorted(
                {
                    (str(namespace or "").strip(), str(value or "").strip())
                    for namespace, value in self.external_identifiers
                    if str(namespace or "").strip() and str(value or "").strip()
                }
            )
        )
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "external_identifiers", identifiers)
        object.__setattr__(self, "canonical_isomeric_smiles", canonical)
        object.__setattr__(self, "molecule_id", stable_domain_id("mol", f"smiles:{canonical}"))
        object.__setattr__(
            self,
            "content_hash",
            stable_content_hash(
                self.SCHEMA_VERSION,
                {
                    "canonical_isomeric_smiles": canonical,
                    "names": names,
                    "external_identifiers": identifiers,
                },
            ),
        )

    def validate(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.canonical_isomeric_smiles:
            reasons.append("invalid_molecule_smiles")
        if self.molecule_id != stable_domain_id("mol", f"smiles:{self.canonical_isomeric_smiles}"):
            reasons.append("unstable_molecule_id")
        return tuple(reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "molecule_id": self.molecule_id,
            "canonical_isomeric_smiles": self.canonical_isomeric_smiles,
            "names": list(self.names),
            "external_identifiers": [
                {"namespace": namespace, "value": value}
                for namespace, value in self.external_identifiers
            ],
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True, slots=True)
class EvidenceClaim:
    """One provenance-preserving claim; correlation is explicit in support_group."""

    SCHEMA_VERSION: ClassVar[str] = "evidence_claim.v2"

    source_channel: str
    support_group: str
    evidence_level: str
    confidence: str
    candidate_id: str = ""
    source_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    report_ref: str = ""
    claim_id: str = field(init=False)
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        channel = str(self.source_channel or "other").strip() or "other"
        support_group = str(self.support_group or "").strip()
        evidence_level = str(self.evidence_level or "model_only").strip() or "model_only"
        confidence = str(self.confidence or "low").strip() or "low"
        candidate_id = str(self.candidate_id or "").strip()
        source_refs = _text_tuple(self.source_refs)
        evidence_refs = _text_tuple(self.evidence_refs)
        report_ref = str(self.report_ref or "").strip()
        payload = {
            "source_channel": channel,
            "support_group": support_group,
            "evidence_level": evidence_level,
            "confidence": confidence,
            "candidate_id": candidate_id,
            "source_refs": source_refs,
            "evidence_refs": evidence_refs,
            "report_ref": report_ref,
        }
        object.__setattr__(self, "source_channel", channel)
        object.__setattr__(self, "support_group", support_group)
        object.__setattr__(self, "evidence_level", evidence_level)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "source_refs", source_refs)
        object.__setattr__(self, "evidence_refs", evidence_refs)
        object.__setattr__(self, "report_ref", report_ref)
        object.__setattr__(
            self,
            "claim_id",
            stable_domain_id(
                "claim",
                channel,
                support_group,
                candidate_id,
                source_refs,
                evidence_refs,
                report_ref,
            ),
        )
        object.__setattr__(self, "content_hash", stable_content_hash(self.SCHEMA_VERSION, payload))

    @classmethod
    def from_source_record(cls, record: Mapping[str, Any]) -> EvidenceClaim:
        return cls(
            source_channel=str(record.get("source_channel") or "other"),
            support_group=str(record.get("support_group") or ""),
            evidence_level=str(record.get("evidence_level") or "model_only"),
            confidence=str(record.get("confidence") or "low"),
            candidate_id=str(record.get("candidate_id") or ""),
            source_refs=tuple(str(value) for value in record.get("source_refs") or []),
            evidence_refs=tuple(str(value) for value in record.get("evidence_refs") or []),
            report_ref=str(record.get("report_ref") or ""),
        )

    def validate(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.source_channel:
            reasons.append("missing_evidence_source_channel")
        if not self.support_group:
            reasons.append("missing_evidence_support_group")
        if self.source_channel.startswith("codex_") and self.support_group != "codex_model":
            reasons.append("codex_claim_has_independent_support_group")
        return tuple(reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "claim_id": self.claim_id,
            "candidate_id": self.candidate_id,
            "source_channel": self.source_channel,
            "support_group": self.support_group,
            "evidence_level": self.evidence_level,
            "confidence": self.confidence,
            "source_refs": list(self.source_refs),
            "evidence_refs": list(self.evidence_refs),
            "report_ref": self.report_ref,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True, slots=True)
class ReactionCandidateEnvelope:
    """A source candidate bound to canonical molecule identities and evidence."""

    SCHEMA_VERSION: ClassVar[str] = "reaction_candidate_envelope.v2"

    product: MoleculeIdentity
    precursors: tuple[MoleculeIdentity, ...]
    reaction_family: str
    source_candidate_ids: tuple[str, ...] = ()
    evidence_claims: tuple[EvidenceClaim, ...] = ()
    transformation_rationale: str = ""
    conditions: tuple[str, ...] = ()
    catalysts: tuple[str, ...] = ()
    enzymes: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    required_validation: tuple[str, ...] = ()
    envelope_id: str = field(init=False)
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        precursors = tuple(sorted(self.precursors, key=lambda row: row.molecule_id))
        family = str(self.reaction_family or "unspecified").strip() or "unspecified"
        source_ids = _text_tuple(self.source_candidate_ids)
        claims = tuple(sorted(self.evidence_claims, key=lambda row: row.claim_id))
        rationale = str(self.transformation_rationale or "").strip()
        conditions = _text_tuple(self.conditions)
        catalysts = _text_tuple(self.catalysts)
        enzymes = _text_tuple(self.enzymes)
        limitations = _text_tuple(self.limitations)
        required_validation = _text_tuple(self.required_validation)
        object.__setattr__(self, "precursors", precursors)
        object.__setattr__(self, "reaction_family", family)
        object.__setattr__(self, "source_candidate_ids", source_ids)
        object.__setattr__(self, "evidence_claims", claims)
        object.__setattr__(self, "transformation_rationale", rationale)
        object.__setattr__(self, "conditions", conditions)
        object.__setattr__(self, "catalysts", catalysts)
        object.__setattr__(self, "enzymes", enzymes)
        object.__setattr__(self, "limitations", limitations)
        object.__setattr__(self, "required_validation", required_validation)
        object.__setattr__(
            self,
            "envelope_id",
            stable_domain_id(
                "envelope",
                self.product.molecule_id,
                tuple(row.molecule_id for row in precursors),
                source_ids,
                tuple(row.claim_id for row in claims),
            ),
        )
        object.__setattr__(
            self,
            "content_hash",
            stable_content_hash(
                self.SCHEMA_VERSION,
                {
                    "product_molecule_id": self.product.molecule_id,
                    "precursor_molecule_ids": tuple(row.molecule_id for row in precursors),
                    "reaction_family": family,
                    "source_candidate_ids": source_ids,
                    "evidence_claim_ids": tuple(row.claim_id for row in claims),
                    "transformation_rationale": rationale,
                    "conditions": conditions,
                    "catalysts": catalysts,
                    "enzymes": enzymes,
                    "limitations": limitations,
                    "required_validation": required_validation,
                },
            ),
        )

    def validate(self) -> tuple[str, ...]:
        reasons = [f"product:{reason}" for reason in self.product.validate()]
        if not self.precursors:
            reasons.append("missing_candidate_precursors")
        for index, precursor in enumerate(self.precursors):
            reasons.extend(f"precursor:{index}:{reason}" for reason in precursor.validate())
        for index, claim in enumerate(self.evidence_claims):
            reasons.extend(f"evidence:{index}:{reason}" for reason in claim.validate())
        return tuple(reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "envelope_id": self.envelope_id,
            "product_molecule_id": self.product.molecule_id,
            "precursor_molecule_ids": [row.molecule_id for row in self.precursors],
            "reaction_family": self.reaction_family,
            "source_candidate_ids": list(self.source_candidate_ids),
            "evidence_claim_ids": [row.claim_id for row in self.evidence_claims],
            "transformation_rationale": self.transformation_rationale,
            "conditions": list(self.conditions),
            "catalysts": list(self.catalysts),
            "enzymes": list(self.enzymes),
            "limitations": list(self.limitations),
            "required_validation": list(self.required_validation),
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True, slots=True)
class ReactionHyperedge:
    """One canonical product-to-precursor retrosynthetic hyperedge."""

    SCHEMA_VERSION: ClassVar[str] = "reaction_hyperedge.v2"

    product: MoleculeIdentity
    precursors: tuple[MoleculeIdentity, ...]
    candidate_envelope_ids: tuple[str, ...] = ()
    evidence_claim_ids: tuple[str, ...] = ()
    source_channels: tuple[str, ...] = ()
    independent_support_groups: tuple[str, ...] = ()
    reaction_families: tuple[str, ...] = ()
    rank_score: float = 0.0
    advisory_only: bool = True
    hyperedge_id: str = field(init=False)
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        precursors = tuple(sorted(self.precursors, key=lambda row: row.molecule_id))
        envelope_ids = _text_tuple(self.candidate_envelope_ids)
        claim_ids = _text_tuple(self.evidence_claim_ids)
        channels = _text_tuple(self.source_channels)
        groups = _text_tuple(self.independent_support_groups)
        families = _text_tuple(self.reaction_families)
        score = round(max(0.0, min(1.0, float(self.rank_score or 0.0))), 4)
        object.__setattr__(self, "precursors", precursors)
        object.__setattr__(self, "candidate_envelope_ids", envelope_ids)
        object.__setattr__(self, "evidence_claim_ids", claim_ids)
        object.__setattr__(self, "source_channels", channels)
        object.__setattr__(self, "independent_support_groups", groups)
        object.__setattr__(self, "reaction_families", families)
        object.__setattr__(self, "rank_score", score)
        object.__setattr__(
            self,
            "hyperedge_id",
            stable_domain_id(
                "rxn",
                self.product.molecule_id,
                tuple(row.molecule_id for row in precursors),
            ),
        )
        object.__setattr__(
            self,
            "content_hash",
            stable_content_hash(
                self.SCHEMA_VERSION,
                {
                    "product_molecule_id": self.product.molecule_id,
                    "precursor_molecule_ids": tuple(row.molecule_id for row in precursors),
                    "candidate_envelope_ids": envelope_ids,
                    "evidence_claim_ids": claim_ids,
                    "source_channels": channels,
                    "independent_support_groups": groups,
                    "reaction_families": families,
                    "rank_score": score,
                    "advisory_only": bool(self.advisory_only),
                },
            ),
        )

    def validate(self) -> tuple[str, ...]:
        reasons = [f"product:{reason}" for reason in self.product.validate()]
        if not self.precursors:
            reasons.append("missing_hyperedge_precursors")
        if self.product.molecule_id in {row.molecule_id for row in self.precursors} and len(self.precursors) == 1:
            reasons.append("identity_hyperedge")
        if any(channel.startswith("codex_") for channel in self.source_channels):
            codex_groups = {group for group in self.independent_support_groups if group.startswith("codex")}
            if codex_groups != {"codex_model"}:
                reasons.append("codex_roles_not_correlated")
        return tuple(reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "hyperedge_id": self.hyperedge_id,
            "product_molecule_id": self.product.molecule_id,
            "precursor_molecule_ids": [row.molecule_id for row in self.precursors],
            "candidate_envelope_ids": list(self.candidate_envelope_ids),
            "evidence_claim_ids": list(self.evidence_claim_ids),
            "source_channels": list(self.source_channels),
            "independent_support_groups": list(self.independent_support_groups),
            "reaction_families": list(self.reaction_families),
            "rank_score": self.rank_score,
            "advisory_only": bool(self.advisory_only),
            "no_solved_claim": True,
            "not_parent_route_proof": True,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True, slots=True)
class AlternativeSet:
    """Competing hyperedges that produce the same canonical product."""

    SCHEMA_VERSION: ClassVar[str] = "alternative_set.v2"

    product_molecule_id: str
    hyperedge_ids: tuple[str, ...]
    kind: str = "competing_disconnections"
    alternative_set_id: str = field(init=False)
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        product_id = str(self.product_molecule_id or "").strip()
        edge_ids = _text_tuple(self.hyperedge_ids)
        kind = str(self.kind or "competing_disconnections").strip() or "competing_disconnections"
        object.__setattr__(self, "product_molecule_id", product_id)
        object.__setattr__(self, "hyperedge_ids", edge_ids)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "alternative_set_id", stable_domain_id("alternative", product_id))
        object.__setattr__(
            self,
            "content_hash",
            stable_content_hash(
                self.SCHEMA_VERSION,
                {"product_molecule_id": product_id, "hyperedge_ids": edge_ids, "kind": kind},
            ),
        )

    def validate(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.product_molecule_id:
            reasons.append("missing_alternative_product")
        if len(self.hyperedge_ids) < 2:
            reasons.append("alternative_set_requires_multiple_hyperedges")
        return tuple(reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "alternative_set_id": self.alternative_set_id,
            "product_molecule_id": self.product_molecule_id,
            "hyperedge_ids": list(self.hyperedge_ids),
            "kind": self.kind,
            "requires_selection": True,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True, slots=True)
class RouteVariant:
    """One advisory selection of hyperedges through the canonical graph."""

    SCHEMA_VERSION: ClassVar[str] = "route_variant.v2"

    root_molecule_id: str
    retrosynthetic_hyperedge_ids: tuple[str, ...]
    molecule_ids: tuple[str, ...] = ()
    frontier_molecule_ids: tuple[str, ...] = ()
    rank_score: float = 0.0
    route_variant_id: str = field(init=False)
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        root_id = str(self.root_molecule_id or "").strip()
        edge_ids = _ordered_text_tuple(self.retrosynthetic_hyperedge_ids)
        molecule_ids = _ordered_text_tuple(self.molecule_ids)
        frontier_ids = _text_tuple(self.frontier_molecule_ids)
        score = round(max(0.0, min(1.0, float(self.rank_score or 0.0))), 4)
        object.__setattr__(self, "root_molecule_id", root_id)
        object.__setattr__(self, "retrosynthetic_hyperedge_ids", edge_ids)
        object.__setattr__(self, "molecule_ids", molecule_ids)
        object.__setattr__(self, "frontier_molecule_ids", frontier_ids)
        object.__setattr__(self, "rank_score", score)
        object.__setattr__(
            self,
            "route_variant_id",
            stable_domain_id("variant", root_id, edge_ids, frontier_ids),
        )
        object.__setattr__(
            self,
            "content_hash",
            stable_content_hash(
                self.SCHEMA_VERSION,
                {
                    "root_molecule_id": root_id,
                    "retrosynthetic_hyperedge_ids": edge_ids,
                    "molecule_ids": molecule_ids,
                    "frontier_molecule_ids": frontier_ids,
                    "rank_score": score,
                },
            ),
        )

    def validate(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.root_molecule_id:
            reasons.append("missing_route_variant_root")
        if not self.retrosynthetic_hyperedge_ids:
            reasons.append("missing_route_variant_hyperedges")
        return tuple(reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "route_variant_id": self.route_variant_id,
            "root_molecule_id": self.root_molecule_id,
            "retrosynthetic_hyperedge_ids": list(self.retrosynthetic_hyperedge_ids),
            "forward_hyperedge_ids": list(reversed(self.retrosynthetic_hyperedge_ids)),
            "molecule_ids": list(self.molecule_ids),
            "frontier_molecule_ids": list(self.frontier_molecule_ids),
            "rank_score": self.rank_score,
            "advisory_only": True,
            "solved": False,
            "executable": False,
            "not_parent_route_proof": True,
            "content_hash": self.content_hash,
        }


def _identity_text(value: Any) -> str:
    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return str(value or "")


def _text_tuple(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(sorted({str(value or "").strip() for value in values if str(value or "").strip()}))


def _ordered_text_tuple(values: Iterable[Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return tuple(out)
