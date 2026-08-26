"""Stable scientific identities shared by the canonical V4 graph."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

from rdkit import Chem, RDLogger

from cascade_planner.routes.admission import audit_retrosynthetic_candidate
from cascade_planner.application.strategy_contract import normalize_strategy_card


RDLogger.DisableLog("rdApp.*")


def molecule_identity(smiles: Any) -> tuple[str, str]:
    canonical = _canonical_smiles(smiles)
    if not canonical:
        return "", ""
    return f"mol:{_digest({'canonical_smiles': canonical})}", canonical


def reaction_edge_identity(
    product_smiles: Any,
    precursor_smiles: Iterable[Any],
    *,
    mapped_reaction_smiles: Any = "",
    mapped_product_smiles: Any = "",
    reaction_operations: Iterable[Mapping[str, Any]] = (),
    reactionjson_audit: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    audit = audit_retrosynthetic_candidate(
        product_smiles,
        precursor_smiles,
        mapped_reaction_smiles=mapped_reaction_smiles,
        mapped_product_smiles=mapped_product_smiles,
        reaction_operations=reaction_operations,
        reactionjson_audit=reactionjson_audit,
    )
    digest = str(audit.get("edge_digest") or "")
    return (f"edge:{digest}" if digest else ""), audit


def source_binding_identity(value: Mapping[str, Any]) -> str:
    row = dict(value)
    identity = {
        "source_kind": str(row.get("source_kind") or ""),
        "source_ref": str(row.get("source_ref") or ""),
        "registry_id": str(row.get("registry_id") or ""),
        "artifact_sha256": str(row.get("artifact_sha256") or ""),
        "independence_group": str(row.get("independence_group") or ""),
        "content_scope": str(row.get("content_scope") or ""),
    }
    return f"source:{_digest(identity)}"


def stock_observation_identity(value: Mapping[str, Any]) -> str:
    row = dict(value)
    identity = {
        "leaf_id": str(row.get("leaf_id") or ""),
        "canonical_smiles": str(row.get("canonical_smiles") or ""),
        "inventory_snapshot_set_id": str(row.get("inventory_snapshot_set_id") or ""),
        "audited_as_of": str(row.get("audited_as_of") or ""),
        "provider_content_hash": str(
            dict(row.get("provider_result") or {}).get("content_hash") or ""
        ),
    }
    return f"stock:{_digest(identity)}"


def route_family_identity(
    value: Mapping[str, Any],
    *,
    target_molecule_id: str,
) -> str:
    row = dict(value)
    strategy_card = normalize_strategy_card(row.get("strategy_card") or {})
    identity = {
        "target_molecule_id": target_molecule_id,
        "family_key": str(
            row.get("family_key")
            or row.get("route_family_id")
            or row.get("family_id")
            or row.get("name")
            or ""
        ),
        "strategic_disconnection": str(
            row.get("strategic_disconnection")
            or row.get("strategy")
            or row.get("rationale")
            or ""
        ),
        # Structured strategy identity is stronger than a prose family label,
        # but legacy families without a card retain their historical ids.
        **(
            {"strategy_digest": strategy_card["strategy_digest"]}
            if any(
                str(strategy_card.get(field) or "")
                for field in (
                    "scaffold_motif",
                    "key_forward_transformation",
                    "key_bond_signature",
                    "reaction_edit_digest",
                )
            )
            else {}
        ),
    }
    return f"route-family:{_digest(identity)}"


def hypothesis_identity(
    product_smiles: Any,
    precursor_smiles: Iterable[Any],
    *,
    mapped_reaction_smiles: Any = "",
    mapped_product_smiles: Any = "",
    reaction_operations: Iterable[Mapping[str, Any]] = (),
    reactionjson_audit: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    edge_id, audit = reaction_edge_identity(
        product_smiles,
        precursor_smiles,
        mapped_reaction_smiles=mapped_reaction_smiles,
        mapped_product_smiles=mapped_product_smiles,
        reaction_operations=reaction_operations,
        reactionjson_audit=reactionjson_audit,
    )
    digest = str(audit.get("edge_digest") or "")
    return (f"hypothesis:{digest}" if edge_id else ""), audit


def _canonical_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return ""
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "hypothesis_identity",
    "molecule_identity",
    "reaction_edge_identity",
    "route_family_identity",
    "source_binding_identity",
    "stock_observation_identity",
]
