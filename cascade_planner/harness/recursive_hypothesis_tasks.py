"""Generate next-generation hypothesis subgoals after failed precursor searches."""
from __future__ import annotations

import hashlib
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem


RDLogger.DisableLog("rdApp.*")

RECURSIVE_HYPOTHESIS_TASK_SCHEMA = "recursive_hypothesis_task.v1"


def recursive_hypothesis_tasks_from_route_expansion(
    *,
    blackboard: dict[str, Any],
    route_expansion_result: dict[str, Any],
    max_tasks_per_parent: int = 3,
) -> list[dict[str, Any]]:
    """Create executable next-frontier tasks from rejected hypothesis subgoals.

    These rows are not route proof. They are bounded search targets that let the
    controller continue down one retrosynthetic branch after a first-level
    hypothesis precursor failed verification.
    """
    constraints = dict((blackboard.get("current_belief") or {}).get("constraints") or {})
    max_depth = int(constraints.get("max_recursive_hypothesis_depth") or 3)
    attempted = _attempted_or_queued_smiles(blackboard)
    target_profile = dict(blackboard.get("target_profile") or {})
    target_smiles = _canonical_smiles(str(target_profile.get("target_smiles") or target_profile.get("canonical_smiles") or ""))
    if target_smiles:
        attempted.add(target_smiles)
    tasks: list[dict[str, Any]] = []
    for row in _rejected_hypothesis_subgoals(route_expansion_result):
        subgoal = dict(row.get("subgoal") or {})
        parent_smiles = _canonical_smiles(str(subgoal.get("smiles") or ""))
        if not parent_smiles:
            continue
        parent_depth = _recursive_depth(subgoal)
        if parent_depth >= max_depth:
            continue
        failure_reasons = _failure_reasons(row, route_expansion_result)
        variants = _recursive_precursor_variants(parent_smiles, limit=max_tasks_per_parent)
        for variant in variants:
            precursor = _canonical_smiles(str(variant.get("smiles") or ""))
            if not precursor or precursor == parent_smiles or precursor in attempted:
                continue
            task = _task_row(
                parent_smiles=parent_smiles,
                precursor_smiles=precursor,
                subgoal=subgoal,
                variant=variant,
                failure_reasons=failure_reasons,
                recursive_depth=parent_depth + 1,
            )
            tasks.append(task)
            attempted.add(precursor)
            if len([item for item in tasks if item.get("parent_smiles") == parent_smiles]) >= max(1, int(max_tasks_per_parent or 1)):
                break
    return _dedupe_tasks(tasks)


def _rejected_hypothesis_subgoals(route_expansion_result: dict[str, Any]) -> list[dict[str, Any]]:
    payload = dict(route_expansion_result.get("result") or route_expansion_result or {})
    rows: list[dict[str, Any]] = []
    for raw in payload.get("subgoals") or []:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        if bool(row.get("accepted") or row.get("solved")):
            continue
        verifier = dict(row.get("verifier") or {})
        if bool(verifier.get("accepted")):
            continue
        subgoal = dict(row.get("subgoal") or {})
        if not _is_hypothesis_subgoal(subgoal):
            continue
        rows.append(row)
    return rows


def _is_hypothesis_subgoal(subgoal: dict[str, Any]) -> bool:
    policy = dict(subgoal.get("policy") or subgoal.get("chem_enzy_search_policy") or {})
    compiler = dict(policy.get("compiler_metadata") or {})
    preferred = dict(policy.get("preferred_subgoal") or {})
    text = " ".join(
        str(value or "")
        for value in (
            subgoal.get("source"),
            subgoal.get("name"),
            subgoal.get("task_id"),
            subgoal.get("recursive_hypothesis_task_id"),
            compiler.get("compiler_schema"),
            preferred.get("schema_version"),
        )
    ).lower()
    return bool(
        subgoal.get("hypothesis_only_not_solved")
        or subgoal.get("recursive_hypothesis_task_id")
        or compiler.get("hypothesis_only_not_solved")
        or compiler.get("recursive_hypothesis_frontier")
        or "hypothesis" in text
        or "visual_connectivity" in text
    )


def _task_row(
    *,
    parent_smiles: str,
    precursor_smiles: str,
    subgoal: dict[str, Any],
    variant: dict[str, Any],
    failure_reasons: list[str],
    recursive_depth: int,
) -> dict[str, Any]:
    task_id = "recursive_hypothesis:" + _short_hash(
        "|".join(
            [
                str(subgoal.get("recursive_hypothesis_task_id") or subgoal.get("name") or ""),
                parent_smiles,
                precursor_smiles,
                str(variant.get("variant_type") or ""),
                str(recursive_depth),
            ]
        )
    )
    precursor_set = _replace_component_in_set(
        str(subgoal.get("precursor_set_smiles") or ""),
        old_component=parent_smiles,
        new_component=precursor_smiles,
    )
    sibling_precursors = _precursor_components(precursor_set)
    sibling_precursors = [item for item in sibling_precursors if item != precursor_smiles]
    return {
        "schema_version": RECURSIVE_HYPOTHESIS_TASK_SCHEMA,
        "task_id": task_id,
        "task_type": "recursive_hypothesis_frontier_expansion",
        "task_scope": str(subgoal.get("task_scope") or "precursor"),
        "status": "pending",
        "source": "rejected_hypothesis_precursor",
        "parent_candidate_id": str(subgoal.get("candidate_id") or subgoal.get("recursive_hypothesis_task_id") or ""),
        "parent_subgoal_name": str(subgoal.get("name") or "hypothesis precursor"),
        "parent_smiles": parent_smiles,
        "precursor_smiles": precursor_smiles,
        "precursor_set_smiles": precursor_set,
        "precursor_component_index": int(subgoal.get("precursor_component_index") or 0),
        "precursor_component_count": int(subgoal.get("precursor_component_count") or 1),
        "multi_component_precursor_set": bool(subgoal.get("multi_component_precursor_set")),
        "requires_precursor_set_stitching": bool(subgoal.get("requires_precursor_set_stitching")),
        "sibling_precursor_smiles": sibling_precursors,
        "name": f"recursive_{variant.get('variant_type') or 'same_core_variant'}",
        "recursive_depth": int(recursive_depth),
        "operation_idea": str(variant.get("operation_idea") or "continue same-core redox/protection-state retrosynthesis"),
        "variant_type": str(variant.get("variant_type") or "same_core_variant"),
        "proposal_granularity": str(subgoal.get("proposal_granularity") or "same_core"),
        "route_objective_type": str(subgoal.get("route_objective_type") or "same_core_redox_or_protection_route"),
        "failure_response_policy": dict(subgoal.get("failure_response_policy") or {}),
        "failure_reasons": failure_reasons,
        "risk_flags": _dedupe(
            [
                "recursive_hypothesis_only",
                "same_core_transform_not_literature_exact",
                "requires_route_expansion_verifier",
                *[str(item) for item in variant.get("risk_flags") or []],
            ]
        ),
        "allowed_use": "route_expansion_subgoal_hint_only",
        "not_exact_literature_segment": True,
        "not_parent_route_proof": True,
        "requires_verifier": True,
        "child_route_cannot_promote_parent": True,
        "no_solved_claim": True,
    }


def _recursive_precursor_variants(smiles: str, *, limit: int) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    variants.extend(
        _reaction_variants(
            smiles,
            "[C:1](=[O:2])[OX2H:3]>>[C:1](=[O:2])Cl",
            variant_type="carboxylic_acid_to_acid_chloride_precursor",
            operation_idea="try the corresponding acid chloride activation state after a failed acid component",
            risk_flags=["alternate_acyl_activation_state", "acid_chloride_scope_hypothetical"],
            max_products=2,
        )
    )
    variants.extend(
        _reaction_variants(
            smiles,
            "[C:1](=[O:2])[OX2H:3]>>[C:1](=[O:2])OC",
            variant_type="carboxylic_acid_to_methyl_ester_precursor",
            operation_idea="try a methyl ester or masked acid frontier after a failed acid component",
            risk_flags=["masked_acid_frontier_hypothetical", "ester_exchange_direction_not_proven"],
            max_products=2,
        )
    )
    variants.extend(
        _reaction_variants(
            smiles,
            "[C:1](=[O:2])[OX2H:3]>>[C:1](=[O:2])OC(C)=O",
            variant_type="carboxylic_acid_to_mixed_anhydride_precursor",
            operation_idea="try a mixed-anhydride-like activated acyl donor after a failed acid component",
            risk_flags=["alternate_acyl_activation_state", "mixed_anhydride_scope_hypothetical"],
            max_products=2,
        )
    )
    variants.extend(
        _reaction_variants(
            smiles,
            "[C:1](=[O:2])Cl>>[C:1](=[O:2])O",
            variant_type="acid_chloride_to_carboxylic_acid_precursor",
            operation_idea="fall back from acid chloride to the carboxylic acid level",
            risk_flags=["alternate_acyl_activation_state", "leaving_group_choice_revised"],
            max_products=2,
        )
    )
    variants.extend(
        _reaction_variants(
            smiles,
            "[OX2H:1]>>[O:1]C(C)=O",
            variant_type="alcohol_to_acetate_protected_precursor",
            operation_idea="try a protected alcohol state before searching further upstream",
            risk_flags=["protecting_group_choice_hypothetical"],
            max_products=3,
            exclude_smarts=["[CX3](=O)[OX2H]"],
        )
    )
    variants.extend(
        _reaction_variants(
            smiles,
            "[OX2:1]C(C)=O>>[OX2H:1]",
            variant_type="acetate_to_free_alcohol_precursor",
            operation_idea="remove acetate protection and continue from the free alcohol frontier",
            risk_flags=["deprotection_direction_hypothetical"],
            max_products=3,
            exclude_smarts=["[CX3](=O)[OX2H]"],
        )
    )
    variants.extend(
        _reaction_variants(
            smiles,
            "[CH2:1][OX2H:2]>>[CH:1]=[O:2]",
            variant_type="primary_alcohol_to_aldehyde_precursor",
            operation_idea="continue through aldehyde-level side-chain oxidation state",
            risk_flags=["primary_alcohol_redox_direction_hypothetical"],
            max_products=2,
        )
    )
    variants.extend(
        _reaction_variants(
            smiles,
            "[C:1](=[O:2])>>[C:1]([O:2])",
            variant_type="carbonyl_to_hydroxy_precursor",
            operation_idea="continue through hydroxy steroid oxidation-state precursor",
            risk_flags=["alcohol_stereochemistry_unassigned"],
            max_products=4,
        )
    )
    for precursor in _enone_saturated_ketone_variants(smiles):
        variants.append(
            {
                "smiles": precursor,
                "variant_type": "enone_to_saturated_ketone_precursor",
                "operation_idea": "continue through a saturated ketone or masked enone frontier",
                "risk_flags": ["enone_regioselectivity_unproven"],
            }
        )
    return _dedupe_variant_rows(variants)[: max(1, int(limit or 1))]


def _reaction_variants(
    smiles: str,
    reaction_smarts: str,
    *,
    variant_type: str,
    operation_idea: str,
    risk_flags: list[str],
    max_products: int,
    exclude_smarts: list[str] | None = None,
) -> list[dict[str, Any]]:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return []
    if _rule_excluded(mol, list(exclude_smarts or [])):
        return []
    try:
        reaction = AllChem.ReactionFromSmarts(reaction_smarts)
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for product_tuple in reaction.RunReactants((mol,)):
        if not product_tuple:
            continue
        product = product_tuple[0]
        try:
            Chem.SanitizeMol(product)
        except Exception:
            continue
        canonical = Chem.MolToSmiles(product, isomericSmiles=True)
        if canonical:
            rows.append(
                {
                    "smiles": canonical,
                    "variant_type": variant_type,
                    "operation_idea": operation_idea,
                    "risk_flags": risk_flags,
                }
            )
        if len(_dedupe_variant_rows(rows)) >= max(1, int(max_products or 1)):
            break
    return _dedupe_variant_rows(rows)[: max(1, int(max_products or 1))]


def _rule_excluded(mol: Chem.Mol, exclude_smarts: list[str]) -> bool:
    for smarts in exclude_smarts:
        query = Chem.MolFromSmarts(str(smarts or ""))
        if query is not None and mol.HasSubstructMatch(query):
            return True
    return False


def _enone_saturated_ketone_variants(smiles: str) -> list[str]:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return []
    variants: list[str] = []
    patterns = [
        (Chem.MolFromSmarts("[#6]=[#6]-[CX3](=O)[#6]"), (0, 1)),
        (Chem.MolFromSmarts("[CX3](=O)-[#6]=[#6]"), (2, 3)),
    ]
    for query, bond_indices in patterns:
        if query is None:
            continue
        for match in mol.GetSubstructMatches(query):
            atom_a = int(match[bond_indices[0]])
            atom_b = int(match[bond_indices[1]])
            rw_mol = Chem.RWMol(mol)
            bond = rw_mol.GetBondBetweenAtoms(atom_a, atom_b)
            if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            bond.SetBondType(Chem.BondType.SINGLE)
            candidate = rw_mol.GetMol()
            try:
                Chem.SanitizeMol(candidate)
            except Exception:
                continue
            canonical = Chem.MolToSmiles(candidate, isomericSmiles=True)
            if canonical:
                variants.append(canonical)
    return _dedupe(variants)


def _recursive_depth(subgoal: dict[str, Any]) -> int:
    for key in ("recursive_depth", "recursive_hypothesis_depth", "hypothesis_generation"):
        try:
            return int(subgoal.get(key) or 0)
        except (TypeError, ValueError):
            continue
    policy = dict(subgoal.get("policy") or subgoal.get("chem_enzy_search_policy") or {})
    preferred = dict(policy.get("preferred_subgoal") or {})
    nested = preferred.get("recursive_hypothesis_task")
    if isinstance(nested, dict):
        try:
            return int(nested.get("recursive_depth") or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def _failure_reasons(row: dict[str, Any], route_expansion_result: dict[str, Any]) -> list[str]:
    verifier = dict(row.get("verifier") or {})
    payload = dict(route_expansion_result.get("result") or route_expansion_result or {})
    return _dedupe(
        [
            *[str(item) for item in row.get("reasons") or []],
            *[str(item) for item in verifier.get("reasons") or []],
            *[str(item) for item in payload.get("reasons") or []],
        ]
    )


def _attempted_or_queued_smiles(blackboard: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for task in blackboard.get("recursive_hypothesis_tasks") or []:
        if isinstance(task, dict):
            canonical = _canonical_smiles(str(task.get("precursor_smiles") or ""))
            if canonical:
                values.add(canonical)
    for row in blackboard.get("action_history") or []:
        if not isinstance(row, dict) or str(row.get("action_type") or "") != "expand_child_target":
            continue
        for smiles in _terminal_smiles_from_action_signature(str(row.get("action_signature") or "")):
            canonical = _canonical_smiles(smiles)
            if canonical:
                values.add(canonical)
    return values


def _terminal_smiles_from_action_signature(signature: str) -> list[str]:
    import json

    try:
        payload = dict(json.loads(signature or "{}").get("payload") or {})
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    out: list[str] = []
    for target in payload.get("subgoal_targets") or []:
        if not isinstance(target, dict):
            continue
        text = str(target.get("smiles") or "").strip()
        if text:
            out.append(text)
    return _dedupe(out)


def _canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _replace_component_in_set(precursor_set: str, *, old_component: str, new_component: str) -> str:
    canonical_set = _canonical_smiles(precursor_set)
    old = _canonical_smiles(old_component)
    new = _canonical_smiles(new_component)
    if not canonical_set or not old or not new:
        return ""
    components = _precursor_components(canonical_set)
    if old not in components:
        return canonical_set
    replaced = [new if item == old else item for item in components]
    return _canonical_smiles(".".join(replaced))


def _precursor_components(smiles: str) -> list[str]:
    canonical = _canonical_smiles(smiles)
    if not canonical:
        return []
    return _dedupe([part for part in canonical.split(".") if part])


def _dedupe_tasks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("precursor_smiles") or row.get("task_id") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _dedupe_variant_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        canonical = _canonical_smiles(str(row.get("smiles") or ""))
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        current = dict(row)
        current["smiles"] = canonical
        out.append(current)
    return out


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _short_hash(value: str) -> str:
    return hashlib.sha1(str(value or "").encode("utf-8")).hexdigest()[:12]
