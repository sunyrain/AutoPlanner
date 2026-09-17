"""Pure canonical route review inputs and mapped-boundary projections.

This compiler reads graph facts; it does not schedule workers, mutate routes,
consume budgets, or grant chemical proof. Shared by Director and target closeout.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Mapping

from cascade_planner.application.stereochemistry import canonical_stereo_smiles as _canonical_smiles

from rdkit import Chem

from cascade_planner.application.reaction_inputs import (
    reaction_input_smiles,
    mapped_reaction_input_smiles,
)
from cascade_planner.application.reaction_proof_versions import active_reaction_proofs
from cascade_planner.application.reactionjson_replay import ReactionJsonReplayError
from cascade_planner.application.routejson_compiler import RouteJSONCompiler
from cascade_planner.application.route_edge_scope import (
    origin_condition_predictions,
    route_family_bound_origin_records,
    route_family_scoped_edge_ids,
)


def review_mentions_changed_maps(text: str, changed_maps: set[int]) -> bool:
    """Refuse stale atom references without treating temperatures as map IDs."""
    references = re.findall(
        r"\b(?i:maps?|atoms?)\s*(?i:numbers?\s*)?[:#]?\s*"
        r"(\d+(?:(?:[,\s–—/-]|and)+\d+)*)|"
        r"\b(?:C|O|N|B|P|S|F|Cl|Br|Si)[:#]?(\d+)\b",
        text,
    )
    return bool(changed_maps & {
        int(number) for groups in references for group in groups
        for number in re.findall(r"\d+", group)
    })


def equivalent_review_atom_numbering(old_prompt: str, new_prompt: str) -> dict[int, int] | None:
    """Recognize only a consistent atom renumbering of an otherwise identical review.

    This preserves ordered operations, isotopes, stereo, prose and strategy.
    Molecule-set order is representational; operation order is not.
    """
    marker = "PaperMatchedRouteCriticInput:\n"
    old_head, sep, old_tail = old_prompt.partition(marker)
    new_head, new_sep, new_tail = new_prompt.partition(marker)
    if not sep or not new_sep or old_head != new_head:
        return None
    try:
        old, new = json.loads(old_tail), json.loads(new_tail)
    except (ValueError, TypeError):
        return None
    translation: dict[int, int] = {}
    inverse: dict[int, int] = {}
    molecules: list[tuple[str, str]] = []
    prose: list[str] = []

    def bind(a: int, b: int) -> bool:
        if not isinstance(a, int) or not isinstance(b, int) or a < 1 or b < 1:
            return False
        if translation.get(a, b) != b or inverse.get(b, a) != a:
            return False
        translation[a], inverse[b] = b, a
        return True

    def compare(a: Any, b: Any, key: str = "") -> bool:
        if type(a) is not type(b):
            return False
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(compare(v, b[k], k) for k, v in a.items())
        if isinstance(a, list):
            if key in {"mapped_precursor_smiles", "mapped_reaction_input_smiles", "mapped_auxiliary_reagent_smiles"}:
                a = sorted(a, key=_canonical_smiles)
                b = sorted(b, key=_canonical_smiles)
            return len(a) == len(b) and all(compare(x, y, key) for x, y in zip(a, b))
        if key in {"map_a", "map_b", "map_idx", "map_indices"}:
            return bind(a, b)
        if isinstance(a, str) and (key.startswith("mapped_") or key == "fragment_smiles"):
            molecules.append((a, b))
            return True
        if isinstance(a, str):
            prose.append(a)
        return a == b

    if not compare(old, new):
        return None
    # Operation identities above constrain molecular symmetry before matching.
    for old_smiles, new_smiles in molecules:
        left, right = Chem.MolFromSmiles(old_smiles), Chem.MolFromSmiles(new_smiles)
        if left is None or right is None:
            return None
        left_maps = [atom.GetAtomMapNum() for atom in left.GetAtoms()]
        right_maps = [atom.GetAtomMapNum() for atom in right.GetAtoms()]
        for molecule in (left, right):
            for atom in molecule.GetAtoms():
                atom.SetAtomMapNum(0)
        if Chem.MolToSmiles(left) != Chem.MolToSmiles(right):
            return None
        candidates = []
        for match in right.GetSubstructMatches(left, useChirality=True, uniquify=False, maxMatches=256):
            pairs = list(zip(left_maps, (right_maps[i] for i in match)))
            if all((a == b == 0) or (
                a > 0 and b > 0 and translation.get(a, b) == b and inverse.get(b, a) == a
            ) for a, b in pairs):
                candidates.append(pairs)
        if not candidates:
            return None
        pairs = min(candidates, key=lambda rows: (sum(a != b for a, b in rows), rows))
        for a, b in pairs:
            if a and not bind(a, b):
                return None
    # Check exact translated molecular encodings too: useChirality substructure
    # matching alone does not demand an unspecified query center stay unspecified.
    for a, b in molecules:
        molecule = Chem.MolFromSmiles(a)
        for atom in molecule.GetAtoms():
            if atom.GetAtomMapNum():
                atom.SetAtomMapNum(translation[atom.GetAtomMapNum()])
        if Chem.MolToSmiles(molecule) != Chem.MolToSmiles(Chem.MolFromSmiles(b)):
            return None
    changed = {value for a, b in translation.items() if a != b for value in (a, b)}
    if review_mentions_changed_maps(" ".join(prose), changed):
        return None
    return translation


def equivalent_whole_route_review(old_prompt: str, new_prompt: str) -> tuple[dict[int, int], dict[str, str]] | None:
    """Align review slots before checking the complete reaction contract.

    Unique products establish correspondence only, never equivalence. The
    strict comparison then checks all operations, inputs, conditions, stereo
    and strategy. A saved repair review may contain additional concerns to
    reassess; it still assessed every current reaction under the same policy.
    New or changed concerns cannot reuse a less informed review.
    """
    marker = "PaperMatchedRouteCriticInput:\n"
    old_head, sep, old_tail = old_prompt.partition(marker)
    new_head, new_sep, new_tail = new_prompt.partition(marker)
    if not sep or not new_sep or old_head != new_head:
        return None
    try:
        old, new = json.loads(old_tail), json.loads(new_tail)
    except (TypeError, ValueError):
        return None
    old_steps, new_steps = old.get("steps", []), new.get("steps", [])
    if not old_steps or len(old_steps) != len(new_steps):
        return None
    old_by_product = {_canonical_smiles(s.get("mapped_product_smiles")): s for s in old_steps}
    new_products = [_canonical_smiles(s.get("mapped_product_smiles")) for s in new_steps]
    if (len(old_by_product) != len(old_steps) or len(set(new_products)) != len(new_steps)
            or set(old_by_product) != set(new_products) or "" in old_by_product):
        return None
    slots = {str(old_by_product[p].get("review_slot")): str(s.get("review_slot"))
             for p, s in zip(new_products, new_steps)}
    if any(not re.fullmatch(r"review-\d+", s) for s in (*slots, *slots.values())):
        return None
    if len(set(slots.values())) != len(new_steps):
        return None
    old["steps"] = [{**old_by_product[p], "review_slot": s["review_slot"]}
                    for p, s in zip(new_products, new_steps)]
    old_concerns = old.get("repair_requirements_to_reassess", [])
    new_concerns = new.get("repair_requirements_to_reassess", [])
    if not isinstance(old_concerns, list) or not isinstance(new_concerns, list):
        return None
    if not set(new_concerns).issubset(old_concerns):
        return None
    old.pop("repair_requirements_to_reassess", None)
    new.pop("repair_requirements_to_reassess", None)
    mapping = equivalent_review_atom_numbering(
        old_head + marker + json.dumps(old), new_head + marker + json.dumps(new),
    )
    return (mapping, slots) if mapping is not None else None


@dataclass(frozen=True, slots=True)
class RevisionBoundRouteCriticContext:
    """One final, target-rooted route revision owned by the Route Critic."""

    target_smiles: str
    route_family_id: str
    route_sha256: str
    graph_revision: int
    branch_index: int
    edge_ids: tuple[str, ...]
    steps: tuple[Mapping[str, Any], ...]
    # ``strategy_card`` is the root steering hypothesis retained for API
    # compatibility.  It is never a route-repair admission constraint.
    strategy_card: Mapping[str, Any]
    selected_strategy_lineage: tuple[Mapping[str, Any], ...] = ()
    strategy_milestone_cards: tuple[Mapping[str, Any], ...] = ()


def _strategy_card_digest(strategy_card: Mapping[str, Any] | None) -> str:
    card = dict(strategy_card or {})
    return str(
        card.get("strategy_digest") or card.get("content_sha256") or card.get("strategy_id") or ""
    )


def _selected_strategy_lineage_from_materialized_steps(
    *,
    root_strategy_card: Mapping[str, Any],
    strategy_milestone_cards: Iterable[Mapping[str, Any]],
    steps: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Recover only Strategy horizons referenced by a materialized route.

    Canonical edges retain Strategy digest/index metadata rather than copying
    whole cards.  The root hypothesis is serialized separately, so this
    projection contains only later branch-local horizons actually referenced
    by selected edges, never a duplicate root or an unrelated milestone merely
    because it was generated during search.
    """

    root = dict(root_strategy_card or {})
    step_rows = [dict(row) for row in steps if isinstance(row, Mapping)]
    referenced_digests = {
        str(row.get("strategy_digest") or "")
        for row in step_rows
        if str(row.get("strategy_digest") or "")
    }
    referenced_indices = {
        max(1, int(row.get("strategy_milestone_index") or 1)) for row in step_rows
    }
    cards = [dict(row) for row in strategy_milestone_cards if isinstance(row, Mapping)]
    selected: list[dict[str, Any]] = []
    root_identity = (
        _strategy_card_digest(root)
        or json.dumps(
            root,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if root
        else ""
    )
    seen: set[str] = {root_identity} if root_identity else set()

    def add(card: Mapping[str, Any]) -> None:
        value = dict(card)
        if not value:
            return
        identity = _strategy_card_digest(value) or json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if identity in seen:
            return
        seen.add(identity)
        selected.append(value)

    for index, card in enumerate(cards, start=1):
        digest = _strategy_card_digest(card)
        if (digest and digest in referenced_digests) or (
            not referenced_digests and index in referenced_indices
        ):
            add(card)
    return selected


def _canonical_mapped_smiles(value: Any) -> str:
    """Canonicalize a mapped boundary without changing its map namespace."""

    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None or any(atom.GetAtomMapNum() <= 0 for atom in molecule.GetAtoms()):
        return ""
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _canonical_mapped_reactant_smiles(value: Any) -> str:
    """Canonicalize a mapped reactant while allowing an unmapped leaving atom.

    The Host reaction verifier deliberately permits bounded, unmapped departing
    atoms (for example chloride in a silylation).  A Route Critic may inspect
    that verified mapping without treating it as reaction proof.  Builder
    replay still uses :func:`_canonical_mapped_smiles`, which requires every
    atom in its mutable product boundary to be mapped.
    """

    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None or not any(atom.GetAtomMapNum() > 0 for atom in molecule.GetAtoms()):
        return ""
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _validated_edge_mapped_boundaries(
    edge: Mapping[str, Any],
) -> tuple[str, list[str]]:
    """Read mapped boundaries from the current Host reaction proof.

    Ordinary route hypotheses are atom-mapped by the Host validator after
    materialization, so they do not necessarily carry a ReactionJSON replay
    audit. The proof already binds one complete mapping to the exact canonical
    product/reactant multiset; consume that authority directly instead of
    copying a second mapped representation onto the edge.
    """

    expected_product = _canonical_smiles(edge.get("product_smiles"))
    expected_precursors = sorted(_canonical_smiles(value) for value in reaction_input_smiles(edge))
    required_checks = (
        "mapped_reaction_present",
        "mapped_product_matches",
        "mapped_reactants_match",
        "atom_maps_complete",
        "product_atom_maps_complete",
        "atom_maps_unique",
    )
    for proof in reversed(active_reaction_proofs(edge.get("reaction_proofs") or ())):
        checks = dict(proof.get("checks") or {})
        if proof.get("accepted") is not True or any(
            checks.get(name) is not True for name in required_checks
        ):
            continue
        parts = str(proof.get("mapped_reaction") or "").split(">")
        if len(parts) != 3:
            continue
        mapped_precursors = [
            value for item in parts[0].split(".") if (value := _canonical_mapped_smiles(item))
        ]
        mapped_products = [
            value for item in parts[2].split(".") if (value := _canonical_mapped_smiles(item))
        ]
        if (
            len(mapped_products) != 1
            or _canonical_smiles(mapped_products[0]) != expected_product
            or len(mapped_precursors) != len(expected_precursors)
            or sorted(_canonical_smiles(value) for value in mapped_precursors)
            != expected_precursors
        ):
            continue
        return mapped_products[0], mapped_precursors
    return "", []


def _route_critic_edge_mapped_boundaries(
    edge: Mapping[str, Any],
) -> tuple[str, list[str]]:
    """Read a Host-bound mapping for Critic inspection, not route admission.

    ``ReactionStepProof.accepted`` means that the reaction itself reached a
    validated-transform or precedent tier.  It is intentionally false for a
    structurally consistent ``L2_mapping_consistent`` edge.  Final route review
    needs the latter mapping as input, while the proof tier and its rejection
    reasons remain unchanged.  Product identity must be fully mapped; bounded
    unmapped leaving atoms are allowed only on precursor components.
    """

    expected_product = _canonical_smiles(edge.get("product_smiles"))
    expected_precursors = sorted(_canonical_smiles(value) for value in reaction_input_smiles(edge))
    # These checks establish that the serialized mapping names the canonical
    # edge and that every product atom has an unambiguous provenance identity.
    # Other verifier checks (departure budget, transform registry, precedent,
    # conditions) remain chemical-proof axes and must not gate Critic input.
    required_checks = (
        "structures_materialized",
        "mapped_reaction_present",
        "mapped_product_matches",
        "mapped_reactants_match",
        "product_atom_maps_complete",
        "atom_maps_unique",
        "mapped_elements_preserved",
        "stereochemical_product_matches",
    )

    for proof in reversed(active_reaction_proofs(edge.get("reaction_proofs") or ())):
        checks = dict(proof.get("checks") or {})
        if any(checks.get(name) is not True for name in required_checks) or not (
            checks.get("product_atoms_have_reactant_provenance") is True
            or checks.get("external_atom_source_replayed") is True
        ):
            continue
        parts = str(proof.get("mapped_reaction") or "").split(">")
        if len(parts) != 3:
            continue
        # Host proofs are normally forward (precursors >> product), but bind
        # direction from canonical identities so imported mapped reactions
        # cannot silently invert the route.
        for precursor_text, product_text in ((parts[0], parts[2]), (parts[2], parts[0])):
            mapped_precursors = [
                value
                for item in precursor_text.split(".")
                if (value := _canonical_mapped_reactant_smiles(item))
            ]
            mapped_products = [
                value
                for item in product_text.split(".")
                if (value := _canonical_mapped_smiles(item))
            ]
            if (
                len(mapped_products) == 1
                and _canonical_smiles(mapped_products[0]) == expected_product
                and len(mapped_precursors) == len(expected_precursors)
                and sorted(_canonical_smiles(value) for value in mapped_precursors)
                == expected_precursors
            ):
                return mapped_products[0], mapped_precursors
    return "", []


def compile_revision_bound_route_critic_context(
    graph: Mapping[str, Any],
    *,
    route_family_id: str,
    include_unselected: bool = False,
) -> tuple[RevisionBoundRouteCriticContext | None, dict[str, Any]]:
    """Compile the exact target-rooted route revision for final Critic review.

    The digest covers only chemistry visible to the Critic.  Unrelated graph
    revisions therefore do not trigger another model call, while any new
    materialized edge, edit program, mapped boundary, condition, or Strategy
    binding does.
    """

    route_id = str(route_family_id or "")
    route = dict(dict(graph.get("route_families") or {}).get(route_id) or {})
    if not route or (route.get("selected") is False and not include_unselected):
        return None, {
            "reason": "final_route_critic_route_unavailable",
            "route_family_id": route_id,
        }
    molecules = dict(graph.get("molecules") or {})
    edges = dict(graph.get("edges") or {})
    target_id = str(graph.get("target_molecule_id") or "")
    target_smiles = _canonical_smiles(dict(molecules.get(target_id) or {}).get("canonical_smiles"))
    if not target_id or not target_smiles:
        return None, {
            "reason": "final_route_critic_target_identity_missing",
            "route_family_id": route_id,
        }

    scoped_edges = {
        str(edge_id): dict(edges.get(str(edge_id)) or {})
        for edge_id in route_family_scoped_edge_ids(graph, family=route)
        if isinstance(edges.get(str(edge_id)), Mapping)
    }
    by_product: dict[str, list[dict[str, Any]]] = {}
    for edge in scoped_edges.values():
        product_id = str(edge.get("product_molecule_id") or "")
        if product_id:
            by_product.setdefault(product_id, []).append(edge)

    ordered_edges: list[dict[str, Any]] = []
    pending = deque([target_id])
    visited_molecules: set[str] = set()
    visited_edges: set[str] = set()
    while pending:
        molecule_id = pending.popleft()
        if molecule_id in visited_molecules:
            continue
        visited_molecules.add(molecule_id)
        for edge in sorted(
            by_product.get(molecule_id, ()),
            key=lambda row: str(row.get("edge_id") or ""),
        ):
            edge_id = str(edge.get("edge_id") or "")
            if not edge_id or edge_id in visited_edges:
                continue
            visited_edges.add(edge_id)
            ordered_edges.append(edge)
            pending.extend(
                str(value) for value in edge.get("precursor_molecule_ids") or () if str(value)
            )
    if not ordered_edges:
        return None, {
            "reason": "final_route_critic_target_rooted_route_missing",
            "route_family_id": route_id,
        }

    steps: list[dict[str, Any]] = []
    for edge in ordered_edges:
        edge_id = str(edge.get("edge_id") or "")
        audit = dict(edge.get("reactionjson_audit") or {})
        mapped_product = str(
            audit.get("mapped_product_smiles") or edge.get("mapped_product_smiles") or ""
        )
        mapped_precursors = mapped_reaction_input_smiles(edge)
        if not mapped_product or not mapped_precursors:
            proof_product, proof_precursors = _route_critic_edge_mapped_boundaries(edge)
            if proof_product and proof_precursors:
                mapped_product = proof_product
                mapped_precursors = proof_precursors
        precursor_ids = [str(value) for value in edge.get("precursor_molecule_ids") or ()]
        precursor_smiles = [
            _canonical_smiles(dict(molecules.get(precursor_id) or {}).get("canonical_smiles"))
            for precursor_id in precursor_ids
        ]
        complete_inputs = reaction_input_smiles(edge)
        mapped_by_identity: dict[str, list[str]] = {}
        for value in mapped_precursors:
            mapped_by_identity.setdefault(_canonical_smiles(value), []).append(value)
        aligned_mapped_precursors: list[str] = []
        for value in precursor_smiles:
            matches = mapped_by_identity.get(value) or []
            if matches:
                aligned_mapped_precursors.append(matches.pop(0))
        if (
            not _canonical_mapped_smiles(mapped_product)
            or _canonical_smiles(mapped_product) != _canonical_smiles(edge.get("product_smiles"))
            or not precursor_ids
            or any(not value for value in precursor_smiles)
            or len(aligned_mapped_precursors) != len(precursor_ids)
            or any(
                not _canonical_mapped_reactant_smiles(value) for value in aligned_mapped_precursors
            )
            or sorted(_canonical_smiles(value) for value in mapped_precursors)
            != sorted(_canonical_smiles(value) for value in complete_inputs)
            or any(not _canonical_mapped_reactant_smiles(value) for value in mapped_precursors)
        ):
            return None, {
                "reason": "final_route_critic_mapped_boundary_incomplete",
                "route_family_id": route_id,
                "edge_id": edge_id,
                "edge_ids": [str(value.get("edge_id") or "") for value in ordered_edges],
            }
        bound_origins = route_family_bound_origin_records(
            edge,
            route_family_id=route_id,
        )
        origin = bound_origins[0] if bound_origins else {}
        biocatalytic_steps = [
            dict(value)
            for value in edge.get("biocatalytic_steps") or ()
            if isinstance(value, Mapping)
        ]
        steps.append(
            {
                "step_id": str(origin.get("proposal_id") or edge_id),
                "product_smiles": _canonical_smiles(edge.get("product_smiles")),
                "precursor_smiles": precursor_smiles,
                "mapped_product_smiles": mapped_product,
                "mapped_precursor_smiles": aligned_mapped_precursors,
                "reaction_input_smiles": complete_inputs,
                "mapped_reaction_input_smiles": mapped_precursors,
                "auxiliary_reagent_smiles": [
                    _canonical_smiles(value)
                    for values in mapped_by_identity.values()
                    for value in values
                ],
                "mapped_auxiliary_reagent_smiles": [
                    value for values in mapped_by_identity.values() for value in values
                ],
                "reaction_operations": [
                    dict(value)
                    for value in edge.get("reaction_operations") or ()
                    if isinstance(value, Mapping)
                ],
                "reaction_family": str(
                    origin.get("reaction_family")
                    or origin.get("transformation_hypothesis")
                    or edge.get("transformation_hypothesis")
                    or ""
                ),
                "transformation_hypothesis": str(
                    origin.get("transformation_hypothesis")
                    or edge.get("transformation_hypothesis")
                    or ""
                ),
                "condition_predictions": origin_condition_predictions(edge, origin),
                "execution_domain": str(
                    origin.get("execution_domain") or edge.get("execution_domain") or "chemical"
                ),
                "strategy_anchor": origin.get("strategy_anchor") is True,
                "strategy_milestone_index": int(origin.get("strategy_milestone_index") or 1),
                "strategy_id": str(origin.get("strategy_id") or route.get("strategy_id") or ""),
                "strategy_digest": str(
                    origin.get("strategy_digest") or route.get("strategy_digest") or ""
                ),
                "biocatalytic_step": (biocatalytic_steps[0] if biocatalytic_steps else {}),
            }
        )

    # Canonical edges own reaction-local mappings.  A final Route Critic,
    # however, receives several connected edges at once and therefore needs
    # one route-level namespace.  Replaying here is the authority boundary:
    # it carries parent maps into child products and deterministically moves
    # fresh atoms introduced on sibling branches when their local numbers
    # collide.  Without this projection a harmless Cl:37 / Br:37 reuse on
    # separate materialized edges looks like an element-transmutation defect
    # to the Critic, and a child row can retain an operation map that no
    # longer matches its parent-produced intermediate.
    if all(row.get("reaction_operations") for row in steps):
        try:
            route_state = RouteJSONCompiler().compile_route_graph_state(
                mapped_target_smiles=str(steps[0].get("mapped_product_smiles") or ""),
                steps=steps,
                minimum_depth=1,
                rebase_materialized_local_maps=True,
            )
        except ReactionJsonReplayError as exc:
            return None, {
                "reason": "final_route_critic_route_namespace_not_replayable",
                "route_family_id": route_id,
                "edge_ids": [str(value.get("edge_id") or "") for value in ordered_edges],
                "compiler_error": str(exc),
            }
        steps = RouteJSONCompiler.assemble_route(
            route_state.reactions,
            metadata=steps,
        )

    strategy_card = dict(route.get("root_strategy_card") or route.get("strategy_card") or {})
    milestone_cards = tuple(
        dict(value)
        for value in route.get("strategy_milestone_cards") or ()
        if isinstance(value, Mapping)
    )
    selected_strategy_lineage = tuple(
        _selected_strategy_lineage_from_materialized_steps(
            root_strategy_card=strategy_card,
            strategy_milestone_cards=milestone_cards,
            steps=steps,
        )
    )
    branch_index = _route_branch_index(
        route,
        step_ids=(str(row.get("step_id") or "") for row in steps),
    )
    if branch_index is None:
        return None, {
            "reason": "final_route_critic_branch_lineage_missing",
            "route_family_id": route_id,
            "edge_ids": [str(value.get("edge_id") or "") for value in ordered_edges],
        }
    chemistry_input = {
        "schema_version": "revision_bound_route_critic_input.v2",
        "target_smiles": target_smiles,
        "route_family_id": route_id,
        "root_strategy_card": strategy_card,
        "selected_strategy_lineage": list(selected_strategy_lineage),
        "steps": steps,
    }
    route_sha256 = _digest(chemistry_input)
    return (
        RevisionBoundRouteCriticContext(
            target_smiles=target_smiles,
            route_family_id=route_id,
            route_sha256=route_sha256,
            graph_revision=int(graph.get("revision") or 0),
            branch_index=branch_index,
            edge_ids=tuple(str(edge.get("edge_id") or "") for edge in ordered_edges),
            steps=tuple(steps),
            strategy_card=strategy_card,
            selected_strategy_lineage=selected_strategy_lineage,
            strategy_milestone_cards=milestone_cards,
        ),
        {},
    )


def _route_branch_index(
    route: Mapping[str, Any],
    *,
    step_ids: Iterable[str] = (),
) -> int | None:
    """Recover one unambiguous Strategy branch as a zero-based index.

    Canonical materialization replaces the Director alias with a content
    identity, and a repair may append its own suffix.  New route families
    therefore persist ``strategy_branch_ids`` explicitly; historical graphs
    are recovered from aliases and materialized proposal ids.  Unknown or
    conflicting lineage must remain unavailable instead of silently becoming
    Branch 1.
    """

    branch_numbers: set[int] = set()
    invalid_lineage = False

    def remember(value: Any) -> None:
        nonlocal invalid_lineage
        try:
            number = int(value)
        except (TypeError, ValueError):
            invalid_lineage = True
            return
        if number < 1:
            invalid_lineage = True
            return
        branch_numbers.add(number)

    for value in route.get("strategy_branch_ids") or ():
        remember(value)
    if route.get("strategy_branch_id") not in (None, ""):
        remember(route.get("strategy_branch_id"))

    aliases = (
        route.get("route_family_id"),
        route.get("route_family_alias_override"),
        *(route.get("aliases") or ()),
    )
    for alias in (str(value) for value in aliases if str(value)):
        match = re.search(
            r"(?:^|:)codex:sequential:family:(\d+)(?::|$)",
            alias,
        )
        if match:
            remember(match.group(1))

    for step_id in (str(value) for value in step_ids if str(value)):
        match = re.search(r"(?:^|:)branch:(\d+)(?::|$)", step_id)
        if match:
            remember(match.group(1))

    if invalid_lineage or len(branch_numbers) != 1:
        return None
    return next(iter(branch_numbers)) - 1


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
