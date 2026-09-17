"""Read reaction participants independently of the route-expansion boundary.

New rows carry complete reaction inputs and a separate route precursor subset.
Legacy rows use the replay audit's complete precursor list, or the original
precursor list when no component split exists.  No chemistry is inferred here.
"""

from typing import Any, Mapping


def reaction_input_smiles(row: Mapping[str, Any]) -> list[str]:
    audit = dict(row.get("reactionjson_audit") or {})
    return [
        str(value)
        for value in (
            row.get("reaction_input_smiles")
            or audit.get("reaction_input_smiles")
            or audit.get("precursor_smiles")
            or row.get("reactant_smiles")
            or row.get("precursor_smiles")
            or ()
        )
    ]


def mapped_reaction_input_smiles(row: Mapping[str, Any]) -> list[str]:
    audit = dict(row.get("reactionjson_audit") or {})
    return [
        str(value)
        for value in (
            row.get("mapped_reaction_input_smiles")
            or audit.get("mapped_reaction_input_smiles")
            or audit.get("mapped_precursor_smiles")
            or row.get("mapped_precursor_smiles")
            or ()
        )
    ]


def forward_reaction_smiles(row: Mapping[str, Any]) -> str:
    return ".".join(reaction_input_smiles(row)) + ">>" + str(row.get("product_smiles") or "")


def reaction_input_context(row: Mapping[str, Any], *, mapped_only: bool = False) -> dict[str, Any]:
    """Preserve the complete forward boundary in compact review prompts."""
    complete = reaction_input_smiles(row)
    if sorted(complete) == sorted(row.get("precursor_smiles") or ()):
        return {}
    audit = dict(row.get("reactionjson_audit") or {})
    result = {
        "mapped_reaction_input_smiles": mapped_reaction_input_smiles(row),
        "mapped_auxiliary_reagent_smiles": list(
            row.get("mapped_auxiliary_reagent_smiles")
            or audit.get("mapped_auxiliary_reagent_smiles")
            or ()
        ),
    }
    if not mapped_only:
        result["reaction_input_smiles"] = complete
        result["auxiliary_reagent_smiles"] = list(
            row.get("auxiliary_reagent_smiles") or audit.get("auxiliary_reagent_smiles") or ()
        )
    return result
