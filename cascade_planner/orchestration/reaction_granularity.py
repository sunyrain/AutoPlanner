"""Shared chemical planning unit; graph replay remains the structure authority.

ReactionJSON is a net retrosynthetic edit program, not laboratory chronology.
Internal stages live in the existing condition hypothesis, not a second graph.
"""

from cascade_planner.orchestration.chemical_reasoning import CATALYSIS_PLANNING_GUIDANCE

TRANSFORMATION_GUIDANCE = (
    "Use one chemically meaningful synthetic transformation per reaction edge: "
    "one principal synthetic objective with its enabling activation and finishing workup. "
    "Do not create route nodes solely for transient reactive states, proton transfers, "
    "a reagent change, or a change of vessel. Enolate generation/addition/protonation, "
    "in situ organometallic preparation/addition/workup, acyl activation/capture, and "
    "deprotection/neutralization may belong to one transformation when their sequence is coherent. "
    "Keep independently useful preparative transformations separate: protection before a coupling, "
    "oxidation before olefination, or a radical-precursor preparation before deoxygenation "
    "remain separate transformations even if telescoped experimentally. A one-pot label, "
    "a short route, or a named cascade does not justify hiding independent synthetic objectives. "
    "A coherent cascade can have several bond changes; encode its complete net transformation. "
    "An isolated or externally supplied intermediate may be an explicit boundary; "
    "never imply that its upstream preparation has been solved."
)

STAGE_GUIDANCE = (
    "ReactionJSON operations are ordered retro graph edits, not mechanistic intermediates "
    "or forward reagent-addition order. They must derive the actual input and delivered "
    "product structures, including covalent handles, atom sources and required stereochemistry. "
    "Retain product-atom donors in the replayed inputs: disconnect and complete the donor "
    "rather than remove it and mention it only in conditions (for O-silylation, break O-Si "
    "and add Cl to Si to supply TMSCl). The Host separates auxiliary reagents from route precursors. "
    "Within each condition hypothesis, state the forward order of any necessary activation, "
    "main event and quench/workup, including removal of incompatible carryover or separate "
    "reagent preparation when required. A transient intermediate need not be a route node, "
    "but its reactivity, geometry, selectivity and compatibility must still be supported. "
    "Do not mix mutually incompatible reagents simultaneously or use a condition note to "
    "hide an unencoded independent protection, redox or skeletal transformation. "
    "Different condition hypotheses are alternatives, not successive stages. "
    "For each forward implementation, identify the reagents and catalyst/ligand or enzyme system, "
    "the solvent or reaction medium, and the basis of stereochemical control when required. "
    "State only operating choices that determine feasibility, selectivity, compatibility or the "
    "task objective: relevant temperature, reagent form/loading, medium or pH, addition order, "
    "atmosphere/pressure, method-specific driver, and necessary workup or solvent exchange. "
    "Planning does not require a laboratory SOP: qualitative choices or plausible proposed ranges "
    "are sufficient unless a quantitative boundary is essential to the chemical judgment. "
    "For example, identify a dry feed before a water-sensitive consumer and a cooled, controlled "
    "oxidant addition when needed; do not invent validated moisture limits, feed rates or calorimetry. "
    "Missing exact concentration, hydration, time or rate alone does not justify uncertain or "
    "another call. Identify the unresolved choice and its consequence. Experimental optimization "
    "and measured performance remain evidence dependencies. "
    + CATALYSIS_PLANNING_GUIDANCE
)

BUILDER_GRANULARITY_GUIDANCE = " ".join((
    TRANSFORMATION_GUIDANCE,
    STAGE_GUIDANCE,
    "Privately challenge and replay the complete net edit against selected_leaf_mapped. "
    "Use the Host's maps; the Host derives both endpoints. Choose accessible, meaningful "
    "precursor boundaries without turning routine activation or workup into an unnecessary search problem.",
))

CRITIC_GRANULARITY_GUIDANCE = " ".join((
    TRANSFORMATION_GUIDANCE,
    STAGE_GUIDANCE,
    "Assess every necessary internal stage even when it has no separate route node. "
    "Reject a concrete structural, selectivity, reagent-compatibility or sequence contradiction; "
    "a missing decisive stage choice can be uncertain. Do not reject merely because activation, "
    "protonation, hydrolysis or neutralization accompanies a coherent transformation. "
    "If a chemically plausible row bundles independent synthetic objectives, identify the "
    "needed split and actual synthetic burden in the condition assessment or revision advice; "
    "a counting convention alone is not chemical failure. Do not claim stock closure or "
    "experimental validation from grouping."
))
