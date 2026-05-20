"""Training data construction for CascadeBoard++.

Builds masked route modeling + edit action + preference training data
from AutoPlanner's 2000 cascade routes.
"""
from __future__ import annotations

import json
import random
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cascade_planner.cascadeboard import CascadeBoard, Slot, EditType
from cascade_planner.cascadeboard.route_encoder import (
    REACTION_TYPE_TO_ID, OBJECTIVE_TO_ID, smiles_to_morgan_fp,
)


DIFFICULTY_EASY = 1
DIFFICULTY_MEDIUM = 2
DIFFICULTY_HARD = 3
DIFFICULTY_HARDEST = 4

CORRUPTION_DIFFICULTY = {
    "route_ok": DIFFICULTY_EASY,
    "mask_field": DIFFICULTY_EASY,
    "mask_step": DIFFICULTY_MEDIUM,
    "shift_T": DIFFICULTY_MEDIUM,
    "shift_pH": DIFFICULTY_MEDIUM,
    "replace_type": DIFFICULTY_MEDIUM,
    "replace_ec": DIFFICULTY_MEDIUM,
    "replace_enzyme": DIFFICULTY_MEDIUM,
    "mask_half": DIFFICULTY_HARD,
    "swap_order": DIFFICULTY_HARD,
    "delete_step": DIFFICULTY_HARD,
    "insert_extra": DIFFICULTY_HARD,
    "mask_all": DIFFICULTY_HARDEST,
}


@dataclass
class TrainingSample:
    """A single training sample for CascadeBoard++."""
    # Input: partially masked board
    target_smiles: str
    slots: list[dict]           # slot dicts with some fields masked
    mask_indices: list[int]     # which slots are masked
    mask_fields: list[str]      # which fields are masked
    true_slots: list[dict]      # ground truth slot values
    objective: str = "balanced"

    # Labels
    edit_action: str | None = None  # for L_edit
    edit_slot: int | None = None
    issue_indices: list[int] = field(default_factory=list)
    issue_fields: list[str] = field(default_factory=list)
    severity: float | None = None
    edit_action_valid: bool = True
    candidate_pools: dict[int, list[dict]] = field(default_factory=dict)
    candidate_labels: dict[int, int] = field(default_factory=dict)

    # v3 real labels — previously unused
    compatibility_label: str = ""          # 4-level: empirically_compatible / with_mitigation / with_compromise / unclear
    issue_types: list[str] = field(default_factory=list)  # real issue labels from v3
    mitigation_strategies: list[str] = field(default_factory=list)
    operation_mode: str = ""               # one_pot_simultaneous / sequential / etc.
    pairwise_modes: list[str] = field(default_factory=list)  # per adjacent step pair

    # Context for skeleton generation
    route_domain: str = ""                 # all_enzymatic / chemoenzymatic / all_chemical
    starting_material: str = ""            # starting material SMILES

    # Metadata
    source_doi: str = ""
    corruption_type: str = ""   # "mask" / "replace_type" / "replace_ec" / "shift_T" / ...
    difficulty: int = DIFFICULTY_EASY


def load_cascade_routes(data_path: str) -> list[dict]:
    """Load cascade routes from AutoPlanner JSON (v2 normalized or v3 flat list)."""
    data = json.loads(Path(data_path).read_text())
    routes = []

    # Detect format: v2 is {records_kept: [...]}, v3 is flat list of records
    if isinstance(data, dict):
        records = data.get("records_kept", [])
    elif isinstance(data, list):
        records = data
    else:
        return routes

    for art in records:
        doi = art.get("doi", "")
        for cascade in art.get("cascades", []):
            steps = cascade.get("steps", [])
            if len(steps) < 2:
                continue

            # Get target from first step's output or cascade target
            target_products = cascade.get("target_products", [])
            target_smi = ""
            if target_products and target_products[0]:
                tp = target_products[0]
                target_smi = tp.get("smiles", "") if isinstance(tp, dict) else str(tp)

            route = {
                "doi": doi,
                "target": target_smi,
                "route_domain": cascade.get("route_domain", ""),
                "operation_mode": cascade.get("operation_mode", ""),
                "steps": [],
                # v3 rich annotations — previously unused
                "compatibility_label": (cascade.get("compatibility_annotation") or {}).get("compatibility_label", ""),
                "issue_types": list((cascade.get("compatibility_annotation") or {}).get("issue_types") or []),
                "mitigation_strategies": list((cascade.get("compatibility_annotation") or {}).get("mitigation_strategies") or []),
                "recommended": bool((cascade.get("purpose_assessment") or {}).get("recommended_for_supervised_training", True)),
            }

            for s in steps:
                conds = s.get("step_conditions") or {}
                rxn = s.get("rxn_smiles") or ""

                # Extract product/reactant from rxn_smiles
                product_smi = ""
                reactant_smi = ""
                if rxn and ">>" in rxn:
                    parts = rxn.split(">>")
                    reactant_smi = parts[0].strip()
                    product_smi = parts[1].strip()

                # v3: try output_species / input_species if rxn_smiles missing
                if not product_smi:
                    for sp in (s.get("output_species") or []):
                        if isinstance(sp, dict) and sp.get("smiles"):
                            product_smi = sp["smiles"]
                            break
                if not reactant_smi:
                    for sp in (s.get("input_species") or []):
                        if isinstance(sp, dict) and sp.get("smiles") and sp.get("role_canonical") == "main_substrate":
                            reactant_smi = sp["smiles"]
                            break

                # Extract EC from catalyst_components
                ec = None
                enzyme_uid = None
                for cat in (s.get("catalyst_components") or []):
                    if cat and cat.get("ec_number"):
                        ec = cat["ec_number"]
                        enzyme_uid = cat.get("uniprot_id") or cat.get("component_name")
                        break

                step_info = {
                    "product": product_smi,
                    "reactant": reactant_smi,
                    "rxn_smiles": rxn,
                    "reaction_type": s.get("transformation_superclass", ""),
                    "transformation_name": s.get("transformation_name", ""),
                    "ec": ec,
                    "enzyme_uid": enzyme_uid,
                    "T": conds.get("temperature_c"),
                    "pH": conds.get("ph"),
                    "solvent": conds.get("solvent_smiles"),
                    "yield_pct": (s.get("step_outcome") or {}).get("step_yield_percent"),
                    "ee_pct": (s.get("step_outcome") or {}).get("step_ee_percent"),
                    "pairwise_mode": s.get("pairwise_mode", ""),
                    "reaction_time_h": conds.get("reaction_time_h"),
                    "cofactor_required": any(
                        (cat.get("cofactor_required") or False)
                        for cat in (s.get("catalyst_components") or []) if cat
                    ),
                    "organism": next(
                        (cat.get("organism", "") for cat in (s.get("catalyst_components") or [])
                         if cat and cat.get("organism")), ""
                    ),
                }
                route["steps"].append(step_info)

            # Use first step product as target if not set
            if not route["target"] and route["steps"]:
                route["target"] = route["steps"][0].get("product", "")

            if route["steps"] and any(s.get("rxn_smiles") or s.get("reaction_type") for s in route["steps"]):
                routes.append(route)

    return routes


def route_to_board(route: dict) -> CascadeBoard:
    """Convert a route dict to a CascadeBoard."""
    steps = route["steps"]
    board = CascadeBoard.from_n_steps(len(steps), route.get("target", ""))

    for i, step in enumerate(steps):
        slot = board.slots[i]
        slot.product = step.get("product", "")
        slot.main_reactant = step.get("reactant", "")
        slot.reaction_smiles = step.get("rxn_smiles", "")
        slot.reaction_type = step.get("reaction_type", "")
        slot.ec = step.get("ec")
        slot.enzyme_uid = step.get("enzyme_uid")
        slot.T = step.get("T")
        slot.pH = step.get("pH")
        slot.solvent = step.get("solvent")

    return board


# ---------------------------------------------------------------------------
# Mask strategies
# ---------------------------------------------------------------------------

def mask_single_field(board: CascadeBoard, rng: random.Random) -> TrainingSample:
    """Mask a single field from a random slot."""
    true_slots = [s.to_dict() for s in board.slots]
    masked = board.copy()

    slot_idx = rng.randint(0, masked.n_steps - 1)
    fields = ["reaction_type", "ec", "T", "pH", "solvent"]
    field = rng.choice(fields)

    original_val = getattr(masked.slots[slot_idx], field)
    setattr(masked.slots[slot_idx], field, None)

    return TrainingSample(
        target_smiles=board.slots[0].product or "",
        slots=[s.to_dict() for s in masked.slots],
        mask_indices=[slot_idx],
        mask_fields=[field],
        true_slots=true_slots,
        issue_indices=[slot_idx],
        issue_fields=[field],
        corruption_type="mask_field",
        source_doi="",
        difficulty=DIFFICULTY_EASY,
    )


def mask_full_step(board: CascadeBoard, rng: random.Random) -> TrainingSample:
    """Mask an entire step."""
    true_slots = [s.to_dict() for s in board.slots]
    masked = board.copy()

    slot_idx = rng.randint(0, masked.n_steps - 1)
    slot = masked.slots[slot_idx]
    for field in ["reaction_type", "ec", "T", "pH", "solvent", "reaction_smiles"]:
        setattr(slot, field, None)

    return TrainingSample(
        target_smiles=board.slots[0].product or "",
        slots=[s.to_dict() for s in masked.slots],
        mask_indices=[slot_idx],
        mask_fields=["reaction_type", "ec", "T", "pH", "solvent"],
        true_slots=true_slots,
        issue_indices=[slot_idx],
        issue_fields=["reaction_type", "ec", "T", "pH", "solvent"],
        corruption_type="mask_step",
        difficulty=DIFFICULTY_MEDIUM,
    )


def mask_half(board: CascadeBoard, rng: random.Random) -> TrainingSample:
    """Mask ~50% of all fields."""
    true_slots = [s.to_dict() for s in board.slots]
    masked = board.copy()
    mask_indices = []
    mask_fields_all = []

    for i, slot in enumerate(masked.slots):
        fields = ["reaction_type", "ec", "T", "pH", "solvent"]
        to_mask = rng.sample(fields, k=rng.randint(1, len(fields)))
        for f in to_mask:
            setattr(slot, f, None)
        if to_mask:
            mask_indices.append(i)
            mask_fields_all.extend(to_mask)

    return TrainingSample(
        target_smiles=board.slots[0].product or "",
        slots=[s.to_dict() for s in masked.slots],
        mask_indices=mask_indices,
        mask_fields=mask_fields_all,
        true_slots=true_slots,
        issue_indices=mask_indices,
        issue_fields=mask_fields_all,
        corruption_type="mask_half",
        difficulty=DIFFICULTY_HARD,
    )


def mask_all(board: CascadeBoard, rng: random.Random) -> TrainingSample:
    """Mask everything (from-scratch planning)."""
    true_slots = [s.to_dict() for s in board.slots]
    masked = board.copy()

    for slot in masked.slots:
        for f in ["reaction_type", "ec", "T", "pH", "solvent", "reaction_smiles"]:
            setattr(slot, f, None)

    return TrainingSample(
        target_smiles=board.slots[0].product or "",
        slots=[s.to_dict() for s in masked.slots],
        mask_indices=list(range(masked.n_steps)),
        mask_fields=["reaction_type", "ec", "T", "pH", "solvent"],
        true_slots=true_slots,
        issue_indices=list(range(masked.n_steps)),
        issue_fields=["reaction_type", "ec", "T", "pH", "solvent"],
        corruption_type="mask_all",
        difficulty=DIFFICULTY_HARDEST,
    )


def route_ok(board: CascadeBoard, rng: random.Random) -> TrainingSample:
    """Clean route sample used for energy/feasibility/issue calibration.

    There is intentionally no edit-action label: the current EditType enum has
    no NO_OP class, so edit loss is masked for this sample instead of pretending
    that a clean route should trigger FILL_FIELD.
    """
    true_slots = [s.to_dict() for s in board.slots]
    return TrainingSample(
        target_smiles=board.slots[0].product or "",
        slots=true_slots,
        mask_indices=[],
        mask_fields=[],
        true_slots=true_slots,
        issue_indices=[],
        issue_fields=[],
        severity=0.0,
        edit_action=None,
        edit_slot=None,
        edit_action_valid=False,
        corruption_type="route_ok",
        difficulty=DIFFICULTY_EASY,
    )


# ---------------------------------------------------------------------------
# Corruption strategies (for edit action training)
# ---------------------------------------------------------------------------

CORRUPTION_TYPES = [
    "replace_type", "replace_ec", "shift_T", "shift_pH", "swap_order",
    "replace_enzyme", "delete_step", "insert_extra",
]


def corrupt_replace_type(board: CascadeBoard, rng: random.Random) -> TrainingSample:
    """Replace a reaction type with a wrong one."""
    true_slots = [s.to_dict() for s in board.slots]
    corrupted = board.copy()

    slot_idx = rng.randint(0, corrupted.n_steps - 1)
    original = corrupted.slots[slot_idx].reaction_type
    wrong_types = [t for t in REACTION_TYPE_TO_ID if t and t != original]
    if wrong_types:
        corrupted.slots[slot_idx].reaction_type = rng.choice(wrong_types)

    return TrainingSample(
        target_smiles=board.slots[0].product or "",
        slots=[s.to_dict() for s in corrupted.slots],
        mask_indices=[slot_idx],
        mask_fields=["reaction_type"],
        true_slots=true_slots,
        edit_action="REPLACE_STEP",
        edit_slot=slot_idx,
        issue_indices=[slot_idx],
        issue_fields=["reaction_type"],
        corruption_type="replace_type",
        difficulty=DIFFICULTY_MEDIUM,
    )


def corrupt_shift_T(board: CascadeBoard, rng: random.Random) -> TrainingSample:
    """Shift temperature by a large amount."""
    true_slots = [s.to_dict() for s in board.slots]
    corrupted = board.copy()

    candidates = [i for i, s in enumerate(corrupted.slots) if s.T is not None]
    if not candidates:
        return mask_single_field(board, rng)

    slot_idx = rng.choice(candidates)
    shift = rng.choice([-30, -20, 20, 30])
    corrupted.slots[slot_idx].T = max(0, corrupted.slots[slot_idx].T + shift)

    return TrainingSample(
        target_smiles=board.slots[0].product or "",
        slots=[s.to_dict() for s in corrupted.slots],
        mask_indices=[slot_idx],
        mask_fields=["T"],
        true_slots=true_slots,
        edit_action="ADJUST_CONDITION",
        edit_slot=slot_idx,
        issue_indices=[slot_idx],
        issue_fields=["T"],
        corruption_type="shift_T",
        difficulty=DIFFICULTY_MEDIUM,
    )


def corrupt_shift_pH(board: CascadeBoard, rng: random.Random) -> TrainingSample:
    """Shift pH by a large amount."""
    true_slots = [s.to_dict() for s in board.slots]
    corrupted = board.copy()

    candidates = [i for i, s in enumerate(corrupted.slots) if s.pH is not None]
    if not candidates:
        return mask_single_field(board, rng)

    slot_idx = rng.choice(candidates)
    shift = rng.choice([-3, -2, 2, 3])
    corrupted.slots[slot_idx].pH = max(1, min(14, corrupted.slots[slot_idx].pH + shift))

    return TrainingSample(
        target_smiles=board.slots[0].product or "",
        slots=[s.to_dict() for s in corrupted.slots],
        mask_indices=[slot_idx],
        mask_fields=["pH"],
        true_slots=true_slots,
        edit_action="ADJUST_CONDITION",
        edit_slot=slot_idx,
        issue_indices=[slot_idx],
        issue_fields=["pH"],
        corruption_type="shift_pH",
        difficulty=DIFFICULTY_MEDIUM,
    )


def corrupt_replace_ec(board: CascadeBoard, rng: random.Random) -> TrainingSample:
    """Replace EC with a wrong one from a different class."""
    true_slots = [s.to_dict() for s in board.slots]
    corrupted = board.copy()
    candidates = [i for i, s in enumerate(corrupted.slots) if s.ec]
    if not candidates:
        return mask_single_field(board, rng)
    slot_idx = rng.choice(candidates)
    original_ec1 = corrupted.slots[slot_idx].ec.split(".")[0]
    wrong_ec1 = rng.choice([str(x) for x in range(1, 7) if str(x) != original_ec1])
    corrupted.slots[slot_idx].ec = f"{wrong_ec1}.1.1.1"
    return TrainingSample(
        target_smiles=board.slots[0].product or "",
        slots=[s.to_dict() for s in corrupted.slots],
        mask_indices=[slot_idx], mask_fields=["ec"],
        true_slots=true_slots,
        edit_action="REPLACE_ENZYME", edit_slot=slot_idx,
        issue_indices=[slot_idx], issue_fields=["ec"],
        corruption_type="replace_ec",
        difficulty=DIFFICULTY_MEDIUM,
    )


def corrupt_replace_enzyme(board: CascadeBoard, rng: random.Random) -> TrainingSample:
    """Replace enzyme identity while preserving the EC class when possible."""
    true_slots = [s.to_dict() for s in board.slots]
    corrupted = board.copy()
    candidates = [i for i, s in enumerate(corrupted.slots) if s.enzyme_uid or s.ec]
    if not candidates:
        return corrupt_replace_ec(board, rng)

    slot_idx = rng.choice(candidates)
    slot = corrupted.slots[slot_idx]
    original_uid = slot.enzyme_uid
    enzyme_pool = sorted({
        s.enzyme_uid for s in board.slots
        if s.enzyme_uid and s.enzyme_uid != original_uid
    })
    if enzyme_pool:
        slot.enzyme_uid = rng.choice(enzyme_pool)
    elif slot.ec:
        original_ec1 = slot.ec.split(".")[0]
        wrong_ec1 = rng.choice([str(x) for x in range(1, 7) if str(x) != original_ec1])
        slot.ec = f"{wrong_ec1}.1.1.1"
        slot.enzyme_uid = f"wrong_ec1_{wrong_ec1}"
    else:
        slot.enzyme_uid = "wrong_enzyme_unknown"

    return TrainingSample(
        target_smiles=board.slots[0].product or "",
        slots=[s.to_dict() for s in corrupted.slots],
        mask_indices=[slot_idx],
        mask_fields=["enzyme_uid"],
        true_slots=true_slots,
        edit_action="REPLACE_ENZYME",
        edit_slot=slot_idx,
        issue_indices=[slot_idx],
        issue_fields=["enzyme_uid"],
        corruption_type="replace_enzyme",
        difficulty=DIFFICULTY_MEDIUM,
    )


def corrupt_swap_order(board: CascadeBoard, rng: random.Random) -> TrainingSample:
    """Swap two adjacent steps."""
    true_slots = [s.to_dict() for s in board.slots]
    corrupted = board.copy()
    if corrupted.n_steps < 2:
        return mask_single_field(board, rng)
    idx = rng.randint(0, corrupted.n_steps - 2)
    corrupted.slots[idx], corrupted.slots[idx + 1] = corrupted.slots[idx + 1], corrupted.slots[idx]
    corrupted.slots[idx].index = idx
    corrupted.slots[idx + 1].index = idx + 1
    return TrainingSample(
        target_smiles=board.slots[0].product or "",
        slots=[s.to_dict() for s in corrupted.slots],
        mask_indices=[idx, idx + 1], mask_fields=["reaction_type", "ec", "T", "pH"],
        true_slots=true_slots,
        edit_action="SWAP_ORDER", edit_slot=idx,
        issue_indices=[idx, idx + 1], issue_fields=["order"],
        corruption_type="swap_order",
        difficulty=DIFFICULTY_HARD,
    )


def corrupt_delete_step(board: CascadeBoard, rng: random.Random) -> TrainingSample:
    """Remove a step (label = INSERT_STEP to repair)."""
    true_slots = [s.to_dict() for s in board.slots]
    if board.n_steps < 3:
        return mask_single_field(board, rng)
    corrupted = board.copy()
    idx = rng.randint(1, corrupted.n_steps - 2)  # don't delete first or last
    corrupted.slots.pop(idx)
    for i, s in enumerate(corrupted.slots):
        s.index = i
    return TrainingSample(
        target_smiles=board.slots[0].product or "",
        slots=[s.to_dict() for s in corrupted.slots],
        mask_indices=list(range(len(corrupted.slots))),
        mask_fields=["reaction_type", "ec", "T", "pH"],
        true_slots=true_slots,
        edit_action="INSERT_STEP", edit_slot=idx,
        issue_indices=[min(idx, len(corrupted.slots) - 1)],
        issue_fields=["missing_step"],
        corruption_type="delete_step",
        difficulty=DIFFICULTY_HARD,
    )


def corrupt_insert_extra(board: CascadeBoard, rng: random.Random) -> TrainingSample:
    """Insert a random extra step (label = DELETE_STEP to repair)."""
    true_slots = [s.to_dict() for s in board.slots]
    corrupted = board.copy()
    idx = rng.randint(0, corrupted.n_steps)
    extra = Slot(index=idx)
    extra.reaction_type = rng.choice(["oxidation", "reduction", "hydrolysis"])
    extra.T = rng.uniform(20, 60)
    extra.pH = rng.uniform(5, 9)
    corrupted.slots.insert(idx, extra)
    for i, s in enumerate(corrupted.slots):
        s.index = i
    return TrainingSample(
        target_smiles=board.slots[0].product or "",
        slots=[s.to_dict() for s in corrupted.slots],
        mask_indices=list(range(len(corrupted.slots))),
        mask_fields=["reaction_type", "ec", "T", "pH"],
        true_slots=true_slots,
        edit_action="DELETE_STEP", edit_slot=idx,
        issue_indices=[idx],
        issue_fields=["extra_step"],
        corruption_type="insert_extra",
        difficulty=DIFFICULTY_HARD,
    )


# ---------------------------------------------------------------------------
# Build full training dataset
# ---------------------------------------------------------------------------

def _attach_route_labels(sample: TrainingSample, route: dict) -> None:
    """Attach v3 rich annotations from route dict to a TrainingSample."""
    sample.source_doi = route.get("doi", "")
    sample.compatibility_label = route.get("compatibility_label", "")
    sample.issue_types = route.get("issue_types", [])
    sample.mitigation_strategies = route.get("mitigation_strategies", [])
    sample.operation_mode = route.get("operation_mode", "")
    sample.route_domain = route.get("route_domain", "")
    # Starting material: first SM SMILES from route steps (last step's reactant)
    steps = route.get("steps", [])
    if steps:
        last_step = steps[-1]
        sample.starting_material = last_step.get("reactant", "") or ""
    # Collect pairwise_mode for each adjacent step pair
    sample.pairwise_modes = [
        steps[i + 1].get("pairwise_mode", "") for i in range(len(steps) - 1)
    ]


def build_training_data(
    routes: list[dict],
    n_mask_per_route: int = 10,
    n_corrupt_per_route: int = 5,
    seed: int = 42,
    filter_recommended: bool = False,
) -> list[TrainingSample]:
    """Build training dataset from cascade routes.

    Returns ~(n_mask + n_corrupt) * len(routes) samples.
    If filter_recommended=True, only use routes with recommended_for_supervised_training=True.
    """
    rng = random.Random(seed)
    samples: list[TrainingSample] = []

    mask_fns = [mask_single_field, mask_full_step, mask_half, mask_all]
    corrupt_fns = [
        corrupt_replace_type, corrupt_shift_T, corrupt_shift_pH,
        corrupt_replace_ec, corrupt_replace_enzyme, corrupt_swap_order,
        corrupt_delete_step, corrupt_delete_step, corrupt_insert_extra,
    ]

    for route in routes:
        if filter_recommended and not route.get("recommended", True):
            continue

        board = route_to_board(route)
        if board.n_steps < 2:
            continue
        if smiles_to_morgan_fp(board.slots[0].product).sum() == 0:
            continue

        clean = route_ok(board, rng)
        _attach_route_labels(clean, route)
        samples.append(clean)

        # Mask samples
        for _ in range(n_mask_per_route):
            fn = rng.choice(mask_fns)
            sample = fn(board, rng)
            _attach_route_labels(sample, route)
            samples.append(sample)

        # Corruption samples
        for _ in range(n_corrupt_per_route):
            fn = rng.choice(corrupt_fns)
            sample = fn(board, rng)
            _attach_route_labels(sample, route)
            samples.append(sample)

    rng.shuffle(samples)
    return samples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build CascadeBoard++ training data")
    ap.add_argument("--data", default="cascade_dataset_v2.normalized.json")
    ap.add_argument("--output", default="results/shared/cascadeboard_training_data.json")
    ap.add_argument("--n-mask", type=int, default=10)
    ap.add_argument("--n-corrupt", type=int, default=5)
    args = ap.parse_args()

    print(f"Loading routes from {args.data}...")
    routes = load_cascade_routes(args.data)
    print(f"  {len(routes)} cascade routes loaded")

    print(f"Building training data (mask={args.n_mask}, corrupt={args.n_corrupt} per route)...")
    samples = build_training_data(routes, args.n_mask, args.n_corrupt)
    print(f"  {len(samples)} training samples generated")

    # Stats
    from collections import Counter
    types = Counter(s.corruption_type for s in samples)
    print(f"  Types: {dict(types)}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = [
        {
            "target": s.target_smiles,
            "slots": s.slots,
            "mask_indices": s.mask_indices,
            "mask_fields": s.mask_fields,
            "true_slots": s.true_slots,
            "edit_action": s.edit_action,
            "edit_slot": s.edit_slot,
            "corruption_type": s.corruption_type,
            "doi": s.source_doi,
        }
        for s in samples
    ]
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"  Saved to {out_path} ({out_path.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    main()
