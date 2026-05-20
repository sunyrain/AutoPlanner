"""Loader v2: same JSON, richer extraction.

Adds:
  - StepRow: temperature_c, ph, solvent_smiles (from step OR global fallback),
             transformation_superclass.
  - StepPairRow: consecutive-step pairs within a cascade (for pairwise_mode
    and compatibility prediction with both reaction contexts).
  - Helpers to derive new task labels (transformation_superclass top-K,
    EC1 first-digit, T regression).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


# ---------------- helpers ----------------

def _rxn_valid(rxn: str | None) -> bool:
    if not rxn or ">>" not in rxn:
        return False
    lhs, rhs = rxn.split(">>", 1)
    if not lhs.strip() or not rhs.strip():
        return False
    for side in (lhs, rhs):
        for s in side.split("."):
            if Chem.MolFromSmiles(s) is None:
                return False
    return True


def _coalesce(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


# Cofactor SMILES (canonical) whose presence as MAIN PRODUCT signals a
# half-reaction / cofactor pseudo-step that should be dropped. The same
# set is used to strip cofactors from reactant lists during downstream
# fingerprinting. Kept conservative: only molecules that are universally
# cofactors, never real substrates.
_COFACTOR_PRODUCT_SMI: frozenset[str] = frozenset({
    "O", "[H]O[H]",                        # water
    "O=O",                                  # O2
    "OO",                                   # H2O2
    "O=C=O",                                # CO2
    "[O-]O",                                # superoxide
    "[H][H]",                               # H2
    "N",                                    # ammonia (often pseudo)
    "[NH4+]",
    "O=P(O)(O)O",                           # phosphate
    "O=P(O)(O)OP(=O)(O)O",                  # diphosphate
})


def _canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def _largest_ha(smi_dot: str) -> str | None:
    best, best_n = None, -1
    for frag in smi_dot.split("."):
        m = Chem.MolFromSmiles(frag)
        if m is None:
            continue
        n = m.GetNumHeavyAtoms()
        if n > best_n:
            best, best_n = frag, n
    return best


def is_cofactor_pseudo_step(rxn_smiles: str) -> bool:
    """Return True iff this step's main product canonicalises to a cofactor SMILES."""
    if not rxn_smiles or ">>" not in rxn_smiles:
        return False
    _, rhs = rxn_smiles.split(">>", 1)
    main = _largest_ha(rhs)
    if main is None:
        return False
    c = _canon(main)
    if c is None:
        return False
    return c in _COFACTOR_PRODUCT_SMI


# ---------------- data classes ----------------

@dataclass
class StepRowV2:
    doi: str
    cascade_id: str
    step_id: str
    step_index: int
    rxn_smiles: str
    pairwise_mode: str
    transformation_superclass: str | None
    ec_number: str | None
    catalyst_class: str | None
    # conditions resolved (step → global fallback)
    temperature_c: float | None
    ph: float | None
    solvent_smiles: str | None


@dataclass
class StepPairRowV2:
    doi: str
    cascade_id: str
    pair_index: int
    rxn_smiles_a: str
    rxn_smiles_b: str
    # pairwise_mode is annotated on the SECOND step in our schema (each step.pairwise_mode
    # describes how it relates to the previous step in the same one-pot cascade).
    pairwise_mode: str
    cascade_compatibility_label: str | None
    cascade_issue_types: list[str]
    cascade_mitigation: list[str]
    # conditions for both sides
    t_a: float | None
    ph_a: float | None
    solv_a: str | None
    t_b: float | None
    ph_b: float | None
    solv_b: str | None
    # auxiliary side annotations (for multi-task heads)
    transformation_a: str | None = None
    transformation_b: str | None = None
    catalyst_class_a: str | None = None
    catalyst_class_b: str | None = None


@dataclass
class CascadeRowV2:
    doi: str
    cascade_id: str
    n_steps: int
    rxn_smiles_list: list[str]
    operation_mode: str | None
    route_domain: str | None
    compatibility_label: str | None
    issue_types: list[str]
    mitigation_strategies: list[str]
    evidence_strength: str | None
    # cascade-level avg conditions (mean of valid step values)
    avg_temperature_c: float | None
    avg_ph: float | None
    solvent_smiles_first: str | None


def _step_T_pH_solv(step: dict, cascade: dict) -> tuple[float | None, float | None, str | None]:
    sc = step.get("step_conditions") or {}
    cr = step.get("conditions_resolved") or {}
    gc = cascade.get("global_conditions") or {}
    T = _coalesce(cr.get("temperature_c"), sc.get("temperature_c"), gc.get("temperature_c"))
    pH = _coalesce(cr.get("ph"), sc.get("ph"), gc.get("ph"))
    solv = _coalesce(sc.get("solvent_smiles"), gc.get("solvent_smiles"))
    def _to_float(v):
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            # Map common qualitative labels; otherwise drop.
            if isinstance(v, str):
                s = v.strip().lower()
                if s in ("room temperature", "rt", "ambient", "ambient temperature"):
                    return 25.0
                if s in ("on ice", "ice bath", "ice"):
                    return 4.0
            return None
    return (
        _to_float(T),
        _to_float(pH),
        solv if isinstance(solv, str) and solv else None,
    )


def load_v2(
    path: str | Path,
    trainable_only: bool = True,
    drop_cofactor_products: bool = False,
) -> tuple[list[StepRowV2], list[StepPairRowV2], list[CascadeRowV2]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    step_rows: list[StepRowV2] = []
    pair_rows: list[StepPairRowV2] = []
    cascade_rows: list[CascadeRowV2] = []

    # Support both v2 (dict with records_kept) and v3 (flat list) formats
    if isinstance(data, list):
        records = data
    else:
        records = data.get("records_kept", [])

    for art in records:
        doi = art.get("doi") or art.get("title", "unknown")
        for c in art.get("cascades", []):
            pa = c.get("purpose_assessment") or {}
            if trainable_only and not pa.get("recommended_for_supervised_training"):
                continue

            steps = c.get("steps", []) or []
            ca = c.get("compatibility_annotation") or {}
            cascade_label = ca.get("compatibility_label")
            cascade_issues = list(ca.get("issue_types") or [])
            cascade_mit = list(ca.get("mitigation_strategies") or [])

            rxn_list: list[str] = []
            cached = []  # (rxn, T, pH, solv, pairwise_mode)
            T_vals, pH_vals = [], []
            first_solv = None

            for s in steps:
                rxn = (s.get("rxn_smiles") or "").strip()
                ok = _rxn_valid(rxn)
                rxn_list.append(rxn if ok else "")
                T, pH, solv = _step_T_pH_solv(s, c)
                pm = (s.get("pairwise_mode") or "not_applicable").strip() or "not_applicable"
                if T is not None:
                    T_vals.append(T)
                if pH is not None:
                    pH_vals.append(pH)
                if first_solv is None and solv:
                    first_solv = solv

                if ok:
                    cats = s.get("catalyst_components") or []
                    cat_class = cats[0].get("catalyst_class") if cats else None
                    ec = cats[0].get("ec_number") if cats else None
                    transf = s.get("transformation_superclass")
                    if drop_cofactor_products and is_cofactor_pseudo_step(rxn):
                        cached.append(None)
                        continue
                    step_rows.append(
                        StepRowV2(
                            doi=doi,
                            cascade_id=c.get("cascade_id", ""),
                            step_id=s.get("step_id", ""),
                            step_index=int(s.get("step_index", 0)),
                            rxn_smiles=rxn,
                            pairwise_mode=pm,
                            transformation_superclass=transf,
                            ec_number=ec,
                            catalyst_class=cat_class,
                            temperature_c=T,
                            ph=pH,
                            solvent_smiles=solv,
                        )
                    )
                    cached.append((rxn, T, pH, solv, pm, transf, cat_class))
                else:
                    cached.append(None)

            # pairs (consecutive steps both with valid rxn)
            for i in range(len(cached) - 1):
                a, b = cached[i], cached[i + 1]
                if a is None or b is None:
                    continue
                pair_rows.append(
                    StepPairRowV2(
                        doi=doi,
                        cascade_id=c.get("cascade_id", ""),
                        pair_index=i,
                        rxn_smiles_a=a[0],
                        rxn_smiles_b=b[0],
                        pairwise_mode=b[4],  # second step's pairwise_mode label
                        cascade_compatibility_label=cascade_label,
                        cascade_issue_types=cascade_issues,
                        cascade_mitigation=cascade_mit,
                        t_a=a[1], ph_a=a[2], solv_a=a[3],
                        t_b=b[1], ph_b=b[2], solv_b=b[3],
                        transformation_a=a[5], transformation_b=b[5],
                        catalyst_class_a=a[6], catalyst_class_b=b[6],
                    )
                )

            cascade_rows.append(
                CascadeRowV2(
                    doi=doi,
                    cascade_id=c.get("cascade_id", ""),
                    n_steps=len(steps),
                    rxn_smiles_list=rxn_list,
                    operation_mode=c.get("operation_mode"),
                    route_domain=c.get("route_domain"),
                    compatibility_label=cascade_label,
                    issue_types=cascade_issues,
                    mitigation_strategies=cascade_mit,
                    evidence_strength=ca.get("evidence_strength"),
                    avg_temperature_c=(sum(T_vals) / len(T_vals)) if T_vals else None,
                    avg_ph=(sum(pH_vals) / len(pH_vals)) if pH_vals else None,
                    solvent_smiles_first=first_solv,
                )
            )

    return step_rows, pair_rows, cascade_rows


if __name__ == "__main__":
    import collections
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "cascade_dataset_v2.normalized.json"
    sr, pr, cr = load_v2(path)
    print(f"steps : {len(sr)}   pairs : {len(pr)}   cascades : {len(cr)}")
    print(f"DOIs : {len({c.doi for c in cr})}")
    print(f"steps with T : {sum(1 for s in sr if s.temperature_c is not None)}")
    print(f"steps with pH: {sum(1 for s in sr if s.ph is not None)}")
    print(f"steps with solvent: {sum(1 for s in sr if s.solvent_smiles)}")
    print("\ntransformation_superclass top-15:")
    for k, v in collections.Counter(s.transformation_superclass for s in sr if s.transformation_superclass).most_common(15):
        print(f"  {k:35s} {v}")
    print("\nec1 (first digit) per-step where ec_number exists:")
    ec1 = [s.ec_number.split(".")[0] for s in sr if s.ec_number]
    for k, v in collections.Counter(ec1).most_common():
        print(f"  EC{k:3s} {v}")
    print(f"\npair pairwise_mode (second step):")
    for k, v in collections.Counter(p.pairwise_mode for p in pr).most_common():
        print(f"  {k:30s} {v}")
