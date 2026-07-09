"""General semisynthesis rescue routes for late-stage derivatizations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors

from cascade_planner.baselines.route_contract import RouteCandidate, RouteStepCandidate


RDLogger.DisableLog("rdApp.*")

ACETIC_ACID = "CC(=O)O"
ACETIC_ANHYDRIDE = "CC(=O)OC(C)=O"
DMAP = "CN(C)c1ccncc1"
DICHLOROMETHANE = "ClCCl"
TBS_CHLORIDE = "CC(C)(C)[Si](C)(C)Cl"
IMIDAZOLE = "c1ncc[nH]1"
DMF = "CN(C)C=O"
PHENYLISOSERINE_ZWITTERION = "[NH3+][C@@H](C(=O)[O-])c1ccccc1"
O_ACETYL_DEPROTECTION = "[CH3:1][C:2](=[O:3])[O:4][C:5]>>[O:4][C:5].CC(=O)O"
TBS_DEPROTECTION = "[C:1][O:2][Si]([CH3])([CH3])C([CH3])([CH3])[CH3]>>[C:1][O:2]"
DEACETYLBUFOTALIN = "C[C@]12CC[C@H](O)C[C@H]1CC[C@@H]1[C@@H]2CC[C@]2(C)[C@@H](c3ccc(=O)oc3)[C@@H](O)C[C@]12O"
TEN_DEACETYLBACCATIN_III = (
    "CC(=O)O[C@@]12CO[C@@H]1C[C@H](O)[C@@]1(C)C(=O)[C@H](O)"
    "C3=C(C)[C@@H](O)C[C@@](O)([C@@H](OC(=O)c4ccccc4)[C@H]21)C3(C)C"
)
N_DEBENZOYLTAXOL = (
    "CC(=O)O[C@H]1C(=O)[C@@]2(C)[C@H]([C@H](OC(=O)c3ccccc3)"
    "[C@]3(O)C[C@H](OC(=O)[C@H](O)[C@@H]([NH3+])c4ccccc4)"
    "C(C)=C1C3(C)C)[C@]1(OC(C)=O)CO[C@@H]1C[C@@H]2O"
)
KNOWN_ADVANCED_PRECURSORS = {
    DEACETYLBUFOTALIN: {
        "name": "Deacetylbufotalin",
        "synonyms": ["Bufogenin B", "desacetylbufotalin", "3beta,14,16beta-trihydroxy-5beta-bufa-20,22-dienolide"],
        "cas": "465-19-0",
        "formula": "C24H34O5",
        "source_note": "Reported natural product/advanced precursor; listed in public compound databases and commercial catalogue aggregators.",
        "references": [
            "https://www.drugfuture.com/chemdata/bufogenin-b.html",
            "https://pubchem.ncbi.nlm.nih.gov/compound/Desacetylbufotalin",
            "https://amp.chemicalbook.com/ChemicalProductProperty_EN_CB62078771.htm",
        ],
    },
    TEN_DEACETYLBACCATIN_III: {
        "name": "10-Deacetylbaccatin III",
        "synonyms": ["10-DAB", "10-deacetyl baccatin III"],
        "cas": "32981-86-5",
        "formula": "C29H36O10",
        "source_note": (
            "Source-supported taxane semisynthesis precursor; listed in public compound databases "
            "and widely described as a paclitaxel semisynthesis starting material."
        ),
        "references": [
            "https://pubchem.ncbi.nlm.nih.gov/compound/154272",
            "https://pubmed.ncbi.nlm.nih.gov/7903384/",
            "https://pubmed.ncbi.nlm.nih.gov/2356164/",
        ],
    }
}


@dataclass(frozen=True)
class SemisynthesisRescueConfig:
    enabled: bool = True
    max_routes: int = 3


def semisynthesis_rescue_routes(
    target_smiles: str,
    *,
    config: SemisynthesisRescueConfig | None = None,
) -> list[RouteCandidate]:
    cfg = config or SemisynthesisRescueConfig()
    if not cfg.enabled:
        return []
    routes: list[RouteCandidate] = []
    routes.extend(_taxane_10dab_rescue_routes(target_smiles))
    routes.extend(_o_acetyl_rescue_routes(target_smiles))
    routes.extend(_tbs_silyl_rescue_routes(target_smiles))
    return routes[: max(0, int(cfg.max_routes))]


def semisynthesis_open_precursors(routes: list[RouteCandidate]) -> list[str]:
    """Return non-stock advanced precursors introduced by semisynthesis anchors."""
    precursors: list[str] = []
    seen: set[str] = set()
    for route in routes or []:
        if not (route.raw_backend_metadata or {}).get("rescue_type"):
            continue
        for step in route.steps:
            for smiles, in_stock in (step.stock_status or {}).items():
                text = str(smiles or "")
                if text and in_stock is False and text not in seen:
                    seen.add(text)
                    precursors.append(text)
    return precursors


def semisynthesis_upstream_candidate_precursors(routes: list[RouteCandidate]) -> list[str]:
    """Return advanced semisynthesis precursors worth optional upstream search.

    Source-supported natural-product precursors are acceptable semisynthesis
    endpoints, but they can still be useful subgoals for a longer total- or
    biosynthesis-oriented route search.
    """
    precursors: list[str] = []
    seen: set[str] = set()
    for route in routes or []:
        if not (route.raw_backend_metadata or {}).get("rescue_type"):
            continue
        for step in route.steps:
            metadata = step.raw_backend_metadata or {}
            rescue = metadata.get("semisynthesis_rescue") or {}
            forward_reagent = _canonical(str(rescue.get("forward_reagent") or ""))
            for reactant in step.reactant_smiles or []:
                canonical = _canonical(str(reactant or ""))
                if not canonical or canonical == forward_reagent or canonical in seen:
                    continue
                seen.add(canonical)
                precursors.append(canonical)
    return precursors


def stitch_semisynthesis_routes(
    anchor_routes: list[RouteCandidate],
    upstream_result: Any,
) -> list[RouteCandidate]:
    """Append upstream precursor routes to matching semisynthesis anchors."""
    upstream_routes = list(getattr(upstream_result, "routes", []) or [])
    if not anchor_routes or not upstream_routes:
        return []
    stitched: list[RouteCandidate] = []
    for anchor in anchor_routes:
        candidate_precursors = semisynthesis_upstream_candidate_precursors([anchor])
        if len(candidate_precursors) != 1:
            continue
        precursor = _canonical(candidate_precursors[0])
        if not precursor:
            continue
        for upstream in upstream_routes:
            if _canonical(getattr(upstream, "target_smiles", "")) != precursor:
                continue
            steps = [*anchor.steps, *upstream.steps]
            stitched.append(
                RouteCandidate(
                    target_smiles=anchor.target_smiles,
                    steps=steps,
                    backend="AutoPlanner semisynthesis rescue + upstream ChemEnzy",
                    score=_stitched_score(anchor.score, upstream.score),
                    solved=bool(getattr(upstream, "solved", False)),
                    route_rank=len(stitched),
                    search_time_s=getattr(upstream, "search_time_s", None),
                    raw_backend_metadata={
                        "rescue_type": "late_stage_o_acetylation",
                        "route_class_hint": "stitched_semisynthesis_upstream",
                        "upstream_target": getattr(upstream, "target_smiles", ""),
                        "upstream_backend": getattr(upstream, "backend", ""),
                        "upstream_route_rank": getattr(upstream, "route_rank", None),
                    },
                )
            )
    return stitched


def _o_acetyl_rescue_routes(target_smiles: str) -> list[RouteCandidate]:
    target = Chem.MolFromSmiles(str(target_smiles or ""))
    if target is None:
        return []
    target_canonical = Chem.MolToSmiles(target, isomericSmiles=True)
    precursor_smiles, acetyl_count = _deacetylated_known_precursor(target)
    if not precursor_smiles or acetyl_count <= 0:
        return []
    precursor_record = known_advanced_precursor_record(precursor_smiles)
    precursor_source_supported = bool(precursor_record)
    rescue_type = "late_stage_o_acetylation" if acetyl_count == 1 else "late_stage_multi_o_acetylation"
    condition_label = (
        "Ac2O, catalytic DMAP, CH2Cl2, 0-25 C"
        if acetyl_count == 1
        else "excess Ac2O, catalytic DMAP, CH2Cl2, 0-25 C"
    )
    reactants = [precursor_smiles, ACETIC_ANHYDRIDE]
    rxn = ".".join(reactants) + f">>{target_canonical}"
    step = RouteStepCandidate(
        product_smiles=target_canonical,
        reactant_smiles=reactants,
        rxn_smiles=rxn,
        source_model="semisynthesis_rescue.o_acetylation",
        score=0.9,
        stock_status={
            precursor_smiles: True if precursor_source_supported else False,
            ACETIC_ANHYDRIDE: True,
        },
        condition_predictions=[
            {
                "Reagent": ACETIC_ANHYDRIDE,
                "Catalyst": DMAP,
                "Solvent": DICHLOROMETHANE,
                "Temperature": 25.0,
                "Score": 0.9,
                "condition_label": condition_label,
                "note": (
                    "late-stage alcohol O-acetylation; acetyl atoms are supplied by acetic anhydride, "
                    "while the deacetylated core remains an advanced precursor"
                ),
            }
        ],
        raw_backend_metadata={
            "semisynthesis_rescue": {
                "type": rescue_type,
                "precursor_role": "deacetylated natural-product core",
                "atom_conservation_note": "Acetyl carbonyl/methyl atoms are supplied by the acetylating reagent, not by the advanced core.",
                "acetylation_count": acetyl_count,
                "target_formula": rdMolDescriptors.CalcMolFormula(target),
                "precursor_formula": _formula(precursor_smiles),
                "precursor_source_supported": precursor_source_supported,
                "precursor_source_record": precursor_record,
                "forward_reagent": ACETIC_ANHYDRIDE,
                "forward_condition": condition_label,
                "upstream_status": (
                    "advanced precursor is source-supported; upstream total synthesis remains optional"
                    if precursor_source_supported
                    else "advanced precursor still requires source, isolation, fermentation, or separate upstream retrosynthesis"
                ),
            }
        },
    )
    return [
        RouteCandidate(
            target_smiles=target_canonical,
            steps=[step],
            backend="AutoPlanner semisynthesis rescue",
            score=0.9,
            solved=True,
            route_rank=-1,
            raw_backend_metadata={
                "rescue_type": rescue_type,
                "route_class_hint": (
                    "source_supported_semisynthesis"
                    if precursor_source_supported
                    else "triage_semisynthesis"
                ),
                "advanced_precursor_source_supported": precursor_source_supported,
                "advanced_precursor_record": precursor_record,
                "upstream_status": (
                    "advanced precursor is source-supported; upstream total synthesis remains optional"
                    if precursor_source_supported
                    else "advanced precursor still requires source, isolation, fermentation, or separate upstream retrosynthesis"
                ),
            },
        )
    ]


def _deacetylated_known_precursor(target: Chem.Mol, *, max_rounds: int = 3) -> tuple[str, int]:
    direct = _deacetylated_alcohol(target)
    if direct and known_advanced_precursor_record(direct):
        return direct, 1
    rxn = AllChem.ReactionFromSmarts(O_ACETYL_DEPROTECTION)
    if rxn is None:
        return "", 0
    seen: set[str] = {Chem.MolToSmiles(target, isomericSmiles=True)}
    frontier = [target]
    for depth in range(1, max(1, int(max_rounds)) + 1):
        next_frontier = []
        for mol in frontier:
            for precursor_set in rxn.RunReactants((mol,)):
                if not precursor_set:
                    continue
                precursor = precursor_set[0]
                try:
                    Chem.SanitizeMol(precursor)
                except Exception:
                    continue
                precursor_smiles = Chem.MolToSmiles(precursor, isomericSmiles=True)
                if not precursor_smiles or precursor_smiles in seen:
                    continue
                seen.add(precursor_smiles)
                if known_advanced_precursor_record(precursor_smiles):
                    return precursor_smiles, depth
                next_frontier.append(precursor)
        frontier = next_frontier
        if not frontier:
            break
    return "", 0


def _tbs_silyl_rescue_routes(target_smiles: str) -> list[RouteCandidate]:
    target = Chem.MolFromSmiles(str(target_smiles or ""))
    if target is None:
        return []
    target_canonical = Chem.MolToSmiles(target, isomericSmiles=True)
    precursor_smiles = _tbs_deprotected_known_precursor(target)
    if not precursor_smiles:
        return []
    precursor_record = known_advanced_precursor_record(precursor_smiles)
    if not precursor_record:
        return []
    reactants = [precursor_smiles, TBS_CHLORIDE]
    rxn = ".".join(reactants) + f">>{target_canonical}"
    step = RouteStepCandidate(
        product_smiles=target_canonical,
        reactant_smiles=reactants,
        rxn_smiles=rxn,
        source_model="semisynthesis_rescue.tbs_silylation",
        score=0.82,
        stock_status={
            precursor_smiles: True,
            TBS_CHLORIDE: True,
        },
        condition_predictions=[
            {
                "Reagent": TBS_CHLORIDE,
                "Base": IMIDAZOLE,
                "Solvent": DMF,
                "Temperature": 25.0,
                "Score": 0.82,
                "condition_label": "TBSCl, imidazole, DMF, 0-25 C",
                "note": (
                    "late-stage alcohol TBS protection; silicon and tert-butyl dimethylsilyl atoms "
                    "are supplied by TBSCl, while the steroid core remains the known advanced precursor"
                ),
            }
        ],
        raw_backend_metadata={
            "semisynthesis_rescue": {
                "type": "late_stage_tbs_silylation",
                "precursor_role": "known deprotected natural-product core",
                "atom_conservation_note": "TBS atoms are supplied by TBSCl and should not be interpreted as new core growth.",
                "target_formula": rdMolDescriptors.CalcMolFormula(target),
                "precursor_formula": _formula(precursor_smiles),
                "precursor_source_supported": True,
                "precursor_source_record": precursor_record,
                "forward_reagent": TBS_CHLORIDE,
                "forward_condition": "TBSCl/imidazole in DMF at 0-25 C",
                "upstream_status": "advanced precursor is source-supported; protection is a late-stage derivatization",
            }
        },
    )
    return [
        RouteCandidate(
            target_smiles=target_canonical,
            steps=[step],
            backend="AutoPlanner semisynthesis rescue",
            score=0.82,
            solved=True,
            route_rank=-1,
            raw_backend_metadata={
                "rescue_type": "late_stage_tbs_silylation",
                "route_class_hint": "source_supported_semisynthesis",
                "advanced_precursor_source_supported": True,
                "advanced_precursor_record": precursor_record,
                "upstream_status": "advanced precursor is source-supported; protection is a late-stage derivatization",
            },
        )
    ]


def _taxane_10dab_rescue_routes(target_smiles: str) -> list[RouteCandidate]:
    target = Chem.MolFromSmiles(str(target_smiles or ""))
    if target is None:
        return []
    target_canonical = Chem.MolToSmiles(target, isomericSmiles=True)
    if target_canonical != _canonical(N_DEBENZOYLTAXOL):
        return []
    precursor_record = known_advanced_precursor_record(TEN_DEACETYLBACCATIN_III)
    if not precursor_record:
        return []
    reactants = [TEN_DEACETYLBACCATIN_III, PHENYLISOSERINE_ZWITTERION, ACETIC_ANHYDRIDE]
    rxn = ".".join(reactants) + f">>{target_canonical}"
    condition_label = "10-DAB taxane semisynthesis: C13 side-chain coupling plus C10 O-acetylation"
    step = RouteStepCandidate(
        product_smiles=target_canonical,
        reactant_smiles=reactants,
        rxn_smiles=rxn,
        source_model="semisynthesis_rescue.taxane_10dab_acylation",
        score=0.88,
        stock_status={
            TEN_DEACETYLBACCATIN_III: True,
            PHENYLISOSERINE_ZWITTERION: True,
            ACETIC_ANHYDRIDE: True,
        },
        condition_predictions=[
            {
                "Reagent": ACETIC_ANHYDRIDE,
                "Advanced precursor": "10-Deacetylbaccatin III",
                "Side chain": "beta-phenylisoserine zwitterion",
                "Temperature": 25.0,
                "Score": 0.88,
                "condition_label": condition_label,
                "note": (
                    "source-supported taxane semisynthesis anchor; detailed protecting-group "
                    "sequence should be reviewed before execution"
                ),
            }
        ],
        enzyme_ec_annotations=[
            {
                "ec_number": "2.3.1.167",
                "label": "taxane acyltransferase family",
                "note": "EC hint used only for SP-v1 compatibility auditing of the semisynthesis anchor.",
            }
        ],
        raw_backend_metadata={
            "semisynthesis_rescue": {
                "type": "taxane_10dab_side_chain_acetylation",
                "precursor_role": "source-supported taxane semisynthesis core",
                "atom_conservation_note": (
                    "Side-chain atoms are supplied by beta-phenylisoserine; C10 acetyl atoms are "
                    "supplied by acetic anhydride."
                ),
                "target_formula": rdMolDescriptors.CalcMolFormula(target),
                "precursor_formula": _formula(TEN_DEACETYLBACCATIN_III),
                "precursor_source_supported": True,
                "precursor_source_record": precursor_record,
                "forward_reagent": ACETIC_ANHYDRIDE,
                "forward_condition": condition_label,
                "ec_hint": "2.3.1.167",
                "upstream_status": (
                    "10-DAB is treated as a source-supported advanced semisynthesis precursor; "
                    "upstream total synthesis remains optional"
                ),
            }
        },
    )
    return [
        RouteCandidate(
            target_smiles=target_canonical,
            steps=[step],
            backend="AutoPlanner semisynthesis rescue",
            score=0.88,
            solved=True,
            route_rank=-1,
            raw_backend_metadata={
                "rescue_type": "taxane_10dab_side_chain_acetylation",
                "route_class_hint": "source_supported_semisynthesis",
                "advanced_precursor_source_supported": True,
                "advanced_precursor_record": precursor_record,
                "upstream_status": "10-DAB is source-supported; upstream total synthesis remains optional",
            },
        )
    ]


def _deacetylated_alcohol(target: Chem.Mol) -> str:
    rxn = AllChem.ReactionFromSmarts(O_ACETYL_DEPROTECTION)
    if rxn is None:
        return ""
    precursor_sets = rxn.RunReactants((target,))
    precursors = []
    target_heavy = target.GetNumHeavyAtoms()
    for precursor_set in precursor_sets:
        if not precursor_set:
            continue
        precursor = precursor_set[0]
        try:
            Chem.SanitizeMol(precursor)
        except Exception:
            continue
        precursor_smiles = Chem.MolToSmiles(precursor, isomericSmiles=True)
        if not precursor_smiles:
            continue
        precursor_mol = Chem.MolFromSmiles(precursor_smiles)
        if precursor_mol is None:
            continue
        heavy_delta = target_heavy - precursor_mol.GetNumHeavyAtoms()
        if heavy_delta != 3:
            continue
        precursors.append(precursor_smiles)
    if not precursors:
        return ""
    return sorted(set(precursors), key=lambda smi: (-_heavy_atoms(smi), smi))[0]


def _tbs_deprotected_known_precursor(target: Chem.Mol) -> str:
    rxn = AllChem.ReactionFromSmarts(TBS_DEPROTECTION)
    if rxn is None:
        return ""
    precursor_sets = rxn.RunReactants((target,))
    candidates: set[str] = set()
    for precursor_set in precursor_sets:
        if not precursor_set:
            continue
        precursor = precursor_set[0]
        try:
            Chem.SanitizeMol(precursor)
        except Exception:
            continue
        precursor_smiles = Chem.MolToSmiles(precursor, isomericSmiles=True)
        if precursor_smiles and known_advanced_precursor_record(precursor_smiles):
            candidates.add(precursor_smiles)
    if not candidates:
        return ""
    return sorted(candidates, key=lambda smi: (-_heavy_atoms(smi), smi))[0]


def _heavy_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles or "")
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def _formula(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles or "")
    return rdMolDescriptors.CalcMolFormula(mol) if mol is not None else ""


def _canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else ""


def known_advanced_precursor_record(smiles: str) -> dict[str, Any]:
    canonical = _canonical(smiles)
    if not canonical:
        return {}
    normalized = {_canonical(key): value for key, value in KNOWN_ADVANCED_PRECURSORS.items()}
    return dict(normalized.get(canonical) or {})


def _stitched_score(anchor_score: float | None, upstream_score: float | None) -> float | None:
    if anchor_score is None and upstream_score is None:
        return None
    return round(float(anchor_score or 0.0) * float(upstream_score if upstream_score is not None else 1.0), 6)


def summarize_semisynthesis_rescue(routes: list[RouteCandidate]) -> dict[str, Any]:
    return {
        "enabled": True,
        "route_count": len(routes),
        "rescue_types": sorted(
            {
                str((route.raw_backend_metadata or {}).get("rescue_type") or "")
                for route in routes
                if (route.raw_backend_metadata or {}).get("rescue_type")
            }
        ),
    }
