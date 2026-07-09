"""Independent raw route verification for harness-level solved claims."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors


RDLogger.DisableLog("rdApp.*")

ROUTE_VERIFIER_SCHEMA = "harness_route_verifier_report.v1"


@dataclass
class RouteVerifierReport:
    accepted: bool
    route_status: str
    reasons: list[str] = field(default_factory=list)
    route_count: int = 0
    accepted_route_count: int = 0
    rejected_route_count: int = 0
    rejected_route_summary: list[dict[str, Any]] = field(default_factory=list)
    rejected_terminal_list: list[dict[str, Any]] = field(default_factory=list)
    failure_events: list[dict[str, Any]] = field(default_factory=list)
    best_route_rank: int | None = None
    target_match: bool = False
    target_equivalence_audit: dict[str, Any] = field(default_factory=dict)
    schema_version: str = ROUTE_VERIFIER_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_chemenzy_raw_routes(
    chemenzy_result: dict[str, Any],
    *,
    target_smiles: str,
    case_id: str = "",
    max_simple_terminal_heavy_atoms: int = 24,
    advanced_terminal_similarity: float = 0.5,
    large_atom_jump_heavy_atoms: int = 15,
) -> dict[str, Any]:
    """Verify that at least one native route is genuinely stock closed.

    ChemEnzy raw routes may mark only small terminal leaves as stock while
    retaining non-stock advanced reactants inside individual steps. This
    verifier audits the raw step graph rather than the summarized solved flag.
    """
    result = dict(chemenzy_result.get("result") or chemenzy_result or {})
    routes = [dict(route) for route in result.get("routes") or [] if isinstance(route, dict)]
    target_audit = _target_equivalence_audit(
        request_target_smiles=target_smiles,
        backend_target_smiles=str(result.get("target") or result.get("target_smiles") or ""),
        routes=routes,
    )
    target = _mol(target_smiles or "")
    if target is None:
        return RouteVerifierReport(
            accepted=False,
            route_status="unresolved",
            reasons=["invalid_target_smiles"],
            route_count=len(routes),
            target_match=False,
            target_equivalence_audit=target_audit,
        ).to_dict()

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rejected_terminals: list[dict[str, Any]] = []
    failure_events: list[dict[str, Any]] = []
    target_fp = AllChem.GetMorganFingerprintAsBitVect(target, 2, nBits=2048)
    target_canonical = str(target_audit.get("request_canonical_isomeric_smiles") or "")
    for route in routes:
        route_report = _verify_one_route(
            route,
            target_fp=target_fp,
            target_canonical=target_canonical,
            case_id=case_id,
            max_simple_terminal_heavy_atoms=max_simple_terminal_heavy_atoms,
            advanced_terminal_similarity=advanced_terminal_similarity,
            large_atom_jump_heavy_atoms=large_atom_jump_heavy_atoms,
        )
        if route_report["accepted"]:
            accepted.append(route_report)
        else:
            rejected.append(route_report)
            rejected_terminals.extend(route_report.get("rejected_terminals") or [])
            failure_events.extend(route_report.get("failure_events") or [])

    reasons: list[str] = []
    if not routes:
        reasons.append("no_raw_routes")
    if not accepted and routes:
        reasons.append("no_verifier_accepted_stock_closed_route")
    if any("hidden_nonstock_reactants" in row.get("reasons", []) for row in rejected):
        reasons.append("hidden_nonstock_reactants")
    if any("large_atom_jump" in row.get("reasons", []) for row in rejected):
        reasons.append("large_atom_jump")
    if any("advanced_same_scaffold_terminal" in row.get("reasons", []) for row in rejected):
        reasons.append("advanced_same_scaffold_terminal")
    if any("route_target_product_mismatch" in row.get("reasons", []) for row in rejected):
        reasons.append("route_target_product_mismatch")
    target_match = bool(target_audit.get("target_match"))
    if not target_match:
        reasons.append("target_equivalence_mismatch")

    final_accepted = bool(accepted) and target_match
    report = RouteVerifierReport(
        accepted=final_accepted,
        route_status=(
            "solved"
            if final_accepted
            else "target_mismatch_rejected"
            if routes and not target_match
            else "fake_closed_rejected"
            if routes
            else "unresolved"
        ),
        reasons=sorted(set(reasons)),
        route_count=len(routes),
        accepted_route_count=len(accepted) if target_match else 0,
        rejected_route_count=(len(rejected) + (len(accepted) if not target_match else 0)),
        rejected_route_summary=_compact_route_reports(rejected),
        rejected_terminal_list=_unique_terminal_rejections(rejected_terminals),
        failure_events=failure_events[:50],
        best_route_rank=accepted[0].get("route_rank") if final_accepted else None,
        target_match=target_match,
        target_equivalence_audit={
            **target_audit,
            "route_candidate_accepted_count_before_target_match": len(accepted),
        },
    )
    return report.to_dict()


def _verify_one_route(
    route: dict[str, Any],
    *,
    target_fp: Any,
    target_canonical: str,
    case_id: str,
    max_simple_terminal_heavy_atoms: int,
    advanced_terminal_similarity: float,
    large_atom_jump_heavy_atoms: int,
) -> dict[str, Any]:
    route_rank = int(route.get("route_rank") or 0)
    steps = [dict(step) for step in route.get("steps") or [] if isinstance(step, dict)]
    terminals = _terminal_reactants(route)
    reasons: list[str] = []
    rejected_terminals: list[dict[str, Any]] = []
    failure_events: list[dict[str, Any]] = []

    hidden_nonstock = _hidden_nonstock_reactants(steps)
    if hidden_nonstock:
        reasons.append("hidden_nonstock_reactants")
        failure_events.append(
            _failure_event(
                case_id,
                route_rank,
                "hidden_nonstock_reactants",
                {"count": len(hidden_nonstock), "sample": _compound_summary(hidden_nonstock[0], target_fp)},
            )
        )

    jumps = _large_atom_jumps(steps, threshold=large_atom_jump_heavy_atoms)
    if jumps:
        reasons.append("large_atom_jump")
        failure_events.append(_failure_event(case_id, route_rank, "large_atom_jump", {"jumps": jumps[:5]}))

    if not terminals:
        reasons.append("missing_terminal_reactants")
    if target_canonical and steps:
        route_products = [_canonical_smiles(_step_product(step)) for step in steps if _step_product(step)]
        if route_products and target_canonical not in route_products:
            reasons.append("route_target_product_mismatch")
            failure_events.append(
                _failure_event(
                    case_id,
                    route_rank,
                    "route_target_product_mismatch",
                    {
                        "request_canonical_isomeric_smiles": target_canonical,
                        "route_product_canonical_isomeric_smiles": route_products[:5],
                    },
                )
            )
    for smiles in terminals:
        summary = _compound_summary(smiles, target_fp)
        if not summary.get("valid"):
            reasons.append("invalid_terminal_smiles")
            rejected_terminals.append({**summary, "route_rank": route_rank, "reason": "invalid_terminal_smiles"})
            continue
        if summary["heavy_atoms"] > max_simple_terminal_heavy_atoms or summary["target_similarity"] >= advanced_terminal_similarity:
            reasons.append("advanced_same_scaffold_terminal")
            rejected_terminals.append(
                {**summary, "route_rank": route_rank, "reason": "advanced_same_scaffold_terminal"}
            )

    return {
        "accepted": not reasons,
        "route_rank": route_rank,
        "score": route.get("score"),
        "n_steps": route.get("n_steps") or len(steps),
        "reasons": sorted(set(reasons)),
        "terminal_count": len(terminals),
        "hidden_nonstock_count": len(hidden_nonstock),
        "large_atom_jump_count": len(jumps),
        "rejected_terminals": rejected_terminals,
        "failure_events": failure_events,
    }


def _terminal_reactants(route: dict[str, Any]) -> list[str]:
    seen: list[str] = []
    for smiles in (route.get("metrics") or {}).get("terminal_reactants") or []:
        text = str(smiles or "")
        if text and text not in seen:
            seen.append(text)
    for step in route.get("steps") or []:
        for smiles, in_stock in (dict(step.get("stock_status") or {})).items():
            text = str(smiles or "")
            if text and bool(in_stock) and text not in seen:
                seen.append(text)
    return seen


def _hidden_nonstock_reactants(steps: list[dict[str, Any]]) -> list[str]:
    generated = {_canonical_smiles(_step_product(step)) for step in steps}
    generated.discard("")
    out: list[str] = []
    for step in steps:
        stock = dict(step.get("stock_status") or {})
        for smiles in _step_reactants(step):
            if _canonical_smiles(smiles) in generated:
                continue
            if _stock_value(stock, smiles) is False and smiles not in out:
                out.append(smiles)
    return out


def _step_product(step: dict[str, Any]) -> str:
    product = str(step.get("product") or "")
    if product:
        return product
    reaction = str(step.get("reaction_smiles") or "")
    if ">>" not in reaction:
        return ""
    return reaction.split(">>", 1)[1].strip()


def _canonical_smiles(smiles: str) -> str:
    mol = _mol(smiles)
    if mol is None:
        return str(smiles or "").strip()
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _stock_value(stock: dict[str, Any], smiles: str) -> Any:
    if smiles in stock:
        return stock.get(smiles)
    target = _canonical_smiles(smiles)
    for key, value in stock.items():
        if _canonical_smiles(str(key or "")) == target:
            return value
    return None


def _large_atom_jumps(steps: list[dict[str, Any]], *, threshold: int) -> list[dict[str, Any]]:
    jumps: list[dict[str, Any]] = []
    for step in steps:
        reactant_heavy = sum(_heavy_atoms(smiles) for smiles in _step_reactants(step))
        product_heavy = _heavy_atoms(str(step.get("product") or ""))
        delta = product_heavy - reactant_heavy
        if delta >= threshold:
            jumps.append(
                {
                    "step_index": step.get("index"),
                    "reactant_heavy_atoms": reactant_heavy,
                    "product_heavy_atoms": product_heavy,
                    "delta_heavy_atoms": delta,
                }
            )
    return jumps


def _step_reactants(step: dict[str, Any]) -> list[str]:
    values = [str(step.get("main_reactant") or "")]
    values.extend(str(item or "") for item in step.get("aux_reactants") or [])
    return [value for value in values if value]


def _compact_route_reports(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "route_rank": row.get("route_rank"),
            "score": row.get("score"),
            "n_steps": row.get("n_steps"),
            "reasons": list(row.get("reasons") or []),
            "terminal_count": row.get("terminal_count"),
            "hidden_nonstock_count": row.get("hidden_nonstock_count"),
            "large_atom_jump_count": row.get("large_atom_jump_count"),
        }
        for row in rows[:50]
    ]


def _unique_terminal_rejections(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("canonical_smiles") or row.get("smiles") or ""), str(row.get("reason") or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out[:50]


def _failure_event(case_id: str, route_rank: int, reason: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "failure_event.v1",
        "failure_id": f"{case_id or 'case'}:route_{route_rank}:{reason}",
        "case_id": case_id or "case",
        "reason": reason,
        "severity": "high",
        "source_artifact_id": "chemenzy_native_raw_result",
        "details": dict(details),
    }


def _compound_summary(smiles: str, target_fp: Any) -> dict[str, Any]:
    mol = _mol(smiles)
    if mol is None:
        return {"smiles": str(smiles or ""), "valid": False}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return {
        "smiles": str(smiles or ""),
        "canonical_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
        "valid": True,
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "mol_weight": round(float(Descriptors.MolWt(mol)), 3),
        "target_similarity": round(float(DataStructs.TanimotoSimilarity(target_fp, fp)), 4),
    }


def _target_equivalence_audit(
    *,
    request_target_smiles: str,
    backend_target_smiles: str,
    routes: list[dict[str, Any]],
) -> dict[str, Any]:
    request = _compound_identity(request_target_smiles)
    backend = _compound_identity(backend_target_smiles)
    reasons: list[str] = []
    if not request["valid"]:
        reasons.append("invalid_request_target_smiles")
    if not str(backend_target_smiles or "").strip():
        reasons.append("missing_backend_target_smiles")
    elif not backend["valid"]:
        reasons.append("invalid_backend_target_smiles")
    target_match = bool(
        request["valid"]
        and backend["valid"]
        and request["canonical_isomeric_smiles"] == backend["canonical_isomeric_smiles"]
        and request["inchikey"] == backend["inchikey"]
    )
    if request["valid"] and backend["valid"] and not target_match:
        reasons.append("request_backend_target_mismatch")
    route_product_audits = _route_target_product_audits(routes, request["canonical_isomeric_smiles"])
    if route_product_audits and not any(bool(row.get("target_match")) for row in route_product_audits):
        reasons.append("no_route_product_matches_request_target")
    return {
        "schema_version": "target_equivalence_audit.v1",
        "request_target_smiles": str(request_target_smiles or ""),
        "request_canonical_isomeric_smiles": request["canonical_isomeric_smiles"],
        "request_inchikey": request["inchikey"],
        "backend_target_smiles": str(backend_target_smiles or ""),
        "backend_canonical_isomeric_smiles": backend["canonical_isomeric_smiles"],
        "backend_inchikey": backend["inchikey"],
        "target_match": target_match,
        "match_basis": "canonical_isomeric_smiles_and_inchikey",
        "route_target_product_audits": route_product_audits[:50],
        "reasons": sorted(set(reasons)),
    }


def _route_target_product_audits(routes: list[dict[str, Any]], request_canonical: str) -> list[dict[str, Any]]:
    if not request_canonical:
        return []
    rows: list[dict[str, Any]] = []
    for route in routes[:50]:
        products = []
        for step in route.get("steps") or []:
            if not isinstance(step, dict):
                continue
            product = _step_product(step)
            if not product:
                continue
            products.append(_compound_identity(product))
        product_cans = [str(item.get("canonical_isomeric_smiles") or "") for item in products if item.get("valid")]
        rows.append(
            {
                "route_rank": route.get("route_rank"),
                "target_match": request_canonical in product_cans,
                "product_count": len(products),
                "product_canonical_isomeric_smiles": product_cans[:5],
            }
        )
    return rows


def _compound_identity(smiles: str) -> dict[str, Any]:
    mol = _mol(smiles)
    if mol is None:
        return {
            "valid": False,
            "canonical_isomeric_smiles": "",
            "inchikey": "",
        }
    return {
        "valid": True,
        "canonical_isomeric_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
        "inchikey": _inchikey(mol),
    }


def _inchikey(mol: Chem.Mol) -> str:
    try:
        return str(Chem.MolToInchiKey(mol) or "")
    except Exception:
        return ""


def _heavy_atoms(smiles: str) -> int:
    mol = _mol(smiles)
    return int(mol.GetNumHeavyAtoms()) if mol is not None else 0


def _mol(smiles: str) -> Chem.Mol | None:
    text = str(smiles or "").strip()
    return Chem.MolFromSmiles(text) if text else None
