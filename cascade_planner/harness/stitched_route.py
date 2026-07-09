"""Deterministic stitching audit for literature chains and subgoal routes."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")

STITCHED_SEMISYNTHESIS_ROUTE_SCHEMA = "stitched_semisynthesis_route.v1"


def compile_stitched_semisynthesis_route(
    *,
    literature_chain_audit: dict[str, Any] | str | Path | None,
    subgoal_verifier: dict[str, Any] | str | Path | None = None,
    subgoal_raw_result: dict[str, Any] | str | Path | None = None,
    route_expansion_result: dict[str, Any] | str | Path | None = None,
    output_dir: str | Path | None = None,
    case_id: str = "",
    target_smiles: str = "",
    target_name: str = "",
    subgoal_name: str = "",
) -> dict[str, Any]:
    """Audit whether a literature terminal and solved subgoal route can be joined.

    A subgoal route may only close the full natural-product route when the
    literature-chain terminal is exactly the same compound as the verified
    subgoal target. Names and labels are advisory; the acceptance gate is
    isomeric canonical SMILES plus InChIKey.
    """
    chain = _load_jsonish(literature_chain_audit)
    expansion = _load_jsonish(route_expansion_result)
    selected = _select_subgoal(expansion)
    verifier = _load_jsonish(subgoal_verifier) or dict(selected.get("verifier") or {})
    raw = _load_jsonish(subgoal_raw_result)
    if not raw and selected.get("raw_result_path"):
        raw = _load_jsonish(selected.get("raw_result_path"))

    reasons: list[str] = []
    warnings: list[str] = []
    artifact_refs = _artifact_refs(
        literature_chain_audit=literature_chain_audit,
        subgoal_verifier=subgoal_verifier,
        subgoal_raw_result=subgoal_raw_result,
        route_expansion_result=route_expansion_result,
        selected_subgoal=selected,
    )

    chain_summary = _literature_chain_summary(chain)
    if not chain:
        reasons.append("literature_chain_missing")
    elif not chain_summary["chain_accepted"]:
        reasons.append("literature_chain_not_accepted")
    if not chain_summary["terminal"]["valid"]:
        reasons.append("literature_terminal_invalid_or_missing")

    target_audit = _target_identity_audit(
        requested_target_smiles=target_smiles,
        literature_target_smiles=str(chain.get("target_smiles") or ""),
    )
    if target_audit["required"] and not target_audit["target_match"]:
        reasons.append("target_input_literature_chain_mismatch")

    subgoal_summary = _subgoal_summary(
        verifier=verifier,
        raw=raw,
        selected_subgoal=selected,
        subgoal_name=subgoal_name,
    )
    if not verifier:
        reasons.append("subgoal_verifier_missing")
    else:
        if not subgoal_summary["verifier_accepted"]:
            reasons.append("subgoal_verifier_not_accepted")
        if subgoal_summary["route_status"] != "solved":
            reasons.append("subgoal_route_not_solved")
        if not subgoal_summary["target_match"]:
            reasons.append("subgoal_target_not_verified")
    if subgoal_summary["verifier_reasons"]:
        warnings.extend(f"subgoal_verifier:{item}" for item in subgoal_summary["verifier_reasons"])

    terminal_match = _terminal_subgoal_match_audit(
        terminal=chain_summary["terminal"],
        subgoal=subgoal_summary["target"],
    )
    if verifier and not terminal_match["accepted"]:
        reasons.append("literature_terminal_subgoal_target_mismatch")

    accepted = not sorted(set(reasons))
    literature_step_count = int(chain_summary["step_count"])
    subgoal_step_count = int(subgoal_summary["best_route_step_count"])
    result = {
        "schema_version": STITCHED_SEMISYNTHESIS_ROUTE_SCHEMA,
        "accepted": accepted,
        "solved": accepted,
        "route_status": "solved" if accepted else _failure_status(reasons),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id or str(chain.get("case_id") or ""),
        "target": {
            "name": target_name,
            "smiles": target_smiles or str(chain.get("target_smiles") or ""),
            "identity_audit": target_audit,
        },
        "literature_chain": chain_summary,
        "subgoal_closure": subgoal_summary,
        "terminal_match_audit": terminal_match,
        "stock_audit_passed": accepted,
        "combined_route": {
            "route_type": "stitched_semisynthesis",
            "direction": "stock_to_subgoal_terminal_then_literature_to_target",
            "literature_step_count": literature_step_count,
            "subgoal_route_step_count": subgoal_step_count,
            "combined_step_count": literature_step_count + subgoal_step_count,
            "segments": [
                {
                    "segment_id": "subgoal_stock_closure",
                    "role": "stock_to_literature_terminal",
                    "status": "verified_solved" if subgoal_summary["verifier_accepted"] else "not_verified",
                    "target_smiles": subgoal_summary["target"]["input_smiles"],
                    "best_route_rank": subgoal_summary["best_route_rank"],
                    "step_count": subgoal_step_count,
                },
                {
                    "segment_id": "source_detail_literature_chain",
                    "role": "literature_terminal_to_target",
                    "status": "accepted" if chain_summary["chain_accepted"] else "not_accepted",
                    "terminal_smiles": chain_summary["terminal"]["input_smiles"],
                    "target_smiles": str(chain.get("target_smiles") or target_smiles or ""),
                    "step_count": literature_step_count,
                    "source_ref": chain_summary["source_ref"],
                },
            ],
        },
        "source_policy": {
            "terminal_identity_match_required": True,
            "subgoal_solved_does_not_imply_target_solved_without_stitch": True,
            "literature_segment_requires_source_detail_chain": True,
            "subgoal_segment_requires_route_verifier": True,
            "final_verdict_authority": "deterministic_validators",
            "production_write_blocked": True,
        },
        "artifact_refs": artifact_refs,
        "warnings": sorted(set(warnings)),
        "reasons": sorted(set(reasons)),
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "stitched_semisynthesis_route.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def _load_jsonish(value: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    path = Path(value)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data) if isinstance(data, dict) else {}


def _select_subgoal(expansion: dict[str, Any]) -> dict[str, Any]:
    rows = [dict(item) for item in expansion.get("subgoals") or [] if isinstance(item, dict)]
    accepted = [
        row
        for row in rows
        if row.get("accepted") or (isinstance(row.get("verifier"), dict) and row["verifier"].get("accepted"))
    ]
    return dict((accepted or rows or [{}])[0])


def _literature_chain_summary(chain: dict[str, Any]) -> dict[str, Any]:
    steps = chain.get("chain") or chain.get("steps") or []
    terminal_smiles = (
        str(chain.get("terminal_smiles") or "")
        or str(chain.get("final_reactant_smiles") or "")
        or _last_main_reactant(steps)
    )
    terminal_name = (
        str(chain.get("terminal_name") or "")
        or str(chain.get("final_reactant_name") or "")
        or _last_main_reactant_name(steps)
    )
    explicit_accepted = chain.get("accepted") if "accepted" in chain else None
    chain_accepted = bool(
        explicit_accepted
        if explicit_accepted is not None
        else chain.get("terminal_reached") or chain.get("chain_complete_to_literature_start")
    )
    if not chain_accepted and steps and not chain.get("reasons"):
        chain_accepted = True
    return {
        "schema_version": str(chain.get("schema_version") or ""),
        "accepted": bool(chain.get("accepted", chain_accepted)),
        "chain_accepted": chain_accepted,
        "source_ref": str(chain.get("source_ref") or ""),
        "step_count": int(chain.get("step_count") or len(steps)),
        "terminal_name": terminal_name,
        "terminal": _compound_identity(terminal_smiles),
        "terminal_reached": bool(chain.get("terminal_reached") or chain.get("chain_complete_to_literature_start")),
        "reasons": [str(item) for item in chain.get("reasons") or []],
    }


def _last_main_reactant(steps: Any) -> str:
    if not isinstance(steps, list) or not steps:
        return ""
    last = steps[-1]
    if not isinstance(last, dict):
        return ""
    return str(last.get("main_reactant_smiles") or last.get("final_reactant_smiles") or "")


def _last_main_reactant_name(steps: Any) -> str:
    if not isinstance(steps, list) or not steps:
        return ""
    last = steps[-1]
    if not isinstance(last, dict):
        return ""
    return str(last.get("main_reactant_name") or last.get("final_reactant_name") or "")


def _subgoal_summary(
    *,
    verifier: dict[str, Any],
    raw: dict[str, Any],
    selected_subgoal: dict[str, Any],
    subgoal_name: str,
) -> dict[str, Any]:
    target_audit = dict(verifier.get("target_equivalence_audit") or {})
    selected_target = dict(selected_subgoal.get("subgoal") or {})
    target_smiles = (
        str(target_audit.get("request_target_smiles") or "")
        or str(raw.get("target") or raw.get("target_smiles") or "")
        or str(selected_target.get("smiles") or "")
    )
    return {
        "accepted": bool(verifier.get("accepted")),
        "verifier_accepted": bool(verifier.get("accepted")),
        "route_status": str(verifier.get("route_status") or ("solved" if verifier.get("accepted") else "")),
        "target_match": bool(verifier.get("target_match")),
        "target": _compound_identity(target_smiles),
        "target_equivalence_audit": target_audit,
        "route_count": int(verifier.get("route_count") or len(raw.get("routes") or [])),
        "accepted_route_count": int(verifier.get("accepted_route_count") or 0),
        "best_route_rank": verifier.get("best_route_rank"),
        "best_route_step_count": _best_route_step_count(raw, verifier),
        "subgoal_name": subgoal_name or str(selected_target.get("name") or ""),
        "raw_solved": bool((raw.get("search_status") or {}).get("solved")),
        "verifier_reasons": [str(item) for item in verifier.get("reasons") or []],
    }


def _best_route_step_count(raw: dict[str, Any], verifier: dict[str, Any]) -> int:
    routes = [dict(item) for item in raw.get("routes") or [] if isinstance(item, dict)]
    if not routes:
        return 0
    best_rank = verifier.get("best_route_rank")
    route = next((item for item in routes if item.get("route_rank") == best_rank), routes[0])
    return int(route.get("n_steps") or len(route.get("steps") or []))


def _target_identity_audit(*, requested_target_smiles: str, literature_target_smiles: str) -> dict[str, Any]:
    requested = _compound_identity(requested_target_smiles)
    literature = _compound_identity(literature_target_smiles)
    required = bool(str(requested_target_smiles or "").strip() and str(literature_target_smiles or "").strip())
    target_match = bool(required and _same_compound(requested, literature))
    reasons: list[str] = []
    if required and not target_match:
        reasons.append("target_identity_mismatch")
    return {
        "schema_version": "stitched_route_target_identity_audit.v1",
        "required": required,
        "target_match": target_match,
        "requested_target": requested,
        "literature_target": literature,
        "match_basis": "canonical_isomeric_smiles_and_inchikey",
        "reasons": reasons,
    }


def _terminal_subgoal_match_audit(*, terminal: dict[str, Any], subgoal: dict[str, Any]) -> dict[str, Any]:
    accepted = _same_compound(terminal, subgoal)
    reasons = [] if accepted else ["terminal_subgoal_identity_mismatch"]
    return {
        "schema_version": "terminal_subgoal_match_audit.v1",
        "accepted": accepted,
        "terminal": terminal,
        "subgoal_target": subgoal,
        "match_basis": "canonical_isomeric_smiles_and_inchikey",
        "reasons": reasons,
    }


def _same_compound(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return bool(
        left.get("valid")
        and right.get("valid")
        and left.get("canonical_isomeric_smiles") == right.get("canonical_isomeric_smiles")
        and left.get("inchikey") == right.get("inchikey")
    )


def _compound_identity(smiles: str) -> dict[str, Any]:
    text = str(smiles or "").strip()
    mol = Chem.MolFromSmiles(text) if text else None
    if mol is None:
        return {
            "valid": False,
            "input_smiles": text,
            "canonical_isomeric_smiles": "",
            "inchikey": "",
        }
    return {
        "valid": True,
        "input_smiles": text,
        "canonical_isomeric_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
        "inchikey": _inchikey(mol),
    }


def _inchikey(mol: Chem.Mol) -> str:
    try:
        return str(Chem.MolToInchiKey(mol) or "")
    except Exception:
        return ""


def _artifact_refs(**items: Any) -> dict[str, str]:
    refs: dict[str, str] = {}
    for key, value in items.items():
        if key == "selected_subgoal" and isinstance(value, dict):
            for nested_key in ("raw_result_path", "request_path"):
                if value.get(nested_key):
                    refs[f"subgoal_{nested_key}"] = str(value[nested_key])
            continue
        if isinstance(value, (str, Path)) and str(value).strip():
            refs[key] = str(value)
    return refs


def _failure_status(reasons: list[str]) -> str:
    reason_set = set(reasons)
    if "literature_terminal_subgoal_target_mismatch" in reason_set:
        return "terminal_mismatch"
    if "subgoal_verifier_not_accepted" in reason_set or "subgoal_route_not_solved" in reason_set:
        return "subgoal_not_verified"
    if "literature_chain_not_accepted" in reason_set:
        return "literature_chain_not_accepted"
    return "stitch_rejected"
