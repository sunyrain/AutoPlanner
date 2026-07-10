"""Independent raw route verification for harness-level solved claims."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import mmap
from pathlib import Path
import re
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors


RDLogger.DisableLog("rdApp.*")

ROUTE_VERIFIER_SCHEMA = "harness_route_verifier_report.v1"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHEMENZY_STOCK_CONFIG = _REPO_ROOT / (
    "vendor/ChemEnzyRetroPlanner/retro_planner/config/config.yaml"
)
_TRUSTED_COMMON_STOCK = frozenset(
    _canonical
    for smiles in ("C", "CC", "CCO", "N", "O", "O=O", "N#N", "O=C=O", "Cl", "ClCl", "Br", "BrBr")
    if (_canonical := Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
)
_COMMON_CATALOG_NAME = "autoplanner_common_commodity.v1"
_COMMON_CATALOG_SHA256 = hashlib.sha256(
    ("\n".join(sorted(_TRUSTED_COMMON_STOCK)) + "\n").encode("utf-8")
).hexdigest()
_CATALOG_HIT_CACHE: dict[tuple[str, int, int, str], bool] = {}
_CATALOG_DIGEST_CACHE: dict[tuple[str, int, int], str] = {}


@dataclass
class RouteVerifierReport:
    accepted: bool
    route_status: str
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    route_count: int = 0
    accepted_route_count: int = 0
    rejected_route_count: int = 0
    rejected_route_summary: list[dict[str, Any]] = field(default_factory=list)
    rejected_terminal_list: list[dict[str, Any]] = field(default_factory=list)
    failure_events: list[dict[str, Any]] = field(default_factory=list)
    best_route_rank: int | None = None
    best_route_step_count: int = 0
    accepted_route: dict[str, Any] = field(default_factory=dict)
    accepted_route_audit: dict[str, Any] = field(default_factory=dict)
    stock_catalog_audit: dict[str, Any] = field(default_factory=dict)
    verification_policy: dict[str, Any] = field(default_factory=dict)
    target_match: bool = False
    target_equivalence_audit: dict[str, Any] = field(default_factory=dict)
    schema_version: str = ROUTE_VERIFIER_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_accepted_route_verifier_report(
    value: Any,
    *,
    expected_target_smiles: str = "",
) -> bool:
    """Validate the complete solved-route report contract.

    In particular, ``accepted`` is insufficient without an explicitly
    materialized best route. ``best_route_step_count`` is derived by this
    module from the raw ``steps`` list and never from backend ``n_steps``.
    """
    if not isinstance(value, dict):
        return False
    report = dict(value)
    if not isinstance(report.get("reasons"), list) or report["reasons"]:
        return False
    if not isinstance(report.get("warnings"), list):
        return False
    if not isinstance(report.get("target_equivalence_audit"), dict):
        return False
    stock_catalog_audit = report.get("stock_catalog_audit")
    if not isinstance(stock_catalog_audit, dict):
        return False
    if stock_catalog_audit.get("schema_version") != "independent_stock_catalog_audit.v1":
        return False
    policy = _strict_verification_policy(report.get("verification_policy"))
    if policy is None:
        return False
    audit = dict(report["target_equivalence_audit"])
    if str(expected_target_smiles or "").strip():
        expected = _canonical_smiles(expected_target_smiles)
        reported = _canonical_smiles(
            str(
                audit.get("request_canonical_isomeric_smiles")
                or audit.get("request_target_smiles")
                or ""
            )
        )
        if not expected or not reported or expected != reported:
            return False
    try:
        route_count = int(report.get("route_count") or 0)
        accepted_route_count = int(report.get("accepted_route_count") or 0)
        rejected_route_count = int(report.get("rejected_route_count") or 0)
        best_route_step_count = int(report.get("best_route_step_count") or 0)
    except (TypeError, ValueError):
        return False
    accepted_route = report.get("accepted_route")
    if not isinstance(accepted_route, dict) or not accepted_route.get("steps"):
        return False
    request_target = str(
        expected_target_smiles
        or audit.get("request_canonical_isomeric_smiles")
        or audit.get("request_target_smiles")
        or ""
    )
    if not request_target:
        return False
    revalidation_context = stock_catalog_audit.get("revalidation_context")
    if not isinstance(revalidation_context, dict):
        return False
    reverified = verify_chemenzy_raw_routes(
        {
            "target": request_target,
            "routes": [dict(accepted_route)],
            "stock_catalog_context": dict(revalidation_context),
        },
        target_smiles=request_target,
        max_simple_terminal_heavy_atoms=policy["max_simple_terminal_heavy_atoms"],
        advanced_terminal_similarity=policy["advanced_terminal_similarity"],
        large_atom_jump_heavy_atoms=policy["large_atom_jump_heavy_atoms"],
    )
    reverified_catalog_audit = dict(reverified.get("stock_catalog_audit") or {})
    materialization_matches = bool(
        reverified.get("accepted") is True
        and reverified.get("best_route_rank") == report.get("best_route_rank")
        and int(reverified.get("best_route_step_count") or 0) == best_route_step_count
        and _catalog_audit_signature(reverified_catalog_audit)
        == _catalog_audit_signature(stock_catalog_audit)
    )
    return bool(
        report.get("schema_version") == ROUTE_VERIFIER_SCHEMA
        and report.get("accepted") is True
        and str(report.get("route_status") or "").strip().lower() == "solved"
        and report.get("target_match") is True
        and audit.get("target_match") is True
        and route_count > 0
        and accepted_route_count > 0
        and accepted_route_count <= route_count
        and rejected_route_count >= 0
        and report.get("best_route_rank") is not None
        and best_route_step_count > 0
        and materialization_matches
    )


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
    envelope = dict(chemenzy_result or {})
    result = dict(envelope.get("result") or envelope)
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
    stock_catalog_context = _effective_stock_catalog_context(envelope, result, routes)
    stock_catalog_audit = _independent_stock_catalog_audit(
        routes,
        stock_catalog_context=stock_catalog_context,
    )
    independent_stock_hits = {
        str(key): bool((value or {}).get("in_stock"))
        for key, value in dict(stock_catalog_audit.get("terminal_evidence") or {}).items()
        if isinstance(value, dict)
    }
    for route in routes:
        route_report = _verify_one_route(
            route,
            target_fp=target_fp,
            target_canonical=target_canonical,
            case_id=case_id,
            max_simple_terminal_heavy_atoms=max_simple_terminal_heavy_atoms,
            advanced_terminal_similarity=advanced_terminal_similarity,
            large_atom_jump_heavy_atoms=large_atom_jump_heavy_atoms,
            independent_stock_hits=independent_stock_hits,
            stock_catalog_audit=stock_catalog_audit,
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
    if any("missing_route_steps" in row.get("reasons", []) for row in rejected):
        reasons.append("missing_route_steps")
    if any("missing_route_target_product" in row.get("reasons", []) for row in rejected):
        reasons.append("missing_route_target_product")
    if any("terminal_stock_status_unproven" in row.get("reasons", []) for row in rejected):
        reasons.append("terminal_stock_status_unproven")
    if any("terminal_stock_status_conflict" in row.get("reasons", []) for row in rejected):
        reasons.append("terminal_stock_status_conflict")
    if any("stock_catalog_binding_unverifiable" in row.get("reasons", []) for row in rejected):
        reasons.append("stock_catalog_binding_unverifiable")
    if any("terminal_summary_materialization_mismatch" in row.get("reasons", []) for row in rejected):
        reasons.append("terminal_summary_materialization_mismatch")
    if any("hidden_nonstock_reactants" in row.get("reasons", []) for row in rejected):
        reasons.append("hidden_nonstock_reactants")
    if any("large_atom_jump" in row.get("reasons", []) for row in rejected):
        reasons.append("large_atom_jump")
    if any("element_inventory_not_conserved" in row.get("reasons", []) for row in rejected):
        reasons.append("element_inventory_not_conserved")
    if any("advanced_same_scaffold_terminal" in row.get("reasons", []) for row in rejected):
        reasons.append("advanced_same_scaffold_terminal")
    if any("route_target_product_mismatch" in row.get("reasons", []) for row in rejected):
        reasons.append("route_target_product_mismatch")
    if any("disconnected_route_steps" in row.get("reasons", []) for row in rejected):
        reasons.append("disconnected_route_steps")
    target_match = bool(target_audit.get("target_match"))
    if not target_match:
        reasons.append("target_equivalence_mismatch")

    final_accepted = bool(accepted) and target_match
    all_reasons = sorted(set(reasons))
    best_route_rank = accepted[0].get("route_rank") if final_accepted else None
    accepted_route = next(
        (
            _materialized_route_record(route)
            for route in routes
            if final_accepted and int(route.get("route_rank") or 0) == best_route_rank
        ),
        {},
    )
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
        # On a solved report, rejected sibling diagnostics are warnings rather
        # than contradictory proof blockers. Strict consumers require the
        # solved report's reasons list to be explicitly empty.
        reasons=[] if final_accepted else all_reasons,
        warnings=all_reasons if final_accepted else [],
        route_count=len(routes),
        accepted_route_count=len(accepted) if target_match else 0,
        rejected_route_count=(len(rejected) + (len(accepted) if not target_match else 0)),
        rejected_route_summary=_compact_route_reports(rejected),
        rejected_terminal_list=_unique_terminal_rejections(rejected_terminals),
        failure_events=failure_events[:50],
        best_route_rank=best_route_rank,
        best_route_step_count=int(accepted[0].get("n_steps") or 0) if final_accepted else 0,
        accepted_route=accepted_route,
        accepted_route_audit=dict(accepted[0]) if final_accepted else {},
        stock_catalog_audit=stock_catalog_audit,
        verification_policy={
            "schema_version": "route_verifier_policy.v1",
            "max_simple_terminal_heavy_atoms": int(max_simple_terminal_heavy_atoms),
            "advanced_terminal_similarity": float(advanced_terminal_similarity),
            "large_atom_jump_heavy_atoms": int(large_atom_jump_heavy_atoms),
        },
        target_match=target_match,
        target_equivalence_audit={
            **target_audit,
            "route_candidate_accepted_count_before_target_match": len(accepted),
        },
    )
    return report.to_dict()


def _strict_verification_policy(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or value.get("schema_version") != "route_verifier_policy.v1":
        return None
    try:
        max_terminal = int(value.get("max_simple_terminal_heavy_atoms"))
        similarity = float(value.get("advanced_terminal_similarity"))
        jump = int(value.get("large_atom_jump_heavy_atoms"))
    except (TypeError, ValueError):
        return None
    # Solved-proof consumers may accept stricter caller settings, never looser
    # settings that would bypass the production verifier defaults.
    if not (1 <= max_terminal <= 24 and 0.0 < similarity <= 0.5 and 1 <= jump <= 15):
        return None
    return {
        "max_simple_terminal_heavy_atoms": max_terminal,
        "advanced_terminal_similarity": similarity,
        "large_atom_jump_heavy_atoms": jump,
    }


def _materialized_route_record(route: dict[str, Any]) -> dict[str, Any]:
    return {
        "route_rank": int(route.get("route_rank") or 0),
        "score": route.get("score"),
        "steps": [dict(step) for step in route.get("steps") or [] if isinstance(step, dict)],
        "metrics": dict(route.get("metrics") or {}),
    }


def _verify_one_route(
    route: dict[str, Any],
    *,
    target_fp: Any,
    target_canonical: str,
    case_id: str,
    max_simple_terminal_heavy_atoms: int,
    advanced_terminal_similarity: float,
    large_atom_jump_heavy_atoms: int,
    independent_stock_hits: dict[str, bool],
    stock_catalog_audit: dict[str, Any],
) -> dict[str, Any]:
    route_rank = int(route.get("route_rank") or 0)
    steps = [dict(step) for step in route.get("steps") or [] if isinstance(step, dict)]
    terminals = _materialized_terminal_reactants(steps)
    stock_catalog_binding_valid = _stock_binding_valid_for_terminals(
        stock_catalog_audit,
        terminals=terminals,
    )
    reasons: list[str] = []
    rejected_terminals: list[dict[str, Any]] = []
    failure_events: list[dict[str, Any]] = []

    if not steps:
        reasons.append("missing_route_steps")

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

    jump_audit = _large_atom_jump_audit(steps, threshold=large_atom_jump_heavy_atoms)
    jumps = list(jump_audit["rejected_jumps"])
    if jumps:
        reasons.append("large_atom_jump")
        failure_events.append(_failure_event(case_id, route_rank, "large_atom_jump", {"jumps": jumps[:5]}))

    element_violations = _element_inventory_violations(steps)
    if element_violations:
        reasons.append("element_inventory_not_conserved")
        failure_events.append(
            _failure_event(
                case_id,
                route_rank,
                "element_inventory_not_conserved",
                {"violations": element_violations[:5]},
            )
        )

    if not terminals:
        reasons.append("missing_terminal_reactants")
    if not stock_catalog_binding_valid:
        reasons.append("stock_catalog_binding_unverifiable")
    terminal_stock_audit = _terminal_stock_audit(
        route,
        steps=steps,
        terminals=terminals,
        independent_stock_hits=independent_stock_hits,
    )
    if terminal_stock_audit["unproven"]:
        reasons.append("terminal_stock_status_unproven")
    if terminal_stock_audit["conflicts"]:
        reasons.append("terminal_stock_status_conflict")
    if not terminal_stock_audit["summary_matches_materialized_leaves"]:
        reasons.append("terminal_summary_materialization_mismatch")
    if target_canonical and steps:
        route_products = [_canonical_smiles(_step_product(step)) for step in steps if _step_product(step)]
        if not route_products:
            reasons.append("missing_route_target_product")
        elif target_canonical not in route_products:
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
        if not _route_steps_connect_to_target(steps, target_canonical=target_canonical):
            reasons.append("disconnected_route_steps")
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
        # The backend-reported n_steps is metadata, not materialization proof.
        "n_steps": len(steps),
        "claimed_n_steps": route.get("n_steps"),
        "reasons": sorted(set(reasons)),
        "terminal_count": len(terminals),
        "terminal_stock_proven_count": int(terminal_stock_audit["proven_count"]),
        "terminal_stock_unproven_count": len(terminal_stock_audit["unproven"]),
        "hidden_nonstock_count": len(hidden_nonstock),
        "large_atom_jump_count": len(jumps),
        "mapped_convergent_assembly_count": len(jump_audit["validated_convergences"]),
        "mapped_convergent_assembly_audit": list(jump_audit["validated_convergences"]),
        "rejected_terminals": rejected_terminals,
        "failure_events": failure_events,
    }


def _materialized_terminal_reactants(steps: list[dict[str, Any]]) -> list[str]:
    generated = {_canonical_smiles(_step_product(step)) for step in steps if _step_product(step)}
    generated.discard("")
    seen: list[str] = []
    seen_canonical: set[str] = set()
    for step in steps:
        for smiles in _step_reactants(step):
            text = str(smiles or "")
            canonical = _canonical_smiles(text)
            if text and canonical and canonical not in generated and canonical not in seen_canonical:
                seen.append(text)
                seen_canonical.add(canonical)
    return seen


def _effective_stock_catalog_context(
    envelope: dict[str, Any],
    result: dict[str, Any],
    routes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Extract the stock boundary actually carried by this request/result.

    A verifier must not silently substitute a different, broader catalog.
    Multiple independently emitted descriptions are therefore required to
    agree.  Catalog files are resolved and hashed later; booleans in the raw
    payload never constitute stock evidence.
    """

    context = dict(envelope.get("stock_catalog_context") or result.get("stock_catalog_context") or {})
    request = dict(
        envelope.get("request")
        or envelope.get("request_payload")
        or result.get("request")
        or result.get("request_payload")
        or {}
    )
    ui_metadata = dict(result.get("ui_metadata") or {})
    failure_search = dict((result.get("failure_analysis") or {}).get("search_config") or {})
    result_search = dict(result.get("search_config") or {})
    boundary = dict(
        request.get("harness_search_boundary")
        or envelope.get("harness_search_boundary")
        or result.get("harness_search_boundary")
        or {}
    )

    name_sources: list[dict[str, Any]] = []
    for source, value in (
        ("stock_catalog_context.effective_stock_names", context.get("effective_stock_names")),
        ("request.stock_names", request.get("stock_names")),
        ("harness_search_boundary.effective_stock_names", boundary.get("effective_stock_names")),
        ("result.stock_names", result.get("stock_names")),
        ("result.ui_metadata.stock_names", ui_metadata.get("stock_names")),
        ("result.search_config.stock_names", result_search.get("stock_names")),
        ("result.failure_analysis.search_config.stock_names", failure_search.get("stock_names")),
    ):
        names = _string_list(value)
        if names:
            name_sources.append({"source": source, "names": names})
    for route_index, route in enumerate(routes):
        route_metrics = dict(route.get("metrics") or {})
        for source, value in (
            (f"routes[{route_index}].stock_names", route.get("stock_names")),
            (f"routes[{route_index}].metrics.stock_names", route_metrics.get("stock_names")),
        ):
            names = _string_list(value)
            if names:
                name_sources.append({"source": source, "names": names})

    binding_rows: list[dict[str, str]] = []
    for source, value in (
        ("stock_catalog_context.catalog_bindings", context.get("catalog_bindings")),
        ("stock_catalog_context.effective_catalogs", context.get("effective_catalogs")),
        ("request.stock_catalogs", request.get("stock_catalogs")),
        ("result.stock_catalogs", result.get("stock_catalogs")),
        ("result.ui_metadata.stock_catalogs", ui_metadata.get("stock_catalogs")),
    ):
        binding_rows.extend(_normalise_catalog_bindings(value, source=source))

    distinct_name_sets = {
        tuple(sorted(row["names"]))
        for row in name_sources
    }
    conflicts: list[str] = []
    if len(distinct_name_sets) > 1:
        conflicts.append("effective_stock_names_conflict")
    effective_names = list(name_sources[0]["names"]) if name_sources else []
    binding_names = list(dict.fromkeys(row["name"] for row in binding_rows if row.get("name")))
    if not effective_names and binding_names:
        effective_names = binding_names
    elif binding_names and set(binding_names) - set(effective_names):
        conflicts.append("catalog_binding_name_not_effective")

    return {
        "schema_version": "effective_stock_catalog_context.v1",
        "effective_stock_names": effective_names,
        "name_sources": name_sources,
        "catalog_bindings": binding_rows,
        "conflicts": sorted(set(conflicts)),
        "route_count": len(routes),
    }


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        return []
    return list(dict.fromkeys(str(item or "").strip() for item in values if str(item or "").strip()))


def _normalise_catalog_bindings(value: Any, *, source: str) -> list[dict[str, str]]:
    if isinstance(value, dict):
        if any(key in value for key in ("name", "catalog_name", "stock_name")):
            rows = [value]
        else:
            rows = [
                {"name": name, **(dict(binding) if isinstance(binding, dict) else {"path": binding})}
                for name, binding in value.items()
            ]
    elif isinstance(value, list):
        rows = [dict(row) for row in value if isinstance(row, dict)]
    else:
        rows = []
    out: list[dict[str, str]] = []
    for row in rows:
        name = str(row.get("name") or row.get("catalog_name") or row.get("stock_name") or "").strip()
        path = str(row.get("path") or row.get("catalog_path") or row.get("actual_path") or "").strip()
        sha256 = str(row.get("sha256") or row.get("catalog_sha256") or "").strip().lower()
        if name:
            out.append({"name": name, "path": path, "sha256": sha256, "source": source})
    return out


def _independent_stock_catalog_audit(
    routes: list[dict[str, Any]],
    *,
    stock_catalog_context: dict[str, Any],
) -> dict[str, Any]:
    """Recheck terminal leaves against the effective, identifiable catalogs."""
    terminals = {
        _canonical_smiles(item)
        for route in routes
        for item in _materialized_terminal_reactants(
            [dict(step) for step in route.get("steps") or [] if isinstance(step, dict)]
        )
        if _canonical_smiles(item)
    }
    evidence: dict[str, dict[str, Any]] = {}
    common_catalog = {
        "catalog_name": _COMMON_CATALOG_NAME,
        "catalog_id": f"sha256:{_COMMON_CATALOG_SHA256}",
        "sha256": _COMMON_CATALOG_SHA256,
        "entry_count": len(_TRUSTED_COMMON_STOCK),
        "lookup_basis": "embedded_canonical_isomeric_smiles_set",
        "catalog_role": "independent_common_commodity_supplement",
    }
    for terminal in sorted(terminals):
        if terminal in _TRUSTED_COMMON_STOCK:
            evidence[terminal] = {
                "in_stock": True,
                "catalog_name": _COMMON_CATALOG_NAME,
                "catalog_id": common_catalog["catalog_id"],
                "catalog_sha256": _COMMON_CATALOG_SHA256,
                "lookup_basis": "canonical_isomeric_smiles",
                "catalog_role": "independent_common_commodity_supplement",
            }

    effective_names = _string_list(stock_catalog_context.get("effective_stock_names"))
    configured_catalogs = _load_chemenzy_stock_catalogs(_CHEMENZY_STOCK_CONFIG)
    supplied_bindings = [
        dict(row)
        for row in stock_catalog_context.get("catalog_bindings") or []
        if isinstance(row, dict)
    ]
    resolved_catalogs: list[dict[str, Any]] = []
    binding_failures = [str(item) for item in stock_catalog_context.get("conflicts") or []]
    for name in effective_names:
        record, failures = _resolve_effective_catalog(
            name,
            configured_catalogs=configured_catalogs,
            supplied_bindings=supplied_bindings,
        )
        binding_failures.extend(failures)
        if record:
            resolved_catalogs.append(record)

    missing = sorted(terminals - set(evidence))
    for terminal in missing:
        hits: list[dict[str, Any]] = []
        for catalog in resolved_catalogs:
            if _catalog_contains(Path(catalog["path"]), terminal):
                hits.append(catalog)
        if hits:
            hit = hits[0]
            evidence[terminal] = {
                "in_stock": True,
                "catalog_name": hit["catalog_name"],
                "catalog_id": hit["catalog_id"],
                "catalog_path": hit["path"],
                "catalog_sha256": hit["sha256"],
                "lookup_basis": "exact_canonical_smiles_first_csv_field",
            }
        else:
            evidence[terminal] = {
                "in_stock": False,
                "catalog_id": "effective_catalog_no_hit" if resolved_catalogs else "catalog_unavailable",
                "lookup_basis": "no_independent_hit",
            }

    proven = sum(1 for row in evidence.values() if row.get("in_stock") is True)
    binding_failures = sorted(set(binding_failures))
    if effective_names:
        catalog_binding_valid = not binding_failures and len(resolved_catalogs) == len(effective_names)
        binding_status = "verified" if catalog_binding_valid else "unverifiable"
    else:
        common_only = bool(terminals) and terminals.issubset(_TRUSTED_COMMON_STOCK)
        # The embedded common catalog is itself recheckable.  Whether it is
        # sufficient is route-local: an unrelated rejected sibling with a
        # non-common leaf must not contaminate a valid common-only route.
        catalog_binding_valid = not binding_failures
        binding_status = (
            "common_catalog_only"
            if common_only
            else "common_catalog_available_effective_catalog_missing"
        )

    revalidation_context = {
        "schema_version": "effective_stock_catalog_context.v1",
        "effective_stock_names": effective_names,
        "catalog_bindings": [
            {
                "name": row["catalog_name"],
                "path": row["path"],
                "sha256": row["sha256"],
            }
            for row in resolved_catalogs
        ],
    }
    return {
        "schema_version": "independent_stock_catalog_audit.v1",
        "terminal_count": len(terminals),
        "proven_terminal_count": proven,
        "all_terminals_proven": bool(terminals and proven == len(terminals)),
        "terminal_evidence": evidence,
        "catalog_binding_valid": catalog_binding_valid,
        "catalog_binding_status": binding_status,
        "binding_failures": binding_failures,
        "effective_stock_names": effective_names,
        "effective_catalogs": resolved_catalogs,
        "common_catalog": common_catalog,
        # Kept as a compatibility view; it is now the first *effective*
        # catalog, never an unconditional Zinc fallback.
        "vendor_catalog": dict(resolved_catalogs[0]) if resolved_catalogs else {},
        "revalidation_context": revalidation_context,
        "policy": "effective_search_catalog_plus_explicit_common_commodity_supplement",
    }


def _stock_binding_valid_for_terminals(
    audit: dict[str, Any],
    *,
    terminals: list[str],
) -> bool:
    if audit.get("catalog_binding_valid") is not True:
        return False
    if _string_list(audit.get("effective_stock_names")):
        return True
    canonical = {_canonical_smiles(item) for item in terminals if _canonical_smiles(item)}
    return bool(canonical and canonical.issubset(_TRUSTED_COMMON_STOCK))


def _load_chemenzy_stock_catalogs(config_path: Path) -> dict[str, Path]:
    """Load the named stock paths without importing the vendor runtime."""
    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    in_stocks = False
    out: dict[str, Path] = {}
    base = config_path.parent.parent
    for line in lines:
        if not in_stocks:
            if re.match(r"^stocks\s*:\s*(?:#.*)?$", line):
                in_stocks = True
            continue
        if line and not line[0].isspace():
            break
        match = re.match(r"^\s{2}([^:#][^:]*)\s*:\s*(.*?)\s*(?:#.*)?$", line)
        if not match:
            continue
        name = match.group(1).strip().strip("'\"")
        raw_path = match.group(2).strip().strip("'\"")
        if not name or not raw_path:
            continue
        path = Path(raw_path).expanduser()
        out[name] = path.resolve() if path.is_absolute() else (base / path).resolve()
    return out


def _resolve_effective_catalog(
    name: str,
    *,
    configured_catalogs: dict[str, Path],
    supplied_bindings: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    declarations = [row for row in supplied_bindings if str(row.get("name") or "") == name]
    configured_path = configured_catalogs.get(name)
    declared_paths = {
        str(_resolve_catalog_path(str(row.get("path") or "")))
        for row in declarations
        if str(row.get("path") or "").strip()
    }
    declared_hashes = {
        str(row.get("sha256") or "").strip().lower()
        for row in declarations
        if str(row.get("sha256") or "").strip()
    }
    if len(declared_paths) > 1:
        failures.append(f"catalog_path_conflict:{name}")
    if len(declared_hashes) > 1:
        failures.append(f"catalog_sha256_conflict:{name}")

    if configured_path is not None:
        path = configured_path.resolve()
        if declared_paths and declared_paths != {str(path)}:
            failures.append(f"configured_catalog_path_mismatch:{name}")
    elif len(declared_paths) == 1 and len(declared_hashes) == 1:
        path = Path(next(iter(declared_paths)))
    else:
        failures.append(f"catalog_not_resolvable:{name}")
        return {}, failures

    if not path.is_file():
        failures.append(f"catalog_file_missing:{name}")
        return {}, failures
    stat = path.stat()
    if stat.st_size <= 0:
        failures.append(f"catalog_file_empty:{name}")
        return {}, failures
    cache_prefix = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
    digest = _catalog_sha256(path, cache_prefix=cache_prefix)
    if declared_hashes and declared_hashes != {digest}:
        failures.append(f"catalog_sha256_mismatch:{name}")
    if failures:
        return {}, failures
    return {
        "catalog_name": name,
        "catalog_id": f"{name}@sha256:{digest}",
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "sha256": digest,
        "lookup_basis": "exact_canonical_smiles_first_csv_field",
        "binding_source": "chem_enzy_config" if configured_path is not None else "explicit_request_binding",
    }, []


def _resolve_catalog_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (_REPO_ROOT / path).resolve()


def _catalog_contains(path: Path, canonical_smiles: str) -> bool:
    stat = path.stat()
    cache_prefix = (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
    cache_key = (*cache_prefix, canonical_smiles)
    cached = _CATALOG_HIT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            hit = _mmap_has_exact_first_field(mapped, canonical_smiles.encode("utf-8"))
    _CATALOG_HIT_CACHE[cache_key] = hit
    return hit


def _catalog_audit_signature(audit: dict[str, Any]) -> tuple[Any, ...]:
    catalogs = tuple(
        sorted(
            (
                str(row.get("catalog_name") or ""),
                str(row.get("path") or ""),
                str(row.get("sha256") or ""),
            )
            for row in audit.get("effective_catalogs") or []
            if isinstance(row, dict)
        )
    )
    common = dict(audit.get("common_catalog") or {})
    return (
        bool(audit.get("catalog_binding_valid")),
        tuple(_string_list(audit.get("effective_stock_names"))),
        catalogs,
        str(common.get("catalog_name") or ""),
        str(common.get("sha256") or ""),
    )


def _mmap_has_exact_first_field(mapped: mmap.mmap, value: bytes) -> bool:
    if not value:
        return False
    terminators = {b"\n", b"\r", b","}
    if mapped[: len(value)] == value and (
        len(mapped) == len(value) or mapped[len(value) : len(value) + 1] in terminators
    ):
        return True
    start = 0
    needle = b"\n" + value
    while True:
        index = mapped.find(needle, start)
        if index < 0:
            return False
        after = index + len(needle)
        if after == len(mapped) or mapped[after : after + 1] in terminators:
            return True
        start = index + 1


def _catalog_sha256(path: Path, *, cache_prefix: tuple[str, int, int]) -> str:
    cached = _CATALOG_DIGEST_CACHE.get(cache_prefix)
    if cached:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    _CATALOG_DIGEST_CACHE[cache_prefix] = value
    return value


def _terminal_stock_audit(
    route: dict[str, Any],
    *,
    steps: list[dict[str, Any]],
    terminals: list[str],
    independent_stock_hits: dict[str, bool],
) -> dict[str, Any]:
    metrics = dict(route.get("metrics") or {})
    summary_terminals = {
        _canonical_smiles(str(item or ""))
        for item in metrics.get("terminal_reactants") or []
        if str(item or "").strip()
    }
    materialized = {_canonical_smiles(item) for item in terminals if _canonical_smiles(item)}
    status_maps = [
        dict(metrics.get("terminal_stock_status") or {}),
        *[dict(step.get("stock_status") or {}) for step in steps],
    ]
    unproven: list[str] = []
    conflicts: list[str] = []
    proven_count = 0
    for terminal in terminals:
        canonical = _canonical_smiles(terminal)
        independently_in_stock = independent_stock_hits.get(canonical) is True
        values = [
            _stock_value(status, terminal)
            for status in status_maps
            if _stock_value(status, terminal) is not None
        ]
        if any(value is False for value in values) and (any(value is True for value in values) or independently_in_stock):
            conflicts.append(terminal)
            continue
        if values and all(value is True for value in values) and independently_in_stock:
            proven_count += 1
        else:
            unproven.append(terminal)
    return {
        "proven_count": proven_count,
        "unproven": unproven,
        "conflicts": conflicts,
        "summary_matches_materialized_leaves": not summary_terminals or summary_terminals == materialized,
    }


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
    product = str(step.get("product") or step.get("product_smiles") or "")
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


def _large_atom_jump_audit(steps: list[dict[str, Any]], *, threshold: int) -> dict[str, list[dict[str, Any]]]:
    rejected: list[dict[str, Any]] = []
    validated: list[dict[str, Any]] = []
    for step in steps:
        reactants = list(dict.fromkeys(_canonical_smiles(item) for item in _step_reactants(step)))
        reactants = [item for item in reactants if item]
        reactant_sizes = [_heavy_atoms(smiles) for smiles in reactants]
        # Summing every listed component lets a route manufacture atom balance
        # by padding the reactant list with repeated tiny fragments.  The
        # conservative proof gate measures growth from the largest materialized
        # precursor scaffold instead.  Large convergent assemblies therefore
        # require a downstream reaction/atom-mapping validator rather than
        # becoming solved solely from backend route syntax.
        reactant_heavy = max(reactant_sizes, default=0)
        product_heavy = _heavy_atoms(_step_product(step))
        delta = product_heavy - reactant_heavy
        if delta >= threshold:
            provenance = _mapped_convergent_assembly_audit(step)
            row = {
                "step_index": step.get("index"),
                "reactant_heavy_atoms": reactant_heavy,
                "largest_reactant_heavy_atoms": reactant_heavy,
                "unique_reactant_count": len(reactants),
                "unique_reactant_heavy_atoms_total": sum(reactant_sizes),
                "product_heavy_atoms": product_heavy,
                "delta_heavy_atoms": delta,
                "atom_provenance_audit": provenance,
            }
            if provenance.get("validated") is True:
                validated.append(row)
            else:
                rejected.append(row)
    return {"rejected_jumps": rejected, "validated_convergences": validated}


def _large_atom_jumps(steps: list[dict[str, Any]], *, threshold: int) -> list[dict[str, Any]]:
    """Compatibility wrapper returning only unexplained large jumps."""
    return list(_large_atom_jump_audit(steps, threshold=threshold)["rejected_jumps"])


def _mapped_convergent_assembly_audit(step: dict[str, Any]) -> dict[str, Any]:
    """Recompute the narrow atom-provenance exception for a large assembly.

    Self-reported flags are deliberately ignored.  The exception requires a
    complete atom-mapped reaction whose unmapped structures are exactly bound
    to this step, globally unique heavy-atom maps, element-preserving product
    provenance, and a newly formed bond joining atoms from distinct reactant
    components.
    """

    reasons: list[str] = []
    mapped_reaction, source = _mapped_reaction_from_step(step)
    if not mapped_reaction or ">>" not in mapped_reaction:
        return {
            "schema_version": "deterministic_atom_provenance_audit.v1",
            "validated": False,
            "reasons": ["complete_atom_mapped_reaction_missing"],
        }
    lhs, rhs = mapped_reaction.split(">>", 1)
    lhs_texts = [item.strip() for item in lhs.split(".") if item.strip()]
    rhs_texts = [item.strip() for item in rhs.split(".") if item.strip()]
    listed_texts = _listed_reactant_components(step)
    if not lhs_texts or len(rhs_texts) != 1:
        reasons.append("mapped_reaction_component_schema_invalid")
    lhs_mols = [_mol(item) for item in lhs_texts]
    rhs_mol = _mol(rhs_texts[0]) if len(rhs_texts) == 1 else None
    listed_mols = [_mol(item) for item in listed_texts]
    product_mol = _mol(_step_product(step))
    if (
        rhs_mol is None
        or product_mol is None
        or not lhs_mols
        or any(mol is None for mol in lhs_mols)
        or not listed_mols
        or any(mol is None for mol in listed_mols)
    ):
        reasons.append("mapped_reaction_structure_invalid")
    if reasons:
        return {
            "schema_version": "deterministic_atom_provenance_audit.v1",
            "validated": False,
            "mapping_source": source,
            "reasons": sorted(set(reasons)),
        }

    assert rhs_mol is not None and product_mol is not None
    valid_lhs_mols = [mol for mol in lhs_mols if mol is not None]
    valid_listed_mols = [mol for mol in listed_mols if mol is not None]
    lhs_structures = Counter(_canonical_mol_without_atom_maps(mol) for mol in valid_lhs_mols)
    listed_structures = Counter(_canonical_mol_without_atom_maps(mol) for mol in valid_listed_mols)
    if lhs_structures != listed_structures:
        reasons.append("mapped_reaction_reactants_not_bound_to_step")
    if _canonical_mol_without_atom_maps(rhs_mol) != _canonical_mol_without_atom_maps(product_mol):
        reasons.append("mapped_reaction_product_not_bound_to_step")

    reactant_map_atoms: dict[int, tuple[int, int]] = {}
    duplicate_reactant_maps: set[int] = set()
    reactant_unmapped_heavy = 0
    for component_index, mol in enumerate(valid_lhs_mols):
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue
            atom_map = int(atom.GetAtomMapNum())
            if atom_map <= 0:
                reactant_unmapped_heavy += 1
                continue
            if atom_map in reactant_map_atoms:
                duplicate_reactant_maps.add(atom_map)
            else:
                reactant_map_atoms[atom_map] = (int(atom.GetAtomicNum()), component_index)

    product_map_atoms: dict[int, int] = {}
    duplicate_product_maps: set[int] = set()
    product_unmapped_heavy = 0
    for atom in rhs_mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        atom_map = int(atom.GetAtomMapNum())
        if atom_map <= 0:
            product_unmapped_heavy += 1
            continue
        if atom_map in product_map_atoms:
            duplicate_product_maps.add(atom_map)
        else:
            product_map_atoms[atom_map] = int(atom.GetAtomicNum())

    if reactant_unmapped_heavy or product_unmapped_heavy:
        reasons.append("heavy_atom_mapping_incomplete")
    if duplicate_reactant_maps or duplicate_product_maps:
        reasons.append("atom_mapping_not_unique")
    new_product_maps = sorted(set(product_map_atoms) - set(reactant_map_atoms))
    if new_product_maps:
        reasons.append("product_heavy_atom_without_reactant_provenance")
    element_mismatches = sorted(
        atom_map
        for atom_map, atomic_number in product_map_atoms.items()
        if atom_map in reactant_map_atoms and reactant_map_atoms[atom_map][0] != atomic_number
    )
    if element_mismatches:
        reasons.append("mapped_atom_element_changed")

    contributing_components = {
        reactant_map_atoms[atom_map][1]
        for atom_map in product_map_atoms
        if atom_map in reactant_map_atoms
    }
    if len(contributing_components) < 2:
        reasons.append("not_a_multi_reactant_convergence")

    cross_component_new_bonds = 0
    for bond in rhs_mol.GetBonds():
        begin_map = int(bond.GetBeginAtom().GetAtomMapNum())
        end_map = int(bond.GetEndAtom().GetAtomMapNum())
        if begin_map not in reactant_map_atoms or end_map not in reactant_map_atoms:
            continue
        if reactant_map_atoms[begin_map][1] != reactant_map_atoms[end_map][1]:
            cross_component_new_bonds += 1
    if cross_component_new_bonds <= 0:
        reasons.append("cross_component_product_bond_missing")

    return {
        "schema_version": "deterministic_atom_provenance_audit.v1",
        "validated": not reasons,
        "mapping_source": source,
        "reasons": sorted(set(reasons)),
        "listed_reactant_component_count": len(valid_listed_mols),
        "mapped_reactant_heavy_atom_count": len(reactant_map_atoms),
        "mapped_product_heavy_atom_count": len(product_map_atoms),
        "contributing_reactant_component_count": len(contributing_components),
        "cross_component_product_bond_count": cross_component_new_bonds,
        "new_product_atom_maps": new_product_maps[:20],
        "element_mismatch_atom_maps": element_mismatches[:20],
        "duplicate_reactant_atom_maps": sorted(duplicate_reactant_maps)[:20],
        "duplicate_product_atom_maps": sorted(duplicate_product_maps)[:20],
    }


def _mapped_reaction_from_step(step: dict[str, Any]) -> tuple[str, str]:
    for field_name in ("atom_mapped_reaction_smiles", "mapped_reaction_smiles"):
        value = str(step.get(field_name) or "").strip()
        if value:
            return value, field_name
    reaction = str(step.get("reaction_smiles") or "").strip()
    if reaction and ":" in reaction:
        return reaction, "reaction_smiles"
    provenance = step.get("atom_provenance")
    if isinstance(provenance, dict):
        for field_name in ("atom_mapped_reaction_smiles", "mapped_reaction_smiles", "reaction_smiles"):
            value = str(provenance.get(field_name) or "").strip()
            if value:
                return value, f"atom_provenance.{field_name}"
    return "", ""


def _listed_reactant_components(step: dict[str, Any]) -> list[str]:
    """Select one non-overlapping materialized reactant representation."""
    for field_name in ("reactant_smiles", "precursor_smiles", "reactants"):
        if field_name not in step or not step.get(field_name):
            continue
        raw = step.get(field_name)
        if isinstance(raw, str):
            return [item for item in raw.split(".") if item]
        if isinstance(raw, list):
            return [str(item or "") for item in raw if str(item or "").strip()]
    values = [str(step.get("main_reactant") or step.get("main_reactant_smiles") or "")]
    raw_aux = step.get("aux_reactants") or []
    if isinstance(raw_aux, str):
        values.extend(item for item in raw_aux.split(".") if item)
    elif isinstance(raw_aux, list):
        values.extend(str(item or "") for item in raw_aux)
    return [item for item in values if item]


def _canonical_mol_without_atom_maps(mol: Any) -> str:
    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(copy, isomericSmiles=True)


def _element_inventory_violations(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reject products containing non-hydrogen atoms absent from precursors.

    This is deliberately a necessary, not sufficient, reaction check.  It
    prevents equal-size element transmutation (for example ``NNN -> CCO``)
    while allowing reactant atoms to leave as by-products.  Hydrogen is omitted
    because it is commonly implicit and supplied by work-up.
    """
    violations: list[dict[str, Any]] = []
    for step in steps:
        product = _mol(_step_product(step))
        reactants = [_mol(item) for item in _step_reactants(step)]
        if product is None or not reactants or any(mol is None for mol in reactants):
            continue
        available: Counter[int] = Counter()
        for mol in reactants:
            available.update(atom.GetAtomicNum() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1)
        required: Counter[int] = Counter(
            atom.GetAtomicNum() for atom in product.GetAtoms() if atom.GetAtomicNum() != 1
        )
        deficits = {
            Chem.GetPeriodicTable().GetElementSymbol(atomic_number): count - available.get(atomic_number, 0)
            for atomic_number, count in required.items()
            if count > available.get(atomic_number, 0)
        }
        if deficits:
            violations.append(
                {
                    "step_index": step.get("index"),
                    "missing_product_elements": deficits,
                    "reactant_inventory": {
                        Chem.GetPeriodicTable().GetElementSymbol(number): count
                        for number, count in sorted(available.items())
                    },
                }
            )
    return violations


def _route_steps_connect_to_target(
    steps: list[dict[str, Any]],
    *,
    target_canonical: str,
) -> bool:
    products = {
        _canonical_smiles(_step_product(step))
        for step in steps
        if _canonical_smiles(_step_product(step))
    }
    frontier = {target_canonical}
    remaining = set(range(len(steps)))
    progressed = True
    while progressed:
        progressed = False
        for index in list(remaining):
            product = _canonical_smiles(_step_product(steps[index]))
            if not product or product not in frontier:
                continue
            frontier.discard(product)
            frontier.update(
                canonical
                for canonical in (_canonical_smiles(item) for item in _step_reactants(steps[index]))
                if canonical
            )
            remaining.remove(index)
            progressed = True
    # Every route step must be consumed exactly downstream of the target and
    # the final frontier must contain only true leaves.  If a generated product
    # (especially the target itself) reappears, the route is cyclic rather than
    # a stock-to-target synthesis.
    return bool(not remaining and not (frontier & products))


def _step_reactants(step: dict[str, Any]) -> list[str]:
    values = [str(step.get("main_reactant") or step.get("main_reactant_smiles") or "")]
    for field_name in ("reactant_smiles", "precursor_smiles", "reactants", "aux_reactants"):
        raw = step.get(field_name) or []
        if isinstance(raw, str):
            values.extend(item for item in raw.split(".") if item)
        elif isinstance(raw, list):
            values.extend(str(item or "") for item in raw)
    return [value for value in values if value]


def _compact_route_reports(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "route_rank": row.get("route_rank"),
            "score": row.get("score"),
            "n_steps": row.get("n_steps"),
            "claimed_n_steps": row.get("claimed_n_steps"),
            "reasons": list(row.get("reasons") or []),
            "terminal_count": row.get("terminal_count"),
            "terminal_stock_proven_count": row.get("terminal_stock_proven_count"),
            "terminal_stock_unproven_count": row.get("terminal_stock_unproven_count"),
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
