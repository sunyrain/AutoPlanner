"""Audit enzyme proposal quality before injecting candidates into search.

This script is intentionally candidate-level rather than route-level.  It
answers whether the enlarged enzyme candidate pool adds search-ready candidates
or just increases proposal noise.  Route-level benchmarks can then consume the
same scoring criteria.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import MolToInchiKey

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.cascade_search.bridge_retriever_v0 import BridgeRetrieverV0
from cascade_planner.cascade_search.enzyme_sp_verifier_v1 import EnzymeSPVerifierV1Scorer
from cascade_planner.cascadeboard.live_retro import build_live_retro_engine, retro_engine_cache_stats
from cascade_planner.cascadeboard.route_recovery import canonical_smiles
from cascade_planner.route_tree.proposals import ProposalContext, RetroEngineProposalTool
from cascade_planner.route_tree.schema import CandidateAction
from cascade_planner.route_tree.source_gate import source_group
from scripts.run_bridge_live_policy_benchmark_v0 import load_targets


RDLogger.DisableLog("rdApp.*")

DEFAULT_PACK_DIR = Path("data/bridge_pack_v0")
DEFAULT_PROBE_ROWS = Path("results/shared/bridge_gate_ablation_v0_20260527/live_enzyme_bridge_target_probe_rows.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/shared/enzyme_candidate_quality_audit_v0_20260528")
DEFAULT_SOURCES = ("enzyme_precedent", "v3_retrieval")
NATIVE_CHEMENZY_SOURCE = "chem_enzy_onestep"
ENZYME_LIKE_SOURCES = {"enzyme_precedent", "v3_retrieval", "enzyformer", "enzexpand", "retrorules"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit enzyme proposal candidate quality")
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--probe-rows", type=Path, default=DEFAULT_PROBE_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--positives", type=int, default=20)
    parser.add_argument("--negatives", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--bridge-top-k", type=int, default=8)
    parser.add_argument("--max-bridge-ec-contexts", type=int, default=2)
    parser.add_argument("--max-targets", type=int, default=40)
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help="Comma-separated proposal sources to audit. Use --include-native-chem-enzy for ChemEnzy one-step.",
    )
    parser.add_argument(
        "--include-native-chem-enzy",
        action="store_true",
        help="Also query ChemEnzy native one-step source. This may initialize heavy vendor models.",
    )
    parser.add_argument("--skip-sp-v1", action="store_true", help="Skip enzyme SP-v1 scoring.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sources = _source_list(args.sources)
    if args.include_native_chem_enzy and NATIVE_CHEMENZY_SOURCE not in sources:
        sources.append(NATIVE_CHEMENZY_SOURCE)
    targets = load_targets(args)
    if args.max_targets > 0:
        targets = targets[: int(args.max_targets)]

    old_chemenzy_flag = os.environ.get("AUTOPLANNER_ENABLE_CHEMENZY_ONESTEP_PROPOSALS")
    if args.include_native_chem_enzy:
        os.environ["AUTOPLANNER_ENABLE_CHEMENZY_ONESTEP_PROPOSALS"] = "1"
    retriever = BridgeRetrieverV0(args.pack_dir, scorer=None)
    blacklist = load_blacklist(args.pack_dir)
    scorer = None if args.skip_sp_v1 else EnzymeSPVerifierV1Scorer()
    try:
        live_engine = build_live_retro_engine() if _needs_live_engine(sources) else {}
    finally:
        if args.include_native_chem_enzy:
            if old_chemenzy_flag is None:
                os.environ.pop("AUTOPLANNER_ENABLE_CHEMENZY_ONESTEP_PROPOSALS", None)
            else:
                os.environ["AUTOPLANNER_ENABLE_CHEMENZY_ONESTEP_PROPOSALS"] = old_chemenzy_flag
    tool = RetroEngineProposalTool(live_engine)

    candidate_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for idx, target in enumerate(targets, start=1):
        smiles = str(target.get("target_smiles") or "")
        print(f"[{idx}/{len(targets)}] auditing {smiles}", flush=True)
        target_row, rows = audit_target(
            target,
            retriever=retriever,
            tool=tool,
            sources=sources,
            scorer=scorer,
            blacklist=blacklist,
            top_k=max(1, int(args.top_k)),
            bridge_top_k=max(1, int(args.bridge_top_k)),
            max_bridge_ec_contexts=max(0, int(args.max_bridge_ec_contexts)),
        )
        target_rows.append(target_row)
        candidate_rows.extend(rows)

    summary = summarize(candidate_rows, target_rows, sources=sources)
    report = {
        "schema_version": "enzyme_candidate_quality_audit.v0",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "inputs": {
            "pack_dir": str(args.pack_dir),
            "probe_rows": str(args.probe_rows),
            "output_dir": str(args.output_dir),
            "positives": int(args.positives),
            "negatives": int(args.negatives),
            "targets": len(targets),
            "top_k": int(args.top_k),
            "bridge_top_k": int(args.bridge_top_k),
            "max_bridge_ec_contexts": int(args.max_bridge_ec_contexts),
            "sources": sources,
            "include_native_chem_enzy": bool(args.include_native_chem_enzy),
            "sp_v1_enabled": scorer is not None,
        },
        "summary": summary,
        "conclusion": conclusion(summary, include_native=bool(args.include_native_chem_enzy)),
        "native_engine_cache_stats": retro_engine_cache_stats(live_engine) if live_engine else {},
    }

    candidates_path = args.output_dir / "enzyme_candidate_quality_rows.jsonl"
    targets_path = args.output_dir / "enzyme_candidate_quality_target_rows.jsonl"
    report_json = args.output_dir / "enzyme_candidate_quality_report.json"
    report_md = args.output_dir / "enzyme_candidate_quality_report.md"
    candidates_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in candidate_rows),
        encoding="utf-8",
    )
    targets_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in target_rows),
        encoding="utf-8",
    )
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    report_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"report": str(report_json), "rows": str(candidates_path), "conclusion": report["conclusion"]}, ensure_ascii=False, indent=2))


def audit_target(
    target: dict[str, Any],
    *,
    retriever: BridgeRetrieverV0,
    tool: RetroEngineProposalTool,
    sources: list[str],
    scorer: EnzymeSPVerifierV1Scorer | None,
    blacklist: set[str],
    top_k: int,
    bridge_top_k: int,
    max_bridge_ec_contexts: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_smiles = str(target.get("target_smiles") or "")
    label = int(target.get("label") or 0)
    bridge_hits = retriever.retrieve(target_smiles, top_k=bridge_top_k, require_verifier_pass=True)
    bridge_ec1s = _bridge_ec1s(bridge_hits)
    contexts = [{"context_id": "root_no_ec", "ec1": 0, "context_source": "root"}]
    contexts.extend(
        {"context_id": f"bridge_ec{ec1}", "ec1": int(ec1), "context_source": "bridge_retriever_v0"}
        for ec1 in bridge_ec1s[:max_bridge_ec_contexts]
    )
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for context in contexts:
        proposal_context = ProposalContext(
            depth=0,
            ec1=int(context["ec1"]),
            route_metadata={"bridge_hits": len(bridge_hits), "quality_audit_context": context["context_id"]},
        )
        for source in sources:
            if int(context["ec1"]) and source not in ENZYME_LIKE_SOURCES:
                continue
            try:
                raw_rows = tool._predict(source, tool.retro_engine.get(source), target_smiles, proposal_context, top_k=top_k)  # noqa: SLF001
            except Exception as exc:  # pragma: no cover - provider-specific runtime failures
                errors.append({"source": source, "context_id": str(context["context_id"]), "error": f"{type(exc).__name__}: {exc}"})
                raw_rows = []
            for rank, raw in enumerate(list(raw_rows or [])[:top_k], start=1):
                if not isinstance(raw, dict):
                    continue
                action = CandidateAction.from_candidate(
                    target_smiles,
                    {**raw, "source": source, "rank": rank},
                    rank=rank,
                    source=source,
                )
                rows.append(
                    quality_payload(
                        target=target,
                        action=action,
                        context=context,
                        bridge_ec1s=bridge_ec1s,
                        bridge_hit_count=len(bridge_hits),
                        scorer=scorer,
                        blacklist=blacklist,
                    )
                )
    target_row = summarize_target(target, rows, bridge_hit_count=len(bridge_hits), bridge_ec1s=bridge_ec1s, errors=errors)
    return target_row, rows


def quality_payload(
    *,
    target: dict[str, Any],
    action: CandidateAction,
    context: dict[str, Any],
    bridge_ec1s: list[int],
    bridge_hit_count: int,
    scorer: EnzymeSPVerifierV1Scorer | None,
    blacklist: set[str],
) -> dict[str, Any]:
    source = action.source
    group = source_group(source)
    ecs = _action_ecs(action)
    ec1s = _ec1s(ecs)
    bridge_ec1_match = bool(bridge_ec1s and any(ec1 in bridge_ec1s for ec1 in ec1s))
    product_similarity = _product_similarity(action)
    evidence = dict(action.metadata.get("evidence") or {})
    source_counts = _dict_from_any(evidence.get("source_counts") or action.metadata.get("source_counts"))
    rhea_ids = _list_from_any(evidence.get("rhea_ids") or action.metadata.get("rhea_ids"))
    occurrences = _safe_int(evidence.get("occurrences") or action.metadata.get("occurrences"))
    transition_payload = _dict_from_any(evidence.get("transition_signature") or action.metadata.get("transition_signature"))
    transition_flags = _list_from_any(transition_payload.get("transition_flags") if transition_payload else None)
    transition_score = _safe_float(transition_payload.get("transition_quality_score") if transition_payload else None)
    sp_payload = None
    if scorer is not None and _is_enzyme_like(source, action):
        try:
            sp_payload = scorer.score_action(product=action.product, action=action).to_dict()
        except Exception as exc:
            sp_payload = {"error": f"{type(exc).__name__}: {exc}", "accepted": False, "score": 0.0}
    product_atoms = heavy_atoms(action.product)
    main_atoms = heavy_atoms(action.main_reactant)
    aux_atoms = [heavy_atoms(smi) for smi in action.aux_reactants]
    substrate_total_atoms = sum(heavy_atoms(smi) for smi in action.reactants)
    product_blacklisted = _is_blacklisted(action.product, blacklist)
    main_blacklisted = _is_blacklisted(action.main_reactant, blacklist)
    aux_blacklisted = sum(1 for smi in action.aux_reactants if _is_blacklisted(smi, blacklist))
    flags = quality_flags(
        source=source,
        sp_payload=sp_payload,
        product_similarity=product_similarity,
        ecs=ecs,
        bridge_hit_count=bridge_hit_count,
        bridge_ec1s=bridge_ec1s,
        bridge_ec1_match=bridge_ec1_match,
        context_ec1=int(context.get("ec1") or 0),
        product_blacklisted=product_blacklisted,
        main_blacklisted=main_blacklisted,
        aux_blacklisted=aux_blacklisted,
        product_atoms=product_atoms,
        main_atoms=main_atoms,
        substrate_total_atoms=substrate_total_atoms,
        component_count=len(action.reactants),
        transition_flags=transition_flags,
    )
    quality = quality_score(
        source=source,
        sp_payload=sp_payload,
        product_similarity=product_similarity,
        transition_score=transition_score,
        transition_flags=transition_flags,
        bridge_ec1s=bridge_ec1s,
        bridge_ec1_match=bridge_ec1_match,
        ecs=ecs,
        source_counts=source_counts,
        rhea_ids=rhea_ids,
        occurrences=occurrences,
        product_blacklisted=product_blacklisted,
        main_blacklisted=main_blacklisted,
        aux_blacklisted=aux_blacklisted,
        flags=flags,
    )
    tier = quality_tier(quality, flags, source=source)
    return {
        "schema_version": "enzyme_candidate_quality_row.v0",
        "target_smiles": str(target.get("target_smiles") or ""),
        "target_canonical": canonical_smiles(str(target.get("target_smiles") or "")) or str(target.get("target_smiles") or ""),
        "label": int(target.get("label") or 0),
        "label_source": str(target.get("label_source") or ""),
        "context_id": str(context.get("context_id") or ""),
        "context_ec1": int(context.get("ec1") or 0),
        "bridge_hit_count": int(bridge_hit_count),
        "bridge_ec1s": list(bridge_ec1s),
        "source": source,
        "source_group": group,
        "rank": int(action.rank or 0),
        "canonical_key": action.canonical_key,
        "reaction_smiles": action.rxn_smiles,
        "main_reactant": action.main_reactant,
        "aux_reactants": list(action.aux_reactants),
        "raw_score": float(action.raw_score or 0.0),
        "product_similarity": product_similarity,
        "ec": action.ec,
        "ec_numbers": ecs,
        "ec1s": ec1s,
        "bridge_ec1_match": bridge_ec1_match,
        "sp_v1": sp_payload,
        "product_heavy_atoms": product_atoms,
        "main_heavy_atoms": main_atoms,
        "aux_heavy_atoms": aux_atoms,
        "substrate_total_heavy_atoms": substrate_total_atoms,
        "product_blacklisted": product_blacklisted,
        "main_blacklisted": main_blacklisted,
        "aux_blacklisted_count": aux_blacklisted,
        "component_count": len(action.reactants),
        "occurrences": occurrences,
        "source_counts": source_counts,
        "rhea_ids": rhea_ids,
        "transition_signature": transition_payload,
        "transition_quality_score": transition_score,
        "quality_score": round(float(quality), 4),
        "quality_tier": tier,
        "search_ready": tier in {"strong", "usable_review"},
        "risk_flags": flags,
    }


def quality_flags(
    *,
    source: str,
    sp_payload: dict[str, Any] | None,
    product_similarity: float,
    ecs: list[str],
    bridge_hit_count: int,
    bridge_ec1s: list[int],
    bridge_ec1_match: bool,
    context_ec1: int,
    product_blacklisted: bool,
    main_blacklisted: bool,
    aux_blacklisted: int,
    product_atoms: int,
    main_atoms: int,
    substrate_total_atoms: int,
    component_count: int,
    transition_flags: list[str] | None = None,
) -> list[str]:
    flags: list[str] = []
    if _is_enzyme_source_name(source):
        if int(bridge_hit_count or 0) <= 0 and int(context_ec1 or 0) <= 0:
            flags.append("no_bridge_or_ec_trigger_for_injection")
        if not ecs:
            flags.append("no_ec_evidence")
        if bridge_ec1s and not bridge_ec1_match:
            flags.append("bridge_ec_mismatch")
        if sp_payload is not None and sp_payload.get("accepted") is False:
            flags.append("sp_v1_reject")
        if product_similarity and product_similarity < 0.35:
            flags.append("low_product_similarity")
        for flag in transition_flags or []:
            if flag and flag not in flags:
                flags.append(str(flag))
    if product_blacklisted:
        flags.append("product_common_or_cofactor")
    if main_blacklisted:
        flags.append("main_common_or_cofactor")
    if aux_blacklisted:
        flags.append("aux_common_or_cofactor")
    if component_count > 4:
        flags.append("many_components")
    if product_atoms >= 20 and main_atoms > 0 and abs(product_atoms - main_atoms) >= 25:
        flags.append("large_heavy_atom_delta_review_only")
    if substrate_total_atoms and product_atoms and substrate_total_atoms > max(90, product_atoms * 2.5):
        flags.append("substrate_side_much_larger_review_only")
    return flags


def quality_score(
    *,
    source: str,
    sp_payload: dict[str, Any] | None,
    product_similarity: float,
    transition_score: float = 0.0,
    transition_flags: list[str] | None = None,
    bridge_ec1s: list[int],
    bridge_ec1_match: bool,
    ecs: list[str],
    source_counts: dict[str, Any],
    rhea_ids: list[str],
    occurrences: int,
    product_blacklisted: bool,
    main_blacklisted: bool,
    aux_blacklisted: int,
    flags: list[str],
) -> float:
    if not _is_enzyme_source_name(source):
        return float(product_similarity or 0.0)
    transition_flags = transition_flags or []
    score = 0.0
    if sp_payload is not None:
        sp_score = float(sp_payload.get("score") or 0.0)
        score += 2.0 * sp_score
        score += 0.75 if sp_payload.get("accepted") else -1.0
    score += 1.35 * max(0.0, min(1.0, float(product_similarity or 0.0)))
    if bridge_ec1s:
        score += 0.75 if bridge_ec1_match else -0.55
    if ecs:
        score += 0.25
    else:
        score -= 0.35
    if rhea_ids:
        score += 0.35
    if occurrences > 0:
        score += min(0.45, math.log10(float(occurrences) + 1.0) * 0.15)
    if source_counts:
        score += min(0.35, math.log10(sum(_safe_int(v) for v in source_counts.values()) + 1.0) * 0.10)
    if transition_score:
        score += 0.25 * (max(0.0, min(1.0, float(transition_score))) - 0.5)
    if product_blacklisted:
        score -= 1.25
    if main_blacklisted:
        score -= 1.5
    if aux_blacklisted:
        score -= min(0.5, aux_blacklisted * 0.15)
    if "low_product_similarity" in flags:
        score -= 0.35
    if "many_components" in flags:
        score -= 0.20
    if "main_transition_self_loop" in transition_flags:
        score -= 0.35
    if "weak_main_transition_similarity" in transition_flags:
        score -= 0.15
    if "large_main_transition_delta_review" in transition_flags:
        score -= 0.10
    if "unexplained_element_gain_review" in transition_flags:
        score -= 0.10
    return score


def quality_tier(score: float, flags: list[str], *, source: str) -> str:
    if not _is_enzyme_source_name(source):
        return "native_chemical_reference"
    hard_flags = {"sp_v1_reject", "main_common_or_cofactor", "product_common_or_cofactor", "no_ec_evidence"}
    if any(flag in hard_flags for flag in flags):
        return "not_search_ready"
    if "no_bridge_or_ec_trigger_for_injection" in flags:
        return "ungated_review" if score >= 1.75 else "not_search_ready"
    if "low_product_similarity" in flags:
        return "weak_review" if score >= 1.75 else "not_search_ready"
    if score >= 2.65 and "bridge_ec_mismatch" not in flags:
        return "strong"
    if score >= 1.75:
        return "usable_review"
    if score >= 0.75:
        return "weak_review"
    return "not_search_ready"


def summarize_target(
    target: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    bridge_hit_count: int,
    bridge_ec1s: list[int],
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    by_source: dict[str, dict[str, Any]] = {}
    for source, subset in _group(rows, "source").items():
        by_source[source] = {
            "candidates": len(subset),
            "unique_candidates": len({row["canonical_key"] for row in subset}),
            "search_ready": sum(1 for row in subset if row["search_ready"]),
            "strong": sum(1 for row in subset if row["quality_tier"] == "strong"),
            "sp_v1_accepted": sum(1 for row in subset if (row.get("sp_v1") or {}).get("accepted") is True),
            "best_quality_score": max((float(row["quality_score"]) for row in subset), default=0.0),
            "top_quality_tier": max((str(row["quality_tier"]) for row in subset), default=""),
        }
    return {
        "schema_version": "enzyme_candidate_quality_target.v0",
        "target_smiles": str(target.get("target_smiles") or ""),
        "target_canonical": canonical_smiles(str(target.get("target_smiles") or "")) or str(target.get("target_smiles") or ""),
        "label": int(target.get("label") or 0),
        "label_source": str(target.get("label_source") or ""),
        "bridge_hit_count": int(bridge_hit_count),
        "bridge_ec1s": bridge_ec1s,
        "candidate_count": len(rows),
        "search_ready_count": sum(1 for row in rows if row["search_ready"]),
        "strong_count": sum(1 for row in rows if row["quality_tier"] == "strong"),
        "by_source": by_source,
        "errors": errors,
    }


def summarize(candidate_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]], *, sources: list[str]) -> dict[str, Any]:
    label_sets = {
        "all": target_rows,
        "positive": [row for row in target_rows if int(row["label"]) == 1],
        "negative": [row for row in target_rows if int(row["label"]) == 0],
    }
    source_rows = _group(candidate_rows, "source")
    by_source = {}
    for source in sources:
        rows = source_rows.get(source, [])
        by_source[source] = summarize_source(rows, target_rows)
    target_summary = {
        name: {
            "targets": len(rows),
            "targets_with_bridge": sum(1 for row in rows if int(row["bridge_hit_count"]) > 0),
            "targets_with_search_ready": sum(1 for row in rows if int(row["search_ready_count"]) > 0),
            "targets_with_strong": sum(1 for row in rows if int(row["strong_count"]) > 0),
            "mean_search_ready": _mean(row["search_ready_count"] for row in rows),
        }
        for name, rows in label_sets.items()
    }
    tier_counts = Counter(str(row["quality_tier"]) for row in candidate_rows)
    flag_counts = Counter(flag for row in candidate_rows for flag in row.get("risk_flags") or [])
    return {
        "targets": len(target_rows),
        "positive_targets": sum(1 for row in target_rows if int(row["label"]) == 1),
        "negative_targets": sum(1 for row in target_rows if int(row["label"]) == 0),
        "candidate_rows": len(candidate_rows),
        "unique_candidates": len({row["canonical_key"] for row in candidate_rows}),
        "search_ready_candidates": sum(1 for row in candidate_rows if row["search_ready"]),
        "strong_candidates": sum(1 for row in candidate_rows if row["quality_tier"] == "strong"),
        "target_summary": target_summary,
        "by_source": by_source,
        "quality_tier_counts": dict(tier_counts),
        "risk_flag_counts": dict(flag_counts.most_common()),
    }


def summarize_source(rows: list[dict[str, Any]], target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    target_labels = {row["target_canonical"]: int(row["label"]) for row in target_rows}
    ready_targets = {row["target_canonical"] for row in rows if row["search_ready"]}
    strong_targets = {row["target_canonical"] for row in rows if row["quality_tier"] == "strong"}
    positive_targets = {key for key, label in target_labels.items() if label == 1}
    negative_targets = {key for key, label in target_labels.items() if label == 0}
    return {
        "candidate_rows": len(rows),
        "unique_candidates": len({row["canonical_key"] for row in rows}),
        "targets_with_candidates": len({row["target_canonical"] for row in rows}),
        "search_ready_candidates": sum(1 for row in rows if row["search_ready"]),
        "strong_candidates": sum(1 for row in rows if row["quality_tier"] == "strong"),
        "positive_targets_with_search_ready": len(ready_targets & positive_targets),
        "negative_targets_with_search_ready": len(ready_targets & negative_targets),
        "positive_targets_with_strong": len(strong_targets & positive_targets),
        "negative_targets_with_strong": len(strong_targets & negative_targets),
        "positive_ready_recall": _ratio(len(ready_targets & positive_targets), len(positive_targets)),
        "negative_ready_rate": _ratio(len(ready_targets & negative_targets), len(negative_targets)),
        "mean_quality_score": _mean(row["quality_score"] for row in rows),
        "mean_sp_v1_score": _mean((row.get("sp_v1") or {}).get("score") for row in rows if row.get("sp_v1")),
    }


def conclusion(summary: dict[str, Any], *, include_native: bool) -> str:
    precedent = (summary.get("by_source") or {}).get("enzyme_precedent") or {}
    v3 = (summary.get("by_source") or {}).get("v3_retrieval") or {}
    native = (summary.get("by_source") or {}).get(NATIVE_CHEMENZY_SOURCE) or {}
    text = (
        "Candidate audit separates coverage from quality: "
        f"enzyme_precedent produced {precedent.get('candidate_rows', 0)} rows, "
        f"{precedent.get('search_ready_candidates', 0)} search-ready candidates, "
        f"positive ready recall {precedent.get('positive_ready_recall', 0.0):.4f}, "
        f"negative ready rate {precedent.get('negative_ready_rate', 0.0):.4f}. "
        f"v3_retrieval produced {v3.get('candidate_rows', 0)} rows, "
        f"{v3.get('search_ready_candidates', 0)} search-ready candidates. "
    )
    if include_native:
        text += (
            "Native ChemEnzy one-step was included as a chemical-proposer reference: "
            f"{native.get('candidate_rows', 0)} rows from {native.get('targets_with_candidates', 0)} targets. "
        )
    else:
        text += "Native ChemEnzy one-step was not initialized in this run; use --include-native-chem-enzy for direct proposer comparison. "
    text += "Rows marked large-heavy-atom review are not hard rejects because auxiliary reagents can legitimately introduce groups."
    return text


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Enzyme Candidate Quality Audit v0",
        "",
        "Purpose: audit whether enlarged enzyme proposal sources add search-ready candidates before injecting them into main search.",
        "",
        "## Inputs",
        "",
        f"- Targets: {summary['targets']} ({summary['positive_targets']} positive, {summary['negative_targets']} negative)",
        f"- Sources: {', '.join(report['inputs']['sources'])}",
        f"- SP-v1 enabled: {report['inputs']['sp_v1_enabled']}",
        f"- Include native ChemEnzy one-step: {report['inputs']['include_native_chem_enzy']}",
        "",
        "## Overall",
        "",
        f"- Candidate rows: {summary['candidate_rows']}",
        f"- Unique candidates: {summary['unique_candidates']}",
        f"- Search-ready candidates: {summary['search_ready_candidates']}",
        f"- Strong candidates: {summary['strong_candidates']}",
        "",
        "## Source Summary",
        "",
        "| source | rows | unique | ready | strong | pos ready recall | neg ready rate | mean quality | mean SP-v1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source, row in (summary.get("by_source") or {}).items():
        lines.append(
            "| {source} | {candidate_rows} | {unique_candidates} | {search_ready_candidates} | {strong_candidates} | "
            "{positive_ready_recall:.4f} | {negative_ready_rate:.4f} | {mean_quality_score:.3f} | {mean_sp_v1_score:.3f} |".format(
                source=source,
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Target Summary",
            "",
            "| split | targets | bridge | ready targets | strong targets | mean ready |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split, row in (summary.get("target_summary") or {}).items():
        lines.append(
            "| {split} | {targets} | {targets_with_bridge} | {targets_with_search_ready} | {targets_with_strong} | {mean_search_ready:.2f} |".format(
                split=split,
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Top Risk Flags",
            "",
        ]
    )
    for flag, count in list((summary.get("risk_flag_counts") or {}).items())[:12]:
        lines.append(f"- {flag}: {count}")
    lines.extend(["", "## Conclusion", "", report["conclusion"], ""])
    return "\n".join(lines)


def load_blacklist(pack_dir: Path) -> set[str]:
    path = pack_dir / "cofactor_common_metabolite_blacklist.parquet"
    if not path.exists():
        return set()
    values: set[str] = set()
    for row in pq.read_table(path, columns=["canonical_smiles"]).to_pylist():
        can = canonical_smiles(str(row.get("canonical_smiles") or ""))
        if can:
            values.add(can)
    return values


def _source_list(value: str) -> list[str]:
    out: list[str] = []
    for item in str(value or "").split(","):
        source = item.strip()
        if source and source not in out:
            out.append(source)
    return out or list(DEFAULT_SOURCES)


def _needs_live_engine(sources: list[str]) -> bool:
    return any(source not in {"enzyme_precedent", "v3_retrieval"} for source in sources)


def _bridge_ec1s(hits: list[Any]) -> list[int]:
    out: list[int] = []
    for hit in hits:
        for ec in getattr(hit, "enzyme_ec_sample", ()) or ():
            head = str(ec or "").split(".", 1)[0]
            if head.isdigit() and 1 <= int(head) <= 7 and int(head) not in out:
                out.append(int(head))
    return out


def _action_ecs(action: CandidateAction) -> list[str]:
    values: list[str] = []
    if action.ec:
        values.append(str(action.ec))
    metadata = action.metadata or {}
    evidence = metadata.get("evidence") or {}
    for raw in (
        metadata.get("enzyme_ec_numbers"),
        metadata.get("ec_numbers"),
        metadata.get("enzyme_ec_sample"),
        evidence.get("ec_numbers") if isinstance(evidence, dict) else None,
    ):
        values.extend(_list_from_any(raw))
    return list(dict.fromkeys(str(ec).strip() for ec in values if str(ec or "").strip()))


def _ec1s(ecs: list[str]) -> list[int]:
    out: list[int] = []
    for ec in ecs:
        head = str(ec or "").split(".", 1)[0]
        if head.isdigit() and 1 <= int(head) <= 7 and int(head) not in out:
            out.append(int(head))
    return out


def _product_similarity(action: CandidateAction) -> float:
    metadata = action.metadata or {}
    evidence = metadata.get("evidence") or {}
    for value in (
        metadata.get("precedent_product_similarity"),
        evidence.get("product_similarity") if isinstance(evidence, dict) else None,
        metadata.get("product_similarity"),
        action.raw_score,
    ):
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _is_enzyme_like(source: str, action: CandidateAction) -> bool:
    return _is_enzyme_source_name(source) or bool(action.ec)


def _is_enzyme_source_name(source: str) -> bool:
    group = source_group(source)
    return source in ENZYME_LIKE_SOURCES or group in {"enzymatic", "rhea_retrorules", "retrieval"}


def _is_blacklisted(smiles: str, blacklist: set[str]) -> bool:
    can = canonical_smiles(str(smiles or ""))
    return bool(can and can in blacklist)


def heavy_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    return int(mol.GetNumHeavyAtoms()) if mol is not None else 0


def inchikey(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return ""
    try:
        return MolToInchiKey(mol)
    except Exception:
        return ""


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _list_from_any(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item or "")]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except Exception:
            return [stripped]
        return _list_from_any(parsed)
    return []


def _dict_from_any(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get(key) or "")].append(row)
    return dict(out)


def _mean(values: Any) -> float:
    vals = []
    for value in values:
        try:
            vals.append(float(value))
        except (TypeError, ValueError):
            continue
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _ratio(num: int, den: int) -> float:
    return round(float(num) / float(den), 4) if den else 0.0


if __name__ == "__main__":
    main()
