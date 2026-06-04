"""Experimental enzyme coverage sidecar for real web/native tasks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cascade_planner.cascade_search.bridge_retriever_v0 import BridgeRetrieverV0
from cascade_planner.cascade_search.enzyme_sp_verifier_v1 import EnzymeSPVerifierV1Scorer
from cascade_planner.cascadeboard.enzyme_precedent_retrieval import retrieve_enzyme_precedents
from cascade_planner.route_tree.schema import CandidateAction


@dataclass(frozen=True)
class EnzymeCoverageSidecarConfig:
    pack_dir: Path = Path("data/bridge_pack_v0")
    top_k: int = 8
    bridge_top_k: int = 8
    max_ec_contexts: int = 2
    enable_sp_v1: bool = True


def build_enzyme_coverage_sidecar(
    target_smiles: str,
    *,
    config: EnzymeCoverageSidecarConfig | None = None,
) -> dict[str, Any]:
    cfg = config or EnzymeCoverageSidecarConfig()
    target = str(target_smiles or "")
    report: dict[str, Any] = {
        "schema_version": "enzyme_coverage_sidecar.v1",
        "enabled": True,
        "mode": "annotation_only",
        "target_smiles": target,
        "source": "enzyme_precedent",
        "top_k": int(cfg.top_k),
        "bridge_top_k": int(cfg.bridge_top_k),
        "max_ec_contexts": int(cfg.max_ec_contexts),
        "error": "",
    }
    try:
        retriever = BridgeRetrieverV0(cfg.pack_dir, scorer=None)
        bridge_hits = retriever.retrieve(target, top_k=cfg.bridge_top_k, require_verifier_pass=True)
        ec1s = _bridge_ec1s(bridge_hits)[: max(0, int(cfg.max_ec_contexts))]
        contexts = [{"context_id": "root_no_ec", "ec1": 0}]
        contexts.extend({"context_id": f"bridge_ec{ec1}", "ec1": int(ec1)} for ec1 in ec1s)
        scorer = EnzymeSPVerifierV1Scorer() if cfg.enable_sp_v1 else None
        context_rows = []
        all_candidates = []
        for context in contexts:
            ec1 = int(context["ec1"])
            candidates = retrieve_enzyme_precedents(
                target,
                ec_class=str(ec1) if ec1 else "",
                top_k=cfg.top_k,
            )
            scored = [
                _candidate_payload(target, row, scorer=scorer, context_id=str(context["context_id"]), ec1=ec1)
                for row in candidates
            ]
            context_rows.append(
                {
                    **context,
                    "candidate_count": len(scored),
                    "sp_v1_accepted": sum(1 for row in scored if (row.get("enzyme_sp_verifier_v1") or {}).get("accepted")),
                    "top_candidates": scored[: cfg.top_k],
                }
            )
            all_candidates.extend(scored)
        accepted = [row for row in all_candidates if (row.get("enzyme_sp_verifier_v1") or {}).get("accepted")]
        report.update(
            {
                "bridge_hit_count": len(bridge_hits),
                "bridge_ec1s": ec1s,
                "bridge_hits": [hit.to_dict() for hit in bridge_hits[: cfg.bridge_top_k]],
                "contexts": context_rows,
                "candidate_count": len(all_candidates),
                "sp_v1_enabled": bool(scorer),
                "sp_v1_accepted_count": len(accepted),
                "sp_v1_rejected_count": sum(1 for row in all_candidates if (row.get("enzyme_sp_verifier_v1") or {}).get("accepted") is False),
                "top_accepted_candidates": sorted(
                    accepted,
                    key=lambda row: (
                        float((row.get("enzyme_sp_verifier_v1") or {}).get("score") or 0.0),
                        float(row.get("score") or 0.0),
                    ),
                    reverse=True,
                )[: cfg.top_k],
            }
        )
    except Exception as exc:  # pragma: no cover - sidecar must not break route search
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def _candidate_payload(
    target: str,
    row: dict[str, Any],
    *,
    scorer: EnzymeSPVerifierV1Scorer | None,
    context_id: str,
    ec1: int,
) -> dict[str, Any]:
    action = CandidateAction.from_candidate(target, row, source=str(row.get("source") or "enzyme_precedent"))
    sp_payload = None
    if scorer is not None:
        score = scorer.score_action(product=target, action=action)
        sp_payload = score.to_dict()
    evidence = dict(row.get("evidence") or {})
    return {
        "context_id": context_id,
        "ec1": int(ec1),
        "source": action.source,
        "main_reactant": action.main_reactant,
        "aux_reactants": list(action.aux_reactants),
        "reaction_smiles": action.rxn_smiles,
        "score": float(action.raw_score or 0.0),
        "ec": action.ec,
        "reaction_type": action.reaction_type,
        "evidence": evidence,
        "enzyme_sp_verifier_v1": sp_payload,
    }


def _bridge_ec1s(bridge_hits: list[Any]) -> list[int]:
    out: list[int] = []
    for hit in bridge_hits:
        for ec in getattr(hit, "enzyme_ec_sample", ()) or ():
            head = str(ec or "").split(".", 1)[0]
            if head.isdigit() and 1 <= int(head) <= 7 and int(head) not in out:
                out.append(int(head))
    return out
