"""Cache live one-step proposal pools for offline policy replay.

This script runs the expensive live proposal providers once and writes their
normalized root/frontier candidate pools to JSONL. Follow-up replay benchmarks
can then compare source gates, bridge gates, and enzyme verifier policies on
the same proposal pool without reinitializing providers or changing candidates.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.cascade_search.bridge_retriever_v0 import BridgeRetrieverV0
from cascade_planner.cascadeboard.live_retro import build_live_retro_engine, retro_engine_cache_stats
from cascade_planner.cascadeboard.route_recovery import canonical_smiles
from cascade_planner.route_tree.proposals import ProposalContext, RetroEngineProposalTool
from cascade_planner.route_tree.schema import CandidateAction
from cascade_planner.route_tree.source_gate import source_group
from scripts.run_bridge_live_policy_benchmark_v0 import load_targets


DEFAULT_PACK_DIR = Path("data/bridge_pack_v0")
DEFAULT_PROBE_ROWS = Path("results/shared/bridge_gate_ablation_v0_20260527/live_enzyme_bridge_target_probe_rows.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/shared/live_proposal_replay_pack_v1_20260528")
DEFAULT_SOURCE_ORDER = (
    "retrochimera",
    "chem_enzy_onestep",
    "chemtemplates",
    "enzyformer",
    "enzexpand",
    "v3_retrieval",
    "retrorules",
)
ENZYMATIC_GROUPS = {"enzymatic", "rhea_retrorules"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a live proposal replay pack")
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--probe-rows", type=Path, default=DEFAULT_PROBE_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--positives", type=int, default=3)
    parser.add_argument("--negatives", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--per-source-top-k", type=int, default=8)
    parser.add_argument("--bridge-top-k", type=int, default=8)
    parser.add_argument("--max-bridge-ec-contexts", type=int, default=3)
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCE_ORDER),
        help="Comma-separated provider names to query when available.",
    )
    parser.add_argument(
        "--no-bridge-ec-contexts",
        action="store_true",
        help="Only query no-EC root context; skip bridge-derived EC contexts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    random.seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = load_targets(args)
    retriever = BridgeRetrieverV0(args.pack_dir, scorer=None)
    sources = [source.strip() for source in str(args.sources or "").split(",") if source.strip()]
    live_engine = build_live_retro_engine() if requires_live_engine(sources) else {}
    tool = RetroEngineProposalTool(live_engine)
    rows: list[dict[str, Any]] = []
    for idx, target in enumerate(targets, start=1):
        target_smiles = str(target["target_smiles"])
        print(f"[{idx}/{len(targets)}] querying {target_smiles}", flush=True)
        rows.append(
            build_target_row(
                target,
                tool=tool,
                retriever=retriever,
                sources=sources,
                per_source_top_k=max(1, int(args.per_source_top_k)),
                bridge_top_k=max(1, int(args.bridge_top_k)),
                max_bridge_ec_contexts=max(0, int(args.max_bridge_ec_contexts)),
                include_bridge_ec_contexts=not bool(args.no_bridge_ec_contexts),
            )
        )
    pack_jsonl = args.output_dir / "live_proposal_replay_pack.jsonl"
    pack_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "live_proposal_replay_pack.v1",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "inputs": {
            "pack_dir": str(args.pack_dir),
            "probe_rows": str(args.probe_rows),
            "positives": int(args.positives),
            "negatives": int(args.negatives),
            "seed": int(args.seed),
            "per_source_top_k": int(args.per_source_top_k),
            "bridge_top_k": int(args.bridge_top_k),
            "max_bridge_ec_contexts": int(args.max_bridge_ec_contexts),
            "include_bridge_ec_contexts": not bool(args.no_bridge_ec_contexts),
            "sources": sources,
        },
        "targets": len(rows),
        "positive_targets": sum(1 for row in rows if int(row.get("label") or 0) == 1),
        "negative_targets": sum(1 for row in rows if int(row.get("label") or 0) == 0),
        "contexts": sum(len(row.get("contexts") or []) for row in rows),
        "actions": sum(
            len(source_row.get("actions") or [])
            for row in rows
            for context in row.get("contexts") or []
            for source_row in (context.get("source_results") or {}).values()
        ),
        "source_action_counts": source_action_counts(rows),
        "retro_cache_stats": retro_engine_cache_stats(live_engine),
        "pack_jsonl": str(pack_jsonl),
    }
    manifest_path = args.output_dir / "manifest.json"
    report_path = args.output_dir / "report.md"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(render_report(manifest), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "pack": str(pack_jsonl)}, ensure_ascii=False, indent=2))


def build_target_row(
    target: dict[str, Any],
    *,
    tool: RetroEngineProposalTool,
    retriever: BridgeRetrieverV0,
    sources: list[str],
    per_source_top_k: int,
    bridge_top_k: int,
    max_bridge_ec_contexts: int,
    include_bridge_ec_contexts: bool,
) -> dict[str, Any]:
    target_smiles = str(target["target_smiles"])
    bridge_hits = retriever.retrieve(target_smiles, top_k=bridge_top_k, require_verifier_pass=True)
    contexts: list[dict[str, Any]] = [
        {
            "context_id": "root_no_ec",
            "ec1": 0,
            "context_source": "root",
            "sources": list(sources),
        }
    ]
    if include_bridge_ec_contexts and max_bridge_ec_contexts > 0:
        for ec1 in bridge_ec1s(bridge_hits)[:max_bridge_ec_contexts]:
            contexts.append(
                {
                    "context_id": f"bridge_ec{ec1}",
                    "ec1": int(ec1),
                    "context_source": "bridge_retriever_v0",
                    "sources": [source for source in sources if source_group(source) in ENZYMATIC_GROUPS],
                }
            )
    out_contexts = []
    for context in contexts:
        proposal_context = ProposalContext(
            depth=0,
            ec1=int(context["ec1"]),
            route_metadata={
                "replay_pack_context": context["context_id"],
                "bridge_hits": len(bridge_hits),
            },
        )
        out_contexts.append(
            {
                **context,
                "source_results": query_sources(
                    tool,
                    target_smiles,
                    proposal_context,
                    sources=list(context["sources"]),
                    per_source_top_k=per_source_top_k,
                ),
            }
        )
    return {
        "schema_version": "live_proposal_replay_pack.target.v1",
        "target_smiles": target_smiles,
        "target_canonical": canonical_smiles(target_smiles) or target_smiles,
        "chemical_inchikey": target.get("chemical_inchikey") or "",
        "label": int(target.get("label") or 0),
        "label_source": target.get("label_source") or "",
        "bridge_hits": [hit.to_dict() for hit in bridge_hits],
        "contexts": out_contexts,
    }


def query_sources(
    tool: RetroEngineProposalTool,
    product: str,
    context: ProposalContext,
    *,
    sources: list[str],
    per_source_top_k: int,
) -> dict[str, dict[str, Any]]:
    source_results: dict[str, dict[str, Any]] = {}
    for source in sources:
        engine = tool.retro_engine.get(source)
        available = source in {"v3_retrieval", "enzyme_precedent"} or engine is not None
        if not available:
            source_results[source] = {
                "available": False,
                "requested_top_k": per_source_top_k,
                "raw_count": 0,
                "actions": [],
                "error": "missing_engine",
            }
            continue
        t0 = time.monotonic()
        error = ""
        try:
            rows = tool._predict(source, engine, product, context, top_k=per_source_top_k)  # noqa: SLF001
            raw_count = len(rows)
            if tool.proposal_rankers is not None:
                rows = tool.proposal_rankers.rerank(product, source, rows, limit=per_source_top_k)
            rows = list(rows or [])[:per_source_top_k]
            actions = [
                action_to_replay_dict(
                    CandidateAction.from_candidate(
                        product,
                        {**dict(row), "source": source, "rank": rank},
                        rank=rank,
                        source=source,
                    )
                )
                for rank, row in enumerate(rows, start=1)
                if isinstance(row, dict)
            ]
        except Exception as exc:  # pragma: no cover - live provider errors are environment-specific
            raw_count = 0
            actions = []
            error = f"{type(exc).__name__}: {exc}"
        source_results[source] = {
            "available": True,
            "requested_top_k": per_source_top_k,
            "raw_count": int(raw_count),
            "kept_count": len(actions),
            "elapsed_ms": round((time.monotonic() - t0) * 1000.0, 3),
            "actions": actions,
            "error": error,
        }
    return source_results


def action_to_replay_dict(action: CandidateAction) -> dict[str, Any]:
    return {
        "canonical_key": action.canonical_key,
        "product": action.product,
        "reactants": list(action.reactants),
        "main_reactant": action.main_reactant,
        "aux_reactants": list(action.aux_reactants),
        "rxn_smiles": action.rxn_smiles,
        "source": action.source,
        "source_group": source_group(action.source),
        "raw_score": float(action.raw_score or 0.0),
        "rank": int(action.rank or 0),
        "reaction_type": action.reaction_type,
        "ec": action.ec,
        "catalyst": action.catalyst,
        "T": action.T,
        "pH": action.pH,
        "solvent": action.solvent,
        "validity_flags": list(action.validity_flags),
        "metadata": dict(action.metadata or {}),
    }


def bridge_ec1s(bridge_hits: list[Any]) -> list[int]:
    out: list[int] = []
    for hit in bridge_hits:
        for ec in getattr(hit, "enzyme_ec_sample", ()) or ():
            head = str(ec or "").split(".", 1)[0]
            if head.isdigit() and 1 <= int(head) <= 7 and int(head) not in out:
                out.append(int(head))
    return out


def source_action_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for context in row.get("contexts") or []:
            for source, source_row in (context.get("source_results") or {}).items():
                counts[source] = counts.get(source, 0) + len(source_row.get("actions") or [])
    return dict(sorted(counts.items()))


def requires_live_engine(sources: list[str]) -> bool:
    virtual_sources = {"v3_retrieval", "enzyme_precedent"}
    return any(source not in virtual_sources for source in sources)


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# Live Proposal Replay Pack v1",
        "",
        "This pack caches live root/frontier one-step proposal pools for offline policy replay.",
        "",
        f"- Targets: {manifest['targets']} ({manifest['positive_targets']} positive, {manifest['negative_targets']} negative)",
        f"- Contexts: {manifest['contexts']}",
        f"- Actions: {manifest['actions']}",
        f"- Pack: `{manifest['pack_jsonl']}`",
        "",
        "| source | actions |",
        "|---|---:|",
    ]
    for source, count in manifest.get("source_action_counts", {}).items():
        lines.append(f"| {source} | {count} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
