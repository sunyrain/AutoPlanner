"""Replay cached proposal pools under bridge/enzyme verifier policies.

This is a root/frontier proposal quality benchmark. It does not claim full
route solved-rate because all policies share a cached one-step candidate pool.
Its job is to expose whether stronger enzyme gates reject bad enzyme proposals
without losing bridge-positive proposal coverage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.cascade_search.bridge_retriever_v0 import BridgeRetrieverV0
from cascade_planner.cascade_search.enzyme_sp_verifier_v1 import EnzymeSPVerifierV1Scorer
from cascade_planner.route_tree.schema import CandidateAction
from cascade_planner.route_tree.source_gate import source_group


DEFAULT_PACK = Path("results/shared/live_proposal_replay_pack_v1_20260528/live_proposal_replay_pack.jsonl")
DEFAULT_BRIDGE_PACK_DIR = Path("data/bridge_pack_v0")
DEFAULT_OUTPUT_DIR = Path("results/shared/enzyme_sp_v1_replay_benchmark_20260528")
ENZYMATIC_GROUPS = {"enzymatic", "rhea_retrorules"}
POLICIES = (
    "ungated_all",
    "bridge_gate_v0",
    "bridge_gate_v0_sp_v1_hard",
    "bridge_gate_v0_sp_v1_soft",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay enzyme SP gate policies on cached proposal pools")
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--bridge-pack-dir", type=Path, default=DEFAULT_BRIDGE_PACK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sp-penalty", type=float, default=0.35)
    parser.add_argument("--no-sp-model", action="store_true", help="Do not load SP-v1; useful for schema checks only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.pack)
    retriever = BridgeRetrieverV0(args.bridge_pack_dir, scorer=None)
    scorer = None if args.no_sp_model else EnzymeSPVerifierV1Scorer()
    output_rows: list[dict[str, Any]] = []
    for target in rows:
        actions = collect_actions(target)
        bridge_hits = retriever.retrieve(
            str(target.get("target_smiles") or ""),
            top_k=8,
            require_verifier_pass=True,
        )
        for policy in POLICIES:
            output_rows.append(
                replay_target(
                    target,
                    actions,
                    policy=policy,
                    bridge_hit_count=len(bridge_hits),
                    scorer=scorer,
                    top_k=max(1, int(args.top_k)),
                    sp_penalty=float(args.sp_penalty),
                )
            )
    summary = [summarize_policy(output_rows, policy) for policy in POLICIES]
    report = {
        "schema_version": "enzyme_sp_v1_replay_benchmark.v1",
        "benchmark_scope": "cached_root_frontier_proposal_replay",
        "inputs": {
            "pack": str(args.pack),
            "bridge_pack_dir": str(args.bridge_pack_dir),
            "top_k": int(args.top_k),
            "sp_penalty": float(args.sp_penalty),
            "sp_model_enabled": scorer is not None,
        },
        "targets": len(rows),
        "policies": summary,
        "conclusion": conclusion(summary),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "enzyme_sp_v1_replay_rows.jsonl"
    report_json = args.output_dir / "enzyme_sp_v1_replay_report.json"
    report_md = args.output_dir / "enzyme_sp_v1_replay_report.md"
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    report_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {"report": str(report_json), "rows": str(rows_path), "conclusion": report["conclusion"]},
            ensure_ascii=False,
            indent=2,
        )
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_actions(target: dict[str, Any]) -> list[CandidateAction]:
    actions: list[CandidateAction] = []
    seen: set[str] = set()
    product = str(target.get("target_smiles") or "")
    for context in target.get("contexts") or []:
        context_id = str(context.get("context_id") or "")
        for source_row in (context.get("source_results") or {}).values():
            for raw in source_row.get("actions") or []:
                if not isinstance(raw, dict):
                    continue
                candidate = {
                    "main_reactant": raw.get("main_reactant") or "",
                    "aux_reactants": raw.get("aux_reactants") or [],
                    "rxn_smiles": raw.get("rxn_smiles") or "",
                    "score": raw.get("raw_score") or 0.0,
                    "rank": raw.get("rank") or 0,
                    "source": raw.get("source") or "",
                    "reaction_type": raw.get("reaction_type") or "",
                    "ec": raw.get("ec") or "",
                    "catalyst": raw.get("catalyst") or "",
                    "T": raw.get("T"),
                    "pH": raw.get("pH"),
                    "solvent": raw.get("solvent") or "",
                }
                action = CandidateAction.from_candidate(product, candidate, rank=int(candidate["rank"] or 0), source=candidate["source"])
                metadata = dict(action.metadata or {})
                metadata["replay_context_id"] = context_id
                metadata["replay_context_ec1"] = int(context.get("ec1") or 0)
                action.metadata = metadata
                key = action.canonical_key
                dedupe_key = f"{action.source}|{context_id}|{key}"
                if key in seen and source_group(action.source) in ENZYMATIC_GROUPS:
                    continue
                seen.add(dedupe_key)
                actions.append(action)
    return actions


def replay_target(
    target: dict[str, Any],
    actions: list[CandidateAction],
    *,
    policy: str,
    bridge_hit_count: int,
    scorer: EnzymeSPVerifierV1Scorer | None,
    top_k: int,
    sp_penalty: float,
) -> dict[str, Any]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    sp_calls = 0
    for action in actions:
        enzymatic = is_enzymatic_action(action)
        if policy.startswith("bridge_gate") and enzymatic and bridge_hit_count <= 0:
            rejected.append(rejection_row(action, reason="no_bridge_hit"))
            continue
        sp_score = None
        sp_accepted = None
        if "sp_v1" in policy and enzymatic and scorer is not None:
            sp_calls += 1
            score = scorer.score_action(product=str(target.get("target_smiles") or ""), action=action)
            sp_score = float(score.score)
            sp_accepted = bool(score.accepted)
            if policy.endswith("_hard") and not score.accepted:
                rejected.append(rejection_row(action, reason="sp_v1_reject", sp_score=sp_score))
                continue
        score_value = action_replay_score(action)
        if policy.endswith("_soft") and sp_accepted is False:
            score_value -= float(sp_penalty)
        kept.append(
            {
                "source": action.source,
                "source_group": source_group(action.source),
                "canonical_key": action.canonical_key,
                "main_reactant": action.main_reactant,
                "reactants": list(action.reactants),
                "raw_score": float(action.raw_score or 0.0),
                "rank": int(action.rank or 0),
                "ec": action.ec,
                "replay_score": round(float(score_value), 6),
                "sp_v1_score": sp_score,
                "sp_v1_accepted": sp_accepted,
                "replay_context_id": action.metadata.get("replay_context_id"),
            }
        )
    kept.sort(key=lambda row: (float(row["replay_score"]), -int(row["rank"] or 0)), reverse=True)
    selected = kept[:top_k]
    selected_enzyme = [row for row in selected if row["source_group"] in ENZYMATIC_GROUPS]
    label = int(target.get("label") or 0)
    return {
        "target_smiles": target.get("target_smiles") or "",
        "target_canonical": target.get("target_canonical") or "",
        "label": label,
        "label_source": target.get("label_source") or "",
        "policy": policy,
        "bridge_hit_count": int(bridge_hit_count),
        "input_actions": len(actions),
        "input_enzyme_actions": sum(1 for action in actions if is_enzymatic_action(action)),
        "kept_actions": len(kept),
        "kept_enzyme_actions": sum(1 for row in kept if row["source_group"] in ENZYMATIC_GROUPS),
        "rejected_actions": len(rejected),
        "sp_v1_calls": sp_calls,
        "sp_v1_rejections": sum(1 for row in rejected if row.get("reason") == "sp_v1_reject"),
        "selected_top_k": selected,
        "selected_enzyme_count": len(selected_enzyme),
        "selected_enzyme_target": bool(selected_enzyme),
        "selected_enzyme_true": bool(selected_enzyme and label == 1),
        "selected_enzyme_false": bool(selected_enzyme and label == 0),
        "rejections": rejected[:20],
    }


def action_replay_score(action: CandidateAction) -> float:
    raw = float(action.raw_score or 0.0)
    if raw > 1.0:
        raw = 1.0
    if raw <= 0.0:
        raw = 0.05
    rank_bonus = 1.0 / max(int(action.rank or 1), 1)
    return 0.80 * raw + 0.20 * rank_bonus


def rejection_row(action: CandidateAction, *, reason: str, sp_score: float | None = None) -> dict[str, Any]:
    return {
        "reason": reason,
        "source": action.source,
        "source_group": source_group(action.source),
        "canonical_key": action.canonical_key,
        "main_reactant": action.main_reactant,
        "raw_score": float(action.raw_score or 0.0),
        "rank": int(action.rank or 0),
        "ec": action.ec,
        "sp_v1_score": sp_score,
        "replay_context_id": action.metadata.get("replay_context_id"),
    }


def is_enzymatic_action(action: CandidateAction) -> bool:
    return source_group(action.source) in ENZYMATIC_GROUPS


def summarize_policy(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    subset = [row for row in rows if row["policy"] == policy]
    positives = [row for row in subset if int(row["label"]) == 1]
    negatives = [row for row in subset if int(row["label"]) == 0]
    selected = [row for row in subset if row["selected_enzyme_target"]]
    true_selected = [row for row in selected if int(row["label"]) == 1]
    false_selected = [row for row in selected if int(row["label"]) == 0]
    return {
        "policy": policy,
        "targets": len(subset),
        "positives": len(positives),
        "negatives": len(negatives),
        "targets_with_selected_enzyme": len(selected),
        "true_selected_targets": len(true_selected),
        "false_selected_targets": len(false_selected),
        "selected_target_precision": _ratio(len(true_selected), len(selected)),
        "selected_target_recall": _ratio(len(true_selected), len(positives)),
        "false_enzyme_target_rate": _ratio(len(false_selected), len(negatives)),
        "mean_input_actions": _mean(row["input_actions"] for row in subset),
        "mean_input_enzyme_actions": _mean(row["input_enzyme_actions"] for row in subset),
        "mean_kept_enzyme_actions": _mean(row["kept_enzyme_actions"] for row in subset),
        "mean_sp_v1_calls": _mean(row["sp_v1_calls"] for row in subset),
        "mean_sp_v1_rejections": _mean(row["sp_v1_rejections"] for row in subset),
    }


def conclusion(summary: list[dict[str, Any]]) -> str:
    by_policy = {row["policy"]: row for row in summary}
    ungated = by_policy.get("ungated_all", {})
    hard = by_policy.get("bridge_gate_v0_sp_v1_hard", {})
    return (
        "Offline replay compares policies on identical cached one-step proposal pools. "
        f"Ungated false enzyme target rate={float(ungated.get('false_enzyme_target_rate') or 0.0):.4f}; "
        f"bridge+SP-v1 hard false rate={float(hard.get('false_enzyme_target_rate') or 0.0):.4f}, "
        f"recall={float(hard.get('selected_target_recall') or 0.0):.4f}, "
        f"mean SP-v1 rejections={float(hard.get('mean_sp_v1_rejections') or 0.0):.2f}."
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Enzyme SP-v1 Replay Benchmark",
        "",
        "Scope: cached root/frontier proposal replay, not full multi-step solved-rate.",
        "",
        f"- Targets: {report['targets']}",
        f"- Pack: `{report['inputs']['pack']}`",
        "",
        "| policy | selected targets | true | false | precision | recall | false rate | kept enzyme | SP calls | SP reject |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["policies"]:
        lines.append(
            "| {policy} | {targets_with_selected_enzyme} | {true_selected_targets} | {false_selected_targets} | "
            "{selected_target_precision:.4f} | {selected_target_recall:.4f} | {false_enzyme_target_rate:.4f} | "
            "{mean_kept_enzyme_actions:.2f} | {mean_sp_v1_calls:.2f} | {mean_sp_v1_rejections:.2f} |".format(**row)
        )
    lines.extend(["", report["conclusion"], ""])
    return "\n".join(lines)


def _ratio(num: int, den: int) -> float:
    return round(float(num) / float(den), 4) if den else 0.0


def _mean(values: Any) -> float:
    vals = [float(value) for value in values]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


if __name__ == "__main__":
    main()
