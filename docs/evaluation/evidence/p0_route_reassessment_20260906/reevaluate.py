"""Reassess saved route families through the production Critic, without editing runs.

Prepare by default; --execute makes one model call per selected family. Conditions
are recovered only from the exact original Builder proposal through today's Host.
This is a new assessment of corrected input, not a replacement historical verdict.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs/evaluation/evidence/current_io_review_20260905"))
# Standalone evidence scripts need the repository and original audit on sys.path.
from audit_io import clean_identity, journal, load_graph, read  # noqa: E402
from verify_p0_fixes import worker_record  # noqa: E402
from cascade_planner.application.condition_predictions import normalize_condition_predictions  # noqa: E402
from cascade_planner.application.route_review_context import compile_revision_bound_route_critic_context  # noqa: E402
from cascade_planner.orchestration.sequential_strategy_director import (  # noqa: E402
    DirectorConfig, SequentialStrategyDirectorRunner, _expansions_from_record, _step_row,
)
from cascade_planner.runtime import AgentSpec  # noqa: E402


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def restore_conditions(case, graph):
    records = {r["record"]["task_id"]: r["record"] for r in journal(Path(case["journal_path"]))}
    inputs = {r["task_id"]: r for r in journal(Path(case["io_path"])) if r.get("event") == "model_input"}
    restored = []
    for probe in case["condition_deletion_probes"]:
        task_id = probe["task_id"]
        record = worker_record(records[task_id])
        source = json.loads(inputs[task_id]["prompt"].split("PaperMatchedRouteBuilderContext:\n", 1)[1])
        candidate = record.output_artifact["payload"]["candidates"][0]
        expansions = _expansions_from_record(
            record, expected_product=candidate["product_smiles"],
            mapped_product_smiles=source["selected_leaf_mapped"],
            require_reaction_operations=True, single_step_only=True,
        )
        if len(expansions) != 1:
            raise ValueError(f"Expected one replayable proposal: {task_id}")
        step = _step_row(expansions[0], step_id=candidate["candidate_id"])
        predictions = normalize_condition_predictions(step["condition_predictions"])
        proposal_id = "codex:branch:" + candidate["candidate_id"].split(":branch:", 1)[1]
        matched = [e for e in graph["edges"].values() if any(
            o.get("proposal_id") == proposal_id for o in e.get("origin_records", []))]
        if len(matched) != 1:
            raise ValueError(f"Ambiguous or absent saved edge: {proposal_id}")
        edge = matched[0]
        if clean_identity(edge["product_smiles"]) != clean_identity(candidate["product_smiles"]):
            raise ValueError(f"Builder/edge product mismatch: {proposal_id}")
        if sorted(map(clean_identity, edge["precursor_smiles"])) != sorted(map(clean_identity, step["precursor_smiles"])):
            raise ValueError(f"Builder/edge precursor mismatch: {proposal_id}")
        if probe["condition"] not in json.dumps(predictions):
            raise ValueError(f"Condition was not recovered: {proposal_id}")
        restored.append({"source_task_id": task_id, "edge_id": edge["edge_id"],
                         "before": edge.get("condition_predictions"), "after": predictions})
        edge["condition_predictions"] = predictions
    return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    case = next(c for c in read(ROOT / "docs/evaluation/evidence/current_io_review_20260905/audit.json")["cases"]
                if c["case_id"] == args.case_id)
    original_config = read(case["report_path"])["config"]
    config = DirectorConfig(
        paper_matched_reach_profile=True,
        model=original_config["model"], reasoning_effort=original_config["reasoning_effort"],
        max_node_prompt_bytes=original_config["max_node_prompt_bytes"],
        critic_call_timeout_s=original_config["critic_call_timeout_s"],
    )
    graph = load_graph(case)
    restored = restore_conditions(case, graph)
    output = args.output_root.resolve()
    write(output / "condition-recovery.json", {"case_id": args.case_id, "label": case["label"],
          "source_report": case["report_path"], "original_graph_unchanged": True,
          "historical_verdicts_unchanged": True, "restorations": restored})
    contexts = []
    for family_id, family in graph["route_families"].items():
        if family.get("selected") is False:
            continue
        context, diagnostic = compile_revision_bound_route_critic_context(graph, route_family_id=family_id)
        if context is None:
            raise ValueError(diagnostic)
        contexts.append(context)
    results = []
    for context in sorted(contexts, key=lambda c: c.branch_index):
        branch = output / f"branch-{context.branch_index + 1}"
        runner = SequentialStrategyDirectorRunner()
        prompt = runner.final_route_critic_prompt_for(context, config)
        if not prompt:
            raise ValueError("Final route prompt exceeds the original bound")
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        write(branch / "context.json", asdict(context))
        (branch / "prompt.txt").write_text(prompt, encoding="utf-8")
        row = {"branch": context.branch_index + 1, "route_family_id": context.route_family_id,
               "step_count": len(context.steps), "prompt_sha256": digest,
               "prompt_bytes": len(prompt.encode("utf-8")), "model": config.model,
               "reasoning_effort": config.reasoning_effort, "status": "prepared"}
        print(json.dumps(row), flush=True)
        if args.execute:
            workspace = branch / "worker"
            workspace.mkdir(parents=True, exist_ok=True)
            spec = AgentSpec(
                run_id=f"p0-reassessment-20260906:{args.case_id}",
                agent_id=f"p0-final-critic:{args.case_id}:{context.branch_index}:{digest[:12]}",
                role="route_critic", objective="Reassess this saved route after Host corrections.",
                idempotency_key=digest, context_hash=digest,
                metadata={"model": config.model, "reasoning_effort": config.reasoning_effort,
                          "allowed_workdir": str(workspace), "durable_worker_journal": True},
            )
            critique, record = runner.run_final_route_critic_once(spec, context=context, config=config, prompt=prompt)
            write(branch / "result.json", {**row, "status": record.status,
                  "completed_at": datetime.now(timezone.utc).isoformat(),
                  "critique": critique, "record": asdict(record)})
            row.update(status=record.status, critique=critique)
            print(json.dumps({"branch": row["branch"], "status": row["status"],
                              "assessment": critique.get("overall_assessment")}), flush=True)
        results.append(row)
        write(output / "summary.json", {"case_id": args.case_id, "label": case["label"],
              "mode": "new_model_reassessment" if args.execute else "prepared_no_model_calls",
              "original_graph_unchanged": True, "family_results": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
