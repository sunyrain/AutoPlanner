"""Replay saved P0 incidents through production functions, with no model calls.

Reads audit.json and original runs; writes only p0_fix_replay.json alongside this
script. Historical graphs, judgments and the pre-fix audit remain unchanged.
"""
from dataclasses import fields
import json
from pathlib import Path

from audit_io import ROOT, journal, load_graph, read

from cascade_planner.agent.codex_worker import WorkerRunRecord
from cascade_planner.application.condition_predictions import normalize_condition_predictions
from cascade_planner.application.route_review_context import compile_revision_bound_route_critic_context
from cascade_planner.orchestration.key_event_review import key_event_review_update
from cascade_planner.orchestration.sequential_strategy_director import (
    _critique_from_record,
    _expansions_from_record,
    _host_route_json_from_steps,
    _paper_critic_step_row,
    _selected_path_strategy_checkpoint_state,
    _step_row,
)
from cascade_planner.web.v4_live_synthesis import _route_condition_texts


def worker_record(raw):
    return WorkerRunRecord(**{field.name: raw[field.name] for field in fields(WorkerRunRecord) if field.name in raw})


def verify_case(case):
    records = {
        row.get("record", row)["task_id"]: row.get("record", row)
        for row in journal(Path(case["journal_path"]))
    }
    inputs = {
        row["task_id"]: row for row in journal(Path(case["io_path"]))
        if row.get("event") == "model_input"
    }
    conditions = []
    for probe in case["condition_deletion_probes"]:
        task_id, condition = probe["task_id"], probe["condition"]
        record = worker_record(records[task_id])
        context = json.loads(inputs[task_id]["prompt"].split("PaperMatchedRouteBuilderContext:\n", 1)[1])
        candidate = record.output_artifact["payload"]["candidates"][0]
        expansions = _expansions_from_record(
            record,
            expected_product=candidate["product_smiles"],
            mapped_product_smiles=context["selected_leaf_mapped"],
            require_reaction_operations=True,
            single_step_only=True,
        )
        assert expansions, (case["label"], task_id, "saved builder did not replay")
        step = _step_row(expansions[0], step_id=candidate["candidate_id"])
        predictions = normalize_condition_predictions(step["condition_predictions"])
        assert condition in predictions[0]["reagents"]
        assert predictions[0]["not_reaction_proof"] is True
        assert predictions[0]["not_source_evidence"] is True
        for compact_level in range(4):
            assert condition in _paper_critic_step_row(step, compact_level=compact_level)["conditions"]
        assert condition in _route_condition_texts(step)
        assert condition in _host_route_json_from_steps([step])[0]["conditions"]
        conditions.append({
            "task_id": task_id, "original_output_lines": probe["output_lines"],
            "condition": condition, "builder_replayed": True,
            "preserved_in_host_critic_and_export": True, "remains_advisory": True,
        })

    graph = load_graph(case)
    contexts = []
    for family_id, family in graph["route_families"].items():
        if family.get("selected") is False:
            continue
        context, diagnostic = compile_revision_bound_route_critic_context(graph, route_family_id=family_id)
        assert context is not None, (case["label"], diagnostic)
        before = next(row for row in case["final_context_diagnostics"] if row["route_family_id"] == family_id)
        contexts.append({"route_family_id": family_id, "before_compiles": before["before_compiles"],
                         "after_compiles": True, "step_count": len(context.steps)})

    judgments = []
    for family in case["families"]:
        history = family.get("key_event_critic_history") or []
        for index, row in enumerate(history):
            if row.get("critic_status") != "unavailable":
                continue
            failed_record = worker_record(records[row["task_id"]])
            update = key_event_review_update(
                _critique_from_record(failed_record), focus_step_id=row["focus_step_id"],
                task_id=failed_record.task_id, worker_status=failed_record.status,
            )
            assert update["status"] == "review_unavailable"
            cards = [family.get("root_strategy_card") or {}] + (family.get("strategy_milestone_cards") or [])
            card = next(c for c in cards if c.get("strategy_digest") == row.get("strategy_digest"))
            obligation = row.get("review_of_obligation_id") or row.get("obligation_id")
            earlier = [h for h in history[:index] if h.get("assessment") and
                       (h.get("review_of_obligation_id") or h.get("obligation_id")) == obligation]
            previous = earlier[-1:]
            ids = set(row.get("required_selected_step_ids") or [row["focus_step_id"]])
            for prior in previous:
                ids.update(prior.get("required_selected_step_ids") or [prior["focus_step_id"]])
            steps = [{"step_id": sid} for sid in sorted(ids)]
            # Verify both old persisted failure rows and newly interpreted attempts.
            for failed in (row, {**row, **update}):
                state = _selected_path_strategy_checkpoint_state(
                    {**family, "key_event_critic_history": previous + [failed]}, strategy_card=card, steps=steps,
                )
                assert state["review_pending"] is True
                expected = previous[-1]["assessment"]["verdict"] if previous else ""
                assert state["chemical_confidence"] == expected
                assert state["checkpoint_critic_passed"] is False
            judgments.append({
                "task_id": row["task_id"], "worker_status": failed_record.status,
                "prior_task_id": previous[-1].get("task_id") if previous else None,
                "retained_confidence": state["chemical_confidence"],
                "review_pending": True, "new_attempt_status": update["status"],
                "selected_step_ids_for_incident_replay": sorted(ids),
            })
    return {"label": case["label"], "conditions": conditions, "route_contexts": contexts, "judgments": judgments}


def main():
    baseline = read(Path(__file__).with_name("audit.json"))
    cases = [verify_case(case) for case in baseline["cases"]]
    contexts = [row for case in cases for row in case["route_contexts"]]
    judgments = [row for case in cases for row in case["judgments"]]
    result = {
        "semantics": {"provider_calls": 0, "original_runs_read_only": True,
                      "no_new_chemical_verdicts": True, "production_code_replay": True},
        "summary": {
            "cases": len(cases), "conditions_restored": sum(len(case["conditions"]) for case in cases),
            "route_contexts_before": sum(row["before_compiles"] for row in contexts),
            "route_contexts_after": len(contexts), "failed_critic_attempts": len(judgments),
            "prior_judgments_preserved": sum(bool(row["prior_task_id"]) for row in judgments),
        },
        "cases": cases,
    }
    output = Path(__file__).with_name("p0_fix_replay.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({**result["summary"], "output": str(output.relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
