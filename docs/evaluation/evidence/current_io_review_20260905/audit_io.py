"""Read saved IO and replay pure review-context compilation; never invoke a model."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from rdkit import Chem
from cascade_planner.application import route_review_context as review_context
from cascade_planner.application.stereochemistry import stereo_molecule
from cascade_planner.application.condition_predictions import normalize_condition_text as _clean_condition_text
from cascade_planner.orchestration.sequential_strategy_director import (
    _selected_path_strategy_checkpoint_state,
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def journal(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clean_identity(value):
    molecule = Chem.MolFromSmiles(str(value or ""))
    if molecule is None:
        return ""
    return Chem.MolToSmiles(stereo_molecule(molecule), canonical=True, isomericSmiles=True)


def identity_negative_controls():
    pairs = {
        "tetrahedral_enantiomers": ("C[C@H](O)C(=O)O", "C[C@@H](O)C(=O)O"),
        "alkene_geometry": ("F/C=C/F", "F/C=C\\F"),
        "isotope_identity": ("[13CH3]CO", "CCO"),
        "specified_vs_unspecified": ("C[C@H](O)C(=O)O", "CC(O)C(=O)O"),
    }
    result = {name: {"input": values, "cleaned": [clean_identity(v) for v in values],
                     "remain_distinct": clean_identity(values[0]) != clean_identity(values[1])}
              for name, values in pairs.items()}
    return result


def failed_review_supersession_probes(families):
    """Minimal real-history replay, not a new final route projection or fix."""
    probes = []
    for family in families:
        history = family.get("key_event_critic_history") or []
        for index, row in enumerate(history):
            if row.get("critic_status") != "unavailable" or row.get("assessment"):
                continue
            obligation = row.get("review_of_obligation_id") or row.get("obligation_id")
            earlier = [h for h in history[:index]
                       if (h.get("review_of_obligation_id") or h.get("obligation_id")) == obligation
                       and h.get("assessment")]
            if not earlier:
                continue
            previous = earlier[-1]
            ids = sorted(set((row.get("required_selected_step_ids") or [])
                             + (previous.get("required_selected_step_ids") or [])))
            cards = [family.get("root_strategy_card") or {}] + (family.get("strategy_milestone_cards") or [])
            card = next(c for c in cards if c.get("strategy_digest") == row.get("strategy_digest"))
            minimal_branch = {**family, "key_event_critic_history": [previous, row]}
            steps = [{"step_id": sid} for sid in ids]
            with_failure = _selected_path_strategy_checkpoint_state(minimal_branch, strategy_card=card, steps=steps)
            without_failure = _selected_path_strategy_checkpoint_state(
                {**minimal_branch, "key_event_critic_history": [previous]}, strategy_card=card, steps=steps)
            # A real later reject must remain authoritative after its evidence edge is pruned.
            reject = {**row, "checkpoint_match": True, "critic_status": "completed",
                      "assessment": {"verdict": "reject"}}
            reject_control = _selected_path_strategy_checkpoint_state(
                {**minimal_branch, "key_event_critic_history": [previous, reject]},
                strategy_card=card, steps=[{"step_id": row["focus_step_id"]}])
            compact = lambda state: {k: state[k] for k in [
                "checkpoint_executed", "checkpoint_critic_passed", "chemical_confidence"]}
            probes.append({"route_family_id": family["route_family_id"], "task_id": row.get("task_id"),
                           "prior_task_id": previous.get("task_id"), "obligation_id": obligation,
                           "selected_step_ids_for_minimal_probe": ids,
                           "with_failed_followup": compact(with_failure),
                           "prior_valid_assessment_only": compact(without_failure),
                           "synthetic_later_reject_control": compact(reject_control)})
    return probes


def load_graph(case):
    registry = Path(case["registry_root"])
    index_path = registry / ("run_index.sqlite3" if case["group"] == "rs_context" else "runtime/run_index.sqlite3")
    with sqlite3.connect(index_path.as_uri() + "?mode=ro", uri=True) as connection:
        row = connection.execute(
            "SELECT ref_json FROM artifacts WHERE run_id=? AND artifact_id='canonical_hypergraph' ORDER BY revision DESC LIMIT 1",
            (case["case_id"],),
        ).fetchone()
    if row is None:
        raise ValueError(f"No canonical graph for {case['case_id']} in {index_path}")
    ref = json.loads(row[0])
    return read(registry / "artifacts" / ref["object_path"])


def context_diagnostics(graph):
    results = []
    original = review_context._canonical_smiles
    for family_id, family in graph["route_families"].items():
        if family.get("selected") is False:
            continue
        before, diagnostic = review_context.compile_revision_bound_route_critic_context(graph, route_family_id=family_id)
        try:
            # A process-local experiment only. No production file or run is changed.
            review_context._canonical_smiles = clean_identity
            after, after_diagnostic = review_context.compile_revision_bound_route_critic_context(graph, route_family_id=family_id)
        finally:
            review_context._canonical_smiles = original
        result = {"route_family_id": family_id, "before_compiles": before is not None,
                  "before_diagnostic": diagnostic, "after_clean_identity_compiles": after is not None,
                  "after_diagnostic": after_diagnostic}
        if diagnostic.get("edge_id"):
            edge = graph["edges"][diagnostic["edge_id"]]
            mapped = (edge.get("reactionjson_audit") or {}).get("mapped_product_smiles") or edge.get("mapped_product_smiles")
            product = edge.get("product_smiles")
            result.update(mapped_product=mapped, product=product,
                          before_product_identities=[original(mapped), original(product)],
                          cleaned_product_identities=[clean_identity(mapped), clean_identity(product)],
                          full_inchikeys=[Chem.MolToInchiKey(Chem.MolFromSmiles(v)) for v in [mapped, product]])
            if any(a.GetAtomMapNum() == 6 for a in Chem.MolFromSmiles(mapped).GetAtoms()):
                # BCH's real boronate-bearing center; test only this observed case.
                if "RDGIWRAUTQMKJZ" in result["full_inchikeys"][0]:
                    inverted = Chem.MolFromSmiles(mapped)
                    next(a for a in inverted.GetAtoms() if a.GetAtomMapNum() == 6).InvertChirality()
                    result["inverting_actual_map6_center_remains_distinct"] = (
                        clean_identity(Chem.MolToSmiles(inverted)) != clean_identity(product))
        results.append(result)
    return results


def summarize(case):
    report_path = Path(case["report_path"])
    workspace = report_path.parent / ".autoplanner/director-workspace"
    report = read(report_path)
    records = [row.get("record", row) for row in journal(workspace / "sequential-director-worker-records.jsonl")]
    ios = journal(workspace / "model-io.jsonl")
    inputs = defaultdict(list)
    outputs = defaultdict(list)
    for line_number, item in enumerate(ios, 1):
        if item.get("event") == "model_input":
            inputs[item["task_id"]].append({"line": line_number, "prompt": item.get("prompt", ""), "task_type": item.get("task_type")})
        elif item.get("event") == "model_output":
            outputs[item["task_id"]].append({"line": line_number, "stdout": item.get("stdout", "")})
    calls, key_reviews, strategy_reviews, strategies, route_reviews, editors = [], [], [], [], [], []
    builder_outputs = []
    for record_index, record in enumerate(records, 1):
        task = record["task_id"]
        artifact = record.get("output_artifact") or {}
        payload = artifact.get("payload") or {}
        metadata = record.get("metadata") or {}
        source_inputs = inputs.get(task, [])
        role = artifact.get("summary") or metadata.get("task_type") or (source_inputs[-1]["task_type"] if source_inputs else "unknown")
        usage = record.get("usage") or {}
        diagnostic = metadata.get("usage_diagnostics") or {}
        requests = diagnostic.get("requests") or {}
        row = {"record_line": record_index, "task_id": task, "role": role, "status": record.get("status"),
               "input_lines": [x["line"] for x in source_inputs],
               "output_lines": [x["line"] for x in outputs.get(task, [])],
               "input_chars": [len(x["prompt"]) for x in source_inputs],
               "usage": usage, "elapsed_s": record.get("elapsed_s"),
               "first_response_input_tokens": requests.get("first_response_input_tokens"),
               "subsequent_response_input_tokens": requests.get("subsequent_response_input_tokens"),
               "response_count": requests.get("observed_completed_responses"),
               "tools": dict(Counter(x.get("tool", "unknown") for x in record.get("tool_calls") or [])),
               "output_validation": record.get("output_validation"),
               "prompt_sizes": diagnostic.get("prompt_sizes")}
        row["is_uncertain_direct_precursor_followup"] = any(
            "new immediate upstream step after an earlier uncertain audit" in x["prompt"] for x in source_inputs)
        row["request_usage_log_path"] = requests.get("event_log_path")
        row["observed_api_attempts"] = requests.get("observed_api_attempts")
        row["observed_response_usage"] = requests.get("responses") or []
        if record.get("status") == "rejected_output":
            row["raw_output_diagnostics"] = []
            for output in outputs.get(task, []):
                try:
                    json.loads(output["stdout"])
                    parse_error = ""
                except (ValueError, TypeError) as exc:
                    parse_error = str(exc)
                row["raw_output_diagnostics"].append({"line": output["line"], "characters": len(output["stdout"]),
                                                      "json_parse_error": parse_error, "tail": output["stdout"][-160:]})
        calls.append(row)
        item = {"task_id": task, "record_line": record_index, "input_lines": row["input_lines"],
                "output_lines": row["output_lines"], "payload": payload}
        if role == "paper_matched_key_event_critic":
            key_reviews.append(item)
        elif role == "paper_matched_strategy_critic":
            strategy_reviews.append(item)
        elif role == "paper_matched_strategy_generator":
            strategies.append(item)
        elif role == "paper_matched_route_critic":
            route_reviews.append(item)
        elif role == "path_repair_editor":
            editors.append(item)
        elif role == "paper_matched_route_step":
            builder_outputs.append(item)
    families = []
    for outcome in report.get("director_outcomes", []):
        for family in (outcome.get("plan") or {}).get("route_families", []):
            families.append({key: family.get(key) for key in [
                "route_family_id", "root_strategy_card", "strategy_milestone_cards", "strategy_milestone_attempts",
                "strategy_anchor_diagnostics", "key_event_critic_history", "pending_key_event_feedback",
                "chemical_critic", "path_repair_transactions", "critic_editor_history", "editor_rejection_diagnostics",
                "materialization_diagnostics", "shared_model_budget_ledger", "paper_policy_budget_failure",
            ]})
    graph = load_graph(case)
    condition_deletion_probes = [{"task_id": r["task_id"], "input_lines": r["input_lines"],
                                 "output_lines": r["output_lines"], "condition": condition,
                                 "cleaned_condition": _clean_condition_text(condition)}
                                for r in builder_outputs for candidate in r["payload"].get("candidates", [])
                                for condition in candidate.get("conditions", [])
                                if str(condition).strip() and not _clean_condition_text(condition)]
    return {**case, "io_path": str(workspace / "model-io.jsonl"), "journal_path": str(workspace / "sequential-director-worker-records.jsonl"),
            "input_count": sum(len(v) for v in inputs.values()), "record_count": len(records), "calls": calls,
            "key_reviews": key_reviews, "strategy_reviews": strategy_reviews, "strategies": strategies,
            "route_reviews": route_reviews, "editors": editors, "builder_outputs": builder_outputs, "families": families,
            "final_context_diagnostics": context_diagnostics(graph),
            "failed_review_supersession_probes": failed_review_supersession_probes(families),
            "condition_deletion_probes": condition_deletion_probes,
            "final_review_stage": next((s for s in reversed(report.get("stages", [])) if s.get("stage") == "final_route_critic"), {}),
            "configuration": {k:report.get("config", {}).get(k) for k in ["max_evidence_tasks", "enable_condition_enrichment", "enable_builtin_patent_evidence"]}}


def group_metrics(cases):
    calls = [r for c in cases for r in c["calls"]]
    known = [r for r in calls if r["usage"]]
    followups = [r for r in calls if r["is_uncertain_direct_precursor_followup"]]
    followup_ids = {r["task_id"] for r in followups}
    reviews = [r for c in cases for r in c["key_reviews"]]
    reasons = [s for r in reviews for a in r["payload"].get("step_assessments", []) for s in a.get("reasons", [])]
    verdicts = lambda rows: dict(Counter(a.get("verdict", "missing") for r in rows
                                       for a in r["payload"].get("step_assessments", [])))
    return {
        "worker_attempts": len(calls), "attempts_by_role_from_input_fallback": dict(Counter(r["role"] for r in calls)),
        "statuses": dict(Counter(r["status"] for r in calls)),
        "known_usage_records": len(known),
        "known_usage": {k: sum(r["usage"].get(k, 0) for r in known) for k in [
            "input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens"]},
        "known_input_by_role": {role: sum(r["usage"].get("input_tokens", 0) for r in calls if r["role"] == role)
                                for role in sorted({r["role"] for r in calls})},
        "known_usage_first_response_input": sum(r["first_response_input_tokens"] or 0 for r in known),
        "known_usage_subsequent_response_input": sum(r["subsequent_response_input_tokens"] or 0 for r in known),
        "partial_response_input_outside_worker_usage": sum(
            sum(response.get("input_tokens", 0) for response in r["observed_response_usage"])
            for r in calls if not r["usage"]),
        "tool_events": dict(sum((Counter(r["tools"]) for r in calls), Counter())),
        "key_verdicts": verdicts(reviews),
        "key_reason_count": len(reasons), "key_reasons_exactly_260_chars": sum(len(s) == 260 for s in reasons),
        "strategy_card_decisions": dict(Counter(card.get("review_decision", "missing")
            for c in cases for r in c["strategy_reviews"]
            for card in (r["payload"].get("strategy_cards") or ([r["payload"]["strategy_card"]]
                          if r["payload"].get("strategy_card") else [])))),
        "condition_lines_deleted_by_current_host_normalizer": sum(len(c["condition_deletion_probes"]) for c in cases),
        "uncertain_direct_precursor_followups": {"attempts": len(followups),
            "known_input_tokens": sum(r["usage"].get("input_tokens", 0) for r in followups),
            "statuses": dict(Counter(r["status"] for r in followups)),
            "valid_verdicts": verdicts([r for r in reviews if r["task_id"] in followup_ids]),
            "task_ids": [r["task_id"] for r in followups]},
    }


def main():
    receipt = read(ROOT / "paper/arxiv/data/current_pilot_refresh_20260905.json")
    cases = []
    for case in receipt["cases"]:
        cases.append({**{k: case[k] for k in ["label", "case_id", "report_path"]},
                      "group": "current_pilots", "registry_root": str(Path(case["report_path"]).parents[2])})
    for label, folder in [
        ("R second Astra run", "absolute-config-R-astra-repeat-medium25-20260905-174851--cb44de171576"),
        ("S second Astra run", "absolute-config-S-astra-repeat-medium25-20260905-174851--f92b50b8a221"),
    ]:
        report_path = ROOT / "results/.autoplanner/runs" / folder / "target-only-solve-report.json"
        report = read(report_path)
        cases.append({"label": label, "case_id": report["run_id"], "report_path": str(report_path),
                      "group": "rs_context", "registry_root": str(ROOT / "results/.autoplanner")})
    result = {"semantics": {"read_only_saved_runs": True, "provider_calls": 0,
                            "clean_identity_probe_is_process_local_not_a_deployed_fix": True,
                            "model_verdicts_are_not_expert_truth": True}, "cases": [summarize(c) for c in cases]}
    result["identity_negative_controls"] = identity_negative_controls()
    result["group_metrics"] = {group: group_metrics([c for c in result["cases"] if c["group"] == group])
                               for group in ["current_pilots", "rs_context"]}
    path = Path(__file__).with_name("audit.json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(cases), "worker_records": sum(c["record_count"] for c in result["cases"]), "output": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
