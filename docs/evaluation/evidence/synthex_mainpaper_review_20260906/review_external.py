"""One source-label-hidden critique of a public route using the production rubric.

Public SMILES have no atom maps/edit programs. Adapt only representation-specific
instructions; do not infer missing maps or inject the analyst's suspected flaws.
This diagnostic is neither an end-to-end planning trial nor independent validation.
"""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
# Resolve repository imports when invoked directly from the evidence directory.
from cascade_planner.orchestration.sequential_strategy_director import (  # noqa: E402
    SequentialStrategyDirectorRunner, _critic_prompt, _critic_task, _critique_from_record,
)
from cascade_planner.runtime import AgentSpec  # noqa: E402


def main():
    source_path = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    steps, bindings = [], []
    for ordinal, item in enumerate(source["steps"], 1):
        reactants, product = item["rxn_smiles"].split(">>")
        steps.append({"review_slot": f"review-{ordinal:03d}",
                      "product_smiles": product, "precursor_smiles": reactants.split("."),
                      "reaction_family": item["class"], "proposal_description": item["strategy"],
                      "conditions": item["conditions"],
                      "original_condition_description": item["conditions_as_given"]})
        bindings.append({"step_id": f"public-step-index-{item['idx']}", "reaction_operations": []})
    prefix = _critic_prompt(target="", branch_index=0, strategy_card={}, steps=[],
                            paper_matched=True).split("PaperMatchedRouteCriticInput:", 1)[0]
    prefix = prefix.replace(
        "Use the exact host-derived mapped products, mapped precursors, ReactionJSON operations, and proposed conditions.",
        "Use the supplied unmapped reaction SMILES and proposed conditions. No atom-map or graph-edit program was supplied; do not invent atom-map defects or infer stereochemical retention from SMILES character order.")
    prefix = "\n".join(line for line in prefix.splitlines() if not line.startswith("Atom maps are host graph-replay identities"))
    body = {"campaign_target": source["target_smiles"], "steps": steps,
            "root_strategy_card": {"strategy_query": source["steer_query"]},
            "selected_strategy_lineage": []}
    prompt = (prefix + "\nThis is a source-label-hidden external representation diagnostic. "
              "Do not browse or use tools or inspect files; judge only the supplied input. "
              "No publisher name, prior rating, risk annotations, or reference route is provided. "
              "The graph-edit-specific provenance checks are unavailable for this representation, not failed.\n"
              "ExternalRouteCriticInput:\n" + json.dumps(body, ensure_ascii=False, separators=(",", ":")))
    output.mkdir(parents=True, exist_ok=True)
    workspace = output / "worker"
    workspace.mkdir(exist_ok=True)
    (output / "prompt.txt").write_text(prompt, encoding="utf-8")
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    spec = AgentSpec(run_id="external-route-diagnostic-20260906", agent_id=f"route-diagnostic:{digest[:16]}",
                     role="route_critic", objective="Review the supplied route.",
                     idempotency_key=digest, context_hash=digest,
                     metadata={"model": "gpt-6-astra", "reasoning_effort": "medium",
                               "allowed_workdir": str(workspace), "durable_worker_journal": True})
    runner = SequentialStrategyDirectorRunner()
    runner._prepare_worker_record_journal(spec)
    task = _critic_task(spec, prompt=prompt, branch_index=0, iteration=0, timeout_s=600,
                        paper_matched=True, target_smiles=source["target_smiles"],
                        task_id_override=spec.agent_id, route_steps=bindings)
    task = replace(task, budget=replace(task.budget, max_tool_calls=0))
    print(json.dumps({"status": "started", "prompt_bytes": len(prompt.encode('utf-8'))}), flush=True)
    record = runner._run_journaled_worker(runner.critic_executor, task)
    critique = _critique_from_record(record, route_steps=bindings)
    result = {"source_path": str(source_path), "prompt_sha256": digest,
              "representation": "unmapped_public_reaction_smiles",
              "previous_assessments_withheld": True, "model_tools_budget": 0,
              "not_a_route_repair": True, "not_independent_validation": True,
              "critique": critique, "record": asdict(record)}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": record.status, "assessment": critique.get("overall_assessment")}), flush=True)


if __name__ == "__main__":
    main()
