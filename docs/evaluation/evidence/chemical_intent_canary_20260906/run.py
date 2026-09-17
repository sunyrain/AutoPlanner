"""Probe a saved route with production causal Critic and one repair transaction.

Default is preparation only. --execute performs a new Critic call, then at most
one Editor transaction with six Builder calls and a reserved whole-route review.
The original run/registry is never opened for writing. This diagnoses integration;
it is not a matched efficacy experiment or a chemical validation of the product.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from cascade_planner.application.campaign_context import CampaignContext, CampaignContextDelta  # noqa: E402
from cascade_planner.application.route_review_context import RevisionBoundRouteCriticContext  # noqa: E402
from cascade_planner.application.run_kernel import RunRevision  # noqa: E402
from cascade_planner.interfaces.live_stock import standard_stock_catalog_builder  # noqa: E402
from cascade_planner.orchestration.sequential_strategy_director import (  # noqa: E402
    DirectorConfig, SequentialStrategyDirectorRunner,
)
from cascade_planner.runtime import AgentSpec, Budget  # noqa: E402


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              default=lambda x: list(x) if isinstance(x, deque) else str(x)) + "\n",
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--critique", type=Path, help="Use a saved critique unchanged to isolate repair execution.")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    source = args.context.resolve()
    output = args.output.resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    context = RevisionBoundRouteCriticContext(**raw)
    config = DirectorConfig(
        paper_matched_reach_profile=True, enable_transactional_path_repair=True,
        strategy_tree_engine="aizynthfinder_mcts", strategy_portfolio_mode="paper_independent",
        strategy_branch_count=1, strategy_branch_workers=1, max_node_expansions_per_branch=6,
        max_route_local_repair_rounds=1, max_node_prompt_bytes=96_000,
        max_node_call_timeout_s=300, critic_call_timeout_s=300, max_wall_time_s=1500,
        max_output_tokens=8000, model="gpt-6-astra", reasoning_effort="medium",
    )
    runner = SequentialStrategyDirectorRunner()
    prompt = runner.final_route_critic_prompt_for(context, config)
    if not prompt:
        raise ValueError("Saved route exceeds the production prompt bound")
    manifest = {"source_context": str(source), "source_sha256": digest(raw),
                "source_route_sha256": context.route_sha256, "config": asdict(config),
                "scope": "isolated saved-route diagnostic; no original run mutation",
                "prepared_at": datetime.now(timezone.utc).isoformat()}
    if args.critique:
        manifest["fixed_critique_source"] = str(args.critique.resolve())
    manifest["implementation_files"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            "cascade_planner/orchestration/chemical_reasoning.py",
            "cascade_planner/orchestration/repair_scope.py",
            "cascade_planner/orchestration/sequential_strategy_director.py",
            "cascade_planner/agent/codex_worker.py",
        )
    }
    write(output / "manifest.json", manifest)
    (output / "critic-prompt.txt").write_text(prompt, encoding="utf-8")
    if not args.execute:
        print(json.dumps({"status": "prepared", "prompt_bytes": len(prompt.encode())}), flush=True)
        return

    run_id = "chemical-intent-probe:" + output.name
    metadata = {"model": config.model, "reasoning_effort": config.reasoning_effort,
                "durable_worker_journal": True}
    critic_spec = AgentSpec(
        run_id=run_id, agent_id="intent-final-critic", role="route_critic",
        objective="Independently review the supplied saved route and its chemical prerequisites.",
        idempotency_key=digest(prompt), context_hash=digest(raw),
        metadata={**metadata, "allowed_workdir": str(output / "critic-worker")},
    )
    if args.critique:
        saved = json.loads(args.critique.read_text(encoding="utf-8"))
        critique = saved["critique"]
        write(output / "fixed-input-critique.json", saved)
        print("Using the saved critique unchanged; no new detection call", flush=True)
    else:
        print("Starting one production final Critic", flush=True)
        critique, record = runner.run_final_route_critic_once(critic_spec, context=context, config=config, prompt=prompt)
        write(output / "critic-result.json", {"critique": critique, "record": asdict(record)})
        print(json.dumps({"status": record.status, "assessment": critique.get("overall_assessment"),
                          "dependencies": critique.get("chemical_dependencies")}), flush=True)
    if critique.get("overall_assessment") != "reject":
        write(output / "result.json", {"status": "no_concrete_reject_for_repair", "critique": critique})
        return

    # Bind the same stock identity to both Director and actual AiZ sidecar.
    stock = standard_stock_catalog_builder()

    def membership(values):
        values = list(values)
        catalog = stock(values, max_molecules=max(1, len(values)))
        members = {row["canonical_smiles"] for row in catalog["members"]}
        from cascade_planner.application.blind_benchmark_contract import canonical_smiles
        return {value: canonical_smiles(value) in members for value in values}

    runner = SequentialStrategyDirectorRunner(
        stock_membership=membership,
        aizynthfinder_strategy_python_executable=str(ROOT / ".venv_aizynth/Scripts/python.exe"),
        aizynthfinder_strategy_stock_index=str(stock.index_path),
    )
    # Minimal isolated projection for production plan assembly. Revision 0 is
    # this diagnostic, not the original canonical campaign's graph revision.
    campaign = CampaignContext(
        run_id=run_id, target={"canonical_smiles": context.target_smiles},
        revision=RunRevision(run_id=run_id, revision=0, state_sha256=digest(raw),
                             graph_revision=0, evidence_revision=0, deficit_sha256=digest([]),
                             acceptance_sha256=digest({}), status="isolated_diagnostic",
                             updated_at=manifest["prepared_at"]),
        topology={"source_route_context_sha256": digest(raw)}, route_portfolio={},
        evidence={}, stock={}, deficits=(), proposal_history=(), failure_history=(),
        budget_state={}, acceptance_state={}, delta=CampaignContextDelta(),
    )
    spec = AgentSpec(
        run_id=run_id, agent_id="intent-repair", role="route_editor",
        objective="Resolve the concrete Critic defect through the ordinary graph repair transaction.",
        idempotency_key=digest([raw, critique]), context_hash=campaign.content_sha256,
        budget=Budget(max_wall_time_s=1500),
        metadata={**metadata, "allowed_workdir": str(output / "repair-worker"),
                  "remaining_model_budget": {"model_invocations": 8, "input_tokens": 500000,
                                             "output_tokens": 64000, "wall_time_s": 1500}},
    )
    print("Starting at most one Editor / six Builder / one re-Critic transaction", flush=True)
    result = runner.run_final_route_repair_once(
        spec, campaign_context=campaign, route_context=context, critique=critique,
        config=config, route_family_alias="isolated:chemical-intent-probe",
    )
    write(output / "result.json", result)
    print(json.dumps({"status": result["status"], "reason": result.get("reason"),
                      "usage": result.get("usage")}), flush=True)


if __name__ == "__main__":
    main()
