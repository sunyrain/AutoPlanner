"""Historical enzyme edits through the actual wire/replay/route/review boundary.

The two examples retain the operations and conditions of the 2026-09-15 statin
IO. execution_domain/catalyst are added ONLY by the test as new-protocol input;
the historical records did not contain them and are never rewritten.
"""
from copy import deepcopy
import json

import pytest

from cascade_planner.agent.codex_worker import (
    WorkerBudget, WorkerProcessResult, WorkerTask, run_codex_worker,
    _worker_model_output_json_schema,
)
from cascade_planner.application.biocatalytic_step_contract import normalize_biocatalytic_step
from cascade_planner.application.strategy_contract import normalize_strategy_card
from cascade_planner.orchestration import sequential_strategy_director as d
from cascade_planner.orchestration.chemical_reasoning import CATALYSIS_PLANNING_GUIDANCE


CASES = [
    # statin-retrosynthesis-full-20260915/pitavastatin: input 44, output 47.
    {
        "id": "kred", "target": "COC(=O)C[C@H](O)C[C@H](O)CO",
        "mapped": "[O:1]=[C:2]([O:3][CH3:32])[CH2:4][C@H:5]([OH:6])[CH2:7][C@H:8]([OH:9])[CH2:10][OH:1021]",
        "precursor": "COC(=O)CC(=O)C[C@H](O)CO",
        "catalyst": "ketoreductase panel (candidate not selected)",
        "wire": {
            "checkpoint_relation": "preparatory",
            "reaction_intent": "Stereoselective ketoreductase reduction of methyl (5S)-5,6-dihydroxy-3-oxohexanoate installs the required 3R alcohol while retaining the 5S center and methyl ester.",
            "reaction_operations": [
                {"op": "clear_stereocenter", "map_idx": 5},
                {"op": "set_explicit_h", "map_idx": 5, "count": 0, "no_implicit": True},
                {"op": "set_explicit_h", "map_idx": 6, "count": 0, "no_implicit": True},
                {"op": "change_bond_order", "map_a": 5, "map_b": 6, "delta": 1},
            ],
            "conditions": ["Hypothesis: screen ketoreductases for selective formation of the (3R,5S) triol from the exact unprotected ketoester; use NADPH with glucose dehydrogenase/glucose cofactor recycling in near-neutral aqueous buffer, then remove protein and isolate under mild conditions. Confirm diastereoselectivity, methyl-ester retention, and substrate stability."],
        },
    },
    # statin-retrosynthesis-20260915/fluvastatin: input 61, output 62.
    {
        "id": "esterase", "target": "COC(=O)C[C@@H](O)CC(=O)O",
        "mapped": "[C:8](=[O:9])([CH2:10][C@H:11]([OH:12])[CH2:13][C:14](=[O:15])[O:16][CH3:31])[OH:52]",
        "precursor": "COC(=O)CC(O)CC(=O)OC",
        "catalyst": "esterase panel (candidate not selected)",
        "wire": {
            "checkpoint_relation": "executes_checkpoint",
            "reaction_intent": "Reverse enantioselective esterase-mediated monohydrolysis of symmetric dimethyl 3-hydroxyglutarate to the exact map-11 S monoacid.",
            "reaction_operations": [
                {"op": "clear_stereocenter", "map_idx": 11},
                {"op": "set_explicit_h", "map_idx": 52, "count": 0, "no_implicit": True},
                {"op": "add_group", "map_idx": 52, "fragment_smiles": "[*]C"},
            ],
            "conditions": [
                "Hypothesis: screen esterases in aqueous buffer for preferential cleavage of the map-8 methyl ester, furnishing the exact map-11 S monoacid while retaining the map-14 methyl ester and free alcohol; substrate-specific selectivity is unverified and search allowance is exhausted.",
                "Forward order: conduct controlled enzymatic hydrolysis, monitor monoacid/diacid formation and enantiomeric purity, stop and remove enzyme before excessive second hydrolysis, then mildly acidify and isolate the monoacid; verify absolute configuration rather than infer it from enzyme identity.",
            ],
        },
    },
]


def _replay(case, wire):
    task = WorkerTask(
        task_id=case["id"], case_id=case["id"], task_type="paper_matched_route_step",
        required_artifact_type="RetrosynthesisProposalReport", budget=WorkerBudget(max_tool_calls=0),
        host_context={"target_smiles": case["target"], "selected_product": case["target"]},
    )
    record = run_codex_worker(task, runner=lambda _: WorkerProcessResult(
        stdout=json.dumps(wire), exit_code=0, backend="runner"))
    assert record.status == "accepted_draft"
    return d._reactionjson_candidates_from_record(
        record, expected_product=case["target"], mapped_product_smiles=case["mapped"],
        require_reaction_operations=True,
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_enzyme_identity_survives_real_replay_serialization_and_review(case):
    wire = {**deepcopy(case["wire"]), "execution_domain": "enzymatic", "catalyst": case["catalyst"]}
    candidates, failures = _replay(case, wire)
    assert not failures and len(candidates) == 1
    step = d._step_row(candidates[0].expansion, step_id=case["id"])
    assert step["precursor_smiles"] == [case["precursor"]]
    route = d._host_route_json_from_steps([step])
    for row in (step, route[0]):
        assert row["execution_domain"] == "enzymatic"
        bio = row["biocatalytic_step"]
        assert bio["catalyst_hypothesis"]["enzyme_label"] == case["catalyst"]
        assert bio["boundary"]["forward_input_smiles"] == [case["precursor"]]
        assert bio["validation_gate"]["accepted"] is False
        assert bio["authority_scope"] == "model_proposed_execution_hypothesis"
    assert route[0]["enzyme"] == route[0]["catalyst"] == case["catalyst"]
    assert route[0]["conditions"] == wire["conditions"]
    for level in range(3):
        for projection in (d._critic_step_row, d._paper_critic_step_row):
            projected = projection(step, compact_level=level)
            assert projected["execution_domain"] == "enzymatic"
            encoded = json.dumps(projected)
            assert case["catalyst"] in encoded
            for condition in wire["conditions"]:
                assert condition in encoded
    for kind in ("key_event", "final_route"):
        prompt = d._critic_prompt(target=case["target"], branch_index=0, strategy_card={},
                                 steps=[step], paper_matched=True, audit_kind=kind,
                                 focus_step_id=case["id"])
        assert CATALYSIS_PLANNING_GUIDANCE in prompt
        assert case["catalyst"] in prompt
    editor = d._minimal_editor_prompt_route_rows([step])[0]
    assert editor["execution_domain"] == "enzymatic"
    assert editor["catalyst"] == case["catalyst"]
    assert editor["conditions"] == wire["conditions"]

    bad_wire = deepcopy(wire)
    bad_wire["reaction_operations"][0]["map_idx"] = 999999
    candidates, failures = _replay(case, bad_wire)
    assert not candidates and failures  # Enzyme metadata cannot bypass structural replay.


def test_domain_is_explicit_not_guessed_from_prose_or_branch():
    case = CASES[0]
    for metadata in ({}, {"execution_domain": "chemical", "catalyst": "Ru catalyst"}):
        candidates, failures = _replay(case, {**deepcopy(case["wire"]), **metadata})
        assert not failures
        step = d._step_row(candidates[0].expansion, step_id="chemical-or-legacy")
        # In particular, historical prose alone must not rewrite old identities.
        assert step["execution_domain"] == "chemical"
        assert step["biocatalytic_step"] == {}


def test_nested_catalyst_and_cofactor_survive_compaction_without_proof_metadata():
    bio, _ = normalize_biocatalytic_step(
        {"enzyme_label": "KRED panel", "cofactor_assessment": "required",
         "cofactor_requirements": ["NADPH"], "cofactor_regenerations": ["GDH/glucose"],
         "cosubstrates": ["glucose"], "selectivity_objective": "3R alcohol"},
        execution_domain="enzymatic", product_smiles=CASES[0]["target"],
        precursor_smiles=[CASES[0]["precursor"]],
    )
    step = {"step_id": "focus", "execution_domain": "enzymatic", "biocatalytic_step": bio,
            "condition_predictions": [{"enzyme": "KRED panel", "reagents": ["Proposed screen"]}]}
    rows = [d._critic_step_row(step, compact_level=1),
            d._critic_step_row(step, compact_level=2),
            d._paper_critic_step_row(step, compact_level=2),
            d._minimal_editor_prompt_route_rows([step])[0]]
    for row in rows:
        compact = row["biocatalytic_step"]
        assert compact["catalyst_hypothesis"] == bio["catalyst_hypothesis"]
        assert compact["cofactor_ledger"] == bio["cofactor_ledger"]
        assert "validation_gate" not in compact
        assert "content_sha256" not in compact


@pytest.mark.parametrize("domain", ["enzymatic", "whole_cell", "hybrid", "chemical"])
def test_missing_enzyme_evidence_stays_uncertain_without_query_loop(domain):
    card = normalize_strategy_card({"strategy_query": "stereoselective reduction", "critical_assumption": "selectivity",
                                    "critic_checkpoint": "carbonyl reduction"})
    row = {"task_id": "initial", "focus_step_id": "focus", "obligation_id": "obligation",
           "strategy_digest": d._strategy_card_digest(card), "checkpoint_match": True,
           "assessment": {"verdict": "uncertain", "uncertainty_source": "evidence_missing"}}
    branch = {"strategy_card": card, "key_event_critic_history": [row]}
    steps = [{"step_id": "focus", "execution_domain": domain}]
    saved = deepcopy(branch)
    review = d._pending_uncertain_key_event_evidence_review(branch, steps=steps, allow_evidence_query=True)
    assert bool(review) == (domain == "chemical")
    assert branch == saved  # No silent conversion to pass/reject or history mutation.
    assert not d._needs_proposal_clarification(row, [])
    row["assessment"]["uncertainty_source"] = "assessment_unresolved"
    assert d._pending_uncertain_key_event_evidence_review(branch, steps=steps)
    row["assessment"]["uncertainty_source"] = "proposal_underspecified"
    assert d._needs_proposal_clarification(row, [])
    assert not d._needs_proposal_clarification(row, [row])
    row["assessment"]["verdict"] = "reject"
    assert not d._needs_proposal_clarification(row, [])
    assert row["assessment"]["verdict"] == "reject"


def test_planning_guidance_reaches_strategy_builder_and_editor():
    prompts = [
        d._paper_strategy_portfolio_prompt(target="CCO"),
        d._paper_strategy_portfolio_critic_prompt(target="CCO", strategy_cards=[]),
        d._node_prompt(target="CCO", branch_index=0, lens="test", selected_product="CCO",
                       selected_product_mapped="[CH3:1][CH2:2][OH:3]", steps=[], open_leaves=["CCO"],
                       prior_rejections=[], repair=False, strategy_card={}, forbidden_strategy_cards=(),
                       host_failure_feedback={}, paper_matched=True),
        d._path_repair_editor_prompt(target="CCO", strategy_card={}, repair_mode="route_span",
                                    steps=[], critic_feedback={}),
    ]
    for prompt in prompts:
        assert CATALYSIS_PLANNING_GUIDANCE in prompt
    for kind in ("paper_matched_route_step", "paper_matched_route_editor"):
        task = WorkerTask(task_id=kind, case_id="test", task_type=kind,
                          required_artifact_type="RetrosynthesisProposalReport")
        schema = _worker_model_output_json_schema(task)
        if kind.endswith("editor"):
            schema = schema["properties"]["replace_span"]["properties"]["revised_steps"]["items"]
        assert {"execution_domain", "catalyst"} <= set(schema["required"])
        assert schema["properties"]["execution_domain"]["enum"] == ["chemical", "enzymatic", "whole_cell", "hybrid"]
        assert "biocatalytic_step" not in schema["properties"]  # No self-authored proof.


def test_compact_enzyme_proposal_has_no_fabricated_missing_data_or_duplicate_review_copy():
    case = CASES[0]
    wire = {**deepcopy(case["wire"]), "execution_domain": "enzymatic", "catalyst": case["catalyst"]}
    candidates, failures = _replay(case, wire)
    assert not failures
    step = d._step_row(candidates[0].expansion, step_id="screen")
    bio = step["biocatalytic_step"]
    assert bio["catalyst_hypothesis"] == {"enzyme_label": case["catalyst"]}
    assert not {"cofactor_ledger", "selectivity_objective", "design_complete", "design_deficits"} & bio.keys()
    assert "biocatalytic_design_deficits" not in step
    assert bio["validation_gate"]["accepted"] is False
    again, reasons = normalize_biocatalytic_step(
        bio, execution_domain="enzymatic", product_smiles=case["target"],
        precursor_smiles=[case["precursor"]], step_id="screen",
    )
    assert not reasons and again == bio
    for level in range(3):
        row = d._paper_critic_step_row(step, compact_level=level)
        assert "biocatalytic_step" not in row
        assert row["catalyst"] == case["catalyst"]
        assert row["conditions"] == wire["conditions"]


def test_process_strategy_diversity_and_handoff_are_not_blanket_evidence_exemptions():
    from cascade_planner.orchestration.chemical_reasoning import STRATEGY_DIVERSITY_GUIDANCE
    generator = d._paper_strategy_portfolio_prompt(target="CCO", enhanced=True)
    critic = d._paper_strategy_portfolio_critic_prompt(target="CCO", strategy_cards=[])
    assert STRATEGY_DIVERSITY_GUIDANCE in generator and STRATEGY_DIVERSITY_GUIDANCE in critic
    assert "For process or selectivity development" in generator
    assert "one graph transformation" in generator
    assert "A missing handoff design is proposal_underspecified" in CATALYSIS_PLANNING_GUIDANCE
