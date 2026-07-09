from cascade_planner.agent.artifact_schemas import ARTIFACT_CLASSES, artifact_json_round_trip
from cascade_planner.agent.literature_templates import (
    ExecutableTemplateCandidate,
    LiteratureTemplateCard,
    LiteratureTemplateLevel,
    LiteratureTriggerReason,
    RouteAnchorExpansionTask,
    audit_native_run_for_literature,
    default_literature_template_cards,
    direct_consumption_allowed,
    production_kb_promotion_gate,
    route_anchor_expansion_tasks_from_templates,
    stitch_parent_child_routes,
    template_card_from_advisory_strategy,
    template_compliance_gate,
    validate_literature_template_card,
)


def test_literature_trigger_skips_native_solved_audit_passed_case():
    report = audit_native_run_for_literature(
        {
            "solved": True,
            "route_count": 1,
            "routes": [{"stock_status": {"CCO": True}, "steps": [{"stock_status": {"CCO": True}}]}],
        },
        route_audit={"route_status": "solved", "stock_audit_passed": True, "reasons": []},
    )

    assert report["should_trigger"] is False
    assert report["native_audit_passed"] is True
    assert report["trigger_reasons"] == []


def test_literature_trigger_reasons_cover_native_failed_unclosed_and_fake_closure():
    failed = audit_native_run_for_literature(
        {"solved": False, "route_count": 0, "routes": [], "failures": [{"category": "no_route_found"}]},
        route_audit={"route_status": "unresolved", "stock_audit_passed": False, "reasons": ["native_failed"]},
    )
    unclosed = audit_native_run_for_literature(
        {
            "solved": False,
            "route_count": 1,
            "routes": [{"stock_status": {"advanced_anchor": False}, "steps": [{"stock_status": {"advanced_anchor": False}}]}],
        },
        route_audit={"route_status": "unresolved", "stock_audit_passed": False, "reasons": ["unclosed_route"]},
    )
    fake = audit_native_run_for_literature(
        {
            "solved": True,
            "route_count": 1,
            "routes": [{"stock_status": {"same_scaffold_leaf": True}, "steps": [{"stock_status": {"same_scaffold_leaf": True}}]}],
        },
        route_audit={
            "route_status": "fake_closed_rejected",
            "stock_audit_passed": False,
            "fake_closure_rejected": True,
            "reasons": ["advanced_same_scaffold", "no_complexity_drop"],
        },
    )

    assert LiteratureTriggerReason.NATIVE_FAILED.value in failed["trigger_reasons"]
    assert LiteratureTriggerReason.UNCLOSED_ROUTE.value in unclosed["trigger_reasons"]
    assert LiteratureTriggerReason.FAKE_CLOSURE_RISK.value in fake["trigger_reasons"]
    assert all(item["should_trigger"] for item in (failed, unclosed, fake))


def test_advisory_and_route_anchor_templates_cannot_be_direct_consumed():
    advisory = template_card_from_advisory_strategy(
        {
            "template_schema": "advisory_strategy_template.v1",
            "reaction_class": "glycosylation",
            "break_bonds": ["anomeric C-O"],
        },
        evidence_refs=["ev_advisory"],
    )
    anchor = next(card for card in default_literature_template_cards() if card.template_level == LiteratureTemplateLevel.ROUTE_ANCHOR_ONLY.value)

    assert validate_literature_template_card(advisory)["accepted"]
    assert not direct_consumption_allowed(advisory)
    assert not direct_consumption_allowed(anchor)
    assert validate_literature_template_card(anchor)["direct_consumption_allowed"] is False


def test_executable_template_card_and_artifacts_are_registered():
    card = next(card for card in default_literature_template_cards() if card.template_id == "lit_tpl_o_glycoside_split_v1")
    validation = validate_literature_template_card(card)
    artifact = ARTIFACT_CLASSES["LiteratureTemplateCard"](
        artifact_id="lit_tpl",
        case_id="case",
        source="unit",
        evidence_refs=["ev"],
        validation_status="validated",
        payload=card.to_dict(),
    )

    assert validation["accepted"], validation
    assert validation["direct_consumption_allowed"] is True
    assert "TemplateApplicabilityReport" in ARTIFACT_CLASSES
    assert "ExecutableTemplateCandidate" in ARTIFACT_CLASSES
    assert "AnalogicalReactionTemplateReport" in ARTIFACT_CLASSES
    assert artifact_json_round_trip(artifact).to_dict() == artifact.to_dict()


def test_analogical_reaction_template_report_artifact_round_trips():
    artifact = ARTIFACT_CLASSES["AnalogicalReactionTemplateReport"](
        artifact_id="analog_tpl_report",
        case_id="case",
        source="unit",
        evidence_refs=["doi:analog"],
        validation_status="draft",
        payload={
            "schema_version": "analogical_reaction_template_report.v1",
            "accepted": True,
            "templates": [],
            "no_solved_claim": True,
        },
    )

    assert artifact_json_round_trip(artifact).to_dict() == artifact.to_dict()


def test_route_anchor_tasks_and_stitching_do_not_claim_solved_for_unclosed_child():
    cards = default_literature_template_cards()
    tasks = route_anchor_expansion_tasks_from_templates(cards, parent_route_reference="parent_1")
    taxane_task = next(task for task in tasks if task.anchor_name == "baccatin_or_10_DAB")
    stitched = stitch_parent_child_routes(
        {"route_id": "parent_1", "route_status": "partial_anchor", "steps": [{"product": "taxane"}]},
        [{"route_status": "unresolved", "steps": []}],
        anchor_tasks=[taxane_task],
        all_leaf_audit_passed=False,
    )

    assert isinstance(taxane_task, RouteAnchorExpansionTask)
    assert taxane_task.native_first is True
    assert stitched["route_status"] == "partial_anchor"
    assert stitched["solved_claim_allowed"] is False
    assert stitched["unresolved_anchor_count"] == 1


def test_compliance_and_kb_promotion_gates_keep_target_run_out_of_production_kb():
    card = LiteratureTemplateCard(
        template_id="danger_tpl",
        evidence_refs=["ev"],
        reaction_class="test",
        template_level=LiteratureTemplateLevel.VALIDATED_EXECUTABLE_TEMPLATE.value,
        product_retron={"retron_type": "o_glycoside"},
        safety_flags=["dual_use"],
        promotion_status="validated",
    )
    compliance = template_compliance_gate(card)
    promotion = production_kb_promotion_gate(
        card,
        replicated_case_count=2,
        negative_controls_passed=True,
        source_evidence_stable=True,
        from_target_run=True,
        validation_reports=[{"allowed_for_one_step_source": True}],
    )

    assert compliance["accepted"] is False
    assert "safety_flag:dual_use" in compliance["reasons"]
    assert promotion["accepted"] is False
    assert "target_run_direct_kb_write_forbidden" in promotion["reasons"]
