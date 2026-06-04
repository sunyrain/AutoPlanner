from cascade_planner.agent.chem_enzy_policy import apply_literature_template_plugin_policy
from cascade_planner.baselines.route_contract import RouteSearchConfig
from scripts.run_literature_template_plugin_benchmark import run_literature_template_plugin_benchmark


def test_literature_template_plugin_benchmark_reports_ab_configs_and_required_metrics():
    payload = run_literature_template_plugin_benchmark()
    summary = payload["summary"]

    assert payload["schema_version"] == "literature_template_plugin_benchmark.v1"
    assert payload["configs"] == ["native", "policy_only", "plugin"]
    assert summary["improved_native_failed_or_unclosed_count"] >= 2
    assert summary["negative_controls_ok"] is True
    assert summary["all_plugin_steps_have_evidence_and_validation"] is True
    assert summary["literature_plugin_step_precision"] == 1.0
    assert summary["reconstruction_pass_rate"] == 1.0
    assert "solved_rate" in payload["metrics"]
    assert "fake_closure_rejection_rate" in payload["metrics"]
    assert "route_audit_pass_rate" in payload["metrics"]


def test_phenolic_glycoside_negative_control_is_not_literature_gain():
    payload = run_literature_template_plugin_benchmark()
    control = next(case for case in payload["cases"] if case["case_id"] == "phenolic_glycoside_native_solved_negative_control")

    assert control["negative_control"] is True
    assert control["trigger_report"]["should_trigger"] is False
    assert control["plugin"]["route_count"] == 0
    assert control["plugin"]["improved_over_native"] is False


def test_apply_literature_template_plugin_policy_only_enables_on_explicit_trigger():
    base = RouteSearchConfig(target_smiles="CCO")
    skipped = apply_literature_template_plugin_policy(base, trigger_report={"should_trigger": False, "trigger_reasons": []})
    enabled = apply_literature_template_plugin_policy(
        base,
        trigger_report={"should_trigger": True, "trigger_reasons": ["native_failed"]},
        top_k=3,
        max_added=2,
    )

    assert "literature_template_plugin" not in skipped.search_flags
    assert enabled.search_flags["literature_template_plugin"]["enabled"] is True
    assert enabled.search_flags["literature_template_plugin"]["top_k"] == 3
    assert enabled.search_flags["literature_template_plugin"]["max_added"] == 2
    assert enabled.search_flags["cascade_search_context"]["literature_template_plugin_enabled"] is True
    assert enabled.search_flags["cascade_source_policy"]["literature_template_plugin"]["domain"] == "literature_chemical"
