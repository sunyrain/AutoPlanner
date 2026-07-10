from types import SimpleNamespace

from scripts.run_chem_enzy_plan_for_web import _semisynthesis_rescue_candidates
from cascade_planner.baselines.chem_enzy_onestep import _rescue_one_step_rows


def test_molecule_specific_semisynthesis_rescue_is_disabled_by_default() -> None:
    result = SimpleNamespace(target_smiles="CCO")

    assert _semisynthesis_rescue_candidates(result, {}) == []
    assert _semisynthesis_rescue_candidates(result, {"enable_semisynthesis_rescue": False}) == []


def test_one_step_provider_does_not_inject_legacy_rescue_without_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("AUTOPLANNER_ENABLE_SEMISYNTHESIS_RESCUE_PROPOSALS", raising=False)

    assert _rescue_one_step_rows("CCO") == []
