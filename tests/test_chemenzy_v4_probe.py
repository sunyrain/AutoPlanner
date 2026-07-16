from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from cascade_planner.interfaces.chemenzy_probe import (
    ChemEnzyProposalRequest,
    _guided_native_search_policy,
    _normalized_routes,
    _provider_capability_snapshot,
    _select_runtime,
)


def _preflight(prefix: Path, *, ready: bool, source: str) -> dict:
    return {
        "env_prefix": str(prefix),
        "env_prefix_selection_source": source,
        "filesystem_accepted": True,
        "production_ready": ready,
        "python_executable": str(prefix / "python.exe"),
        "issues": [] if ready else ["fixture_not_importable"],
    }


def test_explicit_chemenzy_runtime_is_never_silently_replaced(tmp_path: Path) -> None:
    explicit = tmp_path / "operator-runtime"
    report = _preflight(explicit, ready=False, source="target_solve_config")

    with (
        patch(
            "cascade_planner.interfaces.chemenzy_runtime_selection."
            "diagnose_chem_enzy_runtime",
            return_value=report,
        ) as diagnose,
        patch(
            "cascade_planner.interfaces.chemenzy_runtime_selection."
            "_registered_conda_prefixes"
        ) as discover,
    ):
        selected, discovery = _select_runtime(env_prefix=explicit, timeout_s=3.0)

    assert selected == report
    assert discovery["source"] == "target_solve_config"
    assert discovery["selected_env_prefix"] == str(explicit)
    assert diagnose.call_args.kwargs["env_prefix"] == str(explicit)
    discover.assert_not_called()


def test_chemenzy_runtime_can_fall_back_to_bounded_conda_discovery(
    tmp_path: Path,
) -> None:
    repository_default = tmp_path / "packed-linux-runtime"
    conda_runtime = tmp_path / "working-conda-runtime"
    unavailable = _preflight(repository_default, ready=False, source="default")
    available = _preflight(
        conda_runtime, ready=True, source="conda_auto_discovery"
    )

    with (
        patch(
            "cascade_planner.interfaces.chemenzy_runtime_selection."
            "diagnose_chem_enzy_runtime",
            side_effect=[unavailable, available],
        ),
        patch(
            "cascade_planner.interfaces.chemenzy_runtime_selection."
            "_registered_conda_prefixes",
            return_value=[conda_runtime],
        ),
    ):
        selected, discovery = _select_runtime(env_prefix=None, timeout_s=3.0)

    assert selected == available
    assert discovery["source"] == "conda_auto_discovery"
    assert discovery["selected_env_prefix"] == str(conda_runtime)
    assert len(discovery["attempts"]) == 2


def test_chemenzy_runtime_can_use_capability_probed_host_python(
    tmp_path: Path,
) -> None:
    repository_default = tmp_path / "packed-linux-runtime"
    host_runtime = tmp_path / "working-host-python"
    unavailable = _preflight(repository_default, ready=False, source="default")
    available = _preflight(
        host_runtime, ready=True, source="host_python_auto_discovery"
    )

    with (
        patch(
            "cascade_planner.interfaces.chemenzy_runtime_selection."
            "diagnose_chem_enzy_runtime",
            side_effect=[unavailable, available],
        ),
        patch(
            "cascade_planner.interfaces.chemenzy_runtime_selection."
            "_registered_conda_prefixes",
            return_value=[],
        ),
        patch(
            "cascade_planner.interfaces.chemenzy_runtime_selection."
            "_host_python_prefixes",
            return_value=[host_runtime],
        ),
    ):
        selected, discovery = _select_runtime(env_prefix=None, timeout_s=3.0)

    assert selected == available
    assert discovery["source"] == "host_python_auto_discovery"
    assert discovery["selected_env_prefix"] == str(host_runtime)
    assert discovery["semantics"][
        "all_auto_discovered_runtimes_require_capability_probe"
    ]


def test_provider_capability_does_not_call_import_probe_campaign_ready() -> None:
    import_only = _provider_capability_snapshot(
        {
            "runtime_preflight": {
                "filesystem_accepted": True,
                "production_ready": True,
                "env_prefix": "runtime",
                "python_executable": "runtime/python",
            }
        }
    )
    executed = _provider_capability_snapshot(
        {
            "runtime_preflight": {
                "filesystem_accepted": True,
                "production_ready": True,
                "env_prefix": "runtime",
                "python_executable": "runtime/python",
            },
            "search_executed": True,
            "ok": True,
        }
    )

    assert import_only["levels"] == {
        "discovered": True,
        "importable": True,
        "model_loadable": False,
        "smoke_tested": False,
        "campaign_ready": False,
    }
    assert executed["levels"]["campaign_ready"] is True


def test_current_launcher_route_schema_is_normalized_without_solved_flag() -> None:
    routes = _normalized_routes(
        {
            "routes": [
                {
                    "steps": [
                        {
                            "product": "CC(=O)OCC",
                            "main_reactant": "CCO",
                            "aux_reactants": ["CC(=O)Cl"],
                            "reaction_smiles": "CCO.CC(=O)Cl>>CC(=O)OCC",
                            "condition_predictions": [],
                        }
                    ]
                }
            ]
        },
        target_smiles="CC(=O)OCC",
    )

    assert len(routes) == 1
    assert routes[0]["proposal_eligible"] is True
    assert routes[0]["backend_route_status"]["solved"] is None
    assert routes[0]["steps"][0]["product_smiles"] == "CC(=O)OCC"
    assert routes[0]["steps"][0]["reactant_smiles"] == ["CCO", "CC(=O)Cl"]


def test_structurally_invalid_launcher_route_is_rejected_by_host_not_solved_flag() -> None:
    routes = _normalized_routes(
        {
            "routes": [
                {
                    "solved": True,
                    "steps": [
                        {
                            "product": "CC(=O)OCC",
                            "main_reactant": "CC(=O)OCC",
                            "aux_reactants": [],
                        }
                    ],
                }
            ]
        },
        target_smiles="CC(=O)OCC",
    )

    assert routes[0]["proposal_eligible"] is False
    assert "target_or_current_node_self_loop" in routes[0]["admission_reasons"]


def test_guided_chemenzy_request_binds_canonical_frontier_and_stop_contract() -> None:
    request_object = ChemEnzyProposalRequest(
        mode="guided_frontier",
        target_name="parent target",
        target_smiles="CCOC(C)=O",
        frontier_smiles=("CC(=O)Cl",),
        route_family_ids=("route:acyl",),
        retron_hints=("acyl substitution",),
        forbidden_smiles=("CCOC(C)=O",),
        limits={"max_routes": 2, "max_steps": 4},
        stop_conditions={"no_novel_edges": 2},
    )
    request = request_object.to_dict()

    assert request["schema_version"] == "chemenzy_proposal_request.v2"
    assert request["mode"] == "guided_frontier"
    assert request["frontier_smiles"] == ["CC(=O)Cl"]
    assert request["semantics"]["provider_has_no_private_expansion_state"] is True

    policy = _guided_native_search_policy(
        request_object,
        limits={"max_iterations": 4, "max_steps": 3, "expansion_topk": 10},
    )
    assert policy["schema_version"] == "chem_enzy_search_policy.v1"
    assert policy["mode"] == "guided"
    assert policy["terminal_blacklist"] == ["CCOC(C)=O"]
    assert policy["preferred_subgoal"]["preferred_retrons"] == [
        "acyl substitution"
    ]
    assert policy["compiler_metadata"]["not_raw_reaction_injection"] is True
