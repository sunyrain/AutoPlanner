import importlib
import os
from pathlib import Path
import sqlite3
import sys
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cascade_planner.baselines.chem_enzy_adapter import (
    ChemEnzyBackendAdapter,
    _SqliteStockMembership,
    _bounded_materialization_result,
    _install_bounded_vendor_mcts,
    _seed_runtime_state,
    audit_materialized_chem_enzy_route,
    route_candidates_from_chem_enzy_result,
)
from cascade_planner.baselines.chem_enzy_budget import (
    classify_chemenzy_attempt_outcome,
    resolve_chemenzy_budget,
)
from cascade_planner.baselines.chem_enzy_bounded_mcts import bounded_mol_planner
from cascade_planner.baselines.route_contract import RouteStepCandidate
from cascade_planner.baselines.route_contract import BaselineRunResult, RouteSearchConfig
from scripts.run_chem_enzy_plan_for_web import (
    _route_config_from_payload,
    _web_payload_from_result,
)


BAD_PRODUCT = "O=C(O)C(O)(CCO)C(=O)OCc1ccccc1"
BAD_REACTANTS = ["O=C(Cl)OCc1ccccc1", "O=C([O-])[O-]"]


def test_runtime_seed_binding_seeds_python_numpy_and_torch() -> None:
    numpy_seed = Mock()
    torch_seed = Mock()
    cuda_seed = Mock()
    fake_numpy = SimpleNamespace(random=SimpleNamespace(seed=numpy_seed))
    fake_torch = SimpleNamespace(
        manual_seed=torch_seed,
        cuda=SimpleNamespace(is_available=lambda: True, manual_seed_all=cuda_seed),
    )
    with (
        patch.dict(sys.modules, {"numpy": fake_numpy, "torch": fake_torch}),
        patch.dict(os.environ, {"PYTHONHASHSEED": "17"}),
        patch("cascade_planner.baselines.chem_enzy_adapter.random.seed") as python_seed,
    ):
        binding = _seed_runtime_state(17)

    python_seed.assert_called_once_with(17)
    numpy_seed.assert_called_once_with(17)
    torch_seed.assert_called_once_with(17)
    cuda_seed.assert_called_once_with(17)
    assert binding["python_hash_seed_matches"] is True
    assert binding["deterministic_algorithms_enabled"] is False


def test_sqlite_stock_membership_supports_mcts_overlay_and_deepcopy(tmp_path: Path) -> None:
    path = tmp_path / "stock.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE stock (canonical_smiles TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        connection.executemany(
            "INSERT INTO stock(canonical_smiles) VALUES (?)",
            [("CCO",), ("CCN",)],
        )
        connection.commit()

    membership = _SqliteStockMembership(path)
    assert "CCO" in membership
    assert "CCC" not in membership
    membership.discard("CCO")
    membership.add("CCC")
    replay = deepcopy(membership)
    assert "CCO" in replay
    assert "CCC" not in replay
    assert len(replay) == 2


def _raw_one_step_route(
    product: str,
    reactants: list[str],
    *,
    template_id: str,
) -> dict:
    return {
        "all_succ_dict_routes": [
            {
                "type": "mol",
                "smiles": product,
                "children": [
                    {
                        "type": "reaction",
                        "score": 0.99,
                        "template": {
                            "template_id": template_id,
                            "source": "native_graphfp",
                            "model_full_name": "graphfp_models.test",
                        },
                        "children": [
                            {"type": "mol", "smiles": reactant, "in_stock": True}
                            for reactant in reactants
                        ],
                    }
                ],
            }
        ],
        "iter": 1,
    }


def _step(product: str, reactants: list[str], index: int) -> RouteStepCandidate:
    return RouteStepCandidate(
        product_smiles=product,
        reactant_smiles=reactants,
        rxn_smiles=f"{'.'.join(reactants)}>>{product}",
        source_model="graphfp_models.test",
        score=0.9,
        raw_backend_metadata={
            "template": {
                "template_id": f"template-{index}",
                "source": "native_graphfp",
                "model_full_name": "graphfp_models.test",
            }
        },
    )


def test_flattened_bad_edge_is_diagnostic_raw_solved_but_never_host_solved() -> None:
    raw = _raw_one_step_route(
        BAD_PRODUCT,
        BAD_REACTANTS,
        template_id="bad-nirmatrelvir-edge",
    )

    routes = route_candidates_from_chem_enzy_result(raw, target_smiles=BAD_PRODUCT)

    assert len(routes) == 1
    route = routes[0]
    assert len(route.steps) == 1
    assert route.solved is False
    assert route.raw_backend_metadata["raw_solved"] is True
    assert route.raw_backend_metadata["route_outcome"] == "verification_rejected"
    admission = route.raw_backend_metadata["route_materialization_admission"]
    assert admission["accepted"] is False
    assert admission["route_outcome"] == "verification_rejected"
    assert admission["reasons"] == ["element_inventory_not_conserved"]
    assert admission["accepted_edge_count"] == 0
    assert admission["rejected_edge_count"] == 1
    rejection = admission["rejected_edges"][0]
    assert rejection["stage"] == "final_route_materialization"
    assert rejection["source"] == "native_graphfp"
    assert rejection["model"] == "graphfp_models.test"
    assert rejection["template"]["fields"]["template_id"] == "bad-nirmatrelvir-edge"
    assert len(rejection["edge_digest"]) == 64


def test_any_bad_edge_rejects_an_entire_fifteen_step_materialized_route() -> None:
    steps = [_step("CCO", ["CC=O"], index) for index in range(15)]
    steps[9] = _step(BAD_PRODUCT, BAD_REACTANTS, 9)

    admission = audit_materialized_chem_enzy_route(steps, route_index=4)

    assert admission["accepted"] is False
    assert admission["route_outcome"] == "verification_rejected"
    assert admission["raw_solved"] is True
    assert admission["host_admission_accepted"] is False
    assert admission["edge_count"] == 15
    assert admission["accepted_edge_count"] == 14
    assert admission["rejected_edge_count"] == 1
    assert admission["rejected_edges"][0]["candidate_index"] == 9
    assert admission["rejected_edges"][0]["reasons"] == [
        "element_inventory_not_conserved"
    ]


def test_valid_materialized_route_is_not_falsely_rejected() -> None:
    raw = _raw_one_step_route("CCO", ["CC=O"], template_id="valid-oxidation")

    routes = route_candidates_from_chem_enzy_result(raw, target_smiles="CCO")

    assert len(routes) == 1
    route = routes[0]
    assert route.solved is True
    assert route.raw_backend_metadata["raw_solved"] is True
    assert route.raw_backend_metadata["route_outcome"] == "admitted_raw_solved"
    admission = route.raw_backend_metadata["route_materialization_admission"]
    assert admission["accepted"] is True
    assert admission["accepted_edge_count"] == 1
    assert admission["rejected_edge_count"] == 0
    assert admission["reasons"] == []


def test_web_export_keeps_raw_rejection_diagnostic_without_solved_promotion() -> None:
    route = route_candidates_from_chem_enzy_result(
        _raw_one_step_route(
            BAD_PRODUCT,
            BAD_REACTANTS,
            template_id="bad-nirmatrelvir-edge",
        ),
        target_smiles=BAD_PRODUCT,
    )[0]
    result = BaselineRunResult(
        target_smiles=BAD_PRODUCT,
        backend="ChemEnzyRetroPlanner",
        routes=[route],
    )

    payload = _web_payload_from_result(
        result,
        {},
        RouteSearchConfig(target_smiles=BAD_PRODUCT),
        0.1,
    )

    assert payload["raw_solved"] is True
    assert payload["materialization_admission_solved"] is False
    assert payload["search_status"]["raw_solved"] is True
    assert payload["search_status"]["solved"] is False
    assert payload["search_status"]["status"] == "partial"
    assert payload["routes"][0]["metrics"]["raw_backend_solved_not_proof"] is True
    assert payload["routes"][0]["metrics"]["route_solved"] is False
    assert payload["routes"][0]["metrics"]["route_outcome"] == "verification_rejected"
    summary = payload["route_set_metrics"]["route_materialization_admission"]
    assert summary["rejected_route_count"] == 1


def test_explicit_raw_solved_diagnostic_cannot_override_rejected_host_verifier() -> None:
    resolution = resolve_chemenzy_budget(
        target_smiles=BAD_PRODUCT,
        action_kind="child",
        payload={},
        policy={},
        authority="host_profile",
        attempt_index=1,
    )
    outcome = classify_chemenzy_attempt_outcome(
        resolution,
        {
            "raw_solved": True,
            "verified_solved": False,
            "search_status": {"status": "partial", "solved": False},
            "routes": [{"raw_backend_metadata": {"route_outcome": "verification_rejected"}}],
        },
        verifier={
            "accepted": False,
            "route_status": "fake_closed_rejected",
            "reasons": ["element_inventory_not_conserved"],
            "accepted_route_count": 0,
        },
        verified_solved=False,
    )

    assert outcome["outcome"] == "verification_rejected"
    assert outcome["raw_solved"] is True
    assert outcome["verified_solved"] is False
    assert outcome["solved"] is False
    assert outcome["raw_search_status_is_authority"] is False


def test_post_search_materialization_keeps_admitted_reserve_and_bounded_advisory() -> None:
    valid = _raw_one_step_route(
        "CCO",
        ["CC=O"],
        template_id="valid-oxidation",
    )["all_succ_dict_routes"][0]
    invalid = _raw_one_step_route(
        BAD_PRODUCT,
        BAD_REACTANTS,
        template_id="invalid-inventory",
    )["all_succ_dict_routes"][0]
    dict_routes = [invalid, valid, invalid, valid, valid, valid]
    route_objects = [f"route-{index}" for index in range(len(dict_routes))]
    raw = {
        "all_succ_dict_routes": dict_routes,
        "all_succ_routes": route_objects,
        "dict_routes": dict_routes[0],
        "routes": route_objects[0],
        "iter": 200,
    }
    config = RouteSearchConfig(
        target_smiles="CCO",
        search_flags={
            "max_materialized_routes": 2,
            "max_advisory_materialized_routes": 1,
        },
    )

    bounded, metadata = _bounded_materialization_result(raw, config=config)

    assert metadata["search_budget_unchanged"] is True
    assert metadata["raw_route_count"] == 6
    assert metadata["preaudit_scanned_count"] == 4
    assert metadata["selected_host_admitted_count"] == 2
    assert metadata["selected_advisory_count"] == 1
    assert metadata["selected_raw_route_indices"] == [1, 3, 0]
    assert metadata["truncated_route_count"] == 3
    assert bounded["all_succ_routes"] == ["route-1", "route-3", "route-0"]


def test_web_export_honors_output_route_limit_after_bounded_search() -> None:
    routes = [
        route_candidates_from_chem_enzy_result(
            _raw_one_step_route(product, [reactant], template_id=f"valid-{index}"),
            target_smiles=product,
        )[0]
        for index, (product, reactant) in enumerate(
            [("CCO", "CC=O"), ("CCCO", "CCC=O"), ("CCCCO", "CCCC=O")]
        )
    ]
    result = BaselineRunResult(
        target_smiles="CCO",
        backend="ChemEnzyRetroPlanner",
        routes=routes,
        raw_backend_metadata={
            "route_materialization_selection": {"raw_route_count": 1363}
        },
    )

    payload = _web_payload_from_result(
        result,
        {"max_routes": 2},
        RouteSearchConfig(
            target_smiles="CCO",
            search_flags={"max_output_routes": 2},
        ),
        0.1,
    )

    assert payload["n_results"] == 2
    assert len(payload["routes"]) == 2
    limit = payload["route_set_metrics"]["output_limit"]
    assert limit["enabled"] is True
    assert limit["eligible_before_limit"] == 3
    assert limit["eligible_truncated"] == 1
    assert payload["search_status"]["native_raw_n_routes"] == 3
    assert payload["search_status"]["native_search_found_n_routes"] == 1363


def test_launcher_derives_bounded_annotation_pool_from_host_route_limit() -> None:
    with patch(
        "scripts.run_chem_enzy_plan_for_web.missing_template_relevance_models",
        return_value=[],
    ):
        config = _route_config_from_payload(
            {
                "target_smiles": "CCO",
                "search_preset": "thorough",
                "max_routes": 4,
                "max_steps": 20,
                "chem_enzy_iterations": 200,
                "chem_enzy_expansion_topk": 120,
                "chemenzy_seed": 41,
                "one_step_models": ["fixture"],
                "stock_names": ["RetroStar-stock"],
                "stock_paths": {
                    "RetroStar-stock": "data/retrostar-origin.csv",
                },
            },
            -1,
        )

    assert config.max_iterations == 200
    assert config.max_depth == 20
    assert config.expansion_topk == 120
    assert config.search_flags["max_output_routes"] == 4
    assert config.search_flags["max_materialized_routes"] == 32
    assert config.search_flags["max_advisory_materialized_routes"] == 4
    assert config.stock_names == ["RetroStar-stock"]
    assert config.random_seed == 41
    assert config.search_flags["stock_paths"]["RetroStar-stock"].endswith(
        "data\\retrostar-origin.csv"
    )


def test_vendor_config_treats_200_as_cap_and_uses_success_reserve() -> None:
    adapter = ChemEnzyBackendAdapter()
    config = adapter._vendor_config(
        RouteSearchConfig(
            target_smiles="CCO",
            max_iterations=200,
            stock_names=["RetroStar-stock"],
            search_flags={
                "max_materialized_routes": 32,
                "stock_paths": {"RetroStar-stock": "D:/bench/origin_dict.csv"},
            },
        )
    )

    assert config["iterations"] == 200
    assert config["keep_search"] is True
    assert config["max_success_routes"] == 32
    assert config["stocks"]["RetroStar-stock"] == "D:/bench/origin_dict.csv"


def test_vendor_mcts_stops_before_iteration_cap_when_reserve_is_ready() -> None:
    vendor_root = Path(__file__).resolve().parents[1] / "vendor" / "ChemEnzyRetroPlanner"
    sys.path.insert(0, str(vendor_root))
    try:
        module = importlib.import_module(
            "retro_planner.search_frame.mcts_star.molmcts_star"
        )
        _install_bounded_vendor_mcts(max_success_routes=1)
        assert module.mol_planner.func is bounded_mol_planner
        assert module.mol_planner.keywords["max_success_routes"] == 1
    finally:
        sys.path.remove(str(vendor_root))

    class Node:
        def __init__(self, *, molecule: bool, succ: bool = False) -> None:
            if molecule:
                self.mol = "molecule"
            self.succ = succ
            self.open = True
            self.children = []
            self.depth = 0
            self.go_back = False
            self.succ_value = 1.0

        def v_target(self) -> float:
            return 0.0

        def is_terminal(self) -> bool:
            return True

    class FakeRoute:
        optimal = False

    class FakeMolTree:
        def __init__(self, **_kwargs: object) -> None:
            self.root = Node(molecule=True)
            self.mol_nodes = [self.root]
            self.succ = False
            self.search_status = 0.0
            self.cascade_expansion_trace = []

        def call_expand_fn(self, _expand_fn: object, _node: Node) -> dict:
            return {"scores": [1.0]}

        def cascade_context_for_mol(self, _node: Node) -> dict:
            return {}

        def prepare_expansion(self, *_args: object, **_kwargs: object) -> tuple:
            return [], [], [], []

        def expand(self, node: Node, *_args: object, **_kwargs: object) -> tuple:
            leaf = Node(molecule=True, succ=True)
            reaction = Node(molecule=False, succ=True)
            reaction.children = [leaf]
            node.children = [reaction]
            node.succ = True
            node.succ_value = 0.0
            self.succ = True
            return True, None

        def get_best_route(self) -> FakeRoute:
            return FakeRoute()

        def extract_all_succ_routes(self) -> list[FakeRoute]:
            return [FakeRoute()]

    mol_tree_module = importlib.import_module(
        "retro_planner.search_frame.mcts_star.mol_tree"
    )
    with patch.object(mol_tree_module, "MolTree", FakeMolTree):
        solved, result = bounded_mol_planner(
            "target",
            0,
            set(),
            lambda _target: {},
            iterations=200,
            keep_search=True,
            max_success_routes=1,
        )

    stop = result[5]
    assert solved is True
    assert result[1] == 1
    assert stop["reason"] == "success_route_limit_reached"
    assert stop["configured_iteration_limit"] == 200
    assert stop["executed_iterations"] == 1
    assert stop["stopped_early"] is True


def test_vendor_mcts_records_raw_expansions_without_cascade_cost_model() -> None:
    vendor_root = Path(__file__).resolve().parents[1] / "vendor" / "ChemEnzyRetroPlanner"
    sys.path.insert(0, str(vendor_root))
    try:
        mol_tree_module = importlib.import_module(
            "retro_planner.search_frame.mcts_star.mol_tree"
        )
    finally:
        sys.path.remove(str(vendor_root))

    class Node:
        mol = "CCO"
        open = True
        depth = 0
        go_back = False
        succ = False
        succ_value = 1.0

        def v_target(self) -> float:
            return 0.0

        def is_terminal(self) -> bool:
            return True

    class FakeMolTree:
        def __init__(self, **_kwargs: object) -> None:
            self.root = Node()
            self.mol_nodes = [self.root]
            self.succ = False
            self.search_status = 0.0
            self.cascade_cost_model = None
            self.cascade_expansion_trace = []

        def call_expand_fn(self, _expand_fn: object, _node: Node) -> dict:
            return {
                "scores": [0.8],
                "reactants": ["CC=O"],
                "templates": [{"model_full_name": "fixture-model"}],
            }

        def cascade_context_for_mol(self, _node: Node) -> dict:
            return {}

        def prepare_expansion(self, *_args: object, **_kwargs: object) -> tuple:
            return [["CC=O"]], [0.2], [{"model_full_name": "fixture-model"}], [None]

        def expand(self, *_args: object, **_kwargs: object) -> tuple:
            return False, False

    with patch.object(mol_tree_module, "MolTree", FakeMolTree):
        _solved, result = bounded_mol_planner(
            "CCO",
            0,
            set(),
            lambda _target: {},
            iterations=1,
        )

    trace = result[4]
    assert len(trace) == 1
    assert trace[0]["parent_mol"] == "CCO"
    assert trace[0]["reactants"] == ["CC=O"]
    assert trace[0]["base_score"] == 0.8
    assert trace[0]["source_model"] == "fixture-model"
