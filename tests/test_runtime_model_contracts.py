from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from cascade_planner.agent.failure_model_contract import (
    FailureClassifier,
    failure_row_features,
)
from cascade_planner.cascadeboard.skeleton_reranker_contract import (
    SkeletonReranker,
    skeleton_row_features,
)
from cascade_planner.eval.train_failure_classifier_from_pack import (
    FailureClassifier as TrainingFailureClassifier,
)
from cascade_planner.eval.train_failure_classifier_from_pack import (
    row_features as training_failure_row_features,
)
from cascade_planner.eval.train_skeleton_reranker import (
    SkeletonReranker as TrainingSkeletonReranker,
)
from cascade_planner.eval.train_skeleton_reranker import (
    row_features as training_skeleton_row_features,
)
from cascade_planner.route_tree.native_route_selection import (
    chem_route_stock_closed,
    select_chem_routes,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PACKAGE_DIRS = (
    "agent",
    "application",
    "baselines",
    "cascadeboard",
    "cascade_search",
    "interfaces",
    "orchestration",
    "providers",
    "routes",
    "route_tree",
    "runtime",
    "web",
)


def test_failure_model_runtime_contract_matches_trainer_checkpoint() -> None:
    row = {
        "target_smiles": "CCO",
        "route_domain": "chemoenzymatic",
        "depth": 3,
        "n_routes": 2,
        "has_failure_label": True,
        "metrics": {
            "plan": True,
            "strict_stock_solve_any": False,
        },
    }
    training_features = training_failure_row_features(row, n_bits=32)
    runtime_features = failure_row_features(row, n_bits=32)
    training_model = TrainingFailureClassifier(len(training_features), 3, hidden=16)
    runtime_model = FailureClassifier(len(runtime_features), 3, hidden=16)

    assert np.array_equal(training_features, runtime_features)
    assert list(training_model.state_dict()) == list(runtime_model.state_dict())
    runtime_model.load_state_dict(training_model.state_dict())


def test_skeleton_reranker_runtime_contract_matches_trainer_checkpoint() -> None:
    schema = {
        "n_bits": 32,
        "max_steps": 4,
        "type_vocab": ["CHEMICAL", "ENZYMATIC"],
        "ec1_vocab": ["NONE", "1"],
    }
    row = {
        "target_smiles": "CCO",
        "depth": 2,
        "type_sequence": ["CHEMICAL", "ENZYMATIC"],
        "ec1_sequence": ["NONE", "1"],
    }
    training_features = training_skeleton_row_features(row, schema)
    runtime_features = skeleton_row_features(row, schema)
    training_model = TrainingSkeletonReranker(len(training_features), hidden=16)
    runtime_model = SkeletonReranker(len(runtime_features), hidden=16)

    assert np.array_equal(training_features, runtime_features)
    assert list(training_model.state_dict()) == list(runtime_model.state_dict())
    runtime_model.load_state_dict(training_model.state_dict())


def test_native_route_selection_contract_matches_evaluator() -> None:
    routes = [
        {
            "steps": [
                {
                    "product_smiles": "CCO",
                    "reactant_smiles": ["CC", "O"],
                    "stock_status": {"CC": False, "O": True},
                }
            ]
        },
        {
            "steps": [
                {
                    "product_smiles": "CCO",
                    "reactant_smiles": ["CC", "O"],
                    "stock_status": {"CC": True, "O": True},
                }
            ]
        },
    ]

    assert chem_route_stock_closed(routes[0]) is False
    assert chem_route_stock_closed(routes[1]) is True
    assert [
        route["_native_rank"]
        for route in select_chem_routes(routes, topk=1, selection="rank")
    ] == [1]
    assert [
        route["_native_rank"]
        for route in select_chem_routes(routes, topk=1, selection="stock_first")
    ] == [2]
    assert [
        route["_native_rank"]
        for route in select_chem_routes(
            routes,
            topk=1,
            selection="rank_plus_stock",
        )
    ] == [2]


def test_canonical_runtime_packages_do_not_import_eval_modules() -> None:
    violations: dict[str, list[str]] = {}
    for package in CANONICAL_PACKAGE_DIRS:
        for path in (ROOT / "cascade_planner" / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            }
            imports.update(
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            )
            eval_imports = sorted(
                imported
                for imported in imports
                if imported == "cascade_planner.eval"
                or imported.startswith("cascade_planner.eval.")
            )
            if eval_imports:
                violations[path.relative_to(ROOT).as_posix()] = eval_imports

    assert violations == {}
