from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from cascade_planner.interfaces.chemenzy_builtin_runtime import (
    _launcher_request,
    run_builtin_chemenzy_probe,
)
from cascade_planner.interfaces.chemenzy_probe_contract import (
    ChemEnzyProposalRequest,
)


def test_builtin_probe_replays_complete_same_request_without_subprocess(
    tmp_path: Path,
) -> None:
    proposal = ChemEnzyProposalRequest(
        target_name="target",
        target_smiles="CCO",
    )
    limits = {
        "max_routes": 2,
        "max_steps": 6,
        "max_iterations": 10,
        "expansion_topk": 20,
        "timeout_s": 90.0,
        "random_seed": 0,
        "pandarallel_workers": 2,
        "one_step_models": [],
    }
    request = _launcher_request(
        target_name="target",
        target_smiles="CCO",
        proposal_request=proposal,
        limits=limits,
    )
    request_path = tmp_path / "chemenzy-v4-guided-fixture-request.json"
    output_path = tmp_path / "chemenzy-v4-guided-fixture-result.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    output_path.write_text(
        json.dumps({"status": "completed", "routes": [{"route_id": "r1"}]}),
        encoding="utf-8",
    )
    preflight = {
        "production_ready": True,
        "python_executable": str(tmp_path / "python.exe"),
        "launcher_path": str(tmp_path / "launcher.py"),
        "vendor_root": str(tmp_path / "vendor"),
    }

    with (
        patch(
            "cascade_planner.interfaces.chemenzy_builtin_runtime."
            "select_chemenzy_runtime",
            return_value=(preflight, {"source": "test"}),
        ),
        patch(
            "cascade_planner.interfaces.chemenzy_builtin_runtime.subprocess.run"
        ) as run,
    ):
        result = run_builtin_chemenzy_probe(
            tmp_path,
            target_name="target",
            target_smiles="CCO",
            proposal_request=proposal,
            scope="guided-fixture",
            env_prefix=None,
            vendor_root=None,
            limits=limits,
        )

    run.assert_not_called()
    assert result["provider_result_replayed"] is True
    assert result["search_executed"] is True
    assert result["routes"] == [{"route_id": "r1"}]
