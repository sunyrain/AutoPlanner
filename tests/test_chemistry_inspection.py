import json
import subprocess
import sys
from pathlib import Path

from cascade_planner.application.chemistry_inspection import (
    inspect_mapped_smiles,
)


def test_compact_chemistry_inspection_reports_only_requested_mapped_center() -> None:
    result = inspect_mapped_smiles(
        "[CH3:1][C@H:2]([OH:3])[C:4](=[O:5])[OH:6]",
        map_ids=[2],
    )

    assert result["ok"] is True
    assert len(result["centers"]) == 1
    assert result["centers"][0]["map_idx"] == 2
    assert result["centers"][0]["cip"] in {"R", "S"}
    assert result["unassigned_center_maps"] == []
    assert [atom["map_idx"] for atom in result["atoms"]] == [2]
    assert {
        frozenset((bond["map_a"], bond["map_b"])) for bond in result["bonds"]
    } == {frozenset((1, 2)), frozenset((2, 3)), frozenset((2, 4))}


def test_compact_chemistry_inspection_exposes_bounded_ring_topology() -> None:
    result = inspect_mapped_smiles(
        "[CH2:1]1[CH2:2][CH2:3][CH2:4][CH2:5][CH2:6]1",
        map_ids=[1],
    )

    assert result["ok"] is True
    assert result["atoms"] == [
        {
            "map_idx": 1,
            "element": "C",
            "formal_charge": 0,
            "total_h": 2,
            "degree": 2,
            "aromatic": False,
            "ring_sizes": [6],
        }
    ]
    assert len(result["rings"]) == 1
    assert set(result["rings"][0]) == {1, 2, 3, 4, 5, 6}
    assert all(bond["in_ring"] for bond in result["bonds"])


def test_compact_chemistry_inspection_bounds_unassigned_enumeration() -> None:
    result = inspect_mapped_smiles(
        "[CH3:1][CH:2]([OH:3])[C:4](=[O:5])[OH:6]",
        map_ids=[2],
        enumerate_unassigned=True,
        max_isomers=2,
    )

    assert result["ok"] is True
    assert result["unassigned_center_maps"] == [2]
    assert 1 <= len(result["limited_stereoisomers"]) <= 2


def test_compact_chemistry_inspection_rejects_invalid_smiles() -> None:
    assert inspect_mapped_smiles("not-smiles") == {
        "ok": False,
        "reason": "invalid_smiles",
    }


def test_chemistry_inspection_mcp_exposes_only_bounded_structure_tool() -> None:
    server = (
        Path(__file__).resolve().parents[1]
        / "cascade_planner"
        / "application"
        / "chemistry_inspection_mcp.py"
    )
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        },
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "inspect_mapped_smiles",
                "arguments": {
                    "smiles": "[CH3:1][C@H:2]([OH:3])[C:4](=[O:5])[OH:6]",
                    "map_ids": [2],
                },
            },
        },
    ]
    completed = subprocess.run(
        [sys.executable, str(server)],
        input="".join(json.dumps(value) + "\n" for value in requests),
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    )
    responses = [json.loads(line) for line in completed.stdout.splitlines()]

    assert responses[0]["result"]["protocolVersion"] == "2025-06-18"
    tools = responses[1]["result"]["tools"]
    assert [tool["name"] for tool in tools] == ["inspect_mapped_smiles"]
    assert set(tools[0]["inputSchema"]["properties"]) == {
        "smiles",
        "map_ids",
        "enumerate_unassigned",
        "max_isomers",
    }
    assert responses[2]["result"]["structuredContent"]["ok"] is True
    assert responses[2]["result"]["structuredContent"]["centers"][0]["map_idx"] == 2
