from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from cascade_planner.cascade_search.subgoal_evidence_contract import (
    candidate_row,
    evidence_items,
    load_program_splits,
    molecule_fingerprint,
    molecule_properties,
)


def test_subgoal_evidence_contract_builds_serializable_model_rows(
    tmp_path: Path,
) -> None:
    program = {
        "program_id": "program-1",
        "doi": "fixture-doi",
        "cascade_id": "cascade-1",
        "cascade_type": "chemoenzymatic",
        "quality_tier": "gold",
        "target_smiles": "CCOC(=O)c1ccccc1",
        "compatibility": {
            "evidence_strength": "process_evidence",
            "compatibility_label": "compatible",
        },
        "steps": [
            {
                "transition_id": "step-1",
                "transformation_superclass": "acyl transfer",
                "product_smiles": "CCOC(=O)c1ccccc1",
                "reactants": ["CCOC(=O)O", "c1ccccc1"],
            }
        ],
    }
    outputs = {}
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}.jsonl"
        path.write_text(json.dumps(program) + "\n", encoding="utf-8")
        outputs[split] = str(path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"outputs": outputs}), encoding="utf-8")

    splits = load_program_splits(manifest)
    evidence = evidence_items(splits["train"], min_heavy_atoms=2)
    query_props = molecule_properties(program["target_smiles"])
    query = {
        "item_id": "query-1",
        "program_id": "query-program",
        "role": "program_target",
        "smiles": program["target_smiles"],
        "transform": "",
        "cascade_type": "chemoenzymatic",
        "route_transforms": ["acyl_transfer"],
        **query_props,
    }
    row = candidate_row(
        query,
        evidence[0],
        similarity=0.8,
        candidate_rank=1,
        schema={"evidence_transforms": ["acyl_transfer", "reduction"]},
        positive_similarity=0.55,
        strong_positive_similarity=0.72,
    )

    assert set(splits) == {"train", "val", "test"}
    assert evidence
    assert molecule_fingerprint(program["target_smiles"]) is not None
    assert row["training_relevance"] in {1, 2}
    assert len(row["features"]) == 31


def test_runtime_subgoal_helpers_do_not_load_archived_trainers() -> None:
    script = """
import json
import sys
from types import SimpleNamespace

from cascade_planner.cascade_search.proposals import (
    _runtime_subgoal_candidates,
    _runtime_subgoal_query,
)

state = SimpleNamespace(
    target_smiles="CCOC(=O)c1ccccc1",
    step_annotations=[],
    raw_metadata={},
)
_runtime_subgoal_candidates(
    state.target_smiles,
    state,
    max_subgoals=4,
    min_heavy_atoms=3,
)
_runtime_subgoal_query(state.target_smiles, state)
print(json.dumps(sorted(
    name for name in sys.modules
    if name in {
        "cascade_planner.eval.audit_cascade_subgoal_discovery",
        "cascade_planner.eval.train_cascade_subgoal_scorer",
        "cascade_planner.legacy.eval_runtime.audit_cascade_subgoal_discovery",
        "cascade_planner.legacy.eval_runtime.train_cascade_subgoal_scorer",
    }
)))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []
