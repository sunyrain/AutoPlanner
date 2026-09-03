from __future__ import annotations

import json
from pathlib import Path

from cascade_planner.route_tree.enzymatic_retrieval import _load_db


def test_enzymatic_retrieval_loads_utf8_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "cascade_dataset_v3.json"
    dataset.write_text(
        json.dumps(
            {
                "records_kept": [
                    {
                        "doi": "10.0000/example",
                        "title": "\u9176\u50ac\u5316\u6c27\u5316",
                        "cascades": [
                            {
                                "cascade_id": "cascade-1",
                                "steps": [
                                    {
                                        "step_id": "step-1",
                                        "rxn_smiles": "CCO>>CC=O",
                                        "transformation_superclass": "oxidation",
                                        "catalyst_components": [
                                            {
                                                "ec_number": "1.1.1.1",
                                                "component_name": "alcohol dehydrogenase",
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rows = _load_db(str(dataset))

    assert len(rows) == 1
    assert rows[0].rxn_smiles == "CCO>>CC=O"
    assert rows[0].evidence["literature_title"] == "\u9176\u50ac\u5316\u6c27\u5316"
