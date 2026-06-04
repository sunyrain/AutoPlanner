from cascade_planner.baselines.chem_enzy_step_quality import evaluate_enzyme_step_quality


def test_enzyme_step_quality_passes_supported_material_sane_candidate():
    quality = evaluate_enzyme_step_quality(
        product_smiles="CCO",
        reactants=["CC=O"],
        source_model="autoplanner.enzyme_precedent",
        template={
            "ec": "1.1.1.1",
            "evidence": {"transition_signature": "alcohol_dehydrogenase", "source_db": "brenda"},
        },
        sp_payload={"score": 0.9, "threshold": 0.3, "accepted": True, "ec_numbers": ["1.1.1.1"]},
    )

    assert quality["decision"] == "pass"
    assert quality["quality_score"] > 0.85
    assert quality["material_sanity"]["passed"] is True
    assert quality["bridge_or_precedent_evidence"] is True


def test_enzyme_step_quality_rejects_large_material_jump():
    quality = evaluate_enzyme_step_quality(
        product_smiles="CCCCCCCCCCCCCCCCCC",
        reactants=["CCC"],
        source_model="autoplanner.enzyme_precedent",
        template={
            "ec": "2.5.1.1",
            "evidence": {"transition_signature": "prenyl_transfer"},
        },
        sp_payload={"score": 0.9, "threshold": 0.3, "accepted": True, "ec_numbers": ["2.5.1.1"]},
    )

    assert quality["decision"] == "reject"
    assert quality["quality_score"] <= 0.4
    assert "material_sanity_failed" in quality["flags"]


def test_enzyme_step_quality_warns_native_label_without_sp_or_precedent():
    quality = evaluate_enzyme_step_quality(
        product_smiles="CCO",
        reactants=["CC=O"],
        source_model="onmt_models.bionav_one_step",
        ec_numbers=["1.1.1.1"],
    )

    assert quality["decision"] == "warn"
    assert quality["quality_score"] == 0.6
    assert "missing_sp_v1" in quality["flags"]
    assert "missing_bridge_or_precedent_evidence" in quality["flags"]
