from scripts.audit_enzyme_candidate_quality_v0 import quality_flags, quality_tier


def test_no_bridge_or_ec_trigger_is_review_not_search_ready_even_if_sp_high():
    flags = quality_flags(
        source="enzyme_precedent",
        sp_payload={"accepted": True, "score": 0.99},
        product_similarity=0.9,
        ecs=["1.1.1.1"],
        bridge_hit_count=0,
        bridge_ec1s=[],
        bridge_ec1_match=False,
        context_ec1=0,
        product_blacklisted=False,
        main_blacklisted=False,
        aux_blacklisted=0,
        product_atoms=20,
        main_atoms=19,
        substrate_total_atoms=19,
        component_count=1,
    )

    assert "no_bridge_or_ec_trigger_for_injection" in flags
    assert quality_tier(5.0, flags, source="enzyme_precedent") == "ungated_review"


def test_bridge_triggered_high_sp_enzyme_precedent_can_be_search_ready():
    flags = quality_flags(
        source="enzyme_precedent",
        sp_payload={"accepted": True, "score": 0.99},
        product_similarity=0.9,
        ecs=["1.1.1.1"],
        bridge_hit_count=1,
        bridge_ec1s=[1],
        bridge_ec1_match=True,
        context_ec1=0,
        product_blacklisted=False,
        main_blacklisted=False,
        aux_blacklisted=1,
        product_atoms=20,
        main_atoms=19,
        substrate_total_atoms=21,
        component_count=2,
    )

    assert "aux_common_or_cofactor" in flags
    assert "no_bridge_or_ec_trigger_for_injection" not in flags
    assert quality_tier(5.0, flags, source="enzyme_precedent") == "strong"
