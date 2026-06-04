from cascade_planner.baselines.chemical_anchor_rescue import (
    AMINOACETONE,
    BENZOTHIAZINE_CHLORO_CORE,
    BENZOTHIAZINE_TARGET,
    BENZOTHIAZOLE_DIOL_CORE,
    BENZOTHIAZOLE_TARGET,
    BENZYL_BROMIDE,
    chemical_anchor_rescue_routes,
    known_chemical_anchor_precursor_record,
)


def test_benzothiazine_anchor_is_source_supported():
    routes = chemical_anchor_rescue_routes(BENZOTHIAZINE_TARGET)

    assert len(routes) == 1
    route = routes[0]
    step = route.steps[0]

    assert route.solved is True
    assert step.source_model == "chemical_anchor_rescue.benzothiazine_c2_amination"
    assert step.stock_status[BENZOTHIAZINE_CHLORO_CORE] is True
    assert step.stock_status[AMINOACETONE] is True
    assert step.raw_backend_metadata["chemical_anchor_rescue"]["precursor_source_supported"] is True
    assert (
        step.raw_backend_metadata["chemical_anchor_rescue"]["precursor_source_record"]["pubchem_cid"]
        == "20143033"
    )


def test_benzothiazole_anchor_is_source_supported():
    routes = chemical_anchor_rescue_routes(BENZOTHIAZOLE_TARGET)

    assert len(routes) == 1
    route = routes[0]
    step = route.steps[0]

    assert route.solved is True
    assert step.source_model == "chemical_anchor_rescue.benzothiazole_dibenzylation"
    assert step.stock_status[BENZOTHIAZOLE_DIOL_CORE] is True
    assert step.stock_status[BENZYL_BROMIDE] is True
    assert step.raw_backend_metadata["chemical_anchor_rescue"]["precursor_source_supported"] is True
    assert (
        step.raw_backend_metadata["chemical_anchor_rescue"]["precursor_source_record"]["pubchem_cid"]
        == "67480579"
    )


def test_chemical_anchor_routes_are_product_filtered():
    assert chemical_anchor_rescue_routes("CCO") == []


def test_known_chemical_anchor_precursor_record_matches_canonical_variants():
    record = known_chemical_anchor_precursor_record("CN1C2=CC=CC=C2SC(C1=O)(C3=CC=C(C=C3)O)Cl")

    assert record["pubchem_cid"] == "20143033"
    assert known_chemical_anchor_precursor_record("CCO") == {}
