from __future__ import annotations

from cascade_planner.legacy.routes_runtime.signatures import exact_edge_signature


def test_exact_edge_signature_is_order_independent_and_component_safe() -> None:
    first = exact_edge_signature("CCO", ["Cl", "CC.N"])
    second = exact_edge_signature("OCC", ["N.CC", "Cl"])

    assert first == second
    assert first.startswith("edge:sha256:")
    assert exact_edge_signature("not-smiles", ["CC"]) == ""
