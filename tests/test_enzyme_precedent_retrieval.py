import json

import pyarrow as pa
import pyarrow.parquet as pq

from cascade_planner.cascadeboard.enzyme_precedent_retrieval import retrieve_enzyme_precedents, transition_signature
from cascade_planner.route_tree.proposals import ProposalContext, RetroEngineProposalTool


def test_enzyme_precedent_retrieval_returns_ec_filtered_precedents(tmp_path):
    pool = tmp_path / "enzyme_reaction_pool.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                _rxn("r1", "CCO.O=O", "CC=O", ["1.1.1.1"], 5),
                _rxn("r2", "CCN", "CC=O", ["2.7.1.1"], 2),
            ]
        ),
        pool,
    )

    rows = retrieve_enzyme_precedents("CC=O", ec_class="1", top_k=3, pool_path=pool)

    assert len(rows) == 1
    assert rows[0]["source"] == "enzyme_precedent"
    assert rows[0]["ec"] == "1.1.1.1"
    assert rows[0]["main_reactant"] == "CCO"
    assert rows[0]["aux_reactants"] == ["O=O"]


def test_route_tree_proposal_tool_can_query_virtual_enzyme_precedent_source(tmp_path, monkeypatch):
    pool = tmp_path / "enzyme_reaction_pool.parquet"
    pq.write_table(
        pa.Table.from_pylist([_rxn("r1", "CCO", "CC=O", ["1.1.1.1"], 5)]),
        pool,
    )
    monkeypatch.setenv("AUTOPLANNER_ROUTE_TREE_ENZYME_PRECEDENT_RETRIEVAL", "1")
    monkeypatch.setenv("AUTOPLANNER_ENZYME_PRECEDENT_MAX_ROWS", "1")
    monkeypatch.setenv("AUTOPLANNER_ENZYME_PRECEDENT_POOL_PATH", str(pool))

    tool = RetroEngineProposalTool({})
    actions = tool._propose_from_sources("CC=O", ProposalContext(ec1=1), ["enzyme_precedent"], top_k=2)

    assert len(actions) == 1
    assert actions[0].source == "enzyme_precedent"
    assert actions[0].ec == "1.1.1.1"


def test_enzyme_precedent_main_reactant_avoids_blacklisted_large_component(tmp_path):
    pool = tmp_path / "enzyme_reaction_pool.parquet"
    large_carrier = "CCCCCCCCCCCC"
    pq.write_table(
        pa.Table.from_pylist(
            [
                _rxn("r1", f"CCO.{large_carrier}", "CC=O", ["1.1.1.1"], 5),
            ]
        ),
        pool,
    )
    _write_blacklist(tmp_path, [large_carrier])

    rows = retrieve_enzyme_precedents("CC=O", top_k=1, pool_path=pool)

    assert len(rows) == 1
    assert rows[0]["main_reactant"] == "CCO"
    assert rows[0]["aux_reactants"] == [large_carrier]
    selection = rows[0]["evidence"]["substrate_component_selection"]
    assert selection["selection_rule"] == "prefer_noncarrier_nonblacklisted_component"
    assert any(item["blacklisted"] for item in selection["annotations"])


def test_enzyme_precedent_similarity_uses_noncarrier_product_component(tmp_path):
    pool = tmp_path / "enzyme_reaction_pool.parquet"
    phosphate = "O=P(O)(O)O"
    pq.write_table(
        pa.Table.from_pylist(
            [
                _rxn("r1", "CCO", f"CC=O.{phosphate}", ["1.1.1.1"], 5),
                _rxn("r2", "CCCC", "CCCC", ["1.1.1.1"], 50),
            ]
        ),
        pool,
    )
    _write_blacklist(tmp_path, [phosphate])

    rows = retrieve_enzyme_precedents("CC=O", top_k=1, min_similarity=0.99, pool_path=pool)

    assert len(rows) == 1
    assert rows[0]["precedent_reaction_id"] == "r1"
    assert 0.0 < rows[0]["score"] <= 1.0
    assert rows[0]["evidence"]["precedent_product_main_smiles"] == "CC=O"
    assert rows[0]["evidence"]["product_similarity"] == 1.0
    assert rows[0]["evidence"]["product_full_similarity"] < 1.0
    assert rows[0]["evidence"]["retrieval_rank_score"] >= rows[0]["score"]
    assert rows[0]["evidence"]["transition_signature"]["valid"] is True


def test_transition_signature_records_auxiliary_explained_element_gain():
    payload = transition_signature("CC=CC", "CC(O)CC", substrate_aux=["O=O"])

    assert payload["valid"] is True
    assert payload["element_delta"]["O"] == 1
    assert payload["explained_element_gains"]["O"] == 1
    assert "auxiliary_explains_element_gain" in payload["transition_flags"]
    assert "unexplained_element_gain_review" not in payload["transition_flags"]
    assert 0.0 < payload["transition_quality_score"] <= 1.0


def test_transition_signature_marks_self_loop_as_review_signal():
    payload = transition_signature("CCO", "CCO")

    assert "main_transition_self_loop" in payload["transition_flags"]
    assert payload["heavy_atom_delta"] == 0
    assert payload["substrate_product_similarity"] == 1.0


def _rxn(reaction_id, substrate, product, ecs, occurrences):
    return {
        "reaction_id": reaction_id,
        "substrate_smiles": substrate,
        "product_smiles": product,
        "reaction_smiles": f"{substrate}>>{product}",
        "occurrences": occurrences,
        "source_counts_json": json.dumps({"fixture": occurrences}),
        "ec_numbers_json": json.dumps(ecs),
        "rhea_ids_json": "[]",
        "example_ids_json": json.dumps([f"{reaction_id}:example"]),
    }


def _write_blacklist(tmp_path, smiles_list):
    rows = []
    for smi in smiles_list:
        rows.append(
            {
                "canonical_smiles": smi,
                "inchikey": "",
                "formula": "",
                "heavy_atoms": 0,
                "enzyme_occurrences": 1,
                "reasons_json": "[]",
                "sources_json": "[]",
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "cofactor_common_metabolite_blacklist.parquet")
