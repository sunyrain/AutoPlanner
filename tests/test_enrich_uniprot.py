import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.cascadeboard.enz_retrieval import (
    _DB_CACHE,
    _cached_uniprot_evidence,
    _load_db,
)
from cascade_planner.data.enrich_uniprot import (
    _apply_uniprot_result,
    _records,
    parse_uniprot_entry,
)


class UniProtEnrichmentTest(unittest.TestCase):
    def test_records_accepts_dict_and_list_snapshots(self):
        rows = [{"record_uuid": "r1"}]
        self.assertEqual(_records({"records_kept": rows}), rows)
        self.assertEqual(_records({"records": rows}), rows)
        self.assertEqual(_records(rows), rows)
        self.assertEqual(_records({"records_kept": {}}), [])

    def test_parse_uniprot_entry_extracts_sequence_rhea_and_cofactor(self):
        entry = {
            "primaryAccession": "P12345",
            "uniProtkbId": "TEST_ECOLI",
            "entryType": "UniProtKB reviewed (Swiss-Prot)",
            "proteinExistence": "1: Evidence at protein level",
            "organism": {"scientificName": "Escherichia coli", "taxonId": 562},
            "sequence": {"value": "MTEST", "length": 5},
            "proteinDescription": {
                "recommendedName": {
                    "fullName": {"value": "Test enzyme"},
                    "ecNumbers": [{"value": "1.1.1.1"}],
                }
            },
            "comments": [{
                "commentType": "COFACTOR",
                "cofactors": [{"name": "NAD"}],
            }],
            "uniProtKBCrossReferences": [{"database": "Rhea", "id": "RHEA:12345"}],
        }

        parsed = parse_uniprot_entry(entry)

        self.assertEqual(parsed["accession"], "P12345")
        self.assertTrue(parsed["reviewed"])
        self.assertEqual(parsed["sequence"], "MTEST")
        self.assertEqual(parsed["sequence_length"], 5)
        self.assertEqual(parsed["rhea_ids"], ["RHEA:12345"])
        self.assertEqual(parsed["cofactor"], "NAD")

    def test_enz_retrieval_can_merge_cached_uniprot_evidence(self):
        cache = {
            "accession::P12345": {
                "accession": "P12345",
                "entry_name": "TEST_ECOLI",
                "reviewed": True,
                "organism": "Escherichia coli",
                "tax_id": 562,
                "sequence": "MTEST",
                "sequence_length": 5,
                "cofactor": "NAD",
                "rhea_ids": ["RHEA:12345"],
                "protein_existence": "1: Evidence at protein level",
            }
        }

        evidence = _cached_uniprot_evidence(
            {"uniprot_id": "P12345", "organism": "E. coli"},
            "1.1.1.1",
            cache=cache,
        )

        self.assertEqual(evidence["uniprot_accession"], "P12345")
        self.assertEqual(evidence["sequence"], "MTEST")
        self.assertEqual(evidence["rhea_ids"], ["RHEA:12345"])
        self.assertTrue(evidence["reviewed"])
        self.assertTrue(evidence["uniprot_cache_hit"])

    def test_enz_retrieval_load_db_accepts_records_snapshot(self):
        record = {
            "doi": "10.0000/example",
            "title": "Example cascade",
            "cascades": [{
                "cascade_id": "cas-1",
                "steps": [{
                    "step_id": "s1",
                    "rxn_smiles": "CCO>>CC=O",
                    "catalyst_components": [{
                        "ec_number": "1.1.1.1",
                        "component_name": "test enzyme",
                    }],
                }],
            }],
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshot.json"
            path.write_text(json.dumps({"records": [record]}), encoding="utf-8")
            _DB_CACHE.clear()

            db = _load_db(str(path))

        self.assertEqual(len(db), 1)
        self.assertEqual(db[0].ec, "1.1.1.1")
        self.assertEqual(db[0].evidence["doi"], "10.0000/example")

    def test_apply_uniprot_result_details_existing_accession(self):
        cat = {
            "uniprot_id": "P12345",
            "enzyme_external_ids": None,
        }
        res = {
            "accession": "P12345",
            "entry_name": "TEST_ECOLI",
            "reviewed": True,
            "organism": "Escherichia coli",
            "tax_id": 562,
            "sequence": "MTEST",
            "sequence_length": 5,
            "cofactor": "NAD",
            "rhea_ids": ["RHEA:12345"],
        }

        changed = _apply_uniprot_result(
            cat,
            res,
            match_quality="existing_accession",
        )

        self.assertTrue(changed)
        self.assertEqual(cat["uniprot_id"], "P12345")
        self.assertEqual(cat["uniprot_entry_name"], "TEST_ECOLI")
        self.assertEqual(cat["uniprot_status"], "reviewed")
        self.assertEqual(cat["enzyme_seq_length"], 5)
        self.assertEqual(cat["enzyme_external_ids"]["rhea"], ["RHEA:12345"])
        self.assertEqual(cat["uniprot_match_quality"], "existing_accession")


if __name__ == "__main__":
    unittest.main()
