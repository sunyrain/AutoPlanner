from __future__ import annotations

import json
from typing import Any, Mapping

from cascade_planner.interfaces.target_identity import resolve_target_identity
from cascade_planner.interfaces.target_solver import (
    _target_name_requires_identity_resolution,
)


TARGET = "CCOC(C)=O"


def test_opaque_hash_label_still_triggers_structure_identity_lookup() -> None:
    assert _target_name_requires_identity_resolution("target-ae21163b")
    assert _target_name_requires_identity_resolution("blind target")
    assert not _target_name_requires_identity_resolution("public compound name")


def _response(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value).encode("utf-8")


def test_smiles_only_identity_requires_exact_inchikey_and_returns_search_hints() -> None:
    calls: list[tuple[str, str]] = []

    def requester(
        method: str,
        url: str,
        *,
        data: Mapping[str, Any] | None = None,
        timeout_s: float,
        max_bytes: int,
    ) -> tuple[int, bytes]:
        calls.append((method, url))
        if "/property/" in url:
            assert data and data["smiles"] == "CCOC(C)=O"
            return 200, _response(
                {
                    "PropertyTable": {
                        "Properties": [
                            {
                                "CID": 8857,
                                "Title": "Ethyl acetate",
                                "IUPACName": "ethyl acetate",
                                "SMILES": TARGET,
                                "ConnectivitySMILES": TARGET,
                                "InChIKey": "XEKOWRVHYACXOJ-UHFFFAOYSA-N",
                                "MolecularFormula": "C4H8O2",
                            }
                        ]
                    }
                }
            )
        if "/synonyms/" in url:
            return 200, _response(
                {
                    "InformationList": {
                        "Information": [{"CID": 8857, "Synonym": ["Ethyl acetate", "EtOAc"]}]
                    }
                }
            )
        return 200, _response(
            {
                "InformationList": {
                    "Information": [
                        {
                            "CID": 8857,
                            "PubMedID": [123],
                            "PatentID": [
                                "WO-4-A",
                                "CN-3-A",
                                "US-1-A",
                                "EP-5-A",
                                "WO-2-A",
                            ],
                        }
                    ]
                }
            }
        )

    result = resolve_target_identity(TARGET, requester=requester)

    assert result["status"] == "completed"
    assert result["identity"]["preferred_name"] == "Ethyl acetate"
    assert result["identity"]["synonyms"] == ["Ethyl acetate", "EtOAc"]
    assert result["identity"]["pubmed_ids"] == ["123"]
    assert result["identity"]["patent_ids"] == [
        "WO-2-A",
        "US-1-A",
        "EP-5-A",
        "CN-3-A",
        "WO-4-A",
    ]
    assert result["semantics"]["no_local_dossier_or_pdf_used"] is True
    assert [method for method, _url in calls] == ["POST", "GET", "GET"]


def test_pubchem_name_is_rejected_when_structure_identity_does_not_match() -> None:
    result = resolve_target_identity(
        TARGET,
        requester=lambda *_args, **_kwargs: (
            200,
            _response(
                {
                    "PropertyTable": {
                        "Properties": [
                            {
                                "CID": 1,
                                "Title": "wrong compound",
                                "InChIKey": "WRONG-INCHIKEY",
                            }
                        ]
                    }
                }
            ),
        ),
    )

    assert result["status"] == "unresolved"
    assert result["reason"] == "pubchem_exact_inchikey_match_missing"


def test_named_lookup_requires_exact_inchikey_before_completing_identity() -> None:
    calls: list[tuple[str, str]] = []

    def requester(
        method: str,
        url: str,
        **_kwargs: Any,
    ) -> tuple[int, bytes]:
        calls.append((method, url))
        if "/compound/smiles/property/" in url:
            return 200, _response(
                {"PropertyTable": {"Properties": [{"CID": 0}]}}
            )
        if "/compound/name/Ethyl%20acetate/property/" in url:
            return 200, _response(
                {
                    "PropertyTable": {
                        "Properties": [
                            {
                                "CID": 8857,
                                "Title": "Ethyl acetate",
                                "InChIKey": "XEKOWRVHYACXOJ-UHFFFAOYSA-N",
                            }
                        ]
                    }
                }
            )
        if "/synonyms/" in url:
            return 200, _response({"InformationList": {"Information": []}})
        return 200, _response(
            {
                "InformationList": {
                    "Information": [{"CID": 8857, "PatentID": ["EP-5-A1"]}]
                }
            }
        )

    result = resolve_target_identity(
        TARGET,
        target_name="Ethyl acetate",
        requester=requester,
    )

    assert result["status"] == "completed"
    assert result["identity"]["patent_ids"] == ["EP-5-A1"]
    assert result["semantics"]["lookup_strategy"] == (
        "name_verified_by_exact_inchikey"
    )
    assert calls[0][0] == "POST"
    assert calls[1][0] == "GET"


def test_named_structure_mismatch_keeps_patents_as_discovery_hints_only() -> None:
    def requester(
        _method: str,
        url: str,
        **_kwargs: Any,
    ) -> tuple[int, bytes]:
        if "/compound/smiles/property/" in url:
            return 200, _response(
                {"PropertyTable": {"Properties": [{"CID": 0}]}}
            )
        if "/compound/name/" in url:
            return 200, _response(
                {
                    "PropertyTable": {
                        "Properties": [
                            {
                                "CID": 131723104,
                                "Title": "named record",
                                "InChIKey": "A-DIFFERENT-INCHIKEY",
                            }
                        ]
                    }
                }
            )
        if "/synonyms/" in url:
            return 200, _response({"InformationList": {"Information": []}})
        return 200, _response(
            {
                "InformationList": {
                    "Information": [
                        {
                            "CID": 131723104,
                            "PatentID": ["EP-0955305-A1", "US-6342568-B1"],
                        }
                    ]
                }
            }
        )

    result = resolve_target_identity(
        TARGET,
        target_name="named record",
        requester=requester,
    )

    assert result["status"] == "unresolved"
    assert result["reason"] == "pubchem_named_inchikey_mismatch"
    assert result["identity"]["patent_ids"] == []
    assert result["identity"]["name_linked_patent_ids"] == [
        "US-6342568-B1",
        "EP-0955305-A1",
    ]
    assert result["semantics"]["host_structure_validation_required"] is True


def test_smaller_pubchem_patent_view_recovers_when_bulk_xrefs_timeout() -> None:
    def requester(
        _method: str,
        url: str,
        **_kwargs: Any,
    ) -> tuple[int, bytes]:
        if "/property/" in url:
            return 200, _response(
                {
                    "PropertyTable": {
                        "Properties": [
                            {
                                "CID": 8857,
                                "Title": "Ethyl acetate",
                                "InChIKey": "XEKOWRVHYACXOJ-UHFFFAOYSA-N",
                            }
                        ]
                    }
                }
            )
        if "/synonyms/" in url:
            return 200, _response({"InformationList": {"Information": []}})
        if "/xrefs/" in url:
            raise RuntimeError("bulk endpoint timed out")
        assert "/rest/pug_view/data/compound/8857/JSON?heading=Patents" in url
        return 200, _response(
            {
                "Record": {
                    "Section": [
                        {
                            "TOCHeading": "Patents",
                            "Information": [
                                {
                                    "Value": {
                                        "StringWithMarkup": [
                                            {"String": "US 4681893"},
                                            {"String": "WO-2024000123-A1"},
                                            {"String": "not a patent"},
                                        ]
                                    }
                                }
                            ],
                        }
                    ]
                }
            }
        )

    result = resolve_target_identity(TARGET, requester=requester)

    assert result["identity"]["patent_ids"] == ["WO2024000123A1", "US4681893"]
    assert result["response_receipts"]["patent_view_sha256"]
