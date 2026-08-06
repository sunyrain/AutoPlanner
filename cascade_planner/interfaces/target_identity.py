"""Fresh, structure-bound target identity discovery for SMILES-only runs."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Callable, Mapping
from urllib.parse import quote

import requests
from rdkit import Chem
from rdkit.Chem import inchi


TARGET_IDENTITY_OBSERVATION_SCHEMA = "target_identity_observation.v1"
TARGET_IDENTITY_PROVIDER_VERSION = "2026-07.4"
JsonRequester = Callable[..., tuple[int, bytes]]


@dataclass(frozen=True, slots=True)
class PubChemIdentityConfig:
    timeout_s: float = 30.0
    max_response_bytes: int = 2_000_000
    max_synonyms: int = 24
    base_url: str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    def __post_init__(self) -> None:
        if self.timeout_s <= 0 or self.max_response_bytes < 1024:
            raise ValueError("target_identity_transport_limit_invalid")
        if not 1 <= self.max_synonyms <= 64:
            raise ValueError("target_identity_synonym_limit_invalid")


def resolve_target_identity(
    target_smiles: str,
    *,
    target_name: str = "",
    config: PubChemIdentityConfig | None = None,
    requester: JsonRequester | None = None,
) -> dict[str, Any]:
    """Resolve names only when PubChem returns the exact input InChIKey.

    A user-supplied exact name is also queried when PubChem's SMILES endpoint
    cannot index the submitted representation.  A mismatched named record is
    never accepted as target identity, but its patent links remain useful as
    explicitly non-authoritative discovery hints for later structure checks.
    """

    active = config or PubChemIdentityConfig()
    request = requester or _request_json
    molecule = Chem.MolFromSmiles(str(target_smiles or ""))
    if molecule is None:
        return _result("rejected", reason="target_smiles_invalid")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    local_inchikey = inchi.MolToInchiKey(molecule)
    properties_url = (
        active.base_url.rstrip("/")
        + "/compound/smiles/property/"
        + "Title,IUPACName,IsomericSMILES,ConnectivitySMILES,InChIKey,MolecularFormula/JSON"
    )
    try:
        status, property_bytes = request(
            "POST",
            properties_url,
            data={"smiles": canonical},
            timeout_s=active.timeout_s,
            max_bytes=active.max_response_bytes,
        )
        properties_payload = _json_object(property_bytes)
    except (OSError, RuntimeError, ValueError, requests.RequestException) as exc:
        return _result(
            "unresolved",
            reason=f"pubchem_identity_lookup_failed:{type(exc).__name__}:{str(exc)[:300]}",
            canonical_smiles=canonical,
            inchikey=local_inchikey,
        )
    rows = [
        dict(row)
        for row in dict(properties_payload.get("PropertyTable") or {}).get("Properties") or []
        if isinstance(row, Mapping)
    ]
    exact = next(
        (row for row in rows if str(row.get("InChIKey") or "") == local_inchikey),
        None,
    )
    lookup_strategy = "structure_exact_inchikey"
    named_mismatch: dict[str, Any] | None = None
    named_property_bytes = b""
    supplied_name = " ".join(str(target_name or "").split())
    if exact is None and supplied_name:
        named_url = (
            active.base_url.rstrip("/")
            + "/compound/name/"
            + quote(supplied_name, safe="")
            + "/property/"
            + "Title,IUPACName,IsomericSMILES,ConnectivitySMILES,InChIKey,MolecularFormula/JSON"
        )
        try:
            named_status, named_property_bytes = request(
                "GET",
                named_url,
                timeout_s=active.timeout_s,
                max_bytes=active.max_response_bytes,
            )
            named_payload = _json_object(named_property_bytes)
        except (OSError, RuntimeError, ValueError, requests.RequestException):
            named_status, named_payload = 0, {}
        named_rows = [
            dict(row)
            for row in dict(named_payload.get("PropertyTable") or {}).get("Properties") or []
            if isinstance(row, Mapping)
        ]
        exact = next(
            (
                row
                for row in named_rows
                if str(row.get("InChIKey") or "") == local_inchikey
            ),
            None,
        )
        if exact is not None and named_status == 200:
            property_bytes = named_property_bytes
            lookup_strategy = "name_verified_by_exact_inchikey"
        elif named_status == 200 and named_rows:
            named_mismatch = named_rows[0]
    if status != 200 and exact is None and named_mismatch is None:
        return _result(
            "unresolved",
            reason="pubchem_identity_lookup_non_200",
            canonical_smiles=canonical,
            inchikey=local_inchikey,
            property_response_sha256=hashlib.sha256(property_bytes).hexdigest(),
        )
    if exact is None and named_mismatch is not None:
        return _named_mismatch_result(
            named_mismatch,
            supplied_name=supplied_name,
            canonical_smiles=canonical,
            local_inchikey=local_inchikey,
            property_bytes=property_bytes,
            named_property_bytes=named_property_bytes,
            active=active,
            requester=request,
        )
    if exact is None:
        return _result(
            "unresolved",
            reason="pubchem_exact_inchikey_match_missing",
            canonical_smiles=canonical,
            inchikey=local_inchikey,
            property_response_sha256=hashlib.sha256(property_bytes).hexdigest(),
        )
    cid = int(exact.get("CID") or 0)
    synonyms: list[str] = []
    pubmed_ids: list[str] = []
    patent_ids: list[str] = []
    response_receipts = {
        "properties_sha256": hashlib.sha256(property_bytes).hexdigest()
    }
    if cid > 0:
        synonyms, synonym_sha = _synonyms(
            cid,
            active=active,
            requester=request,
        )
        pubmed_ids, patent_ids, xref_sha = _xrefs(
            cid,
            active=active,
            requester=request,
        )
        patent_view_sha = ""
        if not patent_ids:
            patent_ids, patent_view_sha = _patent_view_xrefs(
                cid,
                active=active,
                requester=request,
            )
        if synonym_sha:
            response_receipts["synonyms_sha256"] = synonym_sha
        if xref_sha:
            response_receipts["xrefs_sha256"] = xref_sha
        if patent_view_sha:
            response_receipts["patent_view_sha256"] = patent_view_sha
    preferred = " ".join(str(exact.get("Title") or "").split())
    names = _bounded_names(
        [preferred, str(exact.get("IUPACName") or ""), *synonyms],
        limit=active.max_synonyms,
    )
    observation = {
        "schema_version": TARGET_IDENTITY_OBSERVATION_SCHEMA,
        "status": "completed",
        "provider_id": "pubchem.pug_rest",
        "provider_version": TARGET_IDENTITY_PROVIDER_VERSION,
        "input": {
            "canonical_smiles": canonical,
            "inchikey": local_inchikey,
        },
        "identity": {
            "cid": cid,
            "preferred_name": preferred,
            "iupac_name": str(exact.get("IUPACName") or ""),
            "molecular_formula": str(exact.get("MolecularFormula") or ""),
            "isomeric_smiles": str(exact.get("SMILES") or exact.get("IsomericSMILES") or ""),
            "connectivity_smiles": str(exact.get("ConnectivitySMILES") or ""),
            "inchikey": str(exact.get("InChIKey") or ""),
            "synonyms": names,
            "pubmed_ids": pubmed_ids[:32],
            "patent_ids": _rank_patent_ids(patent_ids)[:64],
        },
        "response_receipts": response_receipts,
        "semantics": {
            "resolved_from_input_structure": True,
            "exact_inchikey_match_required": True,
            "lookup_strategy": lookup_strategy,
            "names_are_search_hints_not_route_evidence": True,
            "no_local_dossier_or_pdf_used": True,
        },
    }
    observation["content_sha256"] = _digest(observation)
    return observation


def _synonyms(
    cid: int,
    *,
    active: PubChemIdentityConfig,
    requester: JsonRequester,
) -> tuple[list[str], str]:
    try:
        status, content = requester(
            "GET",
            f"{active.base_url.rstrip('/')}/compound/cid/{cid}/synonyms/JSON",
            timeout_s=active.timeout_s,
            max_bytes=active.max_response_bytes,
        )
        payload = _json_object(content)
    except (OSError, RuntimeError, ValueError, requests.RequestException):
        return [], ""
    if status != 200:
        return [], hashlib.sha256(content).hexdigest()
    information = dict(payload.get("InformationList") or {}).get("Information") or []
    values = list(dict(information[0]).get("Synonym") or []) if information else []
    return _bounded_names(values, limit=active.max_synonyms), hashlib.sha256(content).hexdigest()


def _xrefs(
    cid: int,
    *,
    active: PubChemIdentityConfig,
    requester: JsonRequester,
) -> tuple[list[str], list[str], str]:
    try:
        status, content = requester(
            "GET",
            f"{active.base_url.rstrip('/')}/compound/cid/{cid}/xrefs/PubMedID,PatentID/JSON",
            timeout_s=active.timeout_s,
            max_bytes=active.max_response_bytes,
        )
        payload = _json_object(content)
    except (OSError, RuntimeError, ValueError, requests.RequestException):
        return [], [], ""
    digest = hashlib.sha256(content).hexdigest()
    if status != 200:
        return [], [], digest
    information = dict(payload.get("InformationList") or {}).get("Information") or []
    row = dict(information[0]) if information else {}
    return (
        [str(value) for value in row.get("PubMedID") or [] if str(value)],
        [str(value) for value in row.get("PatentID") or [] if str(value)],
        digest,
    )


def _named_mismatch_result(
    row: Mapping[str, Any],
    *,
    supplied_name: str,
    canonical_smiles: str,
    local_inchikey: str,
    property_bytes: bytes,
    named_property_bytes: bytes,
    active: PubChemIdentityConfig,
    requester: JsonRequester,
) -> dict[str, Any]:
    """Preserve name-linked patent discovery without accepting wrong identity."""

    cid = int(row.get("CID") or 0)
    synonyms: list[str] = []
    pubmed_ids: list[str] = []
    patent_ids: list[str] = []
    receipts = {
        "properties_sha256": hashlib.sha256(property_bytes).hexdigest(),
        "named_properties_sha256": hashlib.sha256(named_property_bytes).hexdigest(),
    }
    if cid > 0:
        synonyms, synonym_sha = _synonyms(cid, active=active, requester=requester)
        pubmed_ids, patent_ids, xref_sha = _xrefs(
            cid, active=active, requester=requester
        )
        patent_view_sha = ""
        if not patent_ids:
            patent_ids, patent_view_sha = _patent_view_xrefs(
                cid, active=active, requester=requester
            )
        if synonym_sha:
            receipts["synonyms_sha256"] = synonym_sha
        if xref_sha:
            receipts["xrefs_sha256"] = xref_sha
        if patent_view_sha:
            receipts["patent_view_sha256"] = patent_view_sha
    preferred = " ".join(str(row.get("Title") or supplied_name).split())
    result = {
        "schema_version": TARGET_IDENTITY_OBSERVATION_SCHEMA,
        "status": "unresolved",
        "reason": "pubchem_named_inchikey_mismatch",
        "provider_id": "pubchem.pug_rest",
        "provider_version": TARGET_IDENTITY_PROVIDER_VERSION,
        "canonical_smiles": canonical_smiles,
        "inchikey": local_inchikey,
        "identity": {
            "cid": cid,
            "preferred_name": preferred,
            "iupac_name": str(row.get("IUPACName") or ""),
            "inchikey": str(row.get("InChIKey") or ""),
            "synonyms": _bounded_names(
                [preferred, str(row.get("IUPACName") or ""), *synonyms],
                limit=active.max_synonyms,
            ),
            "pubmed_ids": pubmed_ids[:32],
            "patent_ids": [],
            "name_linked_patent_ids": _rank_patent_ids(patent_ids)[:64],
        },
        "response_receipts": receipts,
        "semantics": {
            "resolved_from_input_structure": False,
            "exact_inchikey_match_required": True,
            "named_record_structure_mismatch": True,
            "name_linked_patents_are_search_hints_only": True,
            "host_structure_validation_required": True,
            "no_local_dossier_or_pdf_used": True,
        },
    }
    result["content_sha256"] = _digest(result)
    return result


def _patent_view_xrefs(
    cid: int,
    *,
    active: PubChemIdentityConfig,
    requester: JsonRequester,
) -> tuple[list[str], str]:
    """Use the smaller PUG-View patent section when the bulk xref times out."""

    pug_root = active.base_url.rstrip("/")
    if pug_root.endswith("/rest/pug"):
        pug_root = pug_root[: -len("/rest/pug")]
    url = (
        f"{pug_root}/rest/pug_view/data/compound/{cid}/JSON"
        "?heading=Patents"
    )
    try:
        status, content = requester(
            "GET",
            url,
            timeout_s=max(active.timeout_s, 45.0),
            max_bytes=active.max_response_bytes,
        )
        payload = _json_object(content)
    except (OSError, RuntimeError, ValueError, requests.RequestException):
        return [], ""
    digest = hashlib.sha256(content).hexdigest()
    if status != 200:
        return [], digest
    values: list[str] = []
    for section in _walk_mappings(payload):
        if str(section.get("TOCHeading") or "") != "Patents":
            continue
        for information in section.get("Information") or []:
            if not isinstance(information, Mapping):
                continue
            markup = dict(information.get("Value") or {}).get("StringWithMarkup") or []
            for row in markup:
                if not isinstance(row, Mapping):
                    continue
                token = re.sub(r"[\s-]+", "", str(row.get("String") or "")).upper()
                if re.fullmatch(r"(?:WO|US|EP|CA|AU|CN|JP|KR|IN)[A-Z0-9]{5,}", token):
                    values.append(token)
        break
    return list(dict.fromkeys(values)), digest


def _walk_mappings(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list | tuple):
        for child in value:
            yield from _walk_mappings(child)


def _request_json(
    method: str,
    url: str,
    *,
    data: Mapping[str, Any] | None = None,
    timeout_s: float,
    max_bytes: int,
) -> tuple[int, bytes]:
    response = requests.request(
        method,
        url,
        data=dict(data or {}),
        headers={"Accept": "application/json", "User-Agent": "AutoPlanner/1.0 fresh-identity"},
        timeout=max(1.0, timeout_s),
    )
    content = bytes(response.content)
    if len(content) > max_bytes:
        raise ValueError("target_identity_response_too_large")
    return int(response.status_code), content


def _bounded_names(values: list[Any], *, limit: int) -> list[str]:
    result: list[str] = []
    for value in values:
        text = " ".join(str(value or "").split())[:500]
        if text and text.casefold() not in {row.casefold() for row in result}:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _rank_patent_ids(values: list[str]) -> list[str]:
    """Interleave jurisdictions so one long authority list cannot starve others."""

    authority_order = ("WO", "US", "EP", "CA", "AU", "CN")
    unique = list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    groups: dict[str, list[str]] = {}
    for value in unique:
        match = re.match(r"^([A-Z]{2})", value.upper())
        authority = match.group(1) if match else value.split("-", 1)[0].upper()
        groups.setdefault(authority, []).append(value)
    for rows in groups.values():
        rows.sort()
    ordered_authorities = [
        *[authority for authority in authority_order if authority in groups],
        *sorted(authority for authority in groups if authority not in authority_order),
    ]
    result: list[str] = []
    for index in range(max((len(groups[key]) for key in ordered_authorities), default=0)):
        result.extend(
            groups[authority][index]
            for authority in ordered_authorities
            if index < len(groups[authority])
        )
    return result


def _json_object(content: bytes) -> dict[str, Any]:
    value = json.loads(content.decode("utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("target_identity_response_not_object")
    return dict(value)


def _result(status: str, *, reason: str, **values: Any) -> dict[str, Any]:
    row = {
        "schema_version": TARGET_IDENTITY_OBSERVATION_SCHEMA,
        "status": status,
        "reason": reason,
        "provider_id": "pubchem.pug_rest",
        "provider_version": TARGET_IDENTITY_PROVIDER_VERSION,
        **values,
    }
    row["content_sha256"] = _digest(row)
    return row


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "PubChemIdentityConfig",
    "TARGET_IDENTITY_OBSERVATION_SCHEMA",
    "TARGET_IDENTITY_PROVIDER_VERSION",
    "resolve_target_identity",
]
