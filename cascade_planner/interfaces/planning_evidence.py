"""Fixed-provider discovery adapters for bounded planning queries.

No arbitrary URL opening, automatic fallback fan-out, SI download, figure
extraction, or paid model invocation is performed here.
"""
from __future__ import annotations

import html
import re
import time
from urllib.parse import quote
import xml.etree.ElementTree as ET

import requests
from rdkit import Chem

from cascade_planner.application.planning_evidence import canonical_structure
from cascade_planner.interfaces.literature_search import europe_pmc_metadata_search


def _bounded_get(url: str, **kwargs):
    started = time.monotonic()
    # One request, no implicit redirects/retries. Hard streamed byte bound;
    # connect/read timeouts plus an elapsed check bound slow streaming.
    with requests.get(url, **{**kwargs, "timeout": (5, 15), "stream": True,
                             "allow_redirects": False}) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError("source_did_not_return_content")
        parts = []
        size = 0
        for chunk in response.iter_content(chunk_size=16384):
            size += len(chunk)
            if size > 4_000_000 or time.monotonic() - started > 25:
                raise ValueError("source_response_limit")
            parts.append(chunk)
        # Keep the existing metadata adapter's response interface.
        response._content = b"".join(parts)
        response._content_consumed = True
        return response


def search_planning_literature(arguments: dict) -> dict:
    rows = europe_pmc_metadata_search(
        arguments["query"], 3, requester=_bounded_get, include_abstract=True,
    )
    sources = []
    for row in rows:
        identity = str(row.get("doi") or row.get("source_ref") or "").lower()
        if not identity:
            continue
        sources.append({**row, "source_id": identity,
                        "evidence_kind": "metadata_and_abstract",
                        "provider": "Europe PMC"})
    return {"status": "ok" if sources else "no_hit", "sources": sources,
            "coverage": "Europe PMC only; no hit does not imply no chemical precedent"}


def resolve_planning_compound(arguments: dict) -> dict:
    name = quote(arguments["query"], safe="")
    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/"
           "MolecularFormula,CanonicalSMILES,IsomericSMILES,IUPACName/JSON")
    try:
        payload = _bounded_get(url).json()
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return {"status": "no_hit", "candidates": [],
                    "evidence_kind": "compound_identity_not_availability"}
        raise
    candidates = []
    for row in payload.get("PropertyTable", {}).get("Properties", [])[:3]:
        smiles = str(row.get("SMILES") or row.get("IsomericSMILES") or "")
        smiles = canonical_structure(smiles)
        submitted = arguments.get("smiles", "")
        relation = "not_compared"
        if submitted:
            submitted = canonical_structure(submitted)
            if submitted == smiles:
                relation = "exact"
            else:
                left, right = Chem.MolFromSmiles(submitted), Chem.MolFromSmiles(smiles)
                Chem.RemoveStereochemistry(left)
                Chem.RemoveStereochemistry(right)
                relation = ("connectivity_only_stereo_unresolved_or_different"
                            if Chem.MolToSmiles(left) == Chem.MolToSmiles(right) else "different")
        cid = int(row["CID"])
        candidates.append({"cid": cid, "smiles": smiles, "relation_to_submitted": relation,
                           "formula": str(row.get("MolecularFormula") or ""),
                           "name": str(row.get("IUPACName") or "")[:1000],
                           "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"})
    return {"status": "ok" if candidates else "no_hit", "candidates": candidates,
            "evidence_kind": "compound_identity_not_availability"}


def read_planning_literature(source: dict) -> dict:
    pmcid = str(source.get("pmcid") or "").upper()
    if not re.fullmatch(r"PMC\d+", pmcid):
        return {"status": "unavailable", "reason": "no_repository_fulltext",
                "source_id": source["source_id"]}
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    root = ET.fromstring(_bounded_get(url).content)
    dois = {"".join(node.itertext()).strip().lower() for node in root.findall(".//article-id")
            if node.get("pub-id-type") == "doi"}
    if str(source.get("doi") or "").lower() not in dois:
        return {"status": "unavailable", "reason": "source_identity_mismatch"}
    paragraphs = [" ".join(" ".join(node.itertext()).split()) for node in root.findall(".//body//p")]
    # Put procedure-bearing paragraphs first, retaining body order as a tie-break.
    ranked = sorted(enumerate(paragraphs), key=lambda pair: (
        -int(bool(re.search(r"\b(yield|synthesi[sz]|cataly|reaction|substrate|procedure|prepared)",
                            pair[1], re.I))), pair[0]))
    excerpt = "\n\n".join(f"[body paragraph {index + 1}] {text}" for index, text in ranked)
    return {"status": "ok", "source_id": source["source_id"], "url": url,
            "evidence_kind": "article_text_excerpt_not_verified_reaction",
            "text": html.unescape(excerpt)[:12000], "truncated": len(excerpt) > 12000,
            "supplement_and_figures_read": False}


def planning_evidence_providers() -> dict:
    return {"compound": resolve_planning_compound, "search": search_planning_literature,
            "read": read_planning_literature}
