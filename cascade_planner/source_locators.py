"""Pure helpers for validating and canonicalizing source locators.

This module deliberately has no third-party imports.  It is shared by the
route domain, route-forest projection, and the offline run evaluator so those
surfaces cannot disagree about whether a string identifies a real source.

The checks are syntactic rather than network-backed: a canonical locator is a
traceable identifier, not proof that the referenced document supports a
chemical claim.
"""
from __future__ import annotations

import ipaddress
import re
from typing import Any, Iterable
from urllib.parse import unquote, urlsplit, urlunsplit


_DOI_PATTERN = re.compile(r"(?i)^10\.\d{4,9}/[-._;()/:A-Z0-9]+$")
_PII_PATTERN = re.compile(r"(?i)^S\d{12,24}$")
_DOMAIN_LABEL_PATTERN = re.compile(r"(?i)^[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?$")
_PUBMED_HOSTS = {"pubmed.ncbi.nlm.nih.gov", "www.ncbi.nlm.nih.gov"}
_PMC_HOSTS = {"pmc.ncbi.nlm.nih.gov", "www.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov"}
_DOI_HOSTS = {"doi.org", "www.doi.org", "dx.doi.org"}
_SCIENCEDIRECT_HOSTS = {"sciencedirect.com", "www.sciencedirect.com"}


def canonical_traceable_source_ref(value: Any) -> str:
    """Return a strict canonical DOI/PMID/PMC/URL/PDF/PII locator.

    Arbitrary paths and descriptive strings intentionally return ``""``.
    Local files must use the explicit ``local_pdf:`` namespace and end in
    ``.pdf``.  URLs must use HTTPS and contain a syntactically valid external
    hostname.
    """

    text = unquote(str(value or "").strip()).strip()
    if not text or "\x00" in text:
        return ""
    lowered = text.lower()

    if ";" in text:
        compound_aliases = _compound_locator_aliases(text)
        if compound_aliases:
            return sorted(compound_aliases, key=source_ref_sort_key)[0]

    if lowered.startswith("local_pdf:"):
        return _canonical_local_pdf(text.split(":", 1)[1])

    prefixed_doi = _remove_prefix_case_insensitive(text, "doi:")
    if prefixed_doi is not None:
        return _canonical_doi(prefixed_doi)

    prefixed_pmid = _remove_prefix_case_insensitive(text, "pmid:")
    if prefixed_pmid is not None:
        return _canonical_numeric_identifier("pmid", prefixed_pmid)

    prefixed_pmc = _remove_prefix_case_insensitive(text, "pmc:")
    if prefixed_pmc is not None:
        return _canonical_pmc(prefixed_pmc)

    prefixed_pii = _remove_prefix_case_insensitive(text, "pii:")
    if prefixed_pii is not None:
        return _canonical_pii(prefixed_pii)

    if _DOI_PATTERN.fullmatch(text):
        return f"doi:{text.lower()}"
    if _PII_PATTERN.fullmatch(text):
        return f"pii:{text.upper()}"

    if not lowered.startswith("https://"):
        return ""
    parsed = _validated_https_parts(text)
    if parsed is None:
        return ""
    host = str(parsed.hostname or "").lower()
    path = parsed.path or "/"

    if host in _DOI_HOSTS:
        return _canonical_doi(path.lstrip("/"))
    if host in _PUBMED_HOSTS:
        match = re.fullmatch(r"/(?:pubmed/)?(\d+)/?", path, flags=re.IGNORECASE)
        if match:
            return _canonical_numeric_identifier("pmid", match.group(1))
    if host in _PMC_HOSTS:
        match = re.fullmatch(
            r"/(?:pmc/)?articles/(?:PMC)?(\d+)/?",
            path,
            flags=re.IGNORECASE,
        )
        if match:
            return _canonical_numeric_identifier("pmc", match.group(1))
    if host in _SCIENCEDIRECT_HOSTS:
        match = re.search(r"/pii/(S\d{12,24})(?:/|$)", path, flags=re.IGNORECASE)
        if match:
            return _canonical_pii(match.group(1))

    netloc = f"[{host}]" if ":" in host else host
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    normalized_path = re.sub(r"/{2,}", "/", path) or "/"
    return f"url:{urlunsplit(('https', netloc, normalized_path, parsed.query, ''))}"


def canonical_traceable_source_refs(values: Iterable[Any]) -> list[str]:
    """Return stable, deduplicated canonical locators in preference order."""

    aliases: set[str] = set()
    for value in values:
        text = unquote(str(value or "").strip()).strip()
        compound = _compound_locator_aliases(text) if ";" in text else []
        if compound:
            aliases.update(compound)
            continue
        if alias := canonical_traceable_source_ref(text):
            aliases.add(alias)
    return sorted(aliases, key=source_ref_sort_key)


def source_ref_sort_key(value: str) -> tuple[int, str]:
    priorities = {
        "doi": 0,
        "pmid": 1,
        "pmc": 2,
        "pii": 3,
        "url": 4,
        "local_pdf": 5,
    }
    prefix = str(value).split(":", 1)[0]
    return priorities.get(prefix, 99), str(value)


def source_record_support_group(
    source_channel: Any,
    evidence_level: Any,
    source_refs: Iterable[Any] = (),
    evidence_refs: Iterable[Any] = (),
) -> str:
    """Derive one conservative independence group from a source record.

    Producer-authored ``support_group`` values are intentionally absent from
    this API.  All Codex roles are correlated.  Deterministic computational
    channels remain distinguishable.  Only an exact-literature channel bound
    to a valid locator can form an external independent group.
    """

    channel = str(source_channel or "other").strip().lower().replace("-", "_")
    level = str(evidence_level or "model_only").strip().lower().replace("-", "_")
    if channel.startswith("codex_") or channel == "codex":
        return "codex_model"
    if channel in {"chem_enzy", "template", "stock"}:
        return f"computational:{channel}"

    aliases = canonical_traceable_source_refs(
        [*_locator_values(source_refs), *_locator_values(evidence_refs)]
    )
    preferred = aliases[0] if aliases else ""
    if not preferred:
        # Unknown legacy/model records share one conservative correlation
        # bucket.  They must never increase source diversity by changing a
        # producer-authored channel or support_group label.
        return "codex_model"
    if channel == "literature_exact" and level == "literature_exact":
        return f"literature:{preferred}"
    if channel == "literature_exact" and level == "validated":
        return f"validated:{preferred}"
    # A locator in an analogy/model/unknown record proves only that a document
    # can be found, not that the claimed step was bound to that document.
    return "codex_model"


def _remove_prefix_case_insensitive(text: str, prefix: str) -> str | None:
    if text[: len(prefix)].lower() != prefix:
        return None
    return text[len(prefix) :].strip()


def _locator_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _compound_locator_aliases(text: str) -> list[str]:
    """Extract only explicitly named locator fields from a compound record."""

    aliases: list[str] = []
    allowed = {"doi", "pmid", "pmc", "pii", "url", "local_pdf"}
    ignored_metadata = {"patent_publication", "lines"}
    segments = text.split(";")
    if len(segments) < 2:
        return []
    parsed_segments: list[tuple[str, str]] = []
    for segment in segments:
        key, separator, raw_value = segment.partition(":")
        key = key.strip().lower()
        if (
            not separator
            or key not in allowed | ignored_metadata
            or not raw_value.strip()
        ):
            return []
        parsed_segments.append((key, raw_value.strip()))
    for key, raw_value in parsed_segments:
        if key not in allowed:
            continue
        candidate = raw_value if key == "url" else f"{key}:{raw_value}"
        alias = canonical_traceable_source_ref(candidate)
        if alias and alias not in aliases:
            aliases.append(alias)
    return aliases


def _canonical_doi(value: str) -> str:
    text = str(value or "").strip()
    if not _DOI_PATTERN.fullmatch(text):
        return ""
    return f"doi:{text.lower()}"


def _canonical_numeric_identifier(prefix: str, value: str) -> str:
    text = str(value or "").strip()
    if not text.isascii() or not text.isdigit():
        return ""
    number = int(text)
    return f"{prefix}:{number}" if number > 0 else ""


def _canonical_pmc(value: str) -> str:
    text = str(value or "").strip()
    if text[:3].lower() == "pmc":
        text = text[3:].strip()
    return _canonical_numeric_identifier("pmc", text)


def _canonical_pii(value: str) -> str:
    text = str(value or "").strip().upper()
    return f"pii:{text}" if _PII_PATTERN.fullmatch(text) else ""


def _canonical_local_pdf(value: str) -> str:
    identity = str(value or "").split("#", 1)[0].strip().replace("\\", "/")
    if not identity or any(char in identity for char in ("\x00", "\r", "\n")):
        return ""
    if not identity.lower().endswith(".pdf"):
        return ""
    return f"local_pdf:{identity.lower()}"


def _validated_https_parts(value: str):
    if any(char.isspace() for char in value):
        return None
    try:
        parsed = urlsplit(value)
        host = str(parsed.hostname or "")
        # Accessing port validates its syntax and range.
        _ = parsed.port
    except ValueError:
        return None
    if parsed.scheme.lower() != "https" or not host or parsed.username or parsed.password:
        return None
    if not _valid_external_hostname(host):
        return None
    return parsed


def _valid_external_hostname(host: str) -> bool:
    text = host.rstrip(".")
    try:
        address = ipaddress.ip_address(text)
        return not any(
            (
                address.is_private,
                address.is_loopback,
                address.is_link_local,
                address.is_multicast,
                address.is_unspecified,
                address.is_reserved,
            )
        )
    except ValueError:
        pass
    try:
        ascii_host = text.encode("idna").decode("ascii")
    except UnicodeError:
        return False
    labels = ascii_host.split(".")
    if len(labels) < 2 or len(ascii_host) > 253:
        return False
    if all(label.isdigit() for label in labels):
        return False
    return all(_DOMAIN_LABEL_PATTERN.fullmatch(label) for label in labels)
