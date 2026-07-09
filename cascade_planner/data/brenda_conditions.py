"""Parse BRENDA flat-file for Temperature/pH optima and enrich cascade steps.

BRENDA flat-file format (brenda_download.txt):
  - Organised by EC number, sections delimited by ``///``
  - Field codes used here: ``TO`` (Temperature Optimum), ``PHO`` (pH Optimum)
  - Each data line: ``<FIELD>\t#<org_id># <value> (#<org_id># <organism> <refs>)``

Provides:
  * ``parse_brenda_flatfile``  – raw extraction
  * ``BRENDALookup``          – hierarchical EC lookup with organism fallback
  * ``enrich_steps``          – fill missing T/pH on StepRowV2 from BRENDA
  * CLI (``python -m cascade_planner.data.brenda_conditions``)
"""
from __future__ import annotations

import json
import logging
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Flat-file parser
# ---------------------------------------------------------------------------

# Matches lines like:  TO\t#1# 37 (#1# Homo sapiens <1,2>)
_ENTRY_RE = re.compile(
    r"^(?P<field>\w+)\t"
    r"#(?P<org_id>\d+)#\s+"
    r"(?P<value>[\d.eE+\-]+)"
    r"\s*"
    r"(?:\(#\d+#\s*(?P<organism>[^<)]+?)\s*(?:<[^>]*>)?\s*\))?"
)


def parse_brenda_flatfile(path: str | Path) -> dict[tuple[str, str], dict[str, float]]:
    """Parse a BRENDA flat file and return per-(EC, organism) T_opt / pH_opt.

    Returns
    -------
    dict mapping ``(ec_4level, organism_lower)`` to
    ``{"T_opt": median_float, "pH_opt": median_float}``.
    Keys are only present when at least one valid value was found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    current_ec: str | None = None
    # Accumulate raw values: (ec, organism) -> {"TO": [floats], "PHO": [floats]}
    raw: dict[tuple[str, str], dict[str, list[float]]] = {}

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.rstrip("\n\r")

            # EC header
            if line.startswith("ID\t"):
                current_ec = line.split("\t", 1)[1].strip()
                continue

            if not current_ec:
                continue

            # Only care about TO and PHO
            if not (line.startswith("TO\t") or line.startswith("PHO\t")):
                continue

            m = _ENTRY_RE.match(line)
            if m is None:
                log.debug("Skipping malformed line %d: %s", line_no, line[:120])
                continue

            field_code = m.group("field")
            try:
                value = float(m.group("value"))
            except ValueError:
                continue

            organism = (m.group("organism") or "").strip().lower()
            if not organism:
                organism = "_unknown_"

            key = (current_ec, organism)
            raw.setdefault(key, {}).setdefault(field_code, []).append(value)

    # Collapse to medians
    result: dict[tuple[str, str], dict[str, float]] = {}
    for key, fields in raw.items():
        entry: dict[str, float] = {}
        if "TO" in fields:
            entry["T_opt"] = statistics.median(fields["TO"])
        if "PHO" in fields:
            entry["pH_opt"] = statistics.median(fields["PHO"])
        if entry:
            result[key] = entry

    log.info("Parsed BRENDA: %d (EC, organism) entries from %s", len(result), path.name)
    return result


# ---------------------------------------------------------------------------
# 2. Hierarchical lookup
# ---------------------------------------------------------------------------

def _ec_prefixes(ec: str) -> list[str]:
    """Return EC prefixes from most to least specific.

    ``"1.2.3.4"`` -> ``["1.2.3.4", "1.2.3", "1.2", "1"]``
    """
    parts = ec.split(".")
    out: list[str] = []
    for i in range(len(parts), 0, -1):
        out.append(".".join(parts[:i]))
    return out


@dataclass
class BRENDALookup:
    """Hierarchical BRENDA condition lookup.

    Resolution order for a query ``(ec, organism)``:

    1. Exact (EC4, organism) match
    2. Exact EC4, any-organism median
    3. EC3 prefix median  ->  EC2  ->  EC1
    """

    # ec4 -> organism -> {"T_opt": float, "pH_opt": float}
    _by_ec_org: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    # ec_prefix -> {"T_opt": median, "pH_opt": median}  (pre-computed)
    _by_prefix: dict[str, dict[str, float]] = field(default_factory=dict)

    # -- query API ----------------------------------------------------------

    def get_temperature(self, ec: str, organism: str | None = None) -> float | None:
        return self._get("T_opt", ec, organism)

    def get_ph(self, ec: str, organism: str | None = None) -> float | None:
        return self._get("pH_opt", ec, organism)

    def get_conditions(self, ec: str, organism: str | None = None) -> dict[str, float | None]:
        return {
            "T_opt": self.get_temperature(ec, organism),
            "pH_opt": self.get_ph(ec, organism),
        }

    # -- internals ----------------------------------------------------------

    def _get(self, key: str, ec: str, organism: str | None) -> float | None:
        ec = ec.strip()
        org = (organism or "").strip().lower()

        # 1. exact (ec4, organism)
        if org and ec in self._by_ec_org:
            entry = self._by_ec_org[ec].get(org)
            if entry and key in entry:
                return entry[key]

        # 2. walk prefixes (ec4 any-org median, ec3, ec2, ec1)
        for prefix in _ec_prefixes(ec):
            if prefix in self._by_prefix and key in self._by_prefix[prefix]:
                return self._by_prefix[prefix][key]

        return None


def build_ec_lookup(brenda_data: dict[tuple[str, str], dict[str, float]]) -> BRENDALookup:
    """Build a :class:`BRENDALookup` from parsed BRENDA data."""
    by_ec_org: dict[str, dict[str, dict[str, float]]] = {}
    for (ec, org), vals in brenda_data.items():
        by_ec_org.setdefault(ec, {})[org] = vals

    # Pre-compute prefix medians.  For each prefix length collect all values
    # that fall under it, then take the median.
    prefix_vals: dict[str, dict[str, list[float]]] = {}
    for (ec, _org), vals in brenda_data.items():
        for prefix in _ec_prefixes(ec):
            bucket = prefix_vals.setdefault(prefix, {})
            for k, v in vals.items():
                bucket.setdefault(k, []).append(v)

    by_prefix: dict[str, dict[str, float]] = {}
    for prefix, bucket in prefix_vals.items():
        by_prefix[prefix] = {k: statistics.median(vs) for k, vs in bucket.items()}

    lookup = BRENDALookup(_by_ec_org=by_ec_org, _by_prefix=by_prefix)
    log.info(
        "BRENDALookup: %d EC4 entries, %d prefix entries",
        len(by_ec_org),
        len(by_prefix),
    )
    return lookup


# ---------------------------------------------------------------------------
# 3. Step enrichment
# ---------------------------------------------------------------------------

def enrich_steps(
    steps: list[Any],
    lookup: BRENDALookup,
) -> tuple[list[Any], list[dict[str, str]]]:
    """Fill missing T/pH on *steps* from BRENDA lookup.

    Parameters
    ----------
    steps : list[StepRowV2]
        Mutable step rows.  Modified in-place **and** returned.
    lookup : BRENDALookup

    Returns
    -------
    (steps, sources) where *sources* is a parallel list of dicts recording
    which fields were filled and at what resolution level, e.g.
    ``{"temperature_c": "brenda_ec4_median", "ph": "brenda_ec3_median"}``.
    """
    sources: list[dict[str, str]] = []
    filled_t = filled_ph = 0

    for step in steps:
        src: dict[str, str] = {}
        ec = getattr(step, "ec_number", None)
        if not ec:
            sources.append(src)
            continue

        # Temperature
        if getattr(step, "temperature_c", None) is None:
            t = lookup.get_temperature(ec)
            if t is not None:
                step.temperature_c = t
                src["temperature_c"] = "brenda"
                filled_t += 1

        # pH
        if getattr(step, "ph", None) is None:
            ph = lookup.get_ph(ec)
            if ph is not None:
                step.ph = ph
                src["ph"] = "brenda"
                filled_ph += 1

        sources.append(src)

    log.info("Enriched %d T and %d pH values from BRENDA", filled_t, filled_ph)
    return steps, sources


# ---------------------------------------------------------------------------
# 4. Download instructions
# ---------------------------------------------------------------------------

def download_brenda_instructions() -> str:
    return (
        "BRENDA flat file download\n"
        "=========================\n"
        "1. Register at https://www.brenda-enzymes.org/download.php\n"
        "2. Download 'brenda_download.txt' (full flat file, ~2 GB uncompressed).\n"
        "3. Place it at a convenient path and pass --brenda-file <path>.\n"
        "\n"
        "Alternative: BRENDA SOAP API (programmatic, no flat file needed)\n"
        "----------------------------------------------------------------\n"
        "  pip install zeep\n"
        "\n"
        "  from zeep import Client\n"
        "  WSDL = 'https://www.brenda-enzymes.org/soap/brenda_zeep.wsdl'\n"
        "  client = Client(WSDL)\n"
        "  # Example: get temperature optima for EC 1.1.1.1\n"
        "  result = client.service.getTemperatureOptimum(\n"
        "      'your@email.com', 'your_password',\n"
        "      'ecNumber*1.1.1.1',\n"
        "  )\n"
        "  # Similarly: getPhOptimum, getKmValue, etc.\n"
    )


# ---------------------------------------------------------------------------
# 5. JSON cache helpers
# ---------------------------------------------------------------------------

def _cache_path(brenda_path: Path) -> Path:
    return brenda_path.with_suffix(".lookup.json")


def _save_cache(brenda_path: Path, brenda_data: dict[tuple[str, str], dict[str, float]]) -> None:
    """Serialise parsed data to JSON for fast reload."""
    serialisable = {f"{ec}||{org}": vals for (ec, org), vals in brenda_data.items()}
    _cache_path(brenda_path).write_text(json.dumps(serialisable), encoding="utf-8")
    log.info("Saved BRENDA cache to %s", _cache_path(brenda_path))


def _load_cache(brenda_path: Path) -> dict[tuple[str, str], dict[str, float]] | None:
    cp = _cache_path(brenda_path)
    if not cp.exists():
        return None
    if cp.stat().st_mtime < brenda_path.stat().st_mtime:
        log.info("Cache stale, re-parsing")
        return None
    try:
        raw = json.loads(cp.read_text(encoding="utf-8"))
        return {
            (k.split("||")[0], k.split("||")[1]): v
            for k, v in raw.items()
        }
    except Exception:
        log.warning("Failed to load cache, re-parsing", exc_info=True)
        return None


def load_or_parse(brenda_path: str | Path) -> dict[tuple[str, str], dict[str, float]]:
    """Load from JSON cache if fresh, otherwise parse flat file and cache."""
    bp = Path(brenda_path)
    cached = _load_cache(bp)
    if cached is not None:
        log.info("Loaded BRENDA data from cache (%d entries)", len(cached))
        return cached
    data = parse_brenda_flatfile(bp)
    _save_cache(bp, data)
    return data


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def _print_stats(steps: list[Any], label: str) -> None:
    total = len(steps)
    if total == 0:
        print(f"[{label}] No steps.")
        return
    n_t = sum(1 for s in steps if s.temperature_c is not None)
    n_ph = sum(1 for s in steps if s.ph is not None)
    n_ec = sum(1 for s in steps if s.ec_number)
    print(
        f"[{label}] {total} steps | "
        f"T: {n_t} ({100*n_t/total:.1f}%) | "
        f"pH: {n_ph} ({100*n_ph/total:.1f}%) | "
        f"EC: {n_ec} ({100*n_ec/total:.1f}%)"
    )


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="BRENDA condition enrichment for AutoPlanner steps",
    )
    parser.add_argument("--brenda-file", type=str, default=None, help="Path to brenda_download.txt")
    parser.add_argument("--data", type=str, default=None, help="Path to cascade_dataset_v2.normalized.json")
    parser.add_argument("--stats", action="store_true", help="Print coverage statistics")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # If no brenda file, print instructions and exit
    if args.brenda_file is None or not Path(args.brenda_file).exists():
        print(download_brenda_instructions())
        if args.brenda_file and not Path(args.brenda_file).exists():
            print(f"\nFile not found: {args.brenda_file}")
        return

    # Parse / load BRENDA
    brenda_data = load_or_parse(args.brenda_file)
    lookup = build_ec_lookup(brenda_data)
    print(f"BRENDA lookup ready: {len(brenda_data)} (EC, organism) entries")

    if args.data is None:
        print("No --data provided; lookup built but no enrichment performed.")
        return

    # Load steps
    from cascade_planner.data.loader_v2 import StepRowV2, load_v2

    steps, _pairs, _cascades = load_v2(args.data)

    if args.stats:
        _print_stats(steps, "BEFORE")

    steps, sources = enrich_steps(steps, lookup)

    if args.stats:
        _print_stats(steps, "AFTER")
        filled_t = sum(1 for s in sources if "temperature_c" in s)
        filled_ph = sum(1 for s in sources if "ph" in s)
        print(f"Filled from BRENDA: {filled_t} T values, {filled_ph} pH values")


if __name__ == "__main__":
    main()
