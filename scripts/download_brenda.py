#!/usr/bin/env python3
"""Download BRENDA flat file.

BRENDA requires free registration at https://www.brenda-enzymes.org/register.php
After registration, set environment variables and run this script:

    export BRENDA_EMAIL="your@email.com"
    export BRENDA_PASSWORD="your_password"
    python scripts/download_brenda.py

The script downloads T_opt and pH_opt data via BRENDA SOAP API
and saves a JSON lookup at results/shared/brenda_lookup_full.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent


def main():
    email = os.environ.get("BRENDA_EMAIL", "")
    password = os.environ.get("BRENDA_PASSWORD", "")

    if not email or not password:
        print(
            "BRENDA credentials not set.\n\n"
            "1. Register (free) at https://www.brenda-enzymes.org/register.php\n"
            "2. Activate your account via the confirmation email\n"
            "3. Run:\n"
            "     export BRENDA_EMAIL='your@email.com'\n"
            "     export BRENDA_PASSWORD='your_password'\n"
            "     python scripts/download_brenda.py\n",
            file=sys.stderr,
        )
        sys.exit(1)

    pw_hash = hashlib.sha256(password.encode()).hexdigest()

    try:
        from zeep import Client
    except ImportError:
        print("pip install zeep", file=sys.stderr)
        sys.exit(1)

    # Collect EC numbers from AutoPlanner data
    norm_path = ROOT / "workspace" / "cascade_dataset_v2.normalized.json"
    if not norm_path.exists():
        norm_path = ROOT / "cascade_dataset_v2.normalized.json"
    if norm_path.exists():
        data = json.loads(norm_path.read_text())
        ec_set = set()
        for art in data.get("records_kept", []):
            for c in art.get("cascades", []):
                for s in c.get("steps", []):
                    for cat in s.get("catalyst_components") or []:
                        if cat and cat.get("ec_number"):
                            ec = cat["ec_number"]
                            parts = ec.split(".")
                            if len(parts) == 4 and all(p.isdigit() for p in parts):
                                ec_set.add(ec)
        print(f"EC numbers from AutoPlanner: {len(ec_set)}")
    else:
        # Fallback: common EC classes
        ec_set = set()
        for ec1 in range(1, 8):
            ec_set.add(f"{ec1}.*.*.*")
        print("No dataset found, querying all EC classes")

    WSDL = "https://www.brenda-enzymes.org/soap/brenda_zeep.wsdl"
    print(f"Connecting to BRENDA SOAP API...")
    client = Client(WSDL)

    # Fetch T_opt and pH_opt for each EC
    t_opt_data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    ph_opt_data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    total = len(ec_set)
    fetched = 0
    errors = 0

    for ec in sorted(ec_set):
        # Temperature optimum
        try:
            result = client.service.getTemperatureOptimum(
                email=email, password=pw_hash,
                ecNumber=f"ecNumber*{ec}",
                organism="organism*",
                temperatureOptimum="temperatureOptimum*",
                temperatureOptimumMaximum="temperatureOptimumMaximum*",
                commentary="commentary*",
                literature="literature*",
            )
            if result:
                for entry in result.split("!"):
                    parts = {}
                    for field in entry.split("#"):
                        if "*" in field:
                            k, v = field.split("*", 1)
                            parts[k] = v
                    org = parts.get("organism", "").strip()
                    t_val = parts.get("temperatureOptimum", "").strip()
                    if t_val:
                        try:
                            t = float(t_val)
                            if 0 < t < 120:
                                t_opt_data[ec][org].append(t)
                        except ValueError:
                            pass
        except Exception:
            errors += 1

        # pH optimum
        try:
            result = client.service.getPhOptimum(
                email=email, password=pw_hash,
                ecNumber=f"ecNumber*{ec}",
                organism="organism*",
                phOptimum="phOptimum*",
                phOptimumMaximum="phOptimumMaximum*",
                commentary="commentary*",
                literature="literature*",
            )
            if result:
                for entry in result.split("!"):
                    parts = {}
                    for field in entry.split("#"):
                        if "*" in field:
                            k, v = field.split("*", 1)
                            parts[k] = v
                    org = parts.get("organism", "").strip()
                    ph_val = parts.get("phOptimum", "").strip()
                    if ph_val:
                        try:
                            p = float(ph_val)
                            if 2 < p < 13:
                                ph_opt_data[ec][org].append(p)
                        except ValueError:
                            pass
        except Exception:
            errors += 1

        fetched += 1
        if fetched % 20 == 0:
            print(f"  [{fetched}/{total}] T: {len(t_opt_data)} ECs, pH: {len(ph_opt_data)} ECs, errors: {errors}")
        time.sleep(0.3)  # rate limit

    # Build lookup: "ec||organism" -> {"T_opt": float, "pH_opt": float}
    lookup = {}
    for ec in sorted(set(list(t_opt_data.keys()) + list(ph_opt_data.keys()))):
        # Per-organism entries
        all_orgs = set(list(t_opt_data.get(ec, {}).keys()) + list(ph_opt_data.get(ec, {}).keys()))
        for org in all_orgs:
            key = f"{ec}||{org}"
            entry = {}
            if org in t_opt_data.get(ec, {}):
                entry["T_opt"] = round(median(t_opt_data[ec][org]), 1)
            if org in ph_opt_data.get(ec, {}):
                entry["pH_opt"] = round(median(ph_opt_data[ec][org]), 2)
            if entry:
                lookup[key] = entry

        # EC-level aggregate (no organism)
        all_t = [v for vs in t_opt_data.get(ec, {}).values() for v in vs]
        all_ph = [v for vs in ph_opt_data.get(ec, {}).values() for v in vs]
        agg = {}
        if all_t:
            agg["T_opt"] = round(median(all_t), 1)
        if all_ph:
            agg["pH_opt"] = round(median(all_ph), 2)
        if agg:
            lookup[f"{ec}||"] = agg

    out_path = ROOT / "results" / "shared" / "brenda_lookup_full.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(lookup, indent=2))

    print(f"\nDone: {len(lookup)} entries saved to {out_path}")
    print(f"  ECs with T_opt: {len(t_opt_data)}")
    print(f"  ECs with pH_opt: {len(ph_opt_data)}")
    print(f"  Total organisms: {sum(len(v) for v in t_opt_data.values())}")


if __name__ == "__main__":
    main()
