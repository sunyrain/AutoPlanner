"""Print a bounded ChemEnzy runtime capability preflight report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.baselines.chem_enzy_runtime import diagnose_chem_enzy_runtime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check ChemEnzy interpreter and checkout paths without importing or running models."
    )
    parser.add_argument("--env-prefix", default=None)
    parser.add_argument("--vendor-root", default=None)
    parser.add_argument("--launcher", default=None)
    parser.add_argument(
        "--filesystem-only",
        action="store_true",
        help="Only discover files; this mode never reports production ready.",
    )
    parser.add_argument("--capability-probe-timeout-s", type=float, default=60.0)
    args = parser.parse_args()
    report = diagnose_chem_enzy_runtime(
        env_prefix=args.env_prefix,
        vendor_root=args.vendor_root,
        launcher_path=args.launcher,
        capability_probe=not bool(args.filesystem_only),
        capability_probe_timeout_s=float(args.capability_probe_timeout_s),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
