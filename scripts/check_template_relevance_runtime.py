#!/usr/bin/env python3
"""Report local ASKCOS template_relevance model availability."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.baselines.template_relevance_runtime import (
    DEFAULT_TEMPLATE_RELEVANCE_VENDOR_ROOT,
    check_template_relevance,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-root", type=Path, default=DEFAULT_TEMPLATE_RELEVANCE_VENDOR_ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = check_template_relevance(args.vendor_root)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    print(f"model_dir: {report['model_dir']}")
    print("available:")
    for item in report["models"]:
        if item["available"]:
            print(
                f"  - template_relevance.{item['name']} "
                f"({item['size_mb']:.1f} MB, templates={item.get('template_count')})"
            )
    print("missing:")
    for item in report["models"]:
        if not item["available"]:
            print(f"  - template_relevance.{item['name']} ({item['reason']})")

if __name__ == "__main__":
    main()
