"""Batch-render exported route documents with the linear scheme renderer."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render all *_routes.json docs in a directory.")
    parser.add_argument("--route-doc-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--formats", default="svg,pdf")
    parser.add_argument("--steps-per-row", type=int, default=4)
    parser.add_argument("--aux-mode", default="mini", choices=["mini", "text", "none"])
    args = parser.parse_args()

    route_doc_dir = Path(args.route_doc_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    renderer = Path(__file__).resolve().parent / "render_linear_route_schemes.py"
    rows = []
    for doc_path in sorted(route_doc_dir.glob("*_routes.json")):
        target_dir = output_dir / doc_path.stem.replace("_top5_routes", "")
        cmd = [
            sys.executable,
            str(renderer),
            "--input",
            str(doc_path),
            "--output-dir",
            str(target_dir),
            "--top-k",
            str(args.top_k),
            "--formats",
            args.formats,
            "--steps-per-row",
            str(args.steps_per_row),
            "--aux-mode",
            args.aux_mode,
        ]
        subprocess.run(cmd, check=True)
        rows.append(str(target_dir))
    print("\n".join(rows))


if __name__ == "__main__":
    main()
