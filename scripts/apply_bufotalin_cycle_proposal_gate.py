"""Retrofit bufotalin cycle payloads with the current proposal gate."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_bufotalin_12h_iteration import _apply_cycle_proposal_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the bufotalin cycle proposal gate to existing payloads.")
    parser.add_argument("root", help="Bufotalin run root containing */web_payload.json files")
    parser.add_argument("--mode", default="hard_reject")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    report = apply_cycle_proposal_gate_to_root(
        Path(args.root),
        mode=args.mode,
        backup=not args.no_backup,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def apply_cycle_proposal_gate_to_root(root: Path, *, mode: str = "hard_reject", backup: bool = True) -> dict[str, Any]:
    root = Path(root)
    rows: list[dict[str, Any]] = []
    for payload_path in sorted(root.glob("*/web_payload.json")):
        try:
            current_text = payload_path.read_text(encoding="utf-8")
            backup_path = payload_path.with_name("web_payload_pre_proposal_gate.json")
            source_text = backup_path.read_text(encoding="utf-8") if backup_path.exists() else current_text
            payload = json.loads(source_text)
        except Exception as exc:
            rows.append(
                {
                    "path": str(payload_path),
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if backup:
            if not backup_path.exists():
                backup_path.write_text(current_text, encoding="utf-8")
        gate_report = _apply_cycle_proposal_gate(payload, mode=mode)
        payload_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        cleared_figures = _clear_stale_cycle_figures(payload_path.parent)
        rows.append(
            {
                "path": str(payload_path),
                "ok": True,
                "input_routes": gate_report.get("input_routes"),
                "kept_routes": gate_report.get("kept_routes"),
                "dropped_routes": gate_report.get("dropped_routes"),
                "repaired_routes": gate_report.get("repaired_routes"),
                "repair_reason_counts": gate_report.get("repair_reason_counts") or {},
                "reason_counts": gate_report.get("reason_counts") or {},
                "cleared_figure_files": cleared_figures,
            }
        )
    aggregate: dict[str, int] = {}
    for row in rows:
        for reason, count in (row.get("reason_counts") or {}).items():
            aggregate[str(reason)] = int(aggregate.get(str(reason), 0)) + int(count)
    repair_aggregate: dict[str, int] = {}
    for row in rows:
        for reason, count in (row.get("repair_reason_counts") or {}).items():
            repair_aggregate[str(reason)] = int(repair_aggregate.get(str(reason), 0)) + int(count)
    report = {
        "schema_version": "bufotalin_cycle_proposal_gate_retrofit.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "mode": mode,
        "payload_count": len(rows),
        "ok_count": sum(1 for row in rows if row.get("ok")),
        "input_routes": sum(int(row.get("input_routes") or 0) for row in rows),
        "kept_routes": sum(int(row.get("kept_routes") or 0) for row in rows),
        "dropped_routes": sum(int(row.get("dropped_routes") or 0) for row in rows),
        "repaired_routes": sum(int(row.get("repaired_routes") or 0) for row in rows),
        "repair_reason_counts": dict(sorted(repair_aggregate.items(), key=lambda item: (-item[1], item[0]))),
        "reason_counts": dict(sorted(aggregate.items(), key=lambda item: (-item[1], item[0]))),
        "rows": rows,
    }
    (root / "cycle_proposal_gate_retrofit_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def _clear_stale_cycle_figures(cycle_dir: Path) -> int:
    figures_dir = cycle_dir / "figures"
    if not figures_dir.exists():
        return 0
    removed = 0
    for pattern in ("scheme_route_*.svg", "scheme_route_*.pdf", "manifest.json", "index.html"):
        for path in figures_dir.glob(pattern):
            path.unlink(missing_ok=True)
            removed += 1
    return removed


if __name__ == "__main__":
    main()
