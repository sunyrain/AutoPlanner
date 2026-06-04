"""Summarize already-applied bufotalin proposal gates without mutating payloads."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize proposal_gate reports in bufotalin payloads.")
    parser.add_argument("root", help="Bufotalin run root containing */web_payload.json files")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output) if args.output else root / "cycle_proposal_gate_retrofit_summary.json"
    report = summarize_existing_proposal_gates(root)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "payload_count": report["payload_count"],
                "input_routes": report["input_routes"],
                "kept_routes": report["kept_routes"],
                "dropped_routes": report["dropped_routes"],
                "repaired_routes": report["repaired_routes"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def summarize_existing_proposal_gates(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    repair_reason_counts: Counter[str] = Counter()
    for payload_path in sorted(root.glob("*/web_payload.json")):
        payload = _read_json(payload_path)
        gate = _payload_gate(payload)
        row = _row_from_gate(payload_path, gate)
        rows.append(row)
        reason_counts.update(_int_counts(gate.get("reason_counts") or {}))
        repair_reason_counts.update(_int_counts(gate.get("repair_reason_counts") or {}))
    modes = sorted({str(row.get("mode") or "") for row in rows if row.get("mode")})
    return {
        "schema_version": "bufotalin_cycle_proposal_gate_summary.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "mode": modes[0] if len(modes) == 1 else ("mixed" if modes else ""),
        "payload_count": len(rows),
        "ok_count": sum(1 for row in rows if row.get("ok")),
        "input_routes": sum(int(row.get("input_routes") or 0) for row in rows),
        "kept_routes": sum(int(row.get("kept_routes") or 0) for row in rows),
        "dropped_routes": sum(int(row.get("dropped_routes") or 0) for row in rows),
        "repaired_routes": sum(int(row.get("repaired_routes") or 0) for row in rows),
        "repair_reason_counts": dict(sorted(repair_reason_counts.items(), key=lambda item: (-item[1], item[0]))),
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))),
        "rows": rows,
    }


def _payload_gate(payload: dict[str, Any]) -> dict[str, Any]:
    gate = payload.get("proposal_gate")
    if isinstance(gate, dict):
        return gate
    for parent_key in ("route_set_metrics", "ui_metadata"):
        parent = payload.get(parent_key) or {}
        gate = parent.get("proposal_gate")
        if isinstance(gate, dict):
            return gate
    return {}


def _row_from_gate(path: Path, gate: dict[str, Any]) -> dict[str, Any]:
    if not gate:
        return {
            "path": str(path),
            "ok": False,
            "error": "missing proposal_gate",
            "input_routes": 0,
            "kept_routes": 0,
            "dropped_routes": 0,
            "repaired_routes": 0,
            "repair_reason_counts": {},
            "reason_counts": {},
        }
    return {
        "path": str(path),
        "ok": True,
        "mode": gate.get("mode"),
        "input_routes": int(gate.get("input_routes") or 0),
        "kept_routes": int(gate.get("kept_routes") or 0),
        "dropped_routes": int(gate.get("dropped_routes") or 0),
        "repaired_routes": int(gate.get("repaired_routes") or 0),
        "repair_reason_counts": _int_counts(gate.get("repair_reason_counts") or {}),
        "reason_counts": _int_counts(gate.get("reason_counts") or {}),
        "frontier_count": len(gate.get("frontiers") or []),
        "dropped_row_count": len(gate.get("dropped") or []),
    }


def _int_counts(counts: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in counts.items():
        try:
            out[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return out


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


if __name__ == "__main__":
    main()
