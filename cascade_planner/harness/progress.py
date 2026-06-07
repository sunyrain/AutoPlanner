"""Minimal run progress artifacts for the Codex-entry harness."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def write_progress_panel(
    *,
    run_dir: str | Path,
    target_input: dict[str, Any],
    preflight: dict[str, Any],
    workflow_plan: dict[str, Any],
    final_verdict: dict[str, Any] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> None:
    run_path = Path(run_dir)
    final_verdict = dict(final_verdict or {})
    tool_calls = list(tool_calls or [])
    rows = "\n".join(
        "<tr>"
        f"<td>{_esc(str(row.get('tool_name') or ''))}</td>"
        f"<td>{_esc(str(row.get('status') or ''))}</td>"
        f"<td>{_esc(', '.join(str(item) for item in row.get('reasons') or []))}</td>"
        "</tr>"
        for row in tool_calls
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AutoPlanner Codex Entry Progress</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; background: #f8fafc; }}
    main {{ max-width: 1120px; margin: 0 auto; }}
    section {{ margin-bottom: 20px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #e2e8f0; }}
    code {{ background: #e5e7eb; padding: 1px 4px; border-radius: 3px; }}
    pre {{ background: white; border: 1px solid #cbd5e1; padding: 12px; overflow: auto; }}
  </style>
</head>
<body>
<main>
  <h1>AutoPlanner Codex Entry Progress</h1>
  <section>
    <h2>Target</h2>
    <p><strong>{_esc(str(target_input.get('target_name') or 'target'))}</strong></p>
    <p><code>{_esc(str(target_input.get('target_smiles') or ''))}</code></p>
  </section>
  <section>
    <h2>Preflight</h2>
    <p>Accepted: <code>{_esc(str(bool(preflight.get('accepted'))))}</code></p>
    <p>Risk flags: <code>{_esc(', '.join(str(item) for item in preflight.get('initial_risk_flags') or []))}</code></p>
  </section>
  <section>
    <h2>Plan</h2>
    <p>Strategy: <code>{_esc(str(workflow_plan.get('recommended_strategy') or ''))}</code></p>
    <p>Expected floor: <code>{_esc(str(workflow_plan.get('expected_verdict_floor') or ''))}</code></p>
  </section>
  <section>
    <h2>Tool Calls</h2>
    <table>
      <thead><tr><th>Tool</th><th>Status</th><th>Reasons</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </section>
  <section>
    <h2>Final Verdict</h2>
    <pre>{_esc(json.dumps(final_verdict, indent=2, ensure_ascii=False, sort_keys=True))}</pre>
  </section>
</main>
</body>
</html>
"""
    (run_path / "progress_panel.html").write_text(html_text, encoding="utf-8")
    summary = {
        "schema_version": "codex_entry_run_summary.v1",
        "target_name": target_input.get("target_name"),
        "case_id": preflight.get("case_id") or workflow_plan.get("case_id"),
        "preflight_accepted": bool(preflight.get("accepted")),
        "recommended_strategy": workflow_plan.get("recommended_strategy"),
        "tool_call_count": len(tool_calls),
        "final_verdict": final_verdict.get("verdict"),
        "solved": bool(final_verdict.get("solved")),
    }
    (run_path / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _esc(value: str) -> str:
    return html.escape(str(value or ""), quote=True)
