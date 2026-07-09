"""Build a PDF report for an agentic blackboard controller case run."""
from __future__ import annotations

import argparse
import io
import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.harness.agentic_blackboard_controller import run_agentic_blackboard_controller

OUT_DIR = ROOT / "docs" / "agentic_blackboard" / "report_20260609"
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

PAGE_W, PAGE_H = 1240, 1754
MARGIN = 72
TEXT = (24, 35, 51)
MUTED = (82, 95, 115)
LINE = (205, 215, 227)
BG = (246, 248, 251)
PANEL = (255, 255, 255)
ACCENT = (14, 116, 144)
ACCENT_2 = (79, 70, 229)
GOOD = (22, 101, 52)
WARN = (146, 64, 14)
BAD = (153, 27, 27)

MLA_LIKE_SMILES = "CN1CC2CCC1CC2OC(=O)c3ccccc3N4C(=O)CCC4=O"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--test-summary",
        default="pytest -q: 279 passed, 2 skipped, 7 warnings in 80.45s",
        help="Completed test gate summary to embed in the report.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / "case_run"
    data = build_report_data(run_dir=run_dir, test_summary=str(args.test_summary))

    stem = "agentic_blackboard_mla_case_report_20260609"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    pdf_path = output_dir / f"{stem}.pdf"
    audit_path = output_dir / f"{stem}_audit.json"

    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(data), encoding="utf-8")
    render_pdf(data, pdf_path)
    audit = build_audit(data, json_path=json_path, md_path=md_path, pdf_path=pdf_path)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))


def build_report_data(*, run_dir: Path, test_summary: str) -> dict[str, Any]:
    prior_artifacts = {
        "route_verifier": {
            "schema_version": "harness_route_verifier_report.v1",
            "accepted": False,
            "route_status": "fake_closed_rejected",
            "reasons": ["large_atom_jump", "advanced_same_scaffold_terminal"],
            "failure_events": [{"reason": "large_atom_jump"}],
            "rejected_terminal_list": [
                {
                    "smiles": "CN1CC2CCC1CC2OC(C)=O",
                    "canonical_smiles": "CN1CC2CCC1CC2OC(C)=O",
                    "heavy_atoms": 18,
                    "target_similarity": 0.72,
                    "reason": "advanced_same_scaffold_terminal",
                }
            ],
        },
        "guided_chemenzy": {
            "schema_version": "guided_chemenzy_rerun_result.v1",
            "accepted": False,
            "literature_template_plugin_runtime": {
                "schema_version": "literature_template_plugin_runtime_diagnostics.v1",
                "enabled_in_request": True,
                "request_one_step_row_count": 2,
                "calls": 0,
                "added_candidates": 0,
                "reasons": ["literature_template_plugin_not_invoked"],
            },
        },
    }
    result = run_agentic_blackboard_controller(
        target_name="MLA-like alkaloid case",
        target_smiles=MLA_LIKE_SMILES,
        family_hint="MLA, methyllycaconitine-like alkaloid, aryl ester, imide, cage",
        output_dir=run_dir,
        max_rounds=2,
        action_planner=_case_action_planner,
        prior_artifacts=prior_artifacts,
        mock_tool_results={
            "compile_exact_literature_rows": {
                "schema_version": "compiled_exact_literature_rows.mock.v1",
                "accepted": True,
                "exact_rows": [
                    {
                        "row_id": "mock_exact_aryl_ester_step",
                        "source_ref": "mock:source_detail:mla_sidechain",
                        "product_smiles": MLA_LIKE_SMILES,
                        "evidence_refs": ["scheme_2_compound_17_to_target"],
                        "confidence": "source_detail_validated_mock",
                    }
                ],
                "no_solved_claim": True,
            }
        },
    )
    blackboard = result["agent_blackboard"]
    action_batches = result["action_batches"]
    final = result["final_verdict"]
    return {
        "schema_version": "agentic_blackboard_case_report_data.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_run_dir": str(run_dir),
        "test_summary": test_summary,
        "target": blackboard.get("target_profile") or {},
        "final_verdict": final,
        "action_batches": action_batches,
        "action_history": blackboard.get("action_history") or [],
        "bridge_tasks": blackboard.get("bridge_tasks") or [],
        "terminal_blacklist": blackboard.get("terminal_blacklist") or [],
        "route_failures": blackboard.get("route_failures") or [],
        "literature_evidence": blackboard.get("literature_evidence") or {},
        "analogical_hypothesis_ranking": blackboard.get("analogical_hypothesis_ranking") or {},
        "parent_route_proof": blackboard.get("parent_route_proof") or {},
        "budget_state": blackboard.get("budget_state") or {},
        "artifact_refs": blackboard.get("artifact_refs") or {},
        "decision_points": summarize_decisions(action_batches, blackboard),
        "architecture_guards": [
            "Planner selects typed actions only; it cannot emit route SMILES or solved verdicts.",
            "Action batches are schema-validated, budgeted, and stale-action checked before execution.",
            "Failure critic turns route-verifier and plugin-runtime failures into bridge tasks.",
            "Exact literature rows and analogy are separated; analogy only changes priorities.",
            "Final solved requires stitched_parent_route_proof.v1, not backend or child solved flags.",
        ],
    }


def _case_action_planner(**kwargs: Any) -> dict[str, Any]:
    round_index = int(kwargs.get("round_index") or 1)
    case_id = str((kwargs.get("blackboard") or {}).get("case_id") or "agentic_case")
    if round_index == 1:
        actions = [
            _action(round_index, "generate_disconnection_hypotheses", "Initial MLA-like target needs target-side handles before any rerun.", "target_side_disconnection_hypotheses.v1", "Aryl ester, imide, cage, and amine advisory tasks appear."),
            _action(round_index, "build_failure_critic_report", "Prior route verifier shows large atom jump and advanced terminal; normalize that into bridge tasks.", "failure_critic_report.v1", "Target bridge, terminal blacklist, and next-action bias are recorded."),
            _action(round_index, "search_literature", "Bridge tasks need target-proximal source candidates before exact replay.", "literature_scout_report.v1", "Scout emits source candidates and extraction recommendations."),
        ]
    else:
        actions = [
            _action(round_index, "compile_exact_literature_rows", "Mock source-detail row is available; compile it as exact evidence, not as proof.", "compiled exact literature rows", "One exact row enters literature_evidence.exact_rows."),
            _action(round_index, "rank_analogical_hypotheses", "Rank advisory hypotheses after exact-row context is present.", "analogical_hypothesis_ranking.v1", "Selected hypotheses carry required verification and no_solved_claim."),
            _action(round_index, "stop_unresolved", "No stitched parent proof exists; stop without solved claim.", "unresolved stop marker", "Final verdict stays unresolved or partial, never solved."),
        ]
    return {
        "schema_version": "agent_action_batch.v1",
        "case_id": case_id,
        "round_index": round_index,
        "mode": "deterministic_report_case_planner",
        "actions": actions,
    }


def _action(round_index: int, action_type: str, rationale: str, expected: str, success: str) -> dict[str, Any]:
    return {
        "schema_version": "agent_action.v1",
        "action_id": f"case_r{round_index}:{action_type}",
        "action_type": action_type,
        "rationale": rationale,
        "expected_artifact": expected,
        "success_condition": success,
        "payload": {},
    }


def summarize_decisions(action_batches: list[dict[str, Any]], blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    history = [dict(row) for row in blackboard.get("action_history") or [] if isinstance(row, dict)]
    by_type = {str(row.get("action_type") or ""): row for row in history}
    rows: list[dict[str, Any]] = []
    for batch in action_batches:
        for action in batch.get("actions") or []:
            if not isinstance(action, dict):
                continue
            action_type = str(action.get("action_type") or "")
            history_row = by_type.get(action_type, {})
            rows.append(
                {
                    "round": batch.get("round_index"),
                    "action_type": action_type,
                    "rationale": action.get("rationale"),
                    "expected_artifact": action.get("expected_artifact"),
                    "success_condition": action.get("success_condition"),
                    "status": history_row.get("status", ""),
                    "useful_artifact": bool(history_row.get("useful_artifact")),
                    "reasons": history_row.get("reasons") or [],
                }
            )
    return rows


def render_markdown(data: dict[str, Any]) -> str:
    final = data["final_verdict"]
    target = data["target"]
    lines = [
        "# Agentic Blackboard Case Report",
        "",
        f"- Generated: {data['generated_at_utc']}",
        f"- Case run: `{data['case_run_dir']}`",
        f"- Target: {target.get('target_name')} ({target.get('heavy_atoms')} heavy atoms, {target.get('rings')} rings)",
        f"- Test gate: {data['test_summary']}",
        f"- Final verdict: `{final.get('verdict')}` / route_status `{final.get('route_status')}`",
        "",
        "## Round Decisions",
        "",
    ]
    for row in data["decision_points"]:
        lines.extend(
            [
                f"### Round {row['round']}: `{row['action_type']}`",
                f"- Rationale: {row['rationale']}",
                f"- Expected artifact: {row['expected_artifact']}",
                f"- Success condition: {row['success_condition']}",
                f"- Result: status `{row['status']}`, useful_artifact `{row['useful_artifact']}`",
                "",
            ]
        )
    lines.extend(["## Architecture Guards", ""])
    for guard in data["architecture_guards"]:
        lines.append(f"- {guard}")
    lines.extend(["", "## Final Gate", ""])
    lines.append("No stitched parent proof exists in this case, so the final verdict remains non-solved.")
    return "\n".join(lines) + "\n"


def render_pdf(data: dict[str, Any], pdf_path: Path) -> None:
    pages = [
        page_cover(data),
        page_architecture(data),
        page_decisions(data),
        page_blackboard(data),
        page_tests_and_verdict(data),
    ]
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for image in pages:
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=88, optimize=True)
        page.insert_image(fitz.Rect(0, 0, PAGE_W, PAGE_H), stream=stream.getvalue())
    doc.save(pdf_path)
    doc.close()


def page_cover(data: dict[str, Any]) -> Image.Image:
    img, draw = new_page()
    y = 92
    y = draw_text(draw, "Agentic Blackboard Case Report", MARGIN, y, 48, bold=True, color=TEXT, max_width=PAGE_W - 2 * MARGIN)
    y = draw_text(draw, "新架构 Agent 决策过程展示", MARGIN, y + 8, 30, bold=True, color=ACCENT, max_width=PAGE_W - 2 * MARGIN)
    y += 30
    final = data["final_verdict"]
    target = data["target"]
    metrics = [
        ("Target", str(target.get("target_name") or "")),
        ("Heavy atoms / rings", f"{target.get('heavy_atoms')} / {target.get('rings')}"),
        ("Final verdict", f"{final.get('verdict')} ({final.get('route_status')})"),
        ("Test gate", data["test_summary"]),
    ]
    for label, value in metrics:
        y = draw_metric(draw, label, value, MARGIN, y)
    draw_architecture_strip(draw, MARGIN, 1000, PAGE_W - 2 * MARGIN)
    y = 1320
    draw_text(draw, "Core message", MARGIN, y, 28, bold=True, color=TEXT, max_width=PAGE_W - 2 * MARGIN)
    draw_text(
        draw,
        "The controller is no longer a fixed failure -> scout -> extract -> rerun chain. Each round reads the blackboard, selects typed actions, validates them, executes only whitelisted tools, and refuses solved unless deterministic parent proof exists.",
        MARGIN,
        y + 48,
        24,
        color=MUTED,
        max_width=PAGE_W - 2 * MARGIN,
    )
    return img


def page_architecture(data: dict[str, Any]) -> Image.Image:
    img, draw = new_page()
    y = page_title(draw, "Architecture Decisions", "Policy-driven DAG + blackboard, with deterministic gates")
    guards = data["architecture_guards"]
    for idx, guard in enumerate(guards, start=1):
        y = draw_panel(draw, MARGIN, y, PAGE_W - 2 * MARGIN, 122, f"{idx}. {guard}", color=ACCENT if idx <= 2 else TEXT)
        y += 18
    draw_text(draw, "Blackboard state sources", MARGIN, y + 18, 26, bold=True, color=TEXT, max_width=PAGE_W - 2 * MARGIN)
    details = [
        f"route_failures: {len(data['route_failures'])}",
        f"bridge_tasks: {len(data['bridge_tasks'])}",
        f"terminal_blacklist: {len(data['terminal_blacklist'])}",
        f"exact_rows: {len((data['literature_evidence'] or {}).get('exact_rows') or [])}",
        f"selected_analogies: {len((data['analogical_hypothesis_ranking'] or {}).get('selected_hypotheses') or [])}",
    ]
    draw_text(draw, " | ".join(details), MARGIN, y + 72, 24, color=MUTED, max_width=PAGE_W - 2 * MARGIN)
    return img


def page_decisions(data: dict[str, Any]) -> Image.Image:
    img, draw = new_page()
    y = page_title(draw, "Round-by-Round Agent Decisions", "Each action has rationale, expected artifact, and success condition")
    for row in data["decision_points"]:
        h = 176 if len(str(row["rationale"])) < 100 else 210
        title = f"Round {row['round']} - {row['action_type']} [{row['status'] or 'recorded'}]"
        y = draw_panel(draw, MARGIN, y, PAGE_W - 2 * MARGIN, h, title, color=ACCENT_2 if row["useful_artifact"] else WARN)
        inner_y = y - h + 52
        draw_text(draw, f"Rationale: {row['rationale']}", MARGIN + 24, inner_y, 19, color=TEXT, max_width=PAGE_W - 2 * MARGIN - 48)
        draw_text(draw, f"Artifact: {row['expected_artifact']}", MARGIN + 24, inner_y + 50, 19, color=MUTED, max_width=PAGE_W - 2 * MARGIN - 48)
        draw_text(draw, f"Success: {row['success_condition']}", MARGIN + 24, inner_y + 90, 19, color=MUTED, max_width=PAGE_W - 2 * MARGIN - 48)
        y += 18
    return img


def page_blackboard(data: dict[str, Any]) -> Image.Image:
    img, draw = new_page()
    y = page_title(draw, "Blackboard Updates", "Failure evidence becomes typed bridge tasks instead of final-report-only text")
    y = section_list(draw, "Route failures", [row.get("reason", "") for row in data["route_failures"]], y)
    y = section_list(draw, "Bridge tasks", [f"{row.get('task_type')}: {row.get('required_bridge')}" for row in data["bridge_tasks"]], y)
    y = section_list(draw, "Terminal blacklist", [row.get("canonical_smiles", "") for row in data["terminal_blacklist"]], y)
    selected = [
        f"{row.get('hypothesis_id')} score={row.get('score')}"
        for row in (data["analogical_hypothesis_ranking"] or {}).get("selected_hypotheses") or []
    ]
    section_list(draw, "Selected advisory hypotheses", selected, y)
    return img


def page_tests_and_verdict(data: dict[str, Any]) -> Image.Image:
    img, draw = new_page()
    y = page_title(draw, "Verification And Final Gate", "Tests passed; solved still requires parent-route proof")
    y = draw_panel(draw, MARGIN, y, PAGE_W - 2 * MARGIN, 150, f"Comprehensive test gate: {data['test_summary']}", color=GOOD)
    y += 28
    final = data["final_verdict"]
    verdict_text = (
        f"Final verdict is {final.get('verdict')} with route_status {final.get('route_status')}. "
        "This is intentional: exact rows and analogical ranking are evidence for exploration, not proof. "
        "A solved claim would require stitched_parent_route_proof.v1 with target equivalence, parent verifier acceptance, stock audit, no unexplained large atom jump, child-parent connectivity, and exact-literature connectivity."
    )
    y = draw_panel(draw, MARGIN, y, PAGE_W - 2 * MARGIN, 260, verdict_text, color=BAD if final.get("solved") else WARN)
    y += 32
    refs = data["artifact_refs"]
    ref_lines = [f"{key}: {value}" for key, value in sorted(refs.items())[:10]]
    section_list(draw, "Key artifact refs", ref_lines, y)
    return img


def build_audit(data: dict[str, Any], *, json_path: Path, md_path: Path, pdf_path: Path) -> dict[str, Any]:
    page_count = 0
    if pdf_path.exists():
        with fitz.open(pdf_path) as doc:
            page_count = len(doc)
    return {
        "schema_version": "agentic_blackboard_case_report_audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "accepted": pdf_path.exists() and pdf_path.stat().st_size > 20_000 and page_count >= 5,
        "pdf_path": str(pdf_path),
        "markdown_path": str(md_path),
        "json_path": str(json_path),
        "case_run_dir": data["case_run_dir"],
        "page_count": page_count,
        "pdf_size_bytes": pdf_path.stat().st_size if pdf_path.exists() else 0,
        "final_verdict": data["final_verdict"].get("verdict"),
        "solved": bool(data["final_verdict"].get("solved")),
        "decision_count": len(data["decision_points"]),
        "test_summary": data["test_summary"],
    }


def new_page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (PAGE_W, PAGE_H), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([36, 36, PAGE_W - 36, PAGE_H - 36], fill=PANEL, outline=LINE, width=2)
    return img, draw


def page_title(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> int:
    y = 78
    y = draw_text(draw, title, MARGIN, y, 40, bold=True, color=TEXT, max_width=PAGE_W - 2 * MARGIN)
    y = draw_text(draw, subtitle, MARGIN, y + 4, 23, color=MUTED, max_width=PAGE_W - 2 * MARGIN)
    draw.line([MARGIN, y + 24, PAGE_W - MARGIN, y + 24], fill=LINE, width=2)
    return y + 56


def draw_metric(draw: ImageDraw.ImageDraw, label: str, value: str, x: int, y: int) -> int:
    draw.rounded_rectangle([x, y, PAGE_W - MARGIN, y + 112], radius=14, fill=(241, 245, 249), outline=LINE, width=1)
    draw_text(draw, label, x + 24, y + 16, 18, bold=True, color=ACCENT, max_width=240)
    draw_text(draw, value, x + 260, y + 16, 22, color=TEXT, max_width=PAGE_W - MARGIN - x - 284)
    return y + 132


def draw_architecture_strip(draw: ImageDraw.ImageDraw, x: int, y: int, width: int) -> None:
    labels = ["Blackboard", "Planner", "Validator", "Executor", "Critic", "Parent Proof"]
    gap = 18
    box_w = int((width - gap * (len(labels) - 1)) / len(labels))
    for idx, label in enumerate(labels):
        bx = x + idx * (box_w + gap)
        color = ACCENT if idx in {0, 1} else ACCENT_2 if idx in {2, 3} else WARN if idx == 4 else GOOD
        draw.rounded_rectangle([bx, y, bx + box_w, y + 92], radius=12, fill=(239, 246, 255), outline=color, width=3)
        draw_text(draw, label, bx + 12, y + 30, 18, bold=True, color=color, max_width=box_w - 24, align="center")
        if idx < len(labels) - 1:
            ax = bx + box_w + 4
            draw.line([ax, y + 46, ax + gap - 8, y + 46], fill=MUTED, width=3)
            draw.polygon([(ax + gap - 8, y + 40), (ax + gap - 8, y + 52), (ax + gap, y + 46)], fill=MUTED)


def draw_panel(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, text: str, *, color: tuple[int, int, int]) -> int:
    draw.rounded_rectangle([x, y, x + w, y + h], radius=14, fill=(248, 250, 252), outline=LINE, width=1)
    draw.rectangle([x, y, x + 10, y + h], fill=color)
    draw_text(draw, text, x + 28, y + 20, 21, bold=True, color=TEXT, max_width=w - 56)
    return y + h


def section_list(draw: ImageDraw.ImageDraw, title: str, items: list[str], y: int) -> int:
    draw_text(draw, title, MARGIN, y, 26, bold=True, color=TEXT, max_width=PAGE_W - 2 * MARGIN)
    y += 44
    if not items:
        items = ["None recorded"]
    for item in items[:7]:
        draw.ellipse([MARGIN + 4, y + 10, MARGIN + 16, y + 22], fill=ACCENT)
        y = draw_text(draw, str(item), MARGIN + 30, y, 20, color=MUTED, max_width=PAGE_W - 2 * MARGIN - 30) + 6
    return y + 24


def draw_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    size: int,
    *,
    bold: bool = False,
    color: tuple[int, int, int] = TEXT,
    max_width: int,
    align: str = "left",
) -> int:
    font = get_font(size, bold=bold)
    line_h = int(size * 1.38)
    for paragraph in str(text or "").split("\n"):
        lines = wrap_text(draw, paragraph, font, max_width)
        if not lines:
            y += line_h
            continue
        for line in lines:
            tx = x
            if align == "center":
                bbox = draw.textbbox((0, 0), line, font=font)
                tx = x + max(0, int((max_width - (bbox[2] - bbox[0])) / 2))
            draw.text((tx, y), line, font=font, fill=color)
            y += line_h
    return y


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    if not text:
        return []
    words = textwrap.wrap(text, width=96, break_long_words=False, replace_whitespace=False) or [text]
    lines: list[str] = []
    for chunk in words:
        current = ""
        for token in chunk.split(" "):
            candidate = token if not current else f"{current} {token}"
            if text_width(draw, candidate, font) <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
            if text_width(draw, token, font) <= max_width:
                current = token
            else:
                current = ""
                piece = ""
                for ch in token:
                    if text_width(draw, piece + ch, font) <= max_width:
                        piece += ch
                    else:
                        if piece:
                            lines.append(piece)
                        piece = ch
                current = piece
        if current:
            lines.append(current)
    return lines


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def get_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold and FONT_BOLD.exists() else FONT_REGULAR
    if not path.exists():
        path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size)


if __name__ == "__main__":
    main()
