"""Build a PDF report for the current bufotalin canonical full-flow run."""
from __future__ import annotations

import argparse
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw


RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / "results" / "shared" / "bufotalin_canonical_real_tools_20260607_154411"
DEFAULT_PLANNER_REJECT_DIR = ROOT / "results" / "shared" / "bufotalin_canonical_real_fullflow_20260607_154236"
OUT_DIR = ROOT / "docs" / "bufotalin" / "report_20260607"
ASSET_DIR = OUT_DIR / "assets" / "current_fullflow"

FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

PAGE_W, PAGE_H = 1240, 1754
MARGIN = 72
TEXT = (21, 30, 43)
MUTED = (80, 92, 110)
LINE = (203, 213, 225)
BG = (246, 248, 251)
PANEL = (255, 255, 255)
ACCENT = (15, 118, 110)
WARN = (154, 91, 0)
BAD = (164, 0, 0)
GOOD = (22, 101, 52)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--planner-reject-dir", default=str(DEFAULT_PLANNER_REJECT_DIR))
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    planner_reject_dir = Path(args.planner_reject_dir).resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    data = build_report_data(run_dir, planner_reject_dir)
    data_path = OUT_DIR / "bufotalin_current_fullflow_process_20260607.json"
    md_path = OUT_DIR / "bufotalin_current_fullflow_process_20260607.md"
    pdf_path = OUT_DIR / "bufotalin_current_fullflow_process_20260607.pdf"
    data_path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(data), encoding="utf-8")
    render_pdf(data, pdf_path)
    audit = {
        "schema_version": "bufotalin_current_fullflow_process_report_audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "accepted": pdf_path.exists() and pdf_path.stat().st_size > 50_000,
        "run_dir": str(run_dir),
        "planner_reject_dir": str(planner_reject_dir) if planner_reject_dir.exists() else "",
        "pdf_path": str(pdf_path),
        "markdown_path": str(md_path),
        "data_path": str(data_path),
        "final_verdict": data["final"]["verdict"],
        "solved": data["final"]["solved"],
        "tool_call_count": len(data["tool_calls"]),
        "pdf_request_count": len(data["pdf_requests"]),
    }
    audit_path = OUT_DIR / "bufotalin_current_fullflow_process_report_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))


def build_report_data(run_dir: Path, planner_reject_dir: Path) -> dict[str, Any]:
    preflight = read_json(run_dir / "preflight.json")
    final = read_json(run_dir / "final_verdict.json")
    route_verifier = read_json(run_dir / "route_verifier_report.json")
    route_audit = read_json(run_dir / "route_audit.json")
    open_audit = read_json(run_dir / "open_structure_research" / "open_agent_audit.json")
    downstream = read_json(run_dir / "open_structure_research" / "downstream_consumables.json")
    literature_sources = read_json(run_dir / "open_structure_research" / "evidence" / "literature_sources.json")
    route_expansion = tool_output(run_dir, "run_route_expansion_subgoal_search")
    guided = tool_output(run_dir, "run_guided_chemenzy_rerun")
    native = tool_output(run_dir, "run_chemenzy")
    smiles_first = tool_output(run_dir, "run_smiles_first_literature_workflow")
    open_research = tool_output(run_dir, "run_open_structure_research_agent")
    self_evo = tool_output(run_dir, "run_self_evo_replay_gate")

    planner_reject = read_json(planner_reject_dir / "final_verdict.json") if planner_reject_dir.exists() else {}
    tool_calls = read_tool_calls(run_dir)
    pdf_requests = read_jsonl(run_dir / "open_structure_research" / "evidence" / "local_pdf_proxy" / "pdf_requests.jsonl")
    local_pdf = ROOT / "1-s2.0-S0040402025001668-main.pdf"

    route_expansion_result = route_expansion.get("result") or {}
    subgoals = list(route_expansion_result.get("subgoals") or [])
    source_relation_counts = count_by(literature_sources.get("sources") or [], "source_relation")

    return {
        "schema_version": "bufotalin_current_fullflow_process_report_data.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "planner_reject_dir": str(planner_reject_dir) if planner_reject_dir.exists() else "",
        "planner_reject": {
            "verdict": planner_reject.get("verdict", ""),
            "reasons": planner_reject.get("reasons", []),
        },
        "target": {
            "name": "bufotalin",
            "input_smiles": (preflight.get("target_profile") or {}).get("input_smiles", ""),
            "isomeric_smiles": preflight.get("isomeric_smiles", ""),
            "formula": (preflight.get("target_profile") or {}).get("formula", ""),
            "exact_mw": (preflight.get("target_profile") or {}).get("exact_mw", ""),
            "inchi_key": preflight.get("inchi_key", ""),
            "heavy_atoms": (preflight.get("target_profile") or {}).get("heavy_atoms", ""),
            "rings": (preflight.get("target_profile") or {}).get("rings", ""),
            "stereocenters": (preflight.get("target_profile") or {}).get("stereocenters", ""),
            "risk_flags": preflight.get("initial_risk_flags", []),
        },
        "final": {
            "verdict": final.get("verdict", ""),
            "route_status": final.get("route_status", ""),
            "solved": bool(final.get("solved")),
            "stock_audit_passed": bool(final.get("stock_audit_passed")),
            "reasons": final.get("reasons", []),
        },
        "native_chemenzy": {
            "elapsed_s": native.get("elapsed_s"),
            "accepted": native.get("accepted"),
            "depth_attempts": ((native.get("result") or {}).get("result") or {}).get("depth_attempts", []),
            "route_count": route_verifier.get("accepted_route_count", 0) + route_verifier.get("rejected_route_count", 0),
            "verifier_accepted": route_verifier.get("accepted"),
            "accepted_route_count": route_verifier.get("accepted_route_count"),
            "rejected_route_count": route_verifier.get("rejected_route_count"),
            "route_status": route_verifier.get("route_status"),
            "reasons": route_verifier.get("reasons", []),
        },
        "route_audit": {
            "route_status": route_audit.get("route_status", ""),
            "stock_audit_passed": route_audit.get("stock_audit_passed", False),
            "reasons": route_audit.get("reasons", []),
        },
        "smiles_first": {
            "elapsed_s": smiles_first.get("elapsed_s"),
            "accepted": smiles_first.get("accepted"),
            "route_status": (((smiles_first.get("result") or {}).get("validation") or {}).get("route_status")),
        },
        "open_research": {
            "elapsed_s": open_research.get("elapsed_s"),
            "tool_status": open_research.get("status", ""),
            "tool_reasons": open_research.get("reasons", []),
            "final_status": open_audit.get("final_status", ""),
            "solved": bool(open_audit.get("solved")),
            "production_kb_promotion": bool(open_audit.get("production_kb_promotion")),
            "sources": len(literature_sources.get("sources") or []),
            "excluded_sources": len(literature_sources.get("excluded_sources") or []),
            "search_log": len(literature_sources.get("search_log") or []),
            "source_relation_counts": source_relation_counts,
            "guided_rerun_requests": len(downstream.get("guided_rerun_requests") or []),
            "literature_template_cards": len(downstream.get("literature_template_cards") or []),
            "executable_template_extraction_tasks": len(downstream.get("executable_template_extraction_tasks") or []),
            "source_detail_route_steps": len(downstream.get("source_detail_route_steps") or []),
            "route_expansion_tasks": len(downstream.get("route_expansion_tasks") or []),
            "evolution_candidates": len(downstream.get("evolution_candidates") or []),
        },
        "pdf_requests": pdf_requests,
        "local_pdf": {
            "path": str(local_pdf),
            "exists": local_pdf.exists(),
            "size_bytes": local_pdf.stat().st_size if local_pdf.exists() else 0,
        },
        "guided_chemenzy": {
            "elapsed_s": guided.get("elapsed_s"),
            "accepted": guided.get("accepted"),
            "reasons": guided.get("reasons", []),
        },
        "route_expansion": {
            "elapsed_s": route_expansion.get("elapsed_s"),
            "accepted": route_expansion_result.get("accepted"),
            "status": route_expansion_result.get("status"),
            "solved": route_expansion_result.get("solved"),
            "subgoal_count": route_expansion_result.get("subgoal_count"),
            "accepted_subgoal_count": route_expansion_result.get("accepted_subgoal_count"),
            "rejected_subgoal_count": route_expansion_result.get("rejected_subgoal_count"),
            "subgoals": [
                {
                    "name": ((row.get("subgoal") or {}).get("name")),
                    "accepted": row.get("accepted"),
                    "solved": row.get("solved"),
                    "route_status": row.get("route_status"),
                    "route_count": row.get("route_count"),
                }
                for row in subgoals
            ],
        },
        "self_evo": {
            "elapsed_s": self_evo.get("elapsed_s"),
            "accepted": self_evo.get("accepted"),
            "reasons": self_evo.get("reasons", []),
        },
        "tool_calls": [
            {
                "tool_name": row.get("tool_name"),
                "status": row.get("status"),
                "elapsed_s": row.get("elapsed_s"),
                "reasons": row.get("reasons", []),
            }
            for row in tool_calls
        ],
        "key_artifacts": {
            "final_verdict": str(run_dir / "final_verdict.json"),
            "tool_calls": str(run_dir / "tool_calls.jsonl"),
            "route_verifier": str(run_dir / "route_verifier_report.json"),
            "open_research": str(run_dir / "open_structure_research"),
            "pdf_queue": str(run_dir / "open_structure_research" / "evidence" / "local_pdf_proxy" / "pdf_requests.jsonl"),
            "progress_panel": str(run_dir / "progress_panel.html"),
        },
    }


def render_markdown(data: dict[str, Any]) -> str:
    final = data["final"]
    lines = [
        "# Bufotalin 当前全流程运行报告",
        "",
        f"- 生成时间: {data['generated_at_utc']}",
        f"- run_dir: `{data['run_dir']}`",
        f"- final verdict: `{final['verdict']}`",
        f"- solved: `{final['solved']}`",
        "",
        "## 一个 SMILES 进入 AutoPlanner 后经历什么",
        "",
    ]
    for idx, item in enumerate(process_steps(), start=1):
        lines.append(f"{idx}. {item}")
    lines.extend([
        "",
        "## Bufotalin 本次真实运行结论",
        "",
        f"- Native ChemEnzy: elapsed `{data['native_chemenzy']['elapsed_s']}` s, raw route count `{data['native_chemenzy']['route_count']}`, verifier accepted `{data['native_chemenzy']['accepted_route_count']}`.",
        f"- Native verifier reasons: `{', '.join(data['native_chemenzy']['reasons'])}`.",
        f"- Open research: `{data['open_research']['final_status']}`, sources `{data['open_research']['sources']}`, PDF requests `{len(data['pdf_requests'])}`.",
        f"- Guided ChemEnzy: elapsed `{data['guided_chemenzy']['elapsed_s']}` s, reasons `{', '.join(data['guided_chemenzy']['reasons'])}`.",
        f"- Route expansion: status `{data['route_expansion']['status']}`, accepted subgoals `{data['route_expansion']['accepted_subgoal_count']}` / `{data['route_expansion']['subgoal_count']}`.",
        f"- Final: `{final['verdict']}`, reasons `{', '.join(final['reasons'])}`.",
        "",
        "## PDF / Source Access",
        "",
        f"- 已下载本地 PDF: `{data['local_pdf']['path']}`, exists=`{data['local_pdf']['exists']}`.",
    ])
    for row in data["pdf_requests"]:
        lines.append(f"- queued PDF fallback: `{row.get('doi')}` scope=`{row.get('content_scope')}` status=`{row.get('status')}`")
    return "\n".join(lines) + "\n"


def render_pdf(data: dict[str, Any], pdf_path: Path) -> None:
    pages = [
        draw_cover_page(data),
        draw_pipeline_page(data),
        draw_bufotalin_run_page(data),
        draw_source_pdf_page(data),
        draw_artifacts_page(data),
    ]
    doc = fitz.open()
    for image in pages:
        path = ASSET_DIR / f"page_{len(doc) + 1:02d}.png"
        image.save(path, quality=95)
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        page.insert_image(page.rect, filename=str(path))
    doc.set_metadata({
        "title": "Bufotalin current AutoPlanner full-flow process",
        "author": "AutoPlanner Codex harness",
        "subject": "SMILES intake to final verdict with real ChemEnzy and retrieval flow",
    })
    doc.save(pdf_path, deflate=True, garbage=4)
    doc.close()


def draw_cover_page(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base_canvas("Bufotalin 当前全流程报告", "SMILES intake -> ChemEnzy -> verifier -> literature/PDF -> guided rerun -> final verdict")
    target = data["target"]
    y = 190
    draw_card(draw, MARGIN, y, 520, 520, "Target profile", [
        f"name: bufotalin",
        f"formula: {target['formula']}",
        f"exact MW: {target['exact_mw']}",
        f"heavy atoms: {target['heavy_atoms']}",
        f"rings: {target['rings']}",
        f"stereocenters: {target['stereocenters']}",
        f"InChIKey: {target['inchi_key']}",
        f"risk flags: {', '.join(target['risk_flags'])}",
    ])
    draw_molecule(canvas, draw, target["isomeric_smiles"], 650, y + 20, 450, 360)
    draw_card(draw, 650, y + 390, 450, 230, "Final verdict", [
        f"verdict: {data['final']['verdict']}",
        f"route_status: {data['final']['route_status']}",
        f"solved: {data['final']['solved']}",
        f"stock_audit_passed: {data['final']['stock_audit_passed']}",
    ], status="bad")
    y = 860
    draw_card(draw, MARGIN, y, PAGE_W - 2 * MARGIN, 310, "本次运行边界", [
        "Live planner was attempted first and rejected by schema validation: invalid_run_semantics.",
        "The completed full tool run used the deterministic valid planner, but ChemEnzy, retrieval, open research, guided rerun, route expansion, and PDF queue were real tool executions.",
        "No source full text, credentials, cookies, raw LLM reaction injection, or production KB writes are accepted as route evidence.",
    ])
    draw_footer(draw, data)
    return canvas


def draw_pipeline_page(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base_canvas("现在一个 SMILES 进来经历什么", "Canonical controller contract")
    boxes = [
        ("1. Target intake", "Parse SMILES, canonical/isomeric SMILES, formula, rings, stereocenters, risk flags."),
        ("2. Workflow plan", "Plan must validate strategy and run_semantics before tools execute."),
        ("3. Native ChemEnzy", "Route generator only; raw solved is not final solved."),
        ("4. Route verifier", "Checks target identity, stock closure, hidden non-stock leaves, atom jumps, advanced terminals."),
        ("5. Frontier/literature gate", "If unresolved/fake/advanced, extract frontier and trigger literature/source-detail."),
        ("6. Open research/PDF", "Typed retrieval, source access status, same-scope PDF fallback queue, no credential storage."),
        ("7. Guided rerun/expansion", "Validated downstream handoff can guide ChemEnzy or child target search."),
        ("8. Self-evo staging", "Target-run memory may stage candidates; production remains blocked."),
        ("9. Final verdict", "Only deterministic bundle/verifier emits solved, partial, unresolved, or fake_closed_rejected."),
    ]
    x1, x2 = MARGIN, 650
    y = 170
    for idx, (title, body) in enumerate(boxes):
        x = x1 if idx % 2 == 0 else x2
        if idx and idx % 2 == 0:
            y += 220
        draw_card(draw, x, y, 510, 150, title, [body], status="ok" if idx in {0, 1, 3, 8} else "")
    draw_footer(draw, data)
    return canvas


def draw_bufotalin_run_page(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base_canvas("Bufotalin 真实运行结果", "ChemEnzy and downstream tools")
    native = data["native_chemenzy"]
    open_research = data["open_research"]
    route_expansion = data["route_expansion"]
    draw_card(draw, MARGIN, 170, 520, 250, "Native ChemEnzy", [
        f"elapsed_s: {native['elapsed_s']}",
        f"raw/verifier route count: {native['route_count']}",
        f"verifier accepted routes: {native['accepted_route_count']}",
        f"verifier rejected routes: {native['rejected_route_count']}",
        f"route_status: {native['route_status']}",
        f"reasons: {', '.join(native['reasons'])}",
    ], status="bad")
    draw_card(draw, 650, 170, 520, 250, "SMILES-first + literature", [
        f"route_status: {data['smiles_first']['route_status']}",
        "strategic evidence and route anchors were generated",
        "target-as-frontier is tracked separately from route-failure frontier",
    ], status="warn")
    draw_card(draw, MARGIN, 470, 520, 300, "Open research", [
        f"tool_status: {open_research['tool_status']}",
        f"final_status: {open_research['final_status']}",
        f"sources/excluded/search_log: {open_research['sources']} / {open_research['excluded_sources']} / {open_research['search_log']}",
        f"guided requests: {open_research['guided_rerun_requests']}",
        f"literature template cards: {open_research['literature_template_cards']}",
        f"extraction tasks: {open_research['executable_template_extraction_tasks']}",
        f"source_detail_route_steps: {open_research['source_detail_route_steps']}",
    ], status="warn")
    draw_card(draw, 650, 470, 520, 300, "Guided rerun + expansion", [
        f"guided ChemEnzy elapsed_s: {data['guided_chemenzy']['elapsed_s']}",
        f"guided reasons: {', '.join(data['guided_chemenzy']['reasons'])}",
        f"route expansion status: {route_expansion['status']}",
        f"accepted subgoals: {route_expansion['accepted_subgoal_count']} / {route_expansion['subgoal_count']}",
        *[
            f"{row['name']}: accepted={row['accepted']} status={row['route_status']} routes={row['route_count']}"
            for row in route_expansion["subgoals"]
        ],
    ], status="warn")
    draw_card(draw, MARGIN, 820, PAGE_W - 2 * MARGIN, 260, "Final deterministic verdict", [
        f"verdict: {data['final']['verdict']}",
        f"route_status: {data['final']['route_status']}",
        f"solved: {data['final']['solved']}",
        f"stock_audit_passed: {data['final']['stock_audit_passed']}",
        "reasons: " + ", ".join(data["final"]["reasons"]),
    ], status="bad")
    draw_footer(draw, data)
    return canvas


def draw_source_pdf_page(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base_canvas("真实检索与 PDF fallback", "Agent access first, then same-scope local PDF queue")
    open_research = data["open_research"]
    draw_card(draw, MARGIN, 170, 520, 280, "Retrieval/source triage", [
        f"sources: {open_research['sources']}",
        f"excluded_sources: {open_research['excluded_sources']}",
        f"search_log: {open_research['search_log']}",
        "relation counts: " + ", ".join(f"{k}={v}" for k, v in open_research["source_relation_counts"].items()),
    ])
    draw_card(draw, 650, 170, 520, 280, "Local downloaded PDF", [
        f"path: {data['local_pdf']['path']}",
        f"exists: {data['local_pdf']['exists']}",
        f"size_bytes: {data['local_pdf']['size_bytes']}",
        "The report records the local PDF pointer but does not store source full text.",
    ], status="ok" if data["local_pdf"]["exists"] else "warn")
    y = 510
    pdf_lines = [
        f"{row.get('doi')} | scope={row.get('content_scope')} | status={row.get('status')}"
        for row in data["pdf_requests"]
    ]
    draw_card(draw, MARGIN, y, PAGE_W - 2 * MARGIN, 300, "Queued PDF fallback requests", pdf_lines or ["No PDF requests queued."], status="warn")
    draw_card(draw, MARGIN, 860, PAGE_W - 2 * MARGIN, 230, "Source-detail policy", [
        "PDF fallback is only allowed after a same DOI/URL/source_ref and same content_scope agent-access row records metadata-only, login/paywall, or unavailable.",
        "This run queued article-scope requests only after metadata-only source access rows existed.",
        "Downloaded PDF content is not treated as route evidence until structured extraction produces source-grounded product/reactant SMILES.",
    ])
    draw_footer(draw, data)
    return canvas


def draw_artifacts_page(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base_canvas("Artifact map and known failures", "What to inspect next")
    y = 170
    artifact_lines = [f"{k}: {v}" for k, v in data["key_artifacts"].items()]
    draw_card(draw, MARGIN, y, PAGE_W - 2 * MARGIN, 390, "Key artifacts", artifact_lines)
    y += 440
    tool_lines = [
        f"{idx}. {row['tool_name']} | {row['status']} | {row['elapsed_s']}s | {', '.join(row['reasons'])}"
        for idx, row in enumerate(data["tool_calls"], start=1)
    ]
    draw_card(draw, MARGIN, y, PAGE_W - 2 * MARGIN, 470, "Tool calls", tool_lines)
    draw_card(draw, MARGIN, 1110, PAGE_W - 2 * MARGIN, 230, "Interpretation", [
        "This is a complete real tool-chain execution, but it is not a solved synthesis.",
        "The strongest validated output is partial downstream handoff plus route-expansion evidence.",
        "Blocking issues: exact source-detail route steps are still absent, open research had a boundary violation, and raw/guided ChemEnzy routes failed verifier gates.",
    ], status="bad")
    draw_footer(draw, data)
    return canvas


def base_canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    canvas = Image.new("RGB", (PAGE_W, PAGE_H), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, PAGE_W, 28), fill=ACCENT)
    draw.text((MARGIN, 58), title, font=font(44, bold=True), fill=TEXT)
    draw_wrapped(draw, subtitle, MARGIN, 118, PAGE_W - 2 * MARGIN, font(22), MUTED)
    return canvas, draw


def draw_card(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    lines: list[str],
    *,
    status: str = "",
) -> None:
    outline = LINE
    fill = PANEL
    if status == "bad":
        outline = (248, 113, 113)
        fill = (255, 247, 247)
    elif status == "warn":
        outline = (245, 158, 11)
        fill = (255, 251, 235)
    elif status == "ok":
        outline = (52, 211, 153)
        fill = (240, 253, 244)
    draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=fill, outline=outline, width=2)
    draw.text((x + 22, y + 20), title, font=font(24, bold=True), fill=TEXT)
    ty = y + 64
    for idx, line in enumerate(lines):
        used = draw_wrapped(draw, str(line), x + 22, ty, w - 44, font(18), TEXT)
        ty += used + 8
        if ty > y + h - 28 and idx < len(lines) - 1:
            draw.text((x + 22, y + h - 24), "...", font=font(18), fill=MUTED)
            break


def draw_molecule(canvas: Image.Image, draw: ImageDraw.ImageDraw, smiles: str, x: int, y: int, w: int, h: int) -> None:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=PANEL, outline=LINE, width=2)
    if mol is None:
        draw.text((x + 20, y + 20), "Invalid SMILES", font=font(20), fill=BAD)
        return
    img = Draw.MolToImage(mol, size=(w - 40, h - 70), legend="")
    canvas.paste(img.convert("RGB"), (x + 20, y + 20))
    draw.text((x + 20, y + h - 38), "RDKit depiction from preflight isomeric SMILES", font=font(16), fill=MUTED)


def draw_footer(draw: ImageDraw.ImageDraw, data: dict[str, Any]) -> None:
    draw.line((MARGIN, PAGE_H - 80, PAGE_W - MARGIN, PAGE_H - 80), fill=LINE, width=2)
    draw.text((MARGIN, PAGE_H - 58), f"Run: {data['run_dir']}", font=font(14), fill=MUTED)


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    width: int,
    font_obj: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    *,
    line_gap: int = 6,
) -> int:
    lines: list[str] = []
    for para in str(text).splitlines() or [""]:
        if not para:
            lines.append("")
            continue
        current = ""
        for token in para.split(" "):
            trial = token if not current else current + " " + token
            if text_width(draw, trial, font_obj) <= width:
                current = trial
                continue
            if current:
                lines.append(current)
            if text_width(draw, token, font_obj) <= width:
                current = token
            else:
                wrapped = textwrap.wrap(token, width=max(8, width // 18))
                lines.extend(wrapped[:-1])
                current = wrapped[-1] if wrapped else ""
        if current:
            lines.append(current)
    line_h = int(font_obj.size * 1.22) if hasattr(font_obj, "size") else 24
    ty = y
    for line in lines:
        draw.text((x, ty), line, font=font_obj, fill=fill)
        ty += line_h + line_gap
    return max(line_h, ty - y)


def text_width(draw: ImageDraw.ImageDraw, text: str, font_obj: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=font_obj)
    return int(box[2] - box[0])


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold and FONT_BOLD.exists() else FONT_REGULAR
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def process_steps() -> list[str]:
    return [
        "结构预检：解析 SMILES，生成 canonical/isomeric SMILES、InChIKey、formula、rings、stereocenters 和风险标记。",
        "工作流计划：controller/Codex 只规划工具顺序；计划必须通过 schema、strategy、run_semantics 和 raw-reaction guard。",
        "Native ChemEnzy：真实调用 ChemEnzy 作为 route generator；raw_solved 不等于 solved。",
        "Route verifier/audit：独立检查 target identity、stock closure、hidden non-stock、large atom jump、advanced same-scaffold terminal。",
        "Frontier 与 literature gate：只有 route unresolved/fake/advanced 或合法 literature-first 理由才进入文献/source-detail。",
        "Open research：先读 manifest、prefetch、source-detail pack，记录 agent_access_status/content_scope，再排本地 PDF fallback。",
        "Downstream compiler：把可靠证据编译成 guided rerun、template card、route expansion task、self-evo staging candidate。",
        "Guided rerun / route expansion：再次真实调用 ChemEnzy 或子目标搜索，但仍必须过 verifier。",
        "Final verdict：只有 deterministic artifact bundle 和 verifier 能给 solved/fake_closed/partial/unresolved 结论。",
    ]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_tool_calls(run_dir: Path) -> list[dict[str, Any]]:
    return read_jsonl(run_dir / "tool_calls.jsonl")


def tool_output(run_dir: Path, tool_name: str) -> dict[str, Any]:
    for row in read_tool_calls(run_dir):
        if row.get("tool_name") == tool_name:
            out = dict(row.get("output") or {})
            out["elapsed_s"] = row.get("elapsed_s")
            out["status"] = row.get("status")
            out["reasons"] = row.get("reasons", [])
            return out
    return {}


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "missing")
        counts[value] = counts.get(value, 0) + 1
    return counts


if __name__ == "__main__":
    main()
