"""Build an expert PDF report for the stitched bufotalin full-flow run."""
from __future__ import annotations

import argparse
import json
import math
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
DEFAULT_RUN_DIR = ROOT / "results" / "shared" / "bufotalin_stitched_fullflow_existing_pdf_20260608_0345"
OUT_DIR = ROOT / "docs" / "bufotalin" / "report_20260608"
ASSET_DIR = OUT_DIR / "assets" / "stitched_fullflow_expert"

FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

PAGE_W, PAGE_H = 1240, 1754
MARGIN = 72
TEXT = (23, 31, 42)
MUTED = (79, 91, 109)
LINE = (203, 213, 225)
BG = (246, 248, 251)
PANEL = (255, 255, 255)
ACCENT = (13, 105, 117)
ACCENT_2 = (29, 78, 216)
GOOD = (22, 101, 52)
WARN = (146, 64, 14)
BAD = (153, 27, 27)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    global ASSET_DIR
    ASSET_DIR = output_dir / "assets" / "stitched_fullflow_expert"
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    data = build_report_data(run_dir)
    stem = "bufotalin_stitched_fullflow_expert_report_20260608"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    pdf_path = output_dir / f"{stem}.pdf"
    audit_path = output_dir / f"{stem}_audit.json"

    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(data), encoding="utf-8")
    render_pdf(data, pdf_path)
    audit = build_audit(data, pdf_path=pdf_path, json_path=json_path, md_path=md_path)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))


def build_report_data(run_dir: Path) -> dict[str, Any]:
    summary = read_json(run_dir / "bufotalin_stitched_fullflow_summary.json")
    planner = read_json(run_dir / "codex_planner_run_record.json")
    target_input = read_json(run_dir / "target_input.json")
    preflight = read_json(run_dir / "preflight.json")
    native_verifier = read_json(run_dir / "route_verifier_report.json")
    pdf_evidence = read_json(run_dir / "literature_pdf_structure_evidence.json")
    visual_audit = read_json(run_dir / "visual_scheme_audit" / "visual_scheme_audit.json")
    visual_chain = read_json(run_dir / "visual_structure_chain_validation.json")
    if not visual_chain:
        visual_chain = read_json(run_dir / "literature_intermediate_chain_validation" / "visual_structure_chain_validation.json")
    source_chain = read_json(run_dir / "source_detail_chain_route" / "source_detail_route_chain_audit.json")
    subgoal_verifier = read_json(run_dir / "route_expansion_subgoals" / "01_compound_11_exact_tet2025_terminal_verifier.json")
    stitched = read_json(run_dir / "stitched_semisynthesis_route" / "stitched_semisynthesis_route.json")
    final = read_json(run_dir / "final_verdict.json")
    artifact_bundle = read_json(run_dir / "artifact_bundle_validation.json")
    guided_patched = read_json(run_dir / "guided_chemenzy_patched_tool_call.json")
    guided_verifier = read_json(run_dir / "guided_route_verifier_report.json")
    open_result = read_json(run_dir / "open_structure_research_result.json")
    tool_calls = read_tool_calls(run_dir / "tool_calls.jsonl")

    source_chain_steps = [dict(item) for item in source_chain.get("chain") or [] if isinstance(item, dict)]
    normalized_audit = dict(planner.get("normalization_audit") or {})
    planner_plan = dict(planner.get("workflow_plan") or {})
    target_profile = dict(preflight.get("target_profile") or {})
    patched_guided_result = dict((guided_patched.get("output") or {}).get("result") or {})
    patched_guided_policy = dict((patched_guided_result.get("policy") or {}))

    return {
        "schema_version": "bufotalin_stitched_fullflow_expert_report_data.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "target": {
            "name": target_input.get("target_name") or "bufotalin",
            "input_smiles": target_input.get("target_smiles") or target_profile.get("input_smiles") or "",
            "isomeric_smiles": preflight.get("isomeric_smiles") or target_input.get("target_smiles") or "",
            "formula": target_profile.get("formula") or "",
            "exact_mw": target_profile.get("exact_mw") or "",
            "heavy_atoms": target_profile.get("heavy_atoms") or "",
            "rings": target_profile.get("rings") or "",
            "stereocenters": target_profile.get("stereocenters") or "",
            "inchi_key": preflight.get("inchi_key") or "",
            "risk_flags": preflight.get("initial_risk_flags") or [],
        },
        "planner": {
            "accepted": bool(planner.get("accepted")),
            "elapsed_s": planner.get("elapsed_s"),
            "mode": "live",
            "run_semantics": normalized_audit.get("normalized_run_semantics") or planner_plan.get("run_semantics") or "",
            "run_semantics_changed": bool(normalized_audit.get("run_semantics_changed")),
            "raw_run_semantics_type": normalized_audit.get("raw_run_semantics_type") or "",
            "planned_tools": [str((item or {}).get("tool_name") or item) for item in planner_plan.get("planned_tools") or []],
            "record_path": str(run_dir / "codex_planner_run_record.json"),
        },
        "summary": summary,
        "tool_status": summary.get("tool_status") or [
            {
                "tool_name": row.get("tool_name"),
                "status": row.get("status"),
                "elapsed_s": row.get("elapsed_s"),
                "reasons": row.get("reasons") or [],
            }
            for row in tool_calls
        ],
        "native_chemenzy": {
            "elapsed_s": tool_elapsed(summary, "run_chemenzy"),
            "accepted": native_verifier.get("accepted"),
            "route_status": native_verifier.get("route_status"),
            "route_count": native_verifier.get("route_count"),
            "accepted_route_count": native_verifier.get("accepted_route_count"),
            "rejected_route_count": native_verifier.get("rejected_route_count"),
            "reasons": native_verifier.get("reasons") or [],
        },
        "open_research": {
            "status": (open_result.get("status") or tool_status(summary, "run_open_structure_research_agent")),
            "elapsed_s": tool_elapsed(summary, "run_open_structure_research_agent"),
            "reasons": open_result.get("reasons") or tool_reasons(summary, "run_open_structure_research_agent"),
            "output_dir": str(run_dir / "open_structure_research"),
        },
        "pdf_evidence": {
            "source_pdf_path": pdf_evidence.get("source_pdf_path") or "",
            "source_pdf_sha256": pdf_evidence.get("source_pdf_sha256") or "",
            "status": pdf_evidence.get("status") or "",
            "summary": pdf_evidence.get("summary") or summary.get("pdf_evidence_summary") or {},
            "rendered_pages": pdf_evidence.get("rendered_pages") or [],
            "scheme_crops": pdf_evidence.get("scheme_crops") or [],
            "manifest_path": str(run_dir / "literature_pdf_structure_evidence.json"),
        },
        "visual_audit": {
            "accepted": visual_audit.get("accepted"),
            "elapsed_s": visual_audit.get("elapsed_s"),
            "parsed_output": visual_audit.get("parsed_output") or {},
            "image_paths": visual_audit.get("image_paths") or [],
            "path": str(run_dir / "visual_scheme_audit" / "visual_scheme_audit.json"),
        },
        "visual_chain": {
            "accepted": visual_chain.get("accepted"),
            "summary": visual_chain.get("summary") or summary.get("visual_chain_summary") or {},
            "path": str(run_dir / "visual_structure_chain_validation.json"),
        },
        "source_detail_chain": {
            "accepted": source_chain.get("accepted"),
            "step_count": source_chain.get("step_count"),
            "terminal_reached": source_chain.get("terminal_reached"),
            "terminal_smiles": source_chain.get("terminal_smiles"),
            "terminal_name": source_chain.get("terminal_name"),
            "reasons": source_chain.get("reasons") or [],
            "first_steps": compact_steps(source_chain_steps[:4]),
            "last_steps": compact_steps(source_chain_steps[-4:]),
            "path": str(run_dir / "source_detail_chain_route" / "source_detail_route_chain_audit.json"),
        },
        "subgoal": {
            "accepted": subgoal_verifier.get("accepted"),
            "route_status": subgoal_verifier.get("route_status"),
            "route_count": subgoal_verifier.get("route_count"),
            "accepted_route_count": subgoal_verifier.get("accepted_route_count"),
            "rejected_route_count": subgoal_verifier.get("rejected_route_count"),
            "best_route_rank": subgoal_verifier.get("best_route_rank"),
            "reasons": subgoal_verifier.get("reasons") or [],
            "target_match": subgoal_verifier.get("target_match"),
            "target_equivalence_audit": subgoal_verifier.get("target_equivalence_audit") or {},
            "path": str(run_dir / "route_expansion_subgoals" / "01_compound_11_exact_tet2025_terminal_verifier.json"),
        },
        "stitch": {
            "accepted": stitched.get("accepted"),
            "solved": stitched.get("solved"),
            "route_status": stitched.get("route_status"),
            "stock_audit_passed": stitched.get("stock_audit_passed"),
            "combined_route": stitched.get("combined_route") or {},
            "terminal_match_audit": stitched.get("terminal_match_audit") or {},
            "target_identity_audit": (stitched.get("target") or {}).get("identity_audit") or {},
            "source_policy": stitched.get("source_policy") or {},
            "warnings": stitched.get("warnings") or [],
            "reasons": stitched.get("reasons") or [],
            "path": str(run_dir / "stitched_semisynthesis_route" / "stitched_semisynthesis_route.json"),
        },
        "guided_patched": {
            "elapsed_s": guided_patched.get("elapsed_s"),
            "status": guided_patched.get("status"),
            "record_reasons": guided_patched.get("reasons") or [],
            "result_accepted": patched_guided_result.get("accepted"),
            "route_status": patched_guided_result.get("route_status"),
            "solved": patched_guided_result.get("solved"),
            "policy_id": patched_guided_policy.get("policy_id"),
            "rerun_reason": patched_guided_policy.get("rerun_reason"),
            "verifier": {
                "accepted": guided_verifier.get("accepted"),
                "route_status": guided_verifier.get("route_status"),
                "route_count": guided_verifier.get("route_count"),
                "accepted_route_count": guided_verifier.get("accepted_route_count"),
                "reasons": guided_verifier.get("reasons") or [],
            },
            "path": str(run_dir / "guided_chemenzy_patched_tool_call.json"),
        },
        "artifact_bundle": {
            "accepted": artifact_bundle.get("accepted"),
            "reasons": artifact_bundle.get("reasons") or [],
            "path": str(run_dir / "artifact_bundle_validation.json"),
        },
        "final": {
            "verdict": final.get("verdict"),
            "solved": final.get("solved"),
            "route_status": final.get("route_status"),
            "stock_audit_passed": final.get("stock_audit_passed"),
            "reasons": final.get("reasons") or [],
            "path": str(run_dir / "final_verdict.json"),
        },
        "artifact_refs": {
            **dict(summary.get("artifact_refs") or {}),
            "patched_guided_tool_call": str(run_dir / "guided_chemenzy_patched_tool_call.json"),
            "guided_route_verifier": str(run_dir / "guided_route_verifier_report.json"),
            "artifact_bundle_validation": str(run_dir / "artifact_bundle_validation.json"),
            "summary": str(run_dir / "bufotalin_stitched_fullflow_summary.json"),
        },
    }


def compact_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "step_index": row.get("step_index"),
            "step_id": row.get("step_id"),
            "product_smiles": row.get("product_smiles"),
            "main_reactant_smiles": row.get("main_reactant_smiles"),
            "source_ref": row.get("source_ref"),
        }
        for row in steps
    ]


def build_audit(data: dict[str, Any], *, pdf_path: Path, json_path: Path, md_path: Path) -> dict[str, Any]:
    render_pages = sorted((ASSET_DIR).glob("page_*.png"))
    return {
        "schema_version": "bufotalin_stitched_fullflow_expert_report_audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "accepted": pdf_path.exists() and pdf_path.stat().st_size > 80_000 and len(render_pages) >= 8,
        "pdf_path": str(pdf_path),
        "markdown_path": str(md_path),
        "data_path": str(json_path),
        "rendered_page_count": len(render_pages),
        "pdf_size_bytes": pdf_path.stat().st_size if pdf_path.exists() else 0,
        "run_dir": data["run_dir"],
        "final_verdict": data["final"]["verdict"],
        "solved": data["final"]["solved"],
        "stitched_route_status": data["stitch"]["route_status"],
        "source_pdf_path": data["pdf_evidence"]["source_pdf_path"],
        "key_artifacts": data["artifact_refs"],
    }


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Bufotalin stitched full-flow expert report",
        "",
        f"- generated_at_utc: `{data['generated_at_utc']}`",
        f"- run_dir: `{data['run_dir']}`",
        f"- final verdict: `{data['final']['verdict']}`",
        f"- solved: `{data['final']['solved']}`",
        f"- stitched route: accepted=`{data['stitch']['accepted']}`, status=`{data['stitch']['route_status']}`",
        "",
        "## Architecture",
        "",
        "The current controller treats ChemEnzy, retrieval, PDF extraction, route expansion, and self-evolution as typed tools behind deterministic gates. A solved claim must come from deterministic validators, not from raw planner text or raw route-generator status.",
        "",
        "## Bufotalin example",
        "",
        f"- live planner accepted: `{data['planner']['accepted']}`; run_semantics=`{data['planner']['run_semantics']}`; changed=`{data['planner']['run_semantics_changed']}`.",
        f"- native ChemEnzy: elapsed `{data['native_chemenzy']['elapsed_s']}` s, route_count `{data['native_chemenzy']['route_count']}`, accepted routes `{data['native_chemenzy']['accepted_route_count']}`.",
        f"- local PDF: `{data['pdf_evidence']['source_pdf_path']}`; rendered pages `{data['pdf_evidence']['summary'].get('rendered_page_count')}`, crops `{data['pdf_evidence']['summary'].get('scheme_crop_count')}`.",
        f"- source-detail chain: `{data['source_detail_chain']['step_count']}` steps, terminal reached `{data['source_detail_chain']['terminal_reached']}`.",
        f"- subgoal verifier: route_count `{data['subgoal']['route_count']}`, accepted `{data['subgoal']['accepted_route_count']}`, best rank `{data['subgoal']['best_route_rank']}`.",
        f"- stitched route: `{data['stitch']['combined_route'].get('combined_step_count')}` total steps.",
        f"- patched guided rerun: elapsed `{data['guided_patched']['elapsed_s']}` s, route_status `{data['guided_patched']['route_status']}`.",
        "",
        "## Key artifacts",
        "",
    ]
    for key, value in data["artifact_refs"].items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def render_pdf(data: dict[str, Any], pdf_path: Path) -> None:
    pages = [
        draw_cover(data),
        draw_architecture(data),
        draw_controller_contract(data),
        draw_tool_run(data),
        draw_pdf_evidence(data),
        draw_source_detail_chain(data),
        draw_subgoal_stitch(data),
        draw_negative_controls(data),
        draw_artifact_map(data),
    ]
    doc = fitz.open()
    for idx, image in enumerate(pages, start=1):
        png_path = ASSET_DIR / f"page_{idx:02d}.png"
        image.save(png_path, quality=95)
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        page.insert_image(page.rect, filename=str(png_path))
    doc.set_metadata(
        {
            "title": "Bufotalin stitched full-flow expert report",
            "author": "AutoPlanner Codex harness",
            "subject": "Canonical architecture and bufotalin stitched semisynthesis example",
            "keywords": "AutoPlanner, ChemEnzy, bufotalin, route verifier, source-detail chain, subgoal stitching",
        }
    )
    doc.save(pdf_path, deflate=True, garbage=4)
    doc.close()


def draw_cover(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base_canvas(
        "AutoPlanner 高级天然产物全流程专家汇报",
        "以 bufotalin stitched semisynthesis 为完整运行示例",
        "01",
    )
    target = data["target"]
    draw_card(
        draw,
        MARGIN,
        170,
        500,
        430,
        "Target",
        [
            f"name: {target['name']}",
            f"formula: {target['formula']}",
            f"exact MW: {fmt(target['exact_mw'])}",
            f"heavy atoms / rings: {target['heavy_atoms']} / {target['rings']}",
            f"stereocenters: {target['stereocenters']}",
            f"InChIKey: {target['inchi_key']}",
            "risk flags: " + join_items(target["risk_flags"]),
        ],
    )
    draw_molecule(canvas, draw, target["isomeric_smiles"], 650, 170, 470, 390, "Bufotalin RDKit depiction")
    draw_metric_row(
        draw,
        [
            ("Planner", "accepted", data["planner"]["run_semantics"]),
            ("Source chain", f"{data['source_detail_chain']['step_count']} steps", "terminal compound 11"),
            ("Subgoal", f"{data['subgoal']['accepted_route_count']}/{data['subgoal']['route_count']} accepted", "best rank 2"),
            ("Final", str(data["final"]["verdict"]), f"solved={data['final']['solved']}"),
        ],
        700,
    )
    draw_card(
        draw,
        MARGIN,
        970,
        PAGE_W - 2 * MARGIN,
        285,
        "Executive conclusion",
        [
            "本轮 live planner 通过 schema gate，未使用 fallback plan 作为判断基础。",
            "PDF 使用的是本地既有文件，但 PDF render、crop、visual audit、source-detail validation、subgoal search 和 stitch audit 均重新执行。",
            "最终 solved 不是 ChemEnzy 原生/guided 的直接输出，而是 15 步 source-detail 文献链与 compound 11 子目标 solved route 的 deterministic stitch。",
            f"Combined route: {data['stitch']['combined_route'].get('combined_step_count')} steps = {data['stitch']['combined_route'].get('subgoal_route_step_count')} subgoal + {data['stitch']['combined_route'].get('literature_step_count')} literature.",
        ],
        status="ok",
    )
    draw_footer(draw, data)
    return canvas


def draw_architecture(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base_canvas("当前 canonical 架构", "Typed tools + deterministic gates", "02")
    stages = [
        ("SMILES intake", "RDKit parse, canonical/isomeric identity, formula, stereocenter/ring/risk profile."),
        ("Live planner schema gate", "Plan, strategy, run_semantics, allowed local tools, safety flags; invalid plans stop before tools."),
        ("Native ChemEnzy", "Route generator. Raw solved is evidence, not final authority."),
        ("Route verifier/audit", "Target identity, stock closure, hidden nonstock leaves, large atom jumps, same-scaffold fake closure."),
        ("Frontier/literature gate", "Unresolved or fake closure triggers retrieval/open research/source detail."),
        ("PDF/source-detail", "Source access status first; same-scope local PDF fallback; visual/curator records; no raw text promotion."),
        ("Guided/subgoal", "Plugin-guided ChemEnzy and child target search operate under verifier gates."),
        ("Stitch + final verdict", "Only exact terminal identity + accepted subgoal verifier + accepted literature chain can solve target."),
    ]
    x_left, x_right = MARGIN, 675
    y = 168
    for idx, (title, body) in enumerate(stages):
        x = x_left if idx % 2 == 0 else x_right
        if idx and idx % 2 == 0:
            y += 255
        draw_card(draw, x, y, 495, 180, f"{idx + 1}. {title}", [body], status="ok" if idx in {1, 3, 7} else "")
        if idx < len(stages) - 1:
            draw_small_arrow(draw, x + 510 if idx % 2 == 0 else x - 35, y + 78, reverse=bool(idx % 2))
    draw_card(
        draw,
        MARGIN,
        1260,
        PAGE_W - 2 * MARGIN,
        165,
        "Architecture invariant",
        [
            "Planner 可以发起工具和子目标搜索，但 solved 权威只来自 deterministic validators。",
            "子目标 solved 只说明该 terminal 可解；必须由 stitch audit 证明 terminal 与文献链末端完全同一分子，才能上升为目标 solved。",
        ],
        status="ok",
    )
    draw_footer(draw, data)
    return canvas


def draw_controller_contract(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base_canvas("Planner schema 与审计边界", "Why live planner matters", "03")
    planner = data["planner"]
    draw_card(
        draw,
        MARGIN,
        170,
        530,
        330,
        "Live planner result",
        [
            f"accepted: {planner['accepted']}",
            f"elapsed_s: {fmt(planner['elapsed_s'])}",
            f"raw_run_semantics_type: {planner['raw_run_semantics_type']}",
            f"run_semantics: {planner['run_semantics']}",
            f"run_semantics_changed: {planner['run_semantics_changed']}",
            f"planned tool count: {len(planner['planned_tools'])}",
            f"record: {planner['record_path']}",
        ],
        status="ok",
    )
    draw_card(
        draw,
        650,
        170,
        520,
        330,
        "Fixed contract",
        [
            "run_semantics 现在支持安全 normalization audit，但 prompt 明确要求输出字符串枚举。",
            "allowed_local_tools 新增 stitch_literature_chain_with_subgoal_route。",
            "deterministic fallback 不再作为本次判断路径；完整示例来自 live planner accepted run。",
        ],
    )
    tool_lines = [f"{idx}. {name}" for idx, name in enumerate(planner["planned_tools"], start=1)]
    draw_card(draw, MARGIN, 550, PAGE_W - 2 * MARGIN, 520, "Planned tools from live planner", tool_lines)
    draw_card(
        draw,
        MARGIN,
        1135,
        PAGE_W - 2 * MARGIN,
        255,
        "Subgoal/stitch responsibility",
        [
            "Agent 发起子目标搜索；最终审计和拼接也作为工具链的一部分执行。",
            "stitch 工具要求 target identity audit、terminal/subgoal exact match、subgoal route verifier accepted、source-detail chain accepted。",
            "任何单独的 subgoal solved 都不会自动写成 target solved。",
        ],
        status="ok",
    )
    draw_footer(draw, data)
    return canvas


def draw_tool_run(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base_canvas("Bufotalin 全流程运行概览", "Real tool execution summary", "04")
    native = data["native_chemenzy"]
    guided = data["guided_patched"]
    open_research = data["open_research"]
    draw_card(
        draw,
        MARGIN,
        170,
        520,
        285,
        "Native ChemEnzy",
        [
            f"elapsed_s: {fmt(native['elapsed_s'])}",
            f"route_count: {native['route_count']}",
            f"accepted_route_count: {native['accepted_route_count']}",
            f"route_status: {native['route_status']}",
            "reasons: " + join_items(native["reasons"]),
        ],
        status="bad",
    )
    draw_card(
        draw,
        650,
        170,
        520,
        285,
        "Open research",
        [
            f"elapsed_s: {fmt(open_research['elapsed_s'])}",
            f"status: {open_research['status']}",
            "reasons: " + join_items(open_research["reasons"]),
            f"output_dir: {open_research['output_dir']}",
        ],
        status="warn",
    )
    draw_card(
        draw,
        MARGIN,
        500,
        520,
        310,
        "Patched guided rerun",
        [
            f"elapsed_s: {fmt(guided['elapsed_s'])}",
            f"policy_id: {guided['policy_id']}",
            f"rerun_reason: {guided['rerun_reason']}",
            f"route_status: {guided['route_status']}",
            f"solved: {guided['solved']}",
            "verifier reasons: " + join_items(guided["verifier"]["reasons"]),
        ],
        status="bad",
    )
    draw_card(
        draw,
        650,
        500,
        520,
        310,
        "Route expansion subgoal",
        [
            "target: compound 11 exact TET2025 terminal",
            f"route_status: {data['subgoal']['route_status']}",
            f"route_count: {data['subgoal']['route_count']}",
            f"accepted_route_count: {data['subgoal']['accepted_route_count']}",
            f"best_route_rank: {data['subgoal']['best_route_rank']}",
            f"target_match: {data['subgoal']['target_match']}",
        ],
        status="ok",
    )
    draw_card(
        draw,
        MARGIN,
        870,
        PAGE_W - 2 * MARGIN,
        410,
        "Tool-call sequence",
        [
            f"{idx}. {row.get('tool_name')} | {row.get('status')} | {fmt(row.get('elapsed_s'))}s | {join_items(row.get('reasons') or [])}"
            for idx, row in enumerate(data["tool_status"], start=1)
        ],
    )
    draw_footer(draw, data)
    return canvas


def draw_pdf_evidence(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base_canvas("PDF/source-detail 证据链", "Existing PDF was reused; extraction was rerun", "05")
    pdf_data = data["pdf_evidence"]
    visual = data["visual_audit"]
    draw_card(
        draw,
        MARGIN,
        160,
        520,
        315,
        "PDF extraction",
        [
            f"source_pdf_path: {pdf_data['source_pdf_path']}",
            f"sha256: {shorten(pdf_data['source_pdf_sha256'], 58)}",
            f"status: {pdf_data['status']}",
            f"rendered_page_count: {pdf_data['summary'].get('rendered_page_count')}",
            f"scheme_crop_count: {pdf_data['summary'].get('scheme_crop_count')}",
            f"compound_text_snippet_count: {pdf_data['summary'].get('compound_text_snippet_count')}",
        ],
        status="ok",
    )
    parsed = visual["parsed_output"]
    draw_card(
        draw,
        650,
        160,
        520,
        315,
        "Visual audit",
        [
            f"elapsed_s: {fmt(visual['elapsed_s'])}",
            f"accepted: {visual['accepted']}",
            "visible_sequence: " + join_items(parsed.get("visible_sequence") or []),
            "no_solved_claim: " + str(parsed.get("no_solved_claim")),
            "limitation: crops support late Scheme 4 only; full chain is validated via structured source-detail chain.",
        ],
        status="warn",
    )
    crop_paths = [Path(str(item.get("image_path") or "")) for item in pdf_data.get("scheme_crops") or []]
    draw_image_strip(canvas, draw, [p for p in crop_paths if p.exists()][:3], MARGIN, 535, PAGE_W - 2 * MARGIN, 440)
    draw_card(
        draw,
        MARGIN,
        1040,
        PAGE_W - 2 * MARGIN,
        290,
        "Evidence policy",
        [
            "PDF fallback does not emit SMILES directly; it emits evidence manifest, rendered pages, crops, and snippets.",
            "The source-detail chain becomes route evidence only after product/reactant SMILES pass RDKit, chain-contiguity, target-match, and source-policy checks.",
            "The visual audit intentionally remains conservative and does not claim solved from images alone.",
        ],
    )
    draw_footer(draw, data)
    return canvas


def draw_source_detail_chain(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base_canvas("文献 source-detail 链", "15-step chain from bufotalin to compound 11", "06")
    chain = data["source_detail_chain"]
    draw_card(
        draw,
        MARGIN,
        160,
        520,
        290,
        "Chain validation",
        [
            f"accepted: {chain['accepted']}",
            f"step_count: {chain['step_count']}",
            f"terminal_reached: {chain['terminal_reached']}",
            f"terminal_name: {chain['terminal_name']}",
            f"terminal_smiles: {shorten(chain['terminal_smiles'], 76)}",
            "reasons: " + join_items(chain["reasons"]),
        ],
        status="ok",
    )
    draw_molecule(canvas, draw, chain["terminal_smiles"], 690, 160, 390, 290, "Compound 11 terminal")
    draw_card(
        draw,
        MARGIN,
        510,
        520,
        420,
        "First source-detail steps",
        [
            f"{row['step_index']}. {row['step_id']}"
            for row in chain["first_steps"]
        ],
    )
    draw_card(
        draw,
        650,
        510,
        520,
        420,
        "Last source-detail steps",
        [
            f"{row['step_index']}. {row['step_id']}"
            for row in chain["last_steps"]
        ],
    )
    draw_card(
        draw,
        MARGIN,
        995,
        PAGE_W - 2 * MARGIN,
        315,
        "Why this is not yet final solved by itself",
        [
            "The literature chain runs backward from bufotalin to terminal compound 11 and proves a source-detail terminal.",
            "It still requires an independent route closure from stock/building blocks to compound 11.",
            "The stitch tool is the authority that joins these two segments after exact identity audit.",
        ],
    )
    draw_footer(draw, data)
    return canvas


def draw_subgoal_stitch(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base_canvas("Subgoal search 与 stitched solved", "Where solved is earned", "07")
    stitch = data["stitch"]
    combined = stitch["combined_route"]
    terminal_audit = stitch["terminal_match_audit"]
    draw_card(
        draw,
        MARGIN,
        160,
        520,
        330,
        "Subgoal verifier",
        [
            f"accepted: {data['subgoal']['accepted']}",
            f"route_status: {data['subgoal']['route_status']}",
            f"route_count: {data['subgoal']['route_count']}",
            f"accepted_route_count: {data['subgoal']['accepted_route_count']}",
            f"best_route_rank: {data['subgoal']['best_route_rank']}",
            "reasons/warnings: " + join_items(data["subgoal"]["reasons"]),
        ],
        status="ok",
    )
    draw_card(
        draw,
        650,
        160,
        520,
        330,
        "Terminal identity audit",
        [
            f"accepted: {terminal_audit.get('accepted')}",
            f"match_basis: {terminal_audit.get('match_basis')}",
            "terminal InChIKey: " + str((terminal_audit.get("terminal") or {}).get("inchikey")),
            "subgoal InChIKey: " + str((terminal_audit.get("subgoal_target") or {}).get("inchikey")),
            "reasons: " + join_items(terminal_audit.get("reasons") or []),
        ],
        status="ok",
    )
    draw_route_bar(draw, MARGIN, 560, PAGE_W - 2 * MARGIN, combined)
    draw_card(
        draw,
        MARGIN,
        790,
        PAGE_W - 2 * MARGIN,
        280,
        "Stitched route verdict",
        [
            f"accepted: {stitch['accepted']}",
            f"solved: {stitch['solved']}",
            f"route_status: {stitch['route_status']}",
            f"stock_audit_passed: {stitch['stock_audit_passed']}",
            f"combined_step_count: {combined.get('combined_step_count')}",
            "warnings: " + join_items(stitch["warnings"]),
        ],
        status="ok",
    )
    draw_card(
        draw,
        MARGIN,
        1130,
        PAGE_W - 2 * MARGIN,
        240,
        "Source policy",
        [
            f"terminal_identity_match_required: {stitch['source_policy'].get('terminal_identity_match_required')}",
            f"subgoal_segment_requires_route_verifier: {stitch['source_policy'].get('subgoal_segment_requires_route_verifier')}",
            f"literature_segment_requires_source_detail_chain: {stitch['source_policy'].get('literature_segment_requires_source_detail_chain')}",
            f"final_verdict_authority: {stitch['source_policy'].get('final_verdict_authority')}",
        ],
    )
    draw_footer(draw, data)
    return canvas


def draw_negative_controls(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base_canvas("负例与边界条件", "Why the final solved claim is not overbroad", "08")
    draw_card(
        draw,
        MARGIN,
        165,
        520,
        335,
        "Native ChemEnzy rejected",
        [
            f"route_status: {data['native_chemenzy']['route_status']}",
            f"accepted_route_count: {data['native_chemenzy']['accepted_route_count']}",
            "reasons: " + join_items(data["native_chemenzy"]["reasons"]),
            "Interpretation: route generator output contained fake closure / advanced same-scaffold terminal evidence.",
        ],
        status="bad",
    )
    draw_card(
        draw,
        650,
        165,
        520,
        335,
        "Patched guided rerun rejected",
        [
            f"elapsed_s: {fmt(data['guided_patched']['elapsed_s'])}",
            f"route_status: {data['guided_patched']['route_status']}",
            f"accepted_route_count: {data['guided_patched']['verifier']['accepted_route_count']}",
            "reasons: " + join_items(data["guided_patched"]["verifier"]["reasons"]),
            "Interpretation: plugin guidance ran, but did not by itself solve the target.",
        ],
        status="bad",
    )
    draw_card(
        draw,
        MARGIN,
        555,
        520,
        300,
        "Artifact bundle validation",
        [
            f"accepted: {data['artifact_bundle']['accepted']}",
            "reasons: " + join_items(data["artifact_bundle"]["reasons"]),
            "This preserves negative evidence instead of hiding it behind the stitched solved route.",
        ],
        status="warn",
    )
    draw_card(
        draw,
        650,
        555,
        520,
        300,
        "Final verdict precedence",
        [
            f"verdict: {data['final']['verdict']}",
            f"solved: {data['final']['solved']}",
            f"route_status: {data['final']['route_status']}",
            "The final verdict can be solved because stitched_semisynthesis_route passed exact identity and stock audits.",
        ],
        status="ok",
    )
    draw_card(
        draw,
        MARGIN,
        915,
        PAGE_W - 2 * MARGIN,
        335,
        "Expert interpretation",
        [
            "Native/guided route failures remain visible and auditable.",
            "The positive claim is narrower: a semisynthesis route made from a verified stock-to-compound-11 subroute and an accepted source-detail literature chain from compound 11 to bufotalin.",
            "Subgoal route warnings such as large_atom_jump are carried as warnings because the verifier still found accepted stock-closed routes.",
        ],
    )
    draw_footer(draw, data)
    return canvas


def draw_artifact_map(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base_canvas("审计产物与复现入口", "Every claim has a local artifact path", "09")
    refs = data["artifact_refs"]
    priority = [
        "summary",
        "planner_record",
        "pdf_evidence",
        "visual_chain_validation",
        "source_detail_chain_route",
        "route_expansion_subgoals",
        "stitched_route",
        "final_verdict",
        "patched_guided_tool_call",
        "guided_route_verifier",
        "artifact_bundle_validation",
    ]
    lines = [f"{key}: {artifact_display_path(data, refs.get(key))}" for key in priority if refs.get(key)]
    draw_card(draw, MARGIN, 160, PAGE_W - 2 * MARGIN, 650, "Key artifact paths", lines)
    draw_card(
        draw,
        MARGIN,
        860,
        520,
        300,
        "Code paths changed",
        [
            "schemas.py: run_semantics normalization + stitch tool registration",
            "codex_plan.py: planner prompt/schema and stitch planning instruction",
            "tools.py: stitch handler, plugin-only guided policy, compiled_downstream state publication",
            "runner.py: final verdict can consume stitched solved route",
            "literature_pdf_extraction.py: robust crop source fallback",
        ],
    )
    draw_card(
        draw,
        650,
        860,
        520,
        300,
        "Verification",
        [
            "pytest selected harness contract tests passed.",
            "Full bufotalin run completed with live planner accepted.",
            "Patched guided supplement completed as a real ChemEnzy run and was rejected by verifier.",
        ],
        status="ok",
    )
    draw_card(
        draw,
        MARGIN,
        1220,
        PAGE_W - 2 * MARGIN,
        150,
        "Report scope",
        [
            "This PDF is an expert briefing, not a laboratory procedure. It reports architecture, typed evidence flow, and deterministic audit outcomes.",
        ],
    )
    draw_footer(draw, data)
    return canvas


def base_canvas(title: str, subtitle: str, page_no: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    canvas = Image.new("RGB", (PAGE_W, PAGE_H), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, PAGE_W, 30), fill=ACCENT)
    draw.text((MARGIN, 56), title, font=font(41, bold=True), fill=TEXT)
    draw_wrapped(draw, subtitle, MARGIN, 118, PAGE_W - 2 * MARGIN - 90, font(22), MUTED)
    draw.text((PAGE_W - MARGIN - 50, 65), page_no, font=font(30, bold=True), fill=ACCENT)
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
    title_fill = TEXT
    if status == "bad":
        outline = (248, 113, 113)
        fill = (255, 247, 247)
        title_fill = BAD
    elif status == "warn":
        outline = (245, 158, 11)
        fill = (255, 251, 235)
        title_fill = WARN
    elif status == "ok":
        outline = (52, 211, 153)
        fill = (240, 253, 244)
        title_fill = GOOD
    draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill=fill, outline=outline, width=2)
    draw.text((x + 22, y + 18), title, font=font(24, bold=True), fill=title_fill)
    ty = y + 60
    for idx, line in enumerate(lines):
        used = draw_wrapped(draw, str(line), x + 22, ty, w - 44, font(17), TEXT, line_gap=5)
        ty += used + 9
        if ty > y + h - 26 and idx < len(lines) - 1:
            draw.text((x + 22, y + h - 24), "...", font=font(18), fill=MUTED)
            break


def draw_metric_row(draw: ImageDraw.ImageDraw, metrics: list[tuple[str, str, str]], y: int) -> None:
    gap = 22
    w = (PAGE_W - 2 * MARGIN - gap * (len(metrics) - 1)) // len(metrics)
    x = MARGIN
    for label, value, note in metrics:
        draw.rounded_rectangle((x, y, x + w, y + 165), radius=14, fill=PANEL, outline=LINE, width=2)
        draw.text((x + 18, y + 18), label, font=font(18, bold=True), fill=MUTED)
        draw_wrapped(draw, value, x + 18, y + 58, w - 36, font(25, bold=True), TEXT)
        draw_wrapped(draw, note, x + 18, y + 108, w - 36, font(15), MUTED)
        x += w + gap


def draw_small_arrow(draw: ImageDraw.ImageDraw, x: int, y: int, *, reverse: bool = False) -> None:
    if reverse:
        draw.line((x + 30, y, x, y), fill=ACCENT_2, width=4)
        draw.polygon([(x, y), (x + 14, y - 8), (x + 14, y + 8)], fill=ACCENT_2)
    else:
        draw.line((x, y, x + 30, y), fill=ACCENT_2, width=4)
        draw.polygon([(x + 30, y), (x + 16, y - 8), (x + 16, y + 8)], fill=ACCENT_2)


def draw_route_bar(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, route: dict[str, Any]) -> None:
    draw.rounded_rectangle((x, y, x + w, y + 170), radius=16, fill=PANEL, outline=LINE, width=2)
    sub_w = int(w * 0.42)
    lit_w = w - sub_w - 70
    bar_y = y + 70
    draw.rounded_rectangle((x + 34, bar_y, x + 34 + sub_w, bar_y + 46), radius=10, fill=(220, 252, 231), outline=(52, 211, 153), width=2)
    draw.rounded_rectangle((x + 34 + sub_w + 70, bar_y, x + 34 + sub_w + 70 + lit_w, bar_y + 46), radius=10, fill=(219, 234, 254), outline=(96, 165, 250), width=2)
    draw.text((x + 34, y + 24), "Combined stitched route", font=font(23, bold=True), fill=TEXT)
    draw.text((x + 48, bar_y + 9), f"Subgoal stock closure: {route.get('subgoal_route_step_count')} steps", font=font(16, bold=True), fill=GOOD)
    draw.text((x + 34 + sub_w + 92, bar_y + 9), f"Literature chain: {route.get('literature_step_count')} steps", font=font(16, bold=True), fill=ACCENT_2)
    mid = x + 34 + sub_w + 36
    draw.line((mid - 18, bar_y + 23, mid + 18, bar_y + 23), fill=ACCENT, width=4)
    draw.polygon([(mid + 18, bar_y + 23), (mid + 5, bar_y + 15), (mid + 5, bar_y + 31)], fill=ACCENT)
    draw.text((x + 34, y + 128), f"Direction: {route.get('direction')} | total: {route.get('combined_step_count')} steps", font=font(16), fill=MUTED)


def draw_molecule(canvas: Image.Image, draw: ImageDraw.ImageDraw, smiles: str, x: int, y: int, w: int, h: int, label: str) -> None:
    draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill=PANEL, outline=LINE, width=2)
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        draw.text((x + 18, y + 18), "Invalid SMILES", font=font(18), fill=BAD)
        return
    img = Draw.MolToImage(mol, size=(max(120, w - 42), max(120, h - 78)), legend="")
    canvas.paste(img.convert("RGB"), (x + 20, y + 16))
    draw.text((x + 20, y + h - 40), label, font=font(16), fill=MUTED)


def draw_image_strip(canvas: Image.Image, draw: ImageDraw.ImageDraw, paths: list[Path], x: int, y: int, w: int, h: int) -> None:
    draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill=PANEL, outline=LINE, width=2)
    draw.text((x + 20, y + 18), "Rerendered PDF crops", font=font(23, bold=True), fill=TEXT)
    if not paths:
        draw.text((x + 20, y + 70), "No crop images found.", font=font(18), fill=BAD)
        return
    gap = 18
    box_w = (w - 40 - gap * (len(paths) - 1)) // len(paths)
    box_h = h - 110
    bx = x + 20
    for path in paths:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((box_w, box_h))
            px = bx + (box_w - img.width) // 2
            py = y + 70 + (box_h - img.height) // 2
            canvas.paste(img, (px, py))
            draw.rectangle((bx, y + 70, bx + box_w, y + 70 + box_h), outline=LINE, width=1)
            draw.text((bx, y + h - 30), path.stem[:34], font=font(13), fill=MUTED)
        bx += box_w + gap


def draw_footer(draw: ImageDraw.ImageDraw, data: dict[str, Any]) -> None:
    draw.line((MARGIN, PAGE_H - 82, PAGE_W - MARGIN, PAGE_H - 82), fill=LINE, width=2)
    draw.text((MARGIN, PAGE_H - 59), f"Run: {data['run_dir']}", font=font(13), fill=MUTED)
    draw.text((PAGE_W - 350, PAGE_H - 59), "AutoPlanner harness expert report", font=font(13), fill=MUTED)


def artifact_display_path(data: dict[str, Any], value: Any) -> str:
    raw = str(value or "")
    if not raw:
        return ""
    path = Path(raw)
    run_dir = Path(str(data.get("run_dir") or ""))
    try:
        return "run_dir/" + str(path.relative_to(run_dir))
    except ValueError:
        pass
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return raw


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
    for paragraph in str(text).splitlines() or [""]:
        if not paragraph:
            lines.append("")
            continue
        current = ""
        tokens = paragraph.split(" ")
        if len(tokens) == 1:
            tokens = split_cjk_aware(paragraph)
        for token in tokens:
            trial = token if not current else current + ("" if is_cjk_token(token) else " ") + token
            if text_width(draw, trial, font_obj) <= width:
                current = trial
                continue
            if current:
                lines.append(current)
            if text_width(draw, token, font_obj) <= width:
                current = token
            else:
                for piece in hard_wrap_token(draw, token, width, font_obj):
                    if piece:
                        lines.append(piece)
                current = ""
        if current:
            lines.append(current)
    line_h = max(1, int(font_obj.size * 1.25)) if hasattr(font_obj, "size") else 24
    for idx, line in enumerate(lines):
        draw.text((x, y + idx * (line_h + line_gap)), line, font=font_obj, fill=fill)
    return len(lines) * line_h + max(0, len(lines) - 1) * line_gap


def split_cjk_aware(text: str) -> list[str]:
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return list(text)
    return textwrap.wrap(text, width=18, break_long_words=True) or [text]


def is_cjk_token(token: str) -> bool:
    return len(token) == 1 and any("\u4e00" <= ch <= "\u9fff" for ch in token)


def hard_wrap_token(draw: ImageDraw.ImageDraw, token: str, width: int, font_obj: ImageFont.ImageFont) -> list[str]:
    pieces: list[str] = []
    current = ""
    for ch in token:
        trial = current + ch
        if text_width(draw, trial, font_obj) <= width:
            current = trial
        else:
            if current:
                pieces.append(current)
            current = ch
    if current:
        pieces.append(current)
    return pieces


def text_width(draw: ImageDraw.ImageDraw, text: str, font_obj: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font_obj)
    return bbox[2] - bbox[0]


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    if path.exists():
        return ImageFont.truetype(str(path), size)
    fallback = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(fallback), size)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return str(value)
    return str(value)


def join_items(items: list[Any], *, empty: str = "-") -> str:
    values = [str(item) for item in items if str(item)]
    return ", ".join(values) if values else empty


def shorten(value: Any, max_len: int = 80) -> str:
    text = str(value or "")
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def tool_elapsed(summary: dict[str, Any], tool_name: str) -> Any:
    for row in summary.get("tool_status") or []:
        if row.get("tool_name") == tool_name:
            return row.get("elapsed_s")
    return None


def tool_status(summary: dict[str, Any], tool_name: str) -> str:
    for row in summary.get("tool_status") or []:
        if row.get("tool_name") == tool_name:
            return str(row.get("status") or "")
    return ""


def tool_reasons(summary: dict[str, Any], tool_name: str) -> list[str]:
    for row in summary.get("tool_status") or []:
        if row.get("tool_name") == tool_name:
            return [str(item) for item in row.get("reasons") or []]
    return []


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def read_tool_calls(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


if __name__ == "__main__":
    main()
