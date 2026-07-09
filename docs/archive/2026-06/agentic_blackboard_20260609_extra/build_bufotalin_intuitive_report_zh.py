"""Build an intuitive Chinese report for the bufotalin autonomous retry."""
from __future__ import annotations

import io
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "agentic_blackboard" / "report_20260609"
V7_DIR = ROOT / "results" / "shared" / "bufotalin_agentic_blackboard_full_retry_v7_20260609"
V6_DIR = ROOT / "results" / "shared" / "bufotalin_agentic_blackboard_full_retry_v6_20260609"
V6_FIXED_DIR = ROOT / "results" / "shared" / "bufotalin_v6_recompile_after_target_anchor_fix_20260609"
OLD_SUCCESS_DIR = ROOT / "results" / "shared" / "bufotalin_fullflow_fresh_visual_existing_pdf_20260608_065053"
PDF_PATH = ROOT / "1-s2.0-S0040402025001668-main.pdf"

PAGE_W, PAGE_H = 1240, 1754
MARGIN = 72
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
BG = (248, 250, 252)
PANEL = (255, 255, 255)
TEXT = (15, 23, 42)
MUTED = (71, 85, 105)
LINE = (203, 213, 225)
BLUE = (14, 116, 144)
GREEN = (22, 101, 52)
AMBER = (146, 64, 14)
RED = (153, 27, 27)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = build_data()
    stem = "bufotalin_intuitive_route_blackboard_zh_20260609"
    json_path = OUT_DIR / f"{stem}.json"
    md_path = OUT_DIR / f"{stem}.md"
    pdf_path = OUT_DIR / f"{stem}.pdf"
    audit_path = OUT_DIR / f"{stem}_audit.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown(data), encoding="utf-8")
    render_pdf(data, pdf_path)
    audit = {
        "accepted": pdf_path.exists() and pdf_path.stat().st_size > 50_000,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "pdf_path": str(pdf_path),
        "pdf_size_bytes": pdf_path.stat().st_size if pdf_path.exists() else 0,
        "v7_final": data["v7"]["final"].get("verdict"),
        "v7_solved": bool(data["v7"]["final"].get("solved")),
        "tests": data["tests"],
        "schema_version": "bufotalin_intuitive_report_audit.v1",
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


def build_data() -> dict[str, Any]:
    v7_final = read_json(V7_DIR / "final_verdict.json")
    v7_board = read_json(V7_DIR / "agent_blackboard.json")
    v7_compile = read_json(V7_DIR / "source_detail_chain_route_result.json")
    v7_expansion = read_json(V7_DIR / "route_expansion_subgoal_search_result.json")
    v7_stitched = read_json(V7_DIR / "stitched_semisynthesis_route_result.json")
    v6_fixed_audit = read_json(V6_FIXED_DIR / "source_detail_chain_route" / "source_detail_route_chain_audit.json")
    old_final = read_json(OLD_SUCCESS_DIR / "final_verdict_strict_visual_clean.json")
    old_chain = read_json(OLD_SUCCESS_DIR / "source_detail_chain_route_strict_visual" / "source_detail_route_chain_audit.json")
    old_stitch = read_json(OLD_SUCCESS_DIR / "stitched_semisynthesis_route_strict_visual" / "stitched_semisynthesis_route.json")
    route_rows = route_rows_from_chain(v6_fixed_audit)
    return {
        "schema_version": "bufotalin_intuitive_route_blackboard_report.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pdf_path": str(PDF_PATH),
        "v7_run_dir": str(V7_DIR),
        "v6_run_dir": str(V6_DIR),
        "v6_fixed_dir": str(V6_FIXED_DIR),
        "old_success_dir": str(OLD_SUCCESS_DIR),
        "route_rows": route_rows,
        "route_sequence": [row["from"] for row in reversed(route_rows)] + ["bufotalin"],
        "v7": {
            "final": v7_final,
            "budget": v7_board.get("budget_state") or {},
            "actions": action_rows(v7_board),
            "visual_summaries": (v7_board.get("literature_evidence") or {}).get("visual_chains") or [],
            "exact_row_count": len((v7_board.get("literature_evidence") or {}).get("exact_rows") or []),
            "compile": {
                "accepted": bool(v7_compile.get("accepted")),
                "row_count": len(((v7_compile.get("compiled_downstream") or {}).get("literature_template_plugin") or {}).get("one_step_rows") or []),
                "summary": (v7_compile.get("chain_audit") or {}).get("summary") or {},
                "reasons": v7_compile.get("reasons") or [],
            },
            "route_expansion": {
                "accepted": bool(v7_expansion.get("accepted")),
                "status": v7_expansion.get("status"),
                "reasons": v7_expansion.get("reasons") or [],
                "subgoal_count": v7_expansion.get("subgoal_count"),
                "accepted_subgoal_count": v7_expansion.get("accepted_subgoal_count"),
            },
            "stitched": {
                "accepted": bool(v7_stitched.get("accepted")),
                "route_status": v7_stitched.get("route_status"),
                "reasons": v7_stitched.get("reasons") or [],
            },
        },
        "v6_fixed": {
            "accepted": bool(v6_fixed_audit.get("accepted")),
            "step_count": int(v6_fixed_audit.get("step_count") or 0),
            "summary": v6_fixed_audit.get("summary") or {},
            "reasons": v6_fixed_audit.get("reasons") or [],
        },
        "old_success": {
            "solved": bool(old_final.get("solved")),
            "verdict": old_final.get("verdict"),
            "chain_step_count": int(old_chain.get("step_count") or 0),
            "terminal_reached": bool(old_chain.get("terminal_reached")),
            "stitched_solved": bool(old_stitch.get("solved")),
            "stock_audit_passed": bool(old_stitch.get("stock_audit_passed")),
        },
        "tests": "python -m pytest tests/test_agentic_blackboard_controller.py tests/test_failure_critic.py tests/test_parent_route_proof.py tests/test_open_research_experience.py tests/test_codex_entry_harness_contract.py -q -> 153 passed, 2 skipped",
    }


def route_rows_from_chain(audit: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in audit.get("chain") or []:
        step_id = str(item.get("step_id") or "")
        if "_from_" in step_id:
            product, reactant = step_id.split("_from_", 1)
        else:
            product, reactant = str(item.get("step_index") or ""), "?"
        cond = item.get("condition_candidate") or {}
        pieces = [str(cond.get(key) or "").strip() for key in ("reagent", "solvent", "temperature", "reported_yield") if str(cond.get(key) or "").strip()]
        rows.append({"to": product, "from": reactant, "condition": "；".join(pieces) or "条件已记录"})
    return rows


def action_rows(board: dict[str, Any]) -> list[dict[str, Any]]:
    names = {
        "generate_disconnection_hypotheses": "提出目标侧断键想法",
        "search_literature": "找文献锚点",
        "extract_pdf_literature_structures": "把 PDF 变成可读页面和裁图",
        "rank_analogical_hypotheses": "给类比方向排序",
        "extract_visual_literature_chain": "读图抽路线",
        "compile_exact_literature_rows": "整理成可审查文献步骤",
        "run_guided_chemenzy": "带约束路线搜索",
        "expand_child_target": "扩展上游子目标",
        "stitch_parent_route": "拼接父路线证明",
        "build_failure_critic_report": "更新失败判断",
    }
    return [
        {
            "round": row.get("round_index"),
            "name": names.get(str(row.get("action_type") or ""), str(row.get("action_type") or "")),
            "useful": bool(row.get("useful_artifact")),
            "reasons": row.get("reasons") or [],
        }
        for row in board.get("action_history") or []
        if isinstance(row, dict)
    ]


def markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Bufotalin 自主探索复盘：路线图与黑板变化",
        "",
        f"结论：v7 完整跑满预算，最终 `{data['v7']['final'].get('verdict')}`，没有通过父路线证明。",
        f"PDF：`{data['pdf_path']}`",
        "",
        "## 15 步文献路线图",
    ]
    for row in data["route_rows"]:
        lines.append(f"- {row['from']} -> {row['to']}：{row['condition']}")
    lines.extend([
        "",
        "## 为什么失败",
        "- v7 真实自主读图只稳定拿到 31 -> 32 -> 33 -> bufotalin 三步；30 到 11 被读图代理保守标为缺口。",
        "- 路线搜索仍出现大重原子跳跃；后端 solved 标志被确定性审查拒绝。",
        "- 子目标没有和父路线、完整文献段、库存闭合同时连通，所以不能宣称 solved。",
        "",
        "## 和旧成功版对比",
        f"- 旧成功版：15 步文献链 accepted，终端 11 reached={data['old_success']['terminal_reached']}，拼接 solved={data['old_success']['stitched_solved']}。",
        f"- 本轮修复后复核：v6 读图结果可整理出 15 步，但这是离线复核，不是 v7 当轮最终证明。",
        "",
        f"验证：{data['tests']}",
    ])
    return "\n".join(lines) + "\n"


def render_pdf(data: dict[str, Any], path: Path) -> None:
    pages = [
        page_summary(data),
        page_route_map(data),
        page_blackboard(data),
        page_failure(data),
        page_comparison(data),
        page_files_tests(data),
    ]
    doc = fitz.open()
    for img in pages:
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        stream = io.BytesIO()
        img.save(stream, format="JPEG", quality=90)
        page.insert_image(fitz.Rect(0, 0, PAGE_W, PAGE_H), stream=stream.getvalue())
    doc.save(path)
    doc.close()


def page_summary(data: dict[str, Any]) -> Image.Image:
    img, draw = canvas()
    y = title(draw, "Bufotalin 自主探索复盘", "完整预算跑完；没有父路线证明，所以没有宣称 solved")
    facts = [
        f"最终状态：{data['v7']['final'].get('verdict')} / {data['v7']['final'].get('route_status')}",
        f"预算：{data['v7']['budget'].get('rounds_completed')}/{data['v7']['budget'].get('max_rounds')} 轮，读图 {data['v7']['budget'].get('visual_calls')} 次，路线搜索 {data['v7']['budget'].get('chemenzy_runs')} 次",
        f"真实 v7 文献步骤：{data['v7']['exact_row_count']} 行；离线修复复核：{data['v6_fixed']['step_count']} 行",
        "最重要的判断：PDF 不是没用上；失败在读图稳定性、文献链接入和父路线闭合三处。",
    ]
    for item in facts:
        y = box(draw, y, item, BLUE)
    draw_text(draw, "本报告用中文描述流程，避免把重点放在内部函数名上。所有最终结论都来自确定性审查，而不是后端搜索器自报成功。", MARGIN, y + 30, 25, MUTED, PAGE_W - 2 * MARGIN)
    return img


def page_route_map(data: dict[str, Any]) -> Image.Image:
    img, draw = canvas()
    y = title(draw, "PDF 路线图", "文献可形成的 15 步方向：从 11 一路到 bufotalin")
    route = " -> ".join(data["route_sequence"])
    y = draw_text(draw, route, MARGIN, y, 23, BLUE, PAGE_W - 2 * MARGIN)
    y += 20
    rows = data["route_rows"]
    col_w = (PAGE_W - 2 * MARGIN - 28) // 2
    left_y = y
    right_y = y
    for idx, row in enumerate(rows):
        text = f"{idx + 1}. {row['from']} -> {row['to']}：{row['condition']}"
        if idx < 8:
            left_y = mini(draw, MARGIN, left_y, col_w, text)
        else:
            right_y = mini(draw, MARGIN + col_w + 28, right_y, col_w, text)
    draw_text(draw, "v7 当轮只稳定进入证据链的是末端 3 步；v6 修复复核证明工具现在可以把 15 步读图结果整理成文献行。", MARGIN, max(left_y, right_y) + 28, 23, AMBER, PAGE_W - 2 * MARGIN)
    return img


def page_blackboard(data: dict[str, Any]) -> Image.Image:
    img, draw = canvas()
    y = title(draw, "黑板变迁", "每轮不是按固定脚本走，而是看当前证据和失败再选下一步")
    grouped: dict[int, list[str]] = {}
    for row in data["v7"]["actions"]:
        grouped.setdefault(int(row["round"] or 0), []).append(row["name"])
    for round_no in sorted(grouped):
        text = f"第 {round_no} 轮：" + "；".join(grouped[round_no])
        y = box(draw, y, text, GREEN if round_no <= 5 else AMBER)
    return img


def page_failure(data: dict[str, Any]) -> Image.Image:
    img, draw = canvas()
    y = title(draw, "为什么这次仍失败", "失败不是单点，而是三层证据没有同时闭合")
    items = [
        "读图层：v7 自主视觉很保守，只把 31 -> 32 -> 33 -> bufotalin 三步放进证据链；30 到 11 没有进入当轮 exact 文献段。",
        "搜索层：路线搜索返回过看似闭合的候选，但确定性审查发现大重原子跳跃，并标记高级同骨架终端不可信。",
        "桥接层：子目标没有与父目标、完整文献段和库存闭合同时连通；旧成功所需的 11 起点没有在 v7 当轮闭合。",
        "终审层：目标等价、父路线审查、库存审计、文献段连通、子目标连通必须同时通过；v7 没有满足。",
    ]
    for item in items:
        y = box(draw, y, item, RED if "终审" in item else AMBER)
    return img


def page_comparison(data: dict[str, Any]) -> Image.Image:
    img, draw = canvas()
    y = title(draw, "和旧成功版对比", "旧成功不是后端直接闭合 bufotalin，而是拼接半合成证明")
    items = [
        f"旧成功：15 步文献链 accepted，终端 11 到达：{data['old_success']['terminal_reached']}。",
        f"旧成功：11 的上游路线被证明，最终拼接 solved：{data['old_success']['stitched_solved']}，库存审计：{data['old_success']['stock_audit_passed']}。",
        f"v7：真实自主运行只得到 {data['v7']['exact_row_count']} 行文献步骤，最终 {data['v7']['final'].get('verdict')}。",
        f"v6 修复复核：同一 PDF 的一次读图结果经格式修复后可得到 {data['v6_fixed']['step_count']} 行；这说明失败有工具适配因素，也有视觉稳定性因素。",
    ]
    for item in items:
        y = box(draw, y, item, BLUE)
    return img


def page_files_tests(data: dict[str, Any]) -> Image.Image:
    img, draw = canvas()
    y = title(draw, "文件与验证", "可复查的运行目录、报告和测试")
    items = [
        f"v7 运行目录：{data['v7_run_dir']}",
        f"v6 运行目录：{data['v6_run_dir']}",
        f"v6 读图修复复核：{data['v6_fixed_dir']}",
        f"旧成功目录：{data['old_success_dir']}",
        f"测试：{data['tests']}",
    ]
    for item in items:
        y = box(draw, y, item, GREEN if item.startswith("测试") else BLUE)
    return img


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (PAGE_W, PAGE_H), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([36, 36, PAGE_W - 36, PAGE_H - 36], fill=PANEL, outline=LINE, width=2)
    return img, draw


def title(draw: ImageDraw.ImageDraw, heading: str, sub: str) -> int:
    y = 80
    y = draw_text(draw, heading, MARGIN, y, 42, TEXT, PAGE_W - 2 * MARGIN, bold=True)
    y = draw_text(draw, sub, MARGIN, y + 8, 24, MUTED, PAGE_W - 2 * MARGIN)
    draw.line([MARGIN, y + 24, PAGE_W - MARGIN, y + 24], fill=LINE, width=2)
    return y + 58


def box(draw: ImageDraw.ImageDraw, y: int, text: str, color: tuple[int, int, int]) -> int:
    h = 124
    draw.rounded_rectangle([MARGIN, y, PAGE_W - MARGIN, y + h], radius=10, fill=(248, 250, 252), outline=LINE, width=1)
    draw.rectangle([MARGIN, y, MARGIN + 10, y + h], fill=color)
    draw_text(draw, text, MARGIN + 28, y + 20, 22, TEXT, PAGE_W - 2 * MARGIN - 56)
    return y + h + 18


def mini(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, text: str) -> int:
    h = 120
    draw.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=(248, 250, 252), outline=LINE, width=1)
    draw_text(draw, text, x + 18, y + 16, 17, TEXT, w - 36)
    return y + h + 12


def draw_text(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, size: int, color: tuple[int, int, int], max_width: int, *, bold: bool = False) -> int:
    font = font_for(size, bold=bold)
    line_h = int(size * 1.38)
    for para in str(text or "").split("\n"):
        for line in wrap(draw, para, font, max_width):
            draw.text((x, y), line, font=font, fill=color)
            y += line_h
    return y


def wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    out: list[str] = []
    for chunk in textwrap.wrap(str(text), width=70, break_long_words=False) or [""]:
        line = ""
        for ch in chunk:
            candidate = line + ch
            if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
                line = candidate
            else:
                if line:
                    out.append(line)
                line = ch
        if line:
            out.append(line)
    return out


def font_for(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold and FONT_BOLD.exists() else FONT_REGULAR
    if not path.exists():
        path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


if __name__ == "__main__":
    main()
