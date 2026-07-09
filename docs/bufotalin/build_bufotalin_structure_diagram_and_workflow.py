"""Generate a standalone bufotalin retrosynthesis structure diagram and workflow note."""
from __future__ import annotations

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
REPORT_DIR = ROOT / "docs" / "bufotalin" / "report_20260607"
DATA_PATH = REPORT_DIR / "bufotalin_retrosynthesis_report_data.json"
OUT_DIR = REPORT_DIR
ASSET_DIR = OUT_DIR / "assets"
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

NODE_ORDER = ["bufotalin", "33", "32", "31", "30", "22", "14", "20", "19", "28", "27", "26", "23", "25", "24", "11"]
SNAKE_POSITIONS = {
    "bufotalin": (0, 0),
    "33": (0, 1),
    "32": (0, 2),
    "31": (0, 3),
    "30": (1, 3),
    "22": (1, 2),
    "14": (1, 1),
    "20": (1, 0),
    "19": (2, 0),
    "28": (2, 1),
    "27": (2, 2),
    "26": (2, 3),
    "23": (3, 3),
    "25": (3, 2),
    "24": (3, 1),
    "11": (3, 0),
}

TEXT = (20, 28, 42)
MUTED = (78, 93, 110)
ACCENT = (15, 118, 110)
LINE = (190, 200, 214)
PANEL = (255, 255, 255)
BG = (246, 248, 251)
WARN = (154, 91, 0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    png_path = draw_structure_route(data)
    pdf_path = png_to_pdf(png_path)
    workflow_path = write_workflow_note(data, png_path=png_path, pdf_path=pdf_path)
    audit = {
        "schema_version": "bufotalin_structure_diagram_workflow_audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "accepted": png_path.exists() and pdf_path.exists() and workflow_path.exists(),
        "structure_diagram_png": str(png_path),
        "structure_diagram_pdf": str(pdf_path),
        "workflow_markdown": str(workflow_path),
        "node_count": len(NODE_ORDER),
        "retrosynthetic_step_count": len(data.get("reverse_steps") or []),
        "one_step_row_count": data.get("harness_summary", {}).get("one_step_row_count"),
        "chain_probe_accepted": data.get("harness_summary", {}).get("chain_probe_accepted"),
    }
    audit_path = OUT_DIR / "bufotalin_structure_diagram_workflow_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


def draw_structure_route(data: dict[str, Any]) -> Path:
    compounds = {str(row["label"]): dict(row) for row in data.get("compounds") or []}
    step_by_edge = build_step_by_edge(data)
    width, height = 2300, 1760
    margin_x, margin_y = 80, 170
    node_w, node_h = 460, 270
    gap_x, gap_y = 80, 105
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    draw.rectangle((0, 0, width, 26), fill=ACCENT)
    draw.text((margin_x, 54), "Bufotalin retrosynthesis structure route", font=font(48, bold=True), fill=TEXT)
    draw.text(
        (margin_x, 112),
        "Validated source-detail literature baseline: bufotalin -> 33 -> 32 -> 31 -> 30 -> 22 -> 14 -> 20 -> 19 -> 28 -> 27 -> 26 -> 23 -> 25 -> 24 -> 11",
        font=font(22),
        fill=MUTED,
    )

    centers: dict[str, tuple[int, int]] = {}
    for index, label in enumerate(NODE_ORDER, start=1):
        row, col = SNAKE_POSITIONS[label]
        x = margin_x + col * (node_w + gap_x)
        y = margin_y + row * (node_h + gap_y)
        centers[label] = (x + node_w // 2, y + node_h // 2)
        draw_node(draw, canvas, label, index, compounds.get(label, {}), x, y, node_w, node_h)

    for idx in range(len(NODE_ORDER) - 1):
        src = NODE_ORDER[idx]
        dst = NODE_ORDER[idx + 1]
        draw_connector(draw, centers[src], centers[dst], step_by_edge.get((src, dst), {}), idx + 1)

    footer_y = height - 130
    draw.rounded_rectangle((margin_x, footer_y, width - margin_x, footer_y + 82), radius=18, fill=(238, 250, 248), outline=(148, 210, 204), width=2)
    footer = (
        "Policy: literature path is a high-weight baseline, not a mandatory replacement. "
        "ChemEnzy exploration is retained, but raw_solved must pass deterministic verifier and stock closure."
    )
    draw_wrapped(draw, footer, margin_x + 24, footer_y + 18, width - 2 * margin_x - 48, font(22), TEXT)

    out = OUT_DIR / "bufotalin_retrosynthesis_structure_route_20260607.png"
    canvas.save(out, quality=95)
    return out


def build_step_by_edge(data: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    mapping: dict[tuple[str, str], dict[str, Any]] = {}
    for step in data.get("reverse_steps") or []:
        label = str(step.get("retrosynthetic_label") or "")
        left_right = label.split(":", 1)[0].strip()
        if "->" not in left_right:
            continue
        forward_left, forward_right = [part.strip() for part in left_right.split("->", 1)]
        # Source labels are synthetic forward labels, so the retrosynthetic edge is product -> reactant.
        src = "bufotalin" if forward_right == "bufotalin" else forward_right
        dst = forward_left
        mapping[(src, dst)] = dict(step)
    return mapping


def draw_node(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    label: str,
    index: int,
    compound: dict[str, Any],
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    draw.rounded_rectangle((x, y, x + w, y + h), radius=20, fill=PANEL, outline=LINE, width=2)
    draw.text((x + 18, y + 16), f"{index}. {label}", font=font(28, bold=True), fill=TEXT)
    formula = str(compound.get("formula") or "")
    draw.text((x + 18, y + 52), formula, font=font(19), fill=MUTED)
    smiles = str(compound.get("smiles") or "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        img = Draw.MolToImage(mol, size=(380, 178), legend="")
        canvas.paste(img.convert("RGB"), (x + 40, y + 78))
    mw = compound.get("exact_mw")
    ha = compound.get("heavy_atom_count")
    draw.text((x + 18, y + h - 34), f"MW {mw} | heavy atoms {ha}", font=font(17), fill=MUTED)


def draw_connector(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    step: dict[str, Any],
    index: int,
) -> None:
    x0, y0 = start
    x1, y1 = end
    if abs(y0 - y1) < 20:
        direction = 1 if x1 > x0 else -1
        sx = x0 + direction * 240
        ex = x1 - direction * 240
        y = y0
        draw_arrow(draw, (sx, y), (ex, y))
        tx = min(sx, ex) + 10
        ty = y - 66
        draw_arrow_label(draw, tx, ty, step, index, width=abs(ex - sx) - 20)
    else:
        direction_y = 1 if y1 > y0 else -1
        x = x0
        sy = y0 + direction_y * 145
        ey = y1 - direction_y * 145
        draw_arrow(draw, (x, sy), (x, ey))
        draw_arrow_label(draw, x + 18, (sy + ey) // 2 - 44, step, index, width=260)


def draw_arrow_label(draw: ImageDraw.ImageDraw, x: int, y: int, step: dict[str, Any], index: int, *, width: int) -> None:
    cond = dict(step.get("condition") or {})
    reagent = str(cond.get("reagent") or "").replace("room temperature", "rt")
    yield_text = str(cond.get("reported_yield") or "")
    text = f"{index}. {short_step_name(step)}"
    sub = " | ".join(part for part in [reagent, yield_text] if part)
    draw.rounded_rectangle((x, y, x + max(160, width), y + 54), radius=10, fill=(255, 252, 244), outline=(233, 210, 165), width=1)
    draw.text((x + 8, y + 6), ellipsize(text, 34), font=font(16, bold=True), fill=TEXT)
    draw.text((x + 8, y + 30), ellipsize(sub, 42), font=font(14), fill=WARN)


def short_step_name(step: dict[str, Any]) -> str:
    label = str(step.get("retrosynthetic_label") or "")
    if ":" in label:
        return label.split(":", 1)[1].strip()
    return label


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    x0, y0 = start
    x1, y1 = end
    draw.line((x0, y0, x1, y1), fill=TEXT, width=4)
    if abs(x1 - x0) >= abs(y1 - y0):
        direction = 1 if x1 > x0 else -1
        draw.polygon([(x1, y1), (x1 - direction * 18, y1 - 10), (x1 - direction * 18, y1 + 10)], fill=TEXT)
    else:
        direction = 1 if y1 > y0 else -1
        draw.polygon([(x1, y1), (x1 - 10, y1 - direction * 18), (x1 + 10, y1 - direction * 18)], fill=TEXT)


def png_to_pdf(png_path: Path) -> Path:
    out = OUT_DIR / "bufotalin_retrosynthesis_structure_route_20260607.pdf"
    image = Image.open(png_path)
    doc = fitz.open()
    page = doc.new_page(width=842, height=595)
    page.insert_image(page.rect, filename=str(png_path))
    doc.set_metadata(
        {
            "title": "Bufotalin retrosynthesis structure route",
            "author": "AutoPlanner Codex harness",
            "subject": "Validated bufotalin source-detail retrosynthesis structure diagram",
        }
    )
    doc.save(out, deflate=True, garbage=4)
    doc.close()
    image.close()
    return out


def write_workflow_note(data: dict[str, Any], *, png_path: Path, pdf_path: Path) -> Path:
    path = OUT_DIR / "bufotalin_full_workflow_20260607.md"
    h = data["harness_summary"]
    c = data["chemenzy_summary"]
    artifact_refs = data["artifact_refs"]
    steps = data["reverse_steps"]
    lines = [
        "# Bufotalin 全流程逆合成工作流",
        "",
        f"生成时间: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 当前结论",
        "",
        f"- 文献 source-detail 链: `{data['route_summary']['step_count']}` 步，完整接到 compound `{data['route_summary']['final_reactant_name']}`。",
        f"- downstream compiler: `compiled_accepted={h['compiled_accepted']}`, `executable_status={h['executable_status']}`。",
        f"- literature one-step plugin: `one_step_row_count={h['one_step_row_count']}`, `plugin_max_added={h['plugin_max_added']}`。",
        f"- chain probe: `accepted={h['chain_probe_accepted']}`, `terminal_reached={h['chain_probe_terminal_reached']}`。",
        f"- ChemEnzy smoke: `raw_solved={c['raw_solved']}`, 但 verifier accepted=`{c['verifier_accepted']}`，原因: `{', '.join(c['verifier_reasons'])}`。",
        "",
        "## 新增结构图",
        "",
        f"- PNG: [{png_path.name}]({png_path.name})",
        f"- PDF: [{pdf_path.name}]({pdf_path.name})",
        "",
        "## Bufotalin 全流程",
        "",
        "1. **目标输入与 preflight**",
        "   - 输入 target name/smiles/family hint。",
        "   - RDKit 校验 SMILES、heavy atom count、复杂度和初始风险标签。",
        "   - 复杂天然产物默认不能只信 ChemEnzy raw solved，后续必须经过 verifier 和文献/模板证据闭环。",
        "",
        "2. **复用或跳过前段 ChemEnzy**",
        "   - bufotalin 这个阶段不需要重复跑最初 ChemEnzy baseline；可复用已有 raw result、route audit 和 verifier feedback。",
        "   - 关键判定: raw_solved=true 只代表 native search 返回了 stock-closed routes，不等于 verified solved。",
        "",
        "3. **开放文献/结构研究队列**",
        "   - 从 literature_sources、retrieval prefetch、source_material_locator 和 open_structure_research artifacts 里找 exact-target / exact-intermediate source。",
        "   - 本 case 的关键 DOI 是 `10.1016/j.tet.2025.134610`；早期 advisory DOI/anchor 只作为方向性先验。",
        "",
        "4. **PDF/图像证据提取**",
        "   - 渲染本地全文 PDF 页面和 scheme crops。",
        "   - source-detail worker 只抽结构化字段: compound label、product_smiles、reactant_smiles、condition_candidate、source_locator、source_excerpt。",
        "   - 不保存全文实验步骤，不写 production KB。",
        "",
        "5. **连续中间体 SMILES 识别与校验**",
        "   - Codex/vision 按论文 scheme 连续识别 11、24、25、23、26、27、28、19、20、14、22、30、31、32、33、bufotalin。",
        "   - RDKit 校验所有候选 SMILES、formula、exact mass/heavy atoms，并检查链连续性。",
        "   - 论文正向链 `11 -> ... -> bufotalin` 在 harness 内倒成逆合成链 `bufotalin -> ... -> 11`。",
        "",
        "6. **生成 source_detail_curator_records.v1**",
        "   - 通过校验的结构链进入 curator records。",
        "   - provenance=`codex_source_text_translation`，structure_derivation 记录 source_locator、confidence 和 tool_checks。",
        "   - 默认 `main_reactant_only=true`，例如 14 -> 22 的 2-pyrone 偶联不会把辅底物当作后续 steroid child target。",
        "",
        "7. **source-detail resolution**",
        "   - resolver 消费 curator records，产出 source_detail_route_steps。",
        f"   - 当前 bufotalin 结果: `source_detail_route_step_count={h['source_detail_route_step_count']}`, `resolution_gap_count={h['resolution_gap_count']}`。",
        "",
        "8. **downstream compiler**",
        "   - compiler 把 exact source_detail_route_steps 晋级为 executable literature one-step rows。",
        "   - 这些 rows 带 source_ref/evidence_refs/condition_candidate/applicability，不是 advisory template。",
        f"   - 当前状态: `one_step_row_count={h['one_step_row_count']}`, `executable_status={h['executable_status']}`。",
        "",
        "9. **literature_template_plugin 接入 ChemEnzy**",
        "   - guided rerun 可以把 compiled `literature_template_plugin` 作为 one-step source。",
        "   - 插件负责在匹配 product_smiles 时返回文献 exact reactant row。",
        "   - 子目标也可由 compiled child_targets 进入 route expansion。",
        "",
        "10. **ChemEnzy rerun / route expansion / verifier**",
        "   - ChemEnzy 保留探索能力，但必须经过 deterministic verifier。",
        "   - 当前 15-row smoke run 仍被 verifier 拒绝，说明 native route fake-closed 或 large atom jump，不能当作 solved。",
        "",
        "11. **hybrid route set 与报告**",
        "   - 文献链作为 high-weight baseline；ChemEnzy 探索路线作为 alternative/exploratory candidates。",
        "   - 报告输出 PDF、Markdown、report_data JSON、completion audit 和结构路线图。",
        "",
        "## 已验证逆合成链",
        "",
        "| # | step_id | retrosynthetic edge | key condition | yield |",
        "|---:|---|---|---|---|",
    ]
    for step in steps:
        cond = step.get("condition") or {}
        condition = "; ".join(str(cond.get(key) or "") for key in ("reagent", "solvent", "temperature", "duration") if cond.get(key))
        lines.append(
            f"| {step['index']} | `{step['step_id']}` | {step['retrosynthetic_label']} | "
            f"{condition} | {cond.get('reported_yield', '')} |"
        )
    lines.extend(
        [
            "",
            "## 关键 artifacts",
            "",
        ]
    )
    for key, value in artifact_refs.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## 工程接口",
            "",
            "- `extract_pdf_literature_structures`: 生成 PDF/page/crop 证据索引。",
            "- `validate_literature_intermediate_chain`: 校验视觉/来源结构候选链并倒成逆合成链。",
            "- `build_source_detail_curator_records`: 生成 curator records 并触发 source-detail resolution / downstream compile。",
            "- `compile_source_detail_chain_route`: 从 one-step rows unroll 文献链并审计 terminal。",
            "- `compile_hybrid_route_set`: 汇总文献 baseline 与 ChemEnzy exploratory routes。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size=size)


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    width: int,
    fnt: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> int:
    lines = []
    current = ""
    for word in str(text).split():
        trial = word if not current else f"{current} {word}"
        if draw.textlength(trial, font=fnt) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    line_h = int(fnt.size * 1.35)
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += line_h + 6
    return y


def ellipsize(text: str, max_chars: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)] + "…"


if __name__ == "__main__":
    main()
