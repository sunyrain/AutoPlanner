from __future__ import annotations

import math
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
WIDE = (2400, 1350)


PALETTE = {
    "ink": "#172026",
    "muted": "#586672",
    "line": "#9aa8b3",
    "bg": "#f8fafb",
    "panel": "#ffffff",
    "codex": "#2f80ed",
    "tool": "#10a37f",
    "validator": "#6f42c1",
    "artifact": "#f2994a",
    "accept": "#239b56",
    "reject": "#c0392b",
    "partial": "#b7791f",
    "gray": "#edf2f5",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


F_TITLE = font(48, True)
F_SUBTITLE = font(25)
F_H = font(29, True)
F_BODY = font(22)
F_SMALL = font(18)
F_TAG = font(17, True)


def round_rect(draw: ImageDraw.ImageDraw, box, radius=24, fill="#fff", outline="#d9e1e7", width=3):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def shadowed_box(draw: ImageDraw.ImageDraw, box, radius=28, fill="#fff", outline="#d9e1e7", width=3):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0 + 8, y0 + 10, x1 + 8, y1 + 10), radius=radius, fill="#d9e3ea")
    round_rect(draw, box, radius=radius, fill=fill, outline=outline, width=width)


def text(draw: ImageDraw.ImageDraw, xy, value: str, fnt, fill=None, anchor=None):
    draw.text(xy, value, font=fnt, fill=fill or PALETTE["ink"], anchor=anchor)


def wrapped_text(draw: ImageDraw.ImageDraw, box, value: str, fnt, fill=None, line_spacing=7):
    x0, y0, x1, _ = box
    max_width = x1 - x0
    words = value.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if draw.textbbox((0, 0), candidate, font=fnt)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    y = y0
    for line in lines:
        draw.text((x0, y), line, font=fnt, fill=fill or PALETTE["ink"])
        y += fnt.size + line_spacing
    return y


def arrow(draw: ImageDraw.ImageDraw, start, end, fill="#52616d", width=5, head=18):
    x0, y0 = start
    x1, y1 = end
    draw.line((x0, y0, x1, y1), fill=fill, width=width)
    ang = math.atan2(y1 - y0, x1 - x0)
    pts = [
        (x1, y1),
        (x1 - head * math.cos(ang - math.pi / 6), y1 - head * math.sin(ang - math.pi / 6)),
        (x1 - head * math.cos(ang + math.pi / 6), y1 - head * math.sin(ang + math.pi / 6)),
    ]
    draw.polygon(pts, fill=fill)


def tag(draw, xy, label, color):
    x, y = xy
    pad_x, pad_y = 13, 7
    bbox = draw.textbbox((0, 0), label, font=F_TAG)
    w = bbox[2] + pad_x * 2
    h = bbox[3] - bbox[1] + pad_y * 2
    draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill=color)
    draw.text((x + pad_x, y + pad_y - 1), label, font=F_TAG, fill="white")
    return x + w


def draw_molecule_icon(draw, cx, cy, color):
    pts = []
    r = 28
    for i in range(6):
        a = math.pi / 6 + i * math.pi / 3
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    draw.line(pts + [pts[0]], fill=color, width=4)
    for x, y in pts[::2]:
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)


def title_block(draw, title, subtitle):
    text(draw, (90, 58), title, F_TITLE)
    text(draw, (92, 122), subtitle, F_SUBTITLE, PALETTE["muted"])


def figure1():
    img = Image.new("RGB", WIDE, PALETTE["bg"])
    draw = ImageDraw.Draw(img)
    title_block(
        draw,
        "Codex-driven retrosynthesis agent",
        "Agency is separated from authority: Codex chooses and curates; validators decide route status.",
    )

    lane_specs = [
        (185, 370, "Codex agency layer", PALETTE["codex"]),
        (470, 655, "Tool execution layer", PALETTE["tool"]),
        (760, 1045, "Deterministic authority layer", PALETTE["validator"]),
        (1115, 1255, "Audit trail and verdicts", PALETTE["artifact"]),
    ]
    for y0, y1, label, color in lane_specs:
        draw.rounded_rectangle((70, y0, 2330, y1), radius=34, fill="#ffffff", outline="#dfe7ed", width=2)
        draw.rectangle((70, y0, 82, y1), fill=color)
        text(draw, (105, y0 + 26), label, F_H, color)

    boxes = {
        "input": (120, 520, 360, 635),
        "preflight": (430, 790, 720, 935),
        "codex": (430, 230, 760, 345),
        "tools": (805, 510, 1130, 640),
        "lit": (1210, 510, 1535, 640),
        "compiler": (1620, 510, 1945, 640),
        "validators": (1995, 785, 2265, 935),
        "verdict": (1900, 1135, 2265, 1260),
    }
    labels = {
        "input": ("Target input", "name, SMILES, family hint"),
        "preflight": ("Deterministic preflight", "canonical identity, risk flags"),
        "codex": ("Codex workflow controller", "strategy + allowed tool plan"),
        "tools": ("ChemEnzy + route tools", "raw route proposals, frontier audit"),
        "lit": ("Literature research", "sources, structures, PDF/visual chains"),
        "compiler": ("Artifact compiler", "one-step rows, templates, subgoals"),
        "validators": ("Validator gates", "schema, RDKit, stock, provenance"),
        "verdict": ("Final verdict", "solved / partial / rejected"),
    }
    colors = {
        "input": "#f7fbff",
        "preflight": "#f3f5ff",
        "codex": "#eef6ff",
        "tools": "#eafaf6",
        "lit": "#fff6e8",
        "compiler": "#f9f1ff",
        "validators": "#f2edff",
        "verdict": "#fff8e7",
    }
    for key, box in boxes.items():
        shadowed_box(draw, box, fill=colors[key])
        h, b = labels[key]
        text(draw, (box[0] + 22, box[1] + 20), h, F_H)
        wrapped_text(draw, (box[0] + 22, box[1] + 61, box[2] - 20, box[3] - 12), b, F_BODY, PALETTE["muted"])

    # Main flow arrows.
    arrow(draw, (360, 578), (430, 858), PALETTE["line"])
    arrow(draw, (720, 858), (805, 578), PALETTE["line"])
    arrow(draw, (1130, 575), (1210, 575), PALETTE["line"])
    arrow(draw, (1535, 575), (1620, 575), PALETTE["line"])
    arrow(draw, (1945, 575), (1995, 858), PALETTE["line"])
    arrow(draw, (2130, 935), (2130, 1135), PALETTE["line"])

    # Codex control links.
    arrow(draw, (595, 345), (595, 790), PALETTE["codex"], width=4)
    arrow(draw, (760, 288), (970, 510), PALETTE["codex"], width=4)
    arrow(draw, (760, 288), (1375, 510), PALETTE["codex"], width=4)
    arrow(draw, (760, 288), (1785, 510), PALETTE["codex"], width=4)

    # Authority feedback links.
    # Validator feedback is intentionally compact: detailed failure feedback is
    # shown in Figure 2, while this panel emphasizes authority boundaries.
    arrow(draw, (1995, 850), (1810, 640), PALETTE["validator"], width=4)
    arrow(draw, (1995, 905), (1130, 640), PALETTE["validator"], width=4)

    tag(draw, (845, 708), "No raw LLM reaction injection", PALETTE["reject"])
    tag(draw, (1235, 708), "No LLM-only solved claim", PALETTE["reject"])
    tag(draw, (1630, 708), "No production KB write without gate", PALETTE["reject"])

    for x, y, c in [(250, 575, PALETTE["codex"]), (1000, 575, PALETTE["tool"]), (1380, 575, PALETTE["artifact"]), (2130, 860, PALETTE["validator"])]:
        draw_molecule_icon(draw, x, y - 22, c)

    img.save(FIG_DIR / "figure1_architecture.png", quality=95)


def figure2():
    img = Image.new("RGB", WIDE, PALETTE["bg"])
    draw = ImageDraw.Draw(img)
    title_block(draw, "End-to-end workflow and decision gates", "Each stage writes machine-readable artifacts; only validators emit the final route state.")

    stages = [
        ("1", "Target intake", "target_input.json", PALETTE["codex"]),
        ("2", "Preflight", "preflight.json", PALETTE["validator"]),
        ("3", "Codex plan", "workflow_plan.json", PALETTE["codex"]),
        ("4", "ChemEnzy search", "raw routes", PALETTE["tool"]),
        ("5", "Route audit", "frontier + failure feedback", PALETTE["validator"]),
        ("6", "Literature extraction", "sources + exact steps", PALETTE["artifact"]),
        ("7", "Compile handoff", "templates + subgoals", PALETTE["tool"]),
        ("8", "Final validation", "verdict.json", PALETTE["validator"]),
    ]
    x0, y0, bw, bh, gap = 155, 250, 360, 145, 75
    centers = []
    for i, (num, h, b, color) in enumerate(stages):
        row = 0 if i < 4 else 1
        col = i if i < 4 else 7 - i
        x = x0 + col * (bw + gap)
        y = y0 + row * 265
        shadowed_box(draw, (x, y, x + bw, y + bh), fill="#ffffff")
        draw.ellipse((x + 18, y + 18, x + 58, y + 58), fill=color)
        text(draw, (x + 38, y + 27), num, F_TAG, "white", anchor="ma")
        text(draw, (x + 70, y + 20), h, F_H)
        wrapped_text(draw, (x + 22, y + 74, x + bw - 18, y + bh - 8), b, F_SMALL, PALETTE["muted"], line_spacing=5)
        centers.append((x + bw / 2, y + bh / 2))

    for a, b in zip(centers[:3], centers[1:4]):
        arrow(draw, (a[0] + 122, a[1]), (b[0] - 122, b[1]), PALETTE["line"], width=4)
    arrow(draw, (centers[3][0], centers[3][1] + 74), (centers[4][0], centers[4][1] - 74), PALETTE["line"], width=4)
    bottom_path = [centers[4], centers[5], centers[6], centers[7]]
    for a, b in zip(bottom_path, bottom_path[1:]):
        arrow(draw, (a[0] - 122, a[1]), (b[0] + 122, b[1]), PALETTE["line"], width=4)

    # Follow-up loop from validation back to Codex planning.
    draw.arc((470, 420, 2020, 795), 15, 172, fill=PALETTE["codex"], width=4)
    arrow(draw, (635, 485), (585, 398), PALETTE["codex"], width=4)
    text(draw, (820, 653), "needs_followup: narrower extraction, guided rerun or subgoal search", F_BODY, PALETTE["codex"])

    gate_y = 920
    gate_boxes = [
        (165, gate_y, 555, gate_y + 180, "Reject", "fake closure, target drift, boundary violation", PALETTE["reject"]),
        (690, gate_y, 1080, gate_y + 180, "Partial", "literature anchor or executable segment without stock closure", PALETTE["partial"]),
        (1215, gate_y, 1605, gate_y + 180, "Needs follow-up", "subgoals, narrower extraction, guided rerun", PALETTE["codex"]),
        (1740, gate_y, 2130, gate_y + 180, "Solved", "stock-closed and validator-accepted route", PALETTE["accept"]),
    ]
    for x1, y1, x2, y2, h, b, color in gate_boxes:
        shadowed_box(draw, (x1, y1, x2, y2), fill="#fff")
        draw.rectangle((x1, y1, x1 + 12, y2), fill=color)
        text(draw, (x1 + 30, y1 + 24), h, F_H, color)
        wrapped_text(draw, (x1 + 30, y1 + 74, x2 - 24, y2 - 16), b, F_BODY, PALETTE["muted"])

    arrow(draw, (centers[7][0], centers[7][1] + 72), (1280, gate_y - 20), PALETTE["validator"], width=5)
    text(draw, (112, 1222), "Safety invariant: Codex can choose tools and draft artifacts, but deterministic validators decide route status.", F_SUBTITLE, PALETTE["ink"])
    img.save(FIG_DIR / "figure2_workflow_gates.png", quality=95)


def figure3():
    img = Image.new("RGB", WIDE, PALETTE["bg"])
    draw = ImageDraw.Draw(img)
    title_block(draw, "Artifact contracts and validation authority", "Structured files make every route claim traceable from source evidence to final verdict.")

    columns = [
        (110, "Run context", PALETTE["codex"], ["target input", "preflight", "workflow plan", "decision trace"]),
        (610, "Proposal artifacts", PALETTE["tool"], ["raw ChemEnzy routes", "route verifier report", "frontier report", "failure feedback"]),
        (1110, "Evidence artifacts", PALETTE["artifact"], ["literature sources", "validated compounds", "source-detail records", "visual/PDF chain"]),
        (1610, "Compiled artifacts", PALETTE["validator"], ["downstream bundle", "template plugin", "route-expansion tasks", "self-evo staging"]),
    ]
    box_w = 400
    for x, title, color, items in columns:
        shadowed_box(draw, (x, 250, x + box_w, 805), fill="#fff")
        draw.rectangle((x, 250, x + box_w, 260), fill=color)
        text(draw, (x + 25, 285), title, F_H, color)
        y = 350
        for item in items:
            draw.rounded_rectangle((x + 24, y, x + box_w - 24, y + 72), radius=16, fill="#f5f8fa", outline="#dce5eb", width=2)
            wrapped_text(draw, (x + 42, y + 17, x + box_w - 38, y + 66), item, F_BODY, PALETTE["ink"], line_spacing=3)
            y += 92

    for x in [510, 1010, 1510]:
        arrow(draw, (x, 530), (x + 90, 530), PALETTE["line"], width=5)

    gate = (485, 905, 1915, 1165)
    shadowed_box(draw, gate, fill="#ffffff")
    text(draw, (gate[0] + 35, gate[1] + 32), "Validation gates", F_H, PALETTE["validator"])
    checks = [
        ("Schema", "required keys and allowed tools"),
        ("Structure", "RDKit-valid SMILES and target identity"),
        ("Route", "stock closure, atom jumps, terminal audit"),
        ("Evidence", "source refs, exact relation, condition support"),
        ("Policy", "no raw reaction injection, no unsafe KB write"),
    ]
    start_x = gate[0] + 35
    for i, (h, b) in enumerate(checks):
        x = start_x + i * 270
        draw.rounded_rectangle((x, gate[1] + 95, x + 235, gate[1] + 210), radius=18, fill="#f3f0ff", outline="#d8caf7", width=2)
        text(draw, (x + 18, gate[1] + 113), h, F_TAG, PALETTE["validator"])
        wrapped_text(draw, (x + 18, gate[1] + 143, x + 218, gate[1] + 202), b, F_SMALL, PALETTE["muted"], line_spacing=3)

    arrow(draw, (1810, 805), (1810, 905), PALETTE["validator"], width=5)
    tag(draw, (2010, 365), "audit evidence", PALETTE["artifact"])
    tag(draw, (2010, 430), "machine-readable", PALETTE["tool"])
    tag(draw, (2010, 495), "verdict-gated", PALETTE["validator"])
    img.save(FIG_DIR / "figure3_artifact_contracts.png", quality=95)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure1()
    figure2()
    figure3()


if __name__ == "__main__":
    main()
