#!/usr/bin/env python3
"""Compose the labelled Figure 1 from the generated image2 background."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = ROOT / "figures" / "generated" / "figure1_image2_background.png"
OUT = ROOT / "figures" / "figure1_cascade_native.png"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def _draw_label(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    panel: str,
    title: str,
    subtitle: str,
) -> None:
    dark = (31, 45, 51)
    teal = (33, 128, 117)
    muted = (86, 101, 107)
    draw.rounded_rectangle((x, y, x + width, y + 86), radius=18, fill=(255, 255, 255, 236), outline=(209, 224, 224), width=2)
    draw.ellipse((x + 16, y + 18, x + 62, y + 64), fill=(230, 246, 240), outline=(94, 168, 156), width=2)
    draw.text((x + 31, y + 25), panel, fill=teal, font=_font(FONT_BOLD, 24))
    draw.text((x + 80, y + 16), title, fill=dark, font=_font(FONT_BOLD, 22))
    draw.text((x + 80, y + 50), subtitle, fill=muted, font=_font(FONT, 16))


def main() -> None:
    image = Image.open(BACKGROUND).convert("RGBA")
    width, height = image.size
    scale = width / 1680
    canvas = Image.new("RGBA", (width, height + 180), (255, 255, 255, 255))
    canvas.alpha_composite(image, (0, 136))
    draw = ImageDraw.Draw(canvas)

    title_font = _font(FONT_BOLD, int(32 * scale))
    body_font = _font(FONT, int(21 * scale))
    small_font = _font(FONT, int(17 * scale))
    dark = (28, 42, 49)
    muted = (84, 101, 108)
    teal = (33, 128, 117)

    draw.text((42, 28), "AutoPlanner-Cascade: process-aware cascade program search", fill=dark, font=title_font)
    draw.text(
        (42, 78),
        "Retrosynthesis is optimized over molecule graph, process graph, condition graph and evidence graph.",
        fill=muted,
        font=body_font,
    )
    draw.rounded_rectangle((1178, 26, 1624, 84), radius=17, fill=(236, 247, 242), outline=(199, 225, 218), width=2)
    draw.text((1204, 44), "coverage-first controller + learned leaf policy", fill=teal, font=small_font)

    y = int(136 * scale)
    _draw_label(draw, int(28 * scale), y, int(360 * scale), "A", "Route-tree proposals", "ChemEnzy, templates, retrieval")
    _draw_label(draw, int(420 * scale), y, int(370 * scale), "B", "Coverage-aware search", "leaf policy; source scheduling")
    _draw_label(draw, int(820 * scale), y, int(440 * scale), "C", "Cascade program state", "stages, conditions, cofactors")
    _draw_label(draw, int(1292 * scale), y, int(350 * scale), "D", "Feasible plan set", "stock and process diagnostics")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(OUT, quality=95, optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()
