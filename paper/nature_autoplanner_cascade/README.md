# Nature-Style Draft: AutoPlanner-Cascade

This directory contains a self-contained Nature-style paper draft generated from
the local AutoPlanner-Cascade state on 2026-05-12. The manuscript uses the local
TeX `nature.cls` class and restores embedded figures so the exported PDF is
presentation-ready.

## Files

- `main.tex` - Nature-style LaTeX manuscript draft.
- `figures/generated/figure1_image2_background.png` - image2-generated no-text
  main-figure background.
- `figures/generated/figure1_prompt.txt` - prompt used for the image2
  background.
- `scripts/make_main_figure.py` - deterministic label overlay for Figure 1.
- `figures/figure1_cascade_native.png` - final labelled Figure 1.
- `tables/full100_metrics.tex` - full100 benchmark comparison table.
- `build/autoplanner_cascade_nature_draft.pdf` - exported draft PDF when built.

## Build

```bash
python scripts/make_main_figure.py
pdflatex -interaction=nonstopmode -halt-on-error -output-directory build main.tex
cp build/main.pdf build/autoplanner_cascade_nature_draft.pdf
```

If a TeX toolchain is unavailable, use the generated PNG figure and `main.tex`
as the source of truth until TeX is installed.
