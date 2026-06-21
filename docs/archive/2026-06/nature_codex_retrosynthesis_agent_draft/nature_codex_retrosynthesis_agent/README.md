# Nature-style Codex Retrosynthesis Agent Manuscript

This directory contains a Nature-style LaTeX draft for the Codex-driven
AutoPlanner full-flow retrosynthesis agent.

## Files

- `main.tex`: manuscript source.
- `references.bib`: BibTeX references.
- `main.pdf`: compiled preview generated locally.
- `figures/figure1_architecture.png`: Codex/control/validator architecture.
- `figures/figure2_workflow_gates.png`: end-to-end workflow and verdict gates.
- `figures/figure3_artifact_contracts.png`: artifact contracts and validators.
- `image2_prompt_architecture.txt`: GPT Image 2 prompt for an AI-generated
  architecture background.
- `generate_image2_architecture.sh`: GPT Image 2 CLI command.
- `scripts/make_architecture_figures.py`: deterministic fallback figure renderer.

## Compile

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Format Note

The draft uses a portable `article` class with Nature-style ordering,
superscript numerical citations and `naturemag.bst`. The local TeX installation
has an old unofficial `nature.cls`, but not the current Springer Nature
`sn-jnl.cls`; migrate the preamble to the official submission template when the
target journal and template version are fixed.

## Figure Note

The user-requested image-2 path is prepared in `image2_prompt_architecture.txt`
and `generate_image2_architecture.sh`. During this drafting run,
`OPENAI_API_KEY` was not available in the shell, so the manuscript uses
deterministically rendered figures from `scripts/make_architecture_figures.py`.
After configuring credentials, run:

```bash
./generate_image2_architecture.sh
```

Then use the generated `figures/figure1_image2_background.png` as a background
or replacement asset if it passes visual inspection.

## Submission Placeholders

Before submission, replace the team placeholder with named authors,
affiliations, funding, contribution statements and a release-grade data/code
availability statement with commit hashes and packaged run artifacts.
