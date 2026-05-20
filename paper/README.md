# Paper Artifacts

This folder contains the working manuscript and simulated review cycle for
AutoPlanner.

- `manuscript.md` - current full paper draft.
- `review_cycle.md` - three simulated AI-area reviewers, author responses, and
  revision checklist.
- `revision_log.md` - concrete changes made during the iterative writing loop.
- `live_benchmark_results.md` - generated table from the full live benchmark
  summary.
- `../results/v2/prior_benchmark_deepseek_gpu10.md` - small real DeepSeek
  prior ablation used to bound LLM claims.

The manuscript is intentionally conservative about metrics. The current numbers
come from the 2026-05-04 dual-GPU run over the frozen 100-target benchmark. Raw
JSON route artifacts are kept as local generated results; the committed paper
table records the reproducible summary.
