# Revision Log

## 2026-05-04 Draft 1

- Wrote complete manuscript structure.
- Reframed project around condition- and compatibility-aware cascade planning.
- Added related work positioning against AiZynthFinder, ASKCOS, Syntheseus,
  ChemEnzyRetroPlanner, AOT*, and chemistry agents.
- Added explicit metric discipline.
- Added full live benchmark protocol.
- Added three simulated reviewers and author responses.

## 2026-05-04 Draft 2 Infrastructure Revision

- Added `route_export.py` as the factual route JSON contract.
- Added deterministic `cascade_compatibility_success` and enzyme evidence
  diagnostics.
- Added `live_benchmark.py` sharding and merge support for dual-GPU runs.
- Added optional agent layer with deterministic prior, optional DeepSeek
  provider, and route JSON critic.
- Fixed multi-route skeleton filling so enzymatic skeleton slots use
  Enzyformer/v3 retrieval/EnzExpand instead of being forced through
  RetroChimera.

## 2026-05-04 Draft 3 Results Revision

- Re-ran the full 100-target dual-GPU benchmark after fixing enzymatic
  provenance.
- Inserted measured benchmark results into abstract and results.
- Added strict stock solve, candidate source breakdown, and per-domain metrics.
- Tightened limitations: compatibility is a deterministic screen, not an
  experimental success label.

## 2026-05-04 Draft 4 Review-Convergence Revision

- Added one accepted and one rejected route-level case study from the full live
  benchmark artifact.
- Ran a second simulated review round across ML benchmarking, retrosynthesis
  chemistry, and AI agent grounding reviewers.
- Reclassified LLM search lift as a future ablation rather than a current
  performance claim.
- Preserved the publishable claim boundary: AutoPlanner is a condition- and
  compatibility-aware cascade planning system, not a generic state-of-the-art
  retrosynthesis solver.

## 2026-05-04 Draft 5 Agent-Benchmark Deployment

- Connected structured agent priors to live planning as skeleton reranking only.
- Added no-agent, deterministic-prior, and DeepSeek-prior comparison entry
  point.
- Added runtime-only DeepSeek provider check; API keys are not stored in git,
  commands, benchmark artifacts, or reports.
- Ran a CPU smoke comparison over three frozen benchmark targets:
  no-agent and deterministic prior both solved the three-target smoke; DeepSeek
  was skipped because `DEEPSEEK_API_KEY` was not present in the shell
  environment.

## 2026-05-05 Draft 6 DeepSeek Ablation Revision

- Ran a real DeepSeek provider check with `resolved_source=deepseek` and no
  fallback.
- Added CPU smoke and GPU 10-target prior comparisons to the result artifacts.
- Inserted the DeepSeek ablation into the manuscript as a deployment and
  claim-boundary result.
- Updated the discussion and limitations: the current LLM prior is grounded but
  does not yet improve equal-budget route metrics.
- Added a third simulated re-review round after the DeepSeek ablation.

## Pending Future Experiments

- Run full 100-target equal-budget no-agent versus DeepSeek-prior ablation with
  cached skeleton sets or deterministic seeds.
- Upgrade LLM use from skeleton reranking to search-budget allocation and
  failure-conditioned resampling.
- Calibrate deterministic compatibility screens against more failed and
  successful experimental cascade outcomes.
- Enrich enzyme evidence with UniProt, sequence, organism, cofactor, and
  literature support.
