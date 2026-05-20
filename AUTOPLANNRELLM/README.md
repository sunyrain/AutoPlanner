# AUTOPLANNRELLM

AUTOPLANNRELLM is an independent experimental system layered on top of the
current AutoPlanner route-tree runtime. It keeps the same proposal providers,
stock checker, route-tree search, reservoir fallback, and cost scoring, but adds
two DeepSeek-mediated changes:

1. The route-tree selection step can ask a DeepSeek agent to rank open leaves
   and candidate actions.
2. Each proposal pool may receive at most one additional DeepSeek-suggested
   retrosynthetic candidate with source `llm_deepseek`.

The integration is opt-in:

```bash
export DEEPSEEK_API_KEY=...
export AUTOPLANNRELLM_ENABLE=1
export AUTOPLANNRELLM_LLM_SELECTION=1
export AUTOPLANNRELLM_ADD_LLM_CANDIDATE=1
export AUTOPLANNRELLM_CACHE=results/shared/autoplannrellm/deepseek_cache.jsonl
```

Example benchmark invocation:

```bash
PYTHONPATH=. python -m AUTOPLANNRELLM.runner -- \
  --bench data/benchmark_v2_100.json \
  --output results/shared/autoplannrellm/full100_A/run.json \
  --model results/shared/skeleton_inpainter/best.pt \
  --search-mode route_tree \
  --check-stock \
  --workers 2 \
  --device cpu \
  --n-results 5 \
  --n-candidates-per-skeleton 1 \
  --skeleton-samples 2 \
  --trace-output results/shared/autoplannrellm/full100_A/run_trace.jsonl \
  --log-dir results/shared/autoplannrellm/full100_A/parallel_logs \
  --extra-env AUTOPLANNER_ENABLE_ROUTE_TREE_PLANNER=1
```

If `DEEPSEEK_API_KEY` is absent or the API fails, the controller falls back to
the underlying AutoPlanner runtime and records the fallback reason in route-tree
diagnostics. LLM output cannot assert stock availability, yields, enzyme
availability, or reaction conditions unless those fields already appear in the
input.
