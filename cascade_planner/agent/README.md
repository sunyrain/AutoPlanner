# AutoPlanner Agent Layer

This package contains the optional LLM-prior layer.

Safety rules:

- API keys are read only from runtime environment variables or explicit function
  arguments.
- API keys are never written to files, prompts, logs, benchmark artifacts, or
  route JSON.
- LLM outputs are strategic priors or critiques only.
- LLM outputs cannot mark stock availability, invent enzyme availability,
  invent yields, invent conditions, or override reaction validators.
- If the LLM provider is unavailable, deterministic motif-based priors are used.

Current entry points:

```bash
python -m cascade_planner.agent.cli prior --target "CC(O)CCc1ccccc1"
python -m cascade_planner.agent.cli critique --input route_payload.json
python -m cascade_planner.agent.cli failure-risk --input route_payload.json
python -m cascade_planner.agent.cli check --provider deepseek --strict
```

DeepSeek can be used with:

```bash
read -rsp "DeepSeek API key: " DEEPSEEK_API_KEY && export DEEPSEEK_API_KEY && echo
python -m cascade_planner.agent.cli prior --target "CC(O)CCc1ccccc1" --provider deepseek
```

The default DeepSeek model is `deepseek-v4-flash`. Override with
`DEEPSEEK_MODEL=deepseek-v4-pro` if needed.

Do not commit `.env` files or API keys.

Benchmark integration:

```bash
python -m cascade_planner.cascadeboard.prior_benchmark \
  --providers none deterministic deepseek \
  --limit 10 --n-results 5 --skeleton-samples 10 --check-stock
```

Prior providers only rerank generated skeletons. They do not create accepted
reaction candidates or override route validators.

Current planning integration:

- `rerank`: no-agent or prior-guided skeleton ordering, followed by deterministic
  route scoring.
- `stock_aware`: bounded beam filling plus stock-aware terminal scoring and
  route reranking.
- `critic_control`: prior-guided skeleton ordering with iterative expansion; the
  deterministic route critic stops early only after filled, stock-compatible,
  condition-compatible, and cascade-compatible routes appear, or after the
  search budget is exhausted.

Failure-risk policy:

- `failure-risk` loads `results/shared/failure_classifier/pack_failure_classifier_20260507.pt`
  by default.
- It predicts likely planner bottlenecks such as generator misses, stock
  dead-ends, selector misses, and cascade condition failures.
- It only produces retry suggestions. It does not validate reactions and does
  not override Enzyformer, RetroChimera, stock checks, or route metrics.
- Live benchmark also supports `--search-mode policy_retry`, which runs a base
  AO* search, predicts failure risk, and executes one bounded retry only when
  the learned policy marks the retry as automatic-safe.
