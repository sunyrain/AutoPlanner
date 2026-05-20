# Simulated Review Cycle

## Reviewers

### Reviewer 1: Machine Learning and Benchmarking

Summary: The paper has a clear system motivation, but the experimental claims
must be tightened. Skeleton-only and filled-route metrics must never be mixed.
The LLM component is currently under-evaluated.

Major concerns:

1. The abstract should not imply full benchmark success until the full live run
   is reported.
2. The benchmark must include equal-budget no-agent versus LLM-prior ablations.
3. Route recovery should be separated from cascade-feasibility scoring.
4. The paper needs a stronger statement of what is learned versus rule-based.

Required revision:

- Add a metric discipline section.
- Add explicit limitations.
- Defer LLM performance claims until ablation exists.

### Reviewer 2: Retrosynthesis and Chemistry

Summary: The novelty is not hybrid retrosynthesis itself. The paper becomes
interesting only if it emphasizes cascade compatibility, experimental
conditions, operation mode, and enzyme evidence.

Major concerns:

1. Do not claim competitors lack condition prediction entirely.
2. Clarify that ChemEnzyRetroPlanner overlaps with hybrid planning and tool
   orchestration.
3. Explain what cascade compatibility means chemically.
4. Enzyme evidence is currently too shallow if it only uses EC class.

Required revision:

- Reframe novelty around route-level cascade feasibility.
- Add a data requirements section or discussion.
- Avoid overclaiming enzyme recommendation.

### Reviewer 3: AI Agent Safety and Grounding

Summary: The paper uses LLMs responsibly, but the safety contract should be
more explicit. The route JSON contract is a useful grounding mechanism.

Major concerns:

1. The LLM must not generate accepted reaction facts.
2. Critiques must cite route fields.
3. Hallucination metrics should be part of the benchmark.
4. The API/provider layer should not store secrets or traces.

Required revision:

- Add a forbidden-actions list.
- Add hallucinated-claims metric.
- State that LLM-hypothesis candidates are excluded from default metrics.

## Author Response and Revisions

### Response to Reviewer 1

We revised the manuscript to separate skeleton recovery, filled-route recovery,
stock solve, condition-window success, and cascade-compatibility success. The
abstract now avoids claiming completed full-benchmark superiority. We added an
explicit full live benchmark protocol and state that LLM search lift remains to
be evaluated.

### Response to Reviewer 2

We reframed the contribution as condition- and compatibility-aware
chemoenzymatic cascade planning, not first hybrid planning. We explicitly
acknowledge related hybrid and tool-agent systems. We added discussion of
temperature, pH, solvent, cofactor, enzyme, and operation-mode compatibility.

### Response to Reviewer 3

We added a route JSON export contract as the factual input to LLM critique. We
state that LLMs cannot invent reaction SMILES, stock availability, enzyme
availability, yield, pH, or temperature. We added hallucinated claims as a
planned metric.

## Round 2 Simulated Re-Review

### Reviewer 1: Machine Learning and Benchmarking

Decision: weak accept for a systems/dataset-track paper, borderline for a
general ML track.

The revised manuscript now includes a completed frozen 100-target dual-GPU live
benchmark and separates skeleton, filled-route, stock, condition, and
compatibility metrics. The paper no longer claims that LLM priors improve
search. The remaining weakness is the absence of an equal-budget LLM-prior
ablation; this is acceptable only because the LLM layer is framed as an
interface and safety contract rather than the source of current performance.

Required before submission:

- Keep the abstract focused on measured planner metrics.
- Move any future LLM-lift statements to future work unless an ablation is run.
- Preserve the benchmark JSON and script versions used to generate the table.

### Reviewer 2: Retrosynthesis and Chemistry

Decision: accept after minor revision for a cheminformatics or computational
catalysis venue.

The revised framing is substantially stronger. The novelty is not generic
hybrid retrosynthesis but route-level cascade feasibility with condition,
enzyme, provenance, and stock diagnostics. The case studies demonstrate why a
filled type sequence is insufficient without stock and compatibility checks.
The manuscript should avoid implying that deterministic condition screens are
experimental validation.

Required before submission:

- Keep compatibility language as "screen" or "diagnostic".
- Add more enzyme evidence fields when UniProt/cofactor annotations are ready.
- Add more failed cascade examples in the next dataset release.

### Reviewer 3: AI Agent Safety and Grounding

Decision: accept for the agent-grounding component if the safety contract
remains explicit.

The revised system uses LLMs conservatively: priors and critique are allowed,
but reaction facts, stock status, conditions, enzyme availability, and yields
remain controlled by deterministic exports and validators. This is a stronger
agent design than systems that let an LLM directly author accepted route facts.

Required before submission:

- Do not store provider API keys or raw provider traces in committed artifacts.
- Report hallucinated-claims counts when LLM critique is benchmarked.
- Keep deterministic fallback behavior enabled.

## Remaining Reviewer Risks

- The LLM-prior ablation is currently small and negative; full 100-target
  equal-budget evaluation is still needed before making any LLM-lift claim.
- Enzyme evidence needs richer data before strong claims.
- Compatibility is currently a deterministic feasibility screen, not an
  empirical cascade success predictor.
- The next strong paper version should include more case studies with real
  experimental condition conflicts and mitigations.

## Round 3 Simulated Re-Review After DeepSeek Ablation

### Reviewer 1: Machine Learning and Benchmarking

Decision: accept for a systems paper if the LLM result remains framed as a
negative deployment ablation.

The manuscript now includes a real provider-backed DeepSeek comparison. This
addresses the previous concern that the LLM layer was entirely unevaluated. The
10-target result does not show performance lift, but that is acceptable because
the paper reports it transparently and does not overstate the LLM contribution.
The next ML-strengthening step is a 100-target equal-budget ablation with
multiple random seeds or deterministic cached skeleton sets.

### Reviewer 2: Retrosynthesis and Chemistry

Decision: unchanged accept after minor revision.

The negative LLM result actually improves the paper's credibility. The
chemistry contribution remains the structured route feasibility interface and
cascade-specific data, not language-model route invention. The manuscript should
continue to emphasize stock, provenance, enzyme evidence, and compatibility
diagnostics over generic agent claims.

### Reviewer 3: AI Agent Safety and Grounding

Decision: accept.

The DeepSeek run demonstrates that provider integration can be tested without
storing provider keys or allowing generated text to become accepted chemistry.
The LLM is correctly limited to prior generation. Future work should measure
unsupported claims when route critique is run through an LLM provider, not only
the deterministic critic.
