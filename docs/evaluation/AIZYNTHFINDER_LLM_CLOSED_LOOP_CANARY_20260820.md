# AiZynthFinder + Codex closed-loop canary (2026-08-20)

## Decision

The local planning path is ready to move from workflow canaries to the frozen
paper-budget arm.  The canary demonstrated the complete executable sequence:

`StrategyCard -> node-local ReactionJSON -> host replay -> target-rooted RouteJSON -> stock audit -> AiZynthFinder short tail -> host ingestion -> stock re-audit`

This run did not solve the target because the deliberately reduced AiZ canary
budget left one precursor outside the exact stock.  It did not fail because of
missing `precursor_smiles`, broken RouteJSON assembly, or a disconnected
provider result.

## Run binding

- Run: `aiz-llm-closed-loop-20260820-v1`
- Target: SynthEx Figure 1 opaque case `synthexfig1-001-9c1f431594a7`
- Model: `gpt-5.6-terra`, reasoning `medium`
- Codex arm: one strategy, two node-local expansions, top-1 candidate
- Native tail: AiZynthFinder 4.4.1 canary, depth 3, 12 iterations, 60 s limit
- Stock: exact full-InChIKey ZINC + eMolecules union, 39,478,827 unique members
- Report: `D:/Autoplanner/canary_runs/aiz-llm-closed-loop-20260820-v1/run/target-only-solve-report.json`

## Observed result

- 3 real Codex invocations
- 52,957 input tokens; 6,808 output tokens
- 467.719 s model time; 472.595 s total run time
- 2 Codex reaction edits accepted by deterministic host replay
- 1 target-rooted materialized route
- 1 AiZynthFinder invocation
- AiZ search: 12 iterations, 40 nodes, 307 reactant generations
- AiZ returned 2 accepted partial routes containing 5 reaction steps
- Two final precursors were in the exact stock; one precursor remained open
- Outcome: paper reach `true`; paper-equivalent solved `false`; B1 `true`, B4 `false`

The two Codex edits were a cationic polyene-cyclization disconnection followed
by macrocycle opening to an acyclic polyene.  Both model responses intentionally
left `precursor_smiles` empty; the host generated the authoritative precursors
by replaying atom-map edits.  This confirms that an empty model precursor field
is no longer evidence of a missing route.

## Changes validated by this run

- AiZynthFinder 4.4.1 now runs in the isolated `.venv_aizynth` environment.
- Paper stock lookup uses the same frozen full-InChIKey SQLite oracle as the host.
- Paper-matched node width is top-1, matching the released SynthEx/SyntheLite
  configuration; three-candidate OR/UCB remains a separately labelled enhanced
  ablation.
- The paper profile selects AiZynthFinder for open-leaf completion.  ChemEnzy is
  retained for ordinary AutoPlanner and enzyme-oriented ablations.
- AiZ route trees are flattened, topology-checked, and ingested with
  `origin_kind=aizynthfinder`.
- Provider-facing stage summaries now distinguish AiZynthFinder from ChemEnzy;
  the old ChemEnzy action enum remains only a scheduler transport compatibility
  layer.

## Formal launch preflight

`D:/Autoplanner/canary_runs/aiz-paper-preflight-20260820-v3/panel-status.json`
passed all three targets.  The frozen protocol reports:

- `ready_for_paid_experiment=true`
- `issues=[]`
- 3/3 blind supervisor preflights passed
- native short-tail engine: AiZynthFinder
- native and host stock boundary equal
- ReactionJSON candidates per node: 1
- three independent branches, 25 node calls per branch
- short tail: depth 6, 500 iterations, 1,200 s

## Remaining experimental question

The next run should answer a scientific question, not another plumbing
question: with the full 3 x 25 Codex budget and full AiZ short tails, how many
of the three frozen targets become paper-equivalent solved?  Reaction
validation, conditions, evidence, and the enzyme companion arm remain separate
reported axes and must not block that reach measurement.

