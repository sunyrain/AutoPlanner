# AutoPlanner Docs

Last update: 2026-06-24.

The active direction is the agentic blackboard controller.

## Active Docs

- [Agentic blackboard mainline](AGENTIC_BLACKBOARD_MAINLINE_2026-06-24.md)
- [AutoPlanner mainline](MAINLINE.md)
- [Codex WellAU streaming runbook](CODEX_WELLAU_STREAMING_RUNBOOK_2026-06-05.md)

## Current Decisions

- Codex chooses compact typed action batches from blackboard state.
- Deterministic validators enforce action schema, budget, source binding, stale
  repetition, raw reaction rejection, and solved-claim rejection.
- Heavy payloads are completed by local builders, not by long prompts.
- Online search is primary; local PDFs are metadata-matched cache/fallback.
- ChemEnzy is a local route generator. It is not final proof.
- Final `solved` requires deterministic parent route proof and stock audit.

## Archive

Legacy fixed-chain/fullflow, SMILES-first fallback, bufotalin report, and statin
template docs live under:

- [legacy Codex-entry fullflow archive](archive/2026-06/legacy_codex_entry_fullflow/)
- [older archive root](archive/)
