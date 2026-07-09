# Legacy Codex-Entry Fullflow Docs

Archived on 2026-06-24.

These documents describe the June 3-7 fixed Codex-entry/fullflow direction,
SMILES-first fallback workflow, and earlier bufotalin/statin reports. They are
kept for provenance, but they are no longer active implementation instructions.

The active direction is:

- `../../../AGENTIC_BLACKBOARD_MAINLINE_2026-06-24.md`
- `../../../MAINLINE.md`

Main reason for archival:

- the current controller is policy-driven blackboard action selection rather
  than an ordered fullflow chain;
- Codex action planner now emits compact typed action skeletons, not complete
  workflow plans or expanded ChemEnzy policies;
- final solved status is gated by deterministic parent route proof.
