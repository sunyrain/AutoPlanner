# AutoPlanner Docs

Last update: 2026-06-07.

The active direction is Codex-entry route control:

```text
target input
-> deterministic preflight
-> Codex controller chooses tools
-> ChemEnzy / route audit / frontier extraction / literature research / structure validation
-> deterministic validators decide solved, partial, rejected, or needs_followup
```

Codex may plan, research, classify sources, and choose the next tool call. It
must not directly mark a route solved or promote artifacts into production KB.

## Active Docs

- [Architecture and repository state](ARCHITECTURE_AND_REPOSITORY_STATE_2026-06-07.md)
- [AutoPlanner mainline](MAINLINE.md)
- [Codex WellAU streaming runbook](CODEX_WELLAU_STREAMING_RUNBOOK_2026-06-05.md)
- [SMILES-first literature strategic workflow](SMILES_FIRST_LITERATURE_STRATEGIC_WORKFLOW_2026-06-03.md)
- [Literature-to-executable template checklist](LITERATURE_TO_EXECUTABLE_TEMPLATE_CHECKLIST_2026-06-04.md)
- [Atorvastatin template extraction full flow](ATORVASTATIN_TEMPLATE_EXTRACTION_FULLFLOW_2026-06-06.md)

## Current Decisions

- Codex moves to the workflow entry point.
- ChemEnzy remains a deterministic route-search tool called by the harness.
- Literature research is Codex-native, with optional local wrappers for DOI,
  PubMed, CrossRef, PubChem, and filesystem evidence.
- Route audit, artifact schemas, structure validation, and production gates are
  deterministic local code.
- No raw LLM reaction injection, no LLM-only solved claim, and no production KB
  promotion without validator approval.

## Archive

Prior planning, feasibility, code-delivery, paper, statin-panel, ONMT/training,
and renderer materials were moved out of the active surface. Local harness-prep
archives live under `archive/harness_prep_*/` and are ignored by git.

Older provenance docs under `docs/archive/` remain historical context only.
