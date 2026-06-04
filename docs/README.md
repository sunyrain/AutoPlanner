# AutoPlanner Docs

Last update: 2026-06-04.

Top-level docs are now intentionally small. The active source of truth is:

- [AutoPlanner mainline](MAINLINE.md)

The EvoChemEnzy/literature-template document set below supports that mainline.
Prior status reports, progress notes, benchmark reports, and discussion drafts
have been absorbed into these documents and archived under `docs/archive/`.

## Active Docs

- [AutoPlanner mainline](MAINLINE.md)
- [SMILES-first literature strategic workflow](SMILES_FIRST_LITERATURE_STRATEGIC_WORKFLOW_2026-06-03.md)
- [EvoChemEnzy agentic CASP plan](EvoChemEnzy_Agentic_CASP_Plan_2026-06-03.md)
- [EvoChemEnzy code delivery checklist](EvoChemEnzy_Code_Delivery_Checklist_2026-06-03.md)
- [EvoChemEnzy plan feasibility audit](EvoChemEnzy_Plan_Feasibility_Audit_2026-06-04.md)
- [Literature-to-executable template checklist](LITERATURE_TO_EXECUTABLE_TEMPLATE_CHECKLIST_2026-06-04.md)
- [Codebase hygiene audit](CODEBASE_HYGIENE_AUDIT_2026-06-04.md)

Start development from the SMILES-first workflow. The code delivery checklist
then turns that workflow into P0a/P0b/P0c implementation tasks. P0 is centered
on target profiling, frontier extraction, Codex literature retrieval, evidence
cards, strategic intermediate/disconnection candidate generation, hybrid route
package, and validation. RouteStatus, blackboard, compiled judge, guided rerun,
enzyme bridge, and evolution work move to P1/P2/P3 unless needed as P0
guardrails.

Current authority order:

```text
Repository mainline: AutoPlanner mainline
P0 execution: SMILES-first literature strategic workflow
P0 engineering checklist: 当前 P0 开发板 + 开发启动交付清单
Long-term architecture: EvoChemEnzy agentic CASP plan
Risk/feasibility constraints: plan feasibility audit
Executable literature-template delivery: Literature-to-executable template checklist
Repository hygiene boundary: Codebase hygiene audit
Archived provenance: docs/archive/
```

## Current Decisions

- ChemEnzy native multi-step search remains the step-level inner engine.
- Codex / LLM work stays at episode level: diagnosis, literature research,
  typed artifact drafts, strategy proposals, audit, and evolution candidates.
- No LLM call, online LLM rerank, online LLM proposal judge, or raw LLM reaction
  injection is allowed inside the ChemEnzy inner loop.
- One-pot cascade condition compatibility is not the main innovation claim; it
  is a condition/route-audit module.
- The main research framing is enzyme-aware chemo-enzymatic bridge planning:
  gated identification of chemical intermediates that can enter enzyme
  substrate/product/EC space.
- Stock closure, EC annotation, planner score, condition prediction, and
  literature summaries do not by themselves prove a route is solved.

## Archive

- [archive/2026-05/](archive/2026-05/) contains May 2026 reports, roadmaps,
  cleanup notes, bridge/verifier progress, benchmark notes, and frozen research
  context.
- [archive/2026-06/](archive/2026-06/) contains June 2026 temporary audit and
  iteration notes plus archived root-level reference materials.

Archived documents are retained for provenance and reproduction. They are no
longer implementation instructions unless an active doc explicitly references
them.

## Static Outputs

Static showcase assets remain under:

- [statins/](statins/)
- [bufotalin/](bufotalin/)

These are rendered artifacts, not planning-source documents.
