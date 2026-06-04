# Archived EvoChemEnzy Documents

Last update: 2026-06-04.

This archive keeps prior reports, temporary audits, progress notes, benchmark
notes, and discussion drafts for provenance. These files are no longer active
implementation instructions unless one of the active docs explicitly references
them.

Active source of truth:

- `../SMILES_FIRST_LITERATURE_STRATEGIC_WORKFLOW_2026-06-03.md`
- `../EvoChemEnzy_Code_Delivery_Checklist_2026-06-03.md`
- `../EvoChemEnzy_Agentic_CASP_Plan_2026-06-03.md`
- `../EvoChemEnzy_Plan_Feasibility_Audit_2026-06-04.md`
- `../LITERATURE_TO_EXECUTABLE_TEMPLATE_CHECKLIST_2026-06-04.md`
- `../CODEBASE_HYGIENE_AUDIT_2026-06-04.md`

Current priority:

```text
P0 = SMILES-first Codex literature retrieval
   + advanced intermediate / strategic disconnection generation
   + route package validation.
```

## 2026-06

Absorbed into active docs:

- `Product_Route_Audit_TODO_Final_2026-06-03.md`
- `STATIN_DEPTH20_MATERIAL_GATE_ITERATION_2026-06-02.md`
- `reference_materials/` root-level presentation and project-support drops.

Main absorbed rules:

- route packages need product/route audit before any solved claim;
- advanced same-scaffold terminals are not stock closure by default;
- route anchors are planning anchors, not single-step reactions;
- forward surrogates must be marked as non-lab-procedure planning artifacts;
- condition prediction and EC annotation are feasibility hints, not route proof.

## 2026-05

Absorbed into active docs:

- AutoPlanner/Cascade progress, cleanup, verifier proof, proposal training, and
  CCTS decision reports;
- ChemEnzy baseline and target architecture notes;
- bridge pack, bridge verifier, enzyme SP-v1, enzyme coverage, P5 evidence, and
  native-vs-enhanced benchmark reports;
- chemo-enzymatic project discussion drafts and module audits.

Main absorbed rules:

- ChemEnzy remains the step-level inner engine;
- Codex/LLM stays at episode level and only emits typed artifacts;
- no online LLM rerank, proposal judge, or raw reaction injection is allowed in
  the ChemEnzy inner loop;
- enzyme bridge evidence and verifier metrics are useful gates, not expert truth;
- naive source ensemble can pollute multi-step search and must be gated.
