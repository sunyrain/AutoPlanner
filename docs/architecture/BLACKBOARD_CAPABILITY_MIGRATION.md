# Blackboard capability migration into the V4 authority path

Status: active migration contract (2026-07-14)

The legacy blackboard was not a failed collection of features.  Its main
architectural defect was that collaboration state, expansion state, evidence
state, and route truth could diverge.  V4 therefore keeps `RunKernel`, the
canonical hypergraph, and `DeficitFrontier` as the only authorities while
reusing the old system's working capabilities as shared services.

Restoring the blackboard controller itself is forbidden.  Reimplementing a
smaller duplicate of a working capability is also forbidden.

## Migration matrix

| Capability | Proven legacy implementation | V4/shared destination | State | Remaining acceptance work |
|---|---|---|---|---|
| Whole-campaign Codex planning | `agentic_blackboard_controller`, action planners | `GlobalCampaignDirector` | integrated | blind portfolio gain and two-call ceiling benchmark |
| Canonical scientific state | several blackboard/route projections | `CanonicalHypergraphStore` | integrated | delete compatibility writes after telemetry window |
| One typed work queue | blackboard bridge tasks and route deficit queue | `DeficitFrontier` | integrated | add condition/extraction deficits and gain metrics |
| Guided ChemEnzy | `run_guided_chemenzy_rerun`, guided payload and failure critic | `interfaces.chemenzy_probe` | migrating | port analogical/failure policy and provider-compute receipts |
| Codex-selected provider tasks | bridge tasks and guided payload | canonical molecule scheduling annotations -> `DeficitFrontier` | integrated | blind run with 1-3 explicit subtargets |
| Literature research | `literature_research`, source material locator | evidence connector provider chain | migrating | port richer query expansion and source ranking |
| Authorized browser/campus PDF access | `local_pdf_proxy`, `browser_pdf_fetch.py`, `tsinghua_pdf_gateway.py` | V4 literature source lifecycle | integrated | automated resume watcher and operator UX |
| Native PDF focus and route-context selection | private helpers in `harness.tools` | `harness.literature_page_selection` | integrated | long-PDF regression corpus |
| Page/scheme visual extraction | visual literature chain agent | `interfaces.visual_evidence` with RunKernel metering | integrated | crop batching and structured root correction |
| Exact reaction/procedure extraction | legacy PDF/HTML exact-row tools | structured evidence workers | migrating | bind complete procedure schema to exact edges |
| Source lifecycle and failure receipts | blackboard evidence manifests | connector receipts + artifact store | migrating | explicit discovered/downloaded/extracted/bound state projection |
| Deterministic chemistry validation | reaction verifier, mapping and stock gates | V4 worker runtime/admission | integrated | broaden large-atom-jump and stereo regression cases |
| Trusted stock closure | deep-leaf audit and inventory helpers | versioned inventory/stock workers | migrating | real supplier snapshot and offer expiry/revocation |
| Patent self-evolution | patent extraction/template reuse | `PatentSelfEvolutionSession` | integrated-v1 | knowledge-record maturity and reuse statistics |
| Auditable route UI | RouteForest | V4 route workbench/display adapter | migrating | proof-vector views, conditions, source locators, layout benchmark |

`integrated` means the capability reaches the V4 canonical ingestion path and
has a focused regression test.  It does not mean a blind complex-molecule
acceptance gate has passed.  Only the benchmark report may make that claim.

## Correct execution order

```text
arbitrary SMILES
  -> target identity/capability snapshot
  -> Codex whole-campaign route families and multi-step skeletons
  -> deterministic host admission/materialization/validation
  -> Codex-selected local frontier tasks
  -> bounded ChemEnzy/template expansion
  -> source discovery (HTML/XML first, then PDF access)
  -> native text focus / OCR / one batched visual fallback
  -> exact row and procedure binding
  -> stock audit; rejected high-level leaves become bounded recovery tasks
  -> at most one material-event Codex replan
  -> proof-vector portfolio and explicit unresolved deficits
```

The final target is not a default ChemEnzy seed.  A target-level ChemEnzy probe
exists only as an explicit diagnostic option.  This preserves Codex's unique
advantage: it chooses route architecture globally, while ChemEnzy supplies
local alternatives where the architecture says they have value.

## State and cost invariants

1. No provider owns a private route/frontier truth.
2. Every proposed edge enters through canonical admission.
3. Model, visual, provider compute, attempts, and accepted expansions have
   separate receipts and limits.
4. Discovery metadata, downloaded bytes, visual hypotheses, exact procedures,
   reaction proof, conditions, and stock are distinct states.
5. A queued campus/browser source is visible and resumable but grants no proof.
6. A visual structure chain is L0/L1 material only until exact source binding.
7. Benchmark stock is not procurement closure.
8. No UI count may mix suggested disconnections, materialized edges, validated
   routes, literature-grounded routes, and stock-closed routes.

## Current migration slice

- [x] Disable target-level ChemEnzy by default; keep an explicit diagnostic flag.
- [x] Move guided ChemEnzy after the initial Codex architecture.
- [x] Compile Codex `frontier_priorities` with `target_smiles` into the single
  canonical deficit frontier.
- [x] Preserve a bounded stock-rejection recovery pass without repeating a
  previously attempted frontier.
- [x] Share legacy focus/context/coverage PDF selection with V4; remove first-N
  page fallback.
- [x] Queue restricted PDFs through the existing authorized browser proxy and
  consume successful downloads on resume.
- [ ] Extract the old guided ChemEnzy failure/analogy policy into a shared module.
- [ ] Port the full literature research/source-ranking loop instead of relying
  on Crossref metadata alone.
- [ ] Port exact PDF/HTML procedure extraction and source lifecycle states.
- [ ] Complete the proof vector, real inventory boundary, UI, and blind suite.
