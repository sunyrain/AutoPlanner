# Blind retrosynthesis: implementation and acceptance record

This document is the release checklist for the real target-only path. A blind
case supplies only a fresh run ID, a display name, a canonical target SMILES,
and generic acceptance/resource limits. Dossiers, replay packs, target-specific
fixtures, precursor lists, source rows, inventories, and generated routes are
forbidden inputs.

## Intended end state

Codex is a bounded global campaign architect, not a repeated single-step
oracle. One call proposes a compact portfolio of complete route families,
shared intermediates, pivots, source-acquisition work, and stopping conditions.
The host then owns every scientific promotion:

1. A single `RunKernel` owns the attempt budget, accepted-expansion budget,
   canonical hypergraph, proof frontier, checkpoints, and terminal decision.
2. Codex proposals enter as L0 hypotheses. Cheap admission rejects invalid
   identities, element jumps, cycles, duplicate edges, and disconnected plans.
3. Local RXNMapper plus the deterministic reaction verifier can promote an
   edge to L2. Narrow product-grounded repairs create a new L0 edge and must be
   mapped and verified normally; the rejected original remains visible.
4. Exact structured source rows alone can grant L3. Search hints and model
   prose cannot.
5. Versioned stock observations close the configured leaf boundary. Benchmark
   search closure is explicitly not procurement closure.
6. The proof stitcher selects a small weakest-link portfolio. The workbench is
   a bounded read model, never a second source of scientific state.
7. A second global model call is optional and event-driven. It is skipped
   unless the remaining input, output, invocation, and wall-time budgets can
   reserve a realistic complete call.

The resulting main path is:

`SMILES -> blind preflight -> global portfolio -> admission -> materialize ->`
`local map/verify -> narrow host repair -> source frontier -> built-in/replaceable evidence connector ->`
`frozen PDF -> deterministic exact-row ingestion -> internal/deep stock-cut audit ->`
`proof stitch -> B0-B5 gates -> bounded workbench -> stop/checkpoint`

## Independent gates

| Gate | Measurement | Pass condition |
| --- | --- | --- |
| B0 | Blind-input integrity | Fresh run and target-only anti-leak preflight |
| B1 | Global multi-route proposal | Required number of exact-target-rooted, connected, distinct skeletons |
| B2 | Host-validated routes | Required number of skeletons whose retained edges all have current host L2 proof |
| B3 | Exact multi-source grade | Required number of canonical routes with exact rows from the configured independent groups |
| B4 | Configured stock boundary | Required number of canonical routes whose every selected leaf is closed |
| B5 | Configured portfolio acceptance | Canonical proof portfolio satisfies its explicit proof/stock policy |

The gates are independent measurements. For example B4 may be true while B3
is false. `highest_contiguous_gate` remains B2 in that case. B5 at an L2 plus
benchmark-stock policy is not an L3 or procurement claim.

## Completed implementation

### P0 - Blind boundary

- [x] Versioned target-only manifest and strict field allow-list.
- [x] Preflight rejects route/source/stock/replay leakage and reused run paths.
- [x] Tracked-tree absence attestation runs before model work.
- [x] Tests cover forbidden fields, duplicate targets, target leakage, and the
  manifest-only exception.

### P1 - Real target-only entrypoint

- [x] `autoplanner solve-target --target-smiles ...` invokes the real global
  campaign path; model-free `run` remains unchanged.
- [x] `solve-case` is explicitly a dossier replay compatibility alias.
- [x] Atomic checkpoints and `--resume` avoid additional model calls after a
  completed checkpoint. Terminal resume never mutates the historical graph;
  if a newer verifier invalidates its proof, the current disposition is
  `terminal_snapshot_requires_revalidation` rather than a solved claim.
- [x] Provider failures produce an unresolved report instead of false closure.
- [x] CLI output is a bounded route-free summary by default; the complete
  content-addressed report stays on disk and is emitted only with
  `--full-output`.
- [x] A no-change resume is content-idempotent: consecutive refreshes keep the
  same report SHA, attempt count, and model-call count.

### P2 - Bounded global Codex campaign

- [x] New target-only runs default to the locally verified `gpt-5.5`, low
  reasoning, one initial campaign call and at most one event replan. The model
  remains explicitly replaceable with `--model`. An attempted `gpt-5.6-sol`
  run failed before chemistry because Codex CLI 0.142.4 requires a newer CLI;
  unsupported model probing is therefore not part of the campaign loop.
- [x] The default provider launches one global director agent. A multi-agent
  coordinator is explicit opt-in, not an implicit multiplier on model cost.
- [x] Prompts require exact-target-rooted connected DAGs, route-family
  diversity, complete leaves, pivots, limitations, and source tasks.
- [x] CampaignContext is compacted by topology/proof summaries while retaining
  complete identities and connectivity.
- [x] Attempt and accepted-expansion budgets are separate; no per-edge model
  loop exists.
- [x] A replan previews actual prompt bytes and reserves observed input,
  output, invocation, and wall-time envelopes before execution.
- [x] Prospective campaigns request at least three route families so one shared
  bad bottleneck is less likely to collapse the whole portfolio.

### P3 - Deterministic host validation and repair

- [x] All new edges are batch-mapped locally and verified through V4 workers.
- [x] Strict generic transform support includes acyl substitution,
  heterocumulene addition, Suzuki-like C-C coupling, symmetry-tolerant
  aryl-heteroatom coupling, and a structure-derived amino-acid/isothiocyanate
  thiohydantoin annulation.
- [x] Product-grounded repair handles exactly one ring-carbon size error and a
  unique Ar-S=C=N versus Ar-N=C=S connectivity error.
- [x] Repairs preserve the rejected edge, grant no proof, and require normal
  admission, mapping, and validation.
- [x] A repaired edge can prune only the now-disconnected upstream tail; a
  rejected middle edge cannot make a route pass.
- [x] Duplicate repairs are skipped before worker reservation, so resume cannot
  collide with an older payload under the same idempotency key.
- [x] Reaction proofs are verifier-versioned. V7 acceptance or rejection is
  authoritative; stale V6 acceptance cannot outvote a current rejection.
- [x] Current negative reaction proofs are cached, so unresolved resume does
  not map and reject the same edge indefinitely.
- [x] `validate-target` creates a model-free, lineage-bound validation fork.
  It replays a blind source plan under the current verifier without mutating
  the original campaign or making another global planning call.

### P4 - Replaceable evidence and stock authority

- [x] Director source plans become an unresolved acquisition frontier; hints
  never become exact evidence.
- [x] `import-evidence` accepts only trusted structured extraction artifacts
  and resumes canonical proof compilation.
- [x] `--evidence-endpoint` provides a bounded HTTPS/loopback typed connector.
  Its provider identity, limits, receipt hash, and structured rows are frozen;
  connector booleans cannot grant L2 or L3.
- [x] With no external endpoint, a built-in zero-model patent connector first
  freezes official Google Patents HTML and digest-bound paragraph ranges, then
  emits exact rows only after deterministic product/reactant reconstruction.
  Search snippets and discovery-only material remain L0 data.
- [x] PDF download, native-text extraction, OCR, and optional page vision are
  strictly unresolved-only fallbacks. Full HTML closure skips PDF entirely;
  partial HTML closure sends only remaining edge IDs down the fallback ladder.
- [x] Image-only primary pages first use bounded local Tesseract OCR. OCR
  companions bind source identity, PDF hash, page number, rendered image hash,
  text hash, and allowlisted engine identity; any replay mismatch fails closed.
- [x] Sparse page vision is explicit opt-in, defaults to zero, admits at most
  one durable campaign task, and runs only while deterministic parsing leaves
  unresolved edges. Host-normalized visual chemistry remains L0 replan data
  and cannot grant L2, L3, source independence, stock, or completion.
- [x] Evidence discovery is event-bearing: newly found source material or
  exact rows can resume the global campaign only when a complete second-call
  budget can be reserved.
- [x] Patent family identity survives normalization and replay, so a binding
  remains digest-idempotent while family independence stays auditable.
- [x] OPSIN/PubChem name resolution uses a versioned persistent non-authority
  cache; replays avoid repeating slow network resolution but cached data can
  never approve a reaction.
- [x] The generic PubChem vendor-catalog adapter records version, retrieval
  time, response hashes, members, and misses.
- [x] `--inventory-snapshot` closes the procurement boundary only from a
  versioned trusted supplier snapshot and audits every selected leaf.
- [x] Every selected deep leaf is audited again after repairs or replans.
- [x] Host-audited purchasable internal intermediates are legitimate route-cut
  boundaries. Their upstream edges are retained as generated hypotheses but
  excluded from the effective accepted route; deep leaves remain mandatory
  and capacity exhaustion fails closed.
- [x] Fresh positive and negative stock observations are reused on resume;
  a miss no longer triggers an identical network audit until the 30-day
  freshness window expires.
- [x] An empty validation batch does not initialize RXNMapper. A no-change
  Ibrutinib resume now completes locally in under one second without consuming
  another attempt (machine-specific timing, not a contractual latency bound).
- [x] B3/B4 fail closed when their authoritative adapters are absent.
- [x] The live primary-source connector boundary is implemented and covered by
  an end-to-end injected-provider test. No trusted endpoint is configured in
  this environment, so the formal cases below still correctly fail B3.

### P5 - Trustworthy bounded display

- [x] The default view is a small selected portfolio, not the full exploratory
  hypergraph.
- [x] Separate filters expose disconnection suggestions, expanded paths,
  reaction-validated routes, and stock-closed routes.
- [x] Edge colors encode the weakest proof level: L0 proposal, L1
  materialized, L2 validated, L3 exact-source, and L4 procurement-ready.
- [x] B0-B5, highest contiguous gate, resource compliance, model calls, stop
  decision, rejections, conflicts, sources, and stock observations are carried
  through the digest-bound workbench projection.
- [x] Campaign gates are measurement-only UI metadata and cannot grant proof.

## Fresh blind benchmark (2026-07-14)

The checked-in manifest contains four structurally different complex targets.
Each source campaign used a fresh external runtime root, one `gpt-5.5`
low-reasoning global call, no executed replan, local mapping/verification, and
generic stock search. The compact result contains no generated route or
precursor answers. Vismodegib was absent from the tracked repository at blind
preflight; it was added to the target-only manifest only after the run.

| Case | B0 | B1 | B2 | B3 | B4 | B5 | Calls | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Enzalutamide formal run 04 | yes | yes | no | no | yes | no | 1 | One of two skeletons passes V7; an older false-positive edge is revoked |
| Ibrutinib run 01 | yes | yes | no | no | no | no | 1 | Shared core/piperidine connection rejected; three selected stock misses |
| Linagliptin + V7 fork 05 | yes | yes | yes | no | yes | yes | 1 + 0 replay | Two skeletons validate under V7; the source terminal graph remains immutable |
| Vismodegib + V7 fork 06 | yes | yes | yes | no | yes | yes | 1 + 0 replay | Three skeletons validate; five canonical routes stock-close; one exact patent row is bound |

Aggregate model use: 4 calls, 72,083 input tokens, 25,477 output tokens, and
918.267 reported model seconds. The Vismodegib source campaign used one call,
18,002 input tokens, 6,178 output tokens, and 225.641 model seconds; its final
validation/evidence/stock fork made zero model calls. All formal runs stayed
within their actual contracts. The historical Enzalutamide run 03 reached two
L2 routes, but is not the formal result because its deliberately tighter 7k
output contract was exceeded; the resource gate correctly disqualifies it.

The benchmark release threshold is now **passed**: all four cases pass B0/B1,
Linagliptin and Vismodegib pass B2/B5 under the current V7 verifier, every case
stays within its resource contract, and no false evidence or procurement claim
is present. Failed Enzalutamide and Ibrutinib cases remain in the release set.
B3 still correctly remains false for every case: Vismodegib's one exact EP
patent row is useful evidence, but it is not two independent source groups
across two routes. Linagliptin's discovered patent route differs from the
proposed route and is likewise not promoted into a false exact binding.

The important Vismodegib result is architectural. Before the stock-cut fix,
the same global proposal reported zero validated skeletons because rejected
speculative chemistry above purchasable acid/acid-chloride intermediates was
counted as part of the route. The zero-model fork now retains the full proposal
for inspection while accepting only the target-reachable, host-validated path
down to each audited purchasable boundary. The system therefore completes a
useful L2 + benchmark-stock portfolio without paying for another model call.

## Cost and stopping contract

- One low-reasoning global call is the normal cost. A second call is allowed
  only after a material event and only when the full measured envelope fits.
- Primary HTML parsing and local OCR are zero-model work. Optional page vision has its own 0/1 hard
  budget and also consumes the shared model budget; a visual-assisted replan
  therefore needs three explicitly allowed calls in total.
- Mapping, validation, stock audit, exact-source parsing, and validation forks
  are host work and report zero model invocations.
- The solver never keeps calling a stronger model to force a bad proposal to
  pass. It returns an explicit unresolved frontier when evidence, inventory,
  or chemistry authority is missing.
- B3 is deliberately harder than B5 under the configured L2 policy. A real
  route can be operationally accepted while exact multi-source coverage is
  still shown as incomplete.

See `benchmarks/results/blind_benchmark_summary.v1.json` for report hashes,
gate counts, costs, and failure codes.

## Release checklist

- [x] Focused target, repair, verifier, route-workbench, worker, evidence, and
  blind-contract tests.
- [x] Full test suite (`1604 passed, 3 skipped`, 2 subtests) and Ruff.
- [x] Static V4 architecture ownership/line-budget audit and repository audit.
- [x] Verified no GitHub Actions/workflows were added or changed.
- [x] Commit the bounded source changes and compact benchmark summary only.
- [x] Push directly to `origin/main`.

## Stop policy

The solver stops only on configured canonical acceptance, a hard host budget,
or an explicit unresolved frontier that cannot progress without new authority.
Round count, branch count, stock search hits, and Codex confidence never imply
scientific completion. A practical system must return an honest unresolved
campaign cheaply when the one-call global portfolio is chemically wrong; it
must not spend hours trying to force every arbitrary molecule into a success.
