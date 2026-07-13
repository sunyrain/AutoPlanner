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
`local map/verify -> narrow host repair -> source frontier -> stock audit ->`
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
  completed checkpoint; terminal resume only refreshes reporting projections.
- [x] Provider failures produce an unresolved report instead of false closure.

### P2 - Bounded global Codex campaign

- [x] Default model policy is `gpt-5.5`, low reasoning, one initial campaign
  call and at most one event replan.
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

### P4 - Replaceable evidence and stock authority

- [x] Director source plans become an unresolved acquisition frontier; hints
  never become exact evidence.
- [x] `import-evidence` accepts only trusted structured extraction artifacts
  and resumes canonical proof compilation.
- [x] The generic PubChem vendor-catalog adapter records version, retrieval
  time, response hashes, members, and misses.
- [x] Every selected deep leaf is audited again after repairs or replans.
- [x] B3/B4 fail closed when their authoritative adapters are absent.
- [ ] A live primary-source acquisition/extraction connector is not configured
  in this environment. Therefore no blind case is reported as B3.

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

The checked-in manifest contains three structurally different complex targets.
All runs used a fresh external runtime root, one `gpt-5.5` low-reasoning global
call, no executed replan, local mapping/verification, and generic stock search.
The compact result contains no generated route or precursor answers.

| Case | B0 | B1 | B2 | B3 | B4 | B5 | Calls | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Enzalutamide formal run 04 | yes | yes | no | no | yes | no | 1 | Two global families retained unvalidated critical transforms; resource-compliant unresolved result |
| Ibrutinib run 01 | yes | yes | no | no | no | no | 1 | Shared core/piperidine connection rejected; three selected stock misses |
| Linagliptin run 02 | yes | yes | yes | no | yes | yes | 1 | Two L2 plus benchmark-stock routes after a generic ring-size repair |

Aggregate model use: 3 calls, 54,081 input tokens, 19,299 output tokens, and
692.626 reported model seconds. All formal runs stayed within their actual
contracts. The historical Enzalutamide run 03 reached two L2 routes, but is not
the formal result because its deliberately tighter 7k output contract was
exceeded; the resource gate correctly disqualifies it.

The benchmark release threshold is intentionally **not passed**: all three
cases pass B0/B1 and no false evidence/procurement claims exist, but only one
formal case currently has two B2 routes. Failed cases remain in the report; the
repository does not cherry-pick a showcase and call it universal success.

See `benchmarks/results/blind_benchmark_summary.v1.json` for report hashes,
gate counts, costs, and failure codes.

## Release checklist

- [x] Focused target, repair, verifier, route-workbench, worker, evidence, and
  blind-contract tests.
- [x] Full test suite (`1552 passed, 3 skipped`) and Ruff.
- [x] Architecture and repository audits.
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
