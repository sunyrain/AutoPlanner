# SynthEx Figure 1 case 001: LLM + AiZ short-tail probe

Date: 2026-08-20

## Decision

The end-to-end path is operational: Codex selected a blind strategy, five
node-local ReactionJSON actions were host-replayed into a target-rooted route,
and AiZynthFinder received the remaining open leaf with the paper short-tail
budget.  The probe reached B1 but did not satisfy the paper-equivalent stock
endpoint (B4).

This is a bounded diagnostic probe, not the full paper arm.  It used one Codex
branch and five node calls rather than three independent branches with up to
25 calls each.

## Target and configuration

- Public snapshot target: `npa000656` / SynthEx Figure 1 case 001.
- Target SMILES: `C=C(C)[C@H]1CC[C@]2(C)C[C@@H]3[C@H](C)CC[C@@H]3/C(C)=C\CC12`.
- Codex: `gpt-5.6-terra`, medium reasoning, one independent strategy branch,
  five sequential node-local ReactionJSON calls, top-1 candidate per node.
- Short tail: AiZynthFinder 4.4.1, depth 6, 500 iterations, 1,200 s limit.
- Stock: frozen exact full-InChIKey ZINC + eMolecules union, 39,478,827 unique
  members.
- Run: `canary_runs/synthexfig1-001-llm-aiz-probe-v2`.

## Observed route

Codex selected a substrate-programmed polyene cyclization strategy and the host
materialized this five-step topology:

1. Tricyclic target -> acyclic C20 polyene by opening three ring bonds and
   restoring three alkenes.
2. C20 polyene -> C5 alkene + C15 polyene by a proposed convergent coupling.
3. C5 alkene -> 2-pentyne by proposed stereoselective semihydrogenation.
4. C15 polyene -> less-unsaturated C15 precursor by proposed allylic
   dehydrogenation in the forward direction.
5. C15 precursor -> `C=C(C)CCCBr` + `CCCC(C)=CC=C(C)B(O)O` by a proposed
   alkyl/alkenyl Suzuki coupling.

All five ReactionJSON programs replayed successfully and the route remained
target-rooted.  Two of the three final leaves were exact stock members:

- `CC#CCC`: in stock.
- `C=C(C)CCCBr`: in stock.
- `CCCC(C)=CC=C(C)B(O)O`: not in stock.

The route is not host reaction-validated.  Important chemistry risks include
the missing initiation handle for the proposed Lewis-acid polycyclization, a
coupling step whose replay-derived hydrocarbon fragments do not carry the
claimed coupling handles, and a Lindlar condition hypothesis that is not
aligned with an E-alkene target.

## AiZ short-tail result

AiZynthFinder searched the non-stock alkenylboronic-acid leaf and completed the
full configured short-tail budget:

- 500 iterations.
- 1,417 search nodes.
- 985 expansion calls.
- 30,179 reactant generations.
- 525 extracted routes; 12 normalized route candidates retained by the old
  adapter projection.
- 0 solved routes.
- Search time: 54.36 s.
- Best route: six steps, with four of five terminal precursors in stock; the
  remaining non-stock precursor was
  `CC(=CC=C(C)B(O)O)CCC=O`.

Therefore the unsolved result is not caused by a missing short-tail call.  The
search ran to its configured iteration cap and approached, but did not cross,
the exact stock boundary.

## Metrics and cost

- B0 blind input: pass.
- B1 target-rooted generated route: pass.
- B2 host reaction validation: fail.
- B4 exact paper-equivalent stock closure: fail (2/3 Codex-route leaves in
  stock; no solved AiZ tail).
- Model calls: 6.
- Model tokens: 108,592 input / 13,995 output.
- Model wall time: 941.125 s.
- Total run wall time: 1,007.50 s.

Most wall time came from Codex transport: each CLI call retried timed-out
WebSocket sampling and then fell back to HTTP.  AiZ itself took about 54 s.

## Defects exposed and repaired after the probe

The probe exposed reporting/orchestration defects that did not alter this
unsolved outcome but would corrupt a formal batch:

1. `aizynthfinder` was missing from the canonical origin-kind allow-list, so
   materialized AiZ candidates could be mislabeled as Codex.
2. AiZ route lineage was not exported into the common candidate-provenance
   report.
3. Importing all partial AiZ routes at once could spend the materialization
   budget across incompatible route fragments.  The adapter now selects one
   coherent solved route when available, otherwise the top partial route for
   diagnosis/local repair.
4. A later `reused` stage could overwrite the settled 500-iteration search
   statistics in the final report.
5. Guided-provider edge grouping now supports both ChemEnzy and AiZ while
   preserving historical ChemEnzy exclusion semantics.

The related regression set passes: 106 tests, 0 failures.

## Interpretation relative to the paper snapshot

The frozen head-to-head protocol binds this target to the three public route
snapshots `npa000656_s1`, `npa000656_s2`, and `npa000656_s3`.  The protocol
records zero self-reported stock-solved routes across the nine Figure 1
reference routes.  Thus this probe reproduces a meaningful generated-route
case, but neither side has a stock-closed reference result for this frozen
Figure 1 set.  A superiority claim requires the full three-branch arm and the
paper-equivalent endpoint, not this one-branch probe.
