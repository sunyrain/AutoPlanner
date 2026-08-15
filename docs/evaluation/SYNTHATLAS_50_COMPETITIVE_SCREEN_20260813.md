# SynthAtlas 50-target competitive screen

Date: 2026-08-13

## Outcome

The frozen 50-target competitive screen is complete. AutoPlanner generated at
least one target-rooted strategic route for 45/50 targets and materialized a
canonical route for the same 45/50. Nine targets reached strict host reaction
validation and six reached strict stock closure. No target closed an exact
procedure or complete exact-condition route; one target satisfied the configured
B5 portfolio acceptance gate. Four targets failed in the runner with
`MemoryError` and remain in every denominator.

This is a competitive screen, not a strict blind claim. Repository-absence
preflight found 16/50 targets with route-intermediate overlap in historical
project artifacts. Results are therefore reported for all frozen 50 and for the
predeclared repository-clean 34-target subset.

| Subset | Status | C0 | C1 | C2 | C3 | C4 | C5 | C6 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all frozen | 46 complete, 4 runner failures | 45/50 | 45/50 | 9/50 | 0/50 | 0/50 | 6/50 | 0/50 |
| repository-clean | 31 complete, 3 runner failures | 30/34 | 30/34 | 6/34 | 0/34 | 0/34 | 5/34 | 0/34 |

The frozen public SynthAtlas snapshot supplied 149 routes for the same targets.
At target level, 50/50 reached C0 and 46/50 reached C1 after provider-neutral
host replay. The import action granted no C2-C6 authority. At route level,
148/149 passed C0 and 87/149 passed C1; 61 failures were dominated by incomplete
structural reagent accounting, and one route contained a duplicate product step.

## Why the reported SynthEx rate is much higher

The arXiv v1 paper reports three different endpoints on 1,098 targets:

- template baseline: 151/1,098 (13.8%);
- SynthEx strategy layer: 275/1,098 (25.0%);
- SynthEx plus short AiZynthFinder stitching: 702/1,098 (63.9%).

The jump from 25.0% to 63.9% is therefore not an LLM-only result. The LLM is used
to propose difficult, global disconnections; a separate short template search
then closes simplified strategy leaves to a ZINC plus eMolecules stock. The
paper's solved endpoint requires stock identity closure, not independent reaction
proof, exact literature procedures, exact complete conditions, procurement
observations, or experimental execution. It is closest to one part of our C5
axis and is not equivalent to C2-C6 jointly.

AutoPlanner's closest current stock endpoint is 6/50 (12.0%), versus the paper's
reported 702/1,098 (63.9%). This difference is real under the nearest available
endpoint, but it does not mean the global route generator failed: AutoPlanner's
C0 rate was 45/50 (90.0%). The loss occurs mainly after strategic route
generation, at short-tail stock closure.

## Runtime diagnosis

Across the 46 completed targets, AutoPlanner used 73 model invocations,
2,257,312 input tokens, 512,850 output tokens, and 49,846 seconds of summed
per-target elapsed time (median 1,103 seconds). It produced 150 candidate routes,
137 canonical materializations, 14 strict host-validated routes, and 10 strict
stock-closed routes distributed across six targets.

The action ledger shows a scheduling mismatch with the stock-closure endpoint:

- 1,998 exact-evidence acquisition/binding actions;
- 731 Program discovery/review actions;
- 339 reaction-validation actions;
- 242 ChemEnzy target/frontier expansion actions.

Of the 46 initial ChemEnzy target calls, 34 timed out, 11 failed and one reported
runtime unavailable. The run issued 202 guided-frontier requests, each configured
for eight iterations and 90 seconds, and accepted zero provider proposals from
them. The paper's reported stitching stage instead permits up to six transforms,
800 iterations and 1,200 seconds per short search. The systems therefore do not
give comparable compute to the capability that dominates the reported solve
rate.

Four full target processes failed because the concurrent run exhausted memory.
The failures occurred while reading event logs or verifying artifacts, not as a
chemical no-route verdict. They remain failures in this screen, but they identify
a runtime robustness defect rather than a planning limit.

## Consequence for the next iteration

The correct improvement is not to relax element conservation, count condition
predictions as evidence, or add more undirected Codex tokens. The next matched
ablation should preserve all host gates and change only the tail-closure policy:

1. route ChemEnzy/template compute to distinct unclosed leaves immediately after
   a canonical strategic skeleton appears;
2. allocate a meaningful bounded short-search budget, with cached model loading
   and progressive search rather than repeated 90-second cold starts;
3. suppress exact-evidence and Program work while no stock-closed candidate
   exists, then restore those actions after structural closure;
4. stream event-log reads and cap concurrent memory so operational failures do
   not erase otherwise valid trajectories;
5. report a paper-compatible stock endpoint separately from C0-C6, under the
   same frozen stock and full resource ledger.

This preserves AutoPlanner's stronger scientific boundary while directly testing
the hybrid decomposition responsible for SynthEx's reported reach.

## Machine-readable evidence

- `benchmarks/synthatlas_strategy_closure50_v2.v1.json`
- `benchmarks/synthatlas_strategy_closure50_v2.protocol.json`
- `results/shared/synthatlas50_external_snapshot_20260813/summary.json`
- `results/shared/synthatlas50_unified_adaptive_20260813/competitive-summary.json`
- `results/shared/synthatlas50_unified_adaptive_20260813/competitive-summary.md`

Competitive-summary digest:
`9b323da5477fad4d3d1b9f0868c41131c000703c4816b8a1bd9dbc6a9e2a53bf`.
