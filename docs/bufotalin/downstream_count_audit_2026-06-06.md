# Bufotalin Downstream Count Audit - 2026-06-06

Run directory:

```text
results/shared/bufotalin_v0_fullflow_harness_20260606_073155
```

Primary compiled downstream artifact:

```text
results/shared/bufotalin_v0_fullflow_harness_20260606_073155/open_structure_research/compiled_downstream_consumables.json
```

## Bottom Line

The count group is internally consistent and should be interpreted as conservative downstream handoff assets, not as solved-route evidence.

```json
{
  "guided_policy_count": 2,
  "template_card_count": 6,
  "route_expansion_task_count": 1,
  "self_evo_staging_candidate_count": 6,
  "one_step_row_count": 0
}
```

The important part is the zero:

- `one_step_row_count=0` is correct.
- No executable reaction row was emitted.
- No production KB promotion is allowed.
- The six template cards are advisory strategy cards only.
- The two guided policies change search bias and terminal blacklists; they do not inject reactions.
- The one route-expansion task is a stuck-node rerun instruction, not a child target set.
- The six self-evo candidates reached staging/shadow bookkeeping only, with `production_write_blocked=true`.

## Route Status Context

The harness final verdict is:

```json
{
  "verdict": "fake_closed_rejected",
  "route_status": "fake_closed_rejected",
  "solved": false,
  "stock_audit_passed": false
}
```

Native ChemEnzy returned `213` raw routes, but the route verifier accepted `0` routes and rejected all `213`.

Verifier reasons:

- `advanced_same_scaffold_terminal`
- `large_atom_jump`
- `no_verifier_accepted_stock_closed_route`

Route audit adds:

- `route_verifier_rejected_raw_routes`

The active terminal blacklist contains the Deacetylbufotalin-like same-scaffold terminal:

```text
C[C@]12CC[C@H](O)C[C@H]1CC[C@@H]1[C@@H]2CC[C@]2(C)[C@@H](c3ccc(=O)oc3)[C@@H](O)C[C@]12O
```

Audit metadata for that terminal:

- heavy atoms: `29`
- target similarity: `0.7656`
- reason: `advanced_same_scaffold_terminal`

So all downstream assets from this run must be treated as route repair / source extraction aids, not closure evidence.

## Count By Count

### `guided_policy_count: 2`

This count comes from `compiled_guided_chemenzy_requests.json`.

There are two compiled ChemEnzy search policies:

| # | policy source | meaning | direct route evidence? |
|---|---|---|---|
| 1 | local literature seed guided rerun | Rerun ChemEnzy with bufadienolide/steroid semisynthesis preferences and the failed terminal blacklist | no |
| 2 | route-expansion stuck-node operator | Convert the route-expansion task into a second guided policy for the stuck frontier | no |

Both policies include:

- `mode: guided`
- `max_depth: 20`
- `max_iterations: 50`
- `expansion_topk: 100`
- terminal blacklist for the advanced same-scaffold terminal
- preferred classes including `bufadienolide_steroid` and `steroid_semisynthesis`

Why the raw seed count was lower:

- `downstream_consumables.json` has `guided_rerun_requests: 1`.
- Compilation adds a second policy from `route_expansion_tasks[0]`.
- Therefore compiled `guided_policy_count=2` is expected.

Operational consequence:

- These are safe to use for another ChemEnzy rerun.
- They are not safe to count as route progress.
- They must not override the fake-closure verdict.

### `template_card_count: 6`

This count comes from `compiled_literature_template_plugin.json`.

All six cards validate as literature template cards, but every one has:

```json
{
  "direct_consumption_allowed": false,
  "direct_one_step_consumption": false,
  "promotion_status": "advisory_only"
}
```

The six cards are:

| # | template | class | relation | direct? |
|---|---|---|---|---|
| 1 | `ev_bufadienolide_c17_pyrone_installation_advisory_template` | `bufadienolide_steroid` | `family_precedent` | `false` |
| 2 | `ev_bufadienolide_c14_c16_late_stage_oxygenation_advisory_template` | `bufadienolide_steroid` | `reaction_precedent` | `false` |
| 3 | `ev_bufadienolide_steroid_chiral_pool_anchor_advisory_template` | `bufadienolide_steroid` | `family_precedent` | `false` |
| 4 | `ev_steroid_late_stage_sidechain_oxidation_halogenation_advisory_template` | `steroid_semisynthesis` | `reaction_precedent` | `false` |
| 5 | `ev_steroid_chiral_pool_semisynthesis_core_policy_advisory_template` | `steroid_semisynthesis` | `family_precedent` | `false` |
| 6 | `ev_saponin_aglycone_glycan_array_disconnection_advisory_template` | `saponin_triterpenoid_steroidal_glycoside` | `reaction_precedent` | `false` |

These are useful as strategic priors:

- prefer C17 2-pyrone installation logic
- prefer late-stage C14/C16 oxygenation / oxidation-state correction
- prefer steroid chiral-pool anchors
- avoid de novo steroid-ring construction as first-pass planning
- avoid accepting advanced product-like steroid terminals as stock

They are not executable templates because none contains a source-grounded full product/reactant SMILES pair.

### `route_expansion_task_count: 1`

This count comes from `compiled_route_expansion_tasks.json`.

The single task is:

```text
bufotalin_v0_fullflow_20260606_local_seed_route_expansion_1
```

Task type:

```text
stuck_node_rerun
```

Target:

```text
route_failure_frontier_or_advanced_intermediate
```

It has `accepted=true`, `no_solved_claim=true`, `production_write_blocked=true`, and `not_raw_reaction_injection=true`.

Key detail:

- `child_targets` is empty.
- The compiled route-expansion result has `child_targets: []`.
- The controller later skipped route expansion with `route_expansion_child_targets_missing`.

So this task is a policy/rerun instruction, not an expanded subgoal tree.

Operational consequence:

- Use it to drive a guided rerun on the failed frontier.
- Do not claim that it created route children.

### `self_evo_staging_candidate_count: 6`

This count comes from the self-evo section of `compiled_downstream_consumables.json`.

The six self-evo candidates correspond one-to-one with the six advisory template cards.

All six candidate validations are `accepted=true`, but the self-evo compile report has:

```json
{
  "production_write_blocked": true,
  "staging_candidate_count": 6
}
```

The KB history records candidate -> shadow -> staging transitions for each candidate. That means the memory/replay layer can inspect them, but production promotion remains blocked.

This is appropriate because the payloads are advisory templates:

- they lack exact product/reactant structures
- they are not one-step rows
- they are derived from target-run local evidence
- they still require source-detail extraction

Operational consequence:

- Keep them as staging candidates for replay / comparison.
- Do not merge them into production KB.
- Do not treat staging as scientific validation.

### `one_step_row_count: 0`

This is the most important gate.

It comes from:

- `compiled_literature_template_plugin.json`: `one_step_rows: []`
- `compiled_executable_template_maturity.json`: `one_step_row_count: 0`
- `source_detail_resolution.summary.source_detail_route_step_count: 0`
- `executable_template_candidates: 0`

The maturity report gives:

```json
{
  "status": "needs_structured_extraction",
  "advisory_template_count": 6,
  "direct_template_card_count": 0,
  "executable_candidate_count": 0,
  "source_detail_route_step_count": 0,
  "route_segment_count": 0,
  "one_step_row_count": 0,
  "production_write_blocked": true
}
```

Required fields before any one-step row can exist:

- `product_smiles`
- `reactant_smiles`
- `source_ref`
- `evidence_refs`
- `relation_type=exact`
- `applicability.product_reconstruction_passed`
- `condition_candidate`

None of the source-detail packs provided those fields for a complete, RDKit-valid exact route step. Therefore zero one-step rows is the correct behavior.

## Source-Detail Blockers

The source-detail resolution pack reports:

```json
{
  "source_detail_route_step_count": 0,
  "curator_record_count": 0,
  "curator_step_count": 0,
  "resolved_queue_count": 0,
  "gap_count": 6
}
```

Rejected / unresolved source-detail items include:

| source | reason | next action |
|---|---|---|
| `6d23b684a5672425` | `metadata_only_source_requires_followup` | typed patent/web connector or curator review |
| `cf4345dc89fb3f23` | `metadata_only_source_requires_followup` | typed patent/web connector or curator review |
| `doi:10.1016/j.tet.2025.134610` | `doi_not_linked_to_pubmed` | DOI landing page, publisher SI, patent connector, or curator record |
| `2a14e41c14da4359` | `metadata_only_source_requires_followup` | typed patent/web connector or curator review |
| `fdc370e8f9709c0a` | `metadata_only_source_requires_followup` | typed patent/web connector or curator review |
| `doi:10.1016/j.steroids.2024.109555` | `no_open_pmc_xml_for_source` | DOI landing page, publisher SI, patent connector, or curator record |

The high-value exact-target / exact-intermediate leads are real leads, but not executable rows in this run:

- `doi:10.1016/j.tet.2025.134610`
- `doi:10.1021/jo00934a013`
- `doi:10.1016/j.steroids.2024.109555`

They require structured extraction into `source_detail_route_step.v1` or curator records before compiling one-step rows.

There is also a smaller source-detail count drift:

- `source_detail_resolution_pack.json` and `open_research_manifest.json` report `gap_count: 6`.
- `open_agent_run_record.json` and `open_agent_audit.json` report `gap_count` / `extraction_gap_count: 5`.
- The authoritative source-detail pack contains six extraction gaps and zero route steps.

This drift does not affect the downstream gate because every source-detail layer agrees on `source_detail_route_step_count: 0`.

## Report Drift Note

There is a count drift between `structure_template_report.md` and the compiled artifacts.

The report text says:

- guided requests: `2`
- template card drafts: `7`
- extraction tasks: `14`
- evolution candidates: `8`

The compiled artifacts say:

- guided policies: `2`
- template cards: `6`
- extraction tasks: `6`
- evolution candidates / staging candidates: `6`
- one-step rows: `0`

For downstream audit, the compiled artifacts should be treated as authoritative because they are the validator/compiler outputs consumed by later steps.

The drift should still be fixed or annotated in the report generator, because it can make users overestimate how much executable work the open-structure agent produced.

## Tool Status Caveat

The open-structure research tool result is `accepted=false` and `status=failed` because of:

- `open_agent_boundary_violation:context_boundary:large_raw_artifact_dump`
- `open_structure_research_nonzero_exit`

However, the harness still compiled downstream artifacts and marked the compiled downstream bundle itself as accepted:

```json
{
  "accepted": true,
  "source": "curator_augmented_downstream",
  "summary": {
    "guided_policy_count": 2,
    "template_card_count": 6,
    "route_expansion_task_count": 1,
    "self_evo_staging_candidate_count": 6,
    "one_step_row_count": 0
  }
}
```

So the right interpretation is:

- The overall open-structure tool run failed audit.
- The compiled downstream handoff is still inspectable and internally gated.
- Nothing in that handoff solves the target or promotes production KB.

## Recommended Next Actions

1. Resolve source details for `doi:10.1021/jo00934a013` and `doi:10.1016/j.tet.2025.134610`.
2. Extract exact product/reactant SMILES pairs into `source_detail_route_step.v1` or curator records.
3. RDKit-validate each product and reactant.
4. Compile only exact, product-reconstructing steps into one-step rows.
5. Replay ChemEnzy with the two guided policies and the terminal blacklist, but keep same-scaffold advanced terminals blocked.
6. Fix the report count drift so narrative counts match compiled artifact counts.
