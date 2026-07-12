# Local testing

AutoPlanner's default test suite is deterministic and offline. It does not
launch Codex, call a hosted model, read `key.txt`, use a GitHub SSH key, fetch
literature, or require model weights. Live provider checks are separate,
explicitly opted-in activities. This repository intentionally has no GitHub
Actions workflow; validation is run locally before a direct release push.

## Test environments

Python 3.11 and 3.12 are supported local development versions. Install the
standalone offline test stack with:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

For a CPU-only workstation, install the matching CPU wheel for PyTorch before
`requirements-dev.txt`. `requirements.txt` is needed for inference development,
not for the deterministic contract suite.

## Fast architecture contracts

This layer exercises the current route graph, source fusion, proof boundary,
provider registry, stock snapshots, persistent frontier scheduler, route
portfolio, campaign recovery, runtime atomicity, and generic hardcoding guard.
Run this selection before committing architecture changes:

```bash
python -m pytest -q -p no:cacheprovider \
  tests/test_child_agent_runtime_contracts.py \
  tests/test_generic_hardcoding_guards.py \
  tests/test_route_consensus.py \
  tests/test_route_consensus_graph.py \
  tests/test_route_source_adapters.py \
  tests/test_route_verifier.py \
  tests/test_reaction_step_verifier.py \
  tests/test_builtin_providers.py \
  tests/test_chem_enzy_runtime_preflight.py \
  tests/test_portfolio_controller_integration.py \
  tests/test_provider_registry.py \
  tests/test_stock_provider.py \
  tests/test_frontier_scheduler.py \
  tests/test_frontier_ledger.py \
  tests/test_codex_edge_verification.py \
  tests/test_blackboard_events.py \
  tests/test_admitted_hyperedges.py \
  tests/test_action_evidence_loop.py \
  tests/test_codex_retrosynthesis_team.py \
  tests/test_codex_team_source_lifecycle.py \
  tests/test_codex_team_controller_integration.py \
  tests/test_evidence_first_controller_integration.py \
  tests/test_literature_source_documents.py \
  tests/test_compound_label_bindings.py \
  tests/test_route_portfolio.py \
  tests/test_portfolio_supplemental_bindings.py \
  tests/test_route_admission.py \
  tests/test_chem_enzy_guidance.py \
  tests/test_route_forest.py \
  tests/test_route_forest_layout.py \
  tests/test_route_forest_delivery.py \
  tests/test_audit_architecture_v2.py
```

PowerShell accepts the same paths; either put the command on one line or use
PowerShell backticks instead of shell backslashes.

## Complete offline suite

The trusted literature registry fixture is deliberately test-only. Point the
test process at it explicitly so production code keeps the packaged empty,
fail-closed registry.

POSIX shell:

```bash
export AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY=tests/fixtures/trusted_literature_step_registry.json
export AUTOPLANNER_LIVE_CODEX_ENTRY_SMOKE=0
python -m pytest -q -p no:cacheprovider --durations=20
```

PowerShell:

```powershell
$env:AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY = 'tests/fixtures/trusted_literature_step_registry.json'
$env:AUTOPLANNER_LIVE_CODEX_ENTRY_SMOKE = '0'
python -m pytest -q -p no:cacheprovider --durations=20
Remove-Item Env:AUTOPLANNER_TRUSTED_LITERATURE_STEP_REGISTRY
Remove-Item Env:AUTOPLANNER_LIVE_CODEX_ENTRY_SMOKE
```

Three live/optional tests are expected to skip in the default environment. A
skip is not a solved-route assertion and does not relax any parent proof gate.

The central P0 tests must cover these failure boundaries:

- campaign attempts are counted from immutable started events across resumes,
  separately from accepted expansions, and global budgets cannot shrink;
- strict child acceptance remains the default, while `valid_subset_l0` accepts
  only a host-observed quorum after complete spawn coverage; a 4/4 run retains
  the strict tier, while an actually used subset fallback caps every recovered
  edge to non-authoritative L0; hard coordinator/runtime/tool failures still
  reject, and both tiers survive immutable commit/restart replay;
- campaign execution and proof reconciliation share a whole-transaction OS
  lock, so concurrent callers cannot consume the final accepted slot twice;
- proof-only reconciliation keeps its per-call expansion delta separate from
  cumulative durable/external input-event and deduplicated canonical-edge
  counts. Architecture audit binds all scheduler facts and the ledger to the
  current CAS reconciliation queue, and a missing CAS reconciliation fails
  closed rather than using a stale team-report or compatibility projection;
- a valid prepared expansion commit can be adopted after an interruption only
  under the exact campaign/job/attempt/lease fence; malformed or terminally
  failed work cannot be adopted;
- a non-root precursor stays proposal-ineligible until a current-host L2 proof
  binds one of its exact inbound parent step IDs;
- a late exact row retains its complete reactant set and mapped reaction,
  unlocks the matching child exactly once, and reconciliation consumes neither
  Agent attempts nor accepted-expansion budget;
- source-bound PDF material may enter the canonical graph only as L0 search
  admission; tampered provenance is quarantined and only the out-of-band
  registry may grant literature/L3 precedent;
- the caller-advisory graph may contain unsupported suggestions, while the
  ledger, RouteForest authority stages, and closeout dependencies bind the
  smaller canonical durable graph;
- blackboard checkpoints use CAS/tombstones and action outbox replay, and only
  a final unterminated crash fragment may be forensically isolated; terminated
  corruption, duplicate keys, non-finite values, and identity/digest drift fail
  closed;
- impossible self/ancestor cycles, element deficits, and large atom jumps are
  rejected by the shared consensus/ChemEnzy admission gate before ranking;
- per-edge materialization cache hits are replayed by the current verifier,
  while tampering, input/version drift, and injected mappers fail closed or
  bypass persistence;
- Codex self-reported evidence/confidence cannot raise authority ranking;
- the action planner receives a digest-bound, directly embedded decision
  snapshot and never needs shell/filesystem reads; its byte limit remains a
  hard bound under arbitrarily many/long dictionary keys, and both main and
  repair tasks fail closed on digest or declared-size drift;
- ChemEnzy's bounded guidance batch is selected from digest-bound canonical
  frontier state before truncation, remains deterministic and structurally
  diverse, and audits selected/dropped IDs while ignoring model-authored
  confidence/evidence/validation flags;
- ChemEnzy filesystem discovery alone is non-production; launch requires an
  isolated-interpreter capability probe with verified vendor imports and
  readable request-effective model/stock paths. CLI, controller, and Web
  selection must agree; cache hits require the same runtime/config/request
  identity, concurrent identical probes coalesce, and a failure is not probed
  twice. Windows tests keep import roots normal, apply device prefixes only to
  overlong concrete I/O paths, prune individual missing models, and fail closed
  for unknown/incomplete model or stock configuration and all-missing models;
- source group, logical document, and concrete representation counts remain
  distinct, and a source-local compound-label/structure conflict fails closed;
- benchmark/search and procurement stock planes are replayed through trusted
  provider instances and all four ledger fixed points are recomputed;
- UI stage membership is derived from current ledger/queue/edge/leaf evidence,
  never from branch count or colour; the fully-expanded stage requires every
  nonempty step to have a succeeded queue binding, while partial i/N progress
  remains non-authoritative and legacy/empty filtered views fail closed.

These are integrity and authority-replay tests, not cryptographic-authentication
tests. A matching SHA-256 detects canonical-content drift; it is not a security
signature for a repository or run directory controlled by an attacker.

Whole-repository strict Ruff is not yet an honest gate: the retained research
and legacy surfaces have pre-existing style debt. Apply syntax/undefined-name
checks to all active Python and normal strict Ruff rules to the upgraded
application, provider, route, runtime, proof, and test surfaces. Move a module
into the strict set when modernizing it; do not silence new findings with
blanket ignores.

## Safety and hardcoding gates

Before a direct push, verify that `key.txt`, `psaaword.txt`, non-example `.env`
files, private-key headers, and live token signatures are not tracked.
`.env.example` and `.env.local.example` may contain placeholders, never usable
credentials.

Also run `git diff --check`, inspect `git status --short`, and verify that
`.github/workflows/` contains no workflow. There is no CI/Actions fallback: a
direct push is allowed only after the local targeted and complete offline tests
have passed.

`tests/test_generic_hardcoding_guards.py` is the fast molecule-hardcoding gate.
New planner behavior should be driven by structure, typed evidence, provider
contracts, or caller policy. A target-name branch or molecule-specific rescue
must not be added to the generic mainline.

## Adding tests

- Put deterministic tests in `tests/test_*.py`; they automatically join the
  complete suite.
- Add a test to the fast architecture list only when it covers a central
  contract and stays small and offline.
- Use temporary directories for artifacts and catalogs. Do not write into
  `results/`, `workspace/`, or the repository root.
- Mock provider/network boundaries. Never make a required test depend on a
  Codex login, an API key, institutional browser state, or live supplier data.
- Mark a truly live test with `@pytest.mark.live`, require an explicit opt-in
  environment flag, and make the default behavior skip rather than fail.
- Preserve fail-closed expectations: advisory/model output alone cannot assert
  stock closure, reaction validation, trusted precedent, or a solved parent
  route.
