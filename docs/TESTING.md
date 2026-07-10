# Testing and CI

AutoPlanner's default test suite is deterministic and offline. It does not
launch Codex, call a hosted model, read `key.txt`, use a GitHub SSH key, fetch
literature, or require model weights. Live provider checks are a separate,
explicitly opted-in activity and are never part of required CI.

## Test environments

Python 3.11 is the cross-platform CI version. Python 3.12 is the primary local
development version. Install the standalone offline test stack with:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

The CI workflow first installs the CPU wheel for PyTorch from the official
PyTorch wheel index, then installs `requirements-dev.txt`. This prevents a test
job from acquiring a GPU runtime. `requirements.txt` is needed for inference development,
not for the deterministic contract suite.

## Fast architecture contracts

This layer exercises the current route graph, source fusion, proof boundary,
provider registry, stock snapshots, persistent frontier scheduler, route
portfolio, runtime atomicity, and generic hardcoding guard. It is the same
selection run on Linux and Windows for every pull request:

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
  tests/test_route_portfolio.py \
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

## CI layers

| Job | Trigger | Scope |
| --- | --- | --- |
| `quality-and-safety` | Every PR and `main` push | Fatal Ruff rules across Python; strict Ruff on upgraded architecture; tracked credential/private-token scan. |
| `architecture-contracts` | Every PR and `main` push | Fast proof/fusion/runtime/hardcoding selection on both Linux and Windows. |
| `full-linux` | Every PR and `main` push | Every offline test, with a JUnit report. |
| `full-windows` | Weekly schedule or manual dispatch | Every offline test on Windows, with a JUnit report. |

The workflow grants only `contents: read`, disables checkout credential
persistence, supplies no API secrets, and explicitly disables the live Codex
smoke flag. Required tests must remain runnable under that policy.

Whole-repository strict Ruff is not yet an honest gate: the retained research
and legacy surfaces have pre-existing style debt. CI therefore applies
syntax/undefined-name rules to all active Python and the normal strict Ruff
rules to the upgraded application, provider, route, runtime, proof, and test
surfaces. Move a module into the strict list when modernizing it; do not silence
new findings with blanket ignores.

## Safety and hardcoding gates

CI rejects tracked `key.txt`, `psaaword.txt`, and non-example `.env` files. It
also scans tracked content for private-key headers and common live token
signatures. `.env.example` and `.env.local.example` may contain placeholders,
never usable credentials.

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
