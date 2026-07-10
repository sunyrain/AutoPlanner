# Scripts

Active repository utilities for the Codex-entry harness direction only.

| Script | Run semantics | Purpose |
|---|---|---|
| `run_codex_entry_agentic_blackboard.py` | `canonical_multi_agent_controller` | Active mainline: Codex coordinators directly spawn specialist agents for the target and bounded precursor frontiers, assemble a typed multi-source route graph, and delegate final proof authority to deterministic validators. |
| `run_codex_entry_controller.py` | `compatibility_controller` | Older controller entry retained for saved-run and integration compatibility; new orchestration belongs in the agentic-blackboard launcher. |
| `run_open_structure_template_agent.py` | `probe` tool surface | Codex/WellAU structure-template research launcher used by the canonical controller after a literature/source-detail gate. It is not a standalone solved-route authority. |
| `run_bufotalin_fullflow_wellau.py` | `showcase` | Bufotalin-focused fullflow runner used as the current hard-case replay/report surface. |
| `run_statin_panel_literature_self_evo.py` | `replay` | Nine-statin literature/fullflow/self-evolution replay runner with optional PubMed, closure follow-up, access-probe, and PMC signal-count execution. It must not write production KB by default. |
| `run_smiles_first_literature_workflow.py` | `probe` tool surface | Deterministic SMILES-first workflow runner for target profiling, ChemEnzy audit, literature evidence, and guarded route package output. |
| `run_chem_enzy_plan_for_web.py` | `probe` tool surface | Execute one ChemEnzy native route search from a WebUI JSON request and emit Web-compatible audited JSON. Legacy molecule-specific semisynthesis rescue is loaded only for an explicit `enable_semisynthesis_rescue=true` historical replay. |
| `run_chem_enzy_smoke.py` | Run or dry-run the external ChemEnzyRetroPlanner baseline and write normalized JSON. |
| `validate_example_runs.py` | Validate representative saved runs, including final verdict expectations and generated route-forest HTML. |
| `validate_legacy_example_runs.py` | Validate older non-blackboard saved examples such as Codex-entry statin runs, statin-panel dossiers, and retrieval/latest probes. |
| `smoke_route_forest_history.py` | Read-only broad smoke over saved `agent_blackboard.json` histories; compiles and renders route forests in memory. |
| `evaluate_agentic_run.py` | Standard-library-only evaluator for one saved agentic run, or a baseline/final comparison with fail-closed parent-proof and advisory-route semantics. |
| `audit_architecture_v2.py` | Report file-presence capability surface, materialized executable-contract evidence, and one run's chemistry acceptance separately. For committed runs it reads CAS proof/verdict/graph/forest first, audits every portfolio item/hash/selected DAG, and reports mutable compatibility drift without overriding CAS authority. |
| `migrate_codex_campaign_v2.py` | Upgrade a saved pre-outbox Codex frontier campaign to immutable fenced per-job expansion commits without rerunning model work. |
| `refresh_agentic_closeout_artifacts.py` | Recompute hypothesis reports and final verdicts for a saved agentic run without rerunning model/PDF/ChemEnzy work. |
| `resume_agentic_blackboard.py` | Continue a saved agentic run from `agent_blackboard.json`, preserving existing artifacts, budget counters, and closeout auditing. |
| `monitor_autoplanner_web.py` | Poll the local WebUI service, queue state, CUDA status, and recent output/rejected artifacts. |
| `run_autoplanner_web_waitress.py` | Start the local Waitress-backed WebUI service. |
| `reaudit_route_pool.py` | Refresh product/condition audit metadata for an exported route-pool JSON after audit-rule changes. |
| `audit_strategic_disconnections.py` | Audit curated strategic-disconnection source files for ID uniqueness, family coverage, traceability, and compliance gates. |
| `query_strategic_disconnections.py` | Query the strategic-disconnection source layer by text or `family_id`. |
| `local_pdf_proxy.py` | Queue DOI/URL PDF requests on the server, then fetch them on a local machine with authorized school/library access. |
| `sync_local_pdf_proxy.py` | Local-machine rsync loop that pulls PDF requests from a server, runs the local downloader, and pushes downloaded PDFs/manifests back. |
| `tsinghua_pdf_gateway.py` | Experimental Tsinghua Library smart-gateway/Eproxy browser-session helper for authorized PDF access. |
| `download_brenda.py` | Download or refresh BRENDA condition data inputs. |
| `setup_chem_enzy_runtime.sh` | Download/unpack the ChemEnzy runtime under `/root/autodl-tmp`. |
| `setup_chem_enzy_vendor.sh` | Clone/update the ignored ChemEnzyRetroPlanner vendor checkout. |

Archived experiment, ONMT, verifier-training, rendering, and statin-panel scripts
were moved locally under `archive/harness_prep_20260605/scripts/`.

## Current Boundary

Codex enters through the multi-agent coordinator and then the workflow action
planner. Specialist children produce typed draft candidates; the consensus
layer retains independent source records and conflicts. Local code owns tool
execution, persistence, validation, stock audit, and parent-route proof. Codex
research/planning output is never solved-route authority by itself.

For a stitched parent route, each exact literature edge must be bound to a
trusted PDF page and every terminal reactant frontier must have its own
reverified stock closure. A child route, consensus DAG, or accepted visual chain
is displayable evidence but not parent-route proof. The route forest creates a
`stitched_verified_route` only by replaying the accepted proof's embedded
inputs.

## Validation Gates

Run these checks after changing blackboard, final-verdict, or route-forest
logic:

```powershell
D:\conda\envs\py312\python.exe scripts\validate_example_runs.py --summary-output results\shared\example_run_validation_extended_default_20260705.json
D:\conda\envs\py312\python.exe scripts\validate_legacy_example_runs.py --summary-output results\shared\legacy_example_run_validation_20260705.json
D:\conda\envs\py312\python.exe scripts\smoke_route_forest_history.py --root results\shared --summary-output results\shared\route_forest_broad_smoke_20260705.json
D:\conda\envs\py312\python.exe -m pytest -q
```

`validate_example_runs.py` checks the current representative examples against
expected final verdicts, solved flags, route status, and route-forest HTML
fragments. `validate_legacy_example_runs.py` covers saved artifacts that predate
`agent_blackboard.json`, so old Codex-entry/statin-panel examples do not fall
out of regression coverage. `smoke_route_forest_history.py` is broader and
read-only: it scans saved blackboard histories, compiles route-forest data,
renders HTML in memory, and reports cases where a non-empty run would otherwise
produce no displayable branch.

Evaluate one run as JSON, or compare a baseline with a candidate/final run:

```powershell
D:\conda\envs\py312\python.exe scripts\evaluate_agentic_run.py results\shared\RUN_DIR --human
D:\conda\envs\py312\python.exe scripts\evaluate_agentic_run.py results\shared\BASELINE --compare-to results\shared\FINAL --output results\shared\agentic_run_comparison.json --human
```

The evaluator reports model/runtime claims separately from deterministic proof
status. Advisory, consensus, process-evidence, and otherwise unverified route
branches never contribute to its strictly usable solved-route count.

When the local, git-ignored paclitaxel run directories are present, compare
them with:

```powershell
D:\conda\envs\py312\python.exe scripts\evaluate_agentic_run.py `
  results\shared\paclitaxel_codex_baseline_20260710 `
  --compare-to results\shared\paclitaxel_codex_improved_20260710 `
  --output results\shared\paclitaxel_codex_improved_20260710\evaluation_vs_baseline.json `
  --human
```

Its final status is intentionally unresolved: two child targets closed, but no
strict literature-to-parent proof survived deterministic verification.

When only closeout/reporting logic changes, refresh a saved run without
rerunning expensive tools:

```powershell
D:\conda\envs\py312\python.exe scripts\refresh_agentic_closeout_artifacts.py results\shared\RUN_DIR
```

The refresh recomputes `hypothesis_only_retrosynthesis_report.json`,
`hypothesis_execution_report.json`, `final_verdict.json`, and
`agentic_final_verdict_validation.json` from the current blackboard and existing
route-expansion/proof artifacts. It is intended for projection/verdict fixes,
not for creating new evidence.

To continue a saved blackboard run with new evidence-producing actions:

```powershell
D:\conda\envs\py312\python.exe scripts\resume_agentic_blackboard.py results\shared\RUN_DIR --plan-only --exhaust-round-budget
D:\conda\envs\py312\python.exe scripts\resume_agentic_blackboard.py results\shared\RUN_DIR --max-new-rounds 1 --exhaust-round-budget --emit-blackboard-steps
```

The resume path rebuilds PDF-evidence indexes from existing artifacts before
planning, so visual extraction can reuse rendered pages from earlier rounds.
Use `--plan-only` first when auditing what the next action batch will do.

## Local PDF Proxy

Use this path when the remote server cannot access your school databases. The
open-research agent should first try native web/source access and record whether
the source is readable. When the agent only gets metadata, a login page, or a
paywall, the server writes metadata-only requests; your local machine performs
authorized downloads through campus VPN/library access; then you sync the
returned PDFs and manifest back to the server. Do not put school credentials,
cookies, or browser profiles in the repo or prompt context.

On the server, queue one or more targets:

```bash
python scripts/local_pdf_proxy.py request --doi '10.1021/example'
python scripts/local_pdf_proxy.py request --url 'https://publisher.example/article'
python scripts/local_pdf_proxy.py request \
  --source-material-locator results/shared/RUN/evidence/source_material_locator_pack.json \
  --output-dir results/shared/RUN
```

Sync `results/shared/RUN/evidence/local_pdf_proxy/` to your laptop. On the
laptop, connect the school VPN/library network and run:

```bash
python scripts/local_pdf_proxy.py fetch --output-dir results/shared/RUN
python scripts/local_pdf_proxy.py status --output-dir results/shared/RUN
```

Sync the same `local_pdf_proxy/` directory back to the server. Successful
downloads appear in `pdf_download_manifest.jsonl`; login/landing-page cases are
marked `needs_manual_access` instead of storing any sensitive session state.

To automate the pull-download-push loop from your laptop:

```bash
python scripts/sync_local_pdf_proxy.py \
  --server USER@SERVER \
  --remote-output-dir /root/autodl-tmp/AutoPlanner/results/shared/RUN \
  --interval-s 300
```

Run it from a local AutoPlanner checkout while connected to the school
VPN/library network. Use `--max-items 3` for a small first trial.

## Tsinghua Library PDF Gateway Probe

`tsinghua_pdf_gateway.py` is a local-only helper for the recommended Tsinghua
Library database navigation / smart gateway / Eproxy route. It does not store a
school username or password. It only reuses a persistent Chromium profile under
`results/shared/tsinghua_pdf_gateway/browser-profile`, which should be treated
as sensitive login state.

Install optional browser dependencies:

```bash
python -m pip install playwright
python -m playwright install chromium
python -m playwright install-deps chromium  # only needed if Chromium reports missing Linux libraries
```

Run the lightweight network/browser check:

```bash
python scripts/tsinghua_pdf_gateway.py check
python scripts/tsinghua_pdf_gateway.py doctor
```

Create or refresh the browser login state:

```bash
python scripts/tsinghua_pdf_gateway.py login
```

On a headless remote server with no `DISPLAY`, a headed browser cannot pop up
directly. If the provider offers noVNC or a remote desktop, run the command
inside that desktop. Otherwise try an experimental DevTools login:

```bash
python scripts/tsinghua_pdf_gateway.py login --headless --cdp-port 9222
```

Then forward/open the DevTools endpoint securely from your own machine. For
example, keep the `login` command running on the server and run this on your
laptop:

```bash
ssh -N -L 9222:127.0.0.1:9222 USER@SERVER
```

Open `http://127.0.0.1:9222/json/list` locally, then open the listed
`devtoolsFrontendUrl` for the Tsinghua page and complete authentication. Press
Enter in the server terminal only after login finishes. Do not expose the CDP
port publicly because it controls the browser session.

Download one authorized target at a time:

```bash
python scripts/tsinghua_pdf_gateway.py download --doi '10.1021/example'
python scripts/tsinghua_pdf_gateway.py download --url 'https://example.publisher/path'
```

For Tsinghua, the most reliable URL is usually the database-navigation or
database-detail "access" link that triggers the smart gateway, especially links
beginning with `http://tlink...`. Bare DOI URLs may not trigger institutional
access by themselves.
