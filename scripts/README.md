# Scripts

Active repository utilities for the Codex-entry harness direction only.

| Script | Run semantics | Purpose |
|---|---|---|
| `run_codex_entry_controller.py` | `canonical_agent_controller` | Production mainline: Codex/agent chooses ChemEnzy, literature, hybrid follow-up, and deterministic validators emit the only canonical final verdict. |
| `run_open_structure_template_agent.py` | `probe` tool surface | Codex/WellAU structure-template research launcher used by the canonical controller after a literature/source-detail gate. It is not a standalone solved-route authority. |
| `run_bufotalin_fullflow_wellau.py` | `showcase` | Bufotalin-focused fullflow runner used as the current hard-case replay/report surface. |
| `run_statin_panel_literature_self_evo.py` | `replay` | Nine-statin literature/fullflow/self-evolution replay runner with optional PubMed, closure follow-up, access-probe, and PMC signal-count execution. It must not write production KB by default. |
| `run_smiles_first_literature_workflow.py` | `probe` tool surface | Deterministic SMILES-first workflow runner for target profiling, ChemEnzy audit, literature evidence, and guarded route package output. |
| `run_chem_enzy_plan_for_web.py` | `probe` tool surface | Execute one ChemEnzy native route search from a WebUI JSON request and emit Web-compatible audited JSON. |
| `run_chem_enzy_smoke.py` | Run or dry-run the external ChemEnzyRetroPlanner baseline and write normalized JSON. |
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

Codex should enter at the workflow controller level and decide which tool to
call next. The local scripts above provide deterministic tool execution,
persistence, and validation surfaces; Codex research/planning output is not a
solved-route authority by itself.

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
