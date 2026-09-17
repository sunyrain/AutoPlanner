# AutoPlanner Web

The V4 Web surface is a thin adapter over the same `CampaignGateway` used by
the canonical CLI.

New target runs default to `gpt-6-astra` with `medium` reasoning. The homepage
displays and submits the model configuration supplied by
`SYNTHEX_MATCHED_PROFILE_DEFAULTS`, which also supplies the HTTP/CLI defaults.
Explicit model choices and historical run labels retain their original values.
Changes to these Python defaults require restarting the service after active
work has finished; a browser refresh alone does not reload the solver.

```bash
python -m cascade_planner serve
```

Open `http://127.0.0.1:7860/`. This is the only homepage and the only
user-facing retrosynthesis launch entry. The Strategy Generator creates three
orthogonal strategies and the live Route Builder creates the same authoritative background
job, shows the three Strategy Generator cards, and consumes an SSE projection
of the append-only director `model-io.jsonl`: every structured model output
refreshes the Route Builder canvas, while pending steps remain visually distinct
until the next host graph replay supplies canonical precursors. The landing page
also reads the same-origin `/api/v4/jobs` queue, so existing local jobs can be
opened directly (or deep-linked with `/?job=...`) and then followed through the
same SSE stream. Active jobs use their own detail/SSE projection; the federated
catalog refreshes at a lower cadence and never overlaps an unfinished refresh.
Paused runs are non-executing snapshots: they render as paused, their SSE stream
closes after the saved snapshot, and they are never presented as live model
activity. Molecule depictions use the local RDKit SVG endpoint inside bounded
image elements. Routes open at 100% with the target in view; CSS zoom repaints
structures and text at the requested size, up to 400%. Each structure can be
opened in a large viewer and saved as SVG. Native offline route, graph, and
replay exports inline the same display assets and molecule vectors. The PNG
endpoint supplies 1920 × 1200 images for bitmap consumers.

The live homepage's **路线语言 · 中文 / English** control switches saved
strategy, reaction, condition and review text without modifying a job or calling
a model. Chinese is the default; the browser remembers the selection. The
presentation boundary also handles SSE updates and replay snapshots. Switching
language retains pan/zoom, branch selection, replay position and open details.
Untranslated content, activity IO and native downloads remain in their original
language. If the dictionary cannot load, the route still works in the original
language; clicking 中文 retries the load.

Publish a completed, reviewed translation set to the existing local service
without restarting it (this command does not call models):

```bash
python scripts/translate_route_exports.py publish-live --output results/discussion/route-zh-20260917
```

This generates `static/route_translations.zh-CN.json` from `catalog.json` and the
reviewed `zh-CN.json`. Only the display-field whitelist and exact source/Chinese
text pairs are published, not source run paths, model attempts or other raw IO.
Refresh the page after updating the frontend. The initial set covers the eight
2026-09-17 statin tasks, including revised strategies and alternative records.

Critic availability is not a chemical verdict. The projection keeps a missing
final binding (`unavailable`) distinct from an explicit step `pass`, `uncertain`
or `reject`. A missing binding preserves an earlier assessment in
`historical_critic`, with its original task identity and reasons; the details
panel labels it as historical, not applicable final approval. Historical whole
route reviews are read from the saved Director plan within the same strategy
family. Blind `review_slot` values resolve only through the corresponding
Critic task's saved slot-to-step mapping, never array position or another call.
Replay exposes judgments only at their original output event, and attaches the
final unbound-history notice only at the final snapshot. An overall viable
review without a per-step verdict displays “整路已评审 · 未单列步骤判定”, not an
invented pass. This is display recovery; it never modifies the canonical graph,
relaxes final review binding, changes scientific acceptance or calls a model.

Refresh existing native exports without replaying or changing their saved
routes, reviews, selections, and event timelines:

```bash
python scripts/refresh_route_exports.py "results/discussion/absolute-config-*.html"
```

The command rewrites matching native files in place, using current templates
and drawings; unrelated HTML files are skipped. Updating a saved replay
collection also updates every embedded viewer and depiction library.

The Web queue is a paginated federation of explicitly registered run indexes.
It never scans `results/**`: ordinary Web/CLI runs use the main registry, while
an isolated panel becomes visible only after its registry location is published
to `RunRegistryCatalog`. The catalog stores project/case labels and filesystem
boundaries only; lifecycle and scientific state are always read from the owning
`RunIndex`/`RunKernel`. The stable identity is `(registry_id, run_id)`, exposed
as `solve:@<registry_id>:<run_id>`. Bare `solve:<run_id>` identities are no
longer accepted.

Publish a new blind panel deliberately with `--publish-registry`, optionally
supplying `--registry-id`, `--registry-label`, `--registry-project-id` and
`--registry-project-label`. Existing panel directories can be registered without
rerunning them:

```bash
python -m cascade_planner.runtime.run_registry_catalog register \
  --registry-root results/.autoplanner/example-panel/case1 \
  --registry-id example-case1 \
  --project-id example-panel \
  --project-label "Example panel"
```

`GET /api/v4/jobs` accepts `limit`, `offset`, `project_id` and `registry_id`,
and returns `total_count`, `has_more`, project summaries and registry diagnostics.
An unavailable explicit registry is reported without blocking the others.

The mutable Web job row and its background worker are process-local. Restarting
the Web gateway does not resume that thread: durable registry state and existing
`model-io.jsonl` remain inspectable through the catalog. Restart continuity
still requires an explicit durable executor or checkpoint contract; discovery
must not be mistaken for execution ownership.

The `/v4` surface is a results-only operations workspace: it lists live and
historical runs, opens route workbenches, and presents showcase and benchmark
artifacts. New synthesis starts only at `/`. Retired page aliases (`/synthesis`,
`/v4/console`, `/v4/showcase`, `/agent`, `/statins`, and `/showcase`) are no
longer routed. Saved-run replay compatibility remains separate and does not own
V4 graph, frontier, proof, budget, or completion state.

There is no built-in user login. Keep the service on loopback or place it
behind an authenticated reverse proxy. Mutation requests require JSON and may
be protected with `AUTOPLANNER_WEB_API_TOKEN`; provider credentials remain
server-owned and cannot be selected by HTTP payloads.

## External experiment HTTP bridge

The bridge is disabled unless the host sets
`AUTOPLANNER_EXPERIMENT_HTTP_BASE_URL`. It exposes explicitly enabled
submit/poll/cancel transport over the same reserved experiment task; it does
not create a queue, and transport success is not scientific success.

Required host configuration for a remote bridge:

- `AUTOPLANNER_EXPERIMENT_HTTP_BASE_URL`: fixed HTTPS origin and optional base path;
- `AUTOPLANNER_EXPERIMENT_HTTP_BEARER_TOKEN_ENV`: name of the environment
  variable containing the Bearer token, never the token itself;
- optional provider ID/version, operator ID, paths, domain allowlist, timeout,
  response-size and cost limits use the `AUTOPLANNER_EXPERIMENT_HTTP_*` prefix.

Plain HTTP is rejected except for an explicitly enabled loopback test bridge.
Redirects are disabled, external job IDs are URL-escaped, responses are size
bounded, and raw response bodies or credential values are never persisted.
The bridge response contract is one JSON object containing exactly
`external_job_id`, `provider_sequence`, `status`, and `status_detail`.

After `dispatch-experiment`, use `submit-experiment-job` and
`poll-experiment-job`. Cancellation remains two-step:
`cancel-experiment-job` records the audited request, then
`transmit-experiment-cancel` sends it to the configured provider. The matching
HTTP routes are `/experiments/transport/submit`, `/poll`, and `/cancel`.
