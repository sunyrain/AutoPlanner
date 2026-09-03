# AutoPlanner Web

The V4 Web surface is a thin adapter over the same `CampaignGateway` used by
the canonical CLI.

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
activity. Molecule depictions are fetched from the local RDKit endpoint,
validated as SVG, and inserted inline rather than loaded as external images.

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
