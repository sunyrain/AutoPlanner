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
also reads the same-origin `/api/v4/jobs` queue every three seconds, so existing
local jobs can be opened directly (or deep-linked with `/?job=...`) and
then followed through the same SSE stream. Molecule depictions are fetched from
the local RDKit endpoint, validated as SVG, and inserted inline rather than
loaded as external images.

The job queue is scoped to this Web gateway. A CLI experiment that creates its
own `run_index.sqlite3` under an isolated output directory is a separate run
registry and is not auto-imported by scanning `results/**`. Launch any smoke
that must appear in the live page through `POST /api/v4/jobs` (or explicitly
register it with the same gateway); its append-only `model-io.jsonl` remains
the sole live progress source.

The mutable job row and its background worker are process-local. Restarting the
Web gateway does not resume that thread: the durable run is shown as a
`historical_snapshot`, and its existing `model-io.jsonl` can still be replayed
for inspection. Restart continuity requires an explicit durable executor or
checkpoint contract; it must not be inferred from a results-directory scan.

The `/v4` surface is now a results-only operations workspace: it lists live and
historical runs, opens route workbenches, and presents showcase and benchmark
artifacts. It no longer owns a ChemEnzy-first launch form. `/synthesis` and
`/v4/console` redirect to the homepage; `/agent`, `/statins`, `/showcase`, and
`/v4/showcase` remain compatibility redirects to result sections of `/v4`. Legacy
APIs remain available during replay migration but do not own V4 graph,
frontier, proof, budget, or completion state.

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
