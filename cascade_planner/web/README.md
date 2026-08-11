# AutoPlanner Web

The V4 Web surface is a thin adapter over the same `CampaignGateway` used by
the canonical CLI.

```bash
python -m cascade_planner serve
```

Open `http://127.0.0.1:7860/v4`. This is the only user-facing page: it launches
SMILES retrosynthesis jobs, lists live and historical runs, opens route
workbenches, and presents showcase and benchmark artifacts. The default server
is Waitress; use `--server flask` for local debugging. The JSON API lives at
`/api/v4/runs`.

The older `/`, `/agent`, `/statins`, `/showcase`, `/v4/console`, and
`/v4/showcase` paths are compatibility redirects to sections of `/v4`. Legacy
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
