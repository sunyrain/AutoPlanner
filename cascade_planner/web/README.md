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
