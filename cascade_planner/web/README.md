# AutoPlanner Web

The V4 Web surface is a thin adapter over the same `CampaignGateway` used by
the canonical CLI.

```bash
python -m cascade_planner serve
```

Open `http://127.0.0.1:7860/v4`. The default server is Waitress; use
`--server flask` for local debugging. The JSON API lives at `/api/v4/runs`.

The older `/`, `/agent`, ChemEnzy job, and saved-result endpoints remain frozen
compatibility surfaces until P10 replay migration. They may read legacy files
but do not own V4 graph, frontier, proof, budget, or completion state.

There is no built-in user login. Keep the service on loopback or place it
behind an authenticated reverse proxy. Mutation requests require JSON and may
be protected with `AUTOPLANNER_WEB_API_TOKEN`; provider credentials remain
server-owned and cannot be selected by HTTP payloads.
