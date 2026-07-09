---
name: autoplanner-route-controller
description: Use when controlling AutoPlanner retrosynthesis runs through the repo-versioned Codex-entry harness, choosing ChemEnzy, SMILES-first literature, or open structure research tools while preserving deterministic final verdict gates.
---

# AutoPlanner Route Controller

Use this skill when acting as Codex inside the AutoPlanner Codex-entry harness.

## Operating Rules

- Start at workflow planning. Choose one strategy: `chem_enzy_first`, `literature_first`, `hybrid`, or `reject_invalid_input`.
- Call only repo-controlled local tools listed in the harness plan. Do not invent tool names.
- Treat literature research as evidence gathering, not route closure.
- Do not emit raw reaction SMILES, raw reaction candidate lists, route-tree mutations, or production KB writes.
- Do not claim `solved`. Deterministic stock/route audit is the only solved gate.
- Keep all generated files inside the supplied run directory.

## References

- Fake closure policy: `references/fake_closure_policy.md`
- Literature anchor policy: `references/literature_anchor_policy.md`
- Executable route contract: `references/executable_route_contract.md`
- Bufotalin breakpoint: `references/bufotalin_breakpoint.md`
