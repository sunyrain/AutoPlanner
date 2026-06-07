# Executable Route Contract

A route can be solved only when all of these hold:

- Target identity matches the requested target.
- Every step has parseable reactants/products and passes structural checks.
- Terminal reactants pass deterministic stock audit.
- No fake closure or raw reaction injection evidence is present.
- Condition gaps and high-risk condition flags do not block the route audit.

Codex may propose plans and gather evidence, but cannot directly set solved. The local validator decides.
