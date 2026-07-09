# Fake Closure Policy

Fake closure is any route state that appears terminal but leaves the hard retrosynthetic problem unresolved.

Reject solved status when:
- The frontier is target-like, same-scaffold, or has no meaningful complexity drop.
- Terminal reactants are not proven stock molecules by deterministic audit.
- A route anchor, analogy, or forward surrogate is presented as a stock-closed executable route.
- Any payload includes raw reaction injection or unvalidated reaction candidates.

Fake closure evidence must force `fake_closed_rejected` or a non-solved follow-up verdict.
