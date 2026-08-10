"""The online routing runtime: given one prompt, embed it, assign it to a cluster, and pick a
model — `predicted_error + lambda * normalised_cost` — per
`../../../docs/specs/2026-08-04-cluster-routing-implementation-plan.md` (Phases 2-4). See each
module's docstring for what's already implemented in `../common/` versus what's still a stub here.

Not yet wired to anything: no host agent, no HTTP service. `cli.py` is the only way to exercise it
today.
"""
