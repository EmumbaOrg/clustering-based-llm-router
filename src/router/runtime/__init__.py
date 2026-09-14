"""The online routing runtime: given one prompt, embed it, assign it to a cluster, and pick a
model — `predicted_error + lambda * normalised_cost`. Not yet wired to anything: no host agent,
no HTTP service. `cli.py` is the only way to exercise it today.
"""
