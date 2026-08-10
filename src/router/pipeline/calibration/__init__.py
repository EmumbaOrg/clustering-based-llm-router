"""Task selection -> grading -> smoothed error rates -> model-profiles.json. Consumes
cluster-map.json (via ..common.assign) rather than owning cluster definitions itself;
../evaluate.py replays this stage's output against a holdout split."""
