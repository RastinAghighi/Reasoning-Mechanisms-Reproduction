# Results

Run directories are created here automatically, one per execution:

```
results/<experiment>/<YYYYmmdd-HHMMSS>_s<seed>_<config-fingerprint>/
    config.yaml     resolved config
    metadata.json   git commit, package versions, overrides, status
    artifacts/      predictions, scores, tables, plots
    logs/
```

Everything here is **git-ignored** except this file and `summaries/`.
Copy small, curated outputs (`.md`, `.csv`, `.json`) that back a claim in
`docs/` into `results/summaries/`, named after the run they came from.
