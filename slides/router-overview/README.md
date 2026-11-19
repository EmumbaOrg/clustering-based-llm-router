# Router overview slides

`router-overview-slides.md` is a [Marp](https://marp.app) deck. Every image it shows is in `img/`,
and everything needed to regenerate those images is in `sources/`. `router-overview-slides.pdf` is
an export of the current deck.

The deck covers the four current candidates only: gpt-6.1-sol, gpt-5.4-mini, gpt-5.6-luna and
gpt-5-nano.

## View or export

```bash
npx @marp-team/marp-cli router-overview-slides.md --pdf --allow-local-files    # or --pptx, --html
```

Or open the file in VS Code with the Marp extension.

## How the images were generated

There are two kinds of images.

### Diagrams (7): Mermaid, rendered with mermaid-cli

| Image | Source |
|---|---|
| `img/big-picture.png` | `sources/big-picture.mmd` |
| `img/offline-pipeline.png` | `sources/offline-pipeline.mmd` |
| `img/calibration.png` | `sources/calibration.mmd` |
| `img/grading.png` | `sources/grading.mmd` |
| `img/artifacts.png` | `sources/artifacts.mmd` |
| `img/runtime-overview.png` | `sources/runtime-overview.mmd` |
| `img/runtime-inputs.png` | `sources/runtime-inputs.mmd` |

Requires Node.js. From this folder:

```bash
cd sources
for n in big-picture offline-pipeline calibration grading artifacts runtime-overview runtime-inputs; do
  npx -y @mermaid-js/mermaid-cli -i $n.mmd -o ../img/$n.png \
    -c mermaid-config.json -p puppeteer.json -s 2 -b white
done
```

- `mermaid-config.json` sets the neutral theme, 18px font and label wrapping width.
- `puppeteer.json` passes `--no-sandbox` to the headless Chrome mermaid-cli uses.
- `-s 2` renders at 2x scale so the PNGs stay sharp on slides.

### Charts (4): matplotlib, built from the real pipeline artifacts

| Image | What it shows | Data read |
|---|---|---|
| `img/corpus-mix.png` | tasks per dataset | `artifacts/task-cluster-map.json` |
| `img/sources-per-cluster.png` | dataset share inside each of the 24 clusters | `artifacts/task-cluster-map.json` |
| `img/profile-heatmap.png` | smoothed error rate per candidate per cluster, plus overall pass rate | `artifacts/model-profiles.json` |
| `img/lambda-crossover.png` | routing score vs. λ for cluster 3 (the four candidates) | `artifacts/model-profiles.json`, `config/models.yaml` |

These need the `clustering-based-llm-router` repo and its virtualenv, because the script reads the
artifacts and reuses the router's own scoring code (`router.common.scoring.score_candidates`). From
the repo root:

```bash
uv run python <path to this folder>/sources/build_figures.py
```

It writes all four PNGs into the `img/` folder next to `sources/`. The candidate list is the
`CANDIDATES` dict at the top of the script. Rerun it whenever
`artifacts/model-profiles.json` or `artifacts/task-cluster-map.json` changes so the slides match
the current data. The figures in this folder were built from the artifacts as of 2026-11-19
(profiles `profiles-2026-11-03-merged`, cluster map `clustermap-2026-09-01-k24-s42`).

### Screenshot (1)

`img/router-ui.png` is a screenshot of `router runtime ui` routing the python-docx example
(`?example=11&lambda=0.1`).

After regenerating any image, re-export the deck. Slide numbers and text were written against
these exact images, so check the takeaways (e.g. "17 / 7 / 0 clusters at λ = 0", "pick flips at
λ ≈ 0.15 and 0.37", "mini never wins") still hold.

If mermaid-cli or Marp can't download its own Chrome, point them at an installed one:
`PUPPETEER_SKIP_DOWNLOAD=1` plus `"executablePath": "/usr/bin/google-chrome"` in `puppeteer.json`
for mermaid-cli, and `CHROME_PATH=/usr/bin/google-chrome` for Marp.
