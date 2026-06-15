# KiCraft auto-tuning worker (homelab i7 / unraid)

Runs the cross-corpus placement/routing **auto-tune** on a beefy box instead of
the 2-core cloud server. It's pure batch compute — **$0 LLM, no API key**, and
nothing on kicraft.io depends on it. The only thing that flows back to
production is the winning `DEFAULT_CONFIG` (a tiny git commit you make yourself
after reviewing the result).

The container mirrors the production stack (Ubuntu 24.04 + KiCad 9.0 +
FreeRouting 1.9.0 + Java + xvfb) so routing behaves identically, then clones the
repo (code **and** the committed `tuning_corpus/`) into a mounted volume and runs
`kicraft.tuning.cli run`.

## Prerequisites (unraid 6.12.6)
- Docker (built in).
- A fast appdata pool for the data volume (cache SSD/NVMe). The default mount is
  `/mnt/user/appdata/kicraft-tune`. **Do not** point it at the spinning array —
  each eval copies a workspace to scratch, so it's small-file I/O heavy.
- ~3 GB for the image (KiCad is large) + a few hundred MB for the repo/venv/runs.

## Get the files onto unraid
Copy this `deploy/tuning-i7/` folder to the box, e.g. to a share or appdata:
```sh
# from a machine that has the repo, or just scp the 4 files:
scp -r deploy/tuning-i7 root@TOWER:/mnt/user/appdata/kicraft-tune-build
```
(or clone the repo on the box and use `KiCraft/deploy/tuning-i7`).

## Run it — plain Docker (most reliable on unraid)
```sh
cd /mnt/user/appdata/kicraft-tune-build       # wherever you put the Dockerfile
docker build -t kicraft-tune .                # first build pulls KiCad; ~10-20 min

docker run -d --name kicraft-tune \
  -v /mnt/user/appdata/kicraft-tune:/data \
  -e GENS=40 -e SEEDS=0,1 -e POPSIZE=8 -e TIMEOUT=600 -e RUN_ID=i7 \
  kicraft-tune
```
First start clones the repo, builds the venv, sanity-checks `pcbnew` + `cma`,
then begins tuning. Watch it:
```sh
docker logs -f kicraft-tune
```
You'll see a baseline pass, then `gen N: ... bestJ=...` lines. Each generation
checkpoints to `/mnt/user/appdata/kicraft-tune/runs/i7/checkpoint.json`, so it's
safe to stop/restart anytime.

### Or with docker compose
If you have the Docker Compose Manager plugin (or `docker compose`):
```sh
docker compose up -d --build      # edit env in docker-compose.yml first
docker compose logs -f
```

## Tuning knobs (env vars)
| var | default | meaning |
|-----|---------|---------|
| `GENS` | 40 | CMA-ES generations (checkpointed; stop anytime) |
| `SEEDS` | `0,1` | routing seeds/board (K); more = less routing noise, more time |
| `POPSIZE` | 8 | candidates per generation |
| `SCAL` | `balanced` | objective weighting: `correctness` \| `balanced` \| `speed` |
| `TIMEOUT` | 600 | per-eval cap (s); a board that can't route in this fails |
| `RUN_ID` | `i7` | run name → `/data/runs/<RUN_ID>` |
| `KICRAFT_BUILD_SLOTS` | cores/4 | concurrent evals (each is itself multi-threaded) |
| `ACTIVE` | 8 spacing/clearance params | override the tuned param set |

On a 16-core/24-thread i7, expect **minutes per generation** (vs hours on the
cloud box) — so K=2 seeds and the full 16-board corpus are comfortable.

## Watch the charts on kicraft.io (optional)
The admin page (`/admin/tuning`) scans `~/.kicraft/tuning` on the **cloud** box.
To see the i7 run there, rsync the run dir over periodically:
```sh
rsync -a /mnt/user/appdata/kicraft-tune/runs/i7/ \
  CLOUD_HOST:~/.kicraft/tuning/i7/
```
(Only `tuning.db` + `checkpoint.json` matter for the charts; `scratch/` can be
excluded with `--exclude scratch`.)

## Promote the winner
When you've run enough generations, validate the best config on the held-out
boards and print the overlay to apply to `DEFAULT_CONFIG`:
```sh
docker run --rm -v /mnt/user/appdata/kicraft-tune:/data kicraft-tune \
  bash -lc 'cd $KICRAFT_DATA/KiCraft && python -m kicraft.tuning.cli \
            report  --out /data/runs/i7 && python -m kicraft.tuning.cli \
            promote --corpus tuning_corpus --out /data/runs/i7'
```
`promote` re-evaluates the candidate vs the current default on the holdout
boards (sign-test + Pareto-dominance) and, if it wins, writes the overlay to
`/data/runs/i7/promoted_overlay.json`. Apply those keys to
`kicraft/autoplacer/config.py` `DEFAULT_CONFIG`, commit, and push — that's the
only change that touches production.

## Stop / update / resume
```sh
docker stop kicraft-tune && docker rm kicraft-tune   # stop
# re-run the same `docker run ...` to resume — cached evals are reused, and
# updating is automatic (the entrypoint pulls the latest main on each start).
docker run --rm -v /mnt/user/appdata/kicraft-tune:/data kicraft-tune \
  bash -lc 'cd $KICRAFT_DATA/KiCraft && python -m kicraft.tuning.cli resume --out /data/runs/i7'
```

## Troubleshooting
- **FreeRouting jar 404 on build** — the v1.9.0 asset URL changed; update it in
  the Dockerfile (`freerouting/freerouting` releases).
- **`pcbnew` import fails** — the KiCad PPA didn't install; check the build log
  for the `add-apt-repository ppa:kicad/kicad-9.0-releases` step.
- **Slow / timeouts** — raise `TIMEOUT`, or drop the 2-3 slowest boards by
  setting a smaller corpus (point `--corpus` at a trimmed dir), or lower
  FreeRouting effort. Each eval's wall-time shows in the admin "build time" chart.
