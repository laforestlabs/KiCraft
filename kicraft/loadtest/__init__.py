"""KiCraft load / stress testing harness.

Drives the design pipeline and build queue under concurrency to find the box's
request and build ceilings -- WITHOUT spending money. Three scenarios:

  - build-storm  : enqueue N `kicraft replay` builds ($0, deterministic) and sweep
                   KICRAFT_BUILD_SLOTS to find the saturation knee (buildstorm.py).
  - pipeline     : run the full LLM design pipeline at $0 via the mock client over
                   the benchmark briefs, then build (pipeline_load.py).
  - web          : external HTTP / websocket load drivers (scripts/loadtest_web*).

All scenarios stream host/process/queue metrics into a LoadResultStore (store.py)
sampled by metrics.py, surfaced live on /admin/loadtest.

The mock LLM (mockllm.py) is selected process-wide by KICRAFT_LLM_MODE=mock|replay
(default `live` -> the real OpenRouter client, so this package is a prod no-op
until explicitly switched on).
"""
