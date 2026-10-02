# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Self-hosted LTX-2.5 video and FLUX.1-schnell image generation service on RunPod
Serverless. A FastAPI backend proxies auth/rate-limiting/moderation/queueing in
front of two independent RunPod Serverless endpoints (video, image), stitches
completed clips into long-form video, and uploads results to Cloudflare R2.

Full design: `docs/superpowers/specs/2026-09-15-ltx-video-flux-image-service-design.md`
Implementation plan: `docs/superpowers/plans/2026-09-15-ltx-video-flux-image-service.md`

## Repo layout

Four independently-versioned Python components, each with its own venv,
`requirements.txt`, and test suite — there is no shared root-level dependency
file or package:

- `backend/` — FastAPI proxy (auth, rate limit, moderation, job queue, DB, stitching)
- `worker-video/` — RunPod handler for LTX-2.5, deployed as its own container
- `worker-image/` — RunPod handler for FLUX.1-schnell, deployed as its own container
- `ops/` — standalone cron scripts (cost alert, RunPod scale-drift guard, revenue tracker)

Each worker/ops directory has a flat `handler.py` / `*.py` at its root (not a
package) — tests import it directly (`from handler import ...`), which is why
each directory carries its own `pytest.ini` with `pythonpath = .`.

## Commands

Each component needs its own venv before tests will run:

```bash
python -m venv backend/.venv && backend/.venv/Scripts/python.exe -m pip install -r backend/requirements.txt
python -m venv worker-video/.venv && worker-video/.venv/Scripts/python.exe -m pip install -r worker-video/requirements.txt
python -m venv worker-image/.venv && worker-image/.venv/Scripts/python.exe -m pip install -r worker-image/requirements.txt
python -m venv ops/.venv && ops/.venv/Scripts/python.exe -m pip install -r ops/requirements.txt
```

Run tests per component (from the component directory, or pass the path):

```bash
cd backend && python -m pytest -v                          # all backend tests
cd backend && python -m pytest tests/test_queue.py -v       # one file
cd backend && python -m pytest tests/test_queue.py::test_process_clip_job_marks_complete_on_success -v  # one test

cd worker-video && python -m pytest -v
cd worker-image && python -m pytest -v
cd ops && python -m pytest -v
```

Local stack (backend + Postgres + Redis + RQ worker + Caddy TLS):

```bash
cp backend/.env.example backend/.env   # fill in real credentials
docker compose up --build
```

`docker compose config` validates the compose file without needing real
secrets in `backend/.env` (unset vars just default to blank with a warning).

## Architecture

**Request flow:** client → `POST /generate-image` or `/generate-video` on the
backend → moderation check → `Job`/`VideoProject` rows created in Postgres →
job enqueued on Redis via RQ → RQ worker calls `RunpodClient.dispatch_*` →
RunPod Serverless container runs `handler.py` → result uploaded to R2 → status
polled via `GET /jobs/{id}` / `GET /projects/{id}`.

**Two trust boundaries, not one.** The backend enforces auth, rate limiting,
and moderation before a job is created — but the RunPod endpoints are a
second, independently-reachable trust boundary with their own Bearer keys
(`runpod_video_key`, `runpod_image_key` in `backend/app/config.py`). Both
worker handlers (`worker-video/handler.py`, `worker-image/handler.py`) carry
their own copy of the keyword blocklist (`BLOCKLIST` — mirrors
`backend/app/moderation.py`'s `DEFAULT_BLOCKLIST`) and reject matching
prompts before dispatch, since a caller with a leaked/independently-held
RunPod key bypasses the backend's moderation and rate-limit layer entirely.
Keep the two blocklists in sync; there is no shared package between the
backend and either worker to enforce this automatically.

**Public API surface is deliberately minimal and this is load-bearing, not
incidental.** Every Pydantic request model uses `extra="forbid"`, and neither
handler nor backend schema accepts a URL-shaped field (no `image_url`) —
this is the specific mechanism that closes off SSRF. When adding a field to
`GenerateImageRequest`/`GenerateVideoRequest` or either handler's
`validate_input`, preserve `extra="forbid"` and do not add caller-controlled
URLs. Resolution is chosen server-side from a closed `aspect_ratio` enum (`16:9` or `9:16`, optional, default `16:9` for callers that predate it) and never from a width/height field; fps and model variant are hardcoded in the handlers.

**Two RunPod workers, one shared shape but independent deploy lifecycles.**
`worker-video/handler.py` and `worker-image/handler.py` both: load their
model once into a module-global on cold start (guarded by a lock against
concurrent cold-start races on a warm RunPod worker), and validate input
strictly. `worker-video/handler.py` loads LTX-2.5's 22B distilled transformer
(bf16) plus a Gemma-3-12B text encoder, video/audio VAEs, and a spatial
upsampler from `/runpod-volume/ltx-2.5` (weights are not baked into the
image) via `ltx_pipelines.distilled.DistilledPipeline`, with
`OffloadMode.CPU` — `OffloadMode.NONE` was confirmed to OOM on a 32GB card.
Each worker has its own `Dockerfile` and is meant to connect directly to
RunPod for auto-build-on-push rather than share a base image.

**GPU sizing for worker-video is tight even with CPU offload.** The 22B
transformer + 12B text encoder alone are >60GB combined at bf16 before VAEs
and activations; a 32GB card OOMs even with offload enabled per the comment
in `_load_pipeline_impl`. Use an 80GB card (A100 80GB or H100 80GB) on the
RunPod endpoint for this worker — this is not encoded in the repo (RunPod
endpoint GPU type is configured outside this codebase) so verify current
endpoint config separately.

**Video length is bounded, not variant-routed.** `worker-video/handler.py`
validates `duration` against `FAST_MAX_DURATION` (20s) and generates directly
via the single `DistilledPipeline` — there is no per-duration model variant
selection in the current handler.

**Video generation fans out into many clip jobs.** `POST /generate-video`
splits `target_duration` into `ceil(target_duration / 10)` separate 10-second
clip `Job` rows tracked under one `VideoProject`, linked via `ProjectClip`
rows ordered by `sequence_index` (not insertion order). The stitching step
(`backend/app/stitcher.py`) reads clips back in that order and concatenates
them with ffmpeg, muxing in whatever `TTSProvider.generate()` returns —
currently only `NullTTSProvider`, which produces a matching-length silent
track. There is no real TTS integration; treat `tts.py`'s `TTSProvider`
interface as the extension point rather than modifying the stitching call
site when one is added.

**Secrets flow through one redaction path.** `backend/app/config.py`'s
`Settings.all_secrets()` returns every sensitive value, and
`backend/app/logging_conf.py`'s `configure_logging()` attaches a redacting
filter to every log *handler* (not the logger) — this is intentional:
`logging.Filter` objects attached to a `Logger` do not run when a record
propagates up from a child logger, only when that logger originates the
record, so handler-level attachment is required for redaction to actually
work across modules using `logging.getLogger(__name__)`. Any new secret field
added to `Settings` must also be added to `all_secrets()` or it will not be
redacted from logs.

**In-memory SQLite tests need `StaticPool`.** Route tests that spin up a
FastAPI `TestClient` against a `sqlite:///:memory:` engine must pass
`poolclass=StaticPool` and `connect_args={"check_same_thread": False}` —
`TestClient` runs the app in a different thread than the one that calls
`Base.metadata.create_all()`, and SQLite's default `SingletonThreadPool`
gives each thread its own separate (and thus empty) in-memory database
otherwise. See `backend/tests/test_routes_image.py` or
`test_routes_jobs.py` for the pattern. Tests using a file-based SQLite DSN
(`test_main.py`) don't need this.

**Ops scripts are standalone, not wired into the backend.** `ops/cost_alert.py`,
`ops/scale_guard.py`, `ops/revenue_tracker.py` each take an already-constructed
API client and webhook object and return a bool/dict — they're meant to be
invoked by external cron, not imported by the FastAPI app.
