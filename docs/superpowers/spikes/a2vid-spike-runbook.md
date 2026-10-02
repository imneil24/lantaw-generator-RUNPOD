# A2Vid spike runbook (Task 7, run by a person on a GPU pod)

Status: **NOT YET RUN.** Nothing in this document records an observed result. It
is a checklist of commands to run and a template for what to record. The
worker's ia2v mode (Task 8) was written before this spike ran, so its A2Vid
constructor and call kwargs are placeholders until this is done (see the
`UNVERIFIED` comment above `A2V_CALL_KWARGS` in `worker-video/handler.py`).

Question being answered: can `A2VidPipelineTwoStage` (LTX-2 repo) run on the
LTX-2.5 weights on `/runpod-volume/ltx-2.5`, with one avatar image and one
narration audio file, and does it fit next to the `DistilledPipeline` the t2v
path already uses?

**The spike script is throwaway.** `worker-video/spike/a2vid_spike.py` is
committed on branch `feat/avatar-ia2v-runpod` only so it can be run. Delete
`worker-video/spike/` before this branch is merged to master.

## Pod setup

- GPU: RTX PRO 6000 (96 GB, Blackwell). The earlier sizing note (80 GB A100/H100)
  was written before this card was chosen; the Blackwell checks below matter.
- Image: build from `worker-video/Dockerfile` on this branch (the Dockerfile is
  deliberately not pinned to an LTX-2 commit yet; Step 6 records the SHA to pin).
- Network volume mounted at `/runpod-volume` (contains `/runpod-volume/ltx-2.5`).
- Get `worker-video/spike/a2vid_spike.py` onto the pod (for example
  `runpodctl send` / `scp`, or paste it into `/worker/a2vid_spike.py`). The
  script does `sys.path.insert(0, "/worker")` and `import handler`, so the image's
  `/worker/handler.py` must be the one from this branch.
- Test inputs on the pod: a real avatar JPEG (`/tmp/avatar.jpg`) and a real
  narration MP3 of about 6 seconds (`/tmp/narration.mp3`).
- Run everything below with the LTX-2 venv: `/worker/ltx2-repo/.venv/bin/python`.

## Checklist

### 1. Print the real signatures (brief Step 1)

```bash
cd /worker/ltx2-repo
.venv/bin/python - <<'PY'
import inspect
from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage
print(inspect.signature(A2VidPipelineTwoStage.__init__))
print(inspect.signature(A2VidPipelineTwoStage.__call__))
import ltx_pipelines.utils.types as t
print([n for n in dir(t) if not n.startswith("_")])
print(inspect.getsource(t.ImageConditioningInput))
PY
git -C /worker/ltx2-repo rev-parse HEAD
ls -R /runpod-volume/ltx-2.5 | head -50
```

- [ ] Paste the full output under `## Signatures` in the findings doc.
- [ ] If `ImageConditioningInput` is not in `ltx_pipelines.utils.types`, record the
      module it lives in (Task 8's `_image_conditioning` import must match).
- [ ] If the script's calls (`distilled_lora=`, `loras=`, `offload_mode=`,
      `spatial_upsampler_path=`, `images=`, `audio_path=`) differ from the printed
      signature, edit `a2vid_spike.py` to match and record each difference.

### 2. (Script already written)

`worker-video/spike/a2vid_spike.py` is the brief's Step 2 shape unchanged.
Usage: `python a2vid_spike.py IMAGE AUDIO OUT_MP4 ASPECT SECONDS [DISTILLED_LORA]`.
Note the script does not read the `DISTILLED_LORA` argument yet; if Step 1 shows
the constructor needs a distilled LoRA, wire it in by hand.

### 3. Run it and iterate until it produces a clip (brief Step 3)

```bash
cd /worker
/worker/ltx2-repo/.venv/bin/python a2vid_spike.py /tmp/avatar.jpg /tmp/narration.mp3 /tmp/out-169.mp4 16:9 6
/worker/ltx2-repo/.venv/bin/python a2vid_spike.py /tmp/avatar.jpg /tmp/narration.mp3 /tmp/out-916.mp4 9:16 6
```

- [ ] On `TypeError` (missing or unknown kwarg) or a missing-LoRA error, fix the
      script from the traceback and record exactly what was needed.
- [ ] Record: does the 2.5 distilled transformer work as the stage-1 checkpoint?
- [ ] Record: is a `distilled_lora` file required, and where does it come from?
      **If one is required and it is not on the volume, STOP and tell the owner.**
- [ ] Record the wall time per 6 s clip (the script prints `load:` and `generate:`).
      The first run is slower (kernel JIT, see Blackwell checks); record a second
      run's time as the real figure and say which is which.
- [ ] Record the working dimensions for both aspect ratios.

### 4. Measure memory with both pipelines resident (brief Step 4)

One process: build the `DistilledPipeline` the way `handler.load_pipeline()` does,
run one t2v clip, record memory, then build and run the A2Vid pipeline and record
again. A sketch (adjust the A2Vid half to the working call from Step 3):

```bash
cd /worker
/worker/ltx2-repo/.venv/bin/python - <<'PY'
import sys, torch
sys.path.insert(0, "/worker")
import handler
import a2vid_spike

def rss_mb():
    for line in open("/proc/self/status"):
        if line.startswith(("VmRSS", "VmHWM")):
            print(line.strip())

req = handler.validate_input({"prompt": "a river at dawn", "duration": 6, "aspect_ratio": "16:9"})
handler._generate_video(req)  # loads the DistilledPipeline and runs one t2v clip
print("after t2v: peak GPU GiB", torch.cuda.max_memory_allocated() / 2**30)
rss_mb()
# second pipeline in the same process (this builds A2VidPipelineTwoStage and runs one clip)
a2vid_spike.main("/tmp/avatar.jpg", "/tmp/narration.mp3", "/tmp/both.mp4", "16:9", "6")
print("after a2v: peak GPU GiB", torch.cuda.max_memory_allocated() / 2**30)
rss_mb()
PY
```

Also watch `nvidia-smi` and `ps -o rss= -p <pid>` from a second shell while it runs.

- [ ] Record peak GPU memory and host RSS after each stage.
- [ ] Decision rule: **same worker** if both fit with headroom on the target card;
      otherwise **separate endpoint**.

### 5. Judge the output (brief Step 5)

Download both clips (`/tmp/out-169.mp4`, `/tmp/out-916.mp4`).

- [ ] (a) The mp4's audio is the input narration: play it, and compare durations
      with `ffprobe -v error -show_entries stream=codec_type,duration -of csv=p=0 FILE`.
- [ ] (b) Mouth movement plausibly follows the speech (watch it; say what you saw).
- [ ] (c) Frame size is `1920x1088` / `1088x1920`
      (`ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 FILE`).
- [ ] (d) Duration is within 0.5 s of the requested seconds (the backend's drift
      tolerance).

### 6. Pin the LTX-2 commit (brief Step 6)

- [ ] Record `git -C /worker/ltx2-repo rev-parse HEAD` from the pod under `## Pin`.
      This is the revision that was proven to work. Task 8 Step 5 (Dockerfile pin)
      was skipped because this SHA did not exist yet.

### 7. Write the decision and stop for review (brief Step 7)

- [ ] End the findings doc with `## Decision` stating exactly one of:
  - `proceed: same worker`
  - `proceed: separate endpoint` (then STOP: this changes backend Task 9; re-plan it)
  - `blocked: <reason>` (then STOP and tell the owner)
- [ ] Copy the findings doc into the repo as
      `docs/superpowers/spikes/2026-10-a2vid-findings.md` and commit **only the
      findings**, not the spike script:

```bash
git add docs/superpowers/spikes/2026-10-a2vid-findings.md
git commit -m "docs: record the A2Vid spike findings"
```

- [ ] Then replace the placeholders in `worker-video/handler.py` (`A2V_CALL_KWARGS`,
      `_a2v_pipeline_kwargs()`, and the `_image_conditioning` import if it moved)
      with the findings doc's `## Working call`, pin the Dockerfile
      (`ARG LTX2_COMMIT=<SHA from ## Pin>` plus
      `git clone ... && git -C ltx2-repo checkout ${LTX2_COMMIT}`), delete
      `worker-video/spike/`, and set `R2_READ_KEY` / `R2_READ_SECRET` on the endpoint
      before the live ia2v check.

## Blackwell checks

The RTX PRO 6000 is a Blackwell card. This repo's image has only been proven on
other architectures, so check these explicitly and record what you see:

- [ ] `nvidia-smi` shows the driver version; the driver must support CUDA 13.2
      (the image base is `nvidia/cuda:13.2.1-cudnn-runtime-ubuntu22.04`). Record the
      driver version and the CUDA version it reports.
- [ ] `/worker/ltx2-repo/.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"`
      runs and reports the card. Record the output.
- [ ] Watch the output of Steps 3 and 4 for `no kernel image is available` (a
      compiled CUDA kernel missing this architecture) and for any `natten` import
      or kernel error (the Dockerfile installs `--extra natten`; the A2Vid path may
      use natten kernels the t2v path never touched). Record exact error text, not
      a paraphrase.
- [ ] The first run is slower because triton/natten JIT-compile kernels for this
      architecture (and need `gcc` and `ldconfig` at container start, see the
      Dockerfile). Do not take the first run's wall time as the steady-state figure.
- [ ] If any of the above fails, that is a finding: write it under `## Decision` as
      `blocked: <exact error>`, do not work around it silently.

## Findings doc template

Copy to `docs/superpowers/spikes/2026-10-a2vid-findings.md`. Task 8 reads these
four headings by name; keep them exactly.

````markdown
# A2Vid spike findings

Run date, pod, GPU, driver and CUDA version, image build commit:

## Signatures

(Paste Step 1's output: `__init__` and `__call__` signatures, the names in
`ltx_pipelines.utils.types`, the `ImageConditioningInput` source and the module it
lives in, and the `ls -R /runpod-volume/ltx-2.5` listing.)

Differences between the brief's script and the real signatures:

-

## Working call

(Copy-pasteable facts Task 8 uses verbatim.)

- Import lines for `A2VidPipelineTwoStage` and `ImageConditioningInput`:
- Exact constructor call:
- Extra `__call__` kwargs beyond prompt/seed/height/width/frame_rate/num_frames/images/audio_path (empty if none):
- Distilled-LoRA file path (or "none needed"):
- 2.5 distilled transformer works as the stage-1 checkpoint: yes/no
- Working dimensions, 16:9:
- Working dimensions, 9:16:
- Wall time per 6 s clip (first run / steady state):
- Peak GPU memory with both pipelines loaded:
- Host RSS with both pipelines loaded:
- Blackwell observations (kernel image / natten errors, JIT slowdown):

## Output judgement

- (a) Audio in the mp4 is the input narration:
- (b) Mouth movement follows speech:
- (c) Frame sizes:
- (d) Duration vs requested seconds:

## Pin

LTX-2 commit SHA (`git rev-parse HEAD` in the pod):

## Decision

(exactly one of: `proceed: same worker` / `proceed: separate endpoint` / `blocked: <reason>`)
````
