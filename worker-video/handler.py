import logging
import math
import os
import re
import tempfile
import threading
import uuid
from dataclasses import dataclass, replace

import boto3
import torch

from ltx_core.model.video_vae import get_video_chunks_number
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.helpers import snap_frames_to_grid
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import OffloadMode

# Two-stage pipelines (DistilledPipeline) require both dimensions divisible by
# 64 (ltx_pipelines.utils.helpers.assert_resolution). 1088 is the standard
# "1080p-safe" height used across video codecs for exactly this reason
# (64 * 17 = 1088); true 1080 is not a multiple of 64. 1920 = 64 * 30.
# The caller picks only an orientation (never a width/height): resolution is
# a server-side decision, so a leaked RunPod key cannot ask for a size that
# exhausts GPU memory.
ASPECT_RATIO_DIMENSIONS = {
    "16:9": (1920, 1088),
    "9:16": (1088, 1920),
}
# Callers that predate aspect_ratio (backend/app/runpod_client.py) send only
# prompt+duration and were always served landscape; keeping that behaviour is
# backward compatibility, not a stand-in for a missing value.
DEFAULT_ASPECT_RATIO = "16:9"
FPS = 24
# The VAE's temporal grid: a valid frame count satisfies (frames - 1) % FRAME_GRID_STEP == 0.
FRAME_GRID_STEP = 8
FAST_MAX_DURATION = 20

MODEL_ROOT = "/runpod-volume/ltx-2.5"
TRANSFORMER_PATH = f"{MODEL_ROOT}/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
TEXT_ENCODER_PATH = f"{MODEL_ROOT}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
VIDEO_VAE_PATH = f"{MODEL_ROOT}/vae/ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE_PATH = f"{MODEL_ROOT}/vae/ltx-2.5-audio-vae-bf16.safetensors"
SPATIAL_UPSAMPLER_PATH = f"{MODEL_ROOT}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

logger = logging.getLogger("worker-video")

# Optional avatar (A2V) tuning, all read at call time. UNSET means the behaviour
# is exactly what it was before these existed, so deploying this changes nothing
# until a variable is set on the RunPod endpoint. Set them to A/B test the
# avatar's unnatural expression and weak lip sync without a code change:
#   A2V_TRANSFORMER_PATH        base transformer for the avatar pipeline (default
#                               TRANSFORMER_PATH, the DISTILLED weights)
#   A2V_DISTILLED_LORA_PATH     distilled LoRA for stage 2 (default none)
#   A2V_DISTILLED_LORA_STRENGTH its strength (default 1.0, the library's default)
#   A2V_NUM_INFERENCE_STEPS     stage 1 steps (default from detect_params)
#   A2V_CFG_SCALE               video CFG (text guidance) scale
#   A2V_STG_SCALE               video STG scale (0 disables)
#   A2V_RESCALE_SCALE           video rescale scale (limits over-saturation)
#   A2V_MODALITY_SCALE          audio-to-video modality scale (lip sync; 1.0 off)
# The meaning of the four guidance values is from the LTX-2 library's
# docs/multimodal-guidance.md (Lightricks/LTX-2 main). Good VALUES for this
# model are not known: they need short test clips.
_A2V_FLOAT_VARS = {
    "A2V_CFG_SCALE": "cfg_scale",
    "A2V_STG_SCALE": "stg_scale",
    "A2V_RESCALE_SCALE": "rescale_scale",
    "A2V_MODALITY_SCALE": "modality_scale",
}


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite number >= 0, got {raw!r}")
    return value


def _a2v_transformer_path() -> str:
    return os.environ.get("A2V_TRANSFORMER_PATH", "").strip() or TRANSFORMER_PATH


def _a2v_call_kwargs() -> dict:
    # Extra required args of A2VidPipelineTwoStage.__call__ beyond
    # prompt/seed/height/width/frame_rate/num_frames/images/audio_path. The
    # real call failed live with a TypeError naming negative_prompt,
    # num_inference_steps and video_guider_params as missing.
    #
    # Source: ltx_pipelines.utils.constants (Lightricks/LTX-2 main). detect_params
    # reads the checkpoint's declared model_version and returns the params of the
    # newest generation at or below it, so these values follow the checkpoint
    # and a model upgrade changes them automatically.
    #
    # UNVERIFIED: the call signature is from library source, but this has not yet
    # run to completion on real hardware; output quality and speed are unverified.
    # Imported lazily (like _image_conditioning) so tests run without the library.
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, detect_params

    params = detect_params(_a2v_transformer_path())
    steps = params.num_inference_steps
    guider = params.video_guider_params

    steps_override = _env_float("A2V_NUM_INFERENCE_STEPS")
    if steps_override is not None:
        if steps_override < 1 or steps_override != int(steps_override):
            raise ValueError(f"A2V_NUM_INFERENCE_STEPS must be a whole number >= 1, got {steps_override}")
        steps = int(steps_override)
    guider_overrides = {
        field: value for name, field in _A2V_FLOAT_VARS.items() if (value := _env_float(name)) is not None
    }
    if guider_overrides:
        guider = replace(guider, **guider_overrides)

    # Logged on every call so the values a clip was really made with can be read
    # from the endpoint's logs, not inferred from code.
    logger.info(
        "A2V settings: transformer=%s steps=%s guider=%s",
        _a2v_transformer_path(), steps, guider,
    )
    return {
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "num_inference_steps": steps,
        "video_guider_params": guider,
    }


def _a2v_pipeline_kwargs() -> dict:
    # distilled_lora: the avatar pipeline's stage 2 applies a distilled LoRA on
    # top of the base model (library docs/pipelines.md section 7). It was []
    # (none), and still is unless A2V_DISTILLED_LORA_PATH is set. Whether a LoRA
    # is needed with these (distilled) base weights is not known; the spike has
    # not run.
    path = os.environ.get("A2V_DISTILLED_LORA_PATH", "").strip()
    if not path:
        return {"distilled_lora": []}
    strength = _env_float("A2V_DISTILLED_LORA_STRENGTH")
    if not os.path.isfile(path):
        raise ValueError(f"A2V_DISTILLED_LORA_PATH does not exist: {path!r}")
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps

    return {
        "distilled_lora": [
            LoraPathStrengthAndSDOps(path, 1.0 if strength is None else strength, LTXV_LORA_COMFY_RENAMING_MAP)
        ]
    }


# Mirrors backend/app/moderation.py's DEFAULT_BLOCKLIST. The RunPod endpoint
# is reachable directly with its own Bearer key, independent of the backend
# proxy — this is a second, independent trust boundary, so moderation must
# be enforced here too, not only in the backend before enqueueing.
BLOCKLIST = [
    "child sexual", "csam", "bomb making", "how to build a bomb",
    "bioweapon", "chemical weapon synthesis",
]


def _is_blocked(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(term in lowered for term in BLOCKLIST)


_PIPELINE = None
_PIPELINE_LOCK = threading.Lock()

_R2_CLIENT = None


def _get_r2_client():
    # Built lazily (not at import time) so the moderation/validation tests
    # can run without R2 credentials set.
    global _R2_CLIENT
    if _R2_CLIENT is None:
        _R2_CLIENT = boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT"],
            aws_access_key_id=os.environ["R2_WRITE_KEY"],
            aws_secret_access_key=os.environ["R2_WRITE_SECRET"],
        )
    return _R2_CLIENT


def _upload_to_r2(key: str, data: bytes, content_type: str) -> None:
    _get_r2_client().put_object(
        Bucket=os.environ["R2_BUCKET"], Key=key, Body=data, ContentType=content_type,
    )


_R2_READ_CLIENT = None


def _get_r2_read_client():
    # The write credential is not assumed to be able to read; the backend's
    # own R2_READ_KEY/R2_READ_SECRET pair works for this bucket.
    global _R2_READ_CLIENT
    if _R2_READ_CLIENT is None:
        _R2_READ_CLIENT = boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT"],
            aws_access_key_id=os.environ["R2_READ_KEY"],
            aws_secret_access_key=os.environ["R2_READ_SECRET"],
        )
    return _R2_READ_CLIENT


def _download_from_r2(key: str, dest_path: str) -> None:
    _get_r2_read_client().download_file(os.environ["R2_BUCKET"], key, dest_path)


def _load_pipeline_impl() -> DistilledPipeline:
    model_paths = ModelPaths.from_split(
        transformer_path=TRANSFORMER_PATH,
        text_encoder_path=TEXT_ENCODER_PATH,
        video_vae_path=VIDEO_VAE_PATH,
        audio_vae_path=AUDIO_VAE_PATH,
    )
    return DistilledPipeline(
        model_paths=model_paths,
        spatial_upsampler_path=SPATIAL_UPSAMPLER_PATH,
        loras=[],
        # Default OffloadMode.NONE keeps every component (transformer, text
        # encoder, VAEs) resident on GPU at once — confirmed via a real OOM on
        # a 32GB card (~30GB allocated with zero headroom for computation).
        # CPU offloading trades some speed for the ~5GB VRAM / ~36GB RAM
        # footprint documented in ltx_pipelines.utils.types.OffloadMode.
        offload_mode=OffloadMode.CPU,
    )


def load_pipeline() -> DistilledPipeline:
    global _PIPELINE
    # RunPod serverless can dispatch concurrent requests to one warm worker
    # process. An unlocked check-then-act here would let two invocations
    # both see _PIPELINE is None and both run the GPU-weight-loading init
    # concurrently — wasted memory at best, corrupted shared state at worst.
    if _PIPELINE is None:
        with _PIPELINE_LOCK:
            if _PIPELINE is None:
                _PIPELINE = _load_pipeline_impl()
    return _PIPELINE


_A2V_PIPELINE = None
_A2V_PIPELINE_LOCK = threading.Lock()


def _load_a2v_pipeline_impl():
    # Imported here (not at module top) so the worker's t2v path and the unit
    # tests never need the A2Vid module; a t2v-only worker never loads it.
    from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage

    model_paths = ModelPaths.from_split(
        transformer_path=_a2v_transformer_path(),
        text_encoder_path=TEXT_ENCODER_PATH,
        video_vae_path=VIDEO_VAE_PATH,
        audio_vae_path=AUDIO_VAE_PATH,
    )
    return A2VidPipelineTwoStage(
        model_paths=model_paths,
        spatial_upsampler_path=SPATIAL_UPSAMPLER_PATH,
        loras=[],
        offload_mode=OffloadMode.CPU,
        **_a2v_pipeline_kwargs(),
    )


def load_a2v_pipeline():
    # Same double-checked lock as load_pipeline: concurrent requests on one
    # warm worker must not both run the weight-loading init.
    global _A2V_PIPELINE
    if _A2V_PIPELINE is None:
        with _A2V_PIPELINE_LOCK:
            if _A2V_PIPELINE is None:
                _A2V_PIPELINE = _load_a2v_pipeline_impl()
    return _A2V_PIPELINE


def _image_conditioning(path: str):
    # UNVERIFIED: the import path and the (path, frame_idx, strength) shape are
    # from the LTX-2 GitHub main source read while planning, not from a run;
    # confirm against the spike findings ("Signatures").
    from ltx_pipelines.utils.types import ImageConditioningInput

    return ImageConditioningInput(path=path, frame_idx=0, strength=1.0)


# Keys the backend stages for an ia2v job: runpod-inputs/<job uuid>/image.<ext>
# and .../audio.<ext>. job ids are UUIDs (every top-level folder of the live
# bucket that matched a jobs.id was UUID-named). Matched with fullmatch so a
# trailing newline or extra path segment cannot slip through; no URL, "..", or
# other prefix can ever match, which is what keeps the worker from fetching
# anything the caller names.
_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
IMAGE_KEY_PATTERN = re.compile(rf"runpod-inputs/({_UUID})/image\.(?:jpg|png|webp)")
AUDIO_KEY_PATTERN = re.compile(rf"runpod-inputs/({_UUID})/audio\.(?:mp3|wav)")


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    duration: float
    width: int
    height: int
    image_key: str | None = None
    audio_key: str | None = None

    @property
    def mode(self) -> str:
        return "ia2v" if self.image_key is not None else "t2v"


def _validated_key(job_input: dict, field: str, pattern: re.Pattern) -> re.Match:
    value = job_input[field]
    match = pattern.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError(f"{field} is not a valid staged input key")
    return match


def validate_input(job_input: dict) -> GenerationRequest:
    allowed_keys = {"prompt", "duration", "aspect_ratio", "image_key", "audio_key"}
    if set(job_input.keys()) - allowed_keys:
        raise ValueError(f"unexpected fields: {set(job_input.keys()) - allowed_keys}")
    if "prompt" not in job_input or not isinstance(job_input["prompt"], str) or not job_input["prompt"].strip():
        raise ValueError("prompt is required and must be a non-empty string")
    if "duration" not in job_input:
        raise ValueError("duration is required")
    duration = float(job_input["duration"])
    if not (0 < duration <= FAST_MAX_DURATION):
        raise ValueError(f"duration must be between 0 and {FAST_MAX_DURATION}")
    aspect_ratio = job_input.get("aspect_ratio", DEFAULT_ASPECT_RATIO)
    if not isinstance(aspect_ratio, str) or aspect_ratio not in ASPECT_RATIO_DIMENSIONS:
        raise ValueError(f"aspect_ratio must be one of {sorted(ASPECT_RATIO_DIMENSIONS)}")
    width, height = ASPECT_RATIO_DIMENSIONS[aspect_ratio]

    image_key = audio_key = None
    if ("image_key" in job_input) != ("audio_key" in job_input):
        raise ValueError("image_key and audio_key must be provided together")
    if "image_key" in job_input:
        image_match = _validated_key(job_input, "image_key", IMAGE_KEY_PATTERN)
        audio_match = _validated_key(job_input, "audio_key", AUDIO_KEY_PATTERN)
        if image_match.group(1) != audio_match.group(1):
            raise ValueError("image_key and audio_key must be in the same job folder")
        image_key, audio_key = job_input["image_key"], job_input["audio_key"]

    return GenerationRequest(
        prompt=job_input["prompt"],
        duration=duration,
        width=width,
        height=height,
        image_key=image_key,
        audio_key=audio_key,
    )


def _frames_for_duration(duration: float) -> int:
    """Smallest valid frame count (8k+1) that covers `duration` at FPS.

    snap_frames_to_grid FLOORS, so a whole-second duration (a multiple of 8
    frames) would come out 7 frames (0.29s) short; adding FRAME_GRID_STEP - 1
    first makes the floor land on the first grid value >= the target.
    """
    # round() guards float noise like 145.00000000000003
    target = math.ceil(round(duration * FPS, 6))
    return snap_frames_to_grid(target + FRAME_GRID_STEP - 1)


def _generate_video(request: GenerationRequest) -> bytes:
    pipeline = load_pipeline()
    # The VAE's causal temporal grid requires (frames - 1) % FRAME_GRID_STEP == 0;
    # _frames_for_duration rounds UP to the first valid value covering the duration.
    num_frames = _frames_for_duration(request.duration)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        output_path = tmp_file.name

    try:
        # inference_mode() still crashed here ("Inference tensors cannot be
        # saved for backward") deep inside the VAE decoder's streamed
        # transformer blocks (rms_norm), even with encode_video wrapped in
        # the same context — the pipeline's model is built once at cold
        # start (load_pipeline, module-global, outside any inference_mode
        # scope) and reused across calls, and its streamed weight loading
        # mutates buffers on that long-lived model during the forward pass.
        # inference_mode's special "inference tensor" tagging does not mix
        # safely across that boundary. torch's own error message names the
        # fix: no_grad() disables grad tracking the same way for our
        # purposes (no backward pass ever runs here) without creating
        # inference tensors, so it doesn't trip this check.
        with torch.no_grad():
            result = pipeline(
                prompt=request.prompt,
                seed=int.from_bytes(os.urandom(4), "big"),
                height=request.height,
                width=request.width,
                frame_rate=FPS,
                images=[],
                num_frames=num_frames,
            )
            encode_video(
                video=result.video,
                fps=FPS,
                audio=result.audio,
                output_path=output_path,
                video_chunks_number=get_video_chunks_number(result.num_frames, result.tiling_config),
            )
        with open(output_path, "rb") as f:
            return f.read()
    finally:
        os.remove(output_path)


def _generate_ia2v_video(request: GenerationRequest) -> bytes:
    pipeline = load_a2v_pipeline()
    # Same helper as t2v: snap_frames_to_grid floors, which would leave a
    # whole-second clip 0.29s short (see _frames_for_duration).
    num_frames = _frames_for_duration(request.duration)

    with tempfile.TemporaryDirectory() as workdir:
        image_path = os.path.join(workdir, "image" + os.path.splitext(request.image_key)[1])
        audio_path = os.path.join(workdir, "audio" + os.path.splitext(request.audio_key)[1])
        output_path = os.path.join(workdir, "out.mp4")
        _download_from_r2(request.image_key, image_path)
        _download_from_r2(request.audio_key, audio_path)

        # no_grad, not inference_mode: see the comment in _generate_video.
        with torch.no_grad():
            result = pipeline(
                prompt=request.prompt,
                seed=int.from_bytes(os.urandom(4), "big"),
                height=request.height,
                width=request.width,
                frame_rate=FPS,
                num_frames=num_frames,
                images=[_image_conditioning(image_path)],
                audio_path=audio_path,
                **_a2v_call_kwargs(),
            )
            # result.audio is the INPUT audio passed through (docs/pipelines.md
            # of ltx-pipelines), which is what a lip-synced clip must carry.
            encode_video(
                video=result.video,
                fps=FPS,
                audio=result.audio,
                output_path=output_path,
                video_chunks_number=get_video_chunks_number(result.num_frames, result.tiling_config),
            )
        with open(output_path, "rb") as f:
            return f.read()


def handler(job: dict) -> dict:
    try:
        request = validate_input(job.get("input", {}))
    except ValueError as e:
        return {"error": str(e)}

    if _is_blocked(request.prompt):
        return {"error": "prompt rejected by moderation"}

    if request.mode == "ia2v":
        video_bytes = _generate_ia2v_video(request)
    else:
        video_bytes = _generate_video(request)
    key =f"clips/{uuid.uuid4().hex}.mp4"
    # Uploaded directly from the worker rather than returned as bytes_b64:
    # RunPod's own /job-done callback rejects a full HD video base64-encoded
    # into the job result with a 400 (exceeds RunPod's sync result size
    # limit), so the backend never sees a completed job at all.
    #
    # These prints exist because RunPod's own container has been observed
    # dying/restarting mid-job with no exception ever logged — "Video saved"
    # followed immediately by RunPod's own "Failed to return job results |
    # 400" with nothing in between. Without a log line bracketing the
    # upload call, there is no way to tell after the fact whether
    # _upload_to_r2 ever ran or whether it completed before the container
    # died.
    print(f"uploading to R2: key={key}")
    _upload_to_r2(key, video_bytes, "video/mp4")
    print(f"R2 upload complete: key={key}")
    return {"key": key}
