import glob
import logging
import math
import os
import re
import tempfile
import threading
import time
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
# Optional smaller size for b-roll (t2v) only, chosen by the T2V_SIZE endpoint
# setting ("full" = ASPECT_RATIO_DIMENSIONS, the default and today's behaviour;
# "small" = this table, about 0.9 megapixels against 2.1). The second (refine)
# pass dominates a b-roll clip's time and its cost grows faster than the pixel
# count, so a smaller clip is much cheaper to make; the backend scales it back up
# to the full size before it is joined to avatar clips. Both sides are still
# multiples of 64 (1280 = 64 * 20, 704 = 64 * 11) for assert_resolution. Avatar
# (ia2v) clips never use this table, so lip sync is untouched. Like the other
# size decisions it is server-side only: a caller still picks an orientation,
# never a width or height.
T2V_SMALL_ASPECT_RATIO_DIMENSIONS = {
    "16:9": (1280, 704),
    "9:16": (704, 1280),
}
_T2V_SIZE_CHOICES = ("full", "small")
# Optional LARGER size for avatar (ia2v) clips only, chosen by the A2V_SIZE
# endpoint setting ("full" = ASPECT_RATIO_DIMENSIONS, the default and today's
# behaviour). A TEST setting: A2VidPipelineTwoStage renders stage 1 at half the
# target (a2vid_two_stage.py: stage_1_width = width // 2), so the 1088x1920
# target gives a 544x960 (0.52 MP) first pass, and a third-party guide claims
# lips "get glued together" below about 0.9 MP. "large" targets about 3.5 MP so
# stage 1 is about 0.88 MP (704x1248). Both sides are multiples of 64 (1408 = 64 *
# 22, 2496 = 64 * 39) for assert_resolution. A clip of this size is NOT the
# size the backend joins avatar clips at (concat_videos stream-copies), so it is
# only for single-clip experiments. Server-side only, like every size decision.
A2V_LARGE_ASPECT_RATIO_DIMENSIONS = {
    "16:9": (2496, 1408),
    "9:16": (1408, 2496),
}
_A2V_SIZE_CHOICES = ("full", "large")
FPS = 24
# The VAE's temporal grid: a valid frame count satisfies (frames - 1) % FRAME_GRID_STEP == 0.
FRAME_GRID_STEP = 8
FAST_MAX_DURATION = 20

# Weights come from RunPod's "Cached model" feature (Lightricks/LTX-2.5), which
# mounts the HF hub cache under /runpod-volume/huggingface-cache. The snapshot
# folder name is a commit hash unknown at build time, so it is globbed at
# runtime. Files inside are symlinks into blobs/ (normal for the HF cache); do
# not realpath them. The HF token lives in the endpoint settings, not here.
SNAPSHOT_GLOB = "/runpod-volume/huggingface-cache/hub/models--Lightricks--LTX-2.5/snapshots/*"

_TRANSFORMER_REL = "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
_TEXT_ENCODER_REL = "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
_VIDEO_VAE_REL = "vae/ltx-2.5-video-vae-bf16.safetensors"
_AUDIO_VAE_REL = "vae/ltx-2.5-audio-vae-bf16.safetensors"
# Optional faster decoder ("Conv VAE ... faster, lighter" in the model card; the
# default video VAE above is the DiffVAE, "higher quality, heavier"). It is NOT in
# _REQUIRED_MODEL_FILES: only a worker that opts in with T2V_VIDEO_VAE / A2V_VIDEO_VAE
# needs it, and that is checked when the pipeline loads.
_CONV_VIDEO_VAE_REL = "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
_SPATIAL_UPSAMPLER_REL = "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
_REQUIRED_MODEL_FILES = (
    _TRANSFORMER_REL, _TEXT_ENCODER_REL, _VIDEO_VAE_REL, _AUDIO_VAE_REL, _SPATIAL_UPSAMPLER_REL,
)

# Set by init_model_paths() at worker startup, not at import: importing this
# module (unit tests, tooling) must not block waiting for a GPU worker's cache.
MODEL_ROOT = None
TRANSFORMER_PATH = None
TEXT_ENCODER_PATH = None
VIDEO_VAE_PATH = None
AUDIO_VAE_PATH = None
SPATIAL_UPSAMPLER_PATH = None
_MODEL_PATHS_LOCK = threading.Lock()


def _missing_model_files(root: str) -> list[str]:
    # isfile follows symlinks, so a dangling blob link counts as missing.
    return [rel for rel in _REQUIRED_MODEL_FILES if not os.path.isfile(f"{root}/{rel}")]


def resolve_model_root(timeout: float = 600, poll_interval: float = 5) -> str:
    """Return the cached LTX-2.5 snapshot folder that holds every required file.

    Polls because the cache can mount before it is fully populated. Waiting for
    the files (not just the folder) means a half-populated snapshot is retried
    instead of failing later, deep inside the weight loader.
    """
    deadline = time.monotonic() + timeout
    incomplete: dict[str, list[str]] = {}
    while True:
        incomplete = {}
        for snapshot in sorted(glob.glob(SNAPSHOT_GLOB)):
            missing = _missing_model_files(snapshot)
            if not missing:
                return snapshot
            incomplete[snapshot] = missing
        if time.monotonic() >= deadline:
            break
        time.sleep(poll_interval)
    if incomplete:
        raise RuntimeError(
            f"LTX-2.5 cached model is incomplete (missing files per snapshot: {incomplete}); "
            "check the Cached model setting and HF token"
        )
    raise RuntimeError(
        f"LTX-2.5 cached model not found (no folder matches {SNAPSHOT_GLOB}); "
        "check Cached model setting and HF token"
    )


def init_model_paths() -> str:
    global MODEL_ROOT, TRANSFORMER_PATH, TEXT_ENCODER_PATH, VIDEO_VAE_PATH
    global AUDIO_VAE_PATH, SPATIAL_UPSAMPLER_PATH
    root = resolve_model_root()
    MODEL_ROOT = root
    TRANSFORMER_PATH = f"{root}/{_TRANSFORMER_REL}"
    TEXT_ENCODER_PATH = f"{root}/{_TEXT_ENCODER_REL}"
    VIDEO_VAE_PATH = f"{root}/{_VIDEO_VAE_REL}"
    AUDIO_VAE_PATH = f"{root}/{_AUDIO_VAE_REL}"
    SPATIAL_UPSAMPLER_PATH = f"{root}/{_SPATIAL_UPSAMPLER_REL}"
    print(f"MODEL_ROOT={MODEL_ROOT}")
    return root


def _ensure_model_paths() -> None:
    # Idempotent; the pipeline loaders call it so they are safe even if
    # startup() was skipped.
    if MODEL_ROOT is None:
        with _MODEL_PATHS_LOCK:
            if MODEL_ROOT is None:
                init_model_paths()


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
#   A2V_SIZE                    "full" (default) or "large": the avatar clip's target
#                               size, read per call (see A2V_LARGE_ASPECT_RATIO_DIMENSIONS)
#   A2V_STAGE1_SIGMAS           "distilled": stage 1 uses the library's fixed 8-step
#                               distilled schedule and A2V_NUM_INFERENCE_STEPS is
#                               ignored (unset = the scheduler's sigmas, as before)
#   A2V_IMAGE_STRENGTH          how firmly frame 0 is pinned to the photo, 0 to 1
#                               (default 1.0; read per call, no restart needed)
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


def _a2v_image_strength() -> float:
    # How firmly frame 0 is pinned to the avatar photo (0 to 1). It was a fixed
    # 1.0 (the most rigid setting); community avatar pipelines default to 0.7 and
    # describe lower values as "more motion". UNSET keeps 1.0, so deploying this
    # changes nothing until A2V_IMAGE_STRENGTH is set. Good values are untested.
    value = _env_float("A2V_IMAGE_STRENGTH")
    if value is None:
        return 1.0
    if value > 1.0:
        raise ValueError(f"A2V_IMAGE_STRENGTH must be between 0 and 1, got {value}")
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

    stage_1_sigmas = _a2v_stage_1_sigmas()

    # Logged on every call so the values a clip was really made with can be read
    # from the endpoint's logs, not inferred from code.
    logger.info(
        "A2V settings: transformer=%s steps=%s guider=%s stage1_sigmas=%s",
        _a2v_transformer_path(), steps, guider,
        "distilled (steps ignored)" if stage_1_sigmas is not None else "scheduler",
    )
    kwargs = {
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "num_inference_steps": steps,
        "video_guider_params": guider,
    }
    if stage_1_sigmas is not None:
        kwargs["stage_1_sigmas"] = stage_1_sigmas
    return kwargs


def _a2v_stage_1_sigmas():
    # A2V_STAGE1_SIGMAS=distilled runs stage 1 on the library's fixed 8-step
    # distilled schedule (DISTILLED_SIGMA_VALUES, the same list both community
    # ComfyUI avatar graphs use) instead of the scheduler's sigmas for
    # A2V_NUM_INFERENCE_STEPS. A2VidPipelineTwoStage.__call__ takes
    # `stage_1_sigmas` and, when given, bypasses num_inference_steps (read from
    # Lightricks/LTX-2 main, a2vid_two_stage.py). UNSET keeps the old behaviour,
    # so deploying this changes nothing until the variable is set. Meant to be
    # combined with A2V_CFG_SCALE=1, A2V_STG_SCALE=0, A2V_MODALITY_SCALE=1;
    # whether that gives a calm, talking face is UNTESTED.
    value = os.environ.get("A2V_STAGE1_SIGMAS", "").strip()
    if not value:
        return None
    if value != "distilled":
        raise ValueError(f"A2V_STAGE1_SIGMAS must be 'distilled' or unset, got {value!r}")
    try:
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
    except ImportError as exc:
        # No hard-coded copy of the list: a library without it should fail the
        # clip, not run a schedule nobody checked.
        raise RuntimeError(
            "A2V_STAGE1_SIGMAS=distilled needs ltx_pipelines.utils.constants.DISTILLED_SIGMA_VALUES, "
            "which this library version does not have"
        ) from exc
    return torch.tensor(list(DISTILLED_SIGMA_VALUES), dtype=torch.float32)


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


# Optional speed options, read when a pipeline is LOADED (not per request). UNSET
# means exactly the behaviour these pipelines had before, so deploying this changes
# nothing until a variable is set on the RunPod endpoint. Scoped separately for
# the b-roll pipeline (T2V_*) and the avatar pipeline (A2V_*), so b-roll can be
# tried first without touching lip sync:
#   <T2V|A2V>_VIDEO_VAE      "diff" (default, the heavier higher-quality decoder) or
#                            "conv" (the model card's "faster, lighter" decoder).
#                            Needs vae/ltx-2.5-video-vae-conv-bf16.safetensors in the
#                            cached model; a missing file fails the load by name.
#   <T2V|A2V>_QUANTIZATION   unset/"none" (default) or "fp8-cast": downcast the bf16
#                            transformer to FP8 while loading (model card "Low-VRAM
#                            tips"; ltx_pipelines.utils.quantization_factory).
#   <T2V|A2V>_OFFLOAD        "cpu" (default, today's behaviour), "none" (all weights on
#                            the GPU: fastest if it fits, a real OOM was seen on a
#                            32GB card with bf16) or "disk".
# Whether any of these is faster on this GPU, and whether FP8 keeps lip sync intact,
# is NOT known: they need test clips.
# T2V_SIZE ("full" default, or "small") is read per request, not at load: it sets
# the b-roll render size (see T2V_SMALL_ASPECT_RATIO_DIMENSIONS), has no A2V_
# counterpart, and is UNVERIFIED on real hardware (speed and picture quality).
_VIDEO_VAE_CHOICES = ("diff", "conv")
_QUANTIZATION_CHOICES = ("none", "fp8-cast")


def _env_choice(name: str, allowed: tuple[str, ...], default: str) -> str:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}, got {raw!r}")
    return raw


def _video_vae_path_for(prefix: str) -> str:
    """The video VAE file for the pipeline scoped by `prefix` ("T2V" or "A2V")."""
    name = f"{prefix}_VIDEO_VAE"
    if _env_choice(name, _VIDEO_VAE_CHOICES, "diff") == "diff":
        return VIDEO_VAE_PATH
    path = f"{MODEL_ROOT}/{_CONV_VIDEO_VAE_REL}"
    if not os.path.isfile(path):
        raise RuntimeError(
            f"{name}=conv needs {_CONV_VIDEO_VAE_REL}, which is not in the cached model "
            f"({MODEL_ROOT}); add it to the cache or unset {name}"
        )
    return path


def _quantization_kwargs_for(prefix: str, transformer_path: str) -> dict:
    """{} (the pipeline's own default) unless <prefix>_QUANTIZATION asks for FP8."""
    name = f"{prefix}_QUANTIZATION"
    if _env_choice(name, _QUANTIZATION_CHOICES, "none") == "none":
        return {}
    try:
        from ltx_pipelines.utils.quantization_factory import QuantizationKind
    except ImportError as exc:
        raise RuntimeError(
            f"{name}=fp8-cast needs ltx_pipelines.utils.quantization_factory, which the "
            f"installed ltx_pipelines does not have; unset {name}"
        ) from exc
    return {"quantization": QuantizationKind.FP8_CAST.to_policy(transformer_path)}


def _offload_mode_for(prefix: str) -> OffloadMode:
    return OffloadMode(_env_choice(f"{prefix}_OFFLOAD", ("cpu", "none", "disk"), "cpu"))


def _describe_speed_options(prefix: str) -> str:
    return (
        f"{prefix} pipeline options: "
        f"video_vae={os.environ.get(f'{prefix}_VIDEO_VAE', '').strip() or 'diff'} "
        f"quantization={os.environ.get(f'{prefix}_QUANTIZATION', '').strip() or 'none'} "
        f"offload={os.environ.get(f'{prefix}_OFFLOAD', '').strip() or 'cpu'}"
    )


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
    _ensure_model_paths()
    model_paths = ModelPaths.from_split(
        transformer_path=TRANSFORMER_PATH,
        text_encoder_path=TEXT_ENCODER_PATH,
        video_vae_path=_video_vae_path_for("T2V"),
        audio_vae_path=AUDIO_VAE_PATH,
    )
    logger.info(_describe_speed_options("T2V"))
    return DistilledPipeline(
        model_paths=model_paths,
        spatial_upsampler_path=SPATIAL_UPSAMPLER_PATH,
        loras=[],
        **_quantization_kwargs_for("T2V", TRANSFORMER_PATH),
        # Default OffloadMode.NONE keeps every component (transformer, text
        # encoder, VAEs) resident on GPU at once — confirmed via a real OOM on
        # a 32GB card (~30GB allocated with zero headroom for computation).
        # CPU offloading trades some speed for the ~5GB VRAM / ~36GB RAM
        # footprint documented in ltx_pipelines.utils.types.OffloadMode.
        # T2V_OFFLOAD can change this (default "cpu" = unchanged).
        offload_mode=_offload_mode_for("T2V"),
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


def startup() -> None:
    # Called once by the container CMD before runpod.serverless.start, so the
    # cache wait and the slow weight load happen at cold start rather than on
    # the first request. The A2Vid pipeline stays lazy (see
    # _load_a2v_pipeline_impl): a t2v-only worker never loads it.
    init_model_paths()
    load_pipeline()


_A2V_PIPELINE = None
_A2V_PIPELINE_LOCK = threading.Lock()


def _load_a2v_pipeline_impl():
    # Imported here (not at module top) so the worker's t2v path and the unit
    # tests never need the A2Vid module; a t2v-only worker never loads it.
    from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage

    _ensure_model_paths()
    model_paths = ModelPaths.from_split(
        transformer_path=_a2v_transformer_path(),
        text_encoder_path=TEXT_ENCODER_PATH,
        video_vae_path=_video_vae_path_for("A2V"),
        audio_vae_path=AUDIO_VAE_PATH,
    )
    logger.info(_describe_speed_options("A2V"))
    return A2VidPipelineTwoStage(
        model_paths=model_paths,
        spatial_upsampler_path=SPATIAL_UPSAMPLER_PATH,
        loras=[],
        offload_mode=_offload_mode_for("A2V"),
        **_quantization_kwargs_for("A2V", _a2v_transformer_path()),
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

    strength = _a2v_image_strength()
    logger.info("A2V image conditioning: frame_idx=0 strength=%s", strength)
    return ImageConditioningInput(path=path, frame_idx=0, strength=strength)


def _a2v_last_frame_strength() -> float:
    # How firmly the LAST frame is pinned to the same avatar photo (0 to 1). The
    # ltx-2.5-template-files avatar workflow pins both ends of a clip at 0.7.
    # UNSET or 0 means off (only frame 0 is pinned, as before), so deploying this
    # changes nothing until A2V_LAST_FRAME_STRENGTH is set. Good values are untested.
    value = _env_float("A2V_LAST_FRAME_STRENGTH")
    if value is None:
        return 0.0
    if value > 1.0:
        raise ValueError(f"A2V_LAST_FRAME_STRENGTH must be between 0 and 1, got {value}")
    return value


def _last_frame_conditioning(path: str, num_frames: int) -> list:
    # An extra pin of the same photo at the final frame index (num_frames - 1; the
    # library's assert_image_frames_in_clip accepts 0 <= frame_idx < num_frames).
    # Returns [] when A2V_LAST_FRAME_STRENGTH is unset or 0, and then never touches
    # the library import.
    strength = _a2v_last_frame_strength()
    if strength == 0.0:
        return []
    if num_frames < 2:
        raise ValueError(f"cannot pin a last frame in a clip of {num_frames} frames")
    from ltx_pipelines.utils.types import ImageConditioningInput

    frame_idx = num_frames - 1
    logger.info("A2V last-frame conditioning: frame_idx=%s strength=%s", frame_idx, strength)
    return [ImageConditioningInput(path=path, frame_idx=frame_idx, strength=strength)]


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
    # i2v: a silent-prompt clip that starts from this still (b-roll with a known
    # person or place). Separate from image_key, which only exists with audio_key.
    start_image_key: str | None = None

    @property
    def mode(self) -> str:
        if self.image_key is not None:
            return "ia2v"
        if self.start_image_key is not None:
            return "i2v"
        return "t2v"


def _validated_key(job_input: dict, field: str, pattern: re.Pattern) -> re.Match:
    value = job_input[field]
    match = pattern.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError(f"{field} is not a valid staged input key")
    return match


def validate_input(job_input: dict) -> GenerationRequest:
    allowed_keys = {"prompt", "duration", "aspect_ratio", "image_key", "audio_key", "start_image_key"}
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

    image_key = audio_key = start_image_key = None
    if "start_image_key" in job_input:
        if "image_key" in job_input or "audio_key" in job_input:
            raise ValueError("start_image_key cannot be combined with image_key or audio_key")
        _validated_key(job_input, "start_image_key", IMAGE_KEY_PATTERN)
        start_image_key = job_input["start_image_key"]
    elif ("image_key" in job_input) != ("audio_key" in job_input):
        raise ValueError("image_key and audio_key must be provided together")
    if "image_key" in job_input:
        image_match = _validated_key(job_input, "image_key", IMAGE_KEY_PATTERN)
        audio_match = _validated_key(job_input, "audio_key", AUDIO_KEY_PATTERN)
        if image_match.group(1) != audio_match.group(1):
            raise ValueError("image_key and audio_key must be in the same job folder")
        image_key, audio_key = job_input["image_key"], job_input["audio_key"]
        # avatar only; a bad A2V_SIZE raises ValueError naming the variable.
        if _env_choice("A2V_SIZE", _A2V_SIZE_CHOICES, "full") == "large":
            width, height = A2V_LARGE_ASPECT_RATIO_DIMENSIONS[aspect_ratio]
    else:
        # b-roll only; a bad T2V_SIZE raises ValueError naming the variable,
        # which the handler returns as the job's error.
        if _env_choice("T2V_SIZE", _T2V_SIZE_CHOICES, "full") == "small":
            width, height = T2V_SMALL_ASPECT_RATIO_DIMENSIONS[aspect_ratio]

    return GenerationRequest(
        prompt=job_input["prompt"],
        duration=duration,
        width=width,
        height=height,
        image_key=image_key,
        audio_key=audio_key,
        start_image_key=start_image_key,
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


def _i2v_image_strength() -> float:
    # How firmly frame 0 is pinned to the start still (0 to 1) for an i2v clip.
    # UNSET keeps 1.0. Good values are untested for this pipeline.
    value = _env_float("I2V_IMAGE_STRENGTH")
    if value is None:
        return 1.0
    if value > 1.0:
        raise ValueError(f"I2V_IMAGE_STRENGTH must be between 0 and 1, got {value}")
    return value


def _start_image_conditioning(path: str):
    # Same constructor as _image_conditioning (the deployed ltx_pipelines accepts
    # path/frame_idx/strength); DistilledPipeline takes the same `images` list.
    from ltx_pipelines.utils.types import ImageConditioningInput

    strength = _i2v_image_strength()
    logger.info("I2V start image: frame_idx=0 strength=%s", strength)
    return ImageConditioningInput(path=path, frame_idx=0, strength=strength)


def _generate_i2v_video(request: GenerationRequest) -> bytes:
    with tempfile.TemporaryDirectory() as workdir:
        image_path = os.path.join(workdir, "start" + os.path.splitext(request.start_image_key)[1])
        _download_from_r2(request.start_image_key, image_path)
        return _generate_video(request, image_path=image_path)


def _generate_video(request: GenerationRequest, image_path: str | None = None) -> bytes:
    pipeline = load_pipeline()
    # The VAE's causal temporal grid requires (frames - 1) % FRAME_GRID_STEP == 0;
    # _frames_for_duration rounds UP to the first valid value covering the duration.
    num_frames = _frames_for_duration(request.duration)
    # Logged on every b-roll clip so the size a clip was really made at can be
    # read from the endpoint's logs, not inferred from the T2V_SIZE setting.
    logger.info("T2V size: %sx%s, %s frames", request.width, request.height, num_frames)

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
                images=[_start_image_conditioning(image_path)] if image_path else [],
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
    # Logged on every avatar clip, like the T2V line, so the size a clip was
    # really made at can be read from the endpoint's logs (target size; the
    # library renders stage 1 at half of it).
    logger.info("A2V size: %sx%s, %s frames", request.width, request.height, num_frames)

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
                images=[
                    _image_conditioning(image_path),
                    *_last_frame_conditioning(image_path, num_frames),
                ],
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
    elif request.mode == "i2v":
        video_bytes = _generate_i2v_video(request)
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
