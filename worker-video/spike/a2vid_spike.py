"""THROWAWAY spike -- delete after Task 8. Runs A2VidPipelineTwoStage against
the LTX-2.5 weights on /runpod-volume with one avatar image + one audio file.

Usage: python a2vid_spike.py IMAGE AUDIO OUT_MP4 ASPECT(16:9|9:16) SECONDS [DISTILLED_LORA]
"""
import os
import sys
import time

import torch

sys.path.insert(0, "/worker")
import handler  # reuse its model path constants and encode helpers
from ltx_core.model.video_vae import get_video_chunks_number
from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage
from ltx_pipelines.utils.helpers import snap_frames_to_grid
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import ImageConditioningInput, OffloadMode


def main(image, audio, out, aspect, seconds, distilled_lora=None):
    width, height = handler.ASPECT_RATIO_DIMENSIONS[aspect]
    model_paths = ModelPaths.from_split(
        transformer_path=handler.TRANSFORMER_PATH,
        text_encoder_path=handler.TEXT_ENCODER_PATH,
        video_vae_path=handler.VIDEO_VAE_PATH,
        audio_vae_path=handler.AUDIO_VAE_PATH,
    )
    loras = []  # constructor kwarg name per Step 1's signature
    distilled = []  # fill from DISTILLED_LORA arg if the signature requires it
    t0 = time.time()
    pipeline = A2VidPipelineTwoStage(
        model_paths=model_paths,
        distilled_lora=distilled,
        spatial_upsampler_path=handler.SPATIAL_UPSAMPLER_PATH,
        loras=loras,
        offload_mode=OffloadMode.CPU,
    )
    print(f"load: {time.time() - t0:.1f}s")
    num_frames = handler._frames_for_duration(float(seconds))
    t0 = time.time()
    with torch.no_grad():
        result = pipeline(
            prompt="A person speaking directly to the camera",
            seed=1234,
            height=height,
            width=width,
            frame_rate=handler.FPS,
            num_frames=num_frames,
            images=[ImageConditioningInput(path=image, frame_idx=0, strength=1.0)],
            audio_path=audio,
        )
        encode_video(
            video=result.video,
            fps=handler.FPS,
            audio=result.audio,
            output_path=out,
            video_chunks_number=get_video_chunks_number(result.num_frames, result.tiling_config),
        )
    print(f"generate: {time.time() - t0:.1f}s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main(*sys.argv[1:])
