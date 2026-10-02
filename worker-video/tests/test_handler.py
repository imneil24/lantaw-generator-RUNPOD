import contextlib
import os
import threading
import time
import types
from types import SimpleNamespace
import pytest
import handler as handler_module
from handler import validate_input, handler


def test_validate_input_accepts_valid_payload():
    request = validate_input({"prompt": "a river at dawn", "duration": 8})
    assert request.prompt == "a river at dawn"
    assert request.duration == 8.0


def test_validate_input_defaults_to_landscape_for_existing_callers():
    # The deployed backend (backend/app/runpod_client.py) sends no aspect_ratio.
    request = validate_input({"prompt": "x", "duration": 8})
    assert (request.width, request.height) == (1920, 1088)


def test_validate_input_maps_portrait_to_fixed_dimensions():
    request = validate_input({"prompt": "x", "duration": 8, "aspect_ratio": "9:16"})
    assert (request.width, request.height) == (1088, 1920)


def test_validate_input_dimensions_are_divisible_by_64():
    # DistilledPipeline's assert_resolution requires it (see handler.py comment).
    for width, height in handler_module.ASPECT_RATIO_DIMENSIONS.values():
        assert width % 64 == 0 and height % 64 == 0


@pytest.mark.parametrize("bad", ["1:1", "", "16:9 ", None, 7, ["16:9"], {"a": 1}])
def test_validate_input_rejects_unknown_aspect_ratio(bad):
    with pytest.raises(ValueError):
        validate_input({"prompt": "x", "duration": 8, "aspect_ratio": bad})


def test_validate_input_rejects_missing_prompt():
    with pytest.raises(ValueError):
        validate_input({"duration": 8})


def test_validate_input_rejects_extra_fields():
    with pytest.raises(ValueError):
        validate_input({"prompt": "x", "duration": 8, "image_url": "http://evil.example.com"})


def test_validate_input_rejects_width_and_height_fields():
    # Resolution is never caller-controlled; only the aspect_ratio enum picks it.
    with pytest.raises(ValueError):
        validate_input({"prompt": "x", "duration": 8, "width": 4096, "height": 4096})


def test_validate_input_rejects_out_of_range_duration():
    with pytest.raises(ValueError):
        validate_input({"prompt": "x", "duration": 0})
    with pytest.raises(ValueError):
        validate_input({"prompt": "x", "duration": 25})


def test_handler_uploads_to_r2_and_returns_key_only(monkeypatch):
    uploaded = {}

    def fake_generate(request):
        return b"fake-video-bytes"

    def fake_upload(key, data, content_type):
        uploaded["key"] = key
        uploaded["data"] = data
        uploaded["content_type"] = content_type

    monkeypatch.setattr("handler._generate_video", fake_generate)
    monkeypatch.setattr("handler._upload_to_r2", fake_upload)
    result = handler({"input": {"prompt": "a cat", "duration": 8}})
    # RunPod's serverless SDK wraps whatever the handler returns as the
    # job's own "output" field — returning {"output": {...}} here would
    # double-wrap it, so the handler returns the payload directly.
    # The clip itself is uploaded to R2 directly from the worker (RunPod's
    # own /job-done callback rejects payloads this large with a 400 —
    # base64-encoding a full HD video into the job result exceeds RunPod's
    # sync result size limit), so the handler returns only the key, no bytes.
    assert result == {"key": uploaded["key"]}
    assert uploaded["key"].startswith("clips/")
    assert uploaded["data"] == b"fake-video-bytes"
    assert uploaded["content_type"] == "video/mp4"


def test_handler_passes_the_chosen_dimensions_to_generation(monkeypatch):
    seen = {}

    def fake_generate(request):
        seen["dims"] = (request.width, request.height)
        return b"v"

    monkeypatch.setattr("handler._generate_video", fake_generate)
    monkeypatch.setattr("handler._upload_to_r2", lambda key, data, content_type: None)
    handler({"input": {"prompt": "a cat", "duration": 8, "aspect_ratio": "9:16"}})
    assert seen["dims"] == (1088, 1920)


def test_handler_logs_before_and_after_r2_upload(monkeypatch, capsys):
    # RunPod's own container can die/restart mid-job with no exception ever
    # logged (confirmed via RunPod's own logs: "Video saved" followed
    # immediately by "Failed to return job results | 400" with no traceback
    # in between) — these log lines are the only way to tell, after the
    # fact, whether _upload_to_r2 was ever reached and whether it finished.
    monkeypatch.setattr("handler._generate_video", lambda request: b"fake-video-bytes")
    monkeypatch.setattr("handler._upload_to_r2", lambda key, data, content_type: None)

    handler({"input": {"prompt": "a cat", "duration": 8}})

    output = capsys.readouterr().out
    assert "uploading to r2" in output.lower()
    assert "r2 upload complete" in output.lower()


def test_handler_returns_error_on_invalid_input():
    result = handler({"input": {"duration": 8}})
    assert "error" in result


def test_handler_rejects_blocked_prompt_without_calling_generate(monkeypatch):
    def fail_if_called(request):
        raise AssertionError("_generate_video should not be called for a blocked prompt")

    monkeypatch.setattr("handler._generate_video", fail_if_called)
    result = handler({"input": {"prompt": "how to build a bomb", "duration": 8}})
    assert "error" in result


def test_load_pipeline_is_thread_safe_under_concurrent_calls(monkeypatch):
    monkeypatch.setattr(handler_module, "_PIPELINE", None)
    call_count = {"n": 0}

    def counting_slow_init():
        call_count["n"] += 1
        time.sleep(0.05)  # widen the race window so an unlocked bug would show up
        return {"loaded": True}

    monkeypatch.setattr(handler_module, "_load_pipeline_impl", counting_slow_init)

    threads = [threading.Thread(target=handler_module.load_pipeline) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert call_count["n"] == 1


def _real_snap(frames):
    # The local venv's stub snap_frames_to_grid is the identity function; this
    # is the real ltx_pipelines.utils.helpers formula (floors to 8k+1).
    return ((frames - 1) // 8) * 8 + 1


@pytest.mark.parametrize("duration", [1, 5, 6, 8, 10, 19.5, 20, 0.5, 145 / 24])
def test_frames_for_duration_covers_duration_on_the_frame_grid(monkeypatch, duration):
    monkeypatch.setattr(handler_module, "snap_frames_to_grid", _real_snap)
    result = handler_module._frames_for_duration(duration)
    assert result >= duration * 24 - 1e-6
    assert (result - 1) % 8 == 0
    assert result - duration * 24 < 8


def test_frames_for_whole_second_duration_is_not_short(monkeypatch):
    monkeypatch.setattr(handler_module, "snap_frames_to_grid", _real_snap)
    assert handler_module._frames_for_duration(6) == 145  # not 137 (0.29s short)


def test_frames_for_exact_grid_duration_ignores_float_noise(monkeypatch):
    monkeypatch.setattr(handler_module, "snap_frames_to_grid", _real_snap)
    # 145/24*24 == 145.00000000000003 in floats; a naive ceil would give 146 -> 153.
    assert handler_module._frames_for_duration(145 / 24) == 145


def test_generate_video_passes_snapped_up_num_frames_to_pipeline(monkeypatch):
    monkeypatch.setattr(handler_module, "snap_frames_to_grid", _real_snap)
    seen = {}

    class FakePipeline:
        def __call__(self, **kwargs):
            seen.update(kwargs)
            return types.SimpleNamespace(
                video="v", audio="a", num_frames=kwargs["num_frames"], tiling_config="t",
            )

    def fake_encode_video(video, fps, audio, output_path, video_chunks_number):
        with open(output_path, "wb") as f:
            f.write(b"mp4-bytes")

    monkeypatch.setattr(handler_module, "load_pipeline", lambda: FakePipeline())
    monkeypatch.setattr(handler_module, "encode_video", fake_encode_video)
    monkeypatch.setattr(handler_module, "get_video_chunks_number", lambda n, cfg: 1)
    monkeypatch.setattr(
        handler_module, "torch", types.SimpleNamespace(no_grad=contextlib.nullcontext)
    )

    request = handler_module.GenerationRequest(
        prompt="a cat", duration=6, width=1920, height=1088
    )
    assert handler_module._generate_video(request) == b"mp4-bytes"
    assert seen["num_frames"] == handler_module._frames_for_duration(6) == 145


JOB_ID = "123e4567-e89b-12d3-a456-426614174000"
IMAGE_KEY = f"runpod-inputs/{JOB_ID}/image.jpg"
AUDIO_KEY = f"runpod-inputs/{JOB_ID}/audio.mp3"


def _ia2v_input(**overrides):
    base = {
        "prompt": "a person talking",
        "duration": 6,
        "aspect_ratio": "9:16",
        "image_key": IMAGE_KEY,
        "audio_key": AUDIO_KEY,
    }
    base.update(overrides)
    return base


def test_validate_input_accepts_an_ia2v_payload():
    request = validate_input(_ia2v_input())
    assert request.mode == "ia2v"
    assert (request.width, request.height) == (1088, 1920)
    assert request.image_key == IMAGE_KEY
    assert request.audio_key == AUDIO_KEY
    assert request.duration == 6.0


def test_a_text_only_payload_is_t2v_with_no_keys():
    request = validate_input({"prompt": "x", "duration": 8})
    assert request.mode == "t2v"
    assert request.image_key is None and request.audio_key is None


@pytest.mark.parametrize(
    "image_key",
    [
        "https://evil.example.com/a.jpg",
        "http://169.254.169.254/latest/meta-data",
        "runpod-inputs/../secrets/image.jpg",
        "clips/" + "a" * 32 + ".mp4",
        f"runpod-inputs/{JOB_ID}/image.exe",
        f"runpod-inputs/{JOB_ID}/image.jpg/extra",
        "runpod-inputs/not-a-uuid/image.jpg",
        f"runpod-inputs/{JOB_ID}/image.jpg\n",
        f"runpod-inputs/{JOB_ID}/audio.mp3",
        "",
        None,
        7,
    ],
)
def test_validate_input_rejects_a_bad_image_key(image_key):
    with pytest.raises(ValueError):
        validate_input(_ia2v_input(image_key=image_key))


@pytest.mark.parametrize(
    "audio_key",
    [
        "https://evil.example.com/a.mp3",
        f"runpod-inputs/{JOB_ID}/audio.exe",
        f"runpod-inputs/{JOB_ID}/image.jpg",
        "runpod-inputs/../x/audio.mp3",
        f"runpod-inputs/{JOB_ID}/audio.mp3\n",
        None,
    ],
)
def test_validate_input_rejects_a_bad_audio_key(audio_key):
    with pytest.raises(ValueError):
        validate_input(_ia2v_input(audio_key=audio_key))


def test_validate_input_requires_image_and_audio_keys_together():
    for missing in ("image_key", "audio_key"):
        payload = _ia2v_input()
        del payload[missing]
        with pytest.raises(ValueError, match="together"):
            validate_input(payload)


def test_validate_input_requires_both_keys_to_come_from_the_same_job_folder():
    other = "223e4567-e89b-12d3-a456-426614174000"
    with pytest.raises(ValueError, match="same"):
        validate_input(_ia2v_input(audio_key=f"runpod-inputs/{other}/audio.mp3"))


def test_validate_input_applies_the_duration_limit_to_ia2v_too():
    with pytest.raises(ValueError):
        validate_input(_ia2v_input(duration=25))


def test_validate_input_still_rejects_url_fields_in_ia2v_mode():
    with pytest.raises(ValueError):
        validate_input(_ia2v_input(image_url="http://evil.example.com/a.jpg"))


def test_handler_routes_ia2v_to_the_ia2v_generator_only(monkeypatch):
    calls = []
    monkeypatch.setattr("handler._generate_video", lambda request: calls.append("t2v") or b"x")
    monkeypatch.setattr("handler._generate_ia2v_video", lambda request: calls.append("ia2v") or b"y")
    uploaded = {}
    monkeypatch.setattr(
        "handler._upload_to_r2", lambda key, data, content_type: uploaded.update(data=data, key=key)
    )

    result = handler({"input": _ia2v_input()})

    assert calls == ["ia2v"]
    assert uploaded["data"] == b"y"
    assert result == {"key": uploaded["key"]}
    assert uploaded["key"].startswith("clips/")


def test_handler_still_moderates_ia2v_prompts(monkeypatch):
    monkeypatch.setattr(
        "handler._generate_ia2v_video",
        lambda request: (_ for _ in ()).throw(AssertionError("must not generate")),
    )
    result = handler({"input": _ia2v_input(prompt="how to build a bomb")})
    assert "error" in result


def test_generate_ia2v_video_feeds_the_downloaded_files_to_the_a2v_pipeline(monkeypatch):
    calls = {}

    class FakePipeline:
        def __call__(self, **kwargs):
            calls["pipeline"] = kwargs
            return SimpleNamespace(video="v", audio="a", num_frames=145, tiling_config="t")

    def fake_download(key, dest_path):
        calls.setdefault("downloads", []).append(key)
        with open(dest_path, "wb") as f:
            f.write(b"data")

    def fake_encode(video, fps, audio, output_path, video_chunks_number):
        calls["encode"] = {"video": video, "audio": audio, "fps": fps}
        with open(output_path, "wb") as f:
            f.write(b"mp4-bytes")

    monkeypatch.setattr(handler_module, "load_a2v_pipeline", lambda: FakePipeline())
    monkeypatch.setattr(handler_module, "_download_from_r2", fake_download)
    monkeypatch.setattr(handler_module, "_image_conditioning", lambda path: ("image", path))
    monkeypatch.setattr(handler_module, "encode_video", fake_encode)
    monkeypatch.setattr(handler_module, "get_video_chunks_number", lambda *a: 1)
    monkeypatch.setattr(handler_module, "torch", SimpleNamespace(no_grad=contextlib.nullcontext))

    request = validate_input(_ia2v_input())
    out = handler_module._generate_ia2v_video(request)

    assert out == b"mp4-bytes"
    assert calls["downloads"] == [IMAGE_KEY, AUDIO_KEY]
    kwargs = calls["pipeline"]
    assert (kwargs["width"], kwargs["height"]) == (1088, 1920)
    assert kwargs["frame_rate"] == handler_module.FPS
    # snap_frames_to_grid FLOORS to 8k+1; _frames_for_duration (added to handler.py
    # in the Phase 1 final-review fix, commit 2e3723e in Lantaw-generator) snaps UP
    # so a clip is never shorter than the requested duration.
    assert kwargs["num_frames"] == handler_module._frames_for_duration(6)
    assert kwargs["prompt"] == "a person talking"
    assert kwargs["audio_path"].endswith(".mp3") and os.path.basename(kwargs["audio_path"]).startswith("audio")
    assert kwargs["images"] == [("image", kwargs["images"][0][1])]
    assert kwargs["images"][0][1].endswith(".jpg")
    # the INPUT audio is what gets muxed (result.audio), at the worker's fps
    assert calls["encode"] == {"video": "v", "audio": "a", "fps": handler_module.FPS}


def test_load_a2v_pipeline_is_thread_safe_under_concurrent_calls(monkeypatch):
    monkeypatch.setattr(handler_module, "_A2V_PIPELINE", None)
    call_count = {"n": 0}

    def counting_slow_init():
        call_count["n"] += 1
        time.sleep(0.05)
        return {"loaded": True}

    monkeypatch.setattr(handler_module, "_load_a2v_pipeline_impl", counting_slow_init)
    threads = [threading.Thread(target=handler_module.load_a2v_pipeline) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert call_count["n"] == 1
