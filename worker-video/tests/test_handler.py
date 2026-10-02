import threading
import time
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
