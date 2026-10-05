import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services import ensemble


@pytest.fixture
def client():
    # Not used as a context manager on purpose -- that would run app.main's lifespan
    # (loads TensorFlow/YAMNet/the traditional-ML models for real), which none of
    # these tests need: they all fail before ensemble.predict() is ever reached.
    return TestClient(app)


def test_disallowed_extension_and_content_type_rejected_with_400(client):
    resp = client.post(
        "/predict",
        files={"file": ("malware.exe", b"not audio at all", "application/octet-stream")},
    )

    assert resp.status_code == 400
    assert "format" in resp.json()["detail"].lower()


def test_empty_file_rejected_with_400(client):
    resp = client.post(
        "/predict",
        files={"file": ("clip.wav", b"", "audio/wav")},
    )

    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_oversized_file_rejected_with_400(client, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 0)

    resp = client.post(
        "/predict",
        files={"file": ("clip.wav", b"x" * 1024, "audio/wav")},
    )

    assert resp.status_code == 400
    assert "large" in resp.json()["detail"].lower()


def test_corrupt_audio_rejected_with_400_not_500(client):
    # A real (bundled) ffmpeg will genuinely fail to decode this -- exercises the
    # actual conversion-failure path, not a mock, and confirms it's classified as
    # bad input (400) rather than the old raw-500 behavior.
    resp = client.post(
        "/predict",
        files={"file": ("clip.wav", b"this is not a real wav file" * 50, "audio/wav")},
    )

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "ffmpeg" not in detail.lower()
    assert "traceback" not in detail.lower()


def test_uploaded_bytes_are_actually_written_to_disk(client, monkeypatch):
    # Regression test: the streamed chunk-write loop in predict.py once computed
    # total_bytes from each chunk but never called f.write(chunk), so every upload
    # silently became a 0-byte temp file and ffmpeg rejected it as corrupt -- the
    # existing "corrupt audio -> 400" test couldn't catch this because a 400 was
    # the expected outcome either way. This asserts the bytes ffmpeg actually sees
    # match what was uploaded, independent of real ffmpeg/model behavior.
    import app.routes.predict as predict_route

    captured = {}

    def fake_convert(raw_path, wav_path):
        with open(raw_path, "rb") as f:
            captured["raw_bytes"] = f.read()
        with open(wav_path, "wb") as f:
            f.write(b"placeholder")

    def fake_run_ensemble(wav_path, extra_model=None):
        return {
            "final_prediction": "belt",
            "final_confidence": 0.9,
            "all_probabilities": {"belt": 0.9, "brake": 0.05, "sway": 0.05},
            "individual_models": [],
        }

    monkeypatch.setattr(predict_route, "convert_to_wav", fake_convert)
    monkeypatch.setattr(predict_route, "run_ensemble", fake_run_ensemble)

    payload = b"fake-audio-bytes-0123456789" * 100
    resp = client.post("/predict", files={"file": ("clip.wav", payload, "audio/wav")})

    assert resp.status_code == 200
    assert captured["raw_bytes"] == payload


def test_temp_files_cleaned_up_on_success(client, monkeypatch):
    # Regression test for moving cleanup from predict()'s own `finally` into the
    # background thread's own finally (see the timeout-leak fix in predict.py) --
    # confirms that move didn't just stop cleaning up temp files altogether.
    import os
    import tempfile
    import uuid as uuid_module
    import app.routes.predict as predict_route

    fixed_id = uuid_module.UUID("12345678-1234-5678-1234-567812345678")
    monkeypatch.setattr(predict_route.uuid, "uuid4", lambda: fixed_id)

    def fake_convert(raw_path, wav_path):
        with open(wav_path, "wb") as f:
            f.write(b"placeholder")

    def fake_run_ensemble(wav_path, extra_model=None):
        return {
            "final_prediction": "belt",
            "final_confidence": 0.9,
            "all_probabilities": {"belt": 0.9, "brake": 0.05, "sway": 0.05},
            "individual_models": [],
        }

    monkeypatch.setattr(predict_route, "convert_to_wav", fake_convert)
    monkeypatch.setattr(predict_route, "run_ensemble", fake_run_ensemble)

    resp = client.post("/predict", files={"file": ("clip.wav", b"fake-audio" * 50, "audio/wav")})

    assert resp.status_code == 200
    raw_path = os.path.join(tempfile.gettempdir(), f"{fixed_id}_raw.wav")
    wav_path = os.path.join(tempfile.gettempdir(), f"{fixed_id}.wav")
    assert not os.path.exists(raw_path)
    assert not os.path.exists(wav_path)


def test_temp_files_cleaned_up_when_inference_raises(client, monkeypatch):
    # Same regression concern as above, but for the error path: cleanup used to live
    # in predict()'s outer `finally` (which ran regardless of success/exception) --
    # after moving it into the background thread's own try/finally, confirm it still
    # runs when run_ensemble raises, not just on the happy path.
    import os
    import tempfile
    import uuid as uuid_module
    import app.routes.predict as predict_route

    fixed_id = uuid_module.UUID("87654321-4321-8765-4321-876543218765")
    monkeypatch.setattr(predict_route.uuid, "uuid4", lambda: fixed_id)

    def fake_convert(raw_path, wav_path):
        with open(wav_path, "wb") as f:
            f.write(b"placeholder")

    def fake_run_ensemble(wav_path, extra_model=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(predict_route, "convert_to_wav", fake_convert)
    monkeypatch.setattr(predict_route, "run_ensemble", fake_run_ensemble)

    resp = client.post("/predict", files={"file": ("clip.wav", b"fake-audio" * 50, "audio/wav")})

    assert resp.status_code == 400
    raw_path = os.path.join(tempfile.gettempdir(), f"{fixed_id}_raw.wav")
    wav_path = os.path.join(tempfile.gettempdir(), f"{fixed_id}.wav")
    assert not os.path.exists(raw_path)
    assert not os.path.exists(wav_path)


def test_feature_dimension_mismatch_raises_clear_error():
    feats = np.zeros((1, 100))
    config = {"mfcc_feature_dim": 269}

    with pytest.raises(ensemble.FeatureDimensionError, match="100-dim"):
        ensemble._check_feature_dim(feats, config)


def test_feature_dimension_check_passes_when_dims_match():
    feats = np.zeros((1, 269))
    config = {"mfcc_feature_dim": 269}

    ensemble._check_feature_dim(feats, config)  # must not raise


def test_feature_dimension_check_skipped_when_key_missing_from_older_config():
    feats = np.zeros((1, 42))
    config = {}

    ensemble._check_feature_dim(feats, config)  # must not raise


def test_traditional_probs_checks_dimension_before_touching_scaler(monkeypatch):
    # state deliberately has no "scaler" key -- if the dimension check didn't run
    # before scaler access, this would blow up with KeyError instead of the
    # intended FeatureDimensionError.
    monkeypatch.setattr(ensemble.fe, "extract_mfcc_vector", lambda y, sr, n_mfcc: np.zeros(50))
    config = {"n_mfcc": 40, "mfcc_feature_dim": 269}
    state = {}

    with pytest.raises(ensemble.FeatureDimensionError):
        ensemble._traditional_probs(y=None, sr=None, config=config, state=state)
