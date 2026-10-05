"""Tests for app/services/model_loader.py's optional-model loading wrapper.

Regression coverage for a real bug: the PANNs/AST/EfficientAT loaders used to catch
only ImportError, so any OTHER failure (no internet on a first-run Hugging Face/
torch.hub download, a corrupted or architecture-mismatched checkpoint, ...) crashed
the whole service at startup instead of just skipping that one optional model. These
tests exercise _load_optional directly with fake loaders, so they don't need real
model files, torch, or transformers installed.
"""

import pytest

from app.services import model_loader


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    monkeypatch.setattr(model_loader, "_state", {})


def test_disabled_model_never_calls_loader(tmp_path):
    checkpoint = tmp_path / "best_ast_model.pkl"
    checkpoint.write_bytes(b"fake")
    calls = []

    model_loader._load_optional(
        "AST", "ENABLE_AST", False, str(checkpoint), lambda data_dir: calls.append(data_dir), str(tmp_path),
    )

    assert calls == []
    assert model_loader._state == {}


def test_missing_checkpoint_file_skips_silently(tmp_path):
    missing_path = tmp_path / "best_ast_model.pkl"  # never created
    calls = []

    model_loader._load_optional(
        "AST", "ENABLE_AST", True, str(missing_path), lambda data_dir: calls.append(data_dir), str(tmp_path),
    )

    assert calls == []
    assert model_loader._state == {}


def test_successful_loader_populates_state(tmp_path):
    checkpoint = tmp_path / "best_ast_model.pkl"
    checkpoint.write_bytes(b"fake")

    def fake_loader(data_dir):
        model_loader._state["ast_model"] = "loaded"

    model_loader._load_optional("AST", "ENABLE_AST", True, str(checkpoint), fake_loader, str(tmp_path))

    assert model_loader._state["ast_model"] == "loaded"


@pytest.mark.parametrize("exc", [RuntimeError("state_dict key mismatch"), OSError("no internet"), ValueError("bad pickle")])
def test_non_import_error_from_loader_does_not_propagate(tmp_path, exc):
    """The bug this guards against: previously only ImportError was caught, so any of
    these (a mismatched checkpoint, a failed Hugging Face download, ...) would crash
    load_all() and take the entire service down instead of just skipping this model."""
    checkpoint = tmp_path / "best_efficientat_ft_model.pt"
    checkpoint.write_bytes(b"fake")

    def failing_loader(data_dir):
        raise exc

    model_loader._load_optional(
        "EfficientAT (fine-tuned)", "ENABLE_EFFICIENTAT", True, str(checkpoint), failing_loader, str(tmp_path),
    )  # must not raise

    assert model_loader._state == {}


def test_other_gate_loader_populates_expected_state_keys(tmp_path):
    import pickle

    with open(tmp_path / "best_traditional_model_with_other.pkl", "wb") as f:
        pickle.dump({"model": "fake-model", "name": "fake"}, f)
    with open(tmp_path / "scaler_with_other.pkl", "wb") as f:
        pickle.dump("fake-scaler", f)
    with open(tmp_path / "label_encoder_with_other.pkl", "wb") as f:
        pickle.dump("fake-encoder", f)

    model_loader._load_other_gate(str(tmp_path))

    assert model_loader._state["other_gate_model"] == "fake-model"
    assert model_loader._state["other_gate_scaler"] == "fake-scaler"
    assert model_loader._state["other_gate_label_encoder"] == "fake-encoder"


def test_import_error_from_loader_still_does_not_propagate(tmp_path):
    checkpoint = tmp_path / "best_panns_model.pkl"
    checkpoint.write_bytes(b"fake")

    def failing_loader(data_dir):
        raise ImportError("panns_inference not installed")

    model_loader._load_optional(
        "PANNs (CNN14)", "ENABLE_PANNS", True, str(checkpoint), failing_loader, str(tmp_path),
    )  # must not raise

    assert model_loader._state == {}
