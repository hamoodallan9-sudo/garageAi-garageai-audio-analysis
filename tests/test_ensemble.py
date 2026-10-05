"""Tests for app/services/ensemble.py's fusion logic.

Uses fake model objects (plain classes with a .predict_proba, not real trained
artifacts) so this runs fast and without needing car_test's processed_data/ --
it's testing the ARITHMETIC of combining model outputs, not the models themselves.
"""

import numpy as np
import pytest

from app.services import ensemble
from app.services.ensemble import FeatureDimensionError, _check_feature_dim


CLASSES = ["belt", "brake", "sway"]


class FakeProbaModel:
    """Stands in for a fitted sklearn-style model: always returns the same probs."""

    def __init__(self, probs):
        self._probs = np.array([probs])

    def predict_proba(self, X):
        return self._probs


class FakeScaler:
    def transform(self, X):
        return X


class FakeKerasModel:
    def __init__(self, probs):
        self._probs = np.array([probs])

    def predict(self, X, verbose=0):
        return self._probs


class FakeLabelEncoder:
    classes_ = np.array(CLASSES)


class _FakeTensor:
    """Stands in for a tf.Tensor -- just enough to support the .numpy() call
    ensemble.py makes on YAMNet's output embeddings."""

    def __init__(self, array):
        self._array = array

    def numpy(self):
        return self._array


class FakeYamnet:
    def __init__(self, embedding_dim=8):
        self._embedding_dim = embedding_dim

    def __call__(self, y):
        n_frames = 4
        embeddings = np.zeros((n_frames, self._embedding_dim), dtype=np.float32)
        return None, _FakeTensor(embeddings), None


def _base_state(traditional_probs, cnn_probs, yamnet_probs, ensemble_mode, ensemble_names, ensemble_weights=None, meta_model=None):
    return {
        "config": {
            "target_sr": 22050, "target_duration": 5.0, "n_mfcc": 40,
            "n_mels": 128, "hop_length": 512, "mfcc_feature_dim": None,
        },
        "label_encoder": FakeLabelEncoder(),
        "traditional_model": FakeProbaModel(traditional_probs),
        "scaler": FakeScaler(),
        "mel_stats": {"mean": 0.0, "std": 1.0},
        "cnn_model": FakeKerasModel(cnn_probs),
        "yamnet": FakeYamnet(),
        "transfer_model": FakeProbaModel(yamnet_probs),
        "transfer_scaler": FakeScaler(),
        "ensemble_mode": ensemble_mode,
        "ensemble_names": ensemble_names,
        "ensemble_weights": ensemble_weights,
        "ensemble_meta_model": meta_model,
    }


@pytest.fixture
def wav_path(tmp_path):
    import soundfile as sf

    sr = 22050
    t = np.linspace(0, 5.0, int(sr * 5.0), endpoint=False)
    y = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    path = tmp_path / "clip.wav"
    sf.write(path, y, sr)
    return str(path)


class TestCheckFeatureDim:
    def test_raises_when_dimension_does_not_match_deployed_model(self):
        feats = np.zeros((1, 84))
        with pytest.raises(FeatureDimensionError):
            _check_feature_dim(feats, {"mfcc_feature_dim": 269})

    def test_passes_when_dimension_matches(self):
        feats = np.zeros((1, 269))
        _check_feature_dim(feats, {"mfcc_feature_dim": 269})  # must not raise

    def test_is_a_no_op_when_config_has_no_expected_dim(self):
        feats = np.zeros((1, 84))
        _check_feature_dim(feats, {})  # must not raise -- nothing to compare against


class TestPredictFusion:
    def test_weighted_average_matches_hand_computed_result(self, monkeypatch, wav_path):
        # belt, brake, sway
        trad = [0.10, 0.80, 0.10]
        cnn = [0.20, 0.70, 0.10]
        yamnet = [0.05, 0.90, 0.05]
        state = _base_state(
            trad, cnn, yamnet,
            ensemble_mode="weights",
            ensemble_names=["Traditional ML", "CNN", "Transfer Learning (YAMNet)"],
            ensemble_weights=(0.5, 0.3, 0.2),
        )
        monkeypatch.setattr(ensemble, "get_state", lambda: state)

        result = ensemble.predict(wav_path)

        expected_brake = 0.5 * 0.80 + 0.3 * 0.70 + 0.2 * 0.90
        assert result["final_prediction"] == "brake"
        assert result["final_confidence"] == pytest.approx(expected_brake, abs=1e-6)
        assert {m["model_name"] for m in result["individual_models"]} == {"traditional_ml", "cnn", "yamnet"}

    def test_stacking_mode_uses_the_meta_model_when_all_inputs_available(self, monkeypatch, wav_path):
        trad = [0.10, 0.80, 0.10]
        cnn = [0.20, 0.70, 0.10]
        yamnet = [0.05, 0.90, 0.05]
        meta_output = [0.02, 0.95, 0.03]  # deliberately different from any weighted average
        meta_model = FakeProbaModel(meta_output)
        state = _base_state(
            trad, cnn, yamnet,
            ensemble_mode="stacking",
            ensemble_names=["Traditional ML", "CNN", "Transfer Learning (YAMNet)"],
            meta_model=meta_model,
        )
        monkeypatch.setattr(ensemble, "get_state", lambda: state)

        result = ensemble.predict(wav_path)

        assert result["final_prediction"] == "brake"
        assert result["final_confidence"] == pytest.approx(0.95, abs=1e-6)

    def test_falls_back_to_unweighted_average_when_saved_weights_are_absent(self, monkeypatch, wav_path):
        trad = [0.10, 0.80, 0.10]
        cnn = [0.70, 0.20, 0.10]
        yamnet = [0.10, 0.10, 0.80]
        state = _base_state(
            trad, cnn, yamnet,
            ensemble_mode="weights",
            ensemble_names=["Traditional ML", "CNN", "Transfer Learning (YAMNet)"],
            ensemble_weights=None,
        )
        monkeypatch.setattr(ensemble, "get_state", lambda: state)

        result = ensemble.predict(wav_path)

        expected = (np.array(trad) + np.array(cnn) + np.array(yamnet)) / 3
        final_idx = int(np.argmax(expected))
        assert result["final_prediction"] == CLASSES[final_idx]
        assert result["final_confidence"] == pytest.approx(expected[final_idx], abs=1e-6)

    def test_final_confidence_never_drops_to_zero_when_a_saved_weight_is_zero(self, monkeypatch, wav_path):
        # Regression guard for the exact confusion this was built to fix: a model
        # scoring 100% on the right class contributing NOTHING to the final result
        # just because ensemble_config.pkl gave it a saved weight of 0.0 -- the
        # fusion math itself is correct (0 really does mean "ignored"), so this test
        # just documents/locks in that expected (if surprising) behavior.
        trad = [0.10, 0.47, 0.43]
        cnn = [0.10, 0.56, 0.34]
        yamnet = [0.00, 1.00, 0.00]
        state = _base_state(
            trad, cnn, yamnet,
            ensemble_mode="weights",
            ensemble_names=["Traditional ML", "CNN", "Transfer Learning (YAMNet)"],
            ensemble_weights=(0.8, 0.2, 0.0),
        )
        monkeypatch.setattr(ensemble, "get_state", lambda: state)

        result = ensemble.predict(wav_path)

        expected_brake = 0.8 * 0.47 + 0.2 * 0.56
        assert result["final_confidence"] == pytest.approx(expected_brake, abs=1e-6)


class FakeGateLabelEncoder:
    classes_ = np.array(["belt", "brake", "other", "sway"])


class TestOtherGate:
    def test_short_circuits_to_other_without_running_any_other_model(self, monkeypatch, wav_path):
        gate_probs = [0.05, 0.03, 0.90, 0.02]  # belt, brake, other, sway
        state = _base_state(
            [0.10, 0.80, 0.10], [0.20, 0.70, 0.10], [0.05, 0.90, 0.05],
            ensemble_mode="weights",
            ensemble_names=["Traditional ML", "CNN", "Transfer Learning (YAMNet)"],
            ensemble_weights=(0.5, 0.3, 0.2),
        )
        state["other_gate_model"] = FakeProbaModel(gate_probs)
        state["other_gate_scaler"] = FakeScaler()
        state["other_gate_label_encoder"] = FakeGateLabelEncoder()
        # Deliberately remove the heavy models -- if the gate doesn't actually
        # short-circuit, ensemble.predict() would crash reaching for these anyway.
        del state["cnn_model"]
        del state["yamnet"]
        monkeypatch.setattr(ensemble, "get_state", lambda: state)

        result = ensemble.predict(wav_path)

        assert result["final_prediction"] == "other"
        assert result["final_confidence"] == pytest.approx(0.90, abs=1e-6)
        assert result["individual_models"] == []
        assert result["all_probabilities"] == {
            "belt": pytest.approx(0.05), "brake": pytest.approx(0.03),
            "other": pytest.approx(0.90), "sway": pytest.approx(0.02),
        }

    def test_gate_does_not_interfere_when_it_predicts_a_known_class(self, monkeypatch, wav_path):
        gate_probs = [0.10, 0.85, 0.03, 0.02]  # gate itself leans "brake", not "other"
        trad = [0.10, 0.80, 0.10]
        cnn = [0.20, 0.70, 0.10]
        yamnet = [0.05, 0.90, 0.05]
        state = _base_state(
            trad, cnn, yamnet,
            ensemble_mode="weights",
            ensemble_names=["Traditional ML", "CNN", "Transfer Learning (YAMNet)"],
            ensemble_weights=(0.5, 0.3, 0.2),
        )
        state["other_gate_model"] = FakeProbaModel(gate_probs)
        state["other_gate_scaler"] = FakeScaler()
        state["other_gate_label_encoder"] = FakeGateLabelEncoder()
        monkeypatch.setattr(ensemble, "get_state", lambda: state)

        result = ensemble.predict(wav_path)

        # falls through to the normal ensemble fusion, completely unaffected by the gate
        expected_brake = 0.5 * 0.80 + 0.3 * 0.70 + 0.2 * 0.90
        assert result["final_prediction"] == "brake"
        assert result["final_confidence"] == pytest.approx(expected_brake, abs=1e-6)
        assert {m["model_name"] for m in result["individual_models"]} == {"traditional_ml", "cnn", "yamnet"}

    def test_absent_gate_falls_back_to_normal_flow(self, monkeypatch, wav_path):
        # "other_gate_model" not in state at all -- e.g. train_other_class_detector.py
        # hasn't been run/deployed yet. Must not raise, must behave exactly as before.
        trad = [0.10, 0.80, 0.10]
        cnn = [0.20, 0.70, 0.10]
        yamnet = [0.05, 0.90, 0.05]
        state = _base_state(
            trad, cnn, yamnet,
            ensemble_mode="weights",
            ensemble_names=["Traditional ML", "CNN", "Transfer Learning (YAMNet)"],
            ensemble_weights=(0.5, 0.3, 0.2),
        )
        assert "other_gate_model" not in state
        monkeypatch.setattr(ensemble, "get_state", lambda: state)

        result = ensemble.predict(wav_path)

        assert result["final_prediction"] == "brake"
