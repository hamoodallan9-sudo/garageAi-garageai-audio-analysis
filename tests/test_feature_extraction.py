"""Regression tests for app/services/feature_extraction.py.

This module exists because of a real production bug: the traditional-ML feature
extractor once produced 84-dim vectors while scaler.pkl/best_traditional_model.pkl
were fit on 269-dim vectors (missing MFCC deltas, spectral contrast, and 5 HPSS
scalars), so every /predict call failed with a scikit-learn feature-count mismatch.
The dimension assertions below exist specifically to catch that class of bug again
before it reaches a running service.
"""

import numpy as np
import soundfile as sf

from app.services.feature_extraction import extract_log_mel, extract_mfcc_vector, load_clean_audio

SR = 22050
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512
TARGET_DURATION = 5.0
EXPECTED_MFCC_DIM = 269  # must match config.pkl's "mfcc_feature_dim"


def _write_wav(tmp_path, duration_sec, sr=SR, freq=220.0):
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    y = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    path = tmp_path / "tone.wav"
    sf.write(path, y, sr)
    return str(path)


class TestLoadCleanAudio:
    def test_pads_short_audio_to_target_length(self, tmp_path):
        path = _write_wav(tmp_path, duration_sec=1.0)
        y, sr = load_clean_audio(path, target_sr=SR, target_duration=TARGET_DURATION)

        assert sr == SR
        assert len(y) == int(SR * TARGET_DURATION)

    def test_trims_long_audio_to_target_length(self, tmp_path):
        path = _write_wav(tmp_path, duration_sec=8.0)
        y, sr = load_clean_audio(path, target_sr=SR, target_duration=TARGET_DURATION)

        assert len(y) == int(SR * TARGET_DURATION)

    def test_falls_back_to_untrimmed_audio_instead_of_crashing_on_near_silence(self, tmp_path):
        # A clip quiet enough that librosa.effects.trim(top_db=25) removes everything --
        # must not raise (np.max on an empty array) and must still return target-length audio.
        sr = SR
        y = (np.random.default_rng(0).normal(0, 1e-6, int(sr * 2.0))).astype(np.float32)
        path = tmp_path / "silence.wav"
        sf.write(path, y, sr)

        result_y, result_sr = load_clean_audio(str(path), target_sr=SR, target_duration=TARGET_DURATION)
        assert result_sr == sr
        assert len(result_y) == int(SR * TARGET_DURATION)
        assert np.all(np.isfinite(result_y))


class TestExtractMfccVector:
    def test_output_dimension_matches_the_deployed_model(self, tmp_path):
        path = _write_wav(tmp_path, duration_sec=TARGET_DURATION)
        y, sr = load_clean_audio(path, target_sr=SR, target_duration=TARGET_DURATION)

        feats = extract_mfcc_vector(y, sr, N_MFCC)

        assert feats.shape == (EXPECTED_MFCC_DIM,)
        assert np.all(np.isfinite(feats))

    def test_dimension_is_independent_of_input_loudness(self, tmp_path):
        # A regression guard for the feature vector's *shape*, not its values -- a
        # louder/quieter clip must not change how many numbers come out.
        path_quiet = tmp_path / "quiet.wav"
        t = np.linspace(0, TARGET_DURATION, int(SR * TARGET_DURATION), endpoint=False)
        sf.write(path_quiet, (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), SR)

        y, sr = load_clean_audio(str(path_quiet), target_sr=SR, target_duration=TARGET_DURATION)
        feats = extract_mfcc_vector(y, sr, N_MFCC)

        assert feats.shape == (EXPECTED_MFCC_DIM,)


class TestExtractLogMel:
    def test_output_shape_matches_cnn_input_expectations(self, tmp_path):
        path = _write_wav(tmp_path, duration_sec=TARGET_DURATION)
        y, sr = load_clean_audio(path, target_sr=SR, target_duration=TARGET_DURATION)

        log_mel = extract_log_mel(y, sr, N_MELS, HOP_LENGTH)

        assert log_mel.shape[0] == N_MELS
        assert np.all(np.isfinite(log_mel))

    def test_values_are_in_db_scale_relative_to_clip_peak(self, tmp_path):
        path = _write_wav(tmp_path, duration_sec=TARGET_DURATION)
        y, sr = load_clean_audio(path, target_sr=SR, target_duration=TARGET_DURATION)

        log_mel = extract_log_mel(y, sr, N_MELS, HOP_LENGTH)

        # ref=np.max means the loudest bin is always ~0dB and everything else is <= 0.
        assert log_mel.max() <= 1e-6
