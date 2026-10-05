import numpy as np
import librosa

# Ported verbatim from C:\Users\hp\Desktop\car_test\preprocessing.py -- the script
# that actually generated features.csv / scaler.pkl / best_traditional_model.pkl.
# The traditional-ML model expects a 269-dim vector (40 mfcc mean + 40 std + 40 delta
# mean + 40 delta std + 40 delta2 mean + 40 delta2 std + 7 contrast mean + 7 contrast
# std + 5 scalar mean/std pairs (10) + 5 HPSS scalars = 269). Dropping any of these
# (e.g. the delta/contrast/HPSS terms) makes scaler.transform() raise a feature-count
# mismatch, since StandardScaler was fit on exactly this vector.


def load_clean_audio(path, target_sr, target_duration):
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)

    # A near-silent clip can trim away to nothing; fall back to the untrimmed
    # audio instead of crashing on an empty array (matches car_test/predict.py's
    # fallback_to_untrimmed=True, appropriate for a single live prediction).
    if len(y_trimmed) == 0:
        y_trimmed = y

    peak = np.max(np.abs(y_trimmed))
    if peak > 0:
        y_trimmed = y_trimmed / peak

    target_len = int(target_sr * target_duration)
    if len(y_trimmed) > target_len:
        y_trimmed = y_trimmed[:target_len]
    else:
        y_trimmed = np.pad(y_trimmed, (0, target_len - len(y_trimmed)))

    return y_trimmed, sr


def extract_mfcc_vector(y, sr, n_mfcc):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)

    # HPSS: harmonic (tonal, e.g. belt squeal) vs percussive (impulsive, e.g. sway
    # clunks) energy split -- a direct, physically-motivated signal for that distinction.
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_rms = librosa.feature.rms(y=y_harmonic)
    percussive_rms = librosa.feature.rms(y=y_percussive)
    harmonic_energy = np.mean(harmonic_rms)
    percussive_energy = np.mean(percussive_rms)
    harmonic_ratio = harmonic_energy / (harmonic_energy + percussive_energy + 1e-8)

    def mean_std(feat, axis=1):
        return np.mean(feat, axis=axis), np.std(feat, axis=axis)

    mfcc_mean, mfcc_std = mean_std(mfcc)
    d1_mean, d1_std = mean_std(mfcc_delta)
    d2_mean, d2_std = mean_std(mfcc_delta2)
    contrast_mean, contrast_std = mean_std(contrast)

    scalars = []
    for feat in (zcr, centroid, bandwidth, rolloff, rms):
        scalars.append(np.mean(feat))
        scalars.append(np.std(feat))

    hpss_scalars = [
        harmonic_energy, np.std(harmonic_rms),
        percussive_energy, np.std(percussive_rms),
        harmonic_ratio,
    ]

    return np.concatenate([
        mfcc_mean, mfcc_std,
        d1_mean, d1_std,
        d2_mean, d2_std,
        contrast_mean, contrast_std,
        scalars,
        hpss_scalars,
    ])


def extract_log_mel(y, sr, n_mels, hop_length):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    return librosa.power_to_db(mel, ref=np.max)
