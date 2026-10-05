import numpy as np
import librosa

from app.services.model_loader import get_state
from app.services import feature_extraction as fe
from app.services import optional_models

NAME_MAP = {
    "Traditional ML": "traditional_ml",
    "CNN": "cnn",
    "Transfer Learning (YAMNet)": "yamnet",
    "PANNs (CNN14)": "panns",
    "AST": "ast",
    "CLAP": "clap",
    "PaSST": "passt",
    "BEATs": "beats",
    "EfficientAT (fine-tuned)": "efficientat",
}


class FeatureDimensionError(RuntimeError):
    """Raised when the traditional-ML feature vector's dimension doesn't match
    what scaler.pkl/best_traditional_model.pkl were fit on."""


def _check_feature_dim(feats, config):
    # Mirrors car_test/predict.py's _check_feature_dim -- fail loudly instead of
    # silently mis-scaling if feature_extraction.py's output dimension ever drifts
    # from what scaler.pkl/best_traditional_model.pkl were fit on.
    expected = config.get("mfcc_feature_dim")
    actual = feats.shape[1]
    if expected is not None and actual != expected:
        raise FeatureDimensionError(
            f"Traditional-ML feature dim mismatch: extractor produced {actual}-dim "
            f"features but scaler.pkl/best_traditional_model.pkl were fit on "
            f"{expected}-dim features. The feature extraction code and the deployed "
            f"model are out of sync -- retrain before serving predictions."
        )


def _traditional_probs(y, sr, config, state):
    feats = fe.extract_mfcc_vector(y, sr, config["n_mfcc"]).reshape(1, -1)
    _check_feature_dim(feats, config)
    feats_scaled = state["scaler"].transform(feats)
    return state["traditional_model"].predict_proba(feats_scaled)[0]


def _cnn_probs(y, sr, config, state):
    log_mel = fe.extract_log_mel(y, sr, config["n_mels"], config["hop_length"])
    log_mel = (log_mel - state["mel_stats"]["mean"]) / (state["mel_stats"]["std"] + 1e-8)
    log_mel = log_mel[np.newaxis, ..., np.newaxis]
    return state["cnn_model"].predict(log_mel, verbose=0)[0]


def _transfer_probs(wav_path, state):
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    target_len = int(16000 * 5.0)
    y = y[:target_len] if len(y) > target_len else np.pad(y, (0, target_len - len(y)))
    _, frame_embeddings, _ = state["yamnet"](y.astype(np.float32))
    embedding = np.mean(frame_embeddings.numpy(), axis=0).reshape(1, -1)
    embedding_scaled = state["transfer_scaler"].transform(embedding)
    return state["transfer_model"].predict_proba(embedding_scaled)[0]


def _panns_probs(wav_path, state):
    y, sr = librosa.load(wav_path, sr=32000, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    target_len = int(32000 * 5.0)
    y = y[:target_len] if len(y) > target_len else np.pad(y, (0, target_len - len(y)))
    _, embedding = state["panns_tagger"].inference(y.astype(np.float32)[np.newaxis, :])
    embedding_scaled = state["panns_scaler"].transform(embedding)
    return state["panns_model"].predict_proba(embedding_scaled)[0]


def _efficientat_probs(wav_path, state):
    import torch
    y, sr = librosa.load(wav_path, sr=32000, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    target_len = int(32000 * 5.0)
    y = y[:target_len] if len(y) > target_len else np.pad(y, (0, target_len - len(y)))
    with torch.no_grad():
        waveform = torch.tensor(y.astype(np.float32)[np.newaxis, :])
        spec = state["efficientat_mel"](waveform).unsqueeze(1)
        logits, _ = state["efficientat_model"](spec)
        probs = torch.softmax(logits, dim=1).numpy()[0]
    return probs


# Mirrors car_test/predict.py's OTHER_GATE_THRESHOLD -- keep both in sync, and see
# that constant's comment for the full story: a first attempt at 0.65 (tuned only
# against the with_other model's TEST-split "other" probabilities) looked safe but
# broke 7/31 held-out generalization probes (novel non-car sounds never seen in
# train/val/test at all, e.g. Windows alarms/rings/silence) -- caught by testing a
# live example before shipping. 0.46 is the corrected value: fixes one known real-
# fault false negative (sway_0128_cut.wav) with zero measured cost to either the
# TEST-split "other" samples (min P(other)=0.49) or the held-out probes (min
# P(other)=0.505). Don't raise this without re-checking BOTH populations.
OTHER_GATE_THRESHOLD = 0.46


def _other_gate_probs(y, sr, config, state):
    """Runs BEFORE any belt/brake/sway model -- a genuine supervised 4-class
    classifier (car_test/train_other_class_detector.py), trained with real "other"
    examples, not a bolt-on anomaly detector (see model_loader._load_other_gate's
    docstring for why the unsupervised approach was abandoned).

    Always returns its full {belt, brake, other, sway} probability vector (or None
    if the gate isn't loaded -- not trained/deployed yet), regardless of whether
    P(other) clears OTHER_GATE_THRESHOLD -- predict() below both uses it to decide
    whether to short-circuit AND surfaces P(other) in every response (even ones
    that end up belt/brake/sway) so the UI can show it as a 4th, always-present
    number instead of only when the gate actually fires."""
    if "other_gate_model" not in state:
        return None

    feats = fe.extract_mfcc_vector(y, sr, config["n_mfcc"]).reshape(1, -1)
    feats_scaled = state["other_gate_scaler"].transform(feats)
    gate_classes = list(state["other_gate_label_encoder"].classes_)
    probs = state["other_gate_model"].predict_proba(feats_scaled)[0]
    return {str(c): float(p) for c, p in zip(gate_classes, probs)}


def predict(wav_path: str, extra_model: str | None = None) -> dict:
    """extra_model, if given, must already be validated by the caller (see
    predict.py route's ALLOWED_EXTRA_MODELS check) -- one of optional_models.
    DISPLAY_NAMES' keys ("ast"/"clap"), loaded fresh for just this request."""
    state = get_state()
    config = state["config"]
    label_encoder = state["label_encoder"]
    classes = list(label_encoder.classes_)

    y, sr = fe.load_clean_audio(wav_path, config["target_sr"], config["target_duration"])

    gate_probs = _other_gate_probs(y, sr, config, state)
    if gate_probs is not None and gate_probs["other"] >= OTHER_GATE_THRESHOLD:
        return {
            "final_prediction": "other",
            "final_confidence": gate_probs["other"],
            "all_probabilities": gate_probs,
            "individual_models": [],
        }

    probs_by_name = {
        "Traditional ML": _traditional_probs(y, sr, config, state),
        "CNN": _cnn_probs(y, sr, config, state),
        "Transfer Learning (YAMNet)": _transfer_probs(wav_path, state),
    }
    if "panns_model" in state:
        probs_by_name["PANNs (CNN14)"] = _panns_probs(wav_path, state)
    if "efficientat_model" in state:
        probs_by_name["EfficientAT (fine-tuned)"] = _efficientat_probs(wav_path, state)
    if extra_model is not None:
        display_name = optional_models.DISPLAY_NAMES[extra_model]
        probs_by_name[display_name] = optional_models.run(extra_model, wav_path)

    individual_models = []
    for name, probs in probs_by_name.items():
        idx = int(np.argmax(probs))
        individual_models.append({
            "model_name": NAME_MAP[name],
            "predicted_class": str(classes[idx]),
            "confidence": float(probs[idx]),
            "all_probabilities": {str(c): float(p) for c, p in zip(classes, probs)},
        })

    # Fusion names/weights come from ensemble_config.pkl (built on whatever models were
    # available when evaluate_ensemble.py last ran) -- filter to models we can ACTUALLY
    # compute right now (e.g. PANNs might be in there but not installed here), and
    # renormalize the remaining weights so they still sum to 1 instead of silently
    # under-weighting the fused result.
    names = [n for n in state["ensemble_names"] if n in probs_by_name]
    if state["ensemble_mode"] == "stacking" and set(names) == set(state["ensemble_names"]):
        stacked_features = np.concatenate([probs_by_name[n] for n in state["ensemble_names"]]).reshape(1, -1)
        fused = state["ensemble_meta_model"].predict_proba(stacked_features)[0]
    else:
        raw_weights = state["ensemble_weights"] if state["ensemble_mode"] == "weights" else None
        if raw_weights is not None and set(names) == set(state["ensemble_names"]):
            weights = dict(zip(state["ensemble_names"], raw_weights))
        else:
            # stacking model unusable (a model it needs is missing) or no weights saved --
            # fall back to a plain unweighted average of whatever models ARE available.
            weights = {n: 1.0 / len(names) for n in names}
        total = sum(weights[n] for n in names) or 1.0
        fused = None
        for name in names:
            w = weights[name] / total
            probs = probs_by_name[name]
            fused = w * probs if fused is None else fused + w * probs

    final_idx = int(np.argmax(fused))
    all_probabilities = {str(c): float(p) for c, p in zip(classes, fused)}
    if gate_probs is not None:
        # Informational only -- P(other) comes from a separate model's own judgment
        # call, not from this softmax, so it's layered on top rather than folded into
        # the belt/brake/sway normalization (that would let a gate score that never
        # even cleared OTHER_GATE_THRESHOLD silently shift the real classes' displayed
        # weights). The bars won't sum to exactly 100% -- expected, not a bug.
        all_probabilities["other"] = gate_probs["other"]
    return {
        "final_prediction": str(classes[final_idx]),
        "final_confidence": float(fused[final_idx]),
        "all_probabilities": all_probabilities,
        "individual_models": individual_models,
    }