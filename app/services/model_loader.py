import logging
import os
import pickle

from app.core.config import settings

logger = logging.getLogger("garageai.model_loader")

_state: dict = {}

# Keys load_all() always populates regardless of ENABLE_PANNS -- used by the /health
# check to report whether the service is actually ready to serve /predict.
CORE_STATE_KEYS = (
    "config", "traditional_model", "scaler", "label_encoder", "mel_stats",
    "transfer_model", "transfer_scaler", "ensemble_names", "cnn_model", "yamnet",
)


def _load_other_gate(data_dir):
    """The "other" gate: a genuine supervised 4-class (belt/brake/sway/other)
    traditional-ML classifier (car_test/train_other_class_detector.py), trained on
    real belt/brake/sway audio PLUS real negative examples (synthetic noise/tones/
    chirps/chords + non-car Windows system sounds) -- not a bolt-on anomaly
    detector. Unsupervised novelty detection (IsolationForest/LOF/EllipticEnvelope,
    on both the MFCC feature space and YAMNet's embedding space) was tried first and
    empirically failed to generalize (2026-08-22 project history: caught at most 1-2
    of 4 clearly non-car test sounds). Cheap (same sklearn model class as the main
    traditional-ML model) so this always loads if present, no ENABLE_* flag needed;
    missing artifacts just mean the gate is skipped (old always-guess-one-of-3
    behavior), same fallback predict.py's CLI already uses."""
    with open(os.path.join(data_dir, "best_traditional_model_with_other.pkl"), "rb") as f:
        _state["other_gate_model"] = pickle.load(f)["model"]
    with open(os.path.join(data_dir, "scaler_with_other.pkl"), "rb") as f:
        _state["other_gate_scaler"] = pickle.load(f)
    with open(os.path.join(data_dir, "label_encoder_with_other.pkl"), "rb") as f:
        _state["other_gate_label_encoder"] = pickle.load(f)


def _load_panns_model(data_dir):
    from panns_inference import AudioTagging
    with open(os.path.join(data_dir, "best_panns_model.pkl"), "rb") as f:
        saved = pickle.load(f)
        _state["panns_model"] = saved["model"]
        _state["panns_scaler"] = saved["scaler"]
    _state["panns_tagger"] = AudioTagging(checkpoint_path=None, device="cpu")


def _load_efficientat_model(data_dir):
    import sys
    import torch
    vendor_dir = os.path.join(os.path.dirname(__file__), "efficientat_vendor")
    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)
    from models.mn.model import get_model as get_mn
    from models.preprocess import AugmentMelSTFT
    from helpers.utils import NAME_TO_WIDTH

    mel = AugmentMelSTFT(n_mels=128, sr=32000)
    mel.eval()
    model = get_mn(num_classes=3, pretrained_name=None, width_mult=NAME_TO_WIDTH("mn10_as"), head_type="mlp")
    state_dict = torch.load(os.path.join(data_dir, "best_efficientat_ft_model.pt"), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    _state["efficientat_model"] = model
    _state["efficientat_mel"] = mel


def _load_optional(display_name, env_flag_name, enabled, checkpoint_path, loader, data_dir):
    """Runs an optional model loader (PANNs/AST/EfficientAT), never letting it take
    down the rest of load_all(). enabled=False and a missing checkpoint file both
    skip silently (nothing to load); any OTHER failure during loader(data_dir) --
    missing package, no internet on a Hugging Face/torch.hub download, a corrupted
    or architecture-mismatched checkpoint -- is caught and logged as a warning
    instead of crashing service startup."""
    if not enabled:
        logger.info(
            "%s loading skipped (%s=false) -- it isn't part of the current "
            "ensemble fusion anyway; set %s=true in .env to load it.",
            display_name, env_flag_name, env_flag_name,
        )
        return
    if not os.path.exists(checkpoint_path):
        return
    try:
        loader(data_dir)
        logger.info("%s model loaded.", display_name)
    except Exception:
        logger.warning(
            "Failed to load %s -- skipping it, it just won't show up in the model "
            "breakdown. Common causes: a required package isn't installed, no internet "
            "on first run (Hugging Face/torch.hub download), or a corrupted/mismatched "
            "checkpoint file.", display_name, exc_info=True,
        )


def load_all():
    data_dir = settings.MODELS_DIR

    with open(os.path.join(data_dir, "config.pkl"), "rb") as f:
        _state["config"] = pickle.load(f)

    with open(os.path.join(data_dir, "best_traditional_model.pkl"), "rb") as f:
        saved = pickle.load(f)
        _state["traditional_model"] = saved["model"]

    with open(os.path.join(data_dir, "scaler.pkl"), "rb") as f:
        _state["scaler"] = pickle.load(f)

    # Fail fast (at startup, in the logs) rather than on the first /predict call if
    # config.pkl's mfcc_feature_dim and the fitted scaler ever fall out of sync.
    expected_dim = _state["config"].get("mfcc_feature_dim")
    scaler_dim = getattr(_state["scaler"], "n_features_in_", None)
    if expected_dim is not None and scaler_dim is not None and scaler_dim != expected_dim:
        logger.warning(
            "config.pkl's mfcc_feature_dim (%s) does not match scaler.pkl's fitted "
            "feature count (%s) -- traditional-ML predictions will fail until these "
            "are back in sync.", expected_dim, scaler_dim,
        )

    with open(os.path.join(data_dir, "label_encoder.pkl"), "rb") as f:
        _state["label_encoder"] = pickle.load(f)

    with open(os.path.join(data_dir, "mel_stats.pkl"), "rb") as f:
        _state["mel_stats"] = pickle.load(f)

    with open(os.path.join(data_dir, "best_transfer_model.pkl"), "rb") as f:
        saved = pickle.load(f)
        _state["transfer_model"] = saved["model"]
        _state["transfer_scaler"] = saved["scaler"]

    with open(os.path.join(data_dir, "ensemble_config.pkl"), "rb") as f:
        ensemble_cfg = pickle.load(f)
        _state["ensemble_mode"] = ensemble_cfg.get("mode", "weights")  # older ensemble_config.pkl files predate "mode"
        _state["ensemble_names"] = ensemble_cfg["names"]
        _state["ensemble_weights"] = ensemble_cfg.get("weights")
        _state["ensemble_meta_model"] = ensemble_cfg.get("meta_model")

    import tensorflow as tf
    from app.services import cnn_custom_layers  # noqa: F401 -- registers the BiGRU layer used by cnn_model.keras
    _state["cnn_model"] = tf.keras.models.load_model(
        os.path.join(data_dir, "cnn_model.keras"), compile=False
    )

    try:
        import tensorflow_hub as hub
        _state["yamnet"] = hub.load("https://tfhub.dev/google/yamnet/1")
    except Exception:
        import kagglehub
        model_path = kagglehub.model_download("google/yamnet/tensorFlow2/yamnet/1")
        _state["yamnet"] = tf.saved_model.load(model_path)

    # PANNs/EfficientAT are optional and OFF by default (see the ENABLE_* comments in
    # app/core/config.py) -- neither is in ensemble_config.pkl's fusion, so a failure
    # loading either must never take down the whole service. _load_optional catches ANY
    # loading failure (missing package, no internet on a Hugging Face/torch.hub download,
    # a corrupted or architecture-mismatched checkpoint, ...), not just ImportError -- a
    # narrower catch here previously meant e.g. an offline machine with ENABLE_PANNS=true
    # would crash the entire service at startup instead of just skipping PANNs.
    # AST is NOT loaded here at all -- unlike PANNs/EfficientAT it isn't safe to keep
    # resident (measured free RAM as low as 62MB with it loaded on this 6GB dev machine).
    # It's loaded fresh per-request instead, only when a user opts in -- see
    # app/services/optional_models.py.
    _load_optional("PANNs (CNN14)", "ENABLE_PANNS", settings.ENABLE_PANNS,
                    os.path.join(data_dir, "best_panns_model.pkl"), _load_panns_model, data_dir)
    _load_optional("EfficientAT (fine-tuned)", "ENABLE_EFFICIENTAT", settings.ENABLE_EFFICIENTAT,
                    os.path.join(data_dir, "best_efficientat_ft_model.pt"), _load_efficientat_model, data_dir)

    # No ENABLE_* flag -- cheap (a small sklearn model, same class as the main
    # traditional-ML one) and always wanted if trained, unlike the RAM-heavy optional
    # models above. Still goes through the same "never let a missing/broken optional
    # artifact take down the whole service" _load_optional wrapper.
    _load_optional('"Other" gate', "(none)", True,
                    os.path.join(data_dir, "best_traditional_model_with_other.pkl"), _load_other_gate, data_dir)


def get_state():
    return _state