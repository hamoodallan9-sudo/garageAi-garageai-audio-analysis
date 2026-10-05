"""On-demand comparison models -- loaded FRESH per request, never kept resident.

All four (AST/CLAP/PaSST/BEATs) are strong frozen-embedding models from
car_test's comprehensive benchmark (see car_test/README.md sections 5.1/5.2) --
on this dev machine (6GB RAM, no GPU) keeping any of them always-loaded pushed
free RAM as low as 62MB during testing with just two. Instead of the
ENABLE_PANNS-style "load once at startup, stay resident" pattern used elsewhere in
model_loader.py, these are loaded from scratch for a SINGLE request (when the
frontend user explicitly opts in via `extra_model`) and then garbage-collected --
trading per-request latency (each takes ~20-30s including load time, BEATs can be
longer the first time it downloads its ~360MB checkpoint) for a RAM footprint that
only ever holds one extra heavy model at a time, not several.

_load_semaphore caps this at ONE in-flight optional-model load across ALL
concurrent requests -- two users simultaneously requesting an extra model would
otherwise transiently double the RAM spike this whole module exists to avoid.

NOTE: plain threading.Semaphore, not asyncio.Semaphore -- ensemble.predict() (which
calls run() below) is itself sync code that predict.py's route already runs inside
a worker thread via `asyncio.to_thread(_convert_and_infer_sync)`, so there's no
event loop here to await against.
"""

import os
import pickle
import threading
import urllib.request
from pathlib import Path

import numpy as np
import librosa

from app.core.config import settings

_load_semaphore = threading.Semaphore(1)

# display name -> internal key, mirrors NAME_MAP's convention in ensemble.py
DISPLAY_NAMES = {
    "ast": "AST",
    "clap": "CLAP",
    "passt": "PaSST",
    "beats": "BEATs",
}

BEATS_VENDOR_DIR = os.path.join(os.path.dirname(__file__), "beats_vendor")
# Same verified Hugging Face mirror car_test/train_beats.py uses (the official
# checkpoint is only distributed via a fragile OneDrive link). Downloaded once,
# cached locally under beats_vendor/ (not committed to git -- see .gitignore),
# same pattern as efficientat_vendor's checkpoint handling elsewhere in this service.
BEATS_CHECKPOINT_URL = (
    "https://huggingface.co/datasets/Bencr/beats-checkpoints/resolve/main/BEATs_iter3_plus_AS2M.pt"
)
BEATS_CHECKPOINT_LOCAL = os.path.join(BEATS_VENDOR_DIR, "BEATs_iter3_plus_AS2M.pt")


def _prep_audio(wav_path, sr):
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    target_len = int(sr * 5.0)
    y = y[:target_len] if len(y) > target_len else np.pad(y, (0, target_len - len(y)))
    return y.astype(np.float32)


def _run_ast(wav_path):
    import torch
    from transformers import ASTFeatureExtractor, ASTModel

    data_dir = Path(settings.MODELS_DIR)
    with open(data_dir / "best_ast_model.pkl", "rb") as f:
        saved = pickle.load(f)

    y = _prep_audio(wav_path, sr=16000)
    feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    backbone = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    backbone.eval()
    with torch.no_grad():
        inputs = feature_extractor([y], sampling_rate=16000, return_tensors="pt")
        embedding = backbone(**inputs).last_hidden_state.mean(dim=1).numpy()
    embedding_scaled = saved["scaler"].transform(embedding)
    return saved["model"].predict_proba(embedding_scaled)[0]


def _run_clap(wav_path):
    import torch
    from transformers import ClapProcessor, ClapModel

    data_dir = Path(settings.MODELS_DIR)
    with open(data_dir / "best_clap_model.pkl", "rb") as f:
        saved = pickle.load(f)

    y = _prep_audio(wav_path, sr=48000)
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    model.eval()
    with torch.no_grad():
        inputs = processor(audio=[y], sampling_rate=48000, return_tensors="pt")
        embedding = model.get_audio_features(**inputs).pooler_output.numpy()
    embedding_scaled = saved["scaler"].transform(embedding)
    return saved["model"].predict_proba(embedding_scaled)[0]


def _run_passt(wav_path):
    import torch
    from hear21passt.base import load_model, get_scene_embeddings

    data_dir = Path(settings.MODELS_DIR)
    with open(data_dir / "best_passt_model.pkl", "rb") as f:
        saved = pickle.load(f)

    y = _prep_audio(wav_path, sr=32000)
    model = load_model(mode="embed_only")
    model.eval()
    with torch.no_grad():
        audio = torch.tensor(y[np.newaxis, :])
        embedding = get_scene_embeddings(audio, model).numpy()
    embedding_scaled = saved["scaler"].transform(embedding)
    return saved["model"].predict_proba(embedding_scaled)[0]


def _ensure_beats_checkpoint_downloaded():
    if os.path.exists(BEATS_CHECKPOINT_LOCAL):
        return
    os.makedirs(BEATS_VENDOR_DIR, exist_ok=True)
    urllib.request.urlretrieve(BEATS_CHECKPOINT_URL, BEATS_CHECKPOINT_LOCAL)


def _run_beats(wav_path):
    import sys
    import torch

    if BEATS_VENDOR_DIR not in sys.path:
        sys.path.insert(0, BEATS_VENDOR_DIR)
    from BEATs import BEATs, BEATsConfig  # from beats_vendor/, added to sys.path above

    data_dir = Path(settings.MODELS_DIR)
    with open(data_dir / "best_beats_model.pkl", "rb") as f:
        saved = pickle.load(f)

    y = _prep_audio(wav_path, sr=16000)
    _ensure_beats_checkpoint_downloaded()
    checkpoint = torch.load(BEATS_CHECKPOINT_LOCAL, map_location="cpu", weights_only=False)
    cfg = BEATsConfig(checkpoint["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        audio = torch.tensor(y[np.newaxis, :])
        features, _ = model.extract_features(audio, padding_mask=None)
        embedding = features.mean(dim=1).numpy()
    embedding_scaled = saved["scaler"].transform(embedding)
    return saved["model"].predict_proba(embedding_scaled)[0]


_RUNNERS = {
    "ast": _run_ast,
    "clap": _run_clap,
    "passt": _run_passt,
    "beats": _run_beats,
}


def run(model_key: str, wav_path: str):
    """model_key must already be validated by the caller (see predict.py's
    ALLOWED_EXTRA_MODELS check) -- this raises KeyError on an unknown key on
    purpose, as a second line of defense rather than silently no-op-ing."""
    runner = _RUNNERS[model_key]
    with _load_semaphore:
        return runner(wav_path)
