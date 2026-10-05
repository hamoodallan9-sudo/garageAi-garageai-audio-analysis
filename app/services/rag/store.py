import hashlib
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from app.services.rag.ingest import Chunk, GLOSSARY_PATH, parse_glossary

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
CACHE_PATH = Path(__file__).resolve().parent / ".cache" / "glossary_embeddings.pkl"

_model = None
_chunks: List[Chunk] = []
_embeddings: Optional[np.ndarray] = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_index(force: bool = False) -> None:
    global _chunks, _embeddings

    chunks = parse_glossary(GLOSSARY_PATH)
    current_hash = _file_hash(GLOSSARY_PATH)

    if not force and CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        if cached.get("hash") == current_hash and cached.get("model") == EMBEDDING_MODEL_NAME:
            _chunks = cached["chunks"]
            _embeddings = cached["embeddings"]
            return

    model = _get_model()
    embeddings = model.encode([f"passage: {c.text}" for c in chunks], normalize_embeddings=True)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(
            {"hash": current_hash, "model": EMBEDDING_MODEL_NAME, "chunks": chunks, "embeddings": embeddings},
            f,
        )

    _chunks = chunks
    _embeddings = embeddings


def warm_up() -> None:
    """Forces the embedding model into memory so the first real request isn't slow.

    load_index() alone doesn't do this: when the on-disk cache is valid it just
    unpickles the precomputed passage embeddings and skips loading the model --
    the model is only pulled into memory lazily, on the first search() call.
    """
    _get_model()


def search(query: str, top_k: int = 8) -> List[Tuple[Chunk, float]]:
    if _embeddings is None:
        load_index()

    model = _get_model()
    query_vec = model.encode([f"query: {query}"], normalize_embeddings=True)[0]

    scores = _embeddings @ query_vec
    top_idx = np.argsort(-scores)[:top_k]

    return [(_chunks[i], float(scores[i])) for i in top_idx]
