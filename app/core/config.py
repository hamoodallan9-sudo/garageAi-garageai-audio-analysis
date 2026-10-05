from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "GarageAI - Audio Analysis Microservice"
    APP_VERSION: str = "1.0.0"
    # No default on purpose: this points at car_test's processed_data output, which is
    # machine-specific. Leaving it required means a missing/wrong .env fails loudly at
    # startup (pydantic-settings validation error) instead of pointing at a path that
    # only ever existed on the original dev machine.
    MODELS_DIR: str
    GEMINI_API_KEY: str = ""
    RAG_MATCH_THRESHOLD: float = 0.35
    # When set, /predict and /diagnose/text require this value in the
    # x-internal-api-key header. Left empty in local dev (no check enforced) since this
    # service is only meant to be reached by the Node gateway, never directly by clients.
    INTERNAL_API_KEY: str = ""
    # Off by default: PANNs (CNN14 via panns_inference/torch) isn't in ensemble_config.pkl's
    # fusion at all (it fuses Traditional ML + CNN + YAMNet only), so loading it costs
    # ~500MB+ of RAM for a model that never affects final_prediction. On a machine with
    # ~6GB total RAM, that's a real cause of the process getting silently killed by the OS
    # under memory pressure. Set ENABLE_PANNS=true in .env to load it anyway (e.g. if the
    # ensemble is retuned to actually use it -- standalone it scores close to CNN, so it's
    # a plausible future win, just not free given the RAM cost).
    ENABLE_PANNS: bool = False
    # Same reasoning again, for the fine-tuned EfficientAT model (car_test/
    # train_efficientat_finetune.py) -- also report-only, also beat the traditional ML
    # model standalone (88.72%), also not in ensemble_config.pkl. Lighter than AST
    # (MobileNetV3 backbone, ~19MB checkpoint) but still extra RAM on a constrained
    # machine. Set ENABLE_EFFICIENTAT=true in .env to load it anyway.
    ENABLE_EFFICIENTAT: bool = False
    # Matches the Node gateway's own multer limit (garageai-backend/src/middleware/upload.js)
    # so a file that would be rejected there doesn't even get a chance to hang this service.
    MAX_UPLOAD_SIZE_MB: int = 20
    # Bounds ffmpeg conversion + full ensemble inference for a single /predict call.
    PREDICT_TIMEOUT_SECONDS: int = 40
    # Used instead of PREDICT_TIMEOUT_SECONDS when the request asks for an optional
    # `extra_model` (AST/CLAP/PaSST/BEATs -- see app/services/optional_models.py). These
    # are loaded from scratch per-request (never kept resident, unlike PANNs/EfficientAT),
    # and measured load+inference time for that alone was ~20-30s on top of the ~12s base
    # ensemble on this dev machine -- comfortably over the default 40s budget. BEATs'
    # ~360MB checkpoint should be pre-downloaded (see optional_models.py) rather than
    # relying on this timeout to cover a cold first-request download.
    PREDICT_TIMEOUT_SECONDS_WITH_EXTRA: int = 90
    # Caps how many /predict requests can be running the (ffmpeg + inference) work at
    # once. Needed because PREDICT_TIMEOUT_SECONDS only stops asyncio from waiting on a
    # slow request -- the underlying OS thread (asyncio.to_thread) keeps running to
    # completion regardless, since asyncio cannot force-cancel work already handed to a
    # thread pool worker. Without a cap, enough overlapping slow/timed-out requests
    # could exhaust the shared default thread pool and starve unrelated
    # asyncio.to_thread work elsewhere in this process.
    PREDICT_MAX_CONCURRENCY: int = 4
    ALLOWED_AUDIO_EXTENSIONS: set[str] = {
        ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus",
        ".flac", ".webm", ".wma", ".3gp", ".3gpp", ".amr", ".aiff", ".aif",
    }

    class Config:
        env_file = ".env"

settings = Settings()  # type: ignore[call-arg]  # BaseSettings fills required fields from env/.env, not constructor args; mypy can't see that