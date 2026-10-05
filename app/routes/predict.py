import asyncio
import logging
import os
import time
import uuid
import tempfile

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Request

from app.core.config import settings
from app.core.security import verify_internal_key
from app.services.audio_converter import convert_to_wav
from app.services.ensemble import FeatureDimensionError, predict as run_ensemble
from app.schemas.audio_schemas import AnalysisResponse

logger = logging.getLogger("garageai.predict")

router = APIRouter()

CHUNK_SIZE = 1024 * 1024
AUDIO_CONTENT_TYPE_PREFIX = "audio/"
# Keys of optional_models.DISPLAY_NAMES (app/services/optional_models.py) -- kept as a
# separate literal set here (not imported) so this route can validate and reject a bad
# value with a clean 400 *before* touching the optional_models module at all.
ALLOWED_EXTRA_MODELS = {"ast", "clap", "passt", "beats"}

# See settings.PREDICT_MAX_CONCURRENCY's comment (app/core/config.py) -- bounds how
# many requests can have (ffmpeg + inference) work in flight on the shared thread pool
# at once, since a timed-out request's underlying thread keeps running regardless.
_predict_semaphore = asyncio.Semaphore(settings.PREDICT_MAX_CONCURRENCY)


@router.post(
    "/predict",
    response_model=AnalysisResponse,
    summary="تحليل صوت سيارة (GarageAI Internal Microservice)",
    description="يستقبل ملف صوتي، يحوله WAV، ويشغل النموذج الرسمي (Traditional ML / XGBoost)؛ بيانات باقي النماذج (CNN + YAMNet وغيرها) ترجع أيضًا للمقارنة فقط.",
    dependencies=[Depends(verify_internal_key)],
)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    extra_model: str | None = Form(None),
):
    start = time.time()

    if extra_model is not None and extra_model not in ALLOWED_EXTRA_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown extra_model '{extra_model}'. Must be one of: {sorted(ALLOWED_EXTRA_MODELS)}.",
        )

    original_name = file.filename or "audio"
    ext = os.path.splitext(original_name)[1].lower() or ".webm"
    content_type = file.content_type or ""
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    looks_like_audio_ext = ext in settings.ALLOWED_AUDIO_EXTENSIONS
    looks_like_audio_mime = content_type.startswith(AUDIO_CONTENT_TYPE_PREFIX)
    if not (looks_like_audio_ext or looks_like_audio_mime):
        logger.warning(
            "rejected upload with unsupported type: name=%s content_type=%s",
            original_name, content_type,
        )
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > max_bytes:
                logger.warning(
                    "rejected oversized upload (content-length): name=%s size=%s",
                    original_name, content_length,
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large (max {settings.MAX_UPLOAD_SIZE_MB}MB).",
                )
        except ValueError:
            pass

    tmp_id = str(uuid.uuid4())
    raw_path = os.path.join(tempfile.gettempdir(), f"{tmp_id}_raw{ext}")
    wav_path = os.path.join(tempfile.gettempdir(), f"{tmp_id}.wav")

    total_bytes = 0
    try:
        with open(raw_path, "wb") as f:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    logger.warning(
                        "rejected oversized upload (streamed): name=%s size>%d",
                        original_name, max_bytes,
                    )
                    raise HTTPException(
                        status_code=400,
                        detail=f"File too large (max {settings.MAX_UPLOAD_SIZE_MB}MB).",
                    )
    except HTTPException:
        if os.path.exists(raw_path):
            os.remove(raw_path)
        raise

    if total_bytes == 0:
        os.remove(raw_path)
        logger.warning("rejected empty upload: name=%s", original_name)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.info("predict request received: name=%s size_bytes=%d", original_name, total_bytes)

    def _convert_and_infer_sync():
        # Owns cleanup of its own temp files, rather than an outer `finally` in
        # predict() below. If asyncio.wait_for times out, THIS thread keeps running to
        # completion regardless (asyncio cannot cancel work already handed to a thread
        # pool worker) -- deleting raw_path/wav_path from the timed-out caller's own
        # handler would race whatever ffmpeg/the model is still doing with them, and on
        # Windows a still-open file being removed out from under it can raise
        # PermissionError, turning a clean 504 into an unhandled 500.
        try:
            convert_to_wav(raw_path, wav_path)
            return run_ensemble(wav_path, extra_model=extra_model)
        finally:
            for p in (raw_path, wav_path):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    timeout = settings.PREDICT_TIMEOUT_SECONDS_WITH_EXTRA if extra_model else settings.PREDICT_TIMEOUT_SECONDS
    async with _predict_semaphore:
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_convert_and_infer_sync), timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error("predict request timed out after %ss: name=%s extra_model=%s", timeout, original_name, extra_model)
            raise HTTPException(status_code=504, detail="Analysis timed out. Please try again.")
        except FeatureDimensionError:
            logger.exception("feature dimension mismatch while analyzing %s", original_name)
            raise HTTPException(status_code=500, detail="Internal model configuration error.")
        except (RuntimeError, ValueError) as e:
            logger.warning("bad input while analyzing %s: %s", original_name, e)
            raise HTTPException(
                status_code=400,
                detail="Could not process the uploaded audio file. Please check that it is a valid, non-corrupt audio file.",
            )
        except Exception:
            logger.exception("unexpected error while analyzing %s", original_name)
            raise HTTPException(status_code=500, detail="Internal server error during analysis.")

    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        "predict request succeeded: name=%s prediction=%s confidence=%.3f elapsed_ms=%.1f",
        original_name, result["final_prediction"], result["final_confidence"], elapsed_ms,
    )
    return AnalysisResponse(
        final_prediction=result["final_prediction"],
        final_confidence=result["final_confidence"],
        all_probabilities=result["all_probabilities"],
        individual_models=result["individual_models"],
        processing_time_ms=elapsed_ms,
    )
