import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.core.config import settings
from app.routes import predict, diagnose
from app.services.model_loader import CORE_STATE_KEYS, get_state, load_all
from app.services.rag.store import load_index as load_rag_index, warm_up as warm_up_rag_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("garageai.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.INTERNAL_API_KEY:
        logger.warning(
            "INTERNAL_API_KEY is not set -- /predict and /diagnose/text are reachable "
            "by anyone who can reach this service, with no auth check at all. Fine for "
            "local dev talking to the Node gateway over localhost, but set "
            "INTERNAL_API_KEY before exposing this service beyond localhost."
        )

    logger.info("Loading GarageAI models... this may take a moment.")
    load_all()
    logger.info("Models loaded successfully.")

    logger.info("Loading GarageAI RAG glossary index...")
    load_rag_index()
    warm_up_rag_model()
    logger.info("RAG index ready.")

    yield


app = FastAPI(
    title="GarageAI - Audio Analysis Microservice",
    description="GarageAI — سيرفر داخلي لتحليل أصوات أعطال السيارات (النموذج الرسمي: Traditional ML/XGBoost؛ CNN + YAMNet وغيرهم للمقارنة فقط)",
    version="1.0.0",
    docs_url="/docs",
    lifespan=lifespan,
)


app.include_router(predict.router, tags=["Predict"])
app.include_router(diagnose.router, tags=["Diagnose"])


def _health_payload():
    state = get_state()
    components = {key: key in state for key in CORE_STATE_KEYS}
    models_loaded = all(components.values())
    return {
        "status": "ok" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "components": components,
    }


@app.get("/")
async def root():
    return _health_payload()


@app.get("/health")
async def health():
    return _health_payload()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8001)
