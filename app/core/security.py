from fastapi import Header, HTTPException

from app.core.config import settings


async def verify_internal_key(x_internal_api_key: str = Header(default="")):
    """Guards routes meant only for the Node gateway.

    No-op when INTERNAL_API_KEY isn't configured, so local dev (where the two
    services just talk over localhost) keeps working without extra setup.
    """
    if not settings.INTERNAL_API_KEY:
        return
    if x_internal_api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid internal API key")
