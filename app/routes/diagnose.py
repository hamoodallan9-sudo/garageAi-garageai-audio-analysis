from fastapi import APIRouter, Depends, HTTPException

from app.core.security import verify_internal_key
from app.schemas.diagnose_schemas import DiagnoseTextRequest, DiagnoseTextResponse
from app.services.rag.diagnose import diagnose_text

router = APIRouter()


@router.post(
    "/diagnose/text",
    response_model=DiagnoseTextResponse,
    summary="تشخيص عطل السيارة من محادثة نصية بالعامية (RAG، متعدد الأدوار)",
    description=(
        "يستقبل قائمة رسائل المحادثة (user/assistant)، يطابق أول رسالة من الزبون دلاليًا "
        "مع قاعدة معرفة الأعطال، ويولّد ردًا مبنيًا على كامل المحادثة."
    ),
    dependencies=[Depends(verify_internal_key)],
)
async def diagnose(payload: DiagnoseTextRequest):
    messages = [m.model_dump() for m in payload.messages]

    if messages[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="The last message must be from the user")

    if not any(m["role"] == "user" and m["content"].strip() for m in messages):
        raise HTTPException(status_code=400, detail="At least one non-empty user message is required")

    audio_result = payload.audio_result.model_dump() if payload.audio_result else None
    form_result = payload.form_result.model_dump() if payload.form_result else None
    return diagnose_text(messages, audio_result=audio_result, form_result=form_result)
