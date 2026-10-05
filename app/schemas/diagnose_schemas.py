from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AudioResult(BaseModel):
    """The audio ensemble's own classification, when the chat was opened from
    an analysis result -- lets diagnose_text ground on it directly instead of
    re-deriving the category from a semantic search over prose that merely
    describes the result (see AUDIO_LABEL_TO_CATEGORY_KEY in rag/diagnose.py)."""

    predicted_class: Literal["belt", "brake", "sway"]
    confidence: float


class SeverityFormResult(BaseModel):
    """The deterministic, client-side severity checklist's own result, when the chat
    was opened from it -- lets diagnose_text ground on the same category and build on
    (rather than re-derive or contradict) an already-computed, rule-based severity
    instead of guessing its own from scratch via the LLM."""

    predicted_class: Literal["belt", "brake", "sway"]
    percent: int
    level: Literal["low", "medium", "high"]
    positive_findings: List[str] = Field(default_factory=list)


class DiagnoseTextRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_length=1)
    audio_result: Optional[AudioResult] = None
    form_result: Optional[SeverityFormResult] = None


class DiagnoseTextResponse(BaseModel):
    predicted_class: Optional[str]
    category_label: Optional[str]
    match_score: float
    matched_phrases: List[str]
    answer: str
    severity: Optional[Literal["low", "medium", "high"]] = None
    severity_reason: Optional[str] = None
