import re
from collections import defaultdict
from typing import Dict, List, Optional

from app.core.config import settings
from app.services.rag.ingest import category_context, category_labels, severity_context
from app.services.rag.store import search

# Local embedding scores for this domain cluster tightly (~0.83-0.94) regardless of true
# relevance -- an absolute cutoff can't reliably gate "no match" (verified empirically:
# unrelated queries like "AC not cooling" score as high as genuine matches). So this is
# only a sanity floor for degenerate input, not a real relevance filter -- the actual
# relevance judgment happens inside the LLM prompt below, which can read the retrieved
# text and decide whether it genuinely matches the customer's description.
MATCH_THRESHOLD = settings.RAG_MATCH_THRESHOLD
GEMINI_MODEL = "gemini-3.5-flash-lite"
MAX_HISTORY_MESSAGES = 20  # keep prompts bounded on long chat sessions
GEMINI_TIMEOUT_MS = 25_000  # fail fast instead of hanging the Node request behind us

# The audio ensemble's label_encoder.pkl classes (car_test/processed_data) are the plain
# strings below -- NOT the "01_sway_bar_bushing" etc. keys used internally by this RAG
# module (ingest.py's comment claiming otherwise is stale). This bridges the two.
AUDIO_LABEL_TO_CATEGORY_KEY = {
    "belt": "02_serpentine_belt",
    "brake": "03_brake_squeal",
    "sway": "01_sway_bar_bushing",
}
# Sanity floor only -- well below what the ensemble calls a confident prediction, just
# enough to avoid grounding on a near-coin-flip result when audio_result is provided.
AUDIO_RESULT_CONFIDENCE_FLOOR = 0.5
# Below this (but above the floor), the ensemble's own classification isn't confident
# enough to commit to straight away -- ask one confirmation question first (see
# _generate_answer) instead of silently trusting a coin-flip-ish audio guess. At/above
# it (e.g. the real 84% case that motivated this), skip straight to advice as before.
AUDIO_RESULT_CONFIRM_THRESHOLD = 0.75

# The LLM appends this as the last line of its answer once it has enough conversation
# detail to judge severity (see _BASE_SYSTEM_PROMPT); absent when it still needs more
# info from the user. Parsed out before the answer is shown, never left in the reply.
_SEVERITY_TAG_RE = re.compile(r"<<SEVERITY:(low|medium|high)>>\s*(.*)\s*$", re.DOTALL)

_SEVERITY_LEVEL_LABELS_AR = {"low": "منخفضة", "medium": "متوسطة", "high": "عالية"}

_NO_MATCH_ANSWER = (
    "ما قدرت أحدد العطل بثقة كافية من الوصف يلي كتبته. "
    "ممكن تجرب تسجل صوت العطل وترفعه عبر خاصية تحليل الصوت، "
    "أو تزور ورشة لفحص السيارة مباشرة."
)
_AI_QUOTA_ANSWER = (
    "وصلت خدمة الذكاء الاصطناعي لحدها المسموح مؤقتًا (حصة مجانية). "
    "جرب كمان شوي، أو استخدم خاصية تحليل الصوت بالوقت الحالي."
)
_AI_ERROR_ANSWER = (
    "صار في مشكلة مؤقتة بخدمة الذكاء الاصطناعي وما قدرنا نتأكد من التشخيص. "
    "جرب كمان شوي، أو استخدم خاصية تحليل الصوت بالوقت الحالي."
)

# Used when no category cleared MATCH_THRESHOLD at all. Still routed through the LLM
# (when a key is configured) instead of a single static string, so a follow-up like
# "ليش ما بتقدر؟" gets an actual reply instead of the exact same sentence repeated
# verbatim -- while staying just as strict about never inventing a diagnosis.
_NO_MATCH_SYSTEM_PROMPT = """أنت مساعد GarageAI لتشخيص أعطال السيارات، وبترد بالعامية الفلسطينية/الأردنية.
قاعدة المعرفة عندك حاليًا بتغطي 3 أعطال بس: قشاط الدينمو، البريك، وجلد عمود التوازن (الميزان).

وصف الزبون (أو أي رسالة تالية بنفس المحادثة) ما طابق ولا وحدة من الفئات التلاتة بثقة كافية -- يعني ما عندك تشخيص فعلي تقدمه حاليًا.

بهاي الحالة:
- لا تخترع تشخيص أو تفترض عطل من الثلاثة، حتى لو حسيت فيه تخمين محتمل.
- إذا الزبون سأل ليش ما قدرت تحدد العطل، جاوبه بصراحة وبأسلوب ودود: الوصف يلي كتبه ما كان فيه تفاصيل كافية (نوع الصوت، متى بيصير، وين بالسيارة) تخليك تربطه بوضوح بواحدة من الفئات التلاتة يلي بتعرفها.
- اسأله سؤال أو سؤالين محددين ممكن يوضحوا الصورة أكتر (مثلاً: الصوت صرير ولا طقطقة ولا صفير؟ بيصير وقت الفرملة، وقت الدوران بالمنعطفات، ولا طول الوقت؟).
- ذكّره كمان إنه ممكن يجرب خاصية تحليل الصوت (تسجيل ورفع الصوت) أو يزور ورشة لفحص مباشر، كخيار بديل مش بس أول اقتراح.
- ما تكرر نفس الجملة حرفيًا لو سبق قلتها بهاي المحادثة -- رد على كلام الزبون تحديدًا.

ردك بيتعرض كنص عادي بدون تنسيق -- بدون ماركداون، بدون نجوم، بدون عناوين، بدون قوائم. فقرة أو فقرتين قصار بس. لا تضيف أي علامة <<SEVERITY:...>> أبدًا بهاي الحالة."""

_BASE_SYSTEM_PROMPT = """أنت مساعد GarageAI لتشخيص أعطال السيارات بناءً على وصف الزبون بالعامية الفلسطينية/الأردنية.
قاعدة المعرفة المتوفرة عندك حاليًا بتغطي 3 أعطال بس: الدينمو/قشاط الدينمو، البريك (الفرامل)، وجلد عمود التوازن (الميزان).

مهم جدًا: أول شي تأكد إذا كان وصف الزبون فعلاً يتطابق منطقيًا مع الأسباب أو الأصوات أو الأعراض المذكورة بالسياق تحت.
إذا كان وصف الزبون مش متعلق فعليًا بالسياق المعطى -- مثلاً بيسأل عن المكيف، الزجاج، تجديد الرخصة، تغيير الزيت، أي عطل تاني مش من الثلاثة،
أو سؤال عام/تقني مش وصف عطل أصلاً (زي سؤال عن دقة النظام أو كيف بيشتغل) --
قله بصراحة إنه هذا مش من ضمن اللي تقدر تساعده فيه حاليًا، وانصحه يزور ورشة أو يجرب خاصية تحليل الصوت. لا تفرض تشخيص من الفئات الثلاث إذا مش منطقي.

تنبيه مهم: لو المحادثة كانت أصلاً بلشت وتمحورت حول وحدة من الفئات الثلاث (مثلاً قشاط الدينمو)، وبعدين الزبون سأل عن فئة **تانية من نفس الفئات الثلاث المدعومة** (مثلاً بلش يسأل عن جلد عمود التوازن أو عن البريك) -- هاي مش حالة "مش مدعوم أصلاً". لا تقله إنه هذا مش من ضمن اللي تقدر تساعده فيه. بالعكس، وضحله بصراحة إنه هاد موضوع فعلاً بتقدر تساعده فيه، بس هو عطل مختلف عن اللي كنا نحكي فيه بأول المحادثة، وانصحه يبلش محادثة أو تحليل جديد مخصص لهاد العطل بالذات عشان ما يتلخبط التشخيصين مع بعض.

بهاي الحالتين (مش متعلق فعليًا، أو فئة تانية مدعومة بس مختلفة عن فئة هاي المحادثة)، لازم ردك يبدأ بالضبط بـ <<NO_MATCH>> كأول شي بالرد قبل أي كلمة تانية، وبعدها اكمل ردك العادي بالعامية بنفس السطر أو اللي بعده.
لا تضيف هذا الوسم إذا كان الوصف فعلاً متعلق بنفس فئة هاي المحادثة -- حتى لو لسا ناقصك تفاصيل وبدك تسأل أسئلة متابعة، هاي مش حالة عدم تطابق.

إذا كان الوصف فعلاً متعلق بواحدة من الفئات الثلاث، استخدم فقط المعلومات المعطاة بالسياق (ولا تخترع معلومات إضافية)، وجاوب باللهجة الفلسطينية/الأردنية البسيطة، بأسلوب ودود ومختصر:
(1) العطل الأكثر احتمالاً بناءً على السياق، (2) نصيحة عملية،
(3) تنويه إنه هذا تقدير أولي وينصح بفحص فعلي بالورشة للتأكد.
لا تستخدم عبارات تشخيص قاطعة 100%.

هاي محادثة مستمرة مع الزبون. لو سأل سؤال متابعة (مثلاً عن التكلفة، أو مدة الإصلاح، أو تفاصيل زيادة)، جاوبه بناءً على نفس العطل والسياق يلي حددناه بأول المحادثة -- ما في داعي تعيد التشخيص من الصفر كل مرة.

بعد التشخيص، استخدم سياق تقييم الخطورة المعطى تحت لتحديد مستوى خطورة العطل. إذا لسا ناقصك تفاصيل ضرورية من الزبون (زي أسئلة المتابعة المذكورة بسياق الخطورة)، اسألها بأدب ضمن ردك قبل ما تحكم على الخطورة -- لا تحكم من أول رسالة إذا مفيهاش تفاصيل كافية. لما يصير عندك معلومات كافية فعلاً (من إجابات الزبون أو من وصفه المباشر)، أضف بآخر سطر من ردك -- وبس آخر سطر، وبعد باقي كلامك العادي -- بالضبط هذا الشكل:
<<SEVERITY:low>> سبب قصير بجملة وحدة بالعامية
أو <<SEVERITY:medium>> ... أو <<SEVERITY:high>> ...
لا تضيف هذا السطر أبدًا إذا لسا ناقصك معلومات، ولا تذكر أي رقم أو نسبة إحصائية مش مذكورة صراحة بسياق الخطورة تحت.

مهم: ردك بيتعرض كنص عادي بدون تنسيق، فلا تستخدم ماركداون أبدًا -- بدون نجوم (**) للنص العريض، بدون عناوين (#)، بدون قوائم بنقاط أو أرقام. اكتب فقرات نص عادية بس."""


def _no_match_conversational_answer(messages: List[dict]) -> str:
    if not settings.GEMINI_API_KEY:
        return _NO_MATCH_ANSWER
    text = _call_gemini(_NO_MATCH_SYSTEM_PROMPT, messages)
    if not text:
        return _NO_MATCH_ANSWER
    clean, _, _ = _split_severity_tag(text)  # defensive: prompt says never add the tag here
    return clean


def _no_match_result(messages: List[dict], best_score: float = 0.0) -> Dict:
    return {
        "predicted_class": None,
        "category_label": None,
        "match_score": round(best_score, 3),
        "matched_phrases": [],
        "answer": _no_match_conversational_answer(messages),
        "severity": None,
        "severity_reason": None,
    }


def _first_user_message(messages: List[dict]) -> str:
    for message in messages:
        if message["role"] == "user" and message["content"].strip():
            return message["content"]
    raise ValueError("messages must contain at least one non-empty user message")


def _latest_user_message(messages: List[dict]) -> Optional[str]:
    for message in reversed(messages):
        if message["role"] == "user" and message["content"].strip():
            return message["content"]
    return None


def _split_severity_tag(text: str):
    """Pulls the trailing <<SEVERITY:...>> marker (see _BASE_SYSTEM_PROMPT) out of the
    model's reply so it's never shown to the user, and returns (clean_answer, severity, reason)."""
    match = _SEVERITY_TAG_RE.search(text)
    if not match:
        return text, None, None
    clean = text[: match.start()].strip()
    reason = match.group(2).strip() or None
    return (clean or text), match.group(1), reason


_NO_MATCH_TAG = "<<NO_MATCH>>"


def _split_llm_answer(text: str):
    """Strips a leading <<NO_MATCH>> marker (the model judged the retrieved category
    doesn't actually match the customer's description -- see _BASE_SYSTEM_PROMPT) and a
    trailing <<SEVERITY:...>> marker. Returns (clean_answer, severity, reason, no_match).

    Retrieval alone can't catch this: embedding scores cluster ~0.8-0.94 regardless of
    relevance (see MATCH_THRESHOLD's comment), so a question like "what's a turbo bearing
    failure" or "what does 90% confidence even mean" routinely clears MATCH_THRESHOLD and
    reaches here with some category attached -- only the LLM, reading the actual retrieved
    text, can tell it doesn't really apply. Without this, predicted_class/category_label
    stayed populated in the API response even when the answer itself said "not in scope."
    """
    stripped = text.lstrip()
    no_match = stripped.startswith(_NO_MATCH_TAG)
    if no_match:
        text = stripped[len(_NO_MATCH_TAG):].lstrip()
    clean, severity, reason = _split_severity_tag(text)
    return clean, severity, reason, no_match


def _call_gemini(system_instruction: str, messages: List[dict]) -> str:
    """Calls Gemini with the given system prompt + trimmed conversation history.
    Returns the response text, or one of the honest AI-unavailable fallback strings
    on quota/API/network errors -- never raises, never invents an answer from nothing.
    Assumes settings.GEMINI_API_KEY is set; callers check that first."""
    trimmed = messages[-MAX_HISTORY_MESSAGES:]
    while trimmed and trimmed[0]["role"] != "user":
        trimmed.pop(0)

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    # Gemini uses "model" for the assistant's own turns, not "assistant".
    contents = [
        types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[types.Part.from_text(text=m["content"])],
        )
        for m in trimmed
    ]

    from google.genai import errors as genai_errors

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,  # type: ignore[arg-type]  # google-genai's stub treats list[Content] as invariant; a plain list of Content is valid at runtime
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=2048,
                http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
            ),
        )
    except genai_errors.APIError as exc:
        print(f"[diagnose] Gemini request failed: {exc}")
        if exc.code == 429:
            # Distinct from _NO_MATCH_ANSWER: the AI hit its (free-tier) quota, it did
            # NOT fail to verify relevance -- telling the user "I can't determine your
            # fault" here would be a straight-up lie about what actually went wrong.
            return _AI_QUOTA_ANSWER
        # Regression: this used to fall through to _NO_MATCH_ANSWER for EVERY non-429
        # API error (auth/permission/server errors included), which silently blamed the
        # customer's wording for what was actually a broken Gemini credential/service --
        # a real production case (403 PERMISSION_DENIED on the configured project) proved
        # this indistinguishable from a genuine low-confidence miss. Any API failure here
        # means we never got a real relevance judgment, so say that honestly instead.
        return _AI_ERROR_ANSWER
    except Exception as exc:  # network or unexpected errors
        print(f"[diagnose] Gemini request failed: {exc}")
        return _AI_ERROR_ANSWER

    return (response.text or "").strip()


def _generate_answer(
    messages: List[dict],
    category_label: str,
    context: str,
    severity_ctx: str,
    audio_confidence: Optional[float] = None,
    form_result: Optional[dict] = None,
):
    if not settings.GEMINI_API_KEY:
        # Without the LLM, there is no reliable way to check whether the retrieved category
        # actually matches what the customer described (embedding scores alone can't tell --
        # see the MATCH_THRESHOLD note above). So the fallback must NOT claim a diagnosis --
        # doing so would confidently misdiagnose irrelevant input (e.g. a greeting) as a real
        # fault. Just say generation isn't configured yet.
        return (
            "(الذكاء الاصطناعي مو مفعّل حاليًا على هالسيرفر - GEMINI_API_KEY غير مضبوط)\n"
            "ما فيني أتأكد فعليًا إذا وصفك متعلق بعطل من قاعدة المعرفة عندي من غيره. "
            "لتفعيل تشخيص حقيقي، لازم يتضاف مفتاح Gemini API بإعدادات السيرفر."
        ), None, None, False

    system_instruction = (
        f"{_BASE_SYSTEM_PROMPT}\n\n"
        f"الفئة المرجّحة بناءً على تحليل دلالي لوصف الزبون: {category_label}\n\n"
        f"السياق المسترجع من قاعدة المعرفة:\n{context}\n\n"
        f"سياق تقييم الخطورة لهاي الفئة (استخدمه فقط، ولا تخترع إحصائيات مش موجودة فيه):\n{severity_ctx}"
    )

    if audio_confidence is not None:
        if audio_confidence < AUDIO_RESULT_CONFIRM_THRESHOLD:
            # Confident enough to ground on (cleared AUDIO_RESULT_CONFIDENCE_FLOOR in
            # diagnose_text), but not confident enough to commit to full advice on a
            # classifier guess alone -- verify it against the customer first using the
            # "أسئلة تأكيد التشخيص" bullets already in the context above (mechanically
            # distinguishing: RPM-linked vs pedal-linked vs bump-linked), same way a
            # mechanic would before trusting an ambiguous sound.
            system_instruction += (
                f"\n\nهاي المحادثة بدأت من نتيجة تحليل صوت آلي، بس بثقة متوسطة بس ({round(audio_confidence * 100)}%) -- "
                "مش عالية كفاية تعتمد عليها لحالها. قبل ما تعطي التشخيص والنصيحة الكاملة، "
                "اسأل الزبون سؤال تأكيد واحد بس من قسم 'أسئلة تأكيد التشخيص' بالسياق فوق (يميّز هاي الفئة عن الفئتين التانيتين)، "
                "وانتظر جوابه قبل ما تكمل. إذا جوابه ما بيتوافق مع الفئة المرجّحة، قوله بصراحة إنه احتمال يكون عطل تاني وينصحه يجرب خاصية تحليل الصوت أو يزور ورشة، ولا تفرض هاي الفئة عليه."
            )
        else:
            system_instruction += (
                f"\n\nهاي المحادثة بدأت من نتيجة تحليل صوت آلي بثقة عالية ({round(audio_confidence * 100)}%) -- "
                "اعتمد عليها مباشرة وابدأ بالتشخيص والنصيحة الكاملة من غير ما تسأل سؤال تأكيد إضافي، "
                "ولا تستخدم وسم <<NO_MATCH>> بهاي الحالة -- الفئة جاية من تحليل صوت آلي حقيقي مش من تخمين نصي."
            )
    elif form_result:
        # The customer already ran the deterministic, rule-based severity checklist (no AI
        # involved -- see severityForm.ts) for this exact category before opening this chat.
        # Build on that fixed answer instead of re-deriving (and risking contradicting) it:
        # skip the confirmation-question flow entirely (the category is the one the form was
        # generated for, not a guess) and skip re-judging severity from scratch. The severity
        # value in the API response is set from form_result directly by the caller regardless
        # of what the model does here -- these instructions are just to keep the prose consistent.
        level_ar = _SEVERITY_LEVEL_LABELS_AR.get(form_result.get("level"), form_result.get("level"))
        findings = form_result.get("positive_findings") or []
        findings_line = ("الأعراض يلي أكّدها الزبون بالفورم: " + "، ".join(findings) + ".") if findings else (
            "الزبون ما أكّد أي من العلامات يلي بترفع الخطورة بالفورم."
        )
        system_instruction += (
            f"\n\nهاي المحادثة بدأت من نتيجة فورم تقييم خطورة ثابت (بدون ذكاء اصطناعي، أسئلة نعم/لأ) عبّاه الزبون بنفسه لنفس هاي الفئة. "
            f"نتيجة الفورم: نسبة خطورة {form_result.get('percent')}% (مستوى {level_ar}). {findings_line}\n"
            "اعتمد على هاي النتيجة كما هي بردك -- لا تعيد تقييم الخطورة من الصفر ولا تسأل نفس الأسئلة يلي الزبون جاوب عليها أصلاً بالفورم (أسئلة تأكيد التشخيص أو أسئلة متابعة الخطورة). "
            "ابني ردك (السبب الأرجح والنصيحة العملية) فوق هاي المعلومات مباشرة، واذكر أعراضه المؤكدة إذا كانت مناسبة. "
            "لا تضيف وسم <<SEVERITY:...>> ولا <<NO_MATCH>> أبدًا بهاي الحالة -- الفئة والخطورة معروفين ومؤكدين مسبقًا من الفورم."
        )

    text = _call_gemini(system_instruction, messages)
    if text in (_AI_QUOTA_ANSWER, _AI_ERROR_ANSWER):
        return text, None, None, False
    if not text:
        return _NO_MATCH_ANSWER, None, None, False
    return _split_llm_answer(text)


def _search_best_category(query: str, top_k: int):
    """Runs retrieval for one query string and picks the best-supported category.
    Returns (category_key, best_score, category_label, matched_phrases) or None if the
    embedding store returned nothing at all (degenerate/empty query)."""
    matches = search(query, top_k=top_k)
    if not matches:
        return None

    scores_by_category: Dict[str, List[float]] = defaultdict(list)
    label_by_category: Dict[str, str] = {}
    phrases_by_category: Dict[str, List[str]] = defaultdict(list)

    for chunk, score in matches:
        scores_by_category[chunk.category_key].append(score)
        label_by_category[chunk.category_key] = chunk.category_label
        phrases_by_category[chunk.category_key].append(chunk.text)

    if not scores_by_category:
        return None

    # Pick the category by SUM of its retrieved chunks' scores, not the single highest-
    # scoring chunk: individual chunk scores cluster tightly (see MATCH_THRESHOLD note
    # above) so a genuine match's margin over a coincidental one-chunk near-miss can be
    # as thin as ~0.003 -- well within floating-point noise from BLAS thread/backend
    # differences between processes, let alone real phrasing variation. A category with
    # several relevant chunks in the top-k (the common case for an actual match) reliably
    # outweighs a single unrelated chunk that happened to score marginally higher.
    best_category = max(scores_by_category, key=lambda k: sum(scores_by_category[k]))
    best_score = max(scores_by_category[best_category])

    return best_category, best_score, label_by_category[best_category], phrases_by_category[best_category][:5]


def _build_result(
    messages: List[dict],
    category_key: str,
    category_label: str,
    match_score: float,
    matched_phrases: List[str],
    audio_confidence: Optional[float] = None,
    form_result: Optional[dict] = None,
) -> Dict:
    context = category_context(category_key)
    severity_ctx = severity_context(category_key)
    answer, severity, severity_reason, no_match = _generate_answer(
        messages, category_label, context, severity_ctx, audio_confidence, form_result
    )

    if not no_match and form_result:
        # Always trust the form's own already-computed, deterministic numbers over whatever
        # the LLM did (or didn't) put in a <<SEVERITY:...>> tag -- the prompt above tells it
        # not to emit one here, but the API response must stay correct even if it doesn't
        # comply, since this is the one number in the whole system that doesn't need an LLM
        # to be right.
        severity = form_result.get("level")
        findings = form_result.get("positive_findings") or []
        severity_reason = (
            f"بناءً على فورم التقييم ({form_result.get('percent')}%): " + "، ".join(findings)
            if findings
            else f"بناءً على فورم التقييم: {form_result.get('percent')}%"
        )

    if no_match:
        # Retrieval cleared MATCH_THRESHOLD (embedding scores cluster too tightly for that
        # alone to gate relevance -- see MATCH_THRESHOLD's comment), but the LLM, reading
        # the actual retrieved text, judged this category doesn't really apply. Don't let
        # the API response claim a diagnosis the answer itself just disclaimed.
        return {
            "predicted_class": None,
            "category_label": None,
            "match_score": 0.0,
            "matched_phrases": [],
            "answer": answer,
            "severity": None,
            "severity_reason": None,
        }

    return {
        "predicted_class": category_key,
        "category_label": category_label,
        "match_score": round(match_score, 3),
        "matched_phrases": matched_phrases,
        "answer": answer,
        "severity": severity,
        "severity_reason": severity_reason,
    }


def diagnose_text(
    messages: List[dict],
    top_k: int = 8,
    audio_result: Optional[dict] = None,
    form_result: Optional[dict] = None,
) -> Dict:
    # If the chat was opened from an audio analysis result, ground directly on the
    # ensemble's own classification instead of semantically searching prose that merely
    # reports that result (which routinely scores below MATCH_THRESHOLD -- see
    # AUDIO_LABEL_TO_CATEGORY_KEY). This is strictly more reliable than embedding similarity.
    if audio_result:
        predicted_label = audio_result.get("predicted_class")
        category_key = AUDIO_LABEL_TO_CATEGORY_KEY.get(predicted_label) if isinstance(predicted_label, str) else None
        confidence = audio_result.get("confidence") or 0.0
        if category_key and confidence >= AUDIO_RESULT_CONFIDENCE_FLOOR:
            label = category_labels().get(category_key, category_key)
            return _build_result(
                messages, category_key, label, confidence, [], audio_confidence=confidence, form_result=form_result
            )

    # Same idea for the standalone severity-checklist form (severityForm.ts): it's always
    # generated for a category the audio classifier already named, so ground on it directly
    # too -- no confirmation question needed (see the elif form_result branch in
    # _generate_answer), match_score=1.0 since the category itself isn't in question here,
    # only severity is, and that's already known.
    if form_result:
        predicted_label = form_result.get("predicted_class")
        category_key = AUDIO_LABEL_TO_CATEGORY_KEY.get(predicted_label) if isinstance(predicted_label, str) else None
        if category_key:
            label = category_labels().get(category_key, category_key)
            return _build_result(messages, category_key, label, 1.0, [], form_result=form_result)

    description = _first_user_message(messages)
    result = _search_best_category(description, top_k)

    if result is None or result[1] < MATCH_THRESHOLD:
        # The first message didn't match -- without this, every later reply in this
        # conversation would replay the exact same failing search forever (retrieval
        # always keyed off the first message), even when a later message is a much
        # clearer symptom description. Give the latest message one fair shot before
        # giving up. Deliberately NOT done when the first message already matched
        # (see test_diagnose_text_follow_up_keeps_using_first_message_for_retrieval) --
        # a content-free follow-up like "how much would it cost?" must not be able to
        # steal a category away from an already-successful match.
        latest = _latest_user_message(messages)
        if latest and latest != description:
            retry = _search_best_category(latest, top_k)
            if retry is not None and retry[1] >= MATCH_THRESHOLD:
                result = retry

    if result is None or result[1] < MATCH_THRESHOLD:
        return _no_match_result(messages, result[1] if result else 0.0)

    best_category, best_score, label, phrases = result
    return _build_result(messages, best_category, label, best_score, phrases)
