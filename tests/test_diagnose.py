import pytest

from app.services.rag import diagnose
from app.services.rag.diagnose import _first_user_message, _split_severity_tag, diagnose_text


@pytest.fixture(autouse=True)
def no_gemini_key(monkeypatch):
    """Every test in this module runs without a real Gemini key so it never
    makes a network call -- diagnose_text must degrade to the honest fallback
    instead of crashing or (worse) inventing a diagnosis."""
    monkeypatch.setattr(diagnose.settings, "GEMINI_API_KEY", "")


def test_first_user_message_skips_blank_and_non_user_entries():
    messages = [
        {"role": "user", "content": "   "},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "في صوت صفير من قدام"},
    ]
    assert _first_user_message(messages) == "في صوت صفير من قدام"


def test_first_user_message_raises_when_no_real_user_content():
    with pytest.raises(ValueError):
        _first_user_message([{"role": "user", "content": "   "}])


def test_diagnose_text_without_api_key_does_not_claim_a_diagnosis():
    # Regression test: the fallback used to say "العطل الأقرب لوصفك: <category>"
    # for ANY input (even "مرحبا") because it trusted the retrieval score alone.
    # It must now admit it can't verify relevance instead of guessing.
    result = diagnose_text([{"role": "user", "content": "مرحبا"}])

    assert "GEMINI_API_KEY" in result["answer"]
    assert result["category_label"] not in result["answer"]


def test_diagnose_text_first_turn_identifies_correct_category():
    messages = [{"role": "user", "content": "البريك عم يصرخ كل ما بفرمل"}]
    result = diagnose_text(messages)

    assert result["predicted_class"] == "03_brake_squeal"
    assert result["match_score"] > 0


def test_diagnose_text_picks_category_with_more_supporting_evidence_not_one_lucky_chunk():
    # Regression test: this query's single highest-scoring chunk was a brake bullet
    # (score ~0.917) that narrowly out-scored the best sway-bar bullet (~0.914) --
    # a margin thin enough to flip from ordinary floating-point noise between runs.
    # Sway had 4 relevant chunks in the top-8 vs brake's 1, so summing per-category
    # evidence (not trusting the single best chunk) must pick sway-bar correctly.
    messages = [{"role": "user", "content": "في صوت نقر وطقطقة من تحت السيارة وقت المطبات، خصوصا بالمنعطفات"}]
    result = diagnose_text(messages)

    assert result["predicted_class"] == "01_sway_bar_bushing"


def test_diagnose_text_follow_up_keeps_using_first_message_for_retrieval():
    # A follow-up with no symptom info ("how much would it cost?") must not
    # reset or confuse the category -- retrieval always keys off the first
    # user message, not the latest one.
    messages = [
        {"role": "user", "content": "البريك عم يصرخ كل ما بفرمل"},
        {"role": "assistant", "content": "يبدو إنه الفحمات خلصانة."},
        {"role": "user", "content": "طيب وكم بده يكلف تبديلها؟"},
    ]
    result = diagnose_text(messages)

    assert result["predicted_class"] == "03_brake_squeal"


def test_diagnose_text_recovers_when_a_later_message_actually_matches(monkeypatch):
    # Regression test for the "dead conversation" bug: once the first message failed
    # to clear MATCH_THRESHOLD, every later reply replayed the exact same no-match
    # answer forever, because retrieval only ever looked at the first message.
    #
    # Real embedding scores in this domain cluster ~0.78-0.94 regardless of content
    # (see MATCH_THRESHOLD's own comment in diagnose.py) -- confirmed empirically here
    # too, nothing short-of-empty text reliably scores under 0.35. So a genuine
    # below-threshold first message can't be reproduced with real text; this stubs
    # _search_best_category to force that condition and exercise the fallback directly.
    first_message = "هاي رسالة أولى ما رح تعطي نتيجة حقيقية لأنها متل ستَب"
    real_search = diagnose._search_best_category

    def stubbed_search(query, top_k):
        if query == first_message:
            return "01_sway_bar_bushing", 0.1, "جلود عمود التوازن / الميزان", []
        return real_search(query, top_k)

    monkeypatch.setattr(diagnose, "_search_best_category", stubbed_search)

    messages = [
        {"role": "user", "content": first_message},
        {"role": "assistant", "content": "ما قدرت أحدد العطل بثقة كافية من الوصف يلي كتبته."},
        {"role": "user", "content": "البريك عم يصرخ كل ما بفرمل"},
    ]
    result = diagnose_text(messages)

    assert result["predicted_class"] == "03_brake_squeal"


def test_diagnose_text_still_no_match_when_no_message_matches(monkeypatch):
    monkeypatch.setattr(
        diagnose,
        "_search_best_category",
        lambda query, top_k: ("03_brake_squeal", 0.1, "البريك / الفرامل", []),
    )

    messages = [
        {"role": "user", "content": "مرحبا"},
        {"role": "assistant", "content": "أهلين، كيف فيني ساعدك؟"},
        {"role": "user", "content": "شو أخبارك اليوم؟"},
    ]
    result = diagnose_text(messages)

    assert result["predicted_class"] is None


def test_diagnose_text_no_match_falls_back_to_static_answer_when_no_api_key(monkeypatch):
    monkeypatch.setattr(diagnose, "_search_best_category", lambda query, top_k: None)

    result = diagnose_text([{"role": "user", "content": "مرحبا"}])

    assert result["predicted_class"] is None
    assert result["answer"] == diagnose._NO_MATCH_ANSWER


def test_diagnose_text_no_match_uses_gemini_for_a_conversational_reply(monkeypatch):
    # Regression test for a real screenshot: a no-match conversation used to repeat the
    # exact same canned sentence forever, even for a legitimate follow-up like "why can't
    # you tell?" -- it must now go through the LLM instead of a single static string,
    # while still never claiming a diagnosis (predicted_class stays None).
    captured = _install_fake_gemini_client(monkeypatch)
    monkeypatch.setattr(diagnose, "_search_best_category", lambda query, top_k: None)

    messages = [
        {"role": "user", "content": "مرحبا"},
        {"role": "assistant", "content": diagnose._NO_MATCH_ANSWER},
        {"role": "user", "content": "ليش ما بتقدر تحدد؟"},
    ]
    result = diagnose_text(messages)

    assert result["predicted_class"] is None
    assert result["answer"] == "جواب وهمي"  # from _FakeGeminiResponse, not the static string
    assert "لا تخترع تشخيص" in captured["system_instruction"]


def test_diagnose_text_grounds_directly_on_a_confident_audio_result():
    # Regression test: this is the exact real-world case that used to break -- a chat
    # seeded from the audio classifier's own result ("Belt, 84% confidence") is prose
    # ABOUT a result, not a symptom description, so semantic search alone can't reliably
    # match it. audio_result must bypass retrieval entirely and ground on the ensemble's
    # own answer.
    messages = [
        {
            "role": "user",
            "content": 'طلع معي من تحليل الصوت إنه في احتمال "Belt" (نسبة ثقة 84%). شو ممكن يكون السبب وشو الحل المناسب؟',
        }
    ]
    result = diagnose_text(messages, audio_result={"predicted_class": "belt", "confidence": 0.84})

    assert result["predicted_class"] == "02_serpentine_belt"
    assert result["match_score"] == 0.84


def test_diagnose_text_ignores_a_low_confidence_audio_result():
    messages = [{"role": "user", "content": "البريك عم يصرخ كل ما بفرمل"}]
    result = diagnose_text(messages, audio_result={"predicted_class": "belt", "confidence": 0.3})

    # Confidence below the sanity floor -- falls back to real retrieval on the message,
    # which correctly finds brake, not the low-confidence "belt" guess.
    assert result["predicted_class"] == "03_brake_squeal"


def test_diagnose_text_grounds_on_and_uses_a_severity_form_result():
    # The severity form (severityForm.ts) is deterministic and client-side -- it must keep
    # working, including its own severity number showing up correctly in the API response,
    # even with no Gemini key at all (this test uses the default no_gemini_key fixture).
    messages = [{"role": "user", "content": "طلع معي من فورم تقييم الخطورة نتيجة"}]
    form_result = {
        "predicted_class": "brake",
        "percent": 80,
        "level": "high",
        "positive_findings": ["الصوت طحن معدني", "الدواسة حاسسها مختلفة"],
    }
    result = diagnose_text(messages, form_result=form_result)

    assert result["predicted_class"] == "03_brake_squeal"
    assert result["match_score"] == 1.0
    assert result["severity"] == "high"
    assert "طحن معدني" in result["severity_reason"]
    assert "80%" in result["severity_reason"]


def test_diagnose_text_form_result_severity_survives_even_if_llm_tries_to_override_it(monkeypatch):
    # The prompt tells the LLM never to emit its own <<SEVERITY:...>> tag when form_result
    # is present, but the API response must stay correct even if it ignores that -- the
    # form's number is the one source of truth here, not the LLM's.
    _install_fake_gemini_client(monkeypatch)
    monkeypatch.setattr(_FakeGeminiResponse, "text", "بعض النصيحة.\n<<SEVERITY:low>> تجاهل هاد")

    messages = [{"role": "user", "content": "شو رأيك بنتيجة الفورم؟"}]
    form_result = {"predicted_class": "sway", "percent": 20, "level": "low", "positive_findings": []}
    result = diagnose_text(messages, form_result=form_result)

    assert result["severity"] == "low"  # happens to agree here, but came from form_result, not the tag
    assert "20%" in result["severity_reason"]


def test_diagnose_text_form_result_prompt_skips_confirmation_and_severity_reasking(monkeypatch):
    captured = _install_fake_gemini_client(monkeypatch)

    messages = [{"role": "user", "content": "شو رأيك بنتيجة الفورم؟"}]
    form_result = {
        "predicted_class": "belt",
        "percent": 60,
        "level": "medium",
        "positive_findings": ["البكرة بتهتز"],
    }
    diagnose_text(messages, form_result=form_result)

    assert "فورم تقييم خطورة ثابت" in captured["system_instruction"]
    assert "البكرة بتهتز" in captured["system_instruction"]
    assert "اسأل الزبون سؤال تأكيد واحد" not in captured["system_instruction"]


def test_split_severity_tag_extracts_marker_and_strips_it_from_the_answer():
    text = "يبدو إنه الفحمات خلصانة، بنصحك تفحصها بأقرب وقت.\n<<SEVERITY:high>> صوت طحن معدني مع تغيّر بإحساس الدواسة"
    clean, severity, reason = _split_severity_tag(text)

    assert clean == "يبدو إنه الفحمات خلصانة، بنصحك تفحصها بأقرب وقت."
    assert severity == "high"
    assert reason == "صوت طحن معدني مع تغيّر بإحساس الدواسة"


def test_split_severity_tag_is_a_noop_when_no_marker_present():
    text = "ممكن توضحلي أكتر، الصوت عم يصير طول الوقت ولا بس أحيانًا؟"
    clean, severity, reason = _split_severity_tag(text)

    assert clean == text
    assert severity is None
    assert reason is None


class _FakeGeminiResponse:
    text = "جواب وهمي"


class _FakeGeminiModels:
    def __init__(self, captured):
        self._captured = captured

    def generate_content(self, model, contents, config):
        self._captured["system_instruction"] = config.system_instruction
        return _FakeGeminiResponse()


def _install_fake_gemini_client(monkeypatch):
    """Points google.genai.Client at a stub that just records the system_instruction
    it was called with, so we can assert on prompt content without a real API call."""
    monkeypatch.setattr(diagnose.settings, "GEMINI_API_KEY", "fake-key-for-tests")
    captured: dict = {}

    import google.genai as genai_module

    class FakeClient:
        def __init__(self, api_key):
            self.models = _FakeGeminiModels(captured)

    monkeypatch.setattr(genai_module, "Client", FakeClient)
    return captured


def test_diagnose_text_asks_for_confirmation_when_audio_confidence_is_medium(monkeypatch):
    captured = _install_fake_gemini_client(monkeypatch)

    messages = [
        {
            "role": "user",
            "content": 'طلع معي من تحليل الصوت إنه في احتمال "Belt" (نسبة ثقة 60%).',
        }
    ]
    diagnose_text(messages, audio_result={"predicted_class": "belt", "confidence": 0.6})

    assert "اسأل الزبون سؤال تأكيد واحد" in captured["system_instruction"]
    assert "60%" in captured["system_instruction"]


def test_diagnose_text_skips_confirmation_when_audio_confidence_is_high(monkeypatch):
    captured = _install_fake_gemini_client(monkeypatch)

    # Real-world case from the screenshot that motivated this whole fix: 84% confidence
    # must go straight to full advice, never pause on a confirmation question.
    messages = [{"role": "user", "content": 'طلع معي من تحليل الصوت إنه في احتمال "Belt" (84%).'}]
    diagnose_text(messages, audio_result={"predicted_class": "belt", "confidence": 0.84})

    assert "اعتمد عليها مباشرة" in captured["system_instruction"]
    assert "اسأل الزبون سؤال تأكيد واحد" not in captured["system_instruction"]


def test_split_llm_answer_strips_leading_no_match_tag():
    text = "<<NO_MATCH>>\nهذا مش من ضمن الأعطال يلي بقدر أساعدك فيها حاليًا."

    clean, severity, reason, no_match = diagnose._split_llm_answer(text)

    assert no_match is True
    assert clean == "هذا مش من ضمن الأعطال يلي بقدر أساعدك فيها حاليًا."
    assert severity is None
    assert reason is None


def test_split_llm_answer_is_a_noop_when_no_tag_present():
    text = "يبدو إنه الفحمات خلصانة، بنصحك تفحصها بأقرب وقت."

    clean, severity, reason, no_match = diagnose._split_llm_answer(text)

    assert no_match is False
    assert clean == text


def test_diagnose_text_nulls_predicted_class_when_llm_flags_no_match_despite_retrieval_hit(monkeypatch):
    # Regression test for a real gap found via live testing: a question like "what's a
    # turbo bearing failure" routinely clears MATCH_THRESHOLD (embedding scores cluster
    # too tightly to gate relevance -- see MATCH_THRESHOLD's comment) and gets attached to
    # some category, even though the model's own answer correctly says it's out of scope.
    # Before this fix, predicted_class/category_label stayed populated with that spurious
    # category despite the answer disclaiming it -- any consumer trusting the structured
    # fields (not just displaying the free-text answer) would see a false diagnosis.
    _install_fake_gemini_client(monkeypatch)
    monkeypatch.setattr(
        _FakeGeminiResponse, "text",
        "<<NO_MATCH>>\nهذا العطل مش من ضمن اللي بقدر أشخصه حاليًا، بنصحك تزور ورشة.",
    )
    monkeypatch.setattr(
        diagnose, "_search_best_category",
        lambda query, top_k: ("03_brake_squeal", 0.88, "البريك / الفرامل", ["صوت طحن"]),
    )

    messages = [{"role": "user", "content": "احكيلي عن عطل اسمه turbo bearing failure"}]
    result = diagnose_text(messages)

    assert result["predicted_class"] is None
    assert result["category_label"] is None
    assert result["match_score"] == 0.0
    assert result["matched_phrases"] == []
    assert "<<NO_MATCH>>" not in result["answer"]
    assert "هذا العطل مش من ضمن اللي بقدر أشخصه" in result["answer"]


def test_diagnose_text_keeps_diagnosis_when_llm_does_not_flag_no_match(monkeypatch):
    # Sanity counterpart: a real match must NOT be nulled out just because this
    # mechanism exists -- _FakeGeminiResponse's default text has no <<NO_MATCH>> tag.
    _install_fake_gemini_client(monkeypatch)
    monkeypatch.setattr(
        diagnose, "_search_best_category",
        lambda query, top_k: ("03_brake_squeal", 0.9, "البريك / الفرامل", ["صوت طحن"]),
    )

    messages = [{"role": "user", "content": "البريك عم يصرخ كل ما بفرمل"}]
    result = diagnose_text(messages)

    assert result["predicted_class"] == "03_brake_squeal"
    assert result["match_score"] == 0.9
