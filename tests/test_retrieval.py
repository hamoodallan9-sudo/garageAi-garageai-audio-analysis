"""
Regression tests for semantic retrieval quality.

These are the exact cases that exposed real bugs during manual testing:
paraphrase-multilingual-MiniLM ranked an unrelated alternator bullet above the
near-verbatim sway-bar match for the "knocking over bumps" query. Swapping to
intfloat/multilingual-e5-small (with query:/passage: prefixes) fixed it -- these
tests pin that behavior down so a future model/prefix change can't silently
regress it. Loads the real embedding model, so this module is slower than the
rest of the suite (a few seconds, once per test session).
"""

from app.services.rag.store import search


def _top_category(query: str) -> str:
    chunk, _score = search(query, top_k=1)[0]
    return chunk.category_key


def test_belt_whistle_description_matches_alternator_belt():
    query = "في صوت صفير من قدام السيارة وقت التشغيل، وبزيد لما بسرع"
    assert _top_category(query) == "02_serpentine_belt"


def test_brake_squeal_description_matches_brake_category():
    query = "البريك عم يصرخ كل ما بفرمل، وحاسس الدواسة طرية"
    assert _top_category(query) == "03_brake_squeal"


def test_knocking_over_bumps_matches_sway_bar_not_alternator():
    # This is the case that was misclassified as the alternator category before
    # switching embedding models -- keep it pinned on its own.
    query = "في صوت خبط من تحت السيارة كل ما بفوت عالمطب"
    assert _top_category(query) == "01_sway_bar_bushing"
