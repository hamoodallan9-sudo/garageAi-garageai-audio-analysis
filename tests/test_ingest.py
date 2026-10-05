from collections import Counter

from app.services.rag.ingest import category_context, category_labels, parse_glossary, severity_context


def test_parse_glossary_finds_all_three_fault_categories():
    chunks = parse_glossary()
    counts = Counter(c.category_key for c in chunks)

    assert set(counts) == {"01_sway_bar_bushing", "02_serpentine_belt", "03_brake_squeal"}
    # Regression guard: every chunk must belong to a known model class -- an unmapped
    # heading used to raise, and a mis-mapped one used to silently corrupt retrieval.
    assert all(count > 0 for count in counts.values())


def test_parse_glossary_chunks_carry_section_and_text():
    chunks = parse_glossary()

    assert all(c.text.strip() for c in chunks)
    assert all(c.section.strip() for c in chunks)
    assert all(c.category_label.strip() for c in chunks)


def test_category_context_only_includes_that_categorys_bullets():
    context = category_context("03_brake_squeal")

    assert "الفحمات خلصانة" in context  # brake-specific cause
    assert "قشاط الدينمو قاطع" not in context  # belongs to the alternator category


def test_severity_context_covers_all_three_categories_and_stays_separated():
    for category_key in ("01_sway_bar_bushing", "02_serpentine_belt", "03_brake_squeal"):
        assert severity_context(category_key).strip()

    brake_severity = severity_context("03_brake_squeal")
    assert "طحن معدني" in brake_severity  # brake-specific high-severity marker
    assert "تنشن" not in brake_severity  # belongs to the belt/tensioner category


def test_category_labels_covers_all_three_categories():
    labels = category_labels()

    assert set(labels) == {"01_sway_bar_bushing", "02_serpentine_belt", "03_brake_squeal"}
    assert all(label.strip() for label in labels.values())
