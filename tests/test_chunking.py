import pytest

from app.ingestion.chunking.splitter import chunk_text


def test_empty_and_whitespace_text_gives_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_is_a_single_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_paragraphs_are_packed_up_to_chunk_size():
    paragraphs = ["a" * 40, "b" * 40, "c" * 40]
    chunks = chunk_text("\n\n".join(paragraphs), chunk_size=100)
    assert len(chunks) == 2
    assert "a" * 40 in chunks[0] and "b" * 40 in chunks[0]
    assert "c" * 40 in chunks[1]


def test_no_text_is_lost():
    paragraphs = [f"paragraph {i} " + "x" * 30 for i in range(10)]
    chunks = chunk_text("\n\n".join(paragraphs), chunk_size=120)
    joined = "\n".join(chunks)
    assert all(p in joined for p in paragraphs)


def test_no_empty_chunks():
    text = "\n\n\n\n".join(["one", "", "two"])
    assert all(c.strip() for c in chunk_text(text, chunk_size=5))


@pytest.mark.xfail(
    reason="Known bug: a paragraph longer than chunk_size is never split, "
    "e.g. YAML or code blocks with no blank lines",
    strict=True,
)
def test_oversize_paragraph_is_split():
    chunks = chunk_text("y" * 5000, chunk_size=1500)
    assert all(len(c) <= 1500 for c in chunks)
