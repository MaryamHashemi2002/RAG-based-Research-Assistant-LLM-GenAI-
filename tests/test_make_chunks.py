from src.make_chunks import (
    ENC,
    OVERLAP,
    SIZE,
    build_text,
    fixed_chunks,
    page_at,
    sentence_chunks,
    split_sentences,
)


def test_split_sentences_basic():
    sentences = [s for s, _ in split_sentences("First sentence. Second sentence! Third one?")]
    assert sentences == ["First sentence.", "Second sentence!", "Third one?"]


def test_split_sentences_ignores_lowercase_after_period():
    sentences = split_sentences("This is a test. and this continues.")
    assert len(sentences) == 1


def test_build_text_tracks_page_starts():
    pages = [{"page": 1, "text": "abc"}, {"page": 2, "text": "def"}]
    text, starts = build_text(pages)
    assert text == "abc def"
    assert starts == [0, 4]


def test_page_at_finds_correct_page():
    pages = [{"page": 1}, {"page": 2}, {"page": 3}]
    starts = [0, 100, 200]
    assert page_at(0, starts, pages) == 1
    assert page_at(150, starts, pages) == 2
    assert page_at(100, starts, pages) == 2
    assert page_at(999, starts, pages) == 3


def test_fixed_chunks_respect_size_and_move_forward():
    chunks = fixed_chunks("word " * 1000)
    assert len(chunks) > 1
    offsets = [offset for _, offset in chunks]
    assert offsets == sorted(offsets)
    for chunk, _ in chunks:
        assert len(ENC.encode(chunk)) <= SIZE


def make_sentences(n):
    return " ".join(f"This is sentence number {i}." for i in range(n))


def test_sentence_chunks_respect_size_and_end_cleanly():
    chunks = sentence_chunks(make_sentences(200))
    assert len(chunks) > 1
    for chunk, _ in chunks:
        assert len(ENC.encode(chunk)) <= SIZE
        assert chunk.startswith("This")
        assert chunk.endswith(".")


def test_sentence_chunks_overlap_neighbours():
    chunks = sentence_chunks(make_sentences(200))
    first, second = chunks[0][0], chunks[1][0]
    last_sentence_of_first = first.split(". ")[-1]
    assert last_sentence_of_first in second
    assert OVERLAP > 0


def test_sentence_chunks_lose_no_sentences():
    chunks = sentence_chunks(make_sentences(200))
    for i in range(200):
        assert any(f"sentence number {i}." in chunk for chunk, _ in chunks)