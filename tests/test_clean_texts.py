from src.clean_text import cut_references, fix_text, remove_repeated_lines, split_pages


def test_fix_text_joins_hyphenated_line_breaks():
    assert fix_text("classi-\nfication works") == "classification works"


def test_fix_text_normalizes_ligatures():
    assert fix_text("\ufb01nd the \ufb02ow") == "find the flow"


def test_fix_text_turns_line_breaks_into_spaces():
    assert fix_text("line one\nline two") == "line one line two"


def test_fix_text_collapses_repeated_spaces():
    assert fix_text("a    b") == "a b"


def test_split_pages_returns_numbers_and_text():
    raw = "[Page 1]\nHello\n\n[Page 2]\nWorld"
    pages = split_pages(raw)
    assert [number for number, _ in pages] == [1, 2]
    assert pages[0][1].strip() == "Hello"
    assert pages[1][1].strip() == "World"


def test_remove_repeated_lines_drops_headers_and_page_numbers():
    pages = [(i, f"Journal Header\nContent number {i}\n{i}") for i in range(1, 6)]
    cleaned = remove_repeated_lines(pages)
    for number, text in cleaned:
        assert text == f"Content number {number}"


def test_cut_references_removes_everything_after_heading():
    pages = [(i, f"Body text {i}") for i in range(1, 5)]
    pages.append((5, "Conclusion text\nReferences\n[1] Some cited paper"))
    pages.append((6, "[2] Another cited paper"))
    result, was_cut = cut_references(pages)
    all_text = " ".join(text for _, text in result)
    assert was_cut is True
    assert "Conclusion text" in all_text
    assert "Some cited paper" not in all_text
    assert "Another cited paper" not in all_text


def test_cut_references_ignores_heading_in_first_half():
    pages = [(1, "References\nsome early text")] + [(i, f"Body {i}") for i in range(2, 7)]
    result, was_cut = cut_references(pages)
    assert was_cut is False
    assert len(result) == 6