from __future__ import annotations

from semantic_chunker import ChunkerConfig, chunk_document, chunk_documents
from document import Document, DocumentPage

PL = "Restated Consolidated Statement of Profit and Loss"
PL_HEAD = ["(₹ in million)", "Particulars", "Sep 30, 2025", "Mar 31, 2025", "Mar 31, 2024", "Mar 31, 2023"]
PL_ROWS_A = [
    "Revenue from operations", "869.18", "1,631.06", "1,537.61", "967.96",
    "Other income 25.09 6.38 14.19 2.32",
]
PL_ROWS_B = [
    "Total income 894.27 1,637.44 1,551.80 970.28",
    "Profit Before Tax 111.59 199.09 125.55 72.50",
]


def mk(pages, doc_id="DOC1", method=None):
    """pages: list of (text, section, subsection)."""
    ps = []
    for i, (text, sec, sub) in enumerate(pages, start=1):
        ps.append(DocumentPage(
            document_id=doc_id, page_number=i, text=text, section=sec, subsection=sub,
            extraction_method=(method or {}).get(i, "pymupdf"), character_count=len(text),
        ))
    return Document(document_id=doc_id, company_name="X Ltd", document_type="RHP",
                    source_file="x.pdf", page_count=len(ps), pages=ps)


def check_invariants(chunks):
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert c.pages == sorted(set(c.pages))
        assert c.page_start == c.pages[0] and c.page_end == c.pages[-1]
        assert c.content.strip()


SEC = "SUMMARY OF FINANCIAL INFORMATION"


# ---------------- required cases
def test_paragraph_subsection():
    txt = ("Our Company is engaged in the manufacture of parenteral products.\n"
           "We operate two manufacturing facilities.\n\n"
           "We sell to institutional and private customers across India.")
    chunks = chunk_document(mk([(txt, "OUR BUSINESS", "Overview")]))
    assert len(chunks) == 1
    c = chunks[0]
    assert (c.section, c.subsection, c.pages) == ("OUR BUSINESS", "Overview", [1])
    assert c.content.startswith("OUR BUSINESS\nOverview\n")
    assert "two manufacturing facilities" in c.content
    check_invariants(chunks)


def test_table_single_page():
    txt = "\n".join([PL] + PL_HEAD + PL_ROWS_A + PL_ROWS_B)
    chunks = chunk_document(mk([(txt, SEC, PL)]))
    assert len(chunks) == 1
    c = chunks[0].content
    assert f"Table title: {PL} (₹ in million)" in c
    assert "Columns: Particulars | Sep 30, 2025 | Mar 31, 2025 | Mar 31, 2024 | Mar 31, 2023" in c
    assert "Revenue from operations | 869.18 | 1,631.06 | 1,537.61 | 967.96" in c
    assert "Other income | 25.09 | 6.38 | 14.19 | 2.32" in c
    assert "Profit Before Tax | 111.59 | 199.09 | 125.55 | 72.50" in c
    assert c.index("Columns:") < c.index("Revenue from operations")
    check_invariants(chunks)


def test_table_spanning_pages():
    p3 = "\n".join([PL] + PL_HEAD + PL_ROWS_A)
    p4 = "\n".join(PL_HEAD[1:] + PL_ROWS_B)           # column headings repeated on continuation page
    chunks = chunk_document(mk([("Intro text.", "OTHER", None), (p3, SEC, PL), (p4, SEC, PL)]))
    tbl = [c for c in chunks if "Table title" in c.content]
    assert len(tbl) == 1
    c = tbl[0]
    assert (c.page_start, c.page_end, c.pages) == (2, 3, [2, 3])
    assert c.content.count("Columns:") == 1
    assert c.content.count("Sep 30, 2025") == 1      # repeated heading not duplicated
    assert "Profit Before Tax | 111.59" in c.content and "Revenue from operations | 869.18" in c.content
    check_invariants(chunks)


def test_long_subsection_multiple_chunks():
    sents = [f"The Company completed expansion project number {i} during the year and commissioned new capacity." for i in range(60)]
    para = " ".join(sents)
    cfg = ChunkerConfig(max_chars=700, min_chars=200)
    chunks = chunk_document(mk([(para[: len(para) // 2], SEC, "Expansion"), (para[len(para) // 2:], SEC, "Expansion")]), cfg)
    assert len(chunks) > 3
    for c in chunks:
        assert len(c.content) <= 700 + 60
        assert c.content.startswith(SEC)             # context kept in every chunk
        assert c.content.rstrip().endswith(".") or c.content.rstrip().endswith("capacity")
    check_invariants(chunks)


def test_section_boundary():
    chunks = chunk_document(mk([
        ("Risk text about demand for our products.", "RISK FACTORS", "Internal Risks"),
        ("The Offer comprises a fresh issue of equity shares.", "THE OFFER", None),
    ]))
    assert [(c.section, c.pages) for c in chunks] == [("RISK FACTORS", [1]), ("THE OFFER", [2])]
    assert "fresh issue" not in chunks[0].content and "demand" not in chunks[1].content


def test_table_qa_year_value():
    txt = "\n".join([PL] + PL_HEAD + PL_ROWS_A + PL_ROWS_B)
    c = chunk_document(mk([(txt, SEC, PL)]))[0].content
    cols = next(l for l in c.splitlines() if l.startswith("Columns:")).removeprefix("Columns:").split("|")
    cols = [x.strip() for x in cols]
    row = next(l for l in c.splitlines() if l.startswith("Revenue from operations")).split("|")
    row = [x.strip() for x in row]
    # the value is read straight off the chunk: label + header position, no calculation
    assert row[cols.index("Mar 31, 2024")] == "1,537.61"


# ---------------- additional cases
def test_repeated_headers_footers_removed():
    pages = []
    for i in range(6):
        topics = ["Pune plant", "export markets", "raw materials", "quality systems", "logistics", "employees"]
        body = "\n".join(f"Paragraph {i} covers {topics[i]} aspect {j} of our operations in detail." for j in range(4))
        pages.append((f"ABC Limited - Red Herring Prospectus\n{body}\nPage {45 + i} of 400\n{45 + i}", "BUSINESS", "Overview"))
    chunks = chunk_document(mk(pages))
    joined = "\n".join(c.content for c in chunks)
    assert "Red Herring Prospectus" not in joined and "of 400" not in joined
    assert "\n45\n" not in joined and not joined.rstrip().endswith("50")
    for i in range(6):
        assert f"Paragraph {i} covers" in joined
    assert [p for c in chunks for p in c.pages] and set(p for c in chunks for p in c.pages) == set(range(1, 7))


def test_numeric_table_value_not_mistaken_for_page_number():
    txt = "\n".join([PL] + PL_HEAD + ["Revenue from operations", "869.18", "1,631.06", "1,537.61", "967.96",
                                    "Number of employees", "72"])
    c = chunk_document(mk([(txt, SEC, PL)]))[0].content
    assert "Number of employees | 72" in c


def test_heading_context_preserved():
    chunks = chunk_document(mk([("Revenue from operations increased from 1,537.61 to 1,631.06.", SEC, None)]))
    assert chunks[0].content.startswith(SEC + "\n")
    # heading already in content -> not duplicated
    chunks = chunk_document(mk([(SEC + "\nRevenue from operations increased.", SEC, None)]))
    assert chunks[0].content.count(SEC) == 1


def test_ocr_page_preserved():
    ocr = "Revenue from operati0ns was 1,63l.06 m1llion"
    chunks = chunk_document(mk([("Normal text.", "BUSINESS", "Overview"), (ocr, "BUSINESS", "Overview")],
                               method={2: "ocr"}))
    assert len(chunks) == 1
    assert ocr in chunks[0].content                  # no correction
    assert chunks[0].pages == [1, 2] and chunks[0].section == "BUSINESS" and chunks[0].subsection == "Overview"


def test_oversized_table_split_repeats_headings():
    rows = []
    names = [chr(97 + i // 26) + chr(97 + i % 26) for i in range(70)]
    for i in range(70):
        rows.append(f"Line item {names[i]} {i}.10 {i}.20 {i}.30 {i}.40")
    p1 = "\n".join([PL] + PL_HEAD + rows[:25])
    p2 = "\n".join(rows[25:50])
    p3 = "\n".join(rows[50:])
    cfg = ChunkerConfig(max_chars=900, min_chars=200)
    chunks = chunk_document(mk([(p1, SEC, PL), (p2, SEC, PL), (p3, SEC, PL)]), cfg)
    assert len(chunks) > 2
    for c in chunks:
        assert f"Table title: {PL} (₹ in million)" in c.content
        assert "Columns: Particulars | Sep 30, 2025" in c.content
        assert "Line item" in c.content
    all_rows = "\n".join(c.content for c in chunks)
    for i in range(70):
        assert f"Line item {names[i]} | {i}.10 | {i}.20 | {i}.30 | {i}.40" in all_rows
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 3
    assert any(c.pages == [2] for c in chunks) or any(c.page_start >= 2 for c in chunks)
    check_invariants(chunks)


def test_table_with_explanatory_context():
    txt = "\n".join([
        "Unrelated paragraph about our marketing strategy.",
        "",
        "The following table presents the restated financial information:",
        PL, *PL_HEAD, *PL_ROWS_A, *PL_ROWS_B,
        "",
        "The figures should be read in conjunction with the notes to the restated financial statements.",
        "",
        "Another unrelated paragraph about dividend policy.",
    ])
    chunks = chunk_document(mk([(txt, SEC, PL)]))
    tbl = next(c for c in chunks if "Table title" in c.content)
    assert "The following table presents" in tbl.content
    assert "should be read in conjunction" in tbl.content
    assert "marketing strategy" not in tbl.content and "dividend policy" not in tbl.content
    check_invariants(chunks)


def test_chunk_index_independent_per_document():
    a = mk([("Doc A text.", "S", None)], doc_id="A")
    b = mk([("Doc B text one.\n\nDoc B text two.", "S", None)], doc_id="B")
    chunks = chunk_documents([a, b])
    assert [(c.document_id, c.chunk_index) for c in chunks] == [("A", 0), ("B", 0)]
    assert chunks[0].chunk_id != chunks[1].chunk_id


def test_heading_at_page_bottom_goes_with_next_content():
    p1 = "We have several subsidiaries.\nCORPORATE STRUCTURE"
    p2 = "Our subsidiaries are listed below with their principal activities."
    chunks = chunk_document(mk([(p1, "BUSINESS", "Overview"), (p2, "BUSINESS", "Structure")]))
    assert any("CORPORATE STRUCTURE" in c.content and "principal activities" in c.content for c in chunks)
    assert not any(c.content.rstrip().endswith("CORPORATE STRUCTURE") for c in chunks)
