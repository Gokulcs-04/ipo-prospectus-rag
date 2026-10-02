from pathlib import Path

import pymupdf
import pytest

from ipo_analyzer.ingestion.document_processor import process_document
from ipo_analyzer.ingestion.ocr import ocr_page
from ipo_analyzer.ingestion.pdf_parser import parse_pdf
from ipo_analyzer.ingestion.section_detector import detect_sections
from ipo_analyzer.schemas.document import Document, DocumentPage


DOCUMENT_ID = "test_rhp"
COMPANY_NAME = "Test Industrial Gases Limited"
DOCUMENT_TYPE = "RHP"


def create_test_pdf(path: Path) -> Path:
    """Create a small synthetic 3-page PDF for unit testing."""
    pdf = pymupdf.open()

    pages = [
        (
            "SECTION I: GENERAL\n"
            "DEFINITIONS AND ABBREVIATIONS\n"
            "This is a test prospectus document.\n"
            "The document contains general information."
        ),
        (
            "SECTION II: RISK FACTORS\n"
            "Investors should consider the information disclosed in this section."
        ),
        (
            "This is a continuation page containing ordinary prospectus text.\n"
            "No new section begins on this page."
        ),
    ]

    for text in pages:
        page = pdf.new_page()
        page.insert_text((72, 72), text)

    pdf.save(str(path))
    pdf.close()

    return path


def test_parse_pdf_returns_document(tmp_path):
    pdf_path = create_test_pdf(tmp_path / "test.pdf")

    document = parse_pdf(
        pdf_path,
        document_id=DOCUMENT_ID,
        company_name=COMPANY_NAME,
        document_type=DOCUMENT_TYPE,
    )

    assert document.document_id == DOCUMENT_ID
    assert document.company_name == COMPANY_NAME
    assert document.document_type == DOCUMENT_TYPE
    assert document.page_count == 3
    assert len(document.pages) == 3


def test_parse_pdf_page_metadata(tmp_path):
    pdf_path = create_test_pdf(tmp_path / "test.pdf")

    document = parse_pdf(
        pdf_path,
        document_id=DOCUMENT_ID,
        company_name=COMPANY_NAME,
        document_type=DOCUMENT_TYPE,
    )

    assert [page.page_number for page in document.pages] == [1, 2, 3]

    for page in document.pages:
        assert page.extraction_method == "pymupdf"
        assert page.character_count == len(page.text)
        assert page.word_count == len(page.text.split())
        assert page.image_count >= 0
        assert page.quality_flag in {"normal", "suspicious"}


def test_parse_pdf_missing_file():
    with pytest.raises(FileNotFoundError):
        parse_pdf(
            "does_not_exist.pdf",
            document_id=DOCUMENT_ID,
            company_name=COMPANY_NAME,
            document_type=DOCUMENT_TYPE,
        )


def test_ocr_page_rejects_invalid_page(tmp_path):
    pdf_path = create_test_pdf(tmp_path / "test.pdf")

    with pytest.raises(ValueError):
        ocr_page(
            pdf_path,
            0,
            document_id=DOCUMENT_ID,
        )

    with pytest.raises(ValueError):
        ocr_page(
            pdf_path,
            4,
            document_id=DOCUMENT_ID,
        )


def test_process_document_keeps_normal_pages(tmp_path):
    pdf_path = create_test_pdf(tmp_path / "test.pdf")

    document = process_document(
        pdf_path,
        document_id=DOCUMENT_ID,
        company_name=COMPANY_NAME,
        document_type=DOCUMENT_TYPE,
    )

    assert document.page_count == 3
    assert len(document.pages) == 3

    assert all(page.extraction_method == "pymupdf" for page in document.pages)
    assert all(page.quality_flag == "normal" for page in document.pages)


def test_section_detector_assigns_sections(tmp_path):
    pdf_path = create_test_pdf(tmp_path / "test.pdf")

    document = process_document(
        pdf_path,
        document_id=DOCUMENT_ID,
        company_name=COMPANY_NAME,
        document_type=DOCUMENT_TYPE,
    )

    document = detect_sections(document)

    assert document.pages[0].section == "GENERAL"
    assert document.pages[0].subsection == "DEFINITIONS AND ABBREVIATIONS"

    assert document.pages[1].section == "RISK FACTORS"

    # Section should propagate to later pages.
    assert document.pages[2].section == "RISK FACTORS"

def test_section_detector_normalizes_heading_punctuation():
    document = Document(
        document_id="heading_test",
        company_name="Test Company Limited",
        document_type="RHP",
        source_file="test.pdf",
        page_count=1,
        pages=[
            DocumentPage(
                document_id="heading_test",
                page_number=1,
                text=(
                    "SECTION IV: - ABOUT OUR COMPANY\n"
                    "- INDUSTRY OVERVIEW\n"
                    "This is sample prospectus text."
                ),
                extraction_method="pymupdf",
                character_count=0,
                word_count=0,
                image_count=0,
                quality_flag="normal",
                section=None,
                subsection=None,
            )
        ],
    )

    document = detect_sections(document)

    assert document.pages[0].section == "ABOUT OUR COMPANY"
    assert document.pages[0].subsection == "INDUSTRY OVERVIEW"

def test_section_detector_handles_nested_numbered_sections():
    document = Document(
        document_id="nested_section_test",
        company_name="Test Company Limited",
        document_type="RHP",
        source_file="test.pdf",
        page_count=3,
        pages=[
            DocumentPage(
                document_id="nested_section_test",
                page_number=1,
                text="SECTION IV: ABOUT OUR COMPANY\nINDUSTRY OVERVIEW\n",
                extraction_method="pymupdf",
                character_count=0,
                word_count=0,
                image_count=0,
                quality_flag="normal",
                section=None,
                subsection=None,
            ),
            DocumentPage(
                document_id="nested_section_test",
                page_number=2,
                text=(
                    "Section 3: Competitive Landscape\n"
                    "There is no single company that offers the same combination."
                ),
                extraction_method="pymupdf",
                character_count=0,
                word_count=0,
                image_count=0,
                quality_flag="normal",
                section=None,
                subsection=None,
            ),
            DocumentPage(
                document_id="nested_section_test",
                page_number=3,
                text=(
                    "Section 4: Industry Threats and Challenges\n"
                    "Several industry-level challenges may affect the sector."
                ),
                extraction_method="pymupdf",
                character_count=0,
                word_count=0,
                image_count=0,
                quality_flag="normal",
                section=None,
                subsection=None,
            ),
        ],
    )

    document = detect_sections(document)

    assert document.pages[0].section == "ABOUT OUR COMPANY"
    assert document.pages[0].subsection == "INDUSTRY OVERVIEW"

    assert document.pages[1].section == "ABOUT OUR COMPANY"
    assert document.pages[1].subsection == "Competitive Landscape"

    assert document.pages[2].section == "ABOUT OUR COMPANY"
    assert document.pages[2].subsection == "Industry Threats and Challenges"


def test_section_detector_ignores_contents_page():
    document = Document(
        document_id="contents_test",
        company_name="Test Company Limited",
        document_type="RHP",
        source_file="test.pdf",
        page_count=1,
        pages=[
            DocumentPage(
                document_id="contents_test",
                page_number=1,
                text=(
                    "CONTENTS\n"
                    "SECTION I – GENERAL ........................................ 1\n"
                    "DEFINITIONS AND ABBREVIATIONS ......................... 2\n"
                ),
                extraction_method="pymupdf",
                character_count=0,
                word_count=0,
                image_count=0,
                quality_flag="normal",
                section=None,
                subsection=None,
            )
        ],
    )

    document = detect_sections(document)

    assert document.pages[0].section is None
    assert document.pages[0].subsection is None

@pytest.mark.integration
def test_real_ellenbarrie_ingestion():
    pdf_path = Path(
        "data/raw/native/Ellenbarrie_RHP_industrial_gas.pdf"
    )

    document = process_document(
        pdf_path,
        document_id="ellenbarrie_rhp",
        company_name="Ellenbarrie Industrial Gases Limited",
        document_type="RHP",
    )

    document = detect_sections(document)

    # Basic document integrity
    assert document.page_count == 417
    assert len(document.pages) == 417

    # Page numbering
    assert document.pages[0].page_number == 1
    assert document.pages[-1].page_number == 417
    assert [page.page_number for page in document.pages] == list(range(1, 418))

    # Document IDs
    assert all(page.document_id == "ellenbarrie_rhp" for page in document.pages)

    # OCR should have replaced the previously suspicious pages
    ocr_pages = [
        page.page_number
        for page in document.pages
        if page.extraction_method == "ocr"
    ]

    assert ocr_pages == list(range(409, 418))

    # No suspicious pages should remain after processing
    assert not any(
        page.quality_flag == "suspicious"
        for page in document.pages
    )

    # Section detection checks
    assert document.pages[3].section == "GENERAL"
    assert document.pages[31].section == "RISK FACTORS"
    assert document.pages[67].section == "INTRODUCTION"
    assert document.pages[138].section == "ABOUT OUR COMPANY"
    assert document.pages[253].section == "FINANCIAL INFORMATION"
    assert document.pages[340].section == "LEGAL AND OTHER INFORMATION"
    assert document.pages[364].section == "OFFER INFORMATION"
    assert (
        document.pages[392].section
        == "DESCRIPTION OF EQUITY SHARES AND TERMS OF THE ARTICLES OF ASSOCIATION"
    )
    assert document.pages[405].section == "OTHER INFORMATION"

    # OCR pages should also have section metadata
    assert document.pages[408].section == "OTHER INFORMATION"
    assert document.pages[408].subsection == "DECLARATION"