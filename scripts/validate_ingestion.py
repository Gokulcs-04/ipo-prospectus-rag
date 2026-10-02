from pathlib import Path
import time

from ipo_analyzer.ingestion.document_processor import process_document
from ipo_analyzer.ingestion.section_detector import detect_sections


RAW_DIR = Path("data/raw/native")


DOCUMENTS = [
    {
        "filename": "Ellenbarrie_RHP_industrial_gas.pdf",
        "document_id": "ellenbarrie_rhp",
        "company_name": "Ellenbarrie Industrial Gases Limited",
        "document_type": "RHP",
    },
    {
        "filename": "SEBI _ AEGIS VOPAK TERMINALS LIMITED - RHP_energy_terminal.pdf",
        "document_id": "aegis_vopak_rhp",
        "company_name": "Aegis Vopak Terminals Limited",
        "document_type": "RHP",
    },
    {
        "filename": "SEBI _ Brigade Hotel Ventures Limited- RHP_hospitality.pdf",
        "document_id": "brigade_hotel_rhp",
        "company_name": "Brigade Hotel Ventures Limited",
        "document_type": "RHP",
    },
    {
        "filename": "SEBI _ Pine Labs Ltd - RHP_fintech.pdf",
        "document_id": "pine_labs_rhp",
        "company_name": "Pine Labs Limited",
        "document_type": "RHP",
    },
    {
        "filename": "SEBI _ Sai Parenteral’s Limited - RHP_pharmaceutical.pdf",
        "document_id": "sai_parenterals_rhp",
        "company_name": "Sai Parenteral's Limited",
        "document_type": "RHP",
    },
]

def print_section_transitions(document):
    """Print pages where the detected major section changes."""
    previous_section = None

    for page in document.pages:
        if page.section != previous_section:
            print(
                f"    Page {page.page_number}: "
                f"{page.section!r}"
            )
            previous_section = page.section


def validate_document(info):
    pdf_path = RAW_DIR / info["filename"]

    print("\n" + "=" * 80)
    print(f"Processing: {info['company_name']}")
    print(f"File:       {info['filename']}")
    print("=" * 80)

    if not pdf_path.exists():
        print("ERROR: File not found.")
        return False

    start_time = time.perf_counter()

    try:
        document = process_document(
            pdf_path,
            document_id=info["document_id"],
            company_name=info["company_name"],
            document_type=info["document_type"],
        )

        document = detect_sections(document)

    except Exception as exc:
        elapsed = time.perf_counter() - start_time

        print(f"ERROR: {type(exc).__name__}: {exc}")
        print(f"Time before failure: {elapsed:.2f} seconds")

        return False

    elapsed = time.perf_counter() - start_time

    native_pages = [
        page for page in document.pages
        if page.extraction_method == "pymupdf"
    ]

    ocr_pages = [
        page for page in document.pages
        if page.extraction_method == "ocr"
    ]

    suspicious_pages = [
        page for page in document.pages
        if page.quality_flag == "suspicious"
    ]

    if suspicious_pages:
        print("\nSuspicious page details:")
        for page in suspicious_pages:
            preview = " ".join(page.text.split())[:200]

            print(
                f"    Page {page.page_number}: "
                f"method={page.extraction_method}, "
                f"chars={page.character_count}, "
                f"words={page.word_count}, "
                f"section={page.section!r}"
            )
            print(f"        Preview: {preview!r}")

    sectioned_pages = [
        page for page in document.pages
        if page.section is not None
    ]

    print(f"Page count:              {document.page_count}")
    print(f"Parsed pages:            {len(document.pages)}")
    print(f"PyMuPDF pages:           {len(native_pages)}")
    print(f"OCR pages:               {len(ocr_pages)}")
    print(
        "OCR page numbers:        "
        f"{[page.page_number for page in ocr_pages]}"
    )
    print(
        "Suspicious remaining:    "
        f"{[page.page_number for page in suspicious_pages]}"
    )
    print(f"Sectioned pages:         {len(sectioned_pages)}")
    print(f"Processing time:         {elapsed:.2f} seconds")

    print("\nMajor section transitions:")
    print_section_transitions(document)

    # Basic integrity checks
    assert document.page_count == len(document.pages)
    assert [page.page_number for page in document.pages] == list(
        range(1, document.page_count + 1)
    )
    assert all(
        page.document_id == info["document_id"]
        for page in document.pages
    )

    # Suspicious pages are reported for review.
    # They are not automatically treated as processing failures,
    # because a page may legitimately contain very little text.
    if suspicious_pages:
        print(
            "WARNING: Suspicious pages remain after OCR: "
            f"{[page.page_number for page in suspicious_pages]}"
        )
    else:
        print("No suspicious pages remain after processing.")

    return True

def main():
    print("\nIPO PROSPECTUS INGESTION VALIDATION")
    print(f"Raw data directory: {RAW_DIR.resolve()}")

    successful = 0
    failed = 0

    overall_start = time.perf_counter()

    for info in DOCUMENTS:
        try:
            success = validate_document(info)
        except AssertionError as exc:
            print(f"VALIDATION FAILED: {exc}")
            success = False

        if success:
            successful += 1
        else:
            failed += 1

    overall_elapsed = time.perf_counter() - overall_start

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Documents tested:  {len(DOCUMENTS)}")
    print(f"Successful:        {successful}")
    print(f"Failed:            {failed}")
    print(f"Total time:        {overall_elapsed:.2f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    main()