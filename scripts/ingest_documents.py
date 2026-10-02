from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal


# Allow imports from the src/ layout when running this script
# directly from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from ipo_analyzer.ingestion.document_processor import process_document
from ipo_analyzer.ingestion.section_detector import detect_sections


DocumentType = Literal["RHP", "DRHP", "FINAL_PROSPECTUS"]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Parse and process IPO prospectus PDF documents."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--pdf",
        type=Path,
        help="Path to a single source PDF.",
    )

    input_group.add_argument(
        "--manifest",
        type=Path,
        help="Path to a JSON manifest describing multiple PDFs.",
    )

    parser.add_argument(
        "--document-id",
        help="Unique identifier for a single document.",
    )

    parser.add_argument(
        "--company-name",
        help="Company name for a single document.",
    )

    parser.add_argument(
        "--document-type",
        choices=["RHP", "DRHP", "FINAL_PROSPECTUS"],
        help="Type of a single prospectus document.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where processed JSON files will be saved.",
    )

    return parser.parse_args()


def ingest_document(
    pdf_path: Path,
    *,
    document_id: str,
    company_name: str,
    document_type: DocumentType,
):
    """
    Run PDF parsing, selective OCR, and section detection.

    Returns
    -------
    Document
        Fully processed document with page-level metadata.
    """

    document = process_document(
        pdf_path,
        document_id=document_id,
        company_name=company_name,
        document_type=document_type,
    )

    document = detect_sections(document)

    return document


def save_document(
    document,
    output_dir: Path,
) -> Path:
    """Save a processed Document as JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{document.document_id}.json"

    output_path.write_text(
        document.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return output_path


def print_summary(
    document,
    output_path: Path,
) -> None:
    """Print a summary of the processed document."""

    ocr_pages = [
        page
        for page in document.pages
        if page.extraction_method == "ocr"
    ]

    suspicious_pages = [
        page
        for page in document.pages
        if page.quality_flag == "suspicious"
    ]

    sectioned_pages = [
        page
        for page in document.pages
        if page.section is not None
    ]

    print("\nINGESTION SUMMARY")
    print("=" * 80)
    print(f"Page count:           {document.page_count}")
    print(
        f"PyMuPDF pages:        "
        f"{document.page_count - len(ocr_pages)}"
    )
    print(f"OCR pages:            {len(ocr_pages)}")
    print(f"Sectioned pages:      {len(sectioned_pages)}")
    print(f"Suspicious remaining: {len(suspicious_pages)}")
    print(f"Output:               {output_path}")
    print("=" * 80)


def validate_single_document_arguments(
    args: argparse.Namespace,
) -> None:
    """Validate metadata required for single-document ingestion."""

    missing = []

    if args.document_id is None:
        missing.append("--document-id")

    if args.company_name is None:
        missing.append("--company-name")

    if args.document_type is None:
        missing.append("--document-type")

    if missing:
        raise ValueError(
            "The following arguments are required when using --pdf: "
            + ", ".join(missing)
        )


def ingest_single_document(
    args: argparse.Namespace,
) -> None:
    """Process one PDF document."""

    validate_single_document_arguments(args)

    pdf_path = args.pdf

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file does not exist: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, got: {pdf_path}"
        )

    print("=" * 80)
    print("IPO PROSPECTUS INGESTION")
    print("=" * 80)
    print(f"PDF:           {pdf_path}")
    print(f"Document ID:   {args.document_id}")
    print(f"Company:       {args.company_name}")
    print(f"Document type: {args.document_type}")
    print("=" * 80)

    document = ingest_document(
        pdf_path,
        document_id=args.document_id,
        company_name=args.company_name,
        document_type=args.document_type,
    )

    output_path = save_document(
        document,
        args.output_dir,
    )

    print_summary(
        document,
        output_path,
    )


def load_manifest(
    manifest_path: Path,
) -> list[dict]:
    """Load and validate a JSON ingestion manifest."""

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest file does not exist: {manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    if not isinstance(manifest, list):
        raise ValueError(
            "Ingestion manifest must contain a JSON list."
        )

    required_fields = {
        "pdf",
        "document_id",
        "company_name",
        "document_type",
    }

    for index, item in enumerate(manifest, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"Manifest entry {index} must be a JSON object."
            )

        missing_fields = required_fields - item.keys()

        if missing_fields:
            raise ValueError(
                f"Manifest entry {index} is missing: "
                + ", ".join(sorted(missing_fields))
            )

        if item["document_type"] not in {
            "RHP",
            "DRHP",
            "FINAL_PROSPECTUS",
        }:
            raise ValueError(
                f"Invalid document_type in manifest entry {index}: "
                f"{item['document_type']}"
            )

    return manifest


def ingest_manifest(
    manifest_path: Path,
    output_dir: Path,
) -> None:
    """Process all documents listed in the manifest."""

    manifest = load_manifest(manifest_path)

    successful = 0
    failed = 0

    print("\n" + "=" * 80)
    print("BATCH IPO PROSPECTUS INGESTION")
    print("=" * 80)
    print(f"Manifest:       {manifest_path}")
    print(f"Documents:      {len(manifest)}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    for index, item in enumerate(manifest, start=1):
        pdf_path = Path(item["pdf"])

        print("\n" + "=" * 80)
        print(
            f"Processing document {index}/{len(manifest)}"
        )
        print(f"Company: {item['company_name']}")
        print(f"PDF:     {pdf_path}")
        print("=" * 80)

        try:
            if not pdf_path.exists():
                raise FileNotFoundError(
                    f"PDF file does not exist: {pdf_path}"
                )

            if pdf_path.suffix.lower() != ".pdf":
                raise ValueError(
                    f"Expected a PDF file, got: {pdf_path}"
                )

            document = ingest_document(
                pdf_path,
                document_id=item["document_id"],
                company_name=item["company_name"],
                document_type=item["document_type"],
            )

            output_path = save_document(
                document,
                output_dir,
            )

            print_summary(
                document,
                output_path,
            )

            successful += 1

        except Exception as error:
            failed += 1

            print(
                f"\nFAILED: {item['company_name']}"
            )
            print(
                f"Reason: {error}"
            )

    print("\n" + "=" * 80)
    print("BATCH INGESTION SUMMARY")
    print("=" * 80)
    print(f"Documents processed: {len(manifest)}")
    print(f"Successful:          {successful}")
    print(f"Failed:              {failed}")
    print("=" * 80)

    if failed > 0:
        raise RuntimeError(
            f"Batch ingestion completed with {failed} failure(s)."
        )


def main() -> None:
    """Run single-document or batch ingestion."""

    args = parse_arguments()

    if args.pdf is not None:
        ingest_single_document(args)
    else:
        ingest_manifest(
            args.manifest,
            args.output_dir,
        )


if __name__ == "__main__":
    main()