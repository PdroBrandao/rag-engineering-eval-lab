import io
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from retrieval_lab.errors import IngestError


def pdf_filename(filename: str | None) -> str:
    name = Path(filename or "document.pdf").name.strip()
    return name or "document.pdf"


def extract_pages(data: bytes) -> list[tuple[int, str]]:
    if not data:
        raise IngestError(400, "The uploaded file is empty.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise IngestError(400, "Encrypted PDFs are not supported.")
        pages: list[tuple[int, str]] = []
        for number, page in enumerate(reader.pages, start=1):
            pages.append((number, page.extract_text() or ""))
        return pages
    except IngestError:
        raise
    except PdfReadError as exc:
        raise IngestError(400, "The uploaded file is not a valid PDF.") from exc
    except Exception as exc:
        raise IngestError(400, "The uploaded file is not a valid PDF.") from exc
