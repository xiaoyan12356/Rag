"""PDF loader.

MarkItDown turns the PDF into text and PyMuPDF finds the images that were
embedded in it. The two answers are combined: every image leaves a
`[IMAGE: {image_id}]` marker in the text where the image stood, and the
image itself is written next to the other extracted images.

Image extraction is best effort. A PDF whose images cannot be read still
gives a document with the text in it.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from core.settings import REPO_ROOT, resolve_path
from core.types import Document, image_placeholder
from libs.loader.base_loader import BaseLoader, LoaderError
from libs.loader.file_integrity import compute_file_sha256

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_STORAGE_DIR = "data/images"
PDF_SUFFIX = ".pdf"
PDF_SIGNATURE = b"%PDF"

# MarkItDown keeps the page breaks of the PDF, so the text can be split back
# into the pages it came from.
PAGE_SEPARATOR = "\f"
PAGE_JOIN = "\n\n"

MARKER_PATTERN = re.compile(r"\[IMAGE: ([^\]]+)\]")

try:
    import pymupdf

    PYMUPDF_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    PYMUPDF_AVAILABLE = False


class PdfLoader(BaseLoader):
    """Load a PDF into a document.

    The text comes from MarkItDown, which flattens the page layout into
    Markdown. The images are found with PyMuPDF and saved under
    `data/images/{doc_hash}/`, and each one leaves a marker in the text at
    the place where the image stood.
    """

    def __init__(
        self,
        extract_images: bool = True,
        image_storage_dir: str | Path = DEFAULT_IMAGE_STORAGE_DIR,
    ) -> None:
        self.extract_images = extract_images
        self.image_storage_dir = Path(image_storage_dir)
        self._converter: Any = None

    def load(self, file_path: str | Path) -> Document:
        """Return the document at `file_path`."""
        path = self._validate_file(file_path)
        if path.suffix.lower() != PDF_SUFFIX:
            raise ValueError(f"File is not a PDF: {path}")
        if not self._looks_like_pdf(path):
            raise LoaderError(f"File is not a PDF: {path}")

        doc_hash = compute_file_sha256(path)
        text = self._extract_text(path)

        metadata: dict[str, Any] = {
            "source_path": str(path),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
        }
        title = self._extract_title(text)
        if title:
            metadata["title"] = title

        images: list[dict[str, Any]] = []
        if not PYMUPDF_AVAILABLE:
            logger.warning(
                "PyMuPDF is not installed, so the images of %s are left out", path
            )
        else:
            try:
                with pymupdf.open(path) as document:
                    metadata["page_count"] = document.page_count
                    if self.extract_images:
                        text, images = self._collect_images(document, text, doc_hash)
            except Exception as exc:
                logger.warning("Could not read the images of %s: %s", path, exc)

        if images:
            metadata["images"] = images

        return Document(id=f"doc_{doc_hash[:16]}", text=text, metadata=metadata)

    def _extract_text(self, path: Path) -> str:
        """Return the text of the PDF."""
        try:
            result = self._get_converter().convert(str(path))
        except LoaderError:
            raise
        except Exception as exc:
            raise LoaderError(f"Failed to parse PDF {path}: {exc}") from exc

        text = getattr(result, "markdown", None)
        if text is None:
            text = getattr(result, "text_content", "")
        return text or ""

    def _get_converter(self) -> Any:
        """Return the MarkItDown converter, creating it the first time it is needed."""
        if self._converter is None:
            try:
                from markitdown import MarkItDown
            except ImportError as exc:
                raise LoaderError(
                    "MarkItDown is required to read PDFs. "
                    "Install it with: pip install 'markitdown[pdf]'"
                ) from exc
            self._converter = MarkItDown()
        return self._converter

    def _collect_images(
        self,
        document: Any,
        text: str,
        doc_hash: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Return the text with the image markers in it, and the image records."""
        page_texts = text.split(PAGE_SEPARATOR)
        image_dir = resolve_path(self.image_storage_dir) / doc_hash

        pages: list[str] = []
        markers: list[list[tuple[str, int]]] = []
        records: list[dict[str, Any]] = []

        for page_index in range(document.page_count):
            raw_text = page_texts[page_index].strip() if page_index < len(page_texts) else ""
            page_text, page_markers, page_records = self._process_page(
                document[page_index],
                raw_text,
                doc_hash,
                page_index + 1,
                image_dir,
            )
            pages.append(page_text)
            markers.append(page_markers)
            records.extend(page_records)

        pages.extend(page_text.strip() for page_text in page_texts[document.page_count :])

        joined, offsets = _join_pages(pages, markers)
        for record in records:
            record["text_offset"] = offsets.get(record["id"], 0)
        return joined, records

    def _process_page(
        self,
        page: Any,
        raw_text: str,
        doc_hash: str,
        page_number: int,
        image_dir: Path,
    ) -> tuple[str, list[tuple[str, int]], list[dict[str, Any]]]:
        """Return the text of one page with the image markers in it."""
        image_list = page.get_images(full=True)
        if not image_list:
            return raw_text, [], []

        anchors = _text_blocks(page)
        placements: list[tuple[int, str, Path, Any]] = []

        for sequence, image_info in enumerate(image_list):
            xref = image_info[0]
            try:
                rects = page.get_image_rects(xref)
                if not rects:
                    continue
                payload = page.parent.extract_image(xref)
                image_id = self._generate_image_id(doc_hash, page_number, sequence)
                target = self._save_image(
                    image_dir, image_id, payload["image"], payload.get("ext", "png")
                )
            except Exception as exc:
                logger.warning(
                    "Skipping image %s on page %s of %s: %s", xref, page_number, doc_hash, exc
                )
                continue

            position = _anchor_position(raw_text, anchors, rects[0])
            placements.append((position, image_id, target, rects[0]))

        if not placements:
            return raw_text, [], []

        placements.sort(key=lambda item: item[0])

        parts: list[str] = []
        markers: list[tuple[str, int]] = []
        length = 0
        cursor = 0
        for position, image_id, target, rect in placements:
            parts.append(raw_text[cursor:position])
            length += len(raw_text[cursor:position])
            parts.append(image_placeholder(image_id))
            markers.append((image_id, length))
            length += len(image_placeholder(image_id))
            cursor = position

        parts.append(raw_text[cursor:])
        page_text = "".join(parts)

        records = [
            {
                "id": image_id,
                "path": _display_path(target),
                "page": page_number,
                "text_offset": 0,
                "text_length": len(image_placeholder(image_id)),
                "position": {
                    "x": int(rect.x0),
                    "y": int(rect.y0),
                    "width": int(rect.width),
                    "height": int(rect.height),
                },
            }
            for _, image_id, target, rect in placements
        ]
        return page_text, markers, records

    def _save_image(
        self,
        image_dir: Path,
        image_id: str,
        payload: bytes,
        extension: str,
    ) -> Path:
        """Write the image to disk and return where it went."""
        image_dir.mkdir(parents=True, exist_ok=True)
        target = image_dir / f"{image_id}.{extension}"
        target.write_bytes(payload)
        return target

    @staticmethod
    def _looks_like_pdf(path: Path) -> bool:
        """Return whether the file starts with the signature of a PDF."""
        try:
            with path.open("rb") as handle:
                return handle.read(len(PDF_SIGNATURE)) == PDF_SIGNATURE
        except OSError:
            return False

    @staticmethod
    def _generate_image_id(doc_hash: str, page: int, sequence: int) -> str:
        """Return the id of an image: the document, the page and the place on the page."""
        return f"{doc_hash[:8]}_{page}_{sequence}"

    @staticmethod
    def _extract_title(text: str) -> str | None:
        """Return the title of the document: its first heading, or its first line."""
        lines = text.splitlines()
        for line in lines[:20]:
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
        for line in lines[:10]:
            stripped = line.strip()
            if stripped:
                return stripped
        return None


def _text_blocks(page: Any) -> list[dict[str, Any]]:
    """Return the text blocks of a page, in the order they are read."""
    blocks = page.get_text("dict").get("blocks", [])
    return [block for block in blocks if block.get("type") == 0]


def _block_text(block: dict[str, Any]) -> str:
    """Return the text of a block."""
    lines = []
    for line in block.get("lines", []):
        lines.append("".join(span.get("text", "") for span in line.get("spans", [])))
    return "\n".join(lines)


def _anchor_position(raw_text: str, anchors: list[dict[str, Any]], rect: Any) -> int:
    """Return the place in the page text where the marker of an image belongs.

    The marker goes in front of the text that follows the image, so it stays
    in the same paragraph as the words that describe it. When that text
    cannot be found, the marker goes to the end of the page.
    """
    for block in anchors:
        if block["bbox"][1] < rect.y1 - 2:
            continue
        key = _block_text(block).strip().splitlines()
        key = key[0].strip() if key else ""
        if not key:
            continue
        position = raw_text.find(key)
        if position >= 0:
            return position
    return len(raw_text)


def _join_pages(
    pages: list[str],
    markers: list[list[tuple[str, int]]],
) -> tuple[str, dict[str, int]]:
    """Join the page texts, and report where each marker ended up."""
    parts: list[str] = []
    offsets: dict[str, int] = {}
    position = 0

    for index, page_text in enumerate(pages):
        if index:
            parts.append(PAGE_JOIN)
            position += len(PAGE_JOIN)
        parts.append(page_text)
        for image_id, local_offset in markers[index] if index < len(markers) else ():
            offsets[image_id] = position + local_offset
        position += len(page_text)

    return "".join(parts), offsets


def _display_path(path: Path) -> str:
    """Return the path as it should be stored: relative to the repository when it is inside it."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)
