"""Unit tests for the PDF loader.

The loader is the first stage of the ingestion pipeline, so the tests pin
what comes out of it: a document with the text of the PDF, metadata that
says where it came from, and the images of the PDF as markers in the text.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest

from core.settings import resolve_path
from core.types import Document, image_placeholder
from libs.loader.base_loader import BaseLoader, LoaderError
from libs.loader.pdf_loader import PdfLoader

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "sample_documents"
SIMPLE_PDF = FIXTURES_DIR / "simple.pdf"
IMAGES_PDF = FIXTURES_DIR / "with_images.pdf"


@pytest.fixture
def loader(tmp_path: Path) -> PdfLoader:
    """Return a loader that keeps the extracted images in a temporary directory."""
    return PdfLoader(image_storage_dir=tmp_path / "images")


@pytest.fixture
def default_loader_images():
    """Remove the images a test wrote under the default storage directory."""
    written: list[Path] = []
    yield written
    for directory in written:
        shutil.rmtree(directory, ignore_errors=True)


class TestBaseLoader:
    def test_the_base_class_cannot_be_used_on_its_own(self):
        with pytest.raises(TypeError, match="abstract"):
            BaseLoader()

    def test_a_subclass_has_to_implement_load(self):
        class Incomplete(BaseLoader):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_an_existing_file_is_accepted(self, tmp_path: Path):
        target = tmp_path / "document.pdf"
        target.write_text("content", encoding="utf-8")

        validated = PdfLoader._validate_file(target)

        assert validated.is_file()

    def test_a_missing_file_is_reported(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            PdfLoader._validate_file(tmp_path / "missing.pdf")

    def test_a_directory_is_reported(self, tmp_path: Path):
        with pytest.raises(OSError, match="not a file"):
            PdfLoader._validate_file(tmp_path)


class TestConfiguration:
    def test_the_defaults_are_used(self):
        instance = PdfLoader()

        assert instance.extract_images is True
        assert instance.image_storage_dir == Path("data/images")

    def test_the_configuration_can_be_changed(self, tmp_path: Path):
        instance = PdfLoader(extract_images=False, image_storage_dir=tmp_path)

        assert instance.extract_images is False
        assert instance.image_storage_dir == tmp_path


class TestValidation:
    def test_a_file_that_is_not_a_pdf_is_rejected(self, tmp_path: Path):
        target = tmp_path / "notes.txt"
        target.write_text("not a pdf", encoding="utf-8")

        with pytest.raises(ValueError, match="not a PDF"):
            PdfLoader().load(target)

    def test_a_missing_file_is_rejected(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            PdfLoader().load(tmp_path / "missing.pdf")

    def test_a_directory_is_rejected(self, tmp_path: Path):
        with pytest.raises(OSError, match="not a file"):
            PdfLoader().load(tmp_path)


class TestHelpers:
    def test_the_image_id_names_the_document_the_page_and_the_place(self):
        assert PdfLoader._generate_image_id("abc123def456", 2, 0) == "abc123de_2_0"

    def test_the_title_comes_from_the_first_heading(self):
        title = PdfLoader._extract_title("# Main Title\n\nParagraph text\n\n## Subtitle")

        assert title == "Main Title"

    def test_the_title_falls_back_to_the_first_line(self):
        title = PdfLoader._extract_title("First Line Title\n\nParagraph text")

        assert title == "First Line Title"

    def test_an_empty_text_has_no_title(self):
        assert PdfLoader._extract_title("") is None


class TestSimplePdf:
    def test_a_document_is_produced(self, loader: PdfLoader):
        document = loader.load(SIMPLE_PDF)

        assert isinstance(document, Document)
        assert document.text.strip()

    def test_the_metadata_says_where_the_document_came_from(self, loader: PdfLoader):
        document = loader.load(SIMPLE_PDF)

        assert document.metadata["source_path"] == str(SIMPLE_PDF.resolve())
        assert document.metadata["doc_type"] == "pdf"
        assert len(document.metadata["doc_hash"]) == 64

    def test_the_text_of_the_pdf_is_in_the_document(self, loader: PdfLoader):
        document = loader.load(SIMPLE_PDF)

        assert "sample document" in document.text.lower()
        assert "ingestion" in document.text.lower()

    def test_the_title_is_taken_from_the_document(self, loader: PdfLoader):
        document = loader.load(SIMPLE_PDF)

        assert "sample" in document.metadata["title"].lower()

    def test_the_number_of_pages_is_recorded(self, loader: PdfLoader):
        document = loader.load(SIMPLE_PDF)

        assert document.metadata["page_count"] == 1

    def test_a_document_without_images_says_so(self, loader: PdfLoader):
        document = loader.load(SIMPLE_PDF)

        assert document.metadata.get("images", []) == []

    def test_the_same_file_gives_the_same_document(self, loader: PdfLoader):
        first = loader.load(SIMPLE_PDF)
        second = loader.load(SIMPLE_PDF)

        assert first.id == second.id
        assert first.metadata["doc_hash"] == second.metadata["doc_hash"]
        assert first.text == second.text

    def test_the_document_can_be_serialised_and_restored(self, loader: PdfLoader):
        document = loader.load(SIMPLE_PDF)

        restored = Document.from_dict(document.to_dict())

        assert restored.id == document.id
        assert restored.text == document.text
        assert restored.metadata == document.metadata


class TestPdfWithImages:
    def test_the_images_are_extracted(self, loader: PdfLoader):
        document = loader.load(IMAGES_PDF)

        assert len(document.metadata["images"]) == 2

    def test_every_image_is_described(self, loader: PdfLoader):
        document = loader.load(IMAGES_PDF)

        for image in document.metadata["images"]:
            assert image["id"]
            assert image["path"]
            assert isinstance(image["page"], int)
            assert isinstance(image["text_offset"], int)
            assert image["text_offset"] >= 0
            assert isinstance(image["text_length"], int)
            assert image["text_length"] > 0
            assert set(image["position"]) == {"x", "y", "width", "height"}

    def test_the_images_are_written_to_disk(self, loader: PdfLoader):
        document = loader.load(IMAGES_PDF)

        for image in document.metadata["images"]:
            assert Path(image["path"]).exists()

    def test_the_images_are_written_under_the_hash_of_the_document(self, loader: PdfLoader):
        document = loader.load(IMAGES_PDF)

        directory = Path(loader.image_storage_dir) / document.metadata["doc_hash"]

        assert directory.is_dir()
        assert len(list(directory.iterdir())) == 2

    def test_the_marker_of_an_image_is_in_the_text(self, loader: PdfLoader):
        document = loader.load(IMAGES_PDF)

        for image in document.metadata["images"]:
            assert image_placeholder(image["id"]) in document.text

    def test_the_offset_of_the_marker_is_right(self, loader: PdfLoader):
        document = loader.load(IMAGES_PDF)

        for image in document.metadata["images"]:
            start = image["text_offset"]
            end = start + image["text_length"]

            assert document.text[start:end] == image_placeholder(image["id"])

    def test_every_image_leaves_exactly_one_marker(self, loader: PdfLoader):
        document = loader.load(IMAGES_PDF)

        for image in document.metadata["images"]:
            assert document.text.count(image_placeholder(image["id"])) == 1

    def test_the_markers_are_on_the_page_the_image_was_on(self, loader: PdfLoader):
        document = loader.load(IMAGES_PDF)

        first, second = document.metadata["images"]
        assert document.text.index(image_placeholder(first["id"])) < document.text.index(
            image_placeholder(second["id"])
        )
        assert "Data Flow" in document.text[first["text_offset"] :]

    def test_the_marker_stands_in_front_of_the_text_of_the_image(self, loader: PdfLoader):
        document = loader.load(IMAGES_PDF)

        image = document.metadata["images"][0]
        start = image["text_offset"]

        assert document.text[start : start + image["text_length"] + 1].endswith("]E")

    def test_the_image_ids_are_unique(self, loader: PdfLoader):
        document = loader.load(IMAGES_PDF)

        ids = [image["id"] for image in document.metadata["images"]]

        assert len(set(ids)) == len(ids)

    def test_the_image_ids_name_the_document_the_page_and_the_place(self, loader: PdfLoader):
        document = loader.load(IMAGES_PDF)

        prefix = document.metadata["doc_hash"][:8]
        assert document.metadata["images"][0]["id"] == f"{prefix}_1_0"
        assert document.metadata["images"][1]["id"] == f"{prefix}_2_0"

    def test_the_text_is_still_complete(self, loader: PdfLoader):
        document = loader.load(IMAGES_PDF)

        assert "Architecture Overview" in document.text
        assert "Data Flow" in document.text
        assert "Nothing leaves the machine." in document.text

    def test_the_images_can_be_left_out(self, tmp_path: Path):
        document = PdfLoader(
            extract_images=False,
            image_storage_dir=tmp_path / "images",
        ).load(IMAGES_PDF)

        assert document.metadata.get("images", []) == []
        assert "Architecture Overview" in document.text
        assert "[IMAGE:" not in document.text

    def test_the_default_storage_directory_is_used(self, default_loader_images):
        document = PdfLoader().load(IMAGES_PDF)

        directory = resolve_path("data/images") / document.metadata["doc_hash"]
        default_loader_images.append(directory)

        assert directory.is_dir()
        assert len(list(directory.iterdir())) == 2

    def test_the_path_of_an_image_is_relative_to_the_repository(
        self, default_loader_images
    ):
        document = PdfLoader().load(IMAGES_PDF)
        default_loader_images.append(
            resolve_path("data/images") / document.metadata["doc_hash"]
        )

        path = document.metadata["images"][0]["path"]

        assert not Path(path).is_absolute()
        assert path.startswith("data/images/")


class TestGracefulDegradation:
    def test_a_failing_image_extraction_does_not_block_the_text(
        self, loader: PdfLoader, monkeypatch: pytest.MonkeyPatch
    ):
        def explode(*args, **kwargs):
            raise RuntimeError("no images today")

        monkeypatch.setattr(PdfLoader, "_collect_images", explode)

        document = loader.load(IMAGES_PDF)

        assert "Architecture Overview" in document.text
        assert document.metadata.get("images", []) == []

    def test_a_failing_image_extraction_is_logged(
        self, loader: PdfLoader, monkeypatch: pytest.MonkeyPatch, caplog
    ):
        def explode(*args, **kwargs):
            raise RuntimeError("no images today")

        monkeypatch.setattr(PdfLoader, "_collect_images", explode)

        with caplog.at_level(logging.WARNING):
            loader.load(IMAGES_PDF)

        assert any("no images today" in record.getMessage() for record in caplog.records)

    def test_a_pdf_that_cannot_be_parsed_is_reported(self, tmp_path: Path):
        broken = tmp_path / "broken.pdf"
        broken.write_text("this is not a pdf at all", encoding="utf-8")

        with pytest.raises(LoaderError):
            PdfLoader().load(broken)
