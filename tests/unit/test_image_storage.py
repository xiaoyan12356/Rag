"""Unit tests for the image storage.

The storage keeps the images of a document on disk and remembers in a
database which file belongs to which image id. The tests pin that the file
is really written, that the note survives, and that the same image saved
twice leaves one file behind.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.storage.image_storage import ImageStorage

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 32


@pytest.fixture
def storage(tmp_path):
    return ImageStorage(db_path=tmp_path / "db" / "image_index.db", images_root=tmp_path / "images")


class TestSaveImage:
    def test_the_image_is_written_to_disk(self, storage):
        path = storage.save_image("img_1", PNG_BYTES, collection="kb")

        assert Path(path).exists()

    def test_the_image_holds_what_it_was_given(self, storage):
        path = storage.save_image("img_1", PNG_BYTES, collection="kb")

        assert Path(path).read_bytes() == PNG_BYTES

    def test_the_image_goes_into_the_folder_of_the_collection(self, storage):
        path = storage.save_image("img_1", PNG_BYTES, collection="kb")

        assert Path(path).parent.name == "kb"

    def test_an_image_can_be_copied_from_a_file(self, storage, tmp_path):
        source = tmp_path / "source.png"
        source.write_bytes(PNG_BYTES)

        path = storage.save_image("img_1", source, collection="kb")

        assert Path(path).read_bytes() == PNG_BYTES

    def test_a_file_that_is_not_there_is_reported(self, storage, tmp_path):
        with pytest.raises(OSError, match="Could not save the image"):
            storage.save_image("img_1", tmp_path / "missing.png", collection="kb")

    def test_an_image_id_that_is_empty_is_refused(self, storage):
        with pytest.raises(ValueError, match="image_id"):
            storage.save_image("  ", PNG_BYTES, collection="kb")


class TestRegisterImage:
    def test_the_note_is_written(self, storage, tmp_path):
        image = tmp_path / "image.png"
        image.write_bytes(PNG_BYTES)

        storage.register_image("img_1", image, collection="kb")

        assert storage.get_image_path("img_1") is not None

    def test_the_file_is_left_where_it_is(self, storage, tmp_path):
        image = tmp_path / "image.png"
        image.write_bytes(PNG_BYTES)

        path = storage.register_image("img_1", image, collection="kb")

        assert Path(path) == image.resolve()

    def test_an_image_that_is_not_there_is_reported(self, storage, tmp_path):
        with pytest.raises(FileNotFoundError):
            storage.register_image("img_1", tmp_path / "missing.png", collection="kb")

    def test_an_image_id_that_is_empty_is_refused(self, storage):
        with pytest.raises(ValueError, match="image_id"):
            storage.register_image("", "whatever.png")


class TestLookup:
    def test_the_path_of_an_image_is_found(self, storage):
        path = storage.save_image("img_1", PNG_BYTES, collection="kb")

        assert storage.get_image_path("img_1") == path

    def test_an_image_that_is_not_there_is_not_found(self, storage):
        assert storage.get_image_path("nothing") is None

    def test_it_is_told_whether_an_image_is_known(self, storage):
        storage.save_image("img_1", PNG_BYTES, collection="kb")

        assert storage.image_exists("img_1") is True
        assert storage.image_exists("nothing") is False

    def test_the_images_of_a_collection_are_listed(self, storage):
        storage.save_image("img_1", PNG_BYTES, collection="kb")
        storage.save_image("img_2", PNG_BYTES, collection="kb")
        storage.save_image("img_3", PNG_BYTES, collection="other")

        assert len(storage.list_images(collection="kb")) == 2

    def test_the_images_of_a_document_are_listed(self, storage):
        storage.save_image("img_1", PNG_BYTES, collection="kb", doc_hash="hash_a")
        storage.save_image("img_2", PNG_BYTES, collection="kb", doc_hash="hash_b")

        assert len(storage.list_images(doc_hash="hash_a")) == 1

    def test_the_page_of_an_image_is_kept(self, storage):
        storage.save_image("img_1", PNG_BYTES, collection="kb", page_num=7)

        assert storage.list_images(collection="kb")[0]["page_num"] == 7


class TestIdempotency:
    def test_the_same_image_saved_twice_leaves_one_file(self, storage):
        storage.save_image("img_1", PNG_BYTES, collection="kb")
        storage.save_image("img_1", PNG_BYTES, collection="kb")

        assert len(storage.list_images(collection="kb")) == 1

    def test_the_same_image_saved_twice_leaves_one_note(self, storage):
        storage.save_image("img_1", PNG_BYTES, collection="kb")
        storage.save_image("img_1", PNG_BYTES, collection="kb")

        assert storage.get_image_path("img_1") is not None
        assert len(storage.list_images()) == 1

    def test_the_note_survives_a_new_run(self, tmp_path):
        db_path = tmp_path / "db" / "image_index.db"
        images_root = tmp_path / "images"

        first = ImageStorage(db_path=db_path, images_root=images_root)
        path = first.save_image("img_1", PNG_BYTES, collection="kb")

        second = ImageStorage(db_path=db_path, images_root=images_root)

        assert second.get_image_path("img_1") == path


class TestDelete:
    def test_an_image_can_be_forgotten(self, storage):
        storage.save_image("img_1", PNG_BYTES, collection="kb")

        assert storage.delete_image("img_1") is True
        assert storage.get_image_path("img_1") is None

    def test_the_file_is_taken_away(self, storage):
        path = storage.save_image("img_1", PNG_BYTES, collection="kb")

        storage.delete_image("img_1")

        assert not Path(path).exists()

    def test_the_file_can_be_kept(self, storage):
        path = storage.save_image("img_1", PNG_BYTES, collection="kb")

        storage.delete_image("img_1", remove_file=False)

        assert Path(path).exists()

    def test_an_image_that_is_not_there_is_reported(self, storage):
        assert storage.delete_image("nothing") is False


class TestStats:
    def test_the_number_of_images_is_counted(self, storage):
        storage.save_image("img_1", PNG_BYTES, collection="kb")
        storage.save_image("img_2", PNG_BYTES, collection="kb")

        stats = storage.get_collection_stats("kb")

        assert stats["total_images"] == 2

    def test_the_room_the_images_take_is_counted(self, storage):
        storage.save_image("img_1", PNG_BYTES, collection="kb")

        stats = storage.get_collection_stats("kb")

        assert stats["total_size_bytes"] == len(PNG_BYTES)
