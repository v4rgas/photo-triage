"""Stage 1: what gets indexed, what row it lands on, and what a re-scan does."""

from __future__ import annotations

import numpy as np
from PIL import Image

from photo_triage.cache import CACHE_DIRNAME, Cache
from photo_triage.embed import embed
from photo_triage.scan import scan

from .conftest import write_image


def rels(cache: Cache) -> list[str]:
    return [record.rel for record in cache.load_index()]


def test_the_cache_directory_is_never_indexed(folder):
    cache = Cache(folder)
    scan(cache)
    cache.thumbs_dir.mkdir(parents=True, exist_ok=True)
    write_image(cache.thumbs_dir / "0.jpg", (10, 10, 10))
    write_image(cache.quarantine_dir / "holidays" / "binned.jpg", (20, 20, 20))

    scan(cache)
    assert not any(CACHE_DIRNAME in rel for rel in rels(cache))


def test_rescanning_keeps_every_row_id_and_appends_new_files(folder):
    cache = Cache(folder)
    before = rels(scan_and_read(cache))
    write_image(folder / "aaa_first_alphabetically.jpg", (5, 5, 5))
    after = rels(scan_and_read(cache))

    # The new file sorts first by name and must still land last by row id.
    assert after[: len(before)] == before
    assert after[-1] == "aaa_first_alphabetically.jpg"


def test_a_quarantined_file_keeps_its_row_across_a_rescan(scanned: Cache):
    from photo_triage.quarantine import Quarantine

    before = rels(scanned)
    row = before.index("holidays/IMG_0042.jpg")
    Quarantine(scanned, scanned.load_index()).quarantine([row])

    scan(scanned)
    assert rels(scanned) == before


def test_format_comes_from_the_bytes_not_the_extension(tmp_path):
    """A WebP named .jpg must be indexed, and reported as what it is."""
    misnamed = tmp_path / "actually_a_webp.jpg"
    Image.new("RGB", (32, 32), (7, 90, 190)).save(misnamed, format="WEBP")
    cache = Cache(tmp_path)
    records = scan(cache)
    assert [r.rel for r in records] == ["actually_a_webp.jpg"]
    assert records[0].fmt == "WEBP"


def test_an_unreadable_file_gets_a_row_and_an_error(tmp_path):
    write_image(tmp_path / "fine.jpg", (1, 2, 3))
    truncated = tmp_path / "broken.jpg"
    truncated.write_bytes(b"\xff\xd8\xff" + b"\x00" * 40)

    records = scan(Cache(tmp_path))
    by_rel = {record.rel: record for record in records}
    assert set(by_rel) == {"fine.jpg", "broken.jpg"}
    assert by_rel["fine.jpg"].readable
    assert not by_rel["broken.jpg"].readable


def test_non_images_are_skipped(tmp_path):
    write_image(tmp_path / "photo.png", (9, 9, 9))
    (tmp_path / "notes.txt").write_text("not a picture")
    assert rels(scan_and_read(Cache(tmp_path))) == ["photo.png"]


def test_embedding_resumes_instead_of_restarting(scanned: Cache):
    """An interrupted embed pass leaves zero rows, and only those get redone."""
    records = scanned.load_index()
    partial = np.zeros((len(records), 512), dtype=np.float32)
    partial[0] = np.eye(512, dtype=np.float32)[0]  # one row already done
    scanned.save_embeddings(partial)

    asked_for = []

    class RecordingEmbedder:
        def images(self, paths, progress=None):
            asked_for.extend(paths)
            vectors = np.zeros((len(paths), 512), dtype=np.float32)
            vectors[:, 1] = 1.0
            return vectors

    embeds = embed(scanned, records, RecordingEmbedder())
    assert len(asked_for) == len(records) - 1
    assert np.array_equal(embeds[0], partial[0])
    assert embeds[1:].any(axis=1).all()


def scan_and_read(cache: Cache) -> Cache:
    scan(cache)
    return cache
