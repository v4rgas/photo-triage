"""Redundancy detection: exact pixels, near hashes, and long-delayed re-sends."""

from __future__ import annotations

import os
import random
import time

from PIL import Image

from photo_triage.cache import Cache
from photo_triage.dedupe import RESEND_GAP_SECONDS, find_duplicates
from photo_triage.library import Library
from photo_triage.scan import scan

from .conftest import write_image


def library_for(path) -> Library:
    cache = Cache(path)
    scan(cache)
    return Library.open(path)


def test_identical_pixels_with_different_exif_are_exact_duplicates(tmp_path):
    """File checksums find nothing here; the decoded buffer is what matches."""
    picture = Image.new("RGB", (80, 60), (120, 40, 200))
    plain = tmp_path / "sent" / "photo.jpg"
    tagged = tmp_path / "received" / "photo.jpg"
    plain.parent.mkdir()
    tagged.parent.mkdir()
    picture.save(plain, quality=95)
    picture.save(tagged, quality=95, exif=Image.Exif().tobytes(), comment=b"forwarded")

    assert plain.read_bytes() != tagged.read_bytes()

    groups = find_duplicates(library_for(tmp_path))
    assert [group.kind for group in groups] == ["exact"]
    assert len(groups[0].others) == 1


def textured(seed: int, size=(160, 120)) -> Image.Image:
    """A picture with structure at every frequency, so its phash is meaningful.

    Smooth ramps will not do here: two different linear gradients share their
    low-frequency content and land within a few bits of each other, which is
    perceptual hashing working correctly rather than a fault to design around.
    """
    generator = random.Random(seed)
    picture = Image.new("RGB", size)
    picture.putdata(
        [tuple(generator.randrange(256) for _ in range(3)) for _ in range(size[0] * size[1])]
    )
    return picture


def test_distinct_pictures_are_not_grouped(tmp_path):
    textured(7).save(tmp_path / "a.png")
    textured(31).save(tmp_path / "b.png")
    assert find_duplicates(library_for(tmp_path)) == []


def test_flat_images_are_not_near_duplicates_of_each_other(tmp_path):
    """A solid colour hashes to almost nothing, and would otherwise match all."""
    write_image(tmp_path / "red.png", (255, 0, 0))
    write_image(tmp_path / "black.png", (0, 0, 0), size=(200, 30))
    write_image(tmp_path / "white.png", (255, 255, 255), size=(90, 90))
    assert find_duplicates(library_for(tmp_path)) == []


def test_the_keeper_is_the_earliest_then_largest_copy(tmp_path):
    picture = Image.new("RGB", (200, 150), (30, 160, 90))
    original = tmp_path / "2019" / "IMG_1.jpg"
    resend = tmp_path / "2026" / "IMG_1_forwarded_copy.jpg"
    original.parent.mkdir()
    resend.parent.mkdir()
    picture.save(original, quality=98)
    picture.save(resend, quality=98)

    old = time.time() - RESEND_GAP_SECONDS * 2
    os.utime(original, (old, old))

    library = library_for(tmp_path)
    groups = find_duplicates(library)
    assert len(groups) == 1
    assert library.records[groups[0].keeper].rel == "2019/IMG_1.jpg"


def test_a_years_apart_copy_is_flagged_as_a_resend(tmp_path):
    """Same picture, re-encoded smaller and re-sent much later."""
    picture = Image.new("RGB", (240, 180))
    for x in range(240):
        for y in range(180):
            picture.putpixel((x, y), ((x * 7) % 256, (y * 5) % 256, (x + y) % 256))
    original = tmp_path / "2019" / "IMG_2.jpg"
    resend = tmp_path / "2026" / "IMG_2.jpg"
    original.parent.mkdir()
    resend.parent.mkdir()
    picture.save(original, quality=95)
    picture.resize((120, 90)).save(resend, quality=60)

    old = time.time() - RESEND_GAP_SECONDS * 2
    os.utime(original, (old, old))

    groups = find_duplicates(library_for(tmp_path))
    assert [group.kind for group in groups] == ["resend"]


def test_quarantined_images_are_not_offered_as_duplicates(tmp_path):
    picture = Image.new("RGB", (80, 60), (10, 10, 200))
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    picture.save(tmp_path / "a" / "x.jpg")
    picture.save(tmp_path / "b" / "x.jpg")

    library = library_for(tmp_path)
    assert len(find_duplicates(library)) == 1

    library.where.quarantine([0])
    assert find_duplicates(library) == []
