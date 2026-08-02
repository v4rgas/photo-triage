"""Stage 1 -- walk the folder and record what each image actually is.

Produces the `index.jsonl` row set that fixes every row id for the rest of the
pipeline. Everything here is decided by looking at file contents rather than
file names, because in a messaging-app export the names carry no information
and the extensions are frequently wrong (PROJECT.md 9.6).

Re-running is incremental: a file whose (path, size, mtime) is unchanged keeps
its existing record and, crucially, its existing row id. New files are appended.
A file that has vanished keeps its row, flagged `missing`, so that no row id
ever shifts underneath `embeds.npy`.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageFile

from .cache import CACHE_DIRNAME, Cache, MediaRecord
from .video import looks_like_video, probe, sample, sample_count

log = logging.getLogger(__name__)

# Panoramas and scanner output routinely exceed Pillow's decompression-bomb
# guard, and this tool only ever reads files the user already has.
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# First bytes of every container we are willing to open. The extension is not
# consulted at all: this is what makes a WebP named `.jpg` scan correctly.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "JPEG"),
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"GIF87a", "GIF"),
    (b"GIF89a", "GIF"),
    (b"BM", "BMP"),
    (b"II*\x00", "TIFF"),
    (b"MM\x00*", "TIFF"),
)
_HEADER_BYTES = 32


def scan(
    cache: Cache, progress: Callable[[int, int], None] | None = None
) -> list[MediaRecord]:
    """Index every image under the cache's root and persist the result.

    `progress(done, total)` is called as work completes, where `total` counts
    only the files that actually need decoding this run. Returns the full row
    set in row order, including rows carried over unchanged.

    An image no decoder can read is recorded with an `error` and no hashes
    rather than being dropped, so the user can find the unreadable bucket in
    the UI instead of wondering why the counts do not add up.
    """
    previous = cache.load_index()
    by_key = {r.key: r for r in previous}
    known_rel = {r.rel: r for r in previous}

    found: list[tuple[str, int, float]] = []
    for path in _walk_images(cache.root):
        stat = path.stat()
        rel = path.relative_to(cache.root).as_posix()
        found.append((rel, stat.st_size, stat.st_mtime))

    fresh: dict[str, MediaRecord] = {}
    pending: list[tuple[str, int, float]] = []
    for rel, size, mtime in found:
        carried = by_key.get((rel, size, mtime))
        if carried is not None:
            fresh[rel] = carried
        else:
            pending.append((rel, size, mtime))

    done = 0
    if pending:
        # Decoding is the whole cost of this stage and it is CPU-bound, so it
        # goes to processes rather than threads.
        with ProcessPoolExecutor() as pool:
            args = [(str(cache.root), rel, size, mtime) for rel, size, mtime in pending]
            for record in pool.map(_inspect, args, chunksize=16):
                fresh[record.rel] = record
                done += 1
                if progress:
                    progress(done, len(pending))
    elif progress:
        progress(0, 0)

    records = _merge(previous, known_rel, fresh)
    cache.save_index(records)
    return records


def _merge(
    previous: list[MediaRecord],
    known_rel: dict[str, MediaRecord],
    fresh: dict[str, MediaRecord],
) -> list[MediaRecord]:
    """Rebuild the row set, preserving every existing row id.

    An existing row is updated if its file was re-inspected and otherwise left
    exactly as it was -- absence from this scan is the *normal* state of a
    quarantined image, not evidence that anything is wrong, and a row is never
    dropped because doing so would renumber every row after it. Genuinely new
    paths are appended in sorted order, so two scans of the same folder assign
    the same ids.
    """
    records = list(previous)
    for row, record in enumerate(records):
        replacement = fresh.get(record.rel)
        if replacement is not None:
            records[row] = replacement
    for rel in sorted(set(fresh) - set(known_rel)):
        records.append(fresh[rel])
    return records


def _walk_images(root: Path) -> Iterator[Path]:
    """Every file under `root` whose first bytes look like an image container.

    Skips `.phototriage/` entirely -- otherwise the tool indexes its own
    thumbnails and everything the user has already binned.
    """
    for path in sorted(root.rglob("*")):
        if CACHE_DIRNAME in path.parts or not path.is_file():
            continue
        if _sniff(path) is not None:
            yield path


def _sniff(path: Path) -> str | None:
    """The container format from magic bytes, or None if this is not media.

    Stills and video are both recognised here, from one read of the header, so
    the walk asks the question once per file.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(_HEADER_BYTES)
    except OSError:
        return None
    for magic, fmt in _MAGIC:
        if head.startswith(magic):
            return fmt
    # RIFF....WEBP -- the four size bytes in between are why this is not a
    # simple prefix match. AVI shares the RIFF prefix, so video is asked last.
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "WEBP"
    if head[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1", b"ftypavif"):
        return "HEIF"
    if looks_like_video(head):
        return "VIDEO"
    return None


def _inspect(args: tuple[str, str, int, float]) -> MediaRecord:
    """Decode one file and measure it. Runs in a worker process.

    Any failure becomes an `error` on the record. One unreadable file among
    twenty thousand must never take down the batch, so nothing here raises.
    """
    root, rel, size, mtime = args
    path = Path(root) / rel
    record = MediaRecord(rel=rel, size=size, mtime=mtime)
    try:
        if _sniff(path) == "VIDEO":
            _measure_video(path, record)
        else:
            _measure_still(path, record)
    except Exception as exc:
        record.error = f"unreadable: {type(exc).__name__}"
    return record


def _measure_still(path: Path, record: MediaRecord) -> None:
    with Image.open(path) as im:
        record.fmt = im.format or ""
        # Animated GIFs and multi-frame TIFFs: judge the first frame.
        if getattr(im, "n_frames", 1) > 1:
            im.seek(0)
        record.width, record.height = im.size
        _hash_frame(im.convert("RGB"), record)


def _measure_video(path: Path, record: MediaRecord) -> None:
    """Measure a clip, and hash one frame of it so dedupe still works.

    The hashed frame is the middle sample rather than the first, because the
    opening frame of a phone video is very often black and every such video
    would then hash identically to every other.
    """
    info = probe(path)
    if info is None:
        record.error = "unreadable: no video stream"
        return
    record.fmt = path.suffix.lstrip(".").upper() or "VIDEO"
    record.width, record.height = info.width, info.height
    record.duration = info.duration
    record.segments = sample_count(info.duration)

    frames = sample(path, record.segments)
    if not frames:
        record.error = "unreadable: no frames decoded"
        return
    record.segments = len(frames)
    _hash_frame(frames[len(frames) // 2].convert("RGB"), record)


def _hash_frame(rgb: Image.Image, record: MediaRecord) -> None:
    import imagehash

    record.phash = str(imagehash.phash(rgb))
    record.dhash = str(imagehash.dhash(rgb))
    # Hash the decoded pixels, not the file: two files can be pixel-identical
    # yet differ on disk through EXIF or encoder choice, and comparing file
    # bytes finds none of those.
    record.pixel_md5 = hashlib.md5(rgb.tobytes()).hexdigest()
