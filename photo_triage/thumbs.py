"""Stage 4 -- 256px JPEGs so the grid is instant.

The UI never opens an original. At 24,000 rows the difference between a 40 KB
thumbnail and a 4 MB photograph is the difference between a grid that scrolls
at 60fps and one that does not, and DESIGN.md names frame timing as the primary
aesthetic.

Thumbnails are keyed by row id alone. That is deliberate: it means a rename or
a quarantine move never invalidates one, since the row id is the thing that
does not change.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .cache import Cache, MediaRecord, segment_spans
from .quarantine import Quarantine
from .video import sample

log = logging.getLogger(__name__)

THUMB_PX = 256
_QUALITY = 82


def build_thumbnails(
    cache: Cache,
    records: list[MediaRecord],
    embeds: np.ndarray | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Generate any thumbnail that does not exist yet. Returns how many it made.

    Resumable and cheap to re-run: an existing thumbnail is never regenerated,
    and an image that cannot be decoded is skipped without stopping the pass.
    Decoding releases the GIL, so threads are enough here and they avoid the
    process-startup cost of a pool that is usually idle.
    """
    cache.thumbs_dir.mkdir(parents=True, exist_ok=True)
    where = Quarantine(cache, records)
    spans = segment_spans(records)
    todo = []
    for row, record in enumerate(records):
        if not record.readable or cache.thumb_path(row).exists():
            continue
        key = None
        if record.is_video and embeds is not None:
            start, stop = spans[row]
            key = embeds[start:stop]
        todo.append((row, where.location(row), key))
    if not todo:
        if progress:
            progress(0, 0)
        return 0

    made = 0
    with ThreadPoolExecutor() as pool:
        for ok in pool.map(
            lambda item: _render(item[1], cache.thumb_path(item[0]), item[2]), todo
        ):
            made += ok
            if progress:
                progress(made, len(todo))
    log.info("wrote %d thumbnails", made)
    return made


def _render(source: Path, target: Path, key: np.ndarray | None = None) -> bool:
    """Write one thumbnail. False if the source could not be read.

    For a video, `key` is that clip's segment vectors, and the frame chosen is
    the one closest to their mean: the most representative moment of the clip
    rather than whatever happened to be on screen first. Phone videos very
    often open on a black or still-focusing frame, so frame zero is close to
    the worst possible choice and costs nothing extra to avoid.
    """
    try:
        picture = _pick_frame(source, key)
        if picture is None:
            return False
        picture = picture.convert("RGB")
        picture.thumbnail((THUMB_PX, THUMB_PX), Image.Resampling.LANCZOS)
        target.parent.mkdir(parents=True, exist_ok=True)
        picture.save(target, "JPEG", quality=_QUALITY, optimize=True)
        return True
    except Exception as exc:
        log.debug("no thumbnail for %s: %s", source, exc)
        return False


def _pick_frame(source: Path, key: np.ndarray | None) -> Image.Image | None:
    if key is None:
        with Image.open(source) as im:
            if getattr(im, "n_frames", 1) > 1:
                im.seek(0)
            return im.convert("RGB")

    frames = sample(source, len(key))
    if not frames:
        return None
    usable = min(len(frames), len(key))
    vectors = key[:usable]
    if not vectors.any():
        return frames[usable // 2]
    mean = vectors.mean(axis=0)
    return frames[int(np.argmax(vectors[:usable] @ mean))]
