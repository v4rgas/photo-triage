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

from PIL import Image

from .cache import Cache, ImageRecord
from .quarantine import Quarantine

log = logging.getLogger(__name__)

THUMB_PX = 256
_QUALITY = 82


def build_thumbnails(
    cache: Cache,
    records: list[ImageRecord],
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
    todo = [
        (row, where.location(row))
        for row, record in enumerate(records)
        if record.readable and not cache.thumb_path(row).exists()
    ]
    if not todo:
        if progress:
            progress(0, 0)
        return 0

    made = 0
    with ThreadPoolExecutor() as pool:
        for ok in pool.map(
            lambda item: _render(item[1], cache.thumb_path(item[0])), todo
        ):
            made += ok
            if progress:
                progress(made, len(todo))
    log.info("wrote %d thumbnails", made)
    return made


def _render(source: Path, target: Path) -> bool:
    """Write one thumbnail. False if the source could not be read."""
    try:
        with Image.open(source) as im:
            if getattr(im, "n_frames", 1) > 1:
                im.seek(0)
            im = im.convert("RGB")
            im.thumbnail((THUMB_PX, THUMB_PX), Image.Resampling.LANCZOS)
            target.parent.mkdir(parents=True, exist_ok=True)
            im.save(target, "JPEG", quality=_QUALITY, optimize=True)
        return True
    except Exception as exc:
        log.debug("no thumbnail for %s: %s", source, exc)
        return False
