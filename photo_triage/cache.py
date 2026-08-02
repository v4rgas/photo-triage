"""The on-disk cache for one triaged folder.

This module is the *only* place that knows what `.phototriage/` looks like:
which files exist, what shape they have, and how a row id relates to them.
Every other stage asks for records, a matrix or scores and never touches a
filename or a dtype. Change a format here and nowhere else.

The invariant the whole system rests on
---------------------------------------
An image's **row id** is its position in `index.jsonl`, and the same integer is
its row in `embeds.npy`, its key in `scored.json` and the stem of its
thumbnail. Rows are only ever appended; a file that disappears from disk keeps
its row (flagged `missing`) so that nothing downstream shifts underneath the
matrix. Nothing in this codebase may re-sort one of these structures alone.

Not-yet-embedded rows are stored as an all-zero vector. Real embeddings are
L2-normalised and so can never be all-zero, which means "which rows still need
work?" is answered by the matrix itself rather than by a second bookkeeping
file that could disagree with it.

Design it twice
---------------
The alternative considered was a single SQLite database holding every per-image
fact, which would have given transactions, ad-hoc queries and referential
integrity for nothing. It was rejected because the embedding matrix has to
reach numpy as one contiguous N x 512 float32 block for the similarity matmul:
in SQLite it would live as N blobs that are reassembled on every start, and the
"row id *is* the position" invariant would degrade into a join that code can
forget to perform. Positional files make the invariant physically true, let the
matrix be memory-mapped, and stay recoverable by hand with `head` and `python
-c` if the tool itself breaks -- which is the same reason quarantine is a plain
directory tree rather than a container format.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

CACHE_DIRNAME = ".phototriage"
EMBED_DIM = 512  # ViT-B-32 output width; see PROJECT.md 4.


@dataclass
class ImageRecord:
    """One image as stage 1 found it.

    `rel` is a POSIX-style path relative to the scan root and is the stable
    human-facing identity of the image -- it survives quarantine, since the
    quarantine tree mirrors it exactly. `size` and `mtime` come from the file
    as scanned and together with `rel` form the key that makes re-scanning
    incremental.

    `fmt` is sniffed from the file's magic bytes, not its extension, so a
    `.jpg` that is really a WebP reports `WEBP` (PROJECT.md 9.6).

    `error` is None for a readable image, otherwise a short reason string; such
    a record has no dimensions and no hashes and is never embedded, but it
    still occupies a row so that the row ids of its neighbours never move.
    """

    rel: str
    size: int
    mtime: float
    fmt: str = ""
    width: int = 0
    height: int = 0
    phash: str = ""
    dhash: str = ""
    pixel_md5: str = ""
    error: str | None = None

    @property
    def key(self) -> tuple[str, int, float]:
        """Identity for incremental re-scans: unchanged key means unchanged file."""
        return (self.rel, self.size, self.mtime)

    @property
    def readable(self) -> bool:
        return self.error is None


@dataclass
class Scores:
    """Stage 3's verdict for every row, in row order.

    `category[row]` is a category name from prompts.py, or "" for a row that was
    never classified (unreadable, or not yet embedded). `confidence[row]` is the
    softmax probability of that category in [0, 1] -- a *relative* figure across
    the category set, and emphatically not a probability of being correct
    (PROJECT.md 9.1).
    """

    category: list[str] = field(default_factory=list)
    confidence: list[float] = field(default_factory=list)


class Cache:
    """Reads and writes the `.phototriage/` directory of one triaged folder.

    Construct with the folder the user pointed at; the cache directory is
    created lazily on first write, so merely opening a Cache leaves no trace.
    Every writer here is atomic (temp file plus `os.replace`), so an interrupt
    at any moment leaves the previous good version in place rather than a
    half-written one.

    Missing files are not an error anywhere in this class: an unwritten cache
    reads back as an empty index, a zero-row matrix and empty scores, which is
    exactly the state a fresh folder is in.
    """

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.dir = self.root / CACHE_DIRNAME
        self.thumbs_dir = self.dir / "thumbs"
        self.quarantine_dir = self.dir / "quarantine"
        self._index_path = self.dir / "index.jsonl"
        self._embeds_path = self.dir / "embeds.npy"
        self._paths_path = self.dir / "paths.json"
        self._scored_path = self.dir / "scored.json"
        self._saved_path = self.dir / "saved.json"
        self._reviewed_path = self.dir / "reviewed.json"
        self.journal_path = self.dir / "journal.jsonl"

    # -- index ------------------------------------------------------------

    def load_index(self) -> list[ImageRecord]:
        """Every scanned image, in row order. Empty list if never scanned."""
        if not self._index_path.exists():
            return []
        records = []
        for line in self._index_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(ImageRecord(**json.loads(line)))
        return records

    def save_index(self, records: list[ImageRecord]) -> None:
        """Replace the index, and with it the row id -> path map.

        Both files are rewritten together because they encode the same fact;
        writing one without the other is how the central invariant gets broken.
        """
        body = "".join(
            json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in records
        )
        self._write_atomic(self._index_path, body.encode("utf-8"))
        self._write_atomic(
            self._paths_path,
            json.dumps([r.rel for r in records], ensure_ascii=False).encode("utf-8"),
        )

    # -- embeddings -------------------------------------------------------

    def load_embeddings(self, row_count: int) -> np.ndarray:
        """An `row_count` x 512 float32 matrix of L2-normalised image vectors.

        Rows that have not been embedded yet -- including every row added since
        the last embed pass -- read back as zeros. A cache that is truncated,
        the wrong width or otherwise unreadable is treated as partial rather
        than as corruption: whatever survives is kept and the rest comes back
        as zeros, so the next embed pass simply refills them.
        """
        blank = np.zeros((row_count, EMBED_DIM), dtype=np.float32)
        if not self._embeds_path.exists():
            return blank
        try:
            stored = np.load(self._embeds_path)
        except Exception as exc:
            log.warning("embeds.npy unreadable (%s); re-embedding from scratch", exc)
            return blank
        if stored.ndim != 2 or stored.shape[1] != EMBED_DIM:
            log.warning("embeds.npy has shape %s; re-embedding from scratch", stored.shape)
            return blank
        keep = min(len(stored), row_count)
        blank[:keep] = stored[:keep].astype(np.float32, copy=False)
        return blank

    def save_embeddings(self, embeds: np.ndarray) -> None:
        self._write_atomic(self._embeds_path, _npy_bytes(embeds))

    # -- scores -----------------------------------------------------------

    def load_scores(self, row_count: int) -> Scores:
        """Stage 3's output, padded to `row_count`. Unclassified rows read as ""."""
        scores = Scores([""] * row_count, [0.0] * row_count)
        if not self._scored_path.exists():
            return scores
        try:
            raw = json.loads(self._scored_path.read_text(encoding="utf-8"))
            stored_cat = raw["category"]
            stored_conf = raw["confidence"]
        except Exception as exc:
            log.warning("scored.json unreadable (%s); re-classify to rebuild", exc)
            return scores
        keep = min(len(stored_cat), len(stored_conf), row_count)
        scores.category[:keep] = stored_cat[:keep]
        scores.confidence[:keep] = stored_conf[:keep]
        return scores

    def save_scores(self, scores: Scores) -> None:
        body = json.dumps(
            {"category": scores.category, "confidence": scores.confidence}
        )
        self._write_atomic(self._scored_path, body.encode("utf-8"))

    def save_saved_snapshot(self, saved_rows: set[int]) -> None:
        """Write `saved.json` for humans and outside tools.

        Deliberately never read back. Protection is replayed from the journal,
        which is the single source of truth (PROJECT.md 5); a second readable
        copy of the same fact is a second copy that can disagree.
        """
        self._write_atomic(
            self._saved_path, json.dumps(sorted(saved_rows)).encode("utf-8")
        )

    def load_reviewed(self) -> set[str]:
        """Category names the user has looked through in review mode.

        Gates the bulk actions the prototype made too easy (PROJECT.md 9.2).
        Absent file means nothing has been reviewed, which is the correct state
        for a folder opened for the first time.
        """
        if not self._reviewed_path.exists():
            return set()
        try:
            return set(json.loads(self._reviewed_path.read_text(encoding="utf-8")))
        except Exception:
            return set()

    def save_reviewed(self, categories: set[str]) -> None:
        self._write_atomic(
            self._reviewed_path, json.dumps(sorted(categories)).encode("utf-8")
        )

    # -- journal ----------------------------------------------------------

    def append_journal(self, op: str, ids: list[int], t: float) -> None:
        """Append one operation to the append-only history. Creates it if absent."""
        self.dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"op": op, "ids": ids, "t": t}) + "\n"
        with open(self.journal_path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

    def read_journal(self) -> list[dict]:
        """Every operation ever performed, oldest first. Empty if none."""
        if not self.journal_path.exists():
            return []
        entries = []
        for line in self.journal_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                # A torn final line from a kill mid-write. Everything before it
                # is still valid history, which is the point of append-only.
                log.warning("ignoring malformed journal line")
        return entries

    # -- thumbnails -------------------------------------------------------

    def thumb_path(self, row: int) -> Path:
        return self.thumbs_dir / f"{row}.jpg"

    # -- internals --------------------------------------------------------

    def _write_atomic(self, path: Path, body: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)


def _npy_bytes(array: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    return buf.getvalue()
