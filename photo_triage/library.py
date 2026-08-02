"""One triaged folder, loaded and queryable.

Everything the UI and the CLI ask about a folder is asked here: which images
match a set of filters, in what order, with what metadata, and what the counts
are. The stages above write files; this reads them and answers questions.

The point of the module is that there is exactly **one** filter implementation.
`/api/search` and `/api/ids` are the same call with and without a page window,
so the number the user reads and the set "select all matching" acts on cannot
drift apart -- which is precisely how someone deletes more than they meant to
(PROJECT.md 7).

Ranking is by cosine similarity and is deliberately never thresholded. There is
no universal cutoff that means "similar enough"; a sorted list always has a
best answer at the top, and an arbitrary floor only ever hides it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .cache import Cache, ImageRecord, Scores
from .prompts import GUESSING_BELOW, group_of
from .quarantine import ACTIVE, Quarantine

log = logging.getLogger(__name__)


@dataclass
class Query:
    """A set of filters plus an ordering. Every field defaults to "no opinion".

    `text` and `like` are ranking signals rather than filters: they reorder the
    matching set, they never shrink it. `like` is a row id whose own vector is
    the query, which is how "find more like this" clears a meme and every
    re-send of it in one sweep.

    `show` is "unsaved" (the default -- protected images drop out, so the pile
    still to be judged shrinks with every pass), "saved" or "all". `view` is
    "active" for the folder or "quarantine" for the bin, which is searchable
    with these same filters so the user can audit what they threw away.
    """

    text: str = ""
    like: int | None = None
    category: str = ""
    folder: str = ""
    group: str = ""
    show: str = "unsaved"
    view: str = "active"
    order: str = "relevance"  # or "confidence", for review mode


@dataclass
class Tile:
    """One image as the UI needs it. Field names match the JSON sent over HTTP."""

    id: int
    rel: str
    folder: str
    name: str
    width: int
    height: int
    category: str
    group: str
    confidence: float
    score: float | None
    saved: bool
    quarantined: bool
    error: str | None


@dataclass
class Stats:
    """Counts the toolbar shows, and the per-category totals review mode needs."""

    total: int = 0
    active: int = 0
    quarantined: int = 0
    saved: int = 0
    unreadable: int = 0
    embedded: int = 0
    categories: dict[str, dict] = field(default_factory=dict)
    reviewed: list[str] = field(default_factory=list)
    folders: list[str] = field(default_factory=list)


class Library:
    """The loaded state of one folder, and every question that can be asked of it.

    Open it with `Library.open(root)`. Construction reads the whole cache,
    including the ~48 MB embedding matrix for a 24,000-image folder, so hold one
    instance rather than reopening per request.

    An empty or absent cache is not an error: an unscanned folder opens as a
    library with nothing in it, and every query returns an empty list.
    """

    def __init__(self, cache: Cache, encode_text=None):
        self.cache = cache
        self.records: list[ImageRecord] = cache.load_index()
        self.embeds: np.ndarray = cache.load_embeddings(len(self.records))
        self.scores: Scores = cache.load_scores(len(self.records))
        self.where = Quarantine(cache, self.records)
        self.encode_text = encode_text
        self._folders = sorted({_folder_of(r.rel) for r in self.records})

    @classmethod
    def open(cls, root: Path, encode_text=None) -> "Library":
        return cls(Cache(root), encode_text)

    @property
    def root(self) -> Path:
        return self.cache.root

    # -- queries ----------------------------------------------------------

    def select(self, query: Query) -> list[int]:
        """Every row matching `query`, in the order the UI should show them.

        This is the single filter implementation. Whatever else changes, the
        page endpoint and the select-all endpoint must both come through here.
        """
        rows = [row for row in range(len(self.records)) if self._matches(row, query)]
        ranking = self._ranking(query)
        if ranking is not None:
            rows.sort(key=lambda row: -ranking[row])
        elif query.order == "confidence":
            rows.sort(key=lambda row: -self.scores.confidence[row])
        return rows

    def page(self, query: Query, limit: int, offset: int) -> tuple[list[Tile], int]:
        """A window of `select`, plus the full match count it was drawn from.

        The count is what the UI must show next to the window, so that "showing
        300" and "1,022 match" always describe the same set.
        """
        rows = self.select(query)
        ranking = self._ranking(query)
        window = rows[offset : offset + limit]
        return (
            [self.tile(row, None if ranking is None else float(ranking[row]))
             for row in window],
            len(rows),
        )

    def tile(self, row: int, score: float | None = None) -> Tile:
        record = self.records[row]
        category = self.scores.category[row]
        return Tile(
            id=row,
            rel=record.rel,
            folder=_folder_of(record.rel),
            name=record.rel.rsplit("/", 1)[-1],
            width=record.width,
            height=record.height,
            category=category,
            group=group_of(category),
            confidence=self.scores.confidence[row],
            score=score,
            saved=row in self.where.saved,
            quarantined=self.where.state[row] != ACTIVE,
            error=record.error,
        )

    def stats(self) -> Stats:
        """Counts for the toolbar and for review mode's gating.

        Category counts cover the *active* folder only: a category's bin
        contents are not what the user is deciding about when they read them.
        """
        stats = Stats(
            total=len(self.records),
            saved=len(self.where.saved),
            reviewed=sorted(self.cache.load_reviewed()),
            folders=list(self._folders),
        )
        for row, record in enumerate(self.records):
            active = self.where.state[row] == ACTIVE
            stats.active += active
            stats.quarantined += not active
            stats.unreadable += not record.readable
            stats.embedded += bool(self.embeds[row].any())
            if not active:
                continue
            category = self.scores.category[row] or "unclassified"
            bucket = stats.categories.setdefault(
                category,
                {"count": 0, "group": group_of(self.scores.category[row]),
                 "above_threshold": 0},
            )
            bucket["count"] += 1
            bucket["above_threshold"] += self.scores.confidence[row] >= GUESSING_BELOW
        return stats

    def mark_reviewed(self, category: str) -> None:
        """Record that the user has looked at a whole category in review mode."""
        reviewed = self.cache.load_reviewed()
        reviewed.add(category)
        self.cache.save_reviewed(reviewed)

    # -- internals --------------------------------------------------------

    def _matches(self, row: int, query: Query) -> bool:
        record = self.records[row]
        quarantined = self.where.state[row] != ACTIVE
        if quarantined != (query.view == "quarantine"):
            return False
        saved = row in self.where.saved
        if query.show == "unsaved" and saved:
            return False
        if query.show == "saved" and not saved:
            return False
        # An empty filter is the filter that matches everything, so there is no
        # "is a filter active?" branch anywhere above this.
        if query.category and self.scores.category[row] != query.category:
            return False
        if query.group and group_of(self.scores.category[row]) != query.group:
            return False
        if query.folder and _folder_of(record.rel) != query.folder:
            return False
        return True

    def _ranking(self, query: Query) -> np.ndarray | None:
        """Cosine similarity of every row against the query, or None if unranked.

        Both sides are L2-normalised, so this is a plain dot product; a row with
        no embedding scores -1 and therefore sinks rather than being dropped.
        """
        vector = None
        if query.like is not None and 0 <= query.like < len(self.records):
            vector = self.embeds[query.like]
            if not vector.any():
                vector = None
        elif query.text.strip() and self.encode_text is not None:
            vector = self.encode_text(query.text.strip())
        if vector is None:
            return None
        similarity = self.embeds @ vector
        similarity[~self.embeds.any(axis=1)] = -1.0
        return similarity


def _folder_of(rel: str) -> str:
    """The containing directory of a relative path; "" for the scan root itself."""
    return rel.rsplit("/", 1)[0] if "/" in rel else ""
