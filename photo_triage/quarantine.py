"""Where every image currently is, and how it got there.

This module owns three design decisions and is the only place any of them is
written down:

1. **The mirroring rule.** A quarantined image lives at
   `.phototriage/quarantine/<rel>`, the same relative path it had under the
   scan root. That is what makes restore unambiguous for duplicate basenames
   and makes the bin recoverable by hand in a file manager. It appears exactly
   once, in `_quarantine_path`.
2. **What the journal means.** `journal.jsonl` is the source of truth for
   protection and for whether a row is on disk or in the bin; both are replayed
   from it at construction, never loaded from a derived file.
3. **Protection is enforced here**, below the HTTP layer, so a POST straight to
   the endpoint cannot bypass it (PROJECT.md 7).

Moves are `rename()` within one filesystem: atomic and instantaneous, no
copying and no half-written files. That is the reason quarantine lives inside
the target folder rather than somewhere tidier.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from .cache import Cache, ImageRecord

log = logging.getLogger(__name__)

ACTIVE = "active"
QUARANTINED = "quarantined"
PURGED = "purged"


@dataclass
class MoveResult:
    """What a move actually did.

    `moved` is the rows that changed state. `refused` maps a row to a short
    reason it did not -- protected, already there, or destination occupied.
    Requests are never silently dropped: every id handed in appears in exactly
    one of the two.
    """

    moved: list[int] = field(default_factory=list)
    refused: dict[int, str] = field(default_factory=dict)


@dataclass
class PurgeReport:
    """What a purge found, or removed if it was not a dry run."""

    rows: list[int] = field(default_factory=list)
    bytes_freed: int = 0
    dry_run: bool = True


class Quarantine:
    """Tracks and changes the location and protection of every indexed row.

    Construct with the folder's cache and its index; history is replayed
    immediately, so a fresh instance always reflects everything that has ever
    happened to the folder. Instances are cheap and stateless beyond that
    replay -- the server holds one for the life of the process and mutates it.

    All four mutators are idempotent. Quarantining a row that is already in the
    bin, restoring one that is already back, or protecting one that is already
    protected are all successes that change nothing, so the UI never has to
    reason about whether an action is still applicable.
    """

    def __init__(self, cache: Cache, records: list[ImageRecord]):
        self.cache = cache
        self._rel = [r.rel for r in records]
        self.state: list[str] = [ACTIVE] * len(records)
        self.saved: set[int] = set()
        self._replay()

    # -- queries ----------------------------------------------------------

    def location(self, row: int) -> Path:
        """Where the file for `row` is right now, whether or not it exists.

        A purged row reports its quarantine path, which no longer exists; that
        keeps callers from needing a fourth case for a file that is simply gone.
        """
        if self.state[row] == ACTIVE:
            return self.cache.root / self._rel[row]
        return self._quarantine_path(row)

    def quarantined_rows(self) -> list[int]:
        return [row for row, st in enumerate(self.state) if st == QUARANTINED]

    def batches(self) -> list[dict]:
        """Every quarantine batch, oldest first, as `{"t": float, "ids": [...]}`.

        This is what makes undo work over any batch rather than only the last
        one (PROJECT.md 9.3): the caller picks a batch and restores its ids.
        """
        return [
            {"t": entry["t"], "ids": entry["ids"]}
            for entry in self.cache.read_journal()
            if entry.get("op") == "quarantine"
        ]

    # -- mutators ---------------------------------------------------------

    def quarantine(self, rows: list[int]) -> MoveResult:
        """Move rows into the mirrored bin. Protected rows are refused."""
        result = MoveResult()
        for row in self._valid(rows, result):
            if row in self.saved:
                result.refused[row] = "protected"
                continue
            if self.state[row] != ACTIVE:
                result.refused[row] = "already quarantined"
                continue
            source = self.cache.root / self._rel[row]
            target = self._quarantine_path(row)
            if not self._move(source, target, row, result):
                continue
            self.state[row] = QUARANTINED
            result.moved.append(row)
        self._record("quarantine", result.moved)
        return result

    def restore(self, rows: list[int]) -> MoveResult:
        """Move rows back to their original paths, from any batch."""
        result = MoveResult()
        for row in self._valid(rows, result):
            if self.state[row] == ACTIVE:
                result.refused[row] = "not quarantined"
                continue
            if self.state[row] == PURGED:
                result.refused[row] = "purged"
                continue
            source = self._quarantine_path(row)
            target = self.cache.root / self._rel[row]
            if target.exists():
                result.refused[row] = "a file already occupies the original path"
                continue
            if not self._move(source, target, row, result):
                continue
            self.state[row] = ACTIVE
            result.moved.append(row)
        self._record("restore", result.moved)
        return result

    def set_saved(self, rows: list[int], protected: bool) -> MoveResult:
        """Protect or unprotect rows. Protection outranks any later delete."""
        result = MoveResult()
        for row in self._valid(rows, result):
            if (row in self.saved) == protected:
                result.refused[row] = "already protected" if protected else "not protected"
                continue
            self.saved.add(row) if protected else self.saved.discard(row)
            result.moved.append(row)
        self._record("save" if protected else "unsave", result.moved)
        if result.moved:
            self.cache.save_saved_snapshot(self.saved)
        return result

    def purge(self, dry_run: bool = True) -> PurgeReport:
        """Permanently delete the bin's contents. The one irreversible operation.

        Deletion happens file by file against the rows this instance believes
        are quarantined, never as a recursive wipe of the directory, so a stale
        or foreign file under `quarantine/` is left alone rather than destroyed
        on the strength of its location. Verify-before-destroy (PROJECT.md 9.7):
        the caller is expected to show a dry run and take a confirmation first.
        """
        report = PurgeReport(dry_run=dry_run)
        for row in self.quarantined_rows():
            path = self._quarantine_path(row)
            if not path.exists():
                continue
            report.rows.append(row)
            report.bytes_freed += path.stat().st_size
            if not dry_run:
                path.unlink()
                self.state[row] = PURGED
        if not dry_run and report.rows:
            _prune_empty_dirs(self.cache.quarantine_dir)
            self._record("purge", report.rows)
        return report

    # -- internals --------------------------------------------------------

    def _quarantine_path(self, row: int) -> Path:
        """The mirroring rule. The only expression of it in the codebase."""
        return self.cache.quarantine_dir / self._rel[row]

    def _valid(self, rows: list[int], result: MoveResult) -> list[int]:
        """Deduplicate and drop ids that name no row, recording why."""
        seen: list[int] = []
        for row in rows:
            if not isinstance(row, int) or not 0 <= row < len(self.state):
                result.refused[row] = "no such image"
            elif row not in seen:
                seen.append(row)
        return seen

    def _move(self, source: Path, target: Path, row: int, result: MoveResult) -> bool:
        """Rename source to target, reporting failure against `row`.

        A source that has already vanished counts as done rather than as an
        error: the user asked for the file not to be at that path, and it isn't.
        """
        if not source.exists():
            return True
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
        except OSError as exc:
            # Different filesystem, or a permissions problem. Copy-then-remove is
            # the honest fallback; it is slower but it is what the user asked for.
            try:
                shutil.move(str(source), str(target))
            except OSError:
                log.warning("could not move %s -> %s: %s", source, target, exc)
                result.refused[row] = "move failed"
                return False
        return True

    def _record(self, op: str, rows: list[int]) -> None:
        if rows:
            self.cache.append_journal(op, sorted(rows), time.time())

    def _replay(self) -> None:
        """Rebuild location and protection from the journal, oldest entry first."""
        apply = {
            "quarantine": lambda row: self._set_state(row, QUARANTINED),
            "restore": lambda row: self._set_state(row, ACTIVE),
            "purge": lambda row: self._set_state(row, PURGED),
            "save": self.saved.add,
            "unsave": self.saved.discard,
        }
        for entry in self.cache.read_journal():
            step = apply.get(entry.get("op", ""))
            if step is None:
                continue
            for row in entry.get("ids", []):
                if isinstance(row, int) and 0 <= row < len(self.state):
                    step(row)

    def _set_state(self, row: int, state: str) -> None:
        self.state[row] = state


def _prune_empty_dirs(top: Path) -> None:
    """Remove directories left empty under `top`, deepest first. `top` survives."""
    if not top.is_dir():
        return
    for path in sorted(top.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
