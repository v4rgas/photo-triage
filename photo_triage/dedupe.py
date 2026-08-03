"""Redundancy: the same picture, more than once.

Three kinds, deliberately kept apart because they warrant different levels of
trust:

- **exact** -- identical decoded pixels. Two files can be pixel-identical and
  still differ byte for byte through EXIF or a different encoder, so this
  compares the hash of the decoded buffer, never the file. Safe to act on.
- **near** -- close perceptual hashes. Catches recompressed and resized
  re-sends, but near-identical is not identical: burst shots and slightly
  different crops land here and are legitimately separate photographs. Always
  needs a human.
- **resend** -- a near group whose members' file dates are more than six months
  apart. Worth surfacing separately because the newer copy is usually the
  *worse* one, having been through another round of compression -- the opposite
  of what a "keep the newest" rule would do (PROJECT.md 9.5).

Nothing here deletes anything. It reports groups and names a suggested keeper;
acting on that is the caller's decision, and the CLI defaults to a dry run.
"""

from __future__ import annotations

from dataclasses import dataclass

from .library import Library
from .quarantine import ACTIVE

NEAR_DISTANCE = 6  # Hamming distance between 64-bit phashes, out of 64.
RESEND_GAP_SECONDS = 182 * 24 * 3600  # ~6 months


@dataclass
class DuplicateGroup:
    """Rows holding the same picture, with the one worth keeping named first.

    `keeper` is a suggestion, chosen as earliest date, then largest file, then
    shortest name -- largest usually means least re-compressed. `kind` is
    "exact", "near" or "resend"; only "exact" is safe to act on unseen.
    """

    kind: str
    keeper: int
    others: list[int]

    @property
    def rows(self) -> list[int]:
        return [self.keeper, *self.others]


def find_duplicates(library: Library) -> list[DuplicateGroup]:
    """Every redundant group in the folder, exact first, then near, then resends.

    Only active, readable images are considered: what is already in the bin is
    not a duplicate the user needs to decide about.
    """
    rows = [
        row
        for row, record in enumerate(library.records)
        if record.readable and library.where.state[row] == ACTIVE
    ]
    groups = [
        _group(library, members, "exact")
        for members in _by_exact_pixels(library, rows).values()
        if len(members) > 1
    ]
    already = {row for group in groups for row in group.rows}
    for members in _by_near_hash(library, [r for r in rows if r not in already]):
        kind = "resend" if _spans_years(library, members) else "near"
        groups.append(_group(library, members, kind))
    order = {"exact": 0, "resend": 1, "near": 2}
    groups.sort(key=lambda g: (order[g.kind], -len(g.others)))
    return groups


def _by_exact_pixels(library: Library, rows: list[int]) -> dict[str, list[int]]:
    buckets: dict[str, list[int]] = {}
    for row in rows:
        digest = library.records[row].pixel_digest
        if digest:
            buckets.setdefault(digest, []).append(row)
    return buckets


def _by_near_hash(library: Library, rows: list[int]) -> list[list[int]]:
    """Cluster rows whose perceptual hashes are within NEAR_DISTANCE.

    Rows sharing an exact hash are bucketed first, which collapses the common
    case cheaply; only the distinct hashes are then compared pairwise, so the
    quadratic part runs over distinct pictures rather than over all files.
    """
    by_hash: dict[int, list[int]] = {}
    for row in rows:
        phash = library.records[row].phash
        if phash and not _degenerate(int(phash, 16)):
            by_hash.setdefault(int(phash, 16), []).append(row)

    hashes = list(by_hash)
    parent = list(range(len(hashes)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(hashes)):
        for right in range(left + 1, len(hashes)):
            if bin(hashes[left] ^ hashes[right]).count("1") <= NEAR_DISTANCE:
                parent[root(left)] = root(right)

    clusters: dict[int, list[int]] = {}
    for index, phash in enumerate(hashes):
        clusters.setdefault(root(index), []).extend(by_hash[phash])
    return [members for members in clusters.values() if len(members) > 1]


def _degenerate(phash: int) -> bool:
    """True for a hash carrying too little signal to compare.

    A perceptual hash describes how a picture's brightness varies. An image
    with almost no variation -- a solid colour, a blank scan, a flat gradient
    -- hashes to nearly all zeros or nearly all ones, and every such image then
    sits within a few bits of every other one. Left in, a folder of plain
    backgrounds would be reported as one enormous near-duplicate group.

    They are excluded from *near* matching only. Two genuinely identical flat
    images still match exactly, on their pixels.
    """
    set_bits = bin(phash).count("1")
    return set_bits <= NEAR_DISTANCE or set_bits >= 64 - NEAR_DISTANCE


def _spans_years(library: Library, members: list[int]) -> bool:
    times = [library.records[row].mtime for row in members]
    return max(times) - min(times) > RESEND_GAP_SECONDS


def _group(library: Library, members: list[int], kind: str) -> DuplicateGroup:
    ordered = sorted(members, key=lambda row: _keeper_rank(library, row))
    return DuplicateGroup(kind=kind, keeper=ordered[0], others=ordered[1:])


def _keeper_rank(library: Library, row: int) -> tuple:
    """Sort key implementing the keeper rule: earliest, then largest, then shortest."""
    record = library.records[row]
    return (record.mtime, -record.size, len(record.rel), record.rel)
