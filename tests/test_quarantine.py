"""Quarantine, restore and protection -- the operations that can lose data."""

from __future__ import annotations

import hashlib

from photo_triage.cache import Cache
from photo_triage.quarantine import ACTIVE, PURGED, QUARANTINED, Quarantine


def open_quarantine(cache: Cache) -> Quarantine:
    return Quarantine(cache, cache.load_index())


def row_of(cache: Cache, rel: str) -> int:
    return [r.rel for r in cache.load_index()].index(rel)


def test_nested_file_survives_a_round_trip_byte_for_byte(scanned: Cache):
    rel = "holidays/IMG_0042.jpg"
    row = row_of(scanned, rel)
    original = (scanned.root / rel).read_bytes()

    where = open_quarantine(scanned)
    assert where.quarantine([row]).moved == [row]
    assert not (scanned.root / rel).exists()
    assert (scanned.quarantine_dir / rel).exists()

    assert where.restore([row]).moved == [row]
    assert (scanned.root / rel).read_bytes() == original


def test_colliding_basenames_restore_to_their_own_folders(scanned: Cache):
    rows = [row_of(scanned, "holidays/IMG_0042.jpg"), row_of(scanned, "trip/IMG_0042.jpg")]
    digests = [
        hashlib.md5((scanned.root / rel).read_bytes()).hexdigest()
        for rel in ("holidays/IMG_0042.jpg", "trip/IMG_0042.jpg")
    ]

    where = open_quarantine(scanned)
    where.quarantine(rows)
    assert (scanned.quarantine_dir / "holidays/IMG_0042.jpg").exists()
    assert (scanned.quarantine_dir / "trip/IMG_0042.jpg").exists()

    where.restore(rows)
    for rel, digest in zip(("holidays/IMG_0042.jpg", "trip/IMG_0042.jpg"), digests):
        assert hashlib.md5((scanned.root / rel).read_bytes()).hexdigest() == digest


def test_an_arbitrary_earlier_batch_can_be_restored(scanned: Cache):
    first = [row_of(scanned, "holidays/IMG_0042.jpg")]
    second = [row_of(scanned, "trip/IMG_0042.jpg")]

    where = open_quarantine(scanned)
    where.quarantine(first)
    where.quarantine(second)
    batches = where.batches()
    assert [batch["ids"] for batch in batches] == [first, second]

    # The older batch, not the most recent one.
    where.restore(batches[0]["ids"])
    assert (scanned.root / "holidays/IMG_0042.jpg").exists()
    assert not (scanned.root / "trip/IMG_0042.jpg").exists()


def test_a_saved_file_cannot_be_quarantined(scanned: Cache):
    row = row_of(scanned, "holidays/IMG_0042.jpg")
    where = open_quarantine(scanned)
    where.set_saved([row], True)

    result = where.quarantine([row])
    assert result.moved == []
    assert result.refused == {row: "protected"}
    assert (scanned.root / "holidays/IMG_0042.jpg").exists()


def test_state_and_protection_survive_a_restart(scanned: Cache):
    quarantined = row_of(scanned, "holidays/IMG_0042.jpg")
    protected = row_of(scanned, "trip/IMG_0042.jpg")
    where = open_quarantine(scanned)
    where.quarantine([quarantined])
    where.set_saved([protected], True)

    reopened = open_quarantine(scanned)
    assert reopened.state[quarantined] == QUARANTINED
    assert reopened.state[protected] == ACTIVE
    assert reopened.saved == {protected}


def test_repeating_an_operation_changes_nothing(scanned: Cache):
    row = row_of(scanned, "holidays/IMG_0099.jpg")
    where = open_quarantine(scanned)
    where.quarantine([row])
    again = where.quarantine([row])
    assert again.moved == []
    assert again.refused == {row: "already quarantined"}
    assert where.state[row] == QUARANTINED


def test_purge_reports_before_it_destroys(scanned: Cache):
    row = row_of(scanned, "screenshots/Screenshot_1.png")
    where = open_quarantine(scanned)
    where.quarantine([row])

    plan = where.purge(dry_run=True)
    assert plan.rows == [row] and plan.bytes_freed > 0
    assert (scanned.quarantine_dir / "screenshots/Screenshot_1.png").exists()

    done = where.purge(dry_run=False)
    assert done.rows == [row]
    assert not (scanned.quarantine_dir / "screenshots/Screenshot_1.png").exists()
    assert where.state[row] == PURGED
    assert open_quarantine(scanned).state[row] == PURGED


def test_restore_refuses_rather_than_overwriting_an_occupied_path(scanned: Cache):
    rel = "holidays/IMG_0042.jpg"
    row = row_of(scanned, rel)
    where = open_quarantine(scanned)
    where.quarantine([row])
    (scanned.root / rel).write_bytes(b"something the user put back by hand")

    result = where.restore([row])
    assert result.moved == []
    assert "occupies" in result.refused[row]
    assert (scanned.root / rel).read_bytes() == b"something the user put back by hand"


def test_unknown_ids_are_reported_not_ignored(scanned: Cache):
    where = open_quarantine(scanned)
    result = where.quarantine([9999])
    assert result.moved == []
    assert result.refused == {9999: "no such image"}
