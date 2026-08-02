"""Filtering, ranking, and the endpoints that must agree about what matches."""

from __future__ import annotations

import numpy as np
import pytest

from photo_triage.cache import Cache, Scores
from photo_triage.library import Library, Query
from photo_triage.server import create_app

from .conftest import fake_embeddings


@pytest.fixture
def library(scanned: Cache) -> Library:
    rows = len(scanned.load_index())
    fake_embeddings(scanned, rows)
    scanned.save_scores(
        Scores(
            category=["screenshot", "people_photo", "people_photo", "meme_text"][:rows],
            confidence=[0.9, 0.8, 0.4, 0.95][:rows],
        )
    )
    return Library.open(scanned.root)


@pytest.fixture
def client(library: Library):
    app = create_app(library.root)
    app.config.update(TESTING=True)
    return app.test_client()


def test_ids_and_search_agree_about_every_filter(client):
    filters = [
        "",
        "group=junk",
        "cat=people_photo",
        "folder=holidays",
        "show=all",
        "view=quarantine",
        "group=keep&show=all",
    ]
    for query in filters:
        page = client.get(f"/api/search?{query}&limit=1000").get_json()
        ids = client.get(f"/api/ids?{query}").get_json()
        assert page["matched"] == ids["matched"], query
        assert [tile["id"] for tile in page["results"]] == ids["ids"], query


def test_a_saved_image_drops_out_of_the_default_view(client, library: Library):
    before = client.get("/api/ids").get_json()["matched"]
    row = library.select(Query())[0]
    client.post("/api/save", json={"ids": [row]})
    after = client.get("/api/ids").get_json()
    assert after["matched"] == before - 1
    assert row not in after["ids"]
    assert row in client.get("/api/ids?show=saved").get_json()["ids"]


def test_a_saved_image_survives_a_direct_post_to_the_delete_endpoint(client, library):
    row = library.select(Query())[0]
    client.post("/api/save", json={"ids": [row]})
    result = client.post("/api/quarantine", json={"ids": [row]}).get_json()
    assert result["moved"] == []
    assert result["refused"][str(row)] == "protected"
    assert (library.root / library.records[row].rel).exists()


def test_like_ranks_the_image_itself_first(library: Library):
    row = 2
    ranked = library.select(Query(like=row, show="all"))
    assert ranked[0] == row


def test_search_ranks_rather_than_filters(scanned: Cache):
    """A text query must reorder the whole matching set, never shrink it."""
    rows = len(scanned.load_index())
    embeds = fake_embeddings(scanned, rows)

    def encode(_phrase):
        return embeds[1]

    library = Library.open(scanned.root, encode_text=encode)
    assert len(library.select(Query(text="anything"))) == rows
    assert library.select(Query(text="anything"))[0] == 1


def test_unembedded_rows_sink_but_are_never_dropped(scanned: Cache):
    rows = len(scanned.load_index())
    embeds = fake_embeddings(scanned, rows)
    embeds[0] = 0.0
    scanned.save_embeddings(embeds)
    library = Library.open(scanned.root, encode_text=lambda _: embeds[1])
    ranked = library.select(Query(text="anything"))
    assert len(ranked) == rows
    assert ranked[-1] == 0


def test_an_unscanned_folder_opens_empty_rather_than_failing(tmp_path):
    library = Library.open(tmp_path)
    assert library.select(Query()) == []
    assert library.stats().total == 0


def test_stats_and_review_marking_round_trip(client):
    assert client.get("/api/stats").get_json()["reviewed"] == []
    client.post("/api/reviewed", json={"category": "meme_text"})
    assert client.get("/api/stats").get_json()["reviewed"] == ["meme_text"]


def test_the_port_is_derived_from_the_folder_and_stays_put(tmp_path):
    """Same folder, same address tomorrow. Different folders, different ones."""
    from photo_triage.server import pick_port

    first = tmp_path / "holidays"
    second = tmp_path / "work"
    first.mkdir()
    second.mkdir()

    assert pick_port(first) == pick_port(first)
    assert pick_port(first) != pick_port(second)


def test_the_derived_port_avoids_the_kernel_ephemeral_range(tmp_path):
    """Below 61000 a derived port could clash with an outgoing connection."""
    from photo_triage.server import pick_port

    for name in ("a", "b", "c", "d", "e", "f", "g", "h"):
        folder = tmp_path / name
        folder.mkdir()
        assert 61000 <= pick_port(folder) <= 65535


def test_a_taken_port_is_stepped_over(tmp_path):
    import socket

    from photo_triage.server import pick_port

    folder = tmp_path / "busy"
    folder.mkdir()
    wanted = pick_port(folder)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as squatter:
        squatter.bind(("127.0.0.1", wanted))
        squatter.listen(1)
        assert pick_port(folder) != wanted
    assert pick_port(folder) == wanted  # and it comes back once released
