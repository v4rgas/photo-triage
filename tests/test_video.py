"""Video: sampling, segment spans, and best-moment ranking."""

from __future__ import annotations

import numpy as np
import pytest

from photo_triage.cache import Cache, MediaRecord, segment_count, segment_spans
from photo_triage.embed import row_vectors
from photo_triage.library import Library, Query
from photo_triage.scan import scan
from photo_triage.video import (
    FRAMES_PER_VIDEO,
    looks_like_video,
    sample,
    sample_count,
)

from .conftest import write_image

av = pytest.importorskip("av")


def write_video(path, seconds=6.0, fps=10, size=(160, 120), scenes=None):
    """Encode a short clip. `scenes` is a list of colours, shown in turn.

    A clip whose halves look nothing alike is the case that matters here: it
    is what mean-pooling destroys and what max-over-moments has to survive.
    """
    scenes = scenes or [(200, 40, 40)]
    path.parent.mkdir(parents=True, exist_ok=True)
    total = int(seconds * fps)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        for index in range(total):
            colour = scenes[min(index * len(scenes) // total, len(scenes) - 1)]
            plane = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            plane[:, :] = colour
            # A little structure per frame, so the encoder cannot collapse the
            # whole clip and the perceptual hash means something.
            plane[(index * 3) % size[1], :] = 255
            frame = av.VideoFrame.from_ndarray(plane, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_a_video_is_indexed_with_a_duration_and_several_segments(tmp_path):
    write_video(tmp_path / "clips" / "VID_0001.mp4", seconds=6.0)
    records = scan(Cache(tmp_path))

    assert [r.rel for r in records] == ["clips/VID_0001.mp4"]
    clip = records[0]
    assert clip.readable and clip.is_video
    assert clip.duration == pytest.approx(6.0, abs=1.0)
    assert clip.segments > 1
    assert (clip.width, clip.height) == (160, 120)
    assert clip.phash  # hashed from a real frame, so dedupe still applies


def test_a_photograph_still_owns_exactly_one_segment(folder):
    records = scan(Cache(folder))
    assert all(record.segments == 1 for record in records)
    assert segment_count(records) == len(records)
    assert segment_spans(records) == [(row, row + 1) for row in range(len(records))]


def test_spans_stay_contiguous_when_stills_and_clips_are_mixed():
    records = [
        MediaRecord("a.jpg", 1, 1.0),
        MediaRecord("b.mp4", 1, 1.0, duration=6.0, segments=8),
        MediaRecord("c.jpg", 1, 1.0),
    ]
    assert segment_spans(records) == [(0, 1), (1, 9), (9, 10)]
    assert segment_count(records) == 10


def test_sampling_never_returns_more_frames_than_the_clip_can_give(tmp_path):
    clip = write_video(tmp_path / "short.mp4", seconds=1.0, fps=10)
    frames = sample(clip)
    assert 1 <= len(frames) <= sample_count(1.0)
    assert all(frame.size == (160, 120) for frame in frames)


def test_a_one_second_clip_is_not_sampled_eight_times():
    """The budget must not turn a very short clip into duplicates of itself."""
    assert sample_count(1.0) <= 5


def test_every_clip_costs_the_same_however_long_it_is():
    """The whole point of a fixed budget: length must not drive up the cost.

    A ten-minute video and a twenty-second one own the same number of vectors,
    so no single clip can dominate the embedding pass.
    """
    assert sample_count(20.0) == FRAMES_PER_VIDEO
    assert sample_count(60.0) == FRAMES_PER_VIDEO
    assert sample_count(600.0) == FRAMES_PER_VIDEO
    assert sample_count(0.0) == FRAMES_PER_VIDEO  # duration the container hid


def test_video_containers_are_recognised_from_their_header():
    assert looks_like_video(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 16)
    assert looks_like_video(b"\x1a\x45\xdf\xa3" + b"\x00" * 28)
    assert not looks_like_video(b"\xff\xd8\xff" + b"\x00" * 29)
    assert not looks_like_video(b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 20)


def test_a_clip_scores_as_its_best_moment_not_its_average(tmp_path):
    """The reason a video keeps several vectors instead of one.

    Two segments point in opposite directions. Averaged, the clip matches
    nothing. Taken as a best moment, it matches whichever half you asked for,
    which is what someone searching their camera roll means.
    """
    write_image(tmp_path / "still.jpg", (10, 10, 10))
    write_video(tmp_path / "clip.mp4", seconds=4.0)
    cache = Cache(tmp_path)
    records = scan(cache)

    clip_row = next(row for row, r in enumerate(records) if r.is_video)
    spans = segment_spans(records)
    start, stop = spans[clip_row]

    embeds = np.zeros((segment_count(records), 512), dtype=np.float32)
    embeds[:, 0] = 1.0                      # the still, and a default direction
    query = np.zeros(512, dtype=np.float32)
    query[1] = 1.0
    # Inside the clip: one moment points at the query, the rest point away.
    embeds[start:stop, 0] = 1.0
    embeds[start:stop, 1] = 0.0
    embeds[start] = query
    cache.save_embeddings(embeds)

    means = row_vectors(embeds, records)
    assert float(means[clip_row] @ query) < 0.6   # the average is diluted

    library = Library.open(tmp_path, encode_text=lambda _: query)
    ranked = library.select(Query(text="whatever", show="all"))
    assert ranked[0] == clip_row                  # the best moment wins
    assert len(ranked) == len(records)            # ranking never filters


def test_embedding_a_clip_fills_every_one_of_its_segments(tmp_path):
    from photo_triage.embed import embed

    write_video(tmp_path / "clip.mp4", seconds=4.0)
    cache = Cache(tmp_path)
    records = scan(cache)
    total = segment_count(records)
    assert total > 1

    class CountingEmbedder:
        def __init__(self):
            self.seen = 0

        def images(self, frames, progress=None):
            self.seen += len(frames)
            vectors = np.zeros((len(frames), 512), dtype=np.float32)
            for index, frame in enumerate(frames):
                # A different direction per moment, so nothing collapses.
                vectors[index, (frame.moment or 0) % 512] = 1.0
            return vectors

    embedder = CountingEmbedder()
    embeds = embed(cache, records, embedder)
    assert embedder.seen == total
    assert embeds.any(axis=1).all()

    # Re-running finds nothing left to do.
    embedder.seen = 0
    embed(cache, records, embedder)
    assert embedder.seen == 0
