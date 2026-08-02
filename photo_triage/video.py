"""Reading video files as a handful of still frames.

The only module that knows how to decode video. Everything downstream asks for
frames and gets PIL images, exactly as it would from a photograph, so scanning,
embedding and thumbnailing all treat a clip as "a picture that happens to have
several moments in it" rather than as a separate kind of thing.

Why sampling rather than watching
---------------------------------
A search for `birthday cake` should find the clip that has a cake in it, and a
clip is found by looking at a few of its moments. Decoding every frame of a
two-minute video to answer that is a thousand times the work for no more
answer, so this seeks to a fixed set of timestamps and decodes one frame at
each. Decode cost is the bottleneck for stills too (PROJECT.md 4a), and the
same lesson applies here with a much larger constant.

Why several vectors rather than one average
-------------------------------------------
Averaging a clip's frames into a single vector is the obvious move and it
quietly destroys the thing being searched for. A video that opens on a
birthday table and ends in the car averages to a direction that points at
neither, and the longer the video the more the moment you wanted is diluted.
So every sampled frame keeps its own vector, and a clip scores as the best of
its moments. The mean is still computed, but only where the overall gist is
what is wanted, which is classification.

Audio is not read at all. CLIP has no ear, so a video that matters because of
what someone says is not findable here, and that is a real limit rather than
an oversight.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)

# Roughly one sample every two seconds, which is where video-retrieval work
# settles for clips of this length: enough to catch a scene change, few enough
# that a long clip stays cheap.
SECONDS_PER_SAMPLE = 2.0
MIN_SAMPLES = 8
MAX_SAMPLES = 32
_MAX_RATE = 5.0  # never sample faster than this, so a 1s clip is not 8 copies

# Phone videos routinely open on a black or still-focusing frame, and it drags
# an average toward nothing. Trim both ends before spacing the samples.
_HEAD_TRIM = 0.5
_TAIL_TRIM = 0.2

# First bytes of the containers worth opening. As with stills, the extension is
# not consulted: a `.mov` that is really an mp4 is common and harmless.
_MAGIC: tuple[tuple[int, bytes], ...] = (
    (4, b"ftyp"),           # mp4, mov, m4v, 3gp
    (0, b"\x1a\x45\xdf\xa3"),  # matroska, webm
    (0, b"RIFF"),           # avi, checked further below
    (0, b"OggS"),           # ogg theora
    (0, b"FLV\x01"),        # flash video, still turns up in old exports
)


@dataclass
class VideoInfo:
    """What a clip is, without decoding any of it.

    `duration` is in seconds and is 0.0 when the container does not say, which
    happens with some phone recordings and with truncated files. Callers treat
    an unknown duration as a short clip rather than as an error.
    """

    width: int
    height: int
    duration: float


def looks_like_video(head: bytes) -> bool:
    """True if these first bytes are a video container this module can open.

    Takes the same header bytes the still-image sniffer reads, so the scanner
    can ask both questions from one read of the file.
    """
    for offset, magic in _MAGIC:
        if head[offset : offset + len(magic)] == magic:
            if magic == b"RIFF":
                return head[8:12] == b"AVI "
            return True
    return False


def sample_count(duration: float) -> int:
    """How many frames to take from a clip of this length.

    Clamped at both ends: below the floor a short clip would be described by
    too little, and above the ceiling a long one costs more than the extra
    moments are worth. Short clips are additionally capped so that a one-second
    video is not sampled eight times into eight near-identical frames.
    """
    wanted = round(duration / SECONDS_PER_SAMPLE) if duration > 0 else MIN_SAMPLES
    wanted = max(MIN_SAMPLES, min(MAX_SAMPLES, int(wanted)))
    if duration > 0:
        wanted = min(wanted, max(1, int(duration * _MAX_RATE)))
    return max(1, wanted)


def probe(path: Path) -> VideoInfo | None:
    """Dimensions and duration, or None if this is not a video we can read.

    Reads the container header only. Cheap enough to call on every candidate
    file during a scan.
    """
    import av

    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                return None
            stream = container.streams.video[0]
            width, height = _display_size(stream)
            return VideoInfo(width, height, _duration(container, stream))
    except Exception as exc:
        log.debug("cannot probe %s: %s", path, exc)
        return None


def sample(path: Path, count: int | None = None) -> list[Image.Image]:
    """Decode `count` frames spread across the clip, oldest first.

    Returns fewer frames than asked for, possibly none, when the file is
    truncated or the decoder gives up part way. Callers size their work from
    what comes back rather than from what they requested, so a half-readable
    video still contributes the half that decoded.

    Frames are rotated upright if the container says they should be. Phone
    video is very often recorded sideways with the correction left as metadata,
    and an unrotated frame embeds as a different picture entirely.
    """
    import av

    frames: list[Image.Image] = []
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                return []
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            duration = _duration(container, stream)
            wanted = count if count is not None else sample_count(duration)
            rotation = _rotation(stream)

            for moment in _timestamps(duration, wanted):
                frame = _frame_at(container, stream, moment)
                if frame is not None:
                    frames.append(_upright(frame.to_image(), rotation))
    except Exception as exc:
        log.debug("cannot sample %s: %s", path, exc)
    return frames


# -- internals -------------------------------------------------------------


def _timestamps(duration: float, count: int) -> list[float]:
    """Evenly spaced moments inside the trimmed span of the clip."""
    if duration <= 0:
        # An unknown duration still deserves a try: take the opening seconds,
        # which is where a short clip's content is anyway.
        return [index * SECONDS_PER_SAMPLE for index in range(count)]
    start = _HEAD_TRIM if duration > _HEAD_TRIM * 4 else 0.0
    stop = max(start, duration - _TAIL_TRIM)
    if count == 1:
        return [(start + stop) / 2]
    step = (stop - start) / (count - 1)
    return [start + step * index for index in range(count)]


def _frame_at(container, stream, moment: float):
    """Seek to `moment` seconds and decode the first frame at or after it."""
    try:
        offset = int(moment / stream.time_base) + (stream.start_time or 0)
        container.seek(offset, stream=stream, any_frame=False, backward=True)
        for frame in container.decode(stream):
            return frame
    except Exception:
        return None
    return None


def _duration(container, stream) -> float:
    import av

    if stream.duration and stream.time_base:
        return float(stream.duration * stream.time_base)
    if container.duration:
        return float(container.duration / av.time_base)
    return 0.0


def _display_size(stream) -> tuple[int, int]:
    """Dimensions as they should be shown, with rotation already applied."""
    width, height = stream.codec_context.width, stream.codec_context.height
    if _rotation(stream) in (90, 270):
        return height, width
    return width, height


def _rotation(stream) -> int:
    """Clockwise rotation the container asks for, as 0, 90, 180 or 270."""
    try:
        raw = stream.metadata.get("rotate")
        if raw is not None:
            return int(float(raw)) % 360
    except (TypeError, ValueError):
        pass
    try:
        for side_data in stream.side_data:
            if "DISPLAYMATRIX" in str(side_data.type):
                return int(-float(side_data.rotation)) % 360
    except Exception:
        pass
    return 0


def _upright(image: Image.Image, rotation: int) -> Image.Image:
    if rotation == 0:
        return image
    return image.rotate(-rotation, expand=True)
