"""Running the four cache-building stages, with progress anyone can watch.

Stages are separate modules because they hold separate knowledge; the *order*
they run in, and what "how far along are we?" means while they do, is one piece
of knowledge and it lives here. Both the CLI and the server drive builds
through this class, so a 20-minute embed pass reports identically whether it is
being watched in a terminal or polled by the browser.

A build runs on the calling thread. `start()` puts it on a background one and
returns immediately, which is what the server needs so the page can load and
poll while the work happens (PROJECT.md 9.8).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import classify as classify_stage
from . import scan as scan_stage
from . import thumbs as thumbs_stage
from .cache import Cache, segment_count
from .embed import Device, Embedder, embed, pick_device, row_vectors

log = logging.getLogger(__name__)

STAGES = ("scan", "embed", "classify", "thumbs")


@dataclass
class Progress:
    """How far a build has got. Safe to read from another thread at any moment.

    `stage` is one of STAGES, or "" before the first starts and after the last
    finishes. `done`/`total` are within the current stage, and `total` of 0
    means the stage had nothing to do. `error` is a message if the build gave
    up, in which case `running` is False.
    """

    running: bool = False
    stage: str = ""
    done: int = 0
    total: int = 0
    device: str = ""
    notes: list[str] = field(default_factory=list)
    error: str | None = None
    started: float = 0.0
    stage_started: float = 0.0

    @property
    def eta_seconds(self) -> float | None:
        """Seconds left in the current stage, or None if it cannot be guessed.

        Measured from when this stage began rather than from the start of the
        build. Using the build's start makes every stage after the first
        inherit its predecessors' time and claim minutes of work remaining for
        something that finishes in seconds.
        """
        if not (self.running and self.done and self.total and self.stage_started):
            return None
        elapsed = time.time() - self.stage_started
        return elapsed / self.done * (self.total - self.done)

    def as_json(self) -> dict:
        return {**asdict(self), "eta_seconds": self.eta_seconds}


class Build:
    """Brings one folder's cache up to date, reporting progress as it goes.

    Construct with the folder and a device preference; call `run()` to work on
    this thread or `start()` to work on another. `progress` is readable
    throughout either way. One Build runs one pass -- create another to rebuild.

    The model is loaded lazily and only if some stage actually needs it, so a
    folder that is already embedded reopens without paying for PyTorch or
    waiting on weights.
    """

    def __init__(self, root: Path, device: str = "auto", stages=STAGES):
        self.cache = Cache(root)
        self.device_preference = device
        self.stages = tuple(stages)
        self.progress = Progress()
        self._embedder: Embedder | None = None
        self._device: Device | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Run the build on a daemon thread. Returns at once.

        `running` is set here, on the calling thread, rather than at the top of
        `run`. Setting it there leaves a window between this returning and the
        thread being scheduled in which the build looks finished, and a caller
        that waits for it sees "not running" and carries straight on.
        """
        self.progress.running = True
        self._thread = threading.Thread(target=self.run, name="build", daemon=True)
        self._thread.start()

    def wait(self) -> None:
        """Block until the build has finished. Safe if it never started."""
        if self._thread is not None:
            self._thread.join()

    def run(self) -> None:
        """Run every requested stage in order, in place.

        Any stage failing ends the build with the reason recorded in
        `progress.error` rather than raising: this is the one high-level handler
        for the whole pipeline, and callers watch progress rather than catching.
        """
        self.progress.running = True
        self.progress.started = time.time()
        try:
            records = self.cache.load_index()
            if "scan" in self.stages:
                records = scan_stage.scan(self.cache, self._reporter("scan"))
            if "embed" in self.stages:
                embeds = embed(
                    self.cache, records, self._model(), self._reporter("embed")
                )
            else:
                embeds = self.cache.load_embeddings(segment_count(records))
            if "classify" in self.stages:
                classify_stage.classify(
                    self.cache,
                    row_vectors(embeds, records),
                    self._model(),
                    self._reporter("classify"),
                )
            if "thumbs" in self.stages:
                thumbs_stage.build_thumbnails(
                    self.cache, records, embeds, self._reporter("thumbs")
                )
        except Exception as exc:
            log.exception("build failed during stage %s", self.progress.stage)
            self.progress.error = f"{self.progress.stage or 'build'}: {exc}"
        finally:
            self.progress.running = False
            self.progress.stage = ""

    def encode_text(self, phrase: str):
        """A 512-vector for one phrase, loading the model on first use.

        This is what the Library uses to rank a search, handed over as a plain
        callable so nothing below this line ever imports PyTorch.
        """
        return self._model().text([phrase])[0]

    # -- internals --------------------------------------------------------

    def _model(self) -> Embedder:
        if self._embedder is None:
            self._device = pick_device(self.device_preference)
            self.progress.device = self._device.label
            self.progress.notes = list(self._device.notes)
            for note in self._device.notes:
                log.warning("%s", note)
            log.info("using %s", self._device.label)
            log.info("loading CLIP (the first run downloads about 600 MB)")
            self._embedder = Embedder(self._device)
        return self._embedder

    def _reporter(self, stage: str):
        self.progress.stage = stage
        self.progress.done = 0
        self.progress.total = 0
        self.progress.stage_started = time.time()

        def report(done: int, total: int) -> None:
            self.progress.done = done
            self.progress.total = total

        return report
