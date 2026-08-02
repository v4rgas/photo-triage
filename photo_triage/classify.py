"""Stage 3 -- give every embedded image a rough category.

Zero-shot: each category's phrasings are encoded, averaged and re-normalised
into one vector, and an image takes the category whose vector it is closest to.
Confidence is that similarity row softmaxed at temperature 100, which is the
conventional CLIP scaling and produces figures comparable across images.

Read the confidence for what it is. It says how much the winning category beat
the others by; it says nothing about whether the winner is right. The
prototype's worst mistakes came in at 0.99 (PROJECT.md 9.1), which is why
nothing in this system lets confidence alone authorise a deletion.

The whole stage is one matmul over cached vectors, so re-running it with edited
prompts costs seconds. Treat the prompt set as tunable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from .cache import Cache, Scores
from .embed import Embedder
from .prompts import load_categories

log = logging.getLogger(__name__)

_TEMPERATURE = 100.0


def classify(
    cache: Cache,
    embeds: np.ndarray,
    embedder: Embedder,
    progress: Callable[[int, int], None] | None = None,
) -> Scores:
    """Classify every embedded row and persist the result.

    Rows with no embedding -- unreadable files, or files added since the last
    embed pass -- come back with an empty category rather than being forced
    into the nearest one.
    """
    categories = load_categories(cache.dir)
    names = list(categories)
    if progress:
        progress(0, 2)

    anchors = np.stack(
        [_average(embedder.text(categories[name])) for name in names]
    )  # len(names) x 512, each row L2-normalised

    scores = Scores([""] * len(embeds), [0.0] * len(embeds))
    embedded = np.flatnonzero(embeds.any(axis=1))
    if len(embedded):
        similarity = embeds[embedded] @ anchors.T  # cosine, both sides normalised
        winners = similarity.argmax(axis=1)
        confidence = _softmax(similarity * _TEMPERATURE)
        for offset, row in enumerate(embedded):
            scores.category[row] = names[winners[offset]]
            scores.confidence[row] = float(confidence[offset, winners[offset]])
    if progress:
        progress(2, 2)

    cache.save_scores(scores)
    log.info("classified %d of %d rows into %d categories",
             len(embedded), len(embeds), len(names))
    return scores


def _average(vectors: np.ndarray) -> np.ndarray:
    """Mean of several phrasings, re-normalised so it stays a direction."""
    mean = vectors.mean(axis=0)
    return mean / max(float(np.linalg.norm(mean)), 1e-12)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)
