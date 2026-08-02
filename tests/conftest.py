"""Shared fixtures: a real folder of real image files on a real filesystem.

These tests deliberately avoid mocking the filesystem. Everything they are
guarding -- atomic renames, a mirrored directory tree, byte-identical restores
-- is behaviour of the filesystem itself, and a fake one would assert only that
the fake behaves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from photo_triage.cache import Cache
from photo_triage.scan import scan


def write_image(path: Path, colour: tuple[int, int, int], size=(64, 48), **save) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path, **save)
    return path


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    """A small folder with nested directories and two colliding basenames."""
    write_image(tmp_path / "holidays" / "IMG_0042.jpg", (200, 30, 30))
    write_image(tmp_path / "holidays" / "IMG_0099.jpg", (30, 200, 30))
    write_image(tmp_path / "trip" / "IMG_0042.jpg", (30, 30, 200))
    write_image(tmp_path / "screenshots" / "Screenshot_1.png", (240, 240, 240))
    return tmp_path


@pytest.fixture
def scanned(folder: Path) -> Cache:
    cache = Cache(folder)
    scan(cache)
    return cache


def fake_embeddings(cache: Cache, rows: int, seed: int = 0) -> np.ndarray:
    """Deterministic L2-normalised vectors, standing in for a CLIP pass."""
    generator = np.random.default_rng(seed)
    embeds = generator.normal(size=(rows, 512)).astype(np.float32)
    embeds /= np.linalg.norm(embeds, axis=1, keepdims=True)
    cache.save_embeddings(embeds)
    return embeds
