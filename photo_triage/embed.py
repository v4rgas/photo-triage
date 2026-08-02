"""Stage 2 -- turn every image into a 512-dimension vector, once.

This is the only expensive stage and the only one that needs a GPU. It is also
the only module that knows anything about PyTorch, open_clip, devices or
batching: the classifier and the search endpoint ask for vectors and never
learn what produced them.

Vectors are L2-normalised before storage, which makes every later similarity
question -- search, "more like this", zero-shot classification -- a plain dot
product against a matrix that fits in RAM.

On AMD, PyTorch reimplements the CUDA surface rather than adding a new one:
`torch.cuda.is_available()` is True on a Radeon and there is no `torch.rocm`.
So all device code here is written against `torch.cuda`, and `torch.version.hip`
is consulted only to label the device and to apply the gfx override
(PROJECT.md 4a).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .cache import EMBED_DIM, Cache, MediaRecord, segment_count, segment_spans
from .quarantine import Quarantine

log = logging.getLogger(__name__)

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

# Consumer Radeons that ship no Tensile kernels of their own but run correctly
# while masquerading as an architecture that does. Pin to the rocm6.3 wheels:
# the override is documented to segfault on gfx1031/1032 from ROCm 6.4.3 up.
_GFX_OVERRIDES = {
    "gfx1031": "10.3.0",
    "gfx1032": "10.3.0",
    "gfx1010": "10.1.0",
    "gfx1012": "10.1.0",
    "gfx1103": "11.0.0",
}

# Batch size is derived, not configured: measured at 0.8 GB of VRAM for batch
# 128 at 224x224, so there is nothing to gain from filling a larger card. JPEG
# decode on the CPU is the bottleneck, which is what the worker counts address.
_TUNING = {
    "cuda": (128, max(1, (os.cpu_count() or 4) - 2)),
    "mps": (64, 4),
    "cpu": (32, max(1, (os.cpu_count() or 4) - 2)),
}

_FLUSH_EVERY = 2000  # rows between cache writes; bounds work lost to a kill


@dataclass
class Device:
    """A chosen compute device, ready to use.

    `label` is for humans and names the vendor and the memory. `notes` are
    warnings worth printing before a long run -- a missing GPU, a PyTorch build
    that does not match the hardware present -- and are empty in the happy case.
    """

    torch_device: object
    label: str
    notes: list[str] = field(default_factory=list)


def pick_device(preference: str = "auto") -> Device:
    """Choose where the embedding pass will run. Never raises.

    `preference` is one of "auto", "cuda", "mps" or "cpu"; anything the machine
    cannot honour falls back to CPU with a note rather than failing, because a
    slow triage beats no triage. The returned device has been proved with a real
    matmul: a card that enumerates but faults on first compute is a genuine
    failure mode on unsupported Radeons, and finding it at second two is much
    better than finding it forty minutes into a run.
    """
    import torch

    if preference == "cpu":
        return Device(torch.device("cpu"), _cpu_label())

    _apply_gfx_override()

    if preference in ("auto", "mps") and torch.backends.mps.is_available():
        candidate = Device(torch.device("mps"), "Apple MPS")
    elif preference in ("auto", "cuda") and torch.cuda.is_available():
        candidate = Device(torch.device("cuda"), _gpu_label(torch))
    else:
        notes = []
        if preference != "auto":
            notes.append(f"{preference} was requested but is not available.")
        notes.extend(_missing_gpu_notes(torch))
        return Device(torch.device("cpu"), _cpu_label(), notes)

    if _device_works(candidate.torch_device):
        return candidate
    return Device(
        torch.device("cpu"),
        _cpu_label(),
        [f"{candidate.label} enumerated but failed a test matmul; using the CPU."],
    )


class Embedder:
    """The CLIP model, loaded once, for both images and text.

    Construction downloads the weights on first use and is slow; hold one
    instance for the life of the process. Both methods return float32 arrays
    whose rows are L2-normalised, so cosine similarity between any two rows
    from either method is their dot product.
    """

    def __init__(self, device: Device):
        import open_clip
        import torch

        self.device = device
        self._torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED, device=device.torch_device
        )
        self.model.eval()
        self._tokenize = open_clip.get_tokenizer(MODEL_NAME)
        self._text_cache: dict[str, np.ndarray] = {}

    def text(self, phrases: list[str]) -> np.ndarray:
        """Encode phrases into an len(phrases) x 512 matrix, in order.

        Results are memoised for the life of the instance: users retype the
        same query constantly, and a repeat encode is pure latency in the one
        interaction that has to feel instant.
        """
        missing = [p for p in phrases if p not in self._text_cache]
        if missing:
            with self._torch.no_grad():
                tokens = self._tokenize(missing).to(self.device.torch_device)
                encoded = self.model.encode_text(tokens)
                encoded = _l2(encoded).cpu().numpy().astype(np.float32)
            for phrase, vector in zip(missing, encoded):
                self._text_cache[phrase] = vector
        return np.stack([self._text_cache[p] for p in phrases])

    def images(
        self,
        frames: list["Frame"],
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Encode frames into a len(frames) x 512 matrix, in order.

        A frame is a file and, for video, which moment of it to take. Whether
        a picture came out of a JPEG or out of the middle of a clip makes no
        difference past this point.

        Anything that cannot be decoded yields an all-zero row rather than an
        exception, matching the convention that zero means "no embedding":
        one bad file among twenty thousand must not end the run.
        """
        from torch.utils.data import DataLoader

        batch_size, workers = _TUNING[self.device.torch_device.type]
        loader = DataLoader(
            _FrameSet(frames, self.preprocess),
            batch_size=batch_size,
            num_workers=workers,
            prefetch_factor=4 if workers else None,
            shuffle=False,
        )
        out = np.zeros((len(frames), EMBED_DIM), dtype=np.float32)
        done = 0
        with self._torch.no_grad():
            for pixels, offsets, ok in loader:
                vectors = _l2(
                    self.model.encode_image(pixels.to(self.device.torch_device))
                )
                vectors = vectors.cpu().numpy().astype(np.float32)
                for vector, offset, decoded in zip(vectors, offsets.tolist(), ok.tolist()):
                    if decoded:
                        out[offset] = vector
                done += len(offsets)
                if progress:
                    progress(done, len(frames))
        return out


@dataclass
class Frame:
    """One picture to encode: a file, and which moment of it to take.

    `moment` is None for a photograph and the index of a sampled frame for a
    video. Nothing downstream of the decoder cares which it was.
    """

    path: Path
    moment: int | None = None
    of: int = 1


def pending(
    cache: Cache,
    records: list[MediaRecord],
    embeds: np.ndarray | None = None,
) -> list[tuple[int, Frame]]:
    """Every picture still waiting for a vector, as (segment offset, frame).

    Cheap, and needs no model. That is the point: the caller can find out
    whether there is any work before paying to load CLIP, which on a first run
    means a 600 MB download. A folder with nothing to embed should not download
    a model to discover that.
    """
    if embeds is None:
        embeds = cache.load_embeddings(segment_count(records))
    spans = segment_spans(records)
    where = Quarantine(cache, records)

    todo: list[tuple[int, Frame]] = []
    for row, record in enumerate(records):
        if not record.readable:
            continue
        start, stop = spans[row]
        path = where.location(row)
        for offset in range(start, stop):
            if not embeds[offset].any():
                moment = offset - start if record.is_video else None
                todo.append((offset, Frame(path, moment, stop - start)))
    return todo


def embed(
    cache: Cache,
    records: list[MediaRecord],
    embedder: Embedder,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Fill in every segment that has no vector yet, and persist the matrix.

    Resumable by construction: an all-zero segment is one that still needs
    work, so an interrupted run simply finds fewer to do next time. The matrix
    is flushed periodically during the pass, bounding what a kill can cost to
    the last few thousand pictures rather than the whole run.
    """
    embeds = cache.load_embeddings(segment_count(records))
    todo = pending(cache, records, embeds)
    if not todo:
        if progress:
            progress(0, 0)
        return embeds

    done = 0
    for chunk_start in range(0, len(todo), _FLUSH_EVERY):
        chunk = todo[chunk_start : chunk_start + _FLUSH_EVERY]
        chunk_progress = None
        if progress:
            base = done
            chunk_progress = lambda inner, _total: progress(base + inner, len(todo))
        vectors = embedder.images([frame for _, frame in chunk], chunk_progress)
        for (offset, _), vector in zip(chunk, vectors):
            embeds[offset] = vector
        cache.save_embeddings(embeds)
        done += len(chunk)
    return embeds


def row_vectors(embeds: np.ndarray, records: list[MediaRecord]) -> np.ndarray:
    """One vector per row: the mean of its segments, re-normalised.

    This is the "overall gist" view of an item, which is what classification
    wants. Search deliberately does not use it, because averaging a clip's
    moments is what loses the moment being searched for.
    """
    spans = segment_spans(records)
    out = np.zeros((len(records), EMBED_DIM), dtype=np.float32)
    for row, (start, stop) in enumerate(spans):
        block = embeds[start:stop]
        filled = block[block.any(axis=1)]
        if len(filled):
            mean = filled.mean(axis=0)
            out[row] = mean / max(float(np.linalg.norm(mean)), 1e-12)
    return out


# -- internals ------------------------------------------------------------


class _FrameSet:
    """Decodes and preprocesses one picture per index, for the DataLoader.

    Video is decoded here, in the loader's worker processes, for the same
    reason stills are: decode is the bottleneck and the GPU starves without
    enough workers feeding it.

    Each picture is an independent unit of work, so batches stay evenly sized
    and a half-readable clip still contributes the moments that did decode. To
    stop that costing a full decode per moment, the frames of the most recent
    clip are kept. The work list runs in segment order, so consecutive items
    are the same clip and one remembered entry turns what would be quadratic
    back into one pass per file.
    """

    def __init__(self, frames: list, preprocess):
        self.frames = frames
        self.preprocess = preprocess
        self._clip_path: Path | None = None
        self._clip_moments: list = []

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, offset: int):
        import torch
        from PIL import Image

        frame = self.frames[offset]
        try:
            if frame.moment is None:
                with Image.open(frame.path) as im:
                    if getattr(im, "n_frames", 1) > 1:
                        im.seek(0)
                    pixels = self.preprocess(im.convert("RGB"))
            else:
                moments = self._moments(frame)
                if frame.moment >= len(moments):
                    return torch.zeros(3, 224, 224), offset, False
                pixels = self.preprocess(moments[frame.moment].convert("RGB"))
            return pixels, offset, True
        except Exception:
            return torch.zeros(3, 224, 224), offset, False

    def _moments(self, frame) -> list:
        if frame.path != self._clip_path:
            from .video import sample

            self._clip_path = frame.path
            self._clip_moments = sample(frame.path, frame.of)
        return self._clip_moments


def _l2(tensor):
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _cpu_label() -> str:
    return f"CPU ({os.cpu_count()} threads)"


def _gpu_label(torch) -> str:
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    if torch.version.hip:
        arch = torch.cuda.get_device_properties(0).gcnArchName
        return f"AMD {name} [{arch}] {vram:.0f}GB"
    return f"NVIDIA {name} {vram:.0f}GB"


def _missing_gpu_notes(torch) -> list[str]:
    """Warnings for the CPU fallback, including a detected wrong-wheel install."""
    notes = [
        "No GPU detected -- the embed pass will be roughly 20x slower.",
        "Consider running with --no-serve and leaving it overnight.",
    ]
    if torch.version.cuda is None and torch.version.hip is None and _has_nvidia_smi():
        notes.insert(
            0,
            "nvidia-smi is present but this is a CPU-only PyTorch build. "
            "Reinstall with: uv pip install torch torchvision",
        )
    return notes


def _has_nvidia_smi() -> bool:
    try:
        subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5, check=True
        )
        return True
    except Exception:
        return False


def _device_works(device) -> bool:
    import torch

    try:
        probe = torch.randn(2048, 2048, device=device)
        (probe @ probe).sum().item()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return True
    except Exception as exc:
        log.warning("test matmul on %s failed: %s", device, exc)
        return False


def _apply_gfx_override() -> None:
    """Set HSA_OVERRIDE_GFX_VERSION for Radeons that need it, before first use.

    The architecture can only be read through a CUDA call, and the override
    must be in place *before* the first one -- so the probe runs in a throwaway
    subprocess, leaving this process's ROCm runtime uninitialised until the
    variable is correct. Respects an existing setting; the value chosen is
    logged so a user can reproduce it by hand.
    """
    import torch

    if os.environ.get("HSA_OVERRIDE_GFX_VERSION") or not torch.version.hip:
        return
    probe = (
        "import torch;"
        "print(torch.cuda.get_device_properties(0).gcnArchName"
        " if torch.cuda.is_available() else '')"
    )
    try:
        arch = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
        ).stdout.strip()
    except Exception:
        return
    override = _GFX_OVERRIDES.get(arch.split(":")[0])
    if override:
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = override
        log.info(
            "%s has no kernels of its own; presenting it as gfx%s "
            "(export HSA_OVERRIDE_GFX_VERSION=%s to reproduce)",
            arch,
            override.replace(".", ""),
            override,
        )
