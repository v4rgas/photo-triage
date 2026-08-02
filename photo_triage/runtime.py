"""Knowing whether the model is installed, and what to run if it is not.

PyTorch ships a different wheel per accelerator and they are not
interchangeable: the wheel on PyPI is a CUDA build, and on a Radeon it gives
you an install that works and runs about twenty times slower than the hardware
can. A plain resolver cannot get this right, because the answer is a property
of the machine rather than of the dependency graph.

`uv` solves it properly. `--torch-backend` rewrites the index for exactly the
PyTorch packages, so the extra can be declared in pyproject.toml like any other
and still resolve to the right build. That makes installation a single
supported command, and leaves this module with one job: notice when the model
is missing or mismatched, and say precisely what to run.

It deliberately installs nothing itself. An earlier version detected the
hardware, downloaded wheels into a private directory and put them on sys.path,
which worked and was a hack: it duplicated a solved problem and depended on
pip's private internals. What survives is the part uv genuinely does not do,
which is notice an AMD card.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# ROCm is pinned rather than left to float: the gfx1031/gfx1032 architecture
# override that makes consumer Radeons work is documented to segfault from
# ROCm 6.4.3 up, and 6.3 is the last line verified against that card family.
_ROCM_PIN = "rocm6.3"

# What to pass uv's --torch-backend for each vendor.
#
# uv knows which build pairs with which driver version and keeping that table
# current is a job it does better than we would, so `auto` is right wherever it
# works. What it does not do is notice an AMD card: its detection reads NVIDIA
# drivers and settles on the CPU build for a Radeon, which is the twenty-times
# slowdown this module exists to prevent. So AMD is named explicitly.
_TORCH_BACKEND = {
    "cuda": "auto",
    "rocm": _ROCM_PIN,
    "mps": "auto",   # the default wheels carry Metal support
    "cpu": "cpu",
}

PACKAGE = "photo-triage"


@dataclass
class Backend:
    """The accelerator this machine has, and how to install for it.

    `name` is one of "cuda", "rocm", "mps" or "cpu". `why` is the evidence it
    was chosen on, quoted back to the user so that an unexpected answer, such
    as a CPU verdict on a machine with a graphics card, is debuggable rather
    than mysterious.
    """

    name: str
    why: str

    @property
    def torch_backend(self) -> str:
        """The value for uv's --torch-backend flag."""
        return _TORCH_BACKEND[self.name]

    def install_command(self) -> str:
        """What to run to get the right PyTorch, for how this was installed.

        A distribution package manager owns its own files, so telling someone
        who installed through pacman to run a uv command would have them
        shadow a tracked install with an untracked one. The advice follows the
        route the user actually took.
        """
        if externally_managed() and shutil.which("pacman"):
            variant = {"rocm": "python-pytorch-rocm", "cuda": "python-pytorch-cuda"}
            package = variant.get(self.name, "python-pytorch")
            return f"sudo pacman -S {package} python-open-clip-torch"
        return (
            f'uv tool install "{PACKAGE}[model]" '
            f"--torch-backend={self.torch_backend}"
        )


def detect_backend() -> Backend:
    """What this machine can accelerate with. Cheap, and never raises.

    A GPU always wins when one is present, because the difference is not a
    tuning detail: the same folder takes 78 seconds on a Radeon and over twenty
    minutes on the CPU. CPU is the answer only when there is nothing else,
    never a cautious default.

    The kernel is asked directly rather than only looking for vendor tools,
    since a working card with no `nvidia-smi` on PATH is common and would
    otherwise be missed.
    """
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return Backend("mps", "Apple Silicon")
    nvidia = _nvidia_evidence()
    if nvidia:
        return Backend("cuda", nvidia)
    if Path("/sys/module/amdgpu").exists() and _any_render_node():
        return Backend("rocm", "the amdgpu kernel driver is loaded")
    return Backend("cpu", "no supported GPU found")


def installed_backend() -> str | None:
    """What PyTorch build is installed, or None if PyTorch is absent.

    Reads the build flags rather than probing devices, because on AMD there is
    no separate API to probe: a ROCm build answers to `torch.cuda` and reports
    `torch.cuda.is_available()` as True on a Radeon (PROJECT.md 4a).
    """
    try:
        import torch
    except ImportError:
        return None
    if torch.version.hip:
        return "rocm"
    if torch.version.cuda:
        return "cuda"
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mps"
    return "cpu"


def ensure_model_runtime(assume_yes: bool = False) -> bool:
    """True when the model can be loaded, otherwise say what to run.

    `assume_yes` is accepted and ignored, since nothing here asks a question
    any more; the caller passes it through from --yes.

    An installed build that merely disagrees with the hardware is reported and
    left alone. A deliberate CPU build on a machine with a GPU is a legitimate
    choice, and replacing it uninvited would be worse than slow.
    """
    del assume_yes
    wanted = detect_backend()
    present = installed_backend()

    if present is not None and _has_open_clip():
        if present != wanted.name and not (present == "cuda" and wanted.name == "mps"):
            log.warning(
                "PyTorch is a %s build but this machine has %s (%s). Embedding "
                "will work and will be slow. To switch:\n  %s",
                present, wanted.name, wanted.why, wanted.install_command(),
            )
        return True

    missing = "PyTorch and open_clip are" if present is None else "open_clip is"
    print(
        f"\n{missing} not installed. They are what reads the images, and "
        f"everything else works without them.\n"
        f"Detected: {wanted.name} ({wanted.why})\n\n"
        f"  {wanted.install_command()}\n\n"
        f"{_uv_hint()}",
        file=sys.stderr,
    )
    return False


# -- internals -------------------------------------------------------------


def _uv_hint() -> str:
    if shutil.which("uv"):
        return ""
    return (
        "uv is not installed. See https://docs.astral.sh/uv/ , or install "
        "torch, torchvision and open_clip_torch yourself.\n"
    )


def _nvidia_evidence() -> str:
    """Why we think there is an NVIDIA GPU here, or "" if we do not."""
    if Path("/proc/driver/nvidia/version").exists():
        return "the nvidia kernel driver is loaded"
    if any(Path("/dev").glob("nvidia[0-9]*")):
        return "an nvidia device node is present"
    if shutil.which("nvidia-smi"):
        return "nvidia-smi is present"
    return ""


def _has_open_clip() -> bool:
    from importlib.util import find_spec

    return find_spec("open_clip") is not None


def _any_render_node() -> bool:
    return any(Path("/dev/dri").glob("renderD*")) if Path("/dev/dri").is_dir() else False


# Kept for the environment probe in tests and for anyone reading logs.
def externally_managed() -> bool:
    """True if this interpreter belongs to a distribution rather than to a venv."""
    if os.environ.get("VIRTUAL_ENV") or sys.prefix != sys.base_prefix:
        return False
    import sysconfig

    return Path(sysconfig.get_path("stdlib"), "EXTERNALLY-MANAGED").exists()
