"""Getting the right PyTorch onto the machine, without asking the user to.

PyTorch ships a different wheel per accelerator and they are not
interchangeable: the default PyPI wheel is a CUDA build, and installing it on a
Radeon yields a working-but-CPU-bound install that is twenty times slower than
the hardware allows. That is a question the user cannot be expected to answer
correctly, and it is one we can answer by looking at the machine -- so this
module looks, and installs.

Which is why `torch` is absent from the project's dependencies. Declaring it
would let pip resolve the *wrong* wheel before this code ever ran, and would
silently replace a correct GPU build that the user had already installed by
hand. The dependency is real; it is simply not one a resolver can get right.

Nothing here runs unless a stage actually needs the model. Browsing, searching
by category, quarantining and restoring an already-embedded folder never touch
it, so a user who only wants to look at their existing triage never waits on a
3 GB download.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# The wheel index for each accelerator. "" means the default PyPI index, whose
# wheels bundle CUDA and, on macOS, MPS.
# ROCm is pinned rather than left to float: the gfx1031/gfx1032 architecture
# override that makes consumer Radeons work is documented to segfault from
# ROCm 6.4.3 up, and 6.3 is the last line verified against that card family.
_ROCM_PIN = "rocm6.3"

_INDEX = {
    "cuda": "",
    "mps": "",
    "rocm": f"https://download.pytorch.org/whl/{_ROCM_PIN}",
    "cpu": "https://download.pytorch.org/whl/cpu",
}


@dataclass
class Backend:
    """The accelerator this machine has, and how to install for it.

    `name` is one of "cuda", "rocm", "mps" or "cpu". `why` is the evidence it
    was chosen on, quoted back to the user so an unexpected answer -- a CPU
    install on a machine with a graphics card -- is debuggable rather than
    mysterious.
    """

    name: str
    why: str

    @property
    def packages(self) -> list[str]:
        return ["torch", "torchvision", "open_clip_torch"]

    @property
    def index_url(self) -> str:
        return _INDEX[self.name]

    def install_command(self, executable: str | None = None) -> list[str]:
        """The exact argv that installs this backend. Never a shell string.

        Paths on this machine may contain spaces, accents and emoji, so
        everything that runs a subprocess in this codebase passes an argument
        list (PROJECT.md 9.8).
        """
        base = _installer(executable)
        index = ["--index-url", self.index_url] if self.index_url else []
        return [*base, *index, *self.packages]


def detect_backend() -> Backend:
    """What this machine can accelerate with. Cheap, and never raises.

    Looks at the hardware rather than at what happens to be installed, so it
    gives the same answer before and after an install and can therefore be used
    to notice a mismatch.
    """
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return Backend("mps", "Apple Silicon")
    if shutil.which("nvidia-smi"):
        return Backend("cuda", "nvidia-smi is present")
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
    """Make sure PyTorch and open_clip are importable and match the hardware.

    Returns True when the model can be loaded. Installs what is missing, after
    printing exactly what it is about to run and taking a yes -- downloading
    three gigabytes into someone's environment is not something to do silently,
    but choosing *which* three gigabytes is not something to ask about.

    A usable install that merely disagrees with the detected hardware is left
    alone with a warning: a deliberate CPU build on a machine that has a GPU is
    a legitimate choice, and replacing it uninvited would be worse than slow.
    """
    wanted = detect_backend()
    present = installed_backend()

    if present is not None and _has_open_clip():
        if present != wanted.name and not (present == "cuda" and wanted.name == "mps"):
            log.warning(
                "PyTorch is a %s build but this machine has %s (%s). "
                "Embedding will work and will be slow. To switch:\n  %s",
                present,
                wanted.name,
                wanted.why,
                " ".join(wanted.install_command()),
            )
        return True

    missing = "PyTorch and open_clip are" if present is None else "open_clip is"

    if _externally_managed():
        # A distribution-managed Python is not ours to install into. Installing
        # here would either fail on PEP 668 or, worse, succeed and shadow the
        # distribution's own build with one that does not match its drivers.
        print(
            f"\n{missing} not installed. They are what reads the images.\n"
            f"Detected: {wanted.name} ({wanted.why})\n"
            f"This Python is managed by your distribution, so photo-triage will "
            f"not install into it. Either:\n"
            f"  {_distro_hint(wanted)}\n"
            f"  or run photo-triage from a virtualenv, where it installs them "
            f"itself.\n",
            file=sys.stderr,
        )
        return False

    print(
        f"\n{missing} not installed yet. They are what reads the images.\n"
        f"Detected: {wanted.name} ({wanted.why})\n"
        f"Will run: {' '.join(wanted.install_command())}\n",
        file=sys.stderr,
    )
    if not assume_yes and not _confirm():
        print(
            "Skipped. Install it yourself and re-run, or pass --yes.",
            file=sys.stderr,
        )
        return False

    result = subprocess.run(wanted.install_command())
    if result.returncode != 0:
        print(
            "\nThe install failed. Run the command above by hand to see why; "
            "everything except embedding and text search works without it.",
            file=sys.stderr,
        )
        return False
    return installed_backend() is not None and _has_open_clip()


# -- internals -------------------------------------------------------------


def _installer(executable: str | None = None) -> list[str]:
    """The installer that owns this environment.

    A `uv` virtualenv has no pip in it at all, so reaching for `pip` there
    fails in a way that reads as a broken tool rather than as a missing
    package. Prefer uv when it is on PATH and this is a venv it could have made.
    """
    python = executable or sys.executable
    if shutil.which("uv") and os.environ.get("VIRTUAL_ENV"):
        return ["uv", "pip", "install", "--python", python]
    return [python, "-m", "pip", "install"]


def _externally_managed() -> bool:
    """True if this interpreter belongs to a distribution rather than to us.

    PEP 668 marks such an install with a file next to the standard library.
    A virtualenv is ours whatever it was built from, so an active one always
    wins over the marker.
    """
    if os.environ.get("VIRTUAL_ENV") or sys.prefix != sys.base_prefix:
        return False
    import sysconfig

    return Path(sysconfig.get_path("stdlib"), "EXTERNALLY-MANAGED").exists()


def _distro_hint(backend: Backend) -> str:
    """The package-manager line to suggest, for distributions we recognise."""
    torch_package = {
        "rocm": "python-pytorch-rocm",
        "cuda": "python-pytorch-cuda",
    }.get(backend.name, "python-pytorch")
    if shutil.which("pacman"):
        return f"install {torch_package} and python-open-clip-torch (AUR)"
    if shutil.which("apt-get"):
        return "install python3-torch and python3-open-clip via apt"
    return f"install {torch_package} and open_clip with your package manager"


def _has_open_clip() -> bool:
    from importlib.util import find_spec

    return find_spec("open_clip") is not None


def _any_render_node() -> bool:
    return any(Path("/dev/dri").glob("renderD*")) if Path("/dev/dri").is_dir() else False


def _confirm() -> bool:
    try:
        return input("Install now? [Y/n] ").strip().lower() in ("", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False
