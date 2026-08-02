"""Drive a real photo-triage session on a virtual X display, for recording.

Everything shown is genuine: a real terminal running the real CLI, a real
Chrome loading the real server on real photographs, and a real pointer moved
through XTEST. The only thing the script does that a person would not is move
the mouse in straight lines at a constant speed.

Beat timings are written to marks.json so the caption track is generated from
what actually happened rather than eyeballed against the finished video.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

DISPLAY = ":78"
SCRATCH = Path(__file__).parent
FOLDER = SCRATCH / "real"

ENV = {"DISPLAY": DISPLAY, "PATH": "/usr/bin:/bin", "HOME": str(Path.home())}

# Measured off a screenshot of the running app at 1600x1000.
TERM_URL = (140, 100)
SEARCH = (400, 38)
GROUP_FILTER = (1032, 38)
DOG_TILE = (1077, 738)
GRID_MIDDLE = (700, 520)

marks: list[dict] = []
start = time.time()


def x(*args: str) -> str:
    return subprocess.run(
        ["xdotool", *args], env=ENV, capture_output=True, text=True
    ).stdout.strip()


def mark(label: str) -> None:
    """Record when a beat happened, so a caption can be timed exactly to it."""
    marks.append({"t": round(time.time() - start, 2), "label": label})
    print(f"{marks[-1]['t']:7.2f}  {label}", flush=True)


def glide(to_x: int, to_y: int, steps: int = 28, pause: float = 0.014) -> None:
    """Move the pointer in a straight line, slowly enough to follow on video."""
    shell = dict(
        line.split("=")
        for line in x("getmouselocation", "--shell").splitlines()
        if "=" in line
    )
    from_x, from_y = int(shell.get("X", to_x)), int(shell.get("Y", to_y))
    for step in range(1, steps + 1):
        x(
            "mousemove",
            str(round(from_x + (to_x - from_x) * step / steps)),
            str(round(from_y + (to_y - from_y) * step / steps)),
        )
        time.sleep(pause)


def _printed_url() -> str:
    """The address the CLI has just printed in the terminal.

    The port is derived from the folder rather than fixed, so this asks the
    same function the CLI asks and gets the same answer, rather than scraping
    it back off the screen.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from photo_triage.server import pick_port

    return f"http://127.0.0.1:{pick_port(FOLDER.resolve())}"


def focus(window: str) -> None:
    x("windowfocus", window)
    x("windowactivate", window)
    x("windowraise", window)


# ── 1. the terminal ────────────────────────────────────────────────────────

term = x("search", "--onlyvisible", "--class", "Alacritty").splitlines()[0]
focus(term)
glide(600, 300, steps=10)
time.sleep(2.0)

mark("start")
x("type", "--delay", "58", "photo-triage ~/photos")
time.sleep(0.9)
x("key", "Return")
mark("run")
time.sleep(6.5)
mark("serving")
url = _printed_url()

# ── 2. open the printed URL ────────────────────────────────────────────────

glide(*TERM_URL, steps=34)
time.sleep(0.7)
x("click", "1")
mark("open")
subprocess.Popen(
    [
        "chromium",
        f"--user-data-dir={SCRATCH / 'chrome-profile'}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-infobars",
        "--window-position=0,0",
        "--window-size=1600,1000",
        f"--app={url}",
    ],
    env=ENV,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
time.sleep(9.0)
chrome = x("search", "--onlyvisible", "--class", "chromium").splitlines()
if chrome:
    focus(chrome[-1])
time.sleep(2.0)
mark("grid")
glide(*GRID_MIDDLE)
time.sleep(2.0)

# ── 3. search by what the picture looks like ───────────────────────────────

glide(*SEARCH)
x("click", "1")
time.sleep(0.7)
mark("search")
x("type", "--delay", "95", "a dog")
time.sleep(4.0)
mark("ranked")
time.sleep(3.0)

# ── 4. find visually similar ───────────────────────────────────────────────

glide(240, 300)
time.sleep(0.6)
x("click", "1")          # move focus out of the search field
time.sleep(0.3)
x("key", "c")            # clear the selection that click made
time.sleep(0.4)
glide(240, 300, steps=6)
mark("hover")
time.sleep(1.2)
x("key", "f")
mark("similar")
time.sleep(4.5)

# ── 5. filter, select, quarantine ──────────────────────────────────────────

glide(*GROUP_FILTER)
time.sleep(0.8)
x("click", "1")
time.sleep(1.2)
x("key", "Down")
time.sleep(0.6)
x("key", "Return")
mark("filter")
time.sleep(3.5)

glide(*GRID_MIDDLE)
time.sleep(0.6)
x("click", "1")
mark("select-one")
time.sleep(2.0)
x("key", "a")
mark("select-all")
time.sleep(6.0)

x("key", "d")
mark("arm")
time.sleep(9.0)
x("key", "d")
mark("quarantine")
time.sleep(5.0)

x("key", "u")
mark("undo")
time.sleep(4.5)

mark("end")
json.dump(marks, open(SCRATCH / "marks.json", "w"), indent=1)
