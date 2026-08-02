"""Burn captions and zoom cut-ins onto the raw screen capture.

Caption times come from marks.json -- the timestamps the driver recorded as it
performed each action -- so the text is synchronised with what actually
happened rather than lined up by eye afterwards.

Caption strings go in files rather than inline in the filter graph: ffmpeg's
filter syntax needs commas, colons and quotes escaped twice over, and one
missed backslash turns into a silently truncated caption.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

SCRATCH = Path(__file__).parent
RAW = SCRATCH / "raw.mkv"
FONT = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"
W, H = 1600, 1000

marks = {m["label"]: m["t"] for m in json.load(open(SCRATCH / "marks.json"))}
end = marks["end"] + 3.0

# (start, stop, text). Stops are the next beat, so nothing overlaps.
LEAD_IN = 2.6  # recorded seconds a caption may start before its own action

# (start, stop, text). Stops are the next beat, so nothing overlaps.
# Each line is written to fit the time its own beat lasts, at roughly twelve
# readable characters a second. Two beats are too short to hold a caption of
# their own at this speed, so they share one with the beat beside them rather
# than flashing something nobody can read.
CAPTIONS = [
    (0.0, marks["run"], "2,225 images, no usable metadata."),
    (marks["run"], marks["open"],
     "One command, and it runs entirely on this machine."),
    (marks["open"], marks["grid"], "Open the URL it prints."),
    (marks["grid"], marks["search"], "Embedded locally with CLIP."),
    (marks["search"], marks["hover"] - LEAD_IN,
     "Type what the picture looks like."),
    # Reaching a photo and pressing F is barely a second apart, so the
    # instruction goes up before the pointer arrives and stays through the
    # keypress, which is the order someone wants to read it in anyway.
    (marks["hover"] - LEAD_IN, marks["similar"] + 1.0,
     "Hover any photo and press F"),
    (marks["similar"] + 1.0, marks["filter"],
     "It finds the near copies and re-sends."),
    (marks["filter"], marks["select-one"], "Now only the junk it found."),
    (marks["select-one"], marks["arm"],
     "Click selects one. A selects them all."),
    (marks["arm"], marks["quarantine"], "D asks first: Quarantine 136?"),
    (marks["quarantine"], marks["undo"], "They move to a mirrored folder."),
    (marks["undo"], end, "U puts them all back."),
]

# How much faster than life the finished piece runs. Every duration below is
# written in recorded seconds and divided by this at the end, so the captions
# stay pinned to the actions no matter what this is set to.
SPEED = 1.55

# Push in where the detail matters: the toolbar while a query is being typed,
# and the selection bar at the moment the destructive key is pressed. Each is
# a centre to move toward and a magnification to reach, not a fixed crop --
# the movement is what makes it read as a camera rather than as a jump cut.
ZOOMS = [
    # Pulls back out as the caption stops talking about the toolbar.
    (marks["search"], marks["hover"] - LEAD_IN, (520, 280), 1.62),
    (marks["arm"], marks["undo"], (800, 700), 1.62),
]

RAMP = 0.75  # seconds of *finished* video spent easing in, and again easing out


def zoompan(cx: int, cy: int, target: float, duration: float) -> str:
    """A zoom that eases in, holds, and eases out, centred on (cx, cy).

    `duration` is in recorded seconds, because this runs before the speed
    change: zoompan's `fps` option re-stamps its output, so putting it after
    setpts silently undoes the speed-up on precisely the zoomed stretches and
    drifts every later caption off its action.

    Smoothstep rather than a linear ramp: a constant-velocity zoom starts and
    stops abruptly and reads as a machine moving the frame. This accelerates
    and decelerates, which is the same easing the UI itself uses.

    The pan needs no separate animation. `x` is the left edge of the visible
    region clamped into the frame, and at magnification 1 that clamp can only
    resolve to 0 -- so the view begins as the full frame and converges on the
    centre as it magnifies, with nothing to keep in step by hand.
    """
    ease = "(P*P*(3-2*P))"
    ramp = RAMP * SPEED  # recorded seconds that become RAMP finished seconds
    rising = f"min(it/{ramp}\\,1)"
    falling = f"min(({duration}-it)/{ramp}\\,1)"
    progress = f"max(0\\,min({rising}\\,{falling}))"
    zoom = f"1+{target - 1:.4f}*{ease.replace('P', progress)}"
    return (
        f"zoompan=z='{zoom}'"
        f":x='max(0\\,min(iw-iw/zoom\\,{cx}-(iw/zoom)/2))'"
        f":y='max(0\\,min(ih-ih/zoom\\,{cy}-(ih/zoom)/2))'"
        f":d=1:s={W}x{H}:fps=30"
    )

# Dead air to drop. Two kinds: Chrome's blank window while it starts up, and
# the lag before a repaint lands. The recording display has no GPU, so the
# blurred selection bar has to be recomposited in software over a grid where
# 136 tiles just changed, and that takes seconds it would not take on real
# hardware. Cutting waiting is ordinary screencast editing; nothing that
# happens is removed, only time in which nothing happens.
CUTS = [(14.8, 17.6), (19.6, 22.9), (52.6, 54.9), (58.6, 61.7)]


def kept() -> list[tuple[float, float]]:
    """The stretches of input that survive the cuts, in order."""
    spans, at = [], 0.0
    for start, stop in sorted(CUTS):
        if start > at:
            spans.append((at, start))
        at = max(at, stop)
    if at < end:
        spans.append((at, end))
    return spans


def out_time(t: float) -> float:
    """Where a recorded timestamp lands on the finished, sped-up timeline."""
    elapsed = 0.0
    for start, stop in kept():
        if t >= stop:
            elapsed += stop - start
        elif t > start:
            return (elapsed + (t - start)) / SPEED
    return elapsed / SPEED


def segments() -> list[tuple[float, float, tuple, float]]:
    """Every kept stretch, split again wherever a zoom starts or stops."""
    edges = sorted({e for span in kept() for e in span}
                   | {e for start, stop, *_ in ZOOMS for e in (start, stop)})
    pieces = []
    for start, stop in zip(edges, edges[1:]):
        if not any(low <= start and stop <= high for low, high in kept()):
            continue
        move = next(
            ((centre, z) for zs, ze, centre, z in ZOOMS if zs <= start and stop <= ze),
            None,
        )
        pieces.append((start, stop, *(move or (None, 1.0))))
    return pieces


chain = [f"[0:v]split={len(segments())}" + "".join(f"[s{i}]" for i in range(len(segments())))]
labels = []
for index, (start, stop, centre, magnify) in enumerate(segments()):
    steps = [f"trim={start}:{stop}", "setpts=PTS-STARTPTS"]
    if centre:
        steps.append(zoompan(*centre, magnify, stop - start))
    steps.append(f"setpts=PTS/{SPEED}")  # last, so the speed-up survives zoompan
    chain.append(f"[s{index}]" + ",".join(steps) + f"[v{index}]")
    labels.append(f"[v{index}]")
chain.append("".join(labels) + f"concat=n={len(labels)}:v=1:a=0[cat]")

# Roughly 12 characters a second is a comfortable subtitle rate, with a floor
# for the very short ones. Anything under this is on screen too briefly to
# read, which is a fault in the edit rather than something to notice later in
# the finished file.
for start, stop, text in CAPTIONS:
    shown = out_time(stop) - out_time(start)
    needed = max(1.6, len(text) / 12.0)
    if shown < needed:
        print(f"  too brief: {shown:.1f}s for {needed:.1f}s of text: {text}")

draws = []
for index, (start, stop, text) in enumerate(CAPTIONS):
    path = SCRATCH / f"cap{index:02d}.txt"
    path.write_text(text, encoding="utf-8")
    draws.append(
        f"drawtext=fontfile={FONT}:textfile={path}:"
        f"fontsize=30:fontcolor=white:box=1:boxcolor=black@0.66:boxborderw=22:"
        # High enough to clear the selection bar, which sits at the bottom of
        # the frame and is exactly what the last few captions point at.
        f"x=(w-text_w)/2:y=h-185:"
        f"enable='between(t\\,{out_time(start):.2f}\\,{out_time(stop):.2f})'"
    )
chain.append("[cat]" + ",".join(draws) + "[out]")

script = SCRATCH / "filter.txt"
script.write_text(";\n".join(chain), encoding="utf-8")

subprocess.run(
    ["ffmpeg", "-loglevel", "error", "-stats", "-i", str(RAW),
     "-filter_complex_script", str(script), "-map", "[out]",
     "-c:v", "libx264", "-preset", "slow", "-crf", "20",
     "-pix_fmt", "yuv420p", "-movflags", "+faststart",
     "-y", str(SCRATCH / "photo-triage-demo.mp4")],
    check=True,
)
print("wrote photo-triage-demo.mp4")
