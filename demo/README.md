# Reproducing the demo

The recording in [../docs/demo.mp4](../docs/demo.mp4) is not a mockup. It is a
real terminal running the real CLI and a real browser driving the real server,
captured on a virtual X display so it can be reproduced without taking over
your desktop. These are the scripts that made it.

You need `ffmpeg`, `xdotool`, `Xvfb`, `chromium` and any terminal emulator
(the scripts use `alacritty`). Linux and X11 only.

## 1. Build a folder worth triaging

```bash
python demo/build_dataset.py ~/photos-demo --count 2000
```

Photographs come from [Lorem Picsum](https://picsum.photos). On top of those it
generates the things a real phone backup is actually full of: image-macro
memes, forwarded quote cards, and 110 duplicates across the three kinds
`photo-triage --dedupe` looks for (identical pixels in different files, app
recompressions, and year-later re-sends that are worse than the original).

Screenshots are the one thing it leaves to you, since a generated one would
look like a generated one. Capture a few full pages yourself, drop the PNGs in
a folder, and point at it:

```bash
python demo/build_dataset.py ~/photos-demo --count 2000 --captures ~/captures
```

## 2. Build the caches

```bash
photo-triage ~/photos-demo --no-serve
```

Do this before recording. The demo shows the everyday case, where the
embeddings already exist and startup is quick.

## 3. Record

```bash
Xvfb :78 -screen 0 1600x1000x24 &
DISPLAY=:78 alacritty --working-directory ~ \
  -o 'window.dimensions.columns=96' -o 'window.dimensions.lines=13' \
  -o 'font.size=13' -e bash --noprofile --rcfile demo/demorc &

ffmpeg -f x11grab -draw_mouse 1 -framerate 30 -video_size 1600x1000 \
  -i :78+0,0 -c:v libx264 -preset ultrafast -qp 0 -pix_fmt yuv444p raw.mkv &

python demo/drive.py
```

`drive.py` types the command, opens the printed URL, then searches, finds
visually similar images, filters, selects and quarantines, all by moving the
real pointer and sending real keystrokes. It writes `marks.json`, a timestamp
for each action.

## 4. Caption and cut

```bash
python demo/caption.py
```

Captions are timed from `marks.json` rather than lined up by eye, so they stay
pinned to the actions they describe. The script also eases a zoom in and out on
two moments, speeds the whole thing up, and drops the dead air.

It warns about any caption that is on screen too briefly to read, reckoning
roughly twelve characters a second. If you see that warning, the edit needs
another pass.

## What the edit changes, and what it does not

The cuts only ever remove waiting. Two of them cover Chrome starting up on a
display with no GPU. Two more cover the seconds that same software renderer
spends repainting the blurred toolbar after 136 tiles change state at once,
which is work real hardware does for free.

Nothing is staged. The 136 images quarantined during the recording show up in
the folder's `journal.jsonl` afterwards, and so does the undo.

## Constants you may need to change

`drive.py` holds pointer coordinates measured against a 1600x1000 window. If
you record at another size, take one screenshot and re-measure them. `SPEED`,
`CUTS`, `ZOOMS` and `LEAD_IN` in `caption.py` are all in recorded seconds, so
they stay meaningful if you re-record.
