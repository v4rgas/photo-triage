"""Assemble the demo folder: real photographs, plus the junk a real one has.

A phone backup is not a photo library. It is photographs mixed with memes,
screenshots, forwarded quote cards, and the same picture four times over. A
demo built only from nice photographs would show search working and prove
nothing about the part that matters, so this builds all of it.

Photographs come from Lorem Picsum, which serves real, openly licensed images.
Screenshots are sliced from full-page captures of real websites, which you
supply yourself (see the README); everything else is generated here.

    python demo/build_dataset.py ~/photos-demo --count 2000

Nothing about this script is needed to use photo-triage. It exists so the demo
recording can be reproduced from scratch.
"""

from __future__ import annotations

import argparse
import os
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

FOLDERS = ("holidays", "trip-2024", "family", "received", "work", "school",
           "garden", "nights-out")

TOP = ["WHEN YOU FINALLY", "NOBODY:", "ME EXPLAINING", "THAT MOMENT WHEN",
       "MY FACE WHEN", "EVERYBODY GANGSTA", "POV:", "HIM: I'M FINE"]
BOTTOM = ["FIND THE PHOTO", "ABSOLUTELY NOBODY", "WHY IT MATTERS",
          "IT ACTUALLY WORKS", "THE WIFI DROPS", "UNTIL THE UPDATE",
          "ALSO HIM:", "IT'S MONDAY AGAIN"]
QUOTES = ["The best time to plant a tree\nwas 20 years ago.\nThe second best "
          "time is now.", "Live simply.\nDream big.\nBe grateful.",
          "Good things take time.", "She believed she could,\nso she did.",
          "Every day is a\nfresh start."]

RESEND_AGE_SECONDS = 400 * 24 * 3600


def font(size: int) -> ImageFont.FreeTypeFont:
    """A bold face for captions, falling back to whatever the system has."""
    for name in ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Arial_Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def fetch_photographs(root: Path, count: int) -> None:
    """Download real photographs, spread across folders with dated names."""
    def one(index: int) -> None:
        folder = root / FOLDERS[index % len(FOLDERS)]
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"IMG-2024{index:04d}.jpg"
        if target.exists():
            return
        try:
            with urllib.request.urlopen(
                f"https://picsum.photos/seed/pt{index}/1024/768", timeout=30
            ) as response:
                target.write_bytes(response.read())
        except Exception:
            target.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=24) as pool:
        list(pool.map(one, range(1, count + 1)))
    for path in root.rglob("*.jpg"):
        if path.stat().st_size < 10_000:
            path.unlink()


def caption(image: Image.Image, text: str, top: int) -> None:
    """Draw impact-style text with a hard outline, centred horizontally."""
    draw = ImageDraw.Draw(image)
    face = font(max(22, image.width // 14))
    box = draw.textbbox((0, 0), text, font=face)
    left = (image.width - (box[2] - box[0])) / 2
    for offset_x in (-3, 0, 3):
        for offset_y in (-3, 0, 3):
            draw.text((left + offset_x, top + offset_y), text, font=face, fill=(0, 0, 0))
    draw.text((left, top), text, font=face, fill=(255, 255, 255))


def add_memes(root: Path, photographs: list[Path], rng: random.Random, count: int) -> None:
    folder = root / "received" / "memes"
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        with Image.open(rng.choice(photographs)) as source:
            image = source.convert("RGB")
        image.thumbnail((720, 720))
        caption(image, rng.choice(TOP), 12)
        caption(image, rng.choice(BOTTOM), image.height - image.width // 14 - 24)
        image.save(folder / f"IMG-2025{index:04d}-WA{index:04d}.jpg", quality=82)


def add_quote_cards(root: Path, rng: random.Random, count: int) -> None:
    folder = root / "received" / "forwarded"
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        image = Image.new("RGB", (720, 720),
                          tuple(rng.randrange(60, 200) for _ in range(3)))
        draw = ImageDraw.Draw(image)
        face = font(40)
        text = rng.choice(QUOTES)
        box = draw.multiline_textbbox((0, 0), text, font=face, align="center")
        draw.multiline_text(
            ((720 - (box[2] - box[0])) / 2, (720 - (box[3] - box[1])) / 2),
            text, font=face, fill=(255, 255, 255), align="center", spacing=14,
        )
        image.save(folder / f"IMG-2025{index:04d}-WA9{index:03d}.jpg", quality=85)


def add_duplicates(root: Path, photographs: list[Path], rng: random.Random) -> None:
    """The three kinds of redundancy a messaging export actually contains.

    Exact copies have identical pixels but different file bytes, which is what
    a forward-and-forward-back produces. Near copies have been resized or
    recompressed on the way through an app. Re-sends are near copies whose
    original is dated a year earlier, and are deliberately the worse of the
    two, because that is the direction real re-sends run in.
    """
    forwarded = root / "received" / "forwarded-back"
    resends = root / "received" / "2026-resends"
    forwarded.mkdir(parents=True, exist_ok=True)
    resends.mkdir(parents=True, exist_ok=True)

    for index, source in enumerate(rng.sample(photographs, 40)):
        with Image.open(source) as original:
            original.convert("RGB").save(
                forwarded / f"IMG-2025{index:04d}-WA{7000 + index}.jpg",
                quality=100, subsampling=0,
            )

    for index, source in enumerate(rng.sample(photographs, 45)):
        with Image.open(source) as original:
            image = original.convert("RGB")
        kind = rng.choice(["resize", "recompress", "brighten", "crop"])
        if kind == "resize":
            image = image.resize((image.width // 2, image.height // 2))
        elif kind == "brighten":
            image = ImageEnhance.Brightness(image).enhance(1.18)
        elif kind == "crop":
            margin = image.width // 40
            image = image.crop((margin, margin,
                                image.width - margin, image.height - margin))
        image.save(forwarded / f"IMG-2025{index:04d}-WA{8000 + index}.jpg",
                   quality=55 if kind == "recompress" else 80)

    aged = time.time() - RESEND_AGE_SECONDS
    for index, source in enumerate(rng.sample(photographs, 25)):
        with Image.open(source) as original:
            image = original.convert("RGB")
        smaller = image.resize((max(image.width // 3, 64), max(image.height // 3, 64)))
        smaller.save(resends / f"IMG-2026{index:04d}-WA{9000 + index}.jpg", quality=58)
        os.utime(source, (aged, aged))


def slice_screenshots(captures: Path, root: Path, limit: int = 40) -> int:
    """Cut full-page website captures into phone-shaped screenshots."""
    if not captures.is_dir():
        return 0
    folder = root / "received"
    folder.mkdir(parents=True, exist_ok=True)
    made = 0
    for capture in sorted(captures.glob("*.png")):
        with Image.open(capture) as page:
            image = page.convert("RGB")
        tile_height = int(image.width * 16 / 9)
        for top in range(0, max(image.height - tile_height, 1), max(tile_height // 2, 1)):
            crop = image.crop((0, top, image.width, min(top + tile_height, image.height)))
            if crop.height < tile_height * 0.6:
                continue
            crop.save(folder / f"Screenshot_2025010{made % 9}-{100000 + made * 137}.jpg",
                      quality=88)
            made += 1
            if made >= limit:
                return made
    return made


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--count", type=int, default=2000,
                        help="how many photographs to download")
    parser.add_argument("--captures", type=Path, default=None,
                        help="folder of full-page website PNGs to slice up")
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    root = args.folder.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"downloading {args.count} photographs")
    fetch_photographs(root, args.count)
    photographs = sorted(p for p in root.rglob("*.jpg") if ".phototriage" not in p.parts)
    print(f"  got {len(photographs)}")

    add_memes(root, photographs, rng, 45)
    add_quote_cards(root, rng, 30)
    add_duplicates(root, photographs, rng)
    shots = slice_screenshots(args.captures, root) if args.captures else 0
    print(f"  added 45 memes, 30 quote cards, 110 duplicates, {shots} screenshots")

    total = len([p for p in root.rglob("*.jpg") if ".phototriage" not in p.parts])
    print(f"{total} images in {root}")


if __name__ == "__main__":
    main()
