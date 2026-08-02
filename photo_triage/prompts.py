"""Category definitions: the words the classifier compares pictures against.

The category set and its grouping into JUNK / REVIEW / KEEP is one design
decision and lives only here. Because image embeddings are cached, editing
these costs seconds to re-apply, so they are meant to be tuned rather than
treated as fixed (PROJECT.md 6).

A user's edits go in `<folder>/.phototriage/prompts.json`, which overrides the
defaults below wholesale. Grouping is derived from the category name's presence
in JUNK/REVIEW below, so a category invented by the user lands in KEEP -- the
conservative default, since nothing unknown should ever be pre-flagged as
rubbish.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_CATEGORIES: dict[str, list[str]] = {
    "meme_text": [
        "a meme with large text captions",
        "an image macro with white impact font text",
        "a funny meme picture shared on social media",
        "a picture with a joke written across it",
    ],
    "screenshot": [
        "a screenshot of a phone screen",
        "a screenshot of a chat conversation",
        "a screenshot of a website or an app interface",
        "a screenshot of social media post",
    ],
    "text_graphic": [
        "a poster with a written message",
        "an inspirational quote written on a background",
        "a flyer or advertisement with text",
    ],
    "sticker_art": [
        "a cartoon drawing or illustration",
        "a sticker or clipart graphic",
        "a digital drawing with a plain background",
    ],
    "document": [
        "a scanned document or paperwork",
        "a photograph of a receipt or an invoice",
    ],
    "people_photo": [
        "a candid photograph of people",
        "a selfie of a person",
        "a group photo of friends or family",
    ],
    "event_photo": [
        "a photograph taken at a party or celebration",
        "a photo from a wedding or a birthday",
    ],
    "place_photo": [
        "a photograph of a landscape or scenery",
        "a travel photo of a place or a building",
    ],
    "food_photo": ["a photograph of a meal or food on a plate"],
    "animal_photo": ["a photograph of a pet dog or cat", "a photo of an animal"],
    "product": [
        "a product photo for online shopping",
        "a catalogue photo of an item for sale",
    ],
}

# JUNK is the only group the UI will offer to sweep in bulk. `product` and
# `document` are deliberately absent from it: CLIP maps "photograph containing
# an object" onto `product`, which covers most indoor personal photos, and it
# was the source of the prototype's worst false positives (PROJECT.md 9.1).
JUNK = frozenset({"meme_text", "screenshot", "text_graphic", "sticker_art"})
REVIEW = frozenset({"product", "document"})

# Below this softmax confidence the model is effectively guessing between
# neighbouring categories. Review mode draws its rule here (DESIGN.md 4.5).
# Measured against the 23,584-image corpus described in PROJECT.md.
GUESSING_BELOW = 0.35


def group_of(category: str) -> str:
    """The UI bucket -- "junk", "review" or "keep" -- for a category name.

    Unknown and empty names group as "keep", so an unclassified or
    user-invented category is never pre-flagged as rubbish.
    """
    if category in JUNK:
        return "junk"
    if category in REVIEW:
        return "review"
    return "keep"


def load_categories(cache_dir: Path) -> dict[str, list[str]]:
    """The prompt set for this folder: the user's `prompts.json`, else defaults.

    A malformed or unreadable override falls back to the defaults with a
    warning rather than failing -- a typo in a tuning file should cost the user
    a re-edit, not a run.
    """
    override = cache_dir / "prompts.json"
    if not override.exists():
        return dict(DEFAULT_CATEGORIES)
    try:
        loaded = json.loads(override.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or not loaded:
            raise ValueError("expected a non-empty object of category -> prompts")
        for name, phrasings in loaded.items():
            if not isinstance(phrasings, list) or not all(
                isinstance(p, str) for p in phrasings
            ):
                raise ValueError(f"category {name!r} must map to a list of strings")
        return loaded
    except Exception as exc:
        log.warning("ignoring %s (%s); using default prompts", override, exc)
        return dict(DEFAULT_CATEGORIES)


def write_default_categories(cache_dir: Path) -> Path:
    """Materialise the defaults as an editable `prompts.json`. Never overwrites."""
    override = cache_dir / "prompts.json"
    if not override.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        override.write_text(
            json.dumps(DEFAULT_CATEGORIES, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return override
