"""photo-triage: local, offline semantic triage for folders full of images.

See PROJECT.md for architecture, DESIGN.md for the UI, STYLE.md for how the
code here is meant to be shaped.
"""

__version__ = "0.2.3"


def _teach_pillow_heic() -> None:
    """Let `Image.open` read HEIC, the format every recent iPhone shoots.

    Pillow cannot decode HEIC on its own, so without this an iCloud export is
    indexed by the scanner -- the magic bytes say "image" -- and then fails to
    decode, landing in the unreadable bucket. The photographs are fine; only
    the decoder was missing.

    This runs on package import rather than at the first `Image.open` because
    the scan pool and the embedding pass both open images in child processes.
    Import is the one moment guaranteed to precede every one of them, whether
    a child was forked or started fresh.
    """
    try:
        import pillow_heif
    except ImportError:  # a source install without the extra; stills still work
        return
    pillow_heif.register_heif_opener()


_teach_pillow_heic()
