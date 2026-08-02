"""Command line entry point.

The default invocation -- `photo-triage <folder>` -- builds every cache that is
missing and then serves the UI, because that is what somebody pointing the tool
at a folder for the first time wants. Every other verb is a deliberate
deviation from that, and the one destructive verb is a separate command that
asks first.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
import webbrowser
from pathlib import Path

from .cache import Cache
from .dedupe import find_duplicates
from .library import Library
from .pipeline import STAGES, Build
from .prompts import write_default_categories
from .quarantine import Quarantine
from .runtime import ensure_model_runtime

log = logging.getLogger(__name__)

HOST = "127.0.0.1"  # never configurable; see server.py.


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv or sys.argv[1:])
    _configure_logging(args.verbose)
    root = Path(args.folder).expanduser().resolve()
    if not root.is_dir():
        print(f"not a folder: {root}", file=sys.stderr)
        return 2

    if args.command == "purge":
        return _purge(root, args.yes)
    if args.command == "restore-all":
        return _restore_all(root)
    if args.status:
        return _status(root)
    if args.dedupe:
        return _dedupe(root)
    return _build_and_serve(root, args)


# -- verbs ----------------------------------------------------------------


def _build_and_serve(root: Path, args) -> int:
    stages = ("classify",) if args.reclassify else STAGES
    write_default_categories(Cache(root).dir)

    # Ask for the model runtime up here, on the main thread, before anything
    # starts: the install prompt needs a terminal, and the build thread has
    # none. If the user declines, the stages that need it are dropped and the
    # rest of the tool still works on whatever is already cached.
    needs_model = any(stage in ("embed", "classify") for stage in stages)
    if needs_model and not ensure_model_runtime(args.yes):
        stages = tuple(stage for stage in stages if stage not in ("embed", "classify"))
        log.warning("continuing without the model: no embedding, no text search")
    build = Build(root, device=args.device, stages=stages)

    if args.no_serve:
        _run_with_progress(build)
        return 1 if build.progress.error else 0

    import flask.cli

    from .server import create_app, pick_port

    # Flask's own two-line banner announces the app's import path and that
    # debug mode is off, neither of which means anything to someone triaging
    # photographs. Our own line above it says where to go.
    flask.cli.show_server_banner = lambda *args, **kwargs: None

    port = args.port or pick_port(root)
    build.start()
    app = create_app(root, build)
    url = f"http://{HOST}:{port}"
    print(f"photo-triage serving {root}\n  {url}", file=sys.stderr)
    if args.open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=HOST, port=port, threaded=True)
    return 0


def _status(root: Path) -> int:
    stats = Library.open(root).stats()
    print(f"folder       {root}")
    print(f"indexed      {stats.total:,}")
    print(f"  embedded   {stats.embedded:,}")
    print(f"  unreadable {stats.unreadable:,}")
    print(f"on disk      {stats.active:,}")
    print(f"quarantined  {stats.quarantined:,}")
    print(f"saved        {stats.saved:,}")
    if stats.categories:
        print("\ncategory            group    count")
        for name, bucket in sorted(
            stats.categories.items(), key=lambda item: -item[1]["count"]
        ):
            print(f"  {name:<18}{bucket['group']:<9}{bucket['count']:>7,}")
    return 0


def _dedupe(root: Path) -> int:
    """Report redundancy. Always a dry run -- this verb never moves a file."""
    library = Library.open(root)
    groups = find_duplicates(library)
    if not groups:
        print("no duplicates found")
        return 0
    redundant = sum(len(group.others) for group in groups)
    print(f"{len(groups):,} groups, {redundant:,} redundant copies (dry run)\n")
    for group in groups:
        print(f"[{group.kind}] keep  {library.records[group.keeper].rel}")
        for row in group.others:
            print(f"           drop  {library.records[row].rel}")
    print("\nNothing was moved. Select these in the UI to quarantine them.")
    return 0


def _purge(root: Path, assume_yes: bool) -> int:
    """Permanently empty the bin. The only destructive operation in the tool.

    Counts and measures first, prints what will go, and only then destroys --
    the create-artifact-and-delete-source-in-one-step mistake in PROJECT.md 9.7
    is exactly what this shape prevents.
    """
    cache = Cache(root)
    where = Quarantine(cache, cache.load_index())
    plan = where.purge(dry_run=True)
    if not plan.rows:
        print("quarantine is already empty")
        return 0
    print(f"{len(plan.rows):,} files, {plan.bytes_freed / 1e6:,.1f} MB")
    print(f"in {cache.quarantine_dir}")
    print("\nThis deletes them permanently. They cannot be restored.")
    if not assume_yes and input("Type 'purge' to confirm: ").strip() != "purge":
        print("cancelled")
        return 1
    done = where.purge(dry_run=False)
    print(f"purged {len(done.rows):,} files, freed {done.bytes_freed / 1e6:,.1f} MB")
    return 0


def _restore_all(root: Path) -> int:
    cache = Cache(root)
    where = Quarantine(cache, cache.load_index())
    result = where.restore(where.quarantined_rows())
    print(f"restored {len(result.moved):,} files")
    for row, reason in result.refused.items():
        print(f"  skipped {row}: {reason}", file=sys.stderr)
    return 0


# -- plumbing --------------------------------------------------------------


def _run_with_progress(build: Build) -> None:
    """Run a build in the foreground, printing a line per stage transition.

    Progress goes to stderr as whole lines rather than as a redrawn bar,
    because a bar piped through `tail` or a log file disappears entirely
    (PROJECT.md 9.8).
    """
    build.start()
    last = ("", -1)
    while build.progress.running or build.progress.stage:
        stage, done, total = (
            build.progress.stage,
            build.progress.done,
            build.progress.total,
        )
        step = done // 500 if total > 2000 else done
        # A stage with nothing to do says nothing: "embed: 0/0" reads as a
        # failure when it means the cache was already complete.
        if stage and total and (stage, step) != last:
            eta = build.progress.eta_seconds
            suffix = f", {eta / 60:.0f} min left" if eta and eta > 90 else ""
            print(f"{stage}: {done:,}/{total:,}{suffix}", file=sys.stderr)
            last = (stage, step)
        time.sleep(0.2)
    if build.progress.error:
        print(f"failed: {build.progress.error}", file=sys.stderr)
    else:
        print("done", file=sys.stderr)


def _configure_logging(verbose: bool) -> None:
    """Show this tool's messages, and nothing else's, unless -v is given.

    open_clip narrates model loading with bare `logging.info` calls, which land
    on the *root* logger -- so raising the level of a logger named after it
    does nothing, and the twenty lines about tokenizer configs bury the one
    line saying where the server is. The fix is the other way round: leave the
    root at WARNING and turn our own package up.
    """
    logging.basicConfig(format="%(message)s", stream=sys.stderr)
    logging.getLogger().setLevel(logging.DEBUG if verbose else logging.WARNING)
    logging.getLogger("photo_triage").setLevel(
        logging.DEBUG if verbose else logging.INFO
    )
    if not verbose:
        logging.getLogger("werkzeug").setLevel(logging.ERROR)


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="photo-triage",
        description="Find the photos that matter inside a folder full of junk.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="",
        metavar="{purge,restore-all}",
        help="omit to scan, embed, classify, thumbnail and serve",
    )
    parser.add_argument("folder", nargs="?", default=".")
    parser.add_argument("--no-serve", action="store_true", help="build caches only")
    parser.add_argument("--reclassify", action="store_true",
                        help="re-run classification with edited prompts.json")
    parser.add_argument("--status", action="store_true", help="print counts and exit")
    parser.add_argument("--dedupe", action="store_true",
                        help="report exact and near duplicates; never moves anything")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--open", action="store_true", help="open a browser")
    parser.add_argument("--port", type=int, default=0,
                        help="override the port derived from the folder")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="skip the purge confirmation")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    # `photo-triage ~/pics` puts the folder in `command`; `photo-triage purge
    # ~/pics` fills both. Sorting that out here keeps every caller below from
    # having to know the argument grammar.
    if args.command not in ("", "purge", "restore-all"):
        args.folder, args.command = args.command, ""
    return args


if __name__ == "__main__":
    sys.exit(main())
