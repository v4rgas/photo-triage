"""Stage 5 -- the HTTP layer.

Thin by design. Every handler does three things: turn request arguments into a
`Query` or a list of row ids, call the Library or the Quarantine, and serialise
what comes back. No filtering, no path arithmetic and no protection logic lives
here, because all three are decisions that belong below the transport and must
hold for a POST made with curl exactly as they do for one made by the page.

Binds to 127.0.0.1 only. This is someone's personal photo library and there is
no authentication anywhere in it; there must never be a host argument.

The port is derived from the folder rather than fixed. 5000 is Flask's default
and is therefore busy on any machine that runs another Flask app, and on macOS
it is taken by AirPlay Receiver out of the box. Deriving it also means two
folders can be triaged at once without either one having to be told about the
other, and that the address for a given folder is the same tomorrow.
"""

from __future__ import annotations

import hashlib
import logging
import socket
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.exceptions import HTTPException

from .library import Library, Query
from .pipeline import Build
from .prompts import GUESSING_BELOW

log = logging.getLogger(__name__)

DEFAULT_PAGE = 300
MAX_PAGE = 1000

# Above Linux's default ephemeral range, which ends at 60999, so a derived port
# cannot collide with a source port the kernel hands out for an outgoing
# connection. Well clear of everything in /etc/services.
_PORT_LOW = 61000
_PORT_SPAN = 4000


def pick_port(root: Path) -> int:
    """A free port for this folder, stable across runs.

    Derived by hashing the folder's path, so the same folder always answers at
    the same address and a bookmark keeps working, while two folders opened at
    once land somewhere different without either being configured. If the
    derived port is taken, the search walks upward; if the whole window is
    somehow full, the kernel picks.

    There is a race between testing a port and Flask binding it, and it is not
    worth closing: losing it means the server fails to start with a clear
    address-in-use error, which is exactly what the user needs to read.
    """
    digest = hashlib.blake2s(str(root).encode("utf-8")).digest()
    first = int.from_bytes(digest[:2], "big") % _PORT_SPAN
    for step in range(64):
        candidate = _PORT_LOW + (first + step) % _PORT_SPAN
        if _free(candidate):
            return candidate
    return _free_ephemeral()


def _free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _free_ephemeral() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def create_app(root: Path, build: Build | None = None) -> Flask:
    """A Flask app serving the UI and API for one folder.

    `build` is an in-flight or completed pipeline run, if the caller started
    one; the app polls it for progress and reloads the library when it
    finishes, so the page can be opened while the embed pass is still going.
    """
    # Served at the root rather than under /static, so the page's own relative
    # links (`./style.css`) resolve without the HTML having to know it is being
    # served by Flask at all -- it opens identically from the filesystem.
    app = Flask(
        __name__,
        static_folder=str(Path(__file__).parent / "static"),
        static_url_path="",
    )
    build = build or Build(root, stages=())
    state: dict = {"library": None, "loaded_while_building": None}

    def library() -> Library:
        """The current library, reloaded once each time a build finishes.

        Reloading is keyed on the build no longer running rather than on a
        completion callback, so a build that fails or is interrupted still
        leaves the UI showing whatever did get written.
        """
        building = build.progress.running
        if state["library"] is None or state["loaded_while_building"] != building:
            state["library"] = Library.open(root, encode_text=build.encode_text)
            state["loaded_while_building"] = building
        return state["library"]

    def query_from_request() -> Query:
        args = request.args
        like = args.get("like", type=int)
        return Query(
            text=args.get("q", ""),
            like=like,
            category=args.get("cat", ""),
            folder=args.get("folder", ""),
            group=args.get("group", ""),
            show=args.get("show", "unsaved"),
            view=args.get("view", "active"),
            order=args.get("order", "relevance"),
        )

    def rows_from_body() -> list[int]:
        body = request.get_json(silent=True) or {}
        return [int(row) for row in body.get("ids", []) if isinstance(row, int)]

    # -- the page ---------------------------------------------------------

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    # -- reading ----------------------------------------------------------

    @app.get("/api/search")
    def search():
        limit = min(request.args.get("limit", DEFAULT_PAGE, type=int), MAX_PAGE)
        offset = max(request.args.get("offset", 0, type=int), 0)
        tiles, matched = library().page(query_from_request(), limit, offset)
        return jsonify(
            {
                "matched": matched,
                "offset": offset,
                "results": [tile.__dict__ for tile in tiles],
            }
        )

    @app.get("/api/ids")
    def ids():
        """Every id matching the filters -- what "select all matching" acts on.

        Shares `Library.select` with /api/search by construction, which is the
        property that keeps the count the user reads equal to the set they are
        about to act on.
        """
        rows = library().select(query_from_request())
        return jsonify({"ids": rows, "matched": len(rows)})

    @app.get("/api/stats")
    def stats():
        payload = library().stats().__dict__.copy()
        payload["guessing_below"] = GUESSING_BELOW
        return jsonify(payload)

    @app.get("/api/progress")
    def progress():
        return jsonify(build.progress.as_json())

    @app.get("/api/batches")
    def batches():
        """Quarantine batches oldest first, so any of them can be undone."""
        return jsonify({"batches": library().where.batches()})

    @app.get("/thumb/<int:row>")
    def thumb(row: int):
        path = library().cache.thumb_path(row)
        if not path.exists():
            return "", 404
        return send_file(path, mimetype="image/jpeg", max_age=86400)

    @app.get("/full/<int:row>")
    def full(row: int):
        """The original file, wherever it currently is -- folder or quarantine."""
        current = library()
        if not 0 <= row < len(current.records):
            return "", 404
        path = current.where.location(row)
        if not path.exists():
            return "", 404
        return send_file(path, max_age=3600)

    # -- writing ----------------------------------------------------------

    @app.post("/api/quarantine")
    def quarantine():
        return jsonify(_report(library().where.quarantine(rows_from_body())))

    @app.post("/api/restore")
    def restore():
        return jsonify(_report(library().where.restore(rows_from_body())))

    @app.post("/api/save")
    def save():
        body = request.get_json(silent=True) or {}
        protect = bool(body.get("protected", True))
        return jsonify(_report(library().where.set_saved(rows_from_body(), protect)))

    @app.post("/api/reviewed")
    def reviewed():
        body = request.get_json(silent=True) or {}
        category = str(body.get("category", ""))
        if category:
            library().mark_reviewed(category)
        return jsonify({"reviewed": sorted(library().cache.load_reviewed())})

    # -- one handler for everything that goes wrong ------------------------

    @app.errorhandler(Exception)
    def on_error(exc: Exception):
        """The one handler for everything that goes wrong below this layer.

        A deliberate HTTP status -- a 404 for a thumbnail that has not been
        generated yet, a 405 -- is an answer, not a failure, and is passed
        through unchanged. Everything else is a bug, and is logged with its
        traceback and reported as a 500 exactly once.
        """
        if isinstance(exc, HTTPException):
            return exc
        log.exception("unhandled error serving %s", request.path)
        return jsonify({"error": str(exc)}), 500

    return app


def _report(result) -> dict:
    """Serialise a MoveResult, keeping refusals visible to the caller.

    A request that was declined -- a protected image, an occupied path -- is
    reported rather than quietly dropped, so the UI can revert the optimistic
    update it already made and say why.
    """
    return {
        "moved": result.moved,
        "refused": {str(row): reason for row, reason in result.refused.items()},
    }
