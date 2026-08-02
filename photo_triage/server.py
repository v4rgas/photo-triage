"""Stage 5 -- the HTTP layer.

Thin by design. Every handler does three things: turn request arguments into a
`Query` or a list of row ids, call the Library or the Quarantine, and serialise
what comes back. No filtering, no path arithmetic and no protection logic lives
here, because all three are decisions that belong below the transport and must
hold for a POST made with curl exactly as they do for one made by the page.

Binds to 127.0.0.1 only. This is someone's personal photo library and there is
no authentication anywhere in it; there must never be a host argument.
"""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.exceptions import HTTPException

from .library import Library, Query
from .pipeline import Build
from .prompts import GUESSING_BELOW

log = logging.getLogger(__name__)

DEFAULT_PAGE = 300
MAX_PAGE = 1000


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
