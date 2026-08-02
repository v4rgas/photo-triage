"""Every module imports, and the CLI parses its own arguments.

This exists because it did not, and a syntax error in `__main__.py` shipped
past a green test suite: nothing else imports it, so nothing else compiles it.
The failure mode is the worst kind, since the command is broken for everybody
while every test still passes.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import photo_triage

# Modules that pull in PyTorch, which the suite deliberately does not have.
_NEEDS_MODEL = {"photo_triage.embed", "photo_triage.classify"}


def _module_names() -> list[str]:
    return [
        f"photo_triage.{info.name}"
        for info in pkgutil.iter_modules(photo_triage.__path__)
    ]


@pytest.mark.parametrize("name", _module_names())
def test_every_module_compiles_and_imports(name):
    if name in _NEEDS_MODEL:
        pytest.importorskip("torch", reason="only the model path needs it")
    importlib.import_module(name)


def test_the_cli_is_reachable_and_parses(capsys):
    """`photo-triage --help` is the first thing anybody runs."""
    from photo_triage.__main__ import main

    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "photo-triage" in capsys.readouterr().out


def test_a_missing_folder_is_reported_rather_than_traced(tmp_path, capsys):
    from photo_triage.__main__ import main

    assert main([str(tmp_path / "nope")]) == 2
    assert "not a folder" in capsys.readouterr().err


def test_start_marks_the_build_running_before_it_returns(tmp_path, monkeypatch):
    """Otherwise a caller that waits for it sees "not running" and carries on.

    Setting the flag inside the thread leaves a window between start()
    returning and the thread being scheduled, and the CLI served an unbuilt
    folder whenever it lost that race.

    The thread is stubbed out rather than raced against: the property under
    test is that the flag is set on the calling thread, and a real thread that
    finishes quickly would clear it again before any assertion could run.
    """
    import threading

    from photo_triage.pipeline import Build

    started = []
    monkeypatch.setattr(
        threading, "Thread",
        lambda *a, **k: type("Stub", (), {"start": lambda self: started.append(1),
                                          "join": lambda self: None})(),
    )
    build = Build(tmp_path, stages=())
    build.start()
    assert build.progress.running is True
    assert started == [1]


def test_wait_is_safe_on_a_build_that_never_started(tmp_path):
    from photo_triage.pipeline import Build

    Build(tmp_path, stages=()).wait()


def test_a_build_that_ran_is_finished_after_wait(tmp_path):
    from photo_triage.pipeline import Build

    build = Build(tmp_path, stages=())
    build.start()
    build.wait()
    assert build.progress.running is False
