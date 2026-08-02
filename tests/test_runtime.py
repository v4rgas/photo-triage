"""Picking the right PyTorch wheel without asking the user."""

from __future__ import annotations

import pytest

from photo_triage.runtime import Backend, detect_backend, installed_backend


def test_detection_answers_something_usable_on_this_machine():
    backend = detect_backend()
    assert backend.name in {"cuda", "rocm", "mps", "cpu"}
    assert backend.why  # the evidence is always quotable back to the user


@pytest.mark.parametrize(
    "name,expects_index",
    [("cuda", False), ("mps", False), ("rocm", True), ("cpu", True)],
)
def test_only_the_non_default_backends_pin_an_index(name, expects_index):
    command = Backend(name, "test").install_command("/usr/bin/python3")
    assert ("--index-url" in command) is expects_index
    assert {"torch", "torchvision", "open_clip_torch"} <= set(command)


def test_the_install_command_is_an_argument_list_never_a_shell_string():
    """Paths here contain spaces and emoji; nothing may be concatenated."""
    command = Backend("rocm", "test").install_command("/home/a b/π/bin/python")
    assert all(isinstance(part, str) for part in command)
    assert "/home/a b/π/bin/python" in command


def test_rocm_stays_pinned_below_the_version_that_breaks_gfx1031():
    """6.4.3+ segfaults under the architecture override; the pin is the fix."""
    command = Backend("rocm", "test").install_command()
    index = command[command.index("--index-url") + 1]
    assert index.endswith("/rocm6.3")


def test_installed_backend_reports_a_known_name_or_nothing():
    assert installed_backend() in {None, "cuda", "rocm", "mps", "cpu"}
