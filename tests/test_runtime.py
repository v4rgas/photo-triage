"""Picking the right PyTorch wheel without asking the user."""

from __future__ import annotations

import pytest

from photo_triage.runtime import Backend, detect_backend, installed_backend


def test_detection_answers_something_usable_on_this_machine():
    backend = detect_backend()
    assert backend.name in {"cuda", "rocm", "mps", "cpu"}
    assert backend.why  # the evidence is always quotable back to the user


@pytest.mark.parametrize(
    "name,expected",
    [("cuda", "auto"), ("mps", "auto"), ("rocm", "rocm6.3"), ("cpu", "cpu")],
)
def test_each_vendor_maps_to_a_torch_backend_uv_understands(name, expected):
    assert Backend(name, "test").torch_backend == expected


def test_amd_is_named_explicitly_rather_than_left_to_auto():
    """uv's auto reads NVIDIA drivers and picks the CPU build on a Radeon."""
    assert Backend("rocm", "test").torch_backend != "auto"


def test_rocm_stays_pinned_below_the_version_that_breaks_gfx1031():
    """6.4.3+ segfaults under the architecture override; the pin is the fix."""
    assert Backend("rocm", "test").torch_backend == "rocm6.3"


def test_the_install_command_is_the_one_supported_command():
    command = Backend("rocm", "test").install_command()
    assert command.startswith("uv tool install")
    assert "photo-triage[model]" in command
    assert "--torch-backend=rocm6.3" in command


def test_installed_backend_reports_a_known_name_or_nothing():
    assert installed_backend() in {None, "cuda", "rocm", "mps", "cpu"}


def test_advice_follows_the_route_the_user_installed_by(monkeypatch):
    """A pacman user told to run uv would shadow a tracked install with an
    untracked one, so the advice has to match how they got here."""
    import shutil

    import photo_triage.runtime as runtime

    monkeypatch.setattr(runtime, "externally_managed", lambda: True)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/pacman")
    assert runtime.Backend("rocm", "t").install_command() == (
        "sudo pacman -S python-pytorch-rocm python-open-clip-torch"
    )

    monkeypatch.setattr(runtime, "externally_managed", lambda: False)
    assert runtime.Backend("rocm", "t").install_command().startswith("uv tool install")
